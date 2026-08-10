"""
train.py
Model training script for the Construction Safety Risk Predictor.

Run:  python src/train.py

Steps:
1. Load Incident Reports
2. Validate columns
3. Build combined text features (Activity + Location + Time + Description)
4. Create TF-IDF + classifier pipelines (Logistic Regression, Linear SVM)
5. 5-fold Stratified Cross-Validation for each candidate
6. Select best model by macro F1
7. Retrain best model on full dataset
8. Save complete pipeline + evaluation results

LEAKAGE GUARDS:
- TF-IDF fitted INSIDE pipeline (per fold), never on full dataset first
- Severity / IS Codes / Precautions / Safety Warning NEVER used as features
- Incident ID NOT used as a feature
"""

import os
import sys
import json
import pickle
import numpy as np

# Allow running from project root or src/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.data_loader import load_incidents, load_safety_codes
from src.preprocessing import TextCombiner
from src.evaluation import run_stratified_cv, compute_full_report, save_evaluation_results

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")
EVAL_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
RANDOM_STATE = 42
N_FOLDS = 5


def build_pipeline(classifier) -> Pipeline:
    """
    Build complete sklearn Pipeline:
    DataFrame → TextCombiner → TF-IDF → Classifier

    TF-IDF is INSIDE the pipeline → fitted per fold → no leakage.
    """
    return Pipeline([
        ("text_combiner", TextCombiner()),
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),        # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True,         # log-scale TF
            stop_words="english",
            min_df=2,                  # ignore very rare terms
        )),
        ("classifier", classifier),
    ])


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    df = load_incidents()
    print(f"  Loaded {len(df)} records.")

    # ── 2. Validate ───────────────────────────────────────────────────────────
    severity_counts = df["Severity"].value_counts().to_dict()
    print(f"  Severity distribution: {severity_counts}")

    # ── 3. Features + Target ──────────────────────────────────────────────────
    # Features: Activity, Location Type, Time, Description only
    # NO Severity, NO IS Codes, NO Precautions, NO Safety Warning, NO Incident ID
    FEATURE_COLS = ["Activity", "Location Type", "Time", "Description"]
    X = df[FEATURE_COLS].copy()
    y = df["Severity"]

    # ── 4. Candidate Models ──────────────────────────────────────────────────
    candidates = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                C=1.0,
                solver="lbfgs",
            )
        ),
        "Linear SVM (calibrated)": build_pipeline(
            CalibratedClassifierCV(
                LinearSVC(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    C=1.0,
                ),
                cv=3,
            )
        ),
    }

    # ── 5. Stratified CV for each candidate ──────────────────────────────────
    print(f"\nRunning {N_FOLDS}-fold Stratified CV...")
    cv_results = {}
    for name, pipeline in candidates.items():
        print(f"  Evaluating: {name}")
        result = run_stratified_cv(pipeline, X, y,
                                   n_splits=N_FOLDS,
                                   random_state=RANDOM_STATE)
        cv_results[name] = result
        print(f"    Accuracy : {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"    Macro F1 : {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}")
        print(f"    Wtd F1   : {result['weighted_f1_mean']:.4f}")

    # ── 6. Select Best Model by Macro F1 ──────────────────────────────────────
    best_name = max(cv_results, key=lambda n: cv_results[n]["macro_f1_mean"])
    best_pipeline = candidates[best_name]
    print(f"\nBest model: {best_name} (Macro F1 = {cv_results[best_name]['macro_f1_mean']:.4f})")

    # ── 7. Retrain on Full Dataset ────────────────────────────────────────────
    print("Retraining best model on full dataset...")
    best_pipeline.fit(X, y)
    print("  Training complete.")

    # ── 8. Full-dataset report (in-sample, labelled as such) ─────────────────
    full_report = compute_full_report(best_pipeline, X, y)

    # ── 9. Save model ─────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"  Model saved -> {MODEL_PATH}")

    # ── 10. Save evaluation results ───────────────────────────────────────────
    eval_output = {
        "best_model_name": best_name,
        "cv_results": cv_results,
        "full_dataset_report": full_report,
        "feature_columns": FEATURE_COLS,
        "classes": sorted(y.unique().tolist()),
        "dataset_info": {
            "total_records": len(df),
            "severity_distribution": {k: int(v) for k, v in severity_counts.items()},
            "n_folds": N_FOLDS,
            "random_state": RANDOM_STATE,
        },
    }
    save_evaluation_results(eval_output, EVAL_PATH)
    print(f"  Evaluation saved -> {EVAL_PATH}")

    print("\nTraining complete.")
    print(f"   Best model : {best_name}")
    print(f"   CV Accuracy: {cv_results[best_name]['accuracy_mean']:.4f}")
    print(f"   CV Macro F1: {cv_results[best_name]['macro_f1_mean']:.4f}")

    return best_pipeline, eval_output


if __name__ == "__main__":
    train()
