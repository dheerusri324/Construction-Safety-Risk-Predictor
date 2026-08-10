"""
evaluation.py
Model evaluation utilities — stratified CV, metrics, confusion matrix.

ALL METRICS COME FROM ACTUAL EVALUATION. Never hard-coded.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from typing import Dict, Any


SEVERITY_CLASSES = ["High", "Low", "Medium"]  # alphabetical as sklearn sorts
SEVERITY_DISPLAY = ["Low", "Medium", "High"]   # display order


def run_stratified_cv(pipeline, X: pd.DataFrame, y: pd.Series,
                      n_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Run stratified K-fold cross-validation.
    TF-IDF is fitted INSIDE the pipeline per fold — no leakage.
    Returns mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro",
               "precision_weighted", "recall_weighted"]

    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring,
                                return_train_score=False)

    return {
        "cv_folds": n_splits,
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "macro_precision_mean": float(np.mean(cv_results["test_precision_macro"])),
        "macro_recall_mean": float(np.mean(cv_results["test_recall_macro"])),
        "macro_f1_mean": float(np.mean(cv_results["test_f1_macro"])),
        "macro_f1_std": float(np.std(cv_results["test_f1_macro"])),
        "weighted_precision_mean": float(np.mean(cv_results["test_precision_weighted"])),
        "weighted_recall_mean": float(np.mean(cv_results["test_recall_weighted"])),
        "weighted_f1_mean": float(np.mean(cv_results["test_f1_weighted"])),
        "per_fold_accuracy": [float(v) for v in cv_results["test_accuracy"]],
        "per_fold_macro_f1": [float(v) for v in cv_results["test_f1_macro"]],
    }


def compute_full_report(pipeline, X: pd.DataFrame, y_true: pd.Series,
                        y_pred: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute full classification metrics on provided data.
    If y_pred is None, uses pipeline.predict(X).
    """
    if y_pred is None:
        y_pred = pipeline.predict(X)

    classes = sorted(y_true.unique())

    report_dict = classification_report(y_true, y_pred,
                                        target_names=classes,
                                        output_dict=True,
                                        zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for cls in classes:
        per_class[cls] = {
            "precision": float(report_dict[cls]["precision"]),
            "recall": float(report_dict[cls]["recall"]),
            "f1": float(report_dict[cls]["f1-score"]),
            "support": int(report_dict[cls]["support"]),
        }

    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
    }


def save_evaluation_results(results: dict, filepath: str):
    """Save evaluation results to JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def load_evaluation_results(filepath: str) -> dict:
    """Load previously saved evaluation results."""
    with open(filepath, "r") as f:
        return json.load(f)
