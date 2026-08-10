"""
prediction.py
Prediction utilities — load saved model, predict severity,
extract risk indicators, model-estimated probabilities.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_model.pkl")

FEATURE_COLS = ["Activity", "Location Type", "Time", "Description"]


def load_model(model_path: str = MODEL_PATH):
    """Load the saved sklearn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run: python src/train.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_severity(model, activity: str, location: str,
                     time: str, description: str) -> Dict[str, Any]:
    """
    Predict severity for a single incident/activity.

    Returns:
        predicted_severity: str  (Low / Medium / High)
        probabilities: dict      {class: probability} if supported
        class_order: list        order of classes
    """
    input_df = pd.DataFrame([{
        "Activity": activity,
        "Location Type": location,
        "Time": time,
        "Description": description,
    }])

    predicted = model.predict(input_df)[0]

    # Probabilities (supported by LR and CalibratedClassifierCV)
    try:
        prob_array = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        probabilities = {cls: float(prob) for cls, prob in zip(classes, prob_array)}
    except AttributeError:
        probabilities = {}

    return {
        "predicted_severity": predicted,
        "probabilities": probabilities,
        "input": {
            "activity": activity,
            "location": location,
            "time": time,
            "description": description,
        },
    }


def get_top_model_indicators(model, class_name: str = "High",
                              top_n: int = 10) -> List[str]:
    """
    Extract top TF-IDF feature names with highest coefficient for
    the given class from a linear model.

    These are STATISTICAL INDICATORS — not proven causes.
    """
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["classifier"]

        feature_names = tfidf.get_feature_names_out()

        # Handle CalibratedClassifierCV wrapper
        if hasattr(clf, "estimator"):
            base_clf = clf.calibrated_classifiers_[0].estimator
        elif hasattr(clf, "calibrated_classifiers_"):
            base_clf = clf.calibrated_classifiers_[0].estimator
        else:
            base_clf = clf

        if not hasattr(base_clf, "coef_"):
            return []

        classes = list(model.classes_)
        if class_name not in classes:
            return []

        class_idx = classes.index(class_name)
        coef = base_clf.coef_[class_idx]
        top_indices = np.argsort(coef)[::-1][:top_n]
        return [feature_names[i] for i in top_indices]

    except Exception:
        return []
