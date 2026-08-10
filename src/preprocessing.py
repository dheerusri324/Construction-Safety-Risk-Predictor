"""
preprocessing.py
Feature engineering and text preparation for the ML pipeline.

Combines Activity, Location Type, Time, and Description into a
single text representation for TF-IDF, plus structured features.

STRICT LEAKAGE RULE: Severity, IS Codes, Precautions, Safety Warning
are NEVER used as input features.
"""

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ─── Text Combination ────────────────────────────────────────────────────────

def build_combined_text(row: pd.Series) -> str:
    """
    Combine Activity, Location Type, Time, and Description into a
    single text string for TF-IDF vectorization.

    No leakage: only uses the four allowed input fields.
    """
    activity = str(row.get("Activity", "")).strip()
    location = str(row.get("Location Type", "")).strip()
    time_val = str(row.get("Time", "")).strip()
    description = str(row.get("Description", "")).strip()

    combined = (
        f"activity {activity} "
        f"location {location} "
        f"time {time_val} "
        f"{description}"
    )
    return combined.lower()


def add_combined_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'combined_text' column to DataFrame."""
    df = df.copy()
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    return df


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove excess whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Time Feature Extraction ─────────────────────────────────────────────────

def extract_hour(time_str: str) -> float:
    """Extract hour as numeric from HH:MM string. Returns 12.0 on failure."""
    try:
        parts = str(time_str).split(":")
        return float(parts[0])
    except Exception:
        return 12.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric hour feature from Time column."""
    df = df.copy()
    df["hour"] = df["Time"].apply(extract_hour)
    return df


# ─── Custom Sklearn Transformer ──────────────────────────────────────────────

class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that combines Activity, Location Type,
    Time, and Description into a single text string.

    Placed inside Pipeline so it is applied consistently during both
    training and prediction — no leakage risk.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(build_combined_text, axis=1).tolist()
        raise ValueError("TextCombiner expects a pandas DataFrame.")


# ─── Label Mapping ────────────────────────────────────────────────────────────

SEVERITY_ORDER = ["Low", "Medium", "High"]

def encode_severity(series: pd.Series) -> pd.Series:
    """Map Severity strings to ordered integers (Low=0, Medium=1, High=2)."""
    mapping = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    return series.map(mapping)

def decode_severity(code: int) -> str:
    return SEVERITY_ORDER[code] if 0 <= code < len(SEVERITY_ORDER) else "Unknown"
