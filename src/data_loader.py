"""
data_loader.py
Load and validate the Construction Safety Risk Predictor dataset.
"""

import os
import pandas as pd

REQUIRED_COLUMNS = ["Activity", "Location Type", "Time", "Description", "Severity"]
OPTIONAL_COLUMNS = [
    "Incident ID",
    "Applicable IS Codes / Standards",
    "Recommended Precautions / Preventive Actions",
    "Safety Warning",
]

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data",
                         "Construction_Safety_Risk_Predictor_501_Final.xlsx")


def get_data_path():
    return os.path.abspath(DATA_FILE)


def load_incidents(filepath: str = None) -> pd.DataFrame:
    """Load and validate the Incident Reports sheet."""
    if filepath is None:
        filepath = get_data_path()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_excel(filepath, sheet_name="Incident Reports")

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    # Basic cleaning — strip whitespace from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert Severity to title case for consistency
    df["Severity"] = df["Severity"].str.strip().str.capitalize()

    return df


def load_safety_codes(filepath: str = None) -> pd.DataFrame:
    """Load the Safety Code Reference sheet."""
    if filepath is None:
        filepath = get_data_path()
    df = pd.read_excel(filepath, sheet_name="Safety Code Reference")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_activity_risk(filepath: str = None) -> pd.DataFrame:
    """Load the Activity vs Risk sheet (informational only)."""
    if filepath is None:
        filepath = get_data_path()
    return pd.read_excel(filepath, sheet_name="Activity vs Risk")


def load_risk_summary(filepath: str = None) -> pd.DataFrame:
    """Load the Risk Summary sheet (informational only)."""
    if filepath is None:
        filepath = get_data_path()
    return pd.read_excel(filepath, sheet_name="Risk Summary")


def load_data_dictionary(filepath: str = None) -> pd.DataFrame:
    """Load the Data Dictionary sheet."""
    if filepath is None:
        filepath = get_data_path()
    return pd.read_excel(filepath, sheet_name="Data Dictionary")


def validate_incidents(df: pd.DataFrame) -> dict:
    """Return a dict of data-quality checks."""
    report = {
        "row_count": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_ids": int(df["Incident ID"].duplicated().sum()) if "Incident ID" in df.columns else "N/A",
        "unique_descriptions": int(df["Description"].nunique()),
        "severity_distribution": df["Severity"].value_counts().to_dict(),
        "activity_distribution": df["Activity"].value_counts().to_dict(),
        "location_distribution": df["Location Type"].value_counts().to_dict(),
        "description_length_stats": df["Description"].str.len().describe().to_dict(),
    }
    return report
