"""
analytics.py
Dashboard analytics — all values calculated dynamically from actual data.

STRICT RULE: No hard-coded statistics. Everything calculated from the dataset.
"""

import pandas as pd
from typing import Dict, Any


def get_kpi_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate KPI metrics dynamically from Incident Reports DataFrame."""
    severity_counts = df["Severity"].value_counts()
    activity_counts = df["Activity"].value_counts()

    high_df = df[df["Severity"] == "High"]
    most_frequent_high_activity = (
        high_df["Activity"].mode()[0] if not high_df.empty else "N/A"
    )

    # Most frequent hazard if hazard column exists
    most_frequent_hazard = "N/A"
    if "primary_hazard" in df.columns:
        most_frequent_hazard = df["primary_hazard"].mode()[0] if not df.empty else "N/A"

    return {
        "total_incidents": len(df),
        "high_count": int(severity_counts.get("High", 0)),
        "medium_count": int(severity_counts.get("Medium", 0)),
        "low_count": int(severity_counts.get("Low", 0)),
        "most_frequent_activity": activity_counts.index[0] if not activity_counts.empty else "N/A",
        "most_frequent_high_activity": most_frequent_high_activity,
        "most_frequent_hazard": most_frequent_hazard,
        "unique_activities": df["Activity"].nunique(),
        "unique_locations": df["Location Type"].nunique(),
    }


def get_severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return severity counts as a DataFrame."""
    counts = df["Severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    # Enforce order: High, Medium, Low
    order = ["High", "Medium", "Low"]
    counts["Severity"] = pd.Categorical(counts["Severity"], categories=order, ordered=True)
    counts = counts.sort_values("Severity").reset_index(drop=True)
    return counts


def get_activity_severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return activity × severity crosstab calculated from Incident Reports."""
    pivot = (
        df.groupby(["Activity", "Severity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure columns exist
    for sev in ["High", "Medium", "Low"]:
        if sev not in pivot.columns:
            pivot[sev] = 0
    return pivot


def get_location_severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return location type × severity crosstab."""
    pivot = (
        df.groupby(["Location Type", "Severity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for sev in ["High", "Medium", "Low"]:
        if sev not in pivot.columns:
            pivot[sev] = 0
    return pivot


def get_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return incident count by hour of day.
    Extracts hour from Time column (HH:MM strings).
    """
    def to_hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return -1

    df_copy = df.copy()
    df_copy["hour"] = df_copy["Time"].apply(to_hour)
    df_copy = df_copy[df_copy["hour"] >= 0]
    hourly = df_copy.groupby(["hour", "Severity"]).size().unstack(fill_value=0).reset_index()
    for sev in ["High", "Medium", "Low"]:
        if sev not in hourly.columns:
            hourly[sev] = 0
    return hourly


def get_high_risk_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Return activities sorted by High-risk incident count."""
    high_df = df[df["Severity"] == "High"]
    counts = high_df["Activity"].value_counts().reset_index()
    counts.columns = ["Activity", "High Risk Count"]
    return counts


def get_recurring_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify recurring patterns from historical incidents.
    All values calculated from actual data — nothing invented.
    """
    # Top activities by incident volume
    top_activities = df["Activity"].value_counts().head(5).to_dict()

    # Top locations
    top_locations = df["Location Type"].value_counts().head(5).to_dict()

    # High-risk activity breakdown
    high_df = df[df["Severity"] == "High"]
    high_by_activity = high_df["Activity"].value_counts().to_dict()

    # Time patterns (morning vs afternoon vs evening)
    def time_band(t):
        try:
            h = int(str(t).split(":")[0])
            if h < 12:
                return "Morning (06:00–11:59)"
            elif h < 17:
                return "Afternoon (12:00–16:59)"
            else:
                return "Evening (17:00+)"
        except Exception:
            return "Unknown"

    df_copy = df.copy()
    df_copy["time_band"] = df_copy["Time"].apply(time_band)
    time_band_counts = df_copy["time_band"].value_counts().to_dict()

    return {
        "top_activities": top_activities,
        "top_locations": top_locations,
        "high_risk_by_activity": high_by_activity,
        "time_band_distribution": time_band_counts,
    }
