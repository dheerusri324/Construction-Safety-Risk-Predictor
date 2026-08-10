"""
planned_activities.py
Planned Activities component.

Generates a synthetic planned-activity dataset and predicts risk
for upcoming activities. Does NOT modify the frozen incident dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# ─── Synthetic Planned Activity Dataset ──────────────────────────────────────
# Representative planned activities (not derived from the 501 historical records)

PLANNED_ACTIVITIES_DATA = [
    {
        "Date": (date.today() + timedelta(days=1)).isoformat(),
        "Activity": "Working at height",
        "Location Type": "roof slab",
        "Time": "09:00",
        "Planned Description": (
            "Installation of formwork near an elevated slab edge. "
            "Workers will operate close to the perimeter without permanent guardrails."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=1)).isoformat(),
        "Activity": "Excavation",
        "Location Type": "foundation pit",
        "Time": "08:00",
        "Planned Description": (
            "Deep trench excavation for foundation work. "
            "Ground conditions to be assessed; no shoring installed yet."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=1)).isoformat(),
        "Activity": "Lifting",
        "Location Type": "crane bay",
        "Time": "10:00",
        "Planned Description": (
            "Heavy precast element to be lifted and placed using tower crane. "
            "Exclusion zone to be established beneath the load path."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=1)).isoformat(),
        "Activity": "Scaffolding",
        "Location Type": "facade platform",
        "Time": "07:30",
        "Planned Description": (
            "Erection of external scaffolding along the building facade. "
            "Scaffold tie points and bracing to be completed during erection."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=1)).isoformat(),
        "Activity": "Electrical work",
        "Location Type": "substation room",
        "Time": "11:00",
        "Planned Description": (
            "Temporary electrical distribution board installation. "
            "Circuit isolation required before commencing connection work."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=2)).isoformat(),
        "Activity": "Concrete pouring",
        "Location Type": "slab casting area",
        "Time": "07:00",
        "Planned Description": (
            "Ground floor slab concrete pour. Pump and vibrator operations "
            "coordinated; formwork inspection completed."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=2)).isoformat(),
        "Activity": "Hot work",
        "Location Type": "steel yard",
        "Time": "09:30",
        "Planned Description": (
            "Welding and cutting of structural steel members. "
            "Hot-work permit to be obtained; combustible materials cleared from area."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=2)).isoformat(),
        "Activity": "Material handling",
        "Location Type": "material laydown area",
        "Time": "13:00",
        "Planned Description": (
            "Unloading and stacking of heavy block materials. "
            "Forklift to be used; pedestrian exclusion zone required."
        ),
    },
    {
        "Date": (date.today() + timedelta(days=2)).isoformat(),
        "Activity": "Housekeeping",
        "Location Type": "general site",
        "Time": "16:00",
        "Planned Description": (
            "End-of-day site cleanup. Removal of debris, clearing access routes, "
            "and securing loose materials before site close."
        ),
    },
]


def get_planned_activities_df() -> pd.DataFrame:
    """Return planned activities as a DataFrame."""
    return pd.DataFrame(PLANNED_ACTIVITIES_DATA)


def predict_planned_activities(model, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run risk predictions for all planned activities.
    Returns DataFrame with predicted severity and probabilities.
    """
    if df is None:
        df = get_planned_activities_df()

    results = []
    for _, row in df.iterrows():
        input_df = pd.DataFrame([{
            "Activity": row["Activity"],
            "Location Type": row["Location Type"],
            "Time": row["Time"],
            "Description": row["Planned Description"],
        }])

        predicted = model.predict(input_df)[0]

        try:
            prob_array = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            probs = {cls: float(p) for cls, p in zip(classes, prob_array)}
            confidence = max(probs.values())
        except AttributeError:
            probs = {}
            confidence = None

        results.append({
            "Date": row["Date"],
            "Activity": row["Activity"],
            "Location Type": row["Location Type"],
            "Time": row["Time"],
            "Planned Description": row["Planned Description"],
            "Predicted Severity": predicted,
            "High Prob": probs.get("High", None),
            "Medium Prob": probs.get("Medium", None),
            "Low Prob": probs.get("Low", None),
            "Confidence": confidence,
        })

    result_df = pd.DataFrame(results)

    # Sort by risk priority: High → Medium → Low
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    result_df["_sort"] = result_df["Predicted Severity"].map(severity_order)
    result_df = result_df.sort_values(["Date", "_sort"]).drop("_sort", axis=1)
    result_df = result_df.reset_index(drop=True)

    return result_df


SEVERITY_EMOJI = {
    "High": "🔴",
    "Medium": "🟠",
    "Low": "🟡",
}

def get_priority_summary(predicted_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return priority list with emoji for display."""
    priorities = []
    for _, row in predicted_df.iterrows():
        sev = row["Predicted Severity"]
        priorities.append({
            "emoji": SEVERITY_EMOJI.get(sev, "⚪"),
            "severity": sev,
            "activity": row["Activity"],
            "location": row["Location Type"],
            "date": row["Date"],
            "time": row["Time"],
        })
    return priorities
