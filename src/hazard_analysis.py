"""
hazard_analysis.py
Controlled hazard taxonomy, keyword extraction, and hazard categorization.

Keyword matching is a statistical indicator only — not proof of causality.
"""

from typing import List, Dict
import pandas as pd
import re

# ─── Hazard Taxonomy ──────────────────────────────────────────────────────────

HAZARD_TAXONOMY: Dict[str, List[str]] = {
    "Fall": [
        "fall", "fell", "edge", "guardrail", "barricade", "unprotected edge",
        "open shaft", "lift shaft", "parapet", "roof edge", "slab edge",
        "height", "elevated", "drop", "fall protection", "perimeter",
        "fall exposure", "missing guardrail", "no guardrail", "open edge",
    ],
    "Crushing / Struck-by": [
        "struck", "crush", "load", "suspended load", "overhead load",
        "falling object", "dropped", "swing", "material falling",
        "load shifting", "rigging", "crane", "hoist", "load control",
        "impact", "exclusion zone", "below the load",
    ],
    "Electrical": [
        "electrical", "cable", "exposed cable", "live wire", "electric",
        "shock", "circuit", "panel", "switchboard", "temporary power",
        "voltage", "arc", "isolation", "conductor", "earthing",
    ],
    "Scaffold Instability": [
        "scaffold", "scaffolding", "platform", "unstable scaffold",
        "incomplete scaffold", "scaffold frame", "ledger", "standard",
        "scaffold inspection", "putlog", "overloaded scaffold",
        "scaffold access", "scaffold tie", "scaffold brace",
    ],
    "Excavation / Collapse": [
        "excavation", "trench", "collapse", "cave-in", "spoil",
        "shoring", "ground condition", "deep trench", "pit",
        "excavated", "soil collapse", "slope", "battering",
        "unsupported excavation", "groundwater",
    ],
    "Fire / Hot Work": [
        "fire", "hot work", "welding", "cutting", "grinding",
        "spark", "flame", "ignition", "combustible", "flammable",
        "gas cylinder", "oxy-acetylene", "heat source", "fire hazard",
    ],
    "Slip / Trip": [
        "slip", "trip", "slippery", "wet surface", "uneven surface",
        "debris", "obstruction", "housekeeping", "spill", "mud",
        "poor lighting", "access route", "walkway", "pathway blocked",
    ],
    "Manual Handling": [
        "manual handling", "lifting manually", "awkward posture",
        "heavy load", "carry", "pushing", "pulling", "ergonomic",
        "strain", "overexertion", "repetitive", "weight",
    ],
    "Equipment-Related": [
        "equipment", "machinery", "plant", "vehicle", "forklift",
        "excavator", "concrete pump", "vibrator", "compactor",
        "moving plant", "reversing", "blind spot", "machinery guard",
        "unsafe equipment", "defective equipment",
    ],
}

ALL_HAZARD_KEYWORDS = {
    kw.lower(): category
    for category, keywords in HAZARD_TAXONOMY.items()
    for kw in keywords
}


def extract_hazard_categories(text: str) -> List[str]:
    """
    Return list of hazard categories detected in text via keyword matching.
    Multiple categories may apply.
    Returns ['Other'] if no keywords match.
    """
    text_lower = text.lower()
    found = set()
    for kw, category in ALL_HAZARD_KEYWORDS.items():
        if kw in text_lower:
            found.add(category)
    return sorted(found) if found else ["Other"]


def extract_risk_indicators(text: str, top_n: int = 6) -> List[str]:
    """
    Extract specific keyword phrases present in text as risk indicators.
    These are statistical indicators, NOT proven causal factors.
    """
    indicator_phrases = [
        "unprotected edge", "missing guardrail", "no guardrail",
        "open shaft", "lift shaft", "slab edge", "elevated",
        "fall exposure", "fall protection", "suspended load",
        "overhead load", "exclusion zone", "rigging",
        "exposed cable", "live wire", "electrical isolation",
        "unstable scaffold", "incomplete scaffold", "scaffold access",
        "excavation", "trench", "unsupported", "collapse",
        "hot work", "welding", "spark", "combustible",
        "slippery", "debris", "poor housekeeping", "obstruction",
        "manual handling", "heavy load", "moving plant",
        "defective equipment", "machinery guard",
        "incomplete protection", "missing control", "incomplete edge",
        "edge protection", "barricade", "perimeter protection",
    ]
    text_lower = text.lower()
    found = [p for p in indicator_phrases if p in text_lower]
    return found[:top_n]


def add_hazard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hazard_categories' and 'primary_hazard' columns from Description."""
    df = df.copy()
    df["hazard_categories"] = df["Description"].apply(extract_hazard_categories)
    df["primary_hazard"] = df["hazard_categories"].apply(lambda x: x[0] if x else "Other")
    return df


def get_hazard_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of hazard category frequencies."""
    from collections import Counter
    all_hazards = []
    for cats in df["hazard_categories"]:
        all_hazards.extend(cats)
    counter = Counter(all_hazards)
    freq_df = pd.DataFrame(list(counter.items()), columns=["Hazard Category", "Count"])
    freq_df = freq_df.sort_values("Count", ascending=False).reset_index(drop=True)
    return freq_df


def get_activity_hazard_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate activities vs hazard categories."""
    rows = []
    for _, row in df.iterrows():
        for haz in row["hazard_categories"]:
            rows.append({"Activity": row["Activity"], "Hazard": haz})
    if not rows:
        return pd.DataFrame()
    temp = pd.DataFrame(rows)
    pivot = temp.groupby(["Activity", "Hazard"]).size().unstack(fill_value=0)
    return pivot
