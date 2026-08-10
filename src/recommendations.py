"""
recommendations.py
Controlled recommendation engine using Safety Code Reference data.

Recommendations are derived from the dataset and controlled reference —
NOT invented by the ML classifier or LLM.
"""

from typing import Dict, List, Optional
import pandas as pd


# ─── Toolbox Talk Mapping ─────────────────────────────────────────────────────

TOOLBOX_TALK_MAPPING: Dict[str, str] = {
    "Working at height":    "Fall prevention and edge protection",
    "Scaffolding":          "Scaffold inspection, stability and safe access",
    "Excavation":           "Excavation protection and safe access",
    "Lifting":              "Safe lifting, rigging and exclusion zones",
    "Electrical work":      "Electrical isolation and temporary electrical safety",
    "Hot work":             "Hot-work controls and fire prevention",
    "Concrete pouring":     "Concrete placement and equipment safety",
    "Formwork/Falsework":   "Formwork stability and inspection",
    "Material handling":    "Safe material handling and manual handling",
    "Housekeeping":         "Slip, trip and access control",
}

# Fallback recommendations per activity (from dataset precautions)
ACTIVITY_PRECAUTIONS: Dict[str, List[str]] = {
    "Working at height": [
        "Provide guardrails and edge protection",
        "Protect all openings and shaft edges",
        "Inspect fall-protection systems and anchor points",
        "Secure ladders and safe access routes",
        "Control falling objects below the work area",
        "Maintain safe perimeter access at all times",
    ],
    "Scaffolding": [
        "Use competent scaffold erection and inspection personnel",
        "Provide stable support, bracing, ties, guardrails and toe boards",
        "Maintain safe access and prevent overloading",
        "Keep platforms clear of obstructions",
        "Inspect scaffold after alteration or adverse conditions",
    ],
    "Excavation": [
        "Assess ground conditions before and during excavation",
        "Provide suitable shoring, sloping or battering protection",
        "Provide safe access and egress from excavations",
        "Barricade excavation edges and control spoil placement",
        "Maintain plant stand-off distances from edges",
        "Inspect after rain and control groundwater accumulation",
    ],
    "Lifting": [
        "Inspect lifting equipment before use",
        "Verify suitable load handling and rigging arrangements",
        "Maintain exclusion zones beneath suspended loads",
        "Use appropriate rigging controls and signalling",
        "Ensure load paths are clear before lifting",
    ],
    "Electrical work": [
        "Isolate and verify electrical circuits before work",
        "Use appropriate temporary electrical safety measures",
        "Protect cables and conductors from damage",
        "Ensure appropriate PPE for electrical work",
        "Inspect temporary electrical installations regularly",
    ],
    "Hot work": [
        "Obtain hot-work permit before commencing",
        "Remove or protect combustible materials from the work area",
        "Provide suitable fire extinguishers nearby",
        "Implement fire watch during and after hot work",
        "Inspect gas cylinders and hoses before use",
    ],
    "Concrete pouring": [
        "Inspect formwork and falsework before concrete placement",
        "Maintain safe access and exclusion zones during pouring",
        "Ensure equipment is inspected and fit for purpose",
        "Coordinate pump and vibrator operations safely",
        "Manage concrete spillage and slippery surfaces",
    ],
    "Formwork/Falsework": [
        "Inspect formwork and falsework before loading",
        "Verify structural stability before and during concrete placement",
        "Control access to falsework areas",
        "Follow engineer-approved erection and stripping procedures",
    ],
    "Material handling": [
        "Assess loads before manual handling",
        "Use mechanical aids where loads are excessive",
        "Follow safe manual handling techniques",
        "Maintain clear access and storage routes",
        "Inspect storage areas for stability and safety",
    ],
    "Housekeeping": [
        "Maintain clear access routes and walkways",
        "Remove waste and debris promptly",
        "Control slip and trip hazards",
        "Ensure adequate lighting in work areas",
        "Report and correct poor housekeeping conditions immediately",
    ],
}

# Safety warnings per activity (from controlled reference)
ACTIVITY_WARNINGS: Dict[str, str] = {
    "Working at height":
        "FALL HAZARD: Do not proceed near an unprotected edge or opening. "
        "Stop work until suitable protection is installed and verified.",
    "Scaffolding":
        "SCAFFOLD HAZARD: Do not use an incomplete, damaged, unstable or "
        "inadequately inspected scaffold. Isolate it until corrected.",
    "Excavation":
        "EXCAVATION HAZARD: Keep people and plant away from unsafe edges. "
        "Do not enter an inadequately protected excavation.",
    "Lifting":
        "LIFTING HAZARD: Never stand or work beneath a suspended load. "
        "Maintain exclusion zones at all times.",
    "Electrical work":
        "ELECTRICAL HAZARD: Do not work on or near live electrical equipment "
        "without isolation and verification. Stop work if uncertain.",
    "Hot work":
        "FIRE HAZARD: Do not commence hot work without a valid permit and "
        "fire-prevention controls in place.",
    "Concrete pouring":
        "CONCRETE HAZARD: Ensure formwork stability is verified before pouring. "
        "Control access to the pour area.",
    "Formwork/Falsework":
        "FORMWORK HAZARD: Do not load formwork before structural verification. "
        "Inspect before each use.",
    "Material handling":
        "HANDLING HAZARD: Assess loads before lifting. Use mechanical aids "
        "for excessive loads. Maintain clear access routes.",
    "Housekeeping":
        "SLIP/TRIP HAZARD: Remove debris and maintain clear walkways. "
        "Report and correct hazardous conditions immediately.",
}


class RecommendationEngine:
    """
    Provides controlled safety recommendations after risk prediction.
    Uses Safety Code Reference from the dataset + controlled mappings.
    Does NOT invent recommendations.
    """

    def __init__(self, safety_codes_df: Optional[pd.DataFrame] = None):
        self.safety_codes_df = safety_codes_df
        self._code_lookup = {}
        if safety_codes_df is not None:
            for _, row in safety_codes_df.iterrows():
                activity = str(row.get("Activity", "")).strip()
                self._code_lookup[activity] = {
                    "codes": str(row.get("Applicable IS Codes / Standards", "")),
                    "precautions": str(row.get("Precautions", "")),
                    "warning": str(row.get("Safety Warning", "")),
                }

    def get_precautions(self, activity: str) -> List[str]:
        """Return controlled precautions for the given activity."""
        # First try dataset reference
        if activity in self._code_lookup:
            raw = self._code_lookup[activity].get("precautions", "")
            if raw and raw != "nan":
                return [p.strip() for p in raw.split(";") if p.strip()]
        # Fallback to built-in mapping
        return ACTIVITY_PRECAUTIONS.get(activity, [
            "Conduct site-specific risk assessment",
            "Implement appropriate controls before commencing work",
            "Consult a qualified safety professional",
        ])

    def get_warning(self, activity: str) -> str:
        """Return controlled safety warning for the given activity."""
        if activity in self._code_lookup:
            raw = self._code_lookup[activity].get("warning", "")
            if raw and raw != "nan":
                return raw
        return ACTIVITY_WARNINGS.get(activity,
            "Ensure all hazards are identified and controlled before work proceeds.")

    def get_is_codes(self, activity: str) -> str:
        """Return applicable IS codes for the given activity."""
        if activity in self._code_lookup:
            raw = self._code_lookup[activity].get("codes", "")
            if raw and raw != "nan":
                return raw
        return "Refer to applicable IS standards and NBC 2016."

    def get_toolbox_topic(self, activity: str) -> str:
        """Return the toolbox talk topic for the given activity."""
        return TOOLBOX_TALK_MAPPING.get(activity, f"Safety practices for {activity}")

    def get_full_recommendation(self, activity: str, hazard_categories: List[str] = None) -> dict:
        """
        Return complete recommendation package for a given activity.
        """
        return {
            "activity": activity,
            "precautions": self.get_precautions(activity),
            "warning": self.get_warning(activity),
            "is_codes": self.get_is_codes(activity),
            "toolbox_topic": self.get_toolbox_topic(activity),
            "hazard_categories": hazard_categories or [],
        }
