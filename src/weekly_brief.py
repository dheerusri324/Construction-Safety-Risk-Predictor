"""
weekly_brief.py
Weekly Safety Brief generator.

Primary: Template-based brief from actual calculated data (no fake stats).
Optional: Azure OpenAI to convert structured facts into readable prose.

Azure OpenAI is OPTIONAL. App works fully without it.
API key must be in environment variable — never hard-coded.
"""

import os
from datetime import date, timedelta
from typing import Dict, Any, List

from src.hazard_analysis import get_hazard_frequency, extract_hazard_categories
from src.recommendations import TOOLBOX_TALK_MAPPING
from src.analytics import get_kpi_summary, get_high_risk_activities, get_recurring_patterns


# ─── Data Aggregation for Brief ──────────────────────────────────────────────

def compute_brief_facts(df, hazard_df=None) -> Dict[str, Any]:
    """
    Compute factual data for the weekly brief from the incident dataset.
    Everything calculated from actual data — nothing invented.
    """
    kpis = get_kpi_summary(hazard_df if hazard_df is not None else df)
    high_risk_activities = get_high_risk_activities(df)
    patterns = get_recurring_patterns(df)

    # Hazard frequency
    df_with_hazards = df.copy()
    if "hazard_categories" not in df_with_hazards.columns:
        from src.hazard_analysis import add_hazard_columns
        df_with_hazards = add_hazard_columns(df_with_hazards)

    hazard_freq = get_hazard_frequency(df_with_hazards)
    top_hazards = hazard_freq.head(5)["Hazard Category"].tolist()

    # Priority toolbox topics (top 3 high-risk activities)
    top_high_activities = high_risk_activities.head(3)["Activity"].tolist()
    toolbox_topics = [TOOLBOX_TALK_MAPPING.get(a, f"Safety for {a}") for a in top_high_activities]

    # Severity breakdown
    sev_counts = df["Severity"].value_counts().to_dict()

    return {
        "period_label": f"Full dataset ({len(df)} incidents)",
        "total_incidents": len(df),
        "high_count": sev_counts.get("High", 0),
        "medium_count": sev_counts.get("Medium", 0),
        "low_count": sev_counts.get("Low", 0),
        "high_pct": round(100 * sev_counts.get("High", 0) / len(df), 1) if len(df) > 0 else 0,
        "medium_pct": round(100 * sev_counts.get("Medium", 0) / len(df), 1) if len(df) > 0 else 0,
        "low_pct": round(100 * sev_counts.get("Low", 0) / len(df), 1) if len(df) > 0 else 0,
        "top_high_risk_activities": high_risk_activities.head(5).to_dict("records"),
        "top_hazard_categories": top_hazards,
        "top_toolbox_topics": toolbox_topics,
        "top_activities": list(patterns["top_activities"].keys())[:5],
        "time_band_distribution": patterns["time_band_distribution"],
    }


# ─── Template-Based Brief (Fallback / Primary) ───────────────────────────────

def generate_template_brief(facts: Dict[str, Any]) -> str:
    """
    Generate a structured weekly safety brief from calculated facts.
    This is the primary/fallback method — no LLM required.
    All statements are derived from actual data.
    """
    top_high = facts.get("top_high_risk_activities", [])
    top_high_str = "\n".join(
        f"  • {row['Activity']} ({row['High Risk Count']} High-risk incidents)"
        for row in top_high[:5]
    ) if top_high else "  • No high-risk activity data available."

    top_hazards = facts.get("top_hazard_categories", [])
    hazards_str = "\n".join(f"  • {h}" for h in top_hazards) if top_hazards else "  • None identified."

    toolbox_str = "\n".join(
        f"  • {t}" for t in facts.get("top_toolbox_topics", [])
    ) if facts.get("top_toolbox_topics") else "  • Conduct general site safety awareness."

    time_bands = facts.get("time_band_distribution", {})
    peak_band = max(time_bands, key=time_bands.get) if time_bands else "Not available"

    brief = f"""
══════════════════════════════════════════════════════
           WEEKLY SAFETY BRIEF — CONSTRUCTION SITE
══════════════════════════════════════════════════════
Generated: {date.today().strftime('%d %B %Y')}
Data period: {facts.get('period_label', 'N/A')}
NOTE: Based on synthetic academic data for demonstration purposes.

────────────────────────────────────────────────────
INCIDENT SUMMARY
────────────────────────────────────────────────────
Total incidents observed:  {facts['total_incidents']}
  🔴 High severity:   {facts['high_count']}  ({facts['high_pct']}%)
  🟠 Medium severity: {facts['medium_count']}  ({facts['medium_pct']}%)
  🟡 Low severity:    {facts['low_count']}  ({facts['low_pct']}%)

────────────────────────────────────────────────────
KEY OBSERVATIONS
────────────────────────────────────────────────────
• Incidents span {len(facts.get('top_activities', []))} primary construction activities.
• Peak incident period: {peak_band}
• High-risk incidents are distributed across multiple activity types,
  indicating the need for broad-based hazard controls.

────────────────────────────────────────────────────
HIGH-RISK ACTIVITIES (Priority Focus)
────────────────────────────────────────────────────
{top_high_str}

────────────────────────────────────────────────────
RECURRING HAZARD INDICATORS
────────────────────────────────────────────────────
{hazards_str}

────────────────────────────────────────────────────
PRIORITY TOOLBOX TALK TOPICS
────────────────────────────────────────────────────
{toolbox_str}

────────────────────────────────────────────────────
PREVENTIVE FOCUS AREAS
────────────────────────────────────────────────────
  • Ensure edge protection and fall prevention controls are in place
    before commencing any elevated work.
  • Verify excavation support and exclusion zones before work starts.
  • Confirm lifting equipment inspection and rigging arrangements.
  • Inspect scaffolding before each use and after adverse conditions.
  • Maintain site housekeeping to prevent slip/trip hazards.

────────────────────────────────────────────────────
DISCLAIMER
────────────────────────────────────────────────────
This brief is generated from synthetic academic data for demonstration
and evaluation. It does not replace site-specific risk assessments,
qualified safety professionals, or applicable safety procedures.

Individual incidents must be assessed by a competent person.
Focus is on hazards and controls — not on individual blame.
══════════════════════════════════════════════════════
"""
    return brief.strip()


# ─── Azure OpenAI (Optional) ─────────────────────────────────────────────────

def try_azure_openai_brief(facts: Dict[str, Any]) -> str:
    """
    Attempt to generate a more readable brief using Azure OpenAI.

    Uses environment variables for credentials — never hard-coded.
    Returns None if Azure OpenAI is unavailable.

    The LLM receives ONLY structured factual data — not asked to invent
    incident counts, percentages, or standards.
    """
    try:
        from openai import AzureOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_KEY", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint or not api_key:
            return None

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        top_high = facts.get("top_high_risk_activities", [])
        top_high_str = ", ".join(r["Activity"] for r in top_high[:3])

        structured_data = f"""
Incident summary:
- Total incidents: {facts['total_incidents']}
- High: {facts['high_count']} ({facts['high_pct']}%)
- Medium: {facts['medium_count']} ({facts['medium_pct']}%)
- Low: {facts['low_count']} ({facts['low_pct']}%)

Top high-risk activities: {top_high_str}
Top hazard categories: {', '.join(facts.get('top_hazard_categories', []))}
Priority toolbox topics: {', '.join(facts.get('top_toolbox_topics', []))}
Data note: This is SYNTHETIC ACADEMIC data for a student project.
"""

        system_msg = (
            "You are a construction safety advisor writing a concise weekly safety brief. "
            "Use ONLY the structured facts provided. Do NOT invent statistics, percentages, "
            "incident counts, or standards not in the data. "
            "Never blame workers. Focus on hazards and controls. "
            "Keep it professional, clear, and under 400 words. "
            "Include a disclaimer that data is synthetic academic data."
        )

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Generate a weekly safety brief from:\n{structured_data}"},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return None


def generate_weekly_brief(df, use_azure: bool = True) -> Dict[str, Any]:
    """
    Generate the weekly safety brief.
    Returns template brief + optional Azure OpenAI brief.
    """
    from src.hazard_analysis import add_hazard_columns
    df_with_hazards = add_hazard_columns(df)

    facts = compute_brief_facts(df, df_with_hazards)
    template_brief = generate_template_brief(facts)

    ai_brief = None
    ai_available = False
    if use_azure:
        ai_brief = try_azure_openai_brief(facts)
        ai_available = ai_brief is not None

    return {
        "facts": facts,
        "template_brief": template_brief,
        "ai_brief": ai_brief,
        "ai_available": ai_available,
    }
