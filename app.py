"""
app.py — Construction Safety Risk Predictor
Streamlit application entry point.

Pages:
  1. 📊 Dashboard
  2. 🔮 Risk Predictor
  3. 📅 Planned Activities
  4. 📈 Model Performance
  5. 📝 Weekly Safety Brief

Run:  streamlit run app.py
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from src.data_loader import load_incidents, load_safety_codes, validate_incidents
from src.hazard_analysis import (
    add_hazard_columns, get_hazard_frequency,
    extract_hazard_categories, extract_risk_indicators,
)
from src.analytics import (
    get_kpi_summary, get_severity_distribution,
    get_activity_severity_matrix, get_location_severity_matrix,
    get_time_distribution, get_high_risk_activities, get_recurring_patterns,
)
from src.recommendations import RecommendationEngine
from src.prediction import load_model, predict_severity, get_top_model_indicators
from src.planned_activities import predict_planned_activities, get_planned_activities_df, SEVERITY_EMOJI
from src.weekly_brief import generate_weekly_brief

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(APP_DIR, "models", "risk_model.pkl")
EVAL_PATH = os.path.join(APP_DIR, "models", "evaluation_results.json")

SEVERITY_COLORS = {"High": "#EF4444", "Medium": "#F97316", "Low": "#22C55E"}
SEVERITY_BG = {"High": "#FEF2F2", "Medium": "#FFF7ED", "Low": "#F0FDF4"}

DISCLAIMER = (
    "⚠️ **Disclaimer:** This application uses **synthetic academic data** for demonstration "
    "and evaluation. Predictions are intended for safety-risk prioritisation and do not replace "
    "qualified safety professionals, site-specific risk assessments, or applicable safety procedures."
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Safety Risk Predictor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1E3A5F;
        border-bottom: 3px solid #E63946; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .kpi-card {
        background: #F8FAFC; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 5px solid #1E3A5F; margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .kpi-card-high { border-left-color: #EF4444; background: #FEF2F2; }
    .kpi-card-medium { border-left-color: #F97316; background: #FFF7ED; }
    .kpi-card-low { border-left-color: #22C55E; background: #F0FDF4; }
    .kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #6B7280; margin: 0; }
    .risk-badge-high {
        background: #EF4444; color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    .risk-badge-medium {
        background: #F97316; color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    .risk-badge-low {
        background: #22C55E; color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #1E3A5F;
        border-bottom: 1px solid #E5E7EB; padding-bottom: 0.3rem; margin: 1rem 0 0.5rem;
    }
    .indicator-chip {
        display: inline-block; background: #EFF6FF; color: #1D4ED8;
        border: 1px solid #BFDBFE; border-radius: 12px;
        padding: 0.2rem 0.6rem; font-size: 0.8rem; margin: 2px;
    }
    .warning-box {
        background: #FEF3C7; border: 1px solid #F59E0B;
        border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem;
        color: #92400E; font-size: 0.9rem;
    }
    .brief-box {
        background: #F8FAFC; border: 1px solid #E5E7EB;
        border-radius: 8px; padding: 1rem; font-family: monospace;
        font-size: 0.82rem; white-space: pre-wrap; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df = load_incidents()
    df = add_hazard_columns(df)
    safety_codes = load_safety_codes()
    return df, safety_codes


@st.cache_resource(show_spinner=False)
def load_ml_model():
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_eval_results():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r") as f:
            return json.load(f)
    return None


# ── Sidebar Navigation ─────────────────────────────────────────────────────────

st.sidebar.markdown("## 🦺 Construction Safety\nRisk Predictor")
st.sidebar.markdown("---")

pages = {
    "📊 Dashboard": "dashboard",
    "🔮 Risk Predictor": "predictor",
    "📅 Planned Activities": "planned",
    "📈 Model Performance": "performance",
    "📝 Weekly Safety Brief": "brief",
}
selected_page = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
page = pages[selected_page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Academic MVP · Synthetic Data · No worker profiling</small>",
    unsafe_allow_html=True,
)

# ── Load Data ──────────────────────────────────────────────────────────────────
try:
    df, safety_codes_df = load_data()
    data_ok = True
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

model = load_ml_model()
eval_results = load_eval_results()
rec_engine = RecommendationEngine(safety_codes_df)

activities = sorted(df["Activity"].unique().tolist())
locations = sorted(df["Location Type"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "dashboard":
    st.markdown('<div class="main-title">📊 Construction Safety Dashboard</div>', unsafe_allow_html=True)
    st.markdown(DISCLAIMER)
    st.markdown("")

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔎 Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_activity = st.multiselect("Activity", ["All"] + activities, default=["All"])
        with col2:
            sel_severity = st.multiselect("Severity", ["All", "High", "Medium", "Low"], default=["All"])
        with col3:
            sel_location = st.multiselect("Location Type", ["All"] + locations, default=["All"])

    filtered = df.copy()
    if sel_activity and "All" not in sel_activity:
        filtered = filtered[filtered["Activity"].isin(sel_activity)]
    if sel_severity and "All" not in sel_severity:
        filtered = filtered[filtered["Severity"].isin(sel_severity)]
    if sel_location and "All" not in sel_location:
        filtered = filtered[filtered["Location Type"].isin(sel_location)]

    if filtered.empty:
        st.warning("No incidents match the selected filters.")
        st.stop()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    kpis = get_kpi_summary(filtered)
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{kpis["total_incidents"]}</p><p class="kpi-label">Total Incidents</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card kpi-card-high"><p class="kpi-value" style="color:#EF4444">{kpis["high_count"]}</p><p class="kpi-label">🔴 High Severity</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card kpi-card-medium"><p class="kpi-value" style="color:#F97316">{kpis["medium_count"]}</p><p class="kpi-label">🟠 Medium Severity</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card kpi-card-low"><p class="kpi-value" style="color:#22C55E">{kpis["low_count"]}</p><p class="kpi-label">🟡 Low Severity</p></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value" style="font-size:1rem">{kpis["most_frequent_high_activity"]}</p><p class="kpi-label">⚠️ Top High-Risk Activity</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Severity Distribution + Activity Severity ─────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        sev_dist = get_severity_distribution(filtered)
        fig_sev = px.pie(
            sev_dist, names="Severity", values="Count",
            title="Severity Distribution",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            hole=0.4,
        )
        fig_sev.update_traces(textposition="inside", textinfo="percent+label")
        fig_sev.update_layout(showlegend=True, height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_b:
        act_sev = get_activity_severity_matrix(filtered)
        fig_act = px.bar(
            act_sev, x="Activity", y=["High", "Medium", "Low"],
            title="Incidents by Activity and Severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
        )
        fig_act.update_layout(height=350, xaxis_tickangle=-30, margin=dict(t=50, b=10))
        st.plotly_chart(fig_act, use_container_width=True)

    # ── Row 2: High-Risk Activities + Hazard Categories ───────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        high_acts = get_high_risk_activities(filtered)
        fig_high = px.bar(
            high_acts, x="High Risk Count", y="Activity",
            title="High-Risk Incidents by Activity",
            orientation="h",
            color="High Risk Count",
            color_continuous_scale=["#FEE2E2", "#EF4444"],
        )
        fig_high.update_layout(height=350, showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_high, use_container_width=True)

    with col_d:
        if "primary_hazard" in filtered.columns:
            hazard_freq = get_hazard_frequency(filtered)
            fig_haz = px.bar(
                hazard_freq.head(10), x="Count", y="Hazard Category",
                title="Recurring Hazard Indicators",
                orientation="h",
                color="Count",
                color_continuous_scale=["#FEF3C7", "#D97706"],
            )
            fig_haz.update_layout(height=350, showlegend=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig_haz, use_container_width=True)

    # ── Row 3: Activity × Severity Heatmap + Location Severity ──────────────
    col_e, col_f = st.columns(2)

    with col_e:
        act_sev_pivot = filtered.pivot_table(
            index="Activity", columns="Severity",
            values="Incident ID", aggfunc="count", fill_value=0
        )
        for sev in ["High", "Medium", "Low"]:
            if sev not in act_sev_pivot.columns:
                act_sev_pivot[sev] = 0
        act_sev_pivot = act_sev_pivot[["High", "Medium", "Low"]]

        fig_heat = px.imshow(
            act_sev_pivot,
            title="Activity × Severity Heatmap",
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            text_auto=True,
        )
        fig_heat.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_f:
        time_dist = get_time_distribution(filtered)
        if not time_dist.empty:
            fig_time = px.bar(
                time_dist, x="hour", y=["High", "Medium", "Low"],
                title="Incidents by Time of Day",
                color_discrete_map=SEVERITY_COLORS,
                barmode="stack",
                labels={"hour": "Hour of Day", "value": "Incidents"},
            )
            fig_time.update_layout(height=380, margin=dict(t=50, b=10))
            st.plotly_chart(fig_time, use_container_width=True)

    # ── Activity Risk Analysis Table ──────────────────────────────────────────
    st.markdown('<div class="section-header">Activity Risk Analysis (from Incident Reports)</div>', unsafe_allow_html=True)
    act_table = get_activity_severity_matrix(filtered)
    act_table["Total"] = act_table["High"] + act_table["Medium"] + act_table["Low"]
    act_table = act_table.sort_values("High", ascending=False).reset_index(drop=True)
    st.dataframe(
        act_table[["Activity", "High", "Medium", "Low", "Total"]],
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "predictor":
    st.markdown('<div class="main-title">🔮 Risk Predictor</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            "⚠️ Model not found. Please train the model first:\n\n"
            "```\ncd construction_safety_predictor\npython src/train.py\n```"
        )
        st.stop()

    st.markdown(DISCLAIMER)
    st.markdown("Enter incident or activity details to predict risk severity.")
    st.markdown("---")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-header">📋 Incident / Activity Details</div>', unsafe_allow_html=True)

        activity = st.selectbox("Activity *", activities, index=0)

        location_options = ["— Enter custom —"] + locations
        loc_select = st.selectbox("Location Type", location_options)
        if loc_select == "— Enter custom —":
            location = st.text_input("Custom location", placeholder="e.g., basement level")
        else:
            location = loc_select

        time_input = st.time_input("Time of Activity", value=None)
        time_str = time_input.strftime("%H:%M") if time_input else "09:00"

        description = st.text_area(
            "Incident / Activity Description *",
            height=150,
            placeholder=(
                "Describe the observed conditions or planned activity in detail.\n"
                "e.g.: A work area near an elevated slab edge has incomplete edge protection "
                "and workers are moving materials through the area."
            ),
        )

        predict_btn = st.button("🔍 PREDICT RISK", type="primary", use_container_width=True)

    with col_output:
        st.markdown('<div class="section-header">⚠️ Risk Assessment</div>', unsafe_allow_html=True)

        if predict_btn:
            if not description.strip():
                st.warning("Please enter an incident description.")
            else:
                with st.spinner("Analysing risk..."):
                    result = predict_severity(model, activity, location, time_str, description)

                sev = result["predicted_severity"]
                probs = result["probabilities"]

                # ── Severity Badge ─────────────────────────────────────────────
                badge_class = f"risk-badge-{sev.lower()}"
                emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(sev, "⚪")
                st.markdown(
                    f'<div style="text-align:center; margin: 1rem 0;">'
                    f'<span class="{badge_class}">{emoji} {sev.upper()}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Probability bars ──────────────────────────────────────────
                if probs:
                    st.markdown("**Model-estimated class probabilities:**")
                    st.caption("*(These are model estimates, not guaranteed probabilities)*")
                    for cls in ["High", "Medium", "Low"]:
                        p = probs.get(cls, 0.0)
                        color = SEVERITY_COLORS.get(cls, "#gray")
                        st.markdown(
                            f'<div style="display:flex; align-items:center; margin:4px 0;">'
                            f'<span style="width:70px;font-weight:600">{cls}</span>'
                            f'<div style="flex:1; background:#E5E7EB; border-radius:6px; height:18px; margin:0 8px;">'
                            f'<div style="width:{p*100:.1f}%; background:{color}; height:100%; border-radius:6px;"></div></div>'
                            f'<span style="width:45px; text-align:right">{p*100:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("")

                # ── Risk Indicators ───────────────────────────────────────────
                indicators = extract_risk_indicators(description)
                if indicators:
                    st.markdown("**Detected risk indicators:**")
                    st.caption("*(Statistical indicators from description text — not proven causes)*")
                    chips = " ".join(f'<span class="indicator-chip">{i}</span>' for i in indicators)
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown("")

                # ── Hazard Categories ─────────────────────────────────────────
                hazards = extract_hazard_categories(description)
                st.markdown(f"**Detected hazard categories:** {', '.join(hazards)}")
                st.markdown("")

                # ── Recommendations ───────────────────────────────────────────
                rec = rec_engine.get_full_recommendation(activity, hazards)

                st.markdown("**Recommended preventive actions:**")
                for p in rec["precautions"]:
                    st.markdown(f"  • {p}")

                st.markdown(f"**Toolbox talk topic:** {rec['toolbox_topic']}")
                st.markdown(f"**IS Codes / Standards:** {rec['is_codes']}")

                # ── Safety Warning ────────────────────────────────────────────
                st.markdown(
                    f'<div class="warning-box">⚠️ <strong>Safety Warning:</strong><br>{rec["warning"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Fill in the details on the left and click **PREDICT RISK**.")

    # ── Model Indicators (top features) ───────────────────────────────────────
    if predict_btn and model is not None:
        st.markdown("---")
        with st.expander("📊 Top model indicators for HIGH risk (statistical, not causal)"):
            top_features = get_top_model_indicators(model, class_name="High", top_n=15)
            if top_features:
                st.caption(
                    "These are the TF-IDF terms with the highest model coefficient for 'High' risk. "
                    "They are **important model indicators** — not proven causes of incidents."
                )
                cols = st.columns(3)
                for i, feat in enumerate(top_features):
                    cols[i % 3].markdown(f"• `{feat}`")
            else:
                st.info("Feature importance not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PLANNED ACTIVITIES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "planned":
    st.markdown('<div class="main-title">📅 Planned Activity Risk Prioritisation</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            "⚠️ Model not found. Please train the model first:\n\n"
            "```\ncd construction_safety_predictor\npython src/train.py\n```"
        )
        st.stop()

    st.markdown(DISCLAIMER)
    st.info(
        "This is a **prioritisation aid** based on predicted risk levels. "
        "It does NOT replace a qualified site risk assessment or professional safety judgment."
    )
    st.markdown("---")

    planned_df = get_planned_activities_df()

    with st.spinner("Predicting risk for planned activities..."):
        result_df = predict_planned_activities(model, planned_df)

    # ── Priority Summary ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Safety Priorities</div>', unsafe_allow_html=True)

    dates = sorted(result_df["Date"].unique())
    for d in dates:
        st.markdown(f"### 📆 {d}")
        day_df = result_df[result_df["Date"] == d]

        for _, row in day_df.iterrows():
            sev = row["Predicted Severity"]
            emoji = SEVERITY_EMOJI.get(sev, "⚪")
            color = SEVERITY_COLORS.get(sev, "#gray")
            bg = SEVERITY_BG.get(sev, "#F8FAFC")

            with st.container():
                st.markdown(
                    f'<div style="background:{bg}; border-left:4px solid {color}; '
                    f'border-radius:6px; padding:0.8rem 1rem; margin:0.4rem 0;">'
                    f'<strong>{emoji} {sev.upper()}</strong> — {row["Activity"]} @ {row["Location Type"]} ({row["Time"]})<br>'
                    f'<small style="color:#6B7280">{row["Planned Description"]}</small>',
                    unsafe_allow_html=True,
                )

                if row.get("High Prob") is not None:
                    hi = row["High Prob"] * 100
                    me = row["Medium Prob"] * 100
                    lo = row["Low Prob"] * 100
                    st.markdown(
                        f'<small>Model-estimated: 🔴 High {hi:.0f}% | 🟠 Medium {me:.0f}% | 🟡 Low {lo:.0f}%</small>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")

    # ── Full Results Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Full Predicted Activity Table</div>', unsafe_allow_html=True)
    display_cols = ["Date", "Time", "Activity", "Location Type", "Predicted Severity", "Planned Description"]
    st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

    # ── CSV Export ─────────────────────────────────────────────────────────────
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="planned_activity_risk.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "performance":
    st.markdown('<div class="main-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.caption("Metrics are calculated from synthetic academic data and should not be interpreted as production safety-system performance.")

    if eval_results is None:
        st.error(
            "⚠️ Evaluation results not found. Please train the model first:\n\n"
            "```\ncd construction_safety_predictor\npython src/train.py\n```"
        )
        st.stop()

    cv = eval_results["cv_results"][eval_results["best_model_name"]]
    full = eval_results["full_dataset_report"]
    ds_info = eval_results["dataset_info"]

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Selected model:** {eval_results['best_model_name']}")
        st.markdown(f"**Cross-validation:** {cv['cv_folds']}-fold Stratified CV")
        st.markdown(f"**Training records:** {ds_info['total_records']}")
        st.markdown(f"**Random state:** {ds_info['random_state']}")
    with col2:
        st.markdown(f"**Features used:** Activity · Location Type · Time · Description")
        st.markdown(f"**Vectorisation:** TF-IDF (unigrams + bigrams, max 5000 features)")
        st.markdown(f"**No leakage:** Severity / IS Codes / Precautions / Warnings excluded")

    # ── CV Metrics ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Cross-Validation Metrics (Primary Evaluation)</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CV Accuracy", f"{cv['accuracy_mean']:.3f}", f"±{cv['accuracy_std']:.3f}")
    m2.metric("Macro Precision", f"{cv['macro_precision_mean']:.3f}")
    m3.metric("Macro Recall", f"{cv['macro_recall_mean']:.3f}")
    m4.metric("Macro F1", f"{cv['macro_f1_mean']:.3f}", f"±{cv['macro_f1_std']:.3f}")
    m5.metric("Weighted F1", f"{cv['weighted_f1_mean']:.3f}")

    # Per-fold accuracy bar chart
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(cv["per_fold_accuracy"]))],
        "Accuracy": cv["per_fold_accuracy"],
        "Macro F1": cv["per_fold_macro_f1"],
    })
    fig_fold = px.bar(
        fold_df.melt(id_vars="Fold", var_name="Metric", value_name="Score"),
        x="Fold", y="Score", color="Metric", barmode="group",
        title="Per-Fold CV Scores",
        color_discrete_sequence=["#3B82F6", "#10B981"],
    )
    fig_fold.update_layout(height=300, yaxis_range=[0, 1], margin=dict(t=40, b=10))
    st.plotly_chart(fig_fold, use_container_width=True)

    # ── Per-Class Metrics ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Per-Class Metrics (Full Dataset)</div>', unsafe_allow_html=True)
    st.caption("These are in-sample full-dataset metrics shown for reference. Cross-validation metrics above are the primary evaluation.")

    per_class = full["per_class"]
    class_rows = []
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        class_rows.append({
            "Class": cls,
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f}",
            "F1": f"{m['f1']:.3f}",
            "Support": m["support"],
        })
    st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    # ── High-Risk Recall Callout ───────────────────────────────────────────────
    if "High" in per_class:
        high_recall = per_class["High"]["recall"]
        st.info(
            f"🔴 **High-Risk Recall: {high_recall:.1%}** — "
            f"This measures how many genuinely High-risk cases the model successfully identified. "
            f"In safety-critical contexts, false negatives (missed High-risk predictions) are "
            f"more concerning than false positives. **This is an academic decision-support MVP.**"
        )

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Confusion Matrix (Full Dataset)</div>', unsafe_allow_html=True)
    cm = np.array(full["confusion_matrix"])
    labels = full["confusion_matrix_labels"]

    fig_cm = px.imshow(
        cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig_cm.update_layout(height=400, margin=dict(t=50, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Model Comparison ──────────────────────────────────────────────────────
    if len(eval_results["cv_results"]) > 1:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        comparison_rows = []
        for name, res in eval_results["cv_results"].items():
            comparison_rows.append({
                "Model": name,
                "CV Accuracy": f"{res['accuracy_mean']:.3f} ±{res['accuracy_std']:.3f}",
                "Macro F1": f"{res['macro_f1_mean']:.3f} ±{res['macro_f1_std']:.3f}",
                "Weighted F1": f"{res['weighted_f1_mean']:.3f}",
                "Selected": "✅" if name == eval_results["best_model_name"] else "",
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        st.caption(f"Best model selected by highest CV Macro F1: **{eval_results['best_model_name']}**")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: WEEKLY SAFETY BRIEF
# ══════════════════════════════════════════════════════════════════════════════

elif page == "brief":
    st.markdown('<div class="main-title">📝 Weekly Safety Brief</div>', unsafe_allow_html=True)
    st.markdown(DISCLAIMER)
    st.markdown("")

    st.info(
        "The brief is generated from **actual calculated data** from the incident dataset. "
        "No statistics are invented or hard-coded."
    )

    azure_available = bool(
        os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_KEY")
    )

    col_opts, _ = st.columns([1, 2])
    with col_opts:
        use_ai = st.checkbox(
            "Use Azure OpenAI (if configured)",
            value=azure_available,
            disabled=not azure_available,
        )
        if not azure_available:
            st.caption("Azure OpenAI not configured. Using template-based brief.")

    if st.button("📄 Generate Weekly Safety Brief", type="primary"):
        with st.spinner("Generating brief from incident data..."):
            brief_result = generate_weekly_brief(df, use_azure=use_ai)

        # ── Facts Summary ─────────────────────────────────────────────────────
        facts = brief_result["facts"]
        st.markdown("---")
        st.markdown('<div class="section-header">Data Summary Used in Brief</div>', unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Total Incidents", facts["total_incidents"])
        fc2.metric("High", f"{facts['high_count']} ({facts['high_pct']}%)")
        fc3.metric("Medium", f"{facts['medium_count']} ({facts['medium_pct']}%)")
        fc4.metric("Low", f"{facts['low_count']} ({facts['low_pct']}%)")

        st.markdown("---")

        # ── Brief Display ─────────────────────────────────────────────────────
        if brief_result["ai_available"] and brief_result["ai_brief"]:
            tab1, tab2 = st.tabs(["🤖 AI-Enhanced Brief (Azure OpenAI)", "📄 Template Brief"])
            with tab1:
                st.markdown(brief_result["ai_brief"])
            with tab2:
                st.markdown(
                    f'<div class="brief-box">{brief_result["template_brief"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            if use_ai and not brief_result["ai_available"]:
                st.warning("Azure OpenAI unavailable. Showing template brief.")
            st.markdown('<div class="section-header">Weekly Safety Brief</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="brief-box">{brief_result["template_brief"]}</div>',
                unsafe_allow_html=True,
            )

        # ── Download ──────────────────────────────────────────────────────────
        brief_text = brief_result.get("ai_brief") or brief_result["template_brief"]
        st.download_button(
            "⬇️ Download Brief as Text",
            data=brief_text.encode("utf-8"),
            file_name="weekly_safety_brief.txt",
            mime="text/plain",
        )

        # ── Toolbox Topics ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Priority Toolbox Talk Topics</div>', unsafe_allow_html=True)
        for topic in facts.get("top_toolbox_topics", []):
            st.markdown(f"  📌 {topic}")
