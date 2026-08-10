# Construction Safety Risk Predictor

> **Academic MVP · Proof of Concept · Synthetic Data**

A machine-learning–powered safety risk prediction system for construction sites.  
Built with Python, scikit-learn, Pandas, Streamlit, and Plotly.

---

## 1. Problem Statement

Construction incidents repeat due to recurring unsafe conditions and high-risk activities (working at height, lifting, scaffolding, excavation). Safety reports contain text and inconsistent tags, limiting learning.

This system:
- Classifies incident risk by activity type and context
- Highlights top contributing factors and recurring patterns
- Recommends preventive actions and toolbox talk topics
- Produces a dashboard and weekly safety brief

---

## 2. Objectives

1. Multi-class severity classification (Low / Medium / High)
2. NLP-based risk indicator identification from incident descriptions
3. Recurring hazard pattern analysis from historical incidents
4. Controlled, evidence-based preventive recommendations
5. Planned activity risk prioritisation
6. Weekly safety brief generation

---

## 3. Dataset Description

| Property | Detail |
|---|---|
| File | `Construction_Safety_Risk_Predictor_501_Final.xlsx` |
| Type | **Synthetic academic data** (not real-world incidents) |
| Records | 501 incidents |
| Severity | Balanced: 167 High / 167 Medium / 167 Low |
| Sheets | Incident Reports, Risk Summary, Activity vs Risk, Safety Code Reference, Data Dictionary |

### Incident Reports Columns

| Column | Role |
|---|---|
| Incident ID | Identifier only (excluded from ML) |
| Activity | ML Feature |
| Location Type | ML Feature |
| Time | ML Feature |
| Description | ML Feature (NLP) |
| **Severity** | **Target** |
| Applicable IS Codes | Reference only (excluded from ML — leakage) |
| Recommended Precautions | Reference only (excluded from ML — leakage) |
| Safety Warning | Reference only (excluded from ML — leakage) |

---

## 4. ML Task

**Multi-class classification** — predict Severity (Low / Medium / High) from activity context and incident description.

---

## 5. NLP Method

Combined text representation:

```
Activity: {activity}
Location: {location_type}
Time: {time}
{description}
```

Then:

```
TF-IDF (unigram + bigram, max 5000 features, sublinear TF, English stopwords)
    ↓
ML Classifier (Logistic Regression or Linear SVM)
```

All vectorisation happens **inside** a scikit-learn `Pipeline` — TF-IDF is fitted only on training folds during cross-validation (no data leakage).

---

## 6. Models Evaluated

| Model | Notes |
|---|---|
| Logistic Regression | Interpretable, calibrated probabilities |
| Linear SVM (calibrated) | Efficient for TF-IDF space, wrapped with `CalibratedClassifierCV` |

Best model selected by highest **CV Macro F1** from actual evaluation.

---

## 7. Evaluation Methodology

- **5-fold Stratified Cross-Validation** (primary evaluation)
- `random_state=42` for reproducibility
- TF-IDF fitted inside pipeline per fold — no leakage

---

## 8. Metrics

- Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Per-class: Precision, Recall, F1, Support (Low / Medium / High)
- Confusion Matrix
- High-Risk Recall highlighted (missed High-risk = false negative, safety concern)

> All metrics come from actual model evaluation — none are hard-coded.

---

## 9. Architecture

```
Excel Dataset (frozen, read-only)
        ↓
Pandas Data Layer (data_loader.py)
        ↓
NLP + Feature Engineering (preprocessing.py)
        ↓
TF-IDF Pipeline → ML Classifier (train.py)
        ↓
Prediction (prediction.py)
        ↓
Hazard Analysis (hazard_analysis.py)
        ↓
Recommendation Engine (recommendations.py)
        ↓
Analytics (analytics.py)
        ↓
Streamlit UI (app.py)
        ↓
Optional: Azure OpenAI → Weekly Brief (weekly_brief.py)
```

---

## 10. Streamlit Pages

| Page | Description |
|---|---|
| 📊 Dashboard | KPI cards, severity distribution, activity analysis, hazard patterns, heatmaps |
| 🔮 Risk Predictor | Enter activity details → predict severity + probabilities + indicators + recommendations |
| 📅 Planned Activities | Predict and prioritise risk for upcoming site activities |
| 📈 Model Performance | CV metrics, per-class metrics, confusion matrix, model comparison |
| 📝 Weekly Safety Brief | Template-based brief from calculated data (+ optional Azure OpenAI) |

---

## 11. Recommendation Mechanism

Recommendations are derived from:
- Safety Code Reference sheet (Applicable IS Codes, Precautions, Safety Warnings)
- Controlled activity-to-precaution mappings
- Controlled toolbox talk topic mappings

The ML classifier does **not** invent recommendations.  
Recommendations are standard safe practices only.

---

## 12. Planned Activities

A synthetic planned-activity dataset (not derived from the 501 historical records) is provided.  
The same ML pipeline predicts risk for each planned activity.  
Output is sorted by priority (High → Medium → Low).

---

## 13. Weekly Safety Brief

Generated from calculated data facts:
- Incident counts and severity distribution
- High-risk activities
- Recurring hazard categories
- Priority toolbox topics
- Preventive focus areas

**No statistics are invented.** Template-based fallback always works.  
Optional Azure OpenAI prose enhancement when configured.

---

## 14. Azure OpenAI (Optional)

Configure via environment variables:

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

Used **only** for converting structured brief facts into readable prose.  
App works fully without Azure OpenAI.

---

## 15. Privacy

- No worker-level profiling
- No worker names, IDs, or risk scores
- Focus: activities, locations, hazards, controls

---

## 16. Limitations

- Dataset is **synthetic** — not real construction incident data
- Small dataset (501 records) — model may not generalise
- Balanced classes are intentional — real distributions may differ
- Model performance on synthetic data does not guarantee real-world performance
- Safety predictions are **decision-support only**
- Professional safety judgment, site-specific assessment, and applicable procedures remain necessary
- No-blame focus: all observations are about hazards and controls, not individuals

---

## 17. How to Install

```bash
cd construction_safety_predictor
pip install -r requirements.txt
```

---

## 18. How to Run

**Step 1: Train the model (first time only)**

```bash
cd construction_safety_predictor
python src/train.py
```

**Step 2: Start the app**

```bash
streamlit run app.py
```

---

## 19. How to Retrain

If the dataset is updated:

```bash
cd construction_safety_predictor
python src/train.py
```

Then restart the Streamlit app.

---

## 20. Disclaimer

This project uses synthetic academic data for demonstration and evaluation.  
Predictions are intended for safety-risk prioritisation and do not replace qualified safety professionals, site-specific risk assessments, or applicable safety procedures.
