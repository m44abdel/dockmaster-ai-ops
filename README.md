# DockMaster AI Ops (case study MVP v2)

**Predictive maintenance** (UCI AI4I 2020 proxy + calibrated Random Forest) plus **constraint-aware scheduling** (Google OR-Tools CP-SAT): skills, bay types, parts ETA slots, promised due dates, customer tier late penalties, technician shifts, and secondary labor cost.

## Quick start

```bash
cd dockmaster-ai-ops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads AI4I and trains `models/failure_risk.joblib`. Training uses **SMOTENC** (partial upsampling of the failure class inside each CV fold), **`class_weight='balanced_subsample'`**, and **isotonic calibration**. In `dockmaster_ai_ops/config.py` you can tune:

- **`SMOTE_TARGET_RATIO_TO_MAJORITY`** (try **0.08–0.14**; **~0.10** is a reasonable default on AI4I).
- **`RISK_MODEL_BACKEND`**: **`lgbm`** (LightGBM, often better PR-AUC on this tabular proxy) or **`rf`** (RandomForest).

Use **Retrain risk model** after changes.

## What’s in v2

- Marina-style synthetic work orders (vessel type, engine, location, storage, service category, seasonality).
- Technician roster with skills, `bay_type`, shifts, hourly cost; optional **dual-skill** jobs (e.g. engine + electrical).
- **Explainability:** percentile-based “why prioritized?” text plus financial exposure (`failure_risk ×` configurable USD).
- Scheduler objective: **priority** (urgency + $ exposure) + **SLA lateness** vs promised slot + labor cost; **parts ETA** as earliest start.
- Baselines: FCFS and promised-date order vs optimized schedule; ROI + utilization / SLA KPIs.
- Scenario simulator: absent techs, parts delays, +30% demand.
- CSV export of scored + scheduled work orders.

## Stack

Python 3.10+, scikit-learn, OR-Tools, Streamlit, Plotly, Gemini.