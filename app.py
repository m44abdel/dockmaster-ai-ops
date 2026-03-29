"""
DockMaster AI Ops — v2 prototype (case study).

Run: streamlit run app.py
"""

from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dockmaster_ai_ops.assistant import ensure_dotenv_loaded

ensure_dotenv_loaded(ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dockmaster_ai_ops.baselines import run_baseline
from dockmaster_ai_ops.config import FEATURE_DISPLAY_NAMES
from dockmaster_ai_ops.maintenance_model import load_model, train_failure_model
from dockmaster_ai_ops.config import ESTIMATED_FAILURE_COST_USD
from dockmaster_ai_ops.schedule_metrics import (
    compare_to_baseline,
    extended_business_kpis,
)
from dockmaster_ai_ops.scenarios import apply_scenario
from dockmaster_ai_ops.scheduler import estimate_backlog_improvement, optimize_schedule
from dockmaster_ai_ops.technicians import generate_technician_roster
from dockmaster_ai_ops.work_orders import enrich_work_orders, random_work_orders
from dockmaster_ai_ops import assistant as dm_assistant

st.set_page_config(
    page_title="DockMaster AI Ops",
    page_icon="⚓",
    layout="wide",
)

st.title("DockMaster AI Ops")
st.caption(
    "Predictive prioritization with interpretable drivers, SLA-aware scheduling, and financial exposure "
    "(AI4I proxy + OR-Tools). For boatyard and marina ops."
)

with st.sidebar:
    st.header("Demo controls")
    n_jobs = st.slider("Work orders", 5, 40, 18, 1)
    seed = st.number_input("Random seed", 0, 99999, 42)
    n_techs = st.slider("Technicians / bays", 2, 8, 4)
    slot_mins = st.selectbox("Slot length (minutes)", [30, 60, 120], index=1)
    scenario = st.selectbox(
        "Scenario simulator",
        [
            "None",
            "Two technicians absent",
            "Parts delayed +1 day (20% of jobs)",
            "Parts delayed +2 slots (all pending parts)",
            "Spring demand +30%",
        ],
    )
    st.caption(
        "Scenarios adjust parts ETAs or capacity; the optimizer re-plans under the same rules."
    )
    if st.button("Retrain risk model (AI4I)"):
        with st.spinner("Training…"):
            train_failure_model()
        st.success("Model saved.")
    st.markdown("---")
    st.markdown(
        "**Data:** [AI4I 2020](https://archive.ics.uci.edu/ml/datasets/ai4i+2020+predictive+maintenance+dataset) · "
        "**Schedule:** [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)"
    )

slots_per_day = max(1, int((8 * 60) / slot_mins))

n_eff = n_jobs
if scenario == "Spring demand +30%":
    n_eff = min(40, int(np.ceil(n_jobs * 1.3)))

techs = generate_technician_roster(n_techs, seed=int(seed))
wo = random_work_orders(n_eff, int(seed), technicians=techs)
wo = enrich_work_orders(wo)

if scenario not in ("None", "Spring demand +30%"):
    wo, techs = apply_scenario(wo, techs, scenario, int(seed), slots_per_day=slots_per_day)

if len(wo) == 0:
    st.error("No feasible work orders after scenario — increase technicians or change seed.")
    st.stop()

st.caption(
    f"{len(wo)} work orders · {len(techs)} technicians/bays · scenario: **{scenario}** "
    "(dual-skill jobs need engine + electrical on a matching bay)"
)

result = optimize_schedule(wo, techs, slot_minutes=slot_mins, seed=int(seed))
sched = result.schedule

baseline_fcfs = run_baseline(wo, techs, slot_mins, result.horizon_slots, "fcfs")
baseline_prom = run_baseline(wo, techs, slot_mins, result.horizon_slots, "promised_date")

cmp_fcfs = compare_to_baseline(wo, baseline_fcfs.schedule, sched)
cmp_prom = compare_to_baseline(wo, baseline_prom.schedule, sched)

opt_makespan = float(sched["end_slot"].max()) if len(sched) else 0.0
base_ms_fcfs = (
    float(baseline_fcfs.schedule["end_slot"].max()) if len(baseline_fcfs.schedule) else 0.0
)
span_imp = estimate_backlog_improvement(base_ms_fcfs, opt_makespan)

kpi_opt = extended_business_kpis(wo, sched, len(techs), float(result.horizon_slots))
kpi_fcfs = extended_business_kpis(
    wo, baseline_fcfs.schedule, len(techs), float(result.horizon_slots)
)

merged_export = sched.merge(wo, on="work_order_id", how="left", suffixes=("_sched", ""))

# --- Grounded assistant context (floating dialog; available from any tab) ---
baseline_bundle = {
    "vs_fcfs_weighted_completion_improve_pct": float(cmp_fcfs["weighted_completion_improve_pct"]),
    "vs_fcfs_overdue_count_reduction_pct": float(cmp_fcfs["overdue_count_reduction_pct"]),
    "vs_fcfs_high_risk_sooner_pct": float(cmp_fcfs["high_risk_sooner_pct"]),
    "baseline_overdue": float(cmp_fcfs["baseline_overdue"]),
    "optimized_overdue": float(cmp_fcfs["optimized_overdue"]),
    "vs_promised_date_weighted_completion_improve_pct": float(cmp_prom["weighted_completion_improve_pct"]),
}
run_meta = {
    "solver_status": result.status_name,
    "objective_value": float(result.objective_value),
    "optimized_makespan_slots": float(opt_makespan),
    "n_work_orders": len(wo),
    "n_schedule_rows": len(sched),
    "slot_minutes": int(slot_mins),
}
_ctx = dm_assistant.build_assistant_context(
    wo,
    sched,
    techs,
    kpi_opt,
    baseline_bundle,
    float(result.horizon_slots),
    scenario,
    run_metadata=run_meta,
)
_gemini_key = dm_assistant.get_gemini_api_key()
try:
    _gemini_key = _gemini_key or str(st.secrets["GEMINI_API_KEY"])
except (KeyError, FileNotFoundError, RuntimeError):
    pass

if "dm_assistant_messages" not in st.session_state:
    st.session_state.dm_assistant_messages = []
if "dm_assistant_dialog_open" not in st.session_state:
    st.session_state.dm_assistant_dialog_open = False


def _on_assistant_dialog_dismiss() -> None:
    """Sync session state when the user closes the modal (X, overlay, Esc).

    Default dialog behavior does not rerun the app on dismiss, so
    ``dm_assistant_dialog_open`` would stay True and the chat would reopen on the
    next full rerun (e.g. sidebar change).
    """
    st.session_state["dm_assistant_dialog_open"] = False


_dialog_kw: dict = {}
_sig = inspect.signature(st.dialog)
if "on_dismiss" in _sig.parameters:
    _dialog_kw["on_dismiss"] = _on_assistant_dialog_dismiss
elif "dismissible" in _sig.parameters:
    # Older Streamlit: no X / outside-click dismiss; Close button still works.
    _dialog_kw["dismissible"] = False


@st.dialog("Ask DockMaster AI Ops", width="large", **_dialog_kw)
def dockmaster_assistant_dialog() -> None:
    st.markdown(
        "The predictive model and optimizer produce **this** plan; the assistant answers using **all** live context "
        "(KPIs, scored work orders, schedule, technician roster, baselines). "
        "Follow-ups use recent chat history; facts always match the current sidebar run."
    )
    _hdr_l, _hdr_r = st.columns([4, 1])
    with _hdr_r:
        if st.button("Close", key="dm_close_assistant_dialog"):
            st.session_state.dm_assistant_dialog_open = False
            st.rerun()
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Clear chat", key="dm_clear_assistant_chat"):
            st.session_state.dm_assistant_messages = []
            st.rerun()

    if not _gemini_key:
        st.warning(
            f"Add **`GEMINI_API_KEY`** to **`{ROOT / '.env'}`** (see `.env.example`) or Streamlit secrets."
        )
    else:
        st.caption("Powered by Gemini · grounded on JSON from this run.")

    for msg in st.session_state.dm_assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if _gemini_key and (user_prompt := st.chat_input("Ask anything about the current plan…")):
        prior = [dict(m) for m in st.session_state.dm_assistant_messages]
        with st.spinner("Generating answer…"):
            answer_text = dm_assistant.answer_question_from_context(
                user_prompt,
                _ctx,
                _gemini_key,
                chat_history=prior,
            )
        st.session_state.dm_assistant_messages.append({"role": "user", "content": user_prompt})
        st.session_state.dm_assistant_messages.append({"role": "assistant", "content": answer_text})
        st.rerun()

    with st.expander("What data is sent to the model?"):
        st.markdown(
            "Each reply uses: **run_metadata**, **kpis**, **work_orders**, **schedule**, **technicians**, "
            "**baseline_comparison**, **scenario**."
        )
        st.json(
            {
                "run_metadata": _ctx.get("run_metadata"),
                "kpis": _ctx["kpis"],
                "horizon_slots": _ctx["horizon_slots"],
                "scenario": _ctx["scenario"],
                "baseline_comparison": _ctx["baseline_comparison"],
                "schedule_row_count": len(_ctx["schedule"]),
                "work_order_count": len(_ctx["work_orders"]),
                "technician_count": len(_ctx["technicians"]),
            }
        )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Operations",
        "ROI vs baseline",
        "Risk & work orders",
        "Model quality",
        "Commercial",
    ]
)

with tab1:
    st.caption(
        "Optimizer minimizes **priority-weighted completion** (ML urgency + financial exposure) "
        "+ **SLA lateness** vs `promised_due_slot` (tier-weighted) + labor cost — under skills, bay, parts ETA, shifts."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solver", result.status_name)
    c2.metric("Planning horizon (slots)", result.horizon_slots)
    c3.metric("Objective (weighted)", f"{result.objective_value:,.0f}")
    c4.metric("Schedule span (slots)", f"{opt_makespan:.0f}")

    st.subheader("Business KPIs (this plan)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "Fleet utilization (proxy)",
        f"{kpi_opt['fleet_utilization_proxy'] * 100:.0f}%",
        help="Sum(job durations) / (horizon × technicians)",
    )
    b2.metric(
        "High-priority jobs on-time",
        f"{kpi_opt['high_priority_on_time_pct']:.0f}%",
        help="Share of top-quartile priority jobs with end_slot ≤ promised_due_slot",
    )
    b3.metric(
        "Total SLA lateness (slots)",
        f"{kpi_opt['total_sla_lateness_slots']:.0f}",
        delta=f"vs FCFS: {kpi_fcfs['total_sla_lateness_slots']:.0f}",
    )
    b4.metric(
        "Est. failure cost basis",
        f"${ESTIMATED_FAILURE_COST_USD:,.0f}",
        help="expected_financial_exposure = failure_risk × this (demo constant)",
    )

    with st.expander("Technician roster (skills, bay type, shift, labor rate)"):
        st.dataframe(
            techs[
                [
                    "display_name",
                    "bay_type",
                    "can_engine",
                    "can_electrical",
                    "can_hull",
                    "can_general",
                    "shift_start",
                    "shift_end",
                    "hourly_cost",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Gantt — skill + bay matched")
    if len(sched):
        sp = sched.copy()
        sp["assigned_technician"] = sp["assigned_technician"].astype(str)
        sp["duration_slots"] = pd.to_numeric(sp["duration_slots"], errors="coerce").fillna(1).astype(float)
        sp["start_slot"] = pd.to_numeric(sp["start_slot"], errors="coerce").fillna(0).astype(float)
        sp["lateness_slots"] = pd.to_numeric(sp["lateness_slots"], errors="coerce").fillna(0).astype(float)
        sp["hover_label"] = (
            sp["work_order_id"].astype(str)
            + "<br>"
            + sp["vessel_id"].astype(str)
            + "<br>"
            + sp["bay_type"].astype(str)
            + " · priority "
            + sp["scheduling_priority_score"].round(1).astype(str)
            + "<br>slots "
            + sp["start_slot"].astype(int).astype(str)
            + "–"
            + sp["end_slot"].astype(int).astype(str)
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=sp["duration_slots"],
                    y=sp["assigned_technician"],
                    base=sp["start_slot"],
                    orientation="h",
                    marker=dict(
                        color=sp["lateness_slots"],
                        colorscale="Redor",
                        showscale=True,
                        colorbar=dict(title="Lateness (slots)"),
                        line=dict(width=0.5, color="rgba(255,255,255,0.25)"),
                    ),
                    hovertext=sp["hover_label"],
                    hoverinfo="text",
                )
            ]
        )
        tech_order = sorted(sp["assigned_technician"].unique())
        fig.update_layout(
            height=min(900, 340 + 38 * max(1, len(techs))),
            xaxis_title="Time slot index",
            yaxis_title="",
            barmode="overlay",
            margin=dict(l=200, r=12, t=12, b=48),
            yaxis=dict(categoryorder="array", categoryarray=tech_order, automargin=True),
            uirevision="gantt",
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.info("No schedule.")

    st.subheader("Why is this job prioritized? (interpretable drivers)")
    top = wo.sort_values("scheduling_priority_score", ascending=False).head(4)
    for _, row in top.iterrows():
        st.markdown(f"**{row['work_order_id']}** · {row['vessel_type'].replace('_', ' ')} · {row['marina_location']}")
        st.markdown(row.get("why_prioritized", row.get("risk_reason", "")))
        st.caption(
            f"Failure risk {row['failure_risk']:.2f} · "
            f"Exposure ~ ${row.get('expected_financial_exposure_usd', 0):,.0f} · "
            f"{row['operational_priority']} · {row['service_window_band']}"
        )

with tab2:
    st.markdown(
        "Compared to **manual-style** baselines on the same work orders and technician roster."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "High-risk jobs finish sooner (vs FCFS)",
        f"{cmp_fcfs['high_risk_sooner_pct']:.0f}%",
        help="Mean completion slot for top-quartile urgency jobs",
    )
    m2.metric(
        "Weighted completion improvement (vs FCFS)",
        f"{cmp_fcfs['weighted_completion_improve_pct']:.0f}%",
        help="Lower is better: sum(urgency × completion slot)",
    )
    m3.metric(
        "Overdue WO count reduction (vs FCFS)",
        f"{cmp_fcfs['overdue_count_reduction_pct']:.0f}%",
        help="Jobs with end_slot after promised_due_slot",
    )
    m4, m5, m6 = st.columns(3)
    m4.metric(
        "Makespan vs FCFS",
        f"{span_imp['pct_backlog_reduction']:.0f}%",
        delta=f"{span_imp['slots_saved']:.0f} slots" if span_imp["slots_saved"] > 0 else None,
    )
    m5.metric(
        "vs promised-date ordering",
        f"{cmp_prom['weighted_completion_improve_pct']:.0f}%",
        help="Weighted completion improvement vs sorting by due date only",
    )
    m6.metric(
        "Baseline overdue (FCFS)",
        f"{int(cmp_fcfs['baseline_overdue'])}",
        delta=f"Optimized: {int(cmp_fcfs['optimized_overdue'])}",
    )

    st.caption(
        "KPIs are illustrative for the case study; a pilot would measure real overdue work, "
        "comeback rate, and labor utilization from DockMaster."
    )

with tab3:
    show_cols = [
        "work_order_id",
        "vessel_id",
        "vessel_type",
        "engine_type",
        "marina_location",
        "failure_risk",
        "urgency_score",
        "expected_financial_exposure_usd",
        "scheduling_priority_score",
        "operational_priority",
        "service_window_band",
        "risk_drivers_percentile",
        "why_prioritized",
        "required_skill",
        "required_skill_secondary",
        "required_bay_type",
        "promised_due_slot",
        "parts_eta_slot",
        "customer_tier",
        "estimated_duration_h",
    ]
    show = wo[show_cols].sort_values("failure_risk", ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Proxy features → marina labels (model inputs)")
    st.table(
        pd.DataFrame(
            [{"Model column": k, "Marina label": v} for k, v in FEATURE_DISPLAY_NAMES.items()]
        )
    )

    csv_buf = io.StringIO()
    merged_export.to_csv(csv_buf, index=False)
    st.download_button(
        "Download scored + scheduled CSV",
        csv_buf.getvalue(),
        file_name="dockmaster_ai_ops_export.csv",
        mime="text/csv",
    )

with tab4:
    payload = load_model()
    metrics = payload.get("metrics", {})
    backend = payload.get("risk_model_backend") or metrics.get("risk_model_backend", "lgbm")
    st.write(
        f"**Classifier:** `{backend}` + SMOTENC pipeline + isotonic calibration · **Labels:** machine failure (AI4I)"
    )
    st.json(metrics)
    st.caption(
        "Product-facing metrics: precision/lift at top decile and recall captured in top 20% — "
        "answers 'if we only review the highest-risk queue, what do we catch?'"
    )

with tab5:
    st.markdown(
        """
### DockMaster AI Ops — premium module

**Buyer:** Marina operator, boatyard service manager, marine service chain ops lead (already on DockMaster).

**Problem:** Preventable breakdowns and inefficient dispatch waste bay time; manual drag-and-drop does not jointly optimize **risk**, **promised dates**, **parts arrival**, and **technician/bay skills**.

**Outcome:** Fewer overdue work orders, higher utilization of skilled techs and matched bays, clearer priority queue for the dock office.

**KPIs to improve:** service throughput, on-time completion, emergency/callback jobs, technician overtime.

**Packaging:** Add-on to Service Management (risk scoring + recommended service window) and Scheduling (constraint optimizer + scenarios). Price as uplift on service ARR, not a generic “AI dashboard.”

**Scale:** Batch scoring on work-order save; re-optimize on schedule board “Optimize” or nightly; fits multi-tenant SaaS.
        """
    )

# Floating assistant (bottom-right) — open from any tab
st.markdown(
    """
<style>
/* Streamlit adds st-key-{key} on the widget wrapper */
div.st-key-dockmaster_fab {
    position: fixed !important;
    bottom: 3.25rem !important;
    right: 1.25rem !important;
    z-index: 99990 !important;
}
div.st-key-dockmaster_fab button {
    border-radius: 999px !important;
    min-width: 3.25rem !important;
    min-height: 3.25rem !important;
    font-size: 1.35rem !important;
    padding: 0.35rem 0.65rem !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}
</style>
""",
    unsafe_allow_html=True,
)
if st.button(
    "💬",
    key="dockmaster_fab",
    type="primary",
    help="Ask DockMaster AI Ops (grounded on this run)",
):
    st.session_state.dm_assistant_dialog_open = True

if st.session_state.get("dm_assistant_dialog_open"):
    dockmaster_assistant_dialog()

st.divider()
st.caption(
    "Production would train on DockMaster work history, OEM intervals, and parts/inventory feeds; "
    "this demo uses public maintenance data and synthetic yard records."
)
