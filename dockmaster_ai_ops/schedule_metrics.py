"""KPIs for comparing schedules (baseline vs optimized)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _priority_column(wo: pd.DataFrame) -> str:
    return (
        "scheduling_priority_score"
        if "scheduling_priority_score" in wo.columns
        else "urgency_score"
    )


def _merge(wo: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    if len(sched) == 0:
        return pd.DataFrame()
    s = sched.drop(
        columns=["scheduling_priority_score", "urgency_score"],
        errors="ignore",
    )
    prio = _priority_column(wo)
    return s.merge(
        wo[
            [
                "work_order_id",
                prio,
                "promised_due_slot",
                "failure_risk",
            ]
        ].rename(columns={prio: "priority_for_kpi"}),
        on="work_order_id",
        how="left",
    )


def compute_schedule_kpis(wo: pd.DataFrame, sched: pd.DataFrame) -> dict[str, float]:
    """Weighted completion, makespan, overdue counts, high-risk delay."""
    if len(sched) == 0:
        return {
            "weighted_completion": 0.0,
            "makespan_slots": 0.0,
            "overdue_count": 0.0,
            "overdue_slots_sum": 0.0,
            "high_risk_mean_end": 0.0,
            "high_risk_mean_delay": 0.0,
            "n_jobs": 0.0,
        }
    m = _merge(wo, sched)
    ends = m["end_slot"].values.astype(float)
    prio = m["priority_for_kpi"].values.astype(float)
    due = m["promised_due_slot"].values.astype(float)
    weighted = float(np.sum(prio * ends))
    makespan = float(np.max(ends))
    late = np.maximum(0.0, ends - due)
    overdue_count = float(np.sum(late > 0))
    overdue_sum = float(np.sum(late))
    thr = np.percentile(
        wo[_priority_column(wo)].values.astype(float),
        75,
    )
    hr = prio >= thr
    if hr.any():
        hr_mean_end = float(np.mean(ends[hr]))
        hr_mean_delay = float(np.mean(late[hr]))
    else:
        hr_mean_end = float(np.mean(ends))
        hr_mean_delay = float(np.mean(late))
    return {
        "weighted_completion": weighted,
        "makespan_slots": makespan,
        "overdue_count": overdue_count,
        "overdue_slots_sum": overdue_sum,
        "high_risk_mean_end": hr_mean_end,
        "high_risk_mean_delay": hr_mean_delay,
        "n_jobs": float(len(m)),
    }


def extended_business_kpis(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    n_technicians: int,
    horizon_slots: float,
) -> dict[str, float]:
    """Utilization proxy, SLA lateness, on-time rate for high-priority slice."""
    if len(sched) == 0 or n_technicians <= 0:
        return {
            "fleet_utilization_proxy": 0.0,
            "total_sla_lateness_slots": 0.0,
            "high_priority_on_time_pct": 100.0,
        }
    m = _merge(wo, sched)
    ends = m["end_slot"].values.astype(float)
    due = m["promised_due_slot"].values.astype(float)
    dur = m["duration_slots"].values.astype(float)
    late = np.maximum(0.0, ends - due)
    prio = m["priority_for_kpi"].values.astype(float)

    thr = np.percentile(wo[_priority_column(wo)].values.astype(float), 75)
    hp = prio >= thr
    if hp.any():
        on_time = float(np.mean(ends[hp] <= due[hp]) * 100.0)
    else:
        on_time = 100.0

    denom = max(1.0, float(horizon_slots) * float(n_technicians))
    util = float(np.sum(dur) / denom)

    return {
        "fleet_utilization_proxy": min(1.0, util),
        "total_sla_lateness_slots": float(np.sum(late)),
        "high_priority_on_time_pct": on_time,
    }


def pct_improvement(before: float, after: float, *, lower_is_better: bool = True) -> float:
    if before <= 0:
        return 0.0
    if lower_is_better:
        return max(0.0, 100.0 * (before - after) / before)
    return max(0.0, 100.0 * (after - before) / before)


def compare_to_baseline(
    wo: pd.DataFrame,
    baseline_sched: pd.DataFrame,
    optimized_sched: pd.DataFrame,
) -> dict[str, float]:
    b = compute_schedule_kpis(wo, baseline_sched)
    o = compute_schedule_kpis(wo, optimized_sched)
    return {
        "weighted_completion_improve_pct": pct_improvement(
            b["weighted_completion"], o["weighted_completion"]
        ),
        "makespan_improve_pct": pct_improvement(
            b["makespan_slots"], o["makespan_slots"]
        ),
        "overdue_count_reduction_pct": pct_improvement(
            b["overdue_count"], o["overdue_count"]
        ),
        "high_risk_sooner_pct": pct_improvement(
            b["high_risk_mean_end"], o["high_risk_mean_end"]
        ),
        "high_risk_delay_drop_pct": pct_improvement(
            b["high_risk_mean_delay"], o["high_risk_mean_delay"]
        ),
        "baseline_overdue": b["overdue_count"],
        "optimized_overdue": o["overdue_count"],
    }
