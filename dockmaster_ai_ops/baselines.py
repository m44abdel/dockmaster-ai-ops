"""Baseline schedulers: FCFS, promised-date-first, and urgency-greedy reference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dockmaster_ai_ops.scheduler import (
    ScheduleResult,
    _compatibility_matrix,
    _duration_slots,
    _greedy_schedule,
    _priority_values,
)


def _order_indices(wo: pd.DataFrame, mode: str) -> np.ndarray:
    n = len(wo)
    if mode == "fcfs":
        return np.arange(n)
    if mode == "promised_date":
        return np.argsort(wo["promised_due_slot"].values)
    if mode == "input":
        return np.arange(n)
    raise ValueError(mode)


def run_baseline(
    wo: pd.DataFrame,
    techs: pd.DataFrame,
    slot_minutes: int,
    horizon_slots: int,
    mode: str,
) -> ScheduleResult:
    """
    Greedy simulation with fixed job order (FCFS or promised-date).
    Uses same feasibility rules as _greedy_schedule but respects `mode` order.
    """
    allowed = _compatibility_matrix(wo, techs)
    order = _order_indices(wo, mode)
    prio = _priority_values(wo)
    n = len(wo)
    m = len(techs)
    durations = [
        _duration_slots(float(h), slot_minutes) for h in wo["estimated_duration_h"].values
    ]
    parts_eta = wo["parts_eta_slot"].values.astype(int)
    promised = wo["promised_due_slot"].values.astype(int)
    shift_st = techs["shift_start"].values.astype(int)
    shift_en = techs["shift_end"].values.astype(int)

    tech_free = np.zeros(m, dtype=int)
    rows = []
    for ii in order:
        dur = durations[ii]
        best_t = -1
        best_end = None
        for t in range(m):
            if not allowed[ii, t]:
                continue
            start = max(int(tech_free[t]), int(parts_eta[ii]), int(shift_st[t]))
            end = start + dur
            if end > shift_en[t]:
                continue
            if best_end is None or end < best_end:
                best_end = end
                best_t = t
        if best_t < 0:
            for t in range(m):
                if not allowed[ii, t]:
                    continue
                start = max(int(tech_free[t]), int(parts_eta[ii]))
                end = start + dur
                if best_end is None or end < best_end:
                    best_end = end
                    best_t = t
        if best_t < 0:
            continue
        start = max(
            int(tech_free[best_t]),
            int(parts_eta[ii]),
            int(shift_st[best_t]),
        )
        end = start + dur
        if end > shift_en[best_t]:
            end = min(horizon_slots, end)
            start = max(int(parts_eta[ii]), end - dur)
        tech_free[best_t] = end
        late = max(0, end - int(promised[ii]))
        tr = techs.iloc[best_t]
        rows.append(
            {
                "work_order_id": wo.iloc[ii]["work_order_id"],
                "vessel_id": wo.iloc[ii]["vessel_id"],
                "assigned_technician": str(tr["display_name"]),
                "technician_id": str(tr["technician_id"]),
                "bay_type": str(tr["bay_type"]),
                "start_slot": start,
                "end_slot": end,
                "duration_slots": dur,
                "scheduling_priority_score": float(prio[ii]),
                "lateness_slots": late,
            }
        )
    if not rows:
        return ScheduleResult(pd.DataFrame(), 0.0, f"BASELINE_{mode.upper()}_EMPTY", horizon_slots)
    sched = pd.DataFrame(rows).sort_values(["start_slot", "assigned_technician"])
    # lightweight objective for logging
    obj = float(
        np.sum(sched["scheduling_priority_score"].values * sched["end_slot"].values)
    )
    return ScheduleResult(sched, obj, f"BASELINE_{mode.upper()}", horizon_slots)


def run_urgency_greedy_reference(
    wo: pd.DataFrame,
    techs: pd.DataFrame,
    slot_minutes: int,
    horizon_slots: int,
) -> ScheduleResult:
    """Same as optimizer fallback: urgency-desc greedy."""
    allowed = _compatibility_matrix(wo, techs)
    return _greedy_schedule(wo, techs, slot_minutes, horizon_slots, allowed)
