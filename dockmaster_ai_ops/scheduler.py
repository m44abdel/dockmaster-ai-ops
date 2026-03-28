"""OR-Tools CP-SAT: marina constraints (skills, bays, parts ETA, due dates, shifts, cost)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from dockmaster_ai_ops.config import TIER_LATE_MULT
from dockmaster_ai_ops.technicians import can_assign

logger = logging.getLogger(__name__)


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame
    objective_value: float
    status_name: str
    horizon_slots: int


def _duration_slots(hours: float, slot_minutes: int) -> int:
    return max(1, int(np.ceil(hours * 60 / slot_minutes)))


def _tier_int(tier: str) -> int:
    return int(TIER_LATE_MULT.get(str(tier), 1.0) * 100)


def _priority_values(wo: pd.DataFrame) -> np.ndarray:
    """Risk + financial exposure when present; else ML urgency only."""
    col = (
        "scheduling_priority_score"
        if "scheduling_priority_score" in wo.columns
        else "urgency_score"
    )
    return wo[col].values.astype(float)


def _compatibility_matrix(wo: pd.DataFrame, techs: pd.DataFrame) -> np.ndarray:
    n = len(wo)
    m = len(techs)
    mat = np.zeros((n, m), dtype=bool)
    for i in range(n):
        for t in range(m):
            mat[i, t] = can_assign(wo.iloc[i], techs.iloc[t])
    return mat


def optimize_schedule(
    wo: pd.DataFrame,
    techs: pd.DataFrame,
    slot_minutes: int = 60,
    horizon_slots: int | None = None,
    seed: int = 42,
    urgency_weight: int = 100,
    late_penalty_weight: int = 50,
    cost_weight: int = 1,
) -> ScheduleResult:
    """
    Minimize:
      urgency_weight * sum(priority_i * end_i)  # priority = ML urgency + optional $ exposure
      + late_penalty_weight * sum(tier_mult_i * lateness_i)  # SLA vs promised_due_slot
      + cost_weight * sum(hourly_cost_t * duration_i)
    Subject to: skill(s), bay, shift, parts ETA, no overlap per tech.
    """
    n = len(wo)
    m = len(techs)
    if n == 0:
        return ScheduleResult(pd.DataFrame(), 0.0, "EMPTY", 0)
    if m == 0:
        return ScheduleResult(pd.DataFrame(), 0.0, "NO_TECHS", 0)

    allowed = _compatibility_matrix(wo, techs)
    for i in range(n):
        if not allowed[i].any():
            logger.error("Job %s has no compatible technician", wo.iloc[i].get("work_order_id"))
            return ScheduleResult(pd.DataFrame(), 0.0, "INFEASIBLE", 0)

    durations = np.array(
        [
            _duration_slots(float(h), slot_minutes)
            for h in wo["estimated_duration_h"].values
        ],
        dtype=int,
    )
    parts_eta = wo["parts_eta_slot"].values.astype(int)
    promised = wo["promised_due_slot"].values.astype(int)
    priority = _priority_values(wo)
    tier_w = np.array([_tier_int(str(x)) for x in wo["customer_tier"].values])

    shift_st = techs["shift_start"].values.astype(int)
    shift_en = techs["shift_end"].values.astype(int)
    hourly = techs["hourly_cost"].values.astype(int)

    sum_dur = int(durations.sum())
    h = horizon_slots or max(120, sum_dur + m * 8 + int(parts_eta.max()) + 24)
    h = max(h, sum_dur // max(m, 1) + int(durations.max()) + int(parts_eta.max()) + 10)
    shift_en = np.minimum(shift_en, h)
    M = h + 5

    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    ends: list[cp_model.IntVar] = []
    presences: list[list[cp_model.IntVar]] = []
    lateness: list[cp_model.IntVar] = []

    for i in range(n):
        s = model.NewIntVar(0, h, f"s_{i}")
        e = model.NewIntVar(0, h, f"e_{i}")
        model.Add(e == s + int(durations[i]))
        model.Add(s >= int(parts_eta[i]))
        starts.append(s)
        ends.append(e)
        late = model.NewIntVar(0, h, f"late_{i}")
        model.Add(late >= e - int(promised[i]))
        lateness.append(late)

        row_pres: list[cp_model.IntVar] = []
        for t in range(m):
            pres = model.NewBoolVar(f"x_{i}_{t}")
            if not allowed[i, t]:
                model.Add(pres == 0)
            row_pres.append(pres)
        model.Add(sum(row_pres) == 1)
        presences.append(row_pres)

    tech_intervals: list[list] = [[] for _ in range(m)]
    for i in range(n):
        dur_i = int(durations[i])
        for t in range(m):
            if not allowed[i, t]:
                continue
            interval = model.NewOptionalIntervalVar(
                starts[i],
                dur_i,
                ends[i],
                presences[i][t],
                f"opt_{i}_{t}",
            )
            tech_intervals[t].append(interval)

    for t in range(m):
        if tech_intervals[t]:
            model.AddNoOverlap(tech_intervals[t])

    # Shift windows when on technician t
    for i in range(n):
        for t in range(m):
            if not allowed[i, t]:
                continue
            p = presences[i][t]
            model.Add(starts[i] >= shift_st[t] - M * (1 - p))
            model.Add(ends[i] <= shift_en[t] + M * (1 - p))

    obj_terms = []
    for i in range(n):
        wu = int(max(1.0, priority[i] * urgency_weight))
        obj_terms.append(wu * ends[i])
        wl = int(late_penalty_weight * tier_w[i] // 100)
        wl = max(1, wl)
        obj_terms.append(wl * lateness[i])

    for i in range(n):
        dur_i = int(durations[i])
        for t in range(m):
            if not allowed[i, t]:
                continue
            coeff = int(hourly[t] * dur_i / 10) + 1
            obj_terms.append(cost_weight * coeff * presences[i][t])

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("CP-SAT status=%s; using greedy fallback", status_name)
        return _greedy_schedule(
            wo,
            techs,
            slot_minutes,
            h,
            allowed,
            urgency_weight,
            late_penalty_weight,
            cost_weight,
        )

    rows = []
    objective_value = float(solver.ObjectiveValue())
    for i in range(n):
        tech_idx = next(t for t in range(m) if solver.Value(presences[i][t]) == 1)
        tr = techs.iloc[tech_idx]
        rows.append(
            {
                "work_order_id": wo.iloc[i]["work_order_id"],
                "vessel_id": wo.iloc[i]["vessel_id"],
                "assigned_technician": str(tr["display_name"]),
                "technician_id": str(tr["technician_id"]),
                "bay_type": str(tr["bay_type"]),
                "start_slot": solver.Value(starts[i]),
                "end_slot": solver.Value(ends[i]),
                "duration_slots": int(durations[i]),
                "scheduling_priority_score": priority[i],
                "lateness_slots": solver.Value(lateness[i]),
            }
        )
    sched = pd.DataFrame(rows).sort_values(["start_slot", "assigned_technician"])
    return ScheduleResult(sched, objective_value, status_name, h)


def _greedy_schedule(
    wo: pd.DataFrame,
    techs: pd.DataFrame,
    slot_minutes: int,
    horizon_slots: int,
    allowed: np.ndarray,
    urgency_weight: int = 100,
    late_penalty_weight: int = 50,
    cost_weight: int = 1,
) -> ScheduleResult:
    """Urgency-ordered greedy with marina constraints."""
    n = len(wo)
    m = len(techs)
    order = np.argsort(-_priority_values(wo))
    durations = [
        _duration_slots(float(h), slot_minutes) for h in wo["estimated_duration_h"].values
    ]
    parts_eta = wo["parts_eta_slot"].values.astype(int)
    promised = wo["promised_due_slot"].values.astype(int)
    priority = _priority_values(wo)
    tier_w = np.array([_tier_int(str(x)) for x in wo["customer_tier"].values])
    shift_st = techs["shift_start"].values.astype(int)
    shift_en = techs["shift_end"].values.astype(int)
    hourly = techs["hourly_cost"].values.astype(int)

    tech_free = np.zeros(m, dtype=int)
    rows = []
    obj = 0.0
    for i in order:
        dur = durations[i]
        best_t = -1
        best_end = None
        for t in range(m):
            if not allowed[i, t]:
                continue
            start = max(
                int(tech_free[t]),
                int(parts_eta[i]),
                int(shift_st[t]),
            )
            end = start + dur
            if end > shift_en[t]:
                continue
            if best_end is None or end < best_end:
                best_end = end
                best_t = t
        if best_t < 0:
            # relax shift end if nothing fits (should not happen with full horizon)
            for t in range(m):
                if not allowed[i, t]:
                    continue
                start = max(int(tech_free[t]), int(parts_eta[i]))
                end = start + dur
                if best_end is None or end < best_end:
                    best_end = end
                    best_t = t
        if best_t < 0:
            continue
        start = max(
            int(tech_free[best_t]),
            int(parts_eta[i]),
            int(shift_st[best_t]),
        )
        end = start + dur
        if end > shift_en[best_t]:
            end = min(horizon_slots, end)
            start = max(int(parts_eta[i]), end - dur)
        tech_free[best_t] = end
        late = max(0, end - int(promised[i]))
        u = float(priority[i])
        wl = max(1, int(late_penalty_weight * tier_w[i] // 100))
        wc = int(hourly[best_t] * dur / 10) + 1
        obj += (
            u * urgency_weight * end
            + wl * late
            + cost_weight * wc
        )
        tr = techs.iloc[best_t]
        rows.append(
            {
                "work_order_id": wo.iloc[i]["work_order_id"],
                "vessel_id": wo.iloc[i]["vessel_id"],
                "assigned_technician": str(tr["display_name"]),
                "technician_id": str(tr["technician_id"]),
                "bay_type": str(tr["bay_type"]),
                "start_slot": start,
                "end_slot": end,
                "duration_slots": dur,
                "scheduling_priority_score": u,
                "lateness_slots": late,
            }
        )
    sched = pd.DataFrame(rows).sort_values(["start_slot", "assigned_technician"])
    return ScheduleResult(sched, float(obj), "GREEDY_FALLBACK", horizon_slots)


def estimate_backlog_improvement(
    baseline_makespan: float, optimized_makespan: float
) -> dict:
    if baseline_makespan <= 0:
        return {"pct_backlog_reduction": 0.0, "slots_saved": 0.0}
    saved = baseline_makespan - optimized_makespan
    pct = 100.0 * saved / baseline_makespan
    return {
        "pct_backlog_reduction": max(0.0, float(pct)),
        "slots_saved": max(0.0, float(saved)),
    }
