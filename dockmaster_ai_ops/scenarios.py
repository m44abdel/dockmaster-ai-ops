"""What-if scenarios for planning demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dockmaster_ai_ops.technicians import can_assign


def filter_feasible(wo: pd.DataFrame, techs: pd.DataFrame) -> pd.DataFrame:
    """Keep work orders that at least one remaining technician can perform."""
    if len(techs) == 0:
        return wo.iloc[0:0].copy()
    keep = []
    for i in range(len(wo)):
        ok = any(
            can_assign(wo.iloc[i], techs.iloc[t]) for t in range(len(techs))
        )
        keep.append(ok)
    return wo.loc[keep].reset_index(drop=True)


def apply_scenario(
    wo: pd.DataFrame,
    techs: pd.DataFrame,
    scenario: str,
    seed: int,
    slots_per_day: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return possibly modified (work_orders, technicians)."""
    rng = np.random.default_rng(seed)
    wo = wo.copy()
    techs = techs.copy()

    if scenario == "Two technicians absent":
        techs = techs.iloc[2:].reset_index(drop=True) if len(techs) > 2 else techs.iloc[0:0]
    elif scenario == "Parts delayed +1 day (20% of jobs)":
        delay = max(1, slots_per_day)
        m = rng.random(len(wo)) < 0.20
        wo.loc[m, "parts_eta_slot"] = wo.loc[m, "parts_eta_slot"].astype(int) + delay
    elif scenario == "Parts delayed +2 slots (all pending parts)":
        wo.loc[wo["parts_eta_slot"] > 0, "parts_eta_slot"] += 2
    elif scenario == "None":
        pass
    else:
        pass

    wo = filter_feasible(wo, techs)
    return wo, techs
