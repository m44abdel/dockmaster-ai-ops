"""Synthetic technician / bay roster with skills, shifts, and cost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dockmaster_ai_ops.config import BAY_TYPES


def generate_technician_roster(
    n: int,
    seed: int = 42,
    horizon_slots: int = 120,
) -> pd.DataFrame:
    """
    Build a roster where skills and bay types cover typical yard mixes.
    Shifts are slot indices [shift_start, shift_end) within [0, horizon_slots).
    """
    rng = np.random.default_rng(seed)
    n = max(2, n)
    names = [f"Tech {i+1} — Bay {chr(65 + i)}" for i in range(n)]

    bay_cycle = list(BAY_TYPES)
    rows = []
    for i in range(n):
        # First bay: always standard + engine + electrical so dual-skill WOs stay feasible
        if i == 0:
            bay = "standard"
            can_engine = True
            can_electrical = True
            can_hull = bool(rng.random() > 0.3)
            can_general = True
        else:
            bay = bay_cycle[i % len(bay_cycle)]
            can_engine = bool(rng.random() > 0.2 or i % 4 == 0)
            can_electrical = bool(rng.random() > 0.25 or i % 4 == 1)
            can_hull = bool(rng.random() > 0.25 or i % 4 == 2)
            can_general = bool(i % 3 == 0 or rng.random() > 0.7)
            if not (can_engine or can_electrical or can_hull):
                can_general = True
        # Default: full planning horizon (solver still exposes shift fields for product demos)
        start = 0
        end = int(horizon_slots)
        hourly_cost = int(rng.integers(85, 140))
        rows.append(
            {
                "technician_id": f"T-{i+1:02d}",
                "display_name": names[i],
                "bay_type": bay,
                "can_engine": can_engine,
                "can_electrical": can_electrical,
                "can_hull": can_hull,
                "can_general": can_general,
                "shift_start": int(start),
                "shift_end": int(end),
                "hourly_cost": hourly_cost,
            }
        )
    return pd.DataFrame(rows)


def _skill_ok(skill: str, row_tech: pd.Series) -> bool:
    skill = str(skill).strip()
    if not skill or skill in ("nan", "None"):
        return True
    if skill == "general":
        return bool(
            row_tech.get("can_general")
            or row_tech.get("can_engine")
            or row_tech.get("can_electrical")
            or row_tech.get("can_hull")
        )
    key = f"can_{skill}"
    if key not in row_tech.index:
        return False
    return bool(row_tech[key])


def can_assign(row_job: pd.Series, row_tech: pd.Series) -> bool:
    """Skill(s) + bay compatibility. Optional ``required_skill_secondary`` = dual-cert work."""
    bay_ok = str(row_job["required_bay_type"]) == str(row_tech["bay_type"])
    if not bay_ok:
        return False
    skills = [str(row_job["required_skill"])]
    sec = row_job.get("required_skill_secondary")
    if sec is not None and str(sec).strip() not in ("", "nan", "None", "NaT"):
        skills.append(str(sec).strip())
    return all(_skill_ok(s, row_tech) for s in skills)
