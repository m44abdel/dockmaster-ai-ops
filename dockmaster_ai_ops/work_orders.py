"""Synthetic DockMaster-style work orders + ML feature matrix + risk explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dockmaster_ai_ops.config import (
    COL_AIR,
    COL_PROC,
    COL_RPM,
    COL_TORQUE,
    COL_TYPE,
    COL_WEAR,
    CUSTOMER_TIERS,
    ENGINE_TYPES,
    ESTIMATED_FAILURE_COST_USD,
    SCHEDULER_FINANCIAL_WEIGHT,
    SEASONALITY,
    SERVICE_CATEGORIES,
    STORAGE_TYPES,
    TRAINING_FEATURE_NAMES,
    VESSEL_TYPES,
)
from dockmaster_ai_ops.explainability import build_risk_drivers_short, build_why_prioritized
from dockmaster_ai_ops.maintenance_model import (
    load_model,
    risk_scores_for_work_orders,
    urgency_score,
)


def _sample_skill_and_bay(techs: pd.DataFrame, rng: np.random.Generator) -> tuple[str, str]:
    idx = int(rng.integers(0, len(techs)))
    t = techs.iloc[idx]
    options: list[str] = []
    if t.get("can_engine"):
        options.append("engine")
    if t.get("can_electrical"):
        options.append("electrical")
    if t.get("can_hull"):
        options.append("hull")
    options.append("general")
    skill = str(rng.choice(np.array(options)))
    return skill, str(t["bay_type"])


def random_work_orders(
    n: int,
    seed: int = 42,
    technicians: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate DockMaster-style records. If technicians is provided, each job's
    required_skill / required_bay_type is drawn so at least one tech on the roster matches.
    """
    rng = np.random.default_rng(seed)
    if technicians is None or len(technicians) == 0:
        from dockmaster_ai_ops.technicians import generate_technician_roster

        technicians = generate_technician_roster(4, seed=seed)

    types = rng.choice(["L", "M", "H"], size=n, p=[0.35, 0.4, 0.25])
    air = rng.normal(300, 2.5, n).clip(295, 305)
    proc = air + rng.normal(10, 1.2, n)
    rpm = rng.normal(1520, 180, n).clip(1150, 2100)
    torque = rng.normal(42, 9, n).clip(18, 68)
    wear = rng.integers(5, 240, size=n)
    wo_ids = [f"WO-2025-{i+1:04d}" for i in range(n)]
    vessel_ids = [f"V-{rng.integers(1000, 9999)}" for _ in range(n)]
    severity = rng.integers(1, 6, size=n)
    est_hours = rng.uniform(1.0, 8.0, size=n)
    promised_due = rng.integers(24, 96, size=n)
    tier = rng.choice(list(CUSTOMER_TIERS), size=n, p=[0.65, 0.25, 0.10])
    # Parts ETA in slots — 0 means ready now; else arrival slot
    parts_eta = np.zeros(n, dtype=int)
    for i in range(n):
        if rng.random() < 0.18:
            parts_eta[i] = int(rng.integers(4, 20))
        elif rng.random() < 0.1:
            parts_eta[i] = int(rng.integers(20, 40))

    req_skill: list[str] = []
    req_sec: list[str] = []
    req_bay: list[str] = []
    for _ in range(n):
        # ~14% jobs need engine + electrical on standard bay (dual-cert), rest single-skill
        if rng.random() < 0.14:
            req_skill.append("engine")
            req_sec.append("electrical")
            req_bay.append("standard")
        else:
            s, b = _sample_skill_and_bay(technicians, rng)
            req_skill.append(s)
            req_sec.append("")
            req_bay.append(b)

    marinas = [
        "Harbor East Marina",
        "Bayline Boatyard",
        "Pelican Marine Center",
        "North Sound Yacht Works",
    ]
    loc = rng.choice(np.array(marinas), size=n)

    return pd.DataFrame(
        {
            "work_order_id": wo_ids,
            "vessel_id": vessel_ids,
            COL_TYPE: types,
            COL_AIR: air,
            COL_PROC: proc,
            COL_RPM: rpm,
            COL_TORQUE: torque,
            COL_WEAR: wear,
            "issue_severity_1_5": severity,
            "estimated_duration_h": np.round(est_hours, 1),
            "promised_due_slot": promised_due,
            "customer_tier": tier,
            "parts_eta_slot": parts_eta,
            "required_skill": req_skill,
            "required_skill_secondary": req_sec,
            "required_bay_type": req_bay,
            "vessel_type": rng.choice(list(VESSEL_TYPES), size=n),
            "engine_type": rng.choice(list(ENGINE_TYPES), size=n),
            "marina_location": loc,
            "storage_type": rng.choice(list(STORAGE_TYPES), size=n),
            "service_category": rng.choice(list(SERVICE_CATEGORIES), size=n),
            "seasonality": rng.choice(list(SEASONALITY), size=n),
        }
    )


def _risk_reason_row(row: pd.Series) -> str:
    """Rules-based explanation from proxy sensor fields (no SHAP)."""
    wear = float(row["wear_min"])
    torque = float(row["torque_nm"])
    rpm = float(row["rpm"])
    air = float(row["air_k"])
    proc = float(row["proc_k"])
    delta = proc - air
    reasons: list[str] = []
    if wear > 180 and torque > 50:
        reasons.append("heavy engine strain (high wear + load)")
    if delta > 12:
        reasons.append("possible cooling inefficiency (large operating vs ambient delta)")
    if rpm > 1900 and wear > 150:
        reasons.append("accelerated degradation pattern (high RPM with elevated wear)")
    if torque > 58 and wear <= 180:
        reasons.append("short-term overload signature on drivetrain")
    if not reasons:
        reasons.append("within normal operating envelope vs fleet baseline")
    return "; ".join(reasons[:2])


def _service_window_band(urgency: float, severity: int) -> str:
    score = urgency + severity * 8.0
    if score >= 130:
        return "Critical — schedule within 1–2 days"
    if score >= 95:
        return "High — within 3–5 days"
    if score >= 65:
        return "Moderate — within 1–2 weeks"
    return "Low — next standard maintenance cycle"


def _operational_priority(
    failure_risk: float, urgency: float, tier: str, severity: int
) -> str:
    w = (
        failure_risk * 40.0
        + urgency * 0.35
        + severity * 5.0
        + (2.0 if tier == "premium" else 1.5 if tier == "fleet" else 0.0)
    )
    if w >= 85:
        return "P1 — dispatch today"
    if w >= 60:
        return "P2 — this week"
    if w >= 40:
        return "P3 — next available window"
    return "P4 — backlog"


def build_feature_matrix(wo: pd.DataFrame) -> pd.DataFrame:
    from dockmaster_ai_ops.config import TYPE_MAP

    out = wo.copy()
    out["type_ord"] = out[COL_TYPE].map(TYPE_MAP).fillna(1).astype(int)
    return pd.DataFrame(
        {
            "vessel_class": out["type_ord"],
            "air_k": out[COL_AIR],
            "proc_k": out[COL_PROC],
            "rpm": out[COL_RPM],
            "torque_nm": out[COL_TORQUE],
            "wear_min": out[COL_WEAR],
        }
    )


def validate_features_for_model(X: pd.DataFrame) -> None:
    cols = list(TRAINING_FEATURE_NAMES)
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    if X[cols].isna().any().any():
        raise ValueError("Model features contain NaN")
    if not np.isfinite(X[cols].values).all():
        raise ValueError("Model features contain non-finite values")


def enrich_work_orders(wo: pd.DataFrame) -> pd.DataFrame:
    load_model()
    out = wo.copy()
    if "required_skill_secondary" not in out.columns:
        out["required_skill_secondary"] = ""
    X = build_feature_matrix(out)
    validate_features_for_model(X)
    risk = risk_scores_for_work_orders(X)
    urg = urgency_score(risk)
    out["failure_risk"] = risk
    out["urgency_score"] = urg
    # Operational priority separate from pure ML risk
    out["service_window_band"] = [
        _service_window_band(float(u), int(s)) for u, s in zip(urg, out["issue_severity_1_5"])
    ]
    out["operational_priority"] = [
        _operational_priority(float(r), float(u), str(t), int(s))
        for r, u, t, s in zip(
            risk,
            urg,
            out["customer_tier"],
            out["issue_severity_1_5"],
        )
    ]
    out["risk_reason"] = X.apply(_risk_reason_row, axis=1)
    out["risk_drivers_percentile"] = build_risk_drivers_short(X).values
    out["expected_financial_exposure_usd"] = risk * float(ESTIMATED_FAILURE_COST_USD)
    sched_pri = urg + float(SCHEDULER_FINANCIAL_WEIGHT) * (
        out["expected_financial_exposure_usd"] / 1000.0
    )
    out["scheduling_priority_score"] = np.clip(sched_pri, 0.0, 200.0)
    out["why_prioritized"] = build_why_prioritized(out, X).values
    return out
