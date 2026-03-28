"""Fleet-relative “why prioritized?” explanations (no SHAP — percentile + rules)."""

from __future__ import annotations

import pandas as pd


def _pct_label(pct: float) -> str:
    p = max(0.0, min(100.0, pct))
    return f"{p:.0f}th percentile"


def build_why_prioritized(wo: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """
    Per work order, compare sensor proxies to the *current batch* (fleet slice).
    Returns bullet-style text for managers / interviews.
    """
    wear = X["wear_min"].astype(float)
    torque = X["torque_nm"].astype(float)
    rpm = X["rpm"].astype(float)
    delta = (X["proc_k"] - X["air_k"]).astype(float)

    wear_pct = wear.rank(pct=True) * 100.0
    torque_pct = torque.rank(pct=True) * 100.0
    rpm_pct = rpm.rank(pct=True) * 100.0
    delta_pct = delta.rank(pct=True) * 100.0

    out: list[str] = []
    for i in range(len(wo)):
        bullets: list[str] = []
        if wear_pct.iloc[i] >= 70:
            bullets.append(
                f"High engine hours since service ({_pct_label(float(wear_pct.iloc[i]))} vs this fleet batch)"
            )
        if torque_pct.iloc[i] >= 70:
            bullets.append(
                f"Elevated engine load ({_pct_label(float(torque_pct.iloc[i]))} vs batch)"
            )
        if rpm_pct.iloc[i] >= 75:
            bullets.append(
                f"High RPM operating point ({_pct_label(float(rpm_pct.iloc[i]))} vs batch)"
            )
        if delta_pct.iloc[i] >= 70:
            bullets.append(
                f"Large ambient vs operating temperature delta ({_pct_label(float(delta_pct.iloc[i]))} vs batch)"
            )
        if torque_pct.iloc[i] >= 65 and wear_pct.iloc[i] >= 65:
            bullets.append("Combined wear + load pattern vs peers (strain signature)")

        fr = float(wo["failure_risk"].iloc[i]) if "failure_risk" in wo.columns else 0.0
        bullets.insert(
            0,
            f"Calibrated failure risk {fr:.0%} (model vs fleet batch)",
        )
        if len(bullets) == 1:
            bullets.append("Signals near batch median — priority driven mainly by model score vs peers")
        out.append("\n".join(f"- {b}" for b in bullets[:5]))
    return pd.Series(out, index=wo.index)


def build_risk_drivers_short(X: pd.DataFrame) -> pd.Series:
    """Single-line companion to legacy risk_reason (percentile-flavored)."""
    wear_pct = X["wear_min"].rank(pct=True) * 100.0
    torque_pct = X["torque_nm"].rank(pct=True) * 100.0
    delta = (X["proc_k"] - X["air_k"]).astype(float)
    delta_pct = delta.rank(pct=True) * 100.0
    parts: list[str] = []
    for i in range(len(X)):
        bits: list[str] = []
        if float(wear_pct.iloc[i]) >= 75:
            bits.append(f"wear {_pct_label(float(wear_pct.iloc[i]))}")
        if float(torque_pct.iloc[i]) >= 75:
            bits.append(f"load {_pct_label(float(torque_pct.iloc[i]))}")
        if float(delta_pct.iloc[i]) >= 75:
            bits.append(f"temp delta {_pct_label(float(delta_pct.iloc[i]))}")
        parts.append("; ".join(bits) if bits else "within batch norms on key signals")
    return pd.Series(parts, index=X.index)
