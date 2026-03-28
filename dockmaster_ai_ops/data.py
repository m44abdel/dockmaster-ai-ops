"""Load AI4I 2020 with offline fallback."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dockmaster_ai_ops.config import (
    AI4I_URL,
    COL_AIR,
    COL_FAILURE,
    COL_PROC,
    COL_RPM,
    COL_TORQUE,
    COL_TYPE,
    COL_WEAR,
)

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR.parent / "data"


def _synthetic_ai4i_like(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.4, 0.35, 0.25])
    air = rng.normal(300, 2, n).clip(295, 305)
    proc = air + rng.normal(10, 1, n)
    rpm = rng.normal(1500, 200, n).clip(1100, 2200)
    torque = rng.normal(40, 10, n).clip(15, 70)
    wear = rng.integers(0, 250, size=n)
    failure = np.zeros(n, dtype=int)
    # Correlated failure signal (proxy for real data)
    risk = (
        (wear > 180).astype(float) * 0.35
        + (torque > 55).astype(float) * 0.2
        + (np.abs(proc - air) > 12).astype(float) * 0.15
        + rng.random(n) * 0.15
    )
    failure = (rng.random(n) < risk).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"P{i % 10000:05d}" for i in range(n)],
            COL_TYPE: types,
            COL_AIR: air,
            COL_PROC: proc,
            COL_RPM: rpm,
            COL_TORQUE: torque,
            COL_WEAR: wear,
            COL_FAILURE: failure,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )


def load_ai4i() -> pd.DataFrame:
    try:
        r = requests.get(AI4I_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        logger.info("Loaded AI4I from UCI (%s rows)", len(df))
        return df
    except Exception as e:
        logger.warning("UCI fetch failed (%s); using bundled/synthetic fallback", e)
        fallback = DATA_DIR / "ai4i2020_fallback.csv"
        if fallback.exists():
            return pd.read_csv(fallback)
        return _synthetic_ai4i_like()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from dockmaster_ai_ops.config import TYPE_MAP

    out = df.copy()
    if COL_TYPE in out.columns:
        out["type_ord"] = out[COL_TYPE].map(TYPE_MAP).fillna(1).astype(int)
    else:
        out["type_ord"] = 1
    feature_cols = ["type_ord", COL_AIR, COL_PROC, COL_RPM, COL_TORQUE, COL_WEAR]
    for c in feature_cols:
        if c not in out.columns:
            raise KeyError(f"Missing column {c}")
    X = out[feature_cols].copy()
    X.columns = ["vessel_class", "air_k", "proc_k", "rpm", "torque_nm", "wear_min"]
    y = out[COL_FAILURE].astype(int)
    return X, y
