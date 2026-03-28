"""Marina and scheduling constants for DockMaster AI Ops."""

from __future__ import annotations

AI4I_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

# AI4I column names (as in CSV)
COL_TYPE = "Type"
COL_AIR = "Air temperature [K]"
COL_PROC = "Process temperature [K]"
COL_RPM = "Rotational speed [rpm]"
COL_TORQUE = "Torque [Nm]"
COL_WEAR = "Tool wear [min]"
COL_FAILURE = "Machine failure"

# Partial SMOTENC: target failure count = max(actual positives, ratio × majority count).
# Quick grid on AI4I (~0.08–0.14): ~0.10 often maximizes PR-AUC; higher ratio pushes more recall in top-k.
SMOTE_TARGET_RATIO_TO_MAJORITY = 0.10

# Failure-risk base learner after SMOTENC (inside calibration CV): "lgbm" or "rf"
RISK_MODEL_BACKEND = "lgbm"

# sklearn pipeline column names (must match train + scoring)
TRAINING_FEATURE_NAMES = (
    "vessel_class",
    "air_k",
    "proc_k",
    "rpm",
    "torque_nm",
    "wear_min",
)

# Marina-facing labels for the same features (UI / interviews)
FEATURE_DISPLAY_NAMES = {
    "vessel_class": "Vessel size class (L/M/H proxy)",
    "air_k": "Ambient temperature (K)",
    "proc_k": "Engine operating temperature (K)",
    "rpm": "Engine RPM",
    "torque_nm": "Engine load (Nm)",
    "wear_min": "Engine hours since last service (proxy)",
}

# Legacy table for docs
FEATURE_MARINA_LABELS = {
    COL_TYPE: "Vessel class (L/M/H)",
    COL_AIR: "Ambient temperature (K)",
    COL_PROC: "Engine operating temperature (K)",
    COL_RPM: "Engine RPM",
    COL_TORQUE: "Engine load (Nm)",
    COL_WEAR: "Engine hours since last service (proxy)",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

SKILLS = ("engine", "electrical", "hull", "general")
BAY_TYPES = ("standard", "heavy_lift", "wet_slip")

CUSTOMER_TIERS = ("standard", "premium", "fleet")
TIER_LATE_MULT = {"standard": 1.0, "premium": 2.0, "fleet": 1.5}

SEASONALITY = (
    "spring_launch",
    "mid_season_repair",
    "winter_storage_prep",
    "routine",
)

SERVICE_CATEGORIES = (
    "engine",
    "hull",
    "electrical",
    "winterization",
    "detailing",
)

VESSEL_TYPES = (
    "center_console",
    "pontoon",
    "sailboat",
    "yacht",
    "fishing_boat",
)

ENGINE_TYPES = ("outboard", "inboard", "diesel", "electric")

STORAGE_TYPES = ("in_water", "dry_stack", "trailer")

SLOT_MINUTES = 60
HORIZON_SLOTS_DEFAULT = 120

# Rough average cost of an unplanned failure / emergency service (USD) — demo only
ESTIMATED_FAILURE_COST_USD = 12500.0
# How much financial exposure (per $1k) feeds into schedule priority on top of urgency
SCHEDULER_FINANCIAL_WEIGHT = 2.5
