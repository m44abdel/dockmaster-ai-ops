"""Train and apply failure-risk model (proxy for DockMaster service history)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dockmaster_ai_ops import config as dm_config
from dockmaster_ai_ops.config import TRAINING_FEATURE_NAMES
from dockmaster_ai_ops.data import load_ai4i, prepare_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "failure_risk.joblib"

# vessel_class is ordinal encoded (0,1,2) — SMOTENC treats it as categorical
_SMOTE_CATEGORICAL_IDX = [0]

def _smote_k_neighbors(y: pd.Series) -> int:
    n_pos = int((y == 1).sum())
    return max(1, min(5, n_pos - 1))


def _smote_sampling_strategy_dict(y: pd.Series) -> dict[int, int]:
    """Target minority count = max(current, ratio×majority); never undersample."""
    yv = np.asarray(y)
    n_maj = int((yv == 0).sum())
    n_min = int((yv == 1).sum())
    desired = int(dm_config.SMOTE_TARGET_RATIO_TO_MAJORITY * n_maj)
    target = max(n_min, desired)
    return {1: target}


def _classifier_step(random_state: int) -> tuple[str, object]:
    backend = (dm_config.RISK_MODEL_BACKEND or "rf").strip().lower()
    if backend == "lgbm":
        try:
            from lightgbm import LGBMClassifier

            clf = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            return "lgbm", clf
        except ImportError:
            logger.warning("lightgbm not installed; falling back to RandomForest")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return "rf", clf


def _make_estimator_pipeline(
    random_state: int,
    k_neighbors: int,
    sampling_strategy: dict[int, int],
) -> tuple[ImbPipeline, str]:
    """
    SMOTENC runs inside each CV fold (only on training split of that fold),
    then RandomForest or LightGBM. class_weight / balanced_subsample handles skew.
    """
    name, estimator = _classifier_step(random_state)
    pipe = ImbPipeline(
        [
            (
                "smote",
                SMOTENC(
                    categorical_features=_SMOTE_CATEGORICAL_IDX,
                    random_state=random_state,
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                ),
            ),
            (name, estimator),
        ]
    )
    return pipe, name


def _imbalance_description(backend_name: str) -> str:
    base = (
        "SMOTENC (per CV fold; target minority count = max(n_pos, smote_ratio×n_neg)) + isotonic calibration"
    )
    if backend_name == "lgbm":
        return base + " + LightGBM (class_weight=balanced)"
    return base + " + RandomForest (class_weight=balanced_subsample)"


def _product_ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Business-facing: precision@top-k and recall captured in top slice vs random."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    pos_rate = float(y_true.mean()) if n else 0.0
    order = np.argsort(-proba)
    out: dict[str, float] = {}

    for frac, label in [(0.10, "10pct"), (0.20, "20pct")]:
        k = max(1, int(round(n * frac)))
        top = order[:k]
        prec = float(y_true[top].mean()) if k else 0.0
        out[f"precision_at_top_{label}"] = prec
        if pos_rate > 0:
            out[f"lift_at_top_{label}_vs_random"] = prec / pos_rate
        else:
            out[f"lift_at_top_{label}_vs_random"] = 0.0

    k20 = max(1, int(round(n * 0.20)))
    top20 = order[:k20]
    positives = int(y_true.sum())
    out["recall_capture_in_top_20pct"] = (
        float(y_true[top20].sum() / positives) if positives > 0 else 0.0
    )
    return out


def train_failure_model(random_state: int = 42) -> dict:
    df = load_ai4i()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    kn = _smote_k_neighbors(y_train)
    strat = _smote_sampling_strategy_dict(y_train)
    # Report imbalance before/after SMOTENC on full training set (for dashboards only)
    smote_stats = SMOTENC(
        categorical_features=_SMOTE_CATEGORICAL_IDX,
        random_state=random_state,
        sampling_strategy=strat,
        k_neighbors=kn,
    )
    X_sm, y_sm = smote_stats.fit_resample(X_train, y_train)

    pipe, backend_used = _make_estimator_pipeline(random_state, kn, strat)
    clf = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    train_pos = int(y_train.sum())
    test_pos = int(y_test.sum())
    metrics = {
        "risk_model_backend": backend_used,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train_raw": int(len(X_train)),
        "n_train_after_smote_preview": int(len(X_sm)),
        "positives_train_raw": train_pos,
        "positives_train_after_smote_preview": int(y_sm.sum()),
        "positive_rate_train_raw": float(y_train.mean()),
        "positive_rate_train_after_smote_preview": float(y_sm.mean()),
        "n_test": int(len(X_test)),
        "positives_test": test_pos,
        "positive_rate_full": float(y.mean()),
        "smote_target_minority_count_preview": int(strat[1]),
        "smote_ratio_to_majority": float(dm_config.SMOTE_TARGET_RATIO_TO_MAJORITY),
        "imbalance_handling": _imbalance_description(backend_used),
    }
    metrics.update(_product_ranking_metrics(y_test.values, proba))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "feature_names": list(X.columns),
        "metrics": metrics,
        "risk_model_backend": backend_used,
    }
    joblib.dump(payload, MODEL_PATH)
    logger.info(
        "Saved model to %s train_pos=%s test_pos=%s metrics=%s",
        MODEL_PATH,
        train_pos,
        test_pos,
        metrics,
    )
    return payload


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train_failure_model()
    return joblib.load(MODEL_PATH)


def risk_scores_for_work_orders(X: pd.DataFrame) -> np.ndarray:
    payload = load_model()
    clf = payload["model"]
    cols = payload["feature_names"]
    for c in cols:
        if c not in X.columns:
            raise ValueError(f"Scoring missing column {c}; expected {cols}")
    Xa = X[cols].copy()
    return clf.predict_proba(Xa)[:, 1]


def urgency_score(risk: np.ndarray) -> np.ndarray:
    """Scale 0–100 for UI."""
    return np.clip(np.asarray(risk, dtype=float) * 100.0, 0.0, 100.0)
