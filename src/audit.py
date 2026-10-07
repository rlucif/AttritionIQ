"""Audit trail: every prediction shown in the app is logged with the model
version, the inputs that mattered and the decision. Answers the question
"why was this person flagged, and by which model?" after the fact."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src import config

COLUMNS = ["timestamp_utc", "model_name", "model_version", "employee_id", "p_leave",
           "threshold", "decision", "top_drivers", "user_action"]


def log_prediction(model_name: str, model_version: str, employee_id, p_leave: float,
                   threshold: float, drivers: pd.DataFrame, user_action: str = "viewed") -> None:
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_version": model_version,
        "employee_id": employee_id,
        "p_leave": round(float(p_leave), 4),
        "threshold": threshold,
        "decision": "flag" if p_leave >= threshold else "no flag",
        "top_drivers": json.dumps({r.feature: round(float(r.shap), 3) for r in drivers.itertuples()}),
        "user_action": user_action,
    }
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        config.AUDIT_LOG, mode="a", header=not config.AUDIT_LOG.exists(), index=False)


def read_log() -> pd.DataFrame:
    if not config.AUDIT_LOG.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(config.AUDIT_LOG)
