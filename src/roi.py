"""Business layer: turns probabilities into money and decisions (Track 2).

Cost model for one employee with leave probability p:
    not flagged          expected cost = p * R
    flagged (intervene)  expected cost = C + p * R * (1 - s)
where R = replacement cost, C = intervention cost, s = intervention success rate.

The threshold that minimises total expected cost is chosen on out-of-fold
TRAINING predictions, then reported on the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def replacement_cost(df: pd.DataFrame, multiplier: float = config.REPLACEMENT_COST_MULTIPLIER) -> np.ndarray:
    return df["MonthlyIncome"].to_numpy() * 12 * multiplier


def intervention_cost(df: pd.DataFrame, months: float = config.INTERVENTION_COST_MONTHS) -> np.ndarray:
    return df["MonthlyIncome"].to_numpy() * months


def realised_cost(y_true, flagged, repl, interv, success: float) -> float:
    """Cost of a flagging policy given what actually happened (y_true)."""
    y, f = np.asarray(y_true), np.asarray(flagged).astype(bool)
    cost_flagged = interv + y * repl * (1 - success)
    cost_not_flagged = y * repl
    return float(np.where(f, cost_flagged, cost_not_flagged).sum())


def threshold_curve(y_true, proba, repl, interv, success: float, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    rows = [{"threshold": t, "total_cost": realised_cost(y_true, proba >= t, repl, interv, success),
             "flagged": int((proba >= t).sum())} for t in grid]
    return pd.DataFrame(rows)


def optimise_threshold(y_true, proba, repl, interv, success: float) -> tuple[float, pd.DataFrame]:
    curve = threshold_curve(y_true, proba, repl, interv, success)
    return float(curve.loc[curve.total_cost.idxmin(), "threshold"]), curve


def roi_summary(y_true, proba, threshold, repl, interv, success: float) -> dict:
    """Compare the model policy with 'do nothing' and 'intervene with everyone'."""
    y = np.asarray(y_true)
    do_nothing = realised_cost(y, np.zeros_like(y), repl, interv, success)
    everyone = realised_cost(y, np.ones_like(y), repl, interv, success)
    model = realised_cost(y, proba >= threshold, repl, interv, success)
    return {
        "cost_do_nothing": do_nothing,
        "cost_intervene_all": everyone,
        "cost_model_policy": model,
        "savings_vs_do_nothing": do_nothing - model,
        "savings_pct": (do_nothing - model) / do_nothing if do_nothing else 0.0,
        "employees_flagged": int((proba >= threshold).sum()),
        "intervention_spend": float(np.asarray(interv)[proba >= threshold].sum()),
    }


def budget_allocation(df: pd.DataFrame, proba, repl, interv, success: float, budget: float) -> pd.DataFrame:
    """Greedy knapsack: rank employees by expected net benefit per unit spent
    and fund interventions until the budget runs out.

    expected benefit = p * R * s   (replacement cost we expect to avoid)
    net benefit      = benefit - C
    """
    d = df.copy()
    d["p_leave"] = proba
    d["replacement_cost"] = repl
    d["intervention_cost"] = interv
    d["expected_benefit"] = d.p_leave * d.replacement_cost * success
    d["net_benefit"] = d.expected_benefit - d.intervention_cost
    d["roi_ratio"] = d.expected_benefit / d.intervention_cost
    d = d[d.net_benefit > 0].sort_values("roi_ratio", ascending=False)
    d["cumulative_spend"] = d.intervention_cost.cumsum()
    return d[d.cumulative_spend <= budget]


# --------------------------------------------------------------------------
# Phase 2: sanity check and sensitivity
# --------------------------------------------------------------------------
def break_even_threshold(months: float = config.INTERVENTION_COST_MONTHS,
                         multiplier: float = config.REPLACEMENT_COST_MULTIPLIER,
                         success: float = config.INTERVENTION_SUCCESS_RATE) -> float:
    """Flag someone when the expected saving beats the intervention cost:
        p * R * s > C   ->   p > C / (R * s)
    Both C and R scale with the same salary (C = months x monthly pay,
    R = 12 x multiplier x monthly pay), so salary cancels out:
        p* = months / (12 * multiplier * success)
    With CALIBRATED probabilities the data-driven threshold should land near p*."""
    return months / (12 * multiplier * success)


def sensitivity_grid(train: pd.DataFrame, y_train, oof, test: pd.DataFrame, y_test, p_test,
                     multipliers=(0.5, 1.0, 2.0), successes=(0.2, 0.4, 0.6),
                     months_grid=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Re-run the whole decision for every combination of the three business
    assumptions: re-optimise the threshold on out-of-fold TRAIN predictions,
    then score the policy on TEST. Shows where the model stops paying off."""
    rows = []
    for m in multipliers:
        for s in successes:
            for mo in months_grid:
                t, _ = optimise_threshold(y_train, oof, replacement_cost(train, m),
                                          intervention_cost(train, mo), s)
                r = roi_summary(y_test, p_test, t, replacement_cost(test, m),
                                intervention_cost(test, mo), s)
                rows.append({"replacement_multiplier": m, "success_rate": s,
                             "intervention_months": mo, "threshold": t,
                             "break_even_threshold": break_even_threshold(mo, m, s),
                             "flagged": r["employees_flagged"],
                             "savings_vs_do_nothing": r["savings_vs_do_nothing"],
                             "savings_pct": r["savings_pct"]})
    return pd.DataFrame(rows)
