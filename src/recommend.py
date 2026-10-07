"""Track 3 - Recommendation Systems: the Retention Recommender.

Hybrid recommender of retention interventions:
* Content-based  match the employee's top SHAP risk drivers to the features
                 each intervention targets. Runs on REAL model output.
* Collaborative  Funk SVD (matrix factorisation) on an employee x intervention
                 effectiveness matrix. The IBM data has no intervention history,
                 so this matrix is SIMULATED from rules + noise. Say so openly.
* Elasticity     the smallest pay rise that brings P(leave) below the
                 threshold (analogue of price elasticity). This is model
                 SENSITIVITY, not a causal effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------
# Intervention catalogue. cost_months = TEAM ASSUMPTION (months of salary).
# --------------------------------------------------------------------------
INTERVENTIONS = pd.DataFrame([
    {"id": "workload", "name": "Workload rebalance / overtime cap", "cost_months": 0.5,
     "targets": ["OverTime", "WorkLifeBalance"]},
    {"id": "promotion", "name": "Promotion / level review", "cost_months": 2.0,
     "targets": ["YearsSinceLastPromotion", "PromotionLagRatio", "JobLevel", "YearsInCurrentRole"]},
    {"id": "pay", "name": "Compensation adjustment", "cost_months": 1.5,
     "targets": ["MonthlyIncome", "PayGapPct", "PercentSalaryHike"]},
    {"id": "equity", "name": "Stock / retention grant", "cost_months": 2.0,
     "targets": ["StockOptionLevel"]},
    {"id": "manager", "name": "Manager 1:1s and team fit review", "cost_months": 0.25,
     "targets": ["RelationshipSatisfaction", "EnvironmentSatisfaction", "ManagerTenureRatio", "YearsWithCurrManager"]},
    {"id": "growth", "name": "Training and mentoring plan", "cost_months": 0.5,
     "targets": ["TrainingTimesLastYear", "JobInvolvement", "JobSatisfaction"]},
    {"id": "flex", "name": "Flexible / remote work, less travel", "cost_months": 0.25,
     "targets": ["DistanceFromHome", "BusinessTravel", "WorkLifeBalance"]},
    {"id": "rotation", "name": "Internal role rotation", "cost_months": 1.0,
     "targets": ["YearsInCurrentRole", "JobSatisfaction", "JobRole"]},
])

# +1: higher value = worse for retention; -1: lower value = worse.
# Used only to SIMULATE the collaborative-filtering matrix.
RISK_DIRECTION = {
    "OverTime": 1, "WorkLifeBalance": -1, "YearsSinceLastPromotion": 1, "PromotionLagRatio": 1,
    "JobLevel": -1, "YearsInCurrentRole": 1, "MonthlyIncome": -1, "PercentSalaryHike": -1,
    "StockOptionLevel": -1, "RelationshipSatisfaction": -1, "EnvironmentSatisfaction": -1,
    "ManagerTenureRatio": -1, "YearsWithCurrManager": -1, "TrainingTimesLastYear": -1,
    "JobInvolvement": -1, "JobSatisfaction": -1, "DistanceFromHome": 1, "BusinessTravel": 1,
}


# --------------------------------------------------------------------------
# Content-based
# --------------------------------------------------------------------------
def content_scores(drivers: pd.DataFrame) -> pd.Series:
    """Score = sum of POSITIVE SHAP values (risk-raising) of each intervention's targets."""
    risk = drivers[drivers.shap > 0].set_index("feature")["shap"]
    scores = INTERVENTIONS.set_index("id")["targets"].apply(lambda t: risk.reindex(t).fillna(0).sum())
    return scores.rename("content_score")


# --------------------------------------------------------------------------
# Collaborative filtering (on SIMULATED data)
# --------------------------------------------------------------------------
# Simulation assumption: employees differ in four underlying NEEDS, computed
# from real features, and each intervention serves some needs more than others.
NEED_GROUPS = {
    "career": ["YearsSinceLastPromotion", "PromotionLagRatio", "YearsInCurrentRole", "JobLevel", "TrainingTimesLastYear"],
    "worklife": ["OverTime", "WorkLifeBalance", "DistanceFromHome", "BusinessTravel"],
    "reward": ["MonthlyIncome", "PercentSalaryHike", "StockOptionLevel"],
    "relationship": ["RelationshipSatisfaction", "EnvironmentSatisfaction", "JobSatisfaction", "JobInvolvement"],
}
INTERVENTION_LOADINGS = pd.DataFrame({  # rows = interventions, columns = needs
    "career":       [0.0, 1.0, 0.2, 0.3, 0.0, 0.8, 0.0, 0.7],
    "worklife":     [1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 0.2],
    "reward":       [0.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "relationship": [0.2, 0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.5],
}, index=INTERVENTIONS.id)


def need_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Average 'badness' z-score per need (higher = stronger need)."""
    cols = [f for f in RISK_DIRECTION if f in df.columns]
    z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0).replace(0, 1)
    badness = z * pd.Series(RISK_DIRECTION)[cols]
    return pd.DataFrame({n: badness[[f for f in fs if f in badness]].mean(axis=1)
                         for n, fs in NEED_GROUPS.items()})


def simulate_response_matrix(df: pd.DataFrame, observed_frac: float = 0.5,
                             seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """SIMULATED employee x intervention ratings (1-5 = how well it worked).

    rating = 3 + needs @ loadings + noise, rounded and clipped to 1-5. Then
    (1 - observed_frac) of cells are hidden, as real interaction data is sparse.

    Q&A point: the simulation is low-rank BY DESIGN (4 needs), so collaborative
    filtering doing well here proves the pipeline works, not that the
    interventions work. Real deployment needs real intervention outcomes.
    """
    rng = np.random.default_rng(seed)
    signal = need_scores(df).to_numpy() @ INTERVENTION_LOADINGS.T.to_numpy()
    R = pd.DataFrame(np.clip(np.round(3 + 1.5 * signal + rng.normal(0, 0.6, signal.shape)), 1, 5),
                     index=df.index, columns=INTERVENTION_LOADINGS.index)
    return R.mask(rng.random(R.shape) > observed_frac)


class FunkSVD:
    """Matrix factorisation trained with SGD on observed cells only.

        r_hat(u, i) = mu + b_u + b_i + p_u . q_i
    mu = global mean, b = biases, p/q = latent factors, L2 penalty `reg`.
    """

    def __init__(self, n_factors: int = 4, lr: float = 0.01, reg: float = 0.05,
                 epochs: int = 30, seed: int = config.RANDOM_STATE):
        self.n_factors, self.lr, self.reg, self.epochs, self.seed = n_factors, lr, reg, epochs, seed

    def fit(self, R: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        M = R.to_numpy(dtype=float)
        n_u, n_i = M.shape
        self.mu_ = np.nanmean(M)
        self.bu_, self.bi_ = np.zeros(n_u), np.zeros(n_i)
        self.P_ = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_i, self.n_factors))
        users, items = np.where(~np.isnan(M))
        for _ in range(self.epochs):
            for k in rng.permutation(len(users)):
                u, i = users[k], items[k]
                err = M[u, i] - (self.mu_ + self.bu_[u] + self.bi_[i] + self.P_[u] @ self.Q_[i])
                self.bu_[u] += self.lr * (err - self.reg * self.bu_[u])
                self.bi_[i] += self.lr * (err - self.reg * self.bi_[i])
                pu = self.P_[u].copy()
                self.P_[u] += self.lr * (err * self.Q_[i] - self.reg * self.P_[u])
                self.Q_[i] += self.lr * (err * pu - self.reg * self.Q_[i])
        self.index_, self.columns_ = R.index, R.columns
        return self

    def predict_all(self) -> pd.DataFrame:
        full = self.mu_ + self.bu_[:, None] + self.bi_[None, :] + self.P_ @ self.Q_.T
        return pd.DataFrame(np.clip(full, 1, 5), index=self.index_, columns=self.columns_)


def evaluate_cf(R: pd.DataFrame, test_frac: float = 0.2, seed: int = config.RANDOM_STATE, **svd_kw) -> dict:
    """Hide 20% of observed ratings, train on the rest, then check:
    * RMSE vs a baseline that predicts each intervention's average rating
    * pairwise ranking accuracy: for two held-out interventions of the same
      employee with different true ratings, how often does the model rank the
      better one first? Random = 0.5; the item-mean baseline is shown too.
    """
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~R.isna().to_numpy())
    test_idx = obs[rng.random(len(obs)) < test_frac]
    M = R.to_numpy(dtype=float, copy=True)
    M[test_idx[:, 0], test_idx[:, 1]] = np.nan
    R_train = pd.DataFrame(M, index=R.index, columns=R.columns)
    pred = FunkSVD(**svd_kw).fit(R_train).predict_all().to_numpy()
    truth = R.to_numpy()[test_idx[:, 0], test_idx[:, 1]]
    svd_hat = pred[test_idx[:, 0], test_idx[:, 1]]
    base_hat = np.nanmean(M, axis=0)[test_idx[:, 1]]

    def pairwise_accuracy(hat):
        d = pd.DataFrame({"u": test_idx[:, 0], "r": truth, "p": hat})
        good = total = 0
        for _, g in d.groupby("u"):
            r, p = g.r.to_numpy(), g.p.to_numpy()
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    if r[a] != r[b]:
                        total += 1
                        good += (p[a] - p[b]) * (r[a] - r[b]) > 0
        return float(good / total) if total else None

    return {
        "rmse_svd": float(np.sqrt(np.mean((truth - svd_hat) ** 2))),
        "rmse_item_mean_baseline": float(np.sqrt(np.mean((truth - base_hat) ** 2))),
        "pairwise_ranking_svd": pairwise_accuracy(svd_hat),
        "pairwise_ranking_item_mean": pairwise_accuracy(base_hat),
        "pairwise_ranking_random": 0.5,
        "n_test_ratings": int(len(truth)),
        "data_note": "SIMULATED interaction matrix",
        # TODO(team): sweep observed_frac (0.2 -> 0.8) and plot RMSE vs density:
        # it shows how much outcome data the company must collect before CF helps.
    }


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------
def hybrid_recommend(content: pd.Series, cf_row: pd.Series, alpha: float = 0.6, top: int = 3) -> pd.DataFrame:
    """alpha * content + (1 - alpha) * CF, each min-max scaled to 0-1."""
    def mm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    out = INTERVENTIONS.set_index("id")[["name", "cost_months", "targets"]].copy()
    out["content_score"] = mm(content)
    out["cf_score"] = mm(cf_row.reindex(out.index))
    out["hybrid_score"] = alpha * out.content_score + (1 - alpha) * out.cf_score
    return out.sort_values("hybrid_score", ascending=False).head(top)


def salary_elasticity(predict_proba, X_row: pd.DataFrame, threshold: float,
                      raises=np.round(np.arange(0, 0.31, 0.02), 2)) -> tuple[pd.DataFrame, float | None]:
    """P(leave) as monthly income rises 0-30%. Returns the curve and the smallest
    rise that crosses below the threshold (None if none does)."""
    rows = []
    for r in raises:
        x = X_row.copy()
        x["MonthlyIncome"] = x["MonthlyIncome"] * (1 + r)
        rows.append({"raise_pct": r, "p_leave": float(predict_proba(x)[0, 1])})
    curve = pd.DataFrame(rows)
    below = curve[curve.p_leave < threshold]
    return curve, (float(below.raise_pct.iloc[0]) if len(below) else None)
