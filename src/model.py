"""Track 2 - Decision Automation: the attrition risk engine.

Steps (all called from train.py):
    1. build_pipeline   PayGapAdder -> scaler/one-hot -> (SMOTE) -> classifier
    2. tune             RandomizedSearchCV, stratified 5-fold, scored on PR-AUC
    3. repeated_cv      5-fold x 3 repeats for mean +/- std of each metric
    4. oof_probabilities  out-of-fold predictions on TRAIN, used to pick the
                          cost-optimal threshold (never the test set)
    5. calibrate        Platt scaling so 0.7 means ~70%
    6. evaluate         final, one-time check on the held-out test set
    7. explain          SHAP values per employee
    8. fairness_report  recall and selection rate by gender / age band
    9. error analysis   group profiles + one row per missed leaver with SHAP drivers

Phase 2 additions: per-fold scores and paired tests (compare_models), the
model-selection rule (select_model), nested CV for the chosen model
(nested_cv), expected calibration error, and bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data import base_feature, build_preprocessor
from src.forecast import PayGapAdder

RS = config.RANDOM_STATE


# --------------------------------------------------------------------------
# 1. Candidate models and their search spaces
# --------------------------------------------------------------------------
def model_specs(pos_weight: float) -> dict:
    """Four candidates, from simplest to most flexible.

    Class imbalance (~16% leavers) is handled per model:
      LR / RF   class_weight='balanced' (errors on leavers cost more)
      XGBoost   scale_pos_weight = negatives / positives
      MLP       SMOTE oversampling (MLPClassifier has no class weights)
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(solver="saga", max_iter=5000,
                                            class_weight="balanced", random_state=RS),
            # l1_ratio: 0 = Ridge-style (L2), 1 = Lasso-style (L1), between = elastic net
            "params": {"C": loguniform(1e-3, 1e2), "l1_ratio": uniform(0, 1)},
            "smote": False,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=RS),
            "params": {"n_estimators": randint(200, 600), "max_depth": randint(3, 15),
                       "min_samples_leaf": randint(1, 20), "max_features": ["sqrt", 0.3, 0.5]},
            "smote": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                       n_jobs=-1, random_state=RS, tree_method="hist"),
            "params": {"n_estimators": randint(100, 600), "max_depth": randint(2, 6),
                       "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
                       "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 10),
                       "reg_lambda": loguniform(0.1, 10)},
            "smote": False,
        },
        "Neural Network (MLP)": {
            "estimator": MLPClassifier(max_iter=1000, early_stopping=True, random_state=RS),
            "params": {"hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
                       "alpha": loguniform(1e-4, 1e0), "learning_rate_init": loguniform(1e-4, 1e-2)},
            "smote": True,
        },
    }


def build_pipeline(estimator, lists: dict, smote: bool = False) -> ImbPipeline:
    steps = [("paygap", PayGapAdder()), ("prep", build_preprocessor(lists))]
    if smote:
        steps.append(("smote", SMOTE(random_state=RS)))
    steps.append(("clf", clone(estimator)))
    return ImbPipeline(steps)


# --------------------------------------------------------------------------
# 2-4. Tuning and cross-validation
# --------------------------------------------------------------------------
def tune(name: str, spec: dict, X, y, lists: dict, n_iter: int = config.N_ITER_SEARCH):
    pipe = build_pipeline(spec["estimator"], lists, spec["smote"])
    search = RandomizedSearchCV(
        pipe, {f"clf__{k}": v for k, v in spec["params"].items()},
        n_iter=n_iter, scoring=config.PRIMARY_METRIC, n_jobs=-1, random_state=RS,
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS),
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": {k.replace("clf__", ""): _jsonable(v) for k, v in search.best_params_.items()},
        "best_cv_pr_auc": float(search.best_score_),
    }


def repeated_cv(estimator, X, y) -> dict:
    """Mean and std across 5 folds x 3 repeats: shows how stable the model is."""
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=RS)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall",
               "precision": "precision", "f1": "f1", "brier": "neg_brier_score"}
    s = cross_validate(clone(estimator), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    out = {}
    for k in scoring:
        vals = s[f"test_{k}"] * (-1 if k == "brier" else 1)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    # Same random_state -> every model sees the SAME 15 folds, so the per-fold
    # scores can be compared pairwise (compare_models).
    out["pr_auc"]["folds"] = [float(v) for v in s["test_pr_auc"]]
    return out


# --------------------------------------------------------------------------
# Model selection (Phase 2). Rule fixed BEFORE the full run:
#   1. rank by mean repeated-CV PR-AUC
#   2. every model whose mean is within ONE std (of the best model's fold
#      scores) of the best counts as tied with it
#   3. among tied models pick the simplest: LR < RF < XGBoost < MLP
#      (fewer moving parts, exact SHAP, easier to defend in the Q&A)
# The corrected paired t-test is reported as supporting evidence only.
# --------------------------------------------------------------------------
COMPLEXITY_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network (MLP)"]


def corrected_paired_ttest(a, b, test_train_ratio: float = 1 / (config.CV_FOLDS - 1)) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores.
    A plain t-test is too optimistic because CV training sets overlap; the
    correction inflates the variance by (1/k + n_test/n_train)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    se = np.sqrt((1 / k + test_train_ratio) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    return {"mean_diff": float(d.mean()), "folds_won": int((d > 0).sum()), "folds": k,
            "p_corrected": float(2 * stats.t.sf(abs(t), df=k - 1))}


def select_model(comparison: dict) -> tuple[str, dict]:
    """Apply the selection rule above. Returns (chosen name, explanation)."""
    means = {n: c["repeated_cv"]["pr_auc"]["mean"] for n, c in comparison.items()}
    best = max(means, key=means.get)
    tol = comparison[best]["repeated_cv"]["pr_auc"]["std"]
    tied = [n for n in means if means[n] >= means[best] - tol]
    chosen = next(n for n in COMPLEXITY_ORDER if n in tied)
    best_folds = comparison[best]["repeated_cv"]["pr_auc"]["folds"]
    vs_best = {n: corrected_paired_ttest(best_folds, c["repeated_cv"]["pr_auc"]["folds"])
               for n, c in comparison.items() if n != best}
    return chosen, {
        "rule": "highest mean CV PR-AUC; models within 1 std of the best are tied; "
                "among tied, simplest wins (LR < RF < XGBoost < MLP)",
        "highest_mean": best, "tolerance_1std": float(tol),
        "tied_with_best": tied, "chosen": chosen,
        "ranking": sorted(means, key=means.get, reverse=True),
        "best_vs_others_paired": vs_best,
    }


def nested_cv(name: str, spec: dict, X, y, lists: dict, n_iter: int = config.N_ITER_SEARCH,
              outer_folds: int = config.CV_FOLDS) -> dict:
    """Tuning inside each outer fold, scoring on the outer fold. The gap to the
    plain repeated-CV score measures how optimistic 'tune and evaluate on the
    same data' was. Outer folds use a different seed from the tuning folds."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=RS + 1)
    X = X.reset_index(drop=True)
    scores = []
    for tr, va in outer.split(X, y):
        best, _ = tune(name, spec, X.iloc[tr], y[tr], lists, n_iter=n_iter)
        scores.append(average_precision_score(y[va], best.predict_proba(X.iloc[va])[:, 1]))
    return {"model": name, "outer_folds": outer_folds,
            "pr_auc": {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                       "folds": [float(s) for s in scores]}}


def oof_probabilities(estimator, X, y) -> np.ndarray:
    """Out-of-fold P(leave) for every training row (each predicted by a model
    that never saw it). Used to choose the decision threshold."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


# --------------------------------------------------------------------------
# 5-6. Calibration and final evaluation
# --------------------------------------------------------------------------
def calibrate(estimator, X, y, method: str = "sigmoid"):
    """Platt (sigmoid) scaling. Isotonic needs more data than ~190 leavers."""
    cal = CalibratedClassifierCV(clone(estimator), method=method,
                                 cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS))
    return cal.fit(X, y)


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baseline_pr_auc": float(np.mean(y_true)),  # a random model scores the base rate
    }


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Weighted average gap between predicted and observed rate across equal-
    size (quantile) bins. 0 = perfectly calibrated. 0.05 = off by 5 points."""
    y, p = np.asarray(y_true), np.asarray(proba)
    bins = np.array_split(np.argsort(p), n_bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def calibration_report(y_true, raw, calibrated) -> dict:
    out = {}
    for label, p in [("uncalibrated", raw), ("calibrated", calibrated)]:
        out[label] = {"brier": float(brier_score_loss(y_true, p)),
                      "ece": expected_calibration_error(y_true, p),
                      "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y_true))}
    return out


def bootstrap_ci(y_true, proba, threshold: float, extra=None, n_boot: int = 1000,
                 seed: int = RS) -> dict:
    """95% percentile bootstrap intervals for test-set metrics. The test set has
    ~47 leavers, so a single number hides a lot of uncertainty.
    extra: optional {name: f(idx) -> float} for metrics that need other arrays."""
    y, p = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    draws = {"pr_auc": [], "roc_auc": [], "recall": [], "precision": []}
    draws.update({k: [] for k in (extra or {})})
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        flag = pb >= threshold
        draws["pr_auc"].append(average_precision_score(yb, pb))
        draws["roc_auc"].append(roc_auc_score(yb, pb))
        draws["recall"].append(flag[yb == 1].mean())
        draws["precision"].append(yb[flag].mean() if flag.any() else 0.0)
        for k, f in (extra or {}).items():
            draws[k].append(f(idx))
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for k, v in draws.items()}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; behaves sensibly when n is small or p is 0."""
    if n == 0:
        return (np.nan, np.nan)
    ph = successes / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# --------------------------------------------------------------------------
# 7. Explainability (SHAP)
# --------------------------------------------------------------------------
class Explainer:
    """SHAP explanations for the UNCALIBRATED best pipeline.

    Q&A point: calibration is a monotonic rescaling of the score, so it does not
    change which features push risk up or down. SHAP values are in log-odds for
    tree / linear models (additive), probability for the MLP.
    """

    def __init__(self, pipeline, X_background: pd.DataFrame, lists: dict, n_background: int = 100):
        self.lists = lists
        self.transform = Pipeline([(n, s) for n, s in pipeline.steps if n in ("paygap", "prep")])
        self.clf = pipeline.named_steps["clf"]
        bg = self.transform.transform(X_background.sample(min(n_background, len(X_background)), random_state=RS))
        self.feature_names = list(bg.columns)
        if isinstance(self.clf, (XGBClassifier, RandomForestClassifier)):
            self._exp = shap.TreeExplainer(self.clf)
            self.kind = "tree"
        elif isinstance(self.clf, LogisticRegression):
            self._exp = shap.LinearExplainer(self.clf, bg)
            self.kind = "linear"
        else:
            self._exp = shap.Explainer(lambda a: self.clf.predict_proba(a)[:, 1], bg)
            self.kind = "model-agnostic"

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        Xt = self.transform.transform(X)
        ex = self._exp(Xt)
        if ex.values.ndim == 3:  # RandomForest returns one set per class
            ex = ex[:, :, 1]
        ex.feature_names = self.feature_names
        return ex

    def drivers(self, X_row: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Top-k features for one employee, grouped back to original features."""
        ex = self.explain(X_row)
        s = pd.Series(ex.values[0], index=self.feature_names)
        grouped = s.groupby([base_feature(f, self.lists) for f in s.index]).sum()
        raw = X_row.iloc[0]
        df = pd.DataFrame({"feature": grouped.index, "shap": grouped.values})
        df["value"] = [raw.get(f, np.nan) if f != "PayGapPct" else
                       float(self.transform.transform(X_row)["PayGapPct"].iloc[0]) for f in df.feature]
        df["direction"] = np.where(df.shap > 0, "raises risk", "lowers risk")
        return df.reindex(df.shap.abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        ex = self.explain(X)
        s = pd.Series(np.abs(ex.values).mean(axis=0), index=self.feature_names)
        return s.groupby([base_feature(f, self.lists) for f in s.index]).sum().sort_values(ascending=False)


# --------------------------------------------------------------------------
# 8. Fairness
# --------------------------------------------------------------------------
def fairness_report(df: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Recall (share of real leavers caught) and selection rate (share flagged)
    by group. Big gaps between groups need an explanation in the Q&A."""
    d = df[[config.TARGET, "Gender", "Age"]].copy()
    d["flagged"] = (proba >= threshold).astype(int)
    d["Gender"] = d["Gender"].map({1: "Male", 0: "Female"}).fillna(d["Gender"])
    # right=False -> [0,30) [30,40) [40,50) [50,100): the labels match the bins.
    # (pd.cut defaults to right-closed bins, which put 50-year-olds in "40-49".)
    d["AgeBand"] = pd.cut(d["Age"], [0, 30, 40, 50, 100], right=False,
                          labels=["<30", "30-39", "40-49", "50+"])
    d["p"] = proba
    rows = []
    for attr in ["Gender", "AgeBand"]:
        for grp, g in d.groupby(attr, observed=True):
            leavers = g[g[config.TARGET] == 1]
            lo, hi = wilson_interval(int(leavers.flagged.sum()), len(leavers))
            rows.append({"attribute": attr, "group": str(grp), "n": len(g),
                         "leavers": len(leavers),
                         "actual_attrition": g[config.TARGET].mean(),
                         "mean_predicted": g.p.mean(),   # under-prediction shows up here
                         "selection_rate": g.flagged.mean(),
                         "recall": leavers.flagged.mean() if len(leavers) else np.nan,
                         "recall_ci_low": lo, "recall_ci_high": hi})
    return pd.DataFrame(rows)


def group_shap_gap(explainer: "Explainer", X: pd.DataFrame, mask_group, mask_ref, k: int = 8) -> pd.DataFrame:
    """Mean SHAP contribution (log-odds) for one group of rows minus a reference
    group, per original feature. Negative = the model pushes this group's risk
    DOWN more than the reference group's. Used to explain the 50+ recall gap."""
    ex = explainer.explain(X)
    s = pd.DataFrame(ex.values, columns=explainer.feature_names, index=X.index)
    s = s.T.groupby([base_feature(f, explainer.lists) for f in s.columns]).sum().T
    out = pd.DataFrame({"group_mean_shap": s[np.asarray(mask_group)].mean(),
                        "reference_mean_shap": s[np.asarray(mask_ref)].mean()})
    out["gap"] = out.group_mean_shap - out.reference_mean_shap
    return out.reindex(out.gap.abs().sort_values(ascending=False).index).head(k)


def error_analysis(df: pd.DataFrame, proba: np.ndarray, threshold: float,
                   features=("OverTime", "MonthlyIncome", "YearsAtCompany", "JobLevel",
                             "SatisfactionIndex", "YearsSinceLastPromotion", "DistanceFromHome")) -> pd.DataFrame:
    """Average profile of true positives, false negatives (missed leavers) and
    false positives (false alarms). Missed leavers who look 'fine' on every
    feature point to causes the data does not capture (e.g. family relocation).
    Individual rows: missed_leavers(). Written patterns: docs/PHASE2_RISK_MODEL.md."""
    d = df[list(features) + [config.TARGET]].copy()
    flagged = proba >= threshold
    y = d[config.TARGET].to_numpy() == 1
    d["outcome"] = np.select([y & flagged, y & ~flagged, ~y & flagged],
                             ["true positive", "false negative (missed)", "false positive (false alarm)"],
                             "true negative")
    d["p_leave"] = proba
    out = d.groupby("outcome")[list(features) + ["p_leave"]].mean()
    out.insert(0, "count", d.groupby("outcome").size())
    return out


MISSED_LEAVER_COLUMNS = [
    "EmployeeNumber", "Age", "Gender", "MaritalStatus", "Department", "JobRole", "JobLevel",
    "MonthlyIncome", "OverTime", "BusinessTravel", "DistanceFromHome", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears",
    "NumCompaniesWorked", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "RelationshipSatisfaction", "JobInvolvement", "StockOptionLevel", "PercentSalaryHike",
    "PerformanceRating", "TrainingTimesLastYear",
]


def missed_leavers(df: pd.DataFrame, X: pd.DataFrame, proba: np.ndarray, threshold: float,
                   explainer: "Explainer", k: int = 3) -> pd.DataFrame:
    """One row per false negative: who they are, P(leave), and the k features
    that pushed their risk DOWN most and UP most. Read these one by one: the
    patterns are the error-analysis write-up."""
    y = df[config.TARGET].to_numpy() == 1
    idx = np.where(y & (proba < threshold))[0]
    if not len(idx):
        return pd.DataFrame()
    ex = explainer.explain(X.iloc[idx])
    s = pd.DataFrame(ex.values, columns=explainer.feature_names)
    s = s.T.groupby([base_feature(f, explainer.lists) for f in s.columns]).sum().T
    lowered = s.apply(lambda r: "; ".join(f"{f} {v:+.2f}" for f, v in r.nsmallest(k).items()), axis=1)
    raised = s.apply(lambda r: "; ".join(f"{f} {v:+.2f}" for f, v in r.nlargest(k).items() if v > 0), axis=1)
    out = df.iloc[idx][[c for c in MISSED_LEAVER_COLUMNS if c in df.columns]].reset_index(drop=True)
    out.insert(1, "p_leave", np.round(proba[idx], 3))
    out["gap_to_threshold"] = np.round(threshold - proba[idx], 3)
    out["top_drivers_lowering_risk"] = lowered.to_numpy()
    out["top_drivers_raising_risk"] = raised.to_numpy()
    return out.sort_values("p_leave").reset_index(drop=True)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v
