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
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
    return out


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
    d["AgeBand"] = pd.cut(d["Age"], [0, 30, 40, 50, 100], labels=["<30", "30-39", "40-49", "50+"])
    rows = []
    for attr in ["Gender", "AgeBand"]:
        for grp, g in d.groupby(attr, observed=True):
            leavers = g[g[config.TARGET] == 1]
            rows.append({"attribute": attr, "group": str(grp), "n": len(g),
                         "actual_attrition": g[config.TARGET].mean(),
                         "selection_rate": g.flagged.mean(),
                         "recall": leavers.flagged.mean() if len(leavers) else np.nan})
    return pd.DataFrame(rows)


def error_analysis(df: pd.DataFrame, proba: np.ndarray, threshold: float,
                   features=("OverTime", "MonthlyIncome", "YearsAtCompany", "JobLevel",
                             "SatisfactionIndex", "YearsSinceLastPromotion", "DistanceFromHome")) -> pd.DataFrame:
    """Average profile of true positives, false negatives (missed leavers) and
    false positives (false alarms). Missed leavers who look 'fine' on every
    feature point to causes the data does not capture (e.g. family relocation).
    TODO(team): read 5-10 individual false negatives with their SHAP drivers
    and write up the patterns for the technical deep dive."""
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


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v
