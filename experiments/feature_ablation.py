"""Phase 1: does each engineered feature (and do the sensitive features) help?

Method
- Train part only (same split as src.data.split), so the test set stays untouched.
- The SAME 15 folds (5-fold x 3 repeats) for every variant, so we can compare
  fold by fold (a paired comparison) instead of just comparing two averages.
- Two model families: Logistic Regression (linear) and XGBoost (trees, fixed
  middle-of-the-search-space settings, not tuned). A ratio feature can be
  useless to one family and useful to the other, so we check both.
- Strength of evidence: Nadeau & Bengio (2003) corrected resampled t-test.
  A plain t-test on CV folds is too optimistic because the training sets
  overlap; the correction inflates the variance by (1/k + n_test/n_train).

Decision rule (fixed BEFORE looking at the results):
  1. Does removing the feature lower mean PR-AUC in BOTH model families?
       no  -> DROP from the model (no consistent benefit)
  2. Is the corrected p-value < 0.10 in at least one family?
       yes -> KEEP
       no  -> DROP from the model (gain is indistinguishable from fold noise;
              the simpler model is easier to explain)
Dropped features are still computed by engineer_features(): the recommender
and the error analysis use them as descriptive columns.

Run from the repo root:  python -m experiments.feature_ablation
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src import config
from src.data import ENGINEERED, feature_lists, load_raw, prepare, split
from src.model import build_pipeline

RS = config.RANDOM_STATE
train, _ = split(prepare(load_raw()))
X, y = train.drop(columns=[config.TARGET]), train[config.TARGET]
pos_weight = float((y == 0).sum() / (y == 1).sum())

MODELS = {
    "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight, eval_metric="logloss",
                             n_jobs=-1, random_state=RS, tree_method="hist"),
}
CV = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=RS)
TEST_TRAIN_RATIO = 1 / (config.CV_FOLDS - 1)


def lists_for(engineered, exclude_sensitive=True):
    """Feature lists with only the given engineered features as model inputs."""
    lists = feature_lists(exclude_sensitive=exclude_sensitive)
    for key in ("numeric", "model_numeric"):
        base = [c for c in lists[key] if c not in ENGINEERED]
        lists[key] = base + list(engineered)
    return lists


def fold_scores(model, lists):
    pipe = build_pipeline(MODELS[model], lists, False)
    return cross_val_score(pipe, X, y, cv=CV, scoring="average_precision", n_jobs=-1)


def paired(base, other):
    """Gain of `base` over `other`: mean, folds won, corrected t-test p-value."""
    d = base - other
    k = len(d)
    se = np.sqrt((1 / k + TEST_TRAIN_RATIO) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_gain": float(d.mean()), "folds_won": int((d > 0).sum()), "folds": k,
            "p_corrected": float(p)}


results = {}
for model in MODELS:
    base = fold_scores(model, lists_for(ENGINEERED))
    r = {"all_engineered": {"mean": float(base.mean()), "std": float(base.std())}}
    none = fold_scores(model, lists_for([]))
    r["no_engineered"] = {"mean": float(none.mean()), "std": float(none.std()),
                          "gain_from_all_engineered": paired(base, none)}
    for f in ENGINEERED:
        s = fold_scores(model, lists_for([e for e in ENGINEERED if e != f]))
        r[f"without_{f}"] = {"mean": float(s.mean()), "std": float(s.std()),
                             "gain_from_feature": paired(base, s)}
    sens = fold_scores(model, lists_for(ENGINEERED, exclude_sensitive=False))
    r["sensitive_included"] = {"mean": float(sens.mean()), "std": float(sens.std()),
                               "gain_from_sensitive": paired(sens, base)}
    results[model] = r

decisions = {}
for f in ENGINEERED:
    g = [results[m][f"without_{f}"]["gain_from_feature"] for m in MODELS]
    helps_both = all(x["mean_gain"] > 0 for x in g)
    strong = any(x["p_corrected"] < 0.10 for x in g)
    decisions[f] = "KEEP" if helps_both and strong else "DROP"
results["decisions"] = decisions

for model in MODELS:
    print(f"\n== {model} ==")
    for k, v in results[model].items():
        line = f"{k:30s} PR-AUC {v['mean']:.3f} ± {v['std']:.3f}"
        g = next((v[x] for x in v if x.startswith("gain")), None)
        if g:
            line += (f"   gain {g['mean_gain']:+.4f}  won {g['folds_won']}/{g['folds']}"
                     f"  p={g['p_corrected']:.2f}")
        print(line)
print("\nDecisions:", decisions)

Path("reports").mkdir(exist_ok=True)
Path("reports/feature_ablation.json").write_text(json.dumps(results, indent=2))