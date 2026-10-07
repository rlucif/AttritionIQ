"""Phase 3, Track 4 (part 1): Ridge vs Lasso for the fair-pay model.

The skeleton's Lasso picked alpha = 1e-4, the smallest value on its grid, so
the true optimum may lie below it. This widens the grid to 1e-6..1e2 and uses
repeated 5x3 CV (same folds for both models) so the comparison is paired.

Rule (docs/PHASE3_PLAN.md): if the R2 difference is within one std of the
fold-to-fold differences, call it a tie and keep Ridge, which shares weight
across correlated predictors instead of dropping one of them arbitrarily.

Train split only, as in train.py. Run:  python -m experiments.phase3_paymodel
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate

from src import config, data, forecast

ALPHAS = np.logspace(-6, 2, 50)


def main():
    train, _ = data.split(data.prepare(data.load_raw()))
    X, y = train[forecast.PAY_MODEL_FEATURES], np.log(train["MonthlyIncome"])
    cv = RepeatedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE)
    models = {"Ridge": RidgeCV(alphas=ALPHAS),
              "Lasso": LassoCV(alphas=ALPHAS, max_iter=50000, random_state=config.RANDOM_STATE)}
    out, folds, coefs = {}, {}, {}
    for name, reg in models.items():
        sc = cross_validate(forecast._pay_pipeline(reg), X, y, cv=cv, scoring=["r2", "neg_root_mean_squared_error"])
        folds[name] = sc["test_r2"]
        fit = forecast._pay_pipeline(reg).fit(X, y)
        coefs[name] = pd.Series(fit["reg"].coef_, index=fit["pre"].get_feature_names_out())
        a = float(fit["reg"].alpha_)
        out[name] = {"r2_mean": float(sc["test_r2"].mean()), "r2_std": float(sc["test_r2"].std()),
                     "rmse_log_mean": float(-sc["test_neg_root_mean_squared_error"].mean()),
                     "alpha": a, "alpha_on_grid_edge": bool(a in (ALPHAS.min(), ALPHAS.max())),
                     "zeroed": coefs[name][coefs[name].abs() < 1e-8].index.tolist()}
    d = folds["Ridge"] - folds["Lasso"]
    out["paired"] = {"ridge_minus_lasso_mean": float(d.mean()), "std": float(d.std()),
                     "ridge_wins": int((d > 0).sum()), "folds": int(len(d)),
                     "tie": bool(abs(d.mean()) < d.std())}
    out["choice"] = "Ridge" if out["paired"]["tie"] else max(("Ridge", "Lasso"), key=lambda m: out[m]["r2_mean"])
    num = train[forecast.PAY_MODEL_NUMERIC]
    out["numeric_correlations"] = num.corr().round(3).to_dict()
    out["coefficients"] = pd.DataFrame(coefs).round(4).sort_values("Ridge", key=abs, ascending=False).to_dict()
    (config.REPORTS_DIR / "phase3_paymodel.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "coefficients"}, indent=1))
    print(pd.DataFrame(coefs).round(4).to_string())


if __name__ == "__main__":
    main()
