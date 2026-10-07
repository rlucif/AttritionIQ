"""Track 4 - Predictive Operations.

1. Fair-pay model (Ridge vs Lasso regression): predicts what an employee
   "should" earn from level, experience and role. The gap between actual and
   predicted pay becomes a feature (PayGapPct) in the attrition model. It is
   built as a sklearn transformer so it is re-fitted inside every CV fold.
2. Macro attrition forecast (ARIMA, optional Prophet) on the US JOLTS quits
   rate, with rolling-origin time-series cross-validation.
3. Workforce forecast: expected leavers and replacement cost per department
   over the next 12 months.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

PAY_MODEL_NUMERIC = ["JobLevel", "TotalWorkingYears", "Education", "YearsAtCompany"]
PAY_MODEL_CATEGORICAL = ["JobRole", "Department"]
PAY_MODEL_FEATURES = PAY_MODEL_NUMERIC + PAY_MODEL_CATEGORICAL


# --------------------------------------------------------------------------
# 1. Fair-pay model
# --------------------------------------------------------------------------
def _pay_pipeline(regressor) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), PAY_MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PAY_MODEL_CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("reg", regressor)])


class PayGapAdder(BaseEstimator, TransformerMixin):
    """Adds PayGapPct = actual / predicted income - 1.

    Negative = paid below what peers with the same level, experience and role
    earn. Fitted on log(income) because pay is right-skewed.
    """

    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.model_ = _pay_pipeline(RidgeCV(alphas=self.alphas))
        self.model_.fit(X[PAY_MODEL_FEATURES], np.log(X["MonthlyIncome"]))
        return self

    def transform(self, X):
        X = X.copy()
        predicted = np.exp(self.model_.predict(X[PAY_MODEL_FEATURES]))
        X["PayGapPct"] = X["MonthlyIncome"] / predicted - 1
        return X

    def predict_income(self, X):
        return np.exp(self.model_.predict(X[PAY_MODEL_FEATURES]))

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.array(base + ["PayGapPct"])


def pay_model_report(df: pd.DataFrame) -> dict:
    """Compare Ridge (L2) and Lasso (L1) with 5-fold CV on log income."""
    X, y = df[PAY_MODEL_FEATURES], np.log(df["MonthlyIncome"])
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    alphas = np.logspace(-4, 2, 30)
    report = {}
    for name, reg in [("Ridge", RidgeCV(alphas=alphas)),
                      ("Lasso", LassoCV(alphas=alphas, max_iter=20000, random_state=config.RANDOM_STATE))]:
        scores = cross_validate(_pay_pipeline(reg), X, y, cv=cv,
                                scoring=["r2", "neg_root_mean_squared_error"])
        fitted = _pay_pipeline(reg).fit(X, y)
        coefs = pd.Series(fitted["reg"].coef_, index=fitted["pre"].get_feature_names_out())
        report[name] = {
            "r2_mean": float(scores["test_r2"].mean()),
            "r2_std": float(scores["test_r2"].std()),
            "rmse_log_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            "alpha": float(fitted["reg"].alpha_),
            "n_zero_coefs": int((coefs.abs() < 1e-8).sum()),
            "coefficients": coefs.round(4).to_dict(),
        }
    return report


# --------------------------------------------------------------------------
# 2. Macro forecast on JOLTS quits rate
# --------------------------------------------------------------------------
def load_jolts(path=None) -> pd.Series | None:
    path = path or config.JOLTS_DATA
    """Read the FRED CSV (columns observation_date/DATE + JTSQUR). None if missing."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("observation_date", "date"))
    value_col = [c for c in df.columns if c != date_col][0]
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col]), name="quits_rate").dropna()
    return s.asfreq("MS")


def _fit_arima(series: pd.Series, order):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order).fit()


def select_arima_order(series: pd.Series, p=range(0, 3), d=range(0, 2), q=range(0, 3)):
    """Grid search over (p, d, q), pick the lowest AIC."""
    results = []
    for order in itertools.product(p, d, q):
        try:
            results.append((order, _fit_arima(series, order).aic))
        except Exception:  # some orders fail to converge; skip them
            continue
    results.sort(key=lambda r: r[1])
    return results[0][0], pd.DataFrame(results, columns=["order", "aic"])


def ts_cross_validate(series: pd.Series, order, horizon: int = 12, n_splits: int = 3) -> dict:
    """Rolling-origin CV: train on the past, forecast the next `horizon` months,
    move the origin forward. Compared against a naive 'last value' baseline."""
    rows = []
    for i in range(n_splits, 0, -1):
        cut = len(series) - i * horizon
        train, test = series.iloc[:cut], series.iloc[cut:cut + horizon]
        pred = _fit_arima(train, order).forecast(len(test))
        naive = np.repeat(train.iloc[-1], len(test))
        rows.append({
            "origin": str(train.index[-1].date()),
            "mape_arima": float(np.mean(np.abs((test - pred.values) / test)) * 100),
            "mape_naive": float(np.mean(np.abs((test - naive) / test)) * 100),
        })
    df = pd.DataFrame(rows)
    return {"folds": rows, "mape_arima_mean": float(df.mape_arima.mean()),
            "mape_naive_mean": float(df.mape_naive.mean())}


def forecast_arima(series: pd.Series, order, steps: int = 12) -> pd.DataFrame:
    res = _fit_arima(series, order).get_forecast(steps)
    ci = res.conf_int(alpha=0.2)  # 80% interval
    return pd.DataFrame({"forecast": res.predicted_mean,
                         "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]})


def forecast_prophet(series: pd.Series, steps: int = 12) -> pd.DataFrame | None:
    """Optional comparison model. Returns None if prophet is not installed."""
    try:
        from prophet import Prophet
    except ImportError:
        return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(series.reset_index().set_axis(["ds", "y"], axis=1))
    fut = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(fut).tail(steps).set_index("ds")
    return fc[["yhat", "yhat_lower", "yhat_upper"]].set_axis(["forecast", "lower", "upper"], axis=1)


def macro_adjustment(series: pd.Series, fc: pd.DataFrame) -> float:
    """Ratio of forecast quits rate (next 12m) to the trailing 12m average.

    ASSUMPTION (state it openly): the company's attrition moves in proportion
    to the US quits rate. This only nudges the base rate; it is not a forecast
    of the company itself.
    """
    return float(fc["forecast"].mean() / series.iloc[-12:].mean())


# --------------------------------------------------------------------------
# 3. Workforce forecast
# --------------------------------------------------------------------------
def workforce_forecast(scored: pd.DataFrame, macro_ratio: float = 1.0,
                       replacement_multiplier: float = config.REPLACEMENT_COST_MULTIPLIER) -> pd.DataFrame:
    """Expected leavers and replacement cost per department.

    `scored` needs columns Department, MonthlyIncome and p_leave. Expected
    leavers = sum of individual probabilities (expected value of a sum of
    Bernoulli variables) x macro ratio.
    """
    df = scored.copy()
    df["p_adj"] = (df["p_leave"] * macro_ratio).clip(0, 1)
    df["exp_cost"] = df["p_adj"] * df["MonthlyIncome"] * 12 * replacement_multiplier
    out = df.groupby("Department").agg(
        headcount=("p_adj", "size"),
        expected_leavers=("p_adj", "sum"),
        expected_replacement_cost=("exp_cost", "sum"),
    )
    out["expected_attrition_rate"] = out["expected_leavers"] / out["headcount"]
    return out.sort_values("expected_replacement_cost", ascending=False)


def monthly_leaver_path(total_expected: float, months: int = config.ATTRITION_HORIZON_MONTHS) -> pd.Series:
    """Spread expected leavers evenly over the horizon (cumulative).
    TODO(team): if you use the non-seasonally-adjusted series (JTUQUR), weight
    months by the seasonal pattern instead of spreading evenly."""
    return pd.Series(np.linspace(total_expected / months, total_expected, months),
                     index=range(1, months + 1), name="cumulative_expected_leavers")
