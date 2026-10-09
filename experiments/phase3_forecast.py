"""Phase 3, Track 4 (part 2): macro forecast of the US quits rate (JOLTS).

Data: FRED series JTSQUR (quits rate, total nonfarm, seasonally adjusted),
saved as data/external/JTSQUR.csv. Downloaded 2026-10-08: Dec 2000 - Aug 2026,
309 months.

Rules (docs/PHASE3_PLAN.md, fixed before running):
  * d from ADF + KPSS (forecast.choose_d), then (p, q) in 0..3 by AIC
  * rolling-origin CV: 6 origins x 12-month horizon, vs naive and seasonal naive
  * ARIMA is used in the app only if its mean MAE beats naive; otherwise naive
  * report with and without the folds whose test year overlaps the 2020-22
    shock (COVID dip and the 2021-22 quits surge)

Run from the repo root:  python -m experiments.phase3_forecast
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, forecast  # noqa: E402

SHOCK_END = "2022-12-01"


def main():
    s = forecast.load_jolts()
    d, d_log = forecast.choose_d(s)
    order, aic = forecast.select_arima_order(s, d=d)
    cv = forecast.ts_cross_validate(s, order, horizon=12, n_splits=6)
    folds = pd.DataFrame(cv["folds"])
    calm = folds[folds.test_start > SHOCK_END]
    fc_arima = forecast.forecast_arima(s, order)
    use = "arima" if cv["mae_arima_mean"] < cv["mae_naive_mean"] else "naive"
    fc = fc_arima if use == "arima" else forecast.forecast_naive(s)
    # Secondary check, reported only (does not change the decision): the most
    # parsimonious candidate near the top of the AIC table.
    simple = forecast.ts_cross_validate(s, (0, d, 1), horizon=12, n_splits=6)
    out = {
        "series": {"n": len(s), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
                   "last_value": float(s.iloc[-1]), "trailing_12m_mean": float(s.iloc[-12:].mean())},
        "d_selection": d_log, "d": d, "order": list(order),
        "aic_top5": aic.head(5).assign(order=aic.order.astype(str)).round(2).to_dict("records"),
        "ts_cv": cv,
        "ts_cv_excluding_shock": {f"mae_{m}_mean": float(calm[f"mae_{m}"].mean())
                                  for m in ["arima", "naive", "seasonal_naive"]} | {"folds": len(calm)},
        "decision": use,
        "secondary_check_arima_0d1": {k: simple[k] for k in ["mae_arima_mean", "mae_naive_mean",
                                                             "arima_beats_naive_folds"]},
        "forecast_12m": fc.round(3).reset_index(names="month").assign(month=lambda x: x.month.astype(str)).to_dict("records"),
        "macro_ratio": forecast.macro_adjustment(s, fc),
    }
    (config.REPORTS_DIR / "phase3_forecast.json").write_text(json.dumps(out, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    s.plot(ax=ax, lw=1, label="US quits rate (JOLTS, % of employment)")
    fc.forecast.plot(ax=ax, label=f"{use} forecast (used)")
    ax.fill_between(fc.index, fc.lower, fc.upper, alpha=0.2, label="80% interval")
    if use != "arima":
        fc_arima.forecast.plot(ax=ax, ls=":", c="grey", label=f"ARIMA{tuple(order)} (lost to naive in CV)")
    ax.set_xlim(pd.Timestamp("2015-01-01"), fc.index[-1]); ax.set_ylim(1, 3.4); ax.legend(fontsize=8)
    ax.set_title("US quits rate - 12-month forecast"); plt.tight_layout()
    plt.savefig(config.FIG_DIR / "jolts_forecast.png", dpi=130); plt.close()
    print(json.dumps({k: out[k] for k in ["d_selection", "order", "aic_top5", "ts_cv_excluding_shock",
                                          "decision", "macro_ratio"]}, indent=1, default=str))
    print(folds.round(3).to_string())


if __name__ == "__main__":
    main()
