"""End-to-end training pipeline. Run once before starting the app:

    python train.py            # full run (a few minutes)
    python train.py --fast     # 5 search iterations per model, for quick checks

Outputs:
    models/attritioniq_bundle.joblib   everything the app needs
    reports/metrics.json               every number you will quote in the deck
    reports/figures/*.png              charts for the technical deep dive
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src import config, data, forecast, model, rag, recommend, roi, segment  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def savefig(name: str):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / f"{name}.png", dpi=130)
    plt.close()


def shap_waterfall(explainer, X_row: pd.DataFrame, lists: dict, title: str, name: str):
    """Waterfall for one employee, with one-hot columns summed back to the
    original feature and the employee's real (unscaled) values on the axis."""
    ex = explainer.explain(X_row)
    s = pd.Series(ex.values[0], index=explainer.feature_names)
    g = s.groupby([data.base_feature(f, lists) for f in s.index]).sum()
    raw = X_row.iloc[0].to_dict()
    raw["PayGapPct"] = round(float(explainer.transform.transform(X_row)["PayGapPct"].iloc[0]), 3)
    e = shap.Explanation(values=g.to_numpy(), base_values=float(np.ravel(ex.base_values)[0]),
                         data=np.array([raw.get(f, "") for f in g.index], dtype=object),
                         feature_names=list(g.index))
    shap.plots.waterfall(e, max_display=10, show=False)
    plt.title(title, fontsize=10)
    plt.gcf().text(0.01, 0.005, "Bars in log-odds of the uncalibrated model; the title shows the "
                   "calibrated probability. Calibration rescales the score, it does not reorder drivers.",
                   fontsize=7, color="grey")
    savefig(name)


def step(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(fast: bool = False):
    metrics: dict = {"run_time": time.strftime("%Y-%m-%d %H:%M:%S"), "fast_mode": fast}
    n_iter = 5 if fast else config.N_ITER_SEARCH

    # ------------------------------------------------------------------ data
    step("Loading and preparing data")
    df = data.prepare(data.load_raw())
    metrics["data"] = data.data_summary(df)
    lists = data.feature_lists()
    train, test = data.split(df)
    X_train, y_train = data.X_y(train, lists)
    X_test, y_test = data.X_y(test, lists)
    metrics["data"].update({"train_rows": len(train), "test_rows": len(test),
                            "excluded_sensitive": config.EXCLUDE_SENSITIVE})

    # ------------------------------------------------- Track 2: risk engine
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    specs = model.model_specs(pos_weight)
    comparison, fitted = {}, {}
    for name, spec in specs.items():
        step(f"Tuning {name}")
        best, info = model.tune(name, spec, X_train, y_train, lists, n_iter=n_iter)
        step(f"Repeated CV for {name}")
        info["repeated_cv"] = model.repeated_cv(best, X_train, y_train)
        comparison[name], fitted[name] = info, best
    metrics["model_comparison"] = comparison

    # Selection rule (fixed before the run, see model.select_model): models within
    # one std of the best mean PR-AUC are tied; among tied, the simplest wins.
    best_name, selection = model.select_model(comparison)
    metrics["model_selection"] = selection
    base_model = fitted[best_name]
    step(f"Chosen model: {best_name} (highest mean: {selection['highest_mean']}, "
         f"tied: {selection['tied_with_best']})")

    if not fast:
        step(f"Nested CV for {best_name} (how optimistic was tune-and-score on the same data?)")
        nested = model.nested_cv(best_name, specs[best_name], X_train, y_train, lists, n_iter=n_iter)
        nested["plain_repeated_cv_mean"] = comparison[best_name]["repeated_cv"]["pr_auc"]["mean"]
        nested["optimism"] = nested["plain_repeated_cv_mean"] - nested["pr_auc"]["mean"]
        metrics["nested_cv"] = nested

    step("Out-of-fold probabilities, calibration and cost-optimal threshold")
    calibrated = model.calibrate(base_model, X_train, y_train)
    oof = model.oof_probabilities(calibrated, X_train, y_train)
    oof_raw = model.oof_probabilities(base_model, X_train, y_train)
    repl_tr = roi.replacement_cost(train)
    int_tr = roi.intervention_cost(train)
    success = config.INTERVENTION_SUCCESS_RATE
    threshold, curve = roi.optimise_threshold(y_train, oof, repl_tr, int_tr, success)

    p_test = calibrated.predict_proba(X_test)[:, 1]
    p_test_raw = base_model.predict_proba(X_test)[:, 1]
    repl_te, int_te = roi.replacement_cost(test), roi.intervention_cost(test)
    metrics["best_model"] = best_name
    metrics["calibration"] = {"oof_train": model.calibration_report(y_train, oof_raw, oof),
                              "test": model.calibration_report(y_test, p_test_raw, p_test)}
    metrics["threshold"] = {
        "cost_optimal_oof_train": threshold,
        "break_even_theory": roi.break_even_threshold(),
        "assumptions": {"replacement_multiplier": config.REPLACEMENT_COST_MULTIPLIER,
                        "intervention_months": config.INTERVENTION_COST_MONTHS,
                        "success_rate": success},
    }
    metrics["test_evaluation"] = model.evaluate(y_test, p_test, threshold)
    metrics["test_evaluation_at_0.5"] = model.evaluate(y_test, p_test, 0.5)
    metrics["roi_test"] = roi.roi_summary(y_test, p_test, threshold, repl_te, int_te, success)

    def savings(idx):  # bootstrap: savings vs do-nothing on a resampled test set
        yb, pb, rb, ib = y_test[idx], p_test[idx], repl_te[idx], int_te[idx]
        return (roi.realised_cost(yb, np.zeros(len(idx), bool), rb, ib, success)
                - roi.realised_cost(yb, pb >= threshold, rb, ib, success))
    metrics["test_ci95"] = model.bootstrap_ci(y_test, p_test, threshold,
                                              extra={"savings_vs_do_nothing": savings})

    step("ROI sensitivity to the business assumptions")
    sens = roi.sensitivity_grid(train, y_train, oof, test, y_test, p_test)
    sens.to_csv(config.REPORTS_DIR / "roi_sensitivity.csv", index=False)
    metrics["roi_sensitivity"] = sens.round(4).to_dict("records")

    metrics["fairness_test"] = model.fairness_report(test, p_test, threshold).round(3).to_dict("records")
    metrics["fairness_oof_train"] = model.fairness_report(train, oof, threshold).round(3).to_dict("records")
    metrics["error_analysis_test"] = model.error_analysis(test, p_test, threshold).round(3).reset_index().to_dict("records")
    metrics["error_analysis_oof_train"] = model.error_analysis(train, oof, threshold).round(3).reset_index().to_dict("records")

    # Figures: threshold cost curve, PR curves, calibration
    plt.figure(figsize=(6, 4))
    plt.plot(curve.threshold, curve.total_cost)
    plt.axvline(threshold, ls="--", c="k", label=f"cost-optimal {threshold:.2f}")
    plt.axvline(roi.break_even_threshold(), ls=":", c="grey",
                label=f"break-even theory {roi.break_even_threshold():.2f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Total expected cost (train, out-of-fold)")
    plt.legend(fontsize=8); plt.title(f"Cost-optimal threshold = {threshold:.2f}")
    savefig("threshold_cost_curve")

    plt.figure(figsize=(6, 4))
    for name, est in fitted.items():
        pr, rc, _ = precision_recall_curve(y_test, est.predict_proba(X_test)[:, 1])
        plt.plot(rc, pr, label=name)
    plt.axhline(y_test.mean(), ls=":", c="grey", label="random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=7); plt.title("Precision-recall (test)")
    savefig("pr_curves_test")

    plt.figure(figsize=(5, 4))
    for label, p in [("uncalibrated", p_test_raw), ("calibrated", p_test)]:
        fr, mp = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        plt.plot(mp, fr, marker="o", label=label)
    plt.plot([0, 1], [0, 1], ls=":", c="grey")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed attrition rate"); plt.legend(); plt.title("Calibration (test)")
    savefig("calibration_test")

    plt.figure(figsize=(6, 4))
    folds = pd.DataFrame({n: c["repeated_cv"]["pr_auc"]["folds"] for n, c in comparison.items()})
    plt.boxplot([folds[c] for c in folds], tick_labels=[c.replace(" (MLP)", "") for c in folds])
    plt.ylabel("PR-AUC per fold (5x3 repeated CV)"); plt.title(f"Model comparison - chosen: {best_name}")
    plt.xticks(fontsize=8); savefig("model_comparison_cv")

    step("SHAP explanations")
    explainer = model.Explainer(base_model, X_train, lists)
    X_shap = X_test if explainer.kind != "model-agnostic" else X_test.sample(100, random_state=config.RANDOM_STATE)
    ex = explainer.explain(X_shap)
    glob = explainer.global_importance(X_shap)
    metrics["shap_global_top10"] = glob.head(10).round(4).to_dict()
    shap.plots.beeswarm(ex, max_display=15, show=False)
    savefig("shap_beeswarm")
    plt.figure(figsize=(6, 5))
    glob.head(15)[::-1].plot.barh()
    plt.xlabel("Mean |SHAP| (one-hot columns summed per feature)"); plt.title("Global feature importance (test)")
    savefig("shap_global_bar")

    # Individual explanations: one caught leaver, one missed leaver, one false alarm
    flag_te = p_test >= threshold
    cases = {
        "caught_leaver": np.where((y_test == 1) & flag_te)[0],
        "missed_leaver": np.where((y_test == 1) & ~flag_te)[0],
        "false_alarm": np.where((y_test == 0) & flag_te)[0],
    }
    metrics["shap_individual_examples"] = {}
    for label, idx in cases.items():
        if not len(idx):
            continue
        # highest-risk caught leaver / median missed leaver / highest-risk false alarm
        i = idx[np.argmax(p_test[idx])] if label != "missed_leaver" else idx[np.argsort(p_test[idx])[len(idx) // 2]]
        emp = int(test[config.ID_COL].iloc[i])
        shap_waterfall(explainer, X_test.iloc[[i]], lists,
                       f"{label.replace('_', ' ')} - employee #{emp}, P(leave) {p_test[i]:.0%}",
                       f"shap_waterfall_{label}")
        metrics["shap_individual_examples"][label] = {
            "employee": emp, "p_leave": float(p_test[i]),
            "drivers": explainer.drivers(X_test.iloc[[i]], k=5)[["feature", "shap", "direction"]]
            .round(3).to_dict("records")}

    # ------------------------------------------- Fairness deep dive (50+)
    step("Fairness: why are older leavers missed?")
    age50 = (train["Age"] >= 50).to_numpy()
    leaver = y_train == 1
    gap = model.group_shap_gap(explainer, X_train, age50 & leaver, ~age50 & leaver)
    metrics["fairness_50plus"] = {
        "leavers_50plus_train": int((age50 & leaver).sum()),
        "mean_oof_p_leavers_50plus": float(oof[age50 & leaver].mean()),
        "mean_oof_p_leavers_under50": float(oof[~age50 & leaver].mean()),
        "shap_gap_vs_younger_leavers": gap.round(4).reset_index(names="feature").to_dict("records"),
    }
    # Counterfactual: identical model and settings, sensitive attributes put back in
    lists_s = data.feature_lists(exclude_sensitive=False)
    Xs_train, _ = data.X_y(train, lists_s)
    pipe_s = clone(base_model).set_params(prep=data.build_preprocessor(lists_s))
    oof_s = model.oof_probabilities(model.calibrate(pipe_s, Xs_train, y_train), Xs_train, y_train)
    thr_s, _ = roi.optimise_threshold(y_train, oof_s, repl_tr, int_tr, success)
    metrics["sensitive_counterfactual"] = {
        "note": "Same model and hyperparameters with Gender, Age, MaritalStatus included. "
                "Out-of-fold on train. Shown to quantify the trade-off; the deployed model excludes them.",
        "oof_pr_auc_excluded": float(average_precision_score(y_train, oof)),
        "oof_pr_auc_included": float(average_precision_score(y_train, oof_s)),
        "threshold_excluded": threshold, "threshold_included": thr_s,
        "fairness_excluded": metrics["fairness_oof_train"],
        "fairness_included": model.fairness_report(train, oof_s, thr_s).round(3).to_dict("records"),
    }

    # ------------------------------------------- Error analysis: missed leavers
    step("Error analysis: exporting every missed leaver with SHAP drivers")
    miss_test = model.missed_leavers(test, X_test, p_test, threshold, explainer)
    miss_oof = model.missed_leavers(train, X_train, oof, threshold, explainer)
    miss_test.to_csv(config.REPORTS_DIR / "missed_leavers_test.csv", index=False)
    miss_oof.to_csv(config.REPORTS_DIR / "missed_leavers_oof_train.csv", index=False)
    metrics["missed_leavers"] = {"test": len(miss_test), "oof_train": len(miss_oof),
                                 "files": ["reports/missed_leavers_test.csv",
                                           "reports/missed_leavers_oof_train.csv"]}

    # ------------------------------------------------- Track 1: segments
    # Phase 3 rules (docs/PHASE3_PLAN.md, results in docs/PHASE3_RESULTS.md)
    step("Workforce segmentation (k rule with bootstrap stability, knee eps)")
    Xs, seg_scaler = segment.scale(df)
    k_table = segment.choose_k(Xs)
    k_rule = segment.select_k(k_table)
    k = k_rule or 2
    km = segment.fit_kmeans(Xs, k)
    h_labels, Z = segment.fit_hierarchical(Xs, k)
    eps, min_samples, kd = segment.choose_eps(Xs)
    db_labels = segment.fit_dbscan(Xs, eps=eps, min_samples=min_samples)
    profile = segment.profile_clusters(df, km.labels_)
    personas = segment.persona_names(profile) if k == 3 else {c: f"Segment {c}" for c in profile.index}
    S_shap = segment.group_shap(ex.values, explainer.feature_names, lambda f: data.base_feature(f, lists))
    why_labels, why_sil = segment.cluster_on_shap(S_shap.to_numpy(), k=k)
    from sklearn.metrics import adjusted_rand_score
    metrics["segmentation"] = {
        "k_selection": k_table.round(4).to_dict("records"), "k_chosen": k,
        "k_rule_result": k_rule if k_rule else "no stable k in 3-6; fell back to 2",
        "silhouette_kmeans": float(k_table.set_index("k").loc[k, "silhouette"]),
        "stability_kmeans": float(k_table.set_index("k").loc[k, "stability_ari_mean"]),
        "agreement_kmeans_vs_hierarchical_ARI": float(adjusted_rand_score(km.labels_, h_labels)),
        "dbscan_min_samples": min_samples, "dbscan_eps": eps,
        "dbscan_outliers": int((db_labels == -1).sum()),
        "dbscan_clusters": int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
        "why_segments_silhouette": float(why_sil),
        "why_segments_note": "raw SHAP (log-odds), one-hot summed, pay & seniority grouped; test rows only",
        "personas_draft": {str(c): n for c, n in personas.items()},
        "cluster_descriptions": segment.describe_clusters(profile, df),
    }
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(k_table.k, k_table.inertia, marker="o"); ax[0].set_title("Elbow (inertia)")
    ax[1].plot(k_table.k, k_table.silhouette, marker="o"); ax[1].set_title("Silhouette")
    ax[2].errorbar(k_table.k, k_table.stability_ari_mean, yerr=k_table.stability_ari_std, marker="o", capsize=3)
    ax[2].axhline(0.60, ls="--", c="grey"); ax[2].set_title("Bootstrap stability (ARI)")
    for a in ax:
        a.set_xlabel("k"); a.axvline(k, ls=":", c="k")
    savefig("kmeans_k_selection")
    plt.figure(figsize=(8, 4)); dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True)
    plt.title("Hierarchical clustering (Ward)"); savefig("dendrogram")
    plt.figure(figsize=(6, 4)); plt.plot(kd); plt.axhline(eps, ls="--", c="k", label=f"knee eps = {eps:.2f}")
    plt.xlabel("employees sorted by distance"); plt.ylabel(f"distance to {min_samples}th neighbour")
    plt.title(f"k-distance plot (k = {min_samples}) for DBSCAN eps"); plt.legend(); savefig("dbscan_k_distance")

    # ------------------------------------------------- Track 4: forecast
    step("Pay model and macro forecast")
    metrics["pay_model"] = forecast.pay_model_report(train)
    jolts = forecast.load_jolts()
    jolts_bundle = None
    if jolts is not None and len(jolts) > 48:
        d, d_log = forecast.choose_d(jolts)
        order, aic_table = forecast.select_arima_order(jolts, d=d)
        cv = forecast.ts_cross_validate(jolts, order, horizon=12, n_splits=6)
        fc_arima = forecast.forecast_arima(jolts, order)
        # Phase 3 rule: use ARIMA only if it beats the naive forecast in rolling CV
        use = "ARIMA" if cv["mae_arima_mean"] < cv["mae_naive_mean"] else "naive"
        fc = fc_arima if use == "ARIMA" else forecast.forecast_naive(jolts)
        ratio = forecast.macro_adjustment(jolts, fc)
        metrics["macro_forecast"] = {"d_selection": d_log, "arima_order": list(order), "ts_cv": cv,
                                     "model_used": use, "macro_ratio": ratio,
                                     "last_observation": str(jolts.index[-1].date())}
        jolts_bundle = {"series": jolts, "forecast": fc, "forecast_arima": fc_arima, "order": order,
                        "model_used": use, "ratio": ratio, "prophet": forecast.forecast_prophet(jolts)}
        plt.figure(figsize=(7, 3.5))
        jolts.iloc[-60:].plot(label="history")
        fc.forecast.plot(label=f"{use} forecast (used)")
        plt.fill_between(fc.index, fc.lower, fc.upper, alpha=0.2, label="80% interval")
        if use != "ARIMA":
            fc_arima.forecast.plot(ls=":", c="grey", label=f"ARIMA{order} (lost to naive in CV)")
        plt.legend(fontsize=8); plt.title("US quits rate (JOLTS) - 12-month forecast"); savefig("jolts_forecast")
    else:
        metrics["macro_forecast"] = {"status": f"skipped - add {config.JOLTS_DATA.name} (see README)"}

    # ------------------------------------------------- Track 3: recommender
    step("Recommender (collaborative filtering on SIMULATED matrix)")
    R = recommend.simulate_response_matrix(df)
    metrics["recommender_cf"] = recommend.evaluate_cf(R)
    cf_pred = recommend.FunkSVD().fit(R).predict_all()

    # ------------------------------------------------- Track 5: RAG
    step("RAG retrieval evaluation")
    retriever = rag.Retriever(rag.load_chunks())
    metrics["rag_retrieval"] = rag.evaluate_retrieval(retriever, pd.read_csv(config.RAG_EVAL))

    # ------------------------------------------------- save
    step("Saving bundle and metrics")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": time.strftime("%Y%m%d-%H%M"),
        "model_name": best_name, "model": calibrated, "base_model": base_model,
        "threshold": threshold, "lists": lists,
        "oof_train": oof,  # lets the app re-optimise the threshold when assumptions change
        "train": train, "test": test,
        "segments": {"scaler": seg_scaler, "kmeans": km, "k": k, "labels": km.labels_, "personas": personas,
                     "hier_labels": h_labels, "dbscan_labels": db_labels, "eps": eps,
                     "profile": profile, "pca": segment.pca_2d(Xs),
                     "rfm": segment.rfm_style_scores(df), "df_ids": df[config.ID_COL].to_numpy(),
                     "why_labels": why_labels, "why_index": X_shap.index.to_numpy(),
                     "min_samples": min_samples},
        "cf_pred": cf_pred.set_axis(df[config.ID_COL].to_numpy()),
        "cf_observed": R.set_axis(df[config.ID_COL].to_numpy()),
        "jolts": jolts_bundle,
    }, config.BUNDLE_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    step(f"Done. Best = {best_name}, threshold = {threshold:.2f}, "
         f"test PR-AUC = {metrics['test_evaluation']['pr_auc']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer search iterations")
    main(ap.parse_args().fast)
