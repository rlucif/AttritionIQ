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
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src import config, data, forecast, model, rag, recommend, roi, segment  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def savefig(name: str):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / f"{name}.png", dpi=130)
    plt.close()


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

    # Pick the winner on mean CV PR-AUC. TODO(team): if two models are within
    # one std of each other, consider preferring the simpler, more explainable one.
    best_name = max(comparison, key=lambda n: comparison[n]["repeated_cv"]["pr_auc"]["mean"])
    base_model = fitted[best_name]
    step(f"Best model: {best_name}")

    step("Out-of-fold probabilities, calibration and cost-optimal threshold")
    calibrated = model.calibrate(base_model, X_train, y_train)
    oof = model.oof_probabilities(calibrated, X_train, y_train)
    repl_tr = roi.replacement_cost(train)
    int_tr = roi.intervention_cost(train)
    threshold, curve = roi.optimise_threshold(y_train, oof, repl_tr, int_tr, config.INTERVENTION_SUCCESS_RATE)

    p_test = calibrated.predict_proba(X_test)[:, 1]
    metrics["best_model"] = best_name
    metrics["test_evaluation"] = model.evaluate(y_test, p_test, threshold)
    metrics["test_evaluation_at_0.5"] = model.evaluate(y_test, p_test, 0.5)
    metrics["roi_test"] = roi.roi_summary(y_test, p_test, threshold, roi.replacement_cost(test),
                                          roi.intervention_cost(test), config.INTERVENTION_SUCCESS_RATE)
    metrics["fairness_test"] = model.fairness_report(test, p_test, threshold).round(3).to_dict("records")
    metrics["error_analysis_test"] = model.error_analysis(test, p_test, threshold).round(3).reset_index().to_dict("records")

    # Figures: threshold cost curve, PR curves, calibration
    plt.figure(figsize=(6, 4))
    plt.plot(curve.threshold, curve.total_cost)
    plt.axvline(threshold, ls="--", c="k")
    plt.xlabel("Decision threshold"); plt.ylabel("Total expected cost (train, out-of-fold)")
    plt.title(f"Cost-optimal threshold = {threshold:.2f}")
    savefig("threshold_cost_curve")

    plt.figure(figsize=(6, 4))
    for name, est in fitted.items():
        pr, rc, _ = precision_recall_curve(y_test, est.predict_proba(X_test)[:, 1])
        plt.plot(rc, pr, label=name)
    plt.axhline(y_test.mean(), ls=":", c="grey", label="random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=7); plt.title("Precision-recall (test)")
    savefig("pr_curves_test")

    plt.figure(figsize=(5, 4))
    for label, p in [("uncalibrated", base_model.predict_proba(X_test)[:, 1]), ("calibrated", p_test)]:
        fr, mp = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        plt.plot(mp, fr, marker="o", label=label)
    plt.plot([0, 1], [0, 1], ls=":", c="grey")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed attrition rate"); plt.legend(); plt.title("Calibration (test)")
    savefig("calibration_test")

    step("SHAP explanations")
    explainer = model.Explainer(base_model, X_train, lists)
    X_shap = X_test if explainer.kind != "model-agnostic" else X_test.sample(100, random_state=config.RANDOM_STATE)
    ex = explainer.explain(X_shap)
    metrics["shap_global_top10"] = explainer.global_importance(X_shap).head(10).round(4).to_dict()
    shap.plots.beeswarm(ex, max_display=15, show=False)
    savefig("shap_beeswarm")

    # ------------------------------------------------- Track 1: segments
    step("Workforce segmentation")
    Xs, seg_scaler = segment.scale(df)
    k_table = segment.choose_k(Xs)
    k = int(k_table[k_table.k.between(3, 6)].sort_values("silhouette", ascending=False).k.iloc[0])
    km = segment.fit_kmeans(Xs, k)
    h_labels, Z = segment.fit_hierarchical(Xs, k)
    kd = segment.k_distance(Xs, k=5)
    eps = float(np.percentile(kd, 90))  # TODO(team): pick eps from the knee of the k-distance plot
    db_labels = segment.fit_dbscan(Xs, eps=eps, min_samples=5)
    profile = segment.profile_clusters(df, km.labels_)
    why_labels, why_sil = segment.cluster_on_shap(ex.values, k=3)
    from sklearn.metrics import adjusted_rand_score
    metrics["segmentation"] = {
        "k_selection": k_table.round(4).to_dict("records"), "k_chosen": k,
        "silhouette_kmeans": float(k_table.set_index("k").loc[k, "silhouette"]),
        "agreement_kmeans_vs_hierarchical_ARI": float(adjusted_rand_score(km.labels_, h_labels)),
        "dbscan_eps": eps, "dbscan_outliers": int((db_labels == -1).sum()),
        "dbscan_clusters": int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
        "why_segments_silhouette": float(why_sil),
        "cluster_descriptions": segment.describe_clusters(profile),
    }
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(k_table.k, k_table.inertia, marker="o"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(k_table.k, k_table.silhouette, marker="o"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
    savefig("kmeans_k_selection")
    plt.figure(figsize=(8, 4)); dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True)
    plt.title("Hierarchical clustering (Ward)"); savefig("dendrogram")
    plt.figure(figsize=(6, 4)); plt.plot(kd); plt.axhline(eps, ls="--", c="k")
    plt.title("k-distance plot (k=5) for DBSCAN eps"); plt.ylabel("distance"); savefig("dbscan_k_distance")

    # ------------------------------------------------- Track 4: forecast
    step("Pay model and macro forecast")
    metrics["pay_model"] = forecast.pay_model_report(train)
    jolts = forecast.load_jolts()
    jolts_bundle = None
    if jolts is not None and len(jolts) > 48:
        order, aic_table = forecast.select_arima_order(jolts)
        cv = forecast.ts_cross_validate(jolts, order)
        fc = forecast.forecast_arima(jolts, order)
        ratio = forecast.macro_adjustment(jolts, fc)
        metrics["macro_forecast"] = {"arima_order": list(order), "ts_cv": cv, "macro_ratio": ratio,
                                     "last_observation": str(jolts.index[-1].date())}
        jolts_bundle = {"series": jolts, "forecast": fc, "order": order, "ratio": ratio,
                        "prophet": forecast.forecast_prophet(jolts)}
        plt.figure(figsize=(7, 3.5))
        jolts.iloc[-60:].plot(label="history")
        fc.forecast.plot(label=f"ARIMA{order}")
        plt.fill_between(fc.index, fc.lower, fc.upper, alpha=0.2)
        plt.legend(); plt.title("US quits rate (JOLTS) - 12-month forecast"); savefig("jolts_forecast")
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
        "segments": {"scaler": seg_scaler, "kmeans": km, "k": k, "labels": km.labels_,
                     "hier_labels": h_labels, "dbscan_labels": db_labels, "eps": eps,
                     "profile": profile, "pca": segment.pca_2d(Xs),
                     "rfm": segment.rfm_style_scores(df), "df_ids": df[config.ID_COL].to_numpy(),
                     "why_labels": why_labels, "why_index": X_shap.index.to_numpy()},
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
