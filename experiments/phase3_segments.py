"""Phase 3, Track 1: how many workforce segments, and are they real?

Decision rules were fixed in docs/PHASE3_PLAN.md BEFORE this script was run:

k (K-Means on the 12 standardised SEGMENT_FEATURES)
  * consider k = 3..6 only (k = 2 just splits senior from junior)
  * drop any k whose bootstrap stability (mean adjusted Rand index) < 0.60
  * among the rest, highest silhouette; within 0.01 -> the smaller k
  * no k passes -> report "no stable segments beyond seniority"

DBSCAN
  * min_samples = 2 x n_features (Sander et al. 1998 rule of thumb)
  * eps = knee of the sorted k-distance curve (k = min_samples), found as the
    point furthest below the straight line joining the curve's two ends
    (the idea behind the Kneedle algorithm, Satopaa et al. 2011)

SHAP "why-segments"
  * SHAP values of the deployed model for all 1,470 employees, one-hot
    columns summed back to the original feature
  * NOT standardised: every SHAP value is already in the same unit (log-odds),
    so standardising would give a feature that barely moves risk the same
    weight as OverTime. The skeleton standardised them; both are reported.

Run from the repo root:  python -m experiments.phase3_segments
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,  # noqa: E402
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config, data, model, segment  # noqa: E402

RS = config.RANDOM_STATE
N_BOOT, BOOT_FRAC = 50, 0.8
K_RANGE = range(2, 9)
K_ALLOWED = range(3, 7)
MIN_STABILITY, SIL_TIE = 0.60, 0.01
PAY_SENIORITY = ["MonthlyIncome", "PayGapPct", "JobLevel", "TotalWorkingYears"]
SHAP_VARIANT = "raw_pay_seniority_grouped"  # profiled in detail; all variants are scored


def k_table(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        s_mean, s_std = segment.stability(X, k)
        rows.append({"k": k, "inertia": lab.inertia_,
                     "silhouette": silhouette_score(X, lab.labels_),
                     "calinski_harabasz": calinski_harabasz_score(X, lab.labels_),
                     "davies_bouldin": davies_bouldin_score(X, lab.labels_),
                     "stability_ari_mean": s_mean, "stability_ari_std": s_std})
    return pd.DataFrame(rows)


def apply_k_rule(t: pd.DataFrame) -> tuple[int | None, str]:
    c = t[t.k.isin(K_ALLOWED) & (t.stability_ari_mean >= MIN_STABILITY)]
    if c.empty:
        return None, "no k in 3-6 is stable (ARI >= 0.60)"
    best = c.silhouette.max()
    k = int(c[c.silhouette >= best - SIL_TIE].k.min())
    return k, f"stable k: {c.k.tolist()}; best silhouette {best:.3f}; smallest k within {SIL_TIE}: {k}"


def ward_gap(Z: np.ndarray) -> pd.DataFrame:
    """Height jump when going from k to k-1 clusters. Big jump = natural cut at k."""
    h = Z[:, 2]
    return pd.DataFrame([{"k": k, "merge_height_jump": float(h[-(k - 1)] - h[-k])} for k in K_RANGE])


def cramers_v(labels, y) -> float:
    ct = pd.crosstab(labels, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))


def attrition_by(labels, y) -> dict:
    s = pd.Series(y).groupby(np.asarray(labels))
    return {str(k): {"size": int(v.size), "attrition_rate": round(float(v.mean()), 3)} for k, v in s}


def main():
    df = data.prepare(data.load_raw())
    y = df[config.TARGET].to_numpy()
    X, _ = segment.scale(df)
    out: dict = {"n_rows": len(df), "features": segment.SEGMENT_FEATURES}

    # ---------------------------------------------------------------- k
    t = k_table(X)
    k, why = apply_k_rule(t)
    out["k_table"] = t.round(4).to_dict("records")
    out["k_rule"] = {"chosen": k, "reason": why}
    k_use = k or 2

    h_labels, Z = segment.fit_hierarchical(X, k_use)
    km = segment.fit_kmeans(X, k_use)
    wg = ward_gap(Z)
    out["ward"] = {"height_jumps": wg.round(3).to_dict("records"),
                   "largest_jump_k": int(wg.loc[wg.merge_height_jump.idxmax(), "k"]),
                   "ari_kmeans_vs_ward": float(adjusted_rand_score(km.labels_, h_labels))}

    # ---------------------------------------------------------------- DBSCAN
    min_samples = 2 * X.shape[1]
    kd = segment.k_distance(X, k=min_samples)
    i = segment.knee_index(kd)
    eps = float(kd[i])
    db = segment.fit_dbscan(X, eps, min_samples)
    sens = []
    for f in (0.9, 1.0, 1.1):
        lab = segment.fit_dbscan(X, eps * f, min_samples)
        sens.append({"eps": round(eps * f, 3), "clusters": int(len(set(lab)) - (-1 in lab)),
                     "outliers": int((lab == -1).sum())})
    out["dbscan"] = {"min_samples": min_samples, "eps_knee": eps, "knee_percentile": 100 * i / len(kd),
                     "clusters": int(len(set(db)) - (-1 in db)), "outliers": int((db == -1).sum()),
                     "outlier_attrition_rate": float(y[db == -1].mean()) if (db == -1).any() else None,
                     "inlier_attrition_rate": float(y[db != -1].mean()),
                     "eps_sensitivity": sens,
                     "skeleton_eps_p90_k5": float(np.percentile(segment.k_distance(X, 5), 90))}

    # ---------------------------------------------------------------- profiles
    prof = segment.profile_clusters(df, km.labels_)
    out["profile"] = prof.round(3).reset_index().to_dict("records")
    out["personas_draft"] = {str(c): n for c, n in segment.persona_names(prof).items()}
    out["descriptions"] = {str(c): d for c, d in segment.describe_clusters(prof, df, top=4).items()}
    out["cramers_v_attrition"] = {"kmeans_features": cramers_v(km.labels_, y)}

    # ---------------------------------------------------------------- SHAP segments
    b = joblib.load(config.BUNDLE_PATH)
    lists = b["lists"]
    X_all, _ = data.X_y(df, lists)
    explainer = model.Explainer(b["base_model"], b["train"].pipe(lambda d: data.X_y(d, lists)[0]), lists)
    ex = explainer.explain(X_all)
    S = pd.DataFrame(ex.values, columns=explainer.feature_names)
    S = S.T.groupby([data.base_feature(f, lists) for f in S.columns]).sum().T  # one column per feature
    shap_tables = {}
    # Open decision 6: MonthlyIncome, PayGapPct, JobLevel and TotalWorkingYears are collinear,
    # so their SHAP values offset each other. Summing them into one driver removes that.
    S_grp = S.copy()
    S_grp["Pay & seniority"] = S_grp[PAY_SENIORITY].sum(axis=1)
    S_grp = S_grp.drop(columns=PAY_SENIORITY)
    variants = {"raw_logodds": S, "standardised": pd.DataFrame(StandardScaler().fit_transform(S), columns=S.columns),
                "raw_pay_seniority_grouped": S_grp}
    for name, Mdf in variants.items():
        M = Mdf.to_numpy()
        rows = []
        for kk in K_ALLOWED:
            lab = KMeans(n_clusters=kk, n_init=10, random_state=RS).fit_predict(M)
            rows.append({"k": kk, "silhouette": silhouette_score(M, lab),
                         "stability_ari_mean": segment.stability(M, kk)[0],
                         "cramers_v_attrition": cramers_v(lab, y)})
        shap_tables[name] = pd.DataFrame(rows)
    out["shap_segments"] = {n: tb.round(4).to_dict("records") for n, tb in shap_tables.items()}
    st = shap_tables[SHAP_VARIANT]
    c = st[st.stability_ari_mean >= MIN_STABILITY]
    k_shap = int(c[c.silhouette >= c.silhouette.max() - SIL_TIE].k.min()) if len(c) else None
    out["shap_k_rule"] = k_shap
    if k_shap:
        Sv = variants[SHAP_VARIANT]
        lab = KMeans(n_clusters=k_shap, n_init=10, random_state=RS).fit_predict(Sv.to_numpy())
        means = Sv.groupby(lab).mean()
        out["shap_segment_profiles"] = {
            str(g): {"attrition": attrition_by(lab, y)[str(g)],
                     "top_risk_drivers": means.loc[g].sort_values(ascending=False).head(4).round(3).to_dict(),
                     "top_protective": means.loc[g].sort_values().head(2).round(3).to_dict()}
            for g in means.index}
        out["cramers_v_attrition"][f"kmeans_shap_{SHAP_VARIANT}"] = cramers_v(lab, y)
        out["shap_variant_profiled"] = SHAP_VARIANT
    out["note_shap_in_sample"] = ("SHAP clusters use a supervised model's output, and 80% of rows were in "
                                  "its training data, so their link to attrition is expected and in-sample.")

    # ---------------------------------------------------------------- RFM bands
    rfm = segment.rfm_style_scores(df)
    out["rfm_bands"] = attrition_by(rfm.RFM_band.astype(str), y)

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(t.k, t.inertia, marker="o"); ax[0].set_title("Elbow (inertia)")
    ax[1].plot(t.k, t.silhouette, marker="o"); ax[1].set_title("Silhouette")
    ax[2].errorbar(t.k, t.stability_ari_mean, yerr=t.stability_ari_std, marker="o", capsize=3)
    ax[2].axhline(MIN_STABILITY, ls="--", c="grey"); ax[2].set_title("Bootstrap stability (ARI)")
    for a in ax:
        a.set_xlabel("k")
        if k:
            a.axvline(k, ls=":", c="k")
    plt.tight_layout(); plt.savefig(config.FIG_DIR / "kmeans_k_selection.png", dpi=130); plt.close()

    plt.figure(figsize=(8, 4)); dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True)
    plt.title(f"Ward dendrogram (largest height jump at k = {out['ward']['largest_jump_k']})")
    plt.tight_layout(); plt.savefig(config.FIG_DIR / "dendrogram.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4)); plt.plot(kd); plt.axhline(eps, ls="--", c="k", label=f"knee eps = {eps:.2f}")
    plt.scatter([i], [eps], c="k", zorder=3)
    plt.xlabel("employees sorted by distance"); plt.ylabel(f"distance to {min_samples}th neighbour")
    plt.title(f"k-distance plot (k = {min_samples}) for DBSCAN eps"); plt.legend()
    plt.tight_layout(); plt.savefig(config.FIG_DIR / "dbscan_k_distance.png", dpi=130); plt.close()

    (config.REPORTS_DIR / "phase3_segments.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k_: out[k_] for k_ in ["k_rule", "ward", "dbscan", "descriptions",
                                              "cramers_v_attrition", "shap_k_rule"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
