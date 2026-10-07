"""Track 1 - Customer Intelligence, applied to employees: Workforce Segments.

* K-Means       main segmentation; k chosen with elbow + silhouette
* Hierarchical  Ward linkage; the dendrogram is a second opinion on k
* DBSCAN        density-based; label -1 = outliers (unusual profiles)
* RFM-style     Recency  = years since last promotion (recent = good)
                Frequency = trainings last year (more = good)
                Monetary  = last salary hike % (higher = good)

Segmentation is unsupervised: the Attrition label is NOT used to form
clusters, only to describe them afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config

SEGMENT_FEATURES = [
    "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
    "JobLevel", "OverTime", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "DistanceFromHome", "PercentSalaryHike", "TrainingTimesLastYear",
]
RS = config.RANDOM_STATE


def scale(df: pd.DataFrame, features=SEGMENT_FEATURES):
    """Clustering is distance-based, so features must share a scale."""
    scaler = StandardScaler().fit(df[features])
    return scaler.transform(df[features]), scaler


def stability(X: np.ndarray, k: int, n_boot: int = 50, frac: float = 0.8) -> tuple[float, float]:
    """Refit K-Means on 80% subsamples; adjusted Rand index vs the full-data
    labels on the same rows. Near 1 = the same groups come back every time."""
    rng = np.random.default_rng(RS)
    full = KMeans(n_clusters=k, n_init=10, random_state=RS).fit_predict(X)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        lab = KMeans(n_clusters=k, n_init=10, random_state=RS + b + 1).fit_predict(X[idx])
        aris.append(adjusted_rand_score(full[idx], lab))
    return float(np.mean(aris)), float(np.std(aris))


def choose_k(X: np.ndarray, k_range=range(2, 9), with_stability: bool = True) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        row = {"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)}
        if with_stability:
            row["stability_ari_mean"], row["stability_ari_std"] = stability(X, k)
        rows.append(row)
    return pd.DataFrame(rows)


def select_k(table: pd.DataFrame, allowed=range(3, 7), min_stability: float = 0.60,
             tie: float = 0.01) -> int | None:
    """Phase 3 rule (docs/PHASE3_PLAN.md): k in 3-6; drop k with stability ARI
    < 0.60; highest silhouette, and within 0.01 of it the smaller k wins.
    None = no stable segmentation (fall back to k = 2 and say so)."""
    c = table[table.k.isin(allowed) & (table.stability_ari_mean >= min_stability)]
    if c.empty:
        return None
    return int(c[c.silhouette >= c.silhouette.max() - tie].k.min())


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)


def fit_hierarchical(X: np.ndarray, k: int):
    """Returns cluster labels and the linkage matrix for a dendrogram plot."""
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    return labels, linkage(X, method="ward")


def k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Sorted distance to each point's k-th neighbour (the point itself counts,
    as in DBSCAN's min_samples). The 'knee' of this curve is a principled eps."""
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(dist[:, -1])


def knee_index(y: np.ndarray) -> int:
    """Knee of a sorted, increasing curve: the point furthest below the straight
    line joining its two ends (the idea behind Kneedle, Satopaa et al. 2011)."""
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return int(np.argmax(x - yn))


def choose_eps(X: np.ndarray, min_samples: int | None = None) -> tuple[float, int, np.ndarray]:
    """eps from the k-distance knee. Default min_samples = 2 x n_features
    (Sander et al. 1998 rule of thumb). Returns (eps, min_samples, curve)."""
    min_samples = min_samples or 2 * X.shape[1]
    kd = k_distance(X, k=min_samples)
    return float(kd[knee_index(kd)]), min_samples, kd


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def rfm_style_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Quintile scores 1-5 (5 = best). rank(method='first') breaks ties so qcut works."""
    def q(s, ascending=True):
        labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
        return pd.qcut(s.rank(method="first"), 5, labels=labels).astype(int)

    out = pd.DataFrame(index=df.index)
    out["R_promotion_recency"] = q(df["YearsSinceLastPromotion"], ascending=False)
    out["F_training_frequency"] = q(df["TrainingTimesLastYear"])
    out["M_salary_hike"] = q(df["PercentSalaryHike"])
    out["RFM_total"] = out.sum(axis=1)
    out["RFM_band"] = pd.cut(out["RFM_total"], [0, 6, 9, 12, 15],
                             labels=["Neglected", "At risk", "Steady", "Invested"])
    return out


def profile_clusters(df: pd.DataFrame, labels, features=SEGMENT_FEATURES) -> pd.DataFrame:
    d = df[features + [config.TARGET]].copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").mean()
    prof.insert(0, "size", d.groupby("cluster").size())
    return prof.rename(columns={config.TARGET: "attrition_rate"})


def describe_clusters(profile: pd.DataFrame, df: pd.DataFrame | None = None,
                      features=SEGMENT_FEATURES, top: int = 3) -> dict:
    """Auto-description: the features where each cluster's average differs most
    from the WORKFORCE average, in workforce standard deviations.

    Phase 3 fix: the skeleton standardised across the 3 cluster means, which
    turned a satisfaction gap of 2.77 vs 2.71 (out of 4) into "high" vs "low".
    Pass `df` (the employee table) to measure against the real spread.
    """
    if df is not None:
        mu, sd = df[features].mean(), df[features].std(ddof=0).replace(0, 1)
    else:  # old behaviour, kept so older bundles still load
        mu, sd = profile[features].mean(), profile[features].std(ddof=0).replace(0, 1)
    z = (profile[features] - mu) / sd
    out = {}
    for c in z.index:
        big = z.loc[c].abs().sort_values(ascending=False).index[:top]
        out[c] = ", ".join(f"{f} {'high' if z.loc[c, f] > 0 else 'low'} ({z.loc[c, f]:+.1f} sd)"
                           for f in big if abs(z.loc[c, f]) >= 0.2) or "close to the workforce average"
    return out


# DRAFT persona names (Phase 3, k = 3), pending team approval (open decision 4).
# Assigned by profile, not by cluster number, so a re-run that swaps the
# numbering cannot attach a name to the wrong group.
PERSONA_RULES = {
    "overtime": "Overtime crew",        # highest share working overtime
    "senior": "Senior veterans",        # highest average job level
    "other": "Steady core",             # everyone else
}
PERSONA_NAMES: dict[int, str] = {}  # kept for backward compatibility; use persona_names()


def persona_names(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster number -> persona name using PERSONA_RULES."""
    names = {}
    senior = profile["JobLevel"].idxmax()
    names[senior] = PERSONA_RULES["senior"]
    rest = profile.drop(index=senior)
    names[rest["OverTime"].idxmax()] = PERSONA_RULES["overtime"]
    for c in profile.index:
        names.setdefault(c, PERSONA_RULES["other"] if len(profile) == 3 else f"Segment {c}")
    return {int(k): v for k, v in names.items()}


def pca_2d(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=RS).fit_transform(X)


PAY_SENIORITY = ["MonthlyIncome", "PayGapPct", "JobLevel", "TotalWorkingYears"]


def group_shap(values: np.ndarray, feature_names, base_feature) -> pd.DataFrame:
    """SHAP matrix with one-hot columns summed back to the original feature and
    the collinear pay/seniority features summed into one driver (open decision 6)."""
    S = pd.DataFrame(values, columns=feature_names)
    S = S.T.groupby([base_feature(f) for f in S.columns]).sum().T
    present = [f for f in PAY_SENIORITY if f in S]
    S["Pay & seniority"] = S[present].sum(axis=1)
    return S.drop(columns=present)


def cluster_on_shap(shap_values, k: int, standardise: bool = False):
    """'Why-segments': cluster employees by WHY the model thinks they may leave,
    instead of by who they are.

    Phase 3: not standardised by default. SHAP values already share one unit
    (log-odds), so standardising would give a feature that barely moves risk
    the same weight as OverTime (silhouette 0.109 standardised vs 0.189 raw).
    Pass SHAP values with one-hot columns summed back to the original feature.
    """
    X = StandardScaler().fit_transform(shap_values) if standardise else np.asarray(shap_values)
    km = fit_kmeans(X, k)
    return km.labels_, silhouette_score(X, km.labels_)
