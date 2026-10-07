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
from sklearn.metrics import silhouette_score
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


def choose_k(X: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)


def fit_hierarchical(X: np.ndarray, k: int):
    """Returns cluster labels and the linkage matrix for a dendrogram plot."""
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    return labels, linkage(X, method="ward")


def k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Sorted distance to each point's k-th neighbour. The 'knee' of this curve
    is a principled starting value for DBSCAN's eps."""
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(dist[:, -1])


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


def describe_clusters(profile: pd.DataFrame, features=SEGMENT_FEATURES, top: int = 3) -> dict:
    """Auto-description: the features where each cluster differs most from the
    overall average (in standard deviations).
    TODO(team): read these, then give each cluster a manager-friendly persona
    name (e.g. 'Overworked juniors') in PERSONA_NAMES below."""
    z = (profile[features] - profile[features].mean()) / profile[features].std(ddof=0).replace(0, 1)
    return {c: ", ".join(f"{f} {'high' if z.loc[c, f] > 0 else 'low'}"
                         for f in z.loc[c].abs().sort_values(ascending=False).index[:top])
            for c in z.index}


PERSONA_NAMES: dict[int, str] = {}  # TODO(team): fill after reading describe_clusters()


def pca_2d(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=RS).fit_transform(X)


def cluster_on_shap(shap_values: np.ndarray, k: int):
    """'Why-segments': cluster employees by WHY the model thinks they may leave,
    instead of by who they are."""
    X = StandardScaler().fit_transform(shap_values)
    km = fit_kmeans(X, k)
    return km.labels_, silhouette_score(X, km.labels_)
