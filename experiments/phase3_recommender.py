"""Phase 3, Track 3: does collaborative filtering earn its place, and how much
weight should it get next to the content-based (SHAP) scores?

The employee x intervention matrix is SIMULATED (src/recommend.py explains how),
because the IBM data has no record of which interventions were tried. So:
  * an ORACLE is available: the noise-free simulated rating. It is the best
    any model could do and gives CF a ceiling as well as a floor (item mean).
  * everything here shows the PIPELINE works and how much data it needs. It
    does not show that any intervention works.

Steps and rules (fixed in docs/PHASE3_PLAN.md before running):
1. Tune Funk SVD on a validation split carved from the OBSERVED cells only
   (early stopping picks the epoch); lowest validation RMSE wins, then refit on
   all observed cells. Tuned separately at every density and seed, because a
   fixed epoch count means 5x fewer updates at 10% density than at 50%.
2. Density sweep: observed share 0.1..0.8, 3 seeds. Scored on EVERY unobserved
   cell against the full simulated matrix (thousands of within-employee pairs;
   the first run scored a 20% hold-out with only ~300 pairs and was too noisy).
   RMSE and pairwise ranking for tuned SVD, item mean and oracle.
3. Hybrid weight alpha in 0..1 (step 0.1). Objective: NDCG@3 of the ranking of
   each employee's NOT-yet-observed interventions against their simulated
   ratings. Best mean wins, unless (best - alpha 0.6) < 1 std across seeds,
   in which case 0.6 is kept and the curve is called flat.

Run from the repo root:  python -m experiments.phase3_recommender
"""
import itertools
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, model, recommend  # noqa: E402

RS = config.RANDOM_STATE
SEEDS = [RS, RS + 1, RS + 2]
GRID = {"n_factors": [2, 4, 8], "lr": [0.005, 0.01, 0.02], "reg": [0.005, 0.01, 0.02, 0.05, 0.1]}
MAX_EPOCHS = 150
SKELETON = {"n_factors": 4, "lr": 0.01, "reg": 0.05, "epochs": 30}
DENSITIES = [0.1, 0.2, 0.3, 0.5, 0.8]


def oracle_matrix(df: pd.DataFrame) -> np.ndarray:
    """Expected rating with no noise: what a perfect model would predict."""
    signal = recommend.need_scores(df).to_numpy() @ recommend.INTERVENTION_LOADINGS.T.to_numpy()
    return np.clip(3 + 1.5 * signal, 1, 5)


def hide(M: np.ndarray, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Hide `frac` of the observed cells. Returns (matrix with them hidden, their positions)."""
    obs = np.argwhere(~np.isnan(M))
    pick = obs[rng.random(len(obs)) < frac]
    out = M.copy()
    out[pick[:, 0], pick[:, 1]] = np.nan
    return out, pick


def fit_predict(M: np.ndarray, **kw) -> np.ndarray:
    return recommend.FunkSVD(**kw).fit(pd.DataFrame(M)).predict_all().to_numpy()


def tune(R_obs: np.ndarray, seed: int) -> tuple[dict, pd.DataFrame]:
    """Grid search with early stopping on a validation split of the observed cells."""
    R_fit, val = hide(R_obs, 0.2, np.random.default_rng(seed + 100))
    v = (val[:, 0], val[:, 1], R_obs[val[:, 0], val[:, 1]])
    rows = []
    for vals in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, vals))
        m = recommend.FunkSVD(**kw, epochs=MAX_EPOCHS, seed=seed).fit(pd.DataFrame(R_fit), val=v)
        rows.append({**kw, "epochs": m.best_epoch_, "val_rmse": min(m.val_rmse_)})
    t = pd.DataFrame(rows).sort_values("val_rmse")
    best = t.iloc[0]
    return ({"n_factors": int(best.n_factors), "lr": float(best.lr), "reg": float(best.reg),
             "epochs": int(best.epochs)}, t)


def unobserved_score(R_obs, R_all, pred, rank_pred=None) -> dict:
    """RMSE and pairwise ranking on every cell that was NOT observed."""
    cells = np.argwhere(np.isnan(R_obs))
    t = R_all[cells[:, 0], cells[:, 1]]
    h = pred[cells[:, 0], cells[:, 1]]
    hr = h if rank_pred is None else rank_pred[cells[:, 0], cells[:, 1]]
    return {"rmse": float(np.sqrt(np.mean((t - h) ** 2))), "pairwise": pairwise(cells[:, 0], t, hr)}


def pairwise(rows, truth, hat) -> float:
    d = pd.DataFrame({"u": rows, "r": truth, "p": hat})
    good = total = 0
    for _, g in d.groupby("u"):
        r, p = g.r.to_numpy(), g.p.to_numpy()
        dr, dp = r[:, None] - r[None, :], p[:, None] - p[None, :]
        m = np.triu(dr != 0, 1)
        total += m.sum()
        good += ((dr * dp) > 0)[m].sum()
    return float(good / total)


def score(M_full_obs, M_train, test, pred) -> dict:
    t = M_full_obs[test[:, 0], test[:, 1]]
    h = pred[test[:, 0], test[:, 1]]
    return {"rmse": float(np.sqrt(np.mean((t - h) ** 2))), "pairwise": pairwise(test[:, 0], t, h)}


def ndcg_at_k(true_rel: np.ndarray, scores: np.ndarray, k: int = 3) -> float:
    order = np.argsort(-scores)[:k]
    ideal = np.sort(true_rel)[::-1][:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = ((2 ** true_rel[order] - 1) * disc[:len(order)]).sum()
    idcg = ((2 ** ideal - 1) * disc[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def minmax_rows(A: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(A, axis=1, keepdims=True), np.nanmax(A, axis=1, keepdims=True)
    return np.where(hi > lo, (A - lo) / np.where(hi > lo, hi - lo, 1), 0.0)


def content_matrix(df: pd.DataFrame) -> np.ndarray:
    """Content score for every employee x intervention, from real SHAP values
    (same formula as recommend.content_scores, vectorised)."""
    b = joblib.load(config.BUNDLE_PATH)
    lists = b["lists"]
    X_all, _ = data.X_y(df, lists)
    ex = model.Explainer(b["base_model"], data.X_y(b["train"], lists)[0], lists).explain(X_all)
    S = pd.DataFrame(ex.values, columns=ex.feature_names)
    S = S.T.groupby([data.base_feature(f, lists) for f in S.columns]).sum().T.clip(lower=0)
    return np.column_stack([S.reindex(columns=t, fill_value=0).sum(axis=1).to_numpy()
                            for t in recommend.INTERVENTIONS.targets])


def main():
    df = data.prepare(data.load_raw())
    oracle = oracle_matrix(df)
    out: dict = {"data_note": "SIMULATED interaction matrix; oracle = noise-free simulated rating"}

    oracle_rank = 3 + 1.5 * (recommend.need_scores(df).to_numpy() @ recommend.INTERVENTION_LOADINGS.T.to_numpy())
    # oracle_rank: unclipped, so ties at the 1/5 bounds do not cost it ranking accuracy

    # ---------------------------------------------- 1+2. tune and score at every density
    sweep, tuned = [], {}
    for frac in DENSITIES:
        for s in SEEDS:
            R_obs = recommend.simulate_response_matrix(df, observed_frac=frac, seed=s).to_numpy()
            R_all = recommend.simulate_response_matrix(df, observed_frac=1.0, seed=s).to_numpy()
            best, table = tune(R_obs, s)
            tuned[(frac, s)] = best
            im = np.broadcast_to(np.nanmean(R_obs, axis=0), R_obs.shape)
            preds = {"svd_tuned": (fit_predict(R_obs, **best, seed=s), None),
                     "item_mean": (im, None), "oracle": (oracle, oracle_rank)}
            if frac == 0.5:
                preds["svd_skeleton"] = (fit_predict(R_obs, **SKELETON, seed=s), None)
                if s == RS:
                    out["tuning_at_0.5_seed42"] = {"grid": GRID, "max_epochs": MAX_EPOCHS, "best": best,
                                                   "top5": table.head(5).round(4).to_dict("records")}
            for name, (p, pr) in preds.items():
                sweep.append({"observed_frac": frac, "seed": s, "model": name,
                              **unobserved_score(R_obs, R_all, p, pr), **({"config": best} if name == "svd_tuned" else {})})
    sw = pd.DataFrame(sweep)
    summ = sw.groupby(["observed_frac", "model"])[["rmse", "pairwise"]].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    out["density_sweep"] = summ.reset_index().to_dict("records")
    out["tuned_configs"] = {f"{f}_{s}": c for (f, s), c in tuned.items()}
    out["random_pairwise"] = 0.5

    # ---------------------------------------------------------- 3. hybrid alpha
    C = minmax_rows(content_matrix(df))
    alphas = np.round(np.arange(0, 1.01, 0.1), 1)
    res = {a: [] for a in alphas}
    for s in SEEDS:
        R_obs = recommend.simulate_response_matrix(df, observed_frac=0.5, seed=s).to_numpy()
        R_all = recommend.simulate_response_matrix(df, observed_frac=1.0, seed=s).to_numpy()
        CF = minmax_rows(fit_predict(R_obs, **tuned[(0.5, s)], seed=s))
        unseen = np.isnan(R_obs)
        users = np.where(unseen.sum(axis=1) >= 3)[0]
        for a in alphas:
            H = a * C + (1 - a) * CF
            res[a].append(np.nanmean([ndcg_at_k(R_all[u, unseen[u]], H[u, unseen[u]]) for u in users]))
    al = pd.DataFrame({"alpha": alphas, "ndcg3_mean": [np.mean(res[a]) for a in alphas],
                       "ndcg3_std": [np.std(res[a]) for a in alphas]})
    b_row = al.loc[al.ndcg3_mean.idxmax()]
    at06 = al.set_index("alpha").loc[0.6]
    flat = (b_row.ndcg3_mean - at06.ndcg3_mean) < b_row.ndcg3_std
    out["alpha"] = {"curve": al.round(4).to_dict("records"), "best_alpha": float(b_row.alpha),
                    "flat_vs_0.6": bool(flat), "chosen": 0.6 if flat else float(b_row.alpha),
                    "note": "NDCG@3 over each employee's unobserved interventions; truth = simulated ratings. "
                            "CF is trained on the same simulation it is scored against, so this objective "
                            "favours CF by construction."}
    # Agreement between content scores and simulated truth, for the write-up
    out["content_vs_truth_spearman_mean"] = float(np.nanmean([
        pd.Series(C[u]).corr(pd.Series(R_all[u]), method="spearman") for u in range(len(C))]))

    # ---------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, st in [("svd_tuned", "-o"), ("item_mean", "--s"), ("oracle", ":^")]:
        d = sw[sw.model == name].groupby("observed_frac")
        ax[0].errorbar(d.rmse.mean().index, d.rmse.mean(), yerr=d.rmse.std(), fmt=st, capsize=3,
                       label={"svd_tuned": "Funk SVD (tuned)", "item_mean": "item mean", "oracle": "oracle"}[name])
        ax[1].errorbar(d.pairwise.mean().index, d.pairwise.mean(), yerr=d.pairwise.std(), fmt=st, capsize=3)
    ax[1].axhline(0.5, c="grey", lw=0.8)
    ax[0].set_ylabel("RMSE (unobserved cells)"); ax[1].set_ylabel("Pairwise ranking accuracy")
    for a in ax:
        a.set_xlabel("Share of employee x intervention outcomes observed")
    ax[0].legend(fontsize=8); fig.suptitle("Collaborative filtering vs data density (SIMULATED matrix)", fontsize=10)
    plt.tight_layout(); plt.savefig(config.FIG_DIR / "cf_density_sweep.png", dpi=130); plt.close()

    plt.figure(figsize=(5.5, 3.8))
    plt.errorbar(al.alpha, al.ndcg3_mean, yerr=al.ndcg3_std, marker="o", capsize=3)
    plt.axvline(out["alpha"]["chosen"], ls="--", c="k", label=f"chosen alpha = {out['alpha']['chosen']}")
    plt.xlabel("alpha (0 = collaborative only, 1 = content only)"); plt.ylabel("NDCG@3 (unobserved interventions)")
    plt.title("Hybrid weight (SIMULATED outcomes)", fontsize=10); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(config.FIG_DIR / "hybrid_alpha.png", dpi=130); plt.close()

    (config.REPORTS_DIR / "phase3_recommender.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ["tuning_at_0.5_seed42", "alpha", "content_vs_truth_spearman_mean"]},
                     indent=1, default=str))
    print(summ.to_string())


if __name__ == "__main__":
    main()
