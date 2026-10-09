# Phase 3 - Supporting tracks: plan

Written 2026-10-07, before running any Phase 3 experiment. Decision rules are fixed here so the results cannot steer them. Starting numbers come from the Phase 2 run (`reports/metrics.json`), which already ran the skeleton version of every track.

Order (solo build): Track 1 -> Track 3 -> Track 4 -> Track 5. Each track ships as an experiment script in `experiments/`, a results file in `reports/`, and a short write-up. `src/` and `train.py` change only after the experiment settles the choice.

## Where each track starts (Phase 2 skeleton run)

| Track | Skeleton result | Problem it shows |
|---|---|---|
| 1 Segments | k = 3, silhouette 0.139 (k = 2 scores 0.278); DBSCAN 1 cluster + 66 outliers; SHAP clusters silhouette 0.078 | Weak cluster structure. k was picked as "best silhouette in 3-6" with no stability check. eps = 90th percentile of k-distance, not the knee |
| 3 Recommender | CF RMSE 0.960 vs item-mean 1.002; pairwise ranking 0.567 vs 0.510 | The simulated matrix is low-rank by design, yet CF barely beats the baseline: Funk SVD is probably under-trained (30 epochs, lr 0.01, no tuning) |
| 4 Forecast | Ridge R2 0.873 vs Lasso 0.873; Lasso alpha = 1e-4 (grid edge); JOLTS skipped | Lasso alpha sits on the grid boundary. The ARIMA grid mixes d = 0 and d = 1, and AIC is not comparable across different differencing orders |
| 5 Copilot | TF-IDF, 20 questions, hit@3 0.90 and MRR 0.875 for both cosine and Euclidean | Placeholder handbook. Cosine = Euclidean is expected, not a finding: TF-IDF rows are L2-normalised, and on unit vectors Euclidean distance is a monotone function of cosine (d^2 = 2 - 2cos), so the rankings are identical |

## Track 1 - Workforce segments

Method (`experiments/phase3_segments.py`):
1. k = 2..8: inertia (elbow), silhouette, Calinski-Harabasz, Davies-Bouldin, and **bootstrap stability** (refit on 50 resamples, adjusted Rand index vs the full-data clustering, on the shared rows).
2. Ward dendrogram: the k where the merge-height jump is largest.
3. DBSCAN: `min_samples = 2 x 12 features = 24` (Sander et al. 1998 rule of thumb), k-distance with the same k, eps = knee found by the maximum-distance-to-chord method (the Kneedle idea, coded directly, no new dependency).
4. SHAP "why-segments": SHAP values of the deployed model for all 1,470 employees, same k-selection table.
5. Compare feature clusters vs SHAP clusters on what HR cares about: spread of attrition rate across segments.

**Decision rule for k:** consider k = 3-6 only (k = 2 just splits senior from junior, too coarse to act on). Drop any k with bootstrap ARI < 0.60. Among the rest, choose the highest silhouette; if silhouettes are within 0.01, choose the smaller k. If no k passes, report that the workforce has no stable segments beyond seniority and present k = 2 plus RFM bands instead.

**Done when:** k and eps justified by figures; persona names drafted from profiles and approved by the team (open decision 4); honest statement of how strong the structure is.

## Track 3 - Retention recommender

Method (`experiments/phase3_recommender.py`):
1. **Oracle benchmark.** The simulation is known, so the best achievable RMSE (noise-free signal vs noisy rating) is computable. CF is judged against item-mean (floor) and oracle (ceiling).
2. **Tune Funk SVD** on a validation split carved from the training cells only (test cells untouched): epochs, learning rate, regularisation, factors.
3. **Density sweep:** observed share 0.1-0.8, 3 seeds each; RMSE and pairwise ranking for tuned SVD, item-mean and oracle. Business reading: how much outcome data the company must log before CF beats "recommend what usually works".
4. **Hybrid weight alpha:** grid 0-1 in 0.1 steps; objective = NDCG@3 against held-out simulated ratings. Caveat stated in the write-up: tuning against simulated outcomes demonstrates the procedure; real alpha needs real outcomes.
5. **Catalogue costs:** stay labelled TEAM ASSUMPTION unless a source is found; no invented figures.

**Decision rule for alpha:** pick the alpha with the best mean NDCG@3; if the curve is flat (best minus alpha = 0.6 below one std across seeds), keep 0.6 and say the data cannot distinguish them.

**Done when:** CF clearly beats item-mean on the simulated data (or the reason it cannot is explained), density plot exists, alpha chosen by the rule.

## Track 4 - Predictive operations

1. **Ridge vs Lasso** (runs now): widen the Lasso alpha grid below 1e-4 so the optimum is not on the edge; coefficient comparison; collinearity note (JobLevel vs TotalWorkingYears). Expected conclusion if R2 stays tied: keep Ridge, as it shares weight across correlated predictors instead of arbitrarily dropping one.
2. **JOLTS (blocked):** the sandbox cannot reach FRED. Needs `data/external/JTSQUR.csv` from https://fred.stlouisfed.org/series/JTSQUR (Download -> CSV).
3. **ARIMA, once the CSV is in:** choose d first with ADF and KPSS tests, then p, q by AIC within that d. Rolling-origin CV with 6 origins x 12-month horizon vs naive last-value and seasonal-naive; report with and without the 2020-21 shock. Prophet only if time allows (install is heavy on Windows).

**Done when:** ARIMA beats naive in rolling CV, or the write-up says it does not and the app uses naive.

## Track 5 - HR Copilot

1. **Handbook:** replace every `[N]` with a value taken from a cited public source (decision needed: which sources). Keep the `## ` headings as retrieval units.
2. **Eval set:** 20 -> 32+ questions, paraphrased so they do not reuse the handbook's words, labelled before running retrieval.
3. **Retrieval comparison:** TF-IDF vs sentence-transformers (`all-MiniLM-L6-v2`), hit@1, hit@3, MRR. Cosine vs Euclidean shown on **un-normalised** vectors too (TF-IDF with `norm=None`), since on normalised vectors the two are provably identical.
4. **LLM (decision needed):** provider and API key, held in an environment variable / host secret, never in the repo.

**Done when:** handbook has no placeholders and every value is cited; 32+ question eval with both embedding backends reported; one live LLM brief generated.

## Blockers needing a decision

1. Phase 2 is not committed yet. Commit it first so Phase 3 lands as its own commit (`train.py` is touched by both).
2. JOLTS CSV download (sandbox has no FRED access).
3. LLM provider for the brief (open decision 3).
4. Public sources the handbook is based on.
5. Persona names (open decision 4) - drafts come out of Track 1.
