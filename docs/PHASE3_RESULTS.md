# Phase 3 - Supporting tracks: results so far

Plan and decision rules: `docs/PHASE3_PLAN.md` (written before these runs). Every number below comes from the `reports/phase3_*.json` file of the matching script in `experiments/` (segments, recommender, paymodel, forecast, rag). All five choices are wired into `train.py`.

Status: Phase 3 closed (2026-10-09). Persona names approved by the team; sentence embeddings compared on a laptop and adopted; a live brief generated with Gemini 3.8 Flash.

## In plain words

- **Segments:** the workforce splits cleanly into three groups, and the split is stable. The group that does overtime leaves at 34%, three times the rate of the other two. Satisfaction scores are almost identical across groups, so the segments are about workload and seniority, not mood.
- **Recommender:** after fixing a hidden leak, collaborative filtering helps only once the company has logged outcomes for about half of all employee-intervention pairs, and even then it closes about a fifth of the gap to a perfect model. With an 8-item catalogue, matching interventions to each employee's risk drivers (content-based) is the part to rely on.
- **Fair-pay model:** Ridge and Lasso are tied (R2 0.872 both). We keep Ridge. Lasso's "feature selection" only removed two redundant one-hot columns.
- **Macro forecast:** a tuned ARIMA could not beat "next month = this month" for the US quits rate, so the app uses that simple forecast: 1.9% a month, a 2.6% nudge down from the past year's average.
- **HR Copilot:** the handbook now has real, cited rules. Meaning-based search (sentence embeddings) puts the right policy first for 88% of 33 test questions and in the top 3 for 97%, up from 79% and 88% with keyword search (TF-IDF), which fails when managers use different words ("quit" vs "leave").

---

## Track 1 - Workforce segments

### Choosing k

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Stability (bootstrap ARI) |
|---|---|---|---|---|
| 2 | 0.278 | 360 | 1.66 | 0.975 |
| **3** | **0.139** | **266** | **2.14** | **0.979** |
| 4 | 0.109 | 215 | 2.35 | 0.639 |
| 5 | 0.129 | 191 | 2.21 | 0.710 |
| 6 | 0.122 | 176 | 2.16 | 0.682 |

Rule result: k = 3 to 6 all pass the 0.60 stability bar; k = 3 has the best silhouette. k = 3 is also by far the most stable of them (0.98 vs 0.64-0.71). Ward's dendrogram has its biggest height jump at k = 2 (excluded by design) and its second biggest at k = 3; K-Means and Ward agree on k = 3 (ARI 0.81). Figures: `kmeans_k_selection.png`, `dendrogram.png`.

Honest caveat: a silhouette of 0.14 means the groups overlap. They are stable, but they are regions of one continuous population, not separate islands.

### The three segments (draft names, need team approval)

| Draft persona | Size | Attrition | What sets them apart (vs workforce average) |
|---|---|---|---|
| Overtime crew | 348 | 33.6% | All work overtime (+1.6 sd); slightly junior |
| Steady core | 855 | 11.5% | No overtime; slightly junior |
| Senior veterans | 267 | 8.2% | Pay +1.7 sd, job level +1.6 sd, experience +1.6 sd |

Names are attached by profile (most overtime, highest job level), not by cluster number, so a re-run cannot put a name on the wrong group (`segment.persona_names`, tested).

**Fix to the auto-descriptions:** the skeleton measured each cluster against the spread of the 3 cluster *means*, which labelled a job-satisfaction average of 2.77 vs 2.71 (out of 4) as "+1.4 sd high". Against the workforce spread, those differences vanish; the descriptions now show only differences of 0.2 sd or more.

### DBSCAN

`min_samples` = 24 (2 x 12 features), eps = 3.27 at the knee of the 24-distance curve (87th percentile). Result: **one dense cluster plus 32 outliers**, and still one cluster at eps +/-10% (99 and 10 outliers). DBSCAN therefore finds no density-separated groups here; we use it as an unusual-profile detector. The 32 outliers leave at 6% vs 16% for everyone else. Figure: `dbscan_k_distance.png`.

### SHAP "why-segments"

| Variant (k = 3) | Silhouette | Stability | Link to attrition (Cramer's V) |
|---|---|---|---|
| Feature clusters (above) | 0.139 | 0.98 | **0.27** |
| SHAP, standardised (skeleton) | 0.109 | 0.96 | 0.12 |
| SHAP, raw log-odds | 0.189 | 0.98 | 0.19 |
| SHAP, raw, pay & seniority summed | 0.194 | 1.00 | 0.20 |

Standardising SHAP values was a mistake (they already share a unit), so the default is now raw. Even then, SHAP clusters separate leavers less well than plain feature clusters, and their profiles are dominated by JobRole and pay/seniority pushing in opposite directions: the same collinearity seen in Phase 2. **Decision: feature clusters are the main segmentation; SHAP clusters are reported as a tested alternative.** Possible follow-up: also fold JobRole into the "Pay & seniority" group (open decision 6).

### RFM-style bands

Attrition is 15-17% in every band (Neglected 16.9%, At risk 15.3%, Steady 16.9%, Invested 15.7%). The RFM analogue does not separate leavers; show it as a finding, not as a risk signal.

---

## Track 3 - Retention recommender

### A leak we found and fixed

The first tuning run showed tuned Funk SVD **beating the oracle** (the noise-free truth) at ranking, which is impossible. Cause: the simulator and Funk SVD both called `np.random.default_rng(42)`, so the model's first random draw (the user factors, 1,470 x 8) was exactly the simulator's noise matrix, scaled down. The model started out knowing each employee's simulated noise, and tuning rewarded 8 factors because that copied all of it. Fix: Funk SVD now draws from its own stream (`default_rng([seed, 1])`), with a regression test. The numba-compiled SGD loop reproduces the old Python loop's results exactly and is about 100x faster.

### How much outcome data does CF need? (scored on every unobserved cell, mean of 3 seeds)

| Outcomes observed | RMSE: item mean | RMSE: tuned SVD | RMSE: oracle | Ranking: SVD | Ranking: oracle |
|---|---|---|---|---|---|
| 10% | 1.005 | 1.004 | 0.650 | 0.50 | 0.80 |
| 20% | 1.005 | 0.985 | 0.650 | 0.50 | 0.80 |
| 30% | 1.002 | 0.977 | 0.648 | 0.53 | 0.80 |
| 50% | 1.003 | 0.929 | 0.648 | 0.62 | 0.80 |
| 80% | 1.013 | 0.869 | 0.647 | 0.69 | 0.81 |

Ranking = share of an employee's intervention pairs put in the right order (random = 0.50). At 50% density CF closes 21% of the RMSE gap between "recommend what usually works" and a perfect model; at 80%, 39%. Below 30% it is no better than the item average. Figure: `cf_density_sweep.png`.

Why so data-hungry: with only 8 interventions, each employee has at most 8 outcomes, so their 4 underlying needs must be inferred from a handful of noisy ratings. Business message: **start logging intervention outcomes now; CF becomes useful only after a large share of employees have tried several interventions.**

In `train.py`'s quick hold-out check (`metrics.json`, 20% of observed ratings hidden): tuned RMSE 0.971 vs item mean 1.002, pairwise ranking 0.60 vs 0.51 (only ~300 pairs, so noisy). The Phase 2 run reported 0.960 and 0.57 with the old settings, but that was inflated by the leak: with the leak fixed, the old settings score 0.971 and rank at chance (0.50).

Tuning: validation RMSE is flat (top 10 settings within 0.007), so the new defaults (8 factors, lr 0.02, reg 0.05, 150 epochs) are sensible rather than sharp.

### Hybrid weight alpha

NDCG@3 on unobserved interventions peaks at alpha = 0.4 (0.837) vs 0.833 at 0.6; the gap is smaller than the seed-to-seed std (0.004), so by the rule **alpha stays 0.6**. Two caveats: CF is scored against the same simulation it was trained on, which favours CF by construction; and content scores agree only weakly with the simulated truth (mean Spearman 0.21), because the simulation's "needs" are rules we invented while content scores come from the real model. Real alpha needs real outcomes. Figure: `hybrid_alpha.png`.

Catalogue costs: unchanged and still labelled TEAM ASSUMPTION (open decision 2).

---

## Track 4 part 1 - Fair-pay model (Ridge vs Lasso)

Repeated 5x3 CV on the training split, same folds for both, alpha grid widened to 1e-6..1e2:

| | R2 (mean +/- std) | RMSE (log pay) | alpha | Zeroed coefficients |
|---|---|---|---|---|
| Ridge | 0.8722 +/- 0.0131 | 0.2346 | 1.10 | none |
| Lasso | 0.8723 +/- 0.0132 | 0.2346 | 9.1e-5 | JobRole_Manufacturing Director, Department_Sales |

Paired difference: 0.00003 +/- 0.00029, Ridge wins 7 of 15 folds: a tie, so **Ridge stays**. Lasso's alpha is no longer on the grid edge, and it is tiny: with about 1,200 rows and 16 inputs, regularisation barely matters.

Lasso's two zeros are not feature selection. The one-hot encoder keeps every category, so each set of dummies sums to 1, and JobRole sits almost entirely inside Department (every role except Manager belongs to one department). Lasso simply dropped one redundant column from each set. JobLevel and TotalWorkingYears correlate at 0.79; Ridge keeps both with shared weight, which is why it is the better story for "fair pay".

---

## Track 4 part 2 - Macro forecast (US quits rate)

Data: FRED series JTSQUR (BLS JOLTS quits rate, total nonfarm, seasonally adjusted), Dec 2000 - Aug 2026, 309 months, saved as `data/external/JTSQUR.csv` (snapshot committed, since FRED revises data). Checked against FRED at download: 309 values, sum 620.4.

| Step | Result |
|---|---|
| Differencing (ADF + KPSS) | Levels: ADF p = 0.20, KPSS p = 0.01, not stationary. First difference: ADF p = 0.0001, KPSS p = 0.10, stationary. d = 1 |
| Order by AIC (p, q in 0-3, d = 1) | ARIMA(2,1,3), AIC -569.6; runner-up (3,1,2) -569.3; simple (0,1,1) -564.1 |
| Rolling CV, 6 origins x 12 months | MAE: ARIMA 0.197, naive 0.179, seasonal naive 0.296. ARIMA beat naive in 1 of 6 folds |
| Excluding folds starting before 2023 | ARIMA 0.119 vs naive 0.092 (3 folds) |
| Secondary check, ARIMA(0,1,1) | MAE 0.178 vs naive 0.179, beat it in 1 of 6 folds: a tie |

**Decision (rule fixed in the plan): naive forecast.** At a 12-month horizon the quits rate behaves like a random walk, and the extra ARIMA terms fit noise. The forecast is 1.9% for every month to Aug 2027, with an 80% interval of 1.77-2.03% next month widening to 1.45-2.35% at 12 months (random-walk interval). Macro ratio = 1.9 / 1.95 (last 12 months' average) = 0.974, so expected attrition in the workforce forecast is scaled down by 2.6%. Prophet was not run (optional; heavy to install on Windows). Figure: `jolts_forecast.png`.

## Track 5 - HR Copilot

### Handbook

Every `[N]` placeholder is gone. Each rule now either cites a public source or is marked "company choice":

- **Law (checked against the EUR-Lex texts):** 48-hour average week over up to four months, 11 h daily rest, 24 + 11 h weekly rest (Working Time Directive 2003/88/EC, Art. 3, 5, 6, 16); 5% pay-gap trigger with six months to fix (Pay Transparency Directive 2023/970, Art. 10); no solely automated decisions (GDPR Art. 22); promotion/termination/monitoring AI is high-risk and workers must be told first (AI Act, Annex III point 4(b), Art. 26(7)); two flexible-working requests a year, decision within two months (UK Employment Relations (Flexible Working) Act 2023).
- **Practice (GitLab's public handbook):** semi-annual promotion calibration; six months in role before internal applications; weekly 1-1s; pay range midpoint at the 50th percentile with the minimum at 80% of it; RSUs vesting over two to four years; up to US$10,000 a year learning fund.

Reference list: `data/knowledge_base/SOURCES.md`.

### Retrieval (33 questions, 10 sections, 13 chunks)

13 new questions were added to the original 20, worded differently from the handbook on purpose and labelled before any retrieval ran.

| Set-up | Metric | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| TF-IDF, L2-normalised | cosine | 0.788 | 0.879 | 0.847 |
| TF-IDF, L2-normalised | Euclidean | 0.788 | 0.879 | 0.847 |
| TF-IDF, raw | cosine | 0.788 | 0.879 | 0.847 |
| TF-IDF, raw | Euclidean | 0.121 | 0.333 | 0.326 |
| **Sentence embeddings (MiniLM)** | cosine | **0.879** | **0.970** | **0.930** |
| Sentence embeddings (MiniLM) | Euclidean | 0.879 | 0.970 | 0.930 |

MiniLM rows: run on Raj's laptop (`python -m experiments.phase3_rag`, `reports/phase3_rag.json`), since Hugging Face is blocked in the build sandbox. TF-IDF rows: after the tie-break fix below, identical on both machines.

- **Cosine = Euclidean on normalised vectors**, exactly as theory says. Without normalisation, Euclidean distance mostly measures chunk length (vector norms 13.8-33.1), so long policy sections are "far" from every short question and retrieval collapses. Cosine ignores length, so it is unaffected.
- **A bug we found:** the first run showed cosine and Euclidean differing on normalised TF-IDF. One question ("How do we talk to someone who might quit?") shares no word with the handbook, so its vector is all zeros and an unstable sort decided the order. Ties now sort stably, empty queries count as misses, and the app says "no policy found" for them.
- **Where TF-IDF fails:** 7 of 33 questions do not get the right section first, all from vocabulary mismatch ("quit"/"resign" vs "leave", "sign off" vs "approval", "log back on" vs "rest"). Sentence embeddings fix most of these: MRR 0.930 vs 0.847. **Decision (rule fixed in the plan): switch to MiniLM with cosine.** The app already prefers it and falls back to TF-IDF automatically when `sentence-transformers` is not installed, so a light deployment still works (Phase 6 decides which one the host runs). MiniLM vectors are unit length too, so cosine and Euclidean again rank identically.
- **Tie-break fix:** on the laptop, TF-IDF cosine and Euclidean first differed slightly (MRR 0.842 vs 0.844). The laptop has `faiss` installed, which returned equal scores in its own order and with float32 noise; the sandbox used the numpy path. `VectorIndex.search` now uses faiss only to pick candidates, re-scores them exactly in float64, and treats scores within 1e-6 as tied, broken by handbook order. Verified in the sandbox with and without faiss: cosine = Euclidean = 0.847 MRR either way.
- **App query fix:** the Copilot used to append raw feature names ("MonthlyIncome") to the question, which match nothing in the handbook. They are now translated to handbook words (`rag.FEATURE_WORDS`, e.g. "salary pay compensation").

### LLM

`rag.generate` supports Gemini and Claude through environment variables (README, "Optional: LLM"). A live brief was generated with **Gemini 3.8 Flash** (`LLM_PROVIDER=gemini`, `LLM_MODEL` set to the 3.8 Flash model name from Google AI Studio). Without a key the app shows the grounded inputs instead of a brief, so the demo still works offline. The key lives only in the shell environment, never in the repo.

## Closed items (2026-10-09)

| Item | Outcome |
|---|---|
| Persona names | Approved by the team: Overtime crew / Steady core / Senior veterans |
| Sentence embeddings vs TF-IDF | MiniLM adopted (MRR 0.930 vs 0.847) |
| Live LLM brief | Generated with Gemini 3.8 Flash |

Carried to Phase 6: whether the host installs `sentence-transformers` (PyTorch is heavy on free tiers) or runs the TF-IDF fallback.
