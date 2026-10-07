# Q&A preparation

Every member should be able to answer all of these. Draft answers are starting points: replace them with your own results from `reports/metrics.json`.

## Data
- **Is the data real?** No. IBM data scientists created it as a synthetic dataset. It is clean and well-known, which lets us focus on method; results would need re-validation on real HR data.
- **Why drop EmployeeCount, StandardHours, Over18, EmployeeNumber?** The first three never vary. EmployeeNumber is an ID; it would let the model memorise rows.
- **Why treat satisfaction scores as numbers?** They are ordered 1-4. Treating them as numeric assumes equal spacing; one-hot would avoid that but add columns. (Test both if challenged.)

## Risk model (Track 2)
- **Why not accuracy?** About 16% leave; predicting "nobody leaves" scores ~84%. PR-AUC focuses on how well we find leavers.
- **How did you handle imbalance?** Class weights (LR, RF), scale_pos_weight (XGBoost), SMOTE (MLP), plus a cost-based threshold.
- **Why did the simpler model win?** With ~1,200 training rows, the extra flexibility of trees and neural nets mostly fits noise. Show the CV mean and std.
- **What is in the search space and why?** Read `model_specs`: regularisation strength, tree depth, learning rate. Explain what each does to bias and variance.
- **What does calibration do?** Rescales scores so a 0.7 means ~70% of such employees actually leave. Needed because the cost model multiplies probabilities by money.
- **How is the threshold chosen?** Minimises total expected cost on out-of-fold training predictions, given replacement cost, intervention cost and success rate.
- **Is SHAP causal?** No. It shows how the model uses each feature. "Overtime raises predicted risk" is not "removing overtime will retain them".
- **Why are SHAP values in log-odds?** For linear and tree models, SHAP is additive in the model's raw output (log-odds). Calibration is monotonic, so the ranking of drivers does not change.
- **Fairness?** Sensitive attributes are excluded as inputs, but other features can act as proxies (e.g. TotalWorkingYears for age). That is why we check recall by group on outputs.

### Phase 2 results (numbers from `docs/PHASE2_RISK_MODEL.md`)
- **Which model and why?** Logistic regression: highest CV PR-AUC (0.625 ± 0.054). XGBoost (0.606) is statistically tied with it (paired corrected t-test p = 0.39), and our pre-set rule says the simplest tied model wins. The rule was in the code before the full run.
- **Isn't tuning and scoring on the same folds optimistic?** We checked with nested CV: 0.654 vs 0.625. There is no sign of inflation, since LR has only two hyperparameters.
- **Why is test PR-AUC (0.562) lower than CV (0.625)?** The test set has 47 leavers, and its 95% CI is 0.42-0.69, which contains the CV value. We did not change anything after looking at the test set.
- **Why a threshold of 0.32 when theory says 0.21?** Theory assumes perfect calibration. The model over-predicts in the 0.21-0.32 band (26% predicted vs 18% actual), and the cost curve is flat from 0.27 to 0.35 (within about 1%).
- **Why does the threshold not depend on salary?** Both the intervention cost and the replacement cost are multiples of the same salary, so it cancels: p* = months / (12 x multiplier x success).
- **What if your cost assumptions are wrong?** Savings are positive in 26 of 27 scenarios (0.5-2x replacement, 0.5-2 months intervention, 20-60% success). The only loss is the all-pessimistic corner.
- **Why does SHAP say low income lowers risk?** Collinearity: income, JobLevel (r = 0.95) and PayGapPct move together, so only their sum is meaningful. Dropping income costs PR-AUC (p = 0.045), so it stays.
- **Is the model unfair to older employees?** Recall falls with age (69% under 30, 40% at 50+), but average predicted risk for 50+ matches their actual rate. The cause is the seniority signal (long careers, senior roles read as "stays"). Adding Age as an input does not fix it (50+ recall stays at 40%) and lowers female recall from 61% to 51%.
- **Who does the model miss, and why?** Experienced, senior, non-overtime employees who look like stayers on every recorded feature. It catches 1 of 19 senior-role leavers. The data has no exit reason, so retirement cannot be told apart from resignation.

## Segments (Track 1)
- **How did you choose k?** A rule fixed before running: k between 3 and 6, drop any k whose bootstrap stability (ARI over 50 resamples) is below 0.60, then highest silhouette. k = 3 wins: stability 0.98 vs 0.64-0.71 for k = 4-6, and Ward's dendrogram agrees (ARI 0.81). Silhouette is only 0.14, so the groups overlap: stable regions of one population, not islands.
- **Why not k = 2?** It has the best silhouette (0.28) but just splits senior from junior; nothing to act on.
- **What are the segments?** Overtime crew (34% attrition), Steady core (11%), Senior veterans (8%). Satisfaction scores barely differ between them, so the split is workload and seniority, not mood.
- **Why scale before clustering?** K-Means uses distances; unscaled income would dominate.
- **What does DBSCAN add?** Here, only outlier detection: with min_samples = 24 (2 x 12 features) and eps at the k-distance knee (3.27) it finds one dense cluster plus 32 unusual profiles, and still one cluster at eps +/-10%.
- **Why not cluster on SHAP values?** We tried. Raw SHAP (not standardised: they already share the log-odds unit) gives stable groups, but they separate leavers less well than feature clusters (Cramer's V 0.20 vs 0.27) and are dominated by job role and pay/seniority pulling in opposite directions (collinearity).

## Recommender (Track 3)
- **Where does the collaborative data come from?** It is simulated: no real record of interventions exists. We say this on screen. The content-based part uses real model output.
- **What is Funk SVD?** Factorises the sparse rating matrix into employee and intervention factors, learned by gradient descent on observed cells only, with L2 regularisation.
- **Does CF beat a baseline?** Only with lots of data. Against item-mean (floor) and the noise-free oracle (ceiling), CF closes 21% of the RMSE gap at 50% of outcomes observed, 39% at 80%, and nothing below 30%. With 8 interventions each employee has at most 8 outcomes, so their needs are hard to infer.
- **What went wrong on the way?** Funk SVD and the simulator both used seed 42, so the model's starting factors were the simulated noise; it "beat" the oracle. Spotted because beating a perfect model is impossible; fixed with a separate random stream and a regression test.
- **How was alpha chosen?** NDCG@3 on unobserved interventions; the curve is flat (0.4 vs 0.6 differ by less than the seed-to-seed std), so 0.6 stays. CF is scored against the simulation it learned from, so this test favours CF anyway.

## Forecast (Track 4)
- **Ridge vs Lasso?** Tied: R2 0.872 both, paired difference 0.00003 +/- 0.00029 over 15 folds. We keep Ridge. Lasso zeroed two columns, but that is not feature selection: each set of one-hot dummies sums to 1 and job role sits inside department, so Lasso just dropped redundant columns.
- **Why log income?** Pay is right-skewed; the log makes errors proportional.
- **How did you pick the ARIMA order?** d first, from ADF and KPSS together (d = 1), then p and q by AIC: ARIMA(2,1,3). AIC is not comparable across different d, which the first version got wrong.
- **How did you validate it?** Rolling-origin CV, 6 origins x 12 months. ARIMA lost to "next month = this month" (MAE 0.197 vs 0.179; won 1 of 6 folds), and so did the simpler ARIMA(0,1,1). The quits rate behaves like a random walk at a 12-month horizon, so the app uses the naive forecast. Random k-fold would leak the future into the past.
- **Is the US quits rate relevant to this company?** Only as a macro signal; we state the proportionality assumption.

## HR Copilot (Track 5)
- **What is RAG?** Retrieve relevant policy text, then make the LLM answer only from it, citing sections. Reduces invented answers.
- **Cosine vs Euclidean?** Cosine compares direction; Euclidean also counts vector length. On L2-normalised vectors (e.g. TF-IDF) they rank results identically, because squared distance = 2 - 2 x cosine.
- **How did you evaluate retrieval?** 33 hand-labelled manager questions, worded differently from the handbook and written before any retrieval run: TF-IDF hit@1 0.79, hit@3 0.88, MRR 0.85. On un-normalised TF-IDF, Euclidean collapses (MRR 0.33) because long chunks sit far from short questions; cosine is unaffected.
- **Where does TF-IDF fail?** Vocabulary mismatch: "quit" vs "leave", "sign off" vs "approval". One question shares no word with the handbook at all; the app now says "no policy found" instead of returning a random section. Sentence embeddings are the fix to test.
- **Where do the policies come from?** Each rule cites the EU Working Time, Pay Transparency, GDPR and AI Act texts, the UK Flexible Working Act 2023, or GitLab's public handbook; the rest is marked "company choice". See data/knowledge_base/SOURCES.md.

## Engineering
- **How do you prevent leakage?** Everything that learns sits inside the pipeline; the threshold uses out-of-fold predictions; the test set is touched once.
- **How is it reproducible?** Fixed random seeds, pinned requirements, one command to train, tests.
