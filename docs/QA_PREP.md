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

## Segments (Track 1)
- **How did you choose k?** Elbow, silhouette and the dendrogram. Silhouette is low (overlapping groups), which is normal for HR data; say so.
- **Why scale before clustering?** K-Means uses distances; unscaled income would dominate.
- **What does DBSCAN add?** It finds dense groups without fixing k and labels unusual profiles as outliers.

## Recommender (Track 3)
- **Where does the collaborative data come from?** It is simulated: no real record of interventions exists. We say this on screen. The content-based part uses real model output.
- **What is Funk SVD?** Factorises the sparse rating matrix into employee and intervention factors, learned by gradient descent on observed cells only, with L2 regularisation.
- **Does CF beat a baseline?** Compare RMSE and pairwise ranking vs the item-mean baseline in metrics.json. Gains are small on sparse data, which is why content-based gets more weight.

## Forecast (Track 4)
- **Ridge vs Lasso?** Ridge (L2) shrinks all coefficients; Lasso (L1) can set some to zero (feature selection). Report CV R² and how many coefficients Lasso zeroed.
- **Why log income?** Pay is right-skewed; the log makes errors proportional.
- **How did you validate ARIMA?** Rolling-origin time-series CV against a naive last-value forecast. Random k-fold would leak the future into the past.
- **Is the US quits rate relevant to this company?** Only as a macro signal; we state the proportionality assumption.

## HR Copilot (Track 5)
- **What is RAG?** Retrieve relevant policy text, then make the LLM answer only from it, citing sections. Reduces invented answers.
- **Cosine vs Euclidean?** Cosine compares direction; Euclidean also counts vector length. On L2-normalised vectors (e.g. TF-IDF) they rank results identically, because squared distance = 2 - 2 x cosine.
- **How did you evaluate retrieval?** Hit@3 and MRR on hand-labelled questions.

## Engineering
- **How do you prevent leakage?** Everything that learns sits inside the pipeline; the threshold uses out-of-fold predictions; the test set is touched once.
- **How is it reproducible?** Fixed random seeds, pinned requirements, one command to train, tests.
