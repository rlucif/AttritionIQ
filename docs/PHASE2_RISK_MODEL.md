# Phase 2 - Core risk model (Track 2)

Run: `python train.py` (full, 25 search iterations per model), 2026-10-07. Every number below comes from `reports/metrics.json`, `reports/roi_sensitivity.csv` or the two `reports/missed_leavers_*.csv` files written by that run. Money is in dataset units ($).

## In plain words

- **Model:** a regularised logistic regression. It scored best in cross-validation, and it is also the simplest and easiest to explain of the four candidates.
- **Who gets flagged:** anyone with a 32%+ chance of leaving within the year. That cut-off minimises total cost (lost employees + retention spend) on training data.
- **What it saves:** on 294 held-out employees it flags 36 and saves about $181k (7%) compared with doing nothing. Even in a bad-luck resample, savings stay positive (95% range $43k-$338k).
- **When it stops paying off:** only when all three business assumptions are pessimistic at once (cheap replacement, expensive intervention, low success rate).
- **What it misses:** experienced, senior people. It catches 69% of leavers under 30 but about 40% of leavers aged 40+, and almost no leavers in director-level roles. Adding age as an input does not fix this, so we keep sensitive attributes out.

---

## 1. Model comparison (p2-1)

All four models were tuned with `RandomizedSearchCV` (25 draws, stratified 5-fold, PR-AUC), then scored on the **same** 15 folds (5-fold x 3 repeats, training data only). Because the folds are identical, the models can be compared fold by fold.

| Model | PR-AUC (mean ± std) | ROC-AUC | Recall @0.5 | Precision @0.5 | Brier (uncalibrated) |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.625 ± 0.054** | 0.827 | 0.737 | 0.380 | 0.157 |
| XGBoost | 0.606 ± 0.055 | 0.819 | 0.632 | 0.463 | 0.128 |
| Neural Network (MLP) | 0.570 ± 0.061 | 0.798 | 0.611 | 0.446 | 0.135 |
| Random Forest | 0.525 ± 0.049 | 0.788 | 0.388 | 0.538 | 0.130 |

Random baseline PR-AUC = attrition rate = 0.16. Figure: `reports/figures/model_comparison_cv.png`.

**Paired comparison, LR vs each model** (Nadeau-Bengio corrected t-test, same as Phase 1):

| vs | LR better by | Folds LR won | p (corrected) |
|---|---|---|---|
| XGBoost | 0.019 | 10 / 15 | 0.39 |
| MLP | 0.055 | 13 / 15 | 0.17 |
| Random Forest | 0.100 | 15 / 15 | 0.006 |

Tuned LR: `C = 2.64`, `l1_ratio = 0.44` (elastic net: about half Lasso, half Ridge penalty).

## 2. Model choice (p2-2)

**Rule (written into `model.select_model` before the full run):** rank by mean PR-AUC. Any model within one standard deviation (0.054) of the best counts as tied. Among tied models, the simplest wins (LR < RF < XGBoost < MLP).

**Result:** LR has the highest mean, and XGBoost is tied with it (0.606 is within 0.054). LR wins on both counts, so the tie rule did not decide anything this time. The MLP misses the tie band by 0.001 (0.5697 vs 0.5707), so it is a borderline case. It doesn't change the choice either way.

**Why the simplest model wins here:** there are about 1,200 training rows and 190 leavers. Trees and neural nets have more freedom to fit noise, and the paired test cannot tell XGBoost from LR (p = 0.39). With performance equal, LR is faster, gives exact SHAP values, and every coefficient can be read out in the Q&A.

**Was tuning on the same data optimistic?** Nested CV (tuning repeated inside each of 5 outer folds) gives PR-AUC **0.654 ± 0.053**, against 0.625 for the plain estimate. There is no sign that tuning inflated the score. The difference comes from a different fold split, which is within one std. LR has only two hyperparameters, so there is little room to overfit the search.

## 3. Calibration, threshold, ROI (p2-3)

### Calibration
`class_weight='balanced'` pushes LR probabilities up: it predicts 34% average risk where 16% actually leave. Platt scaling corrects this.

| | ECE (avg. gap predicted vs actual) | Brier | Mean predicted | Actual rate |
|---|---|---|---|---|
| Test, uncalibrated | 0.192 | 0.159 | 0.342 | 0.160 |
| **Test, calibrated** | **0.044** | **0.103** | 0.148 | 0.160 |
| Out-of-fold train, calibrated | 0.042 | 0.096 | 0.163 | 0.162 |

It isn't perfect: on out-of-fold train, the 0.21-0.32 band averages 26% predicted vs 18% actual, and the 0.5+ band averages 65% predicted vs 85% actual. Figure: `calibration_test.png`.

### Threshold
- **Cost-optimal threshold: 0.32**, chosen on out-of-fold training predictions.
- **Theory check:** flag when p x R x s > C. Both costs scale with the same salary, so salary cancels and **p\* = months / (12 x multiplier x success) = 1 / (12 x 1 x 0.4) = 0.21**.
- **Why 0.32 and not 0.21:** (1) the cost curve is flat between 0.27 and 0.35 (total cost $9.79M-$9.90M, within about 1%), so any value in that range is defensible. (2) In the 0.21-0.32 band the model over-predicts (26% predicted vs 18% actual), and the leavers there are lower-paid (salary-weighted rate 12%). Flagging that band therefore costs more than it saves. Figure: `threshold_cost_curve.png` shows both lines.

### Held-out test (used once, n = 294, 47 leavers)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| PR-AUC | 0.562 (random = 0.160) | 0.42 - 0.69 |
| ROC-AUC | 0.786 | 0.71 - 0.86 |
| Recall @0.32 | 0.404 (19 of 47 leavers caught) | 0.27 - 0.55 |
| Precision @0.32 | 0.528 (19 of 36 flags correct) | 0.36 - 0.70 |

Confusion matrix @0.32: TP 19, FP 17, FN 28, TN 230. At 0.5 the model would catch only 28% of leavers, at 87% precision.

Test PR-AUC is below the CV mean (0.625), but CV sits inside the test CI. With 47 test leavers this gap is expected noise. The run before Phase 1's feature drop scored 0.601 on this test set. That is also inside the CI, and **we do not revisit the Phase 1 decision based on test results** (that would turn the test set into a tuning set).

### ROI on test (replacement = 1x salary, intervention = 1 month, success = 40%)

| Policy | Total cost |
|---|---|
| Do nothing | $2,570,448 |
| Intervene with everyone | $3,405,805 |
| **Model policy (flag 36, spend $145,767)** | **$2,389,316** |
| **Savings vs do nothing** | **$181,132 (7.0%), 95% CI $43k - $338k** |

### Sensitivity: does the conclusion survive other assumptions? (`reports/roi_sensitivity.csv`)
The threshold is re-optimised on train for every combination, then scored on test. Savings vs doing nothing ($):

| Replacement x salary | Intervention months | success 20% | success 40% | success 60% |
|---|---|---|---|---|
| 0.5 | 0.5 | 32,593 | 90,566 | 173,234 |
| 0.5 | 1 | 4,758 | 65,185 | 123,350 |
| 0.5 | 2 | **-2,634** | 9,516 | 77,443 |
| 1 | 0.5 | 90,566 | 338,998 | 591,351 |
| **1** | **1** | 65,185 | **181,132** | 346,469 |
| 1 | 2 | 9,516 | 130,371 | 246,701 |
| 2 | 0.5 | 338,998 | 1,318,282 | 2,212,454 |
| 2 | 1 | 181,132 | 677,995 | 1,182,701 |
| 2 | 2 | 130,371 | 362,264 | 692,938 |

The model saves money in 26 of 27 scenarios. The single loss is the all-pessimistic corner, where the break-even probability exceeds 100% and the right answer is "don't run a programme". The two unsourced assumptions (intervention cost and success rate) change the size of the saving, not its sign. Keep presenting them as scenarios until someone finds sources.

## 4. SHAP explanations (p2-4)

**Global** (mean |SHAP| on the test set, one-hot columns summed per feature): JobRole 1.75, MonthlyIncome 1.19, TotalWorkingYears 0.81, OverTime 0.70, EnvironmentSatisfaction 0.43, PayGapPct 0.40, NumCompaniesWorked 0.38, StockOptionLevel 0.37, YearsSinceLastPromotion 0.34, BusinessTravel 0.33. Figures: `shap_global_bar.png`, `shap_beeswarm.png`.

**Individual** (`shap_waterfall_*.png`, bars in log-odds):
- *Caught leaver*, #478, P(leave) 93%: JobRole +1.79, TotalWorkingYears +1.09, OverTime +1.02, BusinessTravel +0.67.
- *Missed leaver*, #1082, 15%: Healthcare Representative, non-traveller, no overtime. These pull risk down; a long commute (24) and job satisfaction of 1 push it up, but not far enough.
- *False alarm*, #411, 61%: JobRole +1.79, TotalWorkingYears +1.50, PayGapPct +1.12.

### Caveat: pay and seniority drivers are collinear
MonthlyIncome correlates 0.95 with JobLevel and 0.77 with TotalWorkingYears, and PayGapPct is computed from income. In the fitted model the **income coefficient is +1.45** while TotalWorkingYears is -1.07 and PayGapPct is -0.50. Holding level, role and experience fixed, higher pay "raises" risk. So an individual explanation can say "MonthlyIncome lowers risk" for a low earner (#478: -1.36). This happens in 48 of the 80 out-of-fold missed leavers. The model is not wrong: the *sum* of these features is meaningful, but each one alone is not.

- **Tested fix: drop MonthlyIncome.** PR-AUC falls by 0.019 (13 of 15 folds, p = 0.045). That fails the Phase 1 keep/drop rule, and JobLevel's sign flips instead. Rejected.
- **Recommended fix (team decision, not yet applied):** show MonthlyIncome + PayGapPct + JobLevel + TotalWorkingYears as one "Pay & seniority" driver in the Employee View and the waterfalls. SHAP values are additive, so the sum is exact, and the model does not change.

## 5. Fairness (p2-5)

Sensitive attributes (Gender, Age, MaritalStatus) are not model inputs. These checks look at outputs. **Out-of-fold train** is the main read (n = 1,176, 190 leavers). The test set (47 leavers, only 8 aged 50+) is too small on its own. Recall intervals are Wilson 95%.

| Group | n | Leavers | Actual rate | Mean predicted | Flagged | Recall (95% CI) |
|---|---|---|---|---|---|---|
| Female | 472 | 71 | 15.0% | 16.7% | 16.5% | 61% (49-71) |
| Male | 704 | 119 | 16.9% | 16.0% | 15.5% | 56% (47-65) |
| <30 | 258 | 74 | 28.7% | 23.0% | 26.7% | 69% (58-78) |
| 30-39 | 497 | 75 | 15.1% | 15.5% | 14.3% | 56% (45-67) |
| 40-49 | 282 | 26 | 9.2% | 13.7% | 12.1% | 42% (26-61) |
| 50+ | 139 | 15 | 10.8% | 11.7% | 9.4% | 40% (20-64) |

Test set, 50+: 34 people, 8 leavers, 1 caught (recall 12.5%, CI 2-47%). The test 50+ group happens to have twice the train attrition rate (23.5% vs 10.8%), so part of that gap is sampling.

**Findings**
- **Gender:** no material gap. Recall CIs overlap almost entirely, and mean predicted risk tracks the actual rate for both.
- **Age:** recall falls steadily with age (69% → 56% → 42% → 40%). For the 50+ group the model's *average* risk is right (11.7% vs 10.8%), but it doesn't single out *which* older employees will leave. The ranking within the group is weak, not the level.
- **Why (SHAP, 50+ leavers vs younger leavers, train):** TotalWorkingYears pushes 50+ leavers down (-0.99 vs +0.51 for younger leavers), and so do their more senior JobRoles (-0.19 vs +0.76). Long careers and senior roles act as a proxy that says "stays". MonthlyIncome pushes the other way (+0.95), the same collinearity as in section 4.
- **Would including Age fix it?** No. With Gender, Age and MaritalStatus added (same model and settings), out-of-fold PR-AUC moves 0.629 → 0.636, **50+ recall stays at 40%**, and **female recall drops from 61% to 51%**. Including sensitive attributes buys almost no accuracy, doesn't help the group we were worried about, and hurts another. This supports the Phase 1 decision to exclude them.

**What to say in the Q&A:** the model under-serves experienced employees because seniority signals dominate. The gap is a known limitation, shown on screen. The mitigation is process, not a sensitive input: HR reviews 50+ and director-level employees with the same drivers, not the score alone.

## 6. Error analysis (p2-6)

Files: `reports/missed_leavers_test.csv` (28 rows) and `reports/missed_leavers_oof_train.csv` (80 rows). One row per missed leaver, with their profile and the three SHAP drivers that pushed their risk down and up. Out-of-fold train is used for the patterns (80 cases vs 28).

**Profile of the four outcome groups (out-of-fold train):**

| | Caught (TP) | Missed (FN) | False alarm (FP) | Correctly not flagged (TN) |
|---|---|---|---|---|
| Count | 110 | 80 | 77 | 909 |
| Overtime | 64% | 34% | 51% | 22% |
| Tenure ≤ 2 years | 55% | 32% | 34% | 17% |
| Avg. years at company | 3.7 | 6.5 | 6.2 | 7.6 |
| Avg. monthly income | 3,988 | 6,020 | 5,304 | 7,004 |
| Any satisfaction score = 1 | 74% | 69% | 65% | 48% |
| 5+ years since promotion | 9% | 24% | 23% | 18% |

**Patterns**
1. **Missed leavers look like stayers on the strongest signals.** Fewer work overtime, they have been there longer and they earn more. On every feature, the gap between missed leavers and true negatives is under 0.3 standard deviations. Most misses are not model bugs: on the data we have, these people look like people who stay.
2. **Seniority blind spot.** The model catches 1 of 19 out-of-fold leavers in senior roles (Manufacturing Director 1/7, Healthcare Representative 0/6, Manager 0/5, Research Director 0/1). The JobRole coefficients for these roles are large and negative (Research Director -3.26, Manager -1.93). This is the same root cause as the 50+ fairness gap. Test examples: #1572 (53, 33 years at company, P = 0.4%), #165 (58, 40 years, 1.2%), #58 (Research Director, 22 years, 10.8%).
3. **Many misses are near-misses.** 23 of 80 out-of-fold misses (9 of 28 on test) are within 0.10 of the threshold, so they are threshold trade-offs rather than blind spots. 27 of 80 score below 10%, and those are the real blind spots.
4. **Dissatisfaction is present but diluted.** 69% of missed leavers gave at least one satisfaction score of 1, against 48% of stayers. The model treats the 1-4 scales as evenly spaced and averages them out. Example #1210 rated job, environment and work-life balance all 1, yet scored 0.31 because no travel and no overtime pulled it down.
5. **Promotion stagnation.** 24% of misses have gone 5+ years without promotion, against 9% of caught leavers. Long tenure drags their risk down more than stagnation pushes it up.
6. **What the data cannot see.** The dataset records no exit reason, so retirement cannot be told apart from resignation for long-career leavers. Life events (relocation, family) are also absent. Some misses are irreducible with this data.

**Candidate improvements (each must pass the Phase 1 ablation rule before going in):**
- An "any satisfaction score = 1" flag, or one-hot satisfaction scales (tests pattern 4)
- A tenure x promotion-gap interaction (tests pattern 5)

These are optional for the deadline. Patterns 2 and 6 are the ones to present.

## 7. Open decisions for the team

1. **Group the pay and seniority SHAP drivers for display?** (section 4) Recommended.
2. **Present the threshold as a range (0.27-0.35) in the app?** The cost curve is flat there.
3. **Business assumptions:** intervention cost and success rate are still unsourced. Present them as the sensitivity table above, or source them.

## Reproduce
```
python train.py          # ~5 min on 2 cores
pytest -q                # 13 tests
```
New outputs: `reports/roi_sensitivity.csv`, `reports/missed_leavers_test.csv`, `reports/missed_leavers_oof_train.csv`, and figures `model_comparison_cv`, `shap_global_bar`, `shap_waterfall_{caught_leaver,missed_leaver,false_alarm}`.
