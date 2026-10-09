# AttritionIQ: Decision Logic Brief

**Course:** Data Science for Managers (capstone) · **Product:** Employee attrition risk engine for HR leaders

---

## 1. Framing: a decision problem, not a prediction problem

The question HR actually faces is not *"who will leave?"* but:

> **Given a limited retention budget, whom do we approach, with what intervention, and is it worth the money?**

Every design choice in AttritionIQ flows from that framing. Prediction is only the first link in a chain that ends in a **costed, explainable, auditable action**.

### End-to-end logic chain

```
Raw HR data
  → Clean + engineer features (incl. pay gap vs. peers)            [Track 4: fair-pay model]
  → Risk model → calibrated P(leave within 12 months)              [Track 2: core]
  → Cost rule → flag / don't flag; allocate budget                 [Track 2: ROI]
  → SHAP → top risk drivers per employee                           [Track 2: explainability]
  → Drivers → ranked interventions + pay-rise sensitivity          [Track 3: recommender]
  → Segments → group-level policy                                  [Track 1: segmentation]
  → Σ P(leave) × macro trend → expected leavers and cost per dept  [Track 4: forecast]
  → Drivers + question → grounded policy brief                     [Track 5: HR Copilot]
  → Every prediction → audit log                                   [Governance]
```

---

## 2. Data logic

| Step | Logic | Why |
|---|---|---|
| **Source** | IBM HR Analytics dataset: 1,470 employees, 35 columns, 16.1% attrition | Standard, well-documented benchmark (synthetic) |
| **Cleaning** | Drop 3 constant columns and the employee ID; check duplicates, missing values and impossible tenure combinations (none found) | Constants carry no information; an ID would let the model memorise rows |
| **Feature engineering** | `PromotionLagRatio` = years since promotion ÷ tenure; `ManagerTenureRatio`; `YearsPerCompany` (job-hopping); `SatisfactionIndex` (average of 4 satisfaction scales); `PayGapPct` (actual ÷ fair pay − 1) | Ratios a manager can interpret; relative measures beat raw ones (2 years without promotion means different things at 3 vs. 15 years' tenure) |
| **Sensitive attributes** | Gender, Age, MaritalStatus **excluded** from the model | Ethical and legal risk outweighs a negligible accuracy gain (see §9) |
| **Label** | "Attrition = Yes" treated as *left within 12 months* | The dataset doesn't document the window; we state the assumption |
| **Split** | Stratified 80/20. The 20% (294 employees) acts as the "current workforce" in the app | All reported results come from people the model never trained on |

---

## 3. Risk model logic (Track 2 core)

### 3.1 Choosing the right yardstick

With 16% positives, **accuracy is misleading**: predicting "nobody leaves" scores 84%. We optimise **PR-AUC** (precision–recall area), which measures how well the model ranks the minority who actually leave. Random guessing gives PR-AUC ≈ 0.16.

### 3.2 Model selection rule (set before seeing results)

1. Tune four candidates: **Logistic Regression, Random Forest, XGBoost, Neural Network**. Each gets 25-iteration randomized search with stratified 5-fold CV.
2. Score each with **repeated CV** (5 folds × 3 repeats = 15 estimates), so we get a mean and a spread, not one lucky number.
3. **Highest mean PR-AUC wins, but any model within 1 standard deviation counts as tied.**
4. **Among tied models, the simplest wins** (LR < RF < XGBoost < MLP).

**Outcome:** Logistic Regression (0.625) and XGBoost (0.606) tied, so **Logistic Regression** was chosen. A corrected paired t-test confirms LR beats Random Forest (p = 0.006) and is statistically indistinguishable from XGBoost (p = 0.39).

**Why simplicity is the tiebreaker:** a linear model is transparent, stable and defensible to HR, employees and auditors. Complexity has to *earn* its place.

**Check against self-deception:** nested CV (tuning inside each outer fold) gives PR-AUC 0.654, no lower than plain CV. Our tuning did not flatter the model.

### 3.3 Calibration: making scores mean what they say

Raw model scores were inflated (mean predicted 34% vs. 16% actual). **Platt scaling** rescales them so that "0.30" really means about 30 in 100 leave. Calibration error (ECE) fell from **0.19 to 0.04**.

**Why it matters:** the cost rule in §4 multiplies probability by money. Uncalibrated probabilities would mis-price every decision.

### 3.4 Explainability: SHAP

SHAP splits each employee's score into the contribution of each feature ("overtime +0.7, pay −0.4 …").

- **Local:** every employee gets their top 5 drivers. These feed the recommender (§6) and the copilot (§8).
- **Global:** the top drivers are **JobRole, MonthlyIncome, TotalWorkingYears, OverTime, EnvironmentSatisfaction, PayGapPct**.
- **Caveat:** SHAP shows *model sensitivity*, not causation.

---

## 4. Decision and economic logic (Track 2 ROI)

### 4.1 The cost equations

For an employee with leave probability *p*, replacement cost *R*, intervention cost *C* and intervention success rate *s*:

| Policy | Expected cost |
|---|---|
| Don't flag | *p × R* |
| Flag and intervene | *C + p × R × (1 − s)* |

**Intervene when:** *p × R × s > C*, i.e. the expected saving exceeds the cost.

### 4.2 Break-even threshold

Both *C* and *R* scale with the same salary (*C* = months × monthly pay; *R* = 12 × multiplier × monthly pay), so **salary cancels out**:

> *p\** = intervention months ÷ (12 × replacement multiplier × success rate)

With defaults (1 month, 1× salary, 40%): *p\** = 1 ÷ 4.8 ≈ **0.21**. This is the theoretical sanity check.

### 4.3 Data-driven threshold

The app sweeps thresholds from 0.05 to 0.95 and picks the one that **minimises total realised cost** on out-of-fold *training* predictions. It then applies that threshold, unchanged, to the test group. Result: **0.32**.

**Why not 0.50?** 0.50 is a statistical convention with no business meaning. At 0.50 the model catches 13 leavers; at the cost-optimal 0.32 it catches 19 and saves more.

### 4.4 Three-policy comparison (test group, dataset salary units)

| Policy | Expected cost | vs. doing nothing |
|---|---|---|
| Do nothing | $2.57M | — |
| Intervene with everyone | $3.41M | **+32%** |
| Targeted (36 flagged, $146K spend) | **$2.39M** | **−7% ($181K saved; 95% CI $43K–$338K)** |

**Logic:** blanket programmes spend on the 84% who would have stayed anyway. The value comes from **targeting**.

### 4.5 Budget allocation (greedy knapsack)

With a fixed budget:

1. Keep only employees with positive net benefit (*p × R × s − C > 0*).
2. Rank them by **ROI ratio** (*p × R × s ÷ C*).
3. Fund from the top until the budget runs out.

### 4.6 Sensitivity analysis

The whole decision is re-run for **27 scenarios**: replacement cost 0.5/1/2× salary, success rate 20/40/60%, and intervention cost 0.5/1/2 months. In each scenario the threshold is re-optimised.

- Savings are **positive in 26 of 27** scenarios, ranging from about 0% to 43%.
- The single loss (−0.2%) needs cheap replacement, low success *and* expensive interventions at once.
- All three assumptions are **editable live** in the app, which re-optimises instantly.

---

## 5. Segmentation logic (Track 1)

| Step | Logic |
|---|---|
| **Inputs** | 12 behavioural features: pay, tenure, promotion gap, level, overtime, satisfaction scales, distance, salary hike, training |
| **Unsupervised by design** | The attrition label is **not** used to form clusters, only to describe them afterwards, so segments reflect who people are, not who left |
| **Choosing k** | Must pass a **stability bar** (bootstrap ARI ≥ 0.60); among those, best silhouette. k = 2 is excluded as too coarse to act on → **k = 3** (stability 0.98) |
| **Second opinion** | Hierarchical (Ward) clustering agrees with K-Means (ARI 0.81) |
| **DBSCAN** | Finds one dense cluster plus 32 outliers, so it is used as an **unusual-profile detector**, not a segmenter |
| **RFM analogue** | Recency = years since promotion, Frequency = trainings, Monetary = salary hike. Attrition is flat (15–17%) across RFM bands, so it is reported as a **finding, not a risk signal** |

**Result:**

| Segment | Size | Attrition |
|---|---|---|
| Overtime crew | 348 | **33.6%** |
| Steady core | 855 | 11.5% |
| Senior veterans | 267 | 8.2% |

**Insight:** satisfaction is nearly identical across segments. **Workload and seniority**, not sentiment, separate high-risk from low-risk groups. Silhouette is low (0.14), so the segments are regions of one continuous population, not sharply separate islands.

---

## 6. Recommendation logic (Track 3)

| Component | Logic | Data |
|---|---|---|
| **Intervention catalogue** | 8 actions (workload cap, promotion review, pay adjustment, stock grant, manager 1:1s, training, flexible work, role rotation), each mapped to the features it targets, with a cost in months of salary | Team assumption |
| **Content-based** | Score each action by the **sum of the employee's risk-raising SHAP values** on that action's target features. An overtime-driven employee gets "workload cap" first | **Real** model output |
| **Collaborative filtering** | Funk SVD matrix factorisation on an employee × intervention effectiveness matrix ("employees like you responded to …") | **Simulated**, because no intervention history exists |
| **Hybrid** | Final score = 0.6 × content + 0.4 × CF (both scaled 0–1) | Content weighted higher because it uses real data |
| **Pay-rise elasticity** | Raise salary 0–30% in 2% steps and find the **smallest raise that pushes P(leave) below the threshold**, the HR analogue of price elasticity | Model sensitivity, not a causal effect |

**Key finding:** CF only beats a simple baseline once about half of employee–intervention outcomes are logged. Until a firm builds that history, **content-based matching carries the recommendation**.

---

## 7. Predictive operations logic (Track 4)

### 7.1 Fair-pay model

- Predict log(income) from job level, experience, education, tenure, role and department.
- Ridge (L2) and Lasso (L1) tie at **R² ≈ 0.87**, so we keep **Ridge**. Lasso's "feature selection" only dropped two redundant dummy columns.
- **PayGapPct** = actual ÷ predicted pay − 1 feeds the risk model. It is refitted inside every CV fold to avoid leakage.

### 7.2 Workforce forecast

- **Expected leavers** = Σ individual probabilities (the expected value of a sum of yes/no outcomes), grouped by department, plus expected replacement cost.
- **Macro adjustment:** multiply by (forecast US quits rate for the next 12 months ÷ trailing 12-month average). This gives **0.974**, a 2.6% nudge down. *Assumption: the company's attrition moves with the national quits rate.*

### 7.3 Choosing the macro forecast model

- An ARIMA(2,1,3) was tuned by stationarity tests and AIC, then validated with **rolling-origin CV** (6 folds, 12-month horizon).
- It beat the naive "next month = this month" forecast in **only 1 of 6 folds**, so the app uses the **naive forecast** (US quits rate ≈ 1.9%/month).
- **Principle:** the complex model has to beat the simple baseline to earn its place.

---

## 8. Enterprise GenAI logic (Track 5: HR Copilot)

**Retrieval-augmented generation (RAG):**

```
HR handbook → split by section (13 chunks) → vectorise → index
Manager question + employee's top risk drivers → vectorise → retrieve top 3 sections
LLM writes the brief using ONLY the retrieved clauses + SHAP drivers
```

| Design choice | Logic |
|---|---|
| **Grounding** | The LLM sees only retrieved policy text, which cuts hallucinated policy |
| **Query enrichment** | The employee's risk drivers are added to the question in plain words, so retrieval finds policies relevant to *this* person |
| **Cosine vs. Euclidean** | Identical results (hit@1 79%, hit@3 88%, MRR 0.85) because vectors are length-normalised. On unit vectors the two metrics rank identically |
| **Graceful fallback** | TF-IDF if sentence embeddings are unavailable; template brief if there's no LLM key, so the demo never breaks |
| **Known weakness** | Keyword retrieval misses synonyms ("quit" vs. "leave"); semantic embeddings address this |

---

## 9. Governance logic

| Principle | Implementation |
|---|---|
| **Fairness by exclusion** | Gender, age and marital status are excluded. Including them raises PR-AUC by only **0.007**, a negligible gain for a material risk |
| **Fairness by monitoring** | Selection rate and recall are reported by gender and age band, with confidence intervals |
| **Known gap, stated openly** | Leavers aged **50+** are under-detected (1 of 8 caught in the test group). For them, the usual signals (low pay, short careers) point the other way |
| **Explainability** | Every score comes with its drivers; the simplest adequate model is used |
| **Accountability** | Every prediction is logged with timestamp, model version, probability, threshold, decision and top drivers |
| **Human in the loop** | A flag triggers a manager conversation, never an automated decision about an individual |

---

## 10. Evidence that the logic works (held-out test group: 294 employees, 47 leavers)

| Metric | Result | Benchmark |
|---|---|---|
| PR-AUC | 0.56 | Random = 0.16 |
| ROC-AUC | 0.79 | Random = 0.50 |
| Precision at the 0.32 threshold | 53% (19 of 36) | Base rate = 16% |
| Recall at the 0.32 threshold | 40% (19 of 47) | 28% at the default 0.50 |
| Calibration error (ECE) | 0.04 | 0.19 before calibration |
| Savings vs. doing nothing | 7% | Positive in 26 of 27 scenarios |

**Error pattern:** missed leavers are longer-tenured, better-paid and less on overtime than the leavers we catch. These "quiet leavers" don't fit the dominant pattern, which is a structural limit of learning from history.

---

## 11. Assumptions and limitations

1. **Synthetic data:** the IBM dataset is not real employee records; its currency and label window are undocumented.
2. **Team assumptions:** intervention cost (1 month of salary) and success rate (40%) are estimates, which is why they are adjustable and stress-tested.
3. **Simulated CF matrix:** no real intervention history exists, so CF demonstrates the pipeline, not the effectiveness of the interventions.
4. **Template handbook:** the HR policy text is a template; bracketed values are placeholders.
5. **Correlation, not causation:** SHAP, what-if and elasticity show model sensitivity, not proven cause and effect.
6. **Macro link:** we assume company attrition tracks the US quits rate; this only nudges the base rate.

---

## Sources

- **IBM HR Analytics Employee Attrition & Performance** (synthetic, 1,470 × 35). [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **US JOLTS quits rate** (BLS via FRED, series JTSQUR), monthly, Dec 2000 – Aug 2026. [FRED](https://fred.stlouisfed.org/series/JTSQUR)
- **Gallup (2019)**, "This Fixable Problem Costs U.S. Businesses $1 Trillion": replacement cost of 0.5× to 2× annual salary. [Article](https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx)

*All figures are taken from `reports/metrics.json` and `docs/PHASE3_RESULTS.md` (training run of 7 Oct 2026). Setup instructions are in `README.md`.*
