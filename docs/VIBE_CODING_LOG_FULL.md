# Vibe Coding Log - full record

The one-page version for submission is `docs/VIBE_CODING_LOG.md`. This file keeps every entry and every AI mistake we caught, as evidence for the Q&A.

## Platforms used

| Tool                                             | Used for                                         |
| ------------------------------------------------ | ------------------------------------------------ |
| Claude (claude.ai Project + agent sessions on the laptop) | Idea stress-test, repository skeleton, phase plans, code, tests, docs, live-site checks |
| GitHub | Repository, one branch per change, pull requests into `main` |
| Google AI Studio | Gemini API key for the HR copilot brief (part of the product) |

## Prompt strategies that worked

- Give the AI the rubric first, then ask for the design, then the code. Design before code avoided rework.
- Ask for assumptions to be labelled with sources ("don't invent data") so every number is defensible.
- Write the decision rule and pass/fail checks into a plan file before any run, then let the AI implement against them.
- Ask the AI to check every number in a write-up against `metrics.json` by script, not by reading.

## Log

| Date       | Who | Tool   | Task / prompt summary                                                  | Accepted / changed / rejected                                                                                                                      | How we verified                                                                                              |
| ---------- | --- | ------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 2026-10-07 | Raj | Claude | Brainstormed and stress-tested the AttritionIQ idea against the rubric | Accepted cost-based threshold as core; added all 5 tracks                                                                                          | Team review                                                                                                  |
| 2026-10-07 | Raj | Claude | Generated the repo skeleton: pipeline, 5 modules, app, tests, docs     | Accepted as skeleton; TODO(team) items left for us                                                                                                 | Ran `train.py --fast`, `pytest`, app headless test                                                           |
| 2026-10-07 | Raj | Claude | Phase 1: quality checks, EDA notebook, feature ablation script         | Changed: asked for a paired, two-model ablation with a decision rule fixed before seeing results; all 4 engineered features dropped from the model | Reproduced our own ablation numbers exactly (LR 0.6225); `pytest` 8 passed; `train.py --fast` ran end to end |
| 2026-10-07 | Raj | Claude | Phase 2: model comparison with paired tests, selection rule, nested CV, calibration (ECE), threshold sanity check, bootstrap CIs, ROI sensitivity grid, SHAP waterfalls, fairness deep dive, missed-leaver export | Changed: selection rule and threshold rule written into code before the full run; fairness also run on out-of-fold train because the test 50+ group has 6 leavers. Rejected (for now): dropping MonthlyIncome to fix confusing SHAP signs, because CV showed it costs PR-AUC (p = 0.045) | Cross-checked group counts against the raw data; checked the 0.32 threshold against the break-even formula; `pytest` 13 passed; every number in `docs/PHASE2_RISK_MODEL.md` checked against `metrics.json` by script |
| 2026-10-07 | Raj | Claude | Phase 3 plan, then Track 1 (k rule with bootstrap stability, DBSCAN knee eps, SHAP segments), Track 3 (Funk SVD tuning, density sweep vs oracle, hybrid alpha), Track 4 Ridge vs Lasso | Decision rules written into `docs/PHASE3_PLAN.md` before any run. Rejected: SHAP clusters as main segmentation (separate leavers worse); tuning alpha towards CF (curve flat, test favours CF by construction) | Recommender results checked against an oracle ceiling; every number in `docs/PHASE3_RESULTS.md` checked against `reports/phase3_*.json` by script; new regression tests |
| 2026-10-08 | Raj | Claude | Phase 3: JOLTS via FRED, ARIMA (d by ADF+KPSS) vs naive, sourced HR handbook, 33-question retrieval eval, train.py wiring | Changed: app uses the naive forecast because ARIMA lost in rolling CV. Every handbook value cited (EU law, UK Act, GitLab public handbook) or marked "company choice"; questions labelled before retrieval ran | JOLTS snapshot checked against FRED (309 months, values sum 620.4); legal figures read from the EUR-Lex texts; `pytest` 20 passed |
| 2026-10-08 | Raj | Claude | Front-end restructure: 7 flat tabs + sidebar became 4 task-based pages (Overview, Employee deep-dive, Workforce, Model & trust) with a shared assumptions panel; gsap.com-inspired theme and GSAP motion | Planned the page structure and a feature-parity checklist before coding; chose to restyle Streamlit over a separate JS front-end (keeps every feature in Python). GSAP and fonts self-hosted in `static/` so nothing depends on a CDN. Old layout kept as `app_classic.py` | Headless browser run of every page and interaction (assumption change, CSV upload, register row → employee, next employee, what-if, brief, all tabs): no exceptions; what-if 74% → 42% matched a direct model call; `pytest` 20 passed |
| 2026-10-09 | Raj | Claude | Phase 3 close-out: sentence embeddings vs TF-IDF on the laptop, live Gemini brief, persona names, faiss tie-break fix | Adopted MiniLM by the pre-set rule (MRR 0.930 vs 0.847). Team approved the persona names. Kept TF-IDF as automatic fallback for light hosting | Brief generated live with Gemini 3.8 Flash and its [section] citations checked against the retrieved clauses; full `train.py` re-run on Windows reproduced every Phase 2/3 headline number |
| 2026-10-09 | Raj | Claude | Phase 5 freeze: fresh-clone install from the README alone, full `train.py`, metrics diff, headless page tests, architecture and README audit | Pass/fail rules written into `docs/PHASE5_PLAN.md` first. Frozen `metrics.json` = README-only run (reproducible by graders, matches the numbers in the docs). Added `tests/test_app.py`. Fixed stale docs (template handbook, uncommitted JOLTS, ARIMA/Prophet and 4 models shown as live) | Script compared all 1,055 values in old vs new `metrics.json` and sorted the 86 differences by cause; planted an error in a page to confirm the new test fails; `pytest` 25 passed, 0 skipped |
| 2026-10-09 | Raj | Claude | Phase 6 deploy: host comparison (Vercel tried, dropped), committed model bundle, P1-P3 pre-deploy checks, live checks on Streamlit Community Cloud, LLM timeout fix | Rejected Vercel (needed beta features and an ASGI entry point) and the custom domain (no free HTTPS redirect). Changed: added `LLM_TIMEOUT_S` after the live brief hung for 4+ minutes. Accepted: when Gemini is overloaded (503/504) the app shows the grounded template brief | Full `train.py` matched the frozen `metrics.json` (1,111 keys); fresh install `pytest` 26 passed; live site matched the frozen numbers (0.32, 36 flagged, $181,132); hang reproduced locally against a server that never replies, then fixed with a test |

## What the AI got wrong and we fixed

| Assumption                          | Status                   | Action                                                            |
| ----------------------------------- | ------------------------ | ----------------------------------------------------------------- |
| `REPLACEMENT_COST_MULTIPLIER = 1.0` | Sourced (Gallup, 0.5-2x) | Keep                                                              |
| `INTERVENTION_COST_MONTHS = 1.0`    | Team Assumption          | Assigned one person to find a source, or present it as a scenario |
| `INTERVENTION_SUCCESS_RATE = 0.40`  | Team Assumption          | Same                                                              |
| `EXCLUDE_SENSITIVE = True`          | Decision                 | Currently true                                                    |
| `CURRENCY_SYMBOL = "$"`             | Decision                 | Dataset units                                                     |

- Phase 6: neither LLM SDK sets a request timeout by default, so the live brief waited forever when Gemini never replied. Found on the live site; added a 30 s timeout with fallback and a test against a server that never answers.
- The AI's correlation snippet printed hundreds of `NaN` rows: in pandas 3, `.stack()` no longer drops missing values. Fixed with `.stack().dropna()` after reading the output, not trusting the code.
- The skeleton's fairness report labelled age bands wrongly: `pd.cut` uses right-closed bins by default, so "50+" held ages 51+ and 50-year-olds sat in "40-49". Found when the 50+ leaver count did not match a direct `Age >= 50` count. Fixed with `right=False` and a test.
- SHAP said "low MonthlyIncome lowers risk" for low earners. Not a bug: income, JobLevel (r = 0.95) and PayGapPct are collinear, so the linear model's income coefficient is positive *after* holding level and role fixed. Found by reading individual explanations, not the global chart.
- The first SHAP waterfall showed log-odds of the uncalibrated model under a calibrated-probability title. Added a footnote to every waterfall.
- Collaborative filtering "beat" the noise-free oracle, which is impossible. The simulator and Funk SVD both called `default_rng(42)`, so the model's starting user factors were exactly the simulated noise (scaled). Fixed with a separate random stream and a regression test; CF's real gain is much smaller.
- The skeleton picked the ARIMA differencing order inside an AIC grid, but AIC from models fitted to differently differenced data is not comparable. Now d comes from ADF + KPSS first.
- The skeleton standardised SHAP values before clustering, giving tiny drivers the same weight as OverTime. SHAP values already share one unit, so they are now clustered raw.
- Cluster descriptions compared each cluster with the spread of the 3 cluster means, so a 2.77 vs 2.71 satisfaction gap read as "+1.4 sd". Now measured against the workforce spread.
- Retrieval showed cosine and Euclidean "differing" on normalised TF-IDF, which is mathematically impossible. Cause: one question shares no word with the handbook (all-zero vector), so an unstable sort decided its ranking. Now ties sort stably and empty queries count as misses (the app says "no policy found").
- The same cosine/Euclidean gap came back on the laptop (MRR 0.842 vs 0.844) because `faiss` was installed there and returns equal scores in its own order. First fix (rounding to 6 decimals) still left a 0.003 gap with faiss because float32 noise split ties; final fix re-scores faiss candidates exactly and treats scores within 1e-6 as tied (handbook order). Checked with and without faiss.
- Front-end: the first dot field and risk dial were SVG, but `st.html` sanitises with DOMPurify's HTML-only profile, which silently strips SVG. Found when the dots were missing in a screenshot; rebuilt both in plain HTML/CSS.
- Front-end: "Next highest risk" jumped back to the previous employee on the next click anywhere. Streamlit now keys widget identity on `key`, so the selectbox kept its old value. Fixed by setting the widget's state in a button callback.
- Phase 5: the committed `metrics.json` (laptop run) did not match the docs on XGBoost (0.599 vs 0.606) and the ARIMA order ((3,1,3) vs (2,1,3)). Causes: multi-threaded XGBoost and the ARIMA optimiser differ by platform, and the laptop has `sentence-transformers`. Fixed by freezing the README-only run and documenting the differences.
- Phase 5: the AI first explained the ARIMA difference as "AIC candidates within 0.3". Checking the AIC table showed (3,1,3) is 7 points behind on the reference run, so the cause is optimiser convergence. The README was corrected before commit.
- Phase 5: Streamlit's `AppTest.switch_page` only handles file-based pages, not the callable pages `st.navigation` registers here. The app test renders each view's `render(ctx)` directly instead.

## What we wrote or decided ourselves

| Sl No. |    Date    | Name |  Tool  |                            Decision/Task                             |
| :----: | :--------: | :--: | :----: | :------------------------------------------------------------------: |
|   01   | 2026-10-07 | Raj  | GitHub | Created a Repository, different branches for different group members |
| 02 | 2026-10-09 | Team | - | Approved persona names: Overtime crew, Steady core, Senior veterans |
| 03 | 2026-10-09 | Raj | - | Kept the default cost assumptions; excluded sensitive attributes; hosted on Streamlit Community Cloud free tier |
