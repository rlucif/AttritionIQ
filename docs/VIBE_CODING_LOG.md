# Vibe Coding Log (1 page)

Keep entries short. Graders want to see how you directed the AI tools and what you checked yourselves, not a transcript.

## Platforms used

| Tool                                             | Used for                                         |
| ------------------------------------------------ | ------------------------------------------------ |
| Claude (claude.ai)                               | Idea stress-test, full repository skeleton, docs |
| _add: Copilot / Codex / Antigravity / AI Studio_ |                                                  |

## Prompt strategies that worked

- Give the AI the rubric first, then ask for the design, then the code. Design before code avoided rework.
- Ask for assumptions to be labelled with sources ("don't invent data") so every number is defensible.
- _add yours_

## Log

| Date       | Who | Tool   | Task / prompt summary                                                  | Accepted / changed / rejected                                                                                                                      | How we verified                                                                                              |
| ---------- | --- | ------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 2026-10-07 |     | Claude | Brainstormed and stress-tested the AttritionIQ idea against the rubric | Accepted cost-based threshold as core; added all 5 tracks                                                                                          | Team review                                                                                                  |
| 2026-10-07 |     | Claude | Generated the repo skeleton: pipeline, 5 modules, app, tests, docs     | Accepted as skeleton; TODO(team) items left for us                                                                                                 | Ran `train.py --fast`, `pytest`, app headless test                                                           |
| 2026-10-07 | Raj | Claude | Phase 1: quality checks, EDA notebook, feature ablation script         | Changed: asked for a paired, two-model ablation with a decision rule fixed before seeing results; all 4 engineered features dropped from the model | Reproduced our own ablation numbers exactly (LR 0.6225); `pytest` 8 passed; `train.py --fast` ran end to end |
| 2026-10-07 | Raj | Claude | Phase 2: model comparison with paired tests, selection rule, nested CV, calibration (ECE), threshold sanity check, bootstrap CIs, ROI sensitivity grid, SHAP waterfalls, fairness deep dive, missed-leaver export | Changed: selection rule and threshold rule written into code before the full run; fairness also run on out-of-fold train because the test 50+ group has 6 leavers. Rejected (for now): dropping MonthlyIncome to fix confusing SHAP signs, because CV showed it costs PR-AUC (p = 0.045) | Cross-checked group counts against the raw data; checked the 0.32 threshold against the break-even formula; `pytest` 13 passed; every number in `docs/PHASE2_RISK_MODEL.md` checked against `metrics.json` by script |

## What the AI got wrong and we fixed

| Assumption                          | Status                   | Action                                                            |
| ----------------------------------- | ------------------------ | ----------------------------------------------------------------- |
| `REPLACEMENT_COST_MULTIPLIER = 1.0` | Sourced (Gallup, 0.5-2x) | Keep                                                              |
| `INTERVENTION_COST_MONTHS = 1.0`    | Team Assumption          | Assigned one person to find a source, or present it as a scenario |
| `INTERVENTION_SUCCESS_RATE = 0.40`  | Team Assumption          | Same                                                              |
| `EXCLUDE_SENSITIVE = True`          | Decision                 | Currently true                                                    |
| `CURRENCY_SYMBOL = "$"`             | Decision                 | Dataset units                                                     |

- _e.g. a metric that looked fine but measured nothing; a library API that had changed_
- The AI's correlation snippet printed hundreds of `NaN` rows: in pandas 3, `.stack()` no longer drops missing values. Fixed with `.stack().dropna()` after reading the output, not trusting the code.
- The skeleton's fairness report labelled age bands wrongly: `pd.cut` uses right-closed bins by default, so "50+" held ages 51+ and 50-year-olds sat in "40-49". Found when the 50+ leaver count did not match a direct `Age >= 50` count. Fixed with `right=False` and a test.
- SHAP said "low MonthlyIncome lowers risk" for low earners. Not a bug: income, JobLevel (r = 0.95) and PayGapPct are collinear, so the linear model's income coefficient is positive *after* holding level and role fixed. Found by reading individual explanations, not the global chart.
- The first SHAP waterfall showed log-odds of the uncalibrated model under a calibrated-probability title. Added a footnote to every waterfall.

## What we wrote or decided ourselves

| Sl No. |    Date    | Name |  Tool  |                            Decision/Task                             |
| :----: | :--------: | :--: | :----: | :------------------------------------------------------------------: |
|   01   | 2026-10-07 | Raj  | GitHub | Created a Repository, different branches for different group members |

- _persona names, threshold assumptions, handbook text, model choice_
