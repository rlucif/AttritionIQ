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

## What we wrote or decided ourselves

| Sl No. |    Date    | Name |  Tool  |                            Decision/Task                             |
| :----: | :--------: | :--: | :----: | :------------------------------------------------------------------: |
|   01   | 2026-10-07 | Raj  | GitHub | Created a Repository, different branches for different group members |

- _persona names, threshold assumptions, handbook text, model choice_
