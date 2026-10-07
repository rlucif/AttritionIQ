# Vibe Coding Log (1 page)

Keep entries short. Graders want to see how you directed the AI tools and what you checked yourselves, not a transcript.

## Platforms used
| Tool | Used for |
|---|---|
| Claude (claude.ai) | Idea stress-test, full repository skeleton, docs |
| _add: Copilot / Codex / Antigravity / AI Studio_ | |

## Prompt strategies that worked
- Give the AI the rubric first, then ask for the design, then the code. Design before code avoided rework.
- Ask for assumptions to be labelled with sources ("don't invent data") so every number is defensible.
- _add yours_

## Log

| Date | Who | Tool | Task / prompt summary | Accepted / changed / rejected | How we verified |
|---|---|---|---|---|---|
| 2026-10-07 | | Claude | Brainstormed and stress-tested the AttritionIQ idea against the rubric | Accepted cost-based threshold as core; added all 5 tracks | Team review |
| 2026-10-07 | | Claude | Generated the repo skeleton: pipeline, 5 modules, app, tests, docs | Accepted as skeleton; TODO(team) items left for us | Ran `train.py --fast`, `pytest`, app headless test |
| | | | | | |

## What the AI got wrong and we fixed
- _e.g. a metric that looked fine but measured nothing; a library API that had changed_

## What we wrote or decided ourselves
- _e.g. persona names, threshold assumptions, handbook text, model choice_
