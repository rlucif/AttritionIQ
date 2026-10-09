# Phase 5 - Freeze and validate: plan

Written 2026-10-09, before running any Phase 5 check. As in Phase 3, the pass/fail rules are fixed here first so the results cannot bend them. Phase 5 adds no features and changes no modelling choice: it proves the build that exists is reproducible, tested and documented truthfully, then freezes it for deployment (Phase 6) and the defence (Phase 7).

Starting point: branch `eda/phase-1` at `afa1200` (Phase 4), working tree clean and pushed to `origin`. `main` is still at Phase 0, so the PR `eda/phase-1 -> main` is part of the freeze.

## Checks and decision rules

| # | Check | How | Pass rule |
|---|---|---|---|
| C1 | Fresh-machine install | New container with no project state; `git clone` from GitHub; follow the README Quick start **as written**, Python 3.13 (the version the README says it was tested on) | Every Quick start command runs unchanged. Any extra step needed = README bug, fixed in the README, then C1 re-run from scratch |
| C2 | Final full training run | `python train.py` (full, not `--fast`) on the fresh clone | Exits 0; writes the bundle, `reports/metrics.json` and every figure |
| C3 | Reproducibility | Compare the fresh `metrics.json` with the committed one, key by key | Headline numbers identical at stored precision: chosen model, CV PR-AUC mean/std, threshold, test PR-AUC/recall/precision, savings and its 95% CI, k and ARI, eps, ARIMA vs naive MAE, Ridge R2, retrieval scores. Any difference is explained (library or platform) before freezing, never overwritten silently. `run_time` and timings are exempt |
| C4 | Tests | `pytest -q` on the fresh clone, after C2 | 0 failures **and 0 skips** (a skip means a test did not run, e.g. data missing) |
| C5 | App boots | Streamlit's `AppTest` (headless) runs `app.py` and each of the four pages, plus `app_classic.py` | No exception on any page with the default assumptions. If no such test exists, add one to `tests/` so Phase 6 can rerun it on the host |
| C6 | Architecture matches code | Every box and arrow in `docs/ARCHITECTURE.md` traced to a function that exists and to whether the app actually uses it | No box for code that does not exist; compared-but-rejected options (RF/XGB/MLP, ARIMA, Prophet, Euclidean) shown as rejected, not as the live path; the front-end layer (`views/`, `ui/`) present |
| C7 | README truthfulness | README claims checked against the repo (what is committed, handbook status, file layout) | No stale statement left |
| C8 | Slide figures committed | Figures in `reports/figures/` are git-ignored today | The figures used in slides are un-ignored and committed; the rest stay generated |
| C9 | Second machine | Raj runs `pytest -q` and `streamlit run app.py` on the Windows laptop | Same as C4/C5 |

## Freeze

Once C1-C9 pass: one commit "Phase 5: freeze", merge the PR into `main`, tag `v1.0-freeze`. After the tag, any change to `src/`, `train.py` or `requirements.txt` re-opens C2-C5 (rerun and compare). Docs and slides can change freely.

## Out of scope

New features, new experiments, retuning. Open decisions 5-7 (money units, grouped SHAP driver, threshold as a range) are presentation choices: they can be made in the app later without touching the frozen model, so they do not block the freeze. Open decision 9 (embeddings on the host) belongs to Phase 6.
