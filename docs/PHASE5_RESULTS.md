# Phase 5 - Freeze and validate: results

Run 2026-10-09 against `eda/phase-1` at `afa1200`, with the rules fixed beforehand in `docs/PHASE5_PLAN.md`. Phase 5 changed no modelling code: only docs, `.gitignore`, one new test file, and the reference `metrics.json` and figures.

## Summary

| # | Check | Result |
|---|---|---|
| C1 | Fresh-machine install from the README alone | **Pass.** New container, `git clone` from GitHub, Python 3.13, Quick start commands run unchanged. Pinned requirements install cleanly; `python -m src.data --download` fetches the 1,470-row CSV |
| C2 | Final full `python train.py` | **Pass.** Exit 0 in 4 min (2 CPU cores). Writes the bundle, `metrics.json`, 13 figures and the missed-leaver / sensitivity CSVs |
| C3 | Reproducibility vs the committed `metrics.json` | **Pass, with three explained differences** (below). 1,055 values compared |
| C4 | `pytest -q` | **Pass.** 25 passed, 0 skipped (20 existing + 5 new app tests) |
| C5 | App boots headless | **Pass.** Overview, Employee deep-dive, Workforce, Model & trust and `app_classic.py` render with no exception or error. Added as `tests/test_app.py`; a deliberately broken page makes it fail, so the test does catch errors |
| C6 | `docs/ARCHITECTURE.md` matches the code | **Fixed.** See drift list |
| C7 | README truthfulness | **Fixed.** See drift list |
| C8 | Slide figures committed | **Fixed.** `reports/figures/*.png` no longer git-ignored (0.9 MB) |
| C9 | Second machine (Windows laptop) | **Pass** (Raj, 2026-10-09). Frozen as merge `b1c2a19` on `main`, tag `v1.0-freeze`; `pytest -q` re-run on a fresh checkout of the tag: 25 passed |

## C3: what reproduced and what did not

Identical, to the last stored digit or within 1e-15 floating-point noise (24 values): model choice (Logistic Regression), LR CV PR-AUC 0.625 ± 0.054, nested CV, calibration, threshold 0.32, every test metric, savings $181,132 and its bootstrap CI, ROI sensitivity grid, fairness tables, SHAP top 10 and examples, missed leavers, segmentation (k = 3, ARI, eps, DBSCAN, personas), Ridge vs Lasso, recommender CF, macro ratio 0.974 and the naive-forecast decision. The regenerated CSVs (`missed_leavers_*.csv`, `roi_sensitivity.csv`) are byte-identical.

| Difference | Values | Cause | Effect on a decision |
|---|---|---|---|
| Retrieval scores (hit@1 0.88 -> 0.79) | 9 | The old file came from a laptop with `sentence-transformers` installed (MiniLM). The README install is TF-IDF only. `embedding_backend` records which ran | None. MiniLM's adoption rests on `reports/phase3_rag.json`, which has both backends |
| ARIMA order (3,1,3) -> (2,1,3), ARIMA CV MAE 0.190 -> 0.197 | 15 | The optimiser converges differently by platform. On this run (3,1,3) scores AIC -562.2 vs -569.6 for (2,1,3) | None. Naive (MAE 0.179) wins either way; the app uses naive |
| XGBoost tuned parameters and CV scores (0.599 -> 0.606) | 37 | Multi-threaded XGBoost is not bit-identical across CPUs | None. LR has the highest mean in both runs; XGBoost stays tied (p 0.34 -> 0.39) |

**Decision (Raj, 2026-10-09):** the frozen `metrics.json` and figures are the README-only run, because a grader can reproduce it. It also matches the numbers already quoted in `PHASE2_RISK_MODEL.md`, `PHASE3_RESULTS.md`, `QA_PREP.md` and `README_Management.md` (XGBoost 0.606, p = 0.39; ARIMA(2,1,3), MAE 0.197). The old committed file did not match on those points. The README now has a "Reference run" note explaining the three differences.

## Drift fixed

`docs/ARCHITECTURE.md`
- Showed LR / RF / XGB / MLP as one live box. Now LR is the live model and the other three are a dashed "compared, lost" box.
- Showed "ARIMA / Prophet" as the forecast. Now the naive forecast is live; ARIMA (lost in CV) and Prophet (optional, not installed) are dashed.
- Showed SHAP clusters feeding the segments. Now marked as the compared alternative (feature clusters won).
- Added the front-end layer (`app.py` -> `views/` -> `ui/`), a page map that ties each page to its function calls, MiniLM/TF-IDF and Gemini/template fallbacks, and a data-flow note on which optional package changes what.
- "SMOTE in the pipeline" corrected to "SMOTE only for the MLP candidate".
- New design-decision rows: LR choice, k = 3, naive forecast, MiniLM + cosine.

`README.md`, `README_Management.md`, `README_Simple.md`
- "HR handbook is a template; values in [brackets] are placeholders": false since Phase 3 (a test checks there are none). Now says it is sourced (`SOURCES.md`).
- "JOLTS CSV (not committed)": it is committed. Fixed.
- Track 4 listed "MLP": there is none in `forecast.py`. Now "ARIMA vs naive (naive used)".
- Layout now lists `experiments/` and `tests/test_app.py`.

`.gitignore`: figures are committed from the freeze run. They are used by the Model & trust page and the slides, and the deployed site may not run `train.py`.

## Notes for Phase 6

- `pytest -q` (including `tests/test_app.py`) is the check to run on the host after deploying.
- Your laptop's bundle (trained 7 Oct) is still valid: the only code change since then that touches anything is in `rag.py`, and the retriever is built live, not stored in the bundle.
- Your laptop runs Python 3.14 and the README says "tested with 3.13, needs 3.11+". Pick the host's Python version to match one of the two.

## C9 and freeze steps (Raj, on the laptop)

```powershell
.venv\Scripts\activate
pytest -q                       # expect 25 passed, 0 skipped
streamlit run app.py            # click through all four pages + one CSV upload
git add -A
git commit -m "Phase 5: freeze - reference metrics and figures, app tests, architecture and README fixes"
git push
# open the PR eda/phase-1 -> main on GitHub, merge it, then:
git checkout main; git pull
git tag v1.0-freeze; git push origin v1.0-freeze
```

Do not re-run `train.py` on the laptop before committing: it would replace the reference `metrics.json` with the MiniLM / (3,1,3) variant. After the tag, any change to `src/`, `train.py` or `requirements.txt` re-opens C2-C5.
