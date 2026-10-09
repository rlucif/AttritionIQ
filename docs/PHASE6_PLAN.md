# Phase 6 - Deploy: plan

Written 2026-10-09, before any deployment. Pass rules are fixed first, as in Phases 3 and 5.

## Host decision

**Streamlit Community Cloud (free tier).** Vercel was tried first and dropped: the app needed the Large Functions and WebSockets betas, an ASGI entry point and a 300 s connection limit, which is too much setup for a demo.

| Requirement | Community Cloud | Source |
|---|---|---|
| Free | Yes | Streamlit docs |
| Fits the app | Memory limit about 2.7 GB; the app peaks at about 520 MB (measured locally with all 4 pages open) | "Manage your app", limits as of Feb 2024 |
| Python 3.13 | Chosen in Advanced settings. Fallback tested: all pinned packages plus `google-genai` install and import on 3.14.6 | Streamlit docs; local test |
| LLM key | Secrets box; top-level keys become environment variables, which `src/rag.py` already reads | Streamlit secrets docs |
| Course brief | Brief names Streamlit as an allowed framework and sets no hosting rule | `Applied_AI_Data_Science_Group_Project.pdf` |
| Custom domain | **Not supported.** Only `<subdomain>.streamlit.app`. Namecheap's free URL redirect has no HTTPS, so `attritioniq.raj-sarania.me` is dropped; the official URL is `https://attritioniq.streamlit.app` (if the subdomain is free) | Streamlit forum; redirect.pizza on Namecheap |

## Decisions

- **Open decision 9: TF-IDF on the host.** No PyTorch. Matches the frozen `metrics.json` (decision 10); MiniLM stays cited from `reports/phase3_rag.json`.
- **Model on the host:** commit `models/attritioniq_bundle.joblib` (0.9 MB) from a full `python train.py` on a clean checkout (un-ignored in `.gitignore`). The host never trains or downloads the dataset.
- **LLM client:** `google-genai==2.29.0` moved from optional to required in `requirements.txt`, so the live brief can call Gemini (decision 3). No pinned version changed (checked with `uv pip check`).
- **No change to `src/`, `app.py` or `views/`.** Community Cloud has a writable disk, so the audit log works as it does locally (it resets on restart; stated in the README).

`requirements.txt` changed, so Phase 5 checks C2-C5 re-open (P1-P3 below).

## Checks and pass rules

| # | Check | Pass rule |
|---|---|---|
| P1 | Full `python train.py` on a clean checkout of this branch | Every C3 headline number in `metrics.json` identical to the frozen file; `run_time` and timings exempt |
| P2 | Fresh clone of the branch, new Python 3.13 venv, `pip install -r requirements.txt`, `pytest -q` | 0 failures, 0 skips |
| P3 | Host simulation: fresh clone **without** `data/raw/` and without running `train.py`, `streamlit run app.py`, headless browser | All 4 pages render with no exception; CSV upload scores rows; download works |
| P4 | Live: deploy on Community Cloud (Raj) | Build succeeds; P3 passes on the live URL on a laptop and a phone |
| P5 | Live brief | Employee page brief comes from Gemini, not the template |
| P6 | Before the demo | Open the site 10 minutes early to wake it; local `streamlit run app.py` ready as backup |

"Run `pytest` on the host" from the original plan is replaced by P2 + P3: Community Cloud gives no shell.

## Results before deployment (2026-10-09, Linux, Python 3.13.16)

| # | Result |
|---|---|
| P1 | **Pass.** Full run: Logistic Regression, threshold 0.32, test PR-AUC 0.562. 1,111 metric keys compared with the frozen file: one difference, `model_comparison.Random Forest.repeated_cv.brier.std` at the 17th decimal (float rounding, rejected model). The frozen `metrics.json` and figures are kept; only the bundle from this run is committed |
| P2 | **Pass.** Fresh copy of the branch, new venv from `requirements.txt` only, README download step: `pytest -q` 25 passed, 0 skipped (3 library deprecation warnings) |
| P3 | **Pass.** Same fresh copy with no dataset and no `train.py`, `streamlit run app.py` from a cold start: all 4 pages render with no exception (first page 5.5 s cold, 2.5-3.6 s after); 20-row CSV scored (4 flagged at 0.32); download returns 20 rows; audit log written |
| - | `google-genai` 2.29.0 added; all 14 pinned versions unchanged; `uv pip check` reports no conflicts. All pins plus `google-genai` also install and import on Python 3.14.6 |

## Live results (2026-10-09, https://attritioniq-ds-project.streamlit.app)

Deployed from `main` at `51bcdf4` (PR #4). Build log: Python 3.13.16, dependencies from `requirements.txt` via uv. Community Cloud replaced `pyarrow` 25.0.1 with 24.0.0 itself ("known segfault, apache/arrow#50471"); `pyarrow` is a Streamlit dependency, not one of our pins.

| # | Result |
|---|---|
| P4 | **Pass** (checked in a narrow, phone-width browser pane). All 4 pages render, no exception. Overview matches the frozen model: threshold 0.32, 294 employees, 36 flagged, $181,132 saved (Phase 2: $181k). 20-row CSV scored, 4 flagged (same as P3); download button shown. HR copilot retrieval returns handbook clauses. Still to do: Raj on a real phone |
| P5 | **Fail, fixed.** "Generate retention brief" spun on "Drafting the brief" for 4+ minutes with nothing in the logs. Cause: neither LLM SDK sets a timeout by default (`google-genai` 2.29: no timeout, no retries), so a request that gets no reply waits forever; a failed request would have shown the template within seconds. Reproduced locally against a server that accepts and never replies: the old code still waiting after 20 s. Fix (`src/rag.py`): `LLM_TIMEOUT_S` (default 30 s, env override) on the Gemini and Claude clients, and failures are printed to the host log. New test `tests/test_llm_timeout.py`: falls back to the template in about 3 s. Full suite on the fresh copy: 26 passed, 0 skipped. `train.py` never calls the LLM, so `metrics.json` is unaffected. Re-check P5 live after the fix is merged; if the log then shows a timeout or an error, the cause is the model name, key or quota |

### P5 re-check after the fix (2026-10-09, live, `main` at `0f79700`, PR #5)

Two clicks on "Generate retention brief" (employee #478):

| Try | Result | Time to answer |
|---|---|---|
| 1 | Gemini returned `504 DEADLINE_EXCEEDED` (no answer inside the 30 s limit); template brief shown | about 30 s |
| 2 | Gemini returned `503 UNAVAILABLE`, "This model is currently experiencing high demand"; template brief shown | 36 s |

- **The fix works:** the page no longer hangs; it falls back to the grounded template with the error shown.
- **P5 not yet passed, cause outside the code:** both errors come from Google's servers after the request was accepted, so the key, `LLM_PROVIDER` and `LLM_MODEL` in Secrets are correct. The model is overloaded.
- **Next (Raj, no code change):** retry later. If it keeps failing, change `LLM_MODEL` in the Secrets box to a less busy Gemini model listed in Google AI Studio, and/or add `LLM_TIMEOUT_S = "60"` (the env override already exists). Changing Secrets does not touch the repo, so it does not re-open Phase 5.
- **Demo rule:** the brief is shown as "optional live" in the Phase 7 demo script; the template fallback is the planned path if Gemini is busy.

## Sources

- Streamlit: [Manage your app (limits, sleep)](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [Secrets management](https://docs.streamlit.io/develop/concepts/connections/secrets-management), [Upgrade Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)
- Custom domains: [Streamlit forum, Custom Domain + NameCheap](https://discuss.streamlit.io/t/custom-domain-namecheap/121412), [Custom domain menu missing](https://discuss.streamlit.io/t/custom-domain-menu-missing-in-app-settings-community-cloud/120316), [Namecheap forwarding and HTTPS](https://redirect.pizza/support/namecheap-forwarding-with-https-support)
