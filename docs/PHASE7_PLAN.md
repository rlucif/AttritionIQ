# Phase 7 - Technical defence: plan

Written 2026-10-09, before any slide, script or drill is made. Pass rules are fixed first, as in Phases 3, 5 and 6. The executive pitch is planned separately; this plan covers the technical deep dive, the live demo and the individual Q&A.

## What the brief asks for

From `Applied_AI_Data_Science_Group_Project.pdf`, section 2B and rubric rows 2-4:

| Brief item | Where it is covered in Phase 7 |
|---|---|
| 15 minutes + Q&A: 5 min pitch, 10 min technical deep dive | Deck section B below, timed in rule D1 |
| Deep dive: system architecture diagram, feature selection logic, model optimisation, cross-validation results, error analysis | Slides 1-8 |
| Live demo: live inference on realistic test data in the app | Demo script (section C), new `data/demo/demo_batch.csv` |
| Rubric 2 (40): preprocessing, scaling, tuning, CV, SHAP/LIME | Slides 2-7; each named on a slide |
| Rubric 3 (20): every member explains the technical parts and parameter choices | Q&A drills (section D) |
| Rubric 4 (10): AI agents orchestrated well; 1-page Vibe Coding Log; clean repo | Section E |

The brief lists the live demo under the 15 minutes but gives it no separate time. Until the instructor says otherwise, plan for the **demo inside the 10 minutes: about 7 min of slides + 3 min of demo** (open decision A).

## A. Freeze rule

Phase 7 changes only docs, slides, a demo data file and the recording. Any change to `src/`, `train.py`, `requirements.txt`, `app.py` or `views/` re-opens the Phase 5/6 checks (P2-P4). So open decisions 6 (one "Pay & seniority" SHAP driver) and 7 (threshold as a range) from the project plan are **explained on the slides, not built into the app**.

## B. Deep-dive deck (about 7 minutes, 9 slides)

Built from the **Slides** artifact type (downloads as .pptx or PDF), in the same look as the pitch deck. Every number comes from `reports/metrics.json` or a phase write-up; each slide names its source file in the speaker notes.

| # | Slide | Content (source) | Figure |
|---|---|---|---|
| 1 | Architecture | Live path vs "compared and lost"; one command trains, the app only loads (`docs/ARCHITECTURE.md`) | Redrawn architecture diagram |
| 2 | Data and preprocessing | 1,470 synthetic IBM rows, 16.1% leave, no missing values or duplicates; 80/20 stratified split (1,176 / 294); constant and ID columns dropped; scaling + one-hot inside the pipeline; PayGapAdder fitted inside each CV fold (no leakage) (`metrics.json: data`, `src/data.py`) | - |
| 3 | Feature selection logic | Ablation rule fixed before results: all 4 engineered features dropped. Sensitive attributes excluded: including them adds only +0.007 PR-AUC and lowers female recall from 61% to 51% (`reports/feature_ablation.json`, Phase 2 write-up) | - |
| 4 | Model optimisation | 4 models, RandomizedSearchCV, what each hyperparameter does to bias and variance; class weights for imbalance; nested CV 0.654 vs plain 0.625, so no tuning optimism (`metrics.json: nested_cv`) | - |
| 5 | Cross-validation results | Repeated stratified 5x3, PR-AUC (not accuracy: "nobody leaves" scores 84%). LR 0.625 ± 0.054; XGBoost tied (paired corrected t-test p = 0.39), so the pre-set rule picks the simpler model. Test PR-AUC 0.562 (95% CI 0.42-0.69) vs baseline 0.16 | `model_comparison_cv.png`, `pr_curves_test.png` |
| 6 | From score to decision | Platt calibration; cost-optimal threshold 0.32 (theory 0.21; cost curve flat 0.27-0.35); 36 flagged on the test set, $181k saved (95% CI $43k-$338k); positive in 26 of 27 cost scenarios | `threshold_cost_curve.png`, `calibration_test.png` |
| 7 | Explainability (SHAP) | Global drivers; three waterfalls (caught leaver, false alarm, missed leaver); SHAP is not causal; pay/seniority collinearity (income vs JobLevel r = 0.95) | `shap_global_bar.png`, waterfalls |
| 8 | Error analysis and fairness | Test confusion matrix (19 caught, 28 missed, 17 false alarms); who is missed (out-of-fold, train): senior, non-overtime staff (1 of 19 senior-role leavers caught); recall by age 69% under 30 vs 40% at 50+, while the average risk for 50+ is right (11.7% predicted vs 10.8% actual); no exit reason in the data (`docs/PHASE2_RISK_MODEL.md`) | - |
| 9 | Supporting tracks: what we kept and what lost | k = 3 (stability 0.98); CF closes 21-39% of the gap only at 50%+ density; ARIMA lost to naive (MAE 0.197 vs 0.179); retrieval TF-IDF on the host, MiniLM better offline (MRR 0.85 vs 0.93) | - |

## C. Live demo (about 3 minutes)

**Test data:** `data/demo/demo_batch.csv`, 20 rows taken from the 294-row test split (never seen in training), with the `Attrition` column removed. Made by a short script from the same split seed as `train.py`; rows chosen to include caught, missed and false-alarm cases so the demo can show an honest miss.

| Step | Action on the live app | What to say | If it fails |
|---|---|---|---|
| 1 | Overview: 294 employees, threshold 0.32, 36 flagged, $181k | Same numbers as slide 6 | Switch to the local run |
| 2 | Change success rate 40% to 20% in the assumptions panel | Threshold and flagged count move live; the decision depends on the business inputs, not just the model | - |
| 3 | Upload `demo_batch.csv` | Live inference on unseen rows; download the scored file | Local run |
| 4 | Employee deep-dive on one flagged row: drivers, then the what-if (overtime off) | Explanation, then a counterfactual; not causal | - |
| 5 | Retention plan and HR copilot clauses | Recommender + RAG retrieval with [section] citations | - |
| 6 | (Optional) Generate brief | Gemini if it answers; otherwise the template with the error shown is the designed fallback (Phase 6, P5) | Skip; say it falls back by design |

**Fallback order:** live URL (opened 10 minutes early to wake it) → local `streamlit run app.py` already running on the presenting laptop → recorded video of the same script (MP4, saved on the laptop and a USB stick, plays offline).

## D. Q&A drills

- Refresh `docs/QA_PREP.md` first: correct the Track 5 answer (the live host runs TF-IDF, MiniLM is the offline result; decision 9), and add Phase 5-6 questions (reproducibility check, deployment, the LLM timeout bug and fix, why no custom domain).
- Add a short **"explain the algorithm"** section for every model the brief names (logistic regression and its C, Platt scaling, XGBoost, SHAP/Shapley values, K-Means, Ward, DBSCAN eps/min_samples, Funk SVD, Ridge vs Lasso, ADF/KPSS and ARIMA orders, TF-IDF, cosine vs Euclidean, RAG). The brief's golden rule: no black-box answers.
- Drill format: questions drawn at random across all modules, 60-90 seconds per answer, scored on three points: right number, right mechanism, one honest caveat.

## E. Repo and Vibe Coding Log

- `docs/VIBE_CODING_LOG.md` is about 1,590 words with placeholder rows; the brief asks for **one page**. Keep the full log as `docs/VIBE_CODING_LOG_FULL.md` and cut the submitted file to one A4 page: platforms, 4-5 prompt strategies, the decision-rules-first workflow, 3 things the AI got wrong (seed collision, AIC across d, missing LLM timeout) and how we caught them. Add the Phase 6 entry first.
- No placeholder text (`_add yours_`, empty "Who" cells) anywhere in `docs/`.

## Checks and pass rules

| # | Check | Pass rule |
|---|---|---|
| D1 | Timed rehearsal | Slides + demo end within 10:00 on three runs in a row (or within the split the instructor confirms) |
| D2 | Brief coverage | Every brief item and rubric 2 item in the table above has a slide or demo step |
| D3 | Numbers | A script checks every number on the slides and in the speaker notes against `metrics.json` and the phase write-ups: 0 mismatches |
| D4 | Pitch consistency | Savings, threshold, flagged count and model name are the same in the pitch deck and the deep dive |
| D5 | Demo | The full script runs on the live URL and on the local backup; the recording is under 4 minutes and plays offline |
| D6 | Q&A | Each member, on 10 randomly drawn questions (at least 6 outside their own module), gets at least 8 right on number and mechanism |
| D7 | Vibe log | Fits one A4 page as PDF; no placeholders in `docs/` |
| D8 | Freeze | `git diff v1.0-freeze -- src train.py requirements.txt` shows only the Phase 6 changes already checked (bundle, `google-genai`, LLM timeout) |

## Order of work

1. Slide outline and speaker notes (this plan, section B) → Raj reviews → build the deck.
2. `demo_batch.csv` + demo script → rehearse on the live URL → record the backup.
3. QA_PREP refresh → drills with the team.
4. Vibe log to one page.
5. Run D1-D8; write results in `docs/PHASE7_RESULTS.md`.

## Open decisions (Raj)

- **A.** Is the live demo inside the 15 minutes? Default above: yes, 7 min slides + 3 min demo. Worth one line to the instructor.
- **B.** One combined deck (pitch + deep dive) or two? Does the pitch deck exist yet, and in what format?
- **C.** Who presents the deep dive and the demo, now that the build has been solo? The Q&A is individual, so every member needs the drills regardless.
- **D.** Presentation date, to turn the order of work into dates.
- **E.** Project-plan decision 5 (money in dataset units or converted to INR with a cited rate) must be settled before the slides, because it changes every dollar figure.

## Carried over from Phase 6

- P5 (live Gemini brief) waits on Google's capacity, not on code: see `docs/PHASE6_PLAN.md`, "P5 re-check after the fix".
- P4 on a real phone (Raj).
