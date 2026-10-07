# AttritionIQ - Project Plan (build track)

The pitch is planned separately. This plan covers the working prototype, the repository and the technical deep dive. Day numbers are relative: map them to your submission date (end-terms are in late October, so leave buffer).

## 1. Roles (8-9 people)

| Role | Owns | Files |
|---|---|---|
| Integrator / PM | GitHub repo, app.py, README, vibe coding log, merges | `app.py`, `README.md`, `docs/VIBE_CODING_LOG.md` |
| Data & EDA lead | Data quality, EDA notebook, engineered features | `src/data.py`, `notebooks/01_eda.ipynb` |
| Risk model lead (x2) | Tuning, CV, calibration, threshold, SHAP, fairness, error analysis | `src/model.py`, `src/roi.py`, `src/audit.py` |
| Segments lead | Track 1 | `src/segment.py` |
| Recommender lead | Track 3 | `src/recommend.py` |
| Forecast lead | Track 4 | `src/forecast.py` |
| Copilot lead | Track 5, writes the HR handbook | `src/rag.py`, `data/knowledge_base/` |
| QA & testing lead (9th member) | Tests, fresh-machine reproducibility, Q&A drills | `tests/`, `docs/QA_PREP.md` |

Everyone also has a **buddy module** they review and can explain. The Q&A is individual, so every member must be able to explain the core risk model, not only their own module.

## 2. Phases

### Phase 0 - Setup (Day 1-2)
- [ ] Create the GitHub repo, push this skeleton, protect `main` (changes via pull requests)
- [ ] Every member runs the Quick start in the README on their own laptop
- [ ] Start the vibe coding log; record the tool, prompt strategy and what was accepted or rejected
- [ ] Agree the assumptions in `src/config.py`: keep, change or find sources

### Phase 1 - Data understanding (Day 2-4)
- [ ] EDA notebook: distributions, attrition rate by feature, correlations, class imbalance
- [ ] Data-quality checks (TODO in `data.clean`)
- [ ] Test each engineered feature against CV PR-AUC; keep only those that help
- [ ] Decide on sensitive features: excluded (default) vs included, and run both to show the cost

### Phase 2 - Core risk model, Track 2 (Day 3-8)
- [ ] Full `python train.py`; compare four models on repeated CV (mean and std)
- [ ] Choose the model; if scores are within one std, justify picking the simpler one
- [ ] Calibration check, cost-optimal threshold, ROI on the held-out set
- [ ] SHAP global and individual explanations
- [ ] Fairness report: explain any group gaps (look at the 50+ age band)
- [ ] Error analysis: read individual missed leavers and write up patterns

### Phase 3 - Supporting tracks, in parallel (Day 4-10)
- [ ] **Track 1**: justify k (elbow, silhouette, dendrogram); choose DBSCAN eps from the k-distance knee; name the personas in `PERSONA_NAMES`; try clustering on SHAP values
- [ ] **Track 3**: finish the intervention catalogue and costs; sweep matrix density vs CF accuracy; tune the hybrid weight
- [ ] **Track 4**: download JOLTS; ARIMA order selection and time-series CV vs the naive baseline; optional Prophet comparison; Ridge vs Lasso write-up
- [ ] **Track 5**: write the real handbook (cite any public policy you base it on); extend `retrieval_eval.csv` to 30+ questions; compare TF-IDF vs sentence embeddings and cosine vs Euclidean; connect an LLM

### Phase 4 - Integration and usability (Day 9-12)
- [ ] App walkthrough with a non-technical classmate; fix whatever confuses them
- [ ] Plain-language labels and tooltips on every number
- [ ] Batch CSV upload tested with a sample file
- [ ] Remove any placeholder text from the UI

### Phase 5 - Freeze and validate (Day 11-13)
- [ ] Final full `python train.py`; commit `reports/metrics.json` and the figures used in slides
- [ ] `pytest -q` passes; clean install on a fresh machine works from the README alone
- [ ] Architecture diagram (`docs/ARCHITECTURE.md`) matches the code

### Phase 6 - Technical defence (Day 12-15)
- [ ] Technical deep-dive slides: architecture, feature logic, tuning, CV results, error analysis
- [ ] Live demo script, with a recorded backup in case the network fails
- [ ] Q&A drills from `docs/QA_PREP.md`: every member answers questions from every module

## 3. Rubric checklist

| Rubric item | Points | Where it is covered |
|---|---|---|
| Problem scoped, ROI quantified | 30 | `roi.py` cost model, sidebar assumptions, savings KPI |
| Usable by non-technical executives | (in 30) | Streamlit tabs, plain labels, budget view |
| Preprocessing and scaling | 40 | `data.py` pipeline, leakage-safe PayGapAdder |
| Hyperparameter tuning | (in 40) | `model.tune` RandomizedSearchCV |
| Cross-validation | (in 40) | repeated stratified 5x3, time-series CV, CF hold-out |
| Explainability (SHAP/LIME) | (in 40) | `model.Explainer`, Employee View tab |
| Error analysis | (in 40) | `model.error_analysis`, Model & Audit tab |
| Individual Q&A | 20 | `docs/QA_PREP.md`, buddy modules |
| Clean, modular, reproducible repo | 10 | required `src/data.py`, `src/model.py`, `app.py`; pinned requirements; tests |
| Vibe coding log | (in 10) | `docs/VIBE_CODING_LOG.md` |

## 4. Open decisions

1. Exclude or include sensitive attributes? (default: exclude)
2. Values for intervention cost and success rate, and their sources
3. Which LLM provider to use in the demo
4. Persona names for the segments
5. Present money in dataset units or convert (and cite the exchange rate)
