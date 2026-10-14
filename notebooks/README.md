# Notebooks

`01_eda.ipynb` - distributions, attrition rate by feature, correlations, class imbalance and data-quality checks. It imports from `src/` so the logic is not duplicated.

The other analyses were done as scripts, not notebooks, so they re-run with the pipeline:

- **Feature experiments** (CV PR-AUC with and without each engineered feature): `experiments/feature_ablation.py`, results in `reports/feature_ablation.json`. Sensitive features in vs out: `train.py`, results under `sensitive_counterfactual` in `reports/metrics.json`.
- **Error analysis** (individual missed leavers with their SHAP drivers): `train.py` writes `reports/missed_leavers_test.csv` and `reports/missed_leavers_oof_train.csv`; the write-up is section 6 of `docs/PHASE2_RISK_MODEL.md`.
