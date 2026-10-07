# Architecture

```mermaid
flowchart LR
    A[IBM attrition CSV] --> B[data.py<br/>clean + engineer features]
    J[JOLTS quits rate CSV] --> F2
    H[HR handbook .md] --> R1

    subgraph T2[Track 2 - Risk Engine]
        P1[PayGapAdder<br/>Ridge pay model] --> P2[Scale + one-hot] --> P3[LR / RF / XGB / MLP<br/>RandomizedSearchCV]
        P3 --> P4[Calibration] --> P5[Cost-optimal threshold<br/>roi.py]
        P3 --> P6[SHAP explainer]
    end
    B --> P1

    subgraph T1[Track 1 - Segments]
        S1[K-Means / Ward / DBSCAN] --> S2[Personas]
        S3[RFM-style score]
    end
    B --> S1
    B --> S3
    P6 -. SHAP values .-> S1

    subgraph T3[Track 3 - Recommender]
        C1[Content-based<br/>SHAP drivers -> interventions]
        C2[Funk SVD<br/>simulated matrix]
        C1 --> C3[Hybrid ranking]
        C2 --> C3
        C4[Pay-rise elasticity]
    end
    P6 --> C1
    P4 --> C4

    subgraph T4[Track 4 - Forecast]
        F1[Ridge vs Lasso pay model]
        F2[ARIMA / Prophet<br/>macro ratio] --> F3[Workforce forecast<br/>by department]
    end
    P4 --> F3
    F1 -. PayGapPct feature .-> P1

    subgraph T5[Track 5 - HR Copilot]
        R1[Chunk + embed] --> R2[FAISS index<br/>cosine / Euclidean] --> R3[LLM brief]
    end
    P6 --> R3
    C3 --> R3

    P5 --> APP[Streamlit app.py]
    S2 --> APP
    C3 --> APP
    F3 --> APP
    R3 --> APP
    APP --> L[Audit log]
```

## Data flow

1. `train.py` loads and prepares the data, then splits it 80/20 with stratification. The 20% test set is the "current workforce" in the app, so every score shown there is out of sample.
2. Everything that learns from data (pay model, scaler, encoder, SMOTE, classifier) sits inside one pipeline. Cross-validation re-fits it per fold, so no validation information leaks into training.
3. The decision threshold is chosen on **out-of-fold training predictions**. The test set is used once, for the final report.
4. `train.py` saves a single bundle plus `reports/metrics.json`. The app loads the bundle and runs inference live, including on uploaded CSVs.

## Design decisions to defend

| Decision | Why | Alternative considered |
|---|---|---|
| PR-AUC as the primary metric | 16% positives; accuracy is misleading | ROC-AUC (reported too) |
| Cost-based threshold | Errors have different costs | Fixed 0.5 |
| Platt (sigmoid) calibration | ~190 leavers is too few for isotonic | Isotonic |
| Pay gap as a pipeline step | Prevents leakage across CV folds | Pre-computing on all data |
| Sensitive features excluded | Fairness and policy risk | Include, then audit |
| Test set as app workforce | Honest out-of-sample demo | Scoring training rows |
| TF-IDF fallback for embeddings | Demo works offline | Fail without the model |
