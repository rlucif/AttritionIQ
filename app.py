"""AttritionIQ - Streamlit app for HR leaders.

Run:  streamlit run app.py      (after `python train.py`)

The "workforce" shown is the held-out TEST set: employees the model never saw
during training, so every score on screen is a genuine out-of-sample prediction.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import audit, config, data, forecast, model, rag, recommend, roi, segment

st.set_page_config(page_title="AttritionIQ", page_icon="📉", layout="wide")
CUR = config.CURRENCY_SYMBOL


# --------------------------------------------------------------------------
# Loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not config.BUNDLE_PATH.exists():
        return None
    return joblib.load(config.BUNDLE_PATH)


@st.cache_resource
def load_explainer(_b):
    X_train, _ = data.X_y(_b["train"], _b["lists"])
    return model.Explainer(_b["base_model"], X_train, _b["lists"])


@st.cache_resource
def load_retriever():
    return rag.Retriever(rag.load_chunks())


@st.cache_data
def load_metrics():
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}


def money(x: float) -> str:
    return f"{CUR}{x:,.0f}"


b = load_bundle()
if b is None:
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()

lists, threshold_trained = b["lists"], b["threshold"]
metrics = load_metrics()
workforce = b["test"].copy()
X_wf, y_wf = data.X_y(workforce, lists)
workforce["p_leave"] = b["model"].predict_proba(X_wf)[:, 1]   # real-time inference

# --------------------------------------------------------------------------
# Sidebar: assumptions and employee selector
# --------------------------------------------------------------------------
st.sidebar.title("📉 AttritionIQ")
st.sidebar.caption(f"Model: {b['model_name']} · version {b['version']}")
st.sidebar.subheader("Business assumptions")
mult = st.sidebar.slider("Replacement cost (× annual salary)", *config.REPLACEMENT_COST_RANGE,
                         config.REPLACEMENT_COST_MULTIPLIER, 0.1,
                         help="Gallup (2019): one-half to two times annual salary.")
int_months = st.sidebar.slider("Intervention cost (months of salary)", 0.0, 6.0,
                               config.INTERVENTION_COST_MONTHS, 0.25, help="Team assumption.")
success = st.sidebar.slider("Intervention success rate", 0.05, 0.9,
                            config.INTERVENTION_SUCCESS_RATE, 0.05, help="Team assumption.")
budget = st.sidebar.number_input(f"Retention budget ({CUR})", 0, 10_000_000,
                                 config.DEFAULT_RETENTION_BUDGET, 5_000)

# Re-optimise the threshold on TRAINING out-of-fold predictions under the new assumptions
train = b["train"]
threshold, _ = roi.optimise_threshold(train[config.TARGET], b["oof_train"],
                                      roi.replacement_cost(train, mult),
                                      roi.intervention_cost(train, int_months), success)
st.sidebar.metric("Cost-optimal threshold", f"{threshold:.2f}",
                  delta=f"{threshold - threshold_trained:+.2f} vs default assumptions")

workforce["flag"] = workforce.p_leave >= threshold
emp_ids = workforce.sort_values("p_leave", ascending=False)[config.ID_COL].tolist()
emp_id = st.sidebar.selectbox("Employee (sorted by risk)", emp_ids,
                              format_func=lambda i: f"#{i} · {workforce.set_index(config.ID_COL).loc[i, 'p_leave']:.0%}")
emp = workforce[workforce[config.ID_COL] == emp_id]
X_emp = data.X_y(emp, lists)[0]
p_emp = float(emp.p_leave.iloc[0])

tabs = st.tabs(["📊 Risk Dashboard", "👤 Employee View", "🧩 Segments", "🎯 Recommendations",
                "📈 Forecast", "💬 HR Copilot", "🔍 Model & Audit"])

# --------------------------------------------------------------------------
# 1. Risk dashboard (Track 2)
# --------------------------------------------------------------------------
with tabs[0]:
    st.header("Who is likely to leave, and what is it worth acting?")
    repl = roi.replacement_cost(workforce, mult)
    interv = roi.intervention_cost(workforce, int_months)
    summary = roi.roi_summary(y_wf, workforce.p_leave.to_numpy(), threshold, repl, interv, success)
    c = st.columns(4)
    c[0].metric("Employees scored", len(workforce))
    c[1].metric("Flagged at-risk", summary["employees_flagged"])
    c[2].metric("Expected replacement cost at stake", money((workforce.p_leave * repl).sum()))
    c[3].metric("Savings vs doing nothing", money(summary["savings_vs_do_nothing"]),
                f"{summary['savings_pct']:.0%}")
    st.caption("Savings are measured on held-out employees whose actual outcome is known.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Risk register")
        cols = [config.ID_COL, "Department", "JobRole", "MonthlyIncome", "OverTime", "p_leave", "flag"]
        st.dataframe(workforce.sort_values("p_leave", ascending=False)[cols],
                     column_config={"p_leave": st.column_config.ProgressColumn("P(leave)", format="%.2f", min_value=0, max_value=1)},
                     hide_index=True, height=380)
    with right:
        st.subheader("Where to spend the budget")
        alloc = roi.budget_allocation(workforce, workforce.p_leave.to_numpy(), repl, interv, success, budget)
        st.metric("Employees funded", len(alloc),
                  f"expected saving {money(alloc.expected_benefit.sum() - alloc.intervention_cost.sum())}")
        st.dataframe(alloc[[config.ID_COL, "p_leave", "intervention_cost", "expected_benefit", "roi_ratio"]].round(2),
                     hide_index=True, height=300)

    st.subheader("Score a new file")
    up = st.file_uploader("CSV with the same columns as the IBM dataset (Attrition optional)", type="csv")
    if up is not None:
        new = data.prepare(pd.read_csv(up))
        Xn = new[data.raw_input_columns(lists)]
        new["p_leave"] = b["model"].predict_proba(Xn)[:, 1]
        new["flag"] = new.p_leave >= threshold
        st.dataframe(new.sort_values("p_leave", ascending=False).head(50), hide_index=True)
        st.download_button("Download scores", new.to_csv(index=False), "attritioniq_scores.csv")

# --------------------------------------------------------------------------
# 2. Employee view (Track 2: SHAP + what-if)
# --------------------------------------------------------------------------
explainer = load_explainer(b)
drivers_all = explainer.drivers(X_emp, k=50)
with tabs[1]:
    st.header(f"Employee #{emp_id}")
    c = st.columns(4)
    c[0].metric("P(leave in 12 months)", f"{p_emp:.0%}", "flagged" if p_emp >= threshold else "not flagged",
                delta_color="inverse" if p_emp >= threshold else "off")
    c[1].metric("Role", emp.JobRole.iloc[0])
    c[2].metric("Monthly income", money(emp.MonthlyIncome.iloc[0]))
    c[3].metric("Years at company", int(emp.YearsAtCompany.iloc[0]))
    if st.session_state.get("last_logged") != (emp_id, threshold):  # log once per view, not per rerun
        audit.log_prediction(b["model_name"], b["version"], emp_id, p_emp, threshold, drivers_all.head(5))
        st.session_state["last_logged"] = (emp_id, threshold)

    st.subheader("Why the model thinks so")
    top = drivers_all.head(10).iloc[::-1]
    fig = px.bar(top, x="shap", y="feature", orientation="h", color="direction",
                 color_discrete_map={"raises risk": "#d62728", "lowers risk": "#2ca02c"},
                 hover_data=["value"], labels={"shap": "contribution (log-odds)", "feature": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption("SHAP values show what the model learned from the data, not proven causes.")

    st.subheader("What-if (model sensitivity, not a causal estimate)")
    w = st.columns(3)
    ot = w[0].toggle("Overtime", bool(emp.OverTime.iloc[0]))
    raise_pct = w[1].slider("Pay rise %", 0, 30, 0)
    wlb = w[2].select_slider("Work-life balance", [1, 2, 3, 4], int(emp.WorkLifeBalance.iloc[0]))
    sim = emp.copy()
    sim["OverTime"], sim["WorkLifeBalance"] = int(ot), wlb
    sim["MonthlyIncome"] = sim.MonthlyIncome * (1 + raise_pct / 100)
    sim = data.engineer_features(sim)  # recompute derived features
    p_sim = float(b["model"].predict_proba(data.X_y(sim, lists)[0])[:, 1][0])
    st.metric("Simulated P(leave)", f"{p_sim:.0%}", f"{(p_sim - p_emp) * 100:+.1f} pts", delta_color="inverse")

# --------------------------------------------------------------------------
# 3. Segments (Track 1)
# --------------------------------------------------------------------------
with tabs[2]:
    seg = b["segments"]
    st.header("Workforce segments")
    names = {k: segment.PERSONA_NAMES.get(k, f"Segment {k}") for k in range(seg["k"])}
    plot_df = pd.DataFrame(seg["pca"], columns=["PC1", "PC2"])
    plot_df["segment"] = [names[l] for l in seg["labels"]]
    plot_df["outlier (DBSCAN)"] = np.where(seg["dbscan_labels"] == -1, "outlier", "core")
    plot_df[config.ID_COL] = seg["df_ids"]
    st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="segment", symbol="outlier (DBSCAN)",
                               hover_data=[config.ID_COL], opacity=0.7), width="stretch")
    st.caption(f"K-Means with k={seg['k']} on scaled features, shown in 2D via PCA. "
               f"DBSCAN (eps={seg['eps']:.2f}) marks unusual profiles as outliers.")
    prof = seg["profile"].rename(index=names)
    st.dataframe(prof.style.format(precision=2).background_gradient(subset=["attrition_rate"], cmap="Reds"))
    desc = metrics.get("segmentation", {}).get("cluster_descriptions", {})
    for k, d in desc.items():
        st.write(f"**{names.get(int(k), k)}**: {d}")

    st.subheader("RFM-style engagement score")
    st.caption("R = years since last promotion, F = trainings last year, M = last salary hike. 5 = best.")
    rfm = seg["rfm"].copy()
    rfm["Attrition"] = pd.concat([b["train"], b["test"]]).set_index(config.ID_COL).loc[seg["df_ids"], config.TARGET].to_numpy()
    st.dataframe(rfm.groupby("RFM_band", observed=True).agg(employees=("RFM_total", "size"),
                                                           attrition_rate=("Attrition", "mean")).round(3))

# --------------------------------------------------------------------------
# 4. Recommendations (Track 3)
# --------------------------------------------------------------------------
with tabs[3]:
    st.header(f"Retention plan for #{emp_id}")
    alpha = st.slider("Weight on content-based vs collaborative", 0.0, 1.0, 0.6, 0.1)
    content = recommend.content_scores(drivers_all)
    recs = recommend.hybrid_recommend(content, b["cf_pred"].loc[emp_id], alpha=alpha, top=3)
    recs["est_cost"] = recs.cost_months * emp.MonthlyIncome.iloc[0]
    st.dataframe(recs[["name", "hybrid_score", "content_score", "cf_score", "est_cost"]].round(2), hide_index=True)
    st.caption("Collaborative scores come from a SIMULATED intervention-history matrix (no real outcome data exists).")

    st.subheader("Pay-rise sensitivity (price-elasticity analogue)")
    curve, min_raise = recommend.salary_elasticity(b["model"].predict_proba, X_emp, threshold)
    st.plotly_chart(px.line(curve, x="raise_pct", y="p_leave", markers=True)
                    .add_hline(y=threshold, line_dash="dash"), width="stretch")
    st.write(f"Smallest rise that drops risk below the threshold: **{min_raise:.0%}**" if min_raise is not None
             else "No rise up to 30% brings this employee below the threshold - pay is not the main lever.")

# --------------------------------------------------------------------------
# 5. Forecast (Track 4)
# --------------------------------------------------------------------------
with tabs[4]:
    st.header("12-month workforce forecast")
    ratio = 1.0
    if b["jolts"]:
        j = b["jolts"]
        ratio = j["ratio"]
        hist = j["series"].iloc[-72:].rename("history").to_frame()
        fc = j["forecast"].rename(columns={"forecast": "ARIMA forecast"})
        st.plotly_chart(px.line(pd.concat([hist, fc[["ARIMA forecast"]]]), labels={"value": "US quits rate %"}),
                        width="stretch")
        st.caption(f"ARIMA{tuple(j['order'])} on JOLTS quits rate. Macro adjustment ratio = {ratio:.2f} "
                   "(assumes company attrition moves with the US quits rate).")
    else:
        st.info(f"Add {config.JOLTS_DATA.name} from {config.JOLTS_URL} and re-run train.py to enable the macro forecast.")
    wf = forecast.workforce_forecast(workforce, ratio, mult)
    st.dataframe(wf.style.format({"expected_leavers": "{:.1f}", "expected_replacement_cost": CUR + "{:,.0f}",
                                  "expected_attrition_rate": "{:.1%}"}))
    st.subheader("Fair-pay model (Ridge vs Lasso)")
    pm = metrics.get("pay_model", {})
    if pm:
        st.dataframe(pd.DataFrame({k: {m: v[m] for m in ["r2_mean", "r2_std", "rmse_log_mean", "alpha", "n_zero_coefs"]}
                                   for k, v in pm.items()}).T.round(4))
    gap = b["base_model"].named_steps["paygap"].transform(X_wf)["PayGapPct"]
    st.plotly_chart(px.histogram(gap, nbins=40, labels={"value": "Pay gap vs predicted (%)"}), width="stretch")

# --------------------------------------------------------------------------
# 6. HR Copilot (Track 5)
# --------------------------------------------------------------------------
with tabs[5]:
    st.header("HR Copilot")
    retriever = load_retriever()
    q = st.text_input("Ask about policy for this employee", "What can I offer to keep this employee?")
    metric = st.radio("Distance metric", ["cosine", "euclidean"], horizontal=True)
    query = q + " " + " ".join(drivers_all[drivers_all.shap > 0].feature.head(3))
    hits = retriever.retrieve(query, k=3, metric=metric)
    st.dataframe(hits[["section", "score", "text"]], hide_index=True)
    if st.button("Generate retention brief"):
        summary = (f"#{emp_id}, {emp.JobRole.iloc[0]} in {emp.Department.iloc[0]}, "
                   f"{int(emp.YearsAtCompany.iloc[0])} years at company, P(leave) {p_emp:.0%}")
        recs = recommend.hybrid_recommend(recommend.content_scores(drivers_all), b["cf_pred"].loc[emp_id])
        text, provider = rag.generate(rag.build_prompt(q, summary, drivers_all.head(5), recs, hits))
        st.caption(f"Generated by: {provider} · embeddings: {retriever.embedder.backend}")
        st.markdown(text)

# --------------------------------------------------------------------------
# 7. Model card and audit
# --------------------------------------------------------------------------
with tabs[6]:
    st.header("Model card")
    if metrics:
        comp = pd.DataFrame({n: {f"{k} (mean)": v["mean"] for k, v in info["repeated_cv"].items()}
                             for n, info in metrics["model_comparison"].items()}).T
        st.subheader("Model comparison (repeated stratified 5-fold CV on training data)")
        st.dataframe(comp.round(3))
        st.subheader(f"Held-out test set - {metrics['best_model']}")
        st.json(metrics["test_evaluation"], expanded=False)
        st.subheader("Error analysis (test set)")
        st.dataframe(pd.DataFrame(metrics.get("error_analysis_test", [])), hide_index=True)
        st.subheader("Fairness check (test set)")
        st.dataframe(pd.DataFrame(metrics["fairness_test"]), hide_index=True)
    figs = sorted(config.FIG_DIR.glob("*.png")) if config.FIG_DIR.exists() else []
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem)
    st.subheader("Audit log")
    st.dataframe(audit.read_log().tail(50), hide_index=True)
