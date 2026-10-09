"""Employee deep-dive: one guided flow per person.

1 Why the model flags them (SHAP, Track 2)  ->  2 What-if (Track 2)
->  3 Retention plan (hybrid recommender + pay-rise elasticity, Track 3)  ->  4 HR copilot (RAG, Track 5)
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import audit, config, data, model, rag, recommend
from ui import charts, theme
from ui.blocks import Blocks, money

ID = config.ID_COL


@st.cache_resource
def load_explainer(_b, version: str):
    X_train, _ = data.X_y(_b["train"], _b["lists"])
    return model.Explainer(_b["base_model"], X_train, _b["lists"])


def render(ctx) -> None:
    B = Blocks("employee")
    b, wf, t, S = ctx.b, ctx.workforce, ctx.threshold, st.session_state

    B.hero(["Why might they leave,", "and what would keep them?"],
           "Pick an employee, then work down the page: the model's reasons, a what-if test, "
           "a costed retention plan and answers from the HR handbook.",
           ["Track 2: Explainability", "Track 3: Recommendations", "Track 5: HR copilot"], compact=True)

    # ---- employee picker (shared with the Overview risk register) -----------------
    by_id = wf.set_index(ID)
    if S.get("emp_pick") != S.emp:      # picked on the Overview register or with "Next"
        S.emp_pick = S.emp
    pos = ctx.emp_ids.index(S.emp)

    def go_next():
        S.emp = S.emp_pick = ctx.emp_ids[pos + 1]

    pick_col, nav_col = st.columns([4, 1.2], vertical_alignment="bottom")
    with pick_col:
        st.selectbox("Employee, highest risk first", ctx.emp_ids,
                     format_func=lambda i: f"#{i} ({by_id.loc[i, 'p_leave']:.0%}), {by_id.loc[i, 'JobRole']}",
                     key="emp_pick", on_change=lambda: S.update(emp=S.emp_pick))
    with nav_col:
        st.button("Next highest risk", icon=":material/arrow_downward:", width="stretch",
                  disabled=pos == len(ctx.emp_ids) - 1, on_click=go_next)

    emp_id = S.emp
    emp = wf[wf[ID] == emp_id]
    row = emp.iloc[0]
    X_emp = data.X_y(emp, ctx.lists)[0]
    p_emp = float(row.p_leave)
    B.person(row, p_emp, t)

    explainer = load_explainer(b, b["version"])
    drivers_all = explainer.drivers(X_emp, k=50)
    if S.get("last_logged") != (emp_id, t):   # log once per view, not per rerun
        audit.log_prediction(b["model_name"], b["version"], emp_id, p_emp, t, drivers_all.head(5))
        S["last_logged"] = (emp_id, t)

    # ---- 1. why ------------------------------------------------------------------
    B.step(1, "Why the model flags them",
           "The ten factors that moved this score most. Bars to the right push risk up; to the left, down.")
    B.drivers(drivers_all)
    st.caption("SHAP values (contribution in log-odds) show what the model learned from the data, not proven causes.")

    # ---- 2. what-if --------------------------------------------------------------
    B.step(2, "Try a what-if", "Change a lever and see how the model's score responds. "
                               "This is model sensitivity, not a causal estimate.")
    w = st.columns(3, gap="large")
    ot = w[0].toggle("Works overtime", bool(row.OverTime), key=f"ot_{emp_id}")
    raise_pct = w[1].slider("Pay rise %", 0, 30, 0, key=f"raise_{emp_id}")
    wlb = w[2].select_slider("Work-life balance (1 to 4)", [1, 2, 3, 4], int(row.WorkLifeBalance), key=f"wlb_{emp_id}")
    sim = emp.copy()
    sim["OverTime"], sim["WorkLifeBalance"] = int(ot), wlb
    sim["MonthlyIncome"] = sim.MonthlyIncome * (1 + raise_pct / 100)
    sim = data.engineer_features(sim)  # recompute derived features
    p_sim = float(b["model"].predict_proba(data.X_y(sim, ctx.lists)[0])[:, 1][0])
    B.shift(p_emp, p_sim)

    # ---- 3. retention plan -------------------------------------------------------
    B.step(3, "Build a retention plan", "Interventions ranked by how well they match this person's risk drivers "
                                        "and how similar employees responded.")
    alpha = st.slider("Weight on matching drivers vs similar employees", 0.0, 1.0, 0.6, 0.1,
                      help="1.0 = content-based only (this person's SHAP drivers). "
                           "0.0 = collaborative filtering only (similar employees).")
    content = recommend.content_scores(drivers_all)
    recs = recommend.hybrid_recommend(content, b["cf_pred"].loc[emp_id], alpha=alpha, top=3)
    recs["est_cost"] = recs.cost_months * row.MonthlyIncome
    B.recs(recs)
    st.caption("Collaborative scores come from a SIMULATED intervention-history matrix (no real outcome data exists).")

    st.markdown("##### Would a pay rise be enough?")
    curve, min_raise = recommend.salary_elasticity(b["model"].predict_proba, X_emp, t)
    fig = px.line(curve, x="raise_pct", y="p_leave", markers=True,
                  labels={"raise_pct": "Pay rise", "p_leave": "P(leave)"})
    fig.update_traces(line_color=theme.LILAC, line_width=3, marker_size=6)
    fig.add_hline(y=t, line_dash="dash", line_color=theme.MUTED, annotation_text=f"threshold {t:.2f}",
                  annotation_font_color=theme.MUTED)
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    charts.show(fig, height=300)
    if min_raise is not None:
        B.note(f"The smallest rise that brings this employee below the threshold is {min_raise:.0%} "
               f"(about {money(row.MonthlyIncome * min_raise * 12)} a year). This is a price-elasticity analogue.")
    else:
        B.note("No rise up to 30% brings this employee below the threshold, so pay is not the main lever.")

    # ---- 4. copilot --------------------------------------------------------------
    B.step(4, "Ask the HR copilot", "Searches the HR policy handbook using your question plus this person's top "
                                    "risk drivers, then drafts a retention brief.")
    retriever = ctx.load_retriever()
    q_col, m_col = st.columns([3, 1], vertical_alignment="bottom")
    q = q_col.text_input("Ask about policy for this employee", "What can I offer to keep this employee?")
    metric = m_col.radio("Distance metric", ["cosine", "euclidean"], horizontal=True)
    query = rag.driver_query(q, drivers_all[drivers_all.shap > 0].feature.head(3))
    hits = retriever.retrieve(query, k=3, metric=metric)
    if not hits.matched.iloc[0]:
        st.warning("No policy shares any words with this question. Try rephrasing it.")
    for i, (_, h) in enumerate(hits.iterrows()):
        with st.expander(f"{h.section}  (score {h.score:.3f})", expanded=i == 0):
            st.write(h.text)
    if st.button("Generate retention brief", type="primary", icon=":material/auto_awesome:"):
        summary = (f"#{emp_id}, {row.JobRole} in {row.Department}, "
                   f"{int(row.YearsAtCompany)} years at company, P(leave) {p_emp:.0%}")
        recs_all = recommend.hybrid_recommend(recommend.content_scores(drivers_all), b["cf_pred"].loc[emp_id])
        with st.spinner("Drafting the brief"):
            text, provider = rag.generate(rag.build_prompt(q, summary, drivers_all.head(5), recs_all, hits))
        with st.container(border=True):
            st.caption(f"Generated by: {provider}. Embeddings: {retriever.embedder.backend}.")
            st.markdown(text)
