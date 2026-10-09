"""Overview: who is likely to leave, what it is worth, where to spend the budget (Track 2)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data, roi
from ui.blocks import CUR, Blocks, money

ID = config.ID_COL


def render(ctx) -> None:
    B = Blocks("overview")
    wf, t = ctx.workforce, ctx.threshold

    # ---- hero: the workforce as a field of dots ----------------------------------
    ordered = wf.sort_values(ID)
    field, cols = B.dot_field(ordered[ID].tolist(), ordered.p_leave.to_numpy(), t)
    B.hero(["Who is likely", "to leave, and is", "it worth acting?"],
           f"Each dot is one of {len(wf)} employees the model never saw in training. "
           f"Lit dots are flagged at the cost-optimal threshold of {t:.2f} for your current assumptions. "
           "Hover a dot to see its score.",
           ["Track 2: Decision automation", "Out-of-sample scores"], right=field, cols=cols)

    repl = roi.replacement_cost(wf, ctx.mult)
    interv = roi.intervention_cost(wf, ctx.int_months)
    summary = roi.roi_summary(ctx.y_wf, wf.p_leave.to_numpy(), t, repl, interv, ctx.success)
    B.kpis([
        dict(value=len(wf), fmt="int", key="scored", label="Employees scored"),
        dict(value=summary["employees_flagged"], fmt="int", key="flagged", label="Flagged at risk"),
        dict(value=float((wf.p_leave * repl).sum()), fmt="money", key="stake",
             label="Expected replacement cost at stake"),
        dict(value=summary["savings_vs_do_nothing"], fmt="money", key="save", hero=True,
             label="Saved vs doing nothing", delta=f"{summary['savings_pct']:.0%} lower cost"),
    ])
    st.caption("Savings are measured on held-out employees whose actual outcome is known.")

    # ---- risk register -----------------------------------------------------------
    B.section("Risk register", "Sorted by chance of leaving. Select a row to open that employee's deep-dive.")
    cols_ = [ID, "Department", "JobRole", "MonthlyIncome", "OverTime", "p_leave", "flag"]
    reg = wf.sort_values("p_leave", ascending=False)[cols_].reset_index(drop=True)
    reg["OverTime"] = reg.OverTime.astype(bool)
    ev = st.dataframe(
        reg, hide_index=True, height=390, on_select="rerun", selection_mode="single-row", key="register",
        column_config={
            ID: st.column_config.NumberColumn("Employee", format="#%d"),
            "JobRole": "Job role",
            "MonthlyIncome": st.column_config.NumberColumn("Monthly income", format=f"{CUR}%d"),
            "OverTime": st.column_config.CheckboxColumn("Overtime"),
            "p_leave": st.column_config.ProgressColumn("P(leave)", format="%.2f", min_value=0, max_value=1),
            "flag": st.column_config.CheckboxColumn("Flagged"),
        })
    if ev.selection.rows:
        st.session_state["emp"] = int(reg.loc[ev.selection.rows[0], ID])
        st.switch_page(ctx.pages["employee"])

    # ---- budget ------------------------------------------------------------------
    B.section("Where to spend the budget",
              f"Funds the best return on each {CUR} first, until your {money(ctx.budget)} budget runs out.")
    alloc = roi.budget_allocation(wf, wf.p_leave.to_numpy(), repl, interv, ctx.success, ctx.budget)
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        B.kpis([dict(value=len(alloc), fmt="int", key="funded", label="Employees funded"),
                dict(value=float(alloc.expected_benefit.sum() - alloc.intervention_cost.sum()), fmt="money",
                     key="netsave", label="Expected saving after intervention cost", hero=True)], stack=True)
    with right:
        st.dataframe(
            alloc[[ID, "p_leave", "intervention_cost", "expected_benefit", "roi_ratio"]], hide_index=True, height=300,
            column_config={
                ID: st.column_config.NumberColumn("Employee", format="#%d"),
                "p_leave": st.column_config.NumberColumn("P(leave)", format="%.2f"),
                "intervention_cost": st.column_config.NumberColumn("Intervention cost", format=f"{CUR}%d"),
                "expected_benefit": st.column_config.NumberColumn("Expected benefit", format=f"{CUR}%d"),
                "roi_ratio": st.column_config.NumberColumn("Return per $1", format="%.2f"),
            })

    # ---- batch scoring -----------------------------------------------------------
    B.section("Score a new file", "Upload staff records in the IBM dataset format to score them with the same "
              "model and threshold. The Attrition column is optional.")
    up = st.file_uploader("CSV with the same columns as the IBM dataset", type="csv", label_visibility="collapsed")
    if up is not None:
        new = data.prepare(pd.read_csv(up))
        Xn = new[data.raw_input_columns(ctx.lists)]
        new["p_leave"] = ctx.b["model"].predict_proba(Xn)[:, 1]
        new["flag"] = new.p_leave >= t
        st.success(f"Scored {len(new):,} employees: {int(new.flag.sum())} flagged at risk.")
        st.dataframe(new.sort_values("p_leave", ascending=False).head(50), hide_index=True,
                     column_config={"p_leave": st.column_config.ProgressColumn(
                         "P(leave)", format="%.2f", min_value=0, max_value=1)})
        st.download_button("Download scores", new.to_csv(index=False), "attritioniq_scores.csv",
                           icon=":material/download:")
