"""Workforce: segments (Track 1), 12-month forecast and pay fairness (Track 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, forecast, segment
from ui import charts, theme
from ui.blocks import CUR, Blocks

ID = config.ID_COL


def render(ctx) -> None:
    B = Blocks("workforce")
    b, wf, metrics = ctx.b, ctx.workforce, ctx.metrics

    B.hero(["Which groups are drifting,", "and what will next year cost?"],
           "Clusters of similar employees, a 12-month forecast of leavers by department, "
           "and how each person's pay compares with what the market model expects.",
           ["Track 1: Workforce segments", "Track 4: Predictive operations"], compact=True)

    tab_seg, tab_fc, tab_pay = st.tabs([":material/scatter_plot: Segments", ":material/trending_up: 12-month forecast",
                                        ":material/balance: Pay fairness"])

    # ---- segments ----------------------------------------------------------------
    with tab_seg:
        seg = b["segments"]
        names = seg.get("personas") or segment.persona_names(seg["profile"])  # team-approved names
        prof = seg["profile"]
        B.section("Three kinds of employee", "Grouped by K-Means on pay, tenure, workload and satisfaction.")
        cards = st.columns(len(prof))
        desc = metrics.get("segmentation", {}).get("cluster_descriptions", {})
        for col, (k, r) in zip(cards, prof.sort_values("attrition_rate", ascending=False).iterrows()):
            with col.container(border=True):
                st.markdown(f"**{names.get(int(k), k)}**")
                st.metric("Attrition rate", f"{r.attrition_rate:.0%}", f"{int(r['size'])} employees",
                          delta_color="off", delta_arrow="off")
                if str(k) in desc:
                    st.caption(desc[str(k)])

        plot_df = pd.DataFrame(seg["pca"], columns=["PC1", "PC2"])
        plot_df["segment"] = [names[l] for l in seg["labels"]]
        plot_df["outlier (DBSCAN)"] = np.where(seg["dbscan_labels"] == -1, "outlier", "core")
        plot_df[ID] = seg["df_ids"]
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="segment", symbol="outlier (DBSCAN)",
                         hover_data=[ID], opacity=0.75, render_mode="svg")
        fig.update_traces(marker_size=7)
        charts.show(fig, height=460)
        st.caption(f"K-Means with k={seg['k']} on scaled features, shown in 2D via PCA. "
                   f"DBSCAN (eps={seg['eps']:.2f}) marks unusual profiles as outliers.")

        with st.expander("Full segment profile (average of each feature)"):
            st.dataframe(prof.rename(index=names).style.format(precision=2)
                         .background_gradient(subset=["attrition_rate"], cmap="Reds"))

        B.section("Engagement score (RFM-style)",
                  "R = years since last promotion, F = trainings last year, M = last salary hike. 5 = best.")
        rfm = seg["rfm"].copy()
        rfm["Attrition"] = (pd.concat([b["train"], b["test"]]).set_index(ID)
                            .loc[seg["df_ids"], config.TARGET].to_numpy())
        rfm_tab = rfm.groupby("RFM_band", observed=True).agg(employees=("RFM_total", "size"),
                                                              attrition_rate=("Attrition", "mean")).round(3)
        c1, c2 = st.columns([1, 1.4], gap="large")
        c1.dataframe(rfm_tab, column_config={
            "RFM_band": "Engagement band", "employees": "Employees",
            "attrition_rate": st.column_config.ProgressColumn("Attrition rate", format="%.3f", min_value=0,
                                                              max_value=float(max(rfm_tab.attrition_rate.max(), .01)))})
        bar = px.bar(rfm_tab.reset_index(), x="RFM_band", y="attrition_rate",
                     labels={"RFM_band": "", "attrition_rate": "Attrition rate"})
        bar.update_traces(marker_color=theme.ORANGE)
        bar.update_yaxes(tickformat=".0%")
        with c2:
            charts.show(bar, height=260)

    # ---- forecast ----------------------------------------------------------------
    with tab_fc:
        ratio = 1.0
        B.section("Expected leavers over the next 12 months",
                  "Each employee's risk, summed by department and scaled by where the US quits rate is heading.")
        if b["jolts"]:
            j = b["jolts"]
            ratio = j["ratio"]
        wfc = forecast.workforce_forecast(wf, ratio, ctx.mult)
        B.kpis([
            dict(value=float(wfc.expected_leavers.sum()), fmt="int", key="leavers", label="Expected leavers", hero=True),
            dict(value=float(wfc.expected_replacement_cost.sum()), fmt="money", key="fccost",
                 label="Expected replacement cost"),
            dict(value=float(wfc.expected_leavers.sum() / wfc.headcount.sum()), fmt="pct1", key="fcrate",
                 label="Expected attrition rate"),
            dict(value=ratio, fmt="dec2", key="ratio", label="Macro adjustment (US quits trend)"),
        ])
        st.dataframe(wfc.style.format({"expected_leavers": "{:.1f}", "expected_replacement_cost": CUR + "{:,.0f}",
                                       "expected_attrition_rate": "{:.1%}"}),
                     column_config={"headcount": "Headcount", "expected_leavers": "Expected leavers",
                                    "expected_replacement_cost": "Expected replacement cost",
                                    "expected_attrition_rate": "Expected attrition rate"})
        if b["jolts"]:
            j = b["jolts"]
            hist = j["series"].iloc[-72:].rename("history").to_frame()
            used = j.get("model_used", "ARIMA")
            label = f"{used} forecast"
            fc = j["forecast"].rename(columns={"forecast": label})
            B.section("US quits rate", "The macro signal behind the adjustment above (JOLTS, BLS via FRED).")
            fig = px.line(pd.concat([hist, fc[[label]]]), labels={"value": "US quits rate %", "index": ""})
            fig.update_traces(line_width=2.5)
            charts.show(fig, height=340)
            why = ("" if used == "ARIMA" else
                   f" ARIMA{tuple(j['order'])} was tested but did not beat 'next month = this month' in back-tests, "
                   "so the simpler forecast is used.")
            st.caption(f"US quits rate (JOLTS, BLS via FRED).{why} Macro adjustment = {ratio:.2f}: expected "
                       "attrition is scaled by this, assuming the company moves with the US quits rate.")
        else:
            st.info(f"Add {config.JOLTS_DATA.name} from {config.JOLTS_URL} and re-run train.py to enable the macro forecast.")

    # ---- pay fairness ------------------------------------------------------------
    with tab_pay:
        B.section("Who is paid below what the model expects?",
                  "Pay gap = actual pay vs the pay predicted for the same role, level and experience. "
                  "Negative means paid below expectation.")
        gap = b["base_model"].named_steps["paygap"].transform(ctx.X_wf)["PayGapPct"]
        hist = px.histogram(gap, nbins=40, labels={"value": "Pay gap vs predicted (%)"})
        hist.update_traces(marker_color=theme.ORANGE, marker_line_width=0)
        hist.update_layout(showlegend=False, bargap=0.05)
        charts.show(hist, height=340)
        pm = metrics.get("pay_model", {})
        if pm:
            with st.expander("How the fair-pay model was chosen (Ridge vs Lasso)", expanded=False):
                st.dataframe(pd.DataFrame({k: {m: v[m] for m in ["r2_mean", "r2_std", "rmse_log_mean", "alpha",
                                                                    "n_zero_coefs"]}
                                           for k, v in pm.items()}).T.round(4))
