"""Model & trust: model card, fairness, error analysis, figures and the audit log (Track 2)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import audit, config
from ui.blocks import Blocks, nice


def render(ctx) -> None:
    B = Blocks("trust")
    m = ctx.metrics

    B.hero(["Can you trust", "these predictions?"],
           f"How {ctx.b['model_name']} was chosen and tested, whether it treats groups fairly, "
           "where it goes wrong, and a log of every prediction viewed in this app.",
           ["Track 2: Model card", "Fairness", "Audit trail"], compact=True)

    st.caption(f"Model in use: {ctx.b['model_name']}, version {ctx.b['version']}.")
    if not m:
        st.info("No metrics found. Run `python train.py` to create reports/metrics.json.")
    tabs = st.tabs([":material/speed: Performance", ":material/balance: Fairness & errors",
                    ":material/image: Charts", ":material/receipt_long: Audit log"])

    with tabs[0]:
        if m:
            te = m["test_evaluation"]
            B.section(f"Held-out test set: {m['best_model']}",
                      f"Scored once on {sum(te['confusion_matrix'].values())} employees kept out of training, "
                      f"at threshold {te['threshold']:.2f}.")
            B.kpis([
                dict(value=te["pr_auc"], fmt="dec2", key="prauc", label="PR-AUC (main metric)", hero=True,
                     delta=f"random guess = {te['baseline_pr_auc']:.2f}"),
                dict(value=te["roc_auc"], fmt="dec2", key="rocauc", label="ROC-AUC"),
                dict(value=te["recall"], fmt="pct", key="recall", label="Leavers caught (recall)"),
                dict(value=te["precision"], fmt="pct", key="prec", label="Flags that were right (precision)"),
            ])
            cm = te["confusion_matrix"]
            c1, c2 = st.columns([1, 1.6], gap="large")
            with c1:
                st.markdown("##### Confusion matrix")
                st.dataframe(pd.DataFrame({"Predicted stay": [cm["tn"], cm["fn"]],
                                           "Predicted leave": [cm["fp"], cm["tp"]]},
                                          index=["Actually stayed", "Actually left"]))
            with c2:
                st.markdown("##### Full test evaluation")
                st.json(te, expanded=False)

            comp = pd.DataFrame({n: {f"{k} (mean)": v["mean"] for k, v in info["repeated_cv"].items()}
                                 for n, info in m["model_comparison"].items()}).T
            B.section("Model comparison", "Repeated stratified 5-fold cross-validation on training data.")
            st.dataframe(comp.round(3))

    with tabs[1]:
        if m:
            B.section("Fairness check (test set)",
                      "Gender, age and marital status are not model inputs; outcomes are still checked by group.")
            st.dataframe(pd.DataFrame(m["fairness_test"]), hide_index=True)
            B.section("Error analysis (test set)",
                      "Average profile of employees the model caught, missed and wrongly flagged.")
            st.dataframe(pd.DataFrame(m.get("error_analysis_test", [])), hide_index=True)

    with tabs[2]:
        figs = sorted(config.FIG_DIR.glob("*.png")) if config.FIG_DIR.exists() else []
        if not figs:
            st.info("No charts yet. Run `python train.py` to create them in reports/figures/.")
        for i in range(0, len(figs), 2):
            cols = st.columns(2, gap="large")
            for col, f in zip(cols, figs[i:i + 2]):
                col.image(str(f), caption=nice(f.stem.replace("_", " ")).capitalize())

    with tabs[3]:
        B.section("Audit log", "Last 50 predictions viewed in the Employee deep-dive.")
        st.dataframe(audit.read_log().tail(50), hide_index=True)
