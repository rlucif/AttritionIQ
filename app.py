"""AttritionIQ - Streamlit app for HR leaders.

Run:  streamlit run app.py      (after `python train.py`)

The "workforce" shown is the held-out TEST set: employees the model never saw
during training, so every score on screen is a genuine out-of-sample prediction.

Structure
---------
app.py            loads the model once, holds the business assumptions (shared by every
                  page) and sets up the top navigation
views/overview    Overview: who is at risk, what it is worth, where to spend   (Track 2)
views/employee    Employee deep-dive: why -> what-if -> plan -> copilot       (Tracks 2, 3, 5)
views/workforce   Segments, 12-month forecast, pay fairness                   (Tracks 1, 4)
views/trust       Model card, fairness, error analysis, figures, audit log    (Track 2)
ui/               theme (CSS), motion (GSAP) and the HTML blocks the pages use
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import streamlit as st

from src import config, data, rag, roi
from ui import motion, theme
from views import employee, overview, trust, workforce

st.set_page_config(page_title="AttritionIQ", page_icon="📉", layout="wide")
theme.inject()
motion.inject(config.CURRENCY_SYMBOL)


# --------------------------------------------------------------------------
# Loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not config.BUNDLE_PATH.exists():
        return None
    return joblib.load(config.BUNDLE_PATH)


@st.cache_resource
def load_retriever():
    return rag.Retriever(rag.load_chunks())


@st.cache_data
def load_metrics():
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}


b = load_bundle()
if b is None:
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()


@st.cache_data
def score_workforce(version: str):
    wf = b["test"].copy()
    X, y = data.X_y(wf, b["lists"])
    wf["p_leave"] = b["model"].predict_proba(X)[:, 1]   # real-time inference on held-out staff
    return wf, X, y


@st.cache_data
def cost_optimal_threshold(version: str, mult: float, months: float, success: float) -> float:
    """Re-optimise the threshold on TRAINING out-of-fold predictions under the chosen assumptions."""
    train = b["train"]
    t, _ = roi.optimise_threshold(train[config.TARGET], b["oof_train"], roi.replacement_cost(train, mult),
                                  roi.intervention_cost(train, months), success)
    return t


workforce_df, X_wf, y_wf = score_workforce(b["version"])
workforce_df = workforce_df.copy()

# --------------------------------------------------------------------------
# Business assumptions: one panel, shared by every page
# --------------------------------------------------------------------------
for k, v in {"mult": config.REPLACEMENT_COST_MULTIPLIER, "int_months": config.INTERVENTION_COST_MONTHS,
             "success": config.INTERVENTION_SUCCESS_RATE, "budget": config.DEFAULT_RETENTION_BUDGET}.items():
    st.session_state.setdefault(k, v)

bar_l, bar_r = st.columns([5, 1.25], vertical_alignment="center")
with bar_r.popover("Adjust assumptions", icon=":material/tune:", width="stretch"):
    st.markdown("**Business assumptions**")
    st.caption("Every page updates as you change these.")
    st.slider("Replacement cost (× annual salary)", *config.REPLACEMENT_COST_RANGE, step=0.1, key="mult",
              help="Gallup (2019): one-half to two times annual salary.")
    st.slider("Intervention cost (months of salary)", 0.0, 6.0, step=0.25, key="int_months",
              help="Team assumption.")
    st.slider("Intervention success rate", 0.05, 0.9, step=0.05, key="success", help="Team assumption.")
    st.number_input(f"Retention budget ({config.CURRENCY_SYMBOL})", 0, 10_000_000, step=5_000, key="budget",
                    help="Team assumption. Used for the budget plan on the Overview page.")

S = st.session_state
threshold = cost_optimal_threshold(b["version"], S.mult, S.int_months, S.success)
threshold_trained = b["threshold"]
workforce_df["flag"] = workforce_df.p_leave >= threshold

with bar_l:
    st.html(f'<div class="aiq-context" style="--accent:{theme.GREEN}">'
            f'<span class="aiq-chip" title="Re-optimised on training data for these assumptions">Cost-optimal threshold '
            f'<b>{threshold:.2f}</b></span>'
            f'<span class="aiq-chip aiq-chip--quiet">{threshold - threshold_trained:+.2f} vs default assumptions</span>'
            f'<span class="aiq-chip">Replacement <b>{S.mult:.1f}× salary</b></span>'
            f'<span class="aiq-chip">Intervention <b>{S.int_months:g} mo salary</b></span>'
            f'<span class="aiq-chip">Success rate <b>{S.success:.0%}</b></span>'
            f'<span class="aiq-chip">Budget <b>{config.CURRENCY_SYMBOL}{S.budget:,.0f}</b></span></div>')

emp_ids = workforce_df.sort_values("p_leave", ascending=False)[config.ID_COL].tolist()
if S.get("emp") not in emp_ids:
    S["emp"] = emp_ids[0]

ctx = SimpleNamespace(
    b=b, lists=b["lists"], metrics=load_metrics(), workforce=workforce_df, X_wf=X_wf, y_wf=y_wf,
    threshold=threshold, threshold_trained=threshold_trained, mult=S.mult, int_months=S.int_months,
    success=S.success, budget=S.budget, emp_ids=emp_ids, load_retriever=load_retriever,
)


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
def overview_page():
    overview.render(ctx)


def employee_page():
    employee.render(ctx)


def workforce_page():
    workforce.render(ctx)


def trust_page():
    trust.render(ctx)


pages = {
    "overview": st.Page(overview_page, title="Overview", icon=":material/radar:", url_path="overview", default=True),
    "employee": st.Page(employee_page, title="Employee deep-dive", icon=":material/person_search:", url_path="employee"),
    "workforce": st.Page(workforce_page, title="Workforce", icon=":material/groups:", url_path="workforce"),
    "trust": st.Page(trust_page, title="Model & trust", icon=":material/verified:", url_path="trust"),
}
ctx.pages = pages
st.navigation(list(pages.values()), position="top").run()
