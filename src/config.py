"""Central configuration: paths, constants and business assumptions.

Every business number in this file is an ASSUMPTION that the team must be able
to defend in the Q&A. Each one says where it comes from. Lines marked
"TEAM ASSUMPTION" have no external source yet: either find one and cite it,
or present the number as a scenario and show the sensitivity (the app has
sliders for exactly this).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
JOLTS_DATA = DATA_DIR / "external" / "JTSQUR.csv"
KB_DIR = DATA_DIR / "knowledge_base"
HANDBOOK = KB_DIR / "hr_policy_handbook.md"
RAG_EVAL = KB_DIR / "retrieval_eval.csv"
MODELS_DIR = ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "attritioniq_bundle.joblib"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
AUDIT_LOG = REPORTS_DIR / "audit_log.csv"

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
# IBM HR Analytics Employee Attrition & Performance. Synthetic dataset created
# by IBM data scientists (1,470 rows, 35 columns). Kaggle mirror:
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# IBM's own copy (used by `python -m src.data --download`):
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
    "master/data/emp_attrition.csv"
)
# US JOLTS quits rate, total nonfarm, seasonally adjusted, monthly (BLS via FRED).
# Download the CSV from https://fred.stlouisfed.org/series/JTSQUR and save it
# as data/external/JTSQUR.csv
JOLTS_URL = "https://fred.stlouisfed.org/series/JTSQUR"

# --------------------------------------------------------------------------
# Modelling settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # held-out set = the "current workforce" shown in the app
CV_FOLDS = 5
CV_REPEATS = 3            # repeated stratified k-fold -> report mean +/- std
N_ITER_SEARCH = 25        # RandomizedSearchCV iterations per model
PRIMARY_METRIC = "average_precision"   # PR-AUC: right metric for ~16% positives

TARGET = "Attrition"
ID_COL = "EmployeeNumber"
# Constant columns (no information) and the ID column (would let the model memorise rows)
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18"]
# Protected / sensitive attributes. Decide as a team and defend the choice.
SENSITIVE_FEATURES = ["Gender", "Age", "MaritalStatus"]
EXCLUDE_SENSITIVE = True

# --------------------------------------------------------------------------
# Business assumptions (all editable live in the app sidebar)
# --------------------------------------------------------------------------
# Gallup (2019), "This Fixable Problem Costs U.S. Businesses $1 Trillion":
# replacing an employee costs one-half to two times their annual salary.
# https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx
REPLACEMENT_COST_MULTIPLIER = 1.0
REPLACEMENT_COST_RANGE = (0.5, 2.0)

# TEAM ASSUMPTION: average cost of one retention intervention, in months of salary.
INTERVENTION_COST_MONTHS = 1.0
# TEAM ASSUMPTION: share of flagged leavers an intervention actually retains.
INTERVENTION_SUCCESS_RATE = 0.40

# The dataset does not state its currency or the time window of the Attrition
# label. We display MonthlyIncome as-is and treat Attrition as "left within
# 12 months". Say this openly in the presentation.
CURRENCY_SYMBOL = "$"
ATTRITION_HORIZON_MONTHS = 12

# Default retention budget for the budget-allocation view (TEAM ASSUMPTION).
DEFAULT_RETENTION_BUDGET = 50_000
