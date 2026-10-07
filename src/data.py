"""Data loading, cleaning, feature engineering and preprocessing.

Pipeline order (see docs/ARCHITECTURE.md):
    load_raw -> clean -> engineer_features -> split -> build_preprocessor

Anything that LEARNS from data (scaling means, one-hot categories, the pay
model) is fitted inside the sklearn Pipeline, so cross-validation never sees
the validation fold during fitting (no leakage).

Run `python -m src.data --download` to fetch the dataset from IBM's GitHub.
"""
from __future__ import annotations

import argparse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# Ordered categories -> integers (a frequent traveller travels more than a rare one)
BUSINESS_TRAVEL_ORDER = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}

# Nominal categories -> one-hot encoded (no natural order)
NOMINAL_CATEGORICALS = ["Department", "EducationField", "JobRole", "MaritalStatus"]

# 1-4 / 1-5 survey-style scales. Kept as numeric (ordinal) and scaled.
# Q&A point: we assume equal spacing between levels; one-hot would drop that
# assumption at the cost of more columns.
ORDINAL_SCALES = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "WorkLifeBalance", "BusinessTravel",
]

# Features created in engineer_features()
ENGINEERED = [
    "YearsPerCompany", "PromotionLagRatio", "ManagerTenureRatio", "SatisfactionIndex",
]

# Created inside the pipeline by forecast.PayGapAdder (Track 4 -> Track 2)
PAY_GAP_FEATURE = "PayGapPct"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def download_dataset(url: str = config.DATASET_URL, dest=config.RAW_DATA) -> None:
    """Download the IBM attrition CSV to data/raw/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Saved dataset to {dest}")


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_DATA
    """Read the raw CSV. utf-8-sig strips the byte-order mark some copies carry."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data --download` or download "
            "the CSV from Kaggle and place it in data/raw/."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Cleaning and features
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop useless columns and encode binary / ordered text columns as numbers."""
    df = df.copy()
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    is_text = lambda c: c in df.columns and not pd.api.types.is_numeric_dtype(df[c])  # noqa: E731
    if is_text(config.TARGET):  # uploads for scoring may have no Attrition column
        df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    if is_text("OverTime"):
        df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    if is_text("Gender"):
        df["Gender"] = (df["Gender"] == "Male").astype(int)
    if is_text("BusinessTravel"):
        df["BusinessTravel"] = df["BusinessTravel"].map(BUSINESS_TRAVEL_ORDER)
    # TODO(team): add data-quality checks (duplicates, impossible values such as
    # YearsAtCompany > TotalWorkingYears) and report them in the EDA notebook.
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features a manager can understand. +1 in denominators avoids /0."""
    df = df.copy()
    df["YearsPerCompany"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    df["PromotionLagRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["ManagerTenureRatio"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    df["SatisfactionIndex"] = df[
        ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    ].mean(axis=1)
    # TODO(team): test whether each engineered feature improves CV PR-AUC.
    # Keep only the ones that do, and record the result in reports/.
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """clean + engineer_features. Used for training AND for new uploads in the app."""
    return engineer_features(clean(df))


# --------------------------------------------------------------------------
# Feature lists and preprocessing
# --------------------------------------------------------------------------
def feature_lists(exclude_sensitive: bool = config.EXCLUDE_SENSITIVE) -> dict:
    """Return the model's input columns, grouped by how they are encoded."""
    numeric = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "OverTime", "Gender", "PercentSalaryHike",
        "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    ] + ORDINAL_SCALES + ENGINEERED
    categorical = list(NOMINAL_CATEGORICALS)
    if exclude_sensitive:
        numeric = [c for c in numeric if c not in config.SENSITIVE_FEATURES]
        categorical = [c for c in categorical if c not in config.SENSITIVE_FEATURES]
    return {"numeric": numeric, "categorical": categorical,
            "model_numeric": numeric + [PAY_GAP_FEATURE]}


def raw_input_columns(lists: dict) -> list[str]:
    """Columns the pipeline needs as input (pay model needs JobLevel, JobRole etc.)."""
    from src.forecast import PAY_MODEL_FEATURES  # local import avoids a cycle
    cols = lists["numeric"] + lists["categorical"] + ["MonthlyIncome"] + PAY_MODEL_FEATURES
    return list(dict.fromkeys(cols))


def build_preprocessor(lists: dict) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode nominal ones. Output stays a DataFrame
    so SHAP and the app can show real feature names."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), lists["model_numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), lists["categorical"]),
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def split(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the ~16% attrition rate."""
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def X_y(df: pd.DataFrame, lists: dict):
    return df[raw_input_columns(lists)], df[config.TARGET].to_numpy()


def base_feature(name: str, lists: dict) -> str:
    """Map a transformed column name back to the original feature.
    e.g. 'JobRole_Sales Executive' -> 'JobRole'."""
    for cat in lists["categorical"]:
        if name.startswith(cat + "_"):
            return cat
    return name


def data_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "attrition_rate": float(df[config.TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download the dataset")
    args = parser.parse_args()
    if args.download:
        download_dataset()
    df = prepare(load_raw())
    print(data_summary(df))
    np.set_printoptions(suppress=True)
    print(df.describe().T.head(15))
