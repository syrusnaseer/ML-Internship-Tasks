"""Preprocessing helpers reflecting the Task 02 notebook."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    "Age",
    "Experience_Years",
    "Annual_Income",
    "Monthly_Spending",
    "Satisfaction_Score",
    "Online_Hours_Per_Week",
]

CATEGORICAL_COLS = [
    "Gender",
    "City",
    "Education",
    "Employment_Status",
]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Income_Missing"] = df["Annual_Income"].isnull()
    df["Satisfaction_Missing"] = df["Satisfaction_Score"].isnull()
    return df


def impute_numeric_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with column medians, as in the notebook."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the notebook's categorical columns with drop_first=True."""
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=cols, drop_first=True)


def standardize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the notebook's numeric columns with StandardScaler."""
    df = df.copy()
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df
