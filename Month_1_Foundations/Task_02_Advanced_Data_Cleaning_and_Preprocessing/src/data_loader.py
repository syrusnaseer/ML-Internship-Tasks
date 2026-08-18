"""Data loading helpers for Task 02."""

from pathlib import Path
import pandas as pd


def load_data(path: str | Path = "task2.csv") -> pd.DataFrame:
    """Load the Task 02 CSV dataset."""
    return pd.read_csv(Path(path))


def basic_profile(df: pd.DataFrame) -> dict:
    """Return basic dataset dimensions and missing-value information."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing": df.isnull().sum(),
        "duplicates": int(df.duplicated().sum()),
    }
