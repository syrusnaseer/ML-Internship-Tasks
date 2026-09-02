"""Data loading helpers for the Bank Marketing dataset."""
from pathlib import Path
import pandas as pd


def load_bank_marketing(path="bank-full.csv"):
    """Load the UCI Bank Marketing CSV using its semicolon separator."""
    path = Path(path)
    return pd.read_csv(path, sep=";")
