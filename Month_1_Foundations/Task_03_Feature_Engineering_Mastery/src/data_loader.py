from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path("data/employee_attrition_feature_engineering_5000.csv")


def load_dataset(path=DEFAULT_DATA_PATH):
    """Load the employee attrition dataset from CSV."""
    return pd.read_csv(path)


def clean_target(df):
    """Map Attrition from Yes/No to 1/0 and remove the identifier."""
    data = df.copy()
    data["Attrition"] = data["Attrition"].map({"No": 0, "Yes": 1})
    return data.drop(columns=["EmployeeID"], errors="ignore")
