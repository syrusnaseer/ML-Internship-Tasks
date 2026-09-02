"""Preprocessing helpers used by the classification notebook."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Create a transformer that scales numeric and one-hot encodes categorical data."""
    categorical_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(exclude="object").columns
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return preprocessor, list(numeric_cols), list(categorical_cols)
