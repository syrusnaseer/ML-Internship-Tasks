"""Modeling utilities.

Task 02's submitted notebook focuses on cleaning and preprocessing and does not
train or evaluate a predictive model. This module is included because the
submission specification requires a modeling.py helper.
"""

from sklearn.model_selection import train_test_split


def split_features_target(df, target="Purchased", test_size=0.2, random_state=42):
    """Create a reproducible train/test split for a future modeling task."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
