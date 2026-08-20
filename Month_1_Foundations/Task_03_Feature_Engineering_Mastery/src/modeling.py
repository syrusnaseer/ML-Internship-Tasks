import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_classifier(y_true, predictions, probabilities):
    """Return standard binary-classification evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def select_features(X_train, y_train, feature_names, mi_k=25, rfe_k=15, l1_c=0.1):
    """Run Mutual Information, RFE, and L1 selection on training data only."""
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_rank = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    mi_selected = mi_rank.head(min(mi_k, len(mi_rank))).index.tolist()

    rfe = RFE(
        estimator=LogisticRegression(max_iter=3000),
        n_features_to_select=min(rfe_k, X_train.shape[1]),
        step=0.1,
    )
    rfe.fit(X_train, y_train)
    rfe_selected = feature_names[rfe.support_].tolist()

    l1 = LogisticRegression(
        solver="liblinear", l1_ratio=1.0, C=l1_c, max_iter=3000
    )
    l1.fit(X_train, y_train)
    l1_selected = feature_names[np.abs(l1.coef_[0]) > 1e-8].tolist()

    selected = list(dict.fromkeys(mi_selected + rfe_selected + l1_selected))
    return {
        "mi_scores": mi_rank,
        "mi_selected": mi_selected,
        "rfe_selected": rfe_selected,
        "l1_selected": l1_selected,
        "selected": selected,
    }


def fit_feature_importance(X_train, y_train, feature_names):
    """Fit a random forest and return feature importance rankings."""
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return model, importance
