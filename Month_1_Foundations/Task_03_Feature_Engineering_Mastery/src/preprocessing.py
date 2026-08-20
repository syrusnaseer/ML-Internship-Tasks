import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

JOB_LEVEL_MAP = {"Entry": 1, "Mid": 2, "Senior": 3, "Lead": 4}


def add_domain_features(data):
    """Create domain-driven, polynomial, interaction, and binned features."""
    d = data.copy()
    d["CareerStability"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["RoleStability"] = d["YearsInCurrentRole"] / (d["YearsAtCompany"] + 1)
    d["ManagerStability"] = d["YearsWithManager"] / (d["YearsAtCompany"] + 1)
    d["PromotionDelay"] = d["YearsAtCompany"] - d["YearsSincePromotion"]
    d["IncomePerWorkingYear"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    d["AgeSquared"] = d["Age"] ** 2
    d["IncomeSquared"] = d["MonthlyIncome"] ** 2
    d["DistanceSquared"] = d["DistanceFromHome"] ** 2
    d["Overtime_Satisfaction"] = d["Overtime"].eq("Yes").astype(int) * d["JobSatisfaction"]
    d["Income_JobLevel"] = d["MonthlyIncome"] * d["JobLevel"].map(JOB_LEVEL_MAP)
    d["AgeGroup"] = pd.cut(
        d["Age"], bins=[20, 30, 40, 50, 60],
        labels=["21-30", "31-40", "41-50", "51-60"]
    )
    d["DistanceGroup"] = pd.cut(
        d["DistanceFromHome"], bins=[0, 5, 15, 30, 60],
        labels=["Near", "Medium", "Far", "Very Far"]
    )
    return d


def fit_target_encoding(train_df, y_train, column="Department"):
    """Fit target encoding using training data only to prevent leakage."""
    stats = pd.DataFrame({column: train_df[column], "target": y_train}).groupby(column)["target"].mean()
    return stats.to_dict(), float(y_train.mean())


def apply_target_encoding(df, mapping, global_mean, column="Department"):
    """Apply a previously fitted target encoding map to new data."""
    out = df.copy()
    out[f"{column}_TargetEncoded"] = out[column].map(mapping).fillna(global_mean)
    return out


def build_preprocessor(X):
    """Build leakage-safe preprocessing for engineered features."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ],
        sparse_threshold=0,
    )
