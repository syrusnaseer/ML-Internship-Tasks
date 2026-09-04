import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🏦 Bank Marketing Classification App")
st.write("Compare classification models and evaluate their performance.")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("bank-full.csv", sep=";")


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.write("Dataset Preview")
st.header("📊 Dataset")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

st.write("Dataset Columns:")
st.write(df.columns.tolist())


# --------------------------------------------------
# Target and Features
# --------------------------------------------------

X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})

st.write("Features:", X.shape)
st.write("Target:", y.shape)


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.write("Training data:", X_train.shape)
st.write("Testing data:", X_test.shape)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

st.write("Numerical columns:", numeric_cols.tolist())
st.write("Categorical columns:", categorical_cols.tolist())


# --------------------------------------------------
# Classification Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


st.write("Available Models:")
st.write(list(models.keys()))


# --------------------------------------------------
# Model Selection
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model:",
    list(models.keys())
)

st.write("Selected Model:", selected_model)


# --------------------------------------------------
# Create Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[selected_model])
])


# --------------------------------------------------
# Train Model
# --------------------------------------------------

if st.button("Train Model"):

    st.header("🤖 Model Training")
    st.write("Select a classification algorithm from the sidebar and train it.")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # Model Performance
    # --------------------------------------------------

    st.success(f"{selected_model} trained successfully!")

    st.subheader("Model Performance")

    st.write(
        "Accuracy:",
        accuracy_score(y_test, y_pred)
    )

    st.write(
        "Precision:",
        precision_score(y_test, y_pred)
    )

    st.write(
        "Recall:",
        recall_score(y_test, y_pred)
    )

    st.write(
        "F1 Score:",
        f1_score(y_test, y_pred)
    )

    st.write(
        "ROC-AUC:",
        roc_auc_score(y_test, y_prob)
    )


    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)

        # --------------------------------------------------
    # Precision-Recall Curve
    # --------------------------------------------------

    st.subheader("Precision-Recall Curve")

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob
    )

    ap_score = average_precision_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots()

    ax.plot(
        recall,
        precision,
        label=f"Average Precision = {ap_score:.3f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
# Cross-Validation
# --------------------------------------------------

st.subheader("Cross-Validation")

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

cv_results = pd.DataFrame({
    "Fold": ["Fold 1", "Fold 2", "Fold 3"],
    "F1 Score": cv_scores
})

st.dataframe(cv_results)

st.metric(
    "Mean Cross-Validation F1",
    f"{cv_scores.mean():.4f}"
)

    # --------------------------------------------------
    # Model Comparison
    # --------------------------------------------------

st.subheader("Model Comparison")

comparison_results = []

for name, model in models.items():
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        model_pipeline.fit(X_train, y_train)

        model_pred = model_pipeline.predict(X_test)
        model_prob = model_pipeline.predict_proba(X_test)[:, 1]

        comparison_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, model_pred),
            "Precision": precision_score(y_test, model_pred),
            "Recall": recall_score(y_test, model_pred),
            "F1 Score": f1_score(y_test, model_pred),
            "ROC-AUC": roc_auc_score(y_test, model_prob)
        })

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        })
    )


st.subheader("🔮 Make a Prediction")

st.write("Enter customer information below:")

input_data = {}

for col in numeric_cols:
        input_data[col] = st.number_input(
            col,
            value=float(df[col].median())
        )

for col in categorical_cols:
        input_data[col] = st.selectbox(
            col,
            df[col].dropna().unique()
        )

input_df = pd.DataFrame([input_data])

if st.button("Predict Customer"):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(
                f"Prediction: YES — Customer is likely to subscribe."
            )
        else:
            st.warning(
                f"Prediction: NO — Customer is unlikely to subscribe."
            )

        st.write(
            f"Probability of subscription: {probability:.2%}"
        )
