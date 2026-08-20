# Task 03 — Feature Engineering Mastery

**Author:** Mussa Khan (@musagithub1)  
**Task:** Feature Engineering Mastery  
**Dataset:** 5,000-row synthetic Employee Attrition dataset

## Objective

This project demonstrates practical feature engineering and feature selection on a binary employee-attrition problem. The dataset is intentionally different from the original real-estate dataset in the task brief so the techniques can be demonstrated on a separate domain.

The implementation covers:

- Leakage-safe data cleaning and train/test splitting
- Domain-driven feature creation
- Polynomial and interaction features
- Binning
- Target encoding using training data only
- Mutual Information feature selection
- Recursive Feature Elimination (RFE)
- L1-based feature selection
- Random Forest feature importance
- Baseline vs engineered-model evaluation

## Dataset

`data/employee_attrition_feature_engineering_5000.csv` contains 5,000 synthetic employee records and 29 original columns, including the binary target `Attrition`.

The dataset contains intentionally introduced missing values to make preprocessing reproducible and meaningful.

## Run in Google Colab

1. Upload or clone this repository into Colab.
2. Open `notebook.ipynb`.
3. Run the notebook from top to bottom.

If the repository is cloned, the notebook automatically uses the dataset under `data/`.

## Local setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Then open `notebook.ipynb` with Jupyter/VS Code.

## Reproducibility

A fixed random seed (`42`) is used for train/test splitting and stochastic models. Target encoding is fitted only on the training set to prevent target leakage.

## Repository structure

```text
Task_03_Feature_Engineering_Mastery/
├── README.md
├── notebook.ipynb
├── data/
│   └── employee_attrition_feature_engineering_5000.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── modeling.py
├── figures/
├── reports/
│   └── REPORT.md
├── requirements.txt
└── .gitignore
```
