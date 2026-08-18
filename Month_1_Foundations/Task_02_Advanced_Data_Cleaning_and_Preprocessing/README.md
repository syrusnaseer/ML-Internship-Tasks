# Task 02 — Advanced Data Cleaning & Preprocessing

## Overview

This project documents an advanced data cleaning and preprocessing workflow completed in Google Colab.

The notebook works with a customer-level dataset containing demographic, employment, income, spending, satisfaction, online-activity, and purchase fields. The workflow covers data profiling, duplicate handling, missing-value investigation, outlier checks, validity checks, numerical imputation, categorical encoding, and numerical standardization.

## Project Structure

```text
Task_02_Advanced_Data_Cleaning_and_Preprocessing/
├── README.md
├── notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── modeling.py
├── figures/
│   ├── 01_missing_values_heatmap.png
│   ├── 02_numeric_boxplots.png
│   ├── 03_annual_income_boxplot.png
│   ├── 04_missing_value_percentages.png
│   └── 05_income_missingness_by_employment.png
├── reports/
│   └── REPORT.md
├── requirements.txt
└── .gitignore
```

## Dataset

The notebook loads a file named `task2.csv`.

The uploaded notebook shows an original shape of **403 rows × 12 columns**. After duplicate removal, the working dataset contains **400 rows**. The original data includes 18 missing ages, 15 missing experience values, 22 missing annual-income values, 30 missing monthly-spending values, and 13 missing satisfaction scores.

**Important:** the original `task2.csv` was not included with the uploaded notebook in this chat, so it is intentionally not fabricated here. Place your original `task2.csv` beside `notebook.ipynb` before running the notebook.

## Methodology

1. Load and inspect the dataset.
2. Profile data types, shape, missing values, and duplicates.
3. Remove duplicate rows.
4. Visualize missing-value patterns.
5. Add missingness indicators for annual income and satisfaction score.
6. Investigate missingness against employment status and purchase outcome.
7. Inspect numerical variables with boxplots.
8. Detect annual-income outliers using the 1.5 × IQR rule.
9. Check suspicious values such as age > 100, negative experience, satisfaction outside 1–10, and negative monthly spending.
10. Impute numerical missing values with medians.
11. One-hot encode categorical variables with `drop_first=True`.
12. Standardize numerical variables with `StandardScaler`.
13. Validate the final shape, missing values, duplicates, data types, and descriptive statistics.

## Key Findings From the Notebook

- The dataset started with 403 rows and 12 columns.
- Duplicate removal reduced the working dataset to 400 rows.
- Annual income had 22 missing values; 34.92% of records with `Student` employment status had missing annual income in the notebook's cross-tabulation.
- Satisfaction-score missingness was compared with the `Purchased` outcome; the notebook reports a 50/50 split among records where satisfaction was missing.
- The annual-income IQR check reported **6 potential outliers**.
- The age validation check exposed ages of 120, 150, and 200.
- One experience value was negative (-3).
- One satisfaction value was outside the expected 1–10 range (15).
- One monthly-spending value was negative (-5000).
- Numerical missing values were filled using column medians.
- Categorical variables were one-hot encoded.
- Numerical variables were standardized using `StandardScaler`.
- Final preprocessing produced 400 rows and 24 columns.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the original `task2.csv` in this directory, then open `notebook.ipynb` in VS Code or Jupyter and run it from a fresh kernel.

## Note on Modeling

The submitted notebook is a cleaning/preprocessing task and does not train a predictive model. `src/modeling.py` is included because the internship submission structure requires that file; it provides a reusable train/test split helper for a later modeling stage.

## Reproducibility

The notebook's transformations are preserved as submitted. The modular `src/` helpers mirror the main preprocessing operations and are intended to make the workflow reusable.
