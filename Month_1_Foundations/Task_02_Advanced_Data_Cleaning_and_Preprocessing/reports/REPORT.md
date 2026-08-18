# Task 02 — Advanced Data Cleaning & Preprocessing — Report
Author: Mussa Khan (@musagithub1)
Date: 2026-08-17

## 1. Executive Summary

- The notebook profiles and cleans a customer-level dataset containing 403 initial records and 12 columns.
- Duplicate removal reduces the working data to 400 records.
- Missing values are investigated using percentages, a heatmap, missingness indicators, and cross-tabulations.
- Numerical missing values are imputed with medians, categorical fields are one-hot encoded, and numerical features are standardized.
- The notebook also identifies potential outliers and several suspicious/invalid values that should be reviewed before downstream modeling.

## 2. Business Problem & Framing

The task is framed as an advanced data cleaning and preprocessing exercise. The objective is to convert a customer dataset into a cleaner, analysis-ready representation while explicitly investigating data-quality problems before applying transformations.

The target field present in the dataset is `Purchased`. However, the submitted notebook focuses on preprocessing and does not train or evaluate a predictive model.

## 3. Data Overview

The notebook reports an original dataset shape of 403 rows × 12 columns.

The fields include:

- `Customer_ID`
- `Age`
- `Gender`
- `City`
- `Education`
- `Employment_Status`
- `Experience_Years`
- `Annual_Income`
- `Monthly_Spending`
- `Satisfaction_Score`
- `Online_Hours_Per_Week`
- `Purchased`

The initial missing-value counts reported by the notebook are:

| Variable | Missing |
|---|---:|
| Age | 18 |
| Experience_Years | 15 |
| Annual_Income | 22 |
| Monthly_Spending | 30 |
| Satisfaction_Score | 13 |

The dataset also contains duplicate rows; after `drop_duplicates()`, the notebook reports 400 rows.

## 4. Methodology

### Data profiling

The notebook checks data types, dimensions, missing-value counts, missing-value percentages, and duplicate counts.

### Duplicate handling

Duplicate rows are removed with:

```python
df = df.drop_duplicates()
```

### Missing-data investigation

The notebook creates:

- a missing-value heatmap;
- an `Income_Missing` indicator;
- a `Satisfaction_Missing` indicator;
- cross-tabulations against employment status and purchase outcome.

For annual income, the notebook reports that students have 34.92% missing income while the other listed employment groups show 0% in the cross-tabulation.

For satisfaction missingness, the notebook reports a 50/50 purchased split among records where satisfaction is missing.

These checks provide evidence about patterns of missingness but do not by themselves prove that missingness is MCAR, MAR, or MNAR.

### Outlier and validity checks

Boxplots are used to inspect numerical variables.

For annual income, the notebook applies the 1.5 × IQR rule and reports six potential outliers.

Additional validity checks identify:

- three ages above 100 (120, 150, and 200);
- one negative experience value (-3);
- one satisfaction value outside 1–10 (15);
- one negative monthly-spending value (-5000).

The notebook checks these issues but does not show a final correction/removal operation for each invalid value.

### Imputation

Numerical columns are filled with their respective medians:

```python
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
```

### Encoding

`Gender`, `City`, `Education`, and `Employment_Status` are one-hot encoded using `pd.get_dummies(..., drop_first=True)`.

### Scaling

The six numerical variables are standardized with Scikit-learn's `StandardScaler`.

## 5. Results & Key Visualizations

Five figures are included in `figures/`.

1. **Missing-values heatmap** — visualizes the pattern of missing observations.
2. **Numeric boxplots** — compares distributions and highlights potential outliers.
3. **Annual-income boxplot** — focuses on income outliers.
4. **Missing-value percentages** — summarizes the reported missingness percentages.
5. **Annual-income missingness by employment** — visualizes the notebook's cross-tabulation.

After preprocessing, the notebook reports:

- shape: 400 rows × 24 columns;
- zero remaining missing values;
- zero duplicate rows;
- standardized numerical features.

## 6. Limitations & Risks

- The notebook does not prove a specific MCAR/MAR/MNAR mechanism.
- Several invalid values are identified but not corrected in the submitted transformation sequence.
- Potential annual-income outliers are identified but not removed or winsorized.
- The notebook uses `StandardScaler` directly on the full dataset; for a predictive modeling pipeline, scaling should be fitted on training data only to avoid leakage.
- The categorical encoding output reflects the source data as submitted, including category variants visible in the notebook outputs.
- No predictive model or baseline metric is evaluated in this task.

## 7. Recommendation / Next Steps

1. Review and correct confirmed invalid values before modeling.
2. Decide on an explicit treatment for annual-income outliers.
3. Separate train/test data before fitting imputers, encoders, and scalers for a predictive workflow.
4. Preserve preprocessing in a reproducible sklearn pipeline.
5. Add a baseline model and evaluation metrics in the modeling stage.
6. Keep the original dataset or a documented dataset link available so the notebook can be reproduced from a fresh kernel.

## 8. References

- Pandas documentation — data loading, missing-value handling, duplicates, and categorical encoding.
- NumPy documentation — numerical operations.
- Matplotlib documentation — plotting.
- Seaborn documentation — statistical visualization.
- Scikit-learn documentation — `StandardScaler` and preprocessing utilities.
