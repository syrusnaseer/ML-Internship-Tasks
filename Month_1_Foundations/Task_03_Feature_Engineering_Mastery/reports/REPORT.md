# Task 03 — Feature Engineering Mastery — Report
Author: Mussa Khan (@musagithub1)  
Date: 2026-08-18

## 1. Executive Summary

- A 5,000-row synthetic employee-attrition dataset was used to demonstrate the Task 03 feature-engineering techniques on a domain different from the original real-estate brief.
- Domain-driven ratios, polynomial features, interactions, bins, and leakage-safe Department target encoding were created.
- Mutual Information, RFE, and L1 selection were applied using training data only.
- The engineered model achieved a ROC-AUC of approximately 0.742 versus 0.739 for the baseline; however, F1 decreased slightly, so the engineering changes did not produce a broad performance improvement.
- The result demonstrates why feature engineering must be evaluated empirically rather than assuming that more features always improve a model.

## 2. Business Problem & Framing

The adapted business problem is employee attrition prediction: identify employees at higher risk of leaving so a company can prioritize retention actions. The target is `Attrition`, where `1` means the employee left and `0` means the employee stayed.

The original Task 03 brief uses a real-estate price-regression scenario. This submission intentionally uses a different dataset, as requested, while preserving the feature-engineering learning objectives. Because the target is binary, classification metrics are used instead of MAE.

## 3. Data Overview

- Rows: 5,000
- Original columns: 29
- Target: `Attrition`
- Target distribution: 3,801 stayed and 1,199 left
- Intentional missing values: six columns contain missing observations
- Identifier `EmployeeID` was removed because it has no predictive business meaning.

The dataset is synthetic and is included under `data/` for reproducibility.

## 4. Methodology

### Preprocessing

Numerical missing values are imputed with the median. Categorical missing values are imputed with the most frequent category, followed by one-hot encoding. A fixed 80/20 stratified train/test split is used.

### Feature engineering

Examples include:

- `CareerStability`
- `RoleStability`
- `ManagerStability`
- `PromotionDelay`
- `IncomePerWorkingYear`
- `AgeSquared`
- `IncomeSquared`
- `DistanceSquared`
- `Overtime_Satisfaction`
- `Income_JobLevel`
- `AgeGroup`
- `DistanceGroup`
- `Department_TargetEncoded`

Target encoding is fitted only on the training data and then applied to the test data using the training mapping, preventing target leakage.

### Feature selection

Three requested approaches are demonstrated:

1. Mutual Information — ranks features by information about the target.
2. RFE — recursively removes weaker features using a logistic-regression estimator.
3. L1 regularization — drives weak coefficients to zero and identifies a sparse feature set.

The union of selected features is used for the final model.

## 5. Results & Key Visualizations

The baseline Logistic Regression produced approximately:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.780 | 0.609 | 0.233 | 0.337 | 0.739 |
| Engineered + Selection | 0.778 | 0.598 | 0.229 | 0.331 | 0.742 |

The final ROC-AUC increased slightly, but the classification threshold metrics declined. Therefore, the engineered feature set should not be described as a universal improvement. Further tuning, class-weighting, threshold optimization, or alternative models would be reasonable next steps.

Key visuals are stored in `figures/` and include target distribution, age/attrition, overtime/attrition, Mutual Information ranking, feature importance, and baseline-vs-final metrics.

## 6. Limitations & Risks

- The dataset is synthetic, so conclusions should not be interpreted as real-world HR evidence.
- Target encoding can leak target information if calculated before splitting; this implementation avoids that risk.
- The selected feature union is not guaranteed to be optimal for every classifier.
- Logistic Regression may not capture all non-linear relationships.
- Attrition classification is somewhat imbalanced, so accuracy alone is insufficient.

## 7. Recommendation / Next Steps

Test class-weighted models, tune the decision threshold for retention use cases, compare with gradient-boosted trees, and validate the engineered features using cross-validation. For production use, the feature definitions and preprocessing should be versioned and monitored for drift.

## 8. References

- scikit-learn documentation for preprocessing, feature selection, and classification metrics.
- pandas documentation for data manipulation and missing-value handling.
- The synthetic dataset was generated specifically for this educational task.
