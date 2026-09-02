# Task 06 — Classification Algorithms Comparison — Report
Author: Mussa Khan (@musagithub1)
Date: 2026-09-02

## 1. Executive Summary
- The Bank Marketing dataset was used to predict term-deposit subscription (`y`).
- Six classification algorithms were compared with 3-fold cross-validation.
- Random Forest achieved the highest mean ROC-AUC (0.927).
- Logistic Regression achieved the highest mean F1-score (0.551) among the six models in cross-validation.
- Threshold optimization improved the best model's test F1-score to 0.609 at a threshold of 0.25.

## 2. Business Problem & Framing
The objective is to identify customers who are likely to subscribe to a term deposit. Correctly identifying likely subscribers can help focus marketing calls on promising customers while reducing unnecessary calls. Because the target is imbalanced, Precision, Recall, F1-score, and ROC-AUC are more informative than accuracy alone.

## 3. Data Overview
The dataset contains 45,211 customer records and 17 columns. It includes demographic, financial, contact, and previous-campaign information. The target variable `y` contains `yes` and `no` subscription outcomes.

The target is imbalanced: approximately 88% of observations are `no` and 12% are `yes`.

Preprocessing:
- Numerical features were standardized with `StandardScaler`.
- Categorical features were one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`.
- The train/test split was stratified to preserve class proportions.
- Class weighting was used for Logistic Regression, Decision Tree, Random Forest, and SVM.

## 4. Methodology
Six models were evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
5. KNN
6. Gradient Boosting

A 3-fold `StratifiedKFold` cross-validation strategy was used. Metrics were Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Cross-validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.902 | 0.677 | 0.304 | 0.419 | 0.927 |
| Gradient Boosting | 0.906 | 0.658 | 0.409 | 0.504 | 0.925 |
| SVM | 0.837 | 0.404 | 0.836 | 0.545 | 0.910 |
| Logistic Regression | 0.844 | 0.415 | 0.818 | 0.551 | 0.910 |
| KNN | 0.894 | 0.582 | 0.327 | 0.419 | 0.830 |
| Decision Tree | 0.876 | 0.467 | 0.445 | 0.456 | 0.689 |

## 5. Results & Key Visualizations
The Random Forest was selected as the best model by ROC-AUC.

On the held-out test set, the Random Forest achieved:
- Accuracy: 0.902
- Precision for `Yes`: 0.68
- Recall for `Yes`: 0.33
- F1-score for `Yes`: 0.45

The majority-class baseline achieved about 0.88 accuracy while predicting no positive cases, demonstrating why accuracy alone is misleading for this problem.

The notebook includes and saves:
- Target distribution
- Subscription by job
- Subscription by education
- Balance distribution
- Model comparison
- Confusion matrix
- ROC curve
- Precision-Recall curve
- F1-score vs threshold

Threshold testing found **0.25** as the best tested threshold for F1, increasing the test F1-score to approximately **0.609**.

For statistical testing, the Wilcoxon signed-rank test compared the two highest-F1 cross-validation models (Logistic Regression and SVM). The p-value was **0.25**, so the observed difference was not statistically significant at the 0.05 level.

## 6. Limitations & Risks
- The comparison used 3-fold CV instead of 5-fold to keep runtime practical on the full dataset.
- SMOTE was not used in the final comparison because it made the six-model cross-validation substantially slower; class weighting was used where supported.
- The `duration` feature is highly predictive but may not be available before or during a call, depending on the intended deployment timing. This creates a potential target-leakage/business-timing concern.
- Threshold selection was optimized for F1, but a production system should optimize the threshold using explicit business costs for false positives and false negatives.
- The statistical test used only three CV folds, so its evidence is limited.

## 7. Recommendation / Next Steps
Random Forest is the recommended model when ROC-AUC is the main selection criterion. However, the business should choose the operating threshold using the actual cost of unnecessary calls versus missed subscribers. The optimized threshold of 0.25 is a useful experiment because it substantially improves F1, but it should not be treated as the final production threshold without cost-based validation.

Next steps:
- Evaluate models without potentially unavailable post-contact features such as `duration`.
- Perform cost-sensitive threshold optimization.
- Tune hyperparameters for the strongest candidates.
- Validate the final model on a temporally separated holdout set if campaign timing is available.

## 8. References
- UCI Machine Learning Repository — Bank Marketing dataset.
- scikit-learn documentation for classification, preprocessing, model selection, and metrics.
