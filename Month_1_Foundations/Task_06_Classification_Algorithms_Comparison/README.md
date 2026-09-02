# Task 06 — Classification Algorithms Comparison

## Overview
This task compares six classification algorithms on the **Bank Marketing** dataset to predict whether a customer subscribes to a term deposit.

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

## Workflow
1. Load and inspect the dataset.
2. Check missing values, duplicates, and class distribution.
3. Explore the data with visualizations.
4. Separate features and target.
5. Scale numerical features and one-hot encode categorical features.
6. Split data using a stratified 80/20 train-test split.
7. Compare six classifiers using 3-fold stratified cross-validation.
8. Evaluate Accuracy, Precision, Recall, F1-score, and ROC-AUC.
9. Compare against a majority-class baseline.
10. Evaluate the best ROC-AUC model on the test set.
11. Analyze the confusion matrix, ROC curve, and Precision-Recall curve.
12. Optimize the classification threshold for F1-score.
13. Use a Wilcoxon signed-rank test to compare the top two models by CV F1-score.

## Key Results
- Best model by cross-validation ROC-AUC: **Random Forest**
- Cross-validation ROC-AUC: **0.927**
- Test Accuracy: **0.902**
- Test Precision for Yes: **0.68**
- Test Recall for Yes: **0.33**
- Test F1-score for Yes: **0.45**
- Best tested threshold for F1: **0.25**
- F1-score at optimized threshold: **0.609**

## Repository Structure
```text
Task_06_Classification_Algorithms_Comparison/
├── README.md
├── notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── modeling.py
├── figures/
├── reports/
│   └── REPORT.md
├── requirements.txt
├── .gitignore
└── bank-full.csv
```

## Running
Install dependencies:

```bash
pip install -r requirements.txt
```

Then open `notebook.ipynb` and run it from top to bottom.

> The dataset is included because it is only a few MB and is required for the notebook to run locally. No credentials or API keys are used.
