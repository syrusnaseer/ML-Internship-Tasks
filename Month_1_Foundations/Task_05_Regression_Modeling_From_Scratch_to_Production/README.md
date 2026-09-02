# 📊 Statistical Analysis and Hypothesis Testing for Medical Insurance Costs

## Project Overview

This project was completed as **Task 05** during my **Machine Learning Internship**.

The objective is to apply statistical analysis and hypothesis testing techniques to the **Medical Cost Personal / Insurance Dataset** and investigate factors associated with medical insurance charges.

The notebook focuses on:
- Dataset inspection and cleaning
- Descriptive statistics
- Smoker vs non-smoker charge comparison
- Assumption checking
- Welch's t-test
- Mann-Whitney U test
- Cohen's d
- 95% confidence interval
- BMI and medical-charge relationships
- Pearson and Spearman correlation
- Regional charge comparison
- Welch's ANOVA
- Kruskal-Wallis test
- Chi-square test of independence
- Cramér's V
- Bonferroni and Holm multiple-comparison corrections

## Dataset

**Dataset:** Medical Cost Personal / Insurance Dataset

The main outcome variable is:

`charges`

After duplicate removal, the cleaned dataset contains **1,337 records**.

## Main Findings

- Smokers had substantially higher medical charges than non-smokers.
- The smoker/non-smoker difference had an extremely large standardized effect.
- BMI had a weak overall association with charges.
- Among smokers, BMI had a much stronger positive association with charges.
- Regional differences in charges were not statistically significant at the 5% level.
- Region and smoking status did not show a statistically significant association.
- No regional pairwise comparison remained significant after multiple-comparison correction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Jupyter Notebook / Google Colab

## Repository Structure

```text
ml-internship-task-5/
├── ML_TASK_5.ipynb
├── insurance.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AAliAhmedd/ml-internship-task-5.git
cd ml-internship-task-5
```

### 2. Create a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `ML_TASK_5.ipynb` and run the cells from top to bottom.

## Statistical Interpretation

The dataset is observational. Therefore, statistically significant associations should not automatically be interpreted as causal relationships. Results should be considered together with effect sizes, confidence intervals, assumptions, and practical business relevance.
