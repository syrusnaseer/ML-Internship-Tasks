# Task 4 — Feature Engineering for House Price Prediction

## Project Overview

This project was completed as **Task 04** during the Machine Learning Internship.

The notebook applies feature engineering techniques to the **Ames Housing Dataset** for a house-price prediction problem. The work focuses on understanding the dataset, handling missing values using domain knowledge, creating meaningful real-estate features, binning numerical variables, generating polynomial/interaction features, and applying leakage-aware target encoding.

> **Current scope:** The notebook is completed through target encoding. Feature selection and regression-model evaluation are not included in the current notebook.

---

## Business Problem

A real-estate platform needs better house-price estimates. Raw housing variables do not always directly represent useful business concepts, so feature engineering is used to create more meaningful predictors.

Examples include:

- Total house square footage
- Total bathrooms
- House age
- Remodel age
- Garage/basement/fireplace/pool indicators
- Living area per room
- Garage area per car
- Quality × condition
- House-size and quality categories
- Polynomial and interaction features
- Target-encoded categorical features

---

## Dataset

**Dataset:** Ames Housing Dataset

**Target variable:** `SalePrice`

**Notebook dataset filename:** `housing (1).csv`

The notebook expects the CSV file to be in the **same directory as the notebook**.

### Important

The CSV dataset was **not included in the uploaded notebook file** used to build this submission package. Before pushing the project to GitHub, place your original `housing (1).csv` in this `Task 4` folder.

---

## Work Completed

### 1. Dataset inspection

The notebook performs:

- Dataset loading
- Preview
- Shape inspection
- Data-type inspection
- Column inspection
- Numerical/categorical column identification
- Duplicate-row checking
- Missing-value analysis

The unnecessary `Unnamed: 0` column is removed.

### 2. Missing-value treatment

The notebook distinguishes structural missingness from genuinely missing information.

Categorical structural-missing values are filled with `None`, including features such as:

- `Pool QC`
- `Misc Feature`
- `Alley`
- `Fence`
- `Fireplace Qu`
- `Garage Type`
- `Garage Finish`
- `Garage Qual`
- `Garage Cond`
- Basement categorical features

Numerical absence-related features are filled with `0`, including:

- `Mas Vnr Area`
- Basement square-footage/bathroom fields
- `Garage Yr Blt`
- `Garage Cars`
- `Garage Area`

`Lot Frontage` is imputed using neighborhood medians followed by the overall median.

The missing `Electrical` value is filled using the mode.

### 3. Domain-driven feature engineering

The notebook creates:

- `Total Bathrooms`
- `Total SF`
- `House Age`
- `Remodel Age`
- `Total Porch SF`
- `Total Outdoor SF`

### 4. Binary features

The following indicators are created:

- `Has Garage`
- `Has Basement`
- `Has Fireplace`
- `Has Pool`

These use `1` for presence and `0` for absence.

### 5. Ratio and interaction features

Created features:

- `Living Area Per Room`
- `Garage Area Per Car`
- `Quality x Condition`

The garage ratio safely avoids division by zero by replacing zero garage capacity with one for the calculation.

### 6. Binning

`pd.cut()` is used to create interpretable categories:

**House Age Category**
- New
- Relatively New
- Old
- Very Old

**House Size**
- Small
- Medium
- Large
- Luxury

**Quality Category**
- Low
- Average
- Good
- Excellent

### 7. Polynomial features

`PolynomialFeatures(degree=2, include_bias=False)` is applied to:

- `Overall Qual`
- `Gr Liv Area`
- `Total SF`

This generates squared terms and pairwise interaction terms.

### 8. Leakage-aware target encoding

Target encoding is applied to:

- `Neighborhood`
- `Exterior 1st`
- `Exterior 2nd`

The dataset is first split into training and test sets using:

- `test_size=0.20`
- `random_state=42`

The encoder is fitted only on training data and then used to transform the test data. A smoothing value of `10` is used.

This prevents test-set target values from being used to learn the encoding.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Category Encoders
- Jupyter Notebook / Google Colab

---

## Repository Structure

```text
ML-Internship-Tasks/
└── Task 4/
    ├── Task_4_ML.ipynb
    ├── housing (1).csv
    ├── README.md
    └── requirements.txt
```

---

## How to Run

### Google Colab

1. Upload `Task_4_ML.ipynb`.
2. Upload `housing (1).csv`.
3. Make sure the CSV filename matches the path used in the notebook.
4. Run the notebook cells from top to bottom.

### Local Jupyter

Create a virtual environment, activate it, and install the requirements:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `Task_4_ML.ipynb` and run the cells from top to bottom.

---

## Learning Outcomes

Through this task, the following concepts were practiced:

- Inspecting a real-world dataset
- Identifying structural missingness
- Applying domain-based missing-value treatment
- Creating meaningful derived features
- Creating binary indicators
- Creating ratios and interaction features
- Binning numerical variables
- Creating polynomial features
- Understanding high-cardinality categorical variables
- Applying target encoding
- Preventing target leakage during target encoding

---

## Current Scope Limitations

The current notebook does not include:

- Mutual Information feature selection
- Recursive Feature Elimination
- L1/Lasso feature selection
- Baseline regression model
- Final regression model
- MAE/RMSE/R² model evaluation

These can be added in a future phase if the task scope is expanded.
