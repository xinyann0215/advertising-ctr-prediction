# Advertising Click-Through Rate Prediction

## Overview
This project predicts whether users will click on online advertisements using machine learning models and behavioral features.

The analysis includes exploratory data analysis, feature preprocessing, model training, model comparison, and feature importance analysis.

## Dataset
The dataset contains **404,290 advertising impressions** with user, device, site, app, and advertisement-related features.

## Tools & Technologies
- Python
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost

## Modeling
Two machine learning models were evaluated:

- Logistic Regression
- XGBoost

Categorical features were transformed using one-hot encoding, and the dataset was split into training and testing sets.

## Results

| Model | ROC-AUC | Log Loss |
|---|---:|---:|
| Logistic Regression | 0.7059 | 0.4155 |
| XGBoost | **0.7193** | **0.4102** |

XGBoost achieved the strongest performance, improving ROC-AUC while reducing log loss compared with the Logistic Regression baseline.

## Key Findings
- XGBoost outperformed the Logistic Regression baseline.
- Nonlinear relationships and feature interactions improved CTR prediction.
- Feature importance analysis was used to identify influential predictors of advertisement clicks.

## Project Workflow
1. Data cleaning
2. Exploratory data analysis
3. Feature engineering
4. One-hot encoding
5. Logistic Regression baseline
6. XGBoost modeling
7. Model evaluation
8. Feature importance analysis

## Repository
The complete analysis and model development process can be found in:

`advertising_ctr_prediction.ipynb`
