# Advertising Click-Through Rate Prediction

Predicting whether a user will click on a display ad, using the [Avazu CTR Prediction dataset](https://www.kaggle.com/c/avazu-ctr-prediction) (404,290 impressions).

## Why this is harder than a standard classification problem

Two things make CTR prediction different from a generic tabular classification task, and this project is
built around both of them:

1. **Extreme class imbalance.** Click rate is far below 50%, so a naive model can look "accurate" while
   never actually ranking likely clicks correctly. This project uses `class_weight` / `scale_pos_weight`
   during training and reports **PR-AUC** alongside ROC-AUC, since PR-AUC is far more sensitive to
   imbalanced-class performance.
2. **High-cardinality identifier features.** Fields like `site_id`, `app_id`, `device_id`, and `device_ip`
   can have tens of thousands of unique values. One-hot encoding them is impractical; this project uses
   **frequency encoding** (fit on the training set only, to avoid leakage) so this signal isn't thrown away.

## Approach

- Exploratory analysis of CTR by banner position and hour of day
- Low-cardinality categorical features → one-hot encoding
- High-cardinality identifier features → frequency encoding
- Baseline: Logistic Regression with `class_weight="balanced"`
- Main model: XGBoost with `scale_pos_weight` set from the training class balance
- A small randomized hyperparameter search (3-fold CV on ROC-AUC) to tune the XGBoost model
- Evaluation on a held-out test set using ROC-AUC, PR-AUC, and Log Loss
- Feature importance analysis to identify which attributes are most predictive of clicks

## Results

| Model | ROC-AUC | PR-AUC | Log Loss |
|---|---:|---:|---:|
| Logistic Regression (balanced) | 0.670 | 0.296 | 0.645 |
| XGBoost (default params, imbalance-aware) | 0.724 | 0.349 | 0.592 |
| XGBoost (tuned) | 0.727 | 0.349 | 0.578 |

XGBoost clearly outperforms the Logistic Regression baseline. The PR-AUC of 0.349 is roughly 2x the
base click rate (~17%), indicating the model meaningfully ranks likely clicks above random — a more
informative signal than ROC-AUC alone under this level of class imbalance. Hyperparameter tuning produced
only a small further improvement, suggesting the model is close to the performance ceiling for the current
feature set rather than being limited by suboptimal hyperparameters.

## Repo structure

```
.
├── advertising_ctr_prediction.ipynb   # main notebook: EDA, feature engineering, modeling, evaluation
├── data/                              # not committed — put filtered_train.csv here (see Data below)
├── requirements.txt
└── README.md
```

## Data

This project uses a filtered subset of the [Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
dataset from Kaggle. Download `train.csv` from Kaggle, place it (or your filtered subset) at
`data/filtered_train.csv`, and it will be picked up by the `DATA_PATH` variable in the notebook.
The raw data is not committed to this repo due to size and Kaggle's terms of use.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook advertising_ctr_prediction.ipynb
```

## Key takeaway

The top-ranked features were not the newly added high-cardinality identifiers (`site_id`, `app_id`,
`device_id`), but rather Avazu's anonymized contextual features `C18` and `C16` (likely ad position/size
related), along with device type and browser category (`C1`). This suggests that, in this dataset, **the
context and environment in which an ad is shown** is a stronger predictor of clicks than *which specific
site or app* it's shown on. The frequency-encoded `site_id`/`app_id`/`site_domain` features still ranked
within the top 20, but with modest importance — likely because a single frequency value doesn't fully
capture each site's unique click propensity, especially for sites that are relatively rare in the training
data.

A natural next step would be target encoding (rather than frequency encoding) for `site_id`/`app_id` to
better capture per-site click propensity, and interaction features such as `site_id × device_type` to
recover more predictive signal from the identifier fields.
