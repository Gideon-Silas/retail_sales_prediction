# Retail Sales Prediction

Predicting total sales for a product at a given Mart store, using historical product and store attributes. Built for the **DSN Bootcamp Qualification Hackathon 2026 (ML Track)**, a qualifying competition for the DSN AI Bootcamp.

A Gradient Boosting Regressor, trained on a log-transformed target and tuned with **Optuna (Bayesian optimization)**, achieved an **RMSE of ≈ 1123** on held-out validation data, outperforming a Random Forest baseline, manual tuning, and an exhaustive grid search.

## Dataset

DSN Mart operates a chain of stores across Nigeria, from small corner shops to flagship hypermarkets. The dataset contains historical product-store sales records with the following fields:

- 'product_code', 'product_weight_kg', 'fat_content', 'shelf_visibility', 'product_category', 'product_price'
- 'store_code', 'store_age_years', 'store_size', 'store_location_tier', 'store_format'
- 'total_sales' (target)

Source: [DSN Bootcamp Qualification Hackathon 2026 ML Track, Kaggle](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)

## Findings

- The raw target ('total_sales') was strongly right-skewed; log-transforming it ('log1p') and training on the log scale, then inverting predictions with 'expm1', improved model performance.
- 'product_category' initially appeared as 48 "unique" values due to inconsistent casing (e.g. 'Frozen Foods' vs 'FROZEN FOODS'), corrected down to 16 real categories before modeling.
- Missing 'product_weight_kg' and 'store_size' values were imputed using informed lookups (per-product weight averages, per-store-format size mode) rather than blind global defaults.
- **'store_format' and 'product_price' are the two dominant predictors**, together accounting for ~ 79% of the model's feature importance, far ahead of 'product_category', 'store_size', or 'store_age_years', all of which showed surprisingly weak individual signal.
- 'product_price' was confirmed to carry independent predictive value, not simply act as a proxy for 'product_category'. Price ranges overlap heavily across every category.
- Gradient Boosting outperformed Random Forest. Manual tuning and an exhaustive grid search (18 combinations) both failed to beat the original untuned Gradient Boosting configuration, but **Optuna's Bayesian optimization**, searching a much wider and continuous hyperparameter space across 50 trials with 5-fold cross-validation, found a genuinely better configuration, improving RMSE from 1127.73 to ** ≈ 1123**.

## Model Structure

**Pipeline:**
```
Raw input (product + store attributes)
        │
        ▼
Categorical encoding (6 fitted LabelEncoders: fat_content, product_category,
store_code, store_size, store_location_tier, store_format)
        │
        ▼
Feature vector (10 features, fixed order — see feature_cols.pkl)
        │
        ▼
StandardScaler (fitted on training data only)
        │
        ▼
Gradient Boosting Regressor
(Optuna-tuned: n_estimators, learning_rate, max_depth, subsample,
min_samples_split, min_samples_leaf — see best_params in notebook)
        │
        ▼
Prediction on log(1 + total_sales) scale
        │
        ▼
np.expm1() → final predicted total_sales
```

**Why the log transform:** 'total_sales' is strongly right-skewed in the training data. Training directly on the log-transformed target and inverting predictions at the end produced a lower RMSE than training on raw sales values.

**Why Optuna over grid search:** grid search only tests fixed, manually chosen values across a small combination count (18 in this case) and plateaued at RMSE 1131. Optuna's Bayesian optimization searches a wider, continuous space and uses results from earlier trials to intelligently guide later ones, finding a better configuration (RMSE ≈1123) in 50 trials.

## Project Structure

```
DSN_Sales_Prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── DSN_Mart_Sales_Prediction.ipynb
└── models/
    ├── gb_model.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    └── feature_cols.pkl
```

## How to Run

1. Clone the repository
   ```
   git clone https://github.com/Gideon-Silas/DSN_Mart_Sales_Prediction.git
   cd DSN_Mart_Sales_Prediction
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Launch the app
   ```
   streamlit run app.py
   ```

## Tools

- Python
- pandas
- scikit-learn
- Streamlit
- matplotlib
- seaborn

## Full write-up 

Full analysis write-up: [Medium] (https://medium.com/@Just_Gideons/dsn-2026-ai-bootcamp-hackathon-project-participation-predicting-retail-sales-at-dsn-mart-7d4f03c3e317)

## Author 

Connect with me on: [LinkedIn] (https://www.linkedin.com/in/gideonsilas), and [Medium] (https://medium.com/@Just_Gideons/dsn-2026-ai-bootcamp-hackathon-project-participation-predicting-retail-sales-at-dsn-mart-7d4f03c3e317)
