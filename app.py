import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="DSN Mart Sales Prediction",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# Model + artifact loading
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "gb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_cols.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, encoders, feature_cols


load_error = None
try:
    model, scaler, encoders, feature_cols = load_artifacts()
except Exception as e:
    load_error = f"{type(e).__name__}: {e}"
    model = scaler = encoders = feature_cols = None

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🛒 DSN Mart Sales Prediction")
st.markdown(
    "Predict **total sales** for a product at a given store, based on "
    "product and store attributes. Built for the DSN Bootcamp Qualification "
    "Hackathon 2026 (ML Track)."
)

if load_error:
    st.error(
        f"Couldn't load required model files: {load_error}\n\n"
        "Make sure gb_model.pkl, scaler.pkl, encoders.pkl, and "
        "feature_cols.pkl all sit inside a `models/` folder next to this app."
    )
    st.stop()

st.divider()
st.subheader("Product & Store Details")

col1, col2 = st.columns(2)

with col1:
    product_weight_kg = st.number_input(
        "Product weight (kg)", min_value=0.0, max_value=50.0, value=12.5, step=0.1
    )
    shelf_visibility = st.number_input(
        "Shelf visibility (0-1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01
    )
    product_price = st.number_input(
        "Product price", min_value=0.0, max_value=1000.0, value=140.0, step=1.0
    )
    store_age_years = st.slider("Store age (years)", 0, 60, 25)

with col2:
    fat_content = st.selectbox("Fat content", sorted(encoders["fat_content"].classes_))
    product_category = st.selectbox(
        "Product category", sorted(encoders["product_category"].classes_)
    )
    store_size = st.selectbox("Store size", sorted(encoders["store_size"].classes_))
    store_location_tier = st.selectbox(
        "Store location tier", sorted(encoders["store_location_tier"].classes_)
    )
    store_format = st.selectbox("Store format", sorted(encoders["store_format"].classes_))

store_code = st.selectbox("Store code", sorted(encoders["store_code"].classes_))

predict_btn = st.button("Predict total sales →", type="primary")

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if predict_btn:
    try:
        input_dict = {
            "product_weight_kg": product_weight_kg,
            "shelf_visibility": shelf_visibility,
            "product_price": product_price,
            "store_age_years": store_age_years,
            "fat_content_encoded": encoders["fat_content"].transform([fat_content])[0],
            "product_category_encoded": encoders["product_category"].transform(
                [product_category]
            )[0],
            "store_code_encoded": encoders["store_code"].transform([store_code])[0],
            "store_size_encoded": encoders["store_size"].transform([store_size])[0],
            "store_location_tier_encoded": encoders["store_location_tier"].transform(
                [store_location_tier]
            )[0],
            "store_format_encoded": encoders["store_format"].transform([store_format])[0],
        }

        input_df = pd.DataFrame([input_dict])[feature_cols]
        scaled_input = scaler.transform(input_df)

        pred_log = model.predict(scaled_input)[0]
        pred_sales = np.expm1(pred_log)

        st.divider()
        st.metric("Predicted Total Sales", f"{pred_sales:,.2f}")
        st.caption(
            "Prediction made using an Optuna-tuned Gradient Boosting Regressor "
            "(RMSE ≈1123) trained on log-transformed sales. Store format and "
            "product price are the two strongest drivers of this prediction."
        )

    except Exception as e:
        st.error(
            "The model didn't accept these inputs — something about the "
            "feature values or encoding doesn't match what it was trained on.\n\n"
            f"Error: {e}"
        )

st.divider()
st.caption(
    "DSN Mart Sales Prediction · Optuna-tuned Gradient Boosting Regressor · "
    "Dataset: DSN Bootcamp Qualification Hackathon 2026 ML Track"
)
