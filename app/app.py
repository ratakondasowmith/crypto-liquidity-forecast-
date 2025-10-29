import streamlit as st
import joblib
import numpy as np

st.title("💧 Crypto Liquidity Predictor")

model = joblib.load("app/model.pkl")

st.sidebar.header("Input Features")
price_ma_7 = st.sidebar.number_input("Price MA (7 days)", value=0.0)
price_ma_30 = st.sidebar.number_input("Price MA (30 days)", value=0.0)
volatility_7 = st.sidebar.number_input("Volatility (7 days)", value=0.0)
volatility_30 = st.sidebar.number_input("Volatility (30 days)", value=0.0)
liquidity_ratio = st.sidebar.number_input("Liquidity Ratio", value=0.0)
liq_ratio_lag_1 = st.sidebar.number_input("Liquidity Lag (1 day)", value=0.0)
liq_ratio_diff_1 = st.sidebar.number_input("Liquidity Diff (1 day)", value=0.0)

if st.sidebar.button("Predict Liquidity"):
    features = np.array([[price_ma_7, price_ma_30, volatility_7, volatility_30,
                          liquidity_ratio, liq_ratio_lag_1, liq_ratio_diff_1]])
    prediction = model.predict(features)[0]
    st.metric("Predicted Liquidity", f"{prediction:.6f}")
