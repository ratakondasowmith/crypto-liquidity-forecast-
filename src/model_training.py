import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(df, model_path='app/model.pkl'):
    print("🔹 Training model...")

    FEATURES = [
        'price_ma_7','price_ma_30','volatility_7','volatility_30',
        'liquidity_ratio','liq_ratio_lag_1','liq_ratio_diff_1'
    ]
    X = df[FEATURES]
    y = df['target_liquidity']

    split = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }
    print("📊 Model Metrics:", metrics)
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")
    return metrics
