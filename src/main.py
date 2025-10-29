import os
from data_preprocessing import preprocess_data
from feature_engineering import create_features
from model_training import train_model
from eda_analysis import eda_report

def main():
    data_path = "data/coin_gecko_2022-03-16.csv"
    model_path = "app/model.pkl"

    if not os.path.exists(data_path):
        print("❌ Dataset not found! Please add it under /data/")
        return

    df_clean = preprocess_data(data_path)
    df_feat = create_features(df_clean, forecast_horizon=1)
    eda_report(df_feat)
    metrics = train_model(df_feat, model_path)

    print("\n✅ Project complete! Model and visuals generated successfully.")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
