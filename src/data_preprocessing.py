import pandas as pd

def preprocess_data(path):
    print("🔹 Loading data...")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # Detect or create date column
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        df['date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

    df = df.sort_values('date').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    print("✅ Data preprocessing completed.")
    return df
