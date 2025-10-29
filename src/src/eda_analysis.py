import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eda_report(df):
    sns.set(style="whitegrid")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10,7))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    if 'date' in df.columns and 'liquidity' in df.columns:
        plt.figure(figsize=(12,4))
        plt.plot(df['date'], df['liquidity'], color='blue')
        plt.title("Liquidity Over Time")
        plt.show()

    price_col = next((c for c in df.columns if 'price' in c.lower()), None)
    if price_col:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=df[price_col], y=df['liquidity'], alpha=0.7)
        plt.title("Price vs Liquidity")
        plt.show()
