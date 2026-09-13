"""
探索式資料分析 (EDA)：看看降雨跟哪些特徵有關，順便驗證前面對「降雨前兆」的假設
(氣壓下降、濕度上升、夏季午後...) 在資料裡是不是真的成立。

用法:
    python scripts/eda.py
輸出圖表會存在 data/processed/plots/ 底下，備審資料可以直接引用。
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from font_utils import setup_cjk_font

DATA_PATH = "data/processed/weather_hourly.csv"
PLOT_DIR = Path("data/processed/plots")


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df


def plot_rain_rate_by_hour(df):
    rate = df.groupby("hour")["rained"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rate.index, rate.values, color="#3b6ea5")
    ax.set_xlabel("小時")
    ax.set_ylabel("降雨機率")
    ax.set_title("各時段降雨機率")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rain_rate_by_hour.png", dpi=150)
    plt.close(fig)


def plot_rain_rate_by_month(df):
    rate = df.groupby("month")["rained"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rate.index, rate.values, color="#3b6ea5")
    ax.set_xlabel("月份")
    ax.set_ylabel("降雨機率")
    ax.set_title("各月份降雨機率")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rain_rate_by_month.png", dpi=150)
    plt.close(fig)


def plot_pressure_change_vs_rain(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df["pressure_change_3h"], bins=12)
    rate = df.groupby(bins, observed=True)["rained"].mean()
    ax.plot(range(len(rate)), rate.values, marker="o", color="#c1543c")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{iv.mid:.1f}" for iv in rate.index], rotation=45, ha="right")
    ax.set_xlabel("過去3小時氣壓變化 (hPa)")
    ax.set_ylabel("降雨機率")
    ax.set_title("氣壓變化率 vs 降雨機率（驗證「氣壓驟降是降雨前兆」的假設）")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pressure_change_vs_rain.png", dpi=150)
    plt.close(fig)


def plot_humidity_vs_rain(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df["humidity_pct"], bins=10)
    rate = df.groupby(bins, observed=True)["rained"].mean()
    ax.plot(range(len(rate)), rate.values, marker="o", color="#3f8f5f")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{iv.mid:.0f}" for iv in rate.index], rotation=45, ha="right")
    ax.set_xlabel("相對濕度 (%)")
    ax.set_ylabel("降雨機率")
    ax.set_title("濕度 vs 降雨機率")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "humidity_vs_rain.png", dpi=150)
    plt.close(fig)


def print_summary(df):
    print("=== 資料摘要 ===")
    print(f"總筆數: {len(df)}")
    print(f"時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"整體降雨比例: {df['rained'].mean():.2%}")
    print()
    print("=== 各特徵與降雨的相關係數 ===")
    candidate_cols = ["pressure_hpa", "pressure_change_3h", "humidity_pct",
                       "temperature_c", "wind_speed_ms", "rained"]
    cols = [c for c in candidate_cols if c in df.columns]
    corr = df[cols].corr()["rained"].sort_values()
    print(corr)


if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()

    df = load_data()
    print_summary(df)
    plot_rain_rate_by_hour(df)
    plot_rain_rate_by_month(df)
    plot_pressure_change_vs_rain(df)
    plot_humidity_vs_rain(df)
    print(f"\n圖表已輸出到 {PLOT_DIR}/")
