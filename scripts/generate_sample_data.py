"""
在還沒申請到中央氣象署 API 金鑰之前，先產生一份「結構跟真實資料一模一樣」的
模擬氣象資料，讓後面的 EDA / 訓練流程可以先跑起來、先確認邏輯沒問題。

拿到真的資料之後，只要用 fetch_data.py 產生的 data/raw/weather_hourly.csv
蓋掉這裡產生的檔案，後面的腳本完全不用改。

模擬的邏輯（盡量貼近新竹的真實氣候型態，但終究是假資料，正式報告不能拿這份當結果）：
- 夏季（6-9月）午後（13-18點）下雨機率明顯較高，模擬午後雷陣雨
- 氣壓下降時，之後1-2小時降雨機率提高，模擬「氣壓驟降是降雨前兆」
- 濕度越高，降雨機率越高
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

def generate(start="2025-01-01", periods_hours=24 * 365, out_path="data/processed/weather_hourly.csv"):
    idx = pd.date_range(start=start, periods=periods_hours, freq="h")
    df = pd.DataFrame({"datetime": idx})
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    is_summer = df["month"].between(6, 9)
    is_afternoon = df["hour"].between(13, 18)

    # 基礎氣壓 + 隨機遊走(模擬鋒面/低壓系統經過) + 日夜微幅波動
    base_pressure = 1013 + 3 * np.sin(2 * np.pi * df["month"] / 12)
    walk = RNG.normal(0, 0.4, size=len(df)).cumsum()
    walk = walk - pd.Series(walk).rolling(24 * 14, min_periods=1).mean().to_numpy()  # 避免無限飄移
    df["pressure_hpa"] = base_pressure + walk + RNG.normal(0, 0.3, size=len(df))

    # 氣壓變化率(過去3小時)：驟降代表可能有天氣系統接近
    df["pressure_change_3h"] = df["pressure_hpa"].diff(3).fillna(0)

    # 濕度：夏季午後、氣壓下降時濕度較高
    humidity_base = 70 + 10 * is_summer.astype(int) + 8 * is_afternoon.astype(int)
    humidity_pressure_effect = np.clip(-df["pressure_change_3h"] * 4, -10, 20)
    df["humidity_pct"] = np.clip(
        humidity_base + humidity_pressure_effect + RNG.normal(0, 5, size=len(df)), 30, 100
    )

    # 溫度：簡單季節+日夜循環
    seasonal = 23 + 7 * np.sin(2 * np.pi * (df["month"] - 3) / 12)
    diurnal = 4 * np.sin(2 * np.pi * (df["hour"] - 9) / 24)
    df["temperature_c"] = seasonal + diurnal + RNG.normal(0, 1.2, size=len(df))

    df["wind_speed_ms"] = np.clip(RNG.normal(2.5, 1.2, size=len(df)), 0, None)

    # 降雨機率：綜合以上特徵的邏輯式組合(這是「生成假資料」的公式，不是要訓練的模型本身)
    logit = (
        -3.5
        + 2.2 * is_summer.astype(int)
        + 1.3 * is_afternoon.astype(int)
        + 0.06 * (df["humidity_pct"] - 70)
        + -0.9 * df["pressure_change_3h"]
        + 0.15 * df["wind_speed_ms"]
    )
    prob_rain = 1 / (1 + np.exp(-logit))
    rained = RNG.binomial(1, prob_rain)
    df["rain_mm"] = np.where(rained == 1, RNG.gamma(2.0, 2.5, size=len(df)), 0.0)
    df["rained"] = (df["rain_mm"] > 0.5).astype(int)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"已產生模擬資料 {len(df)} 筆 -> {out}")
    print(f"降雨比例: {df['rained'].mean():.2%}")


if __name__ == "__main__":
    generate()
