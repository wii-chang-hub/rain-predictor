"""
用中央氣象署開放資料平台 (CWA Open Data) 抓新竹測站的歷史逐時觀測資料。

使用前：
1. 到 https://opendata.cwa.gov.tw/ 註冊會員、取得 API 授權碼
2. 執行前先設定環境變數：
       export CWA_API_KEY="你的金鑰"

注意：
- CWA Open Data API (opendata.cwa.gov.tw/api) 主要提供「現在天氣觀測」跟「未來預報」，
  單次呼叫抓到的是「當下」的資料，不是任意過去區間的歷史資料。
- 如果要抓「過去半年~一年」的逐時歷史資料(訓練模型用)，建議改用氣候資料服務系統 CODiS
  (https://codis.cwa.gov.tw/StationData) 的「單站資料下載」功能，選新竹站、指定日期區間，
  網站上可以直接匯出CSV，不一定要寫程式抓。
- 這支腳本先示範「即時觀測」的抓法，讓你申請到金鑰後可以馬上測試連線是否成功；
  抓歷史訓練資料的部分，把從CODiS下載下來的CSV放進 data/raw/ 資料夾，
  並依照下面 `normalize_codis_csv()` 的欄位對應調整一下即可接上後面的流程。
"""
import os
import sys
import requests
import pandas as pd
from pathlib import Path

API_KEY = os.environ.get("CWA_API_KEY")
# O-A0001-001: 現在天氣觀測報告-無地面雷達回波(逐時自動氣象站資料)
DATASET_ID = "O-A0001-001"
BASE_URL = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{DATASET_ID}"
STATION_NAME = "新竹"  # 依實際測站名稱調整，例如 "新竹" 或 "竹北"


def fetch_current_observation():
    if not API_KEY:
        print("請先設定環境變數 CWA_API_KEY，參考本檔案開頭的說明。", file=sys.stderr)
        sys.exit(1)

    params = {
        "Authorization": API_KEY,
        "StationName": STATION_NAME,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    out_path = Path("data/raw/current_observation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"已抓取即時觀測資料 -> {out_path}")
    return data


def normalize_codis_csv(codis_csv_path: str, out_path: str = "data/processed/weather_hourly.csv"):
    """
    把從 CODiS 網站下載下來的單站逐時資料 CSV，轉成跟 generate_sample_data.py
    輸出格式一致的欄位，讓後面 eda.py / train_model.py 不用改就能吃。

    CODiS 下載的欄位名稱可能因資料集版本略有不同，下載後先用
    `head data/raw/你的檔名.csv` 看一下實際欄位名稱，再調整下面的 rename 對照表。
    """
    df = pd.read_csv(codis_csv_path)

    # 下面這行的 key 是「CODiS常見欄位名稱」，value 是我們統一使用的欄位名稱，
    # 請對照實際下載下來的檔案欄位名稱調整。
    rename_map = {
        "觀測時間(hour)": "datetime",
        "測站氣壓(hPa)": "pressure_hpa",
        "氣溫(℃)": "temperature_c",
        "相對溼度(%)": "humidity_pct",
        "風速(m/s)": "wind_speed_ms",
        "降水量(mm)": "rain_mm",
    }
    df = df.rename(columns=rename_map)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["pressure_change_3h"] = df["pressure_hpa"].diff(3).fillna(0)
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0)
    df["rained"] = (df["rain_mm"] > 0.5).astype(int)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"已整理成訓練用格式 -> {out} ({len(df)} 筆)")


if __name__ == "__main__":
    fetch_current_observation()
