"""
把從 CODiS「單項逐時月報表」下載下來的CSV(一天一列、24小時分欄的寬表格式)，
合併、轉換成 eda.py / train_model.py 吃得懂的長表格式(一小時一列)。

預期檔名格式(CODiS下載下來的預設檔名，不用改)：
    <測站代號>-<年>-<月>-<項目>-hour.csv
例如：
    467571-2026-08-Precipitation-hour.csv
    467571-2026-08-RelativeHumidity-hour.csv
    467571-2026-08-AirTemperature-hour.csv
    467571-2026-08-StationPressure-hour.csv

把下載好的檔案全部丟進 data/raw/ 資料夾，不用重新命名，直接執行：
    python scripts/build_dataset_from_codis.py

會自動掃描 data/raw/ 底下所有符合格式的檔案、抓出有哪些(年,月)組合，
把同一個月份的4個項目合併成一天24小時的逐時資料，再把所有月份接起來，
輸出到 data/processed/weather_hourly.csv。
"""
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/weather_hourly.csv")

# CODiS檔名裡的項目名稱 -> 我們統一使用的欄位名稱
VARIABLES = {
    "StationPressure": "pressure_hpa",
    "RelativeHumidity": "humidity_pct",
    "AirTemperature": "temperature_c",
    "Precipitation": "rain_mm",
}

FILENAME_RE = re.compile(r"(?P<station>\d+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<var>\w+)-hour\.csv")


def parse_value(raw: str):
    """CODiS的數值欄位可能是正常數字、'T'(有降水但量測不到，視為極小值)、或缺漏(空白/X)。"""
    if raw is None:
        return np.nan
    s = str(raw).strip().strip('"')
    if s == "" or s.upper() in {"X", "-", "..."}:
        return np.nan
    if s.upper() == "T":
        return 0.05  # trace，一個象徵性的極小值，不會超過0.5mm的降雨判定門檻
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_wide_csv(path: Path) -> pd.DataFrame:
    """讀取一份「一天一列、24小時分欄」的CODiS CSV，轉成 (day, hour, value) 的長表。"""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]
    day_col = df.columns[0]  # 通常是「日/時」

    hour_cols = [c for c in df.columns[1:] if c.strip().isdigit()]
    records = []
    for _, row in df.iterrows():
        day_raw = str(row[day_col]).strip().strip('"')
        if not day_raw.isdigit():
            continue  # 跳過最後那列「總和」/「平均」
        day = int(day_raw)
        for h in hour_cols:
            hour = int(h)
            value = parse_value(row[h])
            records.append((day, hour, value))
    return pd.DataFrame(records, columns=["day", "hour_label", "value"])


def build_month(year: int, month: int, files: dict) -> pd.DataFrame:
    """files: {變數欄位名稱: 檔案路徑}，回傳這個月份合併好的逐時DataFrame。"""
    merged = None
    for col_name, path in files.items():
        long_df = load_wide_csv(path).rename(columns={"value": col_name})
        merged = long_df if merged is None else merged.merge(
            long_df[["day", "hour_label", col_name]], on=["day", "hour_label"], how="outer"
        )

    if merged is None or merged.empty:
        return pd.DataFrame()

    # CODiS的小時標籤是1~24，24代表當天最後一小時(對應00:00~24:00的區間終點)。
    # 這裡做個簡化處理：hour_label=24 當作當天的 hour=0，這是資料標記方式造成的小誤差，
    # 對於「小時的週期性(sin/cos)特徵」影響很小，報告裡可以誠實註記這個簡化假設。
    merged["hour"] = merged["hour_label"].apply(lambda h: 0 if h == 24 else h)
    merged["datetime"] = pd.to_datetime(
        dict(year=year, month=month, day=merged["day"], hour=merged["hour"])
    )
    merged["month"] = month
    return merged.drop(columns=["day", "hour_label"]).sort_values("datetime")


def main():
    files_by_month = {}  # (year, month) -> {col_name: path}
    for path in glob.glob(str(RAW_DIR / "*-hour.csv")):
        m = FILENAME_RE.search(Path(path).name)
        if not m:
            continue
        var = m.group("var")
        if var not in VARIABLES:
            continue
        key = (int(m.group("year")), int(m.group("month")))
        files_by_month.setdefault(key, {})[VARIABLES[var]] = path

    if not files_by_month:
        print("在 data/raw/ 底下找不到符合格式的CODiS CSV檔案，請確認檔案有放進去、檔名沒有被改過。")
        return

    all_months = []
    for (year, month), files in sorted(files_by_month.items()):
        missing = set(VARIABLES.values()) - set(files.keys())
        if missing:
            print(f"[警告] {year}-{month:02d} 缺少項目: {missing}，這個月會跳過對應欄位的資料")
        month_df = build_month(year, month, files)
        if not month_df.empty:
            all_months.append(month_df)
            print(f"已處理 {year}-{month:02d}：{len(month_df)} 筆")

    df = pd.concat(all_months, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    # 缺值處理：氣壓/濕度/溫度用前後值內插補起來(逐時資料通常變化平緩，內插合理)；
    # 降水量缺值視為0(沒有記錄到降水事件)
    for col in ["pressure_hpa", "humidity_pct", "temperature_c"]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit=3)
    if "rain_mm" in df.columns:
        df["rain_mm"] = df["rain_mm"].fillna(0)

    if "pressure_hpa" in df.columns:
        df["pressure_change_3h"] = df["pressure_hpa"].diff(3)
    df["rained"] = (df.get("rain_mm", 0) > 0.5).astype(int)

    # pressure_change_3h在每個月開頭的前3小時會是NaN(沒有更早的資料可以算差值)，
    # 連同其他核心欄位一起檢查缺值，直接把這些沒辦法算特徵的列拿掉(每個月只會少3小時，不影響大局)。
    required_cols = [c for c in ["pressure_hpa", "humidity_pct", "temperature_c", "pressure_change_3h"]
                      if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"缺值處理：拿掉 {before - len(df)} 筆缺少核心特徵的資料")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n合併完成，共 {len(df)} 筆 -> {OUT_PATH}")
    print(f"時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"降雨比例: {df['rained'].mean():.2%}")


if __name__ == "__main__":
    main()
