"""
訓練「未來會不會下雨」的分類模型，並把邏輯迴歸模型的參數匯出成JSON，
讓GitHub Pages上的純前端網頁可以直接用JavaScript重新實作sigmoid推論邏輯。

流程：
1. baseline：氣候學基準 (climatology) —— 用「歷史上這個月、這個時段的平均降雨機率」當預測，
   這是氣象預報領域常用的最低標準，任何模型都應該打得贏這個baseline才有意義。
2. 邏輯迴歸 —— 可解釋、係數容易匯出到前端重現。
3. 隨機森林 —— 當作效能上限的參考，順便看特徵重要性。
4. 特別關注 recall：因為「漏報下雨(沒帶傘淋雨)」的代價比「誤報下雨(多帶一把傘)」高，
   寧可模型敏感一點。

用法:
    python scripts/train_model.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from font_utils import setup_cjk_font

DATA_PATH = "data/processed/weather_hourly.csv"
MODEL_OUT = "docs/model.json"
PLOT_DIR = Path("data/processed/plots")

# 完整想用的特徵清單；如果資料裡沒有某個欄位(例如沒有抓風速)，
# main()裡會自動排除，不用手動改這裡。
DESIRED_FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pressure_change_3h", "humidity_pct", "wind_speed_ms", "temperature_c",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def climatology_baseline(train_df, test_df):
    """用「訓練集裡同月份+同時段的平均降雨機率 >= 0.5」當預測。"""
    clim = train_df.groupby(["month", "hour"])["rained"].mean()
    pred_prob = test_df.set_index(["month", "hour"]).index.map(clim).to_numpy()
    pred_prob = np.nan_to_num(pred_prob, nan=train_df["rained"].mean())
    pred = (pred_prob >= 0.5).astype(int)
    return pred, pred_prob


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[{name}] accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion(name, y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["預測不下雨", "預測下雨"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["實際不下雨", "實際下雨"])
    ax.set_title(f"{name} 混淆矩陣")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    setup_cjk_font()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).sort_values("datetime")
    df = build_features(df)

    global FEATURE_COLS
    FEATURE_COLS = [c for c in DESIRED_FEATURE_COLS if c in df.columns]
    dropped = set(DESIRED_FEATURE_COLS) - set(FEATURE_COLS)
    if dropped:
        print(f"[注意] 資料裡沒有這些欄位，模型會跳過: {dropped}")
    print(f"實際使用的特徵: {FEATURE_COLS}")

    # 用「時間切分」而非隨機切分，避免用未來資料預測過去(資料洩漏)。
    # 但如果整份資料只切一刀(例如只拿最後20%當測試集)，測試集會只落在少數幾個月，
    # 訓練集看不到那幾個月的氣候特性，baseline會嚴重失真(這是先試過一次才發現的坑)。
    # 改成「每個月各自取最後20%的時段當測試集」，train能看到每個月的季節性，
    # 同時每個月內部仍保持「用較早的時段預測較晚的時段」，不算資料洩漏。
    df["_rank_in_month"] = df.groupby("month")["datetime"].rank(pct=True)
    train_df = df[df["_rank_in_month"] < 0.8].drop(columns="_rank_in_month")
    test_df = df[df["_rank_in_month"] >= 0.8].drop(columns="_rank_in_month")
    print(f"訓練集: {len(train_df)} 筆 | 測試集: {len(test_df)} 筆 (每月各自以時間切分, 80/20)")

    X_train, y_train = train_df[FEATURE_COLS].to_numpy(), train_df["rained"].to_numpy()
    X_test, y_test = test_df[FEATURE_COLS].to_numpy(), test_df["rained"].to_numpy()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # 1. baseline
    base_pred, _ = climatology_baseline(train_df, test_df)
    results.append(evaluate("氣候學baseline", y_test, base_pred))

    # 2. 邏輯迴歸 (class_weight='balanced' 讓模型更看重「抓出真的會下雨」，呼應recall優先的考量)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train_s, y_train)
    logreg_pred = logreg.predict(X_test_s)
    results.append(evaluate("邏輯迴歸", y_test, logreg_pred))
    plot_confusion("邏輯迴歸", y_test, logreg_pred, PLOT_DIR / "confusion_logreg.png")

    # 3. 隨機森林 (效能上限參考 + 特徵重要性)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # 樹模型不需要標準化
    rf_pred = rf.predict(X_test)
    results.append(evaluate("隨機森林", y_test, rf_pred))
    plot_confusion("隨機森林", y_test, rf_pred, PLOT_DIR / "confusion_rf.png")

    # 特徵重要性圖
    importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance.index, importance.values, color="#3b6ea5")
    ax.set_title("隨機森林特徵重要性")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 模型比較表存成CSV，方便貼進報告
    pd.DataFrame(results).to_csv(PLOT_DIR / "model_comparison.csv", index=False)
    print(f"\n模型比較結果已存到 {PLOT_DIR / 'model_comparison.csv'}")

    # 匯出邏輯迴歸模型參數給前端JS用 (係數 + StandardScaler的mean/scale)
    model_export = {
        "feature_order": FEATURE_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
        "threshold": 0.5,
        "metrics": [r for r in results if r["name"] == "邏輯迴歸"][0],
    }
    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(model_export, f, ensure_ascii=False, indent=2)
    print(f"模型參數已匯出 -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
