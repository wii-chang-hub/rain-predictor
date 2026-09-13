# 出門前的降雨預測小幫手

高中資訊/資安特殊選材備審專題。用中央氣象署開放資料訓練一個「接下來1-3小時會不會下雨」的分類模型，
幫自己判斷出門要不要帶傘，減少每天書包裡的不必要物品。

## 專案結構

```
rain-predictor/
├── data/
│   ├── raw/            # 從CWA抓下來的原始資料 (不進版控，太大)
│   └── processed/       # 清理過、可直接餵給模型的資料
├── scripts/
│   ├── generate_sample_data.py     # 沒有真實資料時，先產生結構相同的模擬資料跑通全流程
│   ├── fetch_data.py               # 測試CWA API金鑰連線用 (抓即時觀測，不是訓練資料)
│   ├── build_dataset_from_codis.py # 把從CODiS下載的「單項逐時月報表」CSV合併成訓練格式
│   ├── font_utils.py               # 讓matplotlib畫圖時中文不會變方框，跨平台共用
│   ├── eda.py                       # 探索式資料分析，輸出圖表到 data/processed/plots/
│   └── train_model.py               # 訓練分類模型，評估效果，把模型參數匯出成 docs/model.json
└── docs/                # GitHub Pages 根目錄 (純前端，直接部署)
    ├── index.html
    ├── style.css
    ├── app.js
    └── model.json        # train_model.py 產生
```

## 開發流程

### 1. 先用模擬資料跑通整條流程（不用等API金鑰）

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/eda.py
python scripts/train_model.py
```

跑完後 `docs/` 資料夾就是一個可以直接用瀏覽器打開的完整網頁 (index.html)，
可以先確認整個系統邏輯是通的。

### 2. 申請中央氣象署開放資料API金鑰（測試連線用）

1. 到 https://opendata.cwa.gov.tw/ 免費註冊會員
2. 登入後在會員中心找「取得授權碼」拿到你的 API Key（格式類似 `CWA-XXXXXXXX-XXXX-...`）
3. 在終端機設定環境變數（不要把金鑰寫死進程式碼、不要commit進Git）：
   ```bash
   set CWA_API_KEY=你的金鑰
   ```
4. 執行 `python scripts/fetch_data.py` 測試連線是否成功

### 3. 下載真正拿來訓練模型的歷史資料

金鑰只能抓「即時觀測」，訓練模型要的「過去幾個月逐時歷史資料」要另外去
氣候資料服務系統（CODiS, https://codis.cwa.gov.tw/StationData）下載：

1. 搜尋測站（例如新竹）
2. 左側選單選「單項逐時月報表」
3. 依序選擇項目（測站氣壓 StationPressure、相對溼度 RelativeHumidity、
   氣溫 AirTemperature、降水量 Precipitation）跟月份，各別下載CSV
4. 把所有下載下來的CSV直接丟進 `data/raw/` 資料夾，檔名不用改

下載完成後執行：
```bash
python scripts/build_dataset_from_codis.py   # 合併成 data/processed/weather_hourly.csv
python scripts/eda.py
python scripts/train_model.py
```

### 4. 部署到 GitHub Pages

```bash
git init
git add .
git commit -m "init: 降雨預測小幫手"
git remote add origin <你的repo網址>
git push -u origin main
```

到 GitHub repo 的 Settings → Pages，Source 選擇 `main` 分支、`/docs` 資料夾，儲存後
幾分鐘內就會有一個 `https://<你的帳號>.github.io/<repo名稱>/` 的網址。

## 技術說明（給備審資料用）

- **需求**：每天出門前很難判斷要不要帶傘，乾脆什麼都帶，書包越來越重。
- **資料**：中央氣象署開放資料平台的新竹測站逐時觀測資料（氣壓、濕度、風速、降雨量）。
- **特徵工程**：除了原始氣象數值，額外計算「氣壓變化率」（氣壓驟降常是降雨前兆）、時間特徵
  （小時、月份、是否為夏季午後雷陣雨好發時段）。
- **模型**：先做「歷史同時段降雨機率」當baseline，再用邏輯迴歸／隨機森林做二元分類，
  並與中央氣象署官方短時預報做準確度比較。
- **不對稱代價**：漏報下雨（沒帶傘淋雨）比誤報下雨（多帶一把傘）代價高，
  所以模型調整時特別關注 recall（抓出真的會下雨的比例），而不是只看整體準確率。
- **部署**：模型訓練完成後，把邏輯迴歸的係數匯出成JSON，前端網頁用JavaScript重新實作
  sigmoid推論邏輯，整個系統是純前端、部署在GitHub Pages上，不需要後端主機。
