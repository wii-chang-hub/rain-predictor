// 出門前的降雨預測小幫手 —— 純前端推論
// 模型參數(model.json)是用 scripts/train_model.py 在Python裡訓練完之後匯出的
// 邏輯迴歸係數，這裡用JavaScript重新實作標準化 + sigmoid，兩邊算出來的結果應該一致。

let MODEL = null;

async function loadModel() {
  const res = await fetch("model.json");
  if (!res.ok) throw new Error("model.json 載入失敗，請確認這個頁面是用HTTP伺服器開的（不能直接雙擊開檔案）");
  MODEL = await res.json();
  renderMetrics(MODEL.metrics);
}

function renderMetrics(m) {
  if (!m) return;
  const grid = document.getElementById("metricsGrid");
  const cells = grid.querySelectorAll(".num");
  const vals = [m.accuracy, m.precision, m.recall, m.f1];
  cells.forEach((el, i) => {
    el.textContent = (vals[i] * 100).toFixed(1) + "%";
  });
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function buildFeatureVector(inputs) {
  const dt = new Date(inputs.datetime);
  const hour = dt.getHours() + dt.getMinutes() / 60;
  const month = dt.getMonth() + 1;

  const raw = {
    hour_sin: Math.sin((2 * Math.PI * hour) / 24),
    hour_cos: Math.cos((2 * Math.PI * hour) / 24),
    month_sin: Math.sin((2 * Math.PI * month) / 12),
    month_cos: Math.cos((2 * Math.PI * month) / 12),
    pressure_change_3h: inputs.pressureChange,
    humidity_pct: inputs.humidity,
    wind_speed_ms: inputs.wind,
    temperature_c: inputs.temp,
  };

  // 依照 model.json 裡紀錄的 feature_order 排好順序，順序錯了係數會對不上
  return MODEL.feature_order.map((name) => raw[name]);
}

function predict(inputs) {
  const x = buildFeatureVector(inputs);
  const z = x.reduce((sum, xi, i) => {
    const standardized = (xi - MODEL.scaler_mean[i]) / MODEL.scaler_scale[i];
    return sum + standardized * MODEL.coef[i];
  }, MODEL.intercept);
  return sigmoid(z);
}

function classify(prob) {
  if (prob < 0.3) return { label: "不太需要帶傘", cls: "low" };
  if (prob < 0.6) return { label: "建議帶一把備用", cls: "mid" };
  return { label: "一定要帶傘", cls: "high" };
}

function setupDefaultDatetime() {
  const el = document.getElementById("datetime");
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  el.value = now.toISOString().slice(0, 16);
}

function onPredictClick() {
  if (!MODEL) {
    alert("模型還沒載入完成，稍等一下再試一次");
    return;
  }
  const inputs = {
    datetime: document.getElementById("datetime").value,
    pressureChange: parseFloat(document.getElementById("pressure_change").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    wind: parseFloat(document.getElementById("wind").value),
    temp: parseFloat(document.getElementById("temp").value),
  };
  if (Object.values(inputs).some((v) => v === "" || Number.isNaN(v))) {
    alert("請把欄位都填完整");
    return;
  }

  const prob = predict(inputs);
  const { label, cls } = classify(prob);

  const box = document.getElementById("resultBox");
  document.getElementById("probText").textContent = (prob * 100).toFixed(1) + "%";
  const badge = document.getElementById("badge");
  badge.textContent = label;
  badge.className = "badge " + cls;
  document.getElementById("adviceText").textContent =
    "這是用你輸入的氣壓變化、濕度、風速、溫度，配合當下時間(小時/月份)算出來的降雨機率，" +
    "模型訓練時刻意提高對降雨的敏感度（漏報比誤報代價高），實際使用時仍建議搭配官方預報判斷。";
  box.classList.add("show");
}

async function onFetchClick() {
  const status = document.getElementById("fetchStatus");
  const apiKey = document.getElementById("apiKey").value.trim();
  if (!apiKey) {
    status.textContent = "請先填入你的CWA API金鑰";
    return;
  }
  localStorage.setItem("cwa_api_key", apiKey);
  status.textContent = "抓取中...";

  const url = `https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization=${encodeURIComponent(apiKey)}&StationName=%E6%96%B0%E7%AB%B9`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const station = data?.records?.Station?.[0];
    if (!station) throw new Error("回傳資料格式不如預期，可能是測站名稱要調整，或資料集ID有異動");

    const weather = station.WeatherElement;
    if (weather.RelativeHumidity != null) document.getElementById("humidity").value = weather.RelativeHumidity;
    if (weather.WindSpeed != null) document.getElementById("wind").value = weather.WindSpeed;
    if (weather.AirTemperature != null) document.getElementById("temp").value = weather.AirTemperature;
    // CWA即時觀測通常不會直接給「過去3小時氣壓變化」，這欄暫時維持手動填寫，
    // 若要自動化，需要自己每小時記錄氣壓、在前端算出差值(可以用localStorage存歷史值)。
    status.textContent = "已帶入濕度/風速/溫度，氣壓變化仍請手動確認（CWA即時觀測沒有直接提供這個欄位）。";
  } catch (err) {
    status.textContent =
      "抓取失敗：" + err.message + "（可能是CORS限制、金鑰錯誤，或資料集ID需要調整，先用手動輸入的欄位測試也可以）";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  setupDefaultDatetime();
  loadModel().catch((err) => {
    console.error(err);
    document.getElementById("fetchStatus").textContent = "";
    alert("模型載入失敗：" + err.message);
  });
  document.getElementById("predictBtn").addEventListener("click", onPredictClick);
  document.getElementById("fetchBtn").addEventListener("click", onFetchClick);

  const savedKey = localStorage.getItem("cwa_api_key");
  if (savedKey) document.getElementById("apiKey").value = savedKey;
});
