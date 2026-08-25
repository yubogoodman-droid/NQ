# NQ 五分 K W 底進場策略

那斯達克期貨（NQ）五分鐘 K 線的 **W 底（雙底）** 做多進場策略。

## 型態定義

```
        頸線 ─────────●─────────
                   /   \
                  /     \
    第一低點 ●───         ───● 第二低點
                              ↑
                         突破進場
```

| 條件 | 說明 |
|------|------|
| 第一低點 | 波段低點（左右各 3 根 K 確認） |
| 頸線 | 兩低點之間的最高波段高點 |
| 第二低點 | 與第一低點價差 ≤ 0.1%，間隔 5~60 根 K |
| 進場 | 收盤突破頸線，做多 NQ |
| 停損 | 第二低點 |
| 停利 | 量度漲幅：目標 = 頸線 + (頸線 − 最低點) |

## NQ 起漲點 · 糾結後連兩根剛站上 MA200

模板是 **08-19 08:15** 與 **16:48**：先跌一段，短均線收成一束貼著 MA200，收盤剛站上。

訊號只在 **1 分鐘** 上抓：

| 條件 | 門檻 |
|------|------|
| 短均排列 | MA5>MA10>MA20>MA30 |
| 站穩 | 連續兩根收盤站上 MA200；第一根還不進 |
| 剛站上 | 收盤高出 MA200 ≤ 15 點 |
| 短均收束 | MA5–MA60 帶寬 ≤ 20 |
| 窄箱 | 近 12 根修剪高低 10–40 點 |
| 貼著 200 | 近 12 根至少 4 根收盤距 MA200 ≤ 8 |
| 先跌 | 近 60 根高低至少 45 點；MA200 近 20 根仍往下 |
| 先在 200 下 | 進場前至少 4 根收在 MA200 下面 |
| 不追 | 進場 K 實體絕對值 ≤ 20 |
| 長均 | MA100 與 MA120 都要在 MA200 下面 |

**07:33 / 11:30 / 開盤追價 / 尾盤刺穿 200** 不進。5 分均線不是進場條件，只對照當時長相。

停損在進場前 20 根修剪低點下方 5 點；目標 2R。走到 **1R 後停損移到 +0.3R**。連續 2 根 1 分收回到進場時 MA200 下，當成站上失敗、用收盤離場。

```bash
# 08-19 08:15 那張理想圖
python3 examples/nq_coil_breakout.py --demo

# Yahoo 1m 回測，每筆附當時 5分K + 手機版 HTML
python3 examples/nq_coil_breakout.py backtest --period 10d --html output/nq_coil.html
python3 examples/nq_coil_breakout.py backtest --period 10d --pages
python3 examples/nq_coil_breakout.py backtest --period 30d --pages

# Telegram 輪詢（憑證放 tg_config.env）
python3 examples/nq_coil_breakout.py alert --dry-run --once
```

| 區間 | 筆數 | 勝率 | 加總 |
|------|------|------|------|
| 10 天 | 2 | 100% | +149.0 |

只留 08-19 08:15（+102）與 16:48（+47）。樣本很小，回測不是保證。

TradingView：把 `pinescript/nq_ma_coil_breakout_1m.pine` 貼進 Pine Editor，套用到 NQ1! / MNQ1! 一分圖，可設「NQ 起漲點」警示。

預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/nq-1m-coil-breakout-36d9/docs/nq-coil-breakout/view.html

## NQ 一分 K 破底翻 MA Reclaim

1 分鐘圖：跌破近 2 小時低點後，15 根內收復 MA20/MA30，且 MA5>MA10>MA20，做多 NQ。  
停損在破底低點下方；目標 2R（MA20 上彎時 3R）。品質看 1m MA5 / 1m MA60 / 5m MA60 斜率。

```bash
# 近 8 天 / 近一個月回測 + 手機版 HTML（每筆一張圖）
python3 examples/nq_ma_reclaim.py backtest --period 8d --html output/nq_ma_reclaim.html
python3 examples/nq_ma_reclaim.py backtest --period 30d --pages

# Telegram 輪詢（憑證放 tg_config.env，勿提交）
python3 examples/nq_ma_reclaim.py alert --test
python3 examples/nq_ma_reclaim.py alert --dry-run --once
python3 examples/nq_ma_reclaim.py alert
```

近一個月 Yahoo 1m（2026-07-24 → 08-21）：**嚴格 8 筆、勝率 62.5%、約 +614 點**。  
漏斗：破底 1197 → 深度≥10點 293 → 收復+排列後還被 hug/MA60/9–10點/風險擋掉，只剩 8。  
關掉 hug／MA60 特例後（核心）變成 **25 筆、+815 點**，但會多出幾筆 −50 停損（含原本要擋的 08-11 12:39）。

外網（合併後）：https://yubogoodman-droid.github.io/NQ/nq-ma-reclaim/  
現在先看圖：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/nq-1m-ma-reclaim-2484/docs/nq-ma-reclaim/view.html

## 台股成交額前 100 · 同一套破底翻（一週）

```bash
python3 examples/scan_tw_ma_reclaim.py --limit 100 --range 7d --pages
```

2026-08-21 成交額前 100（第 100 名約 17 億）：一週 **7 筆**。  
近一個月、**股價 ≤ 600**：100 檔（台積電／聯發科／大立光等已濾掉）**32 筆、勝率 21.9%、−6.55**。

```bash
python3 examples/scan_tw_ma_reclaim.py --days 30 --max-price 600 --limit 100 --pages
```

月報預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/nq-1m-ma-reclaim-2484/docs/tw-ma-reclaim-30d/view.html

## 幣安黏帶三幕 Telegram

1 分鐘圖：圓 U 吻上 MA99/120/200 黏帶後放量離開，會推 Telegram。  
約 15–25 分鐘後若走出「長均更黏、短均被帶走」（NBIS 那種），會再推一則**強訊號**（帶圖）。

在 `examples/watch_binance_ribbon.py` 最上面填：

```
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."
```

```bash
python3 examples/watch_binance_ribbon.py --test   # 先測通不通
python3 examples/watch_binance_ribbon.py          # 每根 1m 收盤掃一次
```

## 快速開始

```bash
pip install -r requirements.txt
python examples/run_backtest.py --demo
```

### 使用自己的五分 K 資料

CSV 需含欄位：`datetime, open, high, low, close`

```bash
python examples/run_backtest.py --csv your_nq_5m.csv
```

### 產生今日 HTML 圖表

```bash
# 交易卡片報告（每筆分開，手機版，推薦）
python3 examples/chart_today.py --report --pages

# 30 天回測報告
python3 examples/chart_today.py --report --days 30 -o output/nq_report_30d.html

# 單一大圖
python3 examples/chart_today.py
```

### 股市進出簿

手機也能用的買賣帳本，資料存在瀏覽器本機，可匯出 JSON / CSV。

- 本機：開 `docs/journal/index.html`
- GitHub Pages：`https://yubogoodman-droid.github.io/NQ/journal/`

可記台股／美股買進賣出、自動帶入台股手續費與證交稅、用先進先出算持倉與已實現損益，並在持倉裡手動設現價看未實現。

## 外網開啟

1. **GitHub Pages（永久）**  
   到 [Repo Settings → Pages](https://github.com/yubogoodman-droid/NQ/settings/pages)，Source 選 **GitHub Actions**，儲存後重新執行 workflow。  
   網址：`https://yubogoodman-droid.github.io/NQ/`

2. **HTML Preview（免部署）**  
   https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/index.html

## TradingView

- `pinescript/nq_w_bottom_5m.pine` — NQ 五分 K W 底，套用至 NQ1! / MNQ1! 五分圖。
- `pinescript/nq_ma_coil_breakout_1m.pine` — NQ 一分 K：5>10>20>30 連兩根站上 MA200。

## 參數調整

在 `nq/strategy.py` 的 `NQWBottomStrategy` 中：

| 參數 | 預設 | 說明 |
|------|------|------|
| `swing_lookback` | 3 | 轉折確認 K 數 |
| `low_tolerance_pct` | 0.001 | 兩低點價差容忍（0.1%） |
| `min_bars_between_lows` | 5 | 兩低點最少間隔 |
| `max_bars_between_lows` | 60 | 兩低點最多間隔（約 5 小時） |

## 風險提示

本策略僅供學習與回測，不構成投資建議。實盤請搭配流動性時段、滑價、保證金與個人風控。
