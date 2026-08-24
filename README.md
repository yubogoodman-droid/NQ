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

## NQ 起漲點（均線糾結突破）· 1 分訊號，對照當時 5 分

訊號只在 **1 分鐘** 上抓：下跌後均線糾結、窄幅盤整。進場是 **第一次收盤站上 MA200**，且 **MA5>MA10>MA20>MA30**，**MA60、MA120 都在 MA200 下方**。不要等後面那根放量長綠。

停損在盤整低點下方 5 點；目標 2R。走到 **1R 後停損移到 +0.3R**（移動停利），後面拉回就鎖一截，2R 還在。連續 2 根 1 分收回到盤整高點下，當成突破失敗、用收盤離場。不追 1 分實體已經超過 40 點的長 K。進場當下 **5 分收盤要在 5 分 MA20 上方，也要站上 5 分 MA200**。

每筆 1 分進場會附上 **當時的 5 分 K**：只用到進場那一分為止，當根 5 分可能還沒收完，不會偷看後面。

```bash
# 模擬那張「07:35 放量突破」的走勢
python3 examples/nq_coil_breakout.py --demo

# Yahoo 1m 回測，每筆附當時 5分K + 手機版 HTML
python3 examples/nq_coil_breakout.py backtest --period 10d --html output/nq_coil.html
python3 examples/nq_coil_breakout.py backtest --period 10d --pages
python3 examples/nq_coil_breakout.py backtest --period 30d --pages

# Telegram 輪詢（憑證放 tg_config.env）
python3 examples/nq_coil_breakout.py alert --dry-run --once
```

近 10 天 Yahoo 1m（2026-08-14 → 08-24）：**7 筆、勝率 57.1%、+192.4 點**（改前 10 筆、60%、+333）。
近一個月 Yahoo 1m（2026-07-26 → 08-24）：**21 筆、勝率 42.9%、+196.9 點**（改前 29 筆、44.8%、+330.5）。樣本仍少，只是回測不是保證。
預覽目前是這份一個月的圖。
你那張圖 **08-24 07:32 @ 29217.75** 當時 5 分 MA200 約 29302，還在下方 85 點，**這條規則下不會觸發**。1 分確實站上了 1 分 MA200，但 5 分還沒站上 5 分 MA200。

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
- `pinescript/nq_ma_coil_breakout_1m.pine` — NQ 一分 K 均線糾結起漲點。

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
