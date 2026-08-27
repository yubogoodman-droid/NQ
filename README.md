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

## 幣安 15m MA200 站穩三根

1. **記號**：MA7 > MA14 > MA25 **張開**（7 比 25 至少高 0.4%），15m **放量陽線**收盤站上 MA200（量比 ≥1.7×、量大於前一根、離 200 ≤2%、台北 08–20 點）。同一根 15m 記號 ≥4 檔視為大盤一起過線，之後 3 小時跟風也不算。
2. **進場**：記號之後連 **三根收盤都沒跌破 MA200**，第三根收完下一根開盤做多。進場時 **MA99、MA120 不能在 MA200 上面**。200 還在直線往下砍、沒有走平的不要。
3. **出場**：收盤跌破 MA200，或 4 小時到期。同一檔 8 小時只算一次。

近一週大賺單的共同點（相對 501 筆未篩選）：記號均量 **2.7×**（輸家 1.9×）、亞洲／歐洲盤明顯較好（08–19 合計 +78%，21–23 合計 −72%）、真的跑得動的單 MFE 常 5%+。另外 **MA7/14/25 要真的張開**：像 POL 08-20 那筆 MA7 只比 MA14 高 0.004%，一根尖兵就把排列湊出來，看起來很怪。**記號根量要大於前一根**：像 HOME 08-23 13:30 是第二棒（13:15 已爆量還在 200 下），不算開始擴張。**同一根 15m 不要一堆標的一起過 200**：像 08-25 10:15 一次 14 檔（報告 #44–#50 的 FIL/SUI 那排），那是大盤一起彈，後面 WLFI／BEAMX 跟風也不算。**進場時 99/120 不能壓在 200 上面**：那是打進長均壓力，不是從 200 線頭頂起漲。**200 不能還在直線往下砍**：像 INTW／DOS 那種斜率太彎、還沒走平；SNDK／PIPPIN 雖然下行但已經彎頭走平。所以量能、離 200 距離、日盤、排列張開都要過，出場也不再用 2R／三根低點把跑動砍掉。

```bash
python3 examples/scan_binance_15m_expansion.py --verify   # 回放四張圖
python3 examples/scan_binance_15m_expansion.py --once     # 掃剛收盤的 15m
python3 examples/scan_binance_15m_expansion.py            # 每根 15m 收盤掃；可推 Telegram
```

Telegram 填法與下面黏帶腳本相同。合成測試：`python3 examples/test_scan_binance_15m_expansion.py`

四張圖的記號與進場（台北時間，括號是離 MA200 多遠）：

| 標的 | 記號 | 進場 |
|---|---|---|
| FIL | 01-01 19:45 (+0.39%) | 01-01 20:30 (+1.08%) |
| PIPPIN | 01-21 14:45 (+1.18%) | 01-21 15:30 (+1.09%) |
| SNDK | 07-30 19:30 (+0.34%) | 07-30 20:15 (+1.56%) |
| CRCL | 08-19 20:30 (+0.09%) | 08-19 21:15 (+1.45%) |

近一週回測：

```bash
python3 examples/scan_binance_15m_expansion.py --backtest --days 7 --pages
```

近一週（2026-08-21 → 08-27，247 檔）：**32 筆、勝率 43.8%、等權合計 +31.1%、均筆 +0.97%**。2 小時純續走勝率 71.9%、均 +1.98%。上一輪沒要求 99/120 在 200 下面時 46 筆、+34.7%。

先看圖（32 筆每筆都有圖，黃菱形＝記號、藍圈＝第三根確認、綠三角＝進場、× ＝出場）：
https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/binance-15m-expansion-c066/docs/binance/expansion-15m-7d/view.html

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

`pinescript/nq_w_bottom_5m.pine` 可直接貼入 TradingView Pine Editor，套用至 NQ1! 或 MNQ1! 五分圖。

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
