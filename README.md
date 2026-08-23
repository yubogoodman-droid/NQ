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

## 幣安 15 分 K：7>25 多頭 + 站上 MA200

15 分圖收盤同時高於 **MA7、MA25、MA200**，且 SMA7 > SMA25（MA14 只畫圖、不擋單）。Telegram 只推「這一根剛站上該週期 MA200」。小時圖同一套。大週期圖只對照，不擋單；不再看量比、底下根數、高低差、BTC。

```bash
python3 examples/backtest_15m_bull.py --demo
python3 examples/backtest_15m_bull.py --days 7 --pages                   # 15 分全市場
python3 examples/backtest_15m_bull.py --days 7 --tf 1h --pages           # 小時圖全市場
python3 examples/backtest_15m_bull.py --days 7 --stocks --pages          # 15 分股票
python3 examples/backtest_15m_bull.py --days 7 --tf 1h --stocks --pages  # 小時圖股票
```

外網看圖（請用這個，圖已內嵌）：  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull.html

只掃股票（15 分）：  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull-stocks.html

小時圖（全市場）：  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma1h-bull.html

只掃股票（小時）：  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma1h-bull-stocks.html

合併進 `main` 後 GitHub Pages：  
https://yubogoodman-droid.github.io/NQ/binance/ma15-bull.html

Telegram 監看（與回測同一套規則）。**不要只拷一個 py**，`nq` 資料夾要在旁邊。

PyCharm：用「Open」打開整個 NQ 專案，或把 `15M多排.py` 和 `nq` 放同一層，在 `15M多排.py` 最上面填 token / chat id：

```
PythonProject2\
  15M多排.py
  nq\
```

```bash
pip install numpy requests matplotlib
python 15M多排.py --test
python 15M多排.py                 # 預設 15 分 + 1 小時都推
python 15M多排.py --tf 1h         # 只推小時圖（剛站上 1h MA200）
python 15M多排.py --once
```

也可以 `pip install -e .` 後跑 `examples/watch_15m_bull.py`。

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

### 外網開啟

1. **GitHub Pages（永久）**  
   到 [Repo Settings → Pages](https://github.com/yubogoodman-droid/NQ/settings/pages)，Source 選 **GitHub Actions**，儲存後重新執行 workflow。  
   網址：`https://yubogoodman-droid.github.io/NQ/`

2. **HTML Preview（免部署）**  
   https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/index.html

3. **15 分 K 7>25 站上 MA200（這次 PR）**  
   https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull.html  
   只掃幣安股票：  
   https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull-stocks.html

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
