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

## 台股五分 K 回測

成交額前 100、濾掉 ETF／金融股／電信股／股價 500 以上。五分 K **MA5 > MA10 > MA20 且均線發散**（不要糾結），當根收盤剛站上五分 **MA200**，收盤高於所有均線，也在十五分 **MA5／10／20** 之上，**十五分K已在 MA200 上至少半小時**，**小時K收在 MA20 之上**，記一則通知。

```bash
python3 examples/backtest_tw_5m.py --days 10
python3 examples/backtest_tw_15m.py --days 10
```

報告：`docs/tw/backtest-5m-15m-1h.html`（五分K）、`docs/tw/backtest-15m-1h.html`（十五分K；同一套發散＋剛站上 MA200，小時K要在 MA5／10／20 之上）

## 永豐盤中監控 → Telegram

同一套掃描池與進場條件。啟動時用永豐抓一次歷史 1 分K 當均線底，盤中改訂閱 tick 合成五分／十五分（不要盤中重覆打 kbars）。符合就推 Telegram，同一根 K 不重發。

本機金鑰請放 `examples/local_secrets.py`（已 gitignore，不要 commit），或設環境變數。腳本上方的空字串不要填真的金鑰。

```
SHIOAJI_API_KEY=...
SHIOAJI_SECRET_KEY=...
TELEGRAM_BOT_TOKEN=...   # 也可用 TG_TOKEN
TELEGRAM_CHAT_ID=...     # 也可用 TG_CHAT_ID
```

```bash
pip install shioaji
python3 examples/watch_tw_shioaji.py --test     # 先測 Telegram
python3 examples/watch_tw_shioaji.py --once     # 盤後用歷史K掃今天
python3 examples/watch_tw_shioaji.py            # 盤中一直盯（五分＋十五分）
python3 examples/watch_tw_shioaji.py --tf 5m
```

建議每個交易日開盤前重開。金鑰不要 commit。

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
