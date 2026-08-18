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

## 台股一分 K 掃描（多頭排列 × 站穩 MA200 兩根）

條件：

1. Yahoo 股市**成交金額前 100 名**（上市＋上櫃）
2. **濾掉 ETF**（代號 00/02 開頭或含英文字）、**金融股**（金控／銀行／保險／證券）與**股價 650 元以上**
3. 一分 K 的 **MA5 > MA10 > MA20**（多頭排列），MA5−MA20 ≥ 0.4%，MA20 與 MA200 差距 **0.4%～1.0%**
4. **從 MA200 下面穿上**，金叉前至少三根收在 200 下；**連續兩根收盤站穩**才通知，站穩時間 **09:10 以後**
5. 含該根的五分 K 收盤也必須高於五分 MA200

`MA200` 是一分 K 的 200 期均線（約 200 分鐘），不是日線 200 日。隔夜跳空與開盤 09:00–09:09 不算站穩。

外網頁面（合併進 main 或 Pages 部署後）：

- https://yubogoodman-droid.github.io/NQ/tw/
- 免部署預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/tw-1m-ma-alert-120c/docs/tw/index.html

```bash
pip install -r requirements.txt
python examples/scan_tw_1m.py
```

預設會列出**今天盤中曾出現**的訊號；`--watch` 只對最新一根跳通知。有永豐 API 金鑰時自動改抓即時一分 K。

```bash
python examples/scan_tw_1m.py --latest-only   # 只看當下這一根
python examples/scan_tw_1m.py --watch         # 盤中守著，符合就通知
```

預設走永豐。**只要一個檔**：把 `scan_tw.py` 存進 PyCharm（檔名不要叫 `tw.py`），最上面四個引號填永豐 Key / Secret 和 Telegram 序號，然後：

```bash
pip install pandas numpy yfinance requests shioaji
python scan_tw.py --watch
```

第一次掃描會按日抓近幾天 1K 當種子（永豐一分 K 單次約 270 根），之後用成交明細補當根。強制 Yahoo：`--source yahoo`。

通知通道（擇一或同時設環境變數）：

- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK_URL`
- Linux 桌面：有 `notify-send` 會一併跳出

報告寫入 `docs/tw/index.html`（GitHub Pages）。

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
