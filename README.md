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

## 幣安 Telegram（15 分同時站上）

15 分鐘 K：前一根收盤完全在 MA7 / 14 / 25 / 99 / 120 下方，這一根收盤同時站上這五條，就推 Telegram（只送文字，不畫圖）。預設全掃流動永續（加密+股票）。不必過 MA200。  
這支腳本是獨立的，複製到別的資料夾當 `小米15分K.py` 也能跑，不必帶 `nq` 套件。

在 `examples/watch_binance_ribbon.py` 最上面填：

```
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."
```

```bash
python3 examples/watch_binance_ribbon.py --demo          # 驗證偵測
python3 examples/watch_binance_ribbon.py --test          # 先測 Telegram 通不通
python3 examples/watch_binance_ribbon.py --once          # 掃剛收的 15 分，推完就結束
python3 examples/watch_binance_ribbon.py                 # 每根 15 分收盤全掃流動盤
python3 examples/watch_binance_ribbon.py --asset stocks  # 只要股票
python3 examples/watch_binance_ribbon.py --also-1m       # 順便跑原本 1m 黏帶
```

## 15 分 K 同時站上 7 / 14 / 25 / 99 / 120

一根 15 分鐘 K：前一根收盤完全在 MA7 / 14 / 25 / 99 / 120 下方，這一根收盤同時站上這五條。不必過 MA200。進場用下一根開盤。圖例每筆底下附同一時間的 1 小時圖，只做對照。圖上仍畫 MA200 方便看位置。

```bash
python3 examples/backtest_15m_ribbon.py --demo
python3 examples/backtest_15m_ribbon.py --days 7 --pages
python3 examples/backtest_15m_ribbon.py --days 7 --asset stocks --pages   # 只要股票
```

報告：
- 全部流動盤：`docs/binance/ma-ribbon-15m.html`
- 只要股票：`docs/binance/ma-ribbon-15m-stocks.html`

外網：  
https://yubogoodman-droid.github.io/NQ/binance/ma-ribbon-15m.html  
https://yubogoodman-droid.github.io/NQ/binance/ma-ribbon-15m-stocks.html

合併進 `main` 後 GitHub Pages：  
https://yubogoodman-droid.github.io/NQ/binance/ma-ribbon-15m.html

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

3. **15 分 K 五均線回測**  
   全部：https://yubogoodman-droid.github.io/NQ/binance/ma-ribbon-15m.html  
   股票：https://yubogoodman-droid.github.io/NQ/binance/ma-ribbon-15m-stocks.html

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
