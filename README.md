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

## 一分 K 棒型態回測

學習單根／雙根／三根 K（錘子、吞噬、晨星、三白兵…），再加上南亞科那種**盤整後放量長紅突破**。型態收盤確認，下一根開盤進場；停損在型態極值，停利 1.5R，最多抱 20 分鐘。當日已漲超過 6% 不追。

```bash
# 模擬南亞科式盤整突破
python examples/run_1m_backtest.py --demo

# 近 7 日真實一分 K（Yahoo：南亞科／華邦電／聯電／台積電／NQ）
python examples/run_1m_backtest.py

# 單元測試
python -m unittest tests.test_candles
```

報告寫到 `docs/1m-candles/index.html`。Yahoo 一分 K 最多約 7 個交易日，結果只供學習。

## 南亞科一分均線回測

對應截圖 MA5/10/20/60/120/200：盤整時短均要黏；進場是短均**剛**排成 5>10>20、價剛離開 MA200。已經排成 436 那種末端多頭不追。停損看 MA20，收盤跌破 MA20 離場。

```bash
python examples/run_nanya_ma.py --demo
python examples/run_nanya_ma.py
python -m unittest tests.test_nanya_ma
```

回測網（每筆一分K＋六條均線）：`docs/backtest/index.html`  
表格版：`docs/nanya-ma/index.html`

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
