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

## 台股 1 小時 K 破底翻（元晶型）

1 小時圖，對齊 6443 元晶 2026-09-16 那波：收盤從 MA20 上方跌到下方後，在下面待 6～36 根（不准假站上），相對過程中最高 MA20 深度 ≥ 3.5%，且最低點是近 16 根新低；破底與翻上不能同一根，從低點至少彈回 3%，再站回 MA20 這波才算。破底後 36 根內，第一根收盤同時大於 MA5 / MA10 / MA20 / **MA60**、且該根 K 棒收在當根振幅上緣 ≥ **65%** 進場（預設取成交額前 **100**、留超過 35% 長上影的衝高倒貨棒整波放棄）。下面天數放寬是為了 6456 GIS-KY 那種先磨再翻。

回測出場：停損放在破底低，但離進場超過 5% 就改停在進場下方 5%；停利 0.8R；否則 20 根時間停。另外，進場那根加權 `^TWII` 過去 5 根 1h 報酬要 ≥ 0，大盤這一段在跌就不進。

```bash
# 成交額前 200、近一週
python3 examples/tw_1h_reclaim.py --days 7 --range 2mo --html docs/tw-1h-reclaim-7d/index.html

# 成交額前 200、近一個月；預設元晶型、股價 1000 以上刪掉
python3 examples/tw_1h_reclaim.py --days 30 --range 2mo --pages --html docs/tw-1h-reclaim-30d/index.html

# 近兩個禮拜
python3 examples/tw_1h_reclaim.py --days 14 --range 2mo --pages

# 不限成交額（仍刪股價 1000 以上）
python3 examples/tw_1h_reclaim.py --limit 0 --days 7 --range 1mo

# 單元測試（不打網路）
python3 examples/test_tw_1h_reclaim.py
```

2026-08-26 → 09-24、成交額前 200 且**股價 < 1000**：元晶型 **149 筆、104 檔**。已平 142 筆勝率 **50.7%**，平均 **+1.30%**；2R 16 筆平均 +12.7%，停損 39 筆平均 −4.8%。6443 元晶 9/16 進場，次日 2R +5.98%。

近一週（2026-10-21 → 10-27、成交額前 200、股價 < 1000）：**66 筆**，已平 26 筆（2R 3、停損 23），未平 40。週報：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/tw-1h-reclaim-7d/view.html

2026-10-17 → 10-30 逐步優化：
1. 原始（2R、停在破底）：137 筆、勝率 17.5%、平均 −3.83%。
2. 出場改 1R、停損最多 4%：137 筆、勝率 37.2%、平均 −0.97%。
3. 再加加權過去 5 根 1h 報酬 ≥ 0：111 筆、勝率 38.7%、平均 −0.84%。
4. 再要求進場收盤 > MA60（紫線，排除壓在季線級均線下的死貓跳）：**52 筆、勝率 55.8%、平均 +0.39%、中位數 +2.67%**（停利 26、停損 21、時間 5），同時 5–8 月（649 筆 +0.93%）、8/26–9/24（207 筆 +1.06%）、9/29–10/16（64 筆 +0.14%）四段全部正期望。預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/tw-1h-reclaim-20261017-1030-ma60/view.html
5. 勝率強化版（2026-11-12 全 6 個月驗證：成交額前 100 + 突破 K 收在當根振幅 ≥ 65% 不留長上影 + 停損最多 5% / 停利 0.8R）：全 6 個月 **388 筆、已平勝率 67.3%（全體 66.8%）、平均 +1.00%**；最新兩週 2026-10-30 → 11-12 **31 筆、已平勝率 80.0%（全體 71.0%）、平均 +1.44%**（停利 20、停損 5、未平 6）。預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/tw-1h-reclaim-20261030-1112/view.html

月報預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/main/docs/tw-1h-reclaim-30d/view.html

## 幣安黏帶三幕 Telegram（活盤優化版）

1 分鐘圖：圓 U 吻上 MA99/120/200 黏帶後放量離開。預設啟用**零未來函數優化濾網**，推播時附上 K 線圖、過高觸價進場價與圓底停損價：
- 離開 K 收在當根振幅上緣 ≥ 70%（不留長上影倒貨）且收過吻後高點
- 前 2 小時高低跌幅 ≥ 1.2%（排除死魚橫盤）、MA200 斜率開始拐頭向上、量能 `1.8x–6.0x`（排除極端爆量針）
- 大盤 `BTCUSDT` 1m 收盤在 `MA25` 之上
- 執行：離開後 4 分鐘內突破離開 K 高點觸價進場（先跌破黏帶則取消），停損守圓底 40 根低點（上限 1.6%），不設緊停利、抱滿 60 分鐘收盤出場。

跨兩段共 51 天驗證（已扣 0.08% 來回手續費、同根 K 先判停損）：
- 原始離開（30m 無停損）：8/15–9/05 樣本外 182 筆平均 `-0.006%`（PF `0.98`）、10/15–11/14 近一個月 199 筆平均 `+0.067%`。
- 優化過高進 + 圓底停損 + 60m：8/15–9/05 樣本外 **42 筆、平均 `+0.304%`、PF `2.03`**；10/15–11/14 近一個月 **53 筆、平均 `+0.317%`、PF `2.08`**（51 天合計 **95 筆、平均 `+0.311%`、PF `2.06`、累計 `+29.55%`**）。

在 `examples/watch_binance_ribbon.py` 最上面填：

```
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."
```

```bash
python3 examples/watch_binance_ribbon.py --test         # 先測通不通
python3 examples/watch_binance_ribbon.py                # 每根 1m 收盤掃一次（預設優化版）
python3 examples/watch_binance_ribbon.py --raw          # 舊版未過濾離開全推
python3 examples/watch_binance_ribbon.py --backtest 7   # 回測近 7 天（原始 vs 優化版）
python3 examples/test_watch_binance_ribbon.py           # 離線單元測試
```

## 幣安橫盤壓縮爆量起漲點 Telegram（`watch_binance_ignition.py`）

1 分鐘圖：專抓死魚窄幅橫盤後**第一根實體爆量收穿平台**的起漲點（例如 `EVAAUSDT` 2026-11-14 22:10 從 `1.154` 收穿 `1.1698`、下一根 `1.2075` 觸價進場），排除長上影假突破針與高位追高：
- **橫盤壓縮未追高**：前 2 小時高低振幅 `2.0%–6.5%`、前 6 小時振幅 `≤ 12%`、前 2 小時均量 `≤ 8 萬 USDT/m`（排除高位寬幅震盪與權值幣）
- **真突破實體棒**：收盤高出前 2 小時箱頂 `≥ 0.60%`、收在當根振幅上緣 `≥ 72%`、當根漲幅 `1.5%–7.0%`
- **黃金點火量能與主動買盤**：當根成交額 `≥ 15 萬 USDT`、相對前 20m 均量 `7.5x–55x`（排除 `>55x` 單根消息面極端倒貨針）、主動吃單買盤佔比 `≥ 56%`
- **均線墊高與執行**：`收盤 > MA200`、`MA25 > MA99`、`MA200` 近 1 小時斜率 `≥ 0`；4 分鐘內突破起漲棒高點觸價進場（先跌回箱頂下緣則取消），初始停損守起漲棒低點（上限 `3.0%`），漲滿 `+4.5%` 停損推保本 `+1.0%`、漲滿 `+10%` 改守高點回撤 `7.5%` 吊燈移動停利（最長 60 分鐘）。

實證驗證（已扣 `0.10%` 來回手續費與滑價、同根 K 先判停損）：
- 跨 45 天（10 月訓練 31 天 `PF 1.70`、11/01–11/14 樣本外 14 天 `PF 3.83`）：合計 **48 筆、平均 `+1.945%`、PF `2.27`、累計 `+93.34%`**。
- 全市場 368 檔純幣永續近 7 天（2026-11-07 → 11-14）：**14 筆、勝率 `50.0%`、平均 `+4.11%`、PF `3.97`、累計 `+57.54%`**（含 `EVAAUSDT` `22:10` 起漲進 `1.2075` → `+55.08%`、`CYBERUSDT` `+12.38%`）。

```bash
python3 examples/watch_binance_ignition.py --test                         # 測試 Telegram 通不通
python3 examples/watch_binance_ignition.py                                # 每根 1m 收盤監看（增量快取）
python3 examples/watch_binance_ignition.py --once                         # 只掃剛收盤這一分
python3 examples/watch_binance_ignition.py --symbol EVAAUSDT --replay 240 # 回放單幣抓起漲點並輸出 K 線圖
python3 examples/watch_binance_ignition.py --backtest 7                   # 回測近 7 天全市場表現
python3 examples/test_watch_binance_ignition.py                          # 離線單元測試
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
