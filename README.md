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

## SOXS 一分 K 假突破做空

幣安 `SOXSUSDT` 1 分 K。價格刺過近 120 根前高、超漲不超過 3.5%，接著收盤跌破 MA25 與 MA30，且 MA7 已跌破 MA14。下一到十根內，前一根收盤還在 MA25 下，就在前一根 MA14／MA25 掛空。v3 優化版在 v2 防暴衝基礎上再加入兩項跨 4 檔反向／波動率 ETF（`SOXS` + 外樣本 `SQQQ` / `UVXY` / `TZA`）驗證通過的結構升級：

- **v2 防暴衝與防掃損**：掛單價離下方 MA99 須落在 **0.8%–3.5%** 合理中空帶、前一根 24h 漲幅 **> +20%** 熔斷暫停做空、停損緩衝放寬至假高上方 **0.2%**。
- **v3 過濾 1 根針刺反轉**：假高至跌破至少隔 **2 根**（`min_peak_to_bd=2`），過濾開盤 1 分鐘插針洗盤。
- **v3 條件式瀑布續抱出場**：停利放寬至 **4.0R**；第 30 分鐘檢核時，若收盤仍在 MA25 下方且浮盈 **≥ +0.8R**，則把保護停損移至 **+0.3R** 續抱瀑布（收盤站回 MA25 即落袋、最長至 60 分鐘），其餘盤整單第 30 分鐘準時結算。

只在**美東 09:20–16:00**（冬令台北 22:20–隔日 05:00）掛單進場，停損距離 0.6%–2.5%，來回扣 0.06%，未扣資金費。

```bash
python3 examples/soxs_fake_breakout.py --days 30 --pages
python3 examples/soxs_fake_breakout.py --days 30 --sniper --html docs/soxs-sniper-30d/index.html
python3 examples/soxs_fake_breakout.py --days 140 --ma14-below-ma25   # 極穩空排確認模式（MA14 < MA25）
python3 examples/soxs_fake_breakout.py --days 30 --loose              # 初版寬鬆對照
python3 examples/test_soxs_fake_breakout.py

# 專門抓 S 級大賺單（回測 + 每分鐘即時監看推播 Telegram）
python3 examples/watch_soxs_big_win_tg.py --backtest --days 30
python3 examples/watch_soxs_big_win_tg.py --backtest --days 140 --strict
python3 examples/watch_soxs_big_win_tg.py --test
python3 examples/watch_soxs_big_win_tg.py
```

全上市 4 個多月實測（2026-07-16 → 11-22 台北時間，184,784 根 1 分 K，含 11/13–11/22 真實前進測）＋ 3 檔外樣本 ETF（`SQQQUSDT`、`UVXYUSDT`、`TZAUSDT` 共 59.4 萬根 1 分 K）：
1. 舊版優化 v1（無 MA99 上限／無 24h 熔斷、停損 +0.1%、TP 2.5R、固定 30m）：SOXS 全期 30 筆、勝率 60.0%、淨報酬 +7.96%、+4.17R（PF 1.52、MaxDD 5.03%）；近 30 天 12 筆 +4.31%。
2. 舊版優化 v2（加防暴衝＋停損 +0.2%、TP 2.0R、固定 30m）：SOXS 全期 26 筆、勝率 69.2%、淨報酬 +17.86%、+9.70R（PF 3.56、MaxDD 2.02%）；外樣本 3 ETF 共 26 筆 +4.15%（+5.74R、PF 1.33）。
3. **新版優化主策略 v3**（+ 過濾 1 根針刺 + 30→60m 瀑布續抱 + TP 4.0R）：SOXS 全期 **25 筆、勝率 72.0%、淨報酬 +22.70%、+12.95R**（PF **4.82**、MaxDD 維持 **2.02%**）；近 30 天（10/23–11/22）**9 筆、勝率 77.8%（7 勝 2 負）、淨報酬 +15.13%、+9.07R**；外樣本 3 ETF 共 **26 筆、勝率 57.7%、+9.16%、+11.22R**（PF **1.72**，`SQQQ +2.37%`、`UVXY +6.25%`、`TZA +0.54%` 四檔全正報酬）。
4. **S 級大賺狙擊 v3**（`watch_soxs_big_win_tg.py` / `--sniper`）：SOXS 全期 **15 筆、勝率 80.0%（12 勝 3 負）、淨報酬 +20.07%、+11.54R**（PF **6.33**、MaxDD **2.02%**）；近 30 天 **6 筆、勝率 83.3%（5 勝 1 負）、淨報酬 +14.06%、+8.18R**。
5. **極穩空排確認 v3**（`--ma14-below-ma25` / `--strict`：進場前要求 `MA14 < MA25`）：SOXS 全期 **11 筆、勝率 81.8%（9 勝 2 負、0 筆打到停損）、淨報酬 +18.70%、+9.89R**（PF **21.21**、MaxDD **0.93%**，5/5 時段全正報酬）；外樣本 3 ETF **11 筆、+6.26%、+5.54R**（PF **2.12**）；近 30 天 **3 筆、勝率 100.0%、+9.18%**。

- 優化版預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/soxs-fake-breakout-backtest-274a/docs/soxs-fake-breakout-30d/view.html
- S 級大賺狙擊預覽：https://htmlpreview.github.io/?https://raw.githubusercontent.com/yubogoodman-droid/NQ/cursor/soxs-fake-breakout-backtest-274a/docs/soxs-sniper-30d/view.html

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
