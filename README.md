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

## NQ 一分 K Gork多＋線性空｜自動版（v2 雙月驗證門檻）

`pinescript/gork_linear_combo_auto.pine` 掛 NQ1! / MNQ1! 一分圖；`examples/nq_gork_linear_combo.py` 同一套規則的 Yahoo 1m 回測。  
綜合 9/8–10/2 與 10/5–11/3 兩個月樣本，預設優化門檻（Pine 內標 ★，各有 input 可開關）：

- **Gork 15m 破底翻位置與方向過濾**：`close − 15m MA60 > 25` 或 `15m MA60 − MA60[4] > 0` 就跳過。Gork 本質是洗盤後的均值回歸，所有 +100 停利單都發生在 15m MA60 下彎且價格不高於 15m MA60+25 點時。
- **Gork 跌破深度拉高至 70 點**（原 50 點）：50–70 點的淺跌破在近一個月全數停損或保本，0 筆停利；拉到 70 點保留全部 +100 贏單。
- **線性空進場風險上限放寬至 65 點**（原 55 點）：`peakHigh − 進場收盤 > 65` 才跳過，保留 10/15（風險 63.75、+60.5）與 11/4（風險 55.5、+62.97）兩筆贏單，同時擋掉 >65 的極端寬停損。

```bash
python3 examples/nq_gork_linear_combo.py --compare      # 基準 vs 優化，逐筆／逐日
python3 examples/nq_gork_linear_combo.py --baseline     # 原規則
python3 examples/nq_gork_linear_combo.py --csv my_1m.csv --lin-max-risk 65 --gork-deep 70
```

Yahoo 1m 最新一個月（2026-10-05 → 11-03 ET，28,555 根）：
- **基準**：65 筆 **+203.08**（最大回撤 −400.47）｜Gork 54 筆 −265.47（6 停利 / 7 保本 / 41 停損）｜線性空 11 筆 +468.55（9 贏 / 2 停損）
- **優化 v2**：37 筆 **+689.46**（最大回撤 −155.20）｜Gork 26 筆 +220.91（6 停利 / 4 保本 / 16 停損）｜線性空 11 筆 +468.55（9 贏 / 2 停損）
- 擋掉 25 筆 Gork 停損（合計省下 +486.38 點）與 3 筆保本，**6 筆 Gork +100 與 9 筆線性空贏單全數保留（0 誤殺）**。前半月（10/6–10/19）從 −224.5 轉正為 +24.8，後半月（10/20–11/4）從 +427.6 提升至 +664.7。

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

1 小時圖，對齊 6443 元晶 2026-09-16 那波：收盤從 MA20 上方跌到下方後，在下面待 6～36 根（不准假站上），相對過程中最高 MA20 深度 ≥ 3.5%，且最低點是近 16 根新低；破底與翻上不能同一根，從低點至少彈回 3%，再站回 MA20 這波才算。破底後 36 根內，第一根收盤同時大於 MA5 / MA10 / MA20 進場。下面天數放寬是為了 6456 GIS-KY 那種先磨再翻。

回測出場（方便看兩個禮拜成績）：停在破底低、目標 2R、或 20 根時間停。

```bash
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
