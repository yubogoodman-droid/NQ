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

3. **上週五假跌破 1 分 K**  
   https://yubogoodman-droid.github.io/NQ/spring/?v=neck

## 假跌破後上拉（1 分 K）

對應「先盤整、假跌破支撐、迅速站回、再放量上拉」的短線結構（金居 8358 那種）。

```
        放量上拉 ────────●
                       /
        箱體高 ──────●──
                   /
    支撐 ────────●──  站回
              \ /
               ● 假跌破（量不大）
```

| 條件 | 說明 |
|------|------|
| 盤整 | 約 18 根 K 箱體振幅 ≤ 2%，MA5/10/20 糾結 |
| 假跌破 | 跌破箱低 0.4%～3%，持續不久，量能不宜爆量恐慌 |
| 站回 | 很快收盤站回原支撐 |
| 進場 | 站回後 24 根內收盤突破頸線（盤整收盤高，不吃上影），且量能 ≥ 20MA 的 1.2 倍 |
| 停損 | 假跌破最低點 |
| 停利 | 預設 2R |

```bash
python examples/run_fake_breakdown.py --demo
python examples/run_fake_breakdown.py --demo --chart output/spring_demo.html
python3 -m unittest tests.test_fake_breakdown -v
```

台股 CSV 需含 `datetime,open,high,low,close,volume`。

掃描上週五成交額前 50 檔（1 分 K）：

```bash
python examples/scan_tw_top50_spring.py --date 20260814 --limit 50
python examples/scan_tw_top50_spring.py --date 20260814 --after 00:00   # 含開盤後立刻的訊號
python examples/chart_spring_top50.py                                  # 出 1 分 K 報告
```

報告：`docs/spring_top50_20260814.html`

外網直接開（手機可）：  
https://yubogoodman-droid.github.io/NQ/spring/?v=neck

## TradingView

- W 底：`pinescript/nq_w_bottom_5m.pine`，NQ1! / MNQ1! 五分圖
- 假跌破：`pinescript/fake_breakdown_spring.pine`，1 分圖（台股或 NQ 皆可）

## 參數調整

在 `nq/strategy.py` 的 `NQWBottomStrategy` 中：

| 參數 | 預設 | 說明 |
|------|------|------|
| `swing_lookback` | 3 | 轉折確認 K 數 |
| `low_tolerance_pct` | 0.001 | 兩低點價差容忍（0.1%） |
| `min_bars_between_lows` | 5 | 兩低點最少間隔 |
| `max_bars_between_lows` | 60 | 兩低點最多間隔（約 5 小時） |

`FakeBreakdownStrategy` 預設對 1 分 K：箱體 18 根、跌破 0.4%～3%、頸線取盤整收盤高（約箱振幅 80%，不吃上影）、站回後最多等 24 根、突破量能 1.2 倍、停利 2R；**箱體不可跨夜**，並略過開盤後 5 分鐘（濾掉跳空雜訊）。

## 風險提示

本策略僅供學習與回測，不構成投資建議。實盤請搭配流動性時段、滑價、保證金與個人風控。
