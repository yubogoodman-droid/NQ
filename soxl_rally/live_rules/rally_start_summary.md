# SOXL 1m 起漲點 rule backtest (BingX NCSKSOXL2USD-USDT)

Data: BingX swap 1m, 86,341 bars, 2026-08-10 12:58 → 2026-10-09 11:58 (UTC+8), ~60 days, no gaps (24/7 perp).
Cache: bingx_1m.csv (refresh: fetch_bingx_1m.py). Script: backtest_rally_start.py. Signals: rally_start_signals.csv. Stats: rally_start_summary.csv, rally_start_split.csv. Chart: rally_start_1008.png.
Entry = close of signal bar. Exits: TP +2%/+3% vs SL = 30-bar swing low −0.1% ("swing") or −1% ("pct1"); 120-min timeout; SL assumed first if both hit in one bar. No fees (≈0.1% round trip would come off every trade).

## 10/8 event on BingX (UTC+8, bar open times)
- 21:54 capitulation bar 148.44→147.09, vol 1133 (10.7× 20-bar avg).
- 21:55 low **146.59** (BingX matches Binance low). MA200 149.56, MA99 149.94 → low 2.0% below MA200; 3.3% below 90-bar high.
- 21:56–21:58 three green bars; 22:00 higher low 147.52; 22:02 green 148.44→149.08 on 3.1× vol breaks swing high 148.18.
- 22:03 close > MA25; 22:05–22:06 close > MA99/120/200 cluster (~149.9); 22:07 MA7 crosses MA25.
- 22:29 high **155.38** (+6.0% from low in 34 min). The MA99 150.75 / MA120 150.54 / MA200 150.10 values in the screenshot match BingX at 22:36.

## Rules (all long, dip = low ≥1.5% below MA200 within last 30 bars, 60-bar cooldown)
- A: first close back above MA25 with MA7 rising
- B: MA7 crosses above MA25 while MA25 < MA200, a ≥2× vol bar in last 10
- C: first close above max(MA99,MA120,MA200) (dip within 60 bars)
- D: higher low (2-bar pivot) after the dip low, then close above the swing high between the low and the HL
- E: capitulation bar (dip + vol ≥4×) in last 15 bars, then first close above MA14 with MA7 rising
- "+F" filter: US regular hours (09:30–16:00 NY) AND dip low ≥2.5% below the 90-bar high (sharp drop)

## Results (TP3 / swing SL unless noted)
| rule | n | win | avgR | avg ret% | TP2 swing win/avgR | maxgain60 / maxdd60 | 10/8 signal |
|---|---|---|---|---|---|---|---|
| A | 177 | 40% | 0.10 | 0.11 | 40% / 0.05 | 1.12 / −1.09 | 22:02 @149.08 (+7m) |
| B | 131 | 35% | −0.04 | 0.00 | 36% / −0.11 | 0.98 / −1.20 | 22:06 @150.73 (+11m) |
| C | 75 | 47% | 0.02 | 0.15 | 51% / 0.01 | 1.33 / −1.31 | 22:05 @149.95 (+10m) |
| D | 164 | 41% | 0.06 | 0.05 | 41% / 0.02 | 1.18 / −1.12 | 22:02 @149.08 (+7m) |
| E | 138 | 33% | 0.04 | 0.01 | 33% / −0.04 | 1.12 / −1.24 | 22:01 @148.44 (+6m) |
| A+F | 47 | 55% | 0.27 | 0.36 | 55% / 0.19 | 1.61 / −1.50 | 22:02 @149.08 (+7m) |
| B+F | 33 | 55% | 0.45 | 0.41 | 58% / 0.35 | 1.54 / −1.61 | 22:06 @150.73 (+11m) |
| C+F | 29 | 52% | 0.06 | 0.09 | 55% / 0.08 | 1.77 / −1.56 | 22:05 @149.95 (+10m) |
| E+F | 40 | 45% | 0.35 | 0.28 | 45% / 0.20 | 1.66 / −1.88 | 22:01 @148.44 (+6m) |
| **D+F** | 45 | **62%** | 0.33 | **0.45** | 62% / 0.25 | 1.76 / **−1.33** | 22:02 @149.08 (+7m) |
| baseline: any RTH bar | 17160 | | | | | 1.00 / −1.04 | |

Split-half avgR (Aug10–Sep9 / Sep10–Oct9): D+F 0.28/0.39, A+F 0.20/0.38, B+F 0.15/0.77, E+F 0.05/0.98, C+F −0.11/0.27. Unfiltered rules ≈ 0 in both halves.
All rules hit TP +3% on 10/8.

## Recommendation
D+F (higher low + swing-high break after a sharp dip below MA200, US session only): best win rate, smallest drawdown, positive in both halves, ~1 signal/trading day, caught 10/8 at 22:02 @149.08, 7 min after the low. Stop = swing low (avg risk ~2%). E+F fires 1 bar earlier but has a deeper avg DD and an unstable H1.
Caveats: the "+F" filter was chosen after looking at the same data (in-sample); n≈45 is small; no fees/slippage; BingX ≠ Binance prices.
