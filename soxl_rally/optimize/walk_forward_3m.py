"""3-month walk-forward & regime analysis for SOXL 1m r3 (2026-07-31 .. 2026-10-31 UTC+8).

Evaluates:
1. How the 1-month candidates (A/B/C/D) perform on the newly fetched out-of-sample months M1 (Aug) and M2 (Sep)!
2. 1D sensitivity across all 3 months (M1, M2, M3 separately and combined).
3. Structural / regime features (volatility ATR/range, drop depth, TP scaling, MA alignment, L2-L1, etc.)
   with strict month-to-month walk-forward (Train M1+M2 -> Test M3, Train M2+M3 -> Test M1, Rolling M_k -> M_{k+1}).
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import r3_engine as E
from r3_engine import DEFAULT, Params, replace

RAW_3M = E.HERE.parent / "backtest_3m" / "bingx_1m_3m_raw.csv"
S3M = pd.Timestamp("2026-07-31 09:30", tz="Asia/Taipei")
E3M = pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")
M1_END = pd.Timestamp("2026-08-31 09:30", tz="Asia/Taipei")
M2_END = pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei")

df = E.load(RAW_3M)
m1_ms, m2_ms = int(M1_END.timestamp() * 1000), int(M2_END.timestamp() * 1000)


def eval_3m(p):
    idx = E.signals(df, p, S3M, E3M)
    T = E.simulate(df, idx, p)
    t_ms = df.t.values[T.i.values] if len(T) else np.array([], dtype="int64")
    s_all = E.stats(T)
    s1 = E.stats(T[t_ms < m1_ms])
    s2 = E.stats(T[(t_ms >= m1_ms) & (t_ms < m2_ms)])
    s3 = E.stats(T[t_ms >= m2_ms])
    s12 = E.stats(T[t_ms < m2_ms])
    return dict(
        n_3m=s_all["n"], wr_3m=round(s_all["win_rate"] * 100, 1) if s_all["n"] else np.nan,
        avg_3m=round(s_all["avg"], 3) if s_all["n"] else np.nan, tot_3m=round(s_all["total"], 2),
        n_m1=s1["n"], avg_m1=round(s1["avg"], 2) if s1["n"] else np.nan, tot_m1=round(s1["total"], 2),
        n_m2=s2["n"], avg_m2=round(s2["avg"], 2) if s2["n"] else np.nan, tot_m2=round(s2["total"], 2),
        n_m3=s3["n"], avg_m3=round(s3["avg"], 2) if s3["n"] else np.nan, tot_m3=round(s3["total"], 2),
        n_m12=s12["n"], avg_m12=round(s12["avg"], 3) if s12["n"] else np.nan, tot_m12=round(s12["total"], 2),
    )


# 1) True out-of-sample test of the 1-month candidates on M1 (Aug) and M2 (Sep)
CANDS_1M = [
    ("0_現行基準 (Baseline)", DEFAULT),
    ("A_收緊MA200距離帶 (band=0.75%)", replace(DEFAULT, band=0.0075)),
    ("B_排除1-tick假墊高 (L2>=L1×1.0005)", replace(DEFAULT, hl_min=0.0005)),
    ("C_時間出場縮為60分 (max_bars=60)", replace(DEFAULT, max_bars=60)),
    ("D_10月組合 (band=0.75%+hl5bp+60m)", replace(DEFAULT, band=0.0075, hl_min=0.0005, max_bars=60)),
]
print("=== 1) 10月挑出的 A/B/C/D 方案在 8月(M1)、9月(M2) 真正樣本外檢驗 ===")
c_df = pd.DataFrame([dict(name=nm, **eval_3m(p)) for nm, p in CANDS_1M])
print(c_df.to_string(index=False))

# 2) One-at-a-time across all 3 months
LEVERS = [
    ("baseline", "default", DEFAULT),
    ("lookback", "60", replace(DEFAULT, lookback=60)),
    ("lookback", "90*", replace(DEFAULT, lookback=90)),
    ("lookback", "120", replace(DEFAULT, lookback=120)),
    ("drop", "1.5%", replace(DEFAULT, drop=0.015)),
    ("drop", "2.0%", replace(DEFAULT, drop=0.020)),
    ("drop", "2.5%*", replace(DEFAULT, drop=0.025)),
    ("drop", "3.0%", replace(DEFAULT, drop=0.030)),
    ("drop", "3.5%", replace(DEFAULT, drop=0.035)),
    ("hl_min (L2墊高)", "0bp*", replace(DEFAULT, hl_min=0.0)),
    ("hl_min (L2墊高)", "5bp (0.05%)", replace(DEFAULT, hl_min=0.0005)),
    ("hl_min (L2墊高)", "10bp (0.10%)", replace(DEFAULT, hl_min=0.0010)),
    ("hl_min (L2墊高)", "15bp (0.15%)", replace(DEFAULT, hl_min=0.0015)),
    ("hl_min (L2墊高)", "20bp (0.20%)", replace(DEFAULT, hl_min=0.0020)),
    ("l2_ma_buf", "0.5%", replace(DEFAULT, l2_ma_buf=0.005)),
    ("l2_ma_buf", "1.0%", replace(DEFAULT, l2_ma_buf=0.010)),
    ("l2_ma_buf", "1.5%*", replace(DEFAULT, l2_ma_buf=0.015)),
    ("l2_ma_buf", "不限(10%)", replace(DEFAULT, l2_ma_buf=0.10)),
    ("band", "0.5%", replace(DEFAULT, band=0.005)),
    ("band", "0.75%", replace(DEFAULT, band=0.0075)),
    ("band", "1.0%*", replace(DEFAULT, band=0.010)),
    ("band", "1.5%", replace(DEFAULT, band=0.015)),
    ("band", "不限(10%)", replace(DEFAULT, band=0.10)),
    ("band_side", "both*", replace(DEFAULT, band_side="both")),
    ("band_side", "above (收>=MA200)", replace(DEFAULT, band_side="above")),
    ("band_side", "below (收<=MA200)", replace(DEFAULT, band_side="below")),
    ("cool", "30m*", replace(DEFAULT, cool=30)),
    ("cool", "60m", replace(DEFAULT, cool=60)),
    ("session", "us 21:30-04:00*", replace(DEFAULT, session="us")),
    ("session", "us前半 21:30-01:00", replace(DEFAULT, session="us_early")),
    ("session", "us後半 01:00-04:00", replace(DEFAULT, session="us_late")),
    ("session", "全天 24h", replace(DEFAULT, session="all")),
    ("tp", "1.0%", replace(DEFAULT, tp=0.010)),
    ("tp", "1.5%", replace(DEFAULT, tp=0.015)),
    ("tp", "2.0%", replace(DEFAULT, tp=0.020)),
    ("tp", "2.5%", replace(DEFAULT, tp=0.025)),
    ("tp", "3.0%*", replace(DEFAULT, tp=0.030)),
    ("tp", "4.0%", replace(DEFAULT, tp=0.040)),
    ("sl", "swing30×0.999*", replace(DEFAULT, sl_mode="swing", sl_bars=30, sl_pad=0.001)),
    ("sl", "max(L2, -1%)", replace(DEFAULT, sl_mode="l2", sl_pct=0.01)),
    ("sl", "固定 -1.0%", replace(DEFAULT, sl_mode="pct", sl_pct=0.010)),
    ("sl", "固定 -1.5%", replace(DEFAULT, sl_mode="pct", sl_pct=0.015)),
    ("max_bars", "30m", replace(DEFAULT, max_bars=30)),
    ("max_bars", "60m", replace(DEFAULT, max_bars=60)),
    ("max_bars", "90m", replace(DEFAULT, max_bars=90)),
    ("max_bars", "120m*", replace(DEFAULT, max_bars=120)),
    ("max_bars", "180m", replace(DEFAULT, max_bars=180)),
]
oat3 = pd.DataFrame([dict(lever=lv, value=val, **eval_3m(p)) for lv, val, p in LEVERS])
oat3.to_csv(E.HERE / "one_at_a_time_3m.csv", index=False)
print("\n=== 2) 三個月單因子敏感度 (M1=8月, M2=9月, M3=10月) ===")
print(oat3[["lever", "value", "n_3m", "wr_3m", "avg_3m", "tot_3m", "n_m1", "tot_m1", "n_m2", "tot_m2", "n_m3", "tot_m3"]].to_string(index=False))

# 3) Joint 3-month grid (576 combos) and month-to-month walk-forward
grid_rows = []
for lookback, drop, hl_min, band, band_side, tp, max_bars in product(
    [60, 90],
    [0.025, 0.030],
    [0.0, 0.0005],
    [0.0075, 0.010, 0.015],
    ["both", "above"],
    [0.020, 0.025, 0.030, 0.040],
    [60, 90, 120],
):
    p = Params(lookback=lookback, drop=drop, hl_min=hl_min, band=band, band_side=band_side, tp=tp, max_bars=max_bars)
    grid_rows.append(dict(lookback=lookback, drop=drop, hl_min=hl_min, band=band, band_side=band_side,
                          tp=tp, max_bars=max_bars, is_default=(p == DEFAULT), **eval_3m(p)))

G3 = pd.DataFrame(grid_rows)
G3.to_csv(E.HERE / "grid_3m.csv", index=False)

# Scale-out exit (half at +1.8% -> move SL to breakeven+0.1%, half at +3.0%) vs fixed TP
cl, lo, hi = df.close.values, df.low.values, df.high.values
swing30 = pd.Series(lo).rolling(30).min().values * 0.999


def sim_scale(idx, tp1=0.018, tp2=0.030, max_bars=90, split=True):
    rows = []
    for i in idx:
        e = cl[i]; sl = swing30[i]; hit1 = False; r1 = r2 = None
        for k in range(i + 1, min(i + 1 + max_bars, len(df))):
            if lo[k] <= sl:
                if not hit1:
                    r1 = r2 = sl / e - 1
                else:
                    r2 = sl / e - 1
                break
            if not hit1 and hi[k] >= e * (1 + tp1):
                hit1 = True; r1 = tp1
                if not split:
                    r2 = tp1; break
                sl = max(sl, e * 1.001)
                if hi[k] >= e * (1 + tp2):
                    r2 = tp2; break
            elif hit1 and hi[k] >= e * (1 + tp2):
                r2 = tp2; break
        if r2 is None:
            k = min(i + max_bars, len(df) - 1)
            rem = cl[k] / e - 1
            if not hit1:
                r1 = r2 = rem
            else:
                r2 = rem
        rows.append(dict(i=int(i), net_pct=0.5 * (r1 + r2) * 100 - 0.1))
    return pd.DataFrame(rows)


STRATS_3M = [
    ("0_現行基準 (90K, both, band=1.0%, TP3%, 120m)", DEFAULT, False, 0.03, 0.03, 120),
    ("1_只改 收>=MA200 (band_side=above)", replace(DEFAULT, band_side="above"), False, 0.03, 0.03, 120),
    ("2_只改 H視窗 60K (lookback=60)", replace(DEFAULT, lookback=60), False, 0.03, 0.03, 120),
    ("3_只改 L2>=L1+5bp (hl_min=0.05%)", replace(DEFAULT, hl_min=0.0005), False, 0.03, 0.03, 120),
    ("4_只改 分批停利 (+1.8%出一半移保本, 另一半+3%)", DEFAULT, True, 0.018, 0.03, 120),
    ("5_穩健進場A (90K + 收>=MA200 + L2>=L1+5bp, band=1.0%) + TP3%/120m",
     Params(lookback=90, hl_min=0.0005, band=0.010, band_side="above"), False, 0.03, 0.03, 120),
    ("6_穩健進場B (60K + 收>=MA200 + L2>=L1+5bp, band=1.0%) + TP3%/90m",
     Params(lookback=60, hl_min=0.0005, band=0.010, band_side="above", max_bars=90), False, 0.03, 0.03, 90),
    ("7_穩健進場C (60K + 收>=MA200 + L2>=L1+5bp, band=1.5%) + TP2.5%/90m",
     Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", tp=0.025, max_bars=90), False, 0.025, 0.025, 90),
    ("8_穩健進場C + 分批停利 (+1.8%半倉保本 / +3.0%半倉, 90m)",
     Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above"), True, 0.018, 0.03, 90),
]

s_rows = []
for nm, p, sp, tp1, tp2, mb in STRATS_3M:
    idx = E.signals(df, p, S3M, E3M)
    T = sim_scale(idx, tp1, tp2, mb, sp)
    t_ms = df.t.values[T.i.values]
    sa = E.stats(T); s1 = E.stats(T[t_ms < m1_ms]); s2 = E.stats(T[(t_ms >= m1_ms) & (t_ms < m2_ms)]); s3 = E.stats(T[t_ms >= m2_ms])
    s_rows.append(dict(strategy=nm, n_3m=sa["n"], wr_3m=round(sa["win_rate"]*100, 1), avg_3m=round(sa["avg"], 2), tot_3m=round(sa["total"], 2),
                       m1_n=s1["n"], m1_tot=round(s1["total"], 2), m2_n=s2["n"], m2_tot=round(s2["total"], 2), m3_n=s3["n"], m3_tot=round(s3["total"], 2)))
S3 = pd.DataFrame(s_rows)
S3.to_csv(E.HERE / "candidates_3m_summary.csv", index=False)
print("\n=== 3) 三個月跨行情穩健方案對照 (M1=8月盤整緩漲, M2=9月緩漲, M3=10月崩跌高波動) ===")
print(S3.to_string(index=False))
