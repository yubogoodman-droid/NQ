"""4-month walk-forward, LOMO cross-validation, and slippage stress test for SOXL 1m r3.

Period: 2026-07-31 09:30 .. 2026-11-23 09:30 UTC+8 (170,121 bars, 0 gaps).
Includes M4_11 (2026-10-31 09:30 .. 2026-11-23 09:30 UTC+8) as a true prospective OOS check
against the Oct 31 study.

Research / backtest only: never places orders, never calls trading APIs, never modifies live_rules/.

Outputs:
- one_at_a_time_4m.csv       1D sensitivity across M1..M4
- grid_4m.csv                864-combo structured grid across M1..M4
- wf_4m.csv                  Expanding walk-forward + Leave-One-Month-Out (LOMO) CV
- candidates_4m_summary.csv  Conservative candidate table with drawdown, worst loss & slippage stress
"""
from itertools import product
import numpy as np
import pandas as pd
import r3_engine as E
from r3_engine import DEFAULT, Params, replace

RAW_4M = E.HERE.parent / "backtest_4m" / "bingx_1m_4m_raw.csv"
S4M = pd.Timestamp("2026-07-31 09:30", tz="Asia/Taipei")
E4M = pd.Timestamp("2026-11-23 09:30", tz="Asia/Taipei")
MONTHS = [
    ("M1_08", pd.Timestamp("2026-07-31 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-08-31 09:30", tz="Asia/Taipei")),
    ("M2_09", pd.Timestamp("2026-08-31 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei")),
    ("M3_10", pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")),
    ("M4_11", pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-11-23 09:30", tz="Asia/Taipei")),
]
M_MS = [(nm, int(s.timestamp() * 1000), int(e.timestamp() * 1000)) for nm, s, e in MONTHS]

df = E.load(RAW_4M)


def max_dd_and_streak(rets):
    if len(rets) == 0:
        return 0.0, 0
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    mdd = float((eq - peak).min())
    cur = best = 0
    for r in rets:
        if r <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return round(mdd, 2), best


def eval_4m(p):
    T = E.run(df, p, S4M, E4M)
    t_ms = df.t.values[T.i.values] if len(T) else np.array([], dtype="int64")
    sa = E.stats(T)
    mdd, max_loss_streak = max_dd_and_streak(T.net_pct.values)
    out = dict(
        n_4m=sa["n"], wr_4m=round(sa["win_rate"] * 100, 1) if sa["n"] else np.nan,
        avg_4m=round(sa["avg"], 3) if sa["n"] else np.nan, tot_4m=round(sa["total"], 2),
        worst_4m=round(sa["worst"], 2) if sa["n"] else np.nan, mdd_4m=mdd, max_streak=max_loss_streak,
    )
    pos_months = 0
    for nm, s_ms, e_ms in M_MS:
        sm = E.stats(T[(t_ms >= s_ms) & (t_ms < e_ms)])
        out[f"n_{nm}"] = sm["n"]
        out[f"avg_{nm}"] = round(sm["avg"], 3) if sm["n"] else np.nan
        out[f"tot_{nm}"] = round(sm["total"], 2)
        if sm["total"] > 0:
            pos_months += 1
    out["pos_months"] = pos_months
    return out


# 1) One-at-a-time sensitivity across all 4 months
LEVERS = [
    ("baseline", "default*", DEFAULT),
    ("lookback", "45", replace(DEFAULT, lookback=45)),
    ("lookback", "60", replace(DEFAULT, lookback=60)),
    ("lookback", "75", replace(DEFAULT, lookback=75)),
    ("lookback", "90*", replace(DEFAULT, lookback=90)),
    ("lookback", "120", replace(DEFAULT, lookback=120)),
    ("drop", "2.0%", replace(DEFAULT, drop=0.020)),
    ("drop", "2.5%*", replace(DEFAULT, drop=0.025)),
    ("drop", "3.0%", replace(DEFAULT, drop=0.030)),
    ("drop", "3.5%", replace(DEFAULT, drop=0.035)),
    ("hl_min", "0bp*", replace(DEFAULT, hl_min=0.0)),
    ("hl_min", "5bp (0.05%)", replace(DEFAULT, hl_min=0.0005)),
    ("hl_min", "10bp (0.10%)", replace(DEFAULT, hl_min=0.0010)),
    ("hl_min", "15bp (0.15%)", replace(DEFAULT, hl_min=0.0015)),
    ("l2_ma_buf", "1.0%", replace(DEFAULT, l2_ma_buf=0.010)),
    ("l2_ma_buf", "1.5%*", replace(DEFAULT, l2_ma_buf=0.015)),
    ("l2_ma_buf", "2.0%", replace(DEFAULT, l2_ma_buf=0.020)),
    ("l2_ma_buf", "不限(10%)", replace(DEFAULT, l2_ma_buf=0.10)),
    ("band", "0.75%", replace(DEFAULT, band=0.0075)),
    ("band", "1.0%*", replace(DEFAULT, band=0.010)),
    ("band", "1.5%", replace(DEFAULT, band=0.015)),
    ("band", "2.0%", replace(DEFAULT, band=0.020)),
    ("band", "不限(10%)", replace(DEFAULT, band=0.10)),
    ("band_side", "both*", replace(DEFAULT, band_side="both")),
    ("band_side", "above (收>=MA200)", replace(DEFAULT, band_side="above")),
    ("band_side", "below (收<=MA200)", replace(DEFAULT, band_side="below")),
    ("htf_ma", "0 (不限)*", replace(DEFAULT, htf_ma=0)),
    ("htf_ma", "300 (~5m MA60)", replace(DEFAULT, htf_ma=300)),
    ("htf_ma", "495 (~5m MA99)", replace(DEFAULT, htf_ma=495)),
    ("htf_ma", "600 (~5m MA120)", replace(DEFAULT, htf_ma=600)),
    ("htf_ma", "1000 (~5m MA200)", replace(DEFAULT, htf_ma=1000)),
    ("sl_cap", "不限*", replace(DEFAULT, sl_cap=0.0)),
    ("sl_cap", "2.5%", replace(DEFAULT, sl_cap=0.025)),
    ("sl_cap", "2.0%", replace(DEFAULT, sl_cap=0.020)),
    ("sl_cap", "1.5%", replace(DEFAULT, sl_cap=0.015)),
    ("max_sl_skip", "不限*", replace(DEFAULT, max_sl_skip=0.0)),
    ("max_sl_skip", "2.5%", replace(DEFAULT, max_sl_skip=0.025)),
    ("max_sl_skip", "2.0%", replace(DEFAULT, max_sl_skip=0.020)),
    ("session", "us 21:30-04:00*", replace(DEFAULT, session="us")),
    ("session", "全天 24h", replace(DEFAULT, session="all")),
    ("tp", "1.5%", replace(DEFAULT, tp=0.015)),
    ("tp", "2.0%", replace(DEFAULT, tp=0.020)),
    ("tp", "2.5%", replace(DEFAULT, tp=0.025)),
    ("tp", "3.0%*", replace(DEFAULT, tp=0.030)),
    ("tp", "4.0%", replace(DEFAULT, tp=0.040)),
    ("max_bars", "60m", replace(DEFAULT, max_bars=60)),
    ("max_bars", "90m", replace(DEFAULT, max_bars=90)),
    ("max_bars", "120m*", replace(DEFAULT, max_bars=120)),
    ("max_bars", "180m", replace(DEFAULT, max_bars=180)),
]
oat = pd.DataFrame([dict(lever=lv, value=val, **eval_4m(p)) for lv, val, p in LEVERS])
oat.to_csv(E.HERE / "one_at_a_time_4m.csv", index=False)
print("=== 1) 四個月單因子敏感度 (M1=8月, M2=9月, M3=10月, M4=11月OOS) ===")
print(oat[["lever", "value", "n_4m", "wr_4m", "avg_4m", "tot_4m", "tot_M1_08", "tot_M2_09", "tot_M3_10", "tot_M4_11", "worst_4m"]].to_string(index=False))

# 2) Structured 864-combo grid for strict Expanding WF and Leave-One-Month-Out CV
grid_rows = []
for lookback, hl_min, band, band_side, htf_ma, sl_cap, tp, max_bars in product(
    [60, 90],
    [0.0, 0.0005],
    [0.010, 0.015],
    ["both", "above"],
    [0, 495, 600],
    [0.0, 0.020],
    [0.020, 0.025, 0.030],
    [60, 90, 120],
):
    p = Params(lookback=lookback, hl_min=hl_min, band=band, band_side=band_side,
               htf_ma=htf_ma, sl_cap=sl_cap, tp=tp, max_bars=max_bars)
    grid_rows.append(dict(lookback=lookback, hl_min=hl_min, band=band, band_side=band_side,
                          htf_ma=htf_ma, sl_cap=sl_cap, tp=tp, max_bars=max_bars,
                          is_default=(p == DEFAULT), **eval_4m(p)))
G = pd.DataFrame(grid_rows)
G.to_csv(E.HERE / "grid_4m.csv", index=False)

print(f"\n=== 2) 網格統計 ({len(G)} 組) ===")
print(f"4M tot > 0: {(G.tot_4m > 0).mean()*100:.1f}% | 4 個月全正: {(G.pos_months == 4).sum()} 組 ({(G.pos_months == 4).mean()*100:.1f}%) | 至少 3 個月為正: {(G.pos_months >= 3).sum()} 組 ({(G.pos_months >= 3).mean()*100:.1f}%)")
for col in ["band_side", "htf_ma", "sl_cap", "lookback", "tp"]:
    print(f"\nMedian by {col}:")
    print(G.groupby(col)[["tot_M1_08", "tot_M2_09", "tot_M3_10", "tot_M4_11", "tot_4m", "avg_4m"]].median().round(2).to_string())

# 3) Expanding Walk-Forward (M1->M2, M1+M2->M3, M1+M2+M3->M4) AND Leave-One-Month-Out (LOMO) CV
base_row = G[G.is_default].iloc[0]
m_names = [m[0] for m in MONTHS]
wf_rows = []


def run_fold(kind, train_ms, test_m, min_tr=8):
    n_tr = G[[f"n_{m}" for m in train_ms]].sum(axis=1)
    tot_tr = G[[f"tot_{m}" for m in train_ms]].sum(axis=1)
    avg_tr = tot_tr / n_tr.replace(0, np.nan)
    cand = G.assign(n_tr=n_tr, tot_tr=tot_tr, avg_tr=avg_tr).query(f"n_tr >= {min_tr}").copy()
    # Rank correlation between train avg and OOS test avg
    valid_corr = cand[cand[f"n_{test_m}"] >= 2]
    sp_corr = valid_corr[["avg_tr", f"avg_{test_m}"]].corr(method="spearman").iloc[0, 1]
    # Top-1 by train avg_tr
    top1 = cand.sort_values(["avg_tr", "tot_tr"], ascending=False).iloc[0]
    # Top-5% plateau average OOS (to measure robust neighborhood vs single peak)
    top5p = cand.sort_values(["avg_tr", "tot_tr"], ascending=False).head(max(1, len(cand) // 20))
    wf_rows.append(dict(
        kind=kind, train="+".join(train_ms), test=test_m, spearman_is_oos=round(sp_corr, 3),
        top1_params=f"lb={int(top1.lookback)},hl={top1.hl_min},b={top1.band},{top1.band_side},htf={int(top1.htf_ma)},cap={top1.sl_cap},tp={top1.tp},mb={int(top1.max_bars)}",
        tr_n=int(top1.n_tr), tr_tot=round(top1.tot_tr, 2),
        oos_top1_n=int(top1[f"n_{test_m}"]), oos_top1_tot=round(top1[f"tot_{test_m}"], 2),
        oos_top5pct_mean_tot=round(top5p[f"tot_{test_m}"].mean(), 2),
        oos_base_n=int(base_row[f"n_{test_m}"]), oos_base_tot=round(base_row[f"tot_{test_m}"], 2),
    ))


for k in range(1, 4):
    run_fold("expanding_wf", m_names[:k], m_names[k], min_tr=5 if k == 1 else 10)
for hold in m_names:
    run_fold("lomo_cv", [m for m in m_names if m != hold], hold, min_tr=12)

WF = pd.DataFrame(wf_rows)
WF.to_csv(E.HERE / "wf_4m.csv", index=False)
print("\n=== 3) 嚴格 Walk-Forward 與 Leave-One-Month-Out (LOMO) 交叉驗證 ===")
print(WF.to_string(index=False))
for kind in ("expanding_wf", "lomo_cv"):
    sub = WF[WF.kind == kind]
    print(f"{kind:12s} OOS 合計: Top-1={sub.oos_top1_tot.sum():+.2f}% (n={int(sub.oos_top1_n.sum())}) | "
          f"Top-5%均值={sub.oos_top5pct_mean_tot.sum():+.2f}% | 原預設={sub.oos_base_tot.sum():+.2f}% (n={int(sub.oos_base_n.sum())}) | "
          f"平均 Spearman={sub.spearman_is_oos.mean():+.3f}")

# 4) Conservative Candidates + Risk Metrics + Slippage Stress Tests
CANDS = [
    ("0_現行基準 (90K, both, band=1.0%, TP3%, 120m)", DEFAULT),
    ("1_只改 收>=MA200 (band_side=above)", replace(DEFAULT, band_side="above")),
    ("2_只改 H視窗 60K (lookback=60)", replace(DEFAULT, lookback=60)),
    ("3_只改 SL上限 2.0% (sl_cap=2.0%)", replace(DEFAULT, sl_cap=0.020)),
    ("4_只改 大級別均線 close>=MA600 (~5m MA120)", replace(DEFAULT, htf_ma=600)),
    ("5_原10/31方案C (60K+above+hl5bp, b1.5%, TP2.5%, 90m)", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", tp=0.025, max_bars=90)),
    ("6_方案C + SL上限2.0% (sl_cap=2.0%)", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", tp=0.025, max_bars=90, sl_cap=0.020)),
    ("7_方案C + 略過寬停損 (max_sl_skip=2.5%)", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", tp=0.025, max_bars=90, max_sl_skip=0.025)),
    ("8_方案C + 站上5m MA120 (htf_ma=600)", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", htf_ma=600, tp=0.025, max_bars=90)),
    ("9_方案C + 5m MA120 + SL上限2.0%", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", htf_ma=600, sl_cap=0.020, tp=0.025, max_bars=90)),
    ("10_方案C + 5m MA99 (htf_ma=495) + SL上限2.0%", Params(lookback=60, hl_min=0.0005, band=0.015, band_side="above", htf_ma=495, sl_cap=0.020, tp=0.025, max_bars=90)),
]
c_rows = []
for nm, p in CANDS:
    r = eval_4m(p)
    s_fee02 = eval_4m(replace(p, fee_pct=0.2))["tot_4m"]
    s_fee03 = eval_4m(replace(p, fee_pct=0.3))["tot_4m"]
    c_rows.append(dict(strategy=nm, n=r["n_4m"], wr=r["wr_4m"], avg=round(r["avg_4m"], 2),
                       tot_fee01=r["tot_4m"], tot_fee02=s_fee02, tot_fee03=s_fee03,
                       mdd=r["mdd_4m"], worst=r["worst_4m"], max_loss_streak=r["max_streak"],
                       M1_08=f"{r['tot_M1_08']:+.2f}({r['n_M1_08']})",
                       M2_09=f"{r['tot_M2_09']:+.2f}({r['n_M2_09']})",
                       M3_10=f"{r['tot_M3_10']:+.2f}({r['n_M3_10']})",
                       M4_11=f"{r['tot_M4_11']:+.2f}({r['n_M4_11']})",
                       pos_m=f"{r['pos_months']}/4"))
C4 = pd.DataFrame(c_rows)
C4.to_csv(E.HERE / "candidates_4m_summary.csv", index=False)
print("\n=== 4) 四個月穩健候選方案總表（含最大回撤、連敗、與 0.2%/0.3% 滑價壓力測試） ===")
print(C4.to_string(index=False))
