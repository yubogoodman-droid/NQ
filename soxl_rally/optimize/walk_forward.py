"""Walk-forward parameter & rule study for SOXL 1m r3 (起漲點).

Research / backtest only. Never places orders or calls any trading API.

Produces in the current directory:
- one_at_a_time.csv       1D sweep of every lever around the live default (full month + IS/OOS halves)
- grid_is_oos.csv         Modest entry×exit grid ranked on IS, evaluated OOS (plus reverse split)
- rolling_wf.csv          Anchored / rolling weekly walk-forward (train prior weeks -> test next week)
- report.md               Traditional Chinese summary of findings and overfitting checks
"""
from itertools import product
import numpy as np
import pandas as pd
import r3_engine as E
from r3_engine import DEFAULT, Params, replace

df = E.load()
MID = pd.Timestamp("2026-10-16 09:30", tz="Asia/Taipei")   # 15d IS (10/01-10/16) / 15d OOS (10/16-10/31)
WEEKS = [
    ("W1_1001_1008", pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-08 09:30", tz="Asia/Taipei")),
    ("W2_1008_1015", pd.Timestamp("2026-10-08 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-15 09:30", tz="Asia/Taipei")),
    ("W3_1015_1022", pd.Timestamp("2026-10-15 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-22 09:30", tz="Asia/Taipei")),
    ("W4_1022_1031", pd.Timestamp("2026-10-22 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")),
]

# Precompute exit matrix once per signal index for fast exit sweeps
EXIT_GRID = [
    ("swing30_x0.999", dict(sl_mode="swing", sl_bars=30, sl_pad=0.001)),
    ("swing15_x0.999", dict(sl_mode="swing", sl_bars=15, sl_pad=0.001)),
    ("swing60_x0.999", dict(sl_mode="swing", sl_bars=60, sl_pad=0.001)),
    ("l2_or_1pct",     dict(sl_mode="l2", sl_pct=0.01)),
    ("fixed_1.0pct",   dict(sl_mode="pct", sl_pct=0.010)),
    ("fixed_1.5pct",   dict(sl_mode="pct", sl_pct=0.015)),
]


def eval_split(p):
    """Run p once over the month and slice into full / first-half IS / second-half OOS and per-week."""
    idx = E.signals(df, p, E.MONTH_START, E.MONTH_END)
    T = E.simulate(df, idx, p)
    t_ms = df.t.values[T.i.values] if len(T) else np.array([], dtype="int64")
    mid_ms = int(MID.timestamp() * 1000)
    s_all = E.stats(T)
    s_is = E.stats(T[t_ms < mid_ms])
    s_oos = E.stats(T[t_ms >= mid_ms])
    out = dict(
        n_all=s_all["n"], wr_all=round(s_all["win_rate"] * 100, 1) if s_all["n"] else np.nan,
        avg_all=round(s_all["avg"], 3) if s_all["n"] else np.nan, tot_all=round(s_all["total"], 2),
        n_is=s_is["n"], wr_is=round(s_is["win_rate"] * 100, 1) if s_is["n"] else np.nan,
        avg_is=round(s_is["avg"], 3) if s_is["n"] else np.nan, tot_is=round(s_is["total"], 2),
        n_oos=s_oos["n"], wr_oos=round(s_oos["win_rate"] * 100, 1) if s_oos["n"] else np.nan,
        avg_oos=round(s_oos["avg"], 3) if s_oos["n"] else np.nan, tot_oos=round(s_oos["total"], 2),
    )
    for wname, ws, we in WEEKS:
        m = (t_ms >= int(ws.timestamp() * 1000)) & (t_ms < int(we.timestamp() * 1000))
        sw = E.stats(T[m])
        out[f"n_{wname}"] = sw["n"]
        out[f"avg_{wname}"] = round(sw["avg"], 3) if sw["n"] else np.nan
        out[f"tot_{wname}"] = round(sw["total"], 2)
    return out


# 1) One-at-a-time sensitivity around the live rule
LEVERS = [
    ("baseline", "default", DEFAULT),
    ("lookback (H視窗)", "60", replace(DEFAULT, lookback=60)),
    ("lookback (H視窗)", "90*", replace(DEFAULT, lookback=90)),
    ("lookback (H視窗)", "120", replace(DEFAULT, lookback=120)),
    ("drop (H→L1跌幅)", "1.5%", replace(DEFAULT, drop=0.015)),
    ("drop (H→L1跌幅)", "2.0%", replace(DEFAULT, drop=0.020)),
    ("drop (H→L1跌幅)", "2.5%*", replace(DEFAULT, drop=0.025)),
    ("drop (H→L1跌幅)", "3.0%", replace(DEFAULT, drop=0.030)),
    ("drop (H→L1跌幅)", "3.5%", replace(DEFAULT, drop=0.035)),
    ("l2_ma_buf (L2距MA200下限)", "0.5%", replace(DEFAULT, l2_ma_buf=0.005)),
    ("l2_ma_buf (L2距MA200下限)", "1.0%", replace(DEFAULT, l2_ma_buf=0.010)),
    ("l2_ma_buf (L2距MA200下限)", "1.5%*", replace(DEFAULT, l2_ma_buf=0.015)),
    ("l2_ma_buf (L2距MA200下限)", "2.0%", replace(DEFAULT, l2_ma_buf=0.020)),
    ("l2_ma_buf (L2距MA200下限)", "不限(10%)", replace(DEFAULT, l2_ma_buf=0.10)),
    ("band (|收/MA200-1|)", "0.5%", replace(DEFAULT, band=0.005)),
    ("band (|收/MA200-1|)", "0.75%", replace(DEFAULT, band=0.0075)),
    ("band (|收/MA200-1|)", "1.0%*", replace(DEFAULT, band=0.010)),
    ("band (|收/MA200-1|)", "1.5%", replace(DEFAULT, band=0.015)),
    ("band (|收/MA200-1|)", "2.0%", replace(DEFAULT, band=0.020)),
    ("band (|收/MA200-1|)", "不限(10%)", replace(DEFAULT, band=0.10)),
    ("band_side (收在MA200哪側)", "both*", replace(DEFAULT, band_side="both")),
    ("band_side (收在MA200哪側)", "above (收>=MA200)", replace(DEFAULT, band_side="above")),
    ("band_side (收在MA200哪側)", "below (收<=MA200)", replace(DEFAULT, band_side="below")),
    ("cool (冷卻分鐘)", "15m", replace(DEFAULT, cool=15)),
    ("cool (冷卻分鐘)", "30m*", replace(DEFAULT, cool=30)),
    ("cool (冷卻分鐘)", "60m", replace(DEFAULT, cool=60)),
    ("cool (冷卻分鐘)", "120m", replace(DEFAULT, cool=120)),
    ("session (時段)", "us 21:30-04:00*", replace(DEFAULT, session="us")),
    ("session (時段)", "us前半 21:30-01:00", replace(DEFAULT, session="us_early")),
    ("session (時段)", "us後半 01:00-04:00", replace(DEFAULT, session="us_late")),
    ("session (時段)", "全天 24h", replace(DEFAULT, session="all")),
    ("tp (停利)", "1.0%", replace(DEFAULT, tp=0.010)),
    ("tp (停利)", "1.5%", replace(DEFAULT, tp=0.015)),
    ("tp (停利)", "2.0%", replace(DEFAULT, tp=0.020)),
    ("tp (停利)", "2.5%", replace(DEFAULT, tp=0.025)),
    ("tp (停利)", "3.0%*", replace(DEFAULT, tp=0.030)),
    ("tp (停利)", "4.0%", replace(DEFAULT, tp=0.040)),
    ("tp (停利)", "5.0%", replace(DEFAULT, tp=0.050)),
    ("sl (停損方式)", "swing30×0.999*", replace(DEFAULT, sl_mode="swing", sl_bars=30, sl_pad=0.001)),
    ("sl (停損方式)", "swing15×0.999", replace(DEFAULT, sl_mode="swing", sl_bars=15, sl_pad=0.001)),
    ("sl (停損方式)", "swing60×0.999", replace(DEFAULT, sl_mode="swing", sl_bars=60, sl_pad=0.001)),
    ("sl (停損方式)", "max(L2, -1%)", replace(DEFAULT, sl_mode="l2", sl_pct=0.01)),
    ("sl (停損方式)", "固定 -1.0%", replace(DEFAULT, sl_mode="pct", sl_pct=0.010)),
    ("sl (停損方式)", "固定 -1.5%", replace(DEFAULT, sl_mode="pct", sl_pct=0.015)),
    ("max_bars (時間出場)", "30m", replace(DEFAULT, max_bars=30)),
    ("max_bars (時間出場)", "60m", replace(DEFAULT, max_bars=60)),
    ("max_bars (時間出場)", "90m", replace(DEFAULT, max_bars=90)),
    ("max_bars (時間出場)", "120m*", replace(DEFAULT, max_bars=120)),
    ("max_bars (時間出場)", "180m", replace(DEFAULT, max_bars=180)),
    ("max_bars (時間出場)", "240m", replace(DEFAULT, max_bars=240)),
]

oat = pd.DataFrame([dict(lever=lv, value=val, **eval_split(p)) for lv, val, p in LEVERS])
oat.to_csv(E.HERE / "one_at_a_time.csv", index=False)
print("=== One-at-a-time around live default (* = current live / backtest default) ===")
print(oat[["lever", "value", "n_all", "wr_all", "avg_all", "tot_all", "n_is", "avg_is", "tot_is", "n_oos", "avg_oos", "tot_oos"]].to_string(index=False))

# 2) Modest joint grid (3 x 4 x 2 x 2 x 5 x 3 x 3 = 2160 combos) for IS -> OOS walk-forward
grid_rows = []
for drop, band, band_side, cool, tp, (sl_name, sl_kw), max_bars in product(
    [0.020, 0.025, 0.030],
    [0.0075, 0.010, 0.015, 0.020],
    ["both", "above"],
    [30, 60],
    [0.015, 0.020, 0.025, 0.030, 0.040],
    [("swing30", dict(sl_mode="swing", sl_bars=30, sl_pad=0.001)),
     ("l2_1pct", dict(sl_mode="l2", sl_pct=0.01)),
     ("pct1.5",  dict(sl_mode="pct", sl_pct=0.015))],
    [60, 120, 180],
):
    p = Params(drop=drop, band=band, band_side=band_side, cool=cool, tp=tp, max_bars=max_bars, **sl_kw)
    is_default = (p == DEFAULT)
    grid_rows.append(dict(drop=drop, band=band, band_side=band_side, cool=cool, tp=tp, sl=sl_name, max_bars=max_bars,
                          is_default=is_default, **eval_split(p)))

G = pd.DataFrame(grid_rows)
G.to_csv(E.HERE / "grid_is_oos.csv", index=False)

# Filter to combinations with at least min_trades in the training half to avoid 1-trade flukes
MIN_TR = 5
valid_is = G[G.n_is >= MIN_TR].copy()
valid_oos = G[G.n_oos >= MIN_TR].copy()
corr_avg = valid_is[["avg_is", "avg_oos"]].corr().iloc[0, 1]
rank_corr = valid_is[["avg_is", "avg_oos"]].corr(method="spearman").iloc[0, 1]
print(f"\n=== Grid summary ({len(G)} combos; n_is>={MIN_TR}: {len(valid_is)}) ===")
print(f"IS vs OOS avg_pct correlation: Pearson={corr_avg:+.3f}, Spearman={rank_corr:+.3f}")
for side in ("both", "above"):
    sub = valid_is[valid_is.band_side == side]
    print(f"  band_side={side:5s} (n={len(sub)}): IS->OOS Pearson={sub[['avg_is','avg_oos']].corr().iloc[0,1]:+.3f}, "
          f"Spearman={sub[['avg_is','avg_oos']].corr(method='spearman').iloc[0,1]:+.3f}, "
          f"median avg_is={sub.avg_is.median():+.3f}%, median avg_oos={sub.avg_oos.median():+.3f}%")
print(f"Share of all combos with avg_all > 0: {(G.avg_all > 0).mean()*100:.1f}% | avg_is > 0: {(G.avg_is > 0).mean()*100:.1f}% | avg_oos > 0: {(G.avg_oos > 0).mean()*100:.1f}%")

cols = ["drop", "band", "band_side", "cool", "tp", "sl", "max_bars", "is_default", "n_is", "wr_is", "avg_is", "tot_is", "n_oos", "wr_oos", "avg_oos", "tot_oos", "avg_all", "tot_all"]
top_is = valid_is.sort_values(["avg_is", "tot_is"], ascending=False).head(12)
print("\n--- Top 12 selected on FIRST HALF (IS 10/01-10/16, n_is>=5) -> evaluated on SECOND HALF (OOS 10/16-10/31) ---")
print(top_is[cols].to_string(index=False))

top_rev = valid_oos.sort_values(["avg_oos", "tot_oos"], ascending=False).head(12)
print("\n--- Reverse split: Top 12 selected on SECOND HALF (10/16-10/31, n>=5) -> evaluated on FIRST HALF (10/01-10/16) ---")
print(top_rev[cols].to_string(index=False))

print("\n--- Live default row in the grid ---")
print(G[G.is_default][cols].to_string(index=False))

# 3) Anchored & rolling weekly walk-forward (train on prior week(s), test on next week)
wf_rows = []
for k in range(1, len(WEEKS)):
    w_test = WEEKS[k][0]
    for wf_mode in ("anchored_1..k", "rolling_prev1w"):
        train_weeks = [w[0] for w in (WEEKS[:k] if wf_mode == "anchored_1..k" else [WEEKS[k - 1]])]
        n_tr = G[[f"n_{w}" for w in train_weeks]].sum(axis=1)
        tot_tr = G[[f"tot_{w}" for w in train_weeks]].sum(axis=1)
        avg_tr = tot_tr / n_tr.replace(0, np.nan)
        cand = G.assign(n_tr=n_tr, tot_tr=tot_tr, avg_tr=avg_tr).query("n_tr >= 3")
        best = cand.sort_values(["avg_tr", "tot_tr"], ascending=False).iloc[0]
        base = G[G.is_default].iloc[0]
        wf_rows.append(dict(
            wf_mode=wf_mode, train="+".join(train_weeks), test=w_test,
            picked=f"drop={best['drop']},band={best['band']},{best['band_side']},cool={int(best['cool'])},tp={best['tp']},sl={best['sl']},max={int(best['max_bars'])}",
            train_n=int(best["n_tr"]), train_avg=round(best["avg_tr"], 3), train_tot=round(best["tot_tr"], 2),
            test_n_picked=int(best[f"n_{w_test}"]), test_avg_picked=best[f"avg_{w_test}"], test_tot_picked=best[f"tot_{w_test}"],
            test_n_default=int(base[f"n_{w_test}"]), test_avg_default=base[f"avg_{w_test}"], test_tot_default=base[f"tot_{w_test}"],
        ))
WF = pd.DataFrame(wf_rows)
WF.to_csv(E.HERE / "rolling_wf.csv", index=False)
print("\n=== Weekly walk-forward (train -> next week OOS vs untouched live default) ===")
print(WF.to_string(index=False))
for m in ("anchored_1..k", "rolling_prev1w"):
    sub = WF[WF["wf_mode"] == m]
    print(f"{m:15s} OOS sum (W2..W4): picked={sub['test_tot_picked'].sum():+.2f}% (n={int(sub['test_n_picked'].sum())}) "
          f"vs default={sub['test_tot_default'].sum():+.2f}% (n={int(sub['test_n_default'].sum())})")
