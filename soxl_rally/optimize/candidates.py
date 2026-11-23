"""Trade-by-trade diagnosis and conservative candidate comparison for SOXL 1m r3 (起漲點).

Research / backtest only: never places orders or modifies live_rules/.
Compares the untouched live baseline against 4 low-DOF structural improvements across:
  - Full month (2026-10-01 09:30 .. 2026-10-31 09:30 UTC+8)
  - First half IS (10/01 .. 10/16) and second half OOS (10/16 .. 10/31)
  - All 4 individual weeks (W1 .. W4)
Outputs `candidates_summary.csv` and `trade_diagnostics.csv`.
"""
import numpy as np
import pandas as pd
import r3_engine as E
from r3_engine import DEFAULT, replace

df = E.load()
s = E.structure(df, 90)
MID = pd.Timestamp("2026-10-16 09:30", tz="Asia/Taipei")
WEEKS = [
    ("W1(10/01-08)", pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-08 09:30", tz="Asia/Taipei")),
    ("W2(10/08-15)", pd.Timestamp("2026-10-08 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-15 09:30", tz="Asia/Taipei")),
    ("W3(10/15-22)", pd.Timestamp("2026-10-15 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-22 09:30", tz="Asia/Taipei")),
    ("W4(10/22-31)", pd.Timestamp("2026-10-22 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")),
]

CANDIDATES = [
    ("0_現行基準 (Baseline)", DEFAULT),
    ("A_收緊MA200距離帶 (band=0.75%)", replace(DEFAULT, band=0.0075)),
    ("B_排除1-tick假墊高 (L2>=L1×1.0005)", replace(DEFAULT, hl_min=0.0005)),
    ("C_時間出場縮為60分 (max_bars=60)", replace(DEFAULT, max_bars=60)),
    ("D_組合微調 (band=0.75% + L2>=L1+5bp + 60m)", replace(DEFAULT, band=0.0075, hl_min=0.0005, max_bars=60)),
]

rows = []
for name, p in CANDIDATES:
    T = E.run(df, p)
    t_ms = df.t.values[T.i.values]
    mid_ms = int(MID.timestamp() * 1000)
    sa, sis, soos = E.stats(T), E.stats(T[t_ms < mid_ms]), E.stats(T[t_ms >= mid_ms])
    r = dict(
        candidate=name,
        n_all=sa["n"], wr_all=round(sa["win_rate"] * 100, 1), avg_all=round(sa["avg"], 2), tot_all=round(sa["total"], 2),
        n_is=sis["n"], wr_is=round(sis["win_rate"] * 100, 1), avg_is=round(sis["avg"], 2), tot_is=round(sis["total"], 2),
        n_oos=soos["n"], wr_oos=round(soos["win_rate"] * 100, 1), avg_oos=round(soos["avg"], 2), tot_oos=round(soos["total"], 2),
    )
    for wname, ws, we in WEEKS:
        m = (t_ms >= int(ws.timestamp() * 1000)) & (t_ms < int(we.timestamp() * 1000))
        sw = E.stats(T[m])
        r[f"{wname}_n"] = sw["n"]
        r[f"{wname}_tot"] = round(sw["total"], 2)
    rows.append(r)

S = pd.DataFrame(rows)
S.to_csv(E.HERE / "candidates_summary.csv", index=False)
print("=== Candidate comparison (IS / OOS / Weekly) ===")
print(S.to_string(index=False))

# Trade-by-trade table on the 19 baseline signals
T0 = E.run(df, DEFAULT)
T60 = E.run(df, replace(DEFAULT, max_bars=60))
idx = T0.i.values
T0["H"] = s["H"][idx].round(2)
T0["L1"] = s["L1"][idx].round(2)
T0["L2"] = s["L2"][idx].round(2)
T0["hl_bp"] = ((s["L2"][idx] / s["L1"][idx] - 1) * 10000).round(1)
T0["dist_ma200_pct"] = ((df.close.values[idx] / df.ma200.values[idx] - 1) * 100).round(2)
T0["net_120m"] = T0.net_pct.round(2)
T0["net_60m"] = T60.net_pct.round(2)
T0["pass_A_band075"] = np.abs(df.close.values[idx] / df.ma200.values[idx] - 1) <= 0.0075
T0["pass_B_hl5bp"] = s["L2"][idx] >= s["L1"][idx] * 1.0005
T0["pass_D_both"] = T0["pass_A_band075"] & T0["pass_B_hl5bp"]
cols = ["time", "entry", "L1", "L2", "hl_bp", "dist_ma200_pct", "exit_reason", "hold_min", "net_120m", "net_60m",
        "pass_A_band075", "pass_B_hl5bp", "pass_D_both"]
T0[cols].to_csv(E.HERE / "trade_diagnostics.csv", index=False)
print("\n=== 19 Baseline Trades with Structural Diagnostics ===")
print(T0[cols].to_string(index=False))
