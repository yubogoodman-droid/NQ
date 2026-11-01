"""Check r3_engine with default parameters against the original month backtest (signals.csv / signals_allday.csv)."""
import sys
import pandas as pd
import r3_engine as E

df = E.load()
ok = True
for name, p in [("signals.csv", E.DEFAULT), ("signals_allday.csv", E.replace(E.DEFAULT, session="all"))]:
    ref = pd.read_csv(E.HERE.parent / "backtest_month" / name)
    T = E.run(df, p)
    same_time = list(T.time) == list(ref.time)
    max_diff = (T.net_pct.round(3).values - ref.net_pct.values).__abs__().max() if same_time else float("nan")
    same_exit = same_time and list(T.exit_reason) == list(ref.exit_reason)
    s = E.stats(T)
    print(f"{name}: engine n={s['n']} total={s['total']:+.2f}% | ref n={len(ref)} total={ref.net_pct.sum():+.2f}% | "
          f"times match={same_time} exits match={same_exit} max |net diff|={max_diff:.4f}")
    ok &= same_time and same_exit and max_diff < 1e-3
sens = E.run(df, E.replace(E.DEFAULT, tp=0.01, sl_mode="l2", sl_pct=0.01))
ref = pd.read_csv(E.HERE.parent / "backtest_month" / "signals.csv")
d = (sens.net_pct.round(3).values - ref.sens_net_pct.values).__abs__().max()
print(f"sensitivity TP1%: engine total={sens.net_pct.sum():+.2f}% ref total={ref.sens_net_pct.sum():+.2f}% max diff={d:.4f}")
ok &= d < 1e-3
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
