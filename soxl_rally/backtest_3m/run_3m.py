"""3-month backtest of SOXL 1m r3 (起漲點：急跌後更高低點突破), 2026-07-31 09:30 .. 2026-10-31 09:30 UTC+8.

Replays live_rules/rules.py eval_1m (r3) directly (and verifies 100% match with optimize/r3_engine.py).
Research / backtest only: never places orders or modifies live_rules/.
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import types
sys.path.insert(0, str(HERE.parent / "live_rules"))
sys.path.insert(0, str(HERE.parent / "optimize"))
# rules.py imports r8/r9 helper modules at top level; stub them out as no-ops without modifying live_rules/rules.py
sys.modules.setdefault("h4high_ma30_rule", types.SimpleNamespace(eval_1m=lambda *a, **k: None))
sys.modules.setdefault("h4low_ma_rule", types.SimpleNamespace(D1=None, eval_5m=lambda *a, **k: None))
import rules as LR  # noqa: E402
import r3_engine as E  # noqa: E402

RAW_3M = HERE / "bingx_1m_3m_raw.csv"
S3M = pd.Timestamp("2026-07-31 09:30", tz="Asia/Taipei")
E3M = pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")
M_BOUNDS = [
    ("M1_08 (07/31-08/31)", pd.Timestamp("2026-07-31 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-08-31 09:30", tz="Asia/Taipei")),
    ("M2_09 (08/31-10/01)", pd.Timestamp("2026-08-31 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei")),
    ("M3_10 (10/01-10/31)", pd.Timestamp("2026-10-01 09:30", tz="Asia/Taipei"), pd.Timestamp("2026-10-31 09:30", tz="Asia/Taipei")),
]

raw = pd.read_csv(RAW_3M)
d1 = raw.rename(columns={"time": "t"})[["t", "open", "high", "low", "close"]].reset_index(drop=True)
for n in (7, 14, 25, 30, 200):
    d1[f"ma{n}"] = d1.close.rolling(n).mean()
d1["dt"] = pd.to_datetime(d1.t, unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
d1["swinglow"] = d1.low.rolling(30).min()


def replay_live(allday=False):
    orig = LR.us_session
    if allday:
        LR.us_session = lambda ms: True
    st = LR.default_state()
    hits = []

    def cap(sig, t, title, body):
        if sig == "r3":
            hits.append((int(t), body))

    for i in range(1, len(d1)):
        LR.eval_1m(d1, i, st, cap)
    LR.us_session = orig
    return hits


t2i = {int(t): i for i, t in enumerate(d1.t)}
cl, lo, hi = d1.close.values, d1.low.values, d1.high.values


def sim(i, tp, sl, maxb=120):
    e = cl[i]
    for k in range(i + 1, min(i + 1 + maxb, len(d1))):
        if lo[k] <= sl:
            return sl / e - 1, "停損", k
        if hi[k] >= e * (1 + tp):
            return tp, "停利", k
    k = min(i + maxb, len(d1) - 1)
    return cl[k] / e - 1, "時間出場", k


def build_table(hits):
    out = []
    for t, body in hits:
        i = t2i[t]
        tt = d1.dt.iat[i]
        if not (S3M <= tt < E3M):
            continue
        sw = d1.swinglow.iat[i] * 0.999
        ret, why, k = sim(i, 0.03, sw)
        L2 = float(re.search(r"低2 ([\d.]+)", body).group(1))
        sl2 = max(L2, cl[i] * 0.99)
        ret2, why2, k2 = sim(i, 0.01, sl2)
        m_label = next(m[0] for m in M_BOUNDS if m[1] <= tt < m[2])
        out.append(dict(
            time=tt.strftime("%Y-%m-%d %H:%M"), month=m_label,
            week=(tt - pd.Timedelta(hours=6)).strftime("%G-W%V"),
            session_day=(tt - pd.Timedelta(hours=6)).strftime("%m-%d"),
            body=body, entry=cl[i], sl=round(sw, 3), tp=round(cl[i] * 1.03, 3),
            exit_reason=why, exit_time=d1.dt.iat[k].strftime("%m-%d %H:%M"),
            exit_px=round(cl[i] * (1 + ret), 3), hold_min=k - i,
            gross_pct=round(ret * 100, 3), net_pct=round(ret * 100 - 0.1, 3),
            L2=L2, sens_net_pct=round(ret2 * 100 - 0.1, 3), sens_exit=why2, sens_hold=k2 - i,
        ))
    return pd.DataFrame(out)


T_us = build_table(replay_live(False))
T_all = build_table(replay_live(True))
T_us.to_csv(HERE / "signals_3m.csv", index=False)
T_all.to_csv(HERE / "signals_3m_allday.csv", index=False)

# Verify r3_engine on the 3-month dataset matches live rules.eval_1m 100%
df_eng = E.load(RAW_3M)
T_eng = E.run(df_eng, E.DEFAULT, S3M, E3M)
assert list(T_eng.time) == list(T_us.time), "Engine times mismatch!"
assert np.allclose(T_eng.net_pct.round(3).values, T_us.net_pct.values), "Engine net_pct mismatch!"


def fmt(T, col="net_pct", lab=""):
    if len(T) == 0:
        return f"{lab}: 0 筆"
    w = int((T[col] > 0).sum())
    return (f"{lab}: n={len(T)}，勝 {w}/敗 {len(T)-w}，勝率 {w/len(T)*100:.1f}%，"
            f"平均 {T[col].mean():+.2f}%，總 {T[col].sum():+.2f}%")


lines = [
    fmt(T_us, lab="三個月主設定 美股時段(live)"),
    fmt(T_us, "sens_net_pct", "三個月 停利1%敏感度"),
    f"最大賺 {T_us.net_pct.max():+.2f}% 最大賠 {T_us.net_pct.min():+.2f}% 平均持有 {T_us.hold_min.mean():.1f}m 出場 {T_us.exit_reason.value_counts().to_dict()}",
    fmt(T_all, lab="三個月 全部時段(去掉us_session)"),
    fmt(T_all[~T_all.time.isin(T_us.time)], lab="其中非美股時段"),
    "依月:",
]
for m, _, _ in M_BOUNDS:
    sub = T_us[T_us.month == m]
    lines.append("  " + fmt(sub, lab=m) + f" | 敏感度(TP1%): {sub.sens_net_pct.sum():+.2f}% | 出場 {sub.exit_reason.value_counts().to_dict()}")
lines.append("依週:")
for w, g in T_us.groupby("week"):
    lines.append("  " + fmt(g, lab=w))

(HERE / "stats_3m.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("\n=== All 3-Month Signals (US session) ===")
print(T_us[["time", "month", "entry", "sl", "tp", "exit_reason", "hold_min", "net_pct", "sens_net_pct"]].to_string(index=False))
