"""Shared 1m rally-start rule D_HL_break_swinghigh+F (+ MA200 band) for backtest AND live watcher.
Signal only - never places orders. All inputs are CLOSED 1m bars, ascending.
Bar i only uses data up to and including bar i (no lookahead / no repaint)."""
import numpy as np, pandas as pd

DIP = 0.015    # dip low >= 1.5% below MA200 ...
LOOK = 30      # ... within the last 30 bars
DROP = 0.025   # dip low >= 2.5% below highest high of prior 90 bars
COOL = 60      # min bars between accepted D+F signals (greedy)
BAND = 1.0     # live/final filter: |close/MA200-1| <= 1% (rounded to 0.01%)
NEED = 300     # bars of history needed before a bar can be evaluated reliably

def d_raw(lo, hi, cl, dipb, look=LOOK):
    """Higher-low + first close above swing high after a dip low (identical to backtest rule D)."""
    sig = np.zeros(len(cl), bool)
    for i in range(look + 5, len(cl)):
        w0 = i - look
        j = w0 + int(np.argmin(lo[w0:i + 1]))
        if not dipb[j] or i - j < 4: continue
        for k in range(i - 2, j + 1, -1):
            if lo[k] > lo[j] and lo[k] <= lo[k-1] and lo[k] <= lo[k-2 if k-2 > j else k-1] and lo[k] < lo[k+1] and lo[k] < lo[k+2]:
                sh = hi[j:k].max()
                if cl[i] > sh and (k == i - 2 or (cl[k+1:i] <= sh).all()):
                    sig[i] = True
                break
    return sig

def features(time_ms, high, low, close):
    """DataFrame with MAs, filters, raw D and D+F flags for every bar."""
    df = pd.DataFrame({"time": np.asarray(time_ms, dtype="int64"), "high": np.asarray(high, float),
                       "low": np.asarray(low, float), "close": np.asarray(close, float)})
    df["dt"] = pd.to_datetime(df.time, unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    for n in [7, 25, 99, 200]:
        df[f"ma{n}"] = df.close.rolling(n).mean()
    df["dipbar"] = df.low <= df.ma200 * (1 - DIP)
    df["swinglow"] = df.low.rolling(LOOK).min()
    ny = df.dt.dt.tz_convert("America/New_York")
    mins = ny.dt.hour * 60 + ny.dt.minute
    df["rth"] = (ny.dt.dayofweek < 5) & (mins >= 570) & (mins < 960)   # US regular hours (DST-aware)
    df["drop90"] = df.swinglow / df.high.rolling(90).max() - 1
    df["F"] = df.rth & (df.drop90 <= -DROP)
    df["D"] = d_raw(df.low.values, df.high.values, df.close.values, df.dipbar.values)
    df["DF"] = df.D & df.F
    df["dist_ma200_pct"] = ((df.close / df.ma200 - 1) * 100).round(2)
    return df

def signals(df, cool=COOL, band=BAND, last_sig_t=None):
    """Greedy cooldown over D+F (like the backtest), then MA200 band filter.
    Returns list of (index, accepted_in_band) for every cooldown-accepted D+F bar.
    last_sig_t: time (ms) of last accepted D+F signal before df (for live continuity)."""
    out = []; last = -10**9
    t = df.time.values
    if last_sig_t is not None:
        last = int(np.searchsorted(t, last_sig_t)) if last_sig_t >= t[0] else int((last_sig_t - t[0]) // 60000)
    for i in np.where(df.DF.values)[0]:
        if i - last >= cool:
            last = i
            out.append((int(i), bool(abs(df.dist_ma200_pct.iat[i]) <= band)))
    return out

# ---------------- live helpers (used by watch.py and by the verification replay) ----------------
def process(rs, k, allow_alert_after_ms=None):
    """rs: persisted dict {last_closed,last_sig,last_fired}. k: list of (t,o,h,l,c) ascending,
    last element = forming bar (ignored). Evaluates every NEW closed bar once.
    Returns (df, [indices to alert]). Mutates rs."""
    kc = k[:-1]
    if len(kc) < NEED: return None, []
    df = features([x[0] for x in kc], [x[2] for x in kc], [x[3] for x in kc], [x[4] for x in kc])
    df["open"] = [x[1] for x in kc]
    ev = []; t = df.time.values; DF = df.DF.values
    for j in range(NEED - 1, len(df)):
        tj = int(t[j])
        if rs.get("last_closed") is not None and tj <= rs["last_closed"]: continue
        if DF[j] and (rs.get("last_sig") is None or (tj - rs["last_sig"]) // 60000 >= COOL):
            rs["last_sig"] = tj
            if abs(df.dist_ma200_pct.iat[j]) <= BAND and tj != rs.get("last_fired"):
                if allow_alert_after_ms is None or tj >= allow_alert_after_ms:
                    ev.append(j)
                rs["last_fired"] = tj
        rs["last_closed"] = tj
    return df, ev

def chart(df, j, path, bars=120, after=0):
    """1m chart of last `bars` bars ending at j(+after): MA7/25/99/200, entry, SL/TP lines."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w = df.iloc[max(0, j - bars + 1): j + 1 + after]
    e = df.close.iat[j]; sl = df.swinglow.iat[j] * 0.999; tp = e * 1.03
    x = w.dt.dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(12, 6))
    op = w.close.shift(1).fillna(w.close)   # open approximated by prior close if opens absent
    if "open" in w: op = w.open
    for xi, o, h, l, c in zip(x, op, w.high, w.low, w.close):
        col = "tab:green" if c >= o else "tab:red"
        ax.vlines(xi, l, h, color=col, lw=0.8); ax.vlines(xi, min(o, c), max(o, c), color=col, lw=3)
    for n, cc in zip([7, 25, 99, 200], ["gold", "deepskyblue", "saddlebrown", "purple"]):
        ax.plot(x, w[f"ma{n}"], color=cc, lw=1.2, label=f"MA{n}")
    xe = df.dt.iat[j].tz_localize(None)
    ax.scatter(xe, e, marker="^", s=200, color="blue", zorder=5, label=f"entry {e:.2f}")
    ax.axhline(sl, color="red", ls="--", lw=1, label=f"SL {sl:.2f}")
    ax.axhline(tp, color="green", ls="--", lw=1, label=f"TP+3% {tp:.2f}")
    ax.set_title(f"SOXL 1m rally start (BingX) {df.dt.iat[j].strftime('%Y-%m-%d %H:%M')} UTC+8")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=100); plt.close(fig)
    return path
