"""Parameterised, numpy re-implementation of live r3 (起漲點：急跌後更高低點突破) for research.

Research / backtest only: no orders, no broker or exchange trading API, no notifications.
Reads the cached BingX 1m CSV in ../backtest_month/. Does not import or modify any live watcher.

With DEFAULT parameters this reproduces live_rules/rules.py eval_1m r3 bar-for-bar
(verified by verify_engine.py against backtest_month/signals.csv).
"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "backtest_month" / "bingx_1m_raw.csv"
TPE = "Asia/Taipei"
FEE_PCT = 0.1
MONTH_START = pd.Timestamp("2026-10-01 09:30", tz=TPE)
MONTH_END = pd.Timestamp("2026-10-31 09:30", tz=TPE)


@dataclass(frozen=True)
class Params:
    lookback: int = 90          # bars for the reference high H (window i-lookback+1 .. i)
    drop: float = 0.025         # L1 <= H * (1 - drop)
    hl_min: float = 0.0         # L2 >= L1 * (1 + hl_min) (in addition to L2 > L1; e.g. 0.0005 = +5 bp to reject 1-tick noise)
    l2_ma_buf: float = 0.015    # L2 >= MA200[l2] * (1 - l2_ma_buf)
    band: float = 0.01          # |close / MA200 - 1| <= band
    band_side: str = "both"     # "both" | "above" (close >= MA200) | "below" (close <= MA200)
    cool: int = 30              # minutes between accepted signals
    session: str = "us"         # "us" = weekday TPE 21:30-04:00 (live); "all"; "us_early" 21:30-01:00; "us_late" 01:00-04:00
    tp: float = 0.03            # take profit (fraction of entry)
    sl_mode: str = "swing"      # "swing" = min(low, sl_bars) * (1 - sl_pad); "pct" = entry * (1 - sl_pct); "l2" = max(L2, entry*(1-sl_pct))
    sl_bars: int = 30
    sl_pad: float = 0.001
    sl_pct: float = 0.01
    max_bars: int = 120         # time exit after this many bars


DEFAULT = Params()


def load(path=RAW):
    raw = pd.read_csv(path)
    df = pd.DataFrame({"t": raw.time.astype("int64").values, "open": raw.open.values, "high": raw.high.values,
                       "low": raw.low.values, "close": raw.close.values})
    df["ma200"] = df.close.rolling(200).mean()
    df["dt"] = pd.to_datetime(df.t, unit="ms", utc=True).dt.tz_convert(TPE)
    hm = df.dt.dt.hour * 60 + df.dt.dt.minute
    wd = df.dt.dt.weekday.values
    prev_wd = (df.dt - pd.Timedelta(days=1)).dt.weekday.values
    late = hm >= 21 * 60 + 30
    early = hm < 4 * 60
    df["us"] = np.where(late, wd < 5, np.where(early, prev_wd < 5, False))
    df["us_early"] = df.us & (late | (hm < 60))
    df["us_late"] = df.us & (hm >= 60) & early
    # US trading night label (session starts on TPE evening); used for walk-forward splits
    df["night"] = (df.dt - pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")
    return df


_STRUCT_CACHE = {}


def structure(df, lookback):
    """Per-bar pattern geometry that does not depend on any threshold. Mirrors eval_1m index logic."""
    key = (id(df), lookback)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    hi, lo = df.high.values, df.low.values
    n = len(df)
    out = {k: np.full(n, np.nan) for k in ("H", "L1", "L2", "B", "ma_l2")}
    l1a = np.full(n, -1)
    ma = df.ma200.values
    for i in range(lookback, n):
        w0 = i - lookback + 1
        h = w0 + int(np.argmax(hi[w0:i + 1]))
        if i - (h + 1) < 4:
            continue
        l1 = h + 1 + int(np.argmin(lo[h + 1:i]))
        if l1 + 2 > i - 1:
            continue
        l2 = l1 + 2 + int(np.argmin(lo[l1 + 2:i]))
        out["H"][i] = hi[h]; out["L1"][i] = lo[l1]; out["L2"][i] = lo[l2]
        out["B"][i] = hi[l1 + 1:l2].max(); out["ma_l2"][i] = ma[l2]; l1a[i] = l1
    out["l1"] = l1a
    _STRUCT_CACHE[key] = out
    return out


def signals(df, p=DEFAULT, start=None, end=None):
    """Return bar indices of accepted r3 signals (cooldown applied sequentially like live)."""
    s = structure(df, p.lookback)
    cl, ma = df.close.values, df.ma200.values
    prev = np.r_[np.nan, cl[:-1]]
    with np.errstate(invalid="ignore"):
        dist = cl / ma - 1
        if p.band_side == "above":
            band_ok = (dist >= 0) & (dist <= p.band)
        elif p.band_side == "below":
            band_ok = (dist <= 0) & (dist >= -p.band)
        else:
            band_ok = np.abs(dist) <= p.band
        cand = ((s["L1"] <= s["H"] * (1 - p.drop)) & (s["L2"] > s["L1"]) & (s["L2"] >= s["L1"] * (1 + p.hl_min))
                & ~np.isnan(s["ma_l2"]) & (s["L2"] >= s["ma_l2"] * (1 - p.l2_ma_buf)) & (cl > s["B"]) & (prev <= s["B"])
                & band_ok & ~np.isnan(ma))
    if p.session != "all":
        cand &= df[p.session].values
    t = df.t.values
    acc, last = [], -10**18
    for i in np.where(cand)[0]:
        close_ms = t[i] + 60000
        if close_ms - last >= p.cool * 60000:
            acc.append(i); last = close_ms
    acc = np.array(acc, int)
    # cooldown runs over the whole history (as bt.py does); the window is applied afterwards
    if start is not None:
        acc = acc[t[acc] >= int(start.timestamp() * 1000)]
    if end is not None:
        acc = acc[t[acc] < int(end.timestamp() * 1000)]
    return acc


def simulate(df, idx, p=DEFAULT):
    """Exit simulation identical to backtest_month/analyze.py sim(): stop checked before target in the same bar."""
    cl, lo, hi = df.close.values, df.low.values, df.high.values
    n = len(df)
    s = structure(df, p.lookback) if p.sl_mode == "l2" else None
    swing = pd.Series(lo).rolling(p.sl_bars).min().values if p.sl_mode == "swing" else None
    rows = []
    for i in idx:
        e = cl[i]
        if p.sl_mode == "swing":
            sl = swing[i] * (1 - p.sl_pad)
        elif p.sl_mode == "pct":
            sl = e * (1 - p.sl_pct)
        else:
            sl = max(s["L2"][i], e * (1 - p.sl_pct))
        ret, why, k = None, None, None
        for k in range(i + 1, min(i + 1 + p.max_bars, n)):
            if lo[k] <= sl:
                ret, why = sl / e - 1, "停損"; break
            if hi[k] >= e * (1 + p.tp):
                ret, why = p.tp, "停利"; break
        if ret is None:
            k = min(i + p.max_bars, n - 1); ret, why = cl[k] / e - 1, "時間出場"
        rows.append(dict(i=int(i), time=df.dt.iat[i].strftime("%Y-%m-%d %H:%M"), night=df.night.iat[i],
                         entry=e, sl=sl, exit_reason=why, hold_min=k - i, net_pct=ret * 100 - FEE_PCT,
                         complete=i + p.max_bars < n))
    return pd.DataFrame(rows, columns=["i", "time", "night", "entry", "sl", "exit_reason", "hold_min", "net_pct", "complete"])


def run(df, p=DEFAULT, start=MONTH_START, end=MONTH_END):
    return simulate(df, signals(df, p, start, end), p)


def stats(T):
    if len(T) == 0:
        return dict(n=0, wins=0, win_rate=np.nan, avg=np.nan, total=0.0, worst=np.nan)
    w = int((T.net_pct > 0).sum())
    return dict(n=len(T), wins=w, win_rate=w / len(T), avg=T.net_pct.mean(), total=T.net_pct.sum(), worst=T.net_pct.min())


__all__ = ["Params", "DEFAULT", "replace", "load", "structure", "signals", "simulate", "run", "stats",
           "MONTH_START", "MONTH_END", "FEE_PCT"]
