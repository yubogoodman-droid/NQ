#!/usr/bin/env python3
"""NQ 一分 K「Gork多＋線性空｜自動版」回測 — 對齊 pinescript/gork_linear_combo_auto.pine。

共用：單倉（flat-only）、平倉後冷卻 5 根、ET 16–17 / 18–19 不開新倉。
線性空：創 4H 高 + 全多頭排列 → 回測 MA10 → 收盤跌破 MA20；峰距 MA200 ≥ 100、
        創高當下 MA10 斜率 ≥ 6、不貼上升 MA60、MA20−MA200 ≤ 180、15m 過濾；
        停損四小時高、停利 MA200+5 或靠近 MA200 長下影。
Gork 多：跌破 MA200 > 50 點後站回（連 3 根、五線多頭排列、距 MA200 ≤ 30、
        前 60 根曾連 15 根收在 MA200 下）；停損 MA200−10、停利 +100、+60 改保本。

新增兩個門檻（預設開；--baseline 全關）：
  --lin-max-risk 55        線性空 peakHigh − 進場 > 55 點就跳過
  --gork-ma200-slope 30    Gork 只在 1m MA200 低於 30 根前（下彎）時進場

用法:
  python3 examples/nq_gork_linear_combo.py                 # 近 30 天，優化版
  python3 examples/nq_gork_linear_combo.py --compare       # 基準 vs 優化 逐筆／逐日
  python3 examples/nq_gork_linear_combo.py --baseline      # 原規則
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ET = "America/New_York"
TPE = "Asia/Taipei"

BASE: Dict = dict(
    cooldown=5,
    # 線性空
    lookback4h=240, peak_dist=100, slope_n=5, slope_min=6, near60=20, fan_max=180, tp_buf=5,
    f15_slope=120, f15_dist=120,
    lin_max_risk=0,
    # Gork
    gork_deep=50, gork_window=60, gork_streak=3, gork_max_dist=30, gork_under_need=15,
    gork_sl_below=10, gork_tp=100, gork_be_after=60,
    gork_ma200_slope=0,
)
OPTIMIZED: Dict = dict(BASE, lin_max_risk=55, gork_ma200_slope=30)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float)
    return df


def load_yahoo_1m(symbol: str, days: int) -> pd.DataFrame:
    """Yahoo 1m 一次最多 7 天、最多回看 30 天，所以切塊抓。"""
    import yfinance as yf

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=min(days, 29))
    chunks: List[pd.DataFrame] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        try:
            part = yf.download(symbol, interval="1m", start=cur, end=nxt, progress=False, auto_adjust=False)
        except Exception as ex:  # pragma: no cover - network
            print(f"[data] {cur.date()} → {nxt.date()} error {ex}", file=sys.stderr)
            part = None
        if part is not None and len(part):
            chunks.append(_flatten(part))
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(part)}", file=sys.stderr)
        cur = nxt
        time.sleep(0.3)
    if not chunks:
        raise SystemExit("no data")
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna()
    df.index = df.index.tz_convert(ET)
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _flatten(df)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df.dropna()


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    for n in (5, 10, 20, 30, 60, 120, 200):
        df[f"ma{n}"] = sma(df.close, n)
    return df


def htf_15m(df: pd.DataFrame) -> pd.DataFrame:
    """已收盤 15m，對齊到每根 1m：1m 開盤時間 t 只看得到收盤時間 ≤ t 的 15m（＝Pine 的 [1]）。"""
    c = df.close.resample("15min", label="right", closed="left").last().dropna()
    out = pd.DataFrame({"c": c, "ma10": sma(c, 10), "ma60": sma(c, 60)})
    out["ma10_5"] = out.ma10.shift(5)
    out = out.reindex(out.index.union(df.index)).sort_index().ffill().reindex(df.index)
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run(df: pd.DataFrame, P: Dict):
    h15 = htf_15m(df)
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    ma = {n: df[f"ma{n}"].values for n in (5, 10, 20, 30, 60, 120, 200)}
    t = df.index
    et_min = (t.hour * 60 + t.minute).values
    no_entry = ((et_min >= 16 * 60) & (et_min < 17 * 60)) | ((et_min >= 18 * 60) & (et_min < 19 * 60))
    bull = (ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[30]) & (ma[30] > ma[60]) & (ma[60] > ma[120]) & (ma[120] > ma[200])
    hi4 = pd.Series(h).rolling(P["lookback4h"], min_periods=1).max().values
    made4h = (h >= hi4) & (h > np.r_[np.nan, h[:-1]])
    m15_slope = (h15.ma10 - h15.ma10_5).values
    m15_dist = (h15.c - h15.ma60).values

    trades: List[Dict] = []
    blocks: List[Dict] = []
    pos: Optional[Dict] = None
    last_exit = -10**9
    st = 0
    peak_high = peak_ma200 = peak_slope = np.nan
    deep_bar = -10**9

    def close_trade(i: int, px: float, reason: str):
        nonlocal pos, last_exit, st
        sign = -1 if pos["side"] == "S" else 1
        trades.append(dict(side=pos["side"], ei=pos["i"], xi=i, entry=pos["px"], exit=px,
                           pnl=sign * (px - pos["px"]), reason=reason, **pos["meta"]))
        pos = None
        last_exit = i
        if sign < 0:
            st = 0

    for i in range(201, len(df)):
        if pos is not None:
            if pos["side"] == "S":
                if h[i] >= pos["stop"]:
                    close_trade(i, pos["stop"], "stop"); continue
                tp = ma[200][i] + P["tp_buf"]
                if l[i] <= tp:
                    close_trade(i, tp, "ma200"); continue
                body = max(abs(c[i] - o[i]), 0.25); lw = min(o[i], c[i]) - l[i]; rng = max(h[i] - l[i], 0.25)
                if l[i] >= ma[200][i] and (l[i] - ma[200][i]) <= 15 and lw >= 3 and (lw >= 3 * body or lw / rng >= 0.55):
                    close_trade(i, c[i], "wick"); continue
            else:
                if l[i] <= pos["stop"]:
                    close_trade(i, min(pos["stop"], o[i]), "be" if pos["be"] else "stop"); continue
                if h[i] >= pos["px"] + P["gork_tp"]:
                    close_trade(i, pos["px"] + P["gork_tp"], "tp"); continue
                if not pos["be"] and h[i] >= pos["px"] + P["gork_be_after"]:
                    pos["be"] = True; pos["stop"] = pos["px"]
            continue

        can_enter = (i - last_exit > P["cooldown"]) and not no_entry[i]

        # ---- 線性空狀態機
        if st == 0:
            if made4h[i] and bull[i]:
                st = 1; peak_high = h[i]; peak_ma200 = ma[200][i]; peak_slope = ma[10][i] - ma[10][i - P["slope_n"]]
        elif st == 1:
            if made4h[i] and bull[i] and h[i] > peak_high:
                peak_high = h[i]; peak_ma200 = ma[200][i]; peak_slope = ma[10][i] - ma[10][i - P["slope_n"]]
            if l[i] <= ma[10][i]:
                st = 2
            if c[i] < ma[60][i] and not bull[i]:
                st = 0
        elif st == 2:
            if h[i] > peak_high:
                st = 0
            elif c[i] < ma[20][i] and c[i - 1] >= ma[20][i - 1]:
                meta = dict(peak=peak_high, peakDist=peak_high - peak_ma200, risk=peak_high - c[i], slope=peak_slope,
                            dist60=c[i] - ma[60][i], ma60rise=ma[60][i] - ma[60][i - 5], fan=ma[20][i] - ma[200][i],
                            m15slope=m15_slope[i], m15dist=m15_dist[i])
                why = None
                if meta["peakDist"] < P["peak_dist"]: why = "peakDist"
                elif not can_enter: why = "cooldown/hours"
                elif meta["slope"] < P["slope_min"]: why = "slope"
                elif P["fan_max"] > 0 and meta["fan"] > P["fan_max"]: why = "fan"
                elif P["near60"] > 0 and meta["ma60rise"] > 0 and 0 <= meta["dist60"] <= P["near60"]: why = "near60"
                elif P["f15_slope"] > 0 and meta["m15slope"] > P["f15_slope"]: why = "15m_slope"
                elif P["f15_dist"] > 0 and meta["m15dist"] < P["f15_dist"]: why = "15m_dist"
                elif P["lin_max_risk"] > 0 and meta["risk"] > P["lin_max_risk"]: why = "max_risk"
                st = 0
                if why:
                    blocks.append(dict(side="S", i=i, t=t[i], why=why, px=c[i], **meta))
                else:
                    pos = dict(side="S", i=i, px=c[i], stop=peak_high, meta=meta); st = 3
                    continue

        # ---- Gork 多
        if l[i] < ma[200][i] - P["gork_deep"]:
            deep_bar = i
        if not (1 <= i - deep_bar <= P["gork_window"]):
            continue
        streak = all(c[i - k] > ma[200][i - k] for k in range(P["gork_streak"]))
        stack = ma[5][i] > ma[10][i] > ma[20][i] > ma[30][i] > ma[60][i]
        dist = c[i] - ma[200][i]
        if not (streak and stack and c[i] > ma[60][i] and 0 < dist):
            continue
        run_u = mxu = 0
        for k in range(1, 61):
            if c[i - k] < ma[200][i - k]:
                run_u += 1; mxu = max(mxu, run_u)
            else:
                run_u = 0
        meta = dict(dist200=dist, m15dist60=m15_dist[i], maxunder=mxu, ma200slope=ma[200][i] - ma[200][i - 30])
        why = None
        if not can_enter: why = "cooldown/hours"
        elif dist > P["gork_max_dist"]: why = "far"
        elif mxu < P["gork_under_need"]: why = "under"
        elif P["gork_ma200_slope"] > 0 and ma[200][i] >= ma[200][i - P["gork_ma200_slope"]]: why = "ma200_not_falling"
        if why:
            blocks.append(dict(side="L", i=i, t=t[i], why=why, px=c[i], **meta))
        else:
            pos = dict(side="L", i=i, px=c[i], stop=ma[200][i] - P["gork_sl_below"], be=False, meta=meta)
            deep_bar = -10**9

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["et"] = t[tr.ei.values]
        tr["xt"] = t[tr.xi.values]
        tr["tpe"] = tr.et.dt.tz_convert(TPE)
    bl = pd.DataFrame(blocks)
    if len(bl):
        bl["tpe"] = bl.t.dt.tz_convert(TPE)
    return tr, bl


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summary(tr: pd.DataFrame, name: str) -> str:
    if not len(tr):
        return f"{name}: 0 trades"
    g = tr[tr.side == "L"]; s = tr[tr.side == "S"]
    dd = (tr.pnl.cumsum() - tr.pnl.cumsum().cummax()).min()
    return (f"{name:10s} n={len(tr):3d} net={tr.pnl.sum():8.2f} maxDD={dd:8.2f} | "
            f"Gork n={len(g)} {g.pnl.sum():8.2f} (tp {int((g.reason == 'tp').sum())}/be {int((g.reason == 'be').sum())}/stop {int((g.reason == 'stop').sum())}) | "
            f"線性空 n={len(s)} {s.pnl.sum():8.2f} (停損 {int((s.reason == 'stop').sum())} {s[s.reason == 'stop'].pnl.sum():8.2f})")


def trade_table(tr: pd.DataFrame) -> str:
    if not len(tr):
        return "(none)"
    d = tr.copy()
    d["TPE"] = d.tpe.dt.strftime("%m-%d %H:%M")
    d["bars"] = d.xi - d.ei
    cols = ["side", "TPE", "entry", "exit", "pnl", "reason", "bars", "risk", "m15dist", "dist200", "ma200slope"]
    cols = [x for x in cols if x in d.columns]
    return d[cols].round(2).to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NQ=F")
    ap.add_argument("--days", type=int, default=29, help="Yahoo 1m 最多 29 天")
    ap.add_argument("--csv", help="改用本地 1m OHLC CSV（index=時間）")
    ap.add_argument("--baseline", action="store_true", help="關閉兩個新門檻")
    ap.add_argument("--compare", action="store_true", help="基準 vs 優化")
    ap.add_argument("--lin-max-risk", type=float, default=OPTIMIZED["lin_max_risk"])
    ap.add_argument("--gork-ma200-slope", type=int, default=OPTIMIZED["gork_ma200_slope"])
    args = ap.parse_args()

    df = load_csv(args.csv) if args.csv else load_yahoo_1m(args.symbol, args.days)
    df = add_mas(df)
    print(f"bars={len(df)}  {df.index.min()} → {df.index.max()} ET", file=sys.stderr)
    pd.set_option("display.width", 220)

    opt = dict(BASE, lin_max_risk=args.lin_max_risk, gork_ma200_slope=args.gork_ma200_slope)
    if args.compare:
        tb, bb = run(df, BASE)
        to, bo = run(df, opt)
        print(summary(tb, "基準")); print(summary(to, "優化"))
        lost = tb[~tb.ei.isin(set(to.ei))]
        added = to[~to.ei.isin(set(tb.ei))]
        print("\n被新門檻擋掉的（基準有、優化沒有）:"); print(trade_table(lost))
        print("\n優化多出來的（路徑改變）:"); print(trade_table(added))
        for name, tr in (("基準", tb), ("優化", to)):
            tr["day"] = tr.tpe.dt.strftime("%m-%d")
        daily = pd.DataFrame({"基準": tb.groupby("day").pnl.sum(), "優化": to.groupby("day").pnl.sum()}).fillna(0).round(2)
        daily.loc["TOTAL"] = daily.sum()
        print("\n逐日（TPE 日期）:"); print(daily.to_string())
        if len(bo):
            print("\n優化版擋單原因:"); print(bo.groupby(["side", "why"]).size().to_string())
    else:
        P = BASE if args.baseline else opt
        tr, bl = run(df, P)
        print(summary(tr, "基準" if args.baseline else "優化"))
        print(trade_table(tr))
        if len(bl):
            print("\n擋單原因:"); print(bl.groupby(["side", "why"]).size().to_string())


if __name__ == "__main__":
    main()
