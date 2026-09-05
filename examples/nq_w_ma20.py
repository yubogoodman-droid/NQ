#!/usr/bin/env python3
"""NQ 五分 K 雙底：創兩小時低 → 站上 MA20 → 再跌破 → 回測守三根。

對齊 2026-09-04 圖 29：
  11:30 創兩小時低 29478 → 12:05 站上 MA20 → 13:05 跌破 MA20
  → 13:45 回到低點附近 29468 → 連續三根五分 K 沒新低，14:00 做多。

用法:
  python3 examples/nq_w_ma20.py
  python3 examples/nq_w_ma20.py backtest --period 7d --pages
  python3 examples/nq_w_ma20.py backtest --period 60d --pages
  python3 examples/test_nq_w_ma20.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-w-ma20" / "index.html"
VIEW_BRANCH = "cursor/nq-5m-w-ma20-26e0"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_yfinance(symbol: str = "NQ=F", interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


def to_et(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(ET)
    else:
        out.index = out.index.tz_convert(ET)
    return out


def summarize_trades(trades: Sequence) -> dict:
    pnls = [float(getattr(t, "pnl_points", 0.0)) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_q: Dict[str, List[float]] = {}
    for t in trades:
        by_q.setdefault(getattr(t, "quality", "?"), []).append(float(getattr(t, "pnl_points", 0.0)))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "pnl": float(sum(pnls)),
        "n": n,
        "avg": float(sum(pnls) / n) if n else 0.0,
        "by_quality": {
            q: {
                "n": len(v),
                "wins": sum(1 for p in v if p > 0),
                "pnl": float(sum(v)),
            }
            for q, v in sorted(by_q.items())
        },
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    stand_idx: int
    break_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    first_low: float
    second_low: float
    neckline: float
    ma20: float
    stand_pts: float
    low_gap_pts: float
    neck_pts: float
    hold_bars: int
    quality: str = "C"
    quality_score: int = 0


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_points: float
    exit_reason: str
    quality: str


def sma(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().to_numpy(float)


def rolling_min_prev(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).shift(1).rolling(n, min_periods=n).min().to_numpy(float)


def detect_params(interval: str = "5m") -> dict:
    """15 分 K：兩小時 = 8 根、持倉 16 根；五分 K：兩小時 = 24 根、持倉 48 根。"""
    iv = (interval or "5m").lower().replace("min", "m")
    if iv == "15m":
        return dict(two_hour_bars=8, max_hold=16)
    return dict(two_hour_bars=24, max_hold=48)


def quality_from_w(low_gap_pts: float, neck_pts: float, stand_pts: float) -> Tuple[int, str]:
    score = 0
    if abs(low_gap_pts) <= 15.0:
        score += 1
    if neck_pts >= 40.0:
        score += 1
    if stand_pts >= 4.0:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def detect_signals(
    df,
    *,
    two_hour_bars: int = 24,
    min_break_depth: float = 8.0,
    max_bars_to_stand: int = 36,
    max_bars_after_stand: int = 36,
    max_bars_to_retest: int = 36,
    max_hold_wait: int = 24,
    hold_bars: int = 3,
    near_pts: float = 25.0,
    spring_pts: float = 20.0,
    stop_buffer: float = 4.0,
    target_r: float = 2.0,
    max_risk: float = 50.0,
    min_entry_gap: int = 12,
    ma_period: int = 20,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    創下兩小時低點 → 收盤站上 MA20 → 再跌破 MA20 →
    回到兩小時低點附近 → 連續 hold_bars 根五分 K 沒新低，進場做多。
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma20 = sma(close, ma_period)
    two_hr = rolling_min_prev(low, two_hour_bars)
    n = len(close)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    signals: List[Signal] = []
    last_entry = -(10**9)
    i = max(ma_period, two_hour_bars) + 1

    while i < n - hold_bars:
        if np.isnan(two_hr[i]) or low[i] >= two_hr[i] or (two_hr[i] - low[i]) < min_break_depth:
            i += 1
            continue

        dump_idx = i
        dump_low = float(low[i])
        bump("dump")

        stand_idx: Optional[int] = None
        j = i + 1
        end_stand = min(i + max_bars_to_stand + 1, n)
        while j < end_stand:
            if float(low[j]) < dump_low:
                dump_low = float(low[j])
                dump_idx = j
            m = ma20[j]
            if not np.isnan(m) and close[j] > m and close[j - 1] <= ma20[j - 1]:
                stand_idx = j
                break
            j += 1
        if stand_idx is None:
            bump("skip_no_stand")
            i = dump_idx + 1
            continue
        bump("stand")

        break_idx: Optional[int] = None
        k = stand_idx + 1
        end_brk = min(stand_idx + max_bars_after_stand + 1, n)
        while k < end_brk:
            if float(low[k]) < dump_low - spring_pts:
                bump("skip_dump_after_stand")
                break_idx = None
                break
            mk = ma20[k]
            prev_m = ma20[k - 1]
            if not np.isnan(mk) and not np.isnan(prev_m) and close[k] < mk and close[k - 1] >= prev_m:
                break_idx = k
                break
            k += 1
        if break_idx is None:
            bump("skip_no_break")
            i = stand_idx + 1
            continue
        bump("broke")

        neck_slice = high[stand_idx : break_idx + 1]
        neckline_idx = stand_idx + int(np.argmax(neck_slice))
        neckline_price = float(high[neckline_idx])

        near_idx: Optional[int] = None
        retest_low = dump_low
        retest_idx = dump_idx
        m = break_idx + 1
        end_rt = min(break_idx + max_bars_to_retest + 1, n)
        while m < end_rt:
            lv = float(low[m])
            if lv < dump_low - spring_pts:
                bump("skip_spring_too_deep")
                near_idx = None
                break
            if lv <= dump_low + near_pts:
                near_idx = m
                retest_low = lv
                retest_idx = m
                break
            m += 1
        if near_idx is None:
            bump("skip_no_retest")
            i = break_idx + 1
            continue
        bump("retest")

        hold = 0
        entry_idx: Optional[int] = None
        p = near_idx
        end_hold = min(near_idx + max_hold_wait + 1, n)
        while p < end_hold:
            lv = float(low[p])
            if lv < retest_low - 1e-9:
                if lv < dump_low - spring_pts:
                    bump("skip_new_low")
                    entry_idx = None
                    break
                retest_low = lv
                retest_idx = p
                hold = 0
            elif p != retest_idx:
                hold += 1
                if hold >= hold_bars:
                    entry_idx = p
                    break
            p += 1
        if entry_idx is None:
            bump("skip_no_hold")
            i = near_idx + 1
            continue
        if entry_idx - last_entry < min_entry_gap:
            bump("skip_gap")
            i = entry_idx + 1
            continue

        entry = float(close[entry_idx])
        floor = min(dump_low, retest_low)
        stop = floor - stop_buffer
        risk = entry - stop
        if risk <= 0:
            bump("skip_bad_risk")
            i = entry_idx + 1
            continue
        if max_risk > 0 and risk > max_risk:
            bump("skip_max_risk")
            i = entry_idx + 1
            continue

        neck_pts = neckline_price - floor
        measured = neckline_price + max(neck_pts, 0.0)
        r_tgt = entry + risk * target_r
        target = max(measured, r_tgt)
        if target <= entry:
            bump("skip_bad_target")
            i = entry_idx + 1
            continue

        stand_pts = float(close[stand_idx] - ma20[stand_idx]) if not np.isnan(ma20[stand_idx]) else 0.0
        low_gap = retest_low - dump_low
        q_score, q_grade = quality_from_w(low_gap, neck_pts, stand_pts)
        bump("taken")
        signals.append(
            Signal(
                first_low_idx=dump_idx,
                second_low_idx=retest_idx,
                neckline_idx=neckline_idx,
                stand_idx=stand_idx,
                break_idx=break_idx,
                entry_idx=entry_idx,
                entry_price=entry,
                stop_price=stop,
                target_price=float(target),
                first_low=dump_low,
                second_low=retest_low,
                neckline=neckline_price,
                ma20=float(ma20[entry_idx]) if not np.isnan(ma20[entry_idx]) else 0.0,
                stand_pts=stand_pts,
                low_gap_pts=low_gap,
                neck_pts=neck_pts,
                hold_bars=hold_bars,
                quality=q_grade,
                quality_score=q_score,
            )
        )
        last_entry = entry_idx
        i = entry_idx + 1

    return _dedupe_signals(signals)


def _signal_rank(sig: Signal) -> tuple:
    recency = -(sig.entry_idx - sig.second_low_idx)
    equal = -abs(sig.low_gap_pts)
    return (recency, equal, sig.neck_pts)


def _dedupe_signals(signals: Iterable[Signal], min_gap: int = 12) -> List[Signal]:
    by_entry: dict[int, Signal] = {}
    for sig in signals:
        existing = by_entry.get(sig.entry_idx)
        if existing is None or _signal_rank(sig) > _signal_rank(existing):
            by_entry[sig.entry_idx] = sig
    kept: List[Signal] = []
    for sig in sorted(by_entry.values(), key=lambda s: s.entry_idx):
        if kept and sig.entry_idx - kept[-1].entry_idx < min_gap:
            continue
        kept.append(sig)
    return kept


def simulate(
    df,
    signals: List[Signal],
    max_hold: int = 48,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    n = len(close)
    results: List[TradeResult] = []
    busy_until = -1

    for sig in signals:
        e = sig.entry_idx
        if e <= busy_until:
            continue
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        exit_idx = min(e + max_hold, n - 1)
        exit_price = float(close[exit_idx])
        reason = "time"

        for k in range(e + 1, min(e + max_hold, n - 1) + 1):
            if float(low[k]) <= stop:
                exit_idx, exit_price, reason = k, stop, "stop"
                break
            if float(high[k]) >= target:
                exit_idx, exit_price, reason = k, target, "target"
                break

        busy_until = exit_idx
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=e,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(exit_price - entry),
                exit_reason=reason,
                quality=sig.quality,
            )
        )
    return results


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    60: "#42a5f5",
    120: "#26c6da",
    200: "#ab47bc",
}


def _equity_svg(pnls: List[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(1.0, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys_src[i]):.1f}" for i in range(len(ys_src)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def resample_ohlc(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    cols: Dict[str, str] = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        cols["Volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(cols)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _asof_bar(df: pd.DataFrame, ts) -> int:
    pos = int(df.index.searchsorted(ts, side="right")) - 1
    return max(0, min(pos, len(df) - 1))


def _inline_mpl_svg(fig, prefix: str) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
    raw = buf.getvalue().decode("utf-8")
    raw = re.sub(r"<\?xml[^>]*>", "", raw)
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', raw)
    raw = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", raw)
    raw = re.sub(
        r'(href|xlink:href)="#([^"]+)"',
        lambda m: f'{m.group(1)}="#{prefix}{m.group(2)}"',
        raw,
    )
    raw = raw.replace(
        "<svg ",
        '<svg style="width:100%;height:auto;display:block;background:#0c1210" ',
        1,
    )
    return raw.strip()


def draw_trade_chart(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    plt.rcParams["svg.fonttype"] = "path"
    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    start = max(0, sig.first_low_idx - 12)
    end = min(len(df) - 1, max(trade.exit_idx + 8, sig.entry_idx + 16, sig.second_low_idx + 14))
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for nper, col in MA_COLORS.items():
        ma = close_full.rolling(nper, min_periods=nper).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.55 if nper == 20 else (1.25 if nper <= 10 else 1.0), label=f"MA{nper}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.neckline, color="#f59e0b", ls="--", lw=1.05, alpha=0.85)

    l1x, l2x, nx, sx, bx, ex, xx = (
        sig.first_low_idx - start,
        sig.second_low_idx - start,
        sig.neckline_idx - start,
        sig.stand_idx - start,
        sig.break_idx - start,
        trade.entry_idx - start,
        trade.exit_idx - start,
    )
    w_x, w_y = [], []
    if 0 <= l1x < len(window):
        w_x.append(l1x)
        w_y.append(sig.first_low)
        ax.scatter([l1x], [sig.first_low], s=36, color="#facc15", zorder=6)
        ax.annotate("2h低", (l1x, sig.first_low), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= sx < len(window):
        ax.scatter([sx], [window["Close"].iloc[sx]], s=32, color="#22c55e", zorder=5, marker="^")
        ax.annotate("站上MA20", (sx, window["Close"].iloc[sx]), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#86efac", fontsize=7)
    if 0 <= nx < len(window):
        w_x.append(nx)
        w_y.append(sig.neckline)
        ax.scatter([nx], [sig.neckline], s=32, color="#f59e0b", zorder=5)
        ax.annotate("反彈高", (nx, sig.neckline), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#fbbf24", fontsize=8)
    if 0 <= bx < len(window):
        ax.scatter([bx], [window["Close"].iloc[bx]], s=28, color="#fb7185", zorder=5, marker="v")
        ax.annotate("跌破MA20", (bx, window["Close"].iloc[bx]), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#fda4af", fontsize=7)
    if 0 <= l2x < len(window):
        w_x.append(l2x)
        w_y.append(sig.second_low)
        ax.scatter([l2x], [sig.second_low], s=36, color="#facc15", zorder=6)
        ax.annotate("回測", (l2x, sig.second_low), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= ex < len(window):
        ax.scatter([ex], [trade.entry_price], s=44, color="#22c55e", zorder=7, marker="^")
        ax.annotate("三根沒新低", (ex, trade.entry_price), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#86efac", fontsize=8)
    if 0 <= xx < len(window):
        ax.scatter([xx], [trade.exit_price], s=36, color="#fb7185", zorder=7, marker="v")
    if len(w_x) >= 2:
        ax.plot(w_x, w_y, color="#f59e0b", lw=1.1, alpha=0.7)

    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    labels = [window.index[i].strftime("%m-%d %H:%M") for i in ticks]
    axv.set_xticks(ticks)
    axv.set_xticklabels(labels, rotation=0)
    ax.set_title(f"#{trade_no}  五分K  雙底回測守三根  {window.index[0].strftime('%m-%d %H:%M')} ET",
                 color="#d7e3d4", fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.35, labelcolor="#d7e3d4")
    fig.tight_layout(pad=0.6)
    svg = _inline_mpl_svg(fig, f"t{trade_no:02d}_")
    plt.close(fig)
    return svg


def draw_15m_chart(df_5m: pd.DataFrame, df_15m: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    """同一筆五分進場，對照 15 分 K（綠線是 15m MA20）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    plt.rcParams["svg.fonttype"] = "path"
    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    dump_ts = df_5m.index[sig.first_low_idx]
    stand_ts = df_5m.index[sig.stand_idx]
    break_ts = df_5m.index[sig.break_idx]
    retest_ts = df_5m.index[sig.second_low_idx]
    entry_ts = df_5m.index[trade.entry_idx]
    exit_ts = df_5m.index[trade.exit_idx]

    i_dump = _asof_bar(df_15m, dump_ts)
    i_stand = _asof_bar(df_15m, stand_ts)
    i_break = _asof_bar(df_15m, break_ts)
    i_retest = _asof_bar(df_15m, retest_ts)
    i_entry = _asof_bar(df_15m, entry_ts)
    i_exit = _asof_bar(df_15m, exit_ts)

    start = max(0, i_dump - 8)
    end = min(len(df_15m) - 1, max(i_entry + 8, i_exit + 3, i_dump + 16))
    window = df_15m.iloc[start : end + 1]
    if window.empty:
        return ""
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df_15m["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.85)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for nper, col in MA_COLORS.items():
        ma = close_full.rolling(nper, min_periods=nper).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=2.2 if nper == 20 else (1.2 if nper <= 10 else 1.0), label=f"15m MA{nper}")

    ax.axhline(sig.first_low, color="#facc15", ls=":", lw=1.0, alpha=0.75)
    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=0.9, alpha=0.65)

    marks = (
        (i_dump - start, sig.first_low, "2h低", (0, -13), "#fde68a"),
        (i_stand - start, float(df_15m["Close"].iloc[i_stand]), "站上", (0, 8), "#86efac"),
        (i_break - start, float(df_15m["Close"].iloc[i_break]), "跌破", (0, -12), "#fda4af"),
        (i_retest - start, sig.second_low, "回測", (0, -13), "#fde68a"),
    )
    for x, y, lab, off, col in marks:
        if 0 <= x < len(window):
            ax.scatter([x], [y], s=30, color=col, zorder=6)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, ha="center", color=col, fontsize=7)
    ex = i_entry - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=1.0, alpha=0.7)
        ax.scatter([ex], [trade.entry_price], s=48, color="#22c55e", marker="^", zorder=7)
        ax.annotate("5m進場", (ex, trade.entry_price), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#86efac", fontsize=8)
    xx = i_exit - start
    if 0 <= xx < len(window):
        ax.scatter([xx], [trade.exit_price], s=32, color="#fb7185", marker="v", zorder=7)

    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], rotation=0)
    ax.set_title(f"#{trade_no}  15分K  {entry_ts.strftime('%m-%d %H:%M')} ET", color="#d7e3d4", fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.35, labelcolor="#d7e3d4")
    fig.tight_layout(pad=0.6)
    svg = _inline_mpl_svg(fig, f"t15_{trade_no:02d}_")
    plt.close(fig)
    return svg


def _ts(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    t = df.index[idx]
    return t.tz_convert(ET) if getattr(t, "tzinfo", None) else t


def _render_trade_cards(df: pd.DataFrame, trades: List[TradeResult]) -> str:
    df_15m = resample_ohlc(df, "15min") if len(df) else df
    cards = []
    for i, t in enumerate(trades, 1):
        et = _ts(df, t.entry_idx)
        xt = _ts(df, t.exit_idx)
        l2t = _ts(df, t.signal.second_low_idx)
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk else 0.0
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-loss" if t.pnl_points < 0 else "pnl-flat")
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        chart = draw_trade_chart(df, t, i)
        chart15 = draw_15m_chart(df, df_15m, t, i) if len(df_15m) else ""
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div><div class='trade-no'>#{i}  Q{escape(t.quality)}</div>"
            f"<span class='trade-time'>{escape(et.strftime('%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>雙底</span>"
            f"<span class='tag tag-info'>2h低回測</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"<span class='tag tag-info'>15分K</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"2h低 {t.signal.first_low:.2f} / 回測 {t.signal.second_low:.2f}  "
            f"差 {t.signal.low_gap_pts:+.1f}pt\n"
            f"反彈高 {t.signal.neckline:.2f}  高度 {t.signal.neck_pts:.1f}pt\n"
            f"回測 {l2t.strftime('%m-%d %H:%M')}  三根沒新低  "
            f"MA20 {t.signal.ma20:.2f}"
            "</pre>"
            "<div class='chart-label'>五分K</div>"
            f"<div class='mini-chart'>{chart}</div>"
            "<div class='chart-label'>15分K</div>"
            f"<div class='mini-chart'>{chart15}</div>"
            "</article>"
        )
    return "".join(cards)


def _render_15m_trade_cards(df_15m: pd.DataFrame, trades: List[TradeResult]) -> str:
    cards = []
    for i, t in enumerate(trades, 1):
        et = _ts(df_15m, t.entry_idx)
        xt = _ts(df_15m, t.exit_idx)
        l2t = _ts(df_15m, t.signal.second_low_idx)
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk else 0.0
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-loss" if t.pnl_points < 0 else "pnl-flat")
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        chart = draw_trade_chart(df_15m, t, i)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div><div class='trade-no'>15m #{i}  Q{escape(t.quality)}</div>"
            f"<span class='trade-time'>{escape(et.strftime('%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>15分K</span>"
            f"<span class='tag tag-info'>雙底</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"2h低 {t.signal.first_low:.2f} / 回測 {t.signal.second_low:.2f}  "
            f"差 {t.signal.low_gap_pts:+.1f}pt\n"
            f"反彈高 {t.signal.neckline:.2f}  高度 {t.signal.neck_pts:.1f}pt\n"
            f"回測 {l2t.strftime('%m-%d %H:%M')}  三根15分沒新低  "
            f"MA20 {t.signal.ma20:.2f}"
            "</pre>"
            "<div class='chart-label'>15分K</div>"
            f"<div class='mini-chart'>{chart}</div>"
            "</article>"
        )
    return "".join(cards)


def _section_summary(title: str, df: pd.DataFrame, trades: List[TradeResult], funnel: Optional[Dict[str, int]], verdict: str) -> str:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.1f}" for q, info in stats.get("by_quality", {}).items()]
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    start = df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else ""
    end = df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else ""
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：兩小時低 {funnel.get('dump', 0)} → "
            f"站上MA20 {funnel.get('stand', 0)} → 跌破 {funnel.get('broke', 0)} → "
            f"回測 {funnel.get('retest', 0)} → 進場 {funnel.get('taken', 0)}</p>"
        )
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    return (
        "<section class='summary'>"
        f"<h2>{escape(title)}</h2>"
        f"<p class='muted'>{escape(start)} → {escape(end)} ET · bars={len(df)}</p>"
        f"{verdict_html}"
        "<div class='cards'>"
        f"<div class='card'>筆數<b>{stats['count']}</b></div>"
        f"<div class='card'>勝率<b>{stats['win_rate']:.1f}%</b></div>"
        f"<div class='card'>總點數<b class='{total_cls}'>{stats['total_points']:+.1f}</b></div>"
        f"<div class='card'>均筆<b>{stats['avg']:+.1f}</b></div>"
        "</div>"
        f"<p class='muted'>{escape(q_line)}</p>"
        f"{funnel_line}"
        f"<div class='equity'>{_equity_svg(pnls)}</div>"
        "</section>"
    )


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    verdict: str = "",
    df_15m: Optional[pd.DataFrame] = None,
    trades_15m: Optional[List[TradeResult]] = None,
    funnel_15m: Optional[Dict[str, int]] = None,
    verdict_15m: str = "",
) -> Path:
    if df_15m is None:
        df_15m = resample_ohlc(df, "15min") if len(df) else df
    if trades_15m is None and len(df_15m):
        funnel_15m = {} if funnel_15m is None else funnel_15m
        p15 = detect_params("15m")
        sigs15 = detect_signals(df_15m, funnel=funnel_15m, two_hour_bars=p15["two_hour_bars"])
        trades_15m = simulate(df_15m, sigs15, max_hold=p15["max_hold"])
        verdict_15m = verdict_15m or _verdict(summarize_trades(trades_15m), funnel_15m)
    trades_15m = trades_15m or []

    out = Path(path)
    cards5 = _render_trade_cards(df, trades)
    cards15 = _render_15m_trade_cards(df_15m, trades_15m)
    sec5 = _section_summary("五分K 回測", df, trades, funnel, verdict)
    sec15 = _section_summary("15分K 回測", df_15m, trades_15m, funnel_15m, verdict_15m)
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分 / 15分 雙底</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
h2{{font-size:16px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.chart-label{{margin:10px 0 4px;font-size:13px;font-weight:700;color:#79c0ff}}
.mini-chart{{margin:0 -6px 8px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d;margin-bottom:14px}}
.lead{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
</style></head><body>
<div class="page">
<section class="lead">
<h1>{escape(symbol)} 五分 / 15分 雙底</h1>
<p class="muted">{escape(period)} · 同一套：兩小時低 → 站上 MA20 → 再跌破 → 回測附近守三根。15 分 K 的兩小時是 8 根，三根沒新低 = 45 分鐘。</p>
</section>
{sec5}
{cards5 or "<div class='empty'>五分K 無訊號</div>"}
{sec15}
{cards15 or "<div class='empty'>15分K 無訊號</div>"}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    del branch
    out = src.with_name("view.html")
    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_trades(df, trades, tag: str = "") -> None:
    prefix = f"{tag} " if tag else ""
    for i, t in enumerate(trades, 1):
        l2 = df.index[t.signal.second_low_idx]
        print(
            f"  [{prefix}{i}] Q{t.quality} 回測 {l2.strftime('%m-%d %H:%M')} "
            f"進 {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"2h/回測 {t.signal.first_low:.1f}/{t.signal.second_low:.1f} "
            f"反彈 {t.signal.neck_pts:.0f}pt"
        )


def _verdict(stats: dict, funnel: Dict[str, int]) -> str:
    n = stats["count"]
    wr = stats["win_rate"]
    pnl = stats["total_points"]
    if n == 0:
        if funnel.get("retest", 0) and funnel.get("skip_no_hold", 0):
            return "這段有回到兩小時低點附近，但連續三根之前又創新低。"
        if funnel.get("stand", 0) and funnel.get("skip_no_break", 0):
            return "這段有站上 MA20，但之後沒再跌破回測。"
        return "這段樣本沒抓到雙底回測守三根。"
    if pnl > 80 and wr >= 50:
        return "有料：回測守三根比等第二次站上 MA20 更靠近低點。"
    if abs(pnl) <= 40:
        return "抓得到 09-04 那種雙底，但優勢接近零。"
    if pnl > 0:
        return "邊緣：總點數正，但勝率或均筆還不算穩。"
    return "沒料：回測守三根之後仍常再破，停損被掃。"


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} {args.interval} {args.period}", file=sys.stderr)
    df = to_et(load_yfinance(args.symbol, args.interval, args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    print(f"bars={len(df)} {df.index[0]} → {df.index[-1]}", file=sys.stderr)

    params = detect_params(args.interval)
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, two_hour_bars=params["two_hour_bars"])
    trades = simulate(df, sigs, max_hold=params["max_hold"])
    stats = summarize_trades(trades)
    print(
        f"{args.interval} trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_points']:+.1f} avg={stats['avg']:+.1f}"
    )
    print("funnel", funnel)
    _print_trades(df, trades)
    verdict = _verdict(stats, funnel)
    print(f"verdict: {verdict}")

    df_15m = resample_ohlc(df, "15min") if args.interval == "5m" else pd.DataFrame()
    trades_15m: List[TradeResult] = []
    funnel_15m: Dict[str, int] = {}
    verdict_15m = ""
    if not df_15m.empty:
        p15 = detect_params("15m")
        sigs_15m = detect_signals(df_15m, funnel=funnel_15m, two_hour_bars=p15["two_hour_bars"])
        trades_15m = simulate(df_15m, sigs_15m, max_hold=p15["max_hold"])
        stats_15m = summarize_trades(trades_15m)
        print(
            f"15m trades={stats_15m['count']} WR={stats_15m['win_rate']:.1f}% "
            f"pnl={stats_15m['total_points']:+.1f} avg={stats_15m['avg']:+.1f}"
        )
        print("funnel_15m", funnel_15m)
        verdict_15m = _verdict(stats_15m, funnel_15m)
        print(f"verdict_15m: {verdict_15m}")

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(
            html_path,
            df,
            trades,
            args.symbol,
            args.period,
            funnel=funnel,
            verdict=verdict,
            df_15m=df_15m if not df_15m.empty else None,
            trades_15m=trades_15m,
            funnel_15m=funnel_15m,
            verdict_15m=verdict_15m,
        )
        print(f"html={out}")
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 五分 K 雙底：2h低回測守三根")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 5m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--interval", default="5m")
    b.add_argument("--period", default="60d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-w-ma20/index.html")
    b.set_defaults(func=cmd_backtest)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
