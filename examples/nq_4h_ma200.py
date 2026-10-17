#!/usr/bin/env python3
"""NQ 一分 K：破四小時低 → 一小時內站上 MA200，且 MA5>10>20>30。

對齊 2026-09-29 01:35 那波（低 30372.25）：
  01:31 跌破近 4 小時低 30409.25
  02:03 1m 收盤站上 MA200 且 5/10/20/30 多頭
  再等五分 K 也 5>10>20>30（09/29 在 02:20），才進場／通知

用法:
  python3 examples/nq_4h_ma200.py
  python3 examples/nq_4h_ma200.py backtest --period 8d --pages
  python3 examples/nq_4h_ma200.py alert --dry-run --once
  python3 examples/nq_4h_ma200.py alert
  python3 examples/test_nq_4h_ma200.py

Telegram 憑證放 tg_config.env（勿提交）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
except ImportError:
    requests = None  # type: ignore

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
STATE_PATH = ROOT / "tg_alert_state_4h_ma200.json"
CONFIG_ENV = REPO_ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = ROOT / "tg_config.env"
PAGES_HTML = REPO_ROOT / "docs" / "nq-4h-ma200" / "index.html"
VIEW_BRANCH = "cursor/nq-1m-break-bounce-365c"

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    30: "#26a69a",
    60: "#42a5f5",
    200: "#ffffff",
}


def parse_period_days(period: str) -> Optional[int]:
    p = (period or "").strip().lower()
    if p.endswith("mo") and p[:-2].isdigit():
        return int(p[:-2]) * 30
    if p.endswith("d") and p[:-1].isdigit():
        return int(p[:-1])
    if p.endswith("w") and p[:-1].isdigit():
        return int(p[:-1]) * 7
    return None


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "8d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


def load_yahoo_intraday(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_days: int = 7,
) -> pd.DataFrame:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    chunks: List[pd.DataFrame] = []
    cur = start
    delta = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + delta, end)
        part = yf.download(
            symbol,
            interval=interval,
            start=cur,
            end=nxt,
            progress=False,
            auto_adjust=True,
        )
        if part is not None and len(part):
            if isinstance(part.columns, pd.MultiIndex):
                part.columns = part.columns.get_level_values(0)
            part = part.rename(columns=str.title)
            chunks.append(part)
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(part)}", file=sys.stderr)
        else:
            print(f"[data] {cur.date()} → {nxt.date()} empty", file=sys.stderr)
        cur = nxt
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    days = parse_period_days(period)
    if days is not None and days > 8:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        df = load_yahoo_intraday(symbol, interval, start, end, chunk_days=7)
        if not df.empty:
            return df
        print(f"[data] chunked {period} empty, fallback period download", file=sys.stderr)
    return load_yfinance(symbol, interval, period)


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
            q: {"n": len(v), "wins": sum(1 for p in v if p > 0), "pnl": float(sum(v))}
            for q, v in sorted(by_q.items())
        },
    }


@dataclass
class Signal:
    break_idx: int
    flush_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    four_hr_low: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma200: float
    bars_to_entry: int
    m5_close: float = 0.0
    m5_ma5: float = 0.0
    m5_ma10: float = 0.0
    m5_ma20: float = 0.0
    m5_ma30: float = 0.0
    m5_ma200: float = 0.0
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


def quality_from_setup(
    depth: float,
    bars_to_entry: int,
    over_ma200: float,
    m5_over_200: float = 0.0,
) -> Tuple[int, str]:
    score = 0
    if depth >= 20.0:
        score += 1
    if bars_to_entry <= 45:
        score += 1
    if over_ma200 >= 8.0 or m5_over_200 > 0.0:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def _build_m5_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Map each 1m bar to the latest completed 5m bar (no lookahead)."""
    close = df["Close"].astype(float)
    m5 = close.resample("5min", label="right", closed="right").last().dropna()
    feat = {
        "close": m5.to_numpy(float),
        "ma5": m5.rolling(5, min_periods=5).mean().to_numpy(float),
        "ma10": m5.rolling(10, min_periods=10).mean().to_numpy(float),
        "ma20": m5.rolling(20, min_periods=20).mean().to_numpy(float),
        "ma30": m5.rolling(30, min_periods=30).mean().to_numpy(float),
        "ma200": m5.rolling(200, min_periods=200).mean().to_numpy(float),
    }
    n = len(df)
    out = {k: np.full(n, np.nan, dtype=float) for k in feat}
    stack = np.zeros(n, dtype=bool)
    above20 = np.zeros(n, dtype=bool)
    m5_idx = m5.index
    j = 0
    for i, ts in enumerate(df.index):
        while j + 1 < len(m5_idx) and m5_idx[j + 1] <= ts:
            j += 1
        if j < len(m5_idx) and m5_idx[j] <= ts:
            for k in feat:
                out[k][i] = feat[k][j]
            if (
                np.isfinite(out["ma5"][i])
                and np.isfinite(out["ma30"][i])
                and out["ma5"][i] > out["ma10"][i] > out["ma20"][i] > out["ma30"][i]
            ):
                stack[i] = True
            if np.isfinite(out["close"][i]) and np.isfinite(out["ma20"][i]) and out["close"][i] > out["ma20"][i]:
                above20[i] = True
    out["stack"] = stack
    out["above20"] = above20
    return out


def detect_signals(
    df,
    *,
    four_hour_bars: int = 240,
    reclaim_window: int = 60,
    stop_buffer: float = 12.0,
    target_r: float = 2.0,
    min_entry_gap: int = 30,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """破近 4 小時低後，60 根內 1m 收盤 > MA200 且 5>10>20>30，再等五分也多頭。"""
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    ma200 = sma(close, 200)
    four_hr = rolling_min_prev(low, four_hour_bars)
    m5f = _build_m5_features(df)
    signals: List[Signal] = []
    last_entry = -(10**9)
    n = len(close)
    i = max(four_hour_bars, 200)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        if np.isnan(four_hr[i]) or low[i] >= four_hr[i]:
            i += 1
            continue

        bump("break")
        support = float(four_hr[i])
        break_idx = i
        flush_low = float(low[i])
        flush_idx = i
        entered = False
        saw_m1 = False
        limit = min(break_idx + reclaim_window, n - 1)

        for j in range(break_idx + 1, limit + 1):
            if low[j] < flush_low:
                flush_low = float(low[j])
                flush_idx = j
            if np.isnan(ma200[j]) or np.isnan(ma30[j]):
                continue
            stacked = ma5[j] > ma10[j] > ma20[j] > ma30[j]
            above_200 = close[j] > ma200[j]
            m1_ok = stacked and above_200
            if m1_ok and not saw_m1:
                bump("m1_setup")
                saw_m1 = True
            m5_ok = bool(m5f["stack"][j] and m5f["above20"][j])
            if not (m1_ok and m5_ok):
                continue
            bump("setup")
            if j - last_entry < min_entry_gap:
                bump("skip_gap")
                break
            entry = float(close[j])
            stop = flush_low - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                break
            target = entry + risk * target_r
            depth = support - flush_low
            bars = j - break_idx
            m5_over = (
                float(m5f["close"][j] - m5f["ma200"][j])
                if np.isfinite(m5f["ma200"][j])
                else 0.0
            )
            q_score, q_grade = quality_from_setup(depth, bars, entry - float(ma200[j]), m5_over)
            bump("taken")
            signals.append(
                Signal(
                    break_idx,
                    flush_idx,
                    j,
                    entry,
                    stop,
                    target,
                    flush_low,
                    support,
                    float(ma5[j]),
                    float(ma10[j]),
                    float(ma20[j]),
                    float(ma30[j]),
                    float(ma200[j]),
                    bars,
                    m5_close=float(m5f["close"][j]),
                    m5_ma5=float(m5f["ma5"][j]),
                    m5_ma10=float(m5f["ma10"][j]),
                    m5_ma20=float(m5f["ma20"][j]),
                    m5_ma30=float(m5f["ma30"][j]),
                    m5_ma200=float(m5f["ma200"][j]) if np.isfinite(m5f["ma200"][j]) else 0.0,
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = j
            entered = True
            i = j + 5
            break

        if not entered:
            bump("skip_m5" if saw_m1 else "no_reclaim")
            i = break_idx + reclaim_window + 1

    return signals


def simulate(
    df,
    signals: List[Signal],
    *,
    max_hold: int = 90,
    be_after_r: float = 0.70,
    trail_after_r: float = 1.5,
    trail_lock_r: float = 0.5,
    preopen_flat: bool = True,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    results: List[TradeResult] = []

    for sig in signals:
        entry_idx = sig.entry_idx
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = entry - stop
        if risk <= 0:
            continue
        cur_stop = stop
        mfe = 0.0
        entry_hour = df.index[entry_idx].hour
        limit = min(entry_idx + max_hold, len(df) - 1)
        exit_idx = limit
        exit_price = float(close[exit_idx])
        exit_reason = "timeout"

        for k in range(entry_idx + 1, limit + 1):
            mfe = max(mfe, float(high[k] - entry))
            if be_after_r > 0 and mfe / risk >= be_after_r:
                cur_stop = max(cur_stop, entry)
            if trail_after_r > 0 and mfe / risk >= trail_after_r:
                cur_stop = max(cur_stop, entry + trail_lock_r * risk)

            et_h = df.index[k].hour
            et_m = df.index[k].minute
            if preopen_flat and entry_hour < 9 and (et_h > 9 or (et_h == 9 and et_m >= 30)):
                exit_idx, exit_price, exit_reason = k, float(close[k]), "preopen_flat"
                break
            if low[k] <= cur_stop:
                reason = "be" if cur_stop > stop + 1e-9 else "stop"
                exit_idx, exit_price, exit_reason = k, float(cur_stop), reason
                break
            if high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break

        results.append(
            TradeResult(
                sig,
                entry_idx,
                exit_idx,
                entry,
                exit_price,
                stop,
                target,
                float(exit_price - entry),
                exit_reason,
                sig.quality,
            )
        )
    return results


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


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> str:
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
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    start = max(0, sig.break_idx - 25)
    end = min(len(df) - 1, max(trade.exit_idx + 12, sig.entry_idx + 20))
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
        ax.plot(list(xs), ma, color=col, lw=1.55 if nper == 200 else (1.35 if nper <= 20 else 1.05), label=f"MA{nper}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.four_hr_low, color="#8aa193", ls="--", lw=0.85, alpha=0.55)

    bx, fx, ex, xx = (
        sig.break_idx - start,
        sig.flush_idx - start,
        trade.entry_idx - start,
        trade.exit_idx - start,
    )
    if 0 <= fx < len(window):
        ax.scatter([fx], [sig.break_low], s=42, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (fx, sig.break_low),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "MA200",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#86efac",
            fontsize=8,
        )
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  破4h→MA200+5m  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    svg = _inline_mpl_svg(fig, f"t{trade_no:02d}_")
    plt.close(fig)
    return svg


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int, prefix: str = "t") -> str:
    et = df.index[trade.entry_idx]
    return f"{prefix}{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(df: pd.DataFrame, trades: List[TradeResult], html_path: Path) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        bt = df.index[t.signal.break_idx]
        ft = df.index[t.signal.flush_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "be": "tag-time"}.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i)
        svg = draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        ref = ""
        if abs(t.signal.break_low - 30372.25) < 0.3 and et.strftime("%Y-%m-%d") == "2026-09-29":
            ref = "<span class='tag tag-info'>09/29 參考圖</span>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            "<span class='tag tag-info'>1m+5m</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"{ref}"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破4h {bt.strftime('%H:%M')}  低點 {ft.strftime('%H:%M')} {t.signal.break_low:.2f}\n"
            f"4h低 {t.signal.four_hr_low:.2f}  距破底 {t.signal.bars_to_entry} 分\n"
            f"1m MA5 {t.signal.ma5:.1f} > 10 {t.signal.ma10:.1f} > 20 {t.signal.ma20:.1f} > 30 {t.signal.ma30:.1f}\n"
            f"1m MA200 {t.signal.ma200:.1f}  收盤高出 {t.entry_price - t.signal.ma200:.1f}\n"
            f"5m C {t.signal.m5_close:.1f}  MA5 {t.signal.m5_ma5:.1f} > 10 {t.signal.m5_ma10:.1f} "
            f"> 20 {t.signal.m5_ma20:.1f} > 30 {t.signal.m5_ma30:.1f}"
            "</pre>"
            f"<div class='mini-chart'>{svg}</div>"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    verdict: str = "",
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.1f}" for q, info in stats.get("by_quality", {}).items()]
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    img_dir = out.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    cards = _render_trade_cards(df, trades, out)
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破4h低 {funnel.get('break', 0)} → "
            f"1m站上MA200+排列 {funnel.get('m1_setup', 0)} → "
            f"五分也多頭 {funnel.get('setup', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（沒站上 {funnel.get('no_reclaim', 0)} · 等不到5m {funnel.get('skip_m5', 0)} · "
            f"間隔 {funnel.get('skip_gap', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 破4h低站上 MA200</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
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
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 破4小時低 → 1m MA200 + 五分多頭</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">跌破近 4 小時低點後，60 根內 1 分收盤站上 MA200 且 MA5&gt;10&gt;20&gt;30，再等<strong>五分 K 也 5&gt;10&gt;20&gt;30</strong> 才進場。停損在波段低 −12，目標 2R。09/29：01:35 低 30372.25 → 02:03 一分條件到 → 02:20 五分排列後進場。</p>
{verdict_html}
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">均筆<b>{stats['avg']:+.1f}</b></div>
</div>
<p class="muted">{escape(q_line)}</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{cards or "<div class='empty'>這段期間沒有破4h後一小時內站上 MA200 的訊號</div>"}
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


def load_dotenv(path: Path = CONFIG_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def tg_send(token: str, chat_id: str, text: str, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    data = r.json()
    if not data.get("ok"):
        print(f"[tg] API error: {data}", file=sys.stderr)
        return False
    return True


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _ts_et(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize("UTC").tz_convert(ET)
    return ts.tz_convert(ET)


def entry_key(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    return f"{ts.isoformat()}|{sig.entry_price:.2f}"


def fmt_entry(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    br = _ts_et(df.index[sig.break_idx])
    fl = _ts_et(df.index[sig.flush_idx])
    risk = sig.entry_price - sig.stop_price
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>破4h低 站上1m MA200</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"進場: <code>{sig.entry_price:.2f}</code>\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> (2R)\n"
        f"破底: <code>{br.strftime('%H:%M')}</code> 低點 <code>{fl.strftime('%H:%M')}</code> {sig.break_low:.2f}\n"
        f"4h低: <code>{sig.four_hr_low:.2f}</code> · 距破底 {sig.bars_to_entry} 分\n"
        f"1m排列: 5 {sig.ma5:.1f} &gt; 10 {sig.ma10:.1f} &gt; 20 {sig.ma20:.1f} &gt; 30 {sig.ma30:.1f}\n"
        f"5m排列: 5 {sig.m5_ma5:.1f} &gt; 10 {sig.m5_ma10:.1f} &gt; 20 {sig.m5_ma20:.1f} &gt; 30 {sig.m5_ma30:.1f}\n"
        f"1m MA200: <code>{sig.ma200:.2f}</code>  現價 <code>{last:.2f}</code>\n"
        f"品質: Q{sig.quality}\n"
        f"#破底反彈 #NQ #MA200 #5m"
    )


def scan_once(
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    seed_alert: bool,
    lookback_hours: float,
    period: str = "5d",
) -> None:
    df = to_et(load_yfinance("NQ=F", "1m", period))
    sigs = detect_signals(df)
    state = load_state()
    alerted: Set[str] = set(state.get("alerted_entries") or [])
    now = datetime.now(ET)
    cutoff = now.timestamp() - lookback_hours * 3600
    first_run = not STATE_PATH.exists() or (not alerted and not state.get("initialized"))

    new_entries = []
    for sig in sigs:
        k = entry_key(df, sig)
        ts = _ts_et(df.index[sig.entry_idx])
        if ts.timestamp() < cutoff:
            alerted.add(k)
            continue
        if k in alerted:
            continue
        new_entries.append((k, sig, ts))

    if first_run and not seed_alert:
        for k, _, _ in new_entries:
            alerted.add(k)
        state["alerted_entries"] = sorted(alerted)[-200:]
        state["initialized"] = True
        state["last_scan"] = now.isoformat()
        save_state(state)
        print(
            f"[{now.strftime('%H:%M:%S')} ET] init: marked {len(new_entries)} recent signals, "
            f"bars={len(df)} last={df['Close'].iloc[-1]:.2f}"
        )
        return

    sent = 0
    for k, sig, ts in new_entries:
        ok = tg_send(token, chat_id, fmt_entry(df, sig), dry_run=dry_run)
        if ok:
            alerted.add(k)
            sent += 1
            print(f"[entry] {ts} Q{sig.quality} @ {sig.entry_price:.2f}")

    state["alerted_entries"] = sorted(alerted)[-200:]
    state["last_scan"] = now.isoformat()
    state["initialized"] = True
    save_state(state)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] sent={sent} pending={len(new_entries)-sent} "
        f"bars={len(df)} last={df['Close'].iloc[-1]:.2f}"
    )


def _print_trades(df, trades) -> None:
    for i, t in enumerate(trades, 1):
        print(
            f"  [{i}] Q{t.quality} 破 {df.index[t.signal.break_idx].strftime('%m-%d %H:%M')} "
            f"低 {t.signal.break_low:.2f}  進 {df.index[t.entry_idx].strftime('%H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%H:%M')} {t.exit_reason} {t.pnl_points:+.1f}  "
            f"{t.signal.bars_to_entry}分"
        )


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} 1m {args.period}", file=sys.stderr)
    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    print(f"bars={len(df)} {df.index[0]} → {df.index[-1]}", file=sys.stderr)

    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(
        f"trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_points']:+.1f} avg={stats['avg']:+.1f}"
    )
    print("funnel", funnel)
    _print_trades(df, trades)

    if stats["count"] == 0:
        verdict = "這段樣本有破 4h 低，但一小時內沒等到 1m 站上 MA200 且五分也多頭。"
    elif any(abs(t.signal.break_low - 30372.25) < 0.3 for t in trades):
        verdict = (
            "抓得到 09/29：01:31 破 4h 低，02:03 一分站上 MA200，02:20 五分 5>10>20>30 才進場。"
            "五分過濾後勝率比單看一分高。"
        )
    elif stats["win_rate"] >= 60 and stats["total_points"] > 0:
        verdict = "五分確認後樣本較少，勝率上來了。通知仍不是保證優勢。"
    elif stats["total_points"] > 0:
        verdict = "有訊號，總點數為正。這是通知規則，不是保證優勢。"
    else:
        verdict = "抓得到型態，但這段回測總點數為負。通知仍可發，倉位自己看。"
    print(f"verdict: {verdict}")

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(html_path, df, trades, args.symbol, args.period, funnel=funnel, verdict=verdict)
        print(f"html={out}")
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2
    if args.test:
        ok = tg_send(
            token or "",
            chat_id or "",
            f"✅ 破4h低 MA200 bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1
    print(
        f"4h→MA200 TG | interval={args.interval}s | dry_run={args.dry_run} | lookback={args.lookback_hours}h"
    )
    while True:
        try:
            scan_once(
                token or "",
                chat_id or "",
                dry_run=args.dry_run,
                seed_alert=args.seed_alert,
                lookback_hours=args.lookback_hours,
                period=args.period,
            )
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(max(15, args.interval))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 破4小時低後一小時內站上 1m MA200")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-4h-ma200/index.html")
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("alert", help="Telegram 輪詢")
    a.add_argument("--interval", type=int, default=None)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=None)
    a.add_argument("--period", default="5d")
    a.set_defaults(func=cmd_alert)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "alert":
        if args.interval is None:
            args.interval = int(env("POLL_SECONDS", "60") or 60)
        if args.lookback_hours is None:
            args.lookback_hours = float(env("LOOKBACK_HOURS", "36") or 36)
        return cmd_alert(args)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
