#!/usr/bin/env python3
"""NQ 五分 K W 底：右側站上 MA20 進場（最多）。

圖上那種雙底：左腳 L1、中間頸線、右腳 L2 都在五分 MA20 下面，
右腳確認後第一根收盤站上 MA20 就做多，不等頸線、也不等 5/10/20 多排。

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


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == min(window)


def _find_swing_lows(lows: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback)]


def quality_from_w(low_gap_pts: float, neck_pts: float, stand_pts: float) -> Tuple[int, str]:
    score = 0
    if abs(low_gap_pts) <= 15.0:
        score += 1
    if neck_pts >= 30.0:
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
    swing_lookback: int = 2,
    min_bars_between_lows: int = 5,
    max_bars_between_lows: int = 72,
    low_below_pts: float = 25.0,
    low_above_pts: float = 60.0,
    min_neck_pts: float = 15.0,
    min_right_leg_pts: float = 12.0,
    min_neck_offset: int = 2,
    min_right_bars: int = 2,
    max_mid_swing_lows: int = 2,
    min_prior_drop_pts: float = 20.0,
    prior_lookback: int = 36,
    max_bars_to_stand: int = 36,
    invalidate_pts: float = 8.0,
    stop_buffer: float = 4.0,
    target_r: float = 2.0,
    max_risk: float = 100.0,
    min_entry_gap: int = 12,
    ma_period: int = 20,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    視覺 W 底（兩低點在 MA20 下）形成後，右側第一根收盤站上 MA20 進場。

    最多：兩低點容忍寬、不等頸線、不等多排、含盤外。
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma20 = sma(close, ma_period)
    n = len(close)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    swing_lows = _find_swing_lows(low.tolist(), swing_lookback)
    signals: List[Signal] = []
    last_entry = -(10**9)

    for i, first_idx in enumerate(swing_lows):
        first_low = float(low[first_idx])
        look_from = max(0, first_idx - prior_lookback)
        prior_high = float(np.max(high[look_from : first_idx + 1]))
        if prior_high - first_low < min_prior_drop_pts:
            continue
        ma1 = ma20[first_idx]
        if np.isnan(ma1) or first_low >= ma1:
            continue
        bump("left")

        for second_idx in swing_lows[i + 1 :]:
            gap = second_idx - first_idx
            if gap < min_bars_between_lows:
                continue
            if gap > max_bars_between_lows:
                break

            second_low = float(low[second_idx])
            rel = second_low - first_low
            if rel < -low_below_pts or rel > low_above_pts:
                continue

            ma2 = ma20[second_idx]
            if np.isnan(ma2) or second_low >= ma2:
                continue

            if second_idx - first_idx < 2:
                continue
            mid = slice(first_idx + 1, second_idx)
            neckline_price = float(np.max(high[mid]))
            neckline_idx = first_idx + 1 + int(np.argmax(high[mid]))
            floor = min(first_low, second_low)
            neck_pts = neckline_price - floor
            if neck_pts < min_neck_pts:
                continue
            if neckline_idx - first_idx < min_neck_offset:
                continue
            if second_idx - neckline_idx < min_right_bars:
                continue
            extra = sum(1 for j in swing_lows if first_idx < j < second_idx)
            if extra > max_mid_swing_lows:
                continue
            if neckline_price - second_low < min_right_leg_pts:
                continue
            bump("w")

            confirm = second_idx + swing_lookback
            if confirm >= n:
                continue
            start_k = max(confirm, ma_period)
            end_k = min(n, second_idx + max_bars_to_stand + 1)
            fail = floor - invalidate_pts
            entry_idx: Optional[int] = None
            for k in range(start_k, end_k):
                if float(np.min(low[second_idx : k + 1])) < fail:
                    bump("skip_new_low")
                    entry_idx = None
                    break
                m = ma20[k]
                prev_m = ma20[k - 1]
                if np.isnan(m) or np.isnan(prev_m):
                    continue
                if close[k] > m and close[k - 1] <= prev_m:
                    entry_idx = k
                    break
            if entry_idx is None:
                bump("skip_no_stand")
                continue
            if float(np.min(low[second_idx : entry_idx + 1])) < fail:
                bump("skip_new_low")
                continue
            if entry_idx - last_entry < min_entry_gap:
                bump("skip_gap")
                continue

            entry = float(close[entry_idx])
            stop = floor - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                continue
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            measured = neckline_price + neck_pts
            r_tgt = entry + risk * target_r
            target = max(measured, r_tgt)
            if target <= entry:
                bump("skip_bad_target")
                continue

            stand_pts = entry - float(ma20[entry_idx])
            low_gap = second_low - first_low
            q_score, q_grade = quality_from_w(low_gap, neck_pts, stand_pts)
            bump("taken")
            signals.append(
                Signal(
                    first_low_idx=first_idx,
                    second_low_idx=second_idx,
                    neckline_idx=neckline_idx,
                    entry_idx=entry_idx,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=float(target),
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    ma20=float(ma20[entry_idx]),
                    stand_pts=stand_pts,
                    low_gap_pts=low_gap,
                    neck_pts=neck_pts,
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = entry_idx

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

    l1x, l2x, nx, ex, xx = (
        sig.first_low_idx - start,
        sig.second_low_idx - start,
        sig.neckline_idx - start,
        trade.entry_idx - start,
        trade.exit_idx - start,
    )
    w_x, w_y = [], []
    if 0 <= l1x < len(window):
        w_x.append(l1x)
        w_y.append(sig.first_low)
        ax.scatter([l1x], [sig.first_low], s=36, color="#facc15", zorder=6)
        ax.annotate("L1", (l1x, sig.first_low), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= nx < len(window):
        w_x.append(nx)
        w_y.append(sig.neckline)
        ax.scatter([nx], [sig.neckline], s=32, color="#f59e0b", zorder=5)
        ax.annotate("頸線", (nx, sig.neckline), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#fbbf24", fontsize=8)
    if 0 <= l2x < len(window):
        w_x.append(l2x)
        w_y.append(sig.second_low)
        ax.scatter([l2x], [sig.second_low], s=36, color="#facc15", zorder=6)
        ax.annotate("L2", (l2x, sig.second_low), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= ex < len(window):
        ax.scatter([ex], [trade.entry_price], s=44, color="#22c55e", zorder=7, marker="^")
        ax.annotate("進場", (ex, trade.entry_price), textcoords="offset points", xytext=(0, 10),
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
    ax.set_title(f"#{trade_no}  W底右側站上MA20  {window.index[0].strftime('%m-%d %H:%M')} ET",
                 color="#d7e3d4", fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.35, labelcolor="#d7e3d4")
    fig.tight_layout(pad=0.6)
    svg = _inline_mpl_svg(fig, f"t{trade_no:02d}_")
    plt.close(fig)
    return svg


def _ts(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    t = df.index[idx]
    return t.tz_convert(ET) if getattr(t, "tzinfo", None) else t


def _render_trade_cards(df: pd.DataFrame, trades: List[TradeResult]) -> str:
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
            f"<span class='tag tag-info'>W底</span>"
            f"<span class='tag tag-info'>右側MA20</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"L1 {t.signal.first_low:.2f} / L2 {t.signal.second_low:.2f}  "
            f"差 {t.signal.low_gap_pts:+.1f}pt\n"
            f"頸線 {t.signal.neckline:.2f}  高度 {t.signal.neck_pts:.1f}pt\n"
            f"L2 {l2t.strftime('%m-%d %H:%M')}  右側站上 MA20 {t.signal.ma20:.2f}  "
            f"+{t.signal.stand_pts:.1f}pt"
            "</pre>"
            f"<div class='mini-chart'>{chart}</div>"
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
    q_bits = []
    for q, info in stats.get("by_quality", {}).items():
        q_bits.append(f"Q{q} {info['n']}筆 {info['pnl']:+.1f}")
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades)
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：左腳 {funnel.get('left', 0)} → "
            f"W底 {funnel.get('w', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（沒站上 {funnel.get('skip_no_stand', 0)} · 新低 {funnel.get('skip_new_low', 0)} · "
            f"間隔 {funnel.get('skip_gap', 0)} · 風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分 K W底 右側站上MA20</title>
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
<h1>{escape(symbol)} 五分 K W底 右側站上MA20</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">雙底 L1／L2 都在五分 MA20 下面，右腳確認後第一根收盤站上 MA20 進場。不等頸線、不等 5/10/20 多排。停損在兩低點下方 4 點；目標取量度或 2R 較遠者。持倉最多 48 根。含盤外。</p>
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
{cards or "<div class='empty'>無 W 底右側站上 MA20 訊號</div>"}
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
            f"  [{prefix}{i}] Q{t.quality} L2 {l2.strftime('%m-%d %H:%M')} "
            f"進 {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"L1/L2 {t.signal.first_low:.1f}/{t.signal.second_low:.1f} "
            f"頸 {t.signal.neck_pts:.0f}pt 站上+{t.signal.stand_pts:.1f}"
        )


def _verdict(stats: dict, funnel: Dict[str, int]) -> str:
    n = stats["count"]
    wr = stats["win_rate"]
    pnl = stats["total_points"]
    if n == 0:
        if funnel.get("w", 0) and funnel.get("skip_no_stand", 0):
            return "這段有做出 W，但右側一直沒站上 MA20，或站上前又破底。"
        return "這段樣本沒抓到 W 底右側站上 MA20。"
    if pnl > 80 and wr >= 50:
        return "有料：右側站上 MA20 比等頸線更早，這段抓得到且點數為正。"
    if abs(pnl) <= 40:
        return "抓得到雙底翻上均線，但優勢接近零。假 W 破底會把量度吐回去。"
    if pnl > 0:
        return "邊緣：總點數正，但勝率或均筆還不算穩。"
    return "沒料：右側剛站上 MA20 離兩低點已遠，停損偏寬，假 W 一次吐完。"


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} {args.interval} {args.period}", file=sys.stderr)
    df = to_et(load_yfinance(args.symbol, args.interval, args.period))
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
    verdict = _verdict(stats, funnel)
    print(f"verdict: {verdict}")

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
        )
        print(f"html={out}")
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 五分 K W底 右側站上MA20（最多）")
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
