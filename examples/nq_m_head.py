#!/usr/bin/env python3
"""NQ 一分 K 高檔 M 頭：收盤跌破 MA60 做空。

用法:
  python3 examples/nq_m_head.py
  python3 examples/nq_m_head.py --period 30d --pages
  python3 examples/nq_m_head.py --period 8d --html output/nq_m_head.html
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.patterns import MHeadPattern, detect_m_heads

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-m-head" / "index.html"
POINT_VALUE = 20.0
TICK = 0.25

MA_PERIODS = (5, 10, 20, 30, 60, 120, 200)
MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    30: "#26a69a",
    60: "#42a5f5",
    120: "#26c6da",
    200: "#ab47bc",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def parse_period_days(period: str) -> Optional[int]:
    p = (period or "").strip().lower()
    if p.endswith("mo") and p[:-2].isdigit():
        return int(p[:-2]) * 30
    if p.endswith("d") and p[:-1].isdigit():
        return int(p[:-1])
    if p.endswith("w") and p[:-1].isdigit():
        return int(p[:-1]) * 7
    return None


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna()


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    return _normalize_ohlc(df)


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
            chunks.append(_normalize_ohlc(part))
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(chunks[-1])}", file=sys.stderr)
        else:
            print(f"[data] {cur.date()} → {nxt.date()} empty", file=sys.stderr)
        cur = nxt
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo 1m period= 最多約 7–8 天；超過改用 7 日切片（約可回看 30 天）。"""
    days = parse_period_days(period)
    if days is not None and days > 8:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        if interval == "1m":
            # Yahoo 1m 實際只能回看約 30 天，起點再早會整段空掉
            min_start = end - timedelta(days=29, hours=18)
            if start < min_start:
                start = min_start
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


def round_tick(price: float) -> float:
    return round(price / TICK) * TICK


def sma(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    timestamp: pd.Timestamp
    entry: float
    stop_loss: float
    target: float
    pattern: MHeadPattern
    bar_idx: int
    ma60: float
    ma20: float
    ma5: float

    @property
    def risk(self) -> float:
        return self.stop_loss - self.entry

    @property
    def reward(self) -> float:
        return self.entry - self.target


@dataclass
class TradeResult:
    signal: Signal
    exit_price: float
    exit_time: pd.Timestamp
    exit_idx: int
    exit_reason: str
    pnl_points: float
    pnl_dollars: float


def _is_high_level(
    df: pd.DataFrame,
    pattern: MHeadPattern,
    ma60: np.ndarray,
    *,
    lookback: int = 120,
    near_pct: float = 0.002,
) -> bool:
    """高峰需接近近 lookback 根高點，且兩頭頂在 MA60 上方。"""
    h2 = pattern.second_high_idx
    if h2 >= len(ma60) or np.isnan(ma60[pattern.first_high_idx]) or np.isnan(ma60[h2]):
        return False
    if pattern.first_high < float(ma60[pattern.first_high_idx]):
        return False
    if pattern.second_high < float(ma60[h2]):
        return False
    if float(df["close"].iloc[h2]) < float(ma60[h2]):
        return False
    start = max(0, h2 - lookback)
    window_high = float(df["high"].iloc[start : h2 + 1].max())
    if window_high <= 0:
        return False
    return pattern.peak >= window_high * (1.0 - near_pct)


def generate_signals(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 7,
    high_tolerance_pct: float = 0.0012,
    min_bars_between_highs: int = 20,
    max_bars_between_highs: int = 75,
    min_depth_pct: float = 0.0015,
    high_level_lookback: int = 120,
    high_level_pct: float = 0.0015,
    entry_window: int = 35,
    stop_buffer: float = 8.0,
    min_risk: float = 50.0,
    max_risk: float = 220.0,
    min_h2_extension: float = 30.0,
    min_break_pts: float = 1.0,
    session_start: Optional[int] = None,
    session_end: Optional[int] = None,
    target_r: float = 1.5,
    use_measured_target: bool = True,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """高檔 M 頭確認後，收盤跌破 MA60 做空。"""
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    patterns = detect_m_heads(
        df,
        swing_lookback=swing_lookback,
        high_tolerance_pct=high_tolerance_pct,
        min_bars_between_highs=min_bars_between_highs,
        max_bars_between_highs=max_bars_between_highs,
        min_depth_pct=min_depth_pct,
    )
    fun["m_heads"] = len(patterns)

    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    n = len(df)
    signals: List[Signal] = []
    used_entry: set[int] = set()

    for p in patterns:
        confirm = p.second_high_idx + swing_lookback
        if confirm >= n:
            bump("skip_unconfirmed")
            continue
        if not _is_high_level(
            df, p, ma60, lookback=high_level_lookback, near_pct=high_level_pct
        ):
            bump("skip_not_high")
            continue
        h2_ma = float(ma60[p.second_high_idx])
        if float(df["close"].iloc[p.second_high_idx]) - h2_ma < min_h2_extension:
            bump("skip_thin_ext")
            continue
        bump("high_level")

        end = min(n - 1, confirm + entry_window)
        invalidated = False
        entry_idx: int | None = None
        for k in range(confirm, end + 1):
            if high[k] > p.peak:
                invalidated = True
                break
            if np.isnan(ma60[k]):
                continue
            prev_close = close[k - 1] if k > 0 else close[k]
            prev_ma = ma60[k - 1] if k > 0 and not np.isnan(ma60[k - 1]) else ma60[k]
            under = close[k] <= ma60[k] - min_break_pts
            crossed = under and prev_close >= prev_ma
            already_under = k == confirm and under
            if crossed or already_under:
                entry_idx = k
                break
        if invalidated:
            bump("skip_invalidated")
            continue
        if entry_idx is None:
            bump("skip_no_ma60")
            continue
        if session_start is not None and session_end is not None:
            ts = df.index[entry_idx]
            if getattr(ts, "tzinfo", None):
                ts = ts.tz_convert(ET)
            minutes = int(ts.hour) * 60 + int(ts.minute)
            if not (session_start <= minutes < session_end):
                bump("skip_session")
                continue
        if entry_idx in used_entry:
            bump("skip_dup_entry")
            continue

        entry = round_tick(float(close[entry_idx]))
        stop = round_tick(p.peak + stop_buffer)
        risk = stop - entry
        if risk < min_risk:
            bump("skip_tiny_risk")
            continue
        if risk > max_risk:
            bump("skip_wide_risk")
            continue

        measured = round_tick(p.target)
        r_target = round_tick(entry - risk * target_r)
        if use_measured_target:
            target = min(measured, r_target)
        else:
            target = r_target
        if target >= entry:
            bump("skip_bad_target")
            continue

        bump("taken")
        used_entry.add(entry_idx)
        signals.append(
            Signal(
                timestamp=df.index[entry_idx],
                entry=entry,
                stop_loss=stop,
                target=target,
                pattern=p,
                bar_idx=entry_idx,
                ma60=float(ma60[entry_idx]),
                ma20=float(ma20[entry_idx]) if not np.isnan(ma20[entry_idx]) else 0.0,
                ma5=float(ma5[entry_idx]) if not np.isnan(ma5[entry_idx]) else 0.0,
            )
        )

    return sorted(signals, key=lambda s: s.bar_idx)


def run_backtest(
    df: pd.DataFrame,
    signals: Sequence[Signal] | None = None,
    *,
    max_bars_hold: int = 120,
) -> List[TradeResult]:
    """做空：先停損、再停利、逾時以收盤平倉。持倉中不重疊新單。"""
    if signals is None:
        signals = generate_signals(df)
    results: List[TradeResult] = []
    position_open_until = -1

    for sig in signals:
        entry_idx = sig.bar_idx
        if entry_idx <= position_open_until:
            continue
        end_idx = min(entry_idx + max_bars_hold, len(df) - 1)
        exit_price = float(df["close"].iloc[end_idx])
        exit_time = df.index[end_idx]
        exit_reason = "time_stop"
        exit_idx = end_idx

        for i in range(entry_idx + 1, end_idx + 1):
            lo = float(df["low"].iloc[i])
            hi = float(df["high"].iloc[i])
            if hi >= sig.stop_loss:
                exit_price = sig.stop_loss
                exit_time = df.index[i]
                exit_reason = "stop_loss"
                exit_idx = i
                break
            if lo <= sig.target:
                exit_price = sig.target
                exit_time = df.index[i]
                exit_reason = "take_profit"
                exit_idx = i
                break

        position_open_until = exit_idx
        pnl_points = sig.entry - exit_price
        results.append(
            TradeResult(
                signal=sig,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_idx=exit_idx,
                exit_reason=exit_reason,
                pnl_points=pnl_points,
                pnl_dollars=pnl_points * POINT_VALUE,
            )
        )
    return results


def summarize(results: Sequence[TradeResult]) -> dict:
    if not results:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_points": 0.0,
            "total_pnl_dollars": 0.0,
            "avg_pnl_points": 0.0,
        }
    wins = sum(1 for r in results if r.pnl_points > 0)
    total = sum(r.pnl_points for r in results)
    dollars = sum(r.pnl_dollars for r in results)
    return {
        "trades": len(results),
        "wins": wins,
        "losses": len(results) - wins,
        "win_rate": wins / len(results),
        "total_pnl_points": total,
        "total_pnl_dollars": dollars,
        "avg_pnl_points": total / len(results),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _fmt_time(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(ET) if getattr(ts, "tzinfo", None) else ts
    return t.strftime("%m-%d %H:%M")


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


def _chart_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    p = trade.signal.pattern
    start = max(0, p.first_high_idx - 16)
    end = min(
        len(df) - 1,
        max(trade.exit_idx + 10, trade.signal.bar_idx + 18, p.second_high_idx + 16),
    )
    return start, end


def _trade_img_name(trade: TradeResult, trade_no: int) -> str:
    ts = trade.signal.timestamp
    if getattr(ts, "tzinfo", None):
        ts = ts.tz_convert(ET)
    return f"m{trade_no:02d}_{ts.strftime('%m%d_%H%M')}.png"


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

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
    p = sig.pattern
    start, end = _chart_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    close_full = df["close"].astype(float)

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

    for period in MA_PERIODS:
        ma = close_full.rolling(period, min_periods=period).mean().iloc[start : end + 1]
        if ma.notna().sum() == 0:
            continue
        lw = 1.85 if period == 60 else (1.35 if period <= 20 else 1.05)
        ax.plot(list(xs), ma, color=MA_COLORS[period], lw=lw, label=f"MA{period}")

    neck_start = p.first_high_idx - start
    neck_end = sig.bar_idx - start
    ax.hlines(p.neckline, max(0, neck_start), min(len(window) - 1, neck_end),
              colors="#ffa726", linestyles="--", lw=1.0, alpha=0.9)
    ax.axhline(sig.stop_loss, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(sig.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.ma60, color="#42a5f5", ls=":", lw=0.8, alpha=0.45)

    h1_rel = p.first_high_idx - start
    h2_rel = p.second_high_idx - start
    if 0 <= h1_rel < len(window):
        ax.scatter([h1_rel], [p.first_high], s=42, color="#42a5f5", zorder=5)
        ax.annotate("H1", (h1_rel, p.first_high), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= h2_rel < len(window):
        ax.scatter([h2_rel], [p.second_high], s=42, color="#ec407a", zorder=5)
        ax.annotate("H2", (h2_rel, p.second_high), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#f9a8d4", fontsize=8)

    entry_rel = sig.bar_idx - start
    exit_rel = trade.exit_idx - start
    if 0 <= entry_rel < len(window):
        ax.axvline(entry_rel, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([entry_rel], [sig.entry], s=48, color="#ff5252", marker="v", zorder=6)
    if 0 <= exit_rel < len(window):
        ax.axvline(exit_rel, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [exit_rel],
            [trade.exit_price],
            s=44,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    y_min = float(window["low"].min())
    y_max = float(window["high"].max())
    pad = max((y_max - y_min) * 0.06, 2.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  高檔M頭空  {_fmt_time(sig.timestamp)} → {_fmt_time(trade.exit_time)}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=7)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [_fmt_time(window.index[i]) for i in ticks],
        color="#8aa193",
        rotation=20,
        ha="right",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _render_trade_card(
    df: pd.DataFrame,
    trade: TradeResult,
    trade_no: int,
    img_href: str,
) -> str:
    sig = trade.signal
    p = sig.pattern
    pnl_class = "pnl-win" if trade.pnl_points > 0 else ("pnl-flat" if trade.pnl_points == 0 else "pnl-loss")
    tag_class = {
        "take_profit": "tag-tp",
        "stop_loss": "tag-sl",
    }.get(trade.exit_reason, "tag-time")
    gap = abs(p.first_high - p.second_high)
    avg = (p.first_high + p.second_high) / 2
    gap_pct = gap / avg * 100 if avg else 0
    risk = sig.risk
    return (
        "<article class='trade-card'>"
        "<header class='card-header'>"
        f"<div class='card-title'><span class='trade-no'>#{trade_no}</span>"
        f"<span class='trade-time'>{escape(_fmt_time(sig.timestamp))} → {escape(_fmt_time(trade.exit_time))}</span></div>"
        f"<div class='card-pnl {pnl_class}'>{trade.pnl_points:+.1f} pts</div>"
        "</header>"
        "<div class='tags'>"
        f"<span class='tag {tag_class}'>{escape(trade.exit_reason)}</span>"
        "<span class='tag tag-info'>M頭</span>"
        "<span class='tag tag-info'>1m 空</span>"
        "</div>"
        "<pre class='trade-detail'>"
        f"entry(跌破MA60) {sig.entry:.2f}\n"
        f"MA60 {sig.ma60:.2f} / MA20 {sig.ma20:.2f} / MA5 {sig.ma5:.2f}\n"
        f"stop H高點+緩衝 {sig.stop_loss:.2f}  (風險 {risk:.1f})\n"
        f"TP {sig.target:.2f}\n"
        f"exit {trade.exit_price:.2f}  {trade.exit_reason}\n"
        f"M頭 H1 {p.first_high:.2f} / H2 {p.second_high:.2f}\n"
        f"頸線 {p.neckline:.2f} / 深度 {p.depth:.2f}\n"
        f"雙頂價差 {gap_pct:.2f}%\n"
        f"$ {trade.pnl_dollars:+,.2f} NQ×1"
        "</pre>"
        f"<div class='mini-chart'><img src='{escape(img_href)}' alt='#{trade_no}' "
        "style='width:100%;display:block;border-radius:10px'/></div>"
        "</article>"
    )


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    *,
    embed_images: bool = False,
    note: str = "",
) -> Path:
    stats = summarize(trades)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out.parent / "img"
    cards: List[str] = []
    for i, trade in enumerate(trades, 1):
        img_name = _trade_img_name(trade, i)
        png = draw_trade_png(df, trade, img_dir / img_name, i)
        href = _img_data_uri(png) if embed_images else f"img/{img_name}"
        cards.append(_render_trade_card(df, trade, i, href))

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_pnl_points"] >= 0 else "pnl-loss"
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：M頭 {funnel.get('m_heads', 0)} → "
            f"高檔 {funnel.get('high_level', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（非高檔 {funnel.get('skip_not_high', 0)} · "
            f"伸幅不足 {funnel.get('skip_thin_ext', 0)} · "
            f"未破MA60 {funnel.get('skip_no_ma60', 0)} · "
            f"破高失效 {funnel.get('skip_invalidated', 0)} · "
            f"風險過窄 {funnel.get('skip_tiny_risk', 0)} · "
            f"風險過寬 {funnel.get('skip_wide_risk', 0)}）</p>"
        )
    note_line = f"<p class='muted'>{escape(note)}</p>" if note else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 一分K 高檔M頭跌破MA60做空</title>
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
<h1>{escape(symbol)} 一分K 高檔M頭 · 跌破MA60做空</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
{note_line}
<div class="cards">
<div class="card">筆數<b>{stats['trades']}</b></div>
<div class="card">勝率<b>{stats['win_rate'] * 100:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_pnl_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['losses']}</b></div>
</div>
<p class="muted">均損益 {stats['avg_pnl_points']:+.1f} 點 · ${stats['total_pnl_dollars']:+,.0f}（NQ×1）</p>
{funnel_line}
<div class="equity">{_equity_svg([t.pnl_points for t in trades])}</div>
</section>
{''.join(cards) or "<div class='empty'>未偵測到高檔M頭跌破MA60訊號</div>"}
</div>
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    funnel: Dict[str, int] = {}
    extra = {}
    if getattr(args, "rth", False):
        extra["session_start"] = 9 * 60 + 30
        extra["session_end"] = 16 * 60
    sigs = generate_signals(df, funnel=funnel, **extra)
    trades = run_backtest(df, sigs)
    stats = summarize(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(
        f"trades={stats['trades']} WR={stats['win_rate'] * 100:.1f}% "
        f"pnl={stats['total_pnl_points']:+.1f} ${stats['total_pnl_dollars']:+,.0f}"
    )
    print(
        "funnel "
        f"m={funnel.get('m_heads', 0)} high={funnel.get('high_level', 0)} "
        f"taken={funnel.get('taken', 0)} "
        f"not_high={funnel.get('skip_not_high', 0)} "
        f"thin={funnel.get('skip_thin_ext', 0)} "
        f"no_ma60={funnel.get('skip_no_ma60', 0)} "
        f"invalid={funnel.get('skip_invalidated', 0)} "
        f"tiny={funnel.get('skip_tiny_risk', 0)} "
        f"wide={funnel.get('skip_wide_risk', 0)}"
    )
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] {_fmt_time(t.signal.timestamp)} -> {_fmt_time(t.exit_time)} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"entry={t.signal.entry:.2f} stop={t.signal.stop_loss:.2f} tp={t.signal.target:.2f}"
        )

    html_path = args.html
    if args.pages:
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(html_path, df, trades, args.symbol, args.period, funnel=funnel)
        view = Path(html_path).with_name("view.html")
        if args.pages or Path(html_path).name == "index.html":
            write_html_report(
                view,
                df,
                trades,
                args.symbol,
                args.period,
                funnel=funnel,
                embed_images=True,
                note="圖已內嵌，手機請往下捲。",
            )
            print(f"view={view}")
        print(f"html={out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 一分K 高檔M頭，收盤跌破MA60做空")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="30d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-m-head/index.html")
    p.add_argument("--rth", action="store_true", help="只做 09:30–16:00 ET")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
