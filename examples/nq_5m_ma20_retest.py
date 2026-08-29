#!/usr/bin/env python3
"""NQ 一分 / 五分 K 破底翻：右肩在 MA20 上做多。

破底（跌破近 2 小時低點）→ 反彈收復 MA20 → 先離開均線 →
右肩回踩粉紅 MA20 才進場，不在收復當根追。

用法:
  python3 examples/nq_5m_ma20_retest.py --interval 1m --period 30d --pages
  python3 examples/nq_5m_ma20_retest.py --period 30d --pages
  python3 examples/nq_5m_ma20_retest.py --period 5d --html output/nq_5m_ma20.html
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma20_retest import (  # noqa: E402
    INTERVAL_SIMULATE,
    TradeResult,
    detect_kwargs,
    detect_signals,
    drop_open_end_trades,
    simulate_kwargs,
    summarize_trades,
    simulate,
)

ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]
PAGES_HTML = {
    "5m": REPO / "docs" / "nq-5m-ma20-retest" / "index.html",
    "1m": REPO / "docs" / "nq-1m-ma20-retest" / "index.html",
}

# 盡量對齊手機均線顏色；MA20 用粉紅（藍圈那條）
MA_COLORS = {
    5: "#ffa726",
    10: "#26c6da",
    20: "#ec407a",
    30: "#1565c0",
    60: "#66bb6a",
    120: "#c62828",
    200: "#26a69a",
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


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


def load_yfinance(symbol: str = "NQ=F", interval: str = "5m", period: str = "30d") -> pd.DataFrame:
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
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep="last")].dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo 1m period= 最多約 7–8 天；超過改用 7 日切片（約可回看 30 天）。"""
    days = parse_period_days(period)
    if interval == "1m" and days is not None and days > 8:
        end = datetime.now(timezone.utc)
        # Yahoo 1m 只留約 30 曆日；多要一天常會整段空掉。
        lookback = min(int(days), 29)
        start = end - timedelta(days=lookback)
        df = load_yahoo_intraday(symbol, interval, start, end, chunk_days=6)
        if not df.empty:
            return df
        print(f"[data] chunked {period} empty, fallback period download", file=sys.stderr)
    df = load_yfinance(symbol, interval, period)
    if not df.empty:
        return df
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


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df = df.rename(columns=str.title)
    return to_et(df)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


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


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, trade.signal.break_idx - 18)
    end = min(len(df) - 1, trade.exit_idx + 10)
    return start, end


def resample_ohlc(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    cols: Dict[str, str] = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        cols["Volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(cols)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _asof_bar(df: pd.DataFrame, ts) -> int:
    pos = int(df.index.searchsorted(ts, side="right")) - 1
    return max(0, min(pos, len(df) - 1))


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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
    return plt


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Path:
    plt = _setup_mpl()
    from matplotlib.patches import Ellipse, Rectangle

    sig = trade.signal
    start, end = _trade_window(df, trade)
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
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.7)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.15 if n == 20 else (1.25 if n <= 10 else 1.0)
        ax.plot(list(xs), ma, color=col, lw=lw, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    bx = sig.trough_idx - start
    rx = sig.reclaim_idx - start
    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.break_low], s=38, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, sig.break_low),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= rx < len(window):
        ax.scatter([rx], [df["Close"].iloc[sig.reclaim_idx]], s=36, color="#67e8f9", marker="o", zorder=5)
        ax.annotate(
            "收復",
            (rx, df["Close"].iloc[sig.reclaim_idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#67e8f9",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=46, color="#00e676", marker="^", zorder=6)
        yspan = float(window["High"].max() - window["Low"].min()) or 1.0
        ell_h = max(yspan * 0.08, 22.0)
        ax.add_patch(
            Ellipse(
                (ex, trade.entry_price),
                width=3.2,
                height=ell_h,
                fill=False,
                edgecolor="#38bdf8",
                lw=1.7,
                alpha=0.95,
                zorder=7,
            )
        )
        ax.annotate(
            "右肩進場",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(10, 8),
            ha="left",
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
        f"#{trade_no}  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=7)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_5m_at_entry_png(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Path:
    """一分進場時刻的五分 K 對照：粉紅線是 5m MA20（手機圖那條）。"""
    plt = _setup_mpl()
    from matplotlib.patches import Ellipse, Rectangle

    sig = trade.signal
    entry_ts = df_1m.index[trade.entry_idx]
    trough_ts = df_1m.index[sig.trough_idx]
    reclaim_ts = df_1m.index[sig.reclaim_idx]
    exit_ts = df_1m.index[trade.exit_idx]
    i_entry = _asof_bar(df_5m, entry_ts)
    i_trough = _asof_bar(df_5m, trough_ts)
    i_reclaim = _asof_bar(df_5m, reclaim_ts)
    i_exit = _asof_bar(df_5m, exit_ts)
    start = max(0, i_trough - 16)
    end = min(len(df_5m) - 1, max(i_entry + 16, i_exit + 4))
    window = df_5m.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df_5m["Close"].astype(float)
    ma20_5 = float(close_full.rolling(20, min_periods=20).mean().iloc[i_entry])

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
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.85)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.4 if n == 20 else (1.25 if n <= 10 else 1.0)
        ax.plot(list(xs), ma, color=col, lw=lw, label=f"5m MA{n}")

    bx, rx, ex = i_trough - start, i_reclaim - start, i_entry - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [float(window["Low"].iloc[bx])], s=38, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, float(window["Low"].iloc[bx])),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= rx < len(window):
        ax.scatter([rx], [float(window["Close"].iloc[rx])], s=36, color="#67e8f9", zorder=5)
        ax.annotate(
            "1m收復",
            (rx, float(window["Close"].iloc[rx])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#67e8f9",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=1.0)
        ax.scatter([ex], [trade.entry_price], s=52, color="#00e676", marker="^", zorder=6)
        yspan = float(window["High"].max() - window["Low"].min()) or 1.0
        ax.add_patch(
            Ellipse(
                (ex, float(window["Low"].iloc[ex])),
                width=2.4,
                height=max(yspan * 0.10, 28.0),
                fill=False,
                edgecolor="#38bdf8",
                lw=1.8,
                alpha=0.95,
                zorder=7,
            )
        )
        ax.annotate(
            "1m進場",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(10, 8),
            ha="left",
            color="#86efac",
            fontsize=8,
        )
        if not np.isnan(ma20_5):
            ax.axhline(ma20_5, color="#ec407a", ls=":", lw=1.0, alpha=0.55)

    sign = "+" if trade.pnl_points >= 0 else ""
    ma_txt = f"5mMA20 {ma20_5:.1f}" if not np.isnan(ma20_5) else "5mMA20 n/a"
    ax.set_title(
        f"#{trade_no}  進場當下 5分K  {entry_ts.strftime('%m-%d %H:%M')}  "
        f"{ma_txt}  1m進場 {trade.entry_price:.1f}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=7)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int, prefix: str = "t") -> str:
    et = df.index[trade.entry_idx]
    return f"{prefix}{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(
    df: pd.DataFrame,
    trades: List[TradeResult],
    html_path: Path,
    *,
    prefix: str = "t",
    interval: str = "5m",
    df_5m: Optional[pd.DataFrame] = None,
) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        br = df.index[t.signal.trough_idx]
        rc = df.index[t.signal.reclaim_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {
            "target": "tag-tp",
            "stop": "tag-sl",
            "be_stop": "tag-sl",
        }.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i, prefix=prefix)
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        chart = (
            f"<p class='chart-label'>{escape(interval)} K</p>"
            f"<img src='img/{escape(img_name)}' alt='#{i} Q{escape(t.quality)} {escape(interval)}' "
            "style='width:100%;display:block;border-radius:10px'/>"
        )
        slope60 = t.signal.ma60_5m_slope

        def _htf_slope_txt(s: float) -> str:
            if s < 0:
                return "下彎"
            if s > 0:
                return "上彎"
            return "走平"

        ma60_5_line = (
            f"\n5m MA60 {t.signal.ma60_5m:.1f}  {_htf_slope_txt(slope60)} {slope60:+.1f}  "
            f"（進場 − 5mMA60 = {t.entry_price - t.signal.ma60_5m:+.1f}）"
        )
        ma20_30_line = (
            f"\n5m MA20 {t.signal.ma20_5m:.1f}  {_htf_slope_txt(t.signal.ma20_5m_slope)} "
            f"{t.signal.ma20_5m_slope:+.1f}  （進場 − 5mMA20 = {t.entry_price - t.signal.ma20_5m:+.1f}）"
            f"\n5m MA30 {t.signal.ma30_5m:.1f}  {_htf_slope_txt(t.signal.ma30_5m_slope)} "
            f"{t.signal.ma30_5m_slope:+.1f}  （進場 − 5mMA30 = {t.entry_price - t.signal.ma30_5m:+.1f}）"
        )
        if df_5m is not None and len(df_5m):
            img5 = _trade_img_name(df, t, i, prefix=f"{prefix}5m")
            draw_5m_at_entry_png(df, df_5m, t, html_path.parent / "img" / img5, i)
            chart += (
                "<p class='chart-label'>進場當下 5分K（粉紅 = 5m MA20）</p>"
                f"<img src='img/{escape(img5)}' alt='#{i} 5m對照' "
                "style='width:100%;display:block;border-radius:10px;margin-top:8px'/>"
            )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>{escape(interval)}</span>"
            f"<span class='tag tag-info'>右肩MA20</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            + ("<span class='tag tag-info'>5m對照</span>" if df_5m is not None else "")
            + "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  （右肩踩 {escape(interval)} MA20 {t.signal.ma20:.2f}）\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts，{'右肩低點下方' if interval == '1m' else '破底下方'}）\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破底 {br.strftime('%m-%d %H:%M')} low {t.signal.break_low:.2f} / 2h低 {t.signal.support:.2f}\n"
            f"收復 {rc.strftime('%H:%M')} → 右肩 {et.strftime('%H:%M')}\n"
            f"MA5 {t.signal.ma5:.1f} / MA10 {t.signal.ma10:.1f} / MA20 {t.signal.ma20:.1f} / MA60 {t.signal.ma60:.1f}"
            f"{ma20_30_line}"
            f"{ma60_5_line}"
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
    extra_trades: Optional[List[TradeResult]] = None,
    extra_title: str = "",
    interval: str = "5m",
    df_5m: Optional[pd.DataFrame] = None,
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = []
    for q, info in stats.get("by_quality", {}).items():
        q_bits.append(f"Q{q} {info['n']}筆 {info['pnl']:+.1f}")
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    ma_exit_after = INTERVAL_SIMULATE[interval]["ma_exit_after"]
    ma20_note = "約 20 分鐘" if interval == "1m" else "約 100 分鐘"
    cards = _render_trade_cards(df, trades, out, prefix="t", interval=interval, df_5m=df_5m)
    extra_html = ""
    if extra_trades:
        extra_stats = summarize_trades(extra_trades)
        extra_cls = "pnl-win" if extra_stats["total_points"] >= 0 else "pnl-loss"
        extra_html = (
            f"<section class='summary'><h1>{escape(extra_title or '全時段（含夜盤）')}</h1>"
            f"<p class='muted'>同一套規則，不限 09:30–15:45。夜盤較雜，放著對照研究。</p>"
            f"<div class='cards'><div class='card'>筆數<b>{extra_stats['count']}</b></div>"
            f"<div class='card'>勝率<b>{extra_stats['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總點數<b class='{extra_cls}'>{extra_stats['total_points']:+.1f}</b></div>"
            f"<div class='card'>勝/負<b>{extra_stats['wins']}/{extra_stats['count']-extra_stats['wins']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_points for t in extra_trades])}</div></section>"
            + (
                _render_trade_cards(df, extra_trades, out, prefix="a", interval=interval, df_5m=df_5m)
                or "<div class='empty'>無交易</div>"
            )
        )
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破底 {funnel.get('break', 0)} → "
            f"深度夠 {funnel.get('deep_break', 0)} → "
            f"收復MA20 {funnel.get('reclaim', 0)} → "
            f"右肩回踩 {funnel.get('retest', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（沒離開 {funnel.get('no_retest', 0)} · 沒收復 {funnel.get('no_reclaim', 0)} · "
            f"沒守住 {funnel.get('fail_hold', 0)} · 風險 {funnel.get('skip_max_risk', 0)} · "
            f"貼下彎5mMA60 {funnel.get('skip_ma60', 0)} · "
            f"貼下彎5mMA20/30蓋頭 {funnel.get('skip_ma20_30', 0)} · "
            f"遠低下彎5mMA20 {funnel.get('skip_below_5m', 0)} · "
            f"下跌5m位置 {funnel.get('skip_5m', 0)} · "
            f"間隔 {funnel.get('skip_gap', 0)} · "
            f"追刀 {funnel.get('skip_dump', 0)} · "
            f"午後 {funnel.get('skip_late', 0)} · "
            f"時段 {funnel.get('skip_session', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    pullback_note = (
        "右肩要先從反彈高點拉回至少 25 點，離開均線 3 根後即可回踩。"
        "進場收盤須貼著 1m MA20（高於不超過 20 點）；影線掃到但收盤彈走 30 點那種再等下一腳。"
        "大陰線砸上 MA20 不進，等下一根小 K 確認；實體超過 30 點且 5m MA20 下彎則這波作廢。"
        "13:00 後不進（午餐後／尾盤假右肩）。5m MA20 上彎時目標 2R，否則 1.5R。"
        "五分 MA20 下跌時：追在它上方超過 35 點、或壓在斜率比 −15 更負的蓋子下，不進。"
        "只做日盤破底。"
        if interval == "1m"
        else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} {escape(interval)} 破底翻 · 右肩 MA20</title>
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
.chart-label{{margin:8px 4px 6px;color:#8b949e;font-size:12px;font-weight:600}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} {escape(interval)} 破底翻 · 右肩在 MA20 上</h1>
<p class="muted">只做日盤 09:30–15:45 ET。破底翻 → 收復粉紅 MA20（{escape(ma20_note)}）→ 離開後右肩踩回 MA20 進場。停損在{'右肩低點' if interval == '1m' else '破底'}下方，目標 1.5R；持有滿 {ma_exit_after} 根若收破 MA20 出場。進場若貼著下彎的 5m MA60（40 點內），或夾在下彎空頭排列的 5m MA20/MA30 蓋頭底下（45 點內），或遠低於下彎的 5m MA20（超過 45 點），則略過。五分 MA20 下跌時，追在它上方超過 35 點、或壓在斜率比 −15 更負的蓋子下，也不進。{pullback_note}{" 每筆附進場當下五分 K 對照。" if interval == "1m" else ""}</p>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">{escape(q_line)}</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{cards or "<div class='empty'>無交易</div>"}
{extra_html}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


RAW_BRANCH = "cursor/1m-right-shoulder-ma20-d3c2"


def write_view_html(index_path: Path) -> Path:
    """htmlpreview 用：把相對圖片改成 GitHub raw。"""
    text = index_path.read_text(encoding="utf-8")
    rel = index_path.parent.relative_to(REPO).as_posix()
    raw = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{RAW_BRANCH}/{rel}/"
    view = text.replace("src='img/", f"src='{raw}img/")
    out = index_path.parent / "view.html"
    out.write_text(view, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    interval = args.interval
    if args.csv:
        df = load_csv(args.csv)
    else:
        df = to_et(load_bars(args.symbol, interval, args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    dkw = detect_kwargs(interval, session=args.session)
    skw = simulate_kwargs(interval)
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, **dkw)
    trades = simulate(df, sigs, **skw)
    trades, open_trades = drop_open_end_trades(df, trades, skw["max_hold"])
    stats = summarize_trades(trades)
    print(f"{args.symbol} {interval} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if open_trades:
        print(f"open={len(open_trades)} (sample ended, excluded)")
    if funnel:
        print(
            "funnel "
            f"break={funnel.get('break', 0)} deep={funnel.get('deep_break', 0)} "
            f"reclaim={funnel.get('reclaim', 0)} retest={funnel.get('retest', 0)} "
            f"taken={funnel.get('taken', 0)} "
            f"no_reclaim={funnel.get('no_reclaim', 0)} no_retest={funnel.get('no_retest', 0)} "
            f"fail={funnel.get('fail_hold', 0)} risk={funnel.get('skip_max_risk', 0)} "
            f"ma60={funnel.get('skip_ma60', 0)} "
            f"ma20_30={funnel.get('skip_ma20_30', 0)} "
            f"below5m={funnel.get('skip_below_5m', 0)} "
            f"loc5m={funnel.get('skip_5m', 0)} "
            f"gap={funnel.get('skip_gap', 0)} "
            f"dump={funnel.get('skip_dump', 0)} "
            f"late={funnel.get('skip_late', 0)} "
            f"session={funnel.get('skip_session', 0)}"
        )
    for q, info in stats.get("by_quality", {}).items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"entry {t.entry_price:.2f} stop {t.stop_price:.2f} "
            f"破底 {df.index[t.signal.trough_idx].strftime('%H:%M')}@{t.signal.break_low:.2f} "
            f"收復 {df.index[t.signal.reclaim_idx].strftime('%H:%M')}"
        )
    for i, t in enumerate(open_trades, 1):
        print(
            f"[open {i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"entry {t.entry_price:.2f} 仍持倉（樣本結束）"
        )

    html_path = args.html
    if args.pages:
        html_path = html_path or str(PAGES_HTML[interval])
        img_dir = Path(html_path).parent / "img"
        if img_dir.is_dir():
            for stale in img_dir.glob("*.png"):
                stale.unlink()
    if html_path:
        out = write_html_report(
            html_path,
            df,
            trades,
            args.symbol,
            args.period,
            funnel=funnel,
            interval=interval,
            df_5m=(resample_ohlc(df) if interval == "1m" else None),
        )
        print(f"html={out}")
        if args.pages:
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 破底翻 · 右肩在 MA20 上（1m / 5m）")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--interval", default="1m", choices=("5m", "1m"))
    p.add_argument("--period", default="30d")
    p.add_argument("--csv", default="", help="自備 K 線 CSV（datetime,open,high,low,close）")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-{interval}-ma20-retest/index.html")
    p.add_argument(
        "--session",
        default="rth",
        choices=("rth", "day", "all"),
        help="rth=09:30-15:45 ET（預設）",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
