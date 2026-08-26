#!/usr/bin/env python3
"""NQ 五分 K 打底後 5/10/20 多排進場。

對齊券商 App NQmain 那種圖：急跌、低點短打底，然後 5/10/20 **散開**成多排才做多。
黏在一起翻一下、或打底後橫很久才排好的，都不算。

用法:
  python3 examples/nq_5m_base_stack.py
  python3 examples/nq_5m_base_stack.py backtest --period 30d --pages --loose
  python3 examples/nq_5m_base_stack.py backtest --period 60d --html output/nq_5m_base_stack.html
  python3 examples/nq_5m_base_stack.py alert --test
  python3 examples/nq_5m_base_stack.py alert --dry-run --once
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from html import escape
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
STATE_PATH = ROOT / "tg_base_stack_state.json"
CONFIG_ENV = REPO_ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = ROOT / "tg_config.env"
PAGES_HTML = REPO_ROOT / "docs" / "nq-5m-base-stack" / "index.html"
VIEW_BRANCH = "cursor/nq-5m-base-stack-95d5"

# 關掉散開 / 急跌集中 / 紅 K 墊高，只留「有跌、打底、翻成 5/10/20」。
LOOSE_DETECT = dict(
    min_ribbon=0.0,
    min_ma5_ma10=0.0,
    min_ma10_ma20=0.0,
    min_ma5_slope5=-999.0,
    min_dump_conc_frac=0.0,
    max_wait_bars=36,
    min_up_bars=0,
)

STRICT_BLURB = (
    "急跌打底後，第一次 MA5>MA10>MA20 要散開上攻才進（MA5 明顯高於 MA10，近幾根多數收紅）。"
    "黏帶點一下、或低點橫很久才排好的不算。停損打底低下方，目標 2R。"
)
LOOSE_BLURB = (
    "放寬版：跌夠 + 短打底後，只要翻成 MA5>MA10>MA20 就算（不要求散開、可等多一點）。"
    "很多不是截圖那種急跌 U。嚴格規則仍只留散開上攻。"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_yfinance(symbol: str = "NQ=F", interval: str = "5m", period: str = "8d") -> pd.DataFrame:
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
    dump_idx: int
    base_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    dump_high: float
    base_low: float
    drop_pts: float
    recover: float
    ribbon: float
    ma5: float
    ma10: float
    ma20: float
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
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def _is_swing_low(lows: np.ndarray, idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = float(lows[idx])
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == float(np.min(window)) and int(np.sum(np.isclose(window, pivot))) == 1


def quality_from_setup(drop_pts: float, recover: float, ribbon: float, ma5_slope: float = 0.0) -> Tuple[int, str]:
    """對齊 08-25：急跌 U 底、22:20 均線散開上攻。"""
    score = 0
    if drop_pts >= 120.0:
        score += 1
    if 0.25 <= recover <= 0.60:
        score += 1
    if ribbon >= 14.0 and ma5_slope >= 18.0:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def detect_signals(
    df,
    swing_lookback: int = 3,
    drop_lookback: int = 24,
    min_drop_pts: float = 80.0,
    min_base_bars: int = 6,
    max_wait_bars: int = 16,
    stop_buffer: float = 8.0,
    target_r: float = 2.0,
    max_risk: float = 80.0,
    min_ribbon: float = 14.0,
    max_ribbon: float = 40.0,
    min_ma5_ma10: float = 8.0,
    min_ma10_ma20: float = 3.0,
    min_ma5_slope5: float = 15.0,
    dump_conc_bars: int = 16,
    min_dump_conc_frac: float = 0.80,
    up_lookback: int = 6,
    min_up_bars: int = 4,
    max_recover: float = 0.85,
    max_base_range_frac: float = 0.50,
    min_entry_gap: int = 12,
    require_close_gt_ma5: bool = True,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    打底後多排（對齊 08-25 夜盤那張 U）：
      1. 急跌：回看高點跌夠，而且最後 16 根就要跌掉八成（不是慢慢磨）
      2. 低點短打底：至少 6 根不破底，最多再等 16 根
      3. 第一次翻成 MA5>MA10>MA20 時必須散開上攻（不是三條黏在一起點一下）
      4. 進場前幾根多數收紅，像在墊高，不是區間裡翻排
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(60, drop_lookback, swing_lookback) + 1
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    i = warmup
    while i < n - 1:
        base_idx = i - swing_lookback
        if base_idx < swing_lookback:
            i += 1
            continue
        if not _is_swing_low(low, base_idx, swing_lookback):
            i += 1
            continue
        bump("swing_low")

        win0 = max(0, base_idx - drop_lookback)
        dump_rel = int(np.argmax(high[win0 : base_idx + 1]))
        dump_idx = win0 + dump_rel
        dump_high = float(high[dump_idx])
        base_low = float(low[base_idx])
        drop_pts = dump_high - base_low
        if drop_pts < min_drop_pts:
            bump("skip_shallow")
            i += 1
            continue
        bump("deep_drop")

        conc_from = max(0, base_idx - dump_conc_bars)
        conc = float(np.max(high[conc_from : base_idx + 1]) - base_low)
        if drop_pts > 0 and conc / drop_pts < min_dump_conc_frac:
            bump("skip_slow_dump")
            i += 1
            continue

        dump_len = base_idx - dump_idx + 1
        reds = int(np.sum(close[dump_idx : base_idx + 1] < open_[dump_idx : base_idx + 1]))
        if dump_len > 0 and reds / dump_len < 0.55:
            bump("skip_slow_dump")
            i += 1
            continue
        if dump_idx <= base_idx:
            dump_floor = float(np.min(low[dump_idx : base_idx + 1]))
            if base_low > dump_floor + 8.0:
                bump("skip_not_floor")
                i += 1
                continue

        if not np.isnan(ma20[base_idx]) and base_idx >= 5 and not np.isnan(ma20[base_idx - 5]):
            if float(ma20[base_idx] - ma20[base_idx - 5]) >= 0:
                bump("skip_ma20_up")
                i += 1
                continue
        if not np.isnan(ma60[base_idx]) and float(close[base_idx]) >= float(ma60[base_idx]):
            bump("skip_above_ma60")
            i += 1
            continue

        if not np.isnan(ma5[base_idx]) and ma5[base_idx] > ma10[base_idx] > ma20[base_idx]:
            bump("skip_already_stack")
            i += 1
            continue

        base_end = min(base_idx + min_base_bars, n - 1)
        if base_end > base_idx + 1:
            base_span = float(np.max(high[base_idx + 1 : base_end + 1]) - np.min(low[base_idx + 1 : base_end + 1]))
            if drop_pts > 0 and base_span / drop_pts > max_base_range_frac:
                bump("skip_no_base")
                i += 1
                continue
        bump("base_ok")

        entered = False
        for j in range(base_idx + min_base_bars, min(base_idx + max_wait_bars, n)):
            if float(low[j]) < base_low - 1e-9:
                bump("skip_new_low")
                break
            if np.isnan(ma5[j]) or np.isnan(ma10[j]) or np.isnan(ma20[j]):
                continue
            stacked = ma5[j] > ma10[j] > ma20[j]
            if not stacked:
                continue
            bump("stack_flip")
            ma5_s5 = float(ma5[j] - ma5[j - 5]) if j >= 5 and not np.isnan(ma5[j - 5]) else 0.0
            gap_5_10 = float(ma5[j] - ma10[j])
            gap_10_20 = float(ma10[j] - ma20[j])
            ribbon = float(ma5[j] - ma20[j])
            up_from = max(0, j - up_lookback + 1)
            up_bars = int(np.sum(close[up_from : j + 1] >= open_[up_from : j + 1]))
            recover = (float(close[j]) - base_low) / drop_pts if drop_pts else 1.0
            if require_close_gt_ma5 and close[j] <= ma5[j]:
                bump("skip_below_ma5")
                continue
            if close[j] <= ma20[j]:
                bump("skip_below_ma20")
                continue
            fan_ok = (
                ribbon >= min_ribbon
                and ribbon <= max_ribbon
                and gap_5_10 >= min_ma5_ma10
                and gap_10_20 >= min_ma10_ma20
                and ma5_s5 >= min_ma5_slope5
                and up_bars >= min_up_bars
            )
            if not fan_ok:
                bump("skip_knot")
                continue
            if recover > max_recover:
                bump("skip_recover")
                continue
            if j - last_entry < min_entry_gap:
                bump("skip_gap")
                break

            entry = float(close[j])
            stop = base_low - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                break
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                break

            q_score, q_grade = quality_from_setup(drop_pts, recover, ribbon, ma5_s5)
            target = entry + risk * target_r
            bump("taken")
            signals.append(
                Signal(
                    dump_idx=dump_idx,
                    base_idx=base_idx,
                    entry_idx=j,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    dump_high=dump_high,
                    base_low=base_low,
                    drop_pts=drop_pts,
                    recover=recover,
                    ribbon=ribbon,
                    ma5=float(ma5[j]),
                    ma10=float(ma10[j]),
                    ma20=float(ma20[j]),
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = j
            entered = True
            i = j + 1
            break

        if not entered:
            i += 1

    return signals


def simulate(
    df,
    signals: List[Signal],
    max_hold: int = 48,
    be_after_r: float = 0.60,
    trail_after_r: float = 1.20,
    trail_lock_r: float = 0.40,
    use_ma20_time_exit: bool = True,
    ma20_exit_after: int = 12,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma20 = sma(close, 20)
    results: List[TradeResult] = []
    busy_until = -1

    for sig in signals:
        j = sig.entry_idx
        if j <= busy_until:
            continue
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = entry - stop
        if risk <= 0:
            continue
        cur_stop = stop
        mfe = 0.0
        limit = min(j + max_hold, len(df) - 1)
        exit_idx = limit
        exit_price = float(close[limit])
        reason = "timeout"

        for k in range(j + 1, limit + 1):
            mfe = max(mfe, float(high[k] - entry))
            if be_after_r > 0 and mfe / risk >= be_after_r:
                cur_stop = max(cur_stop, entry)
            if trail_after_r > 0 and mfe / risk >= trail_after_r:
                cur_stop = max(cur_stop, entry + trail_lock_r * risk)

            if (
                use_ma20_time_exit
                and (k - j) >= ma20_exit_after
                and not np.isnan(ma20[k])
                and float(close[k]) < float(ma20[k])
            ):
                exit_idx, exit_price, reason = k, float(close[k]), "ma20"
                break
            if float(low[k]) <= cur_stop:
                if cur_stop <= stop + 1e-9:
                    reason = "stop"
                elif abs(cur_stop - entry) < 1e-9:
                    reason = "be"
                else:
                    reason = "trail"
                exit_idx, exit_price = k, float(cur_stop)
                break
            if float(high[k]) >= target:
                exit_idx, exit_price, reason = k, float(target), "target"
                break

        busy_until = exit_idx
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=j,
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
    30: "#26a69a",
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
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    start = max(0, sig.dump_idx - 10)
    end = min(len(df) - 1, trade.exit_idx + 10)
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
        if ma.notna().sum() == 0:
            continue
        ax.plot(list(xs), ma, color=col, lw=1.35 if nper <= 20 else 1.05, label=f"MA{nper}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    bx, ex, xx = sig.base_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.base_low], s=38, color="#facc15", zorder=5)
        ax.annotate("打底", (bx, sig.base_low), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
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


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int, prefix: str = "t") -> str:
    et = df.index[trade.entry_idx]
    return f"{prefix}{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(
    df: pd.DataFrame, trades: List[TradeResult], html_path: Path, prefix: str = "t"
) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "be": "tag-time", "trail": "tag-tp"}.get(
            t.exit_reason, "tag-time"
        )
        img_name = _trade_img_name(df, t, i, prefix=prefix)
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        chart = (
            f"<img src='img/{escape(img_name)}' alt='#{i} Q{escape(t.quality)}' "
            "style='width:100%;display:block;border-radius:10px'/>"
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
            f"<span class='tag tag-info'>5m</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"<span class='tag tag-info'>跌 {t.signal.drop_pts:.0f}pt</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"打底 {t.signal.base_low:.2f} ← 高點 {t.signal.dump_high:.2f}  (−{t.signal.drop_pts:.1f})\n"
            f"收回 {t.signal.recover * 100:.0f}% · 帶寬 MA5−MA20 {t.signal.ribbon:.1f}\n"
            f"MA5 {t.signal.ma5:.1f} > MA10 {t.signal.ma10:.1f} > MA20 {t.signal.ma20:.1f}"
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
    extra_blurb: str = "",
    verdict: str = "",
    blurb: str = "",
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = []
    for q, info in stats.get("by_quality", {}).items():
        q_bits.append(f"Q{q} {info['n']}筆 {info['pnl']:+.1f}")
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades, out)
    extra_html = ""
    if extra_trades is not None:
        extra_stats = summarize_trades(extra_trades)
        extra_cls = "pnl-win" if extra_stats["total_points"] >= 0 else "pnl-loss"
        extra_html = (
            f"<section class='summary'><h1>{escape(extra_title or 'QA 對照')}</h1>"
            f"<p class='muted'>{escape(extra_blurb or '只留品質 A：急跌 ≥120、收回 25–60%、均線散開且 MA5 上彎。')}</p>"
            f"<div class='cards'><div class='card'>筆數<b>{extra_stats['count']}</b></div>"
            f"<div class='card'>勝率<b>{extra_stats['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總點數<b class='{extra_cls}'>{extra_stats['total_points']:+.1f}</b></div>"
            f"<div class='card'>勝/負<b>{extra_stats['wins']}/{extra_stats['count']-extra_stats['wins']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_points for t in extra_trades])}</div></section>"
        )
        extra_html += _render_trade_cards(df, extra_trades, out, prefix="q") or ""
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：波段低 {funnel.get('swing_low', 0)} → "
            f"跌夠 {funnel.get('deep_drop', 0)} → "
            f"打底 {funnel.get('base_ok', 0)} → "
            f"多排翻轉 {funnel.get('stack_flip', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（慢跌 {funnel.get('skip_slow_dump', 0)} · 沒打底 {funnel.get('skip_no_base', 0)} · "
            f"破底 {funnel.get('skip_new_low', 0)} · 黏帶翻排 {funnel.get('skip_knot', 0)} · "
            f"低點還在均線上 {funnel.get('skip_above_ma60', 0)+funnel.get('skip_ma20_up', 0)} · "
            f"收回過多 {funnel.get('skip_recover', 0)} · "
            f"風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分打底 5/10/20 多排</title>
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
<h1>{escape(symbol)} 五分打底後 5/10/20 多排</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)} · 五分 K</p>
<p class="muted">{escape(blurb or STRICT_BLURB)}</p>
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
{cards or "<div class='empty'>無交易</div>"}
{extra_html}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


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
        return {"alerted_entries": [], "alerted_exits": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _ts_et(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize("UTC").tz_convert(ET)
    return ts.tz_convert(ET)


def entry_key(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    return f"{ts.isoformat()}|{sig.entry_price:.2f}"


def exit_key(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    return f"{et.isoformat()}->{xt.isoformat()}|{tr.exit_reason}|{tr.pnl_points:.2f}"


def fmt_entry(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    base = _ts_et(df.index[sig.base_idx])
    risk = sig.entry_price - sig.stop_price
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>五分打底 多排進場</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"品質: <b>Q{sig.quality}</b> ({sig.quality_score}/3)\n"
        f"進場: <code>{sig.entry_price:.2f}</code>\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> ({r_mult:.1f}R)\n"
        f"打底: <code>{base.strftime('%H:%M')}</code> low={sig.base_low:.2f} 跌 {sig.drop_pts:.0f}pt\n"
        f"MA5 {sig.ma5:.1f} &gt; MA10 {sig.ma10:.1f} &gt; MA20 {sig.ma20:.1f}\n"
        f"現價: <code>{last:.2f}</code>\n"
        f"#打底多排 #NQ #Q{sig.quality}"
    )


def fmt_exit(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_points > 0 else ("⚪" if tr.pnl_points == 0 else "🔴")
    return (
        f"{emoji} <b>五分打底 多排出場</b>\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.2f}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.2f}\n"
        f"原因: <b>{tr.exit_reason}</b>\n"
        f"盈虧: <b>{tr.pnl_points:+.1f} pts</b> · Q{tr.quality}\n"
        f"#打底多排 #出場"
    )


def scan_once(
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    alert_exits: bool,
    seed_alert: bool,
    lookback_hours: float,
    period: str = "5d",
) -> None:
    df = to_et(load_yfinance("NQ=F", "5m", period))
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    state = load_state()
    alerted_e: Set[str] = set(state.get("alerted_entries") or [])
    alerted_x: Set[str] = set(state.get("alerted_exits") or [])
    now = datetime.now(ET)
    cutoff = now.timestamp() - lookback_hours * 3600
    first_run = not STATE_PATH.exists() or (not alerted_e and not state.get("initialized"))

    new_entries = []
    for sig in sigs:
        k = entry_key(df, sig)
        ts = _ts_et(df.index[sig.entry_idx])
        if ts.timestamp() < cutoff:
            alerted_e.add(k)
            continue
        if k in alerted_e:
            continue
        new_entries.append((k, sig, ts))

    if first_run and not seed_alert:
        for k, _, _ in new_entries:
            alerted_e.add(k)
        for tr in trades:
            alerted_x.add(exit_key(df, tr))
        state["alerted_entries"] = sorted(alerted_e)[-200:]
        state["alerted_exits"] = sorted(alerted_x)[-200:]
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
            alerted_e.add(k)
            sent += 1
            print(f"[entry] {ts} Q{sig.quality} @ {sig.entry_price:.2f}")

    if alert_exits:
        for tr in trades:
            k = exit_key(df, tr)
            xt = _ts_et(df.index[tr.exit_idx])
            if xt.timestamp() < cutoff:
                alerted_x.add(k)
                continue
            if k in alerted_x:
                continue
            ok = tg_send(token, chat_id, fmt_exit(df, tr), dry_run=dry_run)
            if ok:
                alerted_x.add(k)
                sent += 1
                print(f"[exit] {xt} {tr.exit_reason} {tr.pnl_points:+.1f}")

    state["alerted_entries"] = sorted(alerted_e)[-200:]
    state["alerted_exits"] = sorted(alerted_x)[-200:]
    state["initialized"] = True
    state["last_scan"] = now.isoformat()
    save_state(state)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] scan ok bars={len(df)} "
        f"sigs={len(sigs)} new_sent={sent} last={df['Close'].iloc[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_trades(df, trades, tag: str = "") -> None:
    prefix = f"{tag} " if tag else ""
    for i, t in enumerate(trades, 1):
        print(
            f"  [{prefix}{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  drop={t.signal.drop_pts:.0f}pt "
            f"rec={t.signal.recover:.2f} rib={t.signal.ribbon:.1f}"
        )


def _verdict(stats: dict, *, loose: bool = False) -> str:
    if loose:
        if stats["count"] == 0:
            return "放寬後這段也沒打到「有跌、有翻 5/10/20」。"
        return "這是放寬版：跌夠 + 打底後翻成 5/10/20 就算。多數不是截圖那種急跌散開 U。嚴格規則 30 天通常只有 1 筆。"
    if stats["count"] == 0:
        return "這段沒打到「急跌短打底、5/10/20 散開上攻」的圖。"
    if stats["count"] <= 2 and stats["total_points"] > 0:
        return "只留截圖那種 U：急跌、短打底、均線散開。黏帶翻排和盤很久的已濾掉。"
    if stats["total_points"] > 0:
        return "有抓到 U 底多排，但筆數少，單筆風險仍在打底低下方。"
    return "濾完以後樣本很小。截圖 08-25 22:20 對得上；其餘常是黏帶或假打底。"


def detect_kwargs(args) -> dict:
    if getattr(args, "loose", False):
        return dict(LOOSE_DETECT)
    return {}


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} 5m {args.period}", file=sys.stderr)
    df = to_et(load_yfinance(args.symbol, "5m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    print(f"bars={len(df)} {df.index[0]} → {df.index[-1]}", file=sys.stderr)

    loose = bool(getattr(args, "loose", False))
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, **detect_kwargs(args))
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    tag = "loose" if loose else "strict"
    print(
        f"{tag} trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_points']:+.1f} avg={stats['avg']:+.1f}"
    )
    print("funnel", funnel)
    _print_trades(df, trades)

    extra_trades: List[TradeResult] = []
    extra_title = "只做 QA（急跌 + 收回 25–60% + 均線散開）"
    extra_blurb = "只留品質 A：急跌 ≥120、收回 25–60%、均線散開且 MA5 上彎。"
    if loose:
        extra_funnel: Dict[str, int] = {}
        extra_trades = simulate(df, detect_signals(df, funnel=extra_funnel))
        extra_stats = summarize_trades(extra_trades)
        extra_title = "嚴格（截圖那種 U）"
        extra_blurb = "急跌集中、短打底、均線散開上攻。黏帶翻排和橫很久的都不算。"
        print(
            f"strict trades={extra_stats['count']} WR={extra_stats['win_rate']:.1f}% "
            f"pnl={extra_stats['total_points']:+.1f}"
        )
        print("strict funnel", extra_funnel)
        _print_trades(df, extra_trades, "strict")
    else:
        qa = [t for t in trades if t.quality == "A"]
        if qa and qa != trades:
            extra_trades = qa
            extra_stats = summarize_trades(extra_trades)
            print(
                f"QA    trades={extra_stats['count']} WR={extra_stats['win_rate']:.1f}% "
                f"pnl={extra_stats['total_points']:+.1f}"
            )
            _print_trades(df, extra_trades, "QA")

    verdict = _verdict(stats, loose=loose)
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
            extra_trades=extra_trades or None,
            extra_title=extra_title,
            extra_blurb=extra_blurb,
            verdict=verdict,
            blurb=LOOSE_BLURB if loose else STRICT_BLURB,
        )
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
            f"✅ 五分打底多排 bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    print(
        f"5m base-stack TG | interval={args.interval}s | exits={not args.no_exits} | "
        f"dry_run={args.dry_run} | lookback={args.lookback_hours}h"
    )
    while True:
        try:
            scan_once(
                token or "",
                chat_id or "",
                dry_run=args.dry_run,
                alert_exits=not args.no_exits,
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
    p = argparse.ArgumentParser(description="NQ 五分 K 打底後 5/10/20 多排")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 5m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-5m-base-stack/index.html")
    b.add_argument("--loose", action="store_true", help="關掉散開／急跌集中，只留跌夠後翻 5/10/20")
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("alert", help="Telegram 輪詢")
    a.add_argument("--interval", type=int, default=None)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--no-exits", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=None)
    a.add_argument("--period", default="5d")
    a.set_defaults(func=cmd_alert)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--loose", action="store_true", help="關掉散開／急跌集中，只留跌夠後翻 5/10/20")
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
