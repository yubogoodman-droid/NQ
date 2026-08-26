#!/usr/bin/env python3
"""NQ 一分 K 站上季線（MA60）— 偵測 + 回測 + HTML + Telegram。

進場：1 分圖收盤從季線下方站上（陽線），可加 MACD 多頭與放量。
停損在進場 K 低點／季線下方；目標 2R。

用法:
  python3 examples/nq_ma60_stand.py
  python3 examples/nq_ma60_stand.py backtest --period 8d --html output/nq_ma60_stand.html
  python3 examples/nq_ma60_stand.py backtest --period 30d --pages
  python3 examples/nq_ma60_stand.py alert --dry-run --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    ET,
    env,
    load_bars,
    load_dotenv,
    load_yfinance,
    sma,
    summarize_trades,
    tg_send,
    to_et,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
STATE_PATH = ROOT / "tg_ma60_stand_state.json"
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma60-stand" / "index.html"


def _align_last_completed(series: pd.Series, index) -> np.ndarray:
    """Map a higher-TF series onto 1m bars using only bars that have already closed."""
    s = series.dropna()
    out = np.full(len(index), np.nan, dtype=float)
    if s.empty:
        return out
    idx = s.index
    vals = s.to_numpy(float)
    j = 0
    for i, ts in enumerate(index):
        while j + 1 < len(idx) and idx[j + 1] <= ts:
            j += 1
        if idx[j] <= ts:
            out[i] = vals[j]
    return out


def build_m5_close_ma60(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = df["Close"].astype(float)
    m5 = close.resample("5min", label="right", closed="right").last()
    m5_ma60 = m5.rolling(60, min_periods=60).mean()
    return _align_last_completed(m5, df.index), _align_last_completed(m5_ma60, df.index)


# Tests / 合成資料關掉盤中殺深過濾
SYNTH_DETECT = dict(
    skip_hour_start=None,
    skip_hour_end=None,
    session_start=None,
    session_end=None,
    min_below_bars=8,
    m5_max_dist=None,
    min_drop_pts=0.0,
    use_cluster=False,
)


@dataclass
class Signal:
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    ma60: float
    ma5: float
    ma10: float
    ma20: float
    below_bars: int
    dist_above: float
    cluster_width: float
    slope60: float
    dif: float
    dea: float
    vol_ratio: float
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


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close, dtype=float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2.0
    return dif.to_numpy(float), dea.to_numpy(float), hist.to_numpy(float)


def quality_from_stand(
    slope60: float,
    vol_ratio: float,
    dif: float,
    dea: float,
    dist_above: float,
    cluster_width: float,
    hist: float,
) -> Tuple[int, str]:
    score = 0
    if dif > dea and hist > 0:
        score += 1
    if vol_ratio >= 1.5:
        score += 1
    if slope60 >= -2.0 and dist_above <= 20.0 and cluster_width <= 20.0:
        score += 1
    if score >= 2:
        grade = "A"
    elif score == 1:
        grade = "B"
    else:
        grade = "C"
    return score, grade


def detect_signals(
    df: pd.DataFrame,
    ma60_len: int = 60,
    min_below_bars: int = 40,
    max_stand_pts: float = 40.0,
    min_stand_pts: float = 1.0,
    cluster_pts: float = 40.0,
    stand_touch_pts: float = 22.0,
    use_cluster: bool = False,
    use_stack: bool = True,
    use_macd: bool = True,
    use_macd_cross: bool = False,
    use_vol: bool = True,
    vol_len: int = 20,
    vol_mult: float = 1.2,
    use_slope: bool = True,
    slope_bars: int = 5,
    min_slope: float = -6.0,
    cooldown: int = 20,
    stop_buffer: float = 12.0,
    target_r: float = 2.0,
    max_risk: float = 80.0,
    skip_hour_start: Optional[int] = 9,
    skip_hour_end: Optional[int] = 10,
    session_start: Optional[int] = 10,
    session_end: Optional[int] = 16,
    m5_max_dist: Optional[float] = -60.0,
    min_drop_pts: float = 60.0,
    drop_lookback: int = 60,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """一分收盤站上 SMA60。對齊 #25：先殺深（五分仍遠低於五分季線），再踩上一分季線。"""
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    volume = df["Volume"].to_numpy(float)

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, ma60_len)
    ma100 = sma(close, 100)
    ma120 = sma(close, 120)
    dif, dea, hist = macd(close)
    vol_ma = sma(volume, vol_len)
    m5_close, m5_ma60 = build_m5_close_ma60(df)
    roll_high = pd.Series(high).rolling(drop_lookback, min_periods=max(10, drop_lookback // 3)).max().to_numpy(float)

    n = len(close)
    below = np.zeros(n, dtype=int)
    run = 0
    for i in range(n):
        if np.isnan(ma60[i]):
            run = 0
            below[i] = 0
            continue
        if close[i] <= ma60[i]:
            run += 1
        else:
            run = 0
        below[i] = run

    signals: List[Signal] = []
    last_entry = -(10**9)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    warmup = max(ma60_len, 35, vol_len, slope_bars) + 1
    for i in range(warmup, n):
        if np.isnan(ma60[i]) or np.isnan(ma60[i - 1]):
            continue
        bump("bars")
        if not (close[i - 1] <= ma60[i - 1] and close[i] > ma60[i]):
            continue
        bump("cross")
        if close[i] <= open_[i]:
            bump("skip_bear")
            continue
        dist = float(close[i] - ma60[i])
        if dist < min_stand_pts or dist > max_stand_pts:
            bump("skip_dist")
            continue
        if stand_touch_pts > 0 and float(low[i]) > float(ma60[i]) + stand_touch_pts:
            bump("skip_touch")
            continue
        cluster_width = 0.0
        if not (np.isnan(ma100[i]) or np.isnan(ma120[i])):
            band = (float(ma60[i]), float(ma100[i]), float(ma120[i]))
            cluster_width = max(band) - min(band)
            if use_cluster and cluster_width > cluster_pts:
                bump("skip_cluster")
                continue
            if use_cluster and close[i] <= max(band):
                bump("skip_cluster")
                continue
        elif use_cluster:
            bump("skip_cluster")
            continue
        if use_stack:
            if np.isnan(ma5[i]) or np.isnan(ma10[i]) or not (ma5[i] > ma10[i]):
                bump("skip_stack")
                continue
        below_prev = int(below[i - 1])
        if below_prev < min_below_bars:
            bump("skip_below")
            continue
        slope60 = 0.0
        if i >= slope_bars and not np.isnan(ma60[i - slope_bars]):
            slope60 = float(ma60[i] - ma60[i - slope_bars])
        if use_slope and slope60 < min_slope:
            bump("skip_slope")
            continue
        d_i = float(dif[i]) if not np.isnan(dif[i]) else 0.0
        e_i = float(dea[i]) if not np.isnan(dea[i]) else 0.0
        h_i = float(hist[i]) if not np.isnan(hist[i]) else 0.0
        if use_macd and h_i <= 0:
            bump("skip_macd")
            continue
        if use_macd_cross and d_i <= e_i:
            bump("skip_macd")
            continue
        v_avg = float(vol_ma[i]) if not np.isnan(vol_ma[i]) else 0.0
        v_ratio = float(volume[i] / v_avg) if v_avg > 0 else 0.0
        if use_vol and v_ratio < vol_mult:
            bump("skip_vol")
            continue
        if skip_hour_start is not None and skip_hour_end is not None:
            h = df.index[i].hour
            if skip_hour_start <= h < skip_hour_end:
                bump("skip_open_hour")
                continue
        if session_start is not None and session_end is not None:
            h = df.index[i].hour
            if not (session_start <= h < session_end):
                bump("skip_session")
                continue
        if min_drop_pts > 0:
            drop = float(roll_high[i] - low[i]) if not np.isnan(roll_high[i]) else 0.0
            if drop < min_drop_pts:
                bump("skip_drop")
                continue
        if m5_max_dist is not None:
            if np.isnan(m5_close[i]) or np.isnan(m5_ma60[i]):
                bump("skip_m5")
                continue
            if float(m5_close[i] - m5_ma60[i]) > m5_max_dist:
                bump("skip_m5")
                continue
        if i - last_entry < cooldown:
            bump("skip_cooldown")
            continue

        entry = float(close[i])
        stop = min(float(low[i]), float(ma60[i])) - stop_buffer
        risk = entry - stop
        if risk <= 0 or risk > max_risk:
            bump("skip_risk")
            continue

        q_score, q_grade = quality_from_stand(slope60, v_ratio, d_i, e_i, dist, cluster_width, h_i)
        bump("taken")
        signals.append(
            Signal(
                entry_idx=i,
                entry_price=entry,
                stop_price=stop,
                target_price=entry + risk * target_r,
                ma60=float(ma60[i]),
                ma5=float(ma5[i]) if not np.isnan(ma5[i]) else 0.0,
                ma10=float(ma10[i]) if not np.isnan(ma10[i]) else 0.0,
                ma20=float(ma20[i]) if not np.isnan(ma20[i]) else 0.0,
                below_bars=below_prev,
                dist_above=dist,
                cluster_width=cluster_width,
                slope60=slope60,
                dif=d_i,
                dea=e_i,
                vol_ratio=v_ratio,
                quality=q_grade,
                quality_score=q_score,
            )
        )
        last_entry = i
    return signals


def simulate(
    df: pd.DataFrame,
    signals: List[Signal],
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
                exit_idx, exit_price, exit_reason = k, float(cur_stop), "stop"
                break
            if high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break

        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(exit_price - entry),
                exit_reason=exit_reason,
                quality=sig.quality,
            )
        )
    return results


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


MA_COLORS = {
    5: "#ffa726",
    10: "#81d4fa",
    20: "#ab47bc",
    60: "#26a69a",
    100: "#ffeb3b",
    120: "#ef5350",
}


def resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    """1m → 5m OHLC（label/closed=right，不含未完成的對照用歷史 K）。"""
    out = (
        df.resample("5min", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    return out


def m5_bar_idx(m5: pd.DataFrame, ts) -> int:
    if m5.empty:
        return -1
    i = int(m5.index.searchsorted(ts, side="left"))
    if i >= len(m5):
        return len(m5) - 1
    return i


def m5_context(m5: pd.DataFrame, ts) -> Dict[str, Any]:
    """一分進場當下，落在哪根五分 K、相對五分季線。"""
    i = m5_bar_idx(m5, ts)
    empty = {
        "idx": -1,
        "close": float("nan"),
        "ma60": float("nan"),
        "dist": float("nan"),
        "above": None,
        "label": "5m 資料不足",
    }
    if i < 0:
        return empty
    close = float(m5["Close"].iloc[i])
    ma60_s = m5["Close"].astype(float).rolling(60, min_periods=60).mean()
    ma60 = float(ma60_s.iloc[i]) if not np.isnan(ma60_s.iloc[i]) else float("nan")
    if np.isnan(ma60):
        return {**empty, "idx": i, "close": close, "label": "5m 季線不足"}
    dist = close - ma60
    above = dist > 0
    return {
        "idx": i,
        "close": close,
        "ma60": ma60,
        "dist": dist,
        "above": above,
        "label": f"5m 已站上季線 +{dist:.1f}" if above else f"5m 仍在季線下 {dist:.1f}",
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


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, trade.entry_idx - 40)
    end = min(len(df) - 1, trade.exit_idx + 18)
    return start, end


def _apply_cjk_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break


def _paint_candles(ax, axv, window: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
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


def _style_axes(ax, axv) -> None:
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_cjk_font()

    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    _style_axes(ax, axv)
    _paint_candles(ax, axv, window)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.0 if n == 60 else (1.25 if n <= 20 else 1.0)
        ax.plot(list(xs), ma, color=col, lw=lw, label="季線" if n == 60 else f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(trade.signal.ma60, color="#26a69a", ls="--", lw=0.9, alpha=0.55)

    ex, xx = trade.entry_idx - start, trade.exit_idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate("站上", (ex, trade.entry_price), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#86efac", fontsize=8)
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter([xx], [trade.exit_price], s=40, color="#00c805" if trade.pnl_points > 0 else "#ff5252",
                   marker="x", zorder=6)

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
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
    plt.close(fig)
    return path


def draw_m5_png(
    m5: pd.DataFrame,
    trade: TradeResult,
    entry_ts,
    exit_ts,
    path: Path,
    trade_no: int,
    ctx: Dict[str, Any],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_cjk_font()
    ei = ctx.get("idx", -1)
    if m5.empty or ei < 0:
        return path
    start = max(0, ei - 36)
    xi = int(m5.index.searchsorted(exit_ts, side="left"))
    if xi >= len(m5):
        xi = len(m5) - 1
    end = min(len(m5) - 1, max(ei, xi) + 10)
    window = m5.iloc[start : end + 1]
    xs = range(len(window))
    close_full = m5["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    _style_axes(ax, axv)
    _paint_candles(ax, axv, window)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.0 if n == 60 else (1.25 if n <= 20 else 1.0)
        ax.plot(list(xs), ma, color=col, lw=lw, label="5m季線" if n == 60 else f"MA{n}")

    ex = ei - start
    xx = xi - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [float(window["Close"].iloc[ex])], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate("1m進場", (ex, float(window["Close"].iloc[ex])), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#86efac", fontsize=8)
    if 0 <= xx < len(window) and xx != ex:
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)

    if not np.isnan(ctx.get("ma60", float("nan"))):
        ax.axhline(ctx["ma60"], color="#26a69a", ls="--", lw=0.9, alpha=0.55)

    sign = "+" if ctx.get("dist", 0) >= 0 else ""
    dist = ctx.get("dist", float("nan"))
    dist_s = f"{sign}{dist:.1f}" if not np.isnan(dist) else "n/a"
    ax.set_title(
        f"#{trade_no}  5分K對照  {pd.Timestamp(entry_ts).strftime('%m-%d %H:%M')}  "
        f"{ctx.get('label', '')}  ({dist_s} vs 季線)",
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
    plt.close(fig)
    return path


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int, suffix: str = "") -> str:
    et = df.index[trade.entry_idx]
    extra = f"_{suffix}" if suffix else ""
    return f"t{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}{extra}.png"


def _render_trade_cards(df: pd.DataFrame, trades: List[TradeResult], html_path: Path) -> str:
    m5 = resample_m5(df)
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        ctx = m5_context(m5, et)
        img_name = _trade_img_name(df, t, i)
        img5_name = _trade_img_name(df, t, i, suffix="5m")
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        draw_m5_png(m5, t, et, xt, html_path.parent / "img" / img5_name, i, ctx)
        m5_tag = "tag-tp" if ctx.get("above") else ("tag-sl" if ctx.get("above") is False else "tag-time")
        m5_line = ""
        if not np.isnan(ctx.get("ma60", float("nan"))):
            m5_line = (
                f"5m收 {ctx['close']:.2f} / 5m季線 {ctx['ma60']:.1f}  "
                f"{'+' if ctx['dist'] >= 0 else ''}{ctx['dist']:.1f}  {ctx['label']}\n"
            )
        else:
            m5_line = f"5m {ctx['label']}\n"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>1m 季線</span>"
            f"<span class='tag {m5_tag}'>{escape(ctx['label'])}</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"1m季線 MA60 {t.signal.ma60:.1f}  站上 +{t.signal.dist_above:.1f}  帶寬 {t.signal.cluster_width:.1f}  下方 {t.signal.below_bars} 根\n"
            f"{m5_line}"
            f"MACD DIF {t.signal.dif:.3f} / DEA {t.signal.dea:.3f}  量比 {t.signal.vol_ratio:.2f}x\n"
            f"MA5 {t.signal.ma5:.1f} / MA10 {t.signal.ma10:.1f} / MA20 {t.signal.ma20:.1f}"
            "</pre>"
            "<p class='chart-label'>一分K</p>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i} 1m' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "<p class='chart-label'>五分K對照</p>"
            f"<div class='mini-chart'><img src='img/{escape(img5_name)}' alt='#{i} 5m' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
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
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.1f}" for q, info in stats.get("by_quality", {}).items()]
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades, out)
    m5 = resample_m5(df)
    n_above = n_below = 0
    for t in trades:
        ctx = m5_context(m5, df.index[t.entry_idx])
        if ctx.get("above") is True:
            n_above += 1
        elif ctx.get("above") is False:
            n_below += 1
    m5_line = (
        f"<p class='muted'>一分進場當下，五分收盤已站上季線 <b>{n_above}</b> 筆、"
        f"仍在季線下 <b>{n_below}</b> 筆。</p>"
        if trades
        else ""
    )
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：穿越 {funnel.get('cross', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（陰線 {funnel.get('skip_bear', 0)} · 距離 {funnel.get('skip_dist', 0)} · "
            f"沒踩到線 {funnel.get('skip_touch', 0)} · 季線帶 {funnel.get('skip_cluster', 0)} · "
            f"短均未轉 {funnel.get('skip_stack', 0)} · "
            f"盤外 {funnel.get('skip_session', 0)} · 殺深不夠 {funnel.get('skip_drop', 0)} · "
            f"五分未折夠 {funnel.get('skip_m5', 0)} · "
            f"下方不夠 {funnel.get('skip_below', 0)} · 斜率 {funnel.get('skip_slope', 0)} · "
            f"MACD {funnel.get('skip_macd', 0)} · 量能 {funnel.get('skip_vol', 0)} · "
            f"冷卻 {funnel.get('skip_cooldown', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 一分K 站上季線</title>
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
.chart-label{{margin:10px 2px 6px;font-size:12px;font-weight:600;color:#8b949e}}
.mini-chart{{margin:0 -6px 4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 一分K 站上季線</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">對齊 HTML #25：先殺深（五分收盤仍低於五分季線 ≥60 點、近 60 根至少回撤 60 點、一分季線下至少 40 根），美股 10:00–16:00 ET 陽線站上 1 分季線，MA5&gt;MA10、MACD 柱翻綠、放量。每筆附一分K與五分K對照。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">{escape(q_line)}</p>
{m5_line}
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{cards or "<div class='empty'>無交易</div>"}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    view = out.parent / "view.html"
    if out.name == "index.html":
        view.write_text(
            html.replace(
                f"<p class=\"muted\">{escape(period)}",
                "<p class=\"muted\">每筆一張一分K、一張五分K對照。手機請往下捲。</p>\n\n<p class=\"muted\">" + escape(period),
                1,
            ),
            encoding="utf-8",
        )
    return out


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


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
    risk = sig.entry_price - sig.stop_price
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>站上季線進場</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"品質: <b>Q{sig.quality}</b> ({sig.quality_score}/3)\n"
        f"進場: <code>{sig.entry_price:.2f}</code>\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> ({r_mult:.1f}R)\n"
        f"季線: <code>{sig.ma60:.2f}</code> 站上 +{sig.dist_above:.1f} · 下方 {sig.below_bars} 根\n"
        f"MACD: DIF {sig.dif:.2f} / DEA {sig.dea:.2f} · 量比 {sig.vol_ratio:.2f}x\n"
        f"現價: <code>{last:.2f}</code>\n"
        f"#站上季線 #NQ #1m #Q{sig.quality}"
    )


def fmt_exit(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_points > 0 else ("⚪" if tr.pnl_points == 0 else "🔴")
    return (
        f"{emoji} <b>站上季線出場</b>\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.2f}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.2f}\n"
        f"原因: <b>{tr.exit_reason}</b>\n"
        f"盈虧: <b>{tr.pnl_points:+.1f} pts</b> · Q{tr.quality}\n"
        f"#站上季線 #出場"
    )


def _load_stand_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": [], "alerted_exits": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": []}


def _save_stand_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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
    df = to_et(load_yfinance("NQ=F", "1m", period))
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    state = _load_stand_state()
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
        _save_stand_state(state)
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
    _save_stand_state(state)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] scan ok bars={len(df)} "
        f"sigs={len(sigs)} new_sent={sent} last={df['Close'].iloc[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if funnel:
        print(
            "funnel "
            f"cross={funnel.get('cross', 0)} taken={funnel.get('taken', 0)} "
            f"bear={funnel.get('skip_bear', 0)} dist={funnel.get('skip_dist', 0)} "
            f"touch={funnel.get('skip_touch', 0)} cluster={funnel.get('skip_cluster', 0)} "
            f"stack={funnel.get('skip_stack', 0)} "
            f"sess={funnel.get('skip_session', 0)} drop={funnel.get('skip_drop', 0)} "
            f"m5={funnel.get('skip_m5', 0)} "
            f"below={funnel.get('skip_below', 0)} slope={funnel.get('skip_slope', 0)} "
            f"macd={funnel.get('skip_macd', 0)} vol={funnel.get('skip_vol', 0)} "
            f"cool={funnel.get('skip_cooldown', 0)}"
        )
    for q, info in stats.get("by_quality", {}).items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}"
        )

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(html_path, df, trades, args.symbol, args.period, funnel=funnel)
        print(f"html={out}")
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
            f"✅ 站上季線 bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    print(
        f"MA60 stand TG | interval={args.interval}s | exits={not args.no_exits} | "
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
    p = argparse.ArgumentParser(description="NQ 一分K 站上季線（MA60）")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-stand/index.html")
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
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-stand/index.html")
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
