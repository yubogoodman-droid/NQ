#!/usr/bin/env python3
"""NQ 五分 K V 轉：急跌後幾乎不盤整、同速拉回。

對齊 2026-08-27 那筆（ETH 02:10 低 29401.75 → 04:50 高 29662）：
  1. 16~28 根內從左側高點灌到右側低點，深度 ≥ max(120 點, 4 ATR)
  2. 底部不盤（靠近低點的 K ≤ 3 根）—— V 不是 U、不是 W
  3. 頸線 = 起跌高點；右腿第一次回到頸線（回補 ≥ 98%）時，右腿 K 數須為左腿的 0.50~1.60 倍
  4. 收紅站上 MA5 做多。停損用右腿回撤低，目標頸線再延伸 0.7× dump

含 ETH（那筆 V 在凌晨）；RTH 09:30–10:00 開盤噪音不進。

用法:
  python3 examples/nq_v_turn.py
  python3 examples/nq_v_turn.py backtest --period 30d --pages
  python3 examples/nq_v_turn.py backtest --period 5d
  python3 examples/test_nq_v_turn.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-v-turn" / "index.html"
VIEW_BRANCH = "cursor/nq-5m-v-turn-4941"


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
    dump_start: int
    dump_high_idx: int
    dump_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    dump_high: float
    dump_low: float
    drop_pts: float
    drop_atr: float
    recover_frac: float
    recover_bars: int
    recover_speed: float
    time_ratio: float
    vol_mult: float
    ma5: float
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


def true_range(high, low, close) -> np.ndarray:
    prev = np.r_[close[0], close[:-1]]
    a = high - low
    b = np.abs(high - prev)
    c = np.abs(low - prev)
    return np.maximum(np.maximum(a, b), c)


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return sma(true_range(high, low, close), n)


def _right_wing_stop(low: np.ndarray, dump_idx: int, entry_idx: int, stop_buffer: float) -> float:
    """停損放右腿回撤，不用 V 底（回到頸線時離 V 底太遠）。"""
    rec_bars = entry_idx - dump_idx
    skip = max(1, rec_bars // 3)
    start = min(dump_idx + skip, entry_idx)
    return float(np.min(low[start : entry_idx + 1])) - stop_buffer


def quality_from_v(drop_atr: float, recover_speed: float, vol_mult: float) -> Tuple[int, str]:
    score = 0
    if drop_atr >= 2.5:
        score += 1
    if recover_speed >= 1.0:
        score += 1
    if vol_mult >= 1.3:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def _is_rth_open_noise(ts) -> bool:
    """09:30–10:00 ET 開盤區間，V 很容易被假跌破洗掉。"""
    mins = ts.hour * 60 + ts.minute
    return 9 * 60 + 30 <= mins < 10 * 60


def _best_dump_ending_here(
    i: int,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    *,
    min_dump_bars: int,
    max_dump_bars: int,
    min_drop_pts: float,
    min_drop_atr: float,
    min_drop_pct: float,
    min_red_frac: float,
    max_base_bars: int,
    left_high_frac: float,
) -> Optional[Tuple[int, int, int, float, float, float]]:
    """若 i 是某段急跌的右側低點，回傳最佳 dump 視窗。"""
    a = float(atr14[i])
    if not np.isfinite(a) or a <= 0:
        return None
    px = float(close[i])
    thresh = max(min_drop_pts, min_drop_atr * a, min_drop_pct * px)

    best: Optional[Tuple[int, int, int, float, float, float]] = None
    best_drop = -1.0

    for dump_bars in range(min_dump_bars, max_dump_bars + 1):
        win0 = i - dump_bars + 1
        if win0 < 0:
            continue
        window_high = high[win0 : i + 1]
        window_low = low[win0 : i + 1]
        hi_off = int(np.argmax(window_high))
        lo_off = int(np.argmin(window_low))
        if lo_off != dump_bars - 1:
            continue
        if hi_off > int(dump_bars * left_high_frac):
            continue

        dump_high = float(window_high[hi_off])
        dump_low = float(window_low[lo_off])
        drop = dump_high - dump_low
        if drop < thresh:
            continue

        reds = sum(1 for k in range(win0, i + 1) if close[k] < open_[k])
        if reds < min_red_frac * dump_bars:
            continue

        near = max(8.0, 0.08 * drop)
        base = sum(1 for k in range(win0, i + 1) if (dump_low + near) >= low[k])
        if base > max_base_bars:
            continue

        if drop > best_drop:
            best_drop = drop
            best = (win0, win0 + hi_off, i, dump_high, dump_low, drop)

    return best


def detect_signals(
    df,
    min_dump_bars: int = 16,
    max_dump_bars: int = 28,
    recover_frac: float = 0.98,
    min_recover_bars: int = 3,
    min_recover_speed: float = 0.60,
    min_time_ratio: float = 0.50,
    max_time_ratio: float = 1.60,
    atr_len: int = 14,
    min_drop_pts: float = 120.0,
    min_drop_atr: float = 4.0,
    min_drop_pct: float = 0.0025,
    min_red_frac: float = 0.55,
    max_base_bars: int = 3,
    left_high_frac: float = 0.40,
    vol_lookback: int = 20,
    stop_buffer: float = 8.0,
    target_dump_mult: float = 1.7,
    max_risk: float = 120.0,
    min_entry_gap: int = 24,
    skip_rth_open: bool = True,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    V 轉：頸線（起跌高）兩邊要差不多——右腿回到頸線，時間也接近左腿。
    """
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))

    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    atr14 = atr(high, low, close, atr_len)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(60, atr_len, vol_lookback, max_dump_bars) + 2
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        dump = _best_dump_ending_here(
            i,
            high,
            low,
            open_,
            close,
            atr14,
            min_dump_bars=min_dump_bars,
            max_dump_bars=max_dump_bars,
            min_drop_pts=min_drop_pts,
            min_drop_atr=min_drop_atr,
            min_drop_pct=min_drop_pct,
            min_red_frac=min_red_frac,
            max_base_bars=max_base_bars,
            left_high_frac=left_high_frac,
        )
        if dump is None:
            i += 1
            continue

        bump("dump")
        dump_start, dump_high_idx, dump_idx, dump_high, dump_low, drop = dump
        dump_bars = dump_idx - dump_start + 1
        dump_speed = drop / dump_bars if dump_bars else 0.0
        a = float(atr14[i]) if np.isfinite(atr14[i]) else drop / 3.0

        vol_avg = float(np.mean(volume[max(0, dump_start - vol_lookback) : dump_start])) or 1.0
        dump_vol = float(np.mean(volume[dump_start : dump_idx + 1]))
        vmult = dump_vol / vol_avg if vol_avg > 0 else 0.0

        recover_window = max(dump_bars, int(dump_bars * max_time_ratio) + 1)
        entered = False
        invalidated = False
        first_touch: int | None = None

        for j in range(dump_idx + 1, min(dump_idx + recover_window + 1, n)):
            if low[j] < dump_low - 1e-9:
                bump("skip_new_low")
                invalidated = True
                break
            rec = (float(close[j]) - dump_low) / drop if drop else 0.0
            rec_bars = j - dump_idx
            if rec_bars / dump_bars > max_time_ratio:
                bump("skip_asym")
                invalidated = True
                break
            if rec < recover_frac:
                continue
            if first_touch is None:
                first_touch = j
            time_ratio = (first_touch - dump_idx) / dump_bars
            bump("recover")
            if time_ratio < min_time_ratio:
                bump("skip_early")
                invalidated = True
                break

            rec_pts = float(close[j]) - dump_low
            rec_speed = (rec_pts / rec_bars / dump_speed) if rec_bars and dump_speed else 0.0
            if rec_bars < min_recover_bars:
                continue
            if rec_speed < min_recover_speed:
                bump("skip_slow")
                continue
            if close[j] < open_[j]:
                bump("skip_red_entry")
                continue
            if np.isnan(ma5[j]) or close[j] <= ma5[j]:
                bump("skip_ma5")
                continue
            if skip_rth_open and _is_rth_open_noise(df.index[j]):
                bump("skip_open")
                continue
            if j - last_entry < min_entry_gap:
                bump("skip_entry_gap")
                invalidated = True
                break

            entry = float(close[j])
            stop = _right_wing_stop(low, dump_idx, j, stop_buffer)
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                continue
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            measured = dump_low + drop * target_dump_mult
            target = measured if measured > entry else entry + risk
            if target <= entry:
                bump("skip_bad_target")
                continue

            drop_atr = drop / a if a else 0.0
            q_score, q_grade = quality_from_v(drop_atr, rec_speed, vmult)
            bump("taken")
            signals.append(
                Signal(
                    dump_start=dump_start,
                    dump_high_idx=dump_high_idx,
                    dump_idx=dump_idx,
                    entry_idx=j,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=float(target),
                    dump_high=dump_high,
                    dump_low=dump_low,
                    drop_pts=drop,
                    drop_atr=drop_atr,
                    recover_frac=rec,
                    recover_bars=rec_bars,
                    recover_speed=rec_speed,
                    time_ratio=time_ratio,
                    vol_mult=vmult,
                    ma5=float(ma5[j]),
                    ma20=float(ma20[j]) if not np.isnan(ma20[j]) else 0.0,
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = j
            entered = True
            i = j + 1
            break

        if entered:
            continue
        if invalidated:
            i = dump_idx + 1
            continue
        i += 1

    return signals


def simulate(
    df,
    signals: List[Signal],
    max_hold: int = 36,
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
    """Turn a matplotlib figure into inline SVG that survives htmlpreview / GitHub raw CSP."""
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
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    start = max(0, sig.dump_start - 10)
    end = min(len(df) - 1, max(trade.exit_idx + 10, sig.dump_idx + sig.recover_bars + 8))
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
        ax.plot(list(xs), ma, color=col, lw=1.35 if nper <= 20 else 1.05, label=f"MA{nper}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.dump_high, color="#79c0ff", ls="--", lw=1.05, alpha=0.85)

    hx, vx, ex, xx = (
        sig.dump_high_idx - start,
        sig.dump_idx - start,
        trade.entry_idx - start,
        trade.exit_idx - start,
    )
    v_x, v_y = [], []
    if 0 <= hx < len(window):
        v_x.append(hx)
        v_y.append(sig.dump_high)
        ax.scatter([hx], [sig.dump_high], s=36, color="#79c0ff", zorder=5)
        ax.annotate("起跌/頸線", (hx, sig.dump_high), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= vx < len(window):
        v_x.append(vx)
        v_y.append(sig.dump_low)
        ax.scatter([vx], [sig.dump_low], s=42, color="#facc15", zorder=6)
        ax.annotate("V底", (vx, sig.dump_low), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= ex < len(window):
        v_x.append(ex)
        v_y.append(trade.entry_price)
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
    if len(v_x) >= 2:
        ax.plot(v_x, v_y, color="#facc15", lw=1.4, alpha=0.85, zorder=4)
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
        f"#{trade_no}  V轉 Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
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


def _render_trade_cards(
    df: pd.DataFrame, trades: List[TradeResult], html_path: Path, prefix: str = "t"
) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        vt = df.index[t.signal.dump_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i, prefix=prefix)
        svg = draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        chart = svg or (
            f"<img src='img/{escape(img_name)}' alt='#{i} Q{escape(t.quality)}' "
            "style='width:100%;display:block;border-radius:10px'/>"
        )
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
            f"<span class='tag tag-info'>5m V轉</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"V底 {vt.strftime('%m-%d %H:%M')}  {t.signal.dump_low:.2f}\n"
            f"頸線 {t.signal.dump_high:.2f}  兩邊 {t.signal.time_ratio:.2f}× 時間 / "
            f"{t.signal.recover_frac * 100:.0f}% 高度\n"
            f"dump {t.signal.drop_pts:.0f}pt / {t.signal.drop_atr:.1f}ATR  "
            f"右腿 {t.signal.recover_bars} 根\n"
            f"速度 {t.signal.recover_speed:.2f}×  量 {t.signal.vol_mult:.1f}×\n"
            f"MA5 {t.signal.ma5:.1f} / MA20 {t.signal.ma20:.1f}"
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
    img_dir = out.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    cards = _render_trade_cards(df, trades, out)
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：急跌 {funnel.get('dump', 0)} → "
            f"回補頸線 {funnel.get('recover', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（新低打斷 {funnel.get('skip_new_low', 0)} · 太早 {funnel.get('skip_early', 0)} · "
            f"不對稱 {funnel.get('skip_asym', 0)} · 太慢 {funnel.get('skip_slow', 0)} · "
            f"紅K {funnel.get('skip_red_entry', 0)} · MA5 {funnel.get('skip_ma5', 0)} · "
            f"開盤 {funnel.get('skip_open', 0)} · 風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分 K V轉</title>
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
<h1>{escape(symbol)} 五分 K V轉</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">16~28 根急跌（≥120 點 / 4 ATR）、尖底不盤。頸線=起跌高，右腿第一次回到頸線（≥98%）且時間為左腿的 0.50~1.60 倍。停損右腿回撤，目標頸線再延伸 0.7× dump。含 ETH，09:30–10:00 不進。</p>
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
{cards or "<div class='empty'>無 V 轉訊號</div>"}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    # Charts are inline SVG, so a straight copy survives htmlpreview and GitHub raw CSP.
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
        vt = df.index[t.signal.dump_idx]
        print(
            f"  [{prefix}{i}] Q{t.quality} V底 {vt.strftime('%m-%d %H:%M')} "
            f"進 {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"dump={t.signal.drop_pts:.0f}pt/{t.signal.drop_atr:.1f}ATR "
            f"頸線兩邊 {t.signal.time_ratio:.2f}×/{t.signal.recover_frac * 100:.0f}%"
        )


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

    if stats["count"] == 0:
        verdict = "這段樣本沒抓到 V 轉。多數急跌要嘛底部盤成 U，要嘛回補太慢。"
    elif stats["total_points"] > 50 and stats["win_rate"] >= 45:
        verdict = "有料：頸線兩邊對稱的 V 比接刀清楚。假 V（回到頸線後再破底）仍會一次吐回去。"
    elif abs(stats["total_points"]) <= 50:
        verdict = "抓得到 08-27 那種 V，但樣本交易優勢接近零。真 V 的量度被假 V 破底吃掉。"
    elif stats["total_points"] > 0:
        verdict = "邊緣：總點數正，但勝率不高，還不算穩的優勢。"
    else:
        verdict = "沒料：收復 50% 進場時離 V 底已遠，停損偏寬；假 V 破底會把整段吐回去。"
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
    p = argparse.ArgumentParser(description="NQ 五分 K V轉")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 5m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--interval", default="5m")
    b.add_argument("--period", default="30d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-v-turn/index.html")
    b.set_defaults(func=cmd_backtest)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="30d")
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
