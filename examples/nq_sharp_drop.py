#!/usr/bin/env python3
"""NQ 五分 K 急跌 V 反 — 對齊力成 6239 那種灌殺後買反彈。

用法:
  python3 examples/nq_sharp_drop.py
  python3 examples/nq_sharp_drop.py backtest --period 60d --pages
  python3 examples/nq_sharp_drop.py backtest --period 60d --html output/nq_sharp_drop.html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-sharp-drop" / "index.html"
VIEW_BRANCH = "cursor/nq-sharp-drop-2f6f"

# 力成 5m：約 272→259（~4.8%）在數根內灌穿均線，量能放大，然後 V 回均線叢。
# NQ 不能用同一百分比（4.8% ≈ 千點）；改成 ATR / 點數等比。


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
    dump_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    dump_high: float
    dump_low: float
    drop_pts: float
    drop_atr: float
    vol_mult: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
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


def quality_from_dump(drop_atr: float, vol_mult: float, pierced_ma60: bool) -> Tuple[int, str]:
    score = 0
    if drop_atr >= 3.0:
        score += 1
    if vol_mult >= 2.0:
        score += 1
    if pierced_ma60:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def _is_reversal_bar(o: float, h: float, l: float, c: float) -> bool:
    """灌殺後的第一根反攻：收紅，且收在 K 棒上半。"""
    rng = h - l
    if rng <= 1e-9:
        return False
    if c < o:
        return False
    return (c - l) / rng >= 0.55


def detect_signals(
    df,
    dump_bars: int = 6,
    min_dump_bars: int = 3,
    reclaim_window: int = 8,
    atr_len: int = 14,
    min_drop_pts: float = 40.0,
    min_drop_atr: float = 2.0,
    min_drop_pct: float = 0.0015,
    vol_lookback: int = 20,
    min_vol_mult: float = 1.4,
    ribbon_atr: float = 0.90,
    pre_ma20_atr: float = 1.30,
    stop_buffer: float = 8.0,
    target_r: float = 1.5,
    max_risk: float = 90.0,
    min_room_to_ma20: float = 12.0,
    min_rr: float = 1.0,
    skip_hour_start: Optional[int] = 9,
    skip_hour_end: Optional[int] = 10,
    rth_only: bool = True,
    min_entry_gap: int = 8,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    力成式急跌：
      1. 灌殺前均線黏在一起（5/10/20 帶寬窄），價格貼著 MA20
      2. 6 根內（約 30 分）從窗口高點灌下去，深度 ≥ max(40點, 2 ATR, 0.15%)
      3. 量能放大，收盤跌破 MA20（最好也跌破 MA60）
      4. 之後 8 根內出現反攻 K（收紅且收在上半），且還在 MA20 下方（有回歸空間）
    """
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    atr14 = atr(high, low, close, atr_len)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(60, atr_len, vol_lookback, dump_bars) + 2
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        if np.isnan(atr14[i]) or np.isnan(ma20[i]) or np.isnan(ma5[i]):
            i += 1
            continue

        a = float(atr14[i])
        if a <= 0:
            i += 1
            continue

        win0 = i - dump_bars + 1
        if win0 < 0:
            i += 1
            continue

        dump_high = float(np.max(high[win0 : i + 1]))
        dump_low = float(np.min(low[win0 : i + 1]))
        drop = dump_high - dump_low
        thresh = max(min_drop_pts, min_drop_atr * a, min_drop_pct * float(close[i]))
        if drop < thresh:
            i += 1
            continue

        reds = sum(1 for k in range(win0, i + 1) if close[k] < open_[k])
        if reds < min_dump_bars:
            i += 1
            continue

        bump("dump")

        pre = max(win0, 0)
        ribbon = float(max(ma5[pre], ma10[pre], ma20[pre]) - min(ma5[pre], ma10[pre], ma20[pre]))
        if ribbon > ribbon_atr * float(atr14[pre] if not np.isnan(atr14[pre]) else a):
            bump("skip_ribbon")
            i += 1
            continue
        if abs(float(close[pre]) - float(ma20[pre])) > pre_ma20_atr * a:
            bump("skip_pre_ma20")
            i += 1
            continue

        if close[i] >= ma20[i]:
            bump("skip_not_through_ma20")
            i += 1
            continue
        bump("through_ma")

        vol_avg = float(np.mean(volume[max(0, win0 - vol_lookback) : win0])) or 1.0
        dump_vol = float(np.mean(volume[win0 : i + 1]))
        vmult = dump_vol / vol_avg if vol_avg > 0 else 0.0
        if vmult < min_vol_mult:
            bump("skip_vol")
            i += 1
            continue
        bump("vol_ok")

        dump_idx = i
        dump_start = win0
        entered = False

        for j in range(dump_idx, min(dump_idx + reclaim_window + 1, n)):
            if low[j] < dump_low:
                dump_low = float(low[j])
                dump_idx = j
            if j == dump_idx:
                # 當根可以是錘子反轉；若收在下半則還在灌
                if not _is_reversal_bar(float(open_[j]), float(high[j]), float(low[j]), float(close[j])):
                    continue
            elif not _is_reversal_bar(float(open_[j]), float(high[j]), float(low[j]), float(close[j])):
                continue

            bump("reversal")

            if rth_only:
                ts = df.index[j]
                h = ts.hour
                m = ts.minute
                mins = h * 60 + m
                if mins < 9 * 60 + 30 or mins >= 16 * 60:
                    bump("skip_eth")
                    continue
            if skip_hour_start is not None and skip_hour_end is not None:
                h = df.index[j].hour
                if skip_hour_start <= h < skip_hour_end:
                    bump("skip_open_hour")
                    continue
            if j - last_entry < min_entry_gap:
                bump("skip_entry_gap")
                break

            entry = float(close[j])
            if np.isnan(ma20[j]) or entry >= float(ma20[j]) - min_room_to_ma20:
                bump("skip_late")
                continue

            stop = dump_low - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                break
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            room = float(ma20[j]) - entry
            target_by_r = entry + risk * target_r
            target = min(float(ma20[j]), target_by_r) if room > 0 else target_by_r
            if target <= entry:
                bump("skip_bad_target")
                continue
            rr = (target - entry) / risk
            if min_rr > 0 and rr < min_rr:
                bump("skip_rr")
                continue

            pierced_ma60 = (not np.isnan(ma60[dump_idx])) and dump_low < float(ma60[dump_idx])
            drop_atr = drop / a if a else 0.0
            q_score, q_grade = quality_from_dump(drop_atr, vmult, pierced_ma60)
            bump("taken")
            signals.append(
                Signal(
                    dump_start=dump_start,
                    dump_idx=dump_idx,
                    entry_idx=j,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    dump_high=dump_high,
                    dump_low=dump_low,
                    drop_pts=drop,
                    drop_atr=drop_atr,
                    vol_mult=vmult,
                    ma5=float(ma5[j]),
                    ma10=float(ma10[j]),
                    ma20=float(ma20[j]),
                    ma60=float(ma60[j]) if not np.isnan(ma60[j]) else 0.0,
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = j
            entered = True
            i = j + 4
            break

        if not entered:
            i = dump_idx + 1

    return signals


def simulate(
    df,
    signals: List[Signal],
    max_hold: int = 24,
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
    240: "#ab47bc",
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
    start = max(0, sig.dump_start - 18)
    end = min(len(df) - 1, trade.exit_idx + 12)
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

    dx, ex, xx = sig.dump_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= dx < len(window):
        ax.scatter([dx], [sig.dump_low], s=38, color="#facc15", zorder=5)
        ax.annotate("急跌低", (dx, sig.dump_low), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#fde68a", fontsize=8)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
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
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
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
            f"<span class='tag tag-info'>{t.signal.drop_atr:.1f} ATR</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R) → MA20\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"急跌 {t.signal.dump_high:.2f} → {t.signal.dump_low:.2f}  (−{t.signal.drop_pts:.1f} / {t.signal.drop_atr:.1f} ATR)\n"
            f"量能 {t.signal.vol_mult:.1f}x · MA5 {t.signal.ma5:.1f} / MA20 {t.signal.ma20:.1f} / MA60 {t.signal.ma60:.1f}"
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
    verdict: str = "",
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
            f"<section class='summary'><h1>{escape(extra_title or '核心對照（不過濾開盤／ETH）')}</h1>"
            f"<p class='muted'>同一套急跌＋反攻 K，但不過濾 09–10 與盤外。</p>"
            f"<div class='cards'><div class='card'>筆數<b>{extra_stats['count']}</b></div>"
            f"<div class='card'>勝率<b>{extra_stats['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總點數<b class='{extra_cls}'>{extra_stats['total_points']:+.1f}</b></div>"
            f"<div class='card'>勝/負<b>{extra_stats['wins']}/{extra_stats['count']-extra_stats['wins']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_points for t in extra_trades])}</div></section>"
        )
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：急跌 {funnel.get('dump', 0)} → "
            f"穿 MA20 {funnel.get('through_ma', 0)} → "
            f"放量 {funnel.get('vol_ok', 0)} → "
            f"反攻 {funnel.get('reversal', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（黏帶擋 {funnel.get('skip_ribbon', 0)} · 沒貼 MA20 {funnel.get('skip_pre_ma20', 0)} · "
            f"量能 {funnel.get('skip_vol', 0)} · 開盤檔 {funnel.get('skip_open_hour', 0)} · "
            f"ETH {funnel.get('skip_eth', 0)} · 太晚 {funnel.get('skip_late', 0)} · "
            f"RR不足 {funnel.get('skip_rr', 0)} · 風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 急跌 V 反</title>
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
<h1>{escape(symbol)} 急跌 V 反（力成邏輯）</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)} · 五分 K</p>
<p class="muted">均線黏帶後 30 分內灌穿 MA20、放量、反攻 K 做多。停損急跌低下方，目標 MA20 或 1.5R。09–10 不進、只做 RTH。</p>
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
# CLI
# ---------------------------------------------------------------------------


CORE_DETECT = {
    "skip_hour_start": None,
    "skip_hour_end": None,
    "rth_only": False,
}


def _print_trades(df, trades, tag: str = "") -> None:
    prefix = f"{tag} " if tag else ""
    for i, t in enumerate(trades, 1):
        print(
            f"  [{prefix}{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  drop={t.signal.drop_pts:.0f}pt/{t.signal.drop_atr:.1f}ATR"
        )


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} {args.interval} {args.period}", file=sys.stderr)
    df = to_et(load_yfinance(args.symbol, args.interval, args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    print(f"bars={len(df)} {df.index[0]} → {df.index[-1]}", file=sys.stderr)

    funnel: Dict[str, int] = {}
    detect_kw = {}
    if getattr(args, "loose", False):
        detect_kw.update(CORE_DETECT)
    sigs = detect_signals(df, funnel=funnel, **detect_kw)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(
        f"trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_points']:+.1f} avg={stats['avg']:+.1f}"
    )
    print("funnel", funnel)
    _print_trades(df, trades)

    extra_trades: List[TradeResult] = []
    extra_funnel: Dict[str, int] = {}
    if getattr(args, "pages", False) and not getattr(args, "loose", False):
        core_sigs = detect_signals(df, funnel=extra_funnel, **CORE_DETECT)
        extra_trades = simulate(df, core_sigs)
        extra_stats = summarize_trades(extra_trades)
        print(
            f"core  trades={extra_stats['count']} WR={extra_stats['win_rate']:.1f}% "
            f"pnl={extra_stats['total_points']:+.1f}  (含 ETH / 開盤)"
        )
        _print_trades(df, extra_trades, "core")

    if stats["count"] == 0:
        verdict = "這段樣本沒打到力成那種急跌。NQ 五分很少出現「黏帶後灌 2ATR+、還有 1R 回到 MA20」的 V。"
    elif stats["total_points"] > 0 and stats["win_rate"] >= 45:
        verdict = "有料：RTH 過濾後期望值為正。仍是接刀，樣本少、遇到單邊續跌會一次吐回去。"
    elif stats["total_points"] > 0:
        verdict = "邊緣：總點數正，但勝率不高，比較像偶爾抓到 V，不是穩的優勢。"
    else:
        verdict = "沒料：力成是個股恐慌回補；NQ 急跌後常續跌。贏家回到 MA20，輸家一次吃掉整段急跌。"

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
            extra_title="核心（含 ETH / 開盤）",
            verdict=verdict,
        )
        print(f"html={out}")
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 五分 K 急跌 V 反（力成邏輯）")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 5m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--interval", default="5m")
    b.add_argument("--period", default="60d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-sharp-drop/index.html")
    b.add_argument("--loose", action="store_true", help="含 ETH / 開盤")
    b.set_defaults(func=cmd_backtest)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--loose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
