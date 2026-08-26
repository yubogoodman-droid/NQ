#!/usr/bin/env python3
"""NQ 一分 K：破底後突破 MA60、回踩 MA60 進場。

對齊截圖（1m NQmain）：
  破底（低點靠近 1m MA60）→ 收盤突破 MA60 → 回踩踩住季線 → 進場做多
停損在回踩低點／MA60 下方；目標 2R。

用法:
  python3 examples/nq_ma60_retest.py
  python3 examples/nq_ma60_retest.py backtest --period 8d --html output/nq_ma60_retest.html
  python3 examples/nq_ma60_retest.py backtest --period 30d --pages
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    ET,
    load_bars,
    rolling_min_prev,
    sma,
    summarize_trades,
    to_et,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma60-retest" / "index.html"
VIEW_BRANCH = "cursor/nq-1m-ma60-retest-8fa0"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    break_idx: int
    breakout_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    two_hr_low: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    ma200: float
    extension: float
    slope60: float
    below_ma60: float = 0.0
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


def quality_from_retest(slope60: float, bull_bar: bool, below_ma60: float) -> Tuple[int, str]:
    """近 MA60 的破底加分（對齊截圖，不要離季線太遠）。"""
    score = 0
    if slope60 >= 0:
        score += 1
    if bull_bar:
        score += 1
    if 0 < below_ma60 <= 40.0:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def detect_signals(
    df: pd.DataFrame,
    two_hour_bars: int = 120,
    min_break_depth: float = 10.0,
    min_below_ma60: float = 15.0,
    max_below_ma60: float = 45.0,
    breakout_window: int = 60,
    retest_window: int = 30,
    min_retest_gap: int = 5,
    min_extension: float = 8.0,
    min_clear_pts: float = 0.0,
    touch_pts: float = 15.0,
    pierce_pts: float = 10.0,
    stop_buffer: float = 12.0,
    target_r: float = 2.0,
    max_risk: float = 60.0,
    min_ma60_slope: float = -8.0,
    ma60_slope_bars: int = 5,
    cooldown: int = 25,
    skip_hour_start: Optional[int] = 9,
    skip_hour_end: Optional[int] = 10,
    require_bull: bool = False,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """破 2h 低且低點靠近 1m MA60 → 收盤站上 MA60 → 回踩季線進場。"""
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    ma200 = sma(close, 200)
    two_hr_low = rolling_min_prev(low, two_hour_bars)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    warmup = max(200, two_hour_bars, 60) + 1
    i = warmup
    while i < n - 1:
        if np.isnan(two_hr_low[i]) or np.isnan(ma60[i]):
            i += 1
            continue
        if low[i] >= float(two_hr_low[i]) or close[i] >= float(ma60[i]):
            i += 1
            continue
        break_depth = float(two_hr_low[i]) - float(low[i])
        if break_depth < min_break_depth:
            i += 1
            continue
        below60 = float(ma60[i]) - float(low[i])
        if below60 < min_below_ma60:
            i += 1
            continue

        bump("break")
        break_idx = i
        break_low = float(low[i])
        support = float(two_hr_low[i])

        breakout_idx: Optional[int] = None
        end_bo = min(break_idx + breakout_window, n - 1)
        j = break_idx + 1
        while j <= end_bo:
            if np.isnan(ma60[j]):
                j += 1
                continue
            if float(low[j]) < break_low and float(close[j]) < float(ma60[j]):
                break_idx = j
                break_low = float(low[j])
                below60 = float(ma60[j]) - break_low
                end_bo = min(break_idx + breakout_window, n - 1)
            if j > 0 and float(close[j - 1]) <= float(ma60[j - 1]) and float(close[j]) > float(ma60[j]):
                breakout_idx = j
                break
            j += 1

        if breakout_idx is None:
            bump("no_breakout")
            i = break_idx + 1
            continue
        below60 = float(ma60[break_idx]) - break_low
        if below60 > max_below_ma60:
            bump("too_far")
            i = break_idx + 1
            continue
        bump("breakout")

        ext_high = float(high[breakout_idx])
        cleared = float(low[breakout_idx]) >= float(ma60[breakout_idx]) + min_clear_pts
        entered = False
        lost = False
        end_rt = min(breakout_idx + retest_window, n - 1)
        for k in range(breakout_idx + 1, end_rt + 1):
            if np.isnan(ma60[k]):
                continue
            ext_high = max(ext_high, float(high[k]))
            ma = float(ma60[k])
            if float(close[k]) < ma:
                bump("lost_ma60")
                lost = True
                break
            if float(low[k]) >= ma + min_clear_pts:
                cleared = True
            if k - breakout_idx < min_retest_gap:
                continue
            if not cleared or ext_high < ma + min_extension:
                continue
            if float(low[k]) > ma + touch_pts:
                continue
            if float(low[k]) < ma - pierce_pts:
                bump("pierce_too_deep")
                lost = True
                break
            bull_bar = float(close[k]) >= float(open_[k])
            if require_bull and not bull_bar:
                continue
            if skip_hour_start is not None and skip_hour_end is not None:
                hour = df.index[k].hour
                if skip_hour_start <= hour < skip_hour_end:
                    bump("skip_open_hour")
                    continue
            if k - last_entry < cooldown:
                bump("skip_cooldown")
                lost = True
                break

            slope60 = 0.0
            if k >= ma60_slope_bars and not np.isnan(ma60[k - ma60_slope_bars]):
                slope60 = float(ma60[k]) - float(ma60[k - ma60_slope_bars])
            if slope60 < min_ma60_slope:
                bump("skip_slope")
                continue

            entry = float(close[k])
            stop = min(float(low[k]), ma) - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                continue
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            q_score, q_grade = quality_from_retest(slope60, bull_bar, below60)
            bump("taken")
            signals.append(
                Signal(
                    break_idx=break_idx,
                    breakout_idx=breakout_idx,
                    entry_idx=k,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=entry + risk * target_r,
                    break_low=break_low,
                    two_hr_low=support,
                    ma5=float(ma5[k]) if not np.isnan(ma5[k]) else 0.0,
                    ma10=float(ma10[k]) if not np.isnan(ma10[k]) else 0.0,
                    ma20=float(ma20[k]) if not np.isnan(ma20[k]) else 0.0,
                    ma60=ma,
                    ma200=float(ma200[k]) if not np.isnan(ma200[k]) else 0.0,
                    extension=float(ext_high - ma),
                    slope60=slope60,
                    below_ma60=float(below60),
                    quality=q_grade,
                    quality_score=q_score,
                )
            )
            last_entry = k
            entered = True
            i = k + 1
            break

        if entered:
            continue
        if lost:
            i = breakout_idx + 1
        else:
            bump("no_retest")
            i = breakout_idx + 1

    return signals


def simulate(
    df: pd.DataFrame,
    signals: List[Signal],
    max_hold: int = 90,
    be_after_r: float = 0.0,
    trail_after_r: float = 1.5,
    trail_lock_r: float = 0.5,
    preopen_flat: bool = True,
    exit_on_ma60_lose: bool = False,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma60 = sma(close, 60)
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
            if exit_on_ma60_lose and not np.isnan(ma60[k]) and float(close[k]) < float(ma60[k]):
                exit_idx, exit_price, exit_reason = k, float(close[k]), "ma60_lose"
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
    30: "#42a5f5",
    60: "#26a69a",
    100: "#ffeb3b",
    120: "#ef5350",
    200: "#26c6da",
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
    start = max(0, trade.signal.break_idx - 20)
    end = min(len(df) - 1, trade.exit_idx + 16)
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


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _apply_cjk_font()
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
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.15 if n == 60 else (1.25 if n <= 20 else 0.95)
        ax.plot(list(xs), ma, color=col, lw=lw, label="MA60" if n == 60 else f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    bx = sig.break_idx - start
    ox = sig.breakout_idx - start
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
    if 0 <= ox < len(window):
        ax.axvline(ox, color="#26a69a", ls=":", lw=0.8, alpha=0.7)
        ax.annotate(
            "突破",
            (ox, float(c.iloc[ox])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#80cbc4",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "回踩進場",
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
        f"#{trade_no}  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
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
) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        bt = df.index[t.signal.break_idx]
        ot = df.index[t.signal.breakout_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {
            "target": "tag-tp",
            "stop": "tag-sl",
            "ma60_lose": "tag-sl",
        }.get(t.exit_reason, "tag-time")
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
            f"<span class='tag tag-info'>1m</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破底 {bt.strftime('%H:%M')} low={t.signal.break_low:.2f} / 2h低 {t.signal.two_hr_low:.2f}\n"
            f"破底距MA60 {t.signal.below_ma60:.1f} pts\n"
            f"突破 {ot.strftime('%H:%M')}  回踩 {et.strftime('%H:%M')}\n"
            f"MA5 {t.signal.ma5:.1f} / MA20 {t.signal.ma20:.1f} / MA60 {t.signal.ma60:.1f} / MA200 {t.signal.ma200:.1f}"
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
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = []
    for q, info in stats.get("by_quality", {}).items():
        q_bits.append(f"Q{q} {info['n']}筆 {info['pnl']:+.1f}")
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades, out, prefix="t")
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破底 {funnel.get('break', 0)} → "
            f"突破MA60 {funnel.get('breakout', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（距MA60太遠 {funnel.get('too_far', 0)} · 沒回踩 {funnel.get('no_retest', 0)} · 失守 {funnel.get('lost_ma60', 0)} · "
            f"刺太深 {funnel.get('pierce_too_deep', 0)} · 沒突破 {funnel.get('no_breakout', 0)} · "
            f"斜率 {funnel.get('skip_slope', 0)} · 9點檔 {funnel.get('skip_open_hour', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 破底後回踩 MA60</title>
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
<h1>{escape(symbol)} 破底後回踩 MA60</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">1 分鐘：破 2h 低，低點距 1m MA60 不超過 45 點 → 收盤突破 MA60 → 回踩踩住 MA60 進場。停損在回踩低點／季線下方，目標 2R。</p>
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
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8")
    if "圖是靜態 1m K 線" not in text:
        text = text.replace(
            "</h1>\n<p class=\"muted\">",
            "</h1>\n<p class=\"muted\">圖是靜態 1m K 線。手機請往下捲。</p>\n<p class=\"muted\">",
            1,
        )
    text = text.replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
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
    sigs = detect_signals(df, funnel=funnel)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if funnel:
        print(
            "funnel "
            f"break={funnel.get('break', 0)} breakout={funnel.get('breakout', 0)} "
            f"taken={funnel.get('taken', 0)} too_far={funnel.get('too_far', 0)} "
            f"no_retest={funnel.get('no_retest', 0)} "
            f"lost={funnel.get('lost_ma60', 0)} pierce={funnel.get('pierce_too_deep', 0)} "
            f"no_bo={funnel.get('no_breakout', 0)} slope={funnel.get('skip_slope', 0)} "
            f"hour={funnel.get('skip_open_hour', 0)}"
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
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 一分K 破底後回踩 MA60")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-retest/index.html")
    b.set_defaults(func=cmd_backtest)
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-retest/index.html")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
