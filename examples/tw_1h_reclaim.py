#!/usr/bin/env python3
"""台股 1h 破底翻（寬鬆版）— 認定一波後，破底 36 小時內收盤站上 MA5/10/20 發訊號。

寬鬆版（預設，沒加 --strict）:
  • 1h 收盤從 MA20 上方跌到下方，開始算。
  • 在下面待 4～36 根。中間 1～2 根假站上，不算結束。
  • 之後要再有一根收盤站回 MA20 這波才算數。
  • 相對過程中最高 MA20，最低點深度 ≥ 1.8%。
  • 最低點必須是近 16 根新低（破底）。更高低點的 W 不算。
  • 不要求急殺、ATR、也不要求先做一腳再吻回的筆畫 W。
  • 破底之後 36 根內，第一根同時 收盤 > MA5 MA10 MA20 → 進場／通知。

--strict 只多兩道：不准假站上；進場還要 MA5>MA10>MA20 多頭排列。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    REPO,
    TPE,
    _chart_payload_to_df,
    _get_json,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
)

PAGES = REPO / "docs" / "tw-1h-reclaim" / "index.html"
MA_COLORS = {5: "#f0c14b", 10: "#79c0ff", 20: "#f472b6"}


# ---------------------------------------------------------------------------
# Params / types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReclaimParams:
    min_below: int = 4
    max_below: int = 36
    max_fakes: int = 2
    min_depth: float = 0.018
    lookback: int = 16
    entry_window: int = 36
    require_stack: bool = False
    target_r: float = 2.0
    time_bars: int = 20


def loose_params(**overrides: Any) -> ReclaimParams:
    return ReclaimParams(**overrides)


def strict_params(**overrides: Any) -> ReclaimParams:
    data = dict(max_fakes=0, require_stack=True)
    data.update(overrides)
    return ReclaimParams(**data)


@dataclass(frozen=True)
class Wave:
    start_idx: int
    reclaim_idx: int
    trough_idx: int
    trough_low: float
    max_ma20: float
    depth_pct: float
    bars_below: int
    fake_stands: int


@dataclass(frozen=True)
class Signal:
    entry_idx: int
    entry_price: float
    ma5: float
    ma10: float
    ma20: float
    wave: Wave


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
    pnl_pct: float
    exit_reason: str
    fwd_1d: Optional[float] = None
    fwd_3d: Optional[float] = None
    fwd_5d: Optional[float] = None


@dataclass
class TwHit:
    row: dict
    trade: TradeResult
    df: pd.DataFrame


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def sma(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if n <= 0 or len(values) < n:
        return out
    c = np.cumsum(np.asarray(values, dtype=float))
    out[n - 1] = c[n - 1] / n
    if len(values) > n:
        out[n:] = (c[n:] - c[:-n]) / n
    return out


def _is_16h_new_low(low: np.ndarray, trough_idx: int, trough_low: float, lookback: int) -> bool:
    if trough_idx < lookback:
        return False
    prev = low[trough_idx - lookback : trough_idx]
    if len(prev) < lookback or np.isnan(prev).all():
        return False
    return float(trough_low) < float(np.nanmin(prev)) - 1e-12


def detect_signals(
    df: pd.DataFrame,
    params: Optional[ReclaimParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """依寬鬆版規則抓 1h 破底翻進場點。小時 = 1h 根數。"""
    p = params or loose_params()
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    n = len(close)
    warmup = max(20, p.lookback)
    signals: List[Signal] = []
    i = warmup

    while i < n:
        if np.isnan(ma20[i]) or np.isnan(ma20[i - 1]):
            i += 1
            continue
        if not (close[i - 1] >= ma20[i - 1] and close[i] < ma20[i]):
            i += 1
            continue
        bump("cross_below")

        start = i
        max_ma20 = float(ma20[i])
        trough_low = float(low[i])
        trough_idx = i
        fake_stands = 0
        streak_above = 0
        bars_below = 0
        confirmed: Optional[Wave] = None
        fail_reason = "timeout"
        span_limit = p.max_below + p.max_fakes

        for j in range(start, min(n, start + span_limit + 1)):
            if not np.isnan(ma20[j]) and float(ma20[j]) > max_ma20:
                max_ma20 = float(ma20[j])
            if float(low[j]) < trough_low:
                trough_low = float(low[j])
                trough_idx = j

            standing = (not np.isnan(ma20[j])) and close[j] > ma20[j]
            if not standing:
                bars_below += 1
                if streak_above:
                    fake_stands += streak_above
                    streak_above = 0
                if bars_below > p.max_below:
                    fail_reason = "timeout"
                    break
                continue

            depth = (max_ma20 - trough_low) / max_ma20 if max_ma20 > 0 else 0.0
            new_low = _is_16h_new_low(low, trough_idx, trough_low, p.lookback)
            qualifies = (
                bars_below >= p.min_below
                and bars_below <= p.max_below
                and depth >= p.min_depth
                and new_low
            )
            if qualifies:
                confirmed = Wave(
                    start_idx=start,
                    reclaim_idx=j,
                    trough_idx=trough_idx,
                    trough_low=trough_low,
                    max_ma20=max_ma20,
                    depth_pct=depth,
                    bars_below=bars_below,
                    fake_stands=fake_stands,
                )
                bump("wave_ok")
                break

            streak_above += 1
            if fake_stands + streak_above > p.max_fakes:
                if bars_below < p.min_below:
                    fail_reason = "too_short"
                elif not new_low:
                    fail_reason = "not_16h_low"
                elif depth < p.min_depth:
                    fail_reason = "shallow"
                else:
                    fail_reason = "too_many_fakes"
                break
        else:
            fail_reason = "timeout"

        if confirmed is None:
            bump(fail_reason)
            k = start + 1
            while k < n:
                if not np.isnan(ma20[k]) and close[k] > ma20[k]:
                    break
                k += 1
            i = max(k, start + 1)
            continue

        lo = confirmed.reclaim_idx
        hi = min(confirmed.trough_idx + p.entry_window, n - 1)
        found = False
        if lo <= hi:
            for e in range(lo, hi + 1):
                if np.isnan(ma5[e]) or np.isnan(ma10[e]) or np.isnan(ma20[e]):
                    continue
                stacked = (not p.require_stack) or (ma5[e] > ma10[e] > ma20[e])
                if close[e] > ma5[e] and close[e] > ma10[e] and close[e] > ma20[e] and stacked:
                    signals.append(
                        Signal(
                            entry_idx=e,
                            entry_price=float(close[e]),
                            ma5=float(ma5[e]),
                            ma10=float(ma10[e]),
                            ma20=float(ma20[e]),
                            wave=confirmed,
                        )
                    )
                    bump("entry")
                    found = True
                    break
        if not found:
            bump("entry_timeout")
        i = confirmed.reclaim_idx + 1

    return signals


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def _session_close_indices(index: pd.DatetimeIndex) -> List[int]:
    last: Dict[Any, int] = {}
    for i, ts in enumerate(index):
        last[ts.date()] = i
    return [last[d] for d in sorted(last)]


def _fwd_pct(close: np.ndarray, session_ends: Sequence[int], entry_idx: int, sessions: int) -> Optional[float]:
    after = [i for i in session_ends if i > entry_idx]
    if len(after) < sessions:
        return None
    nxt = after[sessions - 1]
    if close[entry_idx] == 0:
        return None
    return float(close[nxt] / close[entry_idx] - 1.0)


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    params: Optional[ReclaimParams] = None,
) -> List[TradeResult]:
    p = params or loose_params()
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    ends = _session_close_indices(df.index)
    n = len(close)
    trades: List[TradeResult] = []

    for sig in signals:
        entry_idx = sig.entry_idx
        entry = float(sig.entry_price)
        stop = float(sig.wave.trough_low)
        risk = entry - stop
        if risk <= 0:
            stop = entry * 0.99
            risk = entry - stop
        target = entry + p.target_r * risk
        exit_idx = entry_idx
        exit_px = entry
        reason = "open"
        last = min(n - 1, entry_idx + p.time_bars)
        for k in range(entry_idx + 1, last + 1):
            if float(low[k]) <= stop:
                exit_idx, exit_px, reason = k, stop, "stop"
                break
            if float(high[k]) >= target:
                exit_idx, exit_px, reason = k, target, "target"
                break
        else:
            if last > entry_idx:
                exit_idx, exit_px, reason = last, float(close[last]), "time"
                if last == n - 1 and last < entry_idx + p.time_bars:
                    reason = "open"
            else:
                exit_idx, exit_px, reason = entry_idx, entry, "open"
        trades.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_px,
                stop_price=stop,
                target_price=target,
                pnl_points=exit_px - entry,
                pnl_pct=(exit_px / entry - 1.0) if entry else 0.0,
                exit_reason=reason,
                fwd_1d=_fwd_pct(close, ends, entry_idx, 1),
                fwd_3d=_fwd_pct(close, ends, entry_idx, 3),
                fwd_5d=_fwd_pct(close, ends, entry_idx, 5),
            )
        )
    return trades


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    pnls = [float(t.pnl_points) for t in trades]
    pcts = [float(t.pnl_pct) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    closed = [t for t in trades if t.exit_reason != "open"]
    closed_pcts = [float(t.pnl_pct) for t in closed]
    closed_wins = sum(1 for t in closed if t.pnl_points > 0)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    def _avg_fwd(attr: str) -> Optional[float]:
        xs = [getattr(t, attr) for t in trades if getattr(t, attr) is not None]
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "total_pct": float(sum(pcts)),
        "avg_pct": float(sum(pcts) / n) if n else 0.0,
        "closed": len(closed),
        "open": n - len(closed),
        "closed_win_rate": 100.0 * closed_wins / len(closed) if closed else 0.0,
        "closed_avg_pct": float(sum(closed_pcts) / len(closed_pcts)) if closed_pcts else 0.0,
        "reasons": reasons,
        "fwd_1d": _avg_fwd("fwd_1d"),
        "fwd_3d": _avg_fwd("fwd_3d"),
        "fwd_5d": _avg_fwd("fwd_5d"),
    }


def filter_entry_window(df: pd.DataFrame, signals: Sequence[Signal], days: int) -> List[Signal]:
    if not len(df) or days <= 0:
        return list(signals)
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    return [s for s in signals if df.index[s.entry_idx] >= start]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def fetch_yahoo_1h(symbol: str, range_: str = "2mo") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=60m&range={range_}&includePrePost=false"
    )
    return _chart_payload_to_df(_get_json(url))


# ---------------------------------------------------------------------------
# Chart / HTML
# ---------------------------------------------------------------------------


def _setup_cjk() -> None:
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


def _trade_window(df: pd.DataFrame, trade: TradeResult, pad_left: int = 10, pad_right: int = 4) -> tuple[int, int]:
    sig = trade.signal
    start = max(0, min(sig.wave.start_idx, sig.wave.trough_idx, trade.entry_idx) - pad_left)
    end = min(len(df) - 1, max(trade.exit_idx, trade.entry_idx, sig.wave.reclaim_idx) + pad_right)
    return start, end


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    title_extra: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _setup_cjk()
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
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.8)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    bx = sig.wave.trough_idx - start
    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    sx = sig.wave.start_idx - start
    rx = sig.wave.reclaim_idx - start
    if 0 <= sx < len(window):
        ax.axvline(sx, color="#8aa193", ls=":", lw=0.7, alpha=0.6)
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.wave.trough_low], s=38, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, sig.wave.trough_low),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= rx < len(window):
        ax.axvline(rx, color="#79c0ff", ls=":", lw=0.8, alpha=0.7)
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
    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {trade.pnl_pct*100:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=3)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            text=True,
        )
        return out.strip() or "main"
    except Exception:  # noqa: BLE001
        return "main"


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _fmt_fwd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value*100:+.2f}%"


def write_tw_html(
    path: Path,
    hits: List[TwHit],
    universe: List[dict],
    period: str,
    date: str,
    funnel: Optional[Dict[str, int]] = None,
    strict: bool = False,
) -> Path:
    stats = summarize_trades([h.trade for h in hits])
    cards: List[str] = []
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        img_name = f"t{i:02d}_{hit.row['code']}_{et.strftime('%m%d_%H%M')}.png"
        label = f"{hit.row['code']} {hit.row['name']}"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=label)
        w = t.signal.wave
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>深度 {w.depth_pct*100:.1f}%</span>"
            f"<span class='tag'>假站 {w.fake_stands}</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  stop {t.stop_price:.2f} (−{risk:.2f})\n"
            f"target {t.target_price:.2f}  exit {t.exit_price:.2f} {t.exit_reason}  {t.pnl_points:+.2f}\n"
            f"破底 {w.trough_low:.2f} @ {df.index[w.trough_idx].strftime('%m-%d %H:%M')}  "
            f"站回 {df.index[w.reclaim_idx].strftime('%m-%d %H:%M')}\n"
            f"MA5 {t.signal.ma5:.2f} / MA10 {t.signal.ma10:.2f} / MA20 {t.signal.ma20:.2f}\n"
            f"fwd +1d {_fmt_fwd(t.fwd_1d)}  +3d {_fmt_fwd(t.fwd_3d)}  +5d {_fmt_fwd(t.fwd_5d)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    mode = "嚴格" if strict else "寬鬆"
    fun = funnel or {}
    fwd1 = _fmt_fwd(stats.get("fwd_1d"))
    fwd3 = _fmt_fwd(stats.get("fwd_3d"))
    fwd5 = _fmt_fwd(stats.get("fwd_5d"))
    reasons = stats.get("reasons") or {}
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · {mode}版</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 1h 破底翻 · {mode}版 · 成交額前 {len(universe)}</h1>
<p class="muted">{escape(period)} · 基準日 {escape(date)} · {len(universe)} 檔 · 成交額末名約 {cutoff:.1f} 億
<br/>跌破 MA20 後待 4～36 根（中間最多 {0 if strict else 2} 根假站上），深度 ≥ 1.8%，最低點為近 16 根新低，再站回 MA20。
破底後 36 根內第一根收盤 &gt; MA5 / MA10 / MA20 進場。
回測出場：停在破底低、2R、或 20 根時間停。加總％是各筆報酬相加，不是組合複利。</p>
<p class="muted">漏斗：跌破 {fun.get('cross_below', 0)} → 成波 {fun.get('wave_ok', 0)} → 進場 {fun.get('entry', 0)}
· 太短 {fun.get('too_short', 0)} · 不夠深 {fun.get('shallow', 0)} · 非破底 {fun.get('not_16h_low', 0)}
· 假站過多 {fun.get('too_many_fakes', 0)} · 逾時 {fun.get('timeout', 0)} · 沒站上均線 {fun.get('entry_timeout', 0)}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}
· 收盤後 +1d {fwd1} · +3d {fwd3} · +5d {fwd5}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if stats['avg_pct']>=0 else 'pnl-loss'}">{stats['avg_pct']*100:+.2f}%</b></div>
<div class="card">標的<b>{len({h.row['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有寬鬆版破底翻訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_symbol(
    row: dict,
    range_: str,
    params: ReclaimParams,
    days: int,
    funnel: Optional[Dict[str, int]] = None,
) -> tuple[List[TwHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_yahoo_1h(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 40:
        meta["error"] = "too_few_bars"
        return [], meta
    local_fun: Dict[str, int] = {}
    sigs = detect_signals(df, params, funnel=local_fun)
    if funnel is not None:
        for k, v in local_fun.items():
            funnel[k] = funnel.get(k, 0) + v
    sigs = filter_entry_window(df, sigs, days)
    trades = simulate(df, sigs, params)
    meta["n_sig"] = len(sigs)
    meta["n_trade"] = len(trades)
    return [TwHit(row, t, df) for t in trades], meta


def dump_hits_json(path: Path, hits: List[TwHit], stats: dict, funnel: dict, extra: dict) -> Path:
    rows = []
    for hit in hits:
        t = hit.trade
        df = hit.df
        w = t.signal.wave
        rows.append(
            {
                "code": hit.row["code"],
                "name": hit.row["name"],
                "symbol": hit.row["symbol"],
                "entry_time": str(df.index[t.entry_idx]),
                "exit_time": str(df.index[t.exit_idx]),
                "entry": t.entry_price,
                "exit": t.exit_price,
                "stop": t.stop_price,
                "target": t.target_price,
                "pnl": t.pnl_points,
                "pnl_pct": t.pnl_pct,
                "reason": t.exit_reason,
                "depth_pct": w.depth_pct,
                "trough": w.trough_low,
                "trough_time": str(df.index[w.trough_idx]),
                "reclaim_time": str(df.index[w.reclaim_idx]),
                "fake_stands": w.fake_stands,
                "bars_below": w.bars_below,
                "fwd_1d": t.fwd_1d,
                "fwd_3d": t.fwd_3d,
                "fwd_5d": t.fwd_5d,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "funnel": funnel, "extra": extra, "hits": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股成交額前 N · 1h 破底翻回測")
    p.add_argument("--date", default="", help="YYYYMMDD，預設上一個交易日")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--days", type=int, default=14, help="只統計進場落在最近 N 日的訊號")
    p.add_argument("--range", dest="range_", default="2mo", help="Yahoo 1h 下載區間")
    p.add_argument("--sleep", type=float, default=0.18)
    p.add_argument("--strict", action="store_true", help="不准假站上，進場要 MA5>MA10>MA20")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--json", dest="json_path", default="")
    args = p.parse_args(argv)

    params = strict_params() if args.strict else loose_params()
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    pool = max(args.limit, args.pool if args.max_price else args.limit)
    print(
        f"universe date={date} limit={args.limit} days={args.days} range={args.range_} "
        f"strict={args.strict} max_price={args.max_price}"
    )
    raw = fetch_top_turnover(date, pool)
    universe, dropped = filter_by_max_price(raw, args.max_price, args.limit)
    if dropped:
        print(
            "drop price>"
            + str(args.max_price)
            + ": "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:12])
            + (" …" if len(dropped) > 12 else "")
        )
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
        f"{universe[0]['amount']/1e8:.1f}億 / {universe[0]['close']} · "
        f"末 {universe[-1]['code']} {universe[-1]['amount']/1e8:.1f}億 / {universe[-1]['close']}"
    )

    hits: List[TwHit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    scanned = 0
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol(row, args.range_, params, args.days, funnel=funnel)
        scanned += 1
        if meta["error"]:
            errors += 1
        hits.extend(stock_hits)
        flag = f" trades={meta['n_trade']}" if meta["n_trade"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row['name']} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    stats = summarize_trades([h.trade for h in hits])
    print(
        f"done scanned={scanned} errors={errors} trades={stats['count']} "
        f"WR={stats['win_rate']:.1f}% pnl%={stats['total_pct']*100:+.2f} funnel={funnel}"
    )
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        ts = hit.df.index[t.entry_idx]
        print(
            f"  [{i}] {hit.row['code']} {hit.row['name']} {ts.strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_pct*100:+.2f}%"
        )

    extra = {
        "date": date,
        "days": args.days,
        "strict": args.strict,
        "range": args.range_,
        "limit": args.limit,
        "generated": datetime.now(TPE).isoformat(timespec="seconds"),
    }
    html_path = Path(args.html) if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        period_label = f"{args.days}d · Yahoo {args.range_} 1h"
        if args.max_price is not None:
            period_label += f" · 股價≤{args.max_price:g}"
        out = write_tw_html(html_path, hits, universe, period_label, date, funnel=funnel, strict=args.strict)
        write_view_html(out)
        print(f"html={out}")
    json_path = Path(args.json_path) if args.json_path else None
    if json_path is None and html_path:
        json_path = html_path.with_name("hits.json")
    if json_path:
        dump_hits_json(json_path, hits, stats, funnel, extra)
        print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
