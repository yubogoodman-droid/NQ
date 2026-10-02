#!/usr/bin/env python3
"""台股 1h：突破後回踩 MA60。

對齊討論裡那五張小時圖的結構（中美晶最標準）：
  • 先在上升／走平的 MA60 上方走出一段延伸（高點距 MA60 ≥ 3%）。
  • 之後才回踩：低點碰到 MA60 附近（可略刺穿），收盤仍站在均線上。
  • 進場收盤不能離 MA60 太遠（不是追噴出）。
  • 同一段突破只吃第一次回踩。
  • 收盤明顯跌破 MA60 算轉空，不當成回踩。
  • 從下方站回（破底翻）這套不吃。

回測出場：停在回踩低點（至少距進場 0.8%）、目標 2R、或 20 根時間停。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    REPO,
    TPE,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)
from tw_1h_reclaim import (  # noqa: E402
    _fwd_pct,
    _git_branch,
    _session_close_indices,
    _setup_cjk,
    fetch_yahoo_1h,
    filter_entry_window,
    sma,
    summarize_trades as _summarize_trades,
    write_view_html,
)

PAGES = REPO / "docs" / "tw-1h-ma60-retest-7d" / "index.html"
MA_COLORS = {5: "#f0c14b", 10: "#79c0ff", 20: "#f472b6", 60: "#3ddc68"}

# 對話裡那五檔，即使沒進成交額前段也掃進去對照。
WATCH = [
    {"code": "6533", "name": "晶心科", "market": "tse"},
    {"code": "5483", "name": "中美晶", "market": "otc"},
    {"code": "5309", "name": "系統電", "market": "otc"},
    {"code": "6456", "name": "GIS-KY", "market": "tse"},
    {"code": "4968", "name": "立積", "market": "tse"},
]


@dataclass(frozen=True)
class RetestParams:
    ma_n: int = 60
    slope_lookback: int = 10
    min_ext: float = 0.03
    touch_above: float = 0.01
    max_under: float = 0.015
    max_close_below: float = 0.005
    max_entry_stretch: float = 0.025
    max_retest_bars: int = 30
    min_risk_pct: float = 0.008
    target_r: float = 2.0
    time_bars: int = 20


def loose_params(**overrides: Any) -> RetestParams:
    return RetestParams(**overrides)


@dataclass(frozen=True)
class Signal:
    entry_idx: int
    entry_price: float
    ma60: float
    peak_idx: int
    peak_high: float
    ext_pct: float
    retest_low: float
    run_start: int


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


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    return _summarize_trades(trades)


def detect_signals(
    df: pd.DataFrame,
    params: Optional[RetestParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    p = params or loose_params()
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma = sma(close, p.ma_n)
    n = len(close)
    signals: List[Signal] = []
    i = p.ma_n

    while i < n:
        if np.isnan(ma[i]) or close[i] <= ma[i]:
            i += 1
            continue

        bump("above_run")
        run_start = i
        peak_high = float(high[i])
        peak_idx = i
        peak_ma = float(ma[i])
        extended = False
        j = i
        ended = n

        while j < n:
            if np.isnan(ma[j]):
                j += 1
                continue
            if close[j] < ma[j] * (1.0 - p.max_close_below):
                bump("breakdown")
                ended = j + 1
                break

            if float(high[j]) >= peak_high:
                peak_high = float(high[j])
                peak_idx = j
                peak_ma = float(ma[j])

            ext_pct = (peak_high / peak_ma - 1.0) if peak_ma > 0 else 0.0
            if ext_pct >= p.min_ext:
                if not extended:
                    bump("extended")
                    extended = True

            if extended:
                if j - peak_idx > p.max_retest_bars:
                    bump("retest_timeout")
                    ended = j + 1
                    break
                live_ext = (peak_high / float(ma[j]) - 1.0) if float(ma[j]) > 0 else 0.0
                if j > peak_idx:
                    rising = True
                    sl = j - p.slope_lookback
                    if sl >= 0 and not np.isnan(ma[sl]):
                        rising = float(ma[j]) >= float(ma[sl]) - 1e-12
                    if not rising:
                        bump("falling_ma")
                    elif live_ext < p.min_ext:
                        bump("ext_faded")
                    else:
                        touched = (float(low[j]) <= float(ma[j]) * (1.0 + p.touch_above)) and (
                            float(low[j]) >= float(ma[j]) * (1.0 - p.max_under)
                        )
                        held = float(close[j]) >= float(ma[j]) * (1.0 - p.max_close_below)
                        near = float(close[j]) <= float(ma[j]) * (1.0 + p.max_entry_stretch)
                        if touched and held and near:
                            signals.append(
                                Signal(
                                    entry_idx=j,
                                    entry_price=float(close[j]),
                                    ma60=float(ma[j]),
                                    peak_idx=peak_idx,
                                    peak_high=peak_high,
                                    ext_pct=live_ext,
                                    retest_low=float(low[j]),
                                    run_start=run_start,
                                )
                            )
                            bump("entry")
                            ended = j + 1
                            break
            j += 1
        else:
            ended = n

        i = max(ended, run_start + 1)

    return signals


def _stop_price(sig: Signal, params: RetestParams) -> float:
    """停在回踩低／MA60 下方一點；風險太窄就拉到 min_risk_pct。"""
    entry = sig.entry_price
    stop = min(float(sig.retest_low), float(sig.ma60)) * 0.997
    min_stop = entry * (1.0 - params.min_risk_pct)
    if stop > min_stop:
        stop = min_stop
    if stop >= entry:
        stop = entry * (1.0 - params.min_risk_pct)
    return float(stop)


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    params: Optional[RetestParams] = None,
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
        stop = _stop_price(sig, p)
        risk = entry - stop
        if risk <= 0:
            stop = entry * (1.0 - p.min_risk_pct)
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


def _trade_window(df: pd.DataFrame, trade: TradeResult, pad_left: int = 24, pad_right: int = 10) -> tuple[int, int]:
    sig = trade.signal
    start = max(0, min(sig.run_start, sig.peak_idx, trade.entry_idx) - pad_left)
    end = min(len(df) - 1, max(trade.exit_idx, trade.entry_idx, sig.peak_idx) + pad_right)
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
        series = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.0 if n == 60 else 1.2
        ax.plot(list(xs), series, color=col, lw=lw, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.ma60, color="#3ddc68", ls="--", lw=0.7, alpha=0.55)

    px = sig.peak_idx - start
    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= px < len(window):
        ax.scatter([px], [sig.peak_high], s=34, color="#f0c14b", zorder=5)
        ax.annotate(
            "突破高",
            (px, sig.peak_high),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color="#f0c14b",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "回踩MA60",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(0, -14),
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
    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {trade.pnl_pct*100:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


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
        s = t.signal
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>延伸 {s.ext_pct*100:.1f}%</span>"
            f"<span class='tag'>距MA60 {(t.entry_price / s.ma60 - 1)*100:+.2f}%</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  stop {t.stop_price:.2f} (−{risk:.2f})\n"
            f"target {t.target_price:.2f}  exit {t.exit_price:.2f} {t.exit_reason}  {t.pnl_points:+.2f}\n"
            f"突破高 {s.peak_high:.2f} @ {df.index[s.peak_idx].strftime('%m-%d %H:%M')}  "
            f"MA60 {s.ma60:.2f}\n"
            f"fwd +1d {_fmt_fwd(t.fwd_1d)}  +3d {_fmt_fwd(t.fwd_3d)}  +5d {_fmt_fwd(t.fwd_5d)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    fun = funnel or {}
    fwd1 = _fmt_fwd(stats.get("fwd_1d"))
    fwd3 = _fmt_fwd(stats.get("fwd_3d"))
    fwd5 = _fmt_fwd(stats.get("fwd_5d"))
    reasons = stats.get("reasons") or {}
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 突破回踩 MA60</title>
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
<h1>台股 1h 突破後回踩 MA60 · 成交額前 {len(universe)}</h1>
<p class="muted">{escape(period)} · 基準日 {escape(date)} · {len(universe)} 檔 · 成交額末名約 {cutoff:.1f} 億
<br/>先在 MA60 上方延伸 ≥ 3%，再回踩均線附近（可刺穿 1.5%），收盤站上且離 MA60 ≤ 2.5%。MA60 不能下彎。同一段只吃第一次回踩。
回測出場：停在回踩低（至少 0.8%）、2R、或 20 根時間停。加總％是各筆報酬相加，不是組合複利。</p>
<p class="muted">漏斗（Yahoo 全區間，本週只留進場日）：站上 {fun.get('above_run', 0)} → 有延伸 {fun.get('extended', 0)} → 全區間進場 {fun.get('entry', 0)} → 本週 {stats['count']}
· 轉空 {fun.get('breakdown', 0)} · 回踩逾時 {fun.get('retest_timeout', 0)} · MA60 下彎 {fun.get('falling_ma', 0)}
· 延伸被均線追上 {fun.get('ext_faded', 0)}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}
· 收盤後 +1d {fwd1} · +3d {fwd3} · +5d {fwd5}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if stats['avg_pct']>=0 else 'pnl-loss'}">{stats['avg_pct']*100:+.2f}%</b></div>
<div class="card">標的<b>{len({h.row['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有突破回踩 MA60 訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def scan_symbol(
    row: dict,
    range_: str,
    params: RetestParams,
    days: int,
    funnel: Optional[Dict[str, int]] = None,
    max_price: Optional[float] = None,
) -> tuple[List[TwHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_yahoo_1h(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < params.ma_n + 10:
        meta["error"] = "too_few_bars"
        return [], meta
    local_fun: Dict[str, int] = {}
    sigs = detect_signals(df, params, funnel=local_fun)
    if funnel is not None:
        for k, v in local_fun.items():
            funnel[k] = funnel.get(k, 0) + v
    sigs = filter_entry_window(df, sigs, days)
    if max_price is not None:
        n_hi = sum(1 for s in sigs if s.entry_price >= max_price)
        if n_hi and funnel is not None:
            funnel["price_cap"] = funnel.get("price_cap", 0) + n_hi
        sigs = [s for s in sigs if s.entry_price < max_price]
    trades = simulate(df, sigs, params)
    meta["n_sig"] = len(sigs)
    meta["n_trade"] = len(trades)
    return [TwHit(row, t, df) for t in trades], meta


def dump_hits_json(path: Path, hits: List[TwHit], stats: dict, funnel: dict, extra: dict) -> Path:
    rows = []
    for hit in hits:
        t = hit.trade
        df = hit.df
        s = t.signal
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
                "ext_pct": s.ext_pct,
                "ma60": s.ma60,
                "peak_high": s.peak_high,
                "peak_time": str(df.index[s.peak_idx]),
                "retest_low": s.retest_low,
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


def _merge_watch(universe: List[dict]) -> List[dict]:
    have = {r["code"] for r in universe}
    extra: List[dict] = []
    for w in WATCH:
        if w["code"] in have:
            continue
        extra.append(
            {
                "rank": 0,
                "code": w["code"],
                "name": w["name"],
                "market": w["market"],
                "amount": 0,
                "close": None,
                "symbol": yahoo_symbol(w["code"], w["market"]),
                "watch": True,
            }
        )
    return extra + universe


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股 1h 突破後回踩 MA60 回測")
    p.add_argument("--date", default="", help="YYYYMMDD，預設上一個交易日")
    p.add_argument("--limit", type=int, default=200, help="成交額前 N；0 = 不限成交額")
    p.add_argument("--pool", type=int, default=400, help="先取成交額前 N 再套股價過濾")
    p.add_argument("--max-price", type=float, default=1000, help="股價達此值以上剔除；0 不過濾")
    p.add_argument("--days", type=int, default=7, help="只統計進場落在最近 N 日的訊號")
    p.add_argument("--range", dest="range_", default="2mo", help="Yahoo 1h 下載區間")
    p.add_argument("--sleep", type=float, default=0.18)
    p.add_argument("--no-watch", action="store_true", help="不強制納入對話那五檔")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--json", dest="json_path", default="")
    args = p.parse_args(argv)

    params = loose_params()
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    pool = 0 if args.limit <= 0 else max(args.limit, args.pool if args.max_price else args.limit)
    print(
        f"universe date={date} limit={args.limit} days={args.days} range={args.range_} "
        f"max_price={args.max_price}"
    )
    raw = fetch_top_turnover(date, pool)
    price_cap = None if args.max_price is None or args.max_price <= 0 else float(args.max_price)
    helper_cap = None if price_cap is None else price_cap - 1e-9
    universe, dropped = filter_by_max_price(raw, helper_cap, args.limit)
    args.max_price = price_cap
    if dropped:
        print(
            "drop price>="
            + str(args.max_price)
            + ": "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:12])
            + (" …" if len(dropped) > 12 else "")
        )
    if not args.no_watch:
        universe = _merge_watch(universe)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
        f"{(universe[0]['amount'] or 0)/1e8:.1f}億 / {universe[0]['close']} · "
        f"末 {universe[-1]['code']} {universe[-1]['amount']/1e8:.1f}億 / {universe[-1]['close']}"
    )

    hits: List[TwHit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    scanned = 0
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol(
            row, args.range_, params, args.days, funnel=funnel, max_price=price_cap
        )
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
        "range": args.range_,
        "limit": args.limit,
        "max_price": args.max_price,
        "generated": datetime.now(TPE).isoformat(timespec="seconds"),
        "branch": _git_branch(),
    }
    html_path = Path(args.html).resolve() if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        period_label = f"{args.days}d · Yahoo {args.range_} 1h"
        if args.max_price is not None:
            period_label += f" · 股價<{args.max_price:g}"
        out = write_tw_html(html_path, hits, universe, period_label, date, funnel=funnel)
        write_view_html(out)
        print(f"html={out}")
    json_path = Path(args.json_path).resolve() if args.json_path else None
    if json_path is None and html_path:
        json_path = html_path.with_name("hits.json")
    if json_path:
        dump_hits_json(json_path, hits, stats, funnel, extra)
        print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
