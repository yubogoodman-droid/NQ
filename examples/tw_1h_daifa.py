#!/usr/bin/env python3
"""台股 1h 達發多 — 開盤那根跳空、整根站上 MA60，且 MA5>MA10>MA20。

對齊 6526 達發 2026-09-18 開盤那根（券商圖常把 9:00–10:00 標成 10:00）：
  開 643 / 高 648 / 低 640 / 收 646。
  前一根（09-17 13:00）高 634、收 632，在 MA60 下面。
  這根低點 640 高於前高，也高於當根 MA60（約 638），缺口沒補、整根站上。
  同根 MA5 > MA10 > MA20。

進場用這根收盤（10:00 已知）。持有到之後第 5 個時段收盤；
盤中跌破進場價 10% 才停損。開盤那根低點太貼，下一根影線就會把勝率打掉。
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    REPO,
    TPE,
    _get_json,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)
from tw_1h_reclaim import (  # noqa: E402
    _fwd_pct,
    _session_close_indices,
    fetch_yahoo_1h,
    sma,
)

PAGES = REPO / "docs" / "tw-1h-daifa" / "index.html"
MA_COLORS = {5: "#f0c14b", 10: "#79c0ff", 20: "#f472b6", 60: "#e6edf3"}
PINNED = ("6526", "達發", "tse")


@dataclass(frozen=True)
class DaifaParams:
    stop_pct: float = 0.10
    hold_sessions: int = 5


def default_params(**overrides: object) -> DaifaParams:
    return DaifaParams(**overrides)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Signal:
    entry_idx: int
    entry_price: float
    open_price: float
    bar_low: float
    prev_close: float
    prev_high: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float

    @property
    def gap_pct(self) -> float:
        if self.prev_close <= 0:
            return 0.0
        return self.open_price / self.prev_close - 1.0


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: Optional[float]
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
    daily: Optional[pd.DataFrame] = None


def _session_open_indices(index: pd.DatetimeIndex) -> List[int]:
    seen: Dict[object, int] = {}
    for i, ts in enumerate(index):
        seen.setdefault(ts.date(), i)
    return [seen[d] for d in sorted(seen)]


def detect_signals(
    df: pd.DataFrame,
    params: Optional[DaifaParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """開盤 1h：跳空（低 > 前高）、前收在 MA60 下、整根低點站上 MA60、MA5>MA10>MA20。"""
    del params  # 進出場參數不影響認定
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    if df is None or len(df) < 61:
        return []

    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    signals: List[Signal] = []

    for i in _session_open_indices(df.index):
        if i < 60:
            continue
        bump("opens")
        prev = i - 1
        if df.index[prev].date() == df.index[i].date():
            bump("warmup")
            continue
        if any(np.isnan(x) for x in (ma5[i], ma10[i], ma20[i], ma60[i], close[prev], high[prev])):
            bump("warmup")
            continue
        # 跳空：開盤這根低點留在前一根高點之上，缺口沒補。
        if not (float(low[i]) > float(high[prev]) and float(open_[i]) > float(close[prev])):
            bump("no_gap")
            continue
        # 前收還在 MA60 下，這根才算跳空站上，不是本來就在上面又跳。
        if not float(close[prev]) < float(ma60[i]):
            bump("prev_above")
            continue
        # 整根站上：低點與收盤都在當根 MA60 之上（影線刺破不算）。
        if not (float(low[i]) > float(ma60[i]) and float(close[i]) > float(ma60[i])):
            bump("not_above_ma60")
            continue
        if not (float(ma5[i]) > float(ma10[i]) > float(ma20[i])):
            bump("no_stack")
            continue
        signals.append(
            Signal(
                entry_idx=i,
                entry_price=float(close[i]),
                open_price=float(open_[i]),
                bar_low=float(low[i]),
                prev_close=float(close[prev]),
                prev_high=float(high[prev]),
                ma5=float(ma5[i]),
                ma10=float(ma10[i]),
                ma20=float(ma20[i]),
                ma60=float(ma60[i]),
            )
        )
        bump("entry")
    return signals


def _hold_exit_index(ends: Sequence[int], entry_idx: int, sessions: int, n: int) -> tuple[int, bool]:
    """進場後第 N 個時段收盤。當天 13:00 算第 1 個。資料不夠就停在最後一根。"""
    after = [i for i in ends if i > entry_idx]
    if sessions > 0 and len(after) >= sessions:
        return after[sessions - 1], True
    return n - 1, False


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    params: Optional[DaifaParams] = None,
) -> List[TradeResult]:
    p = params or default_params()
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ends = _session_close_indices(df.index)
    n = len(close)
    trades: List[TradeResult] = []

    for sig in signals:
        entry_idx = sig.entry_idx
        entry = float(sig.entry_price)
        stop = entry * (1.0 - p.stop_pct)
        if stop >= entry:
            stop = entry * 0.99
        exit_idx = entry_idx
        exit_px = entry
        reason = "open"
        last, complete = _hold_exit_index(ends, entry_idx, p.hold_sessions, n)
        for k in range(entry_idx + 1, last + 1):
            if float(low[k]) <= stop:
                exit_idx, exit_px, reason = k, stop, "stop"
                break
        else:
            if last > entry_idx and complete:
                exit_idx, exit_px, reason = last, float(close[last]), "time"
            elif last > entry_idx:
                exit_idx, exit_px, reason = last, float(close[last]), "open"
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
                target_price=None,
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


def filter_entry_dates(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    start: str = "",
    end: str = "",
) -> List[Signal]:
    """只留進場日在 start～end（YYYY-MM-DD，含兩端）。空字串表示那一側不限。"""
    if df is None or not len(df) or (not start and not end):
        return list(signals)
    start_d = datetime.strptime(start, "%Y-%m-%d").date() if start else None
    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else None
    kept: List[Signal] = []
    for sig in signals:
        day = df.index[sig.entry_idx].date()
        if start_d is not None and day < start_d:
            continue
        if end_d is not None and day > end_d:
            continue
        kept.append(sig)
    return kept


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


def _trade_window(df: pd.DataFrame, trade: TradeResult, pad_left: int = 36, pad_right: int = 16) -> tuple[int, int]:
    start = max(0, trade.entry_idx - pad_left)
    end = min(len(df) - 1, max(trade.exit_idx, trade.entry_idx) + pad_right)
    return start, end


def fetch_yahoo_1d(symbol: str, range_: str = "2y") -> pd.DataFrame:
    """日 K。不套用分 K 的 09:00–13:30 遮罩。"""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={range_}&includePrePost=false"
    )
    payload = _get_json(url)
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    ts = result[0].get("timestamp") or []
    quote = result[0]["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(TPE),
    )
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df[~df.index.duplicated(keep="last")].sort_index()


def bars_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """沒有日 K 時，用 60 分 K 收成每天一根。"""
    if df is None or df.empty:
        return pd.DataFrame()
    grouped = df.groupby(df.index.date, sort=True)
    out = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum() if "Volume" in df.columns else 0,
        }
    )
    out.index = pd.to_datetime(out.index).tz_localize(TPE) + pd.Timedelta(hours=9)
    return out


def daily_bounds(
    daily: pd.DataFrame,
    entry_ts,
    exit_ts,
    left: int = 64,
    right: int = 10,
) -> Optional[tuple[int, int, int, int]]:
    """日 K 視窗：進場日前 left 根，出場日後 right 根。回傳 start, end, entry_i, exit_i。"""
    if daily is None or len(daily) == 0:
        return None
    entry_day = pd.Timestamp(entry_ts).date()
    exit_day = pd.Timestamp(exit_ts).date()
    dates = [ts.date() for ts in daily.index]
    entry_i = None
    exit_i = None
    for i, day in enumerate(dates):
        if day <= entry_day:
            entry_i = i
        if day <= exit_day:
            exit_i = i
    if entry_i is None:
        return None
    if exit_i is None or exit_i < entry_i:
        exit_i = entry_i
    start = max(0, entry_i - left)
    end = min(len(daily) - 1, max(exit_i, entry_i) + right)
    return start, end, entry_i, exit_i


def _style_ax(ax) -> None:
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")


def _paint_ohlc(ax, axv, window: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
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
    if vol is not None and axv is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)


def _plot_mas(ax, close_full: pd.Series, start: int, end: int) -> None:
    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(range(end - start + 1), ma, color=col, lw=1.15 if n < 60 else 1.35, label=f"MA{n}")


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    title_extra: str = "",
    daily: Optional[pd.DataFrame] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk()
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    daily_src = daily if daily is not None and len(daily) else bars_to_daily(df)
    bounds = daily_bounds(daily_src, et, xt)

    fig = plt.figure(figsize=(10.4, 8.4 if bounds else 5.6), facecolor="#0c1210")
    if bounds:
        gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 0.85, 2.5, 0.75], hspace=0.16)
        ax = fig.add_subplot(gs[0])
        axv = fig.add_subplot(gs[1], sharex=ax)
        axd = fig.add_subplot(gs[2])
        axdv = fig.add_subplot(gs[3], sharex=axd)
        panels = (ax, axv, axd, axdv)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        axv = fig.add_subplot(gs[1], sharex=ax)
        axd = axdv = None
        panels = (ax, axv)
    for a in panels:
        _style_ax(a)

    _paint_ohlc(ax, axv, window)
    _plot_mas(ax, df["Close"].astype(float), start, end)

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    if trade.target_price is not None:
        ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(trade.signal.ma60, color="#e6edf3", ls="--", lw=0.7, alpha=0.35)

    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "開盤",
            (ex, float(window["High"].iloc[ex])),
            textcoords="offset points",
            xytext=(0, 8),
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

    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {trade.pnl_pct * 100:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    ax.tick_params(labelbottom=False)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    axv.set_ylabel("60分", color="#8aa193", fontsize=8)

    if bounds and axd is not None and axdv is not None:
        d0, d1, entry_i, exit_i = bounds
        dwin = daily_src.iloc[d0 : d1 + 1]
        _paint_ohlc(axd, axdv, dwin)
        _plot_mas(axd, daily_src["Close"].astype(float), d0, d1)
        rel_entry = entry_i - d0
        rel_exit = exit_i - d0
        if 0 <= rel_entry < len(dwin):
            axd.axvline(rel_entry, color="#3dba7a", ls="--", lw=0.9)
            axd.scatter(
                [rel_entry],
                [float(dwin["Close"].iloc[rel_entry])],
                s=36,
                color="#00e676",
                marker="^",
                zorder=6,
            )
        if 0 <= rel_exit < len(dwin) and rel_exit != rel_entry:
            axd.axvline(rel_exit, color="#f0c14b", ls=":", lw=0.9)
        axd.tick_params(labelbottom=False)
        axd.set_ylabel("日線", color="#e8f0ea", fontsize=9)
        dstep = max(1, len(dwin) // 6)
        dticks = list(range(0, len(dwin), dstep))
        axdv.set_xticks(dticks)
        axdv.set_xticklabels([dwin.index[i].strftime("%m-%d") for i in dticks], color="#8aa193")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.05, hspace=0.22)
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
    src = src.resolve()
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _fmt_fwd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


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
        sig = t.signal
        img_name = f"t{i:02d}_{hit.row['code']}_{et.strftime('%m%d_%H%M')}.png"
        label = f"{hit.row['code']} {hit.row['name']}"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=label, daily=hit.daily)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct * 100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>跳空 {sig.gap_pct * 100:.1f}%</span>"
            f"<span class='tag'>MA60 {sig.ma60:.2f}</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  stop {t.stop_price:.2f} (−{risk:.2f}，跌破 10% 出場)\n"
            f"exit {t.exit_price:.2f} {t.exit_reason}  {t.pnl_points:+.2f}  持有到第 5 個收盤\n"
            f"開 {sig.open_price:.2f} 低 {sig.bar_low:.2f}  前高 {sig.prev_high:.2f} 前收 {sig.prev_close:.2f}\n"
            f"MA5 {sig.ma5:.2f} / MA10 {sig.ma10:.2f} / MA20 {sig.ma20:.2f} / MA60 {sig.ma60:.2f}\n"
            f"fwd +1d {_fmt_fwd(t.fwd_1d)}  +3d {_fmt_fwd(t.fwd_3d)}  +5d {_fmt_fwd(t.fwd_5d)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    ranked = [r for r in universe if not r.get("pinned")]
    pinned = [r for r in universe if r.get("pinned")]
    basis = ranked or list(universe)
    cutoff = basis[-1]["amount"] / 1e8 if basis else 0
    head_n = len(ranked) if ranked else len(universe)
    pin_txt = ""
    if pinned:
        pin_txt = "，另列 " + "、".join(f"{r['code']} {r['name']}" for r in pinned)
    fun = funnel or {}
    fwd1 = _fmt_fwd(stats.get("fwd_1d"))
    fwd3 = _fmt_fwd(stats.get("fwd_3d"))
    fwd5 = _fmt_fwd(stats.get("fwd_5d"))
    reasons = stats.get("reasons") or {}
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 達發多</title>
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
<h1>台股 1h 達發多 · 成交額前 {head_n}{escape(pin_txt)}</h1>
<p class="muted">{escape(period)} · 基準日 {escape(date)} · {head_n} 檔{escape(pin_txt)} · 成交額末名約 {cutoff:.1f} 億
<br/>開盤那根 60 分 K：低點高於前一根高點（跳空沒補），前收在 MA60 下，低點與收盤都站上 MA60，且 MA5 &gt; MA10 &gt; MA20。
進場為該根收盤。回測出場：持有到進場後第 5 個時段收盤（含當天），盤中跌破進場價 10% 停損。
不再把開盤那根低點當停損。加總％是各筆報酬相加，不是組合複利。
樣本對齊 6526 達發 2026-09-18 09:00（開 643 高 648 低 640 收 646）。</p>
<p class="muted">漏斗（整段下載；近窗筆數見下方）：開盤 {fun.get('opens', 0)} → 進場 {fun.get('entry', 0)}
· 沒跳空 {fun.get('no_gap', 0)} · 前收已在均線上 {fun.get('prev_above', 0)}
· 沒整根站上 {fun.get('not_above_ma60', 0)} · 沒有多頭排列 {fun.get('no_stack', 0)}
· 暖機 {fun.get('warmup', 0)} · 進場價過高 {fun.get('price_cap', 0)}
<br/>出場：持有到期 {reasons.get('time', 0)} · 跌破 10% {reasons.get('stop', 0)} · 未平 {reasons.get('open', 0)}
· 收盤後 +1d {fwd1} · +3d {fwd3} · +5d {fwd5}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if stats['avg_pct']>=0 else 'pnl-loss'}">{stats['avg_pct']*100:+.2f}%</b></div>
<div class="card">標的<b>{len({h.row['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有達發多訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _pct_stats(trades: Sequence[TradeResult]) -> dict:
    pcts = [float(t.pnl_pct) for t in trades]
    n = len(pcts)
    if not n:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "best": 0.0, "worst": 0.0}
    wins = sum(1 for p in pcts if p > 0)
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "avg": 100.0 * sum(pcts) / n,
        "med": 100.0 * float(np.median(pcts)),
        "best": 100.0 * max(pcts),
        "worst": 100.0 * min(pcts),
    }


def _png_data_uri(df: pd.DataFrame, trade: TradeResult, trade_no: int, label: str, daily) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        png_path = Path(handle.name)
    try:
        draw_trade_png(df, trade, png_path, trade_no, title_extra=label, daily=daily)
        encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    finally:
        png_path.unlink(missing_ok=True)
    return f"data:image/png;base64,{encoded}"


def write_embed_html(
    path: Path,
    hits: List[TwHit],
    universe: List[dict],
    period: str,
    date: str,
    sessions: Sequence[str],
    funnel: Optional[Dict[str, int]] = None,
) -> Path:
    """單一 HTML，K 線圖以 data URI 內嵌。報酬未扣手續費與證交稅。"""
    trades = [h.trade for h in hits]
    closed = [t for t in trades if t.exit_reason != "open"]
    opened = [t for t in trades if t.exit_reason == "open"]
    all_stats = _pct_stats(trades)
    closed_stats = _pct_stats(closed)

    def _extreme(pool: Sequence[TradeResult], pick) -> str:
        if not pool:
            return "—"
        chosen = max(pool, key=lambda t: pick(t.pnl_pct))
        hit = next(h for h in hits if h.trade is chosen)
        ts = hit.df.index[chosen.entry_idx]
        return (
            f"{hit.row['code']} {hit.row['name']} {ts.strftime('%m-%d %H:%M')} "
            f"{chosen.exit_reason} {chosen.pnl_pct * 100:+.2f}%"
        )

    best = _extreme(trades, lambda p: p)
    worst = _extreme(trades, lambda p: -p)
    best_c = _extreme(closed, lambda p: p) if closed else "—"
    worst_c = _extreme(closed, lambda p: -p) if closed else "—"
    sess = "、".join(sessions) if sessions else "（這段沒有任何 60 分 K）"
    fun = funnel or {}
    ranked = [r for r in universe if not r.get("pinned")]
    pinned = [r for r in universe if r.get("pinned")]
    head_n = len(ranked) if ranked else len(universe)
    pin_txt = ""
    if pinned:
        pin_txt = "，另列 " + "、".join(f"{r['code']} {r['name']}" for r in pinned)

    cards: List[str] = []
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        sig = t.signal
        label = f"{hit.row['code']} {hit.row['name']}"
        uri = _png_data_uri(df, t, i, label, hit.daily)
        note = "未滿 5 個收盤，這筆用最後一根收盤計，尚未到期。" if t.exit_reason == "open" else "已出場。"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct * 100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>跳空 {sig.gap_pct * 100:.1f}%</span>"
            f"<span class='tag'>MA60 {sig.ma60:.2f}</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {t.entry_price:.2f}（開盤 60 分 K 收盤）  停損 {t.stop_price:.2f}（−10%）\n"
            f"出場 {t.exit_price:.2f} {t.exit_reason}  {t.pnl_points:+.2f}  風險 {risk:.2f}\n"
            f"開 {sig.open_price:.2f} 低 {sig.bar_low:.2f}  前高 {sig.prev_high:.2f} 前收 {sig.prev_close:.2f}\n"
            f"MA5 {sig.ma5:.2f} / MA10 {sig.ma10:.2f} / MA20 {sig.ma20:.2f} / MA60 {sig.ma60:.2f}\n"
            f"{note} 未扣手續費與證交稅。"
            "</pre>"
            f"<div class='mini-chart'><img src='{uri}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 達發多 · 兩週回測</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:14px 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
ol{{margin:6px 0 0;padding-left:1.2em;color:#c9d1d9;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:18px;margin-top:4px}}
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
<h1>台股 1h 達發多 · {escape(period)}</h1>
<p class="muted">基準日 {escape(date)} · 成交額前 {head_n} 檔{escape(pin_txt)} · 股價達 1000 剔除。
只回測、沒有下單。報酬是價格報酬，未扣手續費與證交稅。</p>
<h2>觸發規則</h2>
<ol>
<li>週期：Yahoo 60 分 K。只留台北時間 09:00–13:30 的棒（含 13:00 那根，涵蓋到 13:30）。08:30 試搓、盤前集合競價沒有進這份資料。</li>
<li>標的：證交所當日成交額排名，先取前 400，刪掉收盤價大於等於 1000 或沒有收盤價的，再留前 200。6526 達發若不在名單裡會另列。這次排名日是 {escape(date)}。</li>
<li>進場只看每個有資料的交易日第一根 60 分 K（09:00，代表 09:00–10:00）。前面至少要有 60 根才算均線。休市日沒有棒，直接略過。</li>
<li>跳空：這根最低價高於前一根最高價，而且開盤價高於前一根收盤價。前一根必須是前一個交易日。</li>
<li>站上 MA60：前一根收盤小於當根 MA60；這根最低價與收盤價都大於當根 MA60。均線是收盤價的簡單移動平均，含當根。</li>
<li>多頭排列：MA5 &gt; MA10 &gt; MA20，三條都要嚴格大於。</li>
<li>進場價：這根收盤價。訊號在 10:00 這根走完才知道。</li>
<li>停損：進場價的 90%（跌 10%）。進場之後每一根，最低價小於等於停損價就出場，成交價用停損價，不是那根最低價。同一根先看停損。</li>
<li>停利：沒有。</li>
<li>時間出場：進場之後第 5 個時段收盤才出，用那根收盤價。當天 13:00 那根算第 1 個。資料還沒走到第 5 個收盤就標成未平，出場價用最後一根收盤。</li>
</ol>
<p class="muted">這段 Yahoo 60 分 K 有出現的交易日：{escape(sess)}。
10/17、10/18、10/24、10/25 是週末。10/23 若上面沒有列出，就是當天沒有正規盤棒（颱風休市）。</p>
<h2>統計</h2>
<p class="muted">全部訊號 {all_stats['n']} 筆（含未平 {len(opened)} 筆）。勝率以報酬 &gt; 0 計算，打平不算贏。
未平的報酬是最後一根收盤對進場價，不是已實現。</p>
<div class="cards">
<div class="card">訊號<b>{all_stats['n']}</b></div>
<div class="card">勝率<b>{all_stats['wr']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if all_stats['avg']>=0 else 'pnl-loss'}">{all_stats['avg']:+.2f}%</b></div>
<div class="card">中位數<b>{all_stats['med']:+.2f}%</b></div>
</div>
<p class="muted">最好：{escape(best)}<br/>最差：{escape(worst)}</p>
<p class="muted">只算已出場（停損或第 5 個收盤）{closed_stats['n']} 筆：勝率 {closed_stats['wr']:.1f}%、平均 {closed_stats['avg']:+.2f}%、中位數 {closed_stats['med']:+.2f}%。
最好：{escape(best_c)}。最差：{escape(worst_c)}。
漏斗是整段下載的認定次數，不是只算這兩週：開盤 {fun.get('opens', 0)} → 進場 {fun.get('entry', 0)}
· 沒跳空 {fun.get('no_gap', 0)} · 前收已在均線上 {fun.get('prev_above', 0)}
· 沒整根站上 {fun.get('not_above_ma60', 0)} · 沒有多頭排列 {fun.get('no_stack', 0)}</p>
</section>
{''.join(cards) or "<div class='empty'>這兩個日期之間沒有達發多訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def ensure_pinned(universe: List[dict], code: str, name: str, market: str) -> List[dict]:
    if any(str(r.get("code")) == code for r in universe):
        return universe
    row = {
        "rank": len(universe) + 1,
        "code": code,
        "name": name,
        "market": market,
        "amount": 0,
        "close": None,
        "symbol": yahoo_symbol(code, market),
        "pinned": True,
    }
    return [*universe, row]


def scan_symbol(
    row: dict,
    range_: str,
    params: DaifaParams,
    days: int,
    funnel: Optional[Dict[str, int]] = None,
    max_price: Optional[float] = None,
    start: str = "",
    end: str = "",
) -> tuple[List[TwHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_yahoo_1h(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 80:
        meta["error"] = "too_few_bars"
        return [], meta
    local_fun: Dict[str, int] = {}
    sigs = detect_signals(df, params, funnel=local_fun)
    if funnel is not None:
        for k, v in local_fun.items():
            funnel[k] = funnel.get(k, 0) + v
    if start or end:
        sigs = filter_entry_dates(df, sigs, start, end)
        if start and end:
            meta["sessions"] = sorted(
                {
                    ts.date().isoformat()
                    for ts in df.index
                    if datetime.strptime(start, "%Y-%m-%d").date()
                    <= ts.date()
                    <= datetime.strptime(end, "%Y-%m-%d").date()
                }
            )
    else:
        sigs = filter_entry_window(df, sigs, days)
    if max_price is not None:
        n_hi = sum(1 for s in sigs if s.entry_price >= max_price)
        if n_hi and funnel is not None:
            funnel["price_cap"] = funnel.get("price_cap", 0) + n_hi
        sigs = [s for s in sigs if s.entry_price < max_price]
    trades = simulate(df, sigs, params)
    meta["n_sig"] = len(sigs)
    meta["n_trade"] = len(trades)
    daily = None
    if trades:
        try:
            fetched = fetch_yahoo_1d(row["symbol"])
            if fetched is not None and len(fetched):
                daily = fetched
        except Exception:  # noqa: BLE001
            daily = None
    return [TwHit(row, t, df, daily) for t in trades], meta


def dump_hits_json(path: Path, hits: List[TwHit], stats: dict, funnel: dict, extra: dict) -> Path:
    rows = []
    for hit in hits:
        t = hit.trade
        df = hit.df
        sig = t.signal
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
                "open": sig.open_price,
                "bar_low": sig.bar_low,
                "prev_close": sig.prev_close,
                "prev_high": sig.prev_high,
                "gap_pct": sig.gap_pct,
                "ma5": sig.ma5,
                "ma10": sig.ma10,
                "ma20": sig.ma20,
                "ma60": sig.ma60,
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


def _row_from_symbol(symbol: str) -> dict:
    code = symbol.split(".")[0]
    market = "otc" if symbol.upper().endswith(".TWO") else "tse"
    return {
        "rank": 1,
        "code": code,
        "name": code,
        "market": market,
        "amount": 0,
        "close": None,
        "symbol": symbol,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股 1h 達發多：開盤跳空站上 MA60")
    p.add_argument("--symbol", default="", help="只掃一檔，例如 6526.TW")
    p.add_argument("--date", default="", help="YYYYMMDD，預設上一個交易日")
    p.add_argument("--limit", type=int, default=200, help="成交額前 N；0 = 不限成交額")
    p.add_argument("--pool", type=int, default=400, help="先取成交額前 N 再套股價過濾")
    p.add_argument("--max-price", type=float, default=1000, help="股價達此值以上剔除，預設 1000；0 不過濾")
    p.add_argument("--days", type=int, default=30, help="只統計進場落在最近 N 日的訊號")
    p.add_argument("--start", default="", help="進場日起 YYYY-MM-DD，搭配 --end 時不用 --days")
    p.add_argument("--end", default="", help="進場日迄 YYYY-MM-DD，含當天")
    p.add_argument("--embed", action="store_true", help="HTML 把 K 線圖內嵌成單一檔")
    p.add_argument("--range", dest="range_", default="3mo", help="Yahoo 1h 下載區間")
    p.add_argument("--sleep", type=float, default=0.18)
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--json", dest="json_path", default="")
    args = p.parse_args(argv)

    params = default_params()
    price_cap = None if args.max_price is None or args.max_price <= 0 else float(args.max_price)
    if args.symbol:
        universe = [_row_from_symbol(args.symbol.strip())]
        date = args.date or last_tw_session_yyyymmdd()
        print(f"symbol={args.symbol} days={args.days} range={args.range_}")
    else:
        date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
        pool = 0 if args.limit <= 0 else max(args.limit, args.pool if args.max_price else args.limit)
        print(
            f"universe date={date} limit={args.limit} days={args.days} range={args.range_} "
            f"max_price={args.max_price}"
        )
        raw = fetch_top_turnover(date, pool)
        helper_cap = None if price_cap is None else price_cap - 1e-9
        universe, dropped = filter_by_max_price(raw, helper_cap, args.limit)
        if dropped:
            print(
                "drop price>="
                + str(price_cap)
                + ": "
                + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:12])
                + (" …" if len(dropped) > 12 else "")
            )
        universe = ensure_pinned(universe, *PINNED)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
        f"{universe[0]['amount']/1e8:.1f}億"
    )

    hits: List[TwHit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    scanned = 0
    seen_sessions: set[str] = set()
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol(
            row,
            args.range_,
            params,
            args.days,
            funnel=funnel,
            max_price=price_cap,
            start=args.start,
            end=args.end,
        )
        scanned += 1
        if meta["error"]:
            errors += 1
        hits.extend(stock_hits)
        seen_sessions.update(meta.get("sessions") or [])
        flag = f" trades={meta['n_trade']}" if meta["n_trade"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row['name']} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, args.sleep))

    sessions = sorted(seen_sessions)
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
        "max_price": price_cap,
        "symbol": args.symbol,
        "generated": datetime.now(TPE).isoformat(timespec="seconds"),
    }
    html_path = Path(args.html).resolve() if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        if args.start and args.end:
            period_label = f"{args.start} → {args.end} · Yahoo {args.range_} 1h"
        else:
            period_label = f"{args.days}d · Yahoo {args.range_} 1h"
        if price_cap is not None:
            period_label += f" · 股價<{price_cap:g}"
        if args.embed:
            out = write_embed_html(
                html_path, hits, universe, period_label, date, sessions, funnel=funnel
            )
        else:
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
