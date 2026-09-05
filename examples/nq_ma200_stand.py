#!/usr/bin/env python3
"""NQ 一分 K：破兩小時低後站上 MA200。

現行規則（無五分 MA60 斜率）：
  1. MA5>10>20>30>60
  2. 站上 MA200 連 ≥3，且距離 ≤30
  3. 破兩小時低後 1 小時內
  4. 進場前 1 小時曾連續 ≥15 根在 MA200 下
  5. 美東 9:30–10:00 不進
  6. 紅 K 長上影跳過
  7. 停損 MA200−10，停利 +100
  8. 進場後五分 K 收盤跌破 MA21 提早出場（同根先看停損／停利）

用法:
  python3 examples/nq_ma200_stand.py backtest --period 30d --pages
  python3 examples/nq_ma200_stand.py backtest --period 8d --html output/nq_ma200_stand.html
"""

from __future__ import annotations

import argparse
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

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma200-stand" / "index.html"

TWO_HOUR_BARS = 120
RECLAIM_BARS = 60
UNDER_LOOKBACK = 60
UNDER_STREAK = 15
ABOVE_STREAK = 3
MAX_DIST_MA200 = 30.0
STOP_BELOW_MA200 = 10.0
TAKE_PROFIT = 100.0
MIN_UPPER_WICK = 8.0
MIN_5M_RIBBON = 0.0  # 關閉；五分圖只對照，不當進場條件
EARLY_EXIT_5M_MA = 21
EARLY_EXIT_5M_MA21 = True


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


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


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
            if isinstance(part.columns, pd.MultiIndex):
                part.columns = part.columns.get_level_values(0)
            part = part.rename(columns=str.title)
            chunks.append(part)
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(part)}", file=sys.stderr)
        else:
            print(f"[data] {cur.date()} → {nxt.date()} empty", file=sys.stderr)
        cur = nxt
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    days = parse_period_days(period)
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


def sma(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().to_numpy(float)


def rolling_min_prev(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).shift(1).rolling(n, min_periods=n).min().to_numpy(float)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    break_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    two_hr_low: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    ma200: float
    dist_ma200: float
    under_streak: int
    m5_ribbon: float = 0.0
    m1_ribbon: float = 0.0


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


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    pnls = [float(t.pnl_points) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "avg": float(sum(pnls) / n) if n else 0.0,
    }


def in_open_skip(ts) -> bool:
    """美東 9:30–10:00 不進（含 9:30，不含 10:00）。"""
    t = ts
    if getattr(t, "tzinfo", None) is None:
        t = t.replace(tzinfo=ET)
    else:
        t = t.astimezone(ET)
    minutes = t.hour * 60 + t.minute
    return 9 * 60 + 30 <= minutes < 10 * 60


def is_red_long_upper(o: float, h: float, l: float, c: float, min_wick: float = MIN_UPPER_WICK) -> bool:
    if c >= o:
        return False
    body = o - c
    upper = h - o
    return upper >= body and upper >= min_wick


def max_under_streak(close: np.ndarray, ma200: np.ndarray, end_idx: int, lookback: int) -> int:
    start = max(0, end_idx - lookback)
    run = 0
    best = 0
    for k in range(start, end_idx):
        if not np.isnan(ma200[k]) and close[k] < ma200[k]:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def above_ma200_streak(close: np.ndarray, ma200: np.ndarray, j: int, need: int) -> bool:
    if j < need - 1:
        return False
    for k in range(j - need + 1, j + 1):
        if np.isnan(ma200[k]) or close[k] <= ma200[k]:
            return False
    return True


def detect_signals(
    df: pd.DataFrame,
    *,
    two_hour_bars: int = TWO_HOUR_BARS,
    reclaim_bars: int = RECLAIM_BARS,
    under_lookback: int = UNDER_LOOKBACK,
    under_streak: int = UNDER_STREAK,
    above_streak: int = ABOVE_STREAK,
    max_dist_ma200: float = MAX_DIST_MA200,
    stop_below_ma200: float = STOP_BELOW_MA200,
    take_profit: float = TAKE_PROFIT,
    min_upper_wick: float = MIN_UPPER_WICK,
    min_5m_ribbon: float = MIN_5M_RIBBON,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    opn = df["Open"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    ma60 = sma(close, 60)
    ma200 = sma(close, 200)
    two_hr_low = rolling_min_prev(low, two_hour_bars)
    m5_ribbon = overlay_5m_ribbon(df)

    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    signals: List[Signal] = []
    n = len(close)
    warmup = max(200, two_hour_bars)
    i = warmup
    while i < n - 1:
        if np.isnan(two_hr_low[i]) or np.isnan(ma200[i]):
            i += 1
            continue
        support = float(two_hr_low[i])
        if low[i] >= support:
            i += 1
            continue

        bump("break")
        break_idx = i
        break_low = float(low[i])
        entered = False
        end_j = min(break_idx + reclaim_bars, n - 1)
        for j in range(break_idx + 1, end_j + 1):
            if np.isnan(ma5[j]) or np.isnan(ma60[j]) or np.isnan(ma200[j]):
                continue
            if not (ma5[j] > ma10[j] > ma20[j] > ma30[j] > ma60[j]):
                bump("skip_stack")
                continue
            if not above_ma200_streak(close, ma200, j, above_streak):
                bump("skip_above3")
                continue
            dist = float(close[j] - ma200[j])
            if dist <= 0 or dist > max_dist_ma200:
                bump("skip_dist")
                continue
            streak = max_under_streak(close, ma200, j, under_lookback)
            if streak < under_streak:
                bump("skip_under")
                continue
            if in_open_skip(df.index[j]):
                bump("skip_open")
                continue
            if is_red_long_upper(float(opn[j]), float(high[j]), float(low[j]), float(close[j]), min_upper_wick):
                bump("skip_wick")
                continue
            ribbon = float(m5_ribbon[j])
            ribbon_1m = float(ma5[j] - ma60[j])
            if min_5m_ribbon > 0 and (np.isnan(ribbon) or ribbon < min_5m_ribbon):
                bump("skip_5m_tangle")
                continue
            stop = float(ma200[j]) - stop_below_ma200
            entry = float(close[j])
            if entry <= stop:
                bump("skip_bad_stop")
                continue
            bump("taken")
            signals.append(
                Signal(
                    break_idx=break_idx,
                    entry_idx=j,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=entry + take_profit,
                    break_low=break_low,
                    two_hr_low=support,
                    ma5=float(ma5[j]),
                    ma10=float(ma10[j]),
                    ma20=float(ma20[j]),
                    ma30=float(ma30[j]),
                    ma60=float(ma60[j]),
                    ma200=float(ma200[j]),
                    dist_ma200=dist,
                    under_streak=streak,
                    m5_ribbon=0.0 if (np.isnan(ribbon) or np.isinf(ribbon)) else ribbon,
                    m1_ribbon=0.0 if np.isnan(ribbon_1m) else ribbon_1m,
                )
            )
            entered = True
            i = j + 1
            break
        if not entered:
            i = break_idx + 1
    return signals


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    *,
    early_exit_5m_ma21: bool = EARLY_EXIT_5M_MA21,
) -> List[TradeResult]:
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    opn = df["Open"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    n = len(close)
    ma21_exit = five_min_ma_exit_flags(df) if early_exit_5m_ma21 else np.zeros(n, dtype=bool)
    results: List[TradeResult] = []
    busy_until = -1
    for sig in signals:
        if sig.entry_idx <= busy_until:
            continue
        if sig.entry_idx >= n - 1:
            continue
        stop = sig.stop_price
        target = sig.target_price
        exit_idx = n - 1
        exit_price = float(close[-1])
        exit_reason = "eod"
        for k in range(sig.entry_idx + 1, n):
            o, h, l = float(opn[k]), float(high[k]), float(low[k])
            if o <= stop:
                exit_idx, exit_price, exit_reason = k, o, "stop"
                break
            if o >= target:
                exit_idx, exit_price, exit_reason = k, o, "target"
                break
            hit_sl = l <= stop
            hit_tp = h >= target
            if hit_sl and hit_tp:
                exit_idx, exit_price, exit_reason = k, stop, "stop"
                break
            if hit_sl:
                exit_idx, exit_price, exit_reason = k, stop, "stop"
                break
            if hit_tp:
                exit_idx, exit_price, exit_reason = k, target, "target"
                break
            if ma21_exit[k]:
                exit_idx, exit_price, exit_reason = k, float(close[k]), "ma21_5m"
                break
        pnl = float(exit_price - sig.entry_price)
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=sig.entry_idx,
                exit_idx=exit_idx,
                entry_price=sig.entry_price,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=pnl,
                exit_reason=exit_reason,
            )
        )
        busy_until = exit_idx
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


def ribbon_spread(*values: float) -> float:
    """均線帶寬：max − min。缺值回 nan。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(np.isnan(arr)):
        return float("nan")
    return float(arr.max() - arr.min())


def ribbon_tangled(*values: float, min_spread: float) -> bool:
    spread = ribbon_spread(*values)
    return bool(np.isnan(spread) or spread < min_spread)


def align_htf(df_1m: pd.DataFrame, series_htf: pd.Series) -> np.ndarray:
    """把已收盤的高週期序列對齊到 1m（不偷看未收的那根）。"""
    idx = series_htf.index
    vals = series_htf.to_numpy(float)
    out = np.full(len(df_1m), np.nan, dtype=float)
    j = 0
    for i, ts in enumerate(df_1m.index):
        while j + 1 < len(idx) and idx[j + 1] <= ts:
            j += 1
        if j < len(idx) and idx[j] <= ts:
            out[i] = vals[j]
    return out


def overlay_5m_ribbon(df_1m: pd.DataFrame) -> np.ndarray:
    """已收盤五分 MA5/10/20/30 帶寬。不把 MA60 算進去：第一張短均只差 13，MA60 會把帶寬撐到 40。"""
    df5 = resample_5m(df_1m)
    close5 = df5["Close"].astype(float)
    mas = [align_htf(df_1m, close5.rolling(n, min_periods=n).mean()) for n in (5, 10, 20, 30)]
    stacked = np.vstack(mas)
    with np.errstate(invalid="ignore"):
        spread = stacked.max(axis=0) - stacked.min(axis=0)
    spread[np.isnan(stacked).any(axis=0)] = np.nan
    return spread


def resample_htf(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """1m → N 分 OHLC（右標、右閉，不偷看未收的那根）。"""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample(f"{int(minutes)}min", label="right", closed="right").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    return resample_htf(df, 5)


def resample_15m(df: pd.DataFrame) -> pd.DataFrame:
    return resample_htf(df, 15)


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    return resample_htf(df, 60)


def five_min_ma_exit_flags(
    df: pd.DataFrame,
    ma_n: int = EARLY_EXIT_5M_MA,
) -> np.ndarray:
    """1m 對齊：該根剛好是五分 K 收盤，且收盤跌破五分 MA。"""
    flags = np.zeros(len(df), dtype=bool)
    if df.empty:
        return flags
    df5 = resample_5m(df)
    if df5.empty:
        return flags
    ma = sma(df5["Close"].to_numpy(float), ma_n)
    for ts, close5, ma5 in zip(df5.index, df5["Close"].to_numpy(float), ma):
        if np.isnan(ma5) or close5 >= ma5:
            continue
        pos = int(df.index.searchsorted(ts, side="right") - 1)
        if 0 <= pos < len(df):
            flags[pos] = True
    return flags


def bar_index_at(df: pd.DataFrame, ts) -> Optional[int]:
    if df.empty:
        return None
    pos = int(df.index.searchsorted(ts, side="left"))
    if pos >= len(df):
        return len(df) - 1
    return pos


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, trade.signal.break_idx - 40)
    end = min(len(df) - 1, trade.exit_idx + 20)
    return start, end


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
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
    return plt


def _paint_ohlc(
    ax,
    window: pd.DataFrame,
    close_full: pd.Series,
    start: int,
    end: int,
    trade: TradeResult,
    marks: dict,
    title: str,
) -> None:
    from matplotlib.patches import Rectangle

    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")

    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 20 else 1.05, label=f"MA{n}")
    extra_mas = marks.get("extra_mas") or {}
    for n, col in extra_mas.items():
        ma = close_full.rolling(int(n), min_periods=int(n)).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.45, ls="--", label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(trade.signal.two_hr_low, color="#8aa193", ls="--", lw=0.85, alpha=0.55)

    bx = marks.get("break")
    ex = marks.get("entry")
    xx = marks.get("exit")
    if bx is not None and 0 <= bx < len(window):
        ax.scatter([bx], [trade.signal.break_low], s=38, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, trade.signal.break_low),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if ex is not None and 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
    if xx is not None and 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )
    ax.set_title(title, color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    plt = _setup_mpl()
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    fig, ax = plt.subplots(figsize=(10.4, 4.8), facecolor="#0c1210")
    _paint_ohlc(
        ax,
        window,
        df["Close"].astype(float),
        start,
        end,
        trade,
        {
            "break": trade.signal.break_idx - start,
            "entry": trade.entry_idx - start,
            "exit": trade.exit_idx - start,
        },
        f"#{trade_no}  1m  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_htf_png(
    df_1m: pd.DataFrame,
    df_htf: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    *,
    label: str,
    lookback: int,
    lookforward: int,
) -> Optional[Path]:
    if df_htf.empty:
        return None
    plt = _setup_mpl()
    br = bar_index_at(df_htf, df_1m.index[trade.signal.break_idx])
    en = bar_index_at(df_htf, df_1m.index[trade.entry_idx])
    ex = bar_index_at(df_htf, df_1m.index[trade.exit_idx])
    if en is None:
        return None
    start = max(0, (br if br is not None else en) - lookback)
    end = min(len(df_htf) - 1, (ex if ex is not None else en) + lookforward)
    if end <= start:
        return None
    window = df_htf.iloc[start : end + 1]
    et = df_1m.index[trade.entry_idx]
    xt = df_1m.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    fig, ax = plt.subplots(figsize=(10.4, 4.8), facecolor="#0c1210")
    _paint_ohlc(
        ax,
        window,
        df_htf["Close"].astype(float),
        start,
        end,
        trade,
        {
            "break": None if br is None else br - start,
            "entry": en - start,
            "exit": None if ex is None else ex - start,
            "extra_mas": {21: "#e8eef2"} if label.startswith("5m") else {},
        },
        f"#{trade_no}  {label}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_5m_png(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Optional[Path]:
    return draw_htf_png(df_1m, df_5m, trade, path, trade_no, label="5m 對照", lookback=48, lookforward=8)


def draw_15m_png(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Optional[Path]:
    return draw_htf_png(df_1m, df_15m, trade, path, trade_no, label="15m 對照", lookback=32, lookforward=6)


def draw_1h_png(
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Optional[Path]:
    return draw_htf_png(df_1m, df_1h, trade, path, trade_no, label="1h 對照", lookback=36, lookforward=4)


def display_trades(trades: Sequence[TradeResult]) -> List[TradeResult]:
    """報告排版：賺錢在前，賠錢在後；同組內照進場時間。"""
    return sorted(trades, key=lambda t: (t.pnl_points <= 0, t.entry_idx))


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
    out = Path(path)
    df5 = resample_5m(df)
    df15 = resample_15m(df)
    df1h = resample_1h(df)
    cards: List[str] = []
    shown_loss = False
    n_win = sum(1 for t in trades if t.pnl_points > 0)
    n_loss = sum(1 for t in trades if t.pnl_points <= 0)
    for i, t in enumerate(display_trades(trades), 1):
        if i == 1 and n_win:
            cards.append("<h2 class='section'>賺錢</h2>")
        if t.pnl_points <= 0 and not shown_loss and n_loss:
            cards.append("<h2 class='section'>賠錢</h2>")
            shown_loss = True
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        if t.exit_reason == "target":
            reason_cls = "tag-tp"
        elif t.exit_reason == "stop":
            reason_cls = "tag-sl"
        elif t.exit_reason == "ma21_5m":
            reason_cls = "tag-early"
        else:
            reason_cls = "tag-time"
        img_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}.png"
        img5_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}_5m.png"
        img15_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}_15m.png"
        img1h_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}_1h.png"
        draw_trade_png(df, t, out.parent / "img" / img_name, i)
        png5 = draw_5m_png(df, df5, t, out.parent / "img" / img5_name, i)
        png15 = draw_15m_png(df, df15, t, out.parent / "img" / img15_name, i)
        png1h = draw_1h_png(df, df1h, t, out.parent / "img" / img1h_name, i)
        charts = (
            f"<div class='mini-chart'><div class='chart-label'>1m</div>"
            f"<img src='img/{escape(img_name)}' alt='#{i} 1m' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
        )
        if png5 is not None:
            charts += (
                f"<div class='mini-chart'><div class='chart-label'>5m 對照</div>"
                f"<img src='img/{escape(img5_name)}' alt='#{i} 5m' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        if png15 is not None:
            charts += (
                f"<div class='mini-chart'><div class='chart-label'>15m 對照</div>"
                f"<img src='img/{escape(img15_name)}' alt='#{i} 15m' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        if png1h is not None:
            charts += (
                f"<div class='mini-chart'><div class='chart-label'>1h 對照</div>"
                f"<img src='img/{escape(img1h_name)}' alt='#{i} 1h' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        risk = t.entry_price - t.stop_price
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            "<span class='tag tag-info'>1m</span>"
            "<span class='tag tag-info'>5m 對照</span>"
            "<span class='tag tag-info'>15m 對照</span>"
            "<span class='tag tag-info'>1h 對照</span>"
            f"<span class='tag tag-info'>距MA200 {t.signal.dist_ma200:.1f}</span>"
            f"<span class='tag tag-info'>5m帶寬 {t.signal.m5_ribbon:.1f}</span>"
            f"<span class='tag tag-info'>1m帶寬 {t.signal.m1_ribbon:.1f}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts, MA200−10)\n"
            f"target {t.target_price:.2f}  (+100)\n"
            f"提早  五分收盤跌破 MA21\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破底 {t.signal.break_low:.2f} / 2h低 {t.signal.two_hr_low:.2f}\n"
            f"MA5 {t.signal.ma5:.1f} > MA10 {t.signal.ma10:.1f} > MA20 {t.signal.ma20:.1f} "
            f"> MA30 {t.signal.ma30:.1f} > MA60 {t.signal.ma60:.1f}\n"
            f"MA200 {t.signal.ma200:.1f}  先前連{t.signal.under_streak}根在下\n"
            f"5m MA5–30 帶寬 {t.signal.m5_ribbon:.1f} · 1m MA5–60 帶寬 {t.signal.m1_ribbon:.1f}"
            "</pre>"
            f"{charts}"
            "</article>"
        )

    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破底 {funnel.get('break', 0)} → 進場 {funnel.get('taken', 0)}"
            f"（排列 {funnel.get('skip_stack', 0)} · 未連3 {funnel.get('skip_above3', 0)} · "
            f"距離 {funnel.get('skip_dist', 0)} · 未洗15 {funnel.get('skip_under', 0)} · "
            f"9:30檔 {funnel.get('skip_open', 0)} · 長上影 {funnel.get('skip_wick', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 破底站上 MA200</title>
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
.tag-early{{background:rgba(121,192,255,0.14);color:#79c0ff;border-color:rgba(121,192,255,0.35)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px 8px;border-radius:10px;overflow:hidden}}
.mini-chart:last-child{{margin-bottom:-4px}}
.chart-label{{font-size:11px;color:#8b949e;font-weight:600;padding:8px 10px 4px}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
h2.section{{font-size:15px;margin:18px 0 10px;color:#e6edf3}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 破底站上 MA200</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">MA5&gt;10&gt;20&gt;30&gt;60 · 站上MA200連3且距≤30 · 破2h低後1小時 · 先前連15根在MA200下 · 9:30–10:00不進 · 紅K長上影跳過 · SL=MA200−10 / TP=+100 · 五分收盤跌破MA21提早出 · 5m / 15m / 1h 圖只對照</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">基準對照：一個月 55 筆 / 約 +1301（視 Yahoo 1m 窗口而定）</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{''.join(cards) or "<div class='empty'>無交易</div>"}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    view = out.parent / "view.html"
    if out.name == "index.html":
        view.write_text(html.replace("img/", "./img/"), encoding="utf-8")
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
    print(
        "funnel "
        f"break={funnel.get('break', 0)} taken={funnel.get('taken', 0)} "
        f"stack={funnel.get('skip_stack', 0)} above3={funnel.get('skip_above3', 0)} "
        f"dist={funnel.get('skip_dist', 0)} under={funnel.get('skip_under', 0)} "
        f"open={funnel.get('skip_open', 0)} wick={funnel.get('skip_wick', 0)}"
    )
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NQ 一分K 破底站上 MA200")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="30d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true")
    b.set_defaults(func=cmd_backtest)
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="30d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    args = p.parse_args(argv)
    if args.cmd is None or args.cmd == "backtest":
        return cmd_backtest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
