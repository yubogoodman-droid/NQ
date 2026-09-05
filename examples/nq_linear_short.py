#!/usr/bin/env python3
"""NQ 一分 K 線性空 — 對齊 pinescript/nq_linear_short_1m.pine。

多頭排列創四小時高 → 回測 MA10 不過高 → 收盤跌破 MA20，
且峰距 MA200 ≥ 100 點，做空。停損在四小時高；
停利碰到 MA200，或靠近 MA200 出現長下影。

用法:
  python3 examples/nq_linear_short.py
  python3 examples/nq_linear_short.py --period 7d --html output/nq_linear_short.html
  python3 examples/nq_linear_short.py --period 7d --pages
"""

from __future__ import annotations

import argparse
import base64
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
PAGES_DIR = REPO_ROOT / "docs" / "nq-linear-short"
POINT_VALUE = 20.0
COMMISSION_RT = 5.0  # $2.5 / side

MA_COLORS = {
    5: "#f0b429",
    10: "#ff7a00",
    20: "#e63946",
    30: "#00b4d8",
    60: "#9b5de5",
    120: "#2a9d8f",
    200: "#264653",
}


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


def _flatten_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    rename = {c: str(c).title() for c in df.columns}
    out = df.rename(columns=rename)
    need = {"Open", "High", "Low", "Close"}
    missing = need - set(out.columns)
    if missing:
        lower = {str(c).lower(): c for c in df.columns}
        for name in list(missing):
            src = lower.get(name.lower())
            if src is not None:
                out[name] = df[src]
                missing.discard(name)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    return out[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "7d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return _flatten_ohlc(df).dropna()


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
            chunks.append(_flatten_ohlc(part))
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


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    work = df[["Open", "High", "Low", "Close"]].copy()
    work["Volume"] = df["Volume"] if "Volume" in df.columns else 0.0
    out = work.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    """把 1m 合成 1h（bar 起點對齊整點，不含未收盤小時）。"""
    return resample_ohlc(df, "1h")


def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    """把 1m 合成 5m。"""
    return resample_ohlc(df, "5min")


def map_closed_htf(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    column: str,
    bar_delta: pd.Timedelta,
) -> np.ndarray:
    """每個低週期 bar 對到已經收盤的最近一根高週期，不看未來。"""
    out = np.full(len(df), np.nan, dtype=float)
    if df.empty or df_htf is None or df_htf.empty or column not in df_htf.columns:
        return out
    ends = df_htf.index + bar_delta
    vals = df_htf[column].to_numpy(float)
    j = 0
    n_h = len(ends)
    for i, ts in enumerate(df.index):
        while j + 1 < n_h and ends[j + 1] <= ts:
            j += 1
        if j < n_h and ends[j] <= ts:
            out[i] = vals[j]
    return out


def map_closed_1h(df: pd.DataFrame, df_1h: pd.DataFrame, column: str = "Close") -> np.ndarray:
    return map_closed_htf(df, df_1h, column, pd.Timedelta(hours=1))


def add_mas(df_htf: pd.DataFrame, periods: Sequence[int] = (5, 10, 20, 60, 200)) -> pd.DataFrame:
    out = df_htf.copy()
    close = out["Close"].astype(float)
    for n in periods:
        out[f"ma{n}"] = close.rolling(n, min_periods=n).mean()
    return out


def add_1h_mas(df_1h: pd.DataFrame, periods: Sequence[int] = (5, 10, 20, 60, 200)) -> pd.DataFrame:
    return add_mas(df_1h, periods)


# ---------------------------------------------------------------------------
# Strategy (bar-by-bar, matches Pine process_orders_on_close)
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    entry_idx: int
    entry_price: float
    stop_price: float
    peak_idx: int
    peak_high: float
    peak_ma200: float
    retest_idx: int
    bounce_hi: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    ma120: float
    ma200: float
    h1_close: float = float("nan")
    h1_ma20: float = float("nan")

    @property
    def peak_dist(self) -> float:
        return self.peak_high - self.peak_ma200

    @property
    def risk(self) -> float:
        return self.stop_price - self.entry_price


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    pnl_points: float
    exit_reason: str


@dataclass
class LinearShortParams:
    ma5: int = 5
    ma10: int = 10
    ma20: int = 20
    ma30: int = 30
    ma60: int = 60
    ma120: int = 120
    ma200: int = 200
    lookback4h: int = 240
    stop_buf: float = 0.0
    min_peak_dist: float = 100.0
    near_ma200: float = 15.0
    min_lower_wick: float = 3.0
    wick_body_ratio: float = 3.0
    wick_range_ratio: float = 0.55


def long_lower_wick(
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    min_lower_wick: float = 3.0,
    wick_body_ratio: float = 3.0,
    wick_range_ratio: float = 0.55,
) -> bool:
    body = max(abs(close - open_), 0.25)
    body_bot = min(open_, close)
    lower = body_bot - low
    bar_range = max(high - low, 0.25)
    return lower >= min_lower_wick and (
        lower >= wick_body_ratio * body or lower / bar_range >= wick_range_ratio
    )


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    pnls = [float(t.pnl_points) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_reason: Dict[str, List[float]] = {}
    for t in trades:
        by_reason.setdefault(t.exit_reason, []).append(float(t.pnl_points))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "pnl": float(sum(pnls)),
        "n": n,
        "by_reason": {
            r: {"n": len(v), "wins": sum(1 for p in v if p > 0), "pnl": float(sum(v))}
            for r, v in sorted(by_reason.items())
        },
    }


def run_linear_short(
    df: pd.DataFrame,
    params: LinearShortParams | None = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[TradeResult]:
    """逐 K 狀態機，對齊 Pine：process_orders_on_close、進場當根不檢查出場。"""
    p = params or LinearShortParams()
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(c)
    if n == 0:
        return []

    ma5 = sma(c, p.ma5)
    ma10 = sma(c, p.ma10)
    ma20 = sma(c, p.ma20)
    ma30 = sma(c, p.ma30)
    ma60 = sma(c, p.ma60)
    ma120 = sma(c, p.ma120)
    ma200 = sma(c, p.ma200)
    high4h = pd.Series(h).rolling(p.lookback4h, min_periods=p.lookback4h).max().to_numpy(float)
    df_1h = add_1h_mas(resample_1h(df))
    h1_close = map_closed_1h(df, df_1h, "Close")
    h1_ma20 = map_closed_1h(df, df_1h, "ma20")

    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    def bull_stack(i: int) -> bool:
        vals = (ma5[i], ma10[i], ma20[i], ma30[i], ma60[i], ma120[i], ma200[i])
        if any(np.isnan(v) for v in vals):
            return False
        return vals[0] > vals[1] > vals[2] > vals[3] > vals[4] > vals[5] > vals[6]

    def made_4h_high(i: int) -> bool:
        if i < 1 or np.isnan(high4h[i]):
            return False
        return bool(h[i] == high4h[i] and h[i] > h[i - 1])

    def reset_setup() -> tuple:
        return 0, float("nan"), float("nan"), float("nan"), -1, -1

    st = 0
    peak_high = float("nan")
    peak_ma200 = float("nan")
    bounce_hi = float("nan")
    peak_idx = -1
    retest_idx = -1
    position = 0
    just_closed = False
    open_sig: Optional[Signal] = None
    trades: List[TradeResult] = []
    warmup = max(p.ma200, p.lookback4h)

    for i in range(warmup, n):
        if just_closed:
            st, peak_high, peak_ma200, bounce_hi, peak_idx, retest_idx = reset_setup()
            just_closed = False
            continue

        if position == 0:
            if st == 0:
                if made_4h_high(i) and bull_stack(i):
                    st = 1
                    peak_high = float(h[i])
                    peak_ma200 = float(ma200[i])
                    bounce_hi = float("nan")
                    peak_idx = i
                    retest_idx = -1
                    bump("made_high")
            elif st == 1:
                if made_4h_high(i) and bull_stack(i) and h[i] > peak_high:
                    peak_high = float(h[i])
                    peak_ma200 = float(ma200[i])
                    peak_idx = i
                    bump("higher_high")
                if l[i] <= ma10[i]:
                    st = 2
                    bounce_hi = float(h[i])
                    retest_idx = i
                    bump("retest")
                if c[i] < ma60[i] and not bull_stack(i):
                    bump("cancel_stack")
                    st, peak_high, peak_ma200, bounce_hi, peak_idx, retest_idx = reset_setup()
            elif st == 2:
                bounce_hi = float(max(bounce_hi if not np.isnan(bounce_hi) else h[i], h[i]))
                if h[i] > peak_high:
                    bump("cancel_new_high")
                    st, peak_high, peak_ma200, bounce_hi, peak_idx, retest_idx = reset_setup()
                else:
                    broke = (
                        i >= 1
                        and not np.isnan(ma20[i])
                        and not np.isnan(ma20[i - 1])
                        and c[i - 1] >= ma20[i - 1]
                        and c[i] < ma20[i]
                    )
                    if broke:
                        bump("break_ma20")
                        peak_dist_ok = (
                            not np.isnan(peak_high)
                            and not np.isnan(peak_ma200)
                            and (peak_high - peak_ma200) >= p.min_peak_dist
                        )
                        if peak_dist_ok:
                            stop = float(peak_high + p.stop_buf)
                            sig = Signal(
                                entry_idx=i,
                                entry_price=float(c[i]),
                                stop_price=stop,
                                peak_idx=peak_idx,
                                peak_high=float(peak_high),
                                peak_ma200=float(peak_ma200),
                                retest_idx=retest_idx,
                                bounce_hi=float(bounce_hi),
                                ma5=float(ma5[i]),
                                ma10=float(ma10[i]),
                                ma20=float(ma20[i]),
                                ma30=float(ma30[i]),
                                ma60=float(ma60[i]),
                                ma120=float(ma120[i]),
                                ma200=float(ma200[i]),
                                h1_close=float(h1_close[i]),
                                h1_ma20=float(h1_ma20[i]),
                            )
                            if sig.risk > 0:
                                position = -1
                                open_sig = sig
                                st = 3
                                bump("taken")
                            else:
                                bump("skip_bad_risk")
                                st, peak_high, peak_ma200, bounce_hi, peak_idx, retest_idx = reset_setup()
                        else:
                            bump("skip_dist")
                            st, peak_high, peak_ma200, bounce_hi, peak_idx, retest_idx = reset_setup()
            continue

        # 持倉中：進場當根不檢查（對齊 Pine position_size 仍為 0）
        assert open_sig is not None
        stop = open_sig.stop_price
        reason = ""
        exit_price = float(c[i])
        if h[i] >= stop:
            reason = "stop"
            exit_price = float(stop)
        elif l[i] <= ma200[i]:
            reason = "ma200"
            exit_price = float(c[i])
        elif (l[i] >= ma200[i] and (l[i] - ma200[i]) <= p.near_ma200) and long_lower_wick(
            o[i], h[i], l[i], c[i],
            min_lower_wick=p.min_lower_wick,
            wick_body_ratio=p.wick_body_ratio,
            wick_range_ratio=p.wick_range_ratio,
        ):
            reason = "wick"
            exit_price = float(c[i])

        if not reason:
            continue

        bump(f"exit_{reason}")
        trades.append(
            TradeResult(
                signal=open_sig,
                entry_idx=open_sig.entry_idx,
                exit_idx=i,
                entry_price=open_sig.entry_price,
                exit_price=exit_price,
                stop_price=stop,
                pnl_points=float(open_sig.entry_price - exit_price),
                exit_reason=reason,
            )
        )
        position = 0
        open_sig = None
        just_closed = True

    if position < 0 and open_sig is not None:
        bump("exit_eod")
        trades.append(
            TradeResult(
                signal=open_sig,
                entry_idx=open_sig.entry_idx,
                exit_idx=n - 1,
                entry_price=open_sig.entry_price,
                exit_price=float(c[-1]),
                stop_price=open_sig.stop_price,
                pnl_points=float(open_sig.entry_price - c[-1]),
                exit_reason="eod",
            )
        )
    return trades


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


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    sig = trade.signal
    start = max(0, min(sig.peak_idx, sig.retest_idx if sig.retest_idx >= 0 else sig.peak_idx) - 40)
    end = min(len(df) - 1, trade.exit_idx + 16)
    return start, end


def _setup_mpl() -> None:
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


def _style_axes(axes) -> None:
    for a in axes:
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")


def _paint_candles(ax, axv, window: pd.DataFrame, min_slots: int = 36) -> None:
    """固定實體寬度；K 數太少時左右留白，避免被拉成大方塊。"""
    from matplotlib.patches import Rectangle

    n = len(window)
    xs = range(n)
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    colors_v = []
    for k in range(n):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.7)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if axv is not None and vol is not None and n:
        axv.bar(list(xs), vol.astype(float), width=0.7, color=colors_v, linewidth=0)
    span = max(n, min_slots)
    extra = span - n
    left_pad = extra * 0.35
    ax.set_xlim(-0.7 - left_pad, n - 0.3 + extra - left_pad)
    if axv is not None:
        axv.set_xlim(-0.7 - left_pad, n - 0.3 + extra - left_pad)


def _htf_index(window: pd.DataFrame, ts) -> int:
    if window.empty:
        return -1
    i = int(window.index.searchsorted(ts, side="right") - 1)
    if i < 0 or i >= len(window):
        return -1
    return i


def _m5_window(df_5m: pd.DataFrame, peak_ts, entry_ts, exit_ts) -> pd.DataFrame:
    """只切這筆結構附近，不要把盤前扁平 K 跟遠處均線一起拉進來。"""
    if df_5m.empty:
        return df_5m
    left = min(peak_ts, entry_ts) - pd.Timedelta(minutes=170)
    right = exit_ts + pd.Timedelta(minutes=30)
    w = df_5m.loc[(df_5m.index >= left) & (df_5m.index <= right)]
    if len(w) < 24:
        w = df_5m.loc[
            (df_5m.index >= entry_ts - pd.Timedelta(hours=3))
            & (df_5m.index <= exit_ts + pd.Timedelta(minutes=40))
        ]
    return w


def _fit_price_ylim(ax, window: pd.DataFrame, extra: Sequence[float] = ()) -> tuple[float, float]:
    lo = float(window["Low"].min())
    hi = float(window["High"].max())
    for v in extra:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        if lo - 90 <= fv <= hi + 90:
            lo = min(lo, fv)
            hi = max(hi, fv)
    pad = max(3.0, (hi - lo) * 0.10)
    ax.set_ylim(lo - pad, hi + pad)
    return lo - pad, hi + pad


def _plot_h1_segments(ax, xs, values, label: str = "1H") -> None:
    """只畫水平段，不畫垂直跳，才不會切花 K 線。"""
    labeled = False
    x0 = None
    level = None

    def flush(x1: float) -> None:
        nonlocal labeled
        if level is None or x0 is None or x1 <= x0:
            return
        ax.hlines(
            level,
            x0,
            x1,
            colors="#94a3b8",
            lw=1.15,
            alpha=0.88,
            zorder=3,
            label=label if not labeled else None,
        )
        labeled = True

    for x, y in zip(xs, values):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            flush(float(x))
            x0 = None
            level = None
            continue
        y = float(y)
        if level is None:
            level = y
            x0 = float(x)
        elif abs(y - level) > 1e-9:
            flush(float(x))
            level = y
            x0 = float(x)
    if xs:
        flush(float(list(xs)[-1]) + 1.0)


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    df_5m: pd.DataFrame | None = None,
) -> Path:
    _setup_mpl()
    import matplotlib.pyplot as plt

    sig = trade.signal
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    close_full = df["Close"].astype(float)
    df_1h = add_mas(resample_1h(df))
    h1_on_1m = map_closed_1h(df, df_1h, "Close")

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    _style_axes((ax, axv))

    _paint_candles(ax, axv, window, min_slots=48)
    xs = range(len(window))
    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.45 if n in (10, 20, 200) else 1.0, label=f"MA{n}")

    h1_win = h1_on_1m[start : end + 1]
    if not np.all(np.isnan(h1_win)):
        _plot_h1_segments(ax, list(xs), h1_win, label="1H")

    ax.axhline(trade.stop_price, color="#7f1d1d", ls=":", lw=1.1, alpha=0.9)
    ax.axhline(sig.peak_ma200, color="#264653", ls="--", lw=0.8, alpha=0.45)
    _fit_price_ylim(
        ax,
        window,
        extra=(trade.stop_price, trade.entry_price, trade.exit_price, sig.peak_high, sig.h1_close),
    )

    px, rx, ex, xx = sig.peak_idx - start, sig.retest_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= px < len(window):
        ax.scatter([px], [sig.peak_high], s=40, color="#fb923c", marker="D", zorder=5)
        ax.annotate("4H高", (px, sig.peak_high), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#fdba74", fontsize=8)
    if 0 <= rx < len(window):
        ax.scatter([rx], [float(window["Low"].iloc[rx])], s=36, color="#38bdf8", zorder=5)
        ax.annotate("回測", (rx, float(window["Low"].iloc[rx])), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#7dd3fc", fontsize=8)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#ef4444", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=46, color="#ef4444", marker="v", zorder=6)
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
        f"#{trade_no}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt  峰距{sig.peak_dist:.0f}",
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


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    et = df.index[trade.entry_idx]
    return f"t{trade_no:02d}_{et.strftime('%m%d_%H%M')}.png"


def _fmt_h1(val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.2f}"


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    embed_images: bool = False,
    df_5m: pd.DataFrame | None = None,
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out.parent / "img"

    reason_bits = []
    for r, info in stats.get("by_reason", {}).items():
        reason_bits.append(f"{r} {info['n']}筆 {info['pnl']:+.1f}")
    reason_line = " · ".join(reason_bits) if reason_bits else "無成交"

    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.stop_price - t.entry_price
        reason_cls = {
            "ma200": "tag-tp",
            "wick": "tag-tp",
            "stop": "tag-sl",
        }.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i)
        png = draw_trade_png(df, t, img_dir / img_name, i, df_5m=df_5m)
        src = _img_data_uri(png) if embed_images else f"img/{escape(img_name)}"
        chart = (
            f"<img src='{src}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/>"
        )
        pt = t.signal.peak_idx
        rt = t.signal.retest_idx
        peak_t = df.index[pt].strftime("%m-%d %H:%M") if 0 <= pt < len(df) else "?"
        retest_t = df.index[rt].strftime("%m-%d %H:%M") if 0 <= rt < len(df) else "?"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · 空</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>1m</span>"
            f"<span class='tag tag-info'>峰距 {t.signal.peak_dist:.0f}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  （風險 {risk:.1f} pts）\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"4H高 {t.signal.peak_high:.2f} @ {peak_t}  MA200當時 {t.signal.peak_ma200:.2f}\n"
            f"回測MA10 @ {retest_t}\n"
            f"進場 MA5 {t.signal.ma5:.1f} / MA10 {t.signal.ma10:.1f} / MA20 {t.signal.ma20:.1f}\n"
            f"MA60 {t.signal.ma60:.1f} / MA200 {t.signal.ma200:.1f}\n"
            f"1H {_fmt_h1(t.signal.h1_close)}  /  1H MA20 {_fmt_h1(t.signal.h1_ma20)}"
            "</pre>"
            f"<div class='mini-chart'>{chart}</div>"
            "</article>"
        )

    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：創4H高 {funnel.get('made_high', 0)} → "
            f"回測MA10 {funnel.get('retest', 0)} → "
            f"破MA20 {funnel.get('break_ma20', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（距不夠 {funnel.get('skip_dist', 0)} · "
            f"破排列取消 {funnel.get('cancel_stack', 0)} · "
            f"再創高取消 {funnel.get('cancel_new_high', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else ""
    end = df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else ""
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    dollars = stats["total_points"] * POINT_VALUE - stats["count"] * COMMISSION_RT
    note = "<p class='muted'>圖是靜態 1m K 線。灰線是已收盤小時線。手機請往下捲。</p>" if embed_images else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 線性空</title>
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
<h1>{escape(symbol)} 線性空</h1>
{note}
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">多頭排列創四小時高 → 回測MA10不過高 → 破MA20做空。峰距MA200至少 100 點。停損四小時高；停利 MA200 / 靠近長下影。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">{escape(reason_line)} · 1 口約 {dollars:+.0f} USD（含來回佣 5）</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{''.join(cards) or "<div class='empty'>這一週沒有符合條件的線性空</div>"}
</div>
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    if args.csv:
        raw = pd.read_csv(args.csv, parse_dates=True, index_col=0)
        df = to_et(_flatten_ohlc(raw).dropna())
    else:
        df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1

    params = LinearShortParams(
        lookback4h=args.lookback,
        stop_buf=args.stop_buf,
        min_peak_dist=args.min_peak_dist,
        near_ma200=args.near_ma200,
    )
    funnel: Dict[str, int] = {}
    trades = run_linear_short(df, params, funnel=funnel)
    stats = summarize_trades(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    print(
        "funnel "
        f"high={funnel.get('made_high', 0)} retest={funnel.get('retest', 0)} "
        f"break={funnel.get('break_ma20', 0)} taken={funnel.get('taken', 0)} "
        f"skip_dist={funnel.get('skip_dist', 0)} "
        f"cancel_stack={funnel.get('cancel_stack', 0)} "
        f"cancel_hh={funnel.get('cancel_new_high', 0)}"
    )
    for r, info in stats.get("by_reason", {}).items():
        print(f"  {r}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"entry={t.entry_price:.2f} stop={t.stop_price:.2f} "
            f"peakDist={t.signal.peak_dist:.1f}"
        )

    html_path = args.html
    if args.pages:
        html_path = html_path or str(PAGES_DIR / "index.html")
    if html_path:
        out = Path(html_path)
        write_html_report(out, df, trades, args.symbol, args.period, funnel=funnel, embed_images=False)
        view = out.parent / "view.html"
        write_html_report(view, df, trades, args.symbol, args.period, funnel=funnel, embed_images=True)
        print(f"html={out}")
        print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 一分 K 線性空（對齊 Pine）")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="7d")
    p.add_argument("--csv", default="")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-linear-short/")
    p.add_argument("--lookback", type=int, default=240)
    p.add_argument("--stop-buf", type=float, default=0.0)
    p.add_argument("--min-peak-dist", type=float, default=100.0)
    p.add_argument("--near-ma200", type=float, default=15.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
