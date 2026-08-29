#!/usr/bin/env python3
"""幣安 1 小時 K：MA25 下破底，再重新站上 MA25。

對齊手機圖那種走法：價先跌破 MA25、在下面做出低點（V / W），
收盤再站回 MA25。進場用站回那根收盤；停損在破底低點。

用法:
  python3 examples/binance_1h_ma25_reclaim.py --symbols AVGOUSDT,ONDSUSDT --days 45
  python3 examples/binance_1h_ma25_reclaim.py --limit 80 --days 30 --pages
  python3 examples/binance_1h_ma25_reclaim.py --recent 48
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-1h-ma25" / "index.html"
BRANCH = "cursor/1h-ma25-reclaim-2a6a"
BASE = "https://www.binance.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
KEEP = {"AVGOUSDT", "ONDSUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Clienttype": "web", "Accept": "application/json"})

MA_COLORS = {
    7: "#f0c14a",
    14: "#26a69a",
    25: "#d28cff",
    99: "#42a5f5",
    120: "#26c6da",
    200: "#ef9a9a",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    break_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    bottom: float
    ma25: float
    depth_pct: float
    bars_below: int
    shape: str
    vol_ratio: float
    impulse_pct: float = 0.0
    undercut_pct: float = 0.0
    flush_atr: float = 0.0
    quality: str = "C"
    quality_score: int = 0
    ma7: float = 0.0
    ma99: float = 0.0
    ma200: float = 0.0


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_pct: float
    exit_reason: str
    quality: str


@dataclass
class Hit:
    symbol: str
    df: pd.DataFrame
    trade: TradeResult
    quote_volume: float = 0.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sma(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def atr(high, low, close, n: int = 14) -> np.ndarray:
    prev = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return sma(tr, n)


def flush_metrics(
    high: np.ndarray,
    low: np.ndarray,
    bottom_idx: int,
    bottom: float,
    atr14: np.ndarray,
    *,
    flush_bars: int = 4,
    base_bars: int = 12,
) -> Tuple[float, float, float]:
    """4 根急殺：幅度、ATR 倍數、相對急殺前平台低點的破底。"""
    left = max(0, bottom_idx - flush_bars + 1)
    peak = float(np.max(high[left : bottom_idx + 1]))
    if peak <= 0:
        return 0.0, 0.0, 0.0
    impulse = (peak - bottom) / peak
    atr_v = float(atr14[bottom_idx]) if bottom_idx < len(atr14) and not np.isnan(atr14[bottom_idx]) else 0.0
    flush_atr = (peak - bottom) / atr_v if atr_v > 0 else 0.0
    base_right = left
    base_left = max(0, base_right - base_bars)
    if base_right <= base_left:
        return float(impulse), float(flush_atr), 0.0
    prior = float(np.min(low[base_left:base_right]))
    undercut = (prior - bottom) / prior if prior > 0 else 0.0
    return float(impulse), float(flush_atr), float(undercut)


def get_json(path: str, params=None, retries: int = 6) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.45 * (i + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def universe(min_quote_volume: float = 8_000_000, extra: Sequence[str] = ()) -> List[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out: List[str] = []
    for s in info.get("symbols") or []:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_quote_volume and sym not in KEEP and sym not in extra:
            continue
        out.append(sym)
    for sym in (*KEEP, *extra):
        if sym and sym not in out:
            out.append(sym)
    return out


def ticker_quote_volume() -> Dict[str, float]:
    try:
        rows = get_json("/fapi/v1/ticker/24hr")
    except Exception:  # noqa: BLE001
        return {}
    return {t["symbol"]: float(t.get("quoteVolume") or 0) for t in rows}


def _klines_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_klines(symbol: str, interval: str = "1h", days: int = 45, drop_forming: bool = True) -> pd.DataFrame:
    """Pull 1h klines. One request covers ~62 days (limit 1500)."""
    limit = min(1500, max(80, int(days) * 24 + 40))
    raw = get_json(
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not raw:
        return _klines_to_df([])
    if drop_forming:
        now_ms = int(time.time() * 1000)
        # 1h bar is open for 3_600_000 ms
        if int(raw[-1][0]) + 3_600_000 > now_ms:
            raw = raw[:-1]
    df = _klines_to_df(raw)
    if df.empty:
        return df
    cutoff = datetime.now(TPE) - timedelta(days=days)
    return df.loc[df.index >= cutoff].copy()


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


def classify_shape(low: np.ndarray, high: np.ndarray, start: int, end: int) -> str:
    """V if one clear trough; W if two swing lows with a bounce in between."""
    if end - start < 6:
        return "V"
    mins: List[int] = []
    for i in range(start + 2, end - 1):
        window = low[i - 2 : i + 3]
        if float(low[i]) <= float(np.min(window)) + 1e-12 and int(np.sum(np.isclose(window, low[i]))) == 1:
            mins.append(i)
    if len(mins) < 2:
        return "V"
    left, right = mins[0], mins[-1]
    if right - left < 3:
        return "V"
    peak = float(np.max(high[left : right + 1]))
    trough = min(float(low[left]), float(low[right]))
    if trough <= 0:
        return "V"
    if (peak / trough - 1.0) < 0.008:
        return "V"
    return "W"


def quality_of(depth_pct: float, vol_ratio: float, shape: str, stacked: bool) -> Tuple[int, str]:
    score = 0
    if depth_pct >= 0.025:
        score += 1
    if vol_ratio >= 1.35:
        score += 1
    if shape == "W":
        score += 1
    if stacked:
        score += 1
    if score >= 3:
        return score, "A"
    if score >= 2:
        return score, "B"
    return score, "C"


def _extend_below_run(close: np.ndarray, ma25: np.ndarray, start: int, allow_above: int = 2) -> int:
    """Stay in the dip while mostly below MA25. A 1–2 bar W-bounce above does not end it."""
    n = len(close)
    end = start
    i = start
    while i + 1 < n and not np.isnan(ma25[i + 1]):
        if close[i + 1] < ma25[i + 1]:
            i += 1
            end = i
            continue
        k = 1
        while (
            k <= allow_above
            and i + 1 + k < n
            and not np.isnan(ma25[i + 1 + k])
            and close[i + 1 + k] >= ma25[i + 1 + k]
        ):
            k += 1
        nxt = i + 1 + k
        if k <= allow_above and nxt < n and not np.isnan(ma25[nxt]) and close[nxt] < ma25[nxt]:
            i = nxt
            end = i
            continue
        break
    return end


def detect_signals(
    df: pd.DataFrame,
    *,
    ma_period: int = 25,
    min_bars_below: int = 10,
    max_bars_below: int = 36,
    min_depth_pct: float = 0.028,
    min_impulse_pct: float = 0.023,
    min_undercut_pct: float = 0.014,
    min_flush_atr: float = 2.8,
    flush_bars: int = 4,
    stop_buffer_pct: float = 0.003,
    target_r: float = 2.0,
    min_entry_gap: int = 8,
    vol_lookback: int = 20,
    break_lookback: int = 16,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """1h close 跌破 MA25 → 在下面做出低點 → 收盤重新站上 MA25。"""
    if df is None or len(df) < ma_period + min_bars_below + 5:
        return []

    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
    ma25 = sma(close, ma_period)
    atr14 = atr(high, low, close, 14)
    ma7 = sma(close, 7)
    ma99 = sma(close, 99)
    ma200 = sma(close, 200)
    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    i = ma_period
    while i < n - 1:
        if np.isnan(ma25[i]):
            i += 1
            continue
        # 第一根收在 MA25 下方（前一根還在上方或剛好站上）
        if not (close[i] < ma25[i] and (i == 0 or close[i - 1] >= ma25[i - 1] or np.isnan(ma25[i - 1]))):
            i += 1
            continue

        bump("cross_down")
        start = i
        end = _extend_below_run(close, ma25, start)
        bars_below = end - start + 1
        reclaim = end + 1
        if reclaim >= n or np.isnan(ma25[reclaim]) or close[reclaim] <= ma25[reclaim]:
            bump("no_reclaim")
            i = end + 1
            continue

        if bars_below < min_bars_below:
            bump("too_short")
            i = reclaim
            continue
        if bars_below > max_bars_below:
            bump("too_long")
            i = reclaim
            continue

        bottom_rel = int(np.argmin(low[start : end + 1]))
        bottom_idx = start + bottom_rel
        bottom = float(low[bottom_idx])
        ref_ma = float(np.nanmax(ma25[start : bottom_idx + 1]))
        if ref_ma <= 0:
            i = reclaim
            continue
        depth_pct = (ref_ma - bottom) / ref_ma
        if depth_pct < min_depth_pct:
            bump("shallow")
            i = reclaim
            continue

        impulse_pct, flush_atr, undercut_pct = flush_metrics(
            high, low, bottom_idx, bottom, atr14, flush_bars=flush_bars
        )
        if impulse_pct < min_impulse_pct or flush_atr < min_flush_atr or undercut_pct < min_undercut_pct:
            bump("weak_flush")
            i = reclaim
            continue

        left = max(0, bottom_idx - break_lookback + 1)
        if bottom > float(np.min(low[left : bottom_idx + 1])) + 1e-12:
            bump("not_break")
            i = reclaim
            continue

        bump("reclaim")
        if reclaim - last_entry < min_entry_gap:
            bump("gap")
            i = reclaim
            continue

        entry = float(close[reclaim])
        stop = bottom * (1.0 - stop_buffer_pct)
        risk = entry - stop
        if risk <= 0:
            bump("bad_risk")
            i = reclaim
            continue

        vol_avg = float(np.mean(volume[max(0, reclaim - vol_lookback) : reclaim]) or 1.0)
        vol_ratio = float(volume[reclaim] / vol_avg) if vol_avg > 0 else 1.0
        shape = classify_shape(low, high, start, end)
        stacked = (not np.isnan(ma7[reclaim])) and entry > float(ma7[reclaim]) > float(ma25[reclaim])
        q_score, q_grade = quality_of(depth_pct, vol_ratio, shape, stacked)

        signals.append(
            Signal(
                break_idx=bottom_idx,
                entry_idx=reclaim,
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(entry + risk * target_r),
                bottom=bottom,
                ma25=float(ma25[reclaim]),
                depth_pct=float(depth_pct),
                bars_below=int(bars_below),
                shape=shape,
                vol_ratio=float(vol_ratio),
                impulse_pct=float(impulse_pct),
                undercut_pct=float(undercut_pct),
                flush_atr=float(flush_atr),
                quality=q_grade,
                quality_score=q_score,
                ma7=float(ma7[reclaim]) if not np.isnan(ma7[reclaim]) else 0.0,
                ma99=float(ma99[reclaim]) if not np.isnan(ma99[reclaim]) else 0.0,
                ma200=float(ma200[reclaim]) if not np.isnan(ma200[reclaim]) else 0.0,
            )
        )
        last_entry = reclaim
        bump("taken")
        i = reclaim + 1

    return signals


def simulate(
    df: pd.DataFrame,
    signals: List[Signal],
    *,
    max_hold: int = 36,
    lose_ma_closes: int = 2,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma25 = sma(close, 25)
    results: List[TradeResult] = []

    for sig in signals:
        entry_idx = sig.entry_idx
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        if entry <= stop:
            continue
        limit = min(entry_idx + max_hold, len(df) - 1)
        exit_idx = limit
        exit_price = float(close[exit_idx])
        exit_reason = "timeout"
        below_streak = 0

        for k in range(entry_idx + 1, limit + 1):
            if low[k] <= stop:
                exit_idx, exit_price, exit_reason = k, float(stop), "stop"
                break
            if high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break
            if not np.isnan(ma25[k]) and close[k] < ma25[k]:
                below_streak += 1
                if below_streak >= lose_ma_closes:
                    exit_idx, exit_price, exit_reason = k, float(close[k]), "lost_ma25"
                    break
            else:
                below_streak = 0
        else:
            exit_idx, exit_price, exit_reason = limit, float(close[limit]), "timeout"

        pnl_pct = (exit_price / entry - 1.0) * 100.0
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_pct=float(pnl_pct),
                exit_reason=exit_reason,
                quality=sig.quality,
            )
        )
    return results


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    pnls = [float(t.pnl_pct) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_q: Dict[str, List[float]] = {}
    for t in trades:
        by_q.setdefault(t.quality, []).append(float(t.pnl_pct))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_pct": float(sum(pnls)),
        "avg_pct": float(sum(pnls) / n) if n else 0.0,
        "by_quality": {
            q: {"n": len(v), "wins": sum(1 for p in v if p > 0), "pnl": float(sum(v))}
            for q, v in sorted(by_q.items())
        },
    }


# ---------------------------------------------------------------------------
# Charts / HTML
# ---------------------------------------------------------------------------


def _setup_cjk() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, min(trade.signal.break_idx, trade.entry_idx) - 18)
    end = min(len(df) - 1, max(trade.exit_idx, trade.entry_idx) + 10)
    return start, end


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int, title_extra: str = "") -> Path:
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
        lw = 2.15 if n == 25 else (1.25 if n <= 14 else 1.0)
        ax.plot(list(xs), ma, color=col, lw=lw, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    bx, ex, xx = sig.break_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.bottom], s=42, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, sig.bottom),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=46, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "站上",
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
            color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_pct >= 0 else ""
    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}Q{trade.quality} {sig.shape}  "
        f"{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_pct:.2f}%",
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


def _equity_svg(pnls: List[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(0.2, (ymax - ymin) * 0.12)
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


def select_card_hits(
    hits: List[Hit],
    *,
    recent_hours: int = 72,
    keep_symbols: Sequence[str] = (),
    max_cards: int = 48,
) -> List[Hit]:
    """Keep example symbols plus the latest tape so the phone report stays short."""
    now = datetime.now(TPE)
    keep = {s.upper() for s in keep_symbols}
    picked: List[Hit] = []
    for hit in hits:
        ts = hit.df.index[hit.trade.entry_idx]
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TPE)
        else:
            ts = ts.replace(tzinfo=TPE)
        if hit.symbol.upper() in keep or now - ts <= timedelta(hours=recent_hours):
            picked.append(hit)
    if len(picked) > max_cards:
        keep_hits = [h for h in picked if h.symbol.upper() in keep]
        rest = [h for h in picked if h.symbol.upper() not in keep]
        rest.sort(key=lambda h: h.df.index[h.trade.entry_idx], reverse=True)
        picked = keep_hits + rest[: max(0, max_cards - len(keep_hits))]
        picked.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    return picked


def write_html_report(
    path: Path,
    hits: List[Hit],
    *,
    days: int,
    scanned: int,
    funnel: Optional[Dict[str, int]] = None,
    recent_hours: int = 0,
    all_stats: Optional[dict] = None,
    card_note: str = "",
) -> Path:
    stats = all_stats or summarize_trades([h.trade for h in hits])
    cards: List[str] = []
    now = datetime.now(TPE)
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        fresh = ""
        if recent_hours > 0:
            ts = et.tz_convert(TPE) if getattr(et, "tzinfo", None) else et.replace(tzinfo=TPE)
            if now - ts <= timedelta(hours=recent_hours):
                fresh = f" <span class='tag tag-fresh'>近{recent_hours}h</span>"
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "lost_ma25": "tag-sl"}.get(t.exit_reason, "tag-time")
        img_name = f"t{i:02d}_{hit.symbol}_{et.strftime('%m%d_%H%M')}_q{t.quality.lower()}.png"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=hit.symbol)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(hit.symbol)} · Q{escape(t.quality)} · {escape(t.signal.shape)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>1h MA25</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"{fresh}"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.6g}\n"
            f"stop  {t.stop_price:.6g}  (−{(risk / t.entry_price) * 100:.2f}%)\n"
            f"target {t.target_price:.6g}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.6g}  {t.exit_reason}\n"
            f"破底 {t.signal.bottom:.6g}  深度 {t.signal.depth_pct * 100:.2f}%  在下 {t.signal.bars_below}h\n"
            f"急殺 {t.signal.impulse_pct * 100:.1f}% / {t.signal.flush_atr:.1f}ATR  破前低 {t.signal.undercut_pct * 100:.1f}%\n"
            f"量比 {t.signal.vol_ratio:.2f}x  MA25 {t.signal.ma25:.6g}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i} {escape(hit.symbol)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：跌破 {funnel.get('cross_down', 0)} → "
            f"站回 {funnel.get('reclaim', 0)} → 進場 {funnel.get('taken', 0)}"
            f"（太短 {funnel.get('too_short', 0)} · 太長 {funnel.get('too_long', 0)} · "
            f"太淺 {funnel.get('shallow', 0)} · 急殺不夠 {funnel.get('weak_flush', 0)} · "
            f"未站回 {funnel.get('no_reclaim', 0)}）</p>"
        )
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.2f}%" for q, info in stats.get("by_quality", {}).items()]
    total_cls = "pnl-win" if stats["total_pct"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 1h MA25 下破底再站上</title>
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
.tag-info{{background:rgba(210,140,255,0.12);color:#e2b6ff;border-color:rgba(210,140,255,0.28)}}
.tag-fresh{{background:rgba(88,166,255,0.16);color:#79c0ff;border-color:rgba(88,166,255,0.35)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 1h · MA25 下破底再站上</h1>
<p class="muted">近 {days} 天 · 掃 {scanned} 檔 U 本位永續 · 粉線 MA25<br/>
價先收在 MA25 下，再出現跟截圖差不多的急殺：4 根內至少 2.3%、≥ 2.8 ATR，並跌破急殺前平台低點 1.4%。之後收盤站回 MA25。停損破底低點，目標 2R。{card_note}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">加總<b class="{total_cls}">{stats['total_pct']:+.2f}%</b></div>
<div class="card">均筆<b>{stats['avg_pct']:+.2f}%</b></div>
</div>
<p class="muted">{escape(' · '.join(q_bits) if q_bits else '無品質分組')}</p>
{funnel_line}
<div class="equity">{_equity_svg([h.trade.pnl_pct for h in hits])}</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有 MA25 下破底再站上</div>"}
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = BRANCH) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_symbol(symbol: str, days: int, detect_kw: dict) -> tuple[List[Hit], dict]:
    meta = {"symbol": symbol, "bars": 0, "error": "", "n_trade": 0}
    try:
        df = fetch_klines(symbol, days=days)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 40:
        meta["error"] = "too_few_bars"
        return [], meta
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, **detect_kw)
    trades = simulate(df, sigs)
    meta["n_trade"] = len(trades)
    meta["funnel"] = funnel
    return [Hit(symbol, df, t) for t in trades], meta


def merge_funnels(acc: Dict[str, int], part: Dict[str, int]) -> None:
    for k, v in part.items():
        acc[k] = acc.get(k, 0) + int(v)


def detect_kwargs(args) -> dict:
    return dict(
        min_bars_below=args.min_bars,
        max_bars_below=args.max_bars,
        min_depth_pct=args.min_depth / 100.0,
        min_impulse_pct=args.min_impulse / 100.0,
        min_undercut_pct=args.min_undercut / 100.0,
        min_flush_atr=args.min_atr,
        target_r=args.target_r,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    extras = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if extras and not args.universe:
        symbols = extras
    else:
        print("載入標的…", file=sys.stderr)
        symbols = universe(args.min_quote, extra=extras)
        if args.limit and len(symbols) > args.limit:
            qv = ticker_quote_volume()
            keep = [s for s in symbols if s in KEEP or s in extras]
            rest = [s for s in symbols if s not in keep]
            rest.sort(key=lambda s: qv.get(s, 0.0), reverse=True)
            symbols = keep + rest[: max(0, args.limit - len(keep))]
            # de-dupe preserve order
            seen = set()
            symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    print(f"掃描 {len(symbols)} 檔  1h  近 {args.days} 天", file=sys.stderr)
    kw = detect_kwargs(args)
    hits: List[Hit] = []
    funnel: Dict[str, int] = {}
    errors = 0

    def work(sym: str) -> tuple[List[Hit], dict]:
        return scan_symbol(sym, args.days, kw)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(work, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                part, meta = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{done:3d}/{len(symbols)}] {sym} err {exc}", file=sys.stderr)
                continue
            if meta.get("error"):
                errors += 1
            hits.extend(part)
            merge_funnels(funnel, meta.get("funnel") or {})
            flag = f" trades={meta['n_trade']}" if meta.get("n_trade") else ""
            err = f" {meta['error']}" if meta.get("error") else ""
            print(f"[{done:3d}/{len(symbols)}] {sym} bars={meta.get('bars', 0)}{flag}{err}", file=sys.stderr)

    hits.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    stats = summarize_trades([h.trade for h in hits])
    print(
        f"done symbols={len(symbols)} errors={errors} trades={stats['count']} "
        f"WR={stats['win_rate']:.1f}% sum={stats['total_pct']:+.2f}% avg={stats['avg_pct']:+.2f}%"
    )
    if funnel:
        print(
            "funnel "
            f"down={funnel.get('cross_down', 0)} reclaim={funnel.get('reclaim', 0)} "
            f"taken={funnel.get('taken', 0)} short={funnel.get('too_short', 0)} "
            f"long={funnel.get('too_long', 0)} shallow={funnel.get('shallow', 0)} "
            f"weak={funnel.get('weak_flush', 0)} noreclaim={funnel.get('no_reclaim', 0)}"
        )
    now = datetime.now(TPE)
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        ts = hit.df.index[t.entry_idx]
        age_h = (now - ts) / timedelta(hours=1)
        mark = " *" if args.recent and age_h <= args.recent else ""
        print(
            f"  [{i}] {hit.symbol} Q{t.quality} {t.signal.shape} "
            f"{ts.strftime('%m-%d %H:%M')} {t.exit_reason} {t.pnl_pct:+.2f}% "
            f"depth={t.signal.depth_pct*100:.1f}% flush={t.signal.impulse_pct*100:.1f}%/"
            f"{t.signal.flush_atr:.1f}atr under={t.signal.undercut_pct*100:.1f}%{mark}"
        )

    html_path = Path(args.html) if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        extras = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
        keep_for_cards = set(KEEP) | set(extras)
        cards = select_card_hits(hits, recent_hours=max(args.recent, 72), keep_symbols=keep_for_cards)
        card_note = (
            f"<br/>卡片 {len(cards)} 筆：AVGO/ONDS 等樣本 + 近 {max(args.recent, 72)} 小時。"
            if len(cards) != len(hits)
            else ""
        )
        out = write_html_report(
            html_path,
            cards,
            days=args.days,
            scanned=len(symbols),
            funnel=funnel,
            recent_hours=args.recent,
            all_stats=stats,
            card_note=card_note,
        )
        view = write_view_html(out)
        print(f"html={out}")
        print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="幣安 1h MA25 下破底再站上")
    p.add_argument("--symbols", default="", help="只掃這些，逗號分隔，例如 AVGOUSDT,ONDSUSDT")
    p.add_argument("--universe", action="store_true", help="即使指定 --symbols 也掃流動永續")
    p.add_argument("--limit", type=int, default=80, help="流動永續最多幾檔（成交額由高到低）")
    p.add_argument("--min-quote", type=float, default=8_000_000)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-bars", type=int, default=10, help="在 MA25 下至少幾根 1h")
    p.add_argument("--max-bars", type=int, default=36, help="在 MA25 下最多幾根 1h")
    p.add_argument("--min-depth", type=float, default=2.8, help="相對 MA25 最低深度 %")
    p.add_argument("--min-impulse", type=float, default=2.3, help="4 根急殺最低幅度 %")
    p.add_argument("--min-undercut", type=float, default=1.4, help="跌破急殺前平台低點 %")
    p.add_argument("--min-atr", type=float, default=2.8, help="急殺至少幾個 ATR")
    p.add_argument("--target-r", type=float, default=2.0)
    p.add_argument("--recent", type=int, default=48, help="報告裡標近幾小時的新訊號")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/binance-1h-ma25/index.html")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
