#!/usr/bin/env python3
"""幣安 15 分 K：MA7/14/25 空頭排列，且收盤同時跌破 MA99 與 MA120，做空回測。

對齊截圖 CLOUSDT 那種急殺：短均 7<14<25，同一根收盤穿過 99/120，
且進場價在 1 小時 MA25 下方。

用法:
  python3 examples/binance_15m_short.py
  python3 examples/binance_15m_short.py --symbol CLOUSDT --days 7 --pages
  python3 examples/binance_15m_short.py --days 7 --pages
  python3 examples/test_binance_15m_short.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-15m-short" / "index.html"
BASE = "https://www.binance.com"
KEEP = {"CLOUSDT"}
UA = "Mozilla/5.0"
MA_COLORS = {
    7: "#f0c14a",
    14: "#79c0ff",
    25: "#f472b6",
    99: "#42a5f5",
    120: "#26c6da",
    200: "#c4b5fd",
}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "application/json", "Clienttype": "web"})


@dataclass(frozen=True)
class ShortParams:
    stop_lookback: int = 1
    target_r: float = 2.0
    time_bars: int = 32
    min_risk_pct: float = 0.005
    max_risk_pct: float = 0.18
    require_red: bool = True
    min_break_pct: float = 0.003
    min_body_pct: float = 0.008
    min_vol_ratio: float = 1.5


@dataclass(frozen=True)
class Signal:
    entry_idx: int
    entry_price: float
    ma7: float
    ma14: float
    ma25: float
    ma99: float
    ma120: float
    ma_high: float
    body_pct: float
    vol_ratio: float


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


@dataclass
class Hit:
    symbol: str
    trade: TradeResult
    df: pd.DataFrame
    df_1h: Optional[pd.DataFrame] = None


def sma(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if n <= 0 or len(values) < n:
        return out
    c = np.cumsum(np.asarray(values, dtype=float))
    out[n - 1] = c[n - 1] / n
    if len(values) > n:
        out[n:] = (c[n:] - c[:-n]) / n
    return out


def default_params(**overrides: Any) -> ShortParams:
    return ShortParams(**overrides)


def _below_both(close: float, ma99: float, ma120: float) -> bool:
    return close < ma99 and close < ma120


def detect_signals(
    df: pd.DataFrame,
    params: Optional[ShortParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """收盤確認：7<14<25，且這一根才同時跌破 99 與 120。"""
    p = params or default_params()
    close = df["close"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    vol = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(df), dtype=float)
    n = len(close)
    m7, m14, m25 = sma(close, 7), sma(close, 14), sma(close, 25)
    m99, m120 = sma(close, 99), sma(close, 120)
    v20 = sma(vol, 20)
    counts = {
        "ready": 0,
        "stack": 0,
        "cross": 0,
        "red": 0,
        "shallow": 0,
        "thin": 0,
        "quiet": 0,
        "entry": 0,
    }
    signals: List[Signal] = []
    for i in range(1, n):
        vals = (m7[i], m14[i], m25[i], m99[i], m120[i], m99[i - 1], m120[i - 1])
        if np.isnan(vals).any():
            continue
        counts["ready"] += 1
        if not (m7[i] < m14[i] < m25[i]):
            continue
        counts["stack"] += 1
        now_below = _below_both(close[i], m99[i], m120[i])
        was_below = _below_both(close[i - 1], m99[i - 1], m120[i - 1])
        if not (now_below and not was_below):
            continue
        counts["cross"] += 1
        if p.require_red and close[i] >= open_[i]:
            continue
        counts["red"] += 1
        ma_high = max(float(m99[i]), float(m120[i]))
        if close[i] >= ma_high * (1.0 - p.min_break_pct):
            counts["shallow"] += 1
            continue
        body_pct = (open_[i] - close[i]) / open_[i] if open_[i] else 0.0
        if body_pct < p.min_body_pct:
            counts["thin"] += 1
            continue
        vol_ratio = float(vol[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        if p.min_vol_ratio > 0 and vol_ratio < p.min_vol_ratio:
            counts["quiet"] += 1
            continue
        counts["entry"] += 1
        signals.append(
            Signal(
                entry_idx=i,
                entry_price=float(close[i]),
                ma7=float(m7[i]),
                ma14=float(m14[i]),
                ma25=float(m25[i]),
                ma99=float(m99[i]),
                ma120=float(m120[i]),
                ma_high=ma_high,
                body_pct=float(body_pct),
                vol_ratio=float(vol_ratio),
            )
        )
    if funnel is not None:
        for k, v in counts.items():
            funnel[k] = funnel.get(k, 0) + v
    return signals


def _stop_price(high: np.ndarray, entry_idx: int, lookback: int, entry: float, ma_high: float) -> Optional[float]:
    a = max(0, entry_idx - lookback + 1)
    stop = float(np.max(high[a : entry_idx + 1]))
    stop = max(stop, float(ma_high))
    if not np.isfinite(stop) or stop <= entry:
        return None
    return stop


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    params: Optional[ShortParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[TradeResult]:
    p = params or default_params()
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(close)
    trades: List[TradeResult] = []
    busy_until = -1
    skip_risk = 0
    skip_busy = 0
    for sig in signals:
        if sig.entry_idx <= busy_until:
            skip_busy += 1
            continue
        entry = float(sig.entry_price)
        stop = _stop_price(high, sig.entry_idx, p.stop_lookback, entry, sig.ma_high)
        if stop is None:
            skip_risk += 1
            continue
        risk = stop - entry
        risk_pct = risk / entry if entry else 0.0
        if risk_pct < p.min_risk_pct or risk_pct > p.max_risk_pct:
            skip_risk += 1
            continue
        target = entry - p.target_r * risk
        if target <= 0:
            skip_risk += 1
            continue
        exit_idx = sig.entry_idx
        exit_px = entry
        reason = "open"
        last = min(n - 1, sig.entry_idx + p.time_bars)
        for k in range(sig.entry_idx + 1, last + 1):
            if float(high[k]) >= stop:
                exit_idx, exit_px, reason = k, stop, "stop"
                break
            if float(low[k]) <= target:
                exit_idx, exit_px, reason = k, target, "target"
                break
        else:
            if last > sig.entry_idx:
                exit_idx, exit_px = last, float(close[last])
                reason = "open" if last == n - 1 and last < sig.entry_idx + p.time_bars else "time"
            else:
                exit_idx, exit_px, reason = sig.entry_idx, entry, "open"
        busy_until = exit_idx
        trades.append(
            TradeResult(
                signal=sig,
                entry_idx=sig.entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_px,
                stop_price=stop,
                target_price=target,
                pnl_points=entry - exit_px,
                pnl_pct=(entry - exit_px) / entry if entry else 0.0,
                exit_reason=reason,
            )
        )
    if funnel is not None:
        funnel["skip_risk"] = funnel.get("skip_risk", 0) + skip_risk
        funnel["skip_busy"] = funnel.get("skip_busy", 0) + skip_busy
        funnel["taken"] = funnel.get("taken", 0) + len(trades)
    return trades


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    pcts = [float(t.pnl_pct) for t in trades]
    n = len(pcts)
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    closed = [t for t in trades if t.exit_reason != "open"]
    closed_wins = sum(1 for t in closed if t.pnl_pct > 0)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_pct": float(sum(pcts)),
        "avg_pct": float(sum(pcts) / n) if n else 0.0,
        "closed": len(closed),
        "open": n - len(closed),
        "closed_win_rate": 100.0 * closed_wins / len(closed) if closed else 0.0,
        "closed_avg_pct": float(sum(t.pnl_pct for t in closed) / len(closed)) if closed else 0.0,
        "reasons": reasons,
    }


def filter_entry_window(df: pd.DataFrame, signals: Sequence[Signal], days: int) -> List[Signal]:
    if not len(df) or days <= 0:
        return list(signals)
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    return [s for s in signals if df.index[s.entry_idx] >= start]


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (i + 1))
    raise last


STOCK_UNDERLYING = {"EQUITY", "HK_EQUITY", "KR_EQUITY", "CN_EQUITY", "PREMARKET"}


def is_stock_contract(info: dict) -> bool:
    """幣安 TradFi 股票／ETF／盤前，不含黃金原油等商品。"""
    return str(info.get("underlyingType") or "") in STOCK_UNDERLYING


def universe(min_quote_vol: float = 10_000_000, include_stocks: bool = False) -> List[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out: List[str] = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        if not include_stocks and is_stock_contract(s):
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_quote_vol and sym not in KEEP:
            continue
        out.append(sym)
    if "CLOUSDT" in KEEP and "CLOUSDT" not in out:
        out.append("CLOUSDT")
    return sorted(set(out))


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
}


def fetch_klines(sym: str, interval: str = "15m", limit: int = 1500) -> pd.DataFrame:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    now_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MS.get(interval, 900_000)
    if int(raw[-1][0]) + interval_ms > now_ms:
        raw = raw[:-1]
    idx = pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE)
    return pd.DataFrame(
        {
            "open": [float(x[1]) for x in raw],
            "high": [float(x[2]) for x in raw],
            "low": [float(x[3]) for x in raw],
            "close": [float(x[4]) for x in raw],
            "volume": [float(x[5]) for x in raw],
        },
        index=idx,
    )


def _as_index_ts(df: pd.DataFrame, ts):
    ts = pd.Timestamp(ts)
    if df.index.tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(df.index.tz)
        else:
            ts = ts.tz_convert(df.index.tz)
    return ts


def bar_index_at(df: pd.DataFrame, ts) -> Optional[int]:
    """含 ts 的那根 K（開盤 <= ts 的最後一根）。"""
    if df is None or len(df) == 0:
        return None
    ts = _as_index_ts(df, ts)
    pos = int(df.index.searchsorted(ts, side="right") - 1)
    if pos < 0 or pos >= len(df):
        return None
    return pos


def htf_snapshot(df: pd.DataFrame, ts) -> str:
    i = bar_index_at(df, ts)
    if i is None:
        return "1h 無資料"
    close = df["close"].to_numpy(float)
    m7, m14, m25 = sma(close, 7)[i], sma(close, 14)[i], sma(close, 25)[i]
    m99, m120 = sma(close, 99)[i], sma(close, 120)[i]
    px = float(df["close"].iloc[i])
    t = df.index[i].strftime("%m-%d %H:%M")
    stack = np.isfinite([m7, m14, m25]).all() and m7 < m14 < m25
    below = np.isfinite([m99, m120]).all() and px < m99 and px < m120
    align = "空頭排列" if stack else "非空頭排列"
    brk = "收在99/120下" if below else "尚未同時跌破99/120"
    parts = [f"1h {t}  {align} · {brk}"]
    mas = []
    for name, val in (("MA7", m7), ("MA14", m14), ("MA25", m25), ("MA99", m99), ("MA120", m120)):
        if np.isfinite(val):
            mas.append(f"{name} {val:.5g}")
    if mas:
        parts.append(" / ".join(mas) + f"  close {px:.5g}")
    return "\n".join(parts)


def htf_ma_at_entry(
    df_htf: pd.DataFrame,
    ts,
    price: float,
    n: int = 25,
    interval: str = "1h",
) -> Optional[float]:
    """進場當下看得到的 HTF MA。未收完的 K 用進場價當收盤，不偷看後面。"""
    if df_htf is None or len(df_htf) == 0:
        return None
    i = bar_index_at(df_htf, ts)
    if i is None:
        return None
    closes = df_htf["close"].to_numpy(float)
    ts = _as_index_ts(df_htf, ts)
    bar_close = df_htf.index[i] + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    if ts < bar_close:
        if i < n - 1:
            return None
        window = np.concatenate([closes[i - n + 1 : i], [float(price)]])
        if len(window) != n or not np.isfinite(window).all():
            return None
        return float(window.mean())
    if i < n - 1:
        return None
    val = sma(closes, n)[i]
    if not np.isfinite(val):
        return None
    return float(val)


def filter_below_1h_ma25(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    df_1h: pd.DataFrame,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """15m 進場收盤必須低於當時的 1h MA25。"""
    kept: List[Signal] = []
    above = 0
    nodata = 0
    for sig in signals:
        ts = df.index[sig.entry_idx]
        ma = htf_ma_at_entry(df_1h, ts, sig.entry_price, n=25, interval="1h")
        if ma is None:
            nodata += 1
            continue
        if sig.entry_price >= ma:
            above += 1
            continue
        kept.append(sig)
    if funnel is not None:
        funnel["above_1h_ma25"] = funnel.get("above_1h_ma25", 0) + above
        funnel["no_1h"] = funnel.get("no_1h", 0) + nodata
    return kept


def prefetch_1h(symbols: Sequence[str], workers: int = 8) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    uniq = list(dict.fromkeys(symbols))
    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fetch_klines, s, "1h"): s for s in uniq}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                out[sym] = fut.result()
            except Exception:  # noqa: BLE001
                out[sym] = pd.DataFrame()
    return out


def scan_symbol(
    sym: str,
    days: int,
    params: ShortParams,
    funnel: Optional[Dict[str, int]] = None,
    require_1h_ma25: bool = True,
) -> tuple[List[Hit], dict]:
    meta = {"symbol": sym, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_klines(sym)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:100]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 130:
        meta["error"] = "too_few_bars"
        return [], meta
    local: Dict[str, int] = {}
    sigs = detect_signals(df, params, funnel=local)
    if funnel is not None:
        for k, v in local.items():
            funnel[k] = funnel.get(k, 0) + v
    sigs = filter_entry_window(df, sigs, days)
    df_1h = pd.DataFrame()
    try:
        df_1h = fetch_klines(sym, "1h")
    except Exception:  # noqa: BLE001
        df_1h = pd.DataFrame()
    if require_1h_ma25:
        sigs = filter_below_1h_ma25(df, sigs, df_1h, funnel=funnel)
    trades = simulate(df, sigs, params, funnel=funnel)
    meta["n_sig"] = len(sigs)
    meta["n_trade"] = len(trades)
    h1 = df_1h if df_1h is not None and len(df_1h) else None
    return [Hit(sym, t, df, df_1h=h1) for t in trades], meta


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


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    title_extra: str = "",
    *,
    interval: str = "15m",
    entry_idx: Optional[int] = None,
    exit_idx: Optional[int] = None,
    pad_left: Optional[int] = None,
    pad_right: Optional[int] = None,
    mark_levels: bool = True,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _setup_cjk()
    if entry_idx is None:
        entry_idx = trade.entry_idx
    if exit_idx is None:
        exit_idx = trade.exit_idx
    if pad_left is None:
        pad_left = 48 if interval == "1h" else 36
    if pad_right is None:
        pad_right = 12 if interval == "1h" else 8
    start = max(0, min(entry_idx, exit_idx) - pad_left)
    end = min(len(df) - 1, max(exit_idx, entry_idx) + pad_right)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    close_full = df["close"].astype(float)

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
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 25 else 1.05, label=f"MA{n}")

    if mark_levels:
        ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
        ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
        if interval == "15m":
            ax.axhline(trade.signal.ma99, color="#42a5f5", ls="--", lw=0.6, alpha=0.35)
            ax.axhline(trade.signal.ma120, color="#26c6da", ls="--", lw=0.6, alpha=0.35)

    ex = entry_idx - start
    xx = exit_idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=46, color="#ff5252", marker="v", zorder=6)
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

    et = df.index[entry_idx] if 0 <= entry_idx < len(df) else df.index[trade.entry_idx]
    xt = df.index[exit_idx] if 0 <= exit_idx < len(df) else df.index[trade.exit_idx]
    extra = f"{title_extra}  " if title_extra else ""
    label = "1h 對照" if interval == "1h" else interval
    ax.set_title(
        f"#{trade_no}  {extra}{label}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {trade.pnl_pct*100:+.2f}%",
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
    pad = max(0.01, (ymax - ymin) * 0.12)
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


def write_html(
    path: Path,
    hits: List[Hit],
    symbols: Sequence[str],
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    max_charts: int = 80,
    featured: str = "CLOUSDT",
    require_1h_ma25: bool = True,
) -> Path:
    stats = summarize_trades([h.trade for h in hits])
    featured_hits = [h for h in hits if h.symbol == featured]
    chart_hits = list(hits)
    if len(chart_hits) > max_charts:
        featured_set = {id(h) for h in featured_hits}
        rest = [h for h in hits if id(h) not in featured_set]
        rest.sort(key=lambda h: abs(h.trade.pnl_pct), reverse=True)
        keep = featured_hits + rest[: max(0, max_charts - len(featured_hits))]
        chart_hits = keep

    need_1h = [h.symbol for h in chart_hits if h.df_1h is None or h.df_1h.empty]
    fetched_1h = prefetch_1h(need_1h) if need_1h else {}

    cards: List[str] = []
    for i, hit in enumerate(chart_hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        risk = t.stop_price - t.entry_price
        img_name = f"t{i:02d}_{hit.symbol}_{et.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=hit.symbol)
        df_1h = hit.df_1h if hit.df_1h is not None and len(hit.df_1h) else fetched_1h.get(hit.symbol)
        h1_html = ""
        h1_detail = ""
        if df_1h is not None and len(df_1h):
            e1 = bar_index_at(df_1h, et)
            x1 = bar_index_at(df_1h, xt)
            if e1 is None:
                e1 = 0
            if x1 is None:
                x1 = e1
            img_1h = f"h{i:02d}_{hit.symbol}_{et.strftime('%m%d_%H%M')}_1h.png"
            draw_trade_png(
                df_1h,
                t,
                path.parent / "img" / img_1h,
                i,
                title_extra=hit.symbol,
                interval="1h",
                entry_idx=e1,
                exit_idx=x1,
            )
            h1_html = (
                "<div class='chart-label'>1h 對照 · 同一進場／出場時刻</div>"
                f"<div class='mini-chart'><img src='img/{escape(img_1h)}' alt='{escape(hit.symbol)} 1h' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
            h1_detail = "\n" + htf_snapshot(df_1h, et)
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        h1_tag = "<span class='tag'>1h MA25下</span>" if require_1h_ma25 else ""
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(hit.symbol)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))} TPE</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag tag-info'>{escape(hit.symbol)}</span>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>空 {t.signal.ma7:.5g}&lt;{t.signal.ma14:.5g}&lt;{t.signal.ma25:.5g}</span>"
            f"<span class='tag'>實體 {t.signal.body_pct*100:.1f}%</span>"
            f"<span class='tag'>量 {t.signal.vol_ratio:.1f}x</span>"
            f"{h1_tag}"
            "</div>"
            "<pre class='trade-detail'>"
            f"做空 entry {t.entry_price:.6g}  stop {t.stop_price:.6g} (+{risk:.6g})\n"
            f"target {t.target_price:.6g}  exit {t.exit_price:.6g} {t.exit_reason}  {t.pnl_pct*100:+.2f}%\n"
            f"MA99 {t.signal.ma99:.6g} / MA120 {t.signal.ma120:.6g}  跌破 {t.signal.ma_high:.6g}\n"
            f"實體 {t.signal.body_pct*100:.2f}%  量/MA20 {t.signal.vol_ratio:.2f}x"
            f"{escape(h1_detail)}"
            "</pre>"
            "<div class='chart-label'>15m</div>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(hit.symbol)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            f"{h1_html}"
            "</article>"
        )

    fun = funnel or {}
    reasons = stats.get("reasons") or {}
    total_cls = "pnl-win" if stats["total_pct"] >= 0 else "pnl-loss"
    avg_cls = "pnl-win" if stats["avg_pct"] >= 0 else "pnl-loss"
    feat_line = (
        f"{featured} {len(featured_hits)} 筆"
        if featured_hits
        else f"{featured} 這週沒有訊號"
    )
    omitted = ""
    if len(hits) > len(chart_hits):
        omitted = f"<p class='muted'>圖表只畫 {len(chart_hits)} / {len(hits)} 筆（含 {featured}，其餘依 |報酬|）。</p>"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 15m 空頭排列跌破 99/120</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.chart-label{{font-size:11px;color:#8b949e;margin:10px 0 4px}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15m · 7/14/25 空頭排列跌破 99/120 做空</h1>
<p class="muted">{escape(period)} · 掃 {len(symbols)} 檔 U 本位永續
<br/>進場：收盤 MA7&lt;MA14&lt;MA25，上一根還沒同時低於 MA99 與 MA120、這一根紅 K 收盤同時跌破，且進場價在 <b>1h MA25 下方</b>。對齊截圖急殺：實體 ≥ 0.8%、量 ≥ 1.5×MA20、至少跌破長均 0.3%。
<br/>出場：停在跌破 K 高點與 MA99/120 上緣的較高者、目標 2R、或 32 根（8 小時）時間停。做空報酬＝(進−出)/進。加總％不是組合複利，也沒扣手續費。
<br/>每筆下面附同一時刻的 <b>1h K</b> 對照（1h 均線是 1 小時圖自己的 7/14/25/99/120）。股票／ETF 永續預設不掃。</p>
<p class="muted">漏斗：有均線 {fun.get('ready', 0)} → 空頭排列 {fun.get('stack', 0)} → 同時跌破 {fun.get('cross', 0)}
→ 紅 K {fun.get('red', 0)} → 進場 {fun.get('entry', 0)}
· 太淺 {fun.get('shallow', 0)} · 實體不夠 {fun.get('thin', 0)} · 量不夠 {fun.get('quiet', 0)}
· 不在1h MA25下 {fun.get('above_1h_ma25', 0)} · 無1h {fun.get('no_1h', 0)}
· 風險不合 {fun.get('skip_risk', 0)} · 持倉中 {fun.get('skip_busy', 0)}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}
<br/>{escape(feat_line)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{avg_cls}">{stats['avg_pct']*100:+.2f}%</b></div>
<div class="card">加總<b class="{total_cls}">{stats['total_pct']*100:+.2f}%</b></div>
</div>
<div class="equity">{_equity_svg([h.trade.pnl_pct * 100 for h in hits])}</div>
</section>
{omitted}
{''.join(cards) or "<div class='empty'>這段期間沒有符合的做空訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def dump_hits_json(path: Path, hits: List[Hit], stats: dict, funnel: dict, extra: dict) -> Path:
    rows = []
    for hit in hits:
        t = hit.trade
        df = hit.df
        rows.append(
            {
                "symbol": hit.symbol,
                "entry_time": str(df.index[t.entry_idx]),
                "exit_time": str(df.index[t.exit_idx]),
                "entry": t.entry_price,
                "exit": t.exit_price,
                "stop": t.stop_price,
                "target": t.target_price,
                "pnl_pct": t.pnl_pct,
                "reason": t.exit_reason,
                "ma7": t.signal.ma7,
                "ma14": t.signal.ma14,
                "ma25": t.signal.ma25,
                "ma99": t.signal.ma99,
                "ma120": t.signal.ma120,
                "body_pct": t.signal.body_pct,
                "vol_ratio": t.signal.vol_ratio,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "funnel": funnel, "extra": extra, "hits": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="幣安 15m 空頭排列跌破 MA99/120 做空回測")
    p.add_argument("--symbol", default="", help="單一標的，例如 CLOUSDT；空白則掃流動永續")
    p.add_argument("--days", type=int, default=7, help="只統計進場落在最近 N 日")
    p.add_argument("--min-quote-vol", type=float, default=10_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--loose", action="store_true", help="不做實體/量能過濾，只看均線排列與跌破")
    p.add_argument("--include-stocks", action="store_true", help="不過濾 TradFi 股票／ETF 永續")
    p.add_argument("--no-1h-ma25", action="store_true", help="不要求進場價在 1h MA25 下方")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--json", dest="json_path", default="")
    p.add_argument("--max-charts", type=int, default=80)
    args = p.parse_args(argv)

    params = default_params()
    if args.loose:
        params = default_params(min_body_pct=0.0, min_vol_ratio=0.0, min_break_pct=0.002, min_risk_pct=0.004)
    if args.symbol.strip():
        symbols = [args.symbol.strip().upper()]
    else:
        print("載入標的…", flush=True)
        symbols = universe(args.min_quote_vol, include_stocks=args.include_stocks)
    print(
        f"symbols={len(symbols)} days={args.days} interval=15m loose={args.loose} "
        f"stocks={'on' if args.include_stocks else 'off'} "
        f"h1_ma25={'off' if args.no_1h_ma25 else 'on'}",
        flush=True,
    )

    hits: List[Hit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    lock_funnels: List[Dict[str, int]] = []

    def _one(sym: str) -> tuple[List[Hit], dict, Dict[str, int]]:
        local: Dict[str, int] = {}
        stock_hits, meta = scan_symbol(
            sym, args.days, params, funnel=local, require_1h_ma25=not args.no_1h_ma25
        )
        return stock_hits, meta, local

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_one, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                stock_hits, meta, local = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{done:3d}/{len(symbols)}] {sym} err {exc}", flush=True)
                continue
            if meta["error"]:
                errors += 1
            hits.extend(stock_hits)
            lock_funnels.append(local)
            flag = f" trades={meta['n_trade']}" if meta["n_trade"] else ""
            err = f" {meta['error']}" if meta["error"] else ""
            print(f"[{done:3d}/{len(symbols)}] {sym} bars={meta['bars']}{flag}{err}", flush=True)

    for local in lock_funnels:
        for k, v in local.items():
            funnel[k] = funnel.get(k, 0) + v
    hits.sort(key=lambda h: (h.df.index[h.trade.entry_idx], h.symbol))
    stats = summarize_trades([h.trade for h in hits])
    print(
        f"done scanned={len(symbols)} errors={errors} trades={stats['count']} "
        f"WR={stats['win_rate']:.1f}% pnl%={stats['total_pct']*100:+.2f} funnel={funnel}",
        flush=True,
    )
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        ts = hit.df.index[t.entry_idx]
        print(
            f"  [{i}] {hit.symbol} {ts.strftime('%m-%d %H:%M')} {t.exit_reason} {t.pnl_pct*100:+.2f}%",
            flush=True,
        )

    extra = {
        "days": args.days,
        "symbols": symbols,
        "generated": datetime.now(TPE).isoformat(timespec="seconds"),
        "interval": "15m",
        "include_stocks": bool(args.include_stocks),
        "require_1h_ma25": not bool(args.no_1h_ma25),
    }
    html_path = Path(args.html) if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        label = f"{args.days}d · 幣安 U 永續 15m"
        if args.symbol:
            label += f" · {args.symbol.upper()}"
        if args.loose:
            label += " · 寬鬆（無實體/量能過濾）"
        if not args.include_stocks and not args.symbol.strip():
            label += " · 已濾股票"
        if not args.no_1h_ma25:
            label += " · 1h MA25下"
        out = write_html(
            html_path,
            hits,
            symbols,
            label,
            funnel=funnel,
            max_charts=args.max_charts,
            require_1h_ma25=not args.no_1h_ma25,
        )
        write_view_html(out)
        print(f"html={out}", flush=True)
    json_path = Path(args.json_path) if args.json_path else None
    if json_path is None and html_path:
        json_path = html_path.with_name("hits.json")
    if json_path:
        dump_hits_json(json_path, hits, stats, funnel, extra)
        print(f"json={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
