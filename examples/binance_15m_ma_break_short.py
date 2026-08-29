#!/usr/bin/env python3
"""幣安 15 分 K：同時跌破 MA99/120/200，且 7/14/25 空頭排列，做空回測一週。

對齊截圖 MUBARAK/USDT：一根大陰線從長均黏帶上沿打穿到三條之下，
同時短均 MA7 < MA14 < MA25。

    python3 examples/binance_15m_ma_break_short.py
    python3 examples/binance_15m_ma_break_short.py --symbol MUBARAKUSDT
    python3 examples/binance_15m_ma_break_short.py --pages
"""

from __future__ import annotations

import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TZ = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-15m-ma-short" / "index.html"
PAGES_30D = REPO / "docs" / "binance-15m-ma-short-30d" / "index.html"
PAGES_60D = REPO / "docs" / "binance-15m-ma-short-60d" / "index.html"
BRANCH_VIEW = "cursor/15m-ma-break-short-9d44"

INTERVAL = "15m"
BAR_MS = {"1m": 60_000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}
MA_SHORT = (7, 14, 25)
MA_LONG = (99, 120, 200)
WARMUP = 200
EVAL_DAYS = 7
MAX_HOLD = 32  # 8 小時
RR = 2.0
STOP_BUFFER = 0.001  # 停損在破位 K 高點上方 0.1%
MIN_BREAK_PCT = 0.8  # 收盤至少低於長均上沿 0.8%，濾掉輕觸
MIN_SHORT_FAN_PCT = 0.30  # 7/14/25 張開不到這個，要靠放量才算瀑布
MIN_VOL_IF_TIGHT_FAN = 3.2  # 短均幾乎黏住時，量比至少這麼多
MAX_1H_BULL_FAN_PCT = 2.0  # 上根小時仍 7>14>25 且張開≥2%，多半是漲勢回檔
APPROACH_BARS = 8  # 進場前幾根要多數還在長均之上
MIN_APPROACH_ABOVE = 4  # 少於這個＝在帶裡穿梭或已經破了再追
SKIP_MINUTES = frozenset({15})  # 整點後第 15 分＝小時中段，假跌破多
SKIP_HOURS = frozenset({17, 18, 21, 22, 23})  # 台北：美股盤前／開盤，股指永續容易假破
MIN_QV = 5_000_000
KEEP = ("MUBARAKUSDT",)
MAX_CHARTS = 150
KLINE_PAGE = 1500

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0",
        "Clienttype": "web",
        "Accept": "application/json",
    }
)
FAPI = ("https://fapi.binance.com", "https://www.binance.com")
SPOT = ("https://api.binance.com", "https://data-api.binance.vision")

MA_COLORS = {
    7: "#f0c14a",
    14: "#26c6da",
    25: "#d28cff",
    99: "#5c6bc0",
    120: "#43a047",
    200: "#8d6e63",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def get_json(url: str, params: dict | None = None, retries: int = 5) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.35 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _klines_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    idx = pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TZ)
    df = pd.DataFrame(
        {
            "open": [float(x[1]) for x in raw],
            "high": [float(x[2]) for x in raw],
            "low": [float(x[3]) for x in raw],
            "close": [float(x[4]) for x in raw],
            "volume": [float(x[5]) for x in raw],
        },
        index=idx,
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def kline_limit_needed(days: int, interval: str = INTERVAL) -> int:
    """回測窗口 + 均線暖身。15m 一天 96 根，1h 一天 24 根。"""
    per_day = 24 if interval == "1h" else 96
    return days * per_day + WARMUP + 48


def pages_html_path(days: int) -> Path:
    """週報 / 月報 / 兩個月報分開寫，避免互相覆蓋。"""
    if days >= 50:
        return PAGES_60D
    if days >= 28:
        return PAGES_30D
    return PAGES


def _fetch_klines_page(
    symbol: str,
    limit: int,
    *,
    interval: str,
    futures: bool,
    end_time_ms: int | None,
) -> pd.DataFrame:
    hosts = FAPI if futures else SPOT
    path = "/fapi/v1/klines" if futures else "/api/v3/klines"
    page = min(limit, 1000 if not futures else KLINE_PAGE)
    last: Exception | None = None
    for host in hosts:
        try:
            params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": page}
            if end_time_ms is not None:
                params["endTime"] = end_time_ms
            raw = get_json(host + path, params)
            return _klines_to_df(raw)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"klines page {symbol} {interval}: {last}")


def fetch_klines(
    symbol: str,
    limit: int = 1000,
    *,
    interval: str = INTERVAL,
    futures: bool = True,
) -> pd.DataFrame:
    """往回抓夠多根 K。幣安單次最多 1500，超過就分頁。"""
    parts: list[pd.DataFrame] = []
    got = 0
    end_time_ms: int | None = None
    last_err: Exception | None = None
    while got < limit:
        batch = min(KLINE_PAGE if futures else 1000, limit - got)
        try:
            df = _fetch_klines_page(
                symbol, batch, interval=interval, futures=futures, end_time_ms=end_time_ms
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            break
        if df.empty:
            break
        parts.append(df)
        got += len(df)
        first_open = int(df.index[0].tz_convert("UTC").timestamp() * 1000)
        end_time_ms = first_open - 1
        if len(df) < batch:
            break
    if not parts:
        if futures:
            return fetch_klines(symbol, limit=limit, interval=interval, futures=False)
        raise RuntimeError(f"klines {symbol} {interval}: {last_err}")
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    now_ms = int(time.time() * 1000)
    last_open = int(out.index[-1].tz_convert("UTC").timestamp() * 1000)
    if last_open + BAR_MS[interval] > now_ms:
        out = out.iloc[:-1]
    if len(out) > limit:
        out = out.iloc[-limit:]
    return out


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    """把 15m 合成 UTC 對齊的 1h（抓不到小時線時的備援）。"""
    if df.empty:
        return df
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    src = df[cols]
    utc = src.tz_convert("UTC") if src.index.tz is not None else src
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in src.columns:
        agg["volume"] = "sum"
    hourly = utc.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
    if hourly.index.tz is not None:
        hourly = hourly.tz_convert(TZ)
    return add_mas(hourly)


def _bar_index(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    if df.empty:
        return 0
    mark = ts
    if getattr(mark, "tzinfo", None) is not None and df.index.tz is not None:
        mark = mark.tz_convert(df.index.tz)
    pos = int(df.index.searchsorted(mark, side="right") - 1)
    return max(0, min(pos, len(df) - 1))


def last_closed_1h_idx(df_1h: pd.DataFrame, ts: pd.Timestamp) -> int:
    """進場當下還沒收盤的那根 1h 不能用，退回上一根已收盤小時 K。"""
    if df_1h is None or df_1h.empty:
        return -1
    mark = ts.tz_convert("UTC") if getattr(ts, "tzinfo", None) else pd.Timestamp(ts, tz="UTC")
    closed_open = mark.floor("1h") - pd.Timedelta(hours=1)
    if df_1h.index.tz is not None:
        closed_open = closed_open.tz_convert(df_1h.index.tz)
    pos = int(df_1h.index.searchsorted(closed_open, side="right") - 1)
    return pos


def filter_signals_1h(
    signals: list[Signal],
    df_1h: pd.DataFrame,
    funnel: dict[str, int] | None = None,
) -> list[Signal]:
    """小時確認：上一根已收盤 1h 還在 MA99 之上，這根 15m 收盤第一次打穿 1h MA99。

    回測裡「1h 已經跌破 MA99 再追空」勝率明顯較差；太晚。
    15m 大陰線若沒打到小時 MA99，多半只是 15m 均線自己的回抽。
    上根小時若仍明顯 7>14>25，比較像漲勢回檔，不像瀑布。
    """
    if not signals:
        return []
    work = add_mas(df_1h) if df_1h is not None and not df_1h.empty and "ma99" not in df_1h.columns else df_1h

    def bump(key: str) -> None:
        if funnel is not None:
            funnel[key] = funnel.get(key, 0) + 1

    kept: list[Signal] = []
    for sig in signals:
        if work is None or work.empty:
            bump("skip_1h_data")
            continue
        pos = last_closed_1h_idx(work, sig.timestamp)
        if pos < 0:
            bump("skip_1h_data")
            continue
        row = work.iloc[pos]
        m99 = row.get("ma99")
        if m99 is None or pd.isna(m99):
            bump("skip_1h_data")
            continue
        m99 = float(m99)
        prev_close = float(row["close"])
        if prev_close < m99:
            bump("skip_1h_late")
            continue
        if sig.entry >= m99:
            bump("skip_1h_shallow")
            continue
        m7, m14, m25 = row.get("ma7"), row.get("ma14"), row.get("ma25")
        if (
            m7 is not None
            and m14 is not None
            and m25 is not None
            and not (pd.isna(m7) or pd.isna(m14) or pd.isna(m25))
        ):
            m7, m14, m25 = float(m7), float(m14), float(m25)
            if m7 > m14 > m25 and m25 > 0:
                h_fan = (m7 / m25 - 1.0) * 100.0
                if h_fan >= MAX_1H_BULL_FAN_PCT:
                    bump("skip_1h_bull")
                    continue
        bump("taken_1h")
        kept.append(sig)
    return kept


def load_1h_frames(
    symbols: list[str],
    frames_15m: dict[str, pd.DataFrame],
    workers: int = 8,
    *,
    limit: int = 1000,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    uniq = [s for s in dict.fromkeys(symbols) if s]

    def one(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            return sym, add_mas(fetch_klines(sym, limit=limit, interval="1h"))
        except Exception:
            src = frames_15m.get(sym)
            return sym, resample_1h(src) if src is not None else pd.DataFrame()

    if not uniq:
        return out
    with ThreadPoolExecutor(min(workers, len(uniq))) as ex:
        for fut in as_completed([ex.submit(one, s) for s in uniq]):
            name, df = fut.result()
            out[name] = df
    return out


def universe(min_qv: float = MIN_QV) -> list[str]:
    info = None
    tickers = None
    last: Exception | None = None
    for host in FAPI:
        try:
            info = get_json(host + "/fapi/v1/exchangeInfo")
            tickers = {t["symbol"]: t for t in get_json(host + "/fapi/v1/ticker/24hr")}
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
    if info is None or tickers is None:
        raise RuntimeError(f"universe: {last}")
    out: list[str] = []
    for s in info["symbols"]:
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
        if qv < min_qv and sym not in KEEP:
            continue
        out.append(sym)
    for k in KEEP:
        if k not in out:
            out.append(k)
    return out


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    for n in MA_SHORT + MA_LONG:
        out[f"ma{n}"] = close.rolling(n, min_periods=n).mean()
    if "volume" in out.columns:
        out["vol20"] = out["volume"].rolling(20, min_periods=20).mean()
    return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    symbol: str
    bar_idx: int
    timestamp: pd.Timestamp
    entry: float
    stop: float
    target: float
    ma7: float
    ma14: float
    ma25: float
    ma99: float
    ma120: float
    ma200: float
    cluster_hi: float
    cluster_lo: float
    vol_ratio: float
    break_pct: float
    fan_pct: float


@dataclass
class TradeResult:
    signal: Signal
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_pct: float


def bar_local(ts: pd.Timestamp) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert(TZ)
    return pd.Timestamp(ts, tz=TZ)


def skip_noisy_tod(ts: pd.Timestamp) -> str | None:
    """回傳要濾掉的原因；好時段回 None。"""
    local = bar_local(ts)
    if int(local.minute) in SKIP_MINUTES:
        return "skip_minute"
    if int(local.hour) in SKIP_HOURS:
        return "skip_hour"
    return None


def detect_signals(
    df: pd.DataFrame,
    symbol: str = "",
    *,
    eval_start: pd.Timestamp | None = None,
    funnel: dict[str, int] | None = None,
    min_break_pct: float = MIN_BREAK_PCT,
) -> list[Signal]:
    """一根 15m 從長均黏帶之上同時打穿 MA99/120/200，且 MA7<MA14<MA25。"""
    if len(df) < WARMUP + 2:
        return []
    work = add_mas(df) if "ma200" not in df.columns else df
    c = work["close"].to_numpy(float)
    h = work["high"].to_numpy(float)
    m7 = work["ma7"].to_numpy(float)
    m14 = work["ma14"].to_numpy(float)
    m25 = work["ma25"].to_numpy(float)
    m99 = work["ma99"].to_numpy(float)
    m120 = work["ma120"].to_numpy(float)
    m200 = work["ma200"].to_numpy(float)
    vol = work["volume"].to_numpy(float) if "volume" in work.columns else np.ones(len(work))
    v20 = work["vol20"].to_numpy(float) if "vol20" in work.columns else np.full(len(work), np.nan)

    def bump(key: str) -> None:
        if funnel is not None:
            funnel[key] = funnel.get(key, 0) + 1

    signals: list[Signal] = []
    for i in range(WARMUP, len(work)):
        vals = [m7[i], m14[i], m25[i], m99[i], m120[i], m200[i], m99[i - 1], m120[i - 1], m200[i - 1]]
        if np.isnan(vals).any():
            continue
        if eval_start is not None and work.index[i] < eval_start:
            continue

        cluster_hi = max(m99[i], m120[i], m200[i])
        cluster_lo = min(m99[i], m120[i], m200[i])
        prev_hi = max(m99[i - 1], m120[i - 1], m200[i - 1])
        now_below = c[i] < m99[i] and c[i] < m120[i] and c[i] < m200[i]
        # 同時跌破：前收還在三條長均之上，本根一次收到三條之下
        prev_above_all = c[i - 1] > prev_hi
        if not (now_below and prev_above_all):
            continue
        break_pct = (cluster_hi - c[i]) / cluster_hi * 100.0
        if break_pct < min_break_pct:
            bump("skip_shallow")
            continue
        bump("break")

        if not (m7[i] < m14[i] < m25[i]):
            bump("skip_stack")
            continue

        entry = float(c[i])
        stop = float(h[i]) * (1.0 + STOP_BUFFER)
        risk = stop - entry
        if risk <= 0 or risk / entry > 0.18:
            bump("skip_risk")
            continue
        vr = float(vol[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        fan_pct = (m25[i] / m7[i] - 1.0) * 100.0
        # 短均黏成一條又沒放量：橫盤輕觸，不像瀑布陰線
        if fan_pct < MIN_SHORT_FAN_PCT and vr < MIN_VOL_IF_TIGHT_FAN:
            bump("skip_chop")
            continue
        above = 0
        for j in range(max(0, i - APPROACH_BARS), i):
            hi_j = max(m99[j], m120[j], m200[j])
            if not np.isnan(hi_j) and c[j] > hi_j:
                above += 1
        # 進場前沒騎在黏帶上：橫盤穿梭或已經破了再追，不像瀑布
        if above < MIN_APPROACH_ABOVE:
            bump("skip_weave")
            continue
        why = skip_noisy_tod(work.index[i])
        if why:
            bump(why)
            continue
        bump("taken")
        target = entry - RR * risk
        signals.append(
            Signal(
                symbol=symbol,
                bar_idx=i,
                timestamp=work.index[i],
                entry=entry,
                stop=stop,
                target=target,
                ma7=float(m7[i]),
                ma14=float(m14[i]),
                ma25=float(m25[i]),
                ma99=float(m99[i]),
                ma120=float(m120[i]),
                ma200=float(m200[i]),
                cluster_hi=float(cluster_hi),
                cluster_lo=float(cluster_lo),
                vol_ratio=vr,
                break_pct=(cluster_hi - entry) / cluster_hi * 100.0,
                fan_pct=float(fan_pct),
            )
        )
    return signals


def simulate(df: pd.DataFrame, signals: list[Signal], *, max_hold: int = MAX_HOLD) -> list[TradeResult]:
    if df.empty or not signals:
        return []
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    ma99 = (
        df["ma99"].to_numpy(float)
        if "ma99" in df.columns
        else df["close"].rolling(99, min_periods=99).mean().to_numpy(float)
    )
    results: list[TradeResult] = []
    busy_until = -1
    for sig in sorted(signals, key=lambda s: s.bar_idx):
        if sig.bar_idx <= busy_until:
            continue
        end = min(sig.bar_idx + max_hold, len(df) - 1)
        exit_idx = end
        exit_price = float(close[end])
        reason = "time_stop"
        for i in range(sig.bar_idx + 1, end + 1):
            if high[i] >= sig.stop:
                exit_idx = i
                exit_price = sig.stop
                reason = "stop_loss"
                break
            if low[i] <= sig.target:
                exit_idx = i
                exit_price = sig.target
                reason = "take_profit"
                break
            if not np.isnan(ma99[i]) and close[i] > ma99[i]:
                exit_idx = i
                exit_price = float(close[i])
                reason = "reclaim_ma99"
                break
        busy_until = exit_idx
        pnl = (sig.entry - exit_price) / sig.entry * 100.0
        results.append(
            TradeResult(
                signal=sig,
                exit_idx=exit_idx,
                exit_time=df.index[exit_idx],
                exit_price=exit_price,
                exit_reason=reason,
                pnl_pct=pnl,
            )
        )
    return results


def sequential_equity(
    trades: list[TradeResult],
    *,
    start: float = 100.0,
    leverage: float = 3.0,
) -> tuple[float, list[TradeResult], list[TradeResult]]:
    """全押槓桿、同時只能一單、平倉後才接下一個，複利滾權益。"""
    eq = start
    busy_until: pd.Timestamp | None = None
    taken: list[TradeResult] = []
    skipped: list[TradeResult] = []
    for t in sorted(trades, key=lambda x: (x.signal.timestamp, x.signal.symbol)):
        if busy_until is not None and t.signal.timestamp < busy_until:
            skipped.append(t)
            continue
        eq *= 1.0 + leverage * t.pnl_pct / 100.0
        taken.append(t)
        busy_until = t.exit_time
    return eq, taken, skipped


def summarize(trades: list[TradeResult]) -> dict[str, float | int]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    total = sum(t.pnl_pct for t in trades)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": 100.0 * wins / len(trades),
        "total_pnl": total,
        "avg_pnl": total / len(trades),
    }


# ---------------------------------------------------------------------------
# Charts + HTML
# ---------------------------------------------------------------------------


def _setup_font() -> None:
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


def _paint_ohlcv(
    ax,
    axv,
    df: pd.DataFrame,
    start: int,
    end: int,
    *,
    entry_idx: int | None = None,
    exit_idx: int | None = None,
    entry_price: float | None = None,
    exit_price: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    mark_entry: bool = False,
    pnl_pct: float = 0.0,
) -> None:
    from matplotlib.patches import Rectangle

    window = df.iloc[start : end + 1]
    if window.empty:
        return
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    close_full = df["close"].astype(float)
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
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 25 else 1.05, label=f"MA{n}")
    if stop is not None:
        ax.axhline(stop, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    if target is not None:
        ax.axhline(target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    if entry_idx is not None:
        ex = entry_idx - start
        if 0 <= ex < len(window):
            ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
            if entry_price is not None:
                ax.scatter([ex], [entry_price], s=42, color="#ff5252", marker="v", zorder=6)
            if mark_entry:
                ax.annotate(
                    "做空",
                    (ex, entry_price if entry_price is not None else float(c.iloc[ex])),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="#ff8a80",
                    fontsize=8,
                )
    if exit_idx is not None:
        xx = exit_idx - start
        if 0 <= xx < len(window):
            ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
            if exit_price is not None:
                ax.scatter(
                    [xx],
                    [exit_price],
                    s=40,
                    color="#00c805" if pnl_pct > 0 else "#ff5252",
                    marker="x",
                    zorder=6,
                )
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")


def hourly_snapshot(df_1h: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float | str]:
    if df_1h is None or df_1h.empty:
        return {}
    work = add_mas(df_1h) if "ma200" not in df_1h.columns else df_1h
    i = _bar_index(work, ts)
    row = work.iloc[i]
    out: dict[str, float | str] = {"time": work.index[i].strftime("%m-%d %H:%M")}
    for n in MA_SHORT + MA_LONG:
        val = row.get(f"ma{n}")
        if val is not None and not pd.isna(val):
            out[f"ma{n}"] = float(val)
    m7, m14, m25 = out.get("ma7"), out.get("ma14"), out.get("ma25")
    if isinstance(m7, float) and isinstance(m14, float) and isinstance(m25, float):
        out["stack"] = "7<14<25" if m7 < m14 < m25 else "短均未空頭"
    return out


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    df_1h: pd.DataFrame | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_font()
    sig = trade.signal
    hourly = df_1h if df_1h is not None and not df_1h.empty else resample_1h(df)
    have_1h = hourly is not None and not hourly.empty

    if have_1h:
        fig, (ax, axv, axh, axhv) = plt.subplots(
            4,
            1,
            figsize=(10.4, 9.5),
            gridspec_kw={"height_ratios": [3.05, 0.82, 3.05, 0.82]},
            facecolor="#0c1210",
        )
        _style_axes((ax, axv, axh, axhv))
    else:
        fig, (ax, axv) = plt.subplots(
            2,
            1,
            figsize=(10.4, 5.6),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1]},
            facecolor="#0c1210",
        )
        _style_axes((ax, axv))

    start = max(0, sig.bar_idx - 36)
    end = min(len(df) - 1, max(trade.exit_idx + 10, sig.bar_idx + 16))
    _paint_ohlcv(
        ax,
        axv,
        df,
        start,
        end,
        entry_idx=sig.bar_idx,
        exit_idx=trade.exit_idx,
        entry_price=sig.entry,
        exit_price=trade.exit_price,
        stop=sig.stop,
        target=sig.target,
        mark_entry=True,
        pnl_pct=trade.pnl_pct,
    )

    et = sig.timestamp
    xt = trade.exit_time
    if hasattr(et, "tz_convert"):
        et = et.tz_convert(TZ)
        xt = xt.tz_convert(TZ)
    sign = "+" if trade.pnl_pct >= 0 else ""
    ax.set_title(
        f"#{trade_no}  {sig.symbol}  15m  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_pct:.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)

    if have_1h:
        hi = _bar_index(hourly, sig.timestamp)
        hx = _bar_index(hourly, trade.exit_time)
        h0 = max(0, hi - 48)
        h1 = min(len(hourly) - 1, max(hx + 12, hi + 16))
        _paint_ohlcv(
            axh,
            axhv,
            hourly,
            h0,
            h1,
            entry_idx=hi,
            exit_idx=hx,
            entry_price=sig.entry,
            exit_price=trade.exit_price,
            stop=sig.stop,
            target=sig.target,
            mark_entry=True,
            pnl_pct=trade.pnl_pct,
        )
        axh.set_title("1h 對照（同一進出場時刻）", color="#e8f0ea", fontsize=11)
        axh.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)

    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _equity_svg(pnls: list[float]) -> str:
    if not pnls:
        return ""
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    w, h = 720, 180
    lo, hi = min(eq), max(eq)
    span = hi - lo or 1.0
    pts = []
    for i, v in enumerate(eq):
        x = i / (len(eq) - 1) * w if len(eq) > 1 else 0
        y = h - 16 - (v - lo) / span * (h - 32)
        pts.append(f"{x:.1f},{y:.1f}")
    zero_y = h - 16 - (0 - lo) / span * (h - 32)
    color = "#16a34a" if eq[-1] >= 0 else "#ef4444"
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero_y:.1f}" x2="{w}" y2="{zero_y:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>'
        f"</svg>"
    )


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fmt_px(px: float) -> str:
    if px >= 100:
        return f"{px:.2f}"
    if px >= 1:
        return f"{px:.4f}"
    return f"{px:.6f}".rstrip("0").rstrip(".")


def _render_cards(
    trades: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    html_path: Path,
    *,
    embed: bool,
    prefix: str = "t",
    frames_1h: dict[str, pd.DataFrame] | None = None,
) -> str:
    cards: list[str] = []
    img_dir = html_path.parent / "img"
    hourly_map = frames_1h or {}
    for i, t in enumerate(trades, 1):
        sig = t.signal
        df = frames[sig.symbol]
        df_1h = hourly_map.get(sig.symbol)
        et = sig.timestamp.tz_convert(TZ) if getattr(sig.timestamp, "tzinfo", None) else sig.timestamp
        xt = t.exit_time.tz_convert(TZ) if getattr(t.exit_time, "tzinfo", None) else t.exit_time
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        reason_cls = {
            "take_profit": "tag-tp",
            "stop_loss": "tag-sl",
            "reclaim_ma99": "tag-time",
            "time_stop": "tag-time",
        }.get(t.exit_reason, "tag-time")
        img_name = f"{prefix}{i:02d}_{sig.symbol}_{et.strftime('%m%d_%H%M')}.png"
        chart_html = ""
        if i <= MAX_CHARTS:
            png = draw_trade_png(df, t, img_dir / img_name, i, df_1h=df_1h)
            src = _img_data_uri(png) if embed else f"img/{img_name}"
            chart_html = (
                f"<div class='mini-chart'><img src='{escape(src)}' alt='#{i} {escape(sig.symbol)} 15m+1h' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        risk_pct = (sig.stop - sig.entry) / sig.entry * 100.0
        snap = hourly_snapshot(df_1h if df_1h is not None else resample_1h(df), sig.timestamp)
        h_line = ""
        if snap:
            def _ma_line(periods: tuple[int, ...], prefix: str) -> str:
                bits = [f"MA{n} {_fmt_px(float(snap[f'ma{n}']))}" for n in periods if f"ma{n}" in snap]
                return f"{prefix} {' / '.join(bits)}" if bits else ""

            h_short = _ma_line(MA_SHORT, "1h")
            if snap.get("stack"):
                h_short = f"{h_short}  {snap['stack']}".strip()
            h_long = _ma_line(MA_LONG, "1h")
            body = "\n".join(x for x in (h_short, h_long) if x)
            h_line = f"\n1h {snap.get('time', '')}\n{body}" if body else ""
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(sig.symbol)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))} 台北</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            "<span class='tag tag-info'>15m 做空</span>"
            "<span class='tag tag-info'>1h 對照</span>"
            f"<span class='tag tag-info'>量比 {sig.vol_ratio:.1f}x</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {_fmt_px(sig.entry)}\n"
            f"stop  {_fmt_px(sig.stop)}  (風險 {risk_pct:.2f}%)\n"
            f"target {_fmt_px(sig.target)}  ({RR:.1f}R)\n"
            f"exit  {_fmt_px(t.exit_price)}  {t.exit_reason}\n"
            f"打穿長均 {sig.break_pct:.2f}%  cluster {_fmt_px(sig.cluster_lo)}–{_fmt_px(sig.cluster_hi)}\n"
            f"15m MA7 {_fmt_px(sig.ma7)} < MA14 {_fmt_px(sig.ma14)} < MA25 {_fmt_px(sig.ma25)}  張開 {sig.fan_pct:.2f}%\n"
            f"15m MA99 {_fmt_px(sig.ma99)} / MA120 {_fmt_px(sig.ma120)} / MA200 {_fmt_px(sig.ma200)}\n"
            "1h 確認：上根小時收盤仍在 MA99 上，本 15m 第一次打穿 1h MA99"
            f"{h_line}"
            "</pre>"
            f"{chart_html}"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: Path,
    trades: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    funnel: dict[str, int] | None = None,
    embed: bool = False,
    mubarak_trades: list[TradeResult] | None = None,
    frames_1h: dict[str, pd.DataFrame] | None = None,
) -> Path:
    stats = summarize(trades)
    total_cls = "pnl-win" if float(stats["total_pnl"]) >= 0 else "pnl-loss"
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：從三條之上一次打穿　{funnel.get('break', 0)} → "
            f"7&lt;14&lt;25　{funnel.get('taken', 0)} → "
            f"1h 首次打穿 MA99　{funnel.get('taken_1h', 0)}"
            f"（淺破 {funnel.get('skip_shallow', 0)} · 短均未排列 {funnel.get('skip_stack', 0)} · "
            f"橫盤輕觸 {funnel.get('skip_chop', 0)} · 沒騎在黏帶上 {funnel.get('skip_weave', 0)} · "
            f"小時中段 :15 {funnel.get('skip_minute', 0)} · 美股時段 {funnel.get('skip_hour', 0)} · "
            f"風險過大 {funnel.get('skip_risk', 0)} · "
            f"1h 已破 MA99 太晚 {funnel.get('skip_1h_late', 0)} · "
            f"15m 沒打到 1h MA99 {funnel.get('skip_1h_shallow', 0)} · "
            f"1h 仍多頭張開 {funnel.get('skip_1h_bull', 0)}）</p>"
        )
    extra = ""
    if mubarak_trades is not None:
        ms = summarize(mubarak_trades)
        mcls = "pnl-win" if float(ms["total_pnl"]) >= 0 else "pnl-loss"
        extra = (
            "<section class='summary'><h1>MUBARAKUSDT（截圖標的）</h1>"
            f"<p class='muted'>同一套規則，只看圖上那檔。</p>"
            f"<div class='cards'><div class='card'>筆數<b>{ms['trades']}</b></div>"
            f"<div class='card'>勝率<b>{ms['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總報酬<b class='{mcls}'>{ms['total_pnl']:+.2f}%</b></div>"
            f"<div class='card'>勝/負<b>{ms['wins']}/{ms['losses']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_pct for t in mubarak_trades])}</div></section>"
            + (
                _render_cards(
                    mubarak_trades, frames, path, embed=embed, prefix="m", frames_1h=frames_1h
                )
                or "<div class='empty'>這一週 MUBARAK 沒打出同時跌破＋空頭排列</div>"
            )
        )
    cards = _render_cards(trades, frames, path, embed=embed, prefix="t", frames_1h=frames_1h)
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
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
<h1>{escape(title)}</h1>
<p class="muted">{escape(subtitle)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['trades']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總報酬<b class="{total_cls}">{stats['total_pnl']:+.2f}%</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['losses']}</b></div>
</div>
<p class="muted">停損＝破位 K 高點 +0.1% · 停利 2R · 收復 MA99 或持倉 {MAX_HOLD} 根（8h）平倉。濾掉短均幾乎黏住又沒放量的橫盤輕觸、進場前沒騎在黏帶上的穿梭／追空、整點後第 15 分（小時中段雜訊）、台北 17–18／21–23 點（美股盤前後假跌破），以及上根小時仍明顯 7&gt;14&gt;25 的漲勢回檔。小時過濾：上根已收盤 1h 還在 MA99 之上，這根 15m 收盤第一次跌破 1h MA99。上圖 15m、下圖 1h。報酬是單筆價格百分比，未計資金費。</p>
{funnel_line}
<div class="equity">{_equity_svg([t.pnl_pct for t in trades])}</div>
</section>
{extra}
{cards or "<div class='empty'>這一週沒有同時跌破＋空頭排列的做空訊號</div>"}
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = BRANCH_VIEW) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    trades: list[TradeResult] = field(default_factory=list)
    funnel: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    errors: int = 0


def eval_start_ts(days: int, now: datetime | None = None) -> pd.Timestamp:
    now = now or datetime.now(TZ)
    return pd.Timestamp(now - timedelta(days=days))


def scan_symbol(sym: str, *, days: int, limit: int) -> tuple[str, pd.DataFrame, list[TradeResult], dict[str, int]]:
    df = fetch_klines(sym, limit=limit)
    df = add_mas(df)
    funnel: dict[str, int] = {}
    start = eval_start_ts(days)
    sigs = detect_signals(df, sym, eval_start=start, funnel=funnel)
    if sigs:
        try:
            df_1h = add_mas(fetch_klines(sym, limit=kline_limit_needed(days, "1h"), interval="1h"))
        except Exception:
            df_1h = resample_1h(df)
        sigs = filter_signals_1h(sigs, df_1h, funnel)
    trades = simulate(df, sigs)
    return sym, df, trades, funnel


def scan_many(symbols: list[str], *, days: int, limit: int, workers: int = 10) -> ScanResult:
    out = ScanResult()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(scan_symbol, s, days=days, limit=limit): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                name, df, trades, funnel = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"err {sym} {exc}", flush=True)
                out.errors += 1
                continue
            out.symbols += 1
            out.frames[name] = df
            out.trades.extend(trades)
            for k, v in funnel.items():
                out.funnel[k] = out.funnel.get(k, 0) + v
            print(f"  {name} bars={len(df)} trades={len(trades)}", flush=True)
    out.trades.sort(key=lambda t: (t.signal.timestamp, t.signal.symbol))
    return out


def print_summary(label: str, trades: list[TradeResult], funnel: dict[str, int] | None = None) -> None:
    stats = summarize(trades)
    print(
        f"{label}: trades={stats['trades']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_pnl']:+.2f}% avg={stats['avg_pnl']:+.2f}%"
    )
    if funnel:
        print(
            f"  funnel break={funnel.get('break', 0)} taken={funnel.get('taken', 0)} "
            f"1h={funnel.get('taken_1h', 0)} "
            f"skip_shallow={funnel.get('skip_shallow', 0)} "
            f"skip_stack={funnel.get('skip_stack', 0)} skip_risk={funnel.get('skip_risk', 0)} "
            f"skip_chop={funnel.get('skip_chop', 0)} "
            f"skip_weave={funnel.get('skip_weave', 0)} "
            f"skip_minute={funnel.get('skip_minute', 0)} "
            f"skip_hour={funnel.get('skip_hour', 0)} "
            f"skip_1h_late={funnel.get('skip_1h_late', 0)} "
            f"skip_1h_shallow={funnel.get('skip_1h_shallow', 0)} "
            f"skip_1h_bull={funnel.get('skip_1h_bull', 0)}"
        )
    for i, t in enumerate(trades, 1):
        et = t.signal.timestamp.tz_convert(TZ) if getattr(t.signal.timestamp, "tzinfo", None) else t.signal.timestamp
        xt = t.exit_time.tz_convert(TZ) if getattr(t.exit_time, "tzinfo", None) else t.exit_time
        print(
            f"  [{i}] {t.signal.symbol} {et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_pct:+.2f}%"
        )
    if trades:
        eq, taken, skipped = sequential_equity(trades)
        print(
            f"  100 USDT × 3x 不能同時持倉：做 {len(taken)} 筆、略過 {len(skipped)} → "
            f"{eq:.2f} USDT ({eq - 100.0:+.2f})"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="幣安 15m 同時跌破 99/120/200 + 短均空頭排列 做空回測")
    p.add_argument("--symbol", default="", help="只回測單一標的，例如 MUBARAKUSDT")
    p.add_argument("--days", type=int, default=EVAL_DAYS)
    p.add_argument("--limit", type=int, default=1000, help="15m K 根數（含均線暖身）")
    p.add_argument("--min-qv", type=float, default=MIN_QV)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--html", default="")
    p.add_argument(
        "--pages",
        action="store_true",
        help="寫到 docs/binance-15m-ma-short/（28–49 日寫 30d，50 日以上寫 60d）",
    )
    p.add_argument("--embed", action="store_true", help="圖用 base64 嵌進 HTML")
    args = p.parse_args(argv)

    if args.symbol:
        symbols = [args.symbol.upper().replace("/", "")]
        if not symbols[0].endswith("USDT"):
            symbols[0] += "USDT"
    else:
        print("載入標的…", flush=True)
        symbols = universe(args.min_qv)
        print(f"掃描 {len(symbols)} 個 USDT 永續（含 {', '.join(KEEP)}）", flush=True)

    limit = max(args.limit, kline_limit_needed(args.days, "15m"))
    t0 = time.time()
    result = scan_many(symbols, days=args.days, limit=limit, workers=args.workers)
    print(f"掃完 {result.symbols} 檔 用 {time.time() - t0:.1f}s  失敗 {result.errors}", flush=True)
    print_summary("全市場", result.trades, result.funnel)

    mubarak = [t for t in result.trades if t.signal.symbol == "MUBARAKUSDT"]
    if not args.symbol or args.symbol.upper().replace("/", "") in {"MUBARAK", "MUBARAKUSDT"}:
        if "MUBARAKUSDT" in result.frames and not mubarak:
            print_summary("MUBARAKUSDT", [], result.funnel if args.symbol else None)
        elif mubarak:
            print_summary("MUBARAKUSDT", mubarak)

    html_path = Path(args.html) if args.html else None
    if args.pages:
        html_path = pages_html_path(args.days)
    if html_path:
        now = datetime.now(TZ)
        start = eval_start_ts(args.days, now)
        title = "15m 同時跌破 99/120/200 做空"
        subtitle = (
            f"{args.days} 日 · {start.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} 台北 · "
            f"{result.symbols} 檔 15m · 進場＝15m 同時跌破 99/120/200 且 7<14<25，"
            f"短均黏住要放量、進場前多數還在長均之上、避開 :15 與美股時段，再加 1h：上根小時收盤仍在 MA99 上、本 15m 第一次打穿 1h MA99，"
            f"且上根小時不是明顯 7>14>25"
        )
        show_mubarak = not args.symbol or "MUBARAK" in args.symbol.upper()
        need_1h = sorted({t.signal.symbol for t in result.trades} | ({"MUBARAKUSDT"} if show_mubarak else set()))
        print(f"抓 1h 對照 {len(need_1h)} 檔…", flush=True)
        frames_1h = load_1h_frames(
            need_1h, result.frames, limit=kline_limit_needed(args.days, "1h")
        )
        write_html_report(
            html_path,
            result.trades,
            result.frames,
            title=title,
            subtitle=subtitle,
            funnel=result.funnel,
            embed=args.embed,
            mubarak_trades=mubarak if show_mubarak and not args.symbol else None,
            frames_1h=frames_1h,
        )
        view = write_view_html(html_path)
        print(f"html={html_path}")
        print(f"view={view}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
