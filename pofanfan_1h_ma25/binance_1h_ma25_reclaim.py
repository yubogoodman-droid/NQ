#!/usr/bin/env python3
"""幣安 1 小時 K：MA25 下做出筆畫 W，站回後回測再進。

對齊手機手畫那種走法：先一腳下跌 → 反彈吻 MA25 → 第二腳破前低 →
急拉收盤站回 MA25 → 離開均線後再回測 MA25。進場用回測那根收盤，
不是站回那根；沒回測就放棄。停損＝收盤跌破破底低點（影線不算）。出場預設五分 M 破頸線。不再用連兩根掉 MA25。最長持有 36h。

用法:
  python3 examples/binance_1h_ma25_reclaim.py --symbols AVGOUSDT,ONDSUSDT --days 45
  python3 examples/binance_1h_ma25_reclaim.py --strict --limit 80 --days 30 --pages
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
_REST_BANNED = False
VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

# 對齊你一開始那兩張幣安手機圖的均線帶：
# 黃 MA7 / 青 MA14 / 粉 MA25 / 紫 MA99 / 綠 MA120 / 酒紅 MA200
MA_COLORS = {
    7: "#f0b90b",
    14: "#4dd0e1",
    25: "#d28cff",
    99: "#7e57c2",
    120: "#66bb6a",
    200: "#c62828",
}
MA_WIDTH = {7: 1.15, 14: 1.05, 25: 2.15, 99: 1.25, 120: 1.15, 200: 1.25}
W_COLOR = "#7A4A2A"
# 咖啡色 W 筆畫（1h + 4h 卡片）。False＝卡片不畫；CLI --no-w-overlay。
DRAW_W_OVERLAY = True
# 反向 / 放空槓桿 ETF 股票永續（幣安 exchangeInfo 無標的名稱欄位 → 手動清單；2026-09 TradFi 191 檔核對）。
# --exclude-inverse 時從掃描清單拿掉：做多反向產品＝放空標的，W 破底翻邏輯反了。
INVERSE_SYMBOLS = frozenset({
    "SQQQUSDT",  # -3x Nasdaq-100
    "SOXSUSDT",  # -3x 半導體
    "SKDDUSDT",  # 放空 SK hynix 槓桿
    "TBTUSDT",   # -2x 20Y+ 美債
    "TZAUSDT",   # -3x Russell 2000
    "NVDUSDT",   # -2x NVDA (GraniteShares 2x Short NVDA Daily ETF on BingX)
})
# 波動率商品（做多 VIX 期貨，對股市反向、長期衰減）；--exclude-vol 另外拿掉。
VOL_SYMBOLS = frozenset({"UVXYUSDT"})
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
# 五分 M 破頸線（對齊龍蝦 08-26 08:15）
M5_MIN_PEAK_SEP = 3
M5_MAX_PEAK_SEP = 24
M5_RIGHT_OVERSHOOT = 0.003
M5_MIN_NECK_DEPTH = 0.015
M5_HH_TOL = 0.003
M5_ARM_R = 1.5
M5_MIN_BREAK_PCT = 0.01


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
    break_vol_ratio: float = 0.0
    impulse_pct: float = 0.0
    undercut_pct: float = 0.0
    flush_atr: float = 0.0
    quality: str = "C"
    quality_score: int = 0
    ma7: float = 0.0
    ma14: float = 0.0
    ma99: float = 0.0
    ma200: float = 0.0
    w_start_idx: int = -1
    w_left_idx: int = -1
    w_peak_idx: int = -1
    w_right_idx: int = -1
    w_arm_idx: int = -1
    reclaim_idx: int = -1
    decline_pct: float = 0.0
    flat_bars: int = -1


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
    exit_ts_5m: Optional[Any] = None


@dataclass
class Hit:
    symbol: str
    df: pd.DataFrame
    trade: TradeResult
    quote_volume: float = 0.0


@dataclass
class M5NeckExit:
    fill_ts: Any
    fill_price: float
    left_ts: Any
    trough_ts: Any
    right_ts: Any
    neckline: float
    left_high: float
    right_high: float


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


def break_vol_ratio(volume: np.ndarray, bottom_idx: int, lookback: int = 20) -> float:
    """volume[bottom] / mean(volume[bottom-N:bottom]); exclude the bottom bar itself."""
    if bottom_idx <= 0 or lookback <= 0:
        return 1.0
    left = max(0, bottom_idx - int(lookback))
    avg = float(np.mean(volume[left:bottom_idx]) or 1.0)
    if avg <= 0:
        return 1.0
    return float(volume[bottom_idx] / avg)


def ma200_near_ok(low: float, ma200_val: float, near_pct: float) -> bool:
    """破底在 MA200 之下或附近。MA200 無效（NaN / ≤0）則否。near_pct 為允許高出的 %（0=必須 ≤）。"""
    if not np.isfinite(ma200_val) or ma200_val <= 0:
        return False
    return float(low) <= float(ma200_val) * (1.0 + float(near_pct) / 100.0)


def ma_cluster_sep_atr(ma25_val: float, ma200_val: float, atr_val: float) -> float:
    """破底 |MA200-MA25| / ATR。太小＝長均線貼在 MA25 上（六線糾成一坨）。"""
    if (
        not np.isfinite(ma25_val)
        or not np.isfinite(ma200_val)
        or not np.isfinite(atr_val)
        or atr_val <= 0
    ):
        return float("nan")
    return abs(float(ma200_val) - float(ma25_val)) / float(atr_val)


def ma_cluster_ok(ma25_val: float, ma200_val: float, atr_val: float, min_sep: float) -> bool:
    """長均線不要塌在 MA25 上。min_sep<=0 關閉。MA / ATR 無效則否。"""
    if min_sep <= 0:
        return True
    sep = ma_cluster_sep_atr(ma25_val, ma200_val, atr_val)
    if not np.isfinite(sep):
        return False
    return sep >= float(min_sep)


def mid_ma_span_atr(ma25_val: float, ma99_val: float, ma120_val: float, atr_val: float) -> float:
    """進場 (max-min of MA25/99/120) / ATR。太小＝粉紫綠糾成一坨。"""
    vals = (ma25_val, ma99_val, ma120_val)
    if any(not np.isfinite(v) for v in vals) or not np.isfinite(atr_val) or atr_val <= 0:
        return float("nan")
    return (max(float(v) for v in vals) - min(float(v) for v in vals)) / float(atr_val)


def mid_ma_cluster_ok(
    ma25_val: float,
    ma99_val: float,
    ma120_val: float,
    atr_val: float,
    min_span: float,
) -> bool:
    """進場 MA25 不要坐在 MA99/120 結裡。min_span<=0 關閉。MA / ATR 無效則否。"""
    if min_span <= 0:
        return True
    span = mid_ma_span_atr(ma25_val, ma99_val, ma120_val, atr_val)
    if not np.isfinite(span):
        return False
    return span >= float(min_span)


def overhead_knot_span_atr(
    ma60_val: float,
    ma99_val: float,
    ma120_val: float,
    close: float,
    atr_val: float,
) -> float:
    """MA60/99/120 全部嚴格在收盤之上時，回傳三線帶寬/ATR；否則 nan。

    帶寬很小＝頭上壓著一團中期均線（進場買在天花板裡）。
    三線散開很寬＝空頭排列的長均線，不是糾結，不當成這個濾網。
    """
    vals = (ma60_val, ma99_val, ma120_val)
    if (
        any(not np.isfinite(v) for v in vals)
        or not np.isfinite(close)
        or not np.isfinite(atr_val)
        or atr_val <= 0
    ):
        return float("nan")
    if any(float(v) <= float(close) for v in vals):
        return float("nan")
    return (max(float(v) for v in vals) - min(float(v) for v in vals)) / float(atr_val)


def ribbon_median_atr(
    ma7: np.ndarray,
    ma14: np.ndarray,
    ma25: np.ndarray,
    atr14: np.ndarray,
    start: int,
    end: int,
) -> float:
    """跌破期間 MA7/14/25 帶寬中位數，單位 ATR。太小＝均線糾結成一條。"""
    if end < start or start < 0:
        return float("nan")
    sl = slice(start, end + 1)
    stack = np.vstack((ma7[sl], ma14[sl], ma25[sl]))
    width = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    a = atr14[sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = width / a
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def decline_before_break(high: np.ndarray, start: int, bottom_idx: int, bottom: float, lookback: int = 24) -> float:
    """跌破前一段真的有跌：max(high[start-lookback : bottom]) / 破底低 → 跌幅（分數）。
    start＝第一根收在 MA25 下。一根影線刺破但前面沒跌（TBT/HK0625/VST 那種 2~3%）會很小。"""
    if bottom <= 0 or bottom_idx < 0:
        return 0.0
    left = max(0, int(start) - int(lookback))
    peak = float(np.max(high[left : bottom_idx + 1]))
    if peak <= 0:
        return 0.0
    return (peak - float(bottom)) / peak


def flat_bar_count(
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: int,
    *,
    window: int = 24,
    ref_bars: int = 120,
    frac: float = 0.35,
) -> int:
    """進場前 window 根（含進場棒）裡，高低差 < frac × 前 ref_bars 根高低差中位數 的根數。
    股票永續休市/週末那種一字線死盤（COIN 週六、TBT 週日）會很多；金樣本 0。"""
    if entry_idx < 1:
        return 0
    rng = high - low
    ref = rng[max(0, entry_idx - int(ref_bars)) : entry_idx]
    if ref.size == 0:
        return 0
    med = float(np.median(ref))
    if not (med > 0):
        return 0
    seg = rng[max(0, entry_idx - int(window) + 1) : entry_idx + 1]
    return int(np.sum(seg < float(frac) * med))


def get_json(path: str, params=None, retries: int = 6) -> Any:
    global _REST_BANNED
    last: Exception | None = None
    if _REST_BANNED:
        raise RuntimeError("GET banned")
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code in (418, 429, 403):
                last = RuntimeError(f"HTTP {r.status_code} {r.text[:80]}")
                if r.status_code == 418 and "banned" in (r.text or "").lower():
                    _REST_BANNED = True
                    break
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.45 * (i + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def universe(min_quote_volume: float = 8_000_000, extra: Sequence[str] = ()) -> List[str]:
    """股票永續清單，依 24h 成交額由高到低。min_quote_volume 在股票模式忽略。"""
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    rows: List[tuple] = []
    for s in info.get("symbols") or []:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "TRADIFI_PERPETUAL":
            continue
        if s.get("underlyingType") not in ("EQUITY", "HK_EQUITY", "KR_EQUITY", "CN_EQUITY"):
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        rows.append((qv, sym))
    rows.sort(reverse=True)
    out = [sym for _, sym in rows]
    for sym in extra:
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
    """Pull klines. REST first; data.binance.vision daily zips if IP-banned."""
    if not _REST_BANNED:
        try:
            df = _fetch_klines_rest(symbol, interval=interval, days=days, drop_forming=drop_forming)
            if df is not None and not df.empty:
                return df
        except Exception:  # noqa: BLE001
            pass
    return _fetch_klines_vision(symbol, interval=interval, days=days, drop_forming=drop_forming)


def _fetch_klines_rest(symbol: str, interval: str = "1h", days: int = 45, drop_forming: bool = True) -> pd.DataFrame:
    bar_ms = int(INTERVAL_MS.get(interval, 3_600_000))
    bars_per_day = max(1, int(86_400_000 / bar_ms))
    need = max(80, int(days) * bars_per_day + 40)
    raw: list = []
    end_ms: Optional[int] = None
    now_ms = int(time.time() * 1000)
    while len(raw) < need:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(1500, need - len(raw)),
        }
        if end_ms is not None:
            params["endTime"] = end_ms
        chunk = get_json("/fapi/v1/klines", params=params)
        if not chunk:
            break
        raw = list(chunk) + raw
        end_ms = int(chunk[0][0]) - 1
        if len(chunk) < int(params["limit"]):
            break
        time.sleep(0.08)
    if not raw:
        return _klines_to_df([])
    if drop_forming and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    df = _klines_to_df(raw)
    if df.empty:
        return df
    cutoff = datetime.now(TPE) - timedelta(days=days)
    return df.loc[df.index >= cutoff].copy()


def _vision_zip_url(symbol: str, interval: str, day) -> str:
    from urllib.parse import quote
    sym = quote(symbol, safe="")
    ds = day.strftime("%Y-%m-%d")
    return f"{VISION_BASE}/{sym}/{interval}/{sym}-{interval}-{ds}.zip"


def _parse_vision_csv(blob: bytes) -> list:
    import csv
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(blob))
    name = z.namelist()[0]
    text = z.read(name).decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    if rows[0] and rows[0][0].lower().startswith("open"):
        rows = rows[1:]
    out = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            out.append([int(float(row[0])), row[1], row[2], row[3], row[4], row[5]])
        except (TypeError, ValueError):
            continue
    return out


VISION_CACHE = Path("/tmp/binance-vision-cache")


def _vision_zip_bytes(symbol: str, interval: str, day) -> bytes | None:
    """Daily Vision zip, cached on disk. Does not invent bars."""
    ds = day.strftime("%Y-%m-%d")
    cache_dir = VISION_CACHE / symbol / interval
    cache_path = cache_dir / f"{symbol}-{interval}-{ds}.zip"
    today = datetime.now(TPE).date()
    if day < today and cache_path.is_file() and cache_path.stat().st_size > 64:
        blob = cache_path.read_bytes()
        if blob[:2] == b"PK":
            return blob
    url = _vision_zip_url(symbol, interval, day)
    try:
        r = SESSION.get(url, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code == 200 and r.content[:2] == b"PK":
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(r.content)
        except Exception:  # noqa: BLE001
            pass
        return r.content
    if r.status_code not in (404, 403):
        time.sleep(0.2)
    return None


def _fetch_klines_vision(symbol: str, interval: str = "1h", days: int = 45, drop_forming: bool = True) -> pd.DataFrame:
    bar_ms = int(INTERVAL_MS.get(interval, 3_600_000))
    now_ms = int(time.time() * 1000)
    end_d = datetime.now(TPE).date()
    start_d = end_d - timedelta(days=int(days) + 1)
    raw: list = []
    d = start_d
    while d <= end_d:
        blob = _vision_zip_bytes(symbol, interval, d)
        if blob:
            try:
                raw.extend(_parse_vision_csv(blob))
            except Exception:  # noqa: BLE001
                pass
        d += timedelta(days=1)
    if not raw:
        return _klines_to_df([])
    seen = {int(x[0]): x for x in raw}
    raw = [seen[k] for k in sorted(seen)]
    if drop_forming and raw and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    df = _klines_to_df(raw)
    if df.empty:
        return df
    cutoff = datetime.now(TPE) - timedelta(days=days)
    return df.loc[df.index >= cutoff].copy()



def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """1h → 4h，對齊幣安 UTC 整點開盤。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    utc = df.copy()
    if utc.index.tz is None:
        utc.index = utc.index.tz_localize("UTC")
    else:
        utc.index = utc.index.tz_convert("UTC")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in utc.columns:
        agg["Volume"] = "sum"
    out = utc.resample("4h", label="left", closed="left").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    if out.empty:
        return out
    return out.tz_convert(TPE)


def prior_spike_vs_break(
    high: np.ndarray,
    break_idx: int,
    break_low: float,
    *,
    bars: int = 24,
) -> float:
    """max(high in bars before break) / break_low - 1. 0 if unavailable."""
    if break_idx <= 0 or break_low <= 0 or bars <= 0:
        return 0.0
    left = max(0, break_idx - int(bars))
    if break_idx <= left:
        return 0.0
    peak = float(np.max(high[left:break_idx]))
    if peak <= 0:
        return 0.0
    return peak / float(break_low) - 1.0


def h4_ma25_at_ts(df: pd.DataFrame, ts, *, ma_period: int = 25) -> Tuple[float, float, float]:
    """At 1h timestamp ts, forming 4h close vs 4h MA25 (Binance UTC align).

    Returns (h4_close, h4_ma25, dist_pct) where dist_pct = (close/ma - 1)*100.
    NaNs if not enough 4h history.
    No look-ahead (v4): only 1h bars with open time <= ts are used, so the forming 4h bar's
    close = close of the 1h bar at ts and MA25 includes that partial 4h bar (same as live TG,
    where the df ends at the last closed 1h bar).
    """
    if df is not None and not df.empty and ts is not None:
        _t = pd.Timestamp(ts)
        if df.index.tz is not None and _t.tzinfo is not None:
            _t = _t.tz_convert(df.index.tz)
        df = df.loc[df.index <= _t]
    h4 = resample_4h(df)
    if h4 is None or len(h4) < ma_period:
        return float("nan"), float("nan"), float("nan")
    pos = _loc_on_tf(h4.index, ts)
    if pos is None or pos < 0:
        return float("nan"), float("nan"), float("nan")
    close4 = h4["Close"].to_numpy(float)
    ma = sma(close4, ma_period)
    c = float(close4[pos])
    m = float(ma[pos]) if pos < len(ma) and np.isfinite(ma[pos]) else float("nan")
    if not np.isfinite(m) or m <= 0:
        return c, m, float("nan")
    return c, m, (c / m - 1.0) * 100.0


def h4_ma25_ok(df: pd.DataFrame, ts, *, tol_pct: float = 0.0, ma_period: int = 25) -> bool:
    """True if forming 4h close >= 4h MA25 * (1 - tol_pct/100)."""
    _c, _m, dist = h4_ma25_at_ts(df, ts, ma_period=ma_period)
    if not np.isfinite(dist):
        return False
    return dist >= -float(tol_pct)


def h4_ma_slope_pct(df: pd.DataFrame, entry_idx: int, *, ma_period: int = 120, bars: int = 24) -> float:
    """4h MA{ma_period} % change over last `bars` 4h bars, using 1h bars up to entry_idx only (last 4h partial)."""
    h4 = resample_4h(df.iloc[: int(entry_idx) + 1])
    if h4 is None or len(h4) < ma_period + bars:
        return float("nan")
    ma = sma(h4["Close"].to_numpy(float), ma_period)
    a, b = float(ma[-1]), float(ma[-1 - bars])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return float("nan")
    return (a / b - 1.0) * 100.0


def filter_symbols(symbols: Sequence[str], *, exclude_inverse: bool = False, exclude_vol: bool = False) -> List[str]:
    drop = set()
    if exclude_inverse:
        drop |= INVERSE_SYMBOLS
    if exclude_vol:
        drop |= VOL_SYMBOLS
    return [s for s in symbols if s.upper() not in drop]


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


def _swing_lows(low: np.ndarray, start: int, end: int) -> List[int]:
    mins: List[int] = []
    for i in range(start + 2, end - 1):
        window = low[i - 2 : i + 3]
        if float(low[i]) <= float(np.min(window)) + 1e-12 and int(np.sum(np.isclose(window, low[i]))) == 1:
            mins.append(i)
    return mins


def classify_shape(low: np.ndarray, high: np.ndarray, start: int, end: int) -> str:
    """V if one clear trough; W if two swing lows with a bounce in between."""
    if end - start < 6:
        return "V"
    mins = _swing_lows(low, start, end)
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


def drawn_w_points(
    low: np.ndarray,
    high: np.ndarray,
    ma25: np.ndarray,
    start: int,
    end: int,
    bottom_idx: int,
    *,
    reclaim: Optional[int] = None,
    min_sep: int = 6,
    min_bounce: float = 0.006,
    neck_below: float = 0.015,
    neck_above: float = 0.015,
    arm_bars: int = 10,
    lookback: int = 18,
) -> Optional[Tuple[int, int, int, int, int]]:
    """咖啡色 W：左谷 → 吻 MA25 → 破底 → 拉到站回。"""
    n = len(high)
    if bottom_idx <= start or bottom_idx > end or bottom_idx >= n:
        return None
    search_from = max(0, start - lookback)
    lefts = [i for i in _swing_lows(low, search_from, end) if i <= bottom_idx - min_sep]
    if not lefts:
        return None
    left = min(lefts, key=lambda i: float(low[i]))
    first = float(low[left])
    bot = float(low[bottom_idx])
    if first <= 0 or bot >= first:
        return None
    pk = left + int(np.argmax(high[left : bottom_idx + 1]))
    if pk <= left or pk >= bottom_idx:
        return None
    peak = float(high[pk])
    ma = float(ma25[pk])
    if np.isnan(ma) or ma <= 0:
        return None
    dist = (peak - ma) / ma
    if dist < -neck_below or dist > neck_above:
        return None
    if (peak - first) / first < min_bounce:
        return None
    drop_from = search_from + int(np.argmax(high[search_from : left + 1])) if left > search_from else search_from
    if drop_from >= left:
        drop_from = max(search_from, left - 1)
    rec = reclaim if reclaim is not None else min(end + 1, n - 1)
    rec = max(bottom_idx, min(int(rec), n - 1))
    arm = min(n - 1, rec + 1)
    if arm <= bottom_idx:
        arm = rec
    return drop_from, left, pk, bottom_idx, arm


def drawn_w_ok(
    low: np.ndarray,
    high: np.ndarray,
    ma25: np.ndarray,
    start: int,
    end: int,
    bottom_idx: int,
    *,
    min_sep: int = 6,
    min_bounce: float = 0.006,
    neck_below: float = 0.015,
    neck_above: float = 0.015,
) -> bool:
    """對齊草圖：先穿上 MA25、中間深 V 破底、再拉回均線上。"""
    return (
        drawn_w_points(
            low,
            high,
            ma25,
            start,
            end,
            bottom_idx,
            min_sep=min_sep,
            min_bounce=min_bounce,
            neck_below=neck_below,
            neck_above=neck_above,
        )
        is not None
    )


def bull_stack(close: np.ndarray, ma7: np.ndarray, ma14: np.ndarray, ma25: np.ndarray, i: int) -> bool:
    """1h MA7 > MA14 > MA25，收盤在三條之上。"""
    if i < 0 or i >= len(close):
        return False
    if np.isnan(ma7[i]) or np.isnan(ma14[i]) or np.isnan(ma25[i]):
        return False
    return float(close[i]) > float(ma7[i]) > float(ma14[i]) > float(ma25[i])


def first_bull_stack(
    close: np.ndarray,
    ma7: np.ndarray,
    ma14: np.ndarray,
    ma25: np.ndarray,
    start_i: int,
    end_i: int,
) -> Optional[int]:
    for k in range(start_i, min(end_i, len(close) - 1) + 1):
        if bull_stack(close, ma7, ma14, ma25, k):
            return k
    return None


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


def find_ma25_retest(
    open_: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma25: np.ndarray,
    atr14: np.ndarray,
    reclaim_idx: int,
    bottom: float,
    *,
    max_wait: int = 18,
    tag_pct: float = 0.003,
    tag_atr: float = 0.4,
    require_leave: bool = True,
    require_bullish: bool = True,
) -> Optional[int]:
    """站回之後等回測 MA25。不在站回那根進。

    先要有一根低點完全在 MA25 上方（離開），再在 max_wait 根內出現：
    低點吻到 / 略穿 / 距 MA ≤ tag_pct 或 ≤ tag_atr·ATR，收盤仍在均線上、
    且仍高於破底。預設還要收紅棒（close ≥ open）。沒回測或中途收盤掉下
    MA25 / 打到破底 → None。
    """
    n = len(close)
    if reclaim_idx < 0 or reclaim_idx >= n - 1:
        return None
    left = not require_leave
    last = min(n - 1, reclaim_idx + max_wait)
    for k in range(reclaim_idx + 1, last + 1):
        ma = float(ma25[k]) if k < len(ma25) and not np.isnan(ma25[k]) else float("nan")
        if not (ma > 0):
            continue
        c = float(close[k])
        lo = float(low[k])
        o = float(open_[k]) if k < len(open_) else c
        if c < ma:
            return None
        if lo <= bottom:
            return None
        if not left:
            if lo > ma:
                left = True
            continue
        atr_v = float(atr14[k]) if k < len(atr14) and not np.isnan(atr14[k]) else 0.0
        dist = (lo - ma) / ma
        atr_dist = (lo - ma) / atr_v if atr_v > 0 else float("inf")
        tagged = lo <= ma or dist <= tag_pct or atr_dist <= tag_atr
        if not tagged:
            continue
        if require_bullish and c < o:
            continue
        if c <= ma:
            continue
        return int(k)
    return None


def detect_signals(
    df: pd.DataFrame,
    *,
    ma_period: int = 25,
    min_bars_below: int = 4,
    max_bars_below: int = 36,
    min_depth_pct: float = 0.018,
    min_impulse_pct: float = 0.0,
    min_undercut_pct: float = 0.014,
    min_flush_atr: float = 0.0,
    flush_bars: int = 4,
    target_r: float = 2.0,
    min_entry_gap: int = 8,
    vol_lookback: int = 20,
    break_lookback: int = 16,
    require_drawn_w: bool = False,
    max_stack_wait: int = 36,
    max_retest_wait: int = 18,
    retest_tag_pct: float = 0.003,
    retest_tag_atr: float = 0.4,
    require_retest_leave: bool = True,
    require_retest_bullish: bool = True,
    max_ma_drift_pct: float = 1.5,
    min_ribbon_atr: float = 0.50,
    ma200_near_pct: float = 1.0,
    min_ma_cluster_atr: float = 1.0,
    min_break_vol: float = 0.8,
    min_mid_ma_atr: float = 0.50,
    require_h4_ma25: bool = False,
    h4_ma25_tol_pct: float = 0.0,
    max_prior_spike_pct: float = 0.0,
    prior_spike_bars: int = 24,
    min_reclaim_entry_bars: int = 0,
    min_decline_pct: float = 0.0,
    decline_lookback: int = 24,
    max_flat_bars: int = -1,
    flat_window: int = 24,
    flat_ref_bars: int = 120,
    flat_frac: float = 0.35,
    max_reclaim_entry_bars: int = 0,
    h4_max_ma_drop_pct: float = 0.0,
    h4_drop_ma: int = 120,
    h4_drop_bars: int = 24,
    max_entry_ext_atr: float = 0.0,
    max_overhead_knot_atr: float = 0.0,
    min_entry_vol: float = 0.0,
    require_ma7_above_ma14: bool = False,
    min_overhead_ma_atr: float = 0.0,
    h4_max_ma_rise_pct: float = 999.0,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """1h 跌破 MA25，做出 W、收盤站回後，等回測 MA25 再進。max_stack_wait 只留給舊呼叫。

    破底那根低點須 ≤ MA200×(1+ma200_near_pct%)；MA200 不足則丟。
    破底 |MA200-MA25| 至少 min_ma_cluster_atr 個 ATR（預設 1.0；0=關），避免六線糾成水平義大利麵。
    破前低預設至少 1.4%（急殺前 12 根平台）；影線輕點算 weak_break。
    破底那根量須 >= 前 N 根均量 x min_break_vol（N=vol_lookback，不含破底棒本身；預設 0.8；0=關）。
    進場 MA25/99/120 帶寬至少 min_mid_ma_atr 個 ATR（預設 0.50；0=關），避免回測走進粉紫綠結。
    require_h4_ma25：進場時 4h 收盤須 >= 4h MA25（容許低 h4_ma25_tol_pct%）；預設關。
    max_prior_spike_pct：破底前 prior_spike_bars 根最高價 / 破底低 -1 超過此值則丟（分數，0.15=15%；0=關）。
    min_reclaim_entry_bars：站回→進場至少隔幾根（entry_idx-reclaim_idx；0=關）。過早回測＝假站回/結構不完整。
    min_decline_pct：破底前真的跌過：max(high[跌破起點-decline_lookback : 破底]) → 破底低 跌幅下限（分數；0=關）。
    max_flat_bars：進場前 flat_window 根內「一字線」（高低差 < flat_frac × 前 flat_ref_bars 根中位數）最多幾根（<0=關）。死盤丟掉。
    max_reclaim_entry_bars：站回→進場最多隔幾根（0=關）。站回後拖太久才回測＝結構走樣。
    h4_max_ma_drop_pct：進場時 4h MA{h4_drop_ma} 近 h4_drop_bars 根 4h 跌幅超過此 %% 則丟（0=關；6＝4h 長均線急跌的空頭段）。
      只用進場那根（含）以前的 1h 組 4h，最後一根 4h 為未完成棒，無偷看；歷史不足則放行。
    max_entry_ext_atr：進場收盤高出 MA25 超過此 ATR 則丟（0=關）。低點有吻到均線、收盤卻衝很遠＝追價，不是回測。
    max_overhead_knot_atr：MA60、MA99、MA120 全在進場收盤之上，且三線帶寬 <= 此 ATR 則丟（0=關）。
      頭上壓著一團中期均線。三線散開很寬（空頭排列）不算。
    """
    if df is None or len(df) < ma_period + min_bars_below + 5:
        return []

    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    open_ = df["Open"].to_numpy(float) if "Open" in df.columns else np.r_[close[0], close[:-1]]
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
    ma25 = sma(close, ma_period)
    atr14 = atr(high, low, close, 14)
    ma7 = sma(close, 7)
    ma14 = sma(close, 14)
    ma60 = sma(close, 60)
    ma99 = sma(close, 99)
    ma120 = sma(close, 120)
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

        ma_start = float(ma25[start])
        ma_reclaim = float(ma25[reclaim])
        if max_ma_drift_pct > 0 and ma_start > 0:
            if abs(ma_reclaim - ma_start) / ma_start > max_ma_drift_pct / 100.0:
                bump("ma_drift")
                i = reclaim
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

        m200 = float(ma200[bottom_idx]) if bottom_idx < len(ma200) else float("nan")
        if not ma200_near_ok(bottom, m200, ma200_near_pct):
            bump("above_ma200")
            i = reclaim
            continue

        if min_ribbon_atr > 0:
            rib = ribbon_median_atr(ma7, ma14, ma25, atr14, start, end)
            if np.isfinite(rib) and rib < min_ribbon_atr:
                bump("tangled")
                i = reclaim
                continue

        atr_b = float(atr14[bottom_idx]) if bottom_idx < len(atr14) else float("nan")
        m25_b = float(ma25[bottom_idx]) if bottom_idx < len(ma25) else float("nan")
        if not ma_cluster_ok(m25_b, m200, atr_b, min_ma_cluster_atr):
            bump("ma_cluster")
            i = reclaim
            continue

        impulse_pct, flush_atr, undercut_pct = flush_metrics(
            high, low, bottom_idx, bottom, atr14, flush_bars=flush_bars
        )
        if impulse_pct < min_impulse_pct or flush_atr < min_flush_atr:
            bump("weak_flush")
            i = reclaim
            continue
        if undercut_pct < min_undercut_pct:
            bump("weak_break")
            i = reclaim
            continue

        left = max(0, bottom_idx - break_lookback + 1)
        if bottom > float(np.min(low[left : bottom_idx + 1])) + 1e-12:
            bump("not_break")
            i = reclaim
            continue

        bvol_ratio = break_vol_ratio(volume, bottom_idx, vol_lookback)
        if min_break_vol > 0 and bvol_ratio < min_break_vol:
            bump("quiet_break")
            i = reclaim
            continue

        w_pts = drawn_w_points(low, high, ma25, start, end, bottom_idx, reclaim=reclaim)
        if require_drawn_w and w_pts is None:
            bump("not_w")
            i = reclaim
            continue

        bump("reclaim")
        entry_idx = find_ma25_retest(
            open_,
            low,
            close,
            ma25,
            atr14,
            reclaim,
            bottom,
            max_wait=max_retest_wait,
            tag_pct=retest_tag_pct,
            tag_atr=retest_tag_atr,
            require_leave=require_retest_leave,
            require_bullish=require_retest_bullish,
        )
        if entry_idx is None:
            bump("no_retest")
            i = reclaim
            continue
        if entry_idx - last_entry < min_entry_gap:
            bump("gap")
            i = reclaim
            continue

        entry = float(close[entry_idx])
        stop = bottom  # 破底那根 K 的低點；回測要收盤跌破才出
        risk = entry - stop
        if risk <= 0:
            bump("bad_risk")
            i = reclaim
            continue

        atr_e = float(atr14[entry_idx]) if entry_idx < len(atr14) else float("nan")
        m25_e = float(ma25[entry_idx]) if entry_idx < len(ma25) else float("nan")
        m99_e = float(ma99[entry_idx]) if entry_idx < len(ma99) else float("nan")
        m120_e = float(ma120[entry_idx]) if entry_idx < len(ma120) else float("nan")
        if not mid_ma_cluster_ok(m25_e, m99_e, m120_e, atr_e, min_mid_ma_atr):
            bump("mid_ma_cluster")
            i = reclaim
            continue

        if max_prior_spike_pct > 0:
            spike = prior_spike_vs_break(high, bottom_idx, bottom, bars=prior_spike_bars)
            if spike > max_prior_spike_pct:
                bump("prior_spike")
                i = reclaim
                continue

        if require_h4_ma25:
            if not h4_ma25_ok(df, df.index[entry_idx], tol_pct=h4_ma25_tol_pct):
                bump("h4_bear")
                i = reclaim
                continue

        if h4_max_ma_drop_pct > 0 or h4_max_ma_rise_pct < 900.0:
            drop = h4_ma_slope_pct(df, entry_idx, ma_period=h4_drop_ma, bars=h4_drop_bars)
            if h4_max_ma_drop_pct > 0 and np.isfinite(drop) and drop < -float(h4_max_ma_drop_pct):
                bump("h4_ma_drop")
                i = reclaim
                continue
            if h4_max_ma_rise_pct < 900.0 and np.isfinite(drop) and drop > float(h4_max_ma_rise_pct):
                bump("h4_ma_rise")
                i = reclaim
                continue

        if min_reclaim_entry_bars > 0 and (entry_idx - reclaim) < int(min_reclaim_entry_bars):
            bump("early_retest")
            i = reclaim
            continue

        if max_reclaim_entry_bars > 0 and (entry_idx - reclaim) > int(max_reclaim_entry_bars):
            bump("late_retest")
            i = reclaim
            continue

        decl = decline_before_break(high, start, bottom_idx, bottom, lookback=decline_lookback)
        if min_decline_pct > 0 and decl < float(min_decline_pct):
            bump("small_decline")
            i = reclaim
            continue

        flats = flat_bar_count(high, low, entry_idx, window=flat_window, ref_bars=flat_ref_bars, frac=flat_frac)
        if max_flat_bars >= 0 and flats > int(max_flat_bars):
            bump("dead_tape")
            i = reclaim
            continue

        if max_entry_ext_atr > 0 and np.isfinite(atr_e) and atr_e > 0 and np.isfinite(m25_e):
            if (entry - m25_e) / atr_e > float(max_entry_ext_atr):
                bump("entry_ext")
                i = reclaim
                continue

        if max_overhead_knot_atr > 0 and np.isfinite(atr_e) and atr_e > 0:
            m60_e = float(ma60[entry_idx]) if entry_idx < len(ma60) else float("nan")
            knot = overhead_knot_span_atr(m60_e, m99_e, m120_e, entry, atr_e)
            if np.isfinite(knot) and knot <= float(max_overhead_knot_atr):
                bump("overhead_knot")
                i = reclaim
                continue

        if min_overhead_ma_atr > 0 and np.isfinite(atr_e) and atr_e > 0:
            m60_e = float(ma60[entry_idx]) if entry_idx < len(ma60) else float("nan")
            m200_e = float(ma200[entry_idx]) if entry_idx < len(ma200) else float("nan")
            ceils = [mv for mv in (m60_e, m99_e, m120_e, m200_e) if np.isfinite(mv) and mv > entry]
            if ceils and min((mv - entry) / atr_e for mv in ceils) < float(min_overhead_ma_atr):
                bump("overhead_ceil")
                i = reclaim
                continue

        vol_avg = float(np.mean(volume[max(0, entry_idx - vol_lookback) : entry_idx]) or 1.0)
        vol_ratio = float(volume[entry_idx] / vol_avg) if vol_avg > 0 else 1.0
        if min_entry_vol > 0 and vol_ratio < float(min_entry_vol):
            bump("quiet_entry")
            i = reclaim
            continue

        if require_ma7_above_ma14:
            m7_e = float(ma7[entry_idx]) if entry_idx < len(ma7) else float("nan")
            m14_e = float(ma14[entry_idx]) if entry_idx < len(ma14) else float("nan")
            if not (np.isfinite(m7_e) and np.isfinite(m14_e) and m7_e > m14_e):
                bump("ma7_below_ma14")
                i = reclaim
                continue

        shape = classify_shape(low, high, start, end)
        if w_pts is not None:
            shape = "W"
        stacked = bull_stack(close, ma7, ma14, ma25, entry_idx)
        q_score, q_grade = quality_of(depth_pct, vol_ratio, shape, stacked)
        w_start = w_left = w_peak = w_right = w_arm = -1
        if w_pts is not None:
            w_start, w_left, w_peak, w_right, w_arm = w_pts

        signals.append(
            Signal(
                break_idx=bottom_idx,
                entry_idx=entry_idx,
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(entry + risk * target_r),
                bottom=bottom,
                ma25=float(ma25[entry_idx]),
                depth_pct=float(depth_pct),
                bars_below=int(bars_below),
                shape=shape,
                vol_ratio=float(vol_ratio),
                break_vol_ratio=float(bvol_ratio),
                impulse_pct=float(impulse_pct),
                undercut_pct=float(undercut_pct),
                flush_atr=float(flush_atr),
                quality=q_grade,
                quality_score=q_score,
                ma7=float(ma7[entry_idx]) if not np.isnan(ma7[entry_idx]) else 0.0,
                ma14=float(ma14[entry_idx]) if not np.isnan(ma14[entry_idx]) else 0.0,
                ma99=float(ma99[entry_idx]) if not np.isnan(ma99[entry_idx]) else 0.0,
                ma200=float(ma200[entry_idx]) if not np.isnan(ma200[entry_idx]) else 0.0,
                w_start_idx=int(w_start),
                w_left_idx=int(w_left),
                w_peak_idx=int(w_peak),
                w_right_idx=int(w_right),
                w_arm_idx=int(w_arm),
                reclaim_idx=int(reclaim),
                decline_pct=float(decl),
                flat_bars=int(flats),
            )
        )
        last_entry = entry_idx
        bump("taken")
        i = entry_idx + 1

    return signals


def _fractal_swings(high: np.ndarray, low: np.ndarray) -> Tuple[List[int], List[int]]:
    """1-bar fractal: strictly higher/lower than the bar on each side."""
    sh: List[int] = []
    sl: List[int] = []
    n = len(high)
    for i in range(1, n - 1):
        if high[i] > high[i - 1] and high[i] > high[i + 1]:
            sh.append(i)
        if low[i] < low[i - 1] and low[i] < low[i + 1]:
            sl.append(i)
    return sh, sl


def detect_5m_m_neckline(
    df_5m: pd.DataFrame,
    *,
    entry_ts,
    entry_price: float,
    stop_price: float,
    end_ts=None,
    min_peak_sep: int = M5_MIN_PEAK_SEP,
    max_peak_sep: int = M5_MAX_PEAK_SEP,
    right_overshoot: float = M5_RIGHT_OVERSHOOT,
    min_neck_depth: float = M5_MIN_NECK_DEPTH,
    hh_tol: float = M5_HH_TOL,
    arm_r: float = M5_ARM_R,
    min_break_pct: float = M5_MIN_BREAK_PCT,
    arm_after_ts=None,
) -> Optional[M5NeckExit]:
    """First 5m M (left=HH, lower-right, neck break) after entry.

    Extra constraints chosen on 龍蝦 08-24/26 tape:
    - left swing-high within hh_tol of running 5m HH since entry
    - new HH (above left*(1+overshoot)) invalidates an armed M
    - only arm after arm_after_ts (1h close ≥ 1.5R) or 5m close ≥ entry+arm_r*R
    - close must break the neckline by min_break_pct (skip 0.4% haircuts)
    """
    if df_5m is None or df_5m.empty:
        return None
    sub = df_5m
    if entry_ts is not None:
        sub = sub.loc[sub.index >= entry_ts]
    if end_ts is not None:
        sub = sub.loc[sub.index <= end_ts]
    if len(sub) < min_peak_sep + 4:
        return None
    high = sub["High"].to_numpy(float)
    low = sub["Low"].to_numpy(float)
    close = sub["Close"].to_numpy(float)
    idx = sub.index
    sh, sl = _fractal_swings(high, low)
    risk = float(entry_price) - float(stop_price)
    armed_r = arm_r <= 0 and arm_after_ts is None
    armed: Optional[dict] = None
    n = len(sub)

    def _maybe_arm_r(j: int) -> bool:
        if arm_after_ts is not None:
            return bool(idx[j] >= arm_after_ts)
        if arm_r <= 0:
            return True
        if risk <= 0:
            return False
        return float(close[j]) >= float(entry_price) + float(arm_r) * risk

    for j in range(n):
        if not armed_r and _maybe_arm_r(j):
            armed_r = True
        if armed is not None and float(high[j]) > armed["left_h"] * (1.0 + right_overshoot):
            armed = None
        if j >= 2:
            i = j - 1
            new_sh = high[i] > high[i - 1] and high[i] > high[j]
            if armed_r and new_sh:
                hh = float(np.max(high[: i + 1]))
                for l in sh:
                    if l >= i:
                        break
                    sep = i - l
                    if sep < min_peak_sep or sep > max_peak_sep:
                        continue
                    left_h = float(high[l])
                    right_h = float(high[i])
                    if right_h > left_h * (1.0 + right_overshoot):
                        continue
                    if hh > 0 and left_h < hh * (1.0 - hh_tol):
                        continue
                    troughs = [t for t in sl if l < t < i]
                    if not troughs:
                        continue
                    t = min(troughs, key=lambda k: (float(low[k]), k))
                    neck = float(low[t])
                    if right_h <= neck or left_h <= 0:
                        continue
                    depth = (left_h - neck) / left_h
                    if depth < min_neck_depth:
                        continue
                    cand = {
                        "left_i": l,
                        "trough_i": t,
                        "right_i": i,
                        "left_h": left_h,
                        "neck": neck,
                        "right_h": right_h,
                    }
                    if armed is None or cand["left_h"] >= armed["left_h"] - 1e-15:
                        armed = cand
        if not armed or not armed_r:
            continue
        if j < armed["right_i"] + 1:
            continue
        neck = armed["neck"]
        brk = (neck - float(close[j])) / neck if neck > 0 else 0.0
        if float(close[j]) < neck and brk >= min_break_pct:
            return M5NeckExit(
                fill_ts=idx[j],
                fill_price=float(close[j]),
                left_ts=idx[armed["left_i"]],
                trough_ts=idx[armed["trough_i"]],
                right_ts=idx[armed["right_i"]],
                neckline=float(neck),
                left_high=float(armed["left_h"]),
                right_high=float(armed["right_h"]),
            )
    return None


def _bar_end_ts(index: pd.DatetimeIndex, k: int) -> pd.Timestamp:
    if k + 1 < len(index):
        return pd.Timestamp(index[k + 1])
    return pd.Timestamp(index[k]) + pd.Timedelta(hours=1)


def simulate(
    df: pd.DataFrame,
    signals: List[Signal],
    *,
    max_hold: int = 36,
    lose_ma_closes: int = 0,
    df_5m: Optional[pd.DataFrame] = None,
    m5_exit: Optional[bool] = None,
) -> List[TradeResult]:
    use_m5 = bool(m5_exit) if m5_exit is not None else df_5m is not None
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    ma25 = sma(close, 25)
    results: List[TradeResult] = []
    idx = df.index

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
        exit_ts_5m = None
        below_streak = 0
        use_target = (not use_m5) and float(target) > float(entry)

        m5_hit: Optional[M5NeckExit] = None
        if use_m5 and df_5m is not None and not df_5m.empty:
            risk = float(entry) - float(stop)
            arm_after = None
            if risk > 0:
                thresh = float(entry) + M5_ARM_R * risk
                for k0 in range(entry_idx, limit + 1):
                    if float(close[k0]) >= thresh:
                        arm_after = pd.Timestamp(idx[k0]) + pd.Timedelta(hours=1)
                        break
            if arm_after is not None:
                end_ts = _bar_end_ts(idx, limit) - pd.Timedelta(milliseconds=1)
                m5_hit = detect_5m_m_neckline(
                    df_5m,
                    entry_ts=idx[entry_idx],
                    entry_price=entry,
                    stop_price=stop,
                    end_ts=end_ts,
                    arm_after_ts=arm_after,
                )

        for k in range(entry_idx + 1, limit + 1):
            bar_open = pd.Timestamp(idx[k])
            bar_end = _bar_end_ts(idx, k)
            if m5_hit is not None:
                ft = pd.Timestamp(m5_hit.fill_ts)
                if getattr(ft, "tzinfo", None) is not None and getattr(bar_open, "tzinfo", None) is not None:
                    ft = ft.tz_convert(bar_open.tz)
                if bar_open <= ft < bar_end:
                    exit_idx, exit_price, exit_reason = k, float(m5_hit.fill_price), "m5_neck"
                    exit_ts_5m = ft
                    break
            # 跌破破底那根 K：收盤低於破底棒低點，用該根收盤出場
            if close[k] < stop:
                exit_idx, exit_price, exit_reason = k, float(close[k]), "stop"
                break
            if use_target and high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break
            if not np.isnan(ma25[k]) and close[k] < ma25[k]:
                below_streak += 1
                if lose_ma_closes > 0 and below_streak >= lose_ma_closes:
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
                exit_ts_5m=exit_ts_5m,
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
    # 手機圖大概 80～90 根：W 起點前留一天，出場後再留半天
    sig = trade.signal
    anchors = [sig.break_idx, trade.entry_idx]
    if getattr(sig, "reclaim_idx", -1) >= 0:
        anchors.append(sig.reclaim_idx)
    if sig.w_peak_idx >= 0:
        anchors.append(sig.w_peak_idx)
    right = [trade.exit_idx, trade.entry_idx]
    if sig.w_arm_idx >= 0:
        right.append(sig.w_arm_idx)
    start = max(0, min(anchors) - 28)
    end = min(len(df) - 1, max(right) + 12)
    return start, end


def _w_xy(
    sig: Signal,
    high: np.ndarray,
    low: np.ndarray,
    win_start: int,
    n_win: int,
) -> Optional[Tuple[List[float], List[float]]]:
    seq = [
        (sig.w_start_idx, "h"),
        (sig.w_left_idx, "l"),
        (sig.w_peak_idx, "h"),
        (sig.w_right_idx, "l"),
        (sig.w_arm_idx, "h"),
    ]
    xs: List[float] = []
    ys: List[float] = []
    last_i = None
    for i, kind in seq:
        if i < 0 or i >= len(high):
            continue
        if last_i is not None and i == last_i:
            continue
        y = float(high[i]) if kind == "h" else float(low[i])
        xs.append(float(i - win_start))
        ys.append(y)
        last_i = i
    keep_x: List[float] = []
    keep_y: List[float] = []
    for x, y in zip(xs, ys):
        if -0.5 <= x <= n_win - 0.5:
            keep_x.append(x)
            keep_y.append(y)
    if len(keep_x) < 3:
        return None
    return keep_x, keep_y


def _draw_w_overlay(ax, sig: Signal, high: np.ndarray, low: np.ndarray, win_start: int, n_win: int) -> None:
    if not DRAW_W_OVERLAY:
        return
    xy = _w_xy(sig, high, low, win_start, n_win)
    if xy is None:
        return
    xs, ys = xy
    ax.plot(
        xs,
        ys,
        color=W_COLOR,
        lw=3.8,
        alpha=0.88,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=7,
        label="_nolegend_",
    )


def _loc_on_tf(index: pd.DatetimeIndex, ts) -> Optional[int]:
    if ts is None or len(index) == 0:
        return None
    if getattr(ts, "tzinfo", None) is not None and index.tz is not None:
        ts = ts.tz_convert(index.tz)
    pos = int(index.searchsorted(ts, side="right") - 1)
    if 0 <= pos < len(index):
        return pos
    return None


def _style_ax(ax) -> None:
    # 幣安 App 淺色主題，才跟你那兩張手機截圖同一種均線長相
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#6b7280", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#e5e7eb")
    ax.grid(True, color="#f0f0f0", lw=0.6, axis="y", zorder=0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def _plot_mas(ax, close_full: pd.Series, start: int, end: int, xs) -> None:
    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(
            list(xs),
            ma,
            color=col,
            lw=MA_WIDTH.get(n, 1.1),
            label=f"MA{n}",
            solid_capstyle="round",
            zorder=3,
        )


def _paint_candles(ax, xs, o, h, l, c):
    colors = []
    from matplotlib.patches import Rectangle

    for k in range(len(xs)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#0ecb81" if up else "#f6465d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.7)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors.append("#0ecb8199" if up else "#f6465d99")
    return colors


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int, title_extra: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk()
    sig = trade.signal
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv, ax4) = plt.subplots(
        3,
        1,
        figsize=(10.4, 8.2),
        gridspec_kw={"height_ratios": [3.35, 0.7, 1.95]},
        facecolor="#ffffff",
    )
    ax.sharex(axv)
    for a in (ax, axv, ax4):
        _style_ax(a)

    colors_v = _paint_candles(ax, xs, o, h, l, c)
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0, zorder=2)

    _plot_mas(ax, close_full, start, end, xs)
    _draw_w_overlay(
        ax,
        sig,
        df["High"].to_numpy(float),
        df["Low"].to_numpy(float),
        start,
        len(window),
    )

    # 停損／目標不拉 Y 軸，否則均線帶會被壓扁，跟手機圖不像
    y_lo = float(np.nanmin(l.to_numpy(float)))
    y_hi = float(np.nanmax(h.to_numpy(float)))
    for n in MA_COLORS:
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        if ma.notna().any():
            y_lo = min(y_lo, float(np.nanmin(ma.to_numpy(float))))
            y_hi = max(y_hi, float(np.nanmax(ma.to_numpy(float))))
    pad = max((y_hi - y_lo) * 0.045, abs(y_hi) * 1e-4)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    if y_lo - pad <= trade.stop_price <= y_hi + pad:
        ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.75)
    if trade.exit_reason not in {"m5_neck", "m5_m"} and y_lo - pad <= trade.target_price <= y_hi + pad:
        ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.7)

    bx, ex, xx = sig.break_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.bottom], s=42, color="#f472b6", zorder=5)
        ax.annotate(
            "破底",
            (bx, sig.bottom),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            color="#c2185b",
            fontsize=8,
        )
    rx_i = getattr(sig, "reclaim_idx", -1)
    if rx_i >= 0 and rx_i != trade.entry_idx:
        rx = rx_i - start
        if 0 <= rx < len(window):
            rec_px = float(df["Close"].iloc[rx_i])
            ax.scatter([rx], [rec_px], s=28, color="#7e57c2", marker="o", zorder=5)
            ax.annotate(
                "站回",
                (rx, rec_px),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="#6a1b9a",
                fontsize=7,
            )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        # filled triangle marks retest entry price
        ax.scatter([ex], [trade.entry_price], s=46, color="#00c853", marker="^", zorder=6)
        # prominent open circle so entry is obvious on cards
        ax.scatter(
            [ex],
            [trade.entry_price],
            s=220,
            facecolors="none",
            edgecolors="#ff8c00",
            linewidths=2.4,
            zorder=7,
        )
        ax.annotate(
            "進",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(10, 12),
            ha="left",
            color="#e65100",
            fontsize=9,
            fontweight="bold",
        )
    if 0 <= xx < len(window):
        exit_col = "#0284c7" if trade.pnl_pct >= 0 else "#dc2626"
        ax.axvline(xx, color=exit_col, ls="--", lw=1.0, alpha=0.85)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=46,
            color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
            marker="v",
            zorder=6,
        )
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=220,
            facecolors="none",
            edgecolors=exit_col,
            linewidths=2.4,
            zorder=7,
        )
        held_bars = max(0, int(trade.exit_idx - trade.entry_idx))
        if trade.exit_reason in {"m5_neck", "m5_m"}:
            exit_lbl = "出·M破頸"
        elif trade.exit_reason == "stop":
            exit_lbl = "出·停損"
        elif trade.exit_reason == "target":
            exit_lbl = "出·停利"
        elif trade.exit_reason == "lost_ma25":
            exit_lbl = "出·掉MA25"
        else:
            exit_lbl = f"出·{held_bars}h"
        ax.annotate(
            exit_lbl,
            (xx, trade.exit_price),
            textcoords="offset points",
            xytext=(10, -14),
            ha="left",
            color=exit_col,
            fontsize=9,
            fontweight="bold",
        )

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_pct >= 0 else ""
    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}Q{trade.quality} {sig.shape}  1h  "
        f"{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_pct:.2f}%",
        color="#1e2329",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#4b5563", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#6b7280")

    h4 = resample_4h(df)
    if len(h4) >= 2:
        i4 = _loc_on_tf(h4.index, et) or max(0, len(h4) - 1)
        s4 = max(0, i4 - 42)
        e4 = min(len(h4) - 1, max(i4 + 8, (_loc_on_tf(h4.index, xt) or i4) + 4))
        w4 = h4.iloc[s4 : e4 + 1]
        xs4 = range(len(w4))
        _paint_candles(ax4, xs4, w4["Open"], w4["High"], w4["Low"], w4["Close"])
        close4 = h4["Close"].astype(float)
        _plot_mas(ax4, close4, s4, e4, xs4)
        if DRAW_W_OVERLAY and sig.w_left_idx >= 0:
            w4x: List[float] = []
            w4y: List[float] = []
            for i, price in (
                (sig.w_start_idx, float(df["High"].iloc[sig.w_start_idx]) if sig.w_start_idx >= 0 else None),
                (sig.w_left_idx, float(df["Low"].iloc[sig.w_left_idx])),
                (sig.w_peak_idx, float(df["High"].iloc[sig.w_peak_idx]) if sig.w_peak_idx >= 0 else None),
                (sig.w_right_idx, float(df["Low"].iloc[sig.w_right_idx]) if sig.w_right_idx >= 0 else None),
                (sig.w_arm_idx, float(df["High"].iloc[sig.w_arm_idx]) if sig.w_arm_idx >= 0 else None),
            ):
                if i is None or price is None or i < 0:
                    continue
                px = _loc_on_tf(w4.index, df.index[i])
                if px is not None:
                    w4x.append(float(px))
                    w4y.append(price)
            if len(w4x) >= 3:
                ax4.plot(w4x, w4y, color=W_COLOR, lw=3.2, alpha=0.88, solid_capstyle="round", zorder=6)
        for ts, col, mark in (
            (df.index[sig.break_idx], "#f472b6", "破底"),
            (et, "#00c853", "回測"),
            (xt, "#0284c7", "出"),
        ):
            px = _loc_on_tf(w4.index, ts)
            if px is not None:
                ax4.axvline(px, color=col, ls="--", lw=0.85, alpha=0.85)
                if mark:
                    ax4.scatter([px], [float(w4["Close"].iloc[px])], s=28, color=col, zorder=5)
        ax4.text(
            0.01,
            0.92,
            "4h 對照",
            transform=ax4.transAxes,
            color="#4b5563",
            fontsize=9,
            va="top",
        )
        ax4.legend(loc="upper right", fontsize=6, frameon=False, labelcolor="#4b5563", ncol=6)
        step4 = max(1, len(w4) // 6)
        ticks4 = list(range(0, len(w4), step4))
        ax4.set_xticks(ticks4)
        ax4.set_xticklabels([w4.index[i].strftime("%m-%d %H:%M") for i in ticks4], color="#6b7280")
    else:
        ax4.set_visible(False)

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
    strict: bool = False,
    m5_exit: bool = True,
    lose_ma_closes: int = 0,
    max_hold: int = 36,
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
        reason_cls = {
            "target": "tag-tp",
            "m5_neck": "tag-tp",
            "m5_m": "tag-tp",
            "stop": "tag-sl",
            "lost_ma25": "tag-sl",
        }.get(t.exit_reason, "tag-time")
        img_name = f"t{i:02d}_{hit.symbol}_{et.strftime('%m%d_%H%M')}_q{t.quality.lower()}.png"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=hit.symbol)
        if t.signal.w_left_idx >= 0:
            bounce_px = float(df["High"].iloc[t.signal.w_peak_idx]) if t.signal.w_peak_idx >= 0 else t.signal.bottom
            entry_line = (
                f"進場 回測 MA25  破底 {t.signal.bottom:.6g}  彈 {bounce_px:.6g}  站回後吻均線"
            )
        else:
            entry_line = (
                f"進場 回測 MA25  MA7 {t.signal.ma7:.6g}  MA14 {t.signal.ma14:.6g}  MA25 {t.signal.ma25:.6g}"
            )
        if m5_exit or t.exit_reason in {"m5_neck", "m5_m"}:
            if lose_ma_closes > 0:
                tgt_line = f"出場 五分 M 破頸線（無 2R；備援 MA25×{lose_ma_closes} / {max_hold}h）"
            else:
                tgt_line = f"出場 五分 M 破頸線（無 2R；停損＝收盤跌破破底；{max_hold}h）"
        else:
            tgt_line = f"target {t.target_price:.6g}  ({r_mult:.1f}R)"
        exit_line = f"exit  {t.exit_price:.6g}  {t.exit_reason}"
        if getattr(t, "exit_ts_5m", None) is not None:
            xt5 = pd.Timestamp(t.exit_ts_5m)
            if getattr(xt5, "tzinfo", None) is not None:
                xt5 = xt5.tz_convert(TPE)
            exit_line += f"  5m {xt5.strftime('%m-%d %H:%M')}"
        cards.append(
            "<article class='trade-card' id='t" + str(i) + "'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(hit.symbol)} · Q{escape(t.quality)} · {escape(t.signal.shape)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>1h + 4h</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"{fresh}"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.6g}\n"
            f"stop  {t.stop_price:.6g}  收盤跌破破底K  (−{(risk / t.entry_price) * 100:.2f}%)\n"
            f"{tgt_line}\n"
            f"{exit_line}\n"
            f"破底 {t.signal.bottom:.6g}  深度 {t.signal.depth_pct * 100:.2f}%  在下 {t.signal.bars_below}h\n"
            f"急殺 {t.signal.impulse_pct * 100:.1f}% / {t.signal.flush_atr:.1f}ATR  破前低 {t.signal.undercut_pct * 100:.1f}%\n"
            f"量比 {t.signal.vol_ratio:.2f}x  破底量 {t.signal.break_vol_ratio:.2f}x  MA25 {t.signal.ma25:.6g}\n"
            f"{entry_line}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i} {escape(hit.symbol)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：跌破 {funnel.get('cross_down', 0)} → "
            f"站回 {funnel.get('reclaim', 0)} → 回測進場 {funnel.get('taken', 0)}"
            f"（太短 {funnel.get('too_short', 0)} · 太長 {funnel.get('too_long', 0)} · "
            f"太淺 {funnel.get('shallow', 0)} · 急殺不夠 {funnel.get('weak_flush', 0)} · "
            f"破底不夠 {funnel.get('weak_break', 0)} · "
            f"未站回 {funnel.get('no_reclaim', 0)} · 沒回測 {funnel.get('no_retest', 0)} · "
            f"MA漂移 {funnel.get('ma_drift', 0)} · 均線糾結 {funnel.get('tangled', 0)} · "
            f"長均線糾結 {funnel.get('ma_cluster', 0)} · "
            f"中均線糾結 {funnel.get('mid_ma_cluster', 0)} · "
            f"破底在MA200上 {funnel.get('above_ma200', 0)} · "
            f"破底沒放量 {funnel.get('quiet_break', 0)} · "
            f"追價 {funnel.get('entry_ext', 0)} · "
            f"頭上均線結 {funnel.get('overhead_knot', 0)} · "
            f"進場縮量 {funnel.get('quiet_entry', 0)} · "
            f"MA7未過MA14 {funnel.get('ma7_below_ma14', 0)}）</p>"
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
<h1>幣安 1h · MA25 下破底再站上（{'嚴格 · 筆畫 W' if strict else '寬鬆'}）</h1>
<p class="muted">近 {days} 天 · 掃 {scanned} 檔 U 本位永續 · 均線對齊你那兩張手機圖：黃7 / 青14 / 粉25 / 紫99 / 綠120 / 酒紅200<br/>
{'嚴格版 · 在下至少 10 小時、急殺破底。圖上藍線照手機畫法：摔到破底、彈起、回踩 MA25、再往上。' if strict else '寬鬆版 · 收盤跌破 MA25，在下至少 4 小時、深度 ≥ 1.8%、破前低 ≥ 1.4%（影線輕點不算）。'}破底後先收盤站回 MA25，離開均線後回測吻到再進（沒回測就放棄）。圖上藍線是筆畫 W。停損＝收盤跌破破底低點（影線不算）。{'出場＝五分 M 破頸。' if m5_exit else '目標 2R。'}{'不再用連兩根掉 MA25。' if lose_ma_closes <= 0 else f'備援連 {lose_ma_closes} 根收盤掉 MA25。'}最長 {max_hold}h。每張卡底下附 4h K 對照。{card_note}</p>
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


def one_at_a_time_path(
    hits: Sequence[Hit],
    *,
    start: float = 100.0,
    lev: float = 3.0,
) -> List[dict]:
    """Compound equity: all-in each time, one open position globally."""
    rows: List[tuple] = []
    for h in hits:
        et = h.df.index[h.trade.entry_idx]
        xt = h.df.index[h.trade.exit_idx]
        if getattr(et, "tzinfo", None) is None:
            et = et.replace(tzinfo=TPE)
            xt = xt.replace(tzinfo=TPE)
        else:
            et = et.tz_convert(TPE)
            xt = xt.tz_convert(TPE)
        rows.append((et, xt, h.symbol, h.trade, h))
    rows.sort(key=lambda x: (x[0], x[2]))
    eq = float(start)
    peak = float(start)
    peak_i = -1
    max_dd = 0.0
    dd_peak_i = dd_trough_i = -1
    taken: List[dict] = []
    busy = None
    for et, xt, sym, t, hit in rows:
        if busy is not None and et < busy:
            continue
        ret = lev * (t.pnl_pct / 100.0)
        nxt = max(0.0, eq * (1.0 + ret))
        rec = {
            "et": et,
            "xt": xt,
            "symbol": sym,
            "trade": t,
            "hit": hit,
            "before": eq,
            "after": nxt,
            "ret": ret,
        }
        taken.append(rec)
        i = len(taken) - 1
        if nxt >= peak:
            peak, peak_i = nxt, i
        dd = (peak - nxt) / peak if peak else 0.0
        if dd > max_dd:
            max_dd, dd_peak_i, dd_trough_i = dd, peak_i, i
        eq = nxt
        busy = xt
        if eq <= 0:
            break
    for rec in taken:
        rec["max_dd"] = max_dd
        rec["dd_peak_i"] = dd_peak_i
        rec["dd_trough_i"] = dd_trough_i
    return taken


def _compound_equity_svg(taken: Sequence[dict], width: int = 720, height: int = 220) -> str:
    if not taken:
        return "<p class='muted'>no trades</p>"
    ys = [float(taken[0]["before"])] + [float(x["after"]) for x in taken]
    xs = np.linspace(0, width, len(ys))
    ymin, ymax = float(min(ys)), float(max(ys))
    pad = max(4.0, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys[i]):.1f}" for i in range(len(ys)))
    start_y = yv(float(taken[0]["before"]))
    color = "#16a34a" if ys[-1] >= ys[0] else "#dc2626"
    peak_i = int(taken[0]["dd_peak_i"])
    trough_i = int(taken[0]["dd_trough_i"])
    # ys[0] is start; trade i ends at ys[i+1]
    px, py = xs[peak_i + 1], yv(ys[peak_i + 1])
    tx, ty = xs[trough_i + 1], yv(ys[trough_i + 1])
    peak_lbl = f"{taken[peak_i]['symbol']} {taken[peak_i]['et'].strftime('%m-%d')}"
    trough_lbl = f"{taken[trough_i]['symbol']} {taken[trough_i]['et'].strftime('%m-%d')}"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{start_y:.1f}" x2="{width}" y2="{start_y:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{pts}"/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#f0c14b"/>'
        f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4" fill="#ff7b72"/>'
        f'<text x="{min(px + 6, width - 120):.1f}" y="{max(14, py - 8):.1f}" fill="#f0c14b" font-size="11">{escape(peak_lbl)} 高點</text>'
        f'<text x="{min(tx + 6, width - 140):.1f}" y="{min(height - 8, ty + 16):.1f}" fill="#ff7b72" font-size="11">{escape(trough_lbl)} 谷底</text>'
        f"</svg>"
    )


def write_seq_html(
    path: Path,
    taken: Sequence[dict],
    *,
    days: int,
    start: float = 100.0,
    lev: float = 3.0,
    scanned: int = 80,
    pool: int = 0,
    featured_html: str = "",
    more_k_href: str = "",
) -> Path:
    if not taken:
        html = "<p>no trades</p>"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
    final = float(taken[-1]["after"])
    max_dd = float(taken[0]["max_dd"])
    peak_i = int(taken[0]["dd_peak_i"])
    trough_i = int(taken[0]["dd_trough_i"])
    peak, trough = taken[peak_i], taken[trough_i]
    wins = sum(1 for x in taken if x["trade"].pnl_pct > 0)
    stretch = taken[peak_i : trough_i + 1]
    worst = min(stretch[1:], key=lambda x: x["trade"].pnl_pct) if len(stretch) > 1 else trough
    zoom = [dict(x) for x in stretch]
    for z in zoom:
        z["dd_peak_i"] = 0
        z["dd_trough_i"] = len(zoom) - 1
    rows = []
    for x in stretch:
        cls = "pnl-win" if x["trade"].pnl_pct > 0 else "pnl-loss"
        mark = ""
        if x is peak:
            mark = " 高點"
        if x is trough:
            mark = " 谷底"
        if x is worst:
            mark += " 最痛"
        rows.append(
            f"<tr><td>{escape(x['et'].strftime('%m-%d %H:%M'))}</td>"
            f"<td>{escape(x['symbol'])}{mark}</td>"
            f"<td>{escape(x['trade'].exit_reason)}</td>"
            f"<td class='{cls}'>{x['trade'].pnl_pct:+.2f}%</td>"
            f"<td class='{cls}'>{x['ret']*100:+.2f}%</td>"
            f"<td>{x['after']:.2f}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>100 USDT ×{lev:.0f} 一次一單 · {days} 天</title>
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
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
.mini-chart{{margin:8px -6px 0;border-radius:10px;overflow:hidden}}
.k-block{{margin:14px 0 0}}
.k-block h2{{font-size:15px;margin:0 0 6px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{text-align:left;padding:6px 4px;border-bottom:1px solid #21262d}}
th{{color:#8b949e;font-weight:600}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>100 USDT · {lev:.0f} 倍 · 一次一單</h1>
<p class="muted">近 {days} 天 · {scanned} 檔訊號池 {pool} 筆 · 做成 {len(taken)} 單<br/>
全部本金開倉，平了才能開下一筆。不計資金費。黃點高點、紅點谷底。</p>
<div class="cards">
<div class="card">做成<b>{len(taken)}</b></div>
<div class="card">勝率<b>{100.0 * wins / len(taken):.1f}%</b></div>
<div class="card">最後<b class="{'pnl-win' if final >= start else 'pnl-loss'}">{final:.0f}</b></div>
<div class="card">最大回撤<b class="pnl-loss">{max_dd*100:.1f}%</b></div>
</div>
<div class="equity">{_compound_equity_svg(taken)}</div>
<p class="muted">整段線性圖。後面漲到 {final:.0f}，前面 {peak['after']:.0f}→{trough['after']:.0f} 看起來會扁，回撤放大見下圖。</p>
<div class="equity">{_compound_equity_svg(zoom)}</div>
<p class="muted">高點 {escape(peak['et'].strftime('%m-%d %H:%M'))} {escape(peak['symbol'])} 後 {peak['after']:.2f} USDT<br/>
谷底 {escape(trough['et'].strftime('%m-%d %H:%M'))} {escape(trough['symbol'])} 後 {trough['after']:.2f} USDT<br/>
最痛單筆 {escape(worst['et'].strftime('%m-%d %H:%M'))} {escape(worst['symbol'])} 價格 {worst['trade'].pnl_pct:+.2f}%（帳戶 {worst['ret']*100:+.2f}%）</p>
{featured_html}
{f"<p class='muted'><a href='{escape(more_k_href)}' style='color:#79c0ff'>全部做成單的 K 棒圖</a></p>" if more_k_href else ""}
</section>
<section class="summary">
<h1>高點 → 谷底這段</h1>
<p class="muted">{escape(peak['et'].strftime('%m-%d'))} 到 {escape(trough['et'].strftime('%m-%d'))} · {len(stretch)} 單 · {peak['after']:.0f} → {trough['after']:.0f}</p>
<table>
<thead><tr><th>進場</th><th>標的</th><th>出場</th><th>價格</th><th>帳戶</th><th>餘額</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</section>
</div>
</body></html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _featured_k_html(taken: Sequence[dict], img_prefix: str = "seq-img/") -> str:
    if not taken:
        return ""
    peak_i = int(taken[0]["dd_peak_i"])
    trough_i = int(taken[0]["dd_trough_i"])
    peak, trough = taken[peak_i], taken[trough_i]
    stretch = taken[peak_i : trough_i + 1]
    worst = min(stretch[1:], key=lambda x: x["trade"].pnl_pct) if len(stretch) > 1 else trough
    blocks = []
    for rec, title in (
        (peak, "高點 · 回撤開始前"),
        (worst, "最痛單筆"),
        (trough, "谷底 · 回撤結束"),
    ):
        img = rec.get("img")
        if not img:
            continue
        t = rec["trade"]
        blocks.append(
            f"<div class='k-block'><h2 style='font-size:15px;margin:16px 0 6px'>{title}</h2>"
            f"<p class='muted'>{escape(rec['symbol'])} {escape(rec['et'].strftime('%m-%d %H:%M'))} · "
            f"{escape(t.exit_reason)} · 價格 {t.pnl_pct:+.2f}% · "
            f"{rec['before']:.2f} → {rec['after']:.2f} USDT</p>"
            f"<div class='mini-chart'><img src='{img_prefix}{escape(img)}' alt='{escape(title)}' "
            "style='width:100%;display:block;border-radius:10px'/></div></div>"
        )
    return "".join(blocks)


def draw_seq_pngs(taken: Sequence[dict], img_dir: Path) -> Sequence[dict]:
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(taken, 1):
        hit = rec.get("hit")
        if hit is None:
            continue
        name = f"s{i:02d}_{hit.symbol}_{rec['et'].strftime('%m%d_%H%M')}.png"
        draw_trade_png(hit.df, rec["trade"], img_dir / name, i, title_extra=hit.symbol)
        rec["img"] = name
    return taken


def write_seq_k_html(
    path: Path,
    taken: Sequence[dict],
    *,
    img_src_prefix: str = "seq-img/",
    days: int = 60,
) -> Path:
    cards: List[str] = []
    for i, rec in enumerate(taken, 1):
        t = rec["trade"]
        img = rec.get("img")
        if not img:
            continue
        cls = "pnl-win" if t.pnl_pct > 0 else "pnl-loss"
        cards.append(
            "<article class='summary' style='margin-bottom:14px'>"
            f"<h1>#{i} · {escape(rec['symbol'])} · {escape(t.exit_reason)}</h1>"
            f"<p class='muted'>{escape(rec['et'].strftime('%Y-%m-%d %H:%M'))} → "
            f"{escape(rec['xt'].strftime('%m-%d %H:%M'))} · "
            f"價格 {t.pnl_pct:+.2f}% · 帳戶 {rec['ret']*100:+.2f}% · "
            f"{rec['before']:.2f} → {rec['after']:.2f} USDT</p>"
            f"<div class='mini-chart'><img src='{img_src_prefix}{escape(img)}' alt='#{i} {escape(rec['symbol'])}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>一次一單 K 棒 · {days} 天</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:16px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px}}
.mini-chart{{margin:8px -6px 0;border-radius:10px;overflow:hidden}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
</style></head><body>
<div class="page">
<section class="summary" style="margin-bottom:14px">
<h1>做成的 {len(cards)} 筆 K 棒</h1>
<p class="muted">100 USDT ×3 · 一次一單 · 近 {days} 天。圖大，手機預覽可能慢。</p>
</section>
{''.join(cards)}
</div>
</body></html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = BRANCH, out_name: str = "view.html") -> Path:
    src = Path(src).resolve()
    rel = src.parent.relative_to(REPO.resolve()).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    text = text.replace("src='seq-img/", f"src='{base}seq-img/")
    out = src.with_name(out_name)
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_symbol(
    symbol: str,
    days: int,
    detect_kw: dict,
    *,
    m5_exit: bool = True,
    max_hold: int = 36,
    lose_ma_closes: int = 0,
) -> tuple[List[Hit], dict]:
    meta = {"symbol": symbol, "bars": 0, "error": "", "n_trade": 0}
    try:
        df = fetch_klines(symbol, days=days + 40)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 40:
        meta["error"] = "too_few_bars"
        return [], meta
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, **detect_kw)
    df_5m: Optional[pd.DataFrame] = None
    if m5_exit and sigs:
        try:
            df_5m = fetch_klines(symbol, interval="5m", days=days + 3)
        except Exception as exc:  # noqa: BLE001
            meta["m5_error"] = str(exc)[:80]
            df_5m = None
    trades = simulate(df, sigs, df_5m=df_5m, m5_exit=m5_exit, max_hold=max_hold, lose_ma_closes=lose_ma_closes)
    cutoff = datetime.now(TPE) - timedelta(days=days)
    hits = [Hit(symbol, df, t) for t in trades if df.index[t.entry_idx] >= cutoff]
    meta["n_trade"] = len(hits)
    meta["funnel"] = funnel
    return hits, meta


def merge_funnels(acc: Dict[str, int], part: Dict[str, int]) -> None:
    for k, v in part.items():
        acc[k] = acc.get(k, 0) + int(v)


STRICT_DETECT = dict(
    min_bars_below=10,
    max_bars_below=36,
    min_depth_pct=0.028,
    min_impulse_pct=0.023,
    min_undercut_pct=0.014,
    min_flush_atr=2.8,
    require_drawn_w=True,
)


def detect_kwargs(args) -> dict:
    extra = dict(
        target_r=args.target_r,
        max_retest_wait=getattr(args, "retest_wait", 18),
        retest_tag_pct=getattr(args, "retest_pct", 0.3) / 100.0,
        retest_tag_atr=getattr(args, "retest_atr", 0.4),
        max_ma_drift_pct=getattr(args, "max_ma_drift", 1.5),
        min_ribbon_atr=getattr(args, "min_ribbon_atr", 0.50),
        ma200_near_pct=getattr(args, "ma200_near", 1.0),
        min_ma_cluster_atr=getattr(args, "min_ma_cluster_atr", 1.0),
        min_break_vol=getattr(args, "min_break_vol", 0.8),
        min_mid_ma_atr=getattr(args, "min_mid_ma_atr", 0.50),
        require_h4_ma25=bool(getattr(args, "require_h4_ma25", False)),
        h4_ma25_tol_pct=float(getattr(args, "h4_ma25_tol", 0.0)),
        max_prior_spike_pct=float(getattr(args, "max_prior_spike", 0.0)) / 100.0,
        prior_spike_bars=int(getattr(args, "prior_spike_bars", 24)),
        min_reclaim_entry_bars=int(getattr(args, "min_reclaim_entry_bars", 0)),
        min_decline_pct=float(getattr(args, "min_decline", 0.0)) / 100.0,
        max_flat_bars=int(getattr(args, "max_flat_bars", -1)),
        max_reclaim_entry_bars=int(getattr(args, "max_reclaim_entry_bars", 0)),
        h4_max_ma_drop_pct=float(getattr(args, "h4_max_ma_drop", 0.0)),
        h4_drop_ma=int(getattr(args, "h4_drop_ma", 120)),
        h4_drop_bars=int(getattr(args, "h4_drop_bars", 24)),
        max_entry_ext_atr=float(getattr(args, "max_entry_ext_atr", 0.0)),
        max_overhead_knot_atr=float(getattr(args, "max_overhead_knot_atr", 0.0)),
        min_entry_vol=float(getattr(args, "min_entry_vol", 0.0)),
        require_ma7_above_ma14=bool(getattr(args, "require_ma7_above_ma14", False)),
        min_overhead_ma_atr=float(getattr(args, "min_overhead_ma_atr", 0.0)),
        h4_max_ma_rise_pct=float(getattr(args, "h4_max_ma_rise", 999.0)),
    )
    if getattr(args, "strict", False):
        return dict(STRICT_DETECT, **extra)
    return dict(
        min_bars_below=args.min_bars,
        max_bars_below=args.max_bars,
        min_depth_pct=args.min_depth / 100.0,
        min_impulse_pct=args.min_impulse / 100.0,
        min_undercut_pct=args.min_undercut / 100.0,
        min_flush_atr=args.min_atr,
        **extra,
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

    ex_inv, ex_vol = bool(getattr(args, "exclude_inverse", False)), bool(getattr(args, "exclude_vol", False))
    if ex_inv or ex_vol:
        before = list(symbols)
        symbols = filter_symbols(symbols, exclude_inverse=ex_inv, exclude_vol=ex_vol)
        dropped = [s for s in before if s not in symbols]
        print(f"排除反向/波動率商品 {len(dropped)} 檔: {','.join(dropped)}", file=sys.stderr)
    print(f"掃描 {len(symbols)} 檔  1h  近 {args.days} 天", file=sys.stderr)
    kw = detect_kwargs(args)
    hits: List[Hit] = []
    funnel: Dict[str, int] = {}
    errors = 0

    m5_exit = bool(getattr(args, "m5_exit", True))
    max_hold = int(getattr(args, "max_hold", 36))
    lose_ma_closes = int(getattr(args, "lose_ma_closes", 0))
    def work(sym: str) -> tuple[List[Hit], dict]:
        return scan_symbol(
            sym,
            args.days,
            kw,
            m5_exit=m5_exit,
            max_hold=max_hold,
            lose_ma_closes=lose_ma_closes,
        )

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
            f"weak={funnel.get('weak_flush', 0)} weakbrk={funnel.get('weak_break', 0)} "
            f"notw={funnel.get('not_w', 0)} "
            f"nostack={funnel.get('no_stack', 0)} noreclaim={funnel.get('no_reclaim', 0)} "
            f"noretest={funnel.get('no_retest', 0)} drift={funnel.get('ma_drift', 0)} "
            f"tangled={funnel.get('tangled', 0)} cluster={funnel.get('ma_cluster', 0)} "
            f"midcluster={funnel.get('mid_ma_cluster', 0)} "
            f"above_ma200={funnel.get('above_ma200', 0)} quietbrk={funnel.get('quiet_break', 0)} "
            f"h4bear={funnel.get('h4_bear', 0)} h4drop={funnel.get('h4_ma_drop', 0)} spike={funnel.get('prior_spike', 0)} "
            f"early={funnel.get('early_retest', 0)} late={funnel.get('late_retest', 0)} "
            f"smalldecl={funnel.get('small_decline', 0)} dead={funnel.get('dead_tape', 0)} "
            f"ext={funnel.get('entry_ext', 0)} knot={funnel.get('overhead_knot', 0)} "
            f"qentry={funnel.get('quiet_entry', 0)} ma7lt14={funnel.get('ma7_below_ma14', 0)}"
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
            f"{t.signal.flush_atr:.1f}atr under={t.signal.undercut_pct*100:.1f}% bvol={t.signal.break_vol_ratio:.2f}x{mark}"
        )

    html_path = Path(args.html) if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        extras = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
        keep_for_cards = set(KEEP) | set(extras)
        if args.days <= 8 or len(hits) <= 80:
            cards = hits
            card_note = ""
        else:
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
            strict=bool(args.strict),
            m5_exit=m5_exit,
            lose_ma_closes=lose_ma_closes,
            max_hold=max_hold,
        )
        view = write_view_html(out)
        print(f"html={out}")
        print(f"view={view}")
        if args.days >= 30:
            taken = one_at_a_time_path(hits, start=100.0, lev=3.0)
            draw_seq_pngs(taken, html_path.parent / "seq-img")
            rel = html_path.resolve().parent.relative_to(REPO.resolve()).as_posix()
            more_k = (
                "https://htmlpreview.github.io/?"
                f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{BRANCH}/"
                f"{rel}/seq-k.html"
            )
            write_seq_html(
                html_path.parent / "seq.html",
                taken,
                days=args.days,
                scanned=len(symbols),
                pool=len(hits),
                featured_html=_featured_k_html(taken),
                more_k_href=more_k,
            )
            write_seq_k_html(html_path.parent / "seq-k.html", taken, days=args.days)
            write_view_html(html_path.parent / "seq.html", out_name="seq.html")
            write_view_html(html_path.parent / "seq-k.html", out_name="seq-k.html")
            print(f"seq={html_path.parent / 'seq.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="幣安 1h MA25 筆畫 W，站回後回測再進")
    p.add_argument("--symbols", default="", help="只掃這些，逗號分隔，例如 AVGOUSDT,ONDSUSDT")
    p.add_argument("--universe", action="store_true", help="即使指定 --symbols 也掃流動永續")
    p.add_argument("--limit", type=int, default=80, help="流動永續最多幾檔（成交額由高到低）")
    p.add_argument("--min-quote", type=float, default=8_000_000)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-bars", type=int, default=4, help="在 MA25 下至少幾根 1h")
    p.add_argument("--max-bars", type=int, default=36, help="在 MA25 下最多幾根 1h")
    p.add_argument("--min-depth", type=float, default=1.8, help="相對 MA25 最低深度 %")
    p.add_argument("--min-impulse", type=float, default=0.0, help="4 根急殺最低幅度 %（0=寬鬆）")
    p.add_argument("--min-undercut", type=float, default=1.4, help="跌破急殺前平台低點 %（預設 1.4；0=關）")
    p.add_argument("--min-atr", type=float, default=0.0, help="急殺至少幾個 ATR（0=寬鬆）")
    p.add_argument("--strict", action="store_true", help="筆畫那種：先穿上 MA25，中間深 V 破底，再站回")
    p.add_argument("--retest-wait", type=int, default=18, help="站回後最多等幾根 1h 回測 MA25")
    p.add_argument("--retest-pct", type=float, default=0.3, help="回測吻 MA25 容許 %")
    p.add_argument("--retest-atr", type=float, default=0.4, help="回測吻 MA25 容許 ATR 倍數")
    p.add_argument("--target-r", type=float, default=2.0, help="2R 目標（--no-m5-exit 時才用；0=關閉）")
    p.add_argument(
        "--m5-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="出場用五分 M 破頸線（預設開；關則回到 2R 目標）",
    )
    p.add_argument(
        "--max-ma-drift",
        type=float,
        default=1.5,
        help="跌破到站回 MA25 最大漂移 %（0=關閉）",
    )
    p.add_argument(
        "--min-ribbon-atr",
        type=float,
        default=0.50,
        help="跌破期間 MA7/14/25 帶寬中位數至少幾個 ATR（0=關閉）；均線太糾結就丟掉",
    )
    p.add_argument(
        "--ma200-near",
        type=float,
        default=1.0,
        help="破底相對 MA200 容許高出 %（0=必須 ≤ MA200；預設 1=附近如龍蝦）",
    )
    p.add_argument(
        "--min-ma-cluster-atr",
        type=float,
        default=1.0,
        help="破底 |MA200-MA25| 至少幾個 ATR（0=關閉）；長均線貼在 MA25 上＝六線糾成水平義大利麵",
    )
    p.add_argument(
        "--min-break-vol",
        type=float,
        default=0.8,
        help="破底那根量 / 前 20 根均量（不含破底棒）最低倍數（預設 0.8；0=關閉）",
    )
    p.add_argument(
        "--min-mid-ma-atr",
        type=float,
        default=0.50,
        help="進場 MA25/99/120 帶寬至少幾個 ATR（0=關閉）；粉紫綠糾成一坨＝回測走進長均線結",
    )
    p.add_argument(
        "--require-h4-ma25",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="進場時要求 4h 收盤 >= 4h MA25（可用 --h4-ma25-tol 容許偏低 %%；預設關）",
    )
    p.add_argument(
        "--h4-ma25-tol",
        type=float,
        default=0.0,
        help="4h MA25 容許偏低 %%（僅 --require-h4-ma25 時有效；1.5＝可低 1.5%%）",
    )
    p.add_argument(
        "--max-prior-spike",
        type=float,
        default=0.0,
        help="破底前 N 根最高 / 破底低 -1 上限 %%（0=關；15＝拒暴衝後回檔）",
    )
    p.add_argument(
        "--prior-spike-bars",
        type=int,
        default=24,
        help="prior spike 回看幾根 1h（預設 24）",
    )
    p.add_argument(
        "--min-reclaim-entry-bars",
        type=int,
        default=0,
        help="站回→進場至少隔幾根 1h（entry-reclaim；0=關；3＝拒過早回測/假站回）",
    )
    p.add_argument(
        "--min-decline",
        type=float,
        default=0.0,
        help="破底前跌幅下限 %%：跌破起點前 24 根最高 → 破底低（0=關；3.5＝拒一根影線假破底）",
    )
    p.add_argument(
        "--max-flat-bars",
        type=int,
        default=-1,
        help="進場前 24 根一字線（高低差 < 0.35×前 120 根中位數）最多幾根（-1=關；2＝拒休市死盤）",
    )
    p.add_argument(
        "--max-reclaim-entry-bars",
        type=int,
        default=0,
        help="站回→進場最多隔幾根 1h（0=關；9＝拒拖太久才回測）",
    )
    p.add_argument(
        "--exclude-inverse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="掃描清單拿掉反向/放空槓桿 ETF（SQQQ/SOXS/SKDD/TBT/TZA；預設關）",
    )
    p.add_argument(
        "--exclude-vol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="掃描清單拿掉波動率商品（UVXY；預設關）",
    )
    p.add_argument(
        "--h4-max-ma-drop",
        type=float,
        default=0.0,
        help="進場時 4h MA(--h4-drop-ma) 近 N 根 4h 跌幅上限 %%（0=關；6＝拒 4h 長均線急跌）",
    )
    p.add_argument("--h4-drop-ma", type=int, default=120, help="--h4-max-ma-drop 用哪條 4h 均線（預設 120）")
    p.add_argument("--h4-drop-bars", type=int, default=24, help="--h4-max-ma-drop 回看幾根 4h（預設 24＝4 天）")
    p.add_argument(
        "--max-entry-ext-atr",
        type=float,
        default=0.0,
        help="進場收盤高出 MA25 最多幾個 ATR（0=關；2＝拒回測棒收太遠的追價）",
    )
    p.add_argument(
        "--max-overhead-knot-atr",
        type=float,
        default=0.0,
        help="MA60/99/120 全在進場收盤之上且三線帶寬<=此 ATR 則丟（0=關；1.5＝頭上中期均線結）",
    )
    p.add_argument(
        "--min-entry-vol",
        type=float,
        default=0.0,
        help="進場那根量 / 前 20 根均量最低倍數（0=關；0.15＝拒休市/極度縮量回測）",
    )
    p.add_argument(
        "--require-ma7-above-ma14",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="進場時要求 1h MA7 > MA14（短均線已黃金交叉；預設關）",
    )
    p.add_argument(
        "--min-overhead-ma-atr",
        type=float,
        default=0.0,
        help="進場上方最近的長均線（MA60/99/120/200）距離至少幾個 ATR（0=關；0.8＝拒買在長均線天花板正下方）",
    )
    p.add_argument(
        "--h4-max-ma-rise",
        type=float,
        default=999.0,
        help="進場時 4h MA(--h4-drop-ma) 近 N 根 4h 漲幅上限 %%（999=關；0.5＝拒高位暴漲後主跌段接刀）",
    )
    p.add_argument(
        "--w-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="卡片畫咖啡色 W 筆畫（--no-w-overlay 關）",
    )
    p.add_argument("--recent", type=int, default=48, help="報告裡標近幾小時的新訊號")
    p.add_argument(
        "--lose-ma-closes",
        type=int,
        default=0,
        help="連幾根 1h 收盤掉 MA25 當備援出場（預設 0＝關閉；只留破底停損）",
    )
    p.add_argument(
        "--max-hold",
        type=int,
        default=36,
        help="最長持有幾根 1h（逾時出場）",
    )
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/binance-1h-ma25/index.html")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global DRAW_W_OVERLAY
    args = build_parser().parse_args(argv)
    if not getattr(args, "w_overlay", True):
        DRAW_W_OVERLAY = False
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
