"""
Shadow-neckline detection — back to the original wick/low-break logic,
with an optional volume-spike (爆量) filter.

Tiers:
- raw: original (Low breaks neckline + SMA14)
- volume: original + 爆量 (>=1.5×) + reject rising neckline
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class DetectParams:
    # original structure
    min_span: int = 1
    max_span: int = 30
    shoulder_sym: float = 0.15
    min_bias: float = 0.05
    max_bias: float = 99.0  # effectively uncapped (original had no max)
    # original break: wick/low
    use_close_break: bool = False
    min_break_pct: float = 0.0
    require_red: bool = False
    require_prev_close_below: bool = False
    min_body_pct: float = 0.0
    # unused in original (kept for API compat / off)
    sma25_soft: bool = False
    require_sma25_hard: bool = False
    min_abs_dist_ma99: float = 0.0
    max_near_above_ma99: float = 0.0
    min_abs_dist_ma200: float = 0.0
    max_near_above_ma200: float = 0.0
    # volume spike (爆量)
    min_vol_ratio: float = 0.0  # 0 = off; 1.5 = 爆量
    vol_lookback: int = 20
    # structure quality: classic H&S short prefers flat/descending neck
    reject_rising_neck: bool = False
    # avoid shorting into rising SMA200 support when price is glued to it
    reject_near_rising_sma200: bool = False
    sma200_slope_bars: int = 24  # 2h on 5m
    near_sma200_pct: float = 1.5  # |close-SMA200|/SMA200 < 1.5%
    # 15m context: 15分K 戳破並收在 200均線下 → 不做空（如 1000RATS）
    reject_15m_pierce_sma200: bool = False
    # 進場貼近 15m SMA200
    reject_near_sma200_15m: bool = False
    near_sma200_15m_pct: float = 1.5
    # meta
    max_chg24: float = 99.0
    cooldown_min: int = 30


RAW = DetectParams()
VOLUME = DetectParams(
    min_vol_ratio=1.5,  # 爆量：破位 K 量能 ≥ 1.5× 近 20 均量
    reject_rising_neck=True,  # 上升頸線（右肩抬高）不空
    reject_near_rising_sma200=True,  # 上彎 SMA200 附近不空（如 BEAT）
    reject_15m_pierce_sma200=True,  # 15分K戳破200均線且收下方不空
    reject_near_sma200_15m=True,  # 貼近 15m SMA200 不空
    cooldown_min=30,
)

# Back-compat aliases (recommended path is now original + 爆量)
BALANCED = VOLUME
STRICT = DetectParams(
    min_vol_ratio=2.0,
    reject_rising_neck=True,
    reject_near_rising_sma200=True,
    reject_15m_pierce_sma200=True,
    reject_near_sma200_15m=True,
    cooldown_min=30,
)


def _find_peaks(close: np.ndarray, curr_idx: int) -> list[int]:
    window = 2
    peaks = []
    for i in range(max(window, curr_idx + 1 - 80), curr_idx + 1 - window):
        if i + window > curr_idx:
            break
        if close[i] == close[i - window : i + window + 1].max():
            peaks.append(i)
    return peaks


def detect_at_index(
    close,
    high,
    low,
    open_,
    sma200,
    sma14,
    sma25,
    sma99,
    volume,
    timestamp,
    sma200_15,
    curr_idx: int,
    p: DetectParams,
):
    """Original shadow-neckline at curr_idx, plus optional 爆量 filter."""
    if curr_idx + 1 < 250:
        return False, None

    peaks = _find_peaks(close, curr_idx)
    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1, h2, h3 = high[p1], high[p2], high[p3]

    # Head must be the local 4h high
    if h2 < high[max(0, p2 - 48) : p2].max():
        return False, None

    s200 = sma200[p2]
    if np.isnan(s200) or s200 == 0:
        return False, None
    bias = (h2 - s200) / s200
    if bias < p.min_bias or bias > p.max_bias:
        return False, None

    span = p3 - p1
    if not (p.min_span <= span <= p.max_span):
        return False, None
    if not (h2 > h1 and h2 > h3):
        return False, None
    if abs(h1 - h3) / max(h1, h3) >= p.shoulder_sym:
        return False, None
    if curr_idx <= p3:
        return False, None

    slope = (h3 - h1) / (p3 - p1) if p3 != p1 else 0.0
    # Rising neckline (right shoulder higher than left) → skip
    if p.reject_rising_neck and slope > 0:
        return False, None

    neck = h1 + slope * (curr_idx - p1)
    s14 = sma14[curr_idx]
    if np.isnan(s14):
        return False, None

    # Original trigger: Low pierces neckline + SMA14
    if not (low[curr_idx] < neck and low[curr_idx] < s14):
        return False, None
    if high[curr_idx] >= h2:
        return False, None

    br = (neck - low[curr_idx]) / neck if neck else 0.0
    if br < p.min_break_pct:
        return False, None

    if p.require_red and not (close[curr_idx] < open_[curr_idx]):
        return False, None

    # 爆量：破位 K 量能相對近 N 均量
    vol_ratio = None
    if p.min_vol_ratio > 0:
        lb = p.vol_lookback
        if curr_idx < lb:
            return False, None
        v_avg = float(np.nanmean(volume[curr_idx - lb : curr_idx]))
        if not (v_avg > 0) or np.isnan(v_avg):
            return False, None
        vol_ratio = float(volume[curr_idx]) / v_avg
        if vol_ratio < p.min_vol_ratio:
            return False, None

    s200_e = sma200[curr_idx]
    dist200 = None
    sma200_slope_pct = None
    if not np.isnan(s200_e) and s200_e != 0:
        dist200 = (close[curr_idx] - s200_e) / s200_e
        sb = p.sma200_slope_bars
        if curr_idx >= sb:
            s200_prev = sma200[curr_idx - sb]
            if not np.isnan(s200_prev) and s200_prev != 0:
                sma200_slope_pct = (s200_e - s200_prev) / s200_prev
        # 上彎 SMA200 + 價格貼近 → 易當支撐，不做空
        if p.reject_near_rising_sma200:
            if sma200_slope_pct is None or dist200 is None:
                return False, None
            if sma200_slope_pct > 0 and abs(dist200) * 100.0 < p.near_sma200_pct:
                return False, None
        # 15分K 戳破 200均線且收在下方 → 不做空
        if p.reject_15m_pierce_sma200 and timestamp is not None:
            bucket_ms = 15 * 60 * 1000
            bucket = int(timestamp[curr_idx] // bucket_ms * bucket_ms)
            lo = curr_idx
            while lo > 0 and int(timestamp[lo - 1] // bucket_ms * bucket_ms) == bucket:
                lo -= 1
            low15 = float(np.nanmin(low[lo : curr_idx + 1]))
            if low15 < s200_e and close[curr_idx] < s200_e:
                return False, None

    dist200_15 = None
    s200_15 = None
    if sma200_15 is not None:
        s200_15 = sma200_15[curr_idx]
        if not np.isnan(s200_15) and s200_15 != 0:
            dist200_15 = (close[curr_idx] - s200_15) / s200_15
            if p.reject_near_sma200_15m and abs(dist200_15) * 100.0 < p.near_sma200_15m_pct:
                return False, None

    neck_chg_pct = ((h3 - h1) / h1 * 100.0) if h1 else 0.0
    out = {
        "price": float(close[curr_idx]),
        "bias": round(bias * 100, 2),
        "line_val": round(float(neck), 6),
        "sma14": round(float(s14), 6),
        "close_break_pct": round(br * 100, 2),
        "neck_chg_pct": round(neck_chg_pct, 2),
        "span": int(span),
    }
    if vol_ratio is not None:
        out["vol_ratio"] = round(vol_ratio, 2)
    # optional extras when arrays present
    if sma25 is not None and not np.isnan(sma25[curr_idx]):
        out["sma25"] = round(float(sma25[curr_idx]), 6)
    if sma99 is not None and not np.isnan(sma99[curr_idx]) and sma99[curr_idx] != 0:
        dist99 = (close[curr_idx] - sma99[curr_idx]) / sma99[curr_idx]
        out["sma99"] = round(float(sma99[curr_idx]), 6)
        out["dist_ma99_pct"] = round(dist99 * 100, 2)
    if dist200 is not None:
        out["sma200"] = round(float(s200_e), 6)
        out["dist_ma200_pct"] = round(dist200 * 100, 2)
    if sma200_slope_pct is not None:
        out["sma200_slope_pct"] = round(sma200_slope_pct * 100, 3)
    if dist200_15 is not None:
        out["sma200_15"] = round(float(s200_15), 6)
        out["dist_ma200_15m_pct"] = round(dist200_15 * 100, 2)
    return True, out


def _sma200_15m_on_5m(df: pd.DataFrame) -> np.ndarray:
    """Map 15m SMA200 onto each 5m bar (NaN until 15m SMA200 is ready)."""
    if df.empty:
        return np.full(0, np.nan)
    bucket_ms = 15 * 60 * 1000
    d = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    d["bucket"] = (d["timestamp"].astype("int64") // bucket_ms) * bucket_ms
    g = (
        d.groupby("bucket", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )
    g["sma200"] = ta.sma(g["close"], length=200)
    mapped = d["bucket"].map(g.set_index("bucket")["sma200"])
    return mapped.to_numpy(dtype=float)


def prepare_indicators(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    timestamp = df["timestamp"].to_numpy(dtype=np.int64)
    sma200 = ta.sma(df["close"], length=200).to_numpy(dtype=float)
    sma14 = ta.sma(df["close"], length=14).to_numpy(dtype=float)
    sma25 = ta.sma(df["close"], length=25).to_numpy(dtype=float)
    sma99 = ta.sma(df["close"], length=99).to_numpy(dtype=float)
    sma200_15 = _sma200_15m_on_5m(df)
    return close, high, low, open_, sma200, sma14, sma25, sma99, volume, timestamp, sma200_15


def params_dict(p: DetectParams) -> dict:
    return asdict(p)
