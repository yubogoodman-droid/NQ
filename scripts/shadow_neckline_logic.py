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
    # meta
    max_chg24: float = 99.0
    cooldown_min: int = 30


RAW = DetectParams()
VOLUME = DetectParams(
    min_vol_ratio=1.5,  # 爆量：破位 K 量能 ≥ 1.5× 近 20 均量
    reject_rising_neck=True,  # 上升頸線（右肩抬高）不空
    cooldown_min=30,
)

# Back-compat aliases (recommended path is now original + 爆量)
BALANCED = VOLUME
STRICT = DetectParams(
    min_vol_ratio=2.0,
    reject_rising_neck=True,
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
    s200_e = sma200[curr_idx]
    if not np.isnan(s200_e) and s200_e != 0:
        dist200 = (close[curr_idx] - s200_e) / s200_e
        out["sma200"] = round(float(s200_e), 6)
        out["dist_ma200_pct"] = round(dist200 * 100, 2)
    return True, out


def prepare_indicators(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    sma200 = ta.sma(df["close"], length=200).to_numpy(dtype=float)
    sma14 = ta.sma(df["close"], length=14).to_numpy(dtype=float)
    sma25 = ta.sma(df["close"], length=25).to_numpy(dtype=float)
    sma99 = ta.sma(df["close"], length=99).to_numpy(dtype=float)
    return close, high, low, open_, sma200, sma14, sma25, sma99, volume


def params_dict(p: DetectParams) -> dict:
    return asdict(p)
