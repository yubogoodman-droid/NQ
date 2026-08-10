"""
Shared shadow-neckline detection helpers.

Tiers:
- raw: wick/low break (original)
- balanced: close confirm + soft SMA25 + SMA99 proximity rules
- strict: balanced-style close confirm + prev-bar confirm + SMA14<SMA25 + stricter structure
         + same SMA99 rules
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class DetectParams:
    # structure
    min_span: int = 8
    max_span: int = 50
    shoulder_sym: float = 0.15
    min_bias: float = 0.05
    max_bias: float = 2.0
    # break
    use_close_break: bool = True
    min_break_pct: float = 0.003  # of neck, by close or low
    require_red: bool = True
    require_prev_close_below: bool = False
    # ma filters
    sma25_soft: bool = True  # close<sma25 OR break>=1%
    require_sma25_hard: bool = False  # close<sma25 and sma14<sma25
    # sma99 rules (same across balanced/strict)
    min_abs_dist_ma99: float = 0.02  # |Δ| < 2% skip
    max_near_above_ma99: float = 0.08  # 0<=Δ<8% skip
    # meta
    max_chg24: float = 3.0
    cooldown_min: int = 60


BALANCED = DetectParams()
STRICT = DetectParams(
    min_span=12,
    max_span=60,
    shoulder_sym=0.10,
    min_break_pct=0.008,
    require_red=True,
    require_prev_close_below=True,
    sma25_soft=False,
    require_sma25_hard=True,
    min_abs_dist_ma99=0.02,
    max_near_above_ma99=0.08,
    max_chg24=3.0,  # keep TUT-class names; structure/confirm already stricter
    cooldown_min=150,
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
    curr_idx: int,
    p: DetectParams,
):
    if curr_idx + 1 < 250:
        return False, None

    peaks = _find_peaks(close, curr_idx)
    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1, h2, h3 = high[p1], high[p2], high[p3]

    if h2 < high[max(0, p2 - 48) : p2].max():
        return False, None

    s200 = sma200[p2]
    if np.isnan(s200):
        return False, None
    bias = (h2 - s200) / s200
    if not (p.min_bias <= bias <= p.max_bias):
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
    neck = h1 + slope * (curr_idx - p1)
    s14, s25, s99 = sma14[curr_idx], sma25[curr_idx], sma99[curr_idx]
    if np.isnan(s14) or np.isnan(s25) or np.isnan(s99) or s99 == 0:
        return False, None

    if p.use_close_break:
        if not (close[curr_idx] < neck and close[curr_idx] < s14):
            return False, None
        br = (neck - close[curr_idx]) / neck
    else:
        if not (low[curr_idx] < neck and low[curr_idx] < s14):
            return False, None
        br = (neck - low[curr_idx]) / neck

    if br < p.min_break_pct:
        return False, None
    if p.require_red and not (close[curr_idx] < open_[curr_idx]):
        return False, None

    if p.require_sma25_hard:
        if not (close[curr_idx] < s25 and s14 < s25):
            return False, None
    elif p.sma25_soft:
        if not (close[curr_idx] < s25 or br >= 0.01):
            return False, None

    if p.require_prev_close_below:
        if curr_idx - 1 <= p3:
            return False, None
        neck_prev = h1 + slope * ((curr_idx - 1) - p1)
        if not (close[curr_idx - 1] < neck_prev):
            return False, None

    # Shared SMA99 rules
    dist99 = (close[curr_idx] - s99) / s99
    if abs(dist99) < p.min_abs_dist_ma99:
        return False, None
    if 0 <= dist99 < p.max_near_above_ma99:
        return False, None

    if high[curr_idx] >= h2:
        return False, None

    return True, {
        "price": float(close[curr_idx]),
        "bias": round(bias * 100, 2),
        "line_val": round(float(neck), 6),
        "sma14": round(float(s14), 6),
        "sma25": round(float(s25), 6),
        "sma99": round(float(s99), 6),
        "dist_ma99_pct": round(dist99 * 100, 2),
        "close_break_pct": round(br * 100, 2),
        "span": int(span),
    }


def prepare_indicators(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    sma200 = ta.sma(df["close"], length=200).to_numpy(dtype=float)
    sma14 = ta.sma(df["close"], length=14).to_numpy(dtype=float)
    sma25 = ta.sma(df["close"], length=25).to_numpy(dtype=float)
    sma99 = ta.sma(df["close"], length=99).to_numpy(dtype=float)
    return close, high, low, open_, sma200, sma14, sma25, sma99


def params_dict(p: DetectParams) -> dict:
    return asdict(p)
