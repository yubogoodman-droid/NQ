"""15m MA5 / MA20 / MA99 多頭排列，整排站上 MA200。

剛排好的第一根才發訊號（已經排好的不重複叫）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MA_PERIODS = (5, 20, 99, 200)
MIN_BARS = 200


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_indicators(d: dict) -> dict:
    out = dict(d)
    c = d["c"]
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    return out


@dataclass(frozen=True)
class AlignSignal:
    idx: int
    open: float
    high: float
    low: float
    close: float
    ma5: float
    ma20: float
    ma99: float
    ma200: float
    ext: float

    @property
    def stack_pct(self) -> float:
        """MA5 相對 MA200 的張開幅度。"""
        if self.ma200 <= 0:
            return 0.0
        return self.ma5 / self.ma200 - 1.0


def is_aligned(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    m5, m20, m99, m200 = d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i]
    if any(np.isnan(x) or x <= 0 for x in (m5, m20, m99, m200)):
        return False
    return bool(d["c"][i] > m200 and m5 > m20 > m99 > m200)


def signal_at(d: dict, i: int) -> AlignSignal | None:
    """第 i 根剛變成「5>20>99>200 且收盤站上 200」。"""
    if i < MIN_BARS or i >= len(d["c"]):
        return None
    if not is_aligned(d, i):
        return None
    if is_aligned(d, i - 1):
        return None
    m200 = float(d["m200"][i])
    close = float(d["c"][i])
    return AlignSignal(
        idx=i,
        open=float(d["o"][i]),
        high=float(d["h"][i]),
        low=float(d["l"][i]),
        close=close,
        ma5=float(d["m5"][i]),
        ma20=float(d["m20"][i]),
        ma99=float(d["m99"][i]),
        ma200=m200,
        ext=close / m200 - 1.0,
    )


def detect_signals(d: dict) -> list[AlignSignal]:
    out: list[AlignSignal] = []
    for i in range(MIN_BARS, len(d["c"])):
        sig = signal_at(d, i)
        if sig is not None:
            out.append(sig)
    return out
