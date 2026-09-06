"""15m MA5 / MA20 / MA99 多頭排列，那根剛收盤站上 MA200 才算。

多頭排列 = MA5 > MA20 > MA99。
剛站上 = 前一根收盤還在 200 下面（或剛好貼著），這根收盤才站上。
已經站在 200 上面、只是均線才排好的，不算。
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


def is_stacked(d: dict, i: int) -> bool:
    """MA5 > MA20 > MA99 多頭排列。"""
    if i < 0 or i >= len(d["c"]):
        return False
    m5, m20, m99 = d["m5"][i], d["m20"][i], d["m99"][i]
    if any(np.isnan(x) or x <= 0 for x in (m5, m20, m99)):
        return False
    return bool(m5 > m20 > m99)


def is_above_200(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    m200 = d["m200"][i]
    if np.isnan(m200) or m200 <= 0:
        return False
    return bool(d["c"][i] > m200)


def just_stood_on_200(d: dict, i: int) -> bool:
    """這根才剛收盤站上 200。"""
    return is_above_200(d, i) and not is_above_200(d, i - 1)


def signal_at(d: dict, i: int) -> AlignSignal | None:
    """5>20>99 多頭排列，且第 i 根剛站上 MA200。"""
    if i < MIN_BARS or i >= len(d["c"]):
        return None
    if not is_stacked(d, i):
        return None
    if not just_stood_on_200(d, i):
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
