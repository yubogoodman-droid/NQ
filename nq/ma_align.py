"""15m MA5 / MA20 / MA99 多頭排列，那根剛收盤站上 MA200 才算。

均線距離對齊 ETH 9/3 那張圖：99 與 200 幾乎黏在一起，
5/20/99/200 還縮成一條，不是已經散開的趨勢帶。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MA_PERIODS = (5, 20, 99, 200)
MIN_BARS = 200

# ETH 9/3 20:00 剛站上：ribbon 0.53%、99-200 0.52%、離 200 +0.09%
# 用戶截圖是 22:45 噴到 2488 的那根（離 200 +3.45%、帶寬 2.22%），
# 那時早已站上，不當訊號；圖要往後畫夠長才看得到那波。
MAX_RIBBON = 0.008  # 四條均線寬度 ≤ 0.8%
MAX_MA99_VS_200 = 0.007  # MA99 與 MA200 距離 ≤ 0.7%
MAX_ENTRY_EXT = 0.010  # 剛站上，收盤離 200 仍近


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
    ribbon: float
    gap99: float

    @property
    def stack_pct(self) -> float:
        if self.ma200 <= 0:
            return 0.0
        return self.ma5 / self.ma200 - 1.0


def _mas(d: dict, i: int) -> tuple[float, float, float, float] | None:
    vals = (d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i])
    if any(np.isnan(x) or x <= 0 for x in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def is_stacked(d: dict, i: int) -> bool:
    """MA5 > MA20 > MA99 多頭排列。"""
    if i < 0 or i >= len(d["c"]):
        return False
    mas = _mas(d, i)
    if mas is None:
        return False
    m5, m20, m99, _ = mas
    return bool(m5 > m20 > m99)


def ribbon_pct(d: dict, i: int) -> float | None:
    mas = _mas(d, i)
    if mas is None:
        return None
    lo, hi = min(mas), max(mas)
    return hi / lo - 1.0


def gap99_pct(d: dict, i: int) -> float | None:
    mas = _mas(d, i)
    if mas is None:
        return None
    return abs(mas[2] / mas[3] - 1.0)


def compact_like_eth(d: dict, i: int) -> bool:
    """均線距離類似 ETH 截圖：整排不散、99 貼著 200。"""
    rib = ribbon_pct(d, i)
    gap = gap99_pct(d, i)
    if rib is None or gap is None:
        return False
    return rib <= MAX_RIBBON and gap <= MAX_MA99_VS_200


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
    """5>20>99、均線黏得像 ETH 圖，且第 i 根剛站上 MA200。"""
    if i < MIN_BARS or i >= len(d["c"]):
        return None
    if not is_stacked(d, i):
        return None
    if not just_stood_on_200(d, i):
        return None
    if not compact_like_eth(d, i):
        return None
    m5, m20, m99, m200 = _mas(d, i)
    close = float(d["c"][i])
    ext = close / m200 - 1.0
    if ext < 0 or ext > MAX_ENTRY_EXT:
        return None
    return AlignSignal(
        idx=i,
        open=float(d["o"][i]),
        high=float(d["h"][i]),
        low=float(d["l"][i]),
        close=close,
        ma5=m5,
        ma20=m20,
        ma99=m99,
        ma200=m200,
        ext=ext,
        ribbon=float(ribbon_pct(d, i) or 0.0),
        gap99=float(gap99_pct(d, i) or 0.0),
    )


def detect_signals(d: dict) -> list[AlignSignal]:
    out: list[AlignSignal] = []
    for i in range(MIN_BARS, len(d["c"])):
        sig = signal_at(d, i)
        if sig is not None:
            out.append(sig)
    return out
