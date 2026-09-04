"""15m 黏帶擠壓：均線纏在 200MA 附近，放量突破箱頂且收盤仍靠近 200 才進場。

對應幣安那種圖：先橫盤把 MA7/14/25/99/120/200 黏成一條，價格在 200 附近晃，
然後一根放量陽線打出箱頂，但收盤離 200 還不遠——這時進，不追已經直豎的那一段。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MA_PERIODS = (7, 14, 25, 99, 120, 200)

LOOKBACK = 24  # 盤整回看：24 根 15m ≈ 6 小時
MAX_RIBBON = 0.012  # 六條均線寬度 ≤ 1.2%
MAX_BOX = 0.030  # 箱體高低差 ≤ 3.0%
MAX_MA200_SLOPE = 0.008  # 近 20 根 MA200 斜率絕對值 ≤ 0.8%
NEAR_200 = 0.015  # 收盤離 200 ≤ 1.5% 算「靠近」
MIN_NEAR_FRAC = 0.70
MAX_MA7_VS_200 = 0.012  # 突破前 MA7 仍黏著 200
MAX_MA200_ABOVE_BOX = 0.012  # 200 可以略高於箱頂，但不能遠在天上
MIN_BARS_AT_OR_BELOW = 4

MIN_VOL_RATIO = 1.60
MIN_RANGE_EXPAND = 1.50
MAX_ENTRY_EXT = 0.015  # 進場收盤仍 ≤ 1.5% 高於 200
MIN_BODY_FRAC = 0.35
MIN_RISK = 0.004
MAX_RISK = 0.018
STOP_BUFFER = 0.002
TARGET_R = 2.0
HOLD_BARS = 16
MIN_GAP_BARS = 16


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_indicators(d: dict) -> dict:
    out = dict(d)
    c, v = d["c"], d["v"]
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    out["v20"] = sma(v, 20)
    return out


@dataclass(frozen=True)
class SqueezeSignal:
    idx: int
    open: float
    high: float
    low: float
    close: float
    ma7: float
    ma14: float
    ma25: float
    ma99: float
    ma120: float
    ma200: float
    ribbon: float
    box_high: float
    box_low: float
    box_pct: float
    near_frac: float
    slope20: float
    vol_ratio: float
    expand: float
    ext: float
    entry: float
    stop: float
    target: float

    @property
    def quality(self) -> str:
        if self.ribbon <= 0.006 and self.ext <= 0.010 and self.vol_ratio >= 2.0:
            return "A"
        if self.ribbon <= 0.010 and self.ext <= 0.013:
            return "B"
        return "C"

    @property
    def risk_pct(self) -> float:
        return (self.entry - self.stop) / self.entry if self.entry else 0.0


def _mas_ok(d: dict, i: int) -> bool:
    keys = [f"m{n}" for n in MA_PERIODS] + ["v20"]
    return not any(np.isnan(d[k][i]) for k in keys)


def signal_at(d: dict, i: int) -> SqueezeSignal | None:
    """第 i 根若是「黏帶箱頂放量突破、收盤仍靠近 200」則回傳訊號。"""
    if i < 220 or i >= len(d["c"]):
        return None
    if not _mas_ok(d, i) or not _mas_ok(d, i - 1):
        return None
    if i - LOOKBACK < 200:
        return None

    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    v20 = d["v20"]

    prev = i - 1
    mas_prev = np.array(
        [m7[prev], m14[prev], m25[prev], m99[prev], m120[prev], m200[prev]],
        dtype=float,
    )
    lo_m, hi_m = float(mas_prev.min()), float(mas_prev.max())
    if lo_m <= 0:
        return None
    ribbon = hi_m / lo_m - 1.0
    if ribbon > MAX_RIBBON:
        return None
    if abs(float(m7[prev] / m200[prev] - 1.0)) > MAX_MA7_VS_200:
        return None

    w0, w1 = i - LOOKBACK, i
    if np.isnan(m200[w0:w1]).any() or np.any(m200[w0:w1] <= 0):
        return None

    box_hi = float(h[w0:w1].max())
    box_lo = float(l[w0:w1].min())
    mid = (box_hi + box_lo) / 2.0
    if mid <= 0:
        return None
    box_pct = (box_hi - box_lo) / mid
    if box_pct > MAX_BOX:
        return None

    ma200_prev = float(m200[prev])
    if ma200_prev < box_lo * 0.995:
        return None
    if ma200_prev > box_hi * (1.0 + MAX_MA200_ABOVE_BOX):
        return None

    if i < 20 or np.isnan(m200[i - 20]) or m200[i - 20] <= 0:
        return None
    slope20 = float(m200[i] / m200[i - 20] - 1.0)
    if abs(slope20) > MAX_MA200_SLOPE:
        return None

    near = np.abs(c[w0:w1] / m200[w0:w1] - 1.0)
    near_frac = float((near <= NEAR_200).mean())
    if near_frac < MIN_NEAR_FRAC:
        return None
    at_or_below = int(np.sum(c[w0:w1] <= m200[w0:w1] * 1.003))
    if at_or_below < MIN_BARS_AT_OR_BELOW:
        return None

    if c[i] <= o[i]:
        return None
    if c[i] <= m200[i]:
        return None
    if c[i] <= box_hi:
        return None

    ext = float(c[i] / m200[i] - 1.0)
    if ext < 0 or ext > MAX_ENTRY_EXT:
        return None

    if v20[i] <= 0 or np.isnan(v20[i]):
        return None
    vr = float(v[i] / v20[i])
    if vr < MIN_VOL_RATIO:
        return None

    ranges = h[w0:w1] - l[w0:w1]
    med_rng = float(np.median(ranges))
    this_rng = float(h[i] - l[i])
    if med_rng <= 0 or this_rng < MIN_RANGE_EXPAND * med_rng:
        return None
    body = float(c[i] - o[i])
    if this_rng <= 0 or body / this_rng < MIN_BODY_FRAC:
        return None

    # 只吃第一根突破：前一根已經打出箱頂且站上 200，這根就是追價
    if c[prev] > box_hi and c[prev] > m200[prev]:
        return None

    entry = float(c[i])
    stop = min(box_lo, float(m200[i])) * (1.0 - STOP_BUFFER)
    if entry <= stop:
        return None
    risk = (entry - stop) / entry
    if risk < MIN_RISK or risk > MAX_RISK:
        return None
    target = entry + TARGET_R * (entry - stop)

    return SqueezeSignal(
        idx=i,
        open=float(o[i]),
        high=float(h[i]),
        low=float(l[i]),
        close=float(c[i]),
        ma7=float(m7[i]),
        ma14=float(m14[i]),
        ma25=float(m25[i]),
        ma99=float(m99[i]),
        ma120=float(m120[i]),
        ma200=float(m200[i]),
        ribbon=ribbon,
        box_high=box_hi,
        box_low=box_lo,
        box_pct=box_pct,
        near_frac=near_frac,
        slope20=slope20,
        vol_ratio=vr,
        expand=this_rng / med_rng,
        ext=ext,
        entry=entry,
        stop=stop,
        target=target,
    )


def detect_signals(d: dict, *, min_gap_bars: int = MIN_GAP_BARS) -> list[SqueezeSignal]:
    """全序列掃描。同一段擠壓只留第一根合格突破。"""
    out: list[SqueezeSignal] = []
    last = -10_000
    for i in range(220, len(d["c"])):
        if i - last < min_gap_bars:
            continue
        sig = signal_at(d, i)
        if sig is None:
            continue
        out.append(sig)
        last = i
    return out


@dataclass(frozen=True)
class TradeResult:
    signal: SqueezeSignal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float


def simulate_trades(d: dict, signals: list[SqueezeSignal], *, hold_bars: int = HOLD_BARS) -> list[TradeResult]:
    """下一根開盤進；先停損、再停利、再到時收盤。持倉中不重疊新單。"""
    results: list[TradeResult] = []
    busy_until = -1
    n = len(d["c"])
    for sig in signals:
        entry_idx = sig.idx + 1
        if entry_idx >= n or entry_idx <= busy_until:
            continue
        entry = float(d["o"][entry_idx])
        if entry <= sig.stop:
            continue
        end = min(entry_idx + hold_bars, n - 1)
        exit_idx, exit_price, reason = end, float(d["c"][end]), "time"
        for k in range(entry_idx, end + 1):
            if float(d["l"][k]) <= sig.stop:
                exit_idx, exit_price, reason = k, sig.stop, "stop"
                break
            if float(d["h"][k]) >= sig.target:
                exit_idx, exit_price, reason = k, sig.target, "target"
                break
        pnl = (exit_price / entry - 1.0) * 100.0
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                exit_reason=reason,
                pnl_pct=float(pnl),
            )
        )
        busy_until = exit_idx
    return results


def summarize_trades(trades: list[TradeResult]) -> dict:
    pnls = [t.pnl_pct for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "pnl_pct": float(sum(pnls)),
        "avg_pct": float(sum(pnls) / n) if n else 0.0,
        "by_quality": {
            q: {
                "n": sum(1 for t in trades if t.signal.quality == q),
                "pnl": float(sum(t.pnl_pct for t in trades if t.signal.quality == q)),
            }
            for q in ("A", "B", "C")
        },
    }
