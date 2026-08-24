"""幣安 1 分 K：MA7 > MA14 > MA25 > MA99 多頭排列，上站同一根 1m 圖的 MA200。

所有均線都用 1 分鐘收盤 SMA，不用日線／小時線 MA200。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONS = (5, 15, 30, 60, 240)  # 1m 根數 → 5m / 15m / 30m / 1h / 4h


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_mas(d: dict) -> dict:
    """1 分鐘收盤的 SMA 7/14/25/99/200。"""
    out = dict(d)
    c, v = d["c"], d["v"]
    out["m7"] = sma(c, 7)
    out["m14"] = sma(c, 14)
    out["m25"] = sma(c, 25)
    out["m99"] = sma(c, 99)
    out["m200"] = sma(c, 200)
    out["v20"] = sma(v, 20)
    return out


def stack_ok(c, m7, m14, m25, m99, m200, i: int) -> bool:
    """7/14/25/99 多頭排列，且收盤在 1m MA200 上。不要求 MA99 已高於 MA200。"""
    vals = [c[i], m7[i], m14[i], m25[i], m99[i], m200[i]]
    if np.isnan(vals).any():
        return False
    return bool(c[i] > m7[i] > m14[i] > m25[i] > m99[i] and c[i] > m200[i])


def bars_below_ma200(c, m200, i: int) -> int:
    n = 0
    j = i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] <= m200[j]:
        n += 1
        j -= 1
    return n


@dataclass(frozen=True)
class BullSignal:
    idx: int
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    m7: float
    m14: float
    m25: float
    m99: float
    ma200: float
    vol_ratio: float
    ext_pct: float
    crossed_200: bool
    bars_below: int


@dataclass
class ForwardMove:
    bars: int
    ret_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None


@dataclass
class SignalRow:
    symbol: str
    sig: BullSignal
    time_ms: int
    entry: float
    quote_volume: float = 0.0
    rank: int = 0
    moves: dict[int, ForwardMove] = field(default_factory=dict)

    @property
    def vol_ratio(self) -> float:
        return self.sig.vol_ratio

    @property
    def ext_pct(self) -> float:
        return self.sig.ext_pct

    @property
    def crossed_200(self) -> bool:
        return self.sig.crossed_200

    @property
    def bars_below(self) -> int:
        return self.sig.bars_below


def signal_at(d: dict, i: int) -> BullSignal | None:
    if i < 1 or i >= len(d["c"]):
        return None
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25, m99, m200 = d["m7"], d["m14"], d["m25"], d["m99"], d["m200"]
    v20 = d["v20"]
    if not stack_ok(c, m7, m14, m25, m99, m200, i):
        return None
    vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
    ext = (c[i] / m200[i] - 1.0) * 100.0 if m200[i] else 0.0
    return BullSignal(
        idx=i,
        open=float(o[i]),
        high=float(h[i]),
        low=float(l[i]),
        close=float(c[i]),
        prev_close=float(c[i - 1]),
        m7=float(m7[i]),
        m14=float(m14[i]),
        m25=float(m25[i]),
        m99=float(m99[i]),
        ma200=float(m200[i]),
        vol_ratio=vr,
        ext_pct=float(ext),
        crossed_200=bool(c[i - 1] <= m200[i - 1] and c[i] > m200[i]),
        bars_below=bars_below_ma200(c, m200, i),
    )


def detect_combo(d: dict, *, min_gap_bars: int = 0, cross_only: bool = False) -> list[BullSignal]:
    """本根 收盤>MA7>14>25>99 且收盤>1m MA200；前一根還沒同時成立。

    cross_only：只保留「前收還在 1m MA200 下、本根收盤站上」——這才是上站，而不是已在線上又排一次均線。
    """
    c, m7, m14, m25, m99, m200 = d["c"], d["m7"], d["m14"], d["m25"], d["m99"], d["m200"]
    out: list[BullSignal] = []
    last_i = -10_000
    for i in range(1, len(c)):
        now_ok = stack_ok(c, m7, m14, m25, m99, m200, i)
        prev_ok = stack_ok(c, m7, m14, m25, m99, m200, i - 1)
        if not now_ok or prev_ok:
            continue
        if cross_only:
            if np.isnan(m200[i - 1]) or not (c[i - 1] <= m200[i - 1] and c[i] > m200[i]):
                continue
        if i - last_i < min_gap_bars:
            continue
        sig = signal_at(d, i)
        if sig is None:
            continue
        out.append(sig)
        last_i = i
    return out


def forward_moves(
    d: dict,
    sig: BullSignal,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[float, dict[int, ForwardMove]]:
    """進場用訊號下一根開盤。"""
    i = sig.idx
    nxt = i + 1
    if nxt >= len(d["c"]):
        return float("nan"), {h: ForwardMove(h, None, None, None) for h in horizons}
    entry = float(d["o"][nxt])
    moves: dict[int, ForwardMove] = {}
    for h in horizons:
        j = nxt + h - 1
        if j >= len(d["c"]):
            moves[h] = ForwardMove(h, None, None, None)
            continue
        last = float(d["c"][j])
        hi = float(d["h"][nxt : j + 1].max())
        lo = float(d["l"][nxt : j + 1].min())
        moves[h] = ForwardMove(
            h,
            (last / entry - 1.0) * 100.0,
            (hi / entry - 1.0) * 100.0,
            (lo / entry - 1.0) * 100.0,
        )
    return entry, moves


def summarize_rows(rows: list[SignalRow], horizon: int) -> dict:
    vals = [
        r.moves[horizon].ret_pct
        for r in rows
        if r.moves.get(horizon) and r.moves[horizon].ret_pct is not None
    ]
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "wr": float((arr > 0).mean() * 100.0),
        "avg": float(arr.mean()),
        "med": float(np.median(arr)),
    }
