"""15 分 K：MA7>MA14>MA25 多頭排列，且收盤站上日線 200 日均線。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONS = (1, 2, 4, 8, 16, 32)  # 15m 根數 → 15m / 30m / 1h / 2h / 4h / 8h
DAILY_MS = 24 * 60 * 60_000


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_15m_mas(d: dict) -> dict:
    out = dict(d)
    c, v = d["c"], d["v"]
    out["m7"] = sma(c, 7)
    out["m14"] = sma(c, 14)
    out["m25"] = sma(c, 25)
    out["m200"] = sma(c, 200)
    out["v20"] = sma(v, 20)
    return out


def attach_daily_ma200(d15: dict, daily: dict) -> dict:
    """把「已收盤」日線 SMA200 對齊到每根 15 分 K（不偷看當天未收盤日 K）。"""
    out = dict(d15)
    complete_t = daily["t"].astype(np.int64) + DAILY_MS
    ma200 = sma(daily["c"], 200)
    j = np.searchsorted(complete_t, out["t"], side="right") - 1
    mapped = np.full(len(out["t"]), np.nan, dtype=float)
    ok = j >= 0
    mapped[ok] = ma200[j[ok]]
    out["m200d"] = mapped
    return out


def _aligned(m7, m14, m25, i: int) -> bool:
    a, b, c = m7[i], m14[i], m25[i]
    if np.isnan([a, b, c]).any():
        return False
    return bool(a > b > c)


@dataclass(frozen=True)
class BullSignal:
    """組合剛成立：15m 多頭排列且收盤在 200 日上方。"""

    idx: int
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    m7: float
    m14: float
    m25: float
    ma200d: float
    ma200_15: float | None
    vol_ratio: float
    ext_pct: float
    crossed_200d: bool
    formed_align: bool


def detect_combo(d: dict, *, min_gap_bars: int = 0) -> list[BullSignal]:
    """
    進場：這一根同時滿足 MA7>MA14>MA25 且收盤 > 日線 MA200，
    前一根兩個條件沒有同時成立（避免多頭裡每根都叫）。

    crossed_200d：前收還在 200 日下（或剛好觸及），本根收盤站上。
    formed_align：已經在 200 日上，本根才排成 7>14>25。
    """
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m200d = d["m200d"]
    m200 = d.get("m200")
    v20 = d["v20"]
    out: list[BullSignal] = []
    last_i = -10_000
    for i in range(25, len(c)):
        if np.isnan([m200d[i], m200d[i - 1]]).any():
            continue
        now_ok = _aligned(m7, m14, m25, i) and c[i] > m200d[i]
        prev_ok = _aligned(m7, m14, m25, i - 1) and c[i - 1] > m200d[i - 1]
        if not now_ok or prev_ok:
            continue
        if i - last_i < min_gap_bars:
            continue
        vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        ext = (c[i] / m200d[i] - 1.0) * 100.0 if m200d[i] else 0.0
        ma15 = float(m200[i]) if m200 is not None and not np.isnan(m200[i]) else None
        out.append(
            BullSignal(
                idx=i,
                open=float(o[i]),
                high=float(h[i]),
                low=float(l[i]),
                close=float(c[i]),
                prev_close=float(c[i - 1]),
                m7=float(m7[i]),
                m14=float(m14[i]),
                m25=float(m25[i]),
                ma200d=float(m200d[i]),
                ma200_15=ma15,
                vol_ratio=vr,
                ext_pct=float(ext),
                crossed_200d=bool(c[i - 1] <= m200d[i - 1] and c[i] > m200d[i]),
                formed_align=bool(c[i - 1] > m200d[i - 1] and not _aligned(m7, m14, m25, i - 1)),
            )
        )
        last_i = i
    return out


def detect_15m_ma200_cross(d: dict) -> list[BullSignal]:
    """對照組：15m MA7>14>25，且收盤剛站上 15m MA200。"""
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25, m200 = d["m7"], d["m14"], d["m25"], d["m200"]
    m200d = d.get("m200d")
    v20 = d["v20"]
    out: list[BullSignal] = []
    for i in range(200, len(c)):
        if np.isnan([m7[i], m14[i], m25[i], m200[i], m200[i - 1]]).any():
            continue
        if not _aligned(m7, m14, m25, i):
            continue
        if not (c[i - 1] <= m200[i - 1] and c[i] > m200[i]):
            continue
        vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        ma_d = float(m200d[i]) if m200d is not None and not np.isnan(m200d[i]) else float("nan")
        ext = (c[i] / m200[i] - 1.0) * 100.0 if m200[i] else 0.0
        out.append(
            BullSignal(
                idx=i,
                open=float(o[i]),
                high=float(h[i]),
                low=float(l[i]),
                close=float(c[i]),
                prev_close=float(c[i - 1]),
                m7=float(m7[i]),
                m14=float(m14[i]),
                m25=float(m25[i]),
                ma200d=ma_d,
                ma200_15=float(m200[i]),
                vol_ratio=vr,
                ext_pct=float(ext),
                crossed_200d=False,
                formed_align=True,
            )
        )
    return out


@dataclass
class ForwardMove:
    bars: int
    ret_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    failed: bool | None


@dataclass
class SignalRow:
    symbol: str
    sig: BullSignal
    time_ms: int
    entry: float
    moves: dict[int, ForwardMove] = field(default_factory=dict)

    @property
    def vol_ratio(self) -> float:
        return self.sig.vol_ratio

    @property
    def crossed_200d(self) -> bool:
        return self.sig.crossed_200d

    @property
    def formed_align(self) -> bool:
        return self.sig.formed_align

    @property
    def ext_pct(self) -> float:
        return self.sig.ext_pct


def forward_moves(
    d: dict,
    sig: BullSignal,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[float, dict[int, ForwardMove]]:
    """進場用訊號下一根開盤。假突破 = 下一根 15 分收盤又跌回 200 日下方。"""
    i = sig.idx
    nxt = i + 1
    if nxt >= len(d["c"]):
        return float("nan"), {h: ForwardMove(h, None, None, None, None) for h in horizons}
    entry = float(d["o"][nxt])
    moves: dict[int, ForwardMove] = {}
    ma = d.get("m200d")
    for h in horizons:
        j = nxt + h - 1
        if j >= len(d["c"]):
            moves[h] = ForwardMove(h, None, None, None, None)
            continue
        last = float(d["c"][j])
        hi = float(d["h"][nxt : j + 1].max())
        lo = float(d["l"][nxt : j + 1].min())
        ret = (last / entry - 1.0) * 100.0
        mfe = (hi / entry - 1.0) * 100.0
        mae = (lo / entry - 1.0) * 100.0
        failed = None
        if h == 1 and ma is not None and not np.isnan(ma[nxt]):
            failed = float(d["c"][nxt]) < float(ma[nxt])
        moves[h] = ForwardMove(h, ret, mfe, mae, failed)
    return entry, moves


def summarize_rows(rows: list[SignalRow], horizon: int) -> dict:
    vals = [
        r.moves[horizon].ret_pct
        for r in rows
        if r.moves.get(horizon) and r.moves[horizon].ret_pct is not None
    ]
    mfes = [
        r.moves[horizon].mfe_pct
        for r in rows
        if r.moves.get(horizon) and r.moves[horizon].mfe_pct is not None
    ]
    maes = [
        r.moves[horizon].mae_pct
        for r in rows
        if r.moves.get(horizon) and r.moves[horizon].mae_pct is not None
    ]
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "mfe": 0.0, "mae": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "wr": float((arr > 0).mean() * 100.0),
        "avg": float(arr.mean()),
        "med": float(np.median(arr)),
        "mfe": float(np.mean(mfes)) if mfes else 0.0,
        "mae": float(np.mean(maes)) if maes else 0.0,
    }


def fail_rate(rows: list[SignalRow]) -> float:
    flags = [r.moves[1].failed for r in rows if r.moves.get(1) and r.moves[1].failed is not None]
    if not flags:
        return 0.0
    return float(sum(1 for x in flags if x) / len(flags) * 100.0)


def apply_filter(
    rows: list[SignalRow],
    *,
    crossed: bool | None = None,
    formed: bool | None = None,
    min_vol: float | None = None,
    max_ext: float | None = None,
) -> list[SignalRow]:
    out = rows
    if crossed:
        out = [r for r in out if r.crossed_200d]
    if formed:
        out = [r for r in out if r.formed_align]
    if min_vol is not None:
        out = [r for r in out if r.vol_ratio >= min_vol]
    if max_ext is not None:
        out = [r for r in out if r.ext_pct <= max_ext]
    return out
