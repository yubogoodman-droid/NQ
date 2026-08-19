"""15 分 K：收盤高於 MA7 / MA14 / MA25 / MA200，且 7>14>25。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONS = (1, 2, 4, 8, 16, 32)  # 15m 根數 → 15m / 30m / 1h / 2h / 4h / 8h


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


def _stack_ok(c, m7, m14, m25, m200, i: int) -> bool:
    """收盤同時在 MA7 / MA14 / MA25 / MA200 之上，且 7>14>25。"""
    px, ma7, ma14, ma25, ma200 = c[i], m7[i], m14[i], m25[i], m200[i]
    if np.isnan([px, ma7, ma14, ma25, ma200]).any():
        return False
    return bool(px > ma7 > ma14 > ma25 and px > ma200)


@dataclass(frozen=True)
class BullSignal:
    """15m 收盤在 MA7/14/25/200 之上，且 7>14>25。"""

    idx: int
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    m7: float
    m14: float
    m25: float
    ma200: float
    vol_ratio: float
    ext_pct: float
    crossed_200: bool
    formed_align: bool

    # 舊欄位別名，給既有回測腳本用
    @property
    def ma200d(self) -> float:
        return self.ma200

    @property
    def ma200_15(self) -> float:
        return self.ma200

    @property
    def crossed_200d(self) -> bool:
        return self.crossed_200


def detect_combo(d: dict, *, min_gap_bars: int = 0) -> list[BullSignal]:
    """
    進場：這一根收盤 > MA7 > MA14 > MA25，且收盤 > 15 分 MA200，
    前一根還沒同時成立。

    crossed_200：前收還在 MA200 下，本根收盤站上（通知用這個）。
    formed_align：已經在 MA200 上，本根才收上短均／排成 7>14>25。
    """
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25, m200 = d["m7"], d["m14"], d["m25"], d["m200"]
    v20 = d["v20"]
    out: list[BullSignal] = []
    last_i = -10_000
    for i in range(200, len(c)):
        if np.isnan([m200[i], m200[i - 1]]).any():
            continue
        now_ok = _stack_ok(c, m7, m14, m25, m200, i)
        prev_ok = _stack_ok(c, m7, m14, m25, m200, i - 1)
        if not now_ok or prev_ok:
            continue
        if i - last_i < min_gap_bars:
            continue
        vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
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
                ma200=float(m200[i]),
                vol_ratio=vr,
                ext_pct=float(ext),
                crossed_200=bool(c[i - 1] <= m200[i - 1] and c[i] > m200[i]),
                formed_align=bool(c[i - 1] > m200[i - 1] and not _stack_ok(c, m7, m14, m25, m200, i - 1)),
            )
        )
        last_i = i
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
        return self.sig.crossed_200

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
    """進場用訊號下一根開盤。假突破 = 下一根 15 分收盤又跌回 MA200 下方。"""
    i = sig.idx
    nxt = i + 1
    if nxt >= len(d["c"]):
        return float("nan"), {h: ForwardMove(h, None, None, None, None) for h in horizons}
    entry = float(d["o"][nxt])
    moves: dict[int, ForwardMove] = {}
    ma = d["m200"]
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
        if h == 1 and not np.isnan(ma[nxt]):
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
