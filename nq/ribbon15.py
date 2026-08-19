"""15 分 K 同時穿越 MA7 / 14 / 25 / 99 / 120 / 200。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MA_PERIODS = (7, 14, 25, 99, 120, 200)
HORIZONS = (1, 2, 4, 8, 16, 32)  # 15m 根數 → 15m / 30m / 1h / 2h / 4h / 8h


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_mas(d: dict) -> dict:
    c, v = d["c"], d["v"]
    out = dict(d)
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    out["v20"] = sma(v, 20)
    return out


def _mas_at(d: dict, i: int) -> np.ndarray | None:
    vals = np.array([d[f"m{n}"][i] for n in MA_PERIODS], dtype=float)
    if np.isnan(vals).any():
        return None
    return vals


@dataclass(frozen=True)
class RibbonBreak:
    """一根 K 從帶子下方收到帶子上方。"""

    idx: int
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    ma_min_prev: float
    ma_max_prev: float
    ma_min: float
    ma_max: float
    width_pct: float
    vol_ratio: float
    body_through: bool
    range_pct: float


def detect_long_breaks(d: dict) -> list[RibbonBreak]:
    """
    做多：前一根收盤完全在 6 條均線下方，這一根收盤完全站上 6 條均線。

    這樣一根 15 分 K 的高低區間必然同時穿過 MA7/14/25/99/120/200。
    """
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    v20 = d["v20"]
    out: list[RibbonBreak] = []
    start = max(MA_PERIODS) + 1
    for i in range(start, len(c)):
        prev = _mas_at(d, i - 1)
        curr = _mas_at(d, i)
        if prev is None or curr is None:
            continue
        lo_p, hi_p = float(prev.min()), float(prev.max())
        lo_c, hi_c = float(curr.min()), float(curr.max())
        if not (c[i - 1] < lo_p and c[i] > hi_c):
            continue
        if l[i] > lo_p:
            # 跳空越過帶子，不算「這根 K 穿過」
            continue
        width = (hi_p / lo_p - 1.0) * 100.0 if lo_p > 0 else float("inf")
        vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        body = o[i] < lo_p and c[i] > hi_c
        rng = (h[i] / l[i] - 1.0) * 100.0 if l[i] > 0 else 0.0
        out.append(
            RibbonBreak(
                idx=i,
                open=float(o[i]),
                high=float(h[i]),
                low=float(l[i]),
                close=float(c[i]),
                prev_close=float(c[i - 1]),
                ma_min_prev=lo_p,
                ma_max_prev=hi_p,
                ma_min=lo_c,
                ma_max=hi_c,
                width_pct=width,
                vol_ratio=vr,
                body_through=body,
                range_pct=rng,
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
    break_: RibbonBreak
    time_ms: int
    entry: float
    moves: dict[int, ForwardMove] = field(default_factory=dict)

    @property
    def body_through(self) -> bool:
        return self.break_.body_through

    @property
    def width_pct(self) -> float:
        return self.break_.width_pct

    @property
    def vol_ratio(self) -> float:
        return self.break_.vol_ratio


def forward_moves(d: dict, br: RibbonBreak, horizons: tuple[int, ...] = HORIZONS) -> tuple[float, dict[int, ForwardMove]]:
    """進場用訊號下一根開盤；假突破 = 下一根收盤又跌回最高均線下方。"""
    i = br.idx
    nxt = i + 1
    if nxt >= len(d["c"]):
        return float("nan"), {h: ForwardMove(h, None, None, None, None) for h in horizons}
    entry = float(d["o"][nxt])
    moves: dict[int, ForwardMove] = {}
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
        if h == 1:
            failed = float(d["c"][nxt]) < br.ma_max
        moves[h] = ForwardMove(h, ret, mfe, mae, failed)
    return entry, moves


def summarize_rows(rows: list[SignalRow], horizon: int) -> dict:
    vals = [r.moves[horizon].ret_pct for r in rows if r.moves.get(horizon) and r.moves[horizon].ret_pct is not None]
    mfes = [r.moves[horizon].mfe_pct for r in rows if r.moves.get(horizon) and r.moves[horizon].mfe_pct is not None]
    maes = [r.moves[horizon].mae_pct for r in rows if r.moves.get(horizon) and r.moves[horizon].mae_pct is not None]
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "mfe": 0.0, "mae": 0.0, "p2": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "wr": float((arr > 0).mean() * 100.0),
        "avg": float(arr.mean()),
        "med": float(np.median(arr)),
        "mfe": float(np.mean(mfes)) if mfes else 0.0,
        "mae": float(np.mean(maes)) if maes else 0.0,
        "p2": float((arr >= 2.0).mean() * 100.0),
    }


def fail_rate(rows: list[SignalRow]) -> float:
    flags = [r.moves[1].failed for r in rows if r.moves.get(1) and r.moves[1].failed is not None]
    if not flags:
        return 0.0
    return float(sum(1 for x in flags if x) / len(flags) * 100.0)


def apply_filter(rows: list[SignalRow], *, body: bool | None = None, max_width: float | None = None, min_vol: float | None = None) -> list[SignalRow]:
    out = rows
    if body:
        out = [r for r in out if r.body_through]
    if max_width is not None:
        out = [r for r in out if r.width_pct <= max_width]
    if min_vol is not None:
        out = [r for r in out if r.vol_ratio >= min_vol]
    return out
