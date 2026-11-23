"""15 分 K 同時穿越均線。訊號預設 MA7 / 14 / 25 / 99 / 120（不含 200）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MA_PERIODS = (7, 14, 25, 99, 120, 200)
SIGNAL_MA_PERIODS = (7, 14, 25, 99, 120)
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


def _mas_at(d: dict, i: int, periods: tuple[int, ...] = MA_PERIODS) -> np.ndarray | None:
    vals = np.array([d[f"m{n}"][i] for n in periods], dtype=float)
    if np.isnan(vals).any():
        return None
    return vals


@dataclass(frozen=True)
class RibbonBreak:
    """一根 K 穿過指定均線帶（做多由下往上、做空由上往下）。"""

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
    side: str = "long"
    ma200_aligned: bool = False
    ret24h_pct: float | None = None
    wick_ratio: float = 0.0
    bars_since_any_break: int | None = None


def _break_context(d: dict, i: int, *, side: str) -> tuple[bool, float | None, float]:
    c, o, h, l = d["c"], d["o"], d["h"], d["l"]
    m200 = d.get("m200")
    ma200_ok = False
    if m200 is not None and not np.isnan(m200[i]):
        ma200_ok = bool(c[i] > m200[i]) if side == "long" else bool(c[i] < m200[i])
    ret24h = float((c[i - 1] / c[i - 97] - 1.0) * 100.0) if i >= 97 and c[i - 97] > 0 else None
    span = float(h[i] - l[i])
    if span <= 0:
        wick = 0.0
    elif side == "long":
        wick = float((h[i] - max(o[i], c[i])) / span)
    else:
        wick = float((min(o[i], c[i]) - l[i]) / span)
    return ma200_ok, ret24h, wick


def detect_all_breaks(d: dict, periods: tuple[int, ...] = SIGNAL_MA_PERIODS) -> list[RibbonBreak]:
    """
    同時偵測多空穿越，並記錄距離同幣前一次任一方向穿越的 K 棒數（避免盤整雙巴）。

    預設五條：MA7 / 14 / 25 / 99 / 120（不必過 MA200）。
    """
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    v20 = d["v20"]
    out: list[RibbonBreak] = []
    start = max(periods) + 1
    last_any_idx: int | None = None
    for i in range(start, len(c)):
        prev = _mas_at(d, i - 1, periods)
        curr = _mas_at(d, i, periods)
        if prev is None or curr is None:
            continue
        lo_p, hi_p = float(prev.min()), float(prev.max())
        lo_c, hi_c = float(curr.min()), float(curr.max())
        side: str | None = None
        if c[i - 1] < lo_p and c[i] > hi_c and l[i] <= lo_p:
            side = "long"
        elif c[i - 1] > hi_p and c[i] < lo_c and h[i] >= hi_p:
            side = "short"
        if side is None:
            continue
        width = (hi_p / lo_p - 1.0) * 100.0 if lo_p > 0 else float("inf")
        vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        body = bool(o[i] < lo_p and c[i] > hi_c) if side == "long" else bool(o[i] > hi_p and c[i] < lo_c)
        rng = (h[i] / l[i] - 1.0) * 100.0 if l[i] > 0 else 0.0
        ma200_ok, ret24h, wick = _break_context(d, i, side=side)
        bars_since = (i - last_any_idx) if last_any_idx is not None else None
        last_any_idx = i
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
                side=side,
                ma200_aligned=ma200_ok,
                ret24h_pct=ret24h,
                wick_ratio=wick,
                bars_since_any_break=bars_since,
            )
        )
    return out


def detect_long_breaks(d: dict, periods: tuple[int, ...] = SIGNAL_MA_PERIODS) -> list[RibbonBreak]:
    """
    做多：前一根收盤完全在指定均線下方，這一根收盤完全站上這些均線。

    預設五條：MA7 / 14 / 25 / 99 / 120（不必過 MA200）。
    """
    return [br for br in detect_all_breaks(d, periods=periods) if br.side == "long"]


def detect_short_breaks(d: dict, periods: tuple[int, ...] = SIGNAL_MA_PERIODS) -> list[RibbonBreak]:
    """
    做空：前一根收盤完全在指定均線上方，這一根收盤完全跌破這些均線。

    預設五條：MA7 / 14 / 25 / 99 / 120（不必破 MA200）。
    """
    return [br for br in detect_all_breaks(d, periods=periods) if br.side == "short"]


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

    @property
    def ma200_aligned(self) -> bool:
        return self.break_.ma200_aligned

    @property
    def ret24h_pct(self) -> float | None:
        return self.break_.ret24h_pct

    @property
    def wick_ratio(self) -> float:
        return self.break_.wick_ratio


def forward_moves(d: dict, br: RibbonBreak, horizons: tuple[int, ...] = HORIZONS) -> tuple[float, dict[int, ForwardMove]]:
    """進場用訊號下一根開盤；假突破 = 下一根收盤又退回帶子內（多看 ma_max、空看 ma_min）。"""
    i = br.idx
    nxt = i + 1
    if nxt >= len(d["c"]):
        return float("nan"), {h: ForwardMove(h, None, None, None, None) for h in horizons}
    entry = float(d["o"][nxt])
    is_short = br.side == "short"
    moves: dict[int, ForwardMove] = {}
    for h in horizons:
        j = nxt + h - 1
        if j >= len(d["c"]):
            moves[h] = ForwardMove(h, None, None, None, None)
            continue
        last = float(d["c"][j])
        hi = float(d["h"][nxt : j + 1].max())
        lo = float(d["l"][nxt : j + 1].min())
        if is_short:
            ret = (1.0 - last / entry) * 100.0
            mfe = (1.0 - lo / entry) * 100.0
            mae = (1.0 - hi / entry) * 100.0
            failed = float(d["c"][nxt]) > br.ma_min if h == 1 else None
        else:
            ret = (last / entry - 1.0) * 100.0
            mfe = (hi / entry - 1.0) * 100.0
            mae = (lo / entry - 1.0) * 100.0
            failed = float(d["c"][nxt]) < br.ma_max if h == 1 else None
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


def apply_filter(
    rows: list[SignalRow],
    *,
    body: bool | None = None,
    max_width: float | None = None,
    min_width: float | None = None,
    min_vol: float | None = None,
    ma200_aligned: bool | None = None,
    trend_24h: bool | None = None,
    max_wick: float | None = None,
    cooldown_hours: float | None = None,
) -> list[SignalRow]:
    out = rows
    if cooldown_hours is not None and cooldown_hours > 0:
        cd_bars = int(round(cooldown_hours * 4))
        cd_ms = int(cooldown_hours * 3600 * 1000)
        # 優先用偵測期紀錄的「距任一方向前次穿越根數」去重，同時備援 timestamp 去重
        last_by_sym: dict[str, int] = {}
        kept: set[tuple[str, str, int]] = set()
        for r in sorted(rows, key=lambda x: x.time_ms):
            prev = last_by_sym.get(r.symbol)
            ts_ok = prev is None or (r.time_ms - prev) >= cd_ms
            bar_gap = r.break_.bars_since_any_break
            bars_ok = bar_gap is None or bar_gap >= cd_bars
            if ts_ok and bars_ok:
                kept.add((r.symbol, r.break_.side, r.time_ms))
            last_by_sym[r.symbol] = r.time_ms
        out = [r for r in out if (r.symbol, r.break_.side, r.time_ms) in kept]
    if body:
        out = [r for r in out if r.body_through]
    if max_width is not None:
        out = [r for r in out if r.width_pct <= max_width]
    if min_width is not None:
        out = [r for r in out if r.width_pct > min_width]
    if min_vol is not None:
        out = [r for r in out if r.vol_ratio >= min_vol]
    if ma200_aligned:
        out = [r for r in out if r.ma200_aligned]
    if trend_24h:
        out = [
            r
            for r in out
            if r.ret24h_pct is not None
            and (r.ret24h_pct > 0 if r.break_.side == "long" else r.ret24h_pct < 0)
        ]
    if max_wick is not None:
        out = [r for r in out if r.wick_ratio <= max_wick]
    return out
