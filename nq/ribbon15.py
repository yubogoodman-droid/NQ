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


H1_MS = 60 * 60 * 1000
M15_MS = 15 * 60 * 1000


def last_closed_h1_index(t: np.ndarray, ts_ms: int) -> int | None:
    """15 分收盤當下，最後一根已經收完的 1 小時 K（不含還在走的那根）。"""
    close_ms = int(ts_ms) + M15_MS
    cutoff = close_ms - H1_MS
    i = int(np.searchsorted(t, cutoff, side="right") - 1)
    if i < 0 or i >= len(t):
        return None
    return i


def forming_h1_index(t: np.ndarray, ts_ms: int) -> int | None:
    i = int(np.searchsorted(t, ts_ms, side="right") - 1)
    if i < 0 or i >= len(t):
        return None
    return i


@dataclass(frozen=True)
class HourlyContext:
    """15 分訊號當下看得到的 1 小時狀態（不偷看未收盤小時 K 的收盤價）。"""

    idx: int
    close: float
    ma_min: float
    ma_max: float
    ma7: float
    ma99: float
    ma200: float
    width_pct: float
    above_ribbon: bool
    below_ribbon: bool
    in_ribbon: bool
    above_ma200: bool
    above_ma99: bool
    ma7_gt_ma200: bool
    last_green: bool
    ma7_up: bool
    forming_green: bool
    px_above_ma200: bool
    px_above_ribbon: bool
    dist_ma200_pct: float


def hourly_context(d1: dict, ts_ms: int, px: float) -> HourlyContext | None:
    t = d1["t"]
    i = last_closed_h1_index(t, ts_ms)
    if i is None:
        return None
    mas = _mas_at(d1, i)
    if mas is None:
        return None
    lo, hi = float(mas.min()), float(mas.max())
    c = float(d1["c"][i])
    o = float(d1["o"][i])
    ma7 = float(d1["m7"][i])
    ma99 = float(d1["m99"][i])
    ma200 = float(d1["m200"][i])
    if lo <= 0 or ma200 <= 0 or np.isnan(ma7) or np.isnan(ma200):
        return None
    ma7_prev = float(d1["m7"][i - 1]) if i > 0 else float("nan")
    fi = forming_h1_index(t, ts_ms)
    hour_open = float(d1["o"][fi]) if fi is not None else float("nan")
    return HourlyContext(
        idx=i,
        close=c,
        ma_min=lo,
        ma_max=hi,
        ma7=ma7,
        ma99=ma99,
        ma200=ma200,
        width_pct=(hi / lo - 1.0) * 100.0,
        above_ribbon=c > hi,
        below_ribbon=c < lo,
        in_ribbon=lo <= c <= hi,
        above_ma200=c > ma200,
        above_ma99=c > ma99,
        ma7_gt_ma200=ma7 > ma200,
        last_green=c >= o,
        ma7_up=bool(not np.isnan(ma7_prev) and ma7 > ma7_prev),
        forming_green=bool(not np.isnan(hour_open) and px > hour_open),
        px_above_ma200=px > ma200,
        px_above_ribbon=px > hi,
        dist_ma200_pct=(px / ma200 - 1.0) * 100.0,
    )


def row_h1_match(row: "SignalRow", key: str) -> bool:
    h = row.hourly
    if h is None:
        return False
    if key == "below_ribbon":
        return h.below_ribbon
    if key == "in_ribbon":
        return h.in_ribbon
    if key == "above_ribbon":
        return h.above_ribbon
    if key == "above_ma200":
        return h.above_ma200
    if key == "above_ma99":
        return h.above_ma99
    if key == "ma7_gt_ma200":
        return h.ma7_gt_ma200
    if key == "last_green":
        return h.last_green
    if key == "ma7_up":
        return h.ma7_up
    if key == "forming_green":
        return h.forming_green
    if key == "px_above_ma200":
        return h.px_above_ma200
    if key == "px_above_ribbon":
        return h.px_above_ribbon
    if key == "px_below_ma200":
        return not h.px_above_ma200
    if key == "near_ma200":
        return h.px_above_ma200 and 0.0 <= h.dist_ma200_pct <= 2.0
    if key == "tight":
        return h.width_pct <= 1.5
    if key == "not_below":
        return not h.below_ribbon
    if key == "form_green_px_ma200":
        return h.forming_green and h.px_above_ma200
    if key == "h1_ma200_form_green":
        return h.above_ma200 and h.forming_green
    if key == "h1_ma200_body":
        return h.above_ma200 and row.body_through
    if key == "px_ma200_tight":
        return h.px_above_ma200 and h.width_pct <= 2.0
    if key == "h1_ma200_near":
        return h.above_ma200 and h.px_above_ma200 and 0.0 <= h.dist_ma200_pct <= 2.0
    if key == "trend_continue":
        return h.above_ma200 and h.forming_green and not h.below_ribbon
    if key == "best_bundle":
        return (
            h.forming_green
            and h.px_above_ma200
            and not h.below_ribbon
            and row.body_through
        )
    return False


# 事先定好的 1h 濾網（不是掃完再挑好看的）。
H1_FILTERS = (
    ("（對照）1h 仍在整條帶下", "below_ribbon"),
    ("（對照）15m 收還在 1h MA200 下", "px_below_ma200"),
    ("1h 在帶內", "in_ribbon"),
    ("1h 已站上整條帶", "above_ribbon"),
    ("1h 收在 MA200 上", "above_ma200"),
    ("1h 收在 MA99 上", "above_ma99"),
    ("1h MA7 > MA200", "ma7_gt_ma200"),
    ("上一根完整 1h 是陽線", "last_green"),
    ("1h MA7 向上", "ma7_up"),
    ("當根小時已翻綠", "forming_green"),
    ("15m 收盤站上 1h MA200", "px_above_ma200"),
    ("15m 收盤站上 1h 帶頂", "px_above_ribbon"),
    ("離 1h MA200 0～2%", "near_ma200"),
    ("1h 帶寬 ≤ 1.5%", "tight"),
    ("排除：1h 還在帶下", "not_below"),
    ("當根小時綠 + 價在 1h MA200 上", "form_green_px_ma200"),
    ("1h 收 MA200 上 + 小時已綠", "h1_ma200_form_green"),
    ("1h 收 MA200 上 + 15m 實體穿越", "h1_ma200_body"),
    ("價在 1h MA200 上 + 1h 帶寬≤2%", "px_ma200_tight"),
    ("1h 收 MA200 上且未延伸", "h1_ma200_near"),
    ("順勢：1h 在 MA200 上且小時已綠", "trend_continue"),
    ("組合：小時綠 + 價上 MA200 + 非帶下 + 實體", "best_bundle"),
)

# 7 天與 30 天都把 4h 勝率拉開的濾網（樣本較少的寫在前）。
H1_RECOMMENDED = (
    ("1h 已站上整條帶", "above_ribbon"),
    ("價在 1h MA200 上 + 1h 帶寬≤2%", "px_ma200_tight"),
)


def apply_h1_filter(rows: list["SignalRow"], key: str) -> list["SignalRow"]:
    return [r for r in rows if row_h1_match(r, key)]


@dataclass
class SignalRow:
    symbol: str
    break_: RibbonBreak
    time_ms: int
    entry: float
    moves: dict[int, ForwardMove] = field(default_factory=dict)
    hourly: HourlyContext | None = None

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


def apply_filter(
    rows: list[SignalRow],
    *,
    body: bool | None = None,
    max_width: float | None = None,
    min_width: float | None = None,
    min_vol: float | None = None,
) -> list[SignalRow]:
    out = rows
    if body:
        out = [r for r in out if r.body_through]
    if max_width is not None:
        out = [r for r in out if r.width_pct <= max_width]
    if min_width is not None:
        out = [r for r in out if r.width_pct > min_width]
    if min_vol is not None:
        out = [r for r in out if r.vol_ratio >= min_vol]
    return out
