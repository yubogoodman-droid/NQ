"""幣安 1 分 K：7>14>25>99>120 多頭排列，剛站上 1m MA200。

截圖紅圈那種：短均先黏成帶，收盤穿過 MA200，99/120 還在帶的下沿。
均線都用一分 K 收盤 SMA。不要求 120 已高於 200。
    預設再看 5 分 K（不偷看未走完的分鐘）：MA7>MA14、MA7 向上、收盤站上 MA7。
    不要求 5 分已經站上 MA200，也不強求 5 分 7>14>25（那組會濾掉不少贏家）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONS = (5, 15, 30, 60, 240)  # 1m 根數 → 5m / 15m / 30m / 1h / 4h
FIVE_MIN_MS = 5 * 60_000


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_mas(d: dict) -> dict:
    """收盤 SMA 7/14/25/99/120/200（週期隨 K 線，1m 就是 1m 均、5m 就是 5m 均）。"""
    out = dict(d)
    c, v = d["c"], d["v"]
    out["m7"] = sma(c, 7)
    out["m14"] = sma(c, 14)
    out["m25"] = sma(c, 25)
    out["m99"] = sma(c, 99)
    out["m120"] = sma(c, 120)
    out["m200"] = sma(c, 200)
    out["v20"] = sma(v, 20)
    return out


def resample_ohlcv(d: dict, interval_ms: int = FIVE_MIN_MS) -> dict:
    """把已排序的較細 K 線合成較大週期（預設 5 分）。"""
    t = np.asarray(d["t"], dtype=np.int64)
    if t.size == 0:
        return {k: np.array([]) for k in ("t", "o", "h", "l", "c", "v")}
    bucket = t - (t % interval_ms)
    change = np.ones(t.size, dtype=bool)
    change[1:] = bucket[1:] != bucket[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], t.size]
    return {
        "t": bucket[starts],
        "o": np.asarray(d["o"], dtype=float)[starts],
        "h": np.maximum.reduceat(np.asarray(d["h"], dtype=float), starts),
        "l": np.minimum.reduceat(np.asarray(d["l"], dtype=float), starts),
        "c": np.asarray(d["c"], dtype=float)[ends - 1],
        "v": np.add.reduceat(np.asarray(d["v"], dtype=float), starts),
    }


def resample_ohlcv_upto(d: dict, end_idx: int, interval_ms: int = FIVE_MIN_MS) -> dict:
    """合成到 1m end_idx 為止看得到的較大週期（最後一根可以還沒走完，不偷看後面分鐘）。"""
    n = len(d.get("c", []))
    if n == 0:
        return resample_ohlcv(d, interval_ms)
    hi = max(0, min(int(end_idx) + 1, n))
    sliced = {k: np.asarray(d[k])[:hi] for k in ("t", "o", "h", "l", "c", "v") if k in d}
    return resample_ohlcv(sliced, interval_ms)


def five_m_last_ok(
    closes,
    *,
    slope_bars: int = 3,
    require_short_stack: bool = True,
    require_close_above_ma7: bool = True,
) -> bool:
    """5 分 K 確認：MA7>MA14、MA7 向上、收盤站上 MA7。不要求站上 MA200，也不強求 25。"""
    c = np.asarray(closes, dtype=float)
    j = int(c.size) - 1
    if j < 13:
        return False
    last = float(c[j])
    m7 = float(c[j - 6 : j + 1].mean())
    m14 = float(c[j - 13 : j + 1].mean())
    if require_short_stack and not (m7 > m14):
        return False
    if require_close_above_ma7 and not (last > m7):
        return False
    prev_j = j - max(1, int(slope_bars))
    if prev_j < 6:
        return False
    m7_prev = float(c[prev_j - 6 : prev_j + 1].mean())
    return bool(m7 > m7_prev)


def five_m_ok(
    d: dict,
    i: int,
    *,
    slope_bars: int = 3,
    require_short_stack: bool = True,
    require_close_above_ma7: bool = True,
) -> bool:
    """用「當下這根 1m 為止」的 5 分 K（含未走完那根），不偷看後面的分鐘。"""
    d5 = resample_ohlcv_upto(d, i)
    if len(d5.get("c", [])) == 0:
        return False
    return five_m_last_ok(
        d5["c"],
        slope_bars=slope_bars,
        require_short_stack=require_short_stack,
        require_close_above_ma7=require_close_above_ma7,
    )


def five_m_ok_mask(
    d: dict,
    *,
    end: int | None = None,
    slope_bars: int = 3,
    require_short_stack: bool = True,
    require_close_above_ma7: bool = True,
) -> np.ndarray:
    """每個 1m 指數對應的 5 分確認（形成中的 5m 只用到該分鐘）。"""
    c = np.asarray(d["c"], dtype=float)
    n = len(c)
    t = np.asarray(d["t"], dtype=np.int64) if "t" in d else np.arange(n, dtype=np.int64) * 60_000
    last = n if end is None else max(0, min(int(end) + 1, n))
    out = np.zeros(n, dtype=bool)
    closes: list[float] = []
    cur_b: int | None = None
    kw = dict(
        slope_bars=slope_bars,
        require_short_stack=require_short_stack,
        require_close_above_ma7=require_close_above_ma7,
    )
    for i in range(last):
        b = int(t[i] - (t[i] % FIVE_MIN_MS))
        if cur_b is None or b != cur_b:
            closes.append(float(c[i]))
            cur_b = b
        else:
            closes[-1] = float(c[i])
        out[i] = five_m_last_ok(closes, **kw)
    return out


def bar_index_at(times: np.ndarray, ts: int) -> int:
    """含 ts 的那根（開盤時間 ≤ ts 的最後一根）。"""
    if len(times) == 0:
        return 0
    i = int(np.searchsorted(times, ts, side="right") - 1)
    return max(i, 0)


def stack_ok(d: dict, i: int) -> bool:
    """截圖紅圈：收盤 > MA200 > 7 > 14 > 25 > 99 > 120。"""
    c, m7, m14, m25 = d["c"], d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    vals = [c[i], m7[i], m14[i], m25[i], m99[i], m120[i], m200[i]]
    if np.isnan(vals).any():
        return False
    return bool(c[i] > m200[i] > m7[i] > m14[i] > m25[i] > m99[i] > m120[i])


def ma_widths(d: dict, i: int) -> tuple[float, float, float]:
    """六條全距%、短均 7/14/25 全距%、短均+MA200 包距%。"""
    ms = [d["m7"][i], d["m14"][i], d["m25"][i], d["m99"][i], d["m120"][i], d["m200"][i]]
    pack = [d["m7"][i], d["m14"][i], d["m25"][i], d["m200"][i]]
    if np.isnan(ms).any() or min(ms) <= 0 or min(pack) <= 0:
        return float("nan"), float("nan"), float("nan")
    ribbon = (max(ms) / min(ms) - 1.0) * 100.0
    short = (max(ms[:3]) / min(ms[:3]) - 1.0) * 100.0
    pack_pct = (max(pack) / min(pack) - 1.0) * 100.0
    return float(ribbon), float(short), float(pack_pct)


def ribbon_ok(
    d: dict,
    i: int,
    *,
    max_ribbon_pct: float | None = 0.65,
    max_short_pct: float | None = 0.50,
) -> bool:
    """站上當下短均還沒扇開，仍跟 MA200 黏在一起。"""
    _ribbon, short, pack = ma_widths(d, i)
    if np.isnan(pack):
        return False
    if max_ribbon_pct is not None and pack > max_ribbon_pct:
        return False
    if max_short_pct is not None and short > max_short_pct:
        return False
    return True


def coil_ok(d: dict, i: int, *, lookback: int = 20, max_prior_short: float | None = 0.15) -> bool:
    """站上前短均先黏成帶（紅圈左邊那段）。"""
    if max_prior_short is None or lookback <= 0:
        return True
    prior: list[float] = []
    for j in range(max(1, i - lookback), i):
        _ribbon, short, _pack = ma_widths(d, j)
        if not np.isnan(short):
            prior.append(short)
    return bool(prior) and min(prior) <= max_prior_short


def vol_ok(d: dict, i: int, *, min_vol_ratio: float = 1.4) -> bool:
    """紅圈那根放量。"""
    if min_vol_ratio <= 0:
        return True
    v20 = d["v20"][i]
    if not v20 or np.isnan(v20) or v20 <= 0:
        return False
    return bool(d["v"][i] / v20 >= min_vol_ratio)


def bars_below_ma200(c, m200, i: int) -> int:
    n = 0
    j = i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] <= m200[j]:
        n += 1
        j -= 1
    return n


def bars_above_ma200(c, m200, i: int) -> int:
    """含本根，連續幾根收盤站在 MA200 上。"""
    n = 0
    j = i
    while j >= 0 and not np.isnan(m200[j]) and c[j] > m200[j]:
        n += 1
        j -= 1
    return n


def streak_start(c, m200, i: int) -> int:
    n = bars_above_ma200(c, m200, i)
    return i - n + 1 if n else i


def vol_ok_streak(d: dict, i: int, *, min_vol_ratio: float) -> bool:
    """站穩這段裡有一根放量即可（常在第一根上站，不在確認根）。"""
    if min_vol_ratio <= 0:
        return True
    start = streak_start(d["c"], d["m200"], i)
    return any(vol_ok(d, j, min_vol_ratio=min_vol_ratio) for j in range(start, i + 1))


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
    m120: float
    ma200: float
    vol_ratio: float
    ext_pct: float
    ribbon_pct: float
    short_pct: float
    crossed_200: bool
    bars_below: int
    bars_above: int


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
    kind: str = "stock"
    moves: dict[int, ForwardMove] = field(default_factory=dict)

    @property
    def kind_label(self) -> str:
        return "加密" if self.kind == "crypto" else "股票"

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

    @property
    def bars_above(self) -> int:
        return self.sig.bars_above


def signal_at(d: dict, i: int, *, require_stack: bool = True) -> BullSignal | None:
    if i < 1 or i >= len(d["c"]):
        return None
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    v20 = d["v20"]
    if require_stack and not stack_ok(d, i):
        return None
    vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
    ext = (c[i] / m200[i] - 1.0) * 100.0 if m200[i] else 0.0
    _ribbon, short, pack = ma_widths(d, i)
    above = bars_above_ma200(c, m200, i)
    start = i - above + 1 if above else i
    crossed = bool(start >= 1 and c[start - 1] <= m200[start - 1] and c[start] > m200[start])
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
        m120=float(m120[i]),
        ma200=float(m200[i]),
        vol_ratio=vr,
        ext_pct=float(ext),
        ribbon_pct=float(pack),
        short_pct=float(short),
        crossed_200=crossed,
        bars_below=bars_below_ma200(c, m200, start),
        bars_above=above,
    )


def _bar_ok(
    d: dict,
    i: int,
    *,
    max_ribbon_pct: float | None,
    max_short_pct: float | None,
    max_prior_short: float | None,
    min_vol_ratio: float,
    min_below: int,
    min_above: int,
) -> bool:
    if not stack_ok(d, i):
        return False
    if not ribbon_ok(d, i, max_ribbon_pct=max_ribbon_pct, max_short_pct=max_short_pct):
        return False
    above = bars_above_ma200(d["c"], d["m200"], i)
    if above < max(1, int(min_above)):
        return False
    start = i - above + 1
    if not coil_ok(d, start, max_prior_short=max_prior_short):
        return False
    if not vol_ok_streak(d, i, min_vol_ratio=min_vol_ratio):
        return False
    if min_below > 0 and bars_below_ma200(d["c"], d["m200"], start) < min_below:
        return False
    return True


def detect_combo(
    d: dict,
    *,
    min_gap_bars: int = 0,
    cross_only: bool = False,
    max_ribbon_pct: float | None = 0.65,
    max_short_pct: float | None = 0.50,
    max_prior_short: float | None = 0.15,
    min_vol_ratio: float = 1.4,
    min_below: int = 20,
    min_above: int = 2,
    use_5m: bool = True,
    five_m_slope: int = 3,
) -> list[BullSignal]:
    """短均先黏帶，長期在 MA200 下，放量上站後再連收至少 min_above 根站穩。

    排列：收盤 > MA200 > 7 > 14 > 25 > 99 > 120。
    預設再加 5 分 K 確認（不偷看未走完的分鐘）：MA7>MA14、MA7 向上、收盤站上 5m MA7。
    不要求 5 分收盤已站上 5m MA200（截圖 SNDK 當時還在下面）。
    5 分只用來過濾「1 分剛成立」的那根，不會等到後面才追進。
    """
    c, m200 = d["c"], d["m200"]
    kw = dict(
        max_ribbon_pct=max_ribbon_pct,
        max_short_pct=max_short_pct,
        max_prior_short=max_prior_short,
        min_vol_ratio=min_vol_ratio,
        min_below=min_below,
        min_above=min_above,
    )
    out: list[BullSignal] = []
    last_i = -10_000
    for i in range(1, len(c)):
        now_ok = _bar_ok(d, i, **kw)
        prev_ok = _bar_ok(d, i - 1, **kw)
        if not now_ok or prev_ok:
            continue
        if cross_only:
            n = bars_above_ma200(c, m200, i)
            start = i - n + 1
            if start < 1 or np.isnan(m200[start - 1]):
                continue
            if not (c[start - 1] <= m200[start - 1] and c[start] > m200[start]):
                continue
        if i - last_i < min_gap_bars:
            continue
        if use_5m and not five_m_ok(d, i, slope_bars=five_m_slope):
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
