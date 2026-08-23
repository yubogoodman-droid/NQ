"""15 分 K：收盤高於 MA7 / MA25 / MA200，且 7>25。圖上仍畫 MA14，不當過濾。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H1_MS = 3_600_000
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
    """收盤同時在 MA7 / MA25 / MA200 之上，且 7>25。MA14 不擋單。"""
    px, ma7, ma25, ma200 = c[i], m7[i], m25[i], m200[i]
    if np.isnan([px, ma7, ma25, ma200]).any():
        return False
    return bool(px > ma7 > ma25 and px > ma200)


def bars_below_ma200(c, m200, i: int) -> int:
    """站上前連續幾根收盤仍在 MA200 下（不含本根）。"""
    n = 0
    j = i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] <= m200[j]:
        n += 1
        j -= 1
    return n


def bars_above_ma200(c, m200, i: int) -> int:
    """本根之前連續幾根收盤已在 MA200 上（不含本根）。剛站上那根為 0。"""
    n = 0
    j = i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] > m200[j]:
        n += 1
        j -= 1
    return n


def rng24_pct(h, l, m200, i: int) -> float:
    """站上前 24 根的高低差，相對當時 MA200。"""
    a0 = max(0, i - 24)
    if i <= a0 or np.isnan(m200[i]) or not m200[i]:
        return 0.0
    return float((h[a0:i].max() - l[a0:i].min()) / m200[i] * 100.0)


@dataclass(frozen=True)
class BullSignal:
    """15m 收盤在 MA7/25/200 之上，且 7>25。"""

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
    bars_below: int = 0
    rng24: float = 0.0
    bars_above: int = 0

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
    進場：這一根收盤 > MA7 > MA25，且收盤 > 15 分 MA200，
    前一根還沒同時成立。MA14 只畫圖、不擋單。

    crossed_200：前收還在 MA200 下，本根收盤站上。
    formed_align：已經在 MA200 上，本根才收上短均／排成 7>25。
    15m 通知：crossed_200（本根剛站上 MA200 且 7>25）。
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
                bars_below=bars_below_ma200(c, m200, i),
                rng24=rng24_pct(h, l, m200, i),
                bars_above=bars_above_ma200(c, m200, i),
            )
        )
        last_i = i
    return out


def htf_sma_at(d_htf: dict, time_ms: int, last_price: float, bar_ms: int, n: int) -> float | None:
    """訊號當下的大週期 SMAn。未收完的那根 K 用 last_price，不偷用後面的收盤。"""
    t = d_htf["t"]
    opened = t <= time_ms
    if not opened.any():
        return None
    last_i = int(np.where(opened)[0][-1])
    if last_i + 1 < n:
        return None
    c = np.array(d_htf["c"][: last_i + 1], dtype=float)
    if int(t[last_i]) + bar_ms > time_ms:
        c[-1] = float(last_price)
    window = c[-n:]
    if np.isnan(window).any():
        return None
    return float(window.mean())


def htf_ma200_at(d_htf: dict, time_ms: int, last_price: float, bar_ms: int) -> float | None:
    return htf_sma_at(d_htf, time_ms, last_price, bar_ms, 200)


def htf_ma25_now_prev(
    d_htf: dict | None, time_ms: int, last_price: float, bar_ms: int
) -> tuple[float | None, float | None]:
    """當下 SMA25（未收完用 last_price）與前一根已收完 SMA25。"""
    if d_htf is None or len(d_htf.get("c", [])) < 26:
        return None, None
    now = htf_sma_at(d_htf, time_ms, last_price, bar_ms, 25)
    t = d_htf["t"]
    opened = t <= time_ms
    if not opened.any():
        return now, None
    last_i = int(np.where(opened)[0][-1])
    if last_i < 25:
        return now, None
    prev_win = np.array(d_htf["c"][last_i - 25 : last_i], dtype=float)
    if len(prev_win) < 25 or np.isnan(prev_win).any():
        return now, None
    return now, float(prev_win.mean())


def htf_ma25_not_down(d_htf: dict | None, time_ms: int, last_price: float, bar_ms: int) -> bool:
    """1h MA25 未下彎：當下 ≥ 前一根已收完。"""
    now, prev = htf_ma25_now_prev(d_htf, time_ms, last_price, bar_ms)
    return now is not None and prev is not None and now >= prev


def h1_ma200_at(d1h: dict, time_ms: int, last_price: float) -> float | None:
    return htf_ma200_at(d1h, time_ms, last_price, H1_MS)


def above_htf_ma200(d_htf: dict | None, time_ms: int, last_price: float, bar_ms: int) -> bool:
    if d_htf is None:
        return False
    ma = htf_ma200_at(d_htf, time_ms, last_price, bar_ms)
    return ma is not None and float(last_price) > ma


def above_1h_ma200(d1h: dict | None, time_ms: int, last_price: float) -> bool:
    return above_htf_ma200(d1h, time_ms, last_price, H1_MS)


def bar_above_ma200(d: dict | None, time_ms: int, bar_ms: int) -> bool:
    """該週期自己的收盤是否在自己的 SMA200 上（未收完用當根已走出的收盤）。"""
    if d is None or len(d.get("c", [])) < 200:
        return False
    opened = d["t"] <= time_ms
    if not opened.any():
        return False
    last_i = int(np.where(opened)[0][-1])
    return above_htf_ma200(d, time_ms, float(d["c"][last_i]), bar_ms)


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
    h1_ma200: float | None = None
    h1_ext_pct: float | None = None
    btc_1h_ok: bool | None = None
    h1_ma25: float | None = None
    h1_ma25_prev: float | None = None
    h1_ma25_up: bool | None = None

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

    @property
    def bars_below(self) -> int:
        return self.sig.bars_below

    @property
    def rng24(self) -> float:
        return self.sig.rng24

    @property
    def bars_above(self) -> int:
        return self.sig.bars_above


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
    min_below: int | None = None,
    max_rng24: float | None = None,
    max_bars_above: int | None = None,
    require_btc_1h: bool | None = None,
    require_h1_ma25_up: bool | None = None,
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
    if min_below is not None:
        out = [r for r in out if r.bars_below >= min_below]
    if max_rng24 is not None:
        out = [r for r in out if r.rng24 <= max_rng24]
    if max_bars_above is not None:
        out = [
            r
            for r in out
            if r.crossed_200d or (r.formed_align and r.bars_above <= max_bars_above)
        ]
    if require_btc_1h:
        out = [r for r in out if r.btc_1h_ok]
    if require_h1_ma25_up:
        out = [r for r in out if r.h1_ma25_up]
    return out


def quality_reclaim(
    sig: BullSignal,
    *,
    min_below: int | None = None,
    min_vol: float | None = None,
    max_ext: float | None = None,
    max_rng24: float | None = None,
    max_bars_above: int | None = None,
) -> bool:
    """剛站上 MA200 且 7>25；可再限底下根數、量比、偏離、波動。"""
    if sig.crossed_200:
        pass
    elif max_bars_above is not None and sig.formed_align and sig.bars_above <= max_bars_above:
        pass
    else:
        return False
    if min_below is not None and sig.bars_below < min_below:
        return False
    if min_vol is not None and sig.vol_ratio < min_vol:
        return False
    if max_ext is not None and sig.ext_pct > max_ext:
        return False
    if max_rng24 is not None and sig.rng24 > max_rng24:
        return False
    return True
