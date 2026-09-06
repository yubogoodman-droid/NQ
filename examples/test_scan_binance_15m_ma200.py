#!/usr/bin/env python3
"""15m 黏帶擠壓 · 200MA 附近進場：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma200_squeeze import (  # noqa: E402
    HOLD_BARS,
    LOOKBACK,
    MAX_ENTRY_EXT,
    add_indicators,
    detect_signals,
    signal_at,
    simulate_trades,
    sma,
    summarize_trades,
)

TZ_OFFSET = 8 * 3600


def _bars(close: np.ndarray, vol: np.ndarray | None = None, *, noise: float = 0.0012) -> dict:
    close = np.asarray(close, float)
    n = len(close)
    rng = np.maximum(np.abs(np.diff(close, prepend=close[0])) * 0.35, close * noise)
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + rng * 0.45
    l = np.minimum(o, close) - rng * 0.45
    v = vol if vol is not None else np.full(n, 1000.0)
    t0 = int(datetime(2026, 8, 20, 8, 0).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": v}


def _coil_then_break(
    *,
    break_close: float = 100.40,
    vol_signal: float = 3500.0,
    coil_amp: float = 0.25,
    already_gone: bool = False,
    extra_after: int = 8,
) -> dict:
    """長時間在 100 附近窄幅震盪，讓六條均線黏在一起，再放量打出箱頂。"""
    n_flat = 260
    base = 100.0
    coil = base + coil_amp * np.sin(np.linspace(0, 10 * np.pi, n_flat))
    # 最後幾根略壓在 200 下方，做出「在 200 附近」而不是已經起飛
    coil[-6:] = np.array([99.85, 99.70, 99.90, 99.80, 99.95, 99.88])
    if already_gone:
        pre = np.array([101.40])
        close = np.concatenate([coil, pre, np.array([break_close]), np.full(extra_after, break_close + 0.2)])
        vol = np.full(len(close), 1000.0)
        vol[n_flat] = vol_signal
        vol[n_flat + 1] = vol_signal
    else:
        close = np.concatenate([coil, np.array([break_close]), np.full(extra_after, break_close + 0.15)])
        vol = np.full(len(close), 1000.0)
        vol[n_flat] = vol_signal
    d = add_indicators(_bars(close, vol, noise=0.0024))
    # 突破根對齊 ETH 9/3：量能溫和、振幅約 2～3 倍，不是跳空
    i = n_flat if not already_gone else n_flat + 1
    d["o"][i] = 99.92
    d["l"][i] = 99.86
    d["h"][i] = max(float(d["c"][i]), 100.40) + 0.04
    if already_gone:
        d["o"][n_flat] = 99.90
        d["l"][n_flat] = 99.85
        d["h"][n_flat] = 101.55
        d["c"][n_flat] = 101.40
    return d


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_coil_break_near_200_hits() -> None:
    d = _coil_then_break()
    hits = detect_signals(d)
    assert hits, "黏帶盤整後放量打出箱頂、收盤仍靠近 200，應該進場"
    h = hits[0]
    assert h.close > h.ma200
    assert h.ext <= MAX_ENTRY_EXT
    assert h.ribbon <= 0.006
    assert h.ext <= 0.008
    assert 3.0 <= h.vol_ratio <= 8.0
    assert 99.0 < h.ma200 < 101.5


def test_too_far_from_200_skips() -> None:
    """一根就直豎離開 200，不是「在 200 附近進場」。"""
    d = _coil_then_break(break_close=104.0)
    d["h"][260] = 104.2
    assert detect_signals(d) == []


def test_low_volume_skips() -> None:
    d = _coil_then_break(vol_signal=1050.0)
    assert detect_signals(d) == []


def test_eth_aug28_weak_pop_skips() -> None:
    """ETH 8/28 第一筆：量只有 2.6×，坐在 200 上面的假突破。"""
    d = _coil_then_break(vol_signal=2600.0)
    assert detect_signals(d) == []


def test_wide_ribbon_skips() -> None:
    """均線散開、沒有黏帶，線型不像截圖。"""
    close = np.concatenate(
        [
            np.linspace(90.0, 100.0, 220),
            100.0 + 2.5 * np.sin(np.linspace(0, 6 * np.pi, 40)),
            np.array([103.0]),
        ]
    )
    vol = np.full(len(close), 1000.0)
    vol[-1] = 4000.0
    d = add_indicators(_bars(close, vol, noise=0.004))
    d["o"][-1] = d["c"][-2]
    d["h"][-1] = 104.0
    d["l"][-1] = d["o"][-1] - 0.2
    assert detect_signals(d) == []


def test_chop_without_expansion_skips() -> None:
    rng = np.random.default_rng(3)
    close = 100.0 + 0.25 * rng.standard_normal(300)
    d = add_indicators(_bars(close, np.full(300, 2000.0), noise=0.0006))
    assert detect_signals(d) == []


def test_old_wick_does_not_block_close_break() -> None:
    """舊影線高於第一根放量收盤時，仍應以收盤箱頂判定突破（ETH 9/3 20:30）。"""
    d = _coil_then_break()
    i = 260
    d["h"][i - 8] = 100.55  # 盤整中一根長上影，高於第一根放量收盤
    d["c"][i - 8] = 100.05
    hits = detect_signals(d)
    assert hits, "不該被舊影線擋住靠近 200 的第一根放量陽線"
    assert hits[0].idx == i
    assert hits[0].ribbon <= 0.006


def test_open_gap_expand_skips() -> None:
    """美股開盤那種 19× 振幅，不是 ETH 2.3× 的溫和打出。"""
    d = _coil_then_break()
    i = 260
    d["h"][i] = 104.0
    d["l"][i] = 99.5
    d["c"][i] = 100.65
    assert detect_signals(d) == []


def test_insane_volume_skips() -> None:
    """14× 開盤量不是 ETH 3.7× 那種線。"""
    d = _coil_then_break(vol_signal=14000.0)
    assert detect_signals(d) == []


def test_second_bar_chase_skips() -> None:
    """前一根已經打出箱頂，這根再吃就是追價。"""
    d = _coil_then_break(already_gone=True, break_close=100.70)
    hits = detect_signals(d)
    for h in hits:
        assert h.idx != 261


def test_simulate_stop_and_target() -> None:
    d = _coil_then_break(extra_after=HOLD_BARS + 2)
    hits = detect_signals(d)
    assert hits
    # 把後面做成一路碰到目標
    sig = hits[0]
    entry_i = sig.idx + 1
    for k in range(entry_i, min(entry_i + 6, len(d["c"]))):
        d["h"][k] = sig.target + 0.05
        d["l"][k] = sig.entry - 0.02
        d["c"][k] = sig.target
    trades = simulate_trades(d, hits)
    assert trades
    assert trades[0].exit_reason == "target"
    assert trades[0].pnl_pct > 0

    d2 = _coil_then_break(extra_after=HOLD_BARS + 2)
    hits2 = detect_signals(d2)
    sig2 = hits2[0]
    e2 = sig2.idx + 1
    d2["l"][e2] = sig2.stop - 0.05
    d2["h"][e2] = sig2.entry
    d2["c"][e2] = sig2.stop
    trades2 = simulate_trades(d2, hits2)
    assert trades2[0].exit_reason == "stop"
    assert trades2[0].pnl_pct < 0


def test_summarize() -> None:
    d = _coil_then_break(extra_after=HOLD_BARS + 2)
    hits = detect_signals(d)
    trades = simulate_trades(d, hits)
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    assert "win_rate" in stats


def test_signal_at_needs_history() -> None:
    d = add_indicators(_bars(np.full(50, 100.0)))
    assert signal_at(d, 40) is None


def test_lookback_window_used() -> None:
    assert LOOKBACK >= 16


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"{len(tests)} passed")
