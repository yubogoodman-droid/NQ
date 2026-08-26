#!/usr/bin/env python3
"""15m MA200 站穩三根：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_15m_expansion import (  # noqa: E402
    ALERT_BUCKET_MS,
    CONFIRM_BARS,
    TZ,
    collapse_hits,
    detect_expansion,
    indicators,
    rsi_sma,
    select_alerts,
    simulate_trade,
    sma,
    summarize_trades,
)
from datetime import datetime


def _bars(n: int, close: np.ndarray, vol: np.ndarray | None = None) -> dict:
    close = np.asarray(close, float)
    rng = np.maximum(np.abs(np.diff(close, prepend=close[0])) * 0.35, close * 0.001)
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + rng * 0.25
    l = np.minimum(o, close) - rng * 0.25
    v = vol if vol is not None else np.full(n, 1000.0)
    t0 = int(datetime(2026, 8, 21, 12, 0, tzinfo=TZ).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": v}


def _series(
    *,
    holds: np.ndarray | None = None,
    tail: np.ndarray | None = None,
    mark: float = 100.6,
    vol_signal: float = 4_000.0,
) -> dict:
    """先跌破 200 線、再在線下翹頭（做出 MA7>MA14>MA25），然後一根收盤站上。

    holds 是記號之後那幾根收盤，tail 接在確認棒後面。
    """
    base = np.full(250, 100.0)
    dip = np.linspace(99.6, 97.0, 20)
    curl = np.linspace(97.4, 99.4, 10)
    parts = [base, dip, curl, np.array([mark])]
    if holds is not None:
        parts.append(np.asarray(holds, float))
    if tail is not None:
        parts.append(np.asarray(tail, float))
    close = np.concatenate(parts)
    vol = np.full(len(close), 1_000.0)
    vol[250 + 20 + 10] = vol_signal
    return indicators(_bars(len(close), close, vol))


def _held(n: int = CONFIRM_BARS, px: float = 100.6, step: float = 0.004) -> np.ndarray:
    return px * (1.0 + step * np.arange(1, n + 1))


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_rsi_sma_all_up() -> None:
    c = np.linspace(1.0, 2.0, 40)
    r = rsi_sma(c, 6)
    assert r[-1] > 90


def test_enters_after_three_holds() -> None:
    d = _series(holds=_held())
    hits = detect_expansion(d)
    assert hits, "多頭排列站上 200 並站穩三根應該命中"
    h = hits[-1]
    assert h["kind"] == "hold3"
    assert h["i"] - h["mark_i"] == CONFIRM_BARS
    assert h["close"] > h["ma200"]
    assert h["ma7"] > h["ma14"] > h["ma25"]


def test_mark_bar_alone_is_not_entry() -> None:
    """只有記號那根，還沒站滿三根，不能進。"""
    d = _series(holds=None)
    assert detect_expansion(d) == []


def test_two_holds_not_enough() -> None:
    d = _series(holds=_held(n=CONFIRM_BARS - 1))
    assert detect_expansion(d) == []


def test_break_below_ma200_cancels() -> None:
    """第二根收盤跌回 200 線下方，整個記號作廢。"""
    holds = np.array([100.4, 96.0, 100.6])
    d = _series(holds=holds)
    assert detect_expansion(d) == []


def test_low_volume_skips() -> None:
    d = _series(holds=_held(), vol_signal=1_050.0)
    assert detect_expansion(d) == []


def test_mark_too_far_from_ma200_skips() -> None:
    d = _series(holds=_held(), mark=108.0)
    assert detect_expansion(d) == []


def test_bear_stack_skips() -> None:
    """收盤站上 200，但短均是空頭排列（MA7 < MA25），不算。"""
    close = np.concatenate([np.full(240, 101.0), np.full(30, 99.4), np.array([100.2]), _held(100.2)])
    d = indicators(_bars(len(close), close, np.full(len(close), 2_000.0)))
    for h in detect_expansion(d):
        assert h["ma7"] > h["ma14"] > h["ma25"]


def test_chop_does_not_hit() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * (1.0 + rng.uniform(-0.0015, 0.0015, 280))
    d = indicators(_bars(len(close), close, np.full(280, 2000.0)))
    assert detect_expansion(d) == []


def test_collapse_keeps_best_of_run() -> None:
    hits = [
        {"i": 10, "score": 1.0},
        {"i": 11, "score": 3.0},
        {"i": 12, "score": 2.0},
        {"i": 20, "score": 4.0},
    ]
    out = collapse_hits(hits)
    assert [h["i"] for h in out] == [11, 20]
    assert out[0]["score"] == 3.0


def test_select_alerts_debounce() -> None:
    d = _series(holds=_held(), tail=np.full(8, 101.6))
    alerts = select_alerts(d, int(d["t"][0]), int(d["t"][-1]))
    assert alerts, "合成站穩三根應有訊號"
    buckets = {int(d["t"][h["i"]]) // ALERT_BUCKET_MS for h in alerts}
    assert len(buckets) == len(alerts)


def test_stop_sits_under_the_hold_window() -> None:
    d = _series(holds=_held(), tail=np.full(6, 101.6))
    hits = detect_expansion(d)
    assert hits
    h = hits[0]
    tr = simulate_trade(d, h)
    assert tr is not None
    assert tr["stop"] < tr["entry"]
    assert tr["mark_i"] == h["mark_i"]
    assert tr["t_mark"] == int(d["t"][h["mark_i"]])


def test_simulate_stop_on_breakdown() -> None:
    d = _series(holds=_held(), tail=np.array([96.0, 95.0, 94.5]))
    hits = detect_expansion(d)
    assert hits
    tr = simulate_trade(d, hits[0])
    assert tr is not None
    assert tr["reason"] == "ma_break"
    assert tr["pnl_pct"] < 0


def test_simulate_target_or_time() -> None:
    d = _series(holds=_held(), tail=101.2 * (1.012 ** np.arange(1, 18)))
    hits = detect_expansion(d)
    assert hits
    tr = simulate_trade(d, hits[0])
    assert tr is not None
    assert tr["entry"] > 0
    assert tr["reason"] in {"ma_break", "time", "eod"}


def test_summarize_empty() -> None:
    s = summarize_trades([])
    assert s["count"] == 0
    assert s["pnl"] == 0.0


def main() -> int:
    test_sma()
    test_rsi_sma_all_up()
    test_enters_after_three_holds()
    test_mark_bar_alone_is_not_entry()
    test_two_holds_not_enough()
    test_break_below_ma200_cancels()
    test_low_volume_skips()
    test_mark_too_far_from_ma200_skips()
    test_bear_stack_skips()
    test_chop_does_not_hit()
    test_collapse_keeps_best_of_run()
    test_select_alerts_debounce()
    test_stop_sits_under_the_hold_window()
    test_simulate_stop_on_breakdown()
    test_simulate_target_or_time()
    test_summarize_empty()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
