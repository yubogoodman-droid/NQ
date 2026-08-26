#!/usr/bin/env python3
"""15m 站上 MA200：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_15m_expansion import (  # noqa: E402
    ALERT_BUCKET_MS,
    MAX_EXT,
    MIN_VOL_RATIO,
    collapse_hits,
    detect_expansion,
    indicators,
    rsi_sma,
    select_alerts,
    simulate_trade,
    sma,
    summarize_trades,
)


def _bars(n: int, close: np.ndarray, vol: np.ndarray | None = None) -> dict:
    close = np.asarray(close, float)
    rng = np.maximum(np.abs(np.diff(close, prepend=close[0])) * 0.35, close * 0.001)
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + rng * 0.25
    l = np.minimum(o, close) - rng * 0.25
    v = vol if vol is not None else np.full(n, 1000.0)
    t = np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": v}


def _reclaim_series(
    *,
    n_base: int = 240,
    px: float = 100.0,
    signal_ext: float = 0.008,
    after: np.ndarray | None = None,
    vol_base: float = 1_000.0,
    vol_signal: float = 4_000.0,
    vol_after: float | None = None,
) -> dict:
    """前面貼在均線下方，最後一根放量收盤剛站上 MA200。"""
    close = np.full(n_base, px * 0.997)
    sig = px * (1.0 + signal_ext)
    close = np.concatenate([close, np.array([sig])])
    if after is not None:
        close = np.concatenate([close, np.asarray(after, float)])
    n = len(close)
    vol = np.full(n, vol_base)
    vol[n_base] = vol_signal
    if after is not None and vol_after is not None:
        vol[n_base + 1 :] = vol_after
    return indicators(_bars(n, close, vol))


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_rsi_sma_all_up() -> None:
    c = np.linspace(1.0, 2.0, 40)
    r = rsi_sma(c, 6)
    assert r[-1] > 90


def test_detect_reclaim_near_ma200() -> None:
    d = _reclaim_series()
    hits = detect_expansion(d)
    assert hits, "貼著 200 線放量站上應該命中"
    last = hits[-1]
    assert last["kind"] == "reclaim"
    assert last["ext"] <= MAX_EXT
    assert last["vol_ratio"] >= MIN_VOL_RATIO
    assert last["close"] > last["ma200"]


def test_gap_far_from_ma200_skips() -> None:
    """跳空收在 MA200 上方 8%，不算『附近』。"""
    d = _reclaim_series(signal_ext=0.08, vol_signal=8_000.0)
    assert detect_expansion(d) == []


def test_no_volume_skips() -> None:
    d = _reclaim_series(vol_signal=1_050.0)
    assert detect_expansion(d) == []


def test_already_above_ma200_skips() -> None:
    """一直在 200 線上方緩漲，沒有站上這一根。"""
    close = 50.0 * (1.00025 ** np.arange(260))
    d = indicators(_bars(len(close), close, np.full(260, 3000.0)))
    assert detect_expansion(d) == []


def test_chop_does_not_hit() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * (1.0 + rng.uniform(-0.0015, 0.0015, 260))
    d = indicators(_bars(len(close), close, np.full(260, 2000.0)))
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
    d = _reclaim_series(after=np.full(8, 100.9))
    t0 = int(d["t"][0])
    t1 = int(d["t"][-1])
    alerts = select_alerts(d, t0, t1)
    assert alerts, "合成站上 200 應有訊號"
    buckets = {int(d["t"][h["i"]]) // ALERT_BUCKET_MS for h in alerts}
    assert len(buckets) == len(alerts)


def test_simulate_stop() -> None:
    after = np.array([99.2, 98.5, 98.0])
    d = _reclaim_series(after=after, vol_after=3_000.0)
    hits = detect_expansion(d)
    assert hits
    tr = simulate_trade(d, hits[-1])
    assert tr is not None
    assert tr["reason"] == "stop"
    assert tr["pnl_pct"] < 0


def test_simulate_target_or_time() -> None:
    after = 101.0 * (1.012 ** np.arange(1, 18))
    d = _reclaim_series(after=after, vol_after=3_500.0)
    hits = detect_expansion(d)
    assert hits
    tr = simulate_trade(d, hits[0])
    assert tr is not None
    assert tr["entry"] > 0
    assert tr["reason"] in {"target", "time", "stop", "eod"}
    assert tr["ribbon"] >= 0


def test_summarize_empty() -> None:
    s = summarize_trades([])
    assert s["count"] == 0
    assert s["pnl"] == 0.0


def main() -> int:
    test_sma()
    test_rsi_sma_all_up()
    test_detect_reclaim_near_ma200()
    test_gap_far_from_ma200_skips()
    test_no_volume_skips()
    test_already_above_ma200_skips()
    test_chop_does_not_hit()
    test_collapse_keeps_best_of_run()
    test_select_alerts_debounce()
    test_simulate_stop()
    test_simulate_target_or_time()
    test_summarize_empty()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
