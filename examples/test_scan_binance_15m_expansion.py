#!/usr/bin/env python3
"""15m 盤整後爆量擴張：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_15m_expansion import (  # noqa: E402
    ALERT_BUCKET_MS,
    CLUSTER_COOLDOWN_MS,
    MAX_SAME_MARK,
    TZ,
    bar_index_at,
    collapse_hits,
    detect_expansion,
    drop_market_cluster,
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
    t0 = int(datetime(2026, 8, 21, 12, 0, tzinfo=TZ).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": v}


def _series(
    *,
    rocket: float = 101.9,
    vol_signal: float = 4_000.0,
    grind: bool = True,
    tail: np.ndarray | None = None,
    prev_vol: float | None = None,
) -> dict:
    """長盤整（均線黏、ATR 小）→ 幾根緩推 → 一根爆量長陽。"""
    n_coil = 270
    coil = 100.0 * (1.0 + 0.0008 * np.sin(np.arange(n_coil) / 4.0))
    grind_px = np.array([100.2, 100.4, 100.6, 100.75, 100.9, 101.05]) if grind else np.array([])
    parts = [coil, grind_px, np.array([rocket])]
    if tail is not None:
        parts.append(np.asarray(tail, float))
    close = np.concatenate(parts)
    vol = np.full(len(close), 1_000.0)
    rocket_i = n_coil + len(grind_px)
    if grind:
        vol[n_coil:rocket_i] = 1_600.0
    vol[rocket_i] = vol_signal
    if prev_vol is not None:
        vol[rocket_i - 1] = prev_vol
    return indicators(_bars(len(close), close, vol))


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_rsi_sma_all_up() -> None:
    c = np.linspace(1.0, 2.0, 40)
    r = rsi_sma(c, 6)
    assert r[-1] > 90


def test_expansion_after_coil_hits() -> None:
    d = _series(tail=np.full(4, 102.2))
    hits = detect_expansion(d)
    assert hits, "盤整後在 MA200 附近的放量陽線應該命中"
    h = hits[-1]
    assert h["kind"] == "expand"
    assert h["mark_i"] == h["i"]
    assert h["mark_ext"] <= 0.02
    assert h["body"] >= 0.004
    assert h["vol_ratio"] >= 1.7
    assert h["ma7"] > h["ma14"] > h["ma25"]


def test_tiny_body_skips() -> None:
    """實體太小，不像 ETH 那根長陽。"""
    d = _series(rocket=101.25, tail=np.full(3, 101.3))
    assert detect_expansion(d) == []


def test_low_volume_skips() -> None:
    d = _series(vol_signal=1_050.0, tail=np.full(3, 102.2))
    assert detect_expansion(d) == []


def test_pre_pump_volume_skips() -> None:
    """前一根已經比擴張棒還大聲，不算開始噴。"""
    d = _series(vol_signal=4_000.0, prev_vol=5_000.0, tail=np.full(3, 102.2))
    assert detect_expansion(d) == []


def test_no_grind_skips() -> None:
    """沒有緩推、短均還沒張開，一根尖兵不夠。"""
    d = _series(grind=False, tail=np.full(3, 102.2))
    assert detect_expansion(d) == []


def test_too_far_from_ma200_skips() -> None:
    """已經離開 MA200 超過 2%，不算在 200 附近進。"""
    d = _series(rocket=103.5, tail=np.full(3, 103.6))
    assert detect_expansion(d) == []


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
    d = _series(tail=np.full(8, 102.2))
    alerts = select_alerts(d, int(d["t"][0]), int(d["t"][-1]))
    assert alerts, "合成擴張應有訊號"
    buckets = {int(d["t"][h["i"]]) // ALERT_BUCKET_MS for h in alerts}
    assert len(buckets) == len(alerts)


def test_stop_is_ma200() -> None:
    d = _series(tail=np.full(6, 102.2))
    hits = detect_expansion(d)
    assert hits
    h = hits[0]
    tr = simulate_trade(d, h)
    assert tr is not None
    assert abs(tr["stop"] - h["ma200"]) < 1e-9
    assert tr["stop"] < tr["entry"]
    assert tr["mark_i"] == h["mark_i"]


def test_simulate_stop_on_ma_break() -> None:
    d = _series(tail=np.array([99.0, 98.5, 98.0]))
    hits = detect_expansion(d)
    assert hits
    tr = simulate_trade(d, hits[0])
    assert tr is not None
    assert tr["reason"] == "ma_break"
    assert tr["pnl_pct"] < 0


def test_simulate_target_or_time() -> None:
    d = _series(tail=102.2 * (1.004 ** np.arange(1, 18)))
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


def test_market_cluster_drops_huge_pack() -> None:
    t0 = 1_000_000
    pack = [{"t_mark": t0, "symbol": f"S{i}"} for i in range(MAX_SAME_MARK)]
    lag = {"t_mark": t0 + CLUSTER_COOLDOWN_MS, "symbol": "LAG"}
    later = {"t_mark": t0 + CLUSTER_COOLDOWN_MS + 1, "symbol": "LATER"}
    out = drop_market_cluster(pack + [lag, later])
    assert [x["symbol"] for x in out] == ["LATER"]


def test_small_pack_kept() -> None:
    pack = [{"t_mark": 1_000, "symbol": f"S{i}"} for i in range(3)]
    assert drop_market_cluster(pack) == pack


def test_bar_index_at() -> None:
    d = _bars(10, np.linspace(1, 10, 10))
    ts = int(d["t"][3])
    assert bar_index_at(d, ts) == 3
    assert bar_index_at(d, ts + 1) == 3
    assert bar_index_at(d, ts + 899_999) == 3
    assert bar_index_at(d, int(d["t"][0]) - 1) is None


def main() -> int:
    test_sma()
    test_rsi_sma_all_up()
    test_expansion_after_coil_hits()
    test_tiny_body_skips()
    test_low_volume_skips()
    test_pre_pump_volume_skips()
    test_no_grind_skips()
    test_too_far_from_ma200_skips()
    test_chop_does_not_hit()
    test_collapse_keeps_best_of_run()
    test_select_alerts_debounce()
    test_stop_is_ma200()
    test_simulate_stop_on_ma_break()
    test_simulate_target_or_time()
    test_summarize_empty()
    test_market_cluster_drops_huge_pack()
    test_small_pack_kept()
    test_bar_index_at()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
