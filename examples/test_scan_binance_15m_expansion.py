#!/usr/bin/env python3
"""15m MA200 站穩三根：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_15m_expansion import (  # noqa: E402
    ALERT_BUCKET_MS,
    CLUSTER_COOLDOWN_MS,
    CONFIRM_BARS,
    TZ,
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


def test_pre_pump_volume_skips() -> None:
    """HOME #12：前一根已爆量，記號根只是第二棒穿越 200，不算開始擴張。"""
    mark = 100.6
    close = np.concatenate(
        [
            np.full(250, 100.0),
            np.linspace(99.6, 97.0, 20),
            np.linspace(97.4, 99.4, 10),
            np.array([mark]),
            _held(),
        ]
    )
    vol = np.full(len(close), 1_000.0)
    mark_i = 250 + 20 + 10
    vol[mark_i - 1] = 4_500.0
    vol[mark_i] = 4_000.0
    d = indicators(_bars(len(close), close, vol))
    assert detect_expansion(d) == []


def test_mark_too_far_from_ma200_skips() -> None:
    d = _series(holds=_held(), mark=108.0)
    assert detect_expansion(d) == []


def test_glued_ma_stack_skips() -> None:
    """MA7 只比 MA14 高一點點，看起來像多頭排列其實沒張開。"""
    close = np.concatenate([np.full(250, 100.0), np.full(20, 99.85), np.array([100.15]), _held(100.15)])
    vol = np.full(len(close), 1_000.0)
    vol[270] = 4_000.0
    d = indicators(_bars(len(close), close, vol))
    for h in detect_expansion(d):
        assert h["ma7"] / h["ma14"] - 1.0 >= 0.001
        assert h["ma7"] / h["ma25"] - 1.0 >= 0.004


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


def test_market_cluster_drops_same_mark_and_laggards() -> None:
    """08-25 那種：同一根記號很多檔，後面幾小時跟風也拿掉。"""
    t0 = 1_000_000
    pack = [{"t_mark": t0, "symbol": f"S{i}"} for i in range(4)]
    lag = {"t_mark": t0 + CLUSTER_COOLDOWN_MS, "symbol": "LAG"}
    later = {"t_mark": t0 + CLUSTER_COOLDOWN_MS + 1, "symbol": "LATER"}
    out = drop_market_cluster(pack + [lag, later])
    assert [x["symbol"] for x in out] == ["LATER"]


def test_three_same_mark_kept() -> None:
    pack = [{"t_mark": 1_000, "symbol": f"S{i}"} for i in range(3)]
    assert drop_market_cluster(pack) == pack


def test_overhead_ma99_ma120_skips() -> None:
    """先漲過一截，99/120 還在 200 上面，再跌破後站回 200：那是打進壓力，不算。"""
    plat = np.full(210, 112.0)
    rise = np.linspace(112, 116, 40)
    drop = np.linspace(115.5, 109.0, 20)
    base_low = np.full(10, 109.2)
    curl = np.linspace(109.3, 110.8, 12)
    close0 = np.concatenate([plat, rise, drop, base_low, curl])
    mark = 113.514016
    holds = np.full(CONFIRM_BARS, mark * 1.004)
    close = np.concatenate([close0, np.array([mark]), holds])
    vol = np.full(len(close), 1_500.0)
    vol[len(close0)] = 9_000.0
    d = indicators(_bars(len(close), close, vol))
    i = len(close0) + CONFIRM_BARS
    assert d["m99"][i] > d["m200"][i]
    assert d["m120"][i] > d["m200"][i]
    assert detect_expansion(d) == []


def main() -> int:
    test_sma()
    test_rsi_sma_all_up()
    test_enters_after_three_holds()
    test_mark_bar_alone_is_not_entry()
    test_two_holds_not_enough()
    test_break_below_ma200_cancels()
    test_low_volume_skips()
    test_pre_pump_volume_skips()
    test_mark_too_far_from_ma200_skips()
    test_glued_ma_stack_skips()
    test_bear_stack_skips()
    test_chop_does_not_hit()
    test_collapse_keeps_best_of_run()
    test_select_alerts_debounce()
    test_stop_sits_under_the_hold_window()
    test_simulate_stop_on_breakdown()
    test_simulate_target_or_time()
    test_summarize_empty()
    test_market_cluster_drops_same_mark_and_laggards()
    test_three_same_mark_kept()
    test_overhead_ma99_ma120_skips()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
