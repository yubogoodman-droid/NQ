#!/usr/bin/env python3
"""Synthetic tests for 幣安 5m 多頭排列（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_5m_align import (  # noqa: E402
    add_mas,
    attach_forwards,
    collect_signals,
    detect_new_align,
    drop_unclosed,
    five_align_ok,
    format_alert,
    forward_pct,
    hour_above_at,
    hour_above_ok,
    hour_mas_at,
    key_of,
    parse_klines,
    pick_chart_hits,
    reclaim_ma200,
    sma,
    summarize_hits,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_drop_unclosed() -> None:
    raw = [[0, 1, 1, 1, 1, 1], [300_000, 1, 1, 1, 1, 1]]
    assert len(drop_unclosed(raw, 300_000, now_ms=300_000 + 10_000)) == 1
    assert len(drop_unclosed(raw, 300_000, now_ms=600_000)) == 2


def test_parse_klines() -> None:
    d = parse_klines([[1, "2", "4", "1", "3", "9"]])
    assert d["t"][0] == 1
    assert d["c"][0] == 3.0
    assert d["v"][0] == 9.0


MS = 300_000


def _bars(closes: np.ndarray, t0: int = 1_700_000_000_000, step: int = MS) -> dict:
    n = len(closes)
    rows = []
    for i, c in enumerate(closes):
        px = float(c)
        rows.append([t0 + i * step, px, px + 0.2, px - 0.2, px, 100.0])
    return parse_klines(rows)


def rising(n: int, start: float = 100.0, step: float = 0.15) -> np.ndarray:
    return start + np.arange(n, dtype=float) * step


def reclaim_closes(n: int = 260) -> np.ndarray:
    """長時間在 100，先跌到 MA200 下再爬、最後一根收過 MA200。"""
    c = np.full(n, 100.0)
    c[190:210] = 96.5
    c[210:229] = np.linspace(97.0, 99.35, 19)
    c[229] = 101.4
    c[230:] = 101.6
    return c


def reclaim_5m(t0: int = 1_700_000_000_000):
    return add_mas(_bars(reclaim_closes(), t0=t0), (7, 14, 25, 200))


def hour_ok_hi():
    return add_mas(_bars(rising(240, 50.0, 0.4), step=3_600_000), (99, 200))


def reclaim_live_pair():
    t0 = 1_700_000_000_000
    d5 = add_mas(_bars(reclaim_closes(), t0=t0 + 220 * 3_600_000), (7, 14, 25, 200))
    h1 = add_mas(_bars(np.full(240, 80.0), t0=t0, step=3_600_000), (99, 200))
    return d5, h1


def test_five_align_rising() -> None:
    d = add_mas(_bars(rising(240)), (7, 14, 25, 200))
    assert five_align_ok(d, 230)
    assert not five_align_ok(d, 10)


def test_hour_above() -> None:
    h = add_mas(_bars(rising(240, 50.0, 0.4), step=3_600_000), (99, 200))
    assert hour_above_ok(h, 220)
    flat = add_mas(_bars(np.full(240, 100.0), step=3_600_000), (99, 200))
    assert not hour_above_ok(flat, 220)
    below = add_mas(_bars(np.concatenate([rising(200, 100.0, 0.2), np.full(40, 90.0)]), step=3_600_000), (99, 200))
    assert not hour_above_ok(below, 239)


def test_already_above_does_not_fire() -> None:
    d5 = add_mas(_bars(rising(240)), (7, 14, 25, 200))
    h1 = add_mas(_bars(rising(240, 80.0, 0.3), step=3_600_000), (99, 200))
    assert five_align_ok(d5, 230)
    assert not reclaim_ma200(d5, 230)
    assert all(detect_new_align(d5, i, h1, 230) is None for i in range(len(d5["c"])))


def test_detect_reclaim_bar() -> None:
    d5, h1 = reclaim_5m(), hour_ok_hi()
    hi = len(h1["c"]) - 1
    hits = [i for i in range(len(d5["c"])) if detect_new_align(d5, i, h1, hi)]
    assert hits == [229]
    assert reclaim_ma200(d5, 229)
    assert five_align_ok(d5, 229)
    assert d5["c"][228] < d5["m200"][228]
    assert d5["c"][229] > d5["m200"][229]
    assert detect_new_align(d5, 230, h1, hi) is None


def test_detect_blocks_without_hour_filter() -> None:
    d5 = reclaim_5m()
    weak = add_mas(_bars(np.concatenate([rising(200, 100.0, 0.2), np.full(40, 80.0)]), step=3_600_000), (99, 200))
    assert detect_new_align(d5, 229, weak, len(weak["c"]) - 1) is None


def test_hour_mas_no_lookahead() -> None:
    h_c = rising(240, 50.0, 0.4)
    h_c[-1] = 10.0
    h = add_mas(_bars(h_c, step=3_600_000), (99, 200))
    px = float(h_c[-2])
    t = int(h["t"][-1])
    mas = hour_mas_at(h, t, px)
    assert mas is not None
    assert px > mas[0] and px > mas[1]
    assert hour_above_at(h, t, px)
    assert not hour_above_ok(h, 239)


def test_collect_signals_first_bar_only() -> None:
    d5, h1 = reclaim_live_pair()
    start, end = int(d5["t"][0]), int(d5["t"][-1])
    hits = collect_signals(d5, h1, start, end)
    assert [h["i"] for h in hits] == [229]
    assert reclaim_ma200(d5, hits[0]["i"])


def test_forward_and_summary() -> None:
    closes = rising(240)
    d5 = add_mas(_bars(closes), (7, 14, 25, 200))
    i = 200
    pct = forward_pct(d5, i, 12)
    expect = (closes[212] / closes[200] - 1) * 100
    assert pct is not None and abs(pct - expect) < 1e-9
    assert forward_pct(d5, 230, 20) is None
    sig = {"i": i, "t": int(d5["t"][i]), "close": float(closes[i]), "symbol": "AAAUSDT"}
    row = attach_forwards(d5, sig)
    assert row["15m"] is not None
    hits = [
        {**row, "symbol": "AAAUSDT"},
        {**row, "symbol": "BBBUSDT", "60m": -1.0, "15m": 0.5, "30m": None, "120m": 2.0},
    ]
    stats = summarize_hits(hits)
    assert stats["count"] == 2
    assert stats["symbols"] == 2
    assert stats["60m"]["n"] == 2
    picked = pick_chart_hits(hits, 2)
    assert len(picked) == 2


def test_format_and_key() -> None:
    d5, h1 = reclaim_5m(), hour_ok_hi()
    hi = len(h1["c"]) - 1
    sig = detect_new_align(d5, 229, h1, hi)
    ev = {"symbol": "BTCUSDT", "sig": sig, "d5": d5, "h1": h1}
    text = format_alert(ev)
    assert "BTCUSDT" in text
    assert "多頭排列" in text
    assert "MA200 下" in text
    assert key_of(ev) == f"BTCUSDT:{sig['t']}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
