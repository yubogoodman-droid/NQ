#!/usr/bin/env python3
"""Synthetic tests for HBAR 1h 假突破跌破 MA25（不打幣安）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_hbar_1h_ma25 import (  # noqa: E402
    INTERVAL_MS,
    Params,
    Trade,
    crossed_below_ma25,
    detect_at,
    detect_signals,
    drop_unclosed,
    filter_universe,
    format_alert,
    indicators,
    key_of,
    pages_path,
    pick_chart_indices,
    safe_name,
    skip_overlap,
    simulate,
    sma,
    summarize,
    write_html,
    write_view_html,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_drop_unclosed() -> None:
    raw = [[0, 1, 1, 1, 1, 1], [INTERVAL_MS, 1, 1, 1, 1, 1]]
    assert len(drop_unclosed(raw, now_ms=INTERVAL_MS + 10_000)) == 1
    assert len(drop_unclosed(raw, now_ms=2 * INTERVAL_MS)) == 2


def _bars(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    vols: np.ndarray | None = None,
) -> dict:
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    if highs is None:
        highs = c + 0.002
    if lows is None:
        lows = c - 0.002
    o = np.concatenate([[c[0]], c[:-1]])
    t0 = 1_700_000_000_000
    return {
        "t": np.arange(n, dtype=np.int64) * INTERVAL_MS + t0,
        "o": o,
        "h": np.asarray(highs, dtype=float),
        "l": np.asarray(lows, dtype=float),
        "c": c,
        "v": np.full(n, 1000.0) if vols is None else np.asarray(vols, dtype=float),
    }


def fake_break_then_fail(n: int = 150) -> dict:
    """箱體整理後拉到高點 1.21，右肩壓在 MA14 下（MA14 下彎、離 MA99 ≥ 7%）後跌破 MA25。"""
    c = np.full(n, 0.90)
    c[50:90] = 1.00
    h = c + 0.003
    l = c - 0.003
    pump = [1.03, 1.08, 1.13, 1.17, 1.20, 1.18]
    pump_h = [1.04, 1.10, 1.15, 1.19, 1.24, 1.20]
    a = 90
    for k, px in enumerate(pump):
        c[a + k] = px
        h[a + k] = pump_h[k]
        l[a + k] = px - 0.01
    fade = [1.16, 1.14, 1.11, 1.095, 1.09, 1.085, 1.082, 1.078, 1.074, 1.060]
    fade_h = [x + 0.006 for x in fade]
    for k, px in enumerate(fade):
        c[a + 6 + k] = px
        h[a + 6 + k] = fade_h[k]
        l[a + k + 6] = px - 0.006
    # 跌破後整段留在 MA25 下，避免再交叉
    fail_i = a + 6 + len(fade) - 1
    for k in range(fail_i + 1, n):
        c[k] = 0.99
        h[k] = 1.00
        l[k] = 0.98
    v = np.full(n, 1000.0)
    v[a : a + 6] = 5000.0
    return indicators(_bars(c, h, l, v))


def chop_around_ma(n: int = 80) -> dict:
    """在均線附近來回，沒有 8% 假突破。"""
    c = np.full(n, 1.00)
    for i in range(40, n):
        c[i] = 1.00 + (0.006 if i % 2 == 0 else -0.006)
    h = c + 0.004
    l = c - 0.004
    # 最後一根明確收在 MA25 下，但高點伸不夠
    c[-1] = 0.988
    h[-1] = 1.004
    l[-1] = 0.984
    c[-2] = 1.006
    h[-2] = 1.010
    return indicators(_bars(c, h, l))


def test_detects_hbar_style_dump() -> None:
    d = fake_break_then_fail()
    sigs = detect_signals(d)
    assert len(sigs) == 1, f"expected 1 signal, got {[(s.i, s.close, s.ma25) for s in sigs]}"
    sig = sigs[0]
    assert crossed_below_ma25(d, sig.i)
    assert d["c"][sig.i - 1] >= d["m25"][sig.i - 1]
    assert sig.close < sig.ma25
    assert sig.ext >= 0.08
    assert sig.fail >= 0.05
    assert 1 <= sig.bars_after <= 36
    assert sig.peak_high >= 1.19
    assert sig.range_break >= 0.22
    assert sig.close_below <= 0.025
    assert sig.vol_ratio >= 1.8
    assert sig.break_vol_ratio >= 0.35
    assert sig.ma7_below_ma14 >= 0.01
    assert sig.room_ma99 >= 0.07
    assert 0.15 <= sig.ext <= 0.40
    assert detect_at(d, sig.i + 1) is None


def test_chop_does_not_fire() -> None:
    d = chop_around_ma()
    assert detect_signals(d) == []


def test_already_below_does_not_refire() -> None:
    d = fake_break_then_fail()
    sigs = detect_signals(d)
    i = sigs[0].i
    assert detect_at(d, i) is not None
    # 下一根繼續在 MA25 下不算新跌破
    d["c"][i + 1] = min(float(d["c"][i]) - 0.002, float(d["m25"][i + 1]) - 0.002)
    d["h"][i + 1] = d["c"][i + 1] + 0.002
    d["l"][i + 1] = d["c"][i + 1] - 0.002
    d = indicators(d)
    assert not crossed_below_ma25(d, i + 1)
    assert detect_at(d, i + 1) is None


def test_tight_ext_filter() -> None:
    d = fake_break_then_fail()
    loose = detect_signals(d, p=Params(min_ext=0.08))
    tight = detect_signals(d, p=Params(min_ext=0.50))
    assert loose
    assert tight == []


def test_rejects_meme_extension() -> None:
    d = fake_break_then_fail()
    peak = detect_signals(d)[0].peak_i
    d["h"][peak] = float(d["m25"][peak]) * 1.80
    d = indicators(d)
    assert detect_signals(d) == []


def test_rejects_without_ma14_cap() -> None:
    d = fake_break_then_fail()
    sig = detect_signals(d)[0]
    d["c"][sig.i - 1] = float(d["m14"][sig.i - 1]) + 0.01
    d["h"][sig.i - 1] = d["c"][sig.i - 1] + 0.005
    d = indicators(d)
    assert detect_signals(d) == []


def test_rejects_without_ma99_room() -> None:
    d = fake_break_then_fail()
    assert detect_signals(d, p=Params(min_room_ma99=0.35)) == []


def test_rejects_dead_volume_breakdown() -> None:
    d = fake_break_then_fail()
    sig = detect_signals(d)[0]
    d["v"][sig.i] = 100.0
    d = indicators(d)
    assert detect_signals(d) == []


def test_simulate_short_and_html(tmp_path: Path | None = None) -> None:
    d = fake_break_then_fail()
    sigs = detect_signals(d)
    trades = simulate(d, sigs, symbol="HBARUSDT")
    assert trades
    t = trades[0]
    assert t.entry == sigs[0].close
    assert abs(t.stop - min(sigs[0].peak_high, t.entry * 1.12)) < 1e-9
    assert t.stop > t.entry
    assert t.target < t.entry
    raw_peak = simulate(d, sigs, p=Params(max_stop_pct=0.0), symbol="HBARUSDT")[0]
    assert raw_peak.stop == sigs[0].peak_high
    stats = summarize(trades)
    assert stats["count"] == 1
    text = format_alert("HBARUSDT", d, sigs[0])
    assert "假突破" in text and "MA25" in text
    assert "HBARUSDT" in key_of("HBARUSDT", d, sigs[0])
    out = Path("/tmp/hbar_1h_ma25_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html(
        out,
        trades,
        stats,
        {
            "days": 7,
            "min_ext": 0.08,
            "min_fail": 0.05,
            "target_r": 2.0,
            "time_bars": 24,
            "symbol": "HBARUSDT",
        },
    )
    html = path.read_text(encoding="utf-8")
    assert "假突破跌破 MA25" in html
    assert "<img src='img/" in html
    assert any((path.parent / "img").glob("t01_*.png"))
    vpath = write_view_html(path)
    assert vpath.exists()


def test_filter_universe_keeps_liquid_and_hbar() -> None:
    info = {
        "symbols": [
            {"symbol": "AAAUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "HBARUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "ETHBTC", "quoteAsset": "BTC", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "DEADUSDT", "quoteAsset": "USDT", "status": "BREAK", "contractType": "PERPETUAL"},
            {"symbol": "IDXUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "INDEX"},
        ]
    }
    tickers = [
        {"symbol": "AAAUSDT", "quoteVolume": "1000"},
        {"symbol": "HBARUSDT", "quoteVolume": "100"},
        {"symbol": "BTCUSDT", "quoteVolume": "9000000"},
    ]
    out = filter_universe(info, tickers, min_quote_vol=5_000_000)
    assert out == ["HBARUSDT", "BTCUSDT"]


def test_skip_overlap_and_chart_pick() -> None:
    d = fake_break_then_fail()
    sig = detect_signals(d)[0]
    a = Trade(
        symbol="AAAUSDT",
        signal=sig,
        entry_idx=10,
        exit_idx=20,
        entry=1.0,
        exit=0.9,
        stop=1.2,
        target=0.6,
        pnl_pct=0.10,
        reason="time",
        d=d,
    )
    b = Trade(
        symbol="AAAUSDT",
        signal=sig,
        entry_idx=15,
        exit_idx=25,
        entry=1.0,
        exit=1.1,
        stop=1.2,
        target=0.6,
        pnl_pct=-0.10,
        reason="stop",
        d=d,
    )
    c = Trade(
        symbol="HBARUSDT",
        signal=sig,
        entry_idx=12,
        exit_idx=18,
        entry=1.0,
        exit=0.95,
        stop=1.2,
        target=0.6,
        pnl_pct=0.05,
        reason="time",
        d=d,
    )
    kept = skip_overlap([b, a, c])
    assert [(t.symbol, t.entry_idx) for t in kept] == [("AAAUSDT", 10), ("HBARUSDT", 12)]
    ix = pick_chart_indices([a, b, c], max_charts=2)
    assert 2 in ix  # pin HBAR
    assert safe_name("HBARUSDT") == "HBARUSDT"
    assert "_" in safe_name("A/B")
    assert pages_path(universe=True, days=7, symbol="HBARUSDT").as_posix().endswith("binance-1h-fake-ma25-7d/index.html")
    assert pages_path(universe=True, days=730, symbol="HBARUSDT").as_posix().endswith("binance-1h-fake-ma25-730d/index.html")


def test_two_layer_stop_ma14_and_cap() -> None:
    # 1. 單根或兩根站上 MA14 不會被洗掉
    d1 = fake_break_then_fail()
    sig = detect_signals(d1)[0]
    i = sig.i
    d1["c"][i + 1] = 1.14
    d1["h"][i + 1] = 1.15
    d1["c"][i + 2] = 1.14
    d1["h"][i + 2] = 1.15
    d1 = indicators(d1)
    t1 = simulate(d1, [sig])[0]
    assert t1.reason != "m14"

    # 2. 連 3 根 1h 收上 MA14 → 結構軟停損提早出場，不等摸到硬停損
    d2 = fake_break_then_fail()
    for k in (1, 2, 3):
        d2["c"][i + k] = 1.14
        d2["h"][i + k] = 1.15
        d2["l"][i + k] = 1.12
    d2 = indicators(d2)
    t2 = simulate(d2, [sig])[0]
    assert t2.reason == "m14"
    assert t2.exit_idx == i + 3
    assert t2.pnl_pct > -0.12

    # 3. 單根爆拉超過 +12% → 觸發 12% 硬停損上限（不等 +17% 假突破頂）
    d3 = fake_break_then_fail()
    d3["h"][i + 1] = sig.close * 1.15
    d3 = indicators(d3)
    t3 = simulate(d3, [sig])[0]
    assert t3.reason == "stop"
    assert abs(t3.pnl_pct - (-0.12)) < 1e-9


def main() -> int:
    test_sma()
    test_drop_unclosed()
    test_detects_hbar_style_dump()
    test_chop_does_not_fire()
    test_already_below_does_not_refire()
    test_tight_ext_filter()
    test_rejects_meme_extension()
    test_rejects_without_ma14_cap()
    test_rejects_without_ma99_room()
    test_rejects_dead_volume_breakdown()
    test_simulate_short_and_html()
    test_two_layer_stop_ma14_and_cap()
    test_filter_universe_keeps_liquid_and_hbar()
    test_skip_overlap_and_chart_pick()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
