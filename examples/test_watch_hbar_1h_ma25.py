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


def _bars(closes: np.ndarray, highs: np.ndarray | None = None, lows: np.ndarray | None = None) -> dict:
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
        "v": np.full(n, 1000.0),
    }


def fake_break_then_fail(n: int = 90) -> dict:
    """長時間在 1.00，拉到高點 1.20，再收跌破 MA25，之後停在下面。"""
    c = np.full(n, 1.00)
    h = c + 0.003
    l = c - 0.003
    pump = [1.03, 1.07, 1.11, 1.14, 1.16, 1.155]
    pump_h = [1.04, 1.09, 1.13, 1.17, 1.20, 1.17]
    a = 50
    for k, px in enumerate(pump):
        c[a + k] = px
        h[a + k] = pump_h[k]
        l[a + k] = px - 0.01
    fade = [1.14, 1.12, 1.09, 1.06, 1.04]
    fade_h = [1.15, 1.13, 1.10, 1.07, 1.05]
    for k, px in enumerate(fade):
        c[a + 6 + k] = px
        h[a + 6 + k] = fade_h[k]
        l[a + 6 + k] = px - 0.008
    # 跌破後整段留在 MA25 下，避免再交叉
    fail_i = a + 6 + len(fade) - 1
    for k in range(fail_i + 1, n):
        c[k] = 0.99
        h[k] = 1.00
        l[k] = 0.98
    return indicators(_bars(c, h, l))


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


def test_simulate_short_and_html(tmp_path: Path | None = None) -> None:
    d = fake_break_then_fail()
    sigs = detect_signals(d)
    trades = simulate(d, sigs, symbol="HBARUSDT")
    assert trades
    t = trades[0]
    assert t.entry == sigs[0].close
    assert t.stop == sigs[0].peak_high
    assert t.stop > t.entry
    assert t.target < t.entry
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


def main() -> int:
    test_sma()
    test_drop_unclosed()
    test_detects_hbar_style_dump()
    test_chop_does_not_fire()
    test_already_below_does_not_refire()
    test_tight_ext_filter()
    test_simulate_short_and_html()
    test_filter_universe_keeps_liquid_and_hbar()
    test_skip_overlap_and_chart_pick()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
