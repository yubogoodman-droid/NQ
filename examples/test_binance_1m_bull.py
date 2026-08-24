#!/usr/bin/env python3
"""Synthetic tests for 1m MA7>14>25>99>120 above MA200 stack (no Binance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma1m_bull import (  # noqa: E402
    add_mas,
    detect_combo,
    forward_moves,
    ma_widths,
    ribbon_ok,
    sma,
    stack_ok,
    summarize_rows,
    SignalRow,
)

LOOSE = dict(
    max_ribbon_pct=None,
    max_short_pct=None,
    max_prior_short=None,
    min_vol_ratio=0,
    min_below=0,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def _make_stack_bars(n: int = 360) -> dict:
    """先慢慢跌、再往上黏，讓收盤剛過 MA200 時仍是 200>7>14>25>99>120。"""
    close = np.zeros(n, dtype=float)
    close[0] = 110.0
    for i in range(1, 220):
        close[i] = close[i - 1] - 0.05
    for i in range(220, n):
        close[i] = close[i - 1] + 0.015
    rng = np.random.default_rng(0)
    high = close + 0.04
    low = close - 0.04
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(100, 200, n)
    t0 = 1_700_000_000_000
    return {
        "t": np.arange(n, dtype=np.int64) * 60_000 + t0,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "v": vol,
    }


def test_detects_first_stack_above_ma200() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d, **LOOSE)
    assert len(sigs) >= 1
    first = sigs[0]
    assert first.idx >= 199
    assert stack_ok(d, first.idx)
    assert not stack_ok(d, first.idx - 1)
    assert d["c"][first.idx] > d["m200"][first.idx] > d["m7"][first.idx] > d["m14"][first.idx]
    assert d["m14"][first.idx] > d["m25"][first.idx] > d["m99"][first.idx] > d["m120"][first.idx]


def test_no_repeat_while_stack_holds() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d, **LOOSE)
    # After the first print, the grind keeps the stack; should not spam every bar.
    assert len(sigs) == 1


def _make_rearm_bars() -> dict:
    """第一次站上後跌破，短均再黏、再剛過 MA200。"""
    raw = _make_stack_bars(460)
    for i in range(325, 333):
        raw["c"][i] = raw["c"][i - 1] - 0.15
    for i in range(333, 453):
        raw["c"][i] = raw["c"][i - 1]
    raw["c"][453] = raw["c"][452] + 0.8
    for i in range(325, len(raw["c"])):
        raw["h"][i] = raw["c"][i] + 0.04
        raw["l"][i] = raw["c"][i] - 0.04
        raw["o"][i] = raw["c"][i - 1]
    return raw


def test_rearms_after_break() -> None:
    d = add_mas(_make_rearm_bars())
    sigs = detect_combo(d, **LOOSE)
    assert len(sigs) >= 2
    assert sigs[1].idx > sigs[0].idx + 10
    assert stack_ok(d, sigs[1].idx)
    assert d["m200"][sigs[1].idx] > d["m7"][sigs[1].idx]


def test_min_gap() -> None:
    d = add_mas(_make_rearm_bars())
    all_sigs = detect_combo(d, min_gap_bars=0, **LOOSE)
    gapped = detect_combo(d, min_gap_bars=200, **LOOSE)
    assert len(all_sigs) >= 2
    assert len(gapped) == 1


def test_forward_and_summarize() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d, **LOOSE)
    entry, moves = forward_moves(d, sigs[0])
    assert entry > 0
    assert moves[5].ret_pct is not None
    assert moves[5].ret_pct > 0
    row = SignalRow(symbol="TESTUSDT", sig=sigs[0], time_ms=int(d["t"][sigs[0].idx]), entry=entry, moves=moves)
    stats = summarize_rows([row], 5)
    assert stats["n"] == 1
    assert stats["wr"] == 100.0


def test_cross_only_keeps_ma200_reclaim() -> None:
    d = add_mas(_make_stack_bars())
    all_sigs = detect_combo(d, cross_only=False, **LOOSE)
    crosses = detect_combo(d, cross_only=True, **LOOSE)
    assert len(crosses) >= 1
    assert all(s.crossed_200 for s in crosses)
    assert len(crosses) <= len(all_sigs)


def test_below_ma200_is_not_a_signal() -> None:
    n = 250
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] - 0.05
    d = add_mas(
        {
            "t": np.arange(n, dtype=np.int64) * 60_000,
            "o": close,
            "h": close + 0.02,
            "l": close - 0.02,
            "c": close,
            "v": np.ones(n) * 10,
        }
    )
    assert detect_combo(d, **LOOSE) == []


def test_stack_allows_ma200_still_above_shorts() -> None:
    """剛站上時常見 收盤>200>7>14>25，200 還壓在短均上仍算排列。"""
    d = {
        "c": np.array([10.5] * 3),
        "m7": np.array([10.2] * 3),
        "m14": np.array([10.1] * 3),
        "m25": np.array([10.0] * 3),
        "m99": np.array([9.9] * 3),
        "m120": np.array([9.8] * 3),
        "m200": np.array([10.3] * 3),
    }
    assert stack_ok(d, 2)
    d["c"] = np.array([10.25] * 3)
    assert not stack_ok(d, 2)


def test_screenshot_circle_stack_and_width() -> None:
    """SNDK 08-24 23:35 紅圈：7>14>25>99>120，剛站上 MA200。"""
    d = {
        "c": np.array([1470.0, 1472.97, 1472.97]),
        "m7": np.array([1468.13] * 3),
        "m14": np.array([1464.19] * 3),
        "m25": np.array([1461.97] * 3),
        "m99": np.array([1450.89] * 3),
        "m120": np.array([1448.16] * 3),
        "m200": np.array([1471.07, 1470.89, 1470.89]),
    }
    assert stack_ok(d, 1)
    assert d["m200"][1] > d["m7"][1]
    _ribbon, short, pack = ma_widths(d, 1)
    assert 0.40 < short < 0.45
    assert 0.60 < pack < 0.63
    assert ribbon_ok(d, 1)


def test_ribbon_ok_rejects_fanned_shorts() -> None:
    d = {
        "c": np.array([101.0] * 3),
        "m7": np.array([100.0] * 3),
        "m14": np.array([99.2] * 3),
        "m25": np.array([98.5] * 3),
        "m99": np.array([97.8] * 3),
        "m120": np.array([97.2] * 3),
        "m200": np.array([100.4] * 3),
    }
    assert stack_ok(d, 1)
    _ribbon, short, pack = ma_widths(d, 1)
    assert short > 0.50
    assert pack > 0.65
    assert not ribbon_ok(d, 1)


def test_default_circled_filters_need_volume() -> None:
    d = add_mas(_make_stack_bars())
    assert detect_combo(d, **LOOSE)
    # 合成量沒放量，紅圈預設 vol≥1.4 不該過
    assert detect_combo(d) == []


def test_default_date_uses_yesterday_before_2am() -> None:
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from binance_1m_bull import default_date, day_window_ms, kline_fetch_days, window_label

    tz = timezone(timedelta(hours=8))
    early = datetime(2026, 8, 25, 0, 58, tzinfo=tz)
    assert default_date(early) == "2026-08-24"
    late = datetime(2026, 8, 24, 17, 0, tzinfo=tz)
    assert default_date(late) == "2026-08-24"
    lo, hi = day_window_ms("2026-08-24")
    assert hi - lo == 24 * 60 * 60 * 1000
    lo30, hi30 = day_window_ms("2026-08-25", 30)
    assert hi30 - lo30 == 30 * 24 * 60 * 60 * 1000
    assert window_label("2026-08-25", 30) == "2026-07-27 → 2026-08-25"
    assert kline_fetch_days(30) == 31


def test_is_usdt_stock_perp() -> None:
    from nq.binance import is_usdt_crypto_perp, is_usdt_stock_perp, select_universe

    stock = {
        "symbol": "SNDKUSDT",
        "quoteAsset": "USDT",
        "marginAsset": "USDT",
        "status": "TRADING",
        "contractType": "TRADIFI_PERPETUAL",
        "underlyingType": "EQUITY",
    }
    btc = {**stock, "symbol": "BTCUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}
    assert is_usdt_stock_perp(stock)
    assert is_usdt_stock_perp({**stock, "symbol": "SKHYNIXUSDT", "underlyingType": "KR_EQUITY"})
    assert is_usdt_stock_perp({**stock, "symbol": "OPENAIUSDT", "underlyingType": "PREMARKET"})
    assert not is_usdt_stock_perp(btc)
    assert is_usdt_crypto_perp(btc)
    assert not is_usdt_crypto_perp(stock)
    assert not is_usdt_stock_perp({**stock, "symbol": "XAUUSDT", "underlyingType": "COMMODITY"})
    assert not is_usdt_crypto_perp({**btc, "symbol": "XAUUSDT", "underlyingType": "COMMODITY"})
    assert not is_usdt_stock_perp({**stock, "quoteAsset": "USDC"})
    assert not is_usdt_stock_perp({**stock, "marginAsset": "BTC"})
    assert not is_usdt_stock_perp({**stock, "contractType": "CURRENT_QUARTER"})
    assert not is_usdt_stock_perp({**stock, "underlyingType": "INDEX"})
    assert not is_usdt_stock_perp({**stock, "status": "BREAK"})

    eth = {**btc, "symbol": "ETHUSDT"}
    doge = {**btc, "symbol": "DOGEUSDT"}
    mu = {**stock, "symbol": "MUUSDT"}
    symbols = [stock, mu, btc, eth, doge, {**stock, "symbol": "XAUUSDT", "underlyingType": "COMMODITY"}]
    tickers = {
        "SNDKUSDT": {"quoteVolume": "100"},
        "MUUSDT": {"quoteVolume": "40"},
        "BTCUSDT": {"quoteVolume": "900"},
        "ETHUSDT": {"quoteVolume": "500"},
        "DOGEUSDT": {"quoteVolume": "80"},
        "XAUUSDT": {"quoteVolume": "300"},
    }
    both = select_universe(symbols, tickers, pool="both", top_n=1)
    assert [x[0] for x in both] == ["BTCUSDT", "SNDKUSDT"]
    assert [x[2] for x in both] == ["crypto", "stock"]
    assert select_universe(symbols, tickers, pool="crypto", top_n=2)[0][0] == "BTCUSDT"
    assert "XAUUSDT" not in {x[0] for x in select_universe(symbols, tickers, pool="both", top_n=0)}


def test_write_view_html_uses_pages_urls() -> None:
    import importlib.util

    path = Path(__file__).resolve().parent / "binance_1m_bull.py"
    spec = importlib.util.spec_from_file_location("binance_1m_bull", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = Path("/tmp/ma1m-view-test")
    tmp.mkdir(parents=True, exist_ok=True)
    src = tmp / "ma1m-bull.html"
    src.write_text("<img src='img/ma1m-bull/SNDK_0824_0642.png' alt='SNDKUSDT'/>", encoding="utf-8")
    out = mod.write_view_html(src)
    text = out.read_text(encoding="utf-8")
    assert "https://raw.githubusercontent.com/yubogoodman-droid/NQ/gh-pages/binance/img/ma1m-bull/SNDK_0824_0642.png" in text
    assert "src='img/" not in text
    assert "raw.githubusercontent.com/yubogoodman-droid/NQ/gh-pages/binance/img/ma1m-bull/" in mod.img_src(
        "img/ma1m-bull/SNDK_0825_0156.png"
    )


def main() -> int:
    test_sma()
    test_detects_first_stack_above_ma200()
    test_no_repeat_while_stack_holds()
    test_rearms_after_break()
    test_min_gap()
    test_forward_and_summarize()
    test_cross_only_keeps_ma200_reclaim()
    test_below_ma200_is_not_a_signal()
    test_stack_allows_ma200_still_above_shorts()
    test_screenshot_circle_stack_and_width()
    test_ribbon_ok_rejects_fanned_shorts()
    test_default_circled_filters_need_volume()
    test_default_date_uses_yesterday_before_2am()
    test_is_usdt_stock_perp()
    test_write_view_html_uses_pages_urls()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
