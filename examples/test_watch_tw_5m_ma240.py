#!/usr/bin/env python3
"""Synthetic tests for 台股五分K 回測 240MA (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_tw_5m_ma240 import (  # noqa: E402
    TPE,
    add_mas,
    detect_retests,
    filter_hits_days,
    format_hit,
    format_hit_line,
    hit_from_row,
    market_session,
    next_session_open,
    seconds_until_next_5m,
    seconds_until_next_scan,
    pin_keep,
    session_dates_back,
    touch_band,
    tw_tick,
)


def _bars(close: np.ndarray, *, low_at: dict[int, float] | None = None) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2026-08-03 09:00", periods=n, freq="5min", tz=TPE)
    low = close - 0.15
    high = close + 0.15
    if low_at:
        for i, val in low_at.items():
            low[i] = val
            high[i] = max(high[i], close[i], val)
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _extended_then(last_close: float, last_low: float, n: int = 280) -> pd.DataFrame:
    """前 240 根在 100，接著拉開到 104，最後一根自行指定。"""
    close = np.empty(n, dtype=float)
    close[:240] = 100.0
    close[240:-1] = np.linspace(102.0, 104.5, n - 241)
    close[-1] = last_close
    df = _bars(close, low_at={n - 1: last_low, n - 2: 103.8})
    return add_mas(df)


def test_tw_tick() -> None:
    assert tw_tick(9.9) == 0.01
    assert tw_tick(25) == 0.05
    assert tw_tick(80) == 0.1
    assert tw_tick(124.5) == 0.5
    assert tw_tick(600) == 1.0
    assert tw_tick(1400) == 5.0


def test_touch_band_uses_ticks() -> None:
    # 124 附近 0.35% ≈ 0.43，兩檔 0.5*2=1.0 比較大
    band = touch_band(124.0, 0.0035, ticks=2)
    assert abs(band - 1.0) < 1e-9
    cheap = touch_band(20.0, 0.0035, ticks=2)
    assert cheap >= 20.0 * 0.0035


def test_detect_pullback_to_ma240() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    assert ma > 99
    df.iloc[-1, df.columns.get_loc("Low")] = ma - 0.05
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 0.25
    df.iloc[-1, df.columns.get_loc("High")] = ma + 0.40
    hits = detect_retests(df)
    assert hits == [len(df) - 1], hits


def test_hug_ma_does_not_fire() -> None:
    n = 280
    close = np.full(n, 100.2)
    df = add_mas(_bars(close))
    assert detect_retests(df) == []


def test_close_breakdown_is_not_retest() -> None:
    df = _extended_then(last_close=98.0, last_low=97.5)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma - 2.0
    df.iloc[-1, df.columns.get_loc("Close")] = ma * 0.98
    df.iloc[-1, df.columns.get_loc("High")] = ma - 0.4
    assert detect_retests(df) == []


def test_only_first_touch_fires() -> None:
    df = _extended_then(last_close=101.0, last_low=100.5, n=282)
    ma_a = float(df["MA240"].iloc[-2])
    ma_b = float(df["MA240"].iloc[-1])
    df.iloc[-2, df.columns.get_loc("Low")] = ma_a - 0.05
    df.iloc[-2, df.columns.get_loc("Close")] = ma_a + 0.2
    df.iloc[-2, df.columns.get_loc("High")] = ma_a + 0.4
    df.iloc[-1, df.columns.get_loc("Low")] = ma_b - 0.05
    df.iloc[-1, df.columns.get_loc("Close")] = ma_b + 0.2
    df.iloc[-1, df.columns.get_loc("High")] = ma_b + 0.4
    hits = detect_retests(df)
    assert hits == [len(df) - 2], hits


def test_shallow_dip_not_retest() -> None:
    df = _extended_then(last_close=104.0, last_low=103.8)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma + 1.20
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 1.50
    df.iloc[-1, df.columns.get_loc("High")] = ma + 1.70
    assert detect_retests(df) == []


def test_cooldown_skips_nearby_retest() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4, n=286)
    # 兩根都刺到 240MA，但只隔 3 根
    for j in (-4, -1):
        ma = float(df["MA240"].iloc[j])
        df.iloc[j, df.columns.get_loc("Low")] = ma - 0.05
        df.iloc[j, df.columns.get_loc("Close")] = ma + 0.20
        df.iloc[j, df.columns.get_loc("High")] = ma + 0.40
        df.iloc[j - 1, df.columns.get_loc("Low")] = ma + 1.5
    hits = detect_retests(df, cooldown_bars=6)
    assert hits == [len(df) - 4], hits
    hits2 = detect_retests(df, cooldown_bars=1)
    assert hits2 == [len(df) - 4, len(df) - 1], hits2


def test_skip_open_minutes() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma - 0.05
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 0.25
    df.iloc[-1, df.columns.get_loc("High")] = ma + 0.40
    idx = list(df.index)
    idx[-1] = pd.Timestamp("2026-08-28 09:05", tz=TPE)
    df.index = pd.DatetimeIndex(idx)
    assert detect_retests(df, skip_open_minutes=15) == []
    assert detect_retests(df) == [len(df) - 1]
    idx[-1] = pd.Timestamp("2026-08-28 09:20", tz=TPE)
    df.index = pd.DatetimeIndex(idx)
    assert detect_retests(df, skip_open_minutes=15) == [len(df) - 1]


def test_touch_then_bounce_still_counts() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma + 0.05
    df.iloc[-1, df.columns.get_loc("Close")] = ma * 1.012
    df.iloc[-1, df.columns.get_loc("High")] = ma * 1.015
    idx = list(df.index)
    idx[-1] = pd.Timestamp("2026-08-28 09:05", tz=TPE)
    df.index = pd.DatetimeIndex(idx)
    assert detect_retests(df) == [len(df) - 1]


def test_select_universe_drops_price_over_700() -> None:
    from watch_tw_5m_ma240 import select_universe

    rows = [
        {"code": "2330", "name": "台積電", "close": 1400.0, "amount": 9, "rank": 1, "market": "tse", "symbol": "2330.TW"},
        {"code": "1815", "name": "1815", "close": 125.0, "amount": 8, "rank": 2, "market": "otc", "symbol": "1815.TWO"},
        {"code": "2303", "name": "聯電", "close": 55.0, "amount": 7, "rank": 3, "market": "tse", "symbol": "2303.TW"},
        {"code": "2454", "name": "聯發科", "close": 4100.0, "amount": 6, "rank": 4, "market": "tse", "symbol": "2454.TW"},
    ]
    kept, dropped = select_universe(rows, limit=200, keep=["1815"], max_price=700)
    assert [r["code"] for r in kept] == ["1815", "2303"]
    assert {r["code"] for r in dropped} == {"2330", "2454"}
    assert kept[0]["name"] == "富喬"
    none, dropped0 = select_universe(rows, limit=2, keep=["1815"], max_price=None)
    assert [r["code"] for r in none[:2]] == ["1815", "2330"]
    assert dropped0 == []


def test_cli_max_price_default() -> None:
    from watch_tw_5m_ma240 import build_parser

    args = build_parser().parse_args([])
    assert args.max_price == 700
    assert args.pool == 400
    args0 = build_parser().parse_args(["--max-price", "0"])
    assert args0.max_price == 0


def test_pin_keep_puts_1815_first() -> None:
    rows = [
        {"code": "2330", "name": "台積電", "market": "tse", "amount": 9, "close": 1, "symbol": "2330.TW", "rank": 1},
        {"code": "1815", "name": "1815", "market": "otc", "amount": 8, "close": 1, "symbol": "1815.TWO", "rank": 2},
    ]
    out = pin_keep(rows, ["1815"])
    assert out[0]["code"] == "1815"
    assert out[0]["name"] == "富喬"
    assert out[0]["pinned"] is True
    missing = pin_keep([{"code": "2330", "name": "台積電"}], ["1815"])
    assert missing[0]["code"] == "1815"
    assert missing[0]["symbol"] == "1815.TWO"


def test_hit_payload_and_message() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 0.3
    hits = detect_retests(df)
    assert hits
    row = {"code": "1815", "name": "富喬", "symbol": "1815.TWO", "rank": 20, "amount": 8_000_000_00}
    hit = hit_from_row(df, hits[-1], row)
    assert hit.code == "1815"
    assert hit.ma240 > 0
    assert hit.key.startswith("1815.TWO:")
    text = format_hit(hit)
    assert "1815" in text
    assert "240MA" in text
    assert "五分" in text
    line = format_hit_line(hit)
    assert "1815" in line
    assert "240MA" in line


def test_market_session_hours() -> None:
    assert market_session(datetime(2026, 8, 28, 10, 0, tzinfo=TPE))
    assert market_session(datetime(2026, 8, 28, 13, 30, tzinfo=TPE))
    assert not market_session(datetime(2026, 8, 28, 13, 40, tzinfo=TPE))
    assert not market_session(datetime(2026, 8, 29, 10, 0, tzinfo=TPE))  # Saturday


def test_next_session_skips_weekend() -> None:
    friday_close = datetime(2026, 8, 28, 15, 0, tzinfo=TPE)
    nxt = next_session_open(friday_close)
    assert nxt.date().isoformat() == "2026-08-31"
    assert nxt.hour == 9


def test_session_dates_back() -> None:
    friday = datetime(2026, 8, 28, 16, 0, tzinfo=TPE)
    got = {d.isoformat() for d in session_dates_back(3, friday)}
    assert got == {"2026-08-26", "2026-08-27", "2026-08-28"}
    monday = datetime(2026, 8, 31, 10, 0, tzinfo=TPE)
    got2 = {d.isoformat() for d in session_dates_back(3, monday)}
    assert got2 == {"2026-08-31", "2026-08-28", "2026-08-27"}


def test_filter_hits_days() -> None:
    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 0.3
    row = {"code": "2330", "name": "台積電", "symbol": "2330.TW", "rank": 1, "amount": 1}
    old = hit_from_row(df, len(df) - 1, row)
    old.ts = pd.Timestamp("2026-08-20 11:00", tz=TPE)
    new = hit_from_row(df, len(df) - 1, row)
    new.ts = pd.Timestamp("2026-08-27 11:00", tz=TPE)
    kept = filter_hits_days([old, new], 3, datetime(2026, 8, 28, 16, 0, tzinfo=TPE))
    assert [h.ts.day for h in kept] == [27]


def test_wait_helpers() -> None:
    noon = datetime(2026, 8, 28, 10, 1, 0, tzinfo=TPE)
    wait = seconds_until_next_5m(noon, extra=8)
    assert 60 < wait < 5 * 60
    after = datetime(2026, 8, 28, 14, 0, tzinfo=TPE)
    wait2 = seconds_until_next_scan(after)
    assert wait2 > 3600


def test_write_html_report() -> None:
    from watch_tw_5m_ma240 import write_html_report, write_view_html

    df = _extended_then(last_close=101.2, last_low=100.4)
    ma = float(df["MA240"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Low")] = ma - 0.05
    df.iloc[-1, df.columns.get_loc("Close")] = ma + 0.25
    hits_i = detect_retests(df)
    assert hits_i
    row = {"code": "1815", "name": "富喬", "symbol": "1815.TWO", "rank": 20, "amount": 1}
    hit = hit_from_row(df, hits_i[-1], row)
    hit.ts = pd.Timestamp("2026-08-28 09:05", tz=TPE)
    hit._df = df  # type: ignore[attr-defined]
    out_dir = Path("/tmp/tw-ma240-html-test")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_html_report(
        out_dir / "index.html",
        [hit],
        [row],
        "近10個交易日",
        "20260828",
        chart_mode="pierce",
        max_price=700,
    )
    text = path.read_text(encoding="utf-8")
    assert "1815" in text
    assert "富喬" in text
    assert "回測 240MA" in text
    assert "股價≤700" in text
    assert "data:image/png;base64," in text
    assert "src='img/" not in text
    view = write_view_html(path)
    assert "data:image/png;base64," in view.read_text(encoding="utf-8")
    assert not (out_dir / "img").exists()


def main() -> int:
    test_tw_tick()
    test_touch_band_uses_ticks()
    test_detect_pullback_to_ma240()
    test_hug_ma_does_not_fire()
    test_close_breakdown_is_not_retest()
    test_only_first_touch_fires()
    test_shallow_dip_not_retest()
    test_cooldown_skips_nearby_retest()
    test_skip_open_minutes()
    test_touch_then_bounce_still_counts()
    test_pin_keep_puts_1815_first()
    test_select_universe_drops_price_over_700()
    test_cli_max_price_default()
    test_hit_payload_and_message()
    test_market_session_hours()
    test_next_session_skips_weekend()
    test_wait_helpers()
    test_session_dates_back()
    test_filter_hits_days()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
