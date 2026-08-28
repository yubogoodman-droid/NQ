#!/usr/bin/env python3
"""Synthetic tests for 台股一小時K 站上 MA60 (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_tw_1h_ma60 import (  # noqa: E402
    TPE,
    add_mas,
    completed_bar_count,
    detect_stand_above,
    drop_close_print,
    filter_hits_days,
    format_hit,
    format_hit_line,
    hit_from_row,
    hour_bar_close,
    market_session,
    next_session_open,
    pin_keep,
    seconds_until_next_1h,
    seconds_until_next_scan,
    session_dates_back,
)


def _bars(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2026-06-01 09:00", periods=n, freq="h", tz=TPE)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_detect_stand_above_cross() -> None:
    close = np.full(80, 100.0)
    close[-1] = 102.0
    df = add_mas(_bars(close))
    hits = detect_stand_above(df)
    assert hits == [len(df) - 1], hits
    row = {"code": "1815", "name": "富喬", "symbol": "1815.TWO", "rank": 1, "amount": 1}
    hit = hit_from_row(df, hits[-1], row)
    assert hit.close == 102.0
    assert hit.prev_close == 100.0
    assert hit.dist_pct > 0
    assert hit.prev_dist_pct <= 0
    text = format_hit(hit)
    assert "1815" in text and "60MA" in text and "一小時" in text
    assert "1815" in format_hit_line(hit)


def test_already_above_does_not_fire() -> None:
    close = np.full(80, 102.0)
    df = add_mas(_bars(close))
    assert detect_stand_above(df) == []


def test_only_first_cross_fires() -> None:
    close = np.full(90, 100.0)
    close[70:] = 105.0
    df = add_mas(_bars(close))
    hits = detect_stand_above(df)
    assert hits == [70], hits


def test_close_must_be_above_ma() -> None:
    close = np.full(80, 100.0)
    close[-1] = 99.5
    df = add_mas(_bars(close))
    assert detect_stand_above(df) == []


def test_drop_1330_print() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-28 13:00", tz=TPE),
            pd.Timestamp("2026-08-28 13:30", tz=TPE),
        ]
    )
    df = pd.DataFrame(
        {"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [1, 1], "Volume": [10, 0]},
        index=idx,
    )
    out = drop_close_print(df)
    assert list(out.index) == [idx[0]]


def test_hour_bar_close_and_completed() -> None:
    ts = pd.Timestamp("2026-08-28 10:00", tz=TPE)
    assert hour_bar_close(ts) == pd.Timestamp("2026-08-28 11:00", tz=TPE)
    ts13 = pd.Timestamp("2026-08-28 13:00", tz=TPE)
    assert hour_bar_close(ts13) == pd.Timestamp("2026-08-28 13:30", tz=TPE)
    close = np.full(70, 100.0)
    df = _bars(close)
    df.index = pd.DatetimeIndex(
        [pd.Timestamp("2026-08-28 09:00", tz=TPE) + pd.Timedelta(hours=i) for i in range(5)]
        + list(df.index[5:])
    )
    df = df.iloc[:5]
    # 10:30 → 09:00 bar is closed, 10:00 still forming
    n = completed_bar_count(df, datetime(2026, 8, 28, 10, 30, tzinfo=TPE))
    assert n == 1, n
    n2 = completed_bar_count(df, datetime(2026, 8, 28, 14, 0, tzinfo=TPE))
    assert n2 == 5


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


def test_select_universe_drops_price_over_700() -> None:
    from watch_tw_1h_ma60 import select_universe

    rows = [
        {"code": "2330", "name": "台積電", "close": 1400.0, "amount": 9, "rank": 1, "market": "tse", "symbol": "2330.TW"},
        {"code": "1815", "name": "1815", "close": 125.0, "amount": 8, "rank": 2, "market": "otc", "symbol": "1815.TWO"},
        {"code": "2303", "name": "聯電", "close": 55.0, "amount": 7, "rank": 3, "market": "tse", "symbol": "2303.TW"},
        {"code": "2454", "name": "聯發科", "close": 4100.0, "amount": 6, "rank": 4, "market": "tse", "symbol": "2454.TW"},
    ]
    kept, dropped = select_universe(rows, limit=200, keep=["1815"], max_price=700)
    assert [r["code"] for r in kept] == ["1815", "2303"]
    assert {r["code"] for r in dropped} == {"2330", "2454"}


def test_cli_defaults() -> None:
    from watch_tw_1h_ma60 import build_parser

    args = build_parser().parse_args([])
    assert args.max_price == 700
    assert args.ma == 60
    assert args.range == "3mo"


def test_market_session_hours() -> None:
    assert market_session(datetime(2026, 8, 28, 10, 0, tzinfo=TPE))
    assert market_session(datetime(2026, 8, 28, 13, 30, tzinfo=TPE))
    assert not market_session(datetime(2026, 8, 28, 13, 40, tzinfo=TPE))
    assert not market_session(datetime(2026, 8, 29, 10, 0, tzinfo=TPE))


def test_next_session_skips_weekend() -> None:
    friday_close = datetime(2026, 8, 28, 15, 0, tzinfo=TPE)
    nxt = next_session_open(friday_close)
    assert nxt.date().isoformat() == "2026-08-31"
    assert nxt.hour == 9


def test_wait_helpers() -> None:
    wait = seconds_until_next_1h(datetime(2026, 8, 28, 10, 1, 0, tzinfo=TPE), extra=8)
    assert 50 * 60 < wait < 60 * 60
    after = datetime(2026, 8, 28, 14, 0, tzinfo=TPE)
    wait2 = seconds_until_next_scan(after)
    assert wait2 > 3600


def test_filter_hits_days() -> None:
    close = np.full(80, 100.0)
    close[-1] = 102.0
    df = add_mas(_bars(close))
    row = {"code": "2330", "name": "台積電", "symbol": "2330.TW", "rank": 1, "amount": 1}
    old = hit_from_row(df, len(df) - 1, row)
    old.ts = pd.Timestamp("2026-08-20 11:00", tz=TPE)
    new = hit_from_row(df, len(df) - 1, row)
    new.ts = pd.Timestamp("2026-08-27 11:00", tz=TPE)
    kept = filter_hits_days([old, new], 3, datetime(2026, 8, 28, 16, 0, tzinfo=TPE))
    assert [h.ts.day for h in kept] == [27]


def test_session_dates_back() -> None:
    friday = datetime(2026, 8, 28, 16, 0, tzinfo=TPE)
    got = {d.isoformat() for d in session_dates_back(3, friday)}
    assert got == {"2026-08-26", "2026-08-27", "2026-08-28"}


def test_write_html_report() -> None:
    from watch_tw_1h_ma60 import write_html_report

    close = np.full(80, 100.0)
    close[-1] = 102.0
    df = add_mas(_bars(close))
    hits_i = detect_stand_above(df)
    row = {"code": "1815", "name": "富喬", "symbol": "1815.TWO", "rank": 20, "amount": 1}
    hit = hit_from_row(df, hits_i[-1], row)
    hit.ts = pd.Timestamp("2026-08-28 10:00", tz=TPE)
    hit._df = df  # type: ignore[attr-defined]
    out_dir = Path("/tmp/tw-1h-ma60-html-test")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_html_report(
        out_dir / "index.html",
        [hit],
        [row],
        "近10個交易日",
        "20260828",
        chart_mode="all",
        max_price=700,
    )
    text = path.read_text(encoding="utf-8")
    assert "1815" in text
    assert "站上" in text
    assert "60MA" in text
    assert "data:image/png;base64," in text
    assert "src='img/" not in text
    assert (out_dir / "index.html").exists()


def main() -> int:
    test_detect_stand_above_cross()
    test_already_above_does_not_fire()
    test_only_first_cross_fires()
    test_close_must_be_above_ma()
    test_drop_1330_print()
    test_hour_bar_close_and_completed()
    test_pin_keep_puts_1815_first()
    test_select_universe_drops_price_over_700()
    test_cli_defaults()
    test_market_session_hours()
    test_next_session_skips_weekend()
    test_wait_helpers()
    test_filter_hits_days()
    test_session_dates_back()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
