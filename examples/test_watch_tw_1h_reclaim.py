#!/usr/bin/env python3
"""Session gating, completed-bar, and durable dedup tests for the TW 1h watcher."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_tw_1h_reclaim import ohlc_from_close, valid_closes  # noqa: E402
from tw_1h_reclaim import TPE, detect_signals, loose_params  # noqa: E402
from watch_tw_1h_reclaim import (  # noqa: E402
    bar_close_time,
    drop_incomplete_1h_bars,
    format_alert,
    in_regular_session,
    is_completed_1h_bar,
    load_seen,
    next_eval_at,
    prepare_live_bars,
    save_seen,
    scan_once,
    scan_prepared,
    select_alerts,
    should_scan,
    signal_key,
    to_tpe,
)


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TPE)


def test_session_gating() -> None:
    assert in_regular_session(ts("2026-10-02T09:00:00")) is True
    assert in_regular_session(ts("2026-10-02T13:30:00")) is True
    assert in_regular_session(ts("2026-10-02T13:30:59")) is True
    assert in_regular_session(ts("2026-10-02T08:59:59")) is False  # 盤前試撮
    assert in_regular_session(ts("2026-10-02T08:30:00")) is False
    assert in_regular_session(ts("2026-10-02T13:31:00")) is False
    assert in_regular_session(ts("2026-10-03T10:00:00")) is False  # Saturday
    assert in_regular_session(ts("2026-10-04T10:00:00")) is False  # Sunday
    assert should_scan(ts("2026-10-02T09:30:00")) is False  # 09:00 根還沒收
    assert should_scan(ts("2026-10-02T10:00:00")) is True
    assert should_scan(ts("2026-10-02T13:30:00")) is True
    assert should_scan(ts("2026-10-02T13:31:00")) is False
    assert should_scan(ts("2026-10-03T11:00:00")) is False


def test_completed_1h_bar_times() -> None:
    open_0900 = ts("2026-10-02T09:00:00")
    open_1300 = ts("2026-10-02T13:00:00")
    assert bar_close_time(open_0900) == pd.Timestamp(ts("2026-10-02T10:00:00"))
    assert bar_close_time(open_1300) == pd.Timestamp(ts("2026-10-02T13:30:00"))
    assert is_completed_1h_bar(open_0900, ts("2026-10-02T09:59:59")) is False
    assert is_completed_1h_bar(open_0900, ts("2026-10-02T10:00:00")) is True
    assert is_completed_1h_bar(open_1300, ts("2026-10-02T13:29:59")) is False
    assert is_completed_1h_bar(open_1300, ts("2026-10-02T13:30:00")) is True
    # 13:00 根不能等到 14:00 才算收完（那時已經離開盤中）
    assert is_completed_1h_bar(open_1300, ts("2026-10-02T13:30:00")) is True
    assert bar_close_time(open_1300).hour == 13
    assert bar_close_time(open_1300).minute == 30


def test_drop_incomplete_bars() -> None:
    idx = pd.DatetimeIndex(
        [
            ts("2026-10-02T09:00:00"),
            ts("2026-10-02T10:00:00"),
            ts("2026-10-02T11:00:00"),
            ts("2026-10-02T12:00:00"),
            ts("2026-10-02T13:00:00"),
        ]
    )
    df = pd.DataFrame({"Close": range(5)}, index=idx)
    at_1030 = drop_incomplete_1h_bars(df, ts("2026-10-02T10:30:00"))
    assert list(at_1030.index) == [idx[0]]
    at_1330 = drop_incomplete_1h_bars(df, ts("2026-10-02T13:30:00"))
    assert len(at_1330) == 5
    at_1315 = drop_incomplete_1h_bars(df, ts("2026-10-02T13:15:00"))
    assert list(at_1315.index) == list(idx[:4])


def test_preopen_bars_stripped() -> None:
    idx = pd.DatetimeIndex(
        [
            ts("2026-10-02T08:00:00"),
            ts("2026-10-02T08:30:00"),
            ts("2026-10-02T09:00:00"),
            ts("2026-10-02T10:00:00"),
        ]
    )
    df = pd.DataFrame(
        {"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1},
        index=idx,
    )
    live = prepare_live_bars(df, ts("2026-10-02T10:30:00"))
    assert list(live.index) == [idx[2]]  # only completed 09:00; 08:xx gone, 10:00 still open


def test_in_progress_entry_cannot_fire() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    forming = to_tpe(df.index[sigs[0].entry_idx])
    now = (forming + timedelta(minutes=20)).to_pydatetime()
    assert is_completed_1h_bar(forming, now) is False
    live = prepare_live_bars(df, now)
    assert forming not in set(to_tpe(ts) for ts in live.index)
    live_sigs = detect_signals(live, loose_params())
    live_keys = {signal_key("6443.TW", live.index[s.entry_idx], s.entry_price) for s in live_sigs}
    forming_key = signal_key("6443.TW", forming, sigs[0].entry_price)
    assert forming_key not in live_keys
    row = {"code": "6443", "name": "元晶", "symbol": "6443.TW"}
    scanned = scan_prepared(row, df, now, set(), loose_params())
    assert scanned.error == ""
    assert all(a.key != forming_key for a in scanned.alerts)
    assert [a for a in scanned.alerts if a.send] == []


def test_completed_entry_can_fire() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    entry_ts = to_tpe(df.index[sigs[0].entry_idx])
    now = bar_close_time(df.index[-1]).to_pydatetime()
    row = {"code": "6443", "name": "元晶", "symbol": "6443.TW"}
    scanned = scan_prepared(row, df, now, set(), loose_params())
    sending = [a for a in scanned.alerts if a.send]
    assert scanned.error == ""
    assert len(sending) == 1
    assert sending[0].key == signal_key("6443.TW", entry_ts, sigs[0].entry_price)


def test_select_alerts_skips_other_days() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    entry_ts = to_tpe(df.index[sigs[0].entry_idx])
    later = entry_ts + timedelta(days=3)
    # later is a Monday+3; force a weekday noon so session helpers are unused here
    now = datetime(later.year, later.month, later.day, 11, 0, tzinfo=TPE)
    selected = select_alerts("6443.TW", df, sigs, now, set())
    assert len(selected) == 1
    assert selected[0].send is False
    assert selected[0].key == signal_key("6443.TW", entry_ts, sigs[0].entry_price)


def test_dedup_persists_across_restart(tmp_path: Path) -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    entry_ts = to_tpe(df.index[sigs[0].entry_idx])
    now = bar_close_time(entry_ts).to_pydatetime()
    row = {"code": "6443", "name": "元晶", "symbol": "6443.TW"}
    state_path = tmp_path / "tw_1h_reclaim_alert_state.json"
    seen_order: list[str] = []
    seen_keys: set[str] = set()
    sent_texts: list[str] = []

    orig_tg = __import__("watch_tw_1h_reclaim", fromlist=["tg_send"]).tg_send

    def capture(token, chat_id, text, dry_run=False):
        sent_texts.append(text)
        return True

    import watch_tw_1h_reclaim as w

    w.tg_send = capture  # type: ignore[method-assign]
    try:
        scanned, errors, sent = scan_once(
            token="",
            chat_id="",
            now=now,
            universe=[row],
            seen_order=seen_order,
            seen_keys=seen_keys,
            params=loose_params(),
            range_="2mo",
            dry_run=True,
            workers=1,
            sleep_s=0,
            fetch_symbol=lambda _row: df,
        )
        assert scanned == 1 and errors == 0 and sent == 1
        assert len(sent_texts) == 1
        save_seen(state_path, {"alerted": seen_order, "initialized": True})
        assert state_path.exists()
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["alerted"]

        loaded = load_seen(state_path)
        seen_order2 = list(loaded["alerted"])
        seen_keys2 = set(seen_order2)
        sent_texts.clear()
        scanned2, errors2, sent2 = scan_once(
            token="",
            chat_id="",
            now=now,
            universe=[row],
            seen_order=seen_order2,
            seen_keys=seen_keys2,
            params=loose_params(),
            range_="2mo",
            dry_run=True,
            workers=1,
            sleep_s=0,
            fetch_symbol=lambda _row: df,
        )
        assert scanned2 == 1 and errors2 == 0 and sent2 == 0
        assert sent_texts == []
        assert seen_keys2 == set(payload["alerted"])
    finally:
        w.tg_send = orig_tg  # type: ignore[method-assign]


def test_failed_send_not_marked_seen() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    now = bar_close_time(df.index[sigs[0].entry_idx]).to_pydatetime()
    row = {"code": "6443", "name": "元晶", "symbol": "6443.TW"}
    seen_order: list[str] = []
    seen_keys: set[str] = set()
    import watch_tw_1h_reclaim as w

    orig = w.tg_send
    w.tg_send = lambda *a, **k: False  # type: ignore[method-assign]
    try:
        _scanned, _errors, sent = scan_once(
            token="x",
            chat_id="1",
            now=now,
            universe=[row],
            seen_order=seen_order,
            seen_keys=seen_keys,
            params=loose_params(),
            range_="2mo",
            dry_run=False,
            workers=1,
            sleep_s=0,
            fetch_symbol=lambda _row: df,
        )
        assert sent == 0
        assert seen_keys == set()
        assert seen_order == []
    finally:
        w.tg_send = orig  # type: ignore[method-assign]


def test_format_is_alert_only() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    text = format_alert({"code": "6443", "name": "元晶", "symbol": "6443.TW"}, df, sigs[0])
    assert "只通知、不下單" in text
    assert "6443" in text
    assert "shioaji" not in text.lower()
    assert "下單" in text


def test_next_eval_skips_weekend() -> None:
    friday_after = ts("2026-10-02T13:31:00")  # Friday
    nxt = next_eval_at(friday_after, grace_seconds=20)
    assert nxt.tzinfo is not None
    assert nxt.weekday() == 0  # Monday
    assert (nxt.hour, nxt.minute, nxt.second) == (10, 0, 20)
    during = next_eval_at(ts("2026-10-02T10:05:00"), grace_seconds=20)
    assert during == ts("2026-10-02T11:00:20")


def test_load_seen_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    data = load_seen(path)
    assert data["alerted"] == []
    assert data["initialized"] is False


def main() -> int:
    from tempfile import TemporaryDirectory

    test_session_gating()
    test_completed_1h_bar_times()
    test_drop_incomplete_bars()
    test_preopen_bars_stripped()
    test_in_progress_entry_cannot_fire()
    test_completed_entry_can_fire()
    test_select_alerts_skips_other_days()
    with TemporaryDirectory() as d:
        test_dedup_persists_across_restart(Path(d))
        test_load_seen_corrupt(Path(d))
    test_failed_send_not_marked_seen()
    test_format_is_alert_only()
    test_next_eval_skips_weekend()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
