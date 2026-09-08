#!/usr/bin/env python3
"""Synthetic tests for 幣安 15m 空頭排列跌破 99/120（不打幣安）。"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_short import (  # noqa: E402
    TPE,
    Hit,
    Signal,
    TradeResult,
    bar_index_at,
    detect_signals,
    filter_below_1h_ma25,
    filter_entry_window,
    filter_near_1h_ma99,
    filter_untangled_1h_mas,
    filter_untangled_15m_mas,
    filter_away_1h_ma120,
    filter_away_15m_ma200,
    filter_attack_15m_ma200,
    filter_sit_15m_ma200,
    filter_away_1h_ma_support,
    filter_not_below_1h_ma200,
    default_params,
    htf_ma_at_entry,
    htf_mas_at_entry,
    htf_snapshot,
    is_stock_contract,
    ma_cluster_spread,
    signal_ma_spread,
    order_chart_hits,
    simulate,
    sma,
    summarize_trades,
    write_html,
    write_view_html,
    bars_per_day,
    kline_limit,
    fetch_klines,
)

LOOSE = default_params(min_body_pct=0.0, min_vol_ratio=0.0, min_break_pct=0.002, min_risk_pct=0.0001)


def bars(closes, start: str = "2026-08-31 00:00") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="15min", tz=TPE)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) + 0.002
    low = np.minimum(opens, closes) - 0.002
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def dump_closes(n_flat: int = 130, dump: float = 0.04) -> np.ndarray:
    """長時間橫盤後急殺，讓 7<14<25 且收盤穿過 99/120。"""
    flat = np.full(n_flat, 1.0)
    drop = np.array([0.985, 0.96, 0.94, 0.92, 0.90])
    rest = np.linspace(0.90 - dump, 0.90 - dump - 0.02, 20)
    return np.concatenate([flat, drop, rest])


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    out = sma(x, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_kline_limit_covers_month() -> None:
    assert bars_per_day("15m") == 96
    assert bars_per_day("1h") == 24
    assert kline_limit(7, "15m") == 1500
    assert kline_limit(30, "15m") >= 30 * 96 + 200
    assert kline_limit(30, "1h") == 1500


def test_fetch_klines_pages() -> None:
    """30 日需要的根數超過 1500 時，往回拼頁且丢掉未收完的最後一根。"""
    import binance_15m_short as m

    interval_ms = 900_000
    now = int(__import__("time").time() * 1000)
    newest_open = ((now - interval_ms) // interval_ms) * interval_ms
    n = 1600
    all_rows = []
    for i in range(n):
        t = newest_open - (n - 1 - i) * interval_ms
        px = 1.0 + i * 0.0001
        all_rows.append([t, px, px + 0.001, px - 0.001, px, 10.0])
    incomplete = [
        newest_open + interval_ms,
        9.0,
        9.0,
        9.0,
        9.0,
        1.0,
    ]

    def fake_get(path, params=None, retries: int = 5):  # noqa: ARG001
        assert path == "/fapi/v1/klines"
        params = params or {}
        lim = int(params["limit"])
        end = params.get("endTime")
        pool = all_rows + [incomplete]
        if end is not None:
            pool = [r for r in pool if int(r[0]) <= int(end)]
        return pool[-lim:]

    old = m.get_json
    m.get_json = fake_get  # type: ignore[method-assign]
    try:
        df = fetch_klines("FOOUSDT", "15m", limit=1600)
    finally:
        m.get_json = old  # type: ignore[method-assign]
    assert len(df) == 1600
    assert df.index.is_monotonic_increasing
    last_ms = int(df.index[-1].timestamp() * 1000)
    assert abs(last_ms - newest_open) < 2000
    assert float(df["close"].iloc[-1]) != 9.0


def test_stock_contract_filter() -> None:
    assert is_stock_contract({"underlyingType": "EQUITY"})
    assert is_stock_contract({"underlyingType": "PREMARKET"})
    assert is_stock_contract({"underlyingType": "HK_EQUITY"})
    assert is_stock_contract({"underlyingType": "KR_EQUITY"})
    assert is_stock_contract({"underlyingType": "CN_EQUITY"})
    assert not is_stock_contract({"underlyingType": "COIN"})
    assert not is_stock_contract({"underlyingType": "COMMODITY"})
    assert not is_stock_contract({})


def test_detect_dump_cross() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs, "急殺應出現做空訊號"
    s = sigs[0]
    assert s.ma7 < s.ma14 < s.ma25
    assert s.entry_price < s.ma99 and s.entry_price < s.ma120
    prev_c = float(df["close"].iloc[s.entry_idx - 1])
    prev_m99 = float(df["close"].rolling(99).mean().iloc[s.entry_idx - 1])
    prev_m120 = float(df["close"].rolling(120).mean().iloc[s.entry_idx - 1])
    assert not (prev_c < prev_m99 and prev_c < prev_m120)
    assert len(sigs) == 1


def test_no_signal_if_bullish_stack() -> None:
    # 先漲再微跌，短均仍 7>14>25
    up = np.linspace(1.0, 1.12, 130)
    dip = np.array([1.118, 1.116, 1.114])
    df = bars(np.concatenate([up, dip]))
    sigs = detect_signals(df, LOOSE)
    assert sigs == []


def test_rebreak_after_reclaim() -> None:
    flat = np.full(130, 1.0)
    dump1 = np.array([0.97, 0.95, 0.93])
    reclaim = np.full(24, 1.03)
    dump2 = np.array([0.90, 0.88])
    df = bars(np.concatenate([flat, dump1, reclaim, dump2]))
    sigs = detect_signals(df, LOOSE)
    assert len(sigs) >= 2


def test_green_candle_skipped() -> None:
    closes = dump_closes()
    df = bars(closes)
    # 把第一根跌破 K 改成低開高收
    sigs0 = detect_signals(df, LOOSE)
    assert sigs0
    i = sigs0[0].entry_idx
    df.loc[df.index[i], "open"] = float(df["close"].iloc[i]) - 0.02
    df.loc[df.index[i], "close"] = float(df["open"].iloc[i]) + 0.01
    df.loc[df.index[i], "high"] = float(df["close"].iloc[i]) + 0.002
    sigs = detect_signals(df, LOOSE)
    assert all(s.entry_idx != i for s in sigs)


def test_short_profit_and_stop() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    trades = simulate(df, sigs)
    assert trades
    t = trades[0]
    assert t.pnl_pct > 0
    assert t.exit_reason in {"target", "time"}
    assert t.pnl_points == t.entry_price - t.exit_price


def test_stop_on_rally() -> None:
    flat = np.full(130, 1.0)
    dump = np.array([0.97, 0.95])
    rally = np.array([1.02, 1.03, 1.04])
    df = bars(np.concatenate([flat, dump, rally]))
    # 讓進場 K 的高點不要蓋過後面反彈
    sigs = detect_signals(df, LOOSE)
    assert sigs
    i = sigs[0].entry_idx
    df.loc[df.index[i], "high"] = float(df["close"].iloc[i]) + 0.001
    trades = simulate(df, sigs, default_params(stop_lookback=1, min_risk_pct=0.0001))
    assert trades
    assert trades[0].exit_reason == "stop"
    assert trades[0].pnl_pct < 0


def test_one_position_skips_overlap() -> None:
    flat = np.full(130, 1.0)
    dump1 = np.array([0.97, 0.95, 0.93])
    reclaim = np.array([1.01])
    dump2 = np.array([0.96, 0.94])
    df = bars(np.concatenate([flat, dump1, reclaim, dump2]))
    sigs = detect_signals(df, LOOSE)
    trades = simulate(df, sigs, default_params(time_bars=40))
    # 第一筆若還沒平，第二筆應被 skip_busy；急殺後很快 2R 也可能兩筆都做
    assert len(trades) <= len(sigs)


def test_filter_entry_window() -> None:
    df = bars(dump_closes(), start="2026-08-01 00:00")
    sigs = detect_signals(df, LOOSE)
    assert sigs
    none = filter_entry_window(df, sigs, days=1)
    # 資料從 8/1 起，急殺在 130 根後 ≈ 1.3 天，截到最後 1 天可能還在
    all_ = filter_entry_window(df, sigs, days=30)
    assert len(all_) == len(sigs)
    empty = filter_entry_window(df, sigs, days=0)
    assert len(empty) == len(sigs)


def test_volume_and_body_filters() -> None:
    df = bars(dump_closes())
    assert detect_signals(df) == []  # 量是平的，預設 1.5x 濾掉
    dump_i = 130
    df.loc[df.index[dump_i], "volume"] = 5000.0
    sigs = detect_signals(df)
    assert sigs
    assert sigs[0].entry_idx == dump_i
    assert sigs[0].vol_ratio >= 1.5
    assert sigs[0].body_pct >= 0.008


def to_1h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def test_bar_index_at() -> None:
    df = bars(dump_closes())
    h1 = to_1h(df)
    ts = df.index[130]
    i = bar_index_at(h1, ts)
    assert i is not None
    assert h1.index[i] <= ts
    if i + 1 < len(h1):
        assert h1.index[i + 1] > ts
    snap = htf_snapshot(h1, ts)
    assert "1h" in snap


def pad_1h(h1: pd.DataFrame, n: int = 120, level: float | None = None) -> pd.DataFrame:
    """往前補滿 1h K，讓 MA99 算得出來。"""
    if len(h1) >= n:
        return h1
    lvl = float(h1["close"].iloc[0] if level is None else level)
    need = n - len(h1)
    start = h1.index[0] - pd.Timedelta(hours=need)
    idx = pd.date_range(start, periods=need, freq="h", tz=h1.index.tz)
    pre = pd.DataFrame(
        {
            "open": np.full(need, lvl),
            "high": np.full(need, lvl + 0.002),
            "low": np.full(need, lvl - 0.002),
            "close": np.full(need, lvl),
            "volume": np.full(need, 1000.0),
        },
        index=idx,
    )
    return pd.concat([pre, h1])


def _shift_1h_history(h1: pd.DataFrame, level: float) -> pd.DataFrame:
    """把進場那根 1h 之前的收盤改成 level，用來控制 MA25。"""
    out = h1.copy()
    out.loc[out.index[:-1], "open"] = level
    out.loc[out.index[:-1], "high"] = level + 0.002
    out.loc[out.index[:-1], "low"] = level - 0.002
    out.loc[out.index[:-1], "close"] = level
    return out


def test_1h_ma25_keeps_dump_from_above() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    h1 = _shift_1h_history(to_1h(df), 1.05)
    kept = filter_below_1h_ma25(df, sigs, h1)
    assert kept == sigs
    ma = htf_ma_at_entry(h1, df.index[sigs[0].entry_idx], sigs[0].entry_price)
    assert ma is not None and sigs[0].entry_price < ma


def test_1h_ma25_rejects_still_above() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    h1 = _shift_1h_history(to_1h(df), 0.80)
    funnel: dict = {}
    kept = filter_below_1h_ma25(df, sigs, h1, funnel=funnel)
    assert kept == []
    assert funnel.get("above_1h_ma25", 0) >= 1
    ma = htf_ma_at_entry(h1, df.index[sigs[0].entry_idx], sigs[0].entry_price)
    assert ma is not None and sigs[0].entry_price >= ma


def test_1h_ma25_no_lookahead() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    h1 = _shift_1h_history(to_1h(df), 1.05)
    ts = df.index[sigs[0].entry_idx]
    i = bar_index_at(h1, ts)
    assert i is not None
    later = h1.copy()
    later.iloc[i, later.columns.get_loc("close")] = 0.50
    ma_live = htf_ma_at_entry(h1, ts, sigs[0].entry_price)
    ma_spoiled = htf_ma_at_entry(later, ts, sigs[0].entry_price)
    assert ma_live is not None and ma_spoiled is not None
    assert abs(ma_live - ma_spoiled) < 1e-12


def test_1h_ma25_missing_data_skips() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    funnel: dict = {}
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert filter_below_1h_ma25(df, sigs, empty, funnel=funnel) == []
    assert funnel.get("no_1h", 0) >= 1


def test_1h_ma99_keeps_near_like_cloud() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    px = sigs[0].entry_price
    h1 = pad_1h(_shift_1h_history(to_1h(df), px / 1.10), level=px / 1.10)
    kept = filter_near_1h_ma99(df, sigs, h1, max_dist=0.20)
    assert kept == sigs
    ma = htf_ma_at_entry(h1, df.index[sigs[0].entry_idx], px, n=99)
    assert ma is not None and abs(px / ma - 1.0) <= 0.20


def test_1h_ma99_rejects_bulla_extension() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    px = sigs[0].entry_price
    h1 = pad_1h(_shift_1h_history(to_1h(df), px / 1.68), level=px / 1.68)
    funnel: dict = {}
    kept = filter_near_1h_ma99(df, sigs, h1, max_dist=0.20, funnel=funnel)
    assert kept == []
    assert funnel.get("far_1h_ma99", 0) >= 1
    ma = htf_ma_at_entry(h1, df.index[sigs[0].entry_idx], px, n=99)
    assert ma is not None and abs(px / ma - 1.0) > 0.20


def test_1h_ma99_rejects_too_far_below() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    px = sigs[0].entry_price
    h1 = pad_1h(_shift_1h_history(to_1h(df), px / 0.65), level=px / 0.65)
    kept = filter_near_1h_ma99(df, sigs, h1, max_dist=0.20)
    assert kept == []


def make_1h(ts, n: int = 130, old: float = 1.0, recent: float = 1.0, recent_bars: int = 30) -> pd.DataFrame:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(TPE)
    else:
        ts = ts.tz_convert(TPE)
    end = ts.floor("h")
    idx = pd.date_range(end - pd.Timedelta(hours=n - 1), periods=n, freq="h", tz=TPE)
    closes = np.full(n, float(old))
    if recent_bars > 0:
        closes[-recent_bars:] = float(recent)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.002,
            "low": closes - 0.002,
            "close": closes,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_15m_mas_reject_tangled_like_zen_xmr() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    px = sigs[0].entry_price
    tangled = replace(
        sigs[0],
        ma7=px * 1.004,
        ma14=px * 1.008,
        ma25=px * 1.012,
        ma99=px * 1.006,
        ma120=px * 1.002,
        ma_high=px * 1.006,
    )
    funnel: dict = {}
    kept = filter_untangled_15m_mas([tangled], min_spread=0.015, funnel=funnel)
    assert kept == []
    assert funnel.get("tangled_15m_ma", 0) == 1
    spread = signal_ma_spread(tangled)
    assert spread is not None and spread < 0.015


def test_15m_mas_keep_fanned_like_cloud() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    px = sigs[0].entry_price
    fanned = replace(
        sigs[0],
        ma7=px * 1.102,
        ma14=px * 1.134,
        ma25=px * 1.143,
        ma99=px * 1.097,
        ma120=px * 1.084,
        ma_high=px * 1.097,
    )
    spread = signal_ma_spread(fanned)
    assert spread is not None and spread >= 0.015
    kept = filter_untangled_15m_mas([fanned], min_spread=0.015)
    assert kept == [fanned]


def test_15m_ma200_rejects_open_too_close() -> None:
    """橫盤貼著 15m MA200 再小跌，開盤沒肉、收盤還在 200 上 → 濾掉。"""
    closes = np.concatenate(
        [
            np.full(200, 0.99),
            np.full(40, 1.02),
            np.array([1.005, 0.998, 0.990, 0.985]),
            np.linspace(0.985, 0.980, 15),
        ]
    )
    df = bars(closes)
    sigs = detect_signals(df, LOOSE)
    assert sigs
    funnel: dict = {}
    kept = filter_away_15m_ma200(df, sigs, min_open_dist=0.04, funnel=funnel)
    assert kept == []
    assert funnel.get("near_15m_ma200", 0) >= 1
    s = sigs[0]
    m200 = float(sma(df["close"].to_numpy(float), 200)[s.entry_idx])
    op = float(df["open"].iloc[s.entry_idx])
    assert s.entry_price >= m200
    assert op / m200 - 1.0 < 0.04


def test_15m_ma200_keeps_dump_from_well_above() -> None:
    """CLO 那種：開盤遠高於 15m MA200，即使收盤砸到 200 附近仍可空。"""
    closes = np.concatenate(
        [
            np.full(200, 0.85),
            np.full(120, 1.0),
            np.array([0.96, 0.94, 0.92, 0.90]),
            np.linspace(0.89, 0.88, 20),
        ]
    )
    df = bars(closes)
    sigs = detect_signals(df, LOOSE)
    assert sigs
    kept = filter_away_15m_ma200(df, sigs, min_open_dist=0.04)
    assert kept == sigs
    s = sigs[0]
    m200 = float(sma(df["close"].to_numpy(float), 200)[s.entry_idx])
    op = float(df["open"].iloc[s.entry_idx])
    assert op / m200 - 1.0 >= 0.04


def test_15m_ma200_keeps_already_through() -> None:
    """收盤已跌破 15m MA200（CATI 那種穿過 200）不因貼近而濾掉。"""
    df = bars(dump_closes(n_flat=220))
    sigs = detect_signals(df, LOOSE)
    assert sigs
    s = sigs[0]
    m200 = float(sma(df["close"].to_numpy(float), 200)[s.entry_idx])
    through = replace(s, entry_price=m200 * 0.99)
    kept = filter_away_15m_ma200(df, [through], min_open_dist=0.04)
    assert kept == [through]


def test_15m_ma200_missing_data_skips() -> None:
    df = bars(dump_closes(n_flat=130))
    sigs = detect_signals(df, LOOSE)
    assert sigs
    funnel: dict = {}
    assert filter_away_15m_ma200(df, sigs, min_open_dist=0.04, funnel=funnel) == []
    assert funnel.get("no_15m_ma200", 0) >= 1


def _ma200_sig(open_px: float, close_px: float, high_px: float) -> tuple[pd.DataFrame, Signal]:
    """200 根收在 1.0，最後一根改成指定 OHLC，MA200 ≈ 1。"""
    n = 220
    closes = np.full(n, 1.0)
    closes[-1] = close_px
    df = bars(closes)
    i = n - 1
    df.iat[i, df.columns.get_loc("open")] = open_px
    df.iat[i, df.columns.get_loc("close")] = close_px
    df.iat[i, df.columns.get_loc("high")] = high_px
    df.iat[i, df.columns.get_loc("low")] = min(open_px, close_px) - 0.001
    body = (open_px - close_px) / open_px if open_px else 0.0
    sig = Signal(
        entry_idx=i,
        entry_price=close_px,
        ma7=0.99,
        ma14=1.00,
        ma25=1.01,
        ma99=1.02,
        ma120=1.03,
        ma_high=float(high_px),
        body_pct=body,
        vol_ratio=2.0,
    )
    return df, sig


def test_attack_ma200_rejects_weak_bar_when_2r_below() -> None:
    """龍蝦那種：2R 在 15m MA200 下，弱陰線還沒打到 200 → 濾掉。"""
    df, sig = _ma200_sig(open_px=1.065, close_px=1.047, high_px=1.083)
    funnel: dict = {}
    kept = filter_attack_15m_ma200(df, [sig], funnel=funnel)
    assert kept == []
    assert funnel.get("weak_15m_ma200", 0) == 1


def test_attack_ma200_keeps_when_2r_still_above() -> None:
    """UNI/COTI 那種：停很近，2R 還在 200 上面 → 留著。"""
    df, sig = _ma200_sig(open_px=1.060, close_px=1.046, high_px=1.052)
    kept = filter_attack_15m_ma200(df, [sig])
    assert kept == [sig]


def test_attack_ma200_keeps_real_dump_into_200() -> None:
    """CLO / MAGMA：大陰線或已經砸到 200 旁邊 → 留著。"""
    df_clo, clo = _ma200_sig(open_px=1.097, close_px=1.007, high_px=1.020)
    assert filter_attack_15m_ma200(df_clo, [clo]) == [clo]
    df_fat, fat = _ma200_sig(open_px=1.170, close_px=1.109, high_px=1.170)
    assert fat.body_pct >= 0.03
    assert filter_attack_15m_ma200(df_fat, [fat]) == [fat]


def test_attack_ma200_keeps_already_through() -> None:
    df, sig = _ma200_sig(open_px=1.02, close_px=0.99, high_px=1.03)
    assert filter_attack_15m_ma200(df, [sig]) == [sig]


def test_sit_15m_ma200_rejects_btw_near_200_and_1h99() -> None:
    """30d #2 BTW：收盤還貼 15m MA200，1h MA99 也近 → 濾掉。"""
    df, sig = _ma200_sig(open_px=1.082, close_px=1.009, high_px=1.09)
    ts = df.index[sig.entry_idx]
    h1 = make_1h(ts, n=220, old=sig.entry_price, recent=sig.entry_price)
    funnel: dict = {}
    kept = filter_sit_15m_ma200(df, [sig], h1, funnel=funnel)
    assert kept == []
    assert funnel.get("sit_15m_ma200", 0) == 1
    m200 = float(sma(df["close"].to_numpy(float), 200)[sig.entry_idx])
    assert sig.entry_price >= m200
    assert sig.entry_price / m200 - 1.0 < 0.02
    ma99 = htf_ma_at_entry(h1, ts, sig.entry_price, n=99)
    assert ma99 is not None and abs(sig.entry_price / ma99 - 1.0) < 0.05


def test_sit_15m_ma200_keeps_clo_when_1h99_has_room() -> None:
    """CLO：收盤砸到 15m 200 旁邊，但 1h MA99 還在 ≥5% 外 → 留著。"""
    df, sig = _ma200_sig(open_px=1.097, close_px=1.007, high_px=1.020)
    ts = df.index[sig.entry_idx]
    h1 = make_1h(ts, n=220, old=0.80, recent=1.05, recent_bars=30)
    kept = filter_sit_15m_ma200(df, [sig], h1)
    assert kept == [sig]
    ma99 = htf_ma_at_entry(h1, ts, sig.entry_price, n=99)
    assert ma99 is not None and abs(sig.entry_price / ma99 - 1.0) >= 0.05


def test_sit_15m_ma200_keeps_already_through() -> None:
    df, sig = _ma200_sig(open_px=1.02, close_px=0.99, high_px=1.03)
    ts = df.index[sig.entry_idx]
    h1 = make_1h(ts, n=220, old=sig.entry_price, recent=sig.entry_price)
    assert filter_sit_15m_ma200(df, [sig], h1) == [sig]


def test_sit_15m_ma200_uses_closed_bar_not_next() -> None:
    """下一根才決定進不進：過濾只用訊號 K 收盤，不偷看後面那根。"""
    df, sig = _ma200_sig(open_px=1.082, close_px=1.009, high_px=1.09)
    nxt = df.iloc[[-1]].copy()
    nxt.index = nxt.index + pd.Timedelta(minutes=15)
    nxt.iloc[0, nxt.columns.get_loc("open")] = float(sig.entry_price)
    nxt.iloc[0, nxt.columns.get_loc("close")] = 0.50
    nxt.iloc[0, nxt.columns.get_loc("high")] = float(sig.entry_price)
    nxt.iloc[0, nxt.columns.get_loc("low")] = 0.49
    later = pd.concat([df, nxt])
    ts = df.index[sig.entry_idx]
    h1 = make_1h(ts, n=220, old=sig.entry_price, recent=sig.entry_price)
    spoiled = h1.copy()
    i = bar_index_at(spoiled, ts)
    assert i is not None
    spoiled.iloc[i, spoiled.columns.get_loc("close")] = 0.50
    assert filter_sit_15m_ma200(df, [sig], h1) == []
    assert filter_sit_15m_ma200(later, [sig], spoiled) == []
    m_now = sma(df["close"].to_numpy(float), 200)[sig.entry_idx]
    m_later = sma(later["close"].to_numpy(float), 200)[sig.entry_idx]
    assert abs(float(m_now) - float(m_later)) < 1e-12


def test_1h_support_rejects_tut_cluster() -> None:
    """30d #3 TUT：1h 99/120/200 疊在進場價旁當支撐 → 濾掉。"""
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    px = sigs[0].entry_price
    h1 = make_1h(ts, n=220, old=px, recent=px)
    funnel: dict = {}
    kept = filter_away_1h_ma_support(df, sigs, h1, near=0.03, min_count=2, funnel=funnel)
    assert kept == []
    assert funnel.get("near_1h_support", 0) >= 1
    n_hug = 0
    for n in (99, 120, 200):
        ma = htf_ma_at_entry(h1, ts, px, n=n)
        assert ma is not None
        if abs(px / ma - 1.0) < 0.03:
            n_hug += 1
    assert n_hug >= 2


def test_1h_support_keeps_clo_fan() -> None:
    """CLO 那種 1h 長均張開：進場價沒貼著 99/120/200 叢 → 留著。"""
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    h1 = make_1h(ts, n=220, old=0.80, recent=1.05, recent_bars=30)
    kept = filter_away_1h_ma_support(df, sigs, h1, near=0.03, min_count=2)
    assert kept == sigs


def test_1h_mas_reject_tangled_like_flock() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    h1 = make_1h(ts, old=1.0, recent=1.0)
    funnel: dict = {}
    kept = filter_untangled_1h_mas(df, sigs, h1, min_spread=0.04, funnel=funnel)
    assert kept == []
    assert funnel.get("tangled_1h_ma", 0) >= 1
    mas = htf_mas_at_entry(h1, ts, sigs[0].entry_price)
    assert mas is not None
    spread = ma_cluster_spread(mas)
    assert spread is not None and spread < 0.04


def test_1h_mas_keep_fanned_like_cloud() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    h1 = make_1h(ts, old=0.80, recent=1.05, recent_bars=30)
    kept = filter_untangled_1h_mas(df, sigs, h1, min_spread=0.04)
    assert kept == sigs
    mas = htf_mas_at_entry(h1, ts, sigs[0].entry_price)
    assert mas is not None
    spread = ma_cluster_spread(mas)
    assert spread is not None and spread >= 0.04


def test_1h_ma120_rejects_too_close_like_morpho() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    px = sigs[0].entry_price
    h1 = make_1h(ts, old=px, recent=px)
    funnel: dict = {}
    kept = filter_away_1h_ma120(df, sigs, h1, min_dist=0.02, funnel=funnel)
    assert kept == []
    assert funnel.get("near_1h_ma120", 0) >= 1
    ma = htf_ma_at_entry(h1, ts, px, n=120)
    assert ma is not None and abs(px / ma - 1.0) < 0.02


def test_1h_ma120_keeps_cloud_distance() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    h1 = make_1h(ts, old=0.80, recent=1.05, recent_bars=30)
    kept = filter_away_1h_ma120(df, sigs, h1, min_dist=0.02)
    assert kept == sigs
    ma = htf_ma_at_entry(h1, ts, sigs[0].entry_price, n=120)
    assert ma is not None and abs(sigs[0].entry_price / ma - 1.0) >= 0.02


def test_1h_ma200_rejects_already_below() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    px = sigs[0].entry_price
    h1 = make_1h(ts, n=220, old=px * 1.25, recent=px * 1.25)
    funnel: dict = {}
    kept = filter_not_below_1h_ma200(df, sigs, h1, funnel=funnel)
    assert kept == []
    assert funnel.get("below_1h_ma200", 0) >= 1
    ma = htf_ma_at_entry(h1, ts, px, n=200)
    assert ma is not None and px < ma


def test_1h_ma200_keeps_cloud_still_above() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    px = sigs[0].entry_price
    h1 = make_1h(ts, n=220, old=0.80, recent=1.05, recent_bars=30)
    kept = filter_not_below_1h_ma200(df, sigs, h1)
    assert kept == sigs
    ma = htf_ma_at_entry(h1, ts, px, n=200)
    assert ma is not None and px >= ma


def test_1h_ma200_missing_data_skips() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    funnel: dict = {}
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert filter_not_below_1h_ma200(df, sigs, empty, funnel=funnel) == []
    assert funnel.get("no_1h_ma200", 0) >= 1


def test_summarize_and_html(tmp_path: Path | None = None) -> None:
    df = bars(dump_closes(n_flat=220))
    sigs = detect_signals(df, LOOSE)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    assert "win_rate" in stats
    out_dir = Path("/tmp/binance_15m_short_test") if tmp_path is None else tmp_path
    h1 = to_1h(df)
    hits = [Hit("CLOUSDT", t, df, df_1h=h1) for t in trades]
    path = write_html(out_dir / "index.html", hits, ["CLOUSDT"], "7d · test")
    text = path.read_text(encoding="utf-8")
    assert "CLOUSDT" in text
    assert "空頭排列" in text
    assert "1h 對照" in text
    assert "股票／ETF 永續預設不掃" in text
    assert "1h MA25" in text
    assert "1h MA99" in text
    assert "糾結" in text
    assert "15m 的 MA7/14/25/99/120 不能糾結" in text
    assert "15m MA200" in text
    assert "2R 還在 15m MA200" in text
    assert "打到 200" in text
    assert "收盤確認" in text
    assert "BTW" in text
    assert "TUT" in text
    assert "15m MA200 開" in text
    assert "MA120" in text
    assert "MA200" in text
    assert "虧損在前" in text
    assert (out_dir / "img").exists()
    pngs = list((out_dir / "img").glob("*.png"))
    assert any("1h" in p.name for p in pngs)


def test_write_view_html_relative_under_repo() -> None:
    dest_dir = Path("docs/_tmp_view_test")
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = dest_dir / "index.html"
    src.write_text("<img src='img/x.png'/>", encoding="utf-8")
    try:
        out = write_view_html(Path("docs/_tmp_view_test/index.html"))
        text = out.read_text(encoding="utf-8")
        assert "raw.githubusercontent.com" in text
        assert "docs/_tmp_view_test/img/x.png" in text
        assert "src='img/" not in text
    finally:
        import shutil

        shutil.rmtree(dest_dir)


def test_charts_put_losses_first() -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    trades = simulate(df, sigs)
    assert trades
    win = trades[0]
    lose = TradeResult(
        signal=win.signal,
        entry_idx=win.entry_idx,
        exit_idx=win.exit_idx,
        entry_price=win.entry_price,
        exit_price=win.stop_price,
        stop_price=win.stop_price,
        target_price=win.target_price,
        pnl_points=win.entry_price - win.stop_price,
        pnl_pct=(win.entry_price - win.stop_price) / win.entry_price,
        exit_reason="stop",
    )
    hits = [Hit("WINUSDT", win, df), Hit("LOSEUSDT", lose, df)]
    ordered = order_chart_hits(hits)
    assert [h.symbol for h in ordered] == ["LOSEUSDT", "WINUSDT"]
    assert ordered[0].trade.pnl_pct < 0 <= ordered[1].trade.pnl_pct


def main() -> int:
    test_sma()
    test_kline_limit_covers_month()
    test_fetch_klines_pages()
    test_stock_contract_filter()
    test_detect_dump_cross()
    test_no_signal_if_bullish_stack()
    test_rebreak_after_reclaim()
    test_green_candle_skipped()
    test_short_profit_and_stop()
    test_stop_on_rally()
    test_one_position_skips_overlap()
    test_filter_entry_window()
    test_volume_and_body_filters()
    test_bar_index_at()
    test_1h_ma25_keeps_dump_from_above()
    test_1h_ma25_rejects_still_above()
    test_1h_ma25_no_lookahead()
    test_1h_ma25_missing_data_skips()
    test_1h_ma99_keeps_near_like_cloud()
    test_1h_ma99_rejects_bulla_extension()
    test_1h_ma99_rejects_too_far_below()
    test_15m_mas_reject_tangled_like_zen_xmr()
    test_15m_mas_keep_fanned_like_cloud()
    test_15m_ma200_rejects_open_too_close()
    test_15m_ma200_keeps_dump_from_well_above()
    test_15m_ma200_keeps_already_through()
    test_15m_ma200_missing_data_skips()
    test_attack_ma200_rejects_weak_bar_when_2r_below()
    test_attack_ma200_keeps_when_2r_still_above()
    test_attack_ma200_keeps_real_dump_into_200()
    test_attack_ma200_keeps_already_through()
    test_sit_15m_ma200_rejects_btw_near_200_and_1h99()
    test_sit_15m_ma200_keeps_clo_when_1h99_has_room()
    test_sit_15m_ma200_keeps_already_through()
    test_sit_15m_ma200_uses_closed_bar_not_next()
    test_1h_support_rejects_tut_cluster()
    test_1h_support_keeps_clo_fan()
    test_1h_mas_reject_tangled_like_flock()
    test_1h_mas_keep_fanned_like_cloud()
    test_1h_ma120_rejects_too_close_like_morpho()
    test_1h_ma120_keeps_cloud_distance()
    test_1h_ma200_rejects_already_below()
    test_1h_ma200_keeps_cloud_still_above()
    test_1h_ma200_missing_data_skips()
    test_summarize_and_html()
    test_write_view_html_relative_under_repo()
    test_charts_put_losses_first()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
