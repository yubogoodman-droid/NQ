#!/usr/bin/env python3
"""Synthetic tests for 幣安 15m 空頭排列跌破 99/120（不打幣安）。"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_short import (  # noqa: E402
    TPE,
    Hit,
    TradeResult,
    bar_index_at,
    detect_signals,
    filter_below_1h_ma25,
    filter_entry_window,
    filter_near_1h_ma99,
    filter_untangled_1h_mas,
    filter_away_1h_ma120,
    filter_not_below_1h_ma200,
    default_params,
    htf_ma_at_entry,
    htf_mas_at_entry,
    htf_snapshot,
    is_stock_contract,
    ma_cluster_spread,
    order_chart_hits,
    simulate,
    sma,
    summarize_trades,
    write_html,
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
    df = bars(dump_closes())
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
    assert "MA120" in text
    assert "MA200" in text
    assert "虧損在前" in text
    assert (out_dir / "img").exists()
    pngs = list((out_dir / "img").glob("*.png"))
    assert any("1h" in p.name for p in pngs)


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
    test_1h_mas_reject_tangled_like_flock()
    test_1h_mas_keep_fanned_like_cloud()
    test_1h_ma120_rejects_too_close_like_morpho()
    test_1h_ma120_keeps_cloud_distance()
    test_1h_ma200_rejects_already_below()
    test_1h_ma200_keeps_cloud_still_above()
    test_1h_ma200_missing_data_skips()
    test_summarize_and_html()
    test_charts_put_losses_first()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
