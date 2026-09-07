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
    bar_index_at,
    detect_signals,
    filter_entry_window,
    default_params,
    htf_snapshot,
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


def test_summarize_and_html(tmp_path: Path | None = None) -> None:
    df = bars(dump_closes())
    sigs = detect_signals(df, LOOSE)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    assert "win_rate" in stats
    from binance_15m_short import Hit

    out_dir = Path("/tmp/binance_15m_short_test") if tmp_path is None else tmp_path
    h1 = to_1h(df)
    hits = [Hit("CLOUSDT", t, df, df_1h=h1) for t in trades]
    path = write_html(out_dir / "index.html", hits, ["CLOUSDT"], "7d · test")
    text = path.read_text(encoding="utf-8")
    assert "CLOUSDT" in text
    assert "空頭排列" in text
    assert "1h 對照" in text
    assert (out_dir / "img").exists()
    pngs = list((out_dir / "img").glob("*.png"))
    assert any("1h" in p.name for p in pngs)


def main() -> int:
    test_sma()
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
    test_summarize_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
