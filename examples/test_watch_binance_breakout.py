#!/usr/bin/env python3
"""Offline synthetic unit tests for examples/watch_binance_breakout.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_breakout import (  # noqa: E402
    INTERVAL_MS,
    BreakoutParams,
    confirm_at,
    detect_signals,
    ignition_at,
    indicators,
    render_chart_bytes,
    write_html_report,
)


def make_synthetic_ake_like(
    n: int = 260,
    ign_i: int = 220,
    *,
    wick_dump: bool = False,
    break_ma25_after: bool = False,
) -> dict:
    """構造一段類似 AKEUSDT 的 220 根極窄橫盤 + 第 220 根爆量破箱體 + 縮量回踩續攻序列。"""
    t0 = 1_794_500_000_000
    t = np.array([t0 + k * INTERVAL_MS for k in range(n)], dtype=np.int64)

    # 前段在 0.00260 附近微幅震盪（振幅 < 2%，八均線完全糾結）
    base = 0.00260 + 0.000015 * np.sin(np.arange(n) * 0.35)
    o = base.copy()
    c = base + 0.000005 * np.cos(np.arange(n) * 0.5)
    h = np.maximum(o, c) + 0.000012
    l = np.minimum(o, c) - 0.000012
    v = np.full(n, 40_000_000.0)
    qv = np.full(n, 110_000.0)  # 平時每 5m 約 11 萬 U

    # 點火棒 ign_i：從 0.00262 拉到 0.00286（+9.16%），成交額 340 萬 U（>30x）
    o[ign_i] = 0.002620
    l[ign_i] = 0.002615
    if wick_dump:
        h[ign_i] = 0.003100
        c[ign_i] = 0.002710  # 長上影倒貨，close_pos = 0.196
    else:
        h[ign_i] = 0.002930
        c[ign_i] = 0.002860  # close_pos = 0.778
    qv[ign_i] = 3_400_000.0

    # 點火後回踩與續攻
    if break_ma25_after:
        for k in range(ign_i + 1, min(n, ign_i + 8)):
            o[k] = 0.002600
            c[k] = 0.002550
            h[k] = 0.002620
            l[k] = 0.002530
            qv[k] = 500_000.0
    else:
        # ign_i+1, ign_i+2 縮量回踩守在箱體頂之上（0.00276 ~ 0.00282）
        o[ign_i + 1], h[ign_i + 1], l[ign_i + 1], c[ign_i + 1], qv[ign_i + 1] = (
            0.002860,
            0.002865,
            0.002780,
            0.002810,
            900_000.0,
        )
        o[ign_i + 2], h[ign_i + 2], l[ign_i + 2], c[ign_i + 2], qv[ign_i + 2] = (
            0.002810,
            0.002820,
            0.002755,
            0.002785,
            620_000.0,
        )
        # ign_i+3 收紅轉強確認（MA7 > MA14 > MA25），隨後一路推升翻倍
        o[ign_i + 3], h[ign_i + 3], l[ign_i + 3], c[ign_i + 3], qv[ign_i + 3] = (
            0.002785,
            0.002940,
            0.002780,
            0.002915,
            1_400_000.0,
        )
        px = 0.002915
        for k in range(ign_i + 4, n):
            nxt = px * 1.018
            o[k] = px
            c[k] = nxt
            h[k] = nxt * 1.005
            l[k] = px * 0.995
            qv[k] = 1_800_000.0
            px = nxt

    return indicators({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "qv": qv})


class WatchBinanceBreakoutTests(unittest.TestCase):
    def test_catches_ake_style_ignition_and_pullback_confirm(self) -> None:
        d = make_synthetic_ake_like()
        params = BreakoutParams()
        sigs = detect_signals(
            d,
            params=params,
            btc_ret_map={int(d["t"][220]): 0.15},
            oi_change_fn=lambda _ts: 2.7,
        )
        self.assertEqual(len(sigs), 1)
        sig = sigs[0]
        self.assertEqual(sig["idx"], 220)
        self.assertGreater(sig["vol_ratio"], 25.0)
        self.assertLess(sig["ma_spread_pct"], 2.0)
        self.assertIsNotNone(sig["confirm"])
        self.assertEqual(sig["confirm"]["confirm_idx"], 223)
        self.assertGreater(sig["mfe_12h_pct"], 80.0)

    def test_rejects_long_upper_wick_pump_and_dump(self) -> None:
        d = make_synthetic_ake_like(wick_dump=True)
        ign = ignition_at(d, 220, BreakoutParams())
        self.assertIsNone(ign)

    def test_rejects_macro_btc_synchronized_spike(self) -> None:
        d = make_synthetic_ake_like()
        ts_ign = int(d["t"][220])
        # 若同根 BTC 已大漲 +7.5%，超額漲幅不足 2.5%，視為大盤事件盤而非獨立起漲
        ign = ignition_at(d, 220, BreakoutParams(), btc_ret_map={ts_ign: 7.5})
        self.assertIsNone(ign)

    def test_rejects_negative_oi_short_squeeze_only(self) -> None:
        d = make_synthetic_ake_like()
        sigs = detect_signals(
            d,
            params=BreakoutParams(min_oi_change_pct=0.5),
            btc_ret_map={int(d["t"][220]): 0.0},
            oi_change_fn=lambda _ts: -0.8,
        )
        self.assertEqual(len(sigs), 0)

    def test_rejects_stage2_when_pullback_breaks_ma25(self) -> None:
        d = make_synthetic_ake_like(break_ma25_after=True)
        sigs = detect_signals(
            d,
            params=BreakoutParams(),
            btc_ret_map={int(d["t"][220]): 0.0},
            oi_change_fn=lambda _ts: 2.0,
        )
        self.assertEqual(len(sigs), 1)
        self.assertIsNone(sigs[0]["confirm"])
        self.assertIsNone(confirm_at(d, 220, 223, sigs[0], BreakoutParams()))

    def test_renders_chart_and_html_report(self) -> None:
        d = make_synthetic_ake_like()
        sigs = detect_signals(d, oi_change_fn=lambda _ts: 3.1)
        self.assertEqual(len(sigs), 1)
        rec = sigs[0]
        rec["symbol"] = "AKEUSDT"
        rec["d"] = d

        png = render_chart_bytes("AKEUSDT", d, rec["idx"], rec["box_hi"], conf_i=rec["confirm"]["confirm_idx"])
        self.assertIsNotNone(png)
        assert png is not None
        self.assertTrue(png.startswith(b"\x89PNG"))

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.html"
            write_html_report([rec], out, days=3.0, n_symbols=1)
            html = out.read_text(encoding="utf-8")
            self.assertIn("AKEUSDT", html)
            self.assertIn("回踩確認", html)
            self.assertIn("data:image/png;base64,", html)


if __name__ == "__main__":
    unittest.main()
