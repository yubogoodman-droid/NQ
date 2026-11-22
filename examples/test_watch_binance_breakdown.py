#!/usr/bin/env python3
"""離線測試：5 分起跌要抓到 AMD 2026-11-19 23:10，不要抓後面的大陰線。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_breakdown import (  # noqa: E402
    TZ,
    detect_breakdowns,
    hm,
    path_after,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "amdusdt_5m_20261119.json"


def load_fixture() -> dict:
    rows = json.loads(FIXTURE.read_text())
    return {
        "t": np.array([r["t"] for r in rows], np.int64),
        "o": np.array([r["o"] for r in rows], np.float64),
        "h": np.array([r["h"] for r in rows], np.float64),
        "l": np.array([r["l"] for r in rows], np.float64),
        "c": np.array([r["c"] for r in rows], np.float64),
    }


def make_breakdown() -> dict:
    """平台 → 收在高點的衝高 → 吐回 → 回抽 → 收在低點跌破長均線。"""
    n = 230
    t0 = int(datetime(2026, 11, 19, 0, 0, tzinfo=TZ).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 5 * 60_000
    c = np.full(n, 100.0)
    o = np.full(n, 100.0)
    h = np.full(n, 100.08)
    l = np.full(n, 99.92)
    # 220 衝高，221 吐回，222 回抽，223 仍在均線上，224 跌破
    o[220], h[220], l[220], c[220] = 100.4, 103.2, 100.3, 103.0
    o[221], h[221], l[221], c[221] = 103.0, 103.4, 101.2, 101.3
    o[222], h[222], l[222], c[222] = 101.3, 102.4, 101.3, 102.2
    o[223], h[223], l[223], c[223] = 102.2, 102.3, 101.4, 101.6
    o[224], h[224], l[224], c[224] = 101.6, 101.6, 99.4, 99.5
    return {"t": t, "o": o, "h": h, "l": l, "c": c}


def test_amd_fixture_hits_2310_not_the_waterfall() -> None:
    d = load_fixture()
    sigs = detect_breakdowns(d)
    assert len(sigs) == 1, [(hm(s["t"]), s["close"]) for s in sigs]
    sig = sigs[0]
    assert hm(sig["t"]) == "11-19 23:10"
    assert abs(sig["close"] - 536.07) < 0.02
    assert abs(sig["stop"] - 547.24) < 0.02
    assert hm(sig["spike_t"]) == "11-19 22:45"
    assert hm(sig["reject_t"]) == "11-19 22:50"
    assert hm(sig["bounce_t"]) == "11-19 23:00"
    assert sig["stop"] > sig["close"]
    # 夾具只到 23:35，還看不到 471 的低點；至少要已經離開停損、往下走
    path = path_after(d, sig, bars=5)
    assert path["reason"] == "open"
    assert path["mfe_pct"] > 1.0


def test_synthetic_breakdown_and_a_plain_dip() -> None:
    d = make_breakdown()
    sigs = detect_breakdowns(d)
    assert len(sigs) == 1, [(hm(s["t"]), round(s["close"], 2)) for s in sigs]
    assert sigs[0]["i"] == 224
    assert abs(sigs[0]["stop"] - 102.4) < 1e-9

    # 同樣最後一根直接收破，但前面沒有衝高吐回
    plain = make_breakdown()
    for i in range(220, 225):
        plain["o"][i] = plain["h"][i] = plain["l"][i] = plain["c"][i] = 100.0
    plain["o"][224], plain["h"][224], plain["l"][224], plain["c"][224] = 100.0, 100.05, 99.2, 99.3
    assert detect_breakdowns(plain) == []


def test_reclaim_of_the_spike_cancels() -> None:
    d = make_breakdown()
    # 回抽收過衝高收盤，這波沒有失敗
    d["c"][222] = 103.1
    d["h"][222] = 103.15
    assert detect_breakdowns(d) == []


if __name__ == "__main__":
    test_amd_fixture_hits_2310_not_the_waterfall()
    test_synthetic_breakdown_and_a_plain_dip()
    test_reclaim_of_the_spike_cancels()
    print("ALL TESTS PASSED")
