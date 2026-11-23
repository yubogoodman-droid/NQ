#!/usr/bin/env python3
"""15 分五均線突破／跌破策略回測與優化（動態區間 + 前後半樣本外驗證 + 週別穩定度）。

只讀幣安 U 本位合約 K 線，不下單。

    python3 examples/optimize_15m_ribbon.py --days 56
    python3 examples/optimize_15m_ribbon.py --days 28 --cost 0.10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.backtest_15m_ribbon import fetch_klines, universe  # noqa: E402
from nq.ribbon15 import (  # noqa: E402
    SIGNAL_MA_PERIODS,
    add_mas,
    detect_all_breaks,
)

TZ = timezone(timedelta(hours=8))


@dataclass
class Sig:
    symbol: str
    side: str
    time_ms: int
    first_half: bool
    week: int
    ma200_aligned: bool
    ret24h_pct: float | None
    rs24_pct: float | None
    wick_ratio: float
    width_pct: float
    cooldown_ok: bool
    exits: dict[str, float]


def _exits(d: dict, br, hold: int) -> dict[str, float]:
    nxt = br.idx + 1
    j = nxt + hold - 1
    if j >= len(d["c"]):
        return {}
    entry = float(d["o"][nxt])
    is_short = br.side == "short"

    def pnl(px: float) -> float:
        return (1.0 - px / entry) * 100.0 if is_short else (px / entry - 1.0) * 100.0

    hold_ret = pnl(float(d["c"][j]))
    failed = float(d["c"][nxt]) > br.ma_min if is_short else float(d["c"][nxt]) < br.ma_max
    fail_exit = pnl(float(d["c"][nxt])) if failed else hold_ret

    def stop_at(stop: float) -> float:
        for k in range(nxt, j + 1):
            if is_short:
                if float(d["o"][k]) >= stop:
                    return pnl(float(d["o"][k]))
                if float(d["h"][k]) >= stop:
                    return pnl(stop)
            else:
                if float(d["o"][k]) <= stop:
                    return pnl(float(d["o"][k]))
                if float(d["l"][k]) <= stop:
                    return pnl(stop)
        return hold_ret

    return {
        "hold4h": hold_ret,
        "fail_exit": fail_exit,
        "stop_k": stop_at(br.high if is_short else br.low),
        "stop_band": stop_at(br.ma_max_prev if is_short else br.ma_min_prev),
    }


def _scan_one(
    sym: str,
    *,
    days: int,
    hold: int,
    cutoff_ms: int,
    split_ms: int,
    cooldown_ms: int,
    btc_ret24: dict[int, float],
) -> list[Sig]:
    raw = fetch_klines(sym, days=days, interval="15m")
    if raw is None:
        return []
    d = add_mas(raw)
    out: list[Sig] = []
    cd_bars = int(round(cooldown_ms / 900_000))
    for br in detect_all_breaks(d, periods=SIGNAL_MA_PERIODS):
        ts = int(d["t"][br.idx])
        if ts < cutoff_ms:
            continue
        ex = _exits(d, br, hold)
        if not ex:
            continue
        btc_r = btc_ret24.get(ts)
        rs24 = (br.ret24h_pct - btc_r) if (br.ret24h_pct is not None and btc_r is not None) else None
        week = int((ts - cutoff_ms) // (7 * 86_400_000))
        cd_ok = br.bars_since_any_break is None or br.bars_since_any_break >= cd_bars
        out.append(
            Sig(
                symbol=sym,
                side=br.side,
                time_ms=ts,
                first_half=ts < split_ms,
                week=week,
                ma200_aligned=br.ma200_aligned,
                ret24h_pct=br.ret24h_pct,
                rs24_pct=rs24,
                wick_ratio=br.wick_ratio,
                width_pct=br.width_pct,
                cooldown_ok=cd_ok,
                exits=ex,
            )
        )
    return out


def summarize(vals: list[float], *, cost_pct: float) -> dict:
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "trim": 0.0, "pf": 0.0}
    a = np.array(vals, dtype=float) - cost_pct
    s = np.sort(a)
    trimmed = s[2:-2] if len(s) > 10 else s
    gains = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    return {
        "n": int(len(a)),
        "wr": float((a > 0).mean() * 100.0),
        "avg": float(a.mean()),
        "med": float(np.median(a)),
        "trim": float(trimmed.mean()),
        "pf": float(pf),
    }


def fmt(st: dict) -> str:
    return (
        f"n={st['n']:>4}  勝率 {st['wr']:5.1f}%  均 {st['avg']:+.3f}%  "
        f"中位 {st['med']:+.3f}%  去極端 {st['trim']:+.3f}%  PF {st['pf']:.2f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="15 分五均線突破／跌破回測與優化（含樣本外驗證）")
    p.add_argument("--days", type=int, default=56, help="回測天數，預設 56 天（8 週）")
    p.add_argument("--hold", type=int, default=16, help="持有 15m 根數，預設 16（4 小時）")
    p.add_argument("--cost", type=float, default=0.10, help="單筆來回成本 %，預設 0.10%")
    p.add_argument("--cooldown-hours", type=float, default=8.0, help="同幣冷卻小時數，預設 8 小時")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-symbols", type=int, default=0, help="除錯用，只掃前 N 個幣")
    args = p.parse_args()

    print(f"抓取 BTCUSDT 近 {args.days} 天基準…", flush=True)
    btc_raw = fetch_klines("BTCUSDT", days=args.days, interval="15m")
    if btc_raw is None:
        print("無法取得 BTCUSDT K 線")
        return 1
    btc_t, btc_c = btc_raw["t"], btc_raw["c"]
    end_ms = int(btc_t[-1])
    cutoff_ms = end_ms - args.days * 86_400_000
    split_ms = end_ms - (args.days // 2) * 86_400_000
    btc_ret24 = {
        int(btc_t[i]): float((btc_c[i - 1] / btc_c[i - 97] - 1.0) * 100.0)
        for i in range(97, len(btc_c))
        if btc_c[i - 97] > 0
    }
    i0 = int(np.searchsorted(btc_t, cutoff_ms))
    im = int(np.searchsorted(btc_t, split_ms))
    btc_all = (btc_c[-1] / btc_c[i0] - 1.0) * 100.0
    btc_h1 = (btc_c[im] / btc_c[i0] - 1.0) * 100.0
    btc_h2 = (btc_c[-1] / btc_c[im] - 1.0) * 100.0

    symbols = universe(asset="all")
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    t0_dt = datetime.fromtimestamp(cutoff_ms / 1000, TZ).strftime("%Y-%m-%d")
    t1_dt = datetime.fromtimestamp(end_ms / 1000, TZ).strftime("%Y-%m-%d")
    print(f"掃描 {len(symbols)} 個流動永續，區間 {t0_dt} → {t1_dt}（GMT+8）", flush=True)
    print(f"BTC 同期：{btc_all:+.2f}%（前半 {btc_h1:+.2f}% / 後半 {btc_h2:+.2f}%）", flush=True)

    sigs: list[Sig] = []
    t0 = time.time()
    cd_ms = int(args.cooldown_hours * 3_600_000)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {
            ex.submit(
                _scan_one,
                s,
                days=args.days,
                hold=args.hold,
                cutoff_ms=cutoff_ms,
                split_ms=split_ms,
                cooldown_ms=cd_ms,
                btc_ret24=btc_ret24,
            ): s
            for s in symbols
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sigs.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
            if done % 60 == 0 or done == len(symbols):
                print(f"  進度 {done}/{len(symbols)}  訊號 {len(sigs)}  {time.time()-t0:.1f}s", flush=True)

    long_raw = [s for s in sigs if s.side == "long"]
    print(f"\n=== 原始多方：不同出場（已扣來回 {args.cost:.2f}%）===")
    for key, name in (
        ("hold4h", f"抱 {args.hold * 15 // 60}h 固定出場"),
        ("fail_exit", "下一根假突破就出"),
        ("stop_k", "停損訊號K低點"),
        ("stop_band", "停損帶底"),
    ):
        st = summarize([s.exits[key] for s in long_raw], cost_pct=args.cost)
        print(f"  {name:<16} {fmt(st)}")

    strategies = [
        ("原始多方", lambda s: s.side == "long"),
        ("原始空方", lambda s: s.side == "short"),
        (
            f"優化多方 A（MA200上方 + 前24h漲 + {args.cooldown_hours:g}h冷卻）",
            lambda s: s.side == "long"
            and s.ma200_aligned
            and s.cooldown_ok
            and s.ret24h_pct is not None
            and s.ret24h_pct > 0,
        ),
        (
            f"優化多方推薦（MA200上方 + 前24h漲 + 寬度≤1% + 上影≤25% + {args.cooldown_hours:g}h冷卻）",
            lambda s: s.side == "long"
            and s.ma200_aligned
            and s.cooldown_ok
            and s.ret24h_pct is not None
            and s.ret24h_pct > 0
            and s.width_pct <= 1.0
            and s.wick_ratio <= 0.25,
        ),
        (
            f"優化空方（MA200下方 + 前24h跌 + {args.cooldown_hours:g}h冷卻）",
            lambda s: s.side == "short"
            and s.ma200_aligned
            and s.cooldown_ok
            and s.ret24h_pct is not None
            and s.ret24h_pct < 0,
        ),
        (
            f"雙向順勢（同側MA200 + 前24h順勢 + 寬度≤1% + {args.cooldown_hours:g}h冷卻）",
            lambda s: s.ma200_aligned
            and s.cooldown_ok
            and s.width_pct <= 1.0
            and s.ret24h_pct is not None
            and (s.ret24h_pct > 0 if s.side == "long" else s.ret24h_pct < 0),
        ),
        (
            f"雙向相對強弱（同側MA200 + 相對BTC 24h >+1%/<-1% + {args.cooldown_hours:g}h冷卻）",
            lambda s: s.ma200_aligned
            and s.cooldown_ok
            and s.rs24_pct is not None
            and (s.rs24_pct > 1.0 if s.side == "long" else s.rs24_pct < -1.0),
        ),
    ]

    print(f"\n=== 全期與前後半樣本外對照（抱 {args.hold * 15 // 60}h，已扣來回 {args.cost:.2f}%）===")
    for name, pred in strategies:
        sub = [s for s in sigs if pred(s)]
        all_s = summarize([s.exits["hold4h"] for s in sub], cost_pct=args.cost)
        raw_s = summarize([s.exits["hold4h"] for s in sub], cost_pct=0.0)
        h1_s = summarize([s.exits["hold4h"] for s in sub if s.first_half], cost_pct=args.cost)
        h2_s = summarize([s.exits["hold4h"] for s in sub if not s.first_half], cost_pct=args.cost)
        print(f"\n[{name}]")
        print(f"  未扣費：{fmt(raw_s)}")
        print(f"  扣費後：{fmt(all_s)}")
        print(
            f"  前半 {h1_s['n']}筆 (勝率 {h1_s['wr']:.1f}% / 均 {h1_s['avg']:+.3f}%)  |  "
            f"後半 {h2_s['n']}筆 (勝率 {h2_s['wr']:.1f}% / 均 {h2_s['avg']:+.3f}%)"
        )

    print(f"\n=== 逐週穩定度（已扣來回 {args.cost:.2f}%）===")
    n_weeks = max((s.week for s in sigs), default=0) + 1
    for name, pred in (strategies[0], strategies[3], strategies[5], strategies[6]):
        sub = [s for s in sigs if pred(s)]
        by_w: dict[int, list[float]] = defaultdict(list)
        for s in sub:
            by_w[s.week].append(s.exits["hold4h"])
        pos_w = 0
        parts = []
        for w in range(n_weeks):
            st = summarize(by_w.get(w, []), cost_pct=args.cost)
            if st["n"] and st["avg"] > 0:
                pos_w += 1
            parts.append(f"W{w+1}:{st['avg']:+.2f}%({st['n']})" if st["n"] else f"W{w+1}:—")
        print(f"  {name} [{pos_w}/{n_weeks}週正]")
        print("    " + "  ".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
