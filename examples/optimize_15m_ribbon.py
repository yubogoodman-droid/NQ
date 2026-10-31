#!/usr/bin/env python3
"""15 分五均線訊號的過濾與出場比較：前一週挑、後一週驗證。只讀幣安 K 線，不下單。

    python3 examples/optimize_15m_ribbon.py
"""

from __future__ import annotations

import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.backtest_15m_ribbon import universe  # noqa: E402
from examples.backtest_15m_ribbon_range import END, START, scan_symbol  # noqa: E402

TZ = timezone(timedelta(hours=8))
SPLIT = datetime(2026, 10, 24, 0, 0, tzinfo=TZ)
HOLD = 16
COST_PCT = 0.10  # 來回 taker 0.05% × 2，不含資金費率與滑價
MIN_TRAIN = 60


@dataclass
class Sig:
    symbol: str
    time_ms: int
    train: bool
    feats: dict
    exits: dict


def features(sym: str, d: dict, row) -> dict:
    br = row.break_
    i = br.idx
    c = d["c"]
    m200 = d["m200"][i]
    m200_prev = d["m200"][i - 8] if i >= 8 else np.nan
    m99, m120 = d["m99"][i], d["m120"][i]
    ret24 = (c[i - 1] / c[i - 25] - 1) * 100 if i >= 25 else np.nan
    ret96 = (c[i - 1] / c[i - 97] - 1) * 100 if i >= 97 else np.nan
    return {
        "width": br.width_pct,
        "vol": br.vol_ratio,
        "body": br.body_through,
        "ext": (br.close / br.ma_max - 1) * 100,
        "range": br.range_pct,
        "above200": bool(not np.isnan(m200) and br.close > m200),
        "m200_up": bool(not np.isnan(m200_prev) and m200 > m200_prev),
        "long_up": bool(m99 > m120),
        "ret24": ret24,
        "ret96": ret96,
        "hour": datetime.fromtimestamp(row.time_ms / 1000, TZ).hour,
        "tradfi": sym in TRADFI,
    }


def exits(d: dict, row) -> dict:
    br = row.break_
    i = br.idx
    nxt = i + 1
    j = nxt + HOLD - 1
    if j >= len(d["c"]):
        return {}
    entry = float(d["o"][nxt])
    out = {"hold4h": (float(d["c"][j]) / entry - 1) * 100}

    # 下一根收回最高均線下方就在那根收盤出，否則抱 4 小時
    if float(d["c"][nxt]) < br.ma_max:
        out["fail_exit"] = (float(d["c"][nxt]) / entry - 1) * 100
    else:
        out["fail_exit"] = out["hold4h"]

    # 停損放在訊號 K 最低價；跳空開在停損下方就用開盤成交
    def stop_at(stop: float) -> float:
        for k in range(nxt, j + 1):
            if float(d["o"][k]) <= stop:
                return (float(d["o"][k]) / entry - 1) * 100
            if float(d["l"][k]) <= stop:
                return (stop / entry - 1) * 100
        return out["hold4h"]

    out["stop_low"] = stop_at(br.low)
    out["stop_band"] = stop_at(br.ma_min_prev)
    return out


TRADFI: set[str] = set()


def collect() -> list[Sig]:
    from examples.backtest_15m_ribbon import get_json

    info = get_json("/fapi/v1/exchangeInfo")
    TRADFI.update(s["symbol"] for s in info["symbols"] if s.get("contractType") == "TRADIFI_PERPETUAL")
    symbols = universe(asset="all")
    print(f"掃描 {len(symbols)} 檔，{START:%m-%d} → {(END - timedelta(days=1)):%m-%d}", flush=True)
    sigs: list[Sig] = []
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                sym, d, rows = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            for row in rows:
                ex_ = exits(d, row)
                if not ex_:
                    continue
                ts = datetime.fromtimestamp(row.time_ms / 1000, TZ)
                sigs.append(Sig(sym, row.time_ms, ts < SPLIT, features(sym, d, row), ex_))
    print(f"已結算 4h 訊號 {len(sigs)}  {time.time()-t0:.1f}s", flush=True)
    return sigs


def stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "sum": 0.0}
    a = np.array(vals, float) - COST_PCT
    return {"n": len(a), "wr": float((a > 0).mean() * 100), "avg": float(a.mean()), "med": float(np.median(a)), "sum": float(a.sum())}


FILTERS = {
    "寬度≤1%": lambda f: f["width"] <= 1.0,
    "寬度≤0.6%": lambda f: f["width"] <= 0.6,
    "寬度≤0.4%": lambda f: f["width"] <= 0.4,
    "寬度>0.2%": lambda f: f["width"] > 0.2,
    "量比≥1.5": lambda f: f["vol"] >= 1.5,
    "量比≥2": lambda f: f["vol"] >= 2.0,
    "量比≥3": lambda f: f["vol"] >= 3.0,
    "實體穿越": lambda f: f["body"],
    "收離帶頂≤0.5%": lambda f: f["ext"] <= 0.5,
    "收離帶頂≤1%": lambda f: f["ext"] <= 1.0,
    "收離帶頂>0.3%": lambda f: f["ext"] > 0.3,
    "站上MA200": lambda f: f["above200"],
    "MA200上彎": lambda f: f["m200_up"],
    "MA99>MA120": lambda f: f["long_up"],
    "前6小時跌>1%": lambda f: not np.isnan(f["ret24"]) and f["ret24"] < -1,
    "前24小時漲": lambda f: not np.isnan(f["ret96"]) and f["ret96"] > 0,
    "前24小時跌": lambda f: not np.isnan(f["ret96"]) and f["ret96"] < 0,
    "只加密": lambda f: not f["tradfi"],
    "只股票等TradFi": lambda f: f["tradfi"],
    "亞洲盤08-16": lambda f: 8 <= f["hour"] < 16,
    "美盤20-04": lambda f: f["hour"] >= 20 or f["hour"] < 4,
}
EXITS = {"hold4h": "抱 4 小時", "fail_exit": "假突破就出", "stop_low": "停損訊號K低點", "stop_band": "停損帶底"}


def pick(sigs: list[Sig], train: bool) -> list[Sig]:
    return [s for s in sigs if s.train == train]


def evaluate(sigs: list[Sig], names: tuple[str, ...], exit_key: str) -> tuple[dict, dict]:
    keep = [s for s in sigs if all(FILTERS[n](s.feats) for n in names)]
    tr = stat([s.exits[exit_key] for s in keep if s.train])
    te = stat([s.exits[exit_key] for s in keep if not s.train])
    return tr, te


def fmt(s: dict) -> str:
    return f"n={s['n']:>3} 勝率 {s['wr']:5.1f}% 均 {s['avg']:+.3f}% 中位 {s['med']:+.3f}%"


def main() -> int:
    sigs = collect()
    print(f"\n前一週（挑條件）{len(pick(sigs, True))} 筆，後一週（驗證）{len(pick(sigs, False))} 筆；已扣來回 {COST_PCT}%")
    print("\n=== 原始條件，不同出場 ===")
    for k, label in EXITS.items():
        tr, te = evaluate(sigs, (), k)
        print(f"{label:<12} 前週 {fmt(tr)} | 後週 {fmt(te)}")

    results = []
    for k in EXITS:
        combos = [()] + [(n,) for n in FILTERS] + list(itertools.combinations(FILTERS, 2))
        for names in combos:
            tr, te = evaluate(sigs, names, k)
            if tr["n"] < MIN_TRAIN:
                continue
            results.append((names, k, tr, te))

    print("\n=== 只看前一週挑出的前 15 名，後一週表現 ===")
    results.sort(key=lambda x: x[2]["avg"], reverse=True)
    for names, k, tr, te in results[:15]:
        label = " + ".join(names) or "原始"
        print(f"{label:<28} {EXITS[k]:<10} 前週 {fmt(tr)} | 後週 {fmt(te)}")

    print("\n=== 單一條件，抱 4 小時 ===")
    for n in FILTERS:
        tr, te = evaluate(sigs, (n,), "hold4h")
        print(f"{n:<16} 前週 {fmt(tr)} | 後週 {fmt(te)}")

    both = [r for r in results if r[2]["avg"] > 0 and r[3]["avg"] > 0 and r[3]["n"] >= 40]
    both.sort(key=lambda x: min(x[2]["avg"], x[3]["avg"]), reverse=True)
    print("\n=== 兩週都賺（扣成本後）且後週 ≥40 筆，取兩週較差那週排序 ===")
    for names, k, tr, te in both[:15]:
        label = " + ".join(names) or "原始"
        print(f"{label:<28} {EXITS[k]:<10} 前週 {fmt(tr)} | 後週 {fmt(te)}")

    wr_pick = [r for r in results if r[1] == "hold4h" and r[2]["n"] >= 35]
    wr_pick.sort(key=lambda x: (x[2]["wr"], x[2]["avg"]), reverse=True)
    print("\n=== 依前週勝率挑（抱 4 小時，前週 ≥35 筆），看後週勝率與中位數 ===")
    for names, k, tr, te in wr_pick[:12]:
        label = " + ".join(names) or "原始"
        print(f"{label:<28} 前週 {fmt(tr)} | 後週 {fmt(te)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
