#!/usr/bin/env python3
"""15m 黏帶：用本機快取 K 線比較出場與進場篩選（只讀，不下單）。

先跑 examples/cache_binance_15m.py。訓練段挑參數，驗證段只看不調。
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from nq.ma200_squeeze import HOLD_BARS, TARGET_R, detect_signals  # noqa: E402

_spec = importlib.util.spec_from_file_location("align_scan", REPO / "examples" / "scan_binance_15m_align.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)
_spec2 = importlib.util.spec_from_file_location("kcache", REPO / "examples" / "cache_binance_15m.py")
kcache = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(kcache)

TZ = scan.TZ
CACHE_START = datetime(2026, 8, 31, tzinfo=TZ)
CACHE_END = datetime(2026, 10, 31, tzinfo=TZ)
TRAIN = (datetime(2026, 9, 3, tzinfo=TZ), datetime(2026, 10, 17, tzinfo=TZ))
TEST = (datetime(2026, 10, 17, tzinfo=TZ), datetime(2026, 10, 31, tzinfo=TZ))
FEE_RT = 0.10  # 幣安 U 本位一般帳戶 taker 0.05% × 進出兩次，單位 %
SIG_PKL = Path("/tmp/squeeze_signals.pkl")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def build_signals() -> list[dict]:
    info = scan.get_json("/fapi/v1/exchangeInfo")
    ctype = {s["symbol"]: s.get("contractType", "") for s in info["symbols"]}
    s_ms, e_ms = _ms(CACHE_START), _ms(CACHE_END)
    lo, hi = _ms(TRAIN[0]), _ms(TEST[1])
    root = kcache.CACHE / f"{s_ms}_{e_ms}"
    out: list[dict] = []
    files = sorted(root.glob("*.npz"))
    for f in files:
        sym = f.stem
        raw = kcache.load(sym, s_ms, e_ms)
        if not raw or len(raw["c"]) < scan.MIN_BARS:
            continue
        d = scan.prepare(raw)
        for s in detect_signals(d):
            t = int(d["t"][s.idx])
            if lo <= t < hi and scan.is_stacked(d, s.idx):
                out.append({"symbol": sym, "d": d, "sig": s, "stock": ctype.get(sym) == "TRADIFI_PERPETUAL"})
    print(f"快取 {len(files)} 檔，訊號 {len(out)}", flush=True)
    return out


@dataclass(frozen=True)
class Exit:
    target_r: float = TARGET_R  # 0 = 不設停利
    hold: int = HOLD_BARS
    trail_ma: int = 0  # >0：收盤跌破這條 SMA 就在收盤出場
    be_r: float = 0.0  # >0：最高價到 +be_r R 後停損拉到進場價

    def label(self) -> str:
        parts = [f"{self.target_r:g}R" if self.target_r else "無停利", f"持{self.hold}"]
        if self.trail_ma:
            parts.append(f"破MA{self.trail_ma}")
        if self.be_r:
            parts.append(f"{self.be_r:g}R保本")
        return " ".join(parts)


def simulate(rows: list[dict], ex: Exit) -> list[dict]:
    """與 simulate_trades 同一套：下一根開盤進，同根先停損再停利，同檔不重疊。"""
    res = []
    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)
    for rs in by_sym.values():
        rs.sort(key=lambda r: r["sig"].idx)
        busy = -1
        for r in rs:
            d, s = r["d"], r["sig"]
            n = len(d["c"])
            ei = s.idx + 1
            if ei >= n or ei <= busy:
                continue
            entry = float(d["o"][ei])
            if entry <= s.stop:
                continue
            risk = s.close - s.stop
            target = s.close + ex.target_r * risk if ex.target_r else np.inf
            stop = s.stop
            ma = None
            if ex.trail_ma:
                key = f"m{ex.trail_ma}"
                ma = d[key] if key in d else scan.sma(d["c"], ex.trail_ma)
                d[key] = ma
            end = min(ei + ex.hold, n - 1)
            xi, xp, why = end, float(d["c"][end]), "time"
            for k in range(ei, end + 1):
                if float(d["l"][k]) <= stop:
                    xi, xp, why = k, stop, "stop" if stop < entry else "be"
                    break
                if float(d["h"][k]) >= target:
                    xi, xp, why = k, target, "target"
                    break
                if ma is not None and k > ei and float(d["c"][k]) < float(ma[k]):
                    xi, xp, why = k, float(d["c"][k]), "trail"
                    break
                if ex.be_r and float(d["h"][k]) >= s.close + ex.be_r * risk:
                    stop = max(stop, entry)
            res.append({**r, "entry": entry, "exit_idx": xi, "exit": xp, "why": why,
                        "pnl": (xp / entry - 1.0) * 100.0, "truncated": end < ei + ex.hold})
            busy = xi
    return res


def stats(trades: list[dict]) -> dict:
    p = [t["pnl"] for t in trades]
    if not p:
        return {"n": 0, "win": 0.0, "avg": 0.0, "med": 0.0, "net": 0.0, "sum_net": 0.0, "best": 0.0, "worst": 0.0}
    net = [x - FEE_RT for x in p]
    return {
        "n": len(p),
        "win": 100.0 * sum(1 for x in p if x > 0) / len(p),
        "win_net": 100.0 * sum(1 for x in net if x > 0) / len(p),
        "avg": float(np.mean(p)),
        "med": float(median(p)),
        "net": float(np.mean(net)),
        "sum_net": float(np.sum(net)),
        "best": float(max(p)),
        "worst": float(min(p)),
    }


def split(rows: list[dict], period: tuple[datetime, datetime]) -> list[dict]:
    a, b = _ms(period[0]), _ms(period[1])
    return [r for r in rows if a <= int(r["d"]["t"][r["sig"].idx]) < b]


def fmt(st: dict) -> str:
    if not st["n"]:
        return "n=0"
    return (f"n={st['n']:3d} 勝{st['win']:5.1f}% 均{st['avg']:+.3f}% 中{st['med']:+.3f}% "
            f"扣費均{st['net']:+.3f}% 扣費合計{st['sum_net']:+7.2f}% 好{st['best']:+.2f} 差{st['worst']:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild or not SIG_PKL.exists():
        rows = build_signals()
        pickle.dump(rows, SIG_PKL.open("wb"))
    else:
        rows = pickle.load(SIG_PKL.open("rb"))
    train, test = split(rows, TRAIN), split(rows, TEST)
    print(f"訓練訊號 {len(train)}  驗證訊號 {len(test)}  （扣費假設來回 {FEE_RT:.2f}%）")

    base = Exit()
    print("\n== 現行出場 ==")
    print("訓練", fmt(stats(simulate(train, base))))
    print("驗證", fmt(stats(simulate(test, base))))

    grid = [
        Exit(target_r=tr, hold=h, trail_ma=tm, be_r=be)
        for tr, h, tm, be in itertools.product(
            (1.0, 1.5, 2.0, 3.0, 0.0), (16, 32, 48, 96), (0, 20, 25, 99), (0.0, 1.0)
        )
    ]
    scored = []
    for ex in grid:
        st = stats(simulate(train, ex))
        scored.append((st["net"], ex, st))
    scored.sort(key=lambda x: -x[0])
    print("\n== 出場：訓練段扣費平均前 15 ==")
    for _, ex, st in scored[:15]:
        print(f"{ex.label():22s} 訓練 {fmt(st)}")
    print("\n== 同一批在驗證段 ==")
    for _, ex, _st in scored[:15]:
        print(f"{ex.label():22s} 驗證 {fmt(stats(simulate(test, ex)))}")

    best_ex = scored[0][1]
    print(f"\n== 進場分組（訓練段，出場用 {best_ex.label()} / 現行）==")
    tr_best = simulate(train, best_ex)
    tr_base = simulate(train, base)
    feats = {
        "股票": lambda t: t["stock"],
        "幣": lambda t: not t["stock"],
        "離200≤0.30%": lambda t: t["sig"].ext <= 0.003,
        "離200>0.30%": lambda t: t["sig"].ext > 0.003,
        "黏帶≤0.30%": lambda t: t["sig"].ribbon <= 0.003,
        "黏帶>0.30%": lambda t: t["sig"].ribbon > 0.003,
        "量≤4x": lambda t: t["sig"].vol_ratio <= 4.0,
        "量>4x": lambda t: t["sig"].vol_ratio > 4.0,
        "振幅≤3x": lambda t: t["sig"].expand <= 3.0,
        "振幅>3x": lambda t: t["sig"].expand > 3.0,
        "風險≤0.8%": lambda t: t["sig"].risk_pct <= 0.008,
        "風險>0.8%": lambda t: t["sig"].risk_pct > 0.008,
        "斜率≥0": lambda t: t["sig"].slope20 >= 0,
        "斜率<0": lambda t: t["sig"].slope20 < 0,
        "品質A": lambda t: t["sig"].quality == "A",
        "品質BC": lambda t: t["sig"].quality != "A",
    }
    for name, fn in feats.items():
        a = [t for t in tr_base if fn(t)]
        b = [t for t in tr_best if fn(t)]
        print(f"{name:12s} 現行 {fmt(stats(a))}\n{'':12s} 新出場 {fmt(stats(b))}")
    pickle.dump({"best": best_ex, "scored": [(s, e) for s, e, _ in scored]}, open("/tmp/squeeze_opt.pkl", "wb"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
