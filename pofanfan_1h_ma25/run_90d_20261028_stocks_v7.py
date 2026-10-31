#!/usr/bin/env python3
"""7d stock-perp 破底翻 v7 (2026-07-28 00:00 .. 2026-10-28 15:00 TPE). No TG, no git.
REST (universe refreshed from exchangeInfo); clock + time.time frozen at 2026-10-28 00:00 TPE;
--days 8 then entries filtered to >= 2026-10-21 00:00 TPE.

Window: entries >= 2026-09-03 00:00 TPE through the frozen clock.
Vision daily zips have no 2026-10-03 file, so the clock is frozen at
2026-10-04 00:00 TPE and --days 31 (cutoff = 2026-09-03 00:00). Last bars are
whatever Vision has published (expected last open 2026-10-03 07:00 TPE).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import binance_1h_ma25_reclaim as m  # noqa: E402
import stamp_pnl_badges as stamp  # noqa: E402
from run_7d_20260927_stocks import parse_detailed  # noqa: E402

TPE = ZoneInfo("Asia/Taipei")
ROOT = Path("/workspace/ma25-retest")
OUT = ROOT / "90d-20261028-stocks-v7"
FROZEN = datetime(2026, 10, 28, 15, 0, 0, tzinfo=TPE)
WIN_START = "2026-07-28 00:00"
SRC = ROOT / "7d-20260927-stocks" / "symbols.txt"
DAYS = 93


def freeze(frozen: datetime) -> None:
    class FrozenDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen if tz is None else frozen.astimezone(tz)

    m.datetime = FrozenDT
    m.time.time = lambda: frozen.timestamp()


def main() -> int:
    html, img, stamped = OUT / "index.html", OUT / "img", OUT / "stamped"
    for d in (OUT, img, stamped):
        d.mkdir(parents=True, exist_ok=True)
    for old in list(img.glob("*.png")) + list(stamped.glob("*.png")):
        old.unlink()

    # days>8 and n>80 would otherwise drop cards to 48. Keep every trade PNG.
    m.select_card_hits = lambda hits, **kwargs: list(hits)

    m._REST_BANNED = False
    freeze(FROZEN)
    syms = m.universe()
    syms = m.filter_symbols(syms, exclude_inverse=True, exclude_vol=False)
    (OUT / "symbols.txt").write_text(",".join(syms), encoding="utf-8")
    print(
        f"run out={OUT} n_symbols={len(syms)} frozen={FROZEN} days={DAYS} REST_BANNED={m._REST_BANNED}",
        flush=True,
    )

    probe_notes = []
    last_open = None
    for sym in ("NVDAUSDT", "TSLAUSDT", "AAPLUSDT"):
        if sym not in syms:
            continue
        df = m.fetch_klines(sym, "1h", days=10)
        if df is None or df.empty:
            probe_notes.append(f"{sym}: empty")
            continue
        last = df.index[-1]
        probe_notes.append(f"{sym} last_open={last.strftime('%Y-%m-%d %H:%M')} TPE n={len(df)}")
        if last_open is None or last > last_open:
            last_open = last
    print("freshness:", "; ".join(probe_notes), flush=True)

    argv = [
        "--symbols", ",".join(syms),
        "--days", str(DAYS),
        "--limit", str(max(len(syms), 300)),
        "--workers", "6",
        "--lose-ma-closes", "0",
        "--max-hold", "36",
        "--m5-exit",
        "--require-h4-ma25",
        "--h4-ma25-tol", "2.0",
        "--max-prior-spike", "15",
        "--prior-spike-bars", "24",
        "--min-reclaim-entry-bars", "2",
        "--ma200-near", "0",
        "--min-decline", "3.5",
        "--max-flat-bars", "2",
        "--max-reclaim-entry-bars", "9",
        "--no-w-overlay",
        "--exclude-inverse",
        "--h4-max-ma-drop", "6",
        "--h4-drop-ma", "120",
        "--h4-drop-bars", "24",
        "--max-entry-ext-atr", "2.0",
        "--max-overhead-knot-atr", "1.5",
        "--html", str(html),
    ]
    try:
        rc = m.main(argv)
    except ValueError as exc:
        print(f"WARN view_html/seq skipped: {exc}", flush=True)
        rc = 0
    if rc != 0 or not html.is_file():
        print("ERROR no html", flush=True)
        return 1

    text = html.read_text(encoding="utf-8")
    detailed_all = parse_detailed(text)
    detailed = [t for t in detailed_all if t["entry_tpe"] >= WIN_START]
    keep = {t["fname"] for t in detailed}
    trades = [t for t in stamp.parse_trades(text) if t["fname"] in keep]
    last_s = last_open.strftime("%Y-%m-%d %H:%M") if last_open is not None else ""
    for t in detailed:
        held_h = None
        try:
            e = datetime.strptime(t["entry_tpe"], "%Y-%m-%d %H:%M")
            x = datetime.strptime(t["exit_tpe"], "%Y-%m-%d %H:%M")
            held_h = int((x - e).total_seconds() // 3600)
        except ValueError:
            pass
        t["hold_bars"] = held_h
        t["censored_at_last_bar"] = bool(
            last_s and t["exit_tpe"] == last_s and t["reason"] == "timeout" and (held_h is None or held_h < 36)
        )

    reasons = Counter(t["reason"] for t in detailed)
    n = len(detailed)
    wins = sum(1 for t in detailed if t["pnl_pct"] > 0)
    s = sum(t["pnl_pct"] for t in detailed)
    usdt = sum(t["usdt_100x3"] for t in detailed)
    stamped_paths = []
    for t in trades:
        src = img / t["fname"]
        if src.is_file():
            dst = stamped / t["fname"]
            stamp.stamp(src, dst, t["display"], t["reason"], t["cls"])
            stamped_paths.append(str(dst))
    stamped_paths = sorted(stamped_paths)

    import csv
    csv_path = OUT / "trades.csv"
    fields = ["n", "symbol", "entry_tpe", "exit_tpe", "pnl_pct", "reason", "entry", "exit", "stop",
              "usdt_100x3", "hold_bars", "censored_at_last_bar", "fname"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in detailed:
            w.writerow(row)

    summary = {
        "scanned_at_real_tpe": datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S TPE"),
        "frozen_clock_tpe": FROZEN.strftime("%Y-%m-%d %H:%M:%S"),
        "window_requested": "2026-07-28 00:00 .. 2026-10-28 15:00 TPE",
        "window": "entries >= 2026-10-21 00:00 TPE (--days 93 + filter, clock frozen 2026-10-28 15:00 TPE)",
        "data_note": (
            "REST fapi used (unbanned). time.time frozen so forming bar dropped vs 07:00 TPE. "
            "A timeout exit on that last bar with hold < 36 is marked censored_at_last_bar."
        ),
        "probe": probe_notes,
        "last_bar_open_tpe": last_s,
        "universe": "STOCKS ONLY from live exchangeInfo TRADIFI_PERPETUAL equity, exclude_inverse",
        "n_symbols": len(syms),
        "module": m.__file__,
        "rest_banned": m._REST_BANNED,
        "rules": {
            "detect": "v6 + max_entry_ext_atr=2.0 + max_overhead_knot_atr=1.5",
            "base": {
                "min_bars_below": 4, "max_bars_below": 36, "min_depth_pct": 0.018,
                "min_undercut_pct": 0.014, "max_retest_wait": 18, "retest_tag_pct": 0.003,
                "retest_tag_atr": 0.4, "max_ma_drift_pct": 1.5, "min_ribbon_atr": 0.50,
                "min_ma_cluster_atr": 1.0, "min_break_vol": 0.8, "min_mid_ma_atr": 0.50,
            },
            "pretty": {
                "require_h4_ma25": True, "h4_ma25_tol_pct": 2.0,
                "max_prior_spike_pct": 0.15, "prior_spike_bars": 24,
                "min_reclaim_entry_bars": 2,
            },
            "v7": {
                "max_entry_ext_atr": 2.0,
                "max_overhead_knot_atr": 1.5,
            },
            "v3_v5": {
                "ma200_near_pct": 0.0, "min_decline_pct": 0.035, "max_flat_bars": 2,
                "max_reclaim_entry_bars": 9, "exclude_inverse": True,
                "h4_max_ma_drop_pct": 6, "h4_drop_ma": 120, "h4_drop_bars": 24,
            },
            "exits": "close below break low; m5_neck; max_hold=36; lose_ma_closes=0",
            "cards": "DRAW_W_OVERLAY=False",
            "sizing": "100U x 3x additive (not compounded)",
        },
        "n": n,
        "wins": wins,
        "winrate_pct": round(100.0 * wins / n, 2) if n else 0.0,
        "sum_pct": round(s, 4),
        "avg_pct": round(s / n, 4) if n else 0.0,
        "reason_counts": dict(reasons),
        "usdt_100x3_additive": round(usdt, 4),
        "margin_usdt": 100,
        "leverage": 3,
        "censored_trades": [f"{t['symbol']} {t['entry_tpe']}" for t in detailed if t.get("censored_at_last_bar")],
        "trades": detailed,
        "html": str(html),
        "csv": str(csv_path),
        "stamped_paths": stamped_paths,
    }
    summary_p = OUT / "summary.json"
    summary_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"RESULT n={n} WR={summary['winrate_pct']}% sum={s:+.2f}% avg={summary['avg_pct']:+.2f}% "
        f"usdt={usdt:+.2f} reasons={dict(reasons)} last_bar={last_s} censored={summary['censored_trades']}",
        flush=True,
    )
    for t in detailed:
        print(
            f"  #{t['n']} {t['symbol']} {t['entry_tpe']} -> {t['exit_tpe']} {t['pnl_pct']:+.2f}% "
            f"{t['reason']} usdt={t['usdt_100x3']:+.2f} censored={t['censored_at_last_bar']}",
            flush=True,
        )
    print(f"summary={summary_p} pngs={len(stamped_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
