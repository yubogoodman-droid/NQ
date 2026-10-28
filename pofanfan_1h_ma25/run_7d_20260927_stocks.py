#!/usr/bin/env python3
"""7-day MA25 pretty/TG DETECT on STOCK equity perps only. No TG, no git.
Window ~2026-09-20 .. 2026-09-27 TPE via --days 7.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, "/workspace/binance-1h-ma25/examples")
import binance_1h_ma25_reclaim as m

sys.path.insert(0, "/workspace/ma25-retest")
import stamp_pnl_badges as stamp  # noqa: E402

TPE = ZoneInfo("Asia/Taipei")
OUT = Path("/workspace/ma25-retest/7d-20260927-stocks")
HTML = OUT / "index.html"
IMG = OUT / "img"
STAMPED = OUT / "stamped"
SUMMARY = OUT / "summary.json"

CARD_RE = re.compile(
    r"<article class='trade-card'>.*?"
    r"<span class='trade-no'>#(\d+)\s*·\s*([^·<]+).*?</span>"
    r"<span class='trade-time'>([^<]+)</span>.*?"
    r"class='card-pnl (pnl-\w+)'>([^<]+)</div>.*?"
    r"<span class='tag tag-(?:time|sl|tp)'>([^<]+)</span>.*?"
    r"<pre class='trade-detail'>(.*?)</pre>.*?"
    r"<img src='img/([^']+)'",
    re.DOTALL,
)


def parse_detailed(html: str) -> list[dict]:
    out = []
    for mobj in CARD_RE.finditer(html):
        num, sym, times, cls, pct, reason, detail, fname = mobj.groups()
        sym = sym.strip()
        raw = pct.strip().replace("\u2212", "-")
        try:
            if raw.strip().startswith("-") or raw.strip().startswith("\u2212"):
                pnl = float(raw.replace("%", "").replace("\u2212", "-"))
            else:
                pnl = float(raw.replace("%", "").replace("+", ""))
        except ValueError:
            pnl = 0.0
        entry = exit_p = stop = None
        em = re.search(r"entry\s+([0-9.]+)", detail)
        xm = re.search(r"exit\s+([0-9.]+)", detail)
        sm = re.search(r"stop\s+([0-9.]+)", detail)
        if em:
            entry = float(em.group(1))
        if xm:
            exit_p = float(xm.group(1))
        if sm:
            stop = float(sm.group(1))
        entry_tpe = exit_tpe = ""
        parts = [p.strip() for p in times.replace("&rarr;", "→").split("→")]
        if parts:
            entry_tpe = parts[0].strip()
        if len(parts) >= 2:
            exit_tpe = parts[1].strip()
            if re.match(r"^\d{2}-\d{2}\s", parts[1].strip()) and entry_tpe[:4].isdigit():
                exit_tpe = entry_tpe[:4] + "-" + parts[1].strip()
        out.append(
            {
                "n": int(num),
                "symbol": sym,
                "entry_tpe": entry_tpe,
                "exit_tpe": exit_tpe,
                "pnl_pct": pnl,
                "reason": reason.strip(),
                "entry": entry,
                "exit": exit_p,
                "stop": stop,
                "fname": fname,
                "cls": cls,
                "usdt_100x3": round(pnl / 100.0 * 300.0, 4),
            }
        )
    return out


def probe_freshness(sample_syms: list[str]) -> str:
    notes = []
    latest = None
    for sym in sample_syms[:5]:
        try:
            df = m.fetch_klines(sym, "1h", days=3)
            if df is None or df.empty:
                notes.append(f"{sym}: empty")
                continue
            last = df.index[-1]
            notes.append(f"{sym} last={last.strftime('%Y-%m-%d %H:%M')} TPE n={len(df)}")
            if latest is None or last > latest:
                latest = last
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{sym}: {type(exc).__name__}:{exc}")
    ban = m._REST_BANNED
    if latest is None:
        return f"REST_BANNED={ban}; no sample bars. " + "; ".join(notes)
    return (
        f"REST_BANNED={ban}; sample last bar ~{latest.strftime('%Y-%m-%d %H:%M')} TPE. "
        + "; ".join(notes)
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    STAMPED.mkdir(parents=True, exist_ok=True)
    for old in list(IMG.glob("*.png")) + list(STAMPED.glob("*.png")):
        old.unlink()
    if HTML.is_file():
        HTML.unlink()

    print(f"scanned_at={datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} TPE", flush=True)
    print(f"module={m.__file__} out={OUT} REST_BANNED={m._REST_BANNED}", flush=True)

    try:
        syms_list = m.universe()
        source = "universe() REST exchangeInfo TRADIFI_PERPETUAL equity"
    except Exception as exc:  # noqa: BLE001
        print(f"universe REST failed: {exc}; falling back", flush=True)
        m._REST_BANNED = True
        # known equity perps from prior scans / user hint
        fallback = [x.strip() for x in Path("/workspace/ma25-retest/7d-20260922-stocks/symbols.txt").read_text().split(",") if x.strip()]
        syms_list = fallback
        source = f"fallback 7d-20260922-stocks/symbols.txt after universe fail: {exc}"

    crypto_ban = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"}
    syms_list = [s for s in syms_list if s.upper() not in crypto_ban]
    (OUT / "symbols.txt").write_text(",".join(syms_list), encoding="utf-8")
    print(f"n_symbols={len(syms_list)} source={source}", flush=True)
    print(f"top20={syms_list[:20]}", flush=True)

    freshness = probe_freshness(syms_list)
    print(f"freshness: {freshness}", flush=True)

    argv = [
        "--symbols",
        ",".join(syms_list),
        "--days",
        "7",
        "--limit",
        str(max(len(syms_list), 300)),
        "--workers",
        "6",
        "--lose-ma-closes",
        "0",
        "--max-hold",
        "36",
        "--m5-exit",
        "--require-h4-ma25",
        "--h4-ma25-tol",
        "1.5",
        "--max-prior-spike",
        "15",
        "--prior-spike-bars",
        "24",
        "--min-reclaim-entry-bars",
        "3",
        "--html",
        str(HTML),
    ]
    try:
        rc = m.main(argv)
    except ValueError as exc:
        print(f"WARN: m.main view_html skipped: {exc}", flush=True)
        rc = 0
        if HTML.is_file() and not (OUT / "view.html").is_file():
            (OUT / "view.html").write_text(
                HTML.read_text(encoding="utf-8"), encoding="utf-8"
            )
    if rc != 0:
        return rc

    if not HTML.is_file():
        print("ERROR: no HTML written", flush=True)
        return 1

    # refresh freshness after scan (REST may have flipped)
    freshness2 = probe_freshness(syms_list)
    html = HTML.read_text(encoding="utf-8")
    detailed = parse_detailed(html)
    trades = stamp.parse_trades(html)

    reasons = Counter(t["reason"] for t in detailed)
    n = len(detailed)
    wins = sum(1 for t in detailed if t["pnl_pct"] > 0)
    sum_pct = sum(t["pnl_pct"] for t in detailed)
    avg_pct = (sum_pct / n) if n else 0.0
    usdt_sum = sum(t["usdt_100x3"] for t in detailed)

    summary = {
        "scanned_at_tpe": datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S TPE"),
        "window": "2026-09-20 .. 2026-09-27 TPE (~7d via --days 7)",
        "universe": "STOCKS ONLY: TRADIFI_PERPETUAL + EQUITY/HK/KR/CN_EQUITY",
        "universe_source": source,
        "n_symbols": len(syms_list),
        "symbols": syms_list,
        "data_freshness": freshness2,
        "module": m.__file__,
        "rest_banned": bool(m._REST_BANNED),
        "rules": {
            "entry": "DETECT TG defaults + pretty ON: require_h4_ma25(tol=1.5%) + max_prior_spike=15%/24bars + min_reclaim_entry_bars=3",
            "detect": {
                "min_bars_below": 4,
                "max_bars_below": 36,
                "min_depth_pct": 0.018,
                "min_impulse_pct": 0.0,
                "min_undercut_pct": 0.014,
                "min_flush_atr": 0.0,
                "target_r": 2.0,
                "max_retest_wait": 18,
                "retest_tag_pct": 0.003,
                "retest_tag_atr": 0.4,
                "max_ma_drift_pct": 1.5,
                "min_ribbon_atr": 0.50,
                "ma200_near_pct": 1.0,
                "min_ma_cluster_atr": 1.0,
                "min_break_vol": 0.8,
                "min_mid_ma_atr": 0.50,
                "require_h4_ma25": True,
                "h4_ma25_tol_pct": 1.5,
                "max_prior_spike_pct": 0.15,
                "prior_spike_bars": 24,
                "min_reclaim_entry_bars": 3,
            },
            "stop": "1H close below break low",
            "lose_ma_closes": 0,
            "m5_exit": True,
            "max_hold": 36,
        },
        "n": n,
        "wins": wins,
        "winrate_pct": round(100.0 * wins / n, 2) if n else 0.0,
        "sum_pct": round(sum_pct, 4),
        "avg_pct": round(avg_pct, 4),
        "reason_counts": {
            "timeout": reasons.get("timeout", 0),
            "stop": reasons.get("stop", 0),
            "m5_neck": reasons.get("m5_neck", 0) + reasons.get("m5_m", 0),
            "lost_ma25": reasons.get("lost_ma25", 0),
            **{
                k: v
                for k, v in reasons.items()
                if k not in {"timeout", "stop", "m5_neck", "m5_m", "lost_ma25"}
            },
        },
        "usdt_100x3_additive": round(usdt_sum, 4),
        "margin_usdt": 100,
        "leverage": 3,
        "trades": detailed,
        "html": str(HTML),
        "img_dir": str(IMG),
        "stamped_dir": str(STAMPED),
        "stamped_paths": [],
    }

    print(
        f"summary prep n={n} WR={summary['winrate_pct']}% "
        f"sum={sum_pct:+.2f}% avg={avg_pct:+.2f}% usdt={usdt_sum:+.2f}",
        flush=True,
    )
    for t in detailed:
        print(
            f"  #{t['n']} {t['symbol']} {t['entry_tpe']} → {t['exit_tpe']} "
            f"{t['pnl_pct']:+.2f}% {t['reason']} entry={t['entry']} exit={t['exit']} stop={t['stop']}",
            flush=True,
        )

    stamped_paths = []
    if trades:
        written = set()
        for i, t in enumerate(trades, 1):
            src = IMG / t["fname"]
            if not src.is_file():
                print(f"missing img {src}", flush=True)
                continue
            dst = STAMPED / t["fname"]
            stamp.stamp(src, dst, t["display"], t["reason"], t["cls"])
            written.add(t["fname"])
            stamped_paths.append(str(dst))
            print(f"stamped {i} {dst.name} {t['display']} {t['reason']}", flush=True)
        for leftover in STAMPED.glob("*.png"):
            if leftover.name not in written:
                leftover.unlink()
        print(f"stamped_count={len(written)}", flush=True)
    else:
        print("no trades to stamp (stamp parse empty; detailed=%d)" % n, flush=True)

    summary["stamped_paths"] = stamped_paths
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
