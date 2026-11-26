#!/usr/bin/env python3
"""幣安股票永續，同一套 1m+5m+15m 撥開，回測近 30 天。

五分、十五分 K 由一分 K 對齊時鐘合成，不另外打 API。
出場兩套：固定 +1 點（跟 SOXS 同一支，股價差很多時不好比），以及等比例 +2%。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soxs_1m_ma200_peel import (  # noqa: E402
    FEE_PCT,
    INTERVAL_MS,
    MA_PERIODS,
    TZ,
    draw_stacked,
    entry_indices,
    hm,
    htf_aligned,
    ma60_reclaim,
    simulate,
    sma,
)

BASE = "https://www.binance.com"
EQUITY_TYPES = ("EQUITY", "HK_EQUITY", "KR_EQUITY", "CN_EQUITY")
INVERSE = {"SQQQUSDT", "SOXSUSDT", "SKDDUSDT", "TBTUSDT", "TZAUSDT"}
PCT_TARGET = 0.02
CACHE = Path("/tmp/stock_1m_cache")
OUT = Path("docs/binance-stocks-ma200-peel")


class Limiter:
    def __init__(self, per_min: float) -> None:
        self.gap = 60.0 / per_min
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
            self.next_at = time.monotonic() + self.gap


def equity_symbols() -> list[str]:
    info = requests.get(BASE + "/fapi/v1/exchangeInfo", timeout=30).json()
    syms = []
    for s in info.get("symbols") or []:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "TRADIFI_PERPETUAL":
            continue
        if s.get("underlyingType") not in EQUITY_TYPES:
            continue
        syms.append(s["symbol"])
    return sorted(syms)


def fetch_1m(symbol: str, start_ms: int, end_ms: int, limiter: Limiter) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}.npz"
    if path.is_file():
        z = np.load(path)
        if int(z["t"][0]) <= start_ms + 60_000 and int(z["t"][-1]) >= end_ms - 3 * 60_000:
            return {k: z[k] for k in ("t", "o", "h", "l", "c")}
    step = 60_000
    cur = start_ms
    rows: list = []
    sess = requests.Session()
    while cur < end_ms:
        limiter.wait()
        raw = None
        for attempt in range(6):
            r = sess.get(
                BASE + "/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": cur,
                    "endTime": end_ms - 1,
                    "limit": 1500,
                },
                timeout=25,
            )
            if r.status_code in (429, 418):
                time.sleep(float(r.headers.get("Retry-After") or 1.5 * (attempt + 1)))
                continue
            if r.status_code != 200:
                time.sleep(0.4 * (attempt + 1))
                continue
            raw = r.json()
            break
        if not raw:
            break
        rows.extend(raw)
        last = int(raw[-1][0])
        if len(raw) < 1500:
            break
        cur = last + step
    if not rows:
        return None
    seen = {int(x[0]): x for x in rows if int(x[0]) + step <= end_ms}
    kl = [seen[k] for k in sorted(seen)]
    d = {
        "t": np.array([int(x[0]) for x in kl], np.int64),
        "o": np.array([float(x[1]) for x in kl], np.float64),
        "h": np.array([float(x[2]) for x in kl], np.float64),
        "l": np.array([float(x[3]) for x in kl], np.float64),
        "c": np.array([float(x[4]) for x in kl], np.float64),
    }
    np.savez(path, **d)
    return d


def resample(d: dict, minutes: int) -> dict:
    step = minutes * 60_000
    b = (d["t"] // step) * step
    order = np.argsort(b, kind="mergesort")
    b = b[order]
    o, h, l, c = d["o"][order], d["h"][order], d["l"][order], d["c"][order]
    if len(b) == 0:
        empty = np.array([], dtype=np.float64)
        return {"t": np.array([], np.int64), "o": empty, "h": empty, "l": empty, "c": empty}
    cuts = np.flatnonzero(np.diff(b)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(b)]
    return {
        "t": b[starts],
        "o": o[starts],
        "h": np.maximum.reduceat(h, starts),
        "l": np.minimum.reduceat(l, starts),
        "c": c[ends - 1],
    }


def _pct(net: float, entry: float) -> float:
    return net / entry * 100.0 if entry else 0.0


def scan_symbol(symbol: str, d: dict, t_min: int) -> list[dict]:
    if len(d["c"]) < 500:
        return []
    d5 = resample(d, 5)
    d15 = resample(d, 15)
    if len(d15["c"]) < 220 or len(d5["c"]) < 220:
        return []
    c = d["c"]
    mas = {n: sma(c, n) for n in MA_PERIODS}
    m7, m14, m25, m30, m99, m120, m200 = (mas[n] for n in (7, 14, 25, 30, 99, 120, 200))
    mas5 = {n: sma(d5["c"], n) for n in MA_PERIODS}
    m15_200 = sma(d15["c"], 200)
    htf = (
        d5["t"],
        d5["l"],
        d5["c"],
        mas5[7],
        mas5[14],
        mas5[60],
        mas5[99],
        mas5[120],
        mas5[200],
        d15["t"],
        m15_200,
    )
    rows = []
    for i in entry_indices(c, m7, m14, m25, m30, m99, m120, m200, d["t"], t_min, htf=htf):
        pt = simulate(i, d["h"], d["l"], c, m200)
        pct = simulate(i, d["h"], d["l"], c, m200, target_pts=float(c[i]) * PCT_TARGET)
        t5, l5, c5, m5_7, m5_14, m5_60, m5_99, m5_120, m5_200, t15, m15 = htf
        _, gap5, s15 = htf_aligned(int(d["t"][i]), t5, m5_7, m5_14, m5_99, m5_120, m5_200, t15, m15)
        _, dist = ma60_reclaim(int(d["t"][i]), t5, l5, c5, m5_60, float(m200[i]))
        entry = float(c[i])
        rows.append(
            {
                "symbol": symbol,
                "inverse": symbol in INVERSE,
                "time": hm(int(d["t"][i])),
                "ms": int(d["t"][i]),
                "i": int(i),
                "entry": entry,
                "stop": pt["stop"],
                "risk": pt["risk"],
                "hold": pct["hold"],
                "scale_hold": pct["scale_hold"],
                "mfe_pct": pt["mfe"] / entry * 100.0,
                "pt_scalp": pt["scalp_net"],
                "pt_scale": pt["scale_net"],
                "pt_reason": pt["scalp_reason"],
                "scalp_reason": pct["scalp_reason"],
                "scale_reason": pct["scale_reason"],
                "scalp_pct": _pct(pct["scalp_net"], entry),
                "scale_pct": _pct(pct["scale_net"], entry),
                "gap5": gap5,
                "s15_200": s15,
                "ma60_dist": dist,
            }
        )
    return rows


def write_page(
    rows: list[dict],
    scanned: int,
    skipped: list[str],
    window: str,
    page: Path,
    *,
    title: str = "幣安股票永續，同一套撥開，近 30 天",
    blurb: str | None = None,
    quiet_names: list[str] | None = None,
) -> None:
    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)
    ranked = sorted(by_sym, key=lambda s: sum(x["scale_pct"] for x in by_sym[s]), reverse=True)
    n = len(rows)
    hit = sum(1 for r in rows if r["scalp_reason"] == "目標")
    sc = sum(r["scalp_pct"] for r in rows)
    sk = sum(r["scale_pct"] for r in rows)
    body = []
    for sym in ranked:
        xs = by_sym[sym]
        sk_s = sum(x["scale_pct"] for x in xs)
        sc_s = sum(x["scalp_pct"] for x in xs)
        wins = sum(1 for x in xs if x["scalp_reason"] == "目標")
        tag = "反向" if sym in INVERSE else ""
        cls = "pnl-win" if sk_s > 0 else "pnl-loss" if sk_s < 0 else "pnl-flat"
        body.append(
            f"<article class='trade-card'><header class='card-header'><div class='card-title'>"
            f"<span class='trade-no'>{sym}</span> <span class='trade-time'>{tag}</span></div>"
            f"<div class='card-pnl {cls}'>{sk_s:+.2f}%</div></header>"
            f"<pre class='trade-detail'>{len(xs)} 筆  目標 {wins}/{len(xs)}\n"
            f"整筆 +2% 扣費 {sc_s:+.2f}%\n一半 +2% 扣費 {sk_s:+.2f}%</pre>"
        )
        for r in xs:
            img = f"img/{sym.lower()}_{r['time'].replace('-', '').replace(' ', '_').replace(':', '')}.png"
            rcls = "pnl-win" if r["scale_pct"] > 0 else "pnl-loss"
            pt_line = ""
            if "pt_scale" in r:
                pt_line = f"\n固定 +1 點  整筆 {r['pt_scalp']:+.2f}  一半 {r['pt_scale']:+.2f}"
            body.append(
                f"<pre class='trade-detail'>{r['time']}  進 {r['entry']:.4g}\n"
                f"整筆 {r['scalp_reason']} {r['scalp_pct']:+.2f}%   一半 {r['scale_pct']:+.2f}%  "
                f"<span class='{rcls}'>最大 {r['mfe_pct']:.2f}%</span>{pt_line}</pre>"
                f"<div class='mini-chart'><img src='{img}' alt='{sym} {r['time']}' style='width:100%;display:block;border-radius:10px'/></div>"
            )
        body.append("</article>")
    for sym in quiet_names or []:
        if sym in by_sym or sym in skipped:
            continue
        tag = "反向" if sym in INVERSE else ""
        body.append(
            f"<article class='trade-card'><header class='card-header'><div class='card-title'>"
            f"<span class='trade-no'>{sym}</span> <span class='trade-time'>{tag}</span></div>"
            f"<div class='card-pnl pnl-flat'>沒進場</div></header>"
            f"<pre class='trade-detail'>這 30 天沒有同時滿足進場條件的 K 棒。</pre></article>"
        )
    quiet = scanned - len(by_sym) - len(skipped)
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{title}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px 8px;border-radius:10px;overflow:hidden}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{title}</h1>
<p class="muted">{blurb if blurb is not None else f"台北 {window}。美股、港股、韓股、陸股 USDT 永續共 {scanned} 檔。進場跟 SOXS 同一套。表上的百分比是等比例 +2% 目標、扣來回 {FEE_PCT:.2f}% 手續費。固定 +1 點只在 SOXS 那種價位有意義，這裡不拿來跨股票比。"}</p>
<div class="cards">
<div class="card">有進場<b>{len(by_sym)}</b></div>
<div class="card">筆數<b>{n}</b></div>
<div class="card">打到 +2%<b>{hit}/{n if n else 0}</b></div>
<div class="card">一半合計<b class="{'pnl-win' if sk>0 else 'pnl-loss'}">{sk:+.1f}%</b></div>
<div class="card">整筆合計<b class="{'pnl-win' if sc>0 else 'pnl-loss'}">{sc:+.1f}%</b></div>
<div class="card">沒進場<b>{quiet}</b></div>
</div>
<p class="muted">K 線不夠長、算不出 15 分 MA200 的有 {len(skipped)} 檔。</p>
</section>
{''.join(body)}
</div></body></html>
"""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(html, encoding="utf-8")
    if page.name == "index.html":
        (page.parent / "view.html").write_text(html, encoding="utf-8")
        (page.parent / "month.html").write_text(html, encoding="utf-8")


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = datetime.now(TZ).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    win_start = now - timedelta(days=30)
    t_min = int(win_start.timestamp() * 1000)
    fetch_start = int((win_start - timedelta(days=6)).timestamp() * 1000)
    syms = equity_symbols()
    print(f"股票永續 {len(syms)} 檔  {win_start.strftime('%m-%d %H:%M')} ～ {now.strftime('%m-%d %H:%M')}", flush=True)
    limiter = Limiter(150)
    done = 0
    packs: dict[str, dict | None] = {}

    def job(sym: str) -> tuple[str, dict | None]:
        try:
            return sym, fetch_1m(sym, fetch_start, end_ms, limiter)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {sym} {exc}", flush=True)
            return sym, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(job, s) for s in syms]
        for fut in as_completed(futs):
            sym, d = fut.result()
            packs[sym] = d
            done += 1
            if done % 10 == 0 or done == len(syms):
                print(f"下載 {done}/{len(syms)}", flush=True)

    rows: list[dict] = []
    skipped: list[str] = []
    chart_src: list[tuple[dict, dict, dict, dict[int, np.ndarray]]] = []
    for sym in syms:
        d = packs.get(sym)
        if d is None or len(d["c"]) < 500:
            skipped.append(sym)
            continue
        got = scan_symbol(sym, d, t_min)
        if not got:
            continue
        d5 = resample(d, 5)
        mas = {n: sma(d["c"], n) for n in MA_PERIODS}
        mas5 = {n: sma(d5["c"], n) for n in MA_PERIODS}
        for r in got:
            r["pack_i"] = len(chart_src)
            rows.append(r)
        chart_src.append((d, mas, d5, mas5))
        print(f"{sym} {len(got)} 筆  一半 {sum(x['scale_pct'] for x in got):+.2f}%", flush=True)

    img = OUT / "img"
    img.mkdir(parents=True, exist_ok=True)
    for old in img.glob("*.png"):
        old.unlink()
    for r in rows:
        d, mas, d5, mas5 = chart_src[r["pack_i"]]
        stamp = r["time"].replace("-", "").replace(" ", "_").replace(":", "")
        trade = {
            "i": r["i"],
            "ms": r["ms"],
            "time": r["time"],
            "entry": r["entry"],
            "stop": r["stop"],
            "hold": r["hold"],
            "scale_hold": r["scale_hold"],
            "scalp_reason": r["scalp_reason"],
            "scalp_net": r["scalp_pct"],
            "scale_reason": r["scale_reason"],
            "scale_net": r["scale_pct"],
            "mfe": r["mfe_pct"],
            "target_pts": r["entry"] * PCT_TARGET,
            "mark": "+2%",
            "unit": "%",
        }
        draw_stacked(img / f"{r['symbol'].lower()}_{stamp}.png", d, mas, d5, mas5, trade, 1)
    window = f"{win_start.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%m-%d %H:%M')}"
    slim = [{k: v for k, v in r.items() if k != "pack_i"} for r in rows]
    (OUT / "trades.json").write_text(json.dumps({"window": window, "scanned": len(syms), "skipped": skipped, "trades": slim}, ensure_ascii=False), encoding="utf-8")
    write_page(slim, len(syms), skipped, window, OUT / "index.html")
    hit = sum(1 for r in slim if r["scalp_reason"] == "目標")
    print(
        f"完成 掃描 {len(syms)}  有進場 {len({r['symbol'] for r in slim})}  筆數 {len(slim)}  "
        f"打到+2% {hit}  一半 {sum(r['scale_pct'] for r in slim):+.2f}%  整筆 {sum(r['scalp_pct'] for r in slim):+.2f}%",
        flush=True,
    )
    print(f"圖：{OUT / 'index.html'}", flush=True)


def rebuild_cached() -> None:
    """用已下載的 1 分 K 重算、重畫。視窗鎖住，避免快取被看成過期。"""
    end = datetime(2026, 11, 22, 11, 52, tzinfo=TZ)
    win_start = end - timedelta(days=30)
    t_min = int(win_start.timestamp() * 1000)
    fetch_start = int((win_start - timedelta(days=6)).timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    syms = equity_symbols()
    limiter = Limiter(120)
    rows: list[dict] = []
    skipped: list[str] = []
    chart_src: list[tuple] = []
    for n, sym in enumerate(syms, 1):
        d = fetch_1m(sym, fetch_start, end_ms, limiter)
        if d is None or len(d["c"]) < 500:
            skipped.append(sym)
            print(f"跳過 {sym}", flush=True)
            continue
        got = scan_symbol(sym, d, t_min)
        if got:
            d5 = resample(d, 5)
            mas = {k: sma(d["c"], k) for k in MA_PERIODS}
            mas5 = {k: sma(d5["c"], k) for k in MA_PERIODS}
            for r in got:
                r["pack_i"] = len(chart_src)
                rows.append(r)
            chart_src.append((d, mas, d5, mas5))
            print(f"{sym} {len(got)} 筆", flush=True)
        if n % 40 == 0:
            print(f"重算 {n}/{len(syms)}", flush=True)
    img = OUT / "img"
    img.mkdir(parents=True, exist_ok=True)
    for old in img.glob("*.png"):
        old.unlink()
    for r in rows:
        d, mas, d5, mas5 = chart_src[r["pack_i"]]
        stamp = r["time"].replace("-", "").replace(" ", "_").replace(":", "")
        trade = {
            "i": r["i"],
            "ms": r["ms"],
            "time": r["time"],
            "entry": r["entry"],
            "stop": r["stop"],
            "hold": r["hold"],
            "scale_hold": r["scale_hold"],
            "scalp_reason": r["scalp_reason"],
            "scalp_net": r["scalp_pct"],
            "scale_reason": r["scale_reason"],
            "scale_net": r["scale_pct"],
            "mfe": r["mfe_pct"],
            "target_pts": r["entry"] * PCT_TARGET,
            "mark": "+2%",
            "unit": "%",
        }
        draw_stacked(img / f"{r['symbol'].lower()}_{stamp}.png", d, mas, d5, mas5, trade, 1)
    window = f"{win_start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%m-%d %H:%M')}"
    slim = [{k: v for k, v in r.items() if k != "pack_i"} for r in rows]
    (OUT / "trades.json").write_text(
        json.dumps({"window": window, "scanned": len(syms), "skipped": skipped, "trades": slim}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_page(slim, len(syms), skipped, window, OUT / "index.html")
    hit = sum(1 for r in slim if r["scalp_reason"] == "目標")
    print(
        f"完成 掃描 {len(syms)}  跳過 {len(skipped)}  有進場 {len({r['symbol'] for r in slim})}  "
        f"筆數 {len(slim)}  打到+2% {hit}  "
        f"一半 {sum(r['scale_pct'] for r in slim):+.2f}%  整筆 {sum(r['scalp_pct'] for r in slim):+.2f}%",
        flush=True,
    )


def _norm_symbol(name: str) -> str:
    s = name.strip().upper()
    return s if s.endswith("USDT") else s + "USDT"


def run_named(names: list[str]) -> None:
    """同一套進場，只回測指定的股票永續。"""
    syms = [_norm_symbol(n) for n in names]
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    win_start = now - timedelta(days=30)
    t_min = int(win_start.timestamp() * 1000)
    fetch_start = int((win_start - timedelta(days=6)).timestamp() * 1000)
    out = Path("docs/soxs-soxl-ma200-peel")
    print(f"{' '.join(syms)}  {win_start.strftime('%m-%d %H:%M')} ～ {now.strftime('%m-%d %H:%M')}", flush=True)
    limiter = Limiter(150)
    rows: list[dict] = []
    skipped: list[str] = []
    scanned: list[str] = []
    chart_src: list[tuple] = []
    for sym in syms:
        d = fetch_1m(sym, fetch_start, end_ms, limiter)
        if d is None or len(d["c"]) < 500:
            skipped.append(sym)
            print(f"跳過 {sym}", flush=True)
            continue
        scanned.append(sym)
        got = scan_symbol(sym, d, t_min)
        if not got:
            print(f"{sym} 0 筆", flush=True)
            continue
        d5 = resample(d, 5)
        mas = {n: sma(d["c"], n) for n in MA_PERIODS}
        mas5 = {n: sma(d5["c"], n) for n in MA_PERIODS}
        for r in got:
            r["pack_i"] = len(chart_src)
            rows.append(r)
        chart_src.append((d, mas, d5, mas5))
        print(
            f"{sym} {len(got)} 筆  +2%一半 {sum(x['scale_pct'] for x in got):+.2f}%  "
            f"+1點一半 {sum(x['pt_scale'] for x in got):+.2f}",
            flush=True,
        )
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    for old in img.glob("*.png"):
        old.unlink()
    for r in rows:
        d, mas, d5, mas5 = chart_src[r["pack_i"]]
        stamp = r["time"].replace("-", "").replace(" ", "_").replace(":", "")
        trade = {
            "i": r["i"],
            "ms": r["ms"],
            "time": r["time"],
            "entry": r["entry"],
            "stop": r["stop"],
            "hold": r["hold"],
            "scale_hold": r["scale_hold"],
            "scalp_reason": r["scalp_reason"],
            "scalp_net": r["scalp_pct"],
            "scale_reason": r["scale_reason"],
            "scale_net": r["scale_pct"],
            "mfe": r["mfe_pct"],
            "target_pts": r["entry"] * PCT_TARGET,
            "mark": "+2%",
            "unit": "%",
        }
        draw_stacked(img / f"{r['symbol'].lower()}_{stamp}.png", d, mas, d5, mas5, trade, 1)
    window = f"{win_start.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%m-%d %H:%M')}"
    slim = [{k: v for k, v in r.items() if k != "pack_i"} for r in rows]
    (out / "trades.json").write_text(
        json.dumps({"window": window, "symbols": syms, "skipped": skipped, "trades": slim}, ensure_ascii=False),
        encoding="utf-8",
    )
    shown = "、".join(s.removesuffix("USDT") for s in syms)
    blurb = (
        f"台北 {window}。只跑 {shown}。進場條件跟全股票那次相同。"
        f"百分比是等比例 +2%、扣來回 {FEE_PCT:.2f}% 手續費。固定 +1 點寫在每一筆下面。"
    )
    write_page(
        slim,
        len(syms),
        skipped,
        window,
        out / "pair.html",
        title="SOXS 跟 SOXL，同一套撥開，近 30 天",
        blurb=blurb,
        quiet_names=scanned,
    )
    hit = sum(1 for r in slim if r["scalp_reason"] == "目標")
    print(
        f"完成 筆數 {len(slim)}  打到+2% {hit}  "
        f"+2%一半 {sum(r['scale_pct'] for r in slim):+.2f}%  "
        f"+1點一半 {sum(r['pt_scale'] for r in slim):+.2f}",
        flush=True,
    )
    print(f"圖：{out / 'pair.html'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        rebuild_cached()
    elif len(sys.argv) > 1:
        run_named(sys.argv[1:])
    else:
        main()
