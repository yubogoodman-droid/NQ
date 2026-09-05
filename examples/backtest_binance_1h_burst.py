#!/usr/bin/env python3
"""幣安 1h 多頭爆發回測：收盤進場，停在訊號 K 低、2R、或 N 根時間停。

    python3 examples/backtest_binance_1h_burst.py --days 3 --pages
    python3 examples/test_watch_binance_1h_burst.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_1h_burst import (  # noqa: E402
    MA_PERIODS,
    REPO,
    TZ,
    VOL_MULT,
    fetch_klines,
    find_bursts,
    hm,
    indicators,
    sma,
    universe,
)

PAGES = REPO / "docs" / "binance-1h-burst" / "index.html"
TARGET_R = 2.0
TIME_BARS = 8
MIN_RISK_PCT = 0.005


@dataclass
class Trade:
    symbol: str
    entry_idx: int
    exit_idx: int
    entry: float
    exit: float
    stop: float
    target: float
    pnl_pct: float
    reason: str
    vol_ratio: float
    green: bool
    fwd_1h: Optional[float]
    fwd_4h: Optional[float]
    fwd_8h: Optional[float]
    fwd_24h: Optional[float]
    d: dict


def _fwd(close: np.ndarray, i: int, n: int) -> Optional[float]:
    j = i + n
    if j >= len(close) or close[i] == 0:
        return None
    return float(close[j] / close[i] - 1.0)


def simulate_trade(
    d: dict,
    hit: dict,
    target_r: float = TARGET_R,
    time_bars: int = TIME_BARS,
    min_risk_pct: float = MIN_RISK_PCT,
) -> Trade:
    i = int(hit["i"])
    close, high, low = d["c"], d["h"], d["l"]
    entry = float(close[i])
    stop = float(low[i])
    if entry - stop < entry * min_risk_pct:
        stop = entry * (1.0 - min_risk_pct)
    risk = entry - stop
    if risk <= 0:
        stop = entry * (1.0 - min_risk_pct)
        risk = entry - stop
    target = entry + target_r * risk
    exit_idx, exit_px, reason = i, entry, "open"
    last = min(len(close) - 1, i + time_bars)
    for k in range(i + 1, last + 1):
        if float(low[k]) <= stop:
            exit_idx, exit_px, reason = k, stop, "stop"
            break
        if float(high[k]) >= target:
            exit_idx, exit_px, reason = k, target, "target"
            break
    else:
        if last > i:
            exit_idx, exit_px, reason = last, float(close[last]), "time"
            if last == len(close) - 1 and last < i + time_bars:
                reason = "open"
    return Trade(
        symbol="",
        entry_idx=i,
        exit_idx=exit_idx,
        entry=entry,
        exit=float(exit_px),
        stop=stop,
        target=target,
        pnl_pct=(float(exit_px) / entry - 1.0) if entry else 0.0,
        reason=reason,
        vol_ratio=float(hit["vol_ratio"]),
        green=float(hit["close"]) >= float(hit["open"]),
        fwd_1h=_fwd(close, i, 1),
        fwd_4h=_fwd(close, i, 4),
        fwd_8h=_fwd(close, i, 8),
        fwd_24h=_fwd(close, i, 24),
        d=d,
    )


def drop_overlap(trades: list[Trade]) -> list[Trade]:
    kept: list[Trade] = []
    busy = -1
    for t in sorted(trades, key=lambda x: x.entry_idx):
        if t.entry_idx <= busy:
            continue
        kept.append(t)
        busy = t.exit_idx
    return kept


def summarize(trades: list[Trade]) -> dict:
    n = len(trades)
    closed = [t for t in trades if t.reason != "open"]
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    closed_wins = sum(1 for t in closed if t.pnl_pct > 0)
    pcts = [t.pnl_pct for t in trades]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    def avg_fwd(attr: str) -> Optional[float]:
        xs = [getattr(t, attr) for t in trades if getattr(t, attr) is not None]
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "closed": len(closed),
        "open": n - len(closed),
        "closed_win_rate": 100.0 * closed_wins / len(closed) if closed else 0.0,
        "avg_pct": float(sum(pcts) / n) if n else 0.0,
        "total_pct": float(sum(pcts)),
        "reasons": reasons,
        "fwd_1h": avg_fwd("fwd_1h"),
        "fwd_4h": avg_fwd("fwd_4h"),
        "fwd_8h": avg_fwd("fwd_8h"),
        "fwd_24h": avg_fwd("fwd_24h"),
        "symbols": len({t.symbol for t in trades}),
        "green": sum(1 for t in trades if t.green),
    }


def backtest_symbol(
    sym: str,
    days: int,
    vol_mult: float,
    green_only: bool,
    time_bars: int,
    target_r: float,
    allow_overlap: bool,
) -> tuple[list[Trade], int]:
    need = 200 + days * 24 + time_bars + 30
    raw = fetch_klines(sym, limit=max(280, need))
    if raw is None:
        return [], 0
    d = indicators(raw)
    last = len(d["c"]) - 1
    start = max(MA_PERIODS[-1], last - days * 24 + 1)
    hits = find_bursts(d, start, last, vol_mult=vol_mult, green_only=green_only)
    trades = [simulate_trade(d, hit, target_r=target_r, time_bars=time_bars) for hit in hits]
    for t in trades:
        t.symbol = sym
    if not allow_overlap:
        trades = drop_overlap(trades)
    return trades, len(hits)


def _git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True)
        return out.strip() or "main"
    except Exception:
        return "main"


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _fmt_fwd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


def _draw_trade(t: Trade, path: Path, trade_no: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return
    i, x = t.entry_idx, t.exit_idx
    a0 = max(0, i - 36)
    a1 = min(len(t.d["c"]), max(x, i) + 6)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    d = t.d
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.4, 5.5), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    colors_v = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, v, width=0.8, color=colors_v, linewidth=0)
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}
    for n, col in pal.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    ax.axhline(t.stop, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(t.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ex, xx = i - a0, x - a0
    if 0 <= ex < len(c):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [t.entry], s=42, color="#00e676", marker="^", zorder=6)
    if 0 <= xx < len(c):
        ax.scatter(
            [xx],
            [t.exit],
            s=40,
            color="#00c805" if t.pnl_pct > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )
    et = hm(int(d["t"][i]))
    xt = hm(int(d["t"][x]))
    ax.set_title(
        f"#{trade_no}  {t.symbol}  {et} → {xt}  {t.reason}  {t.pnl_pct * 100:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_html(path: Path, trades: list[Trade], stats: dict, extra: dict, max_charts: int) -> Path:
    cards = []
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    for i, t in enumerate(trades, 1):
        et = hm(int(t.d["t"][t.entry_idx]))
        xt = hm(int(t.d["t"][t.exit_idx]))
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        side = "陽線" if t.green else "陰線"
        img_name = f"t{i:03d}_{t.symbol}_{et.replace(' ', '_').replace(':', '')}.png"
        img_html = ""
        if i <= max_charts:
            _draw_trade(t, img_dir / img_name, i)
            img_html = (
                f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(t.symbol)}' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(t.symbol)}</span>"
            f"<span class='trade-time'>{escape(et)} → {escape(xt)}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct * 100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(t.reason)}</span>"
            f"<span class='tag'>{side}</span>"
            f"<span class='tag'>量 {t.vol_ratio:.2f}×</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry:g}  stop {t.stop:g}  target {t.target:g}\n"
            f"exit {t.exit:g} {t.reason}  {t.pnl_pct * 100:+.2f}%\n"
            f"fwd +1h {_fmt_fwd(t.fwd_1h)}  +4h {_fmt_fwd(t.fwd_4h)}  "
            f"+8h {_fmt_fwd(t.fwd_8h)}  +24h {_fmt_fwd(t.fwd_24h)}"
            "</pre>"
            f"{img_html}"
            "</article>"
        )
    reasons = stats.get("reasons") or {}
    avg = stats["avg_pct"]
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 1h 多頭爆發 · {extra['days']} 天</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 1h 多頭爆發 · 近 {extra['days']} 天</h1>
<p class="muted">USDT 永續 · MA7&gt;14&gt;25&gt;99&gt;120&gt;200 且收盤量 &gt; 前一根 × {extra['vol_mult']:g}
<br/>收盤進場；停在訊號 K 低（太窄則 0.5%）、目標 {extra['target_r']:g}R、或 {extra['time_bars']} 根時間停。
同標的重疊訊號預設不重做。加總％是各筆相加，不是組合複利。
<br/>掃 {extra['scanned']} 檔 · 原始訊號 {extra['raw_hits']} · 進場 {stats['count']}
· 陽線 {stats['green']}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}
· 收盤後 +1h {_fmt_fwd(stats.get('fwd_1h'))} · +4h {_fmt_fwd(stats.get('fwd_4h'))}
· +8h {_fmt_fwd(stats.get('fwd_8h'))} · +24h {_fmt_fwd(stats.get('fwd_24h'))}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if avg >= 0 else 'pnl-loss'}">{avg * 100:+.2f}%</b></div>
<div class="card">標的<b>{stats['symbols']}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有多頭爆發訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def dump_hits(path: Path, trades: list[Trade], stats: dict, extra: dict) -> Path:
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_time": hm(int(t.d["t"][t.entry_idx])),
                "exit_time": hm(int(t.d["t"][t.exit_idx])),
                "entry": t.entry,
                "exit": t.exit,
                "stop": t.stop,
                "target": t.target,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
                "vol_ratio": t.vol_ratio,
                "green": t.green,
                "fwd_1h": t.fwd_1h,
                "fwd_4h": t.fwd_4h,
                "fwd_8h": t.fwd_8h,
                "fwd_24h": t.fwd_24h,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "extra": extra, "hits": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="幣安 1h 多頭爆發回測")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--vol-mult", type=float, default=VOL_MULT)
    p.add_argument("--green-only", action="store_true")
    p.add_argument("--overlap", action="store_true", help="同標的重疊訊號也做")
    p.add_argument("--time-bars", type=int, default=TIME_BARS)
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--json", dest="json_path", default="")
    p.add_argument("--max-charts", type=int, default=80)
    p.add_argument("--symbols", nargs="*")
    args = p.parse_args(argv)

    print("載入標的…", flush=True)
    symbols = list(args.symbols) if args.symbols else universe()
    print(
        f"回測 {len(symbols)} 檔 · {args.days}d 1h · 量>{args.vol_mult:g}× · "
        f"{args.time_bars} 根 / {args.target_r:g}R",
        flush=True,
    )
    trades: list[Trade] = []
    raw_hits = 0
    errors = 0
    with ThreadPoolExecutor(8) as ex:
        futs = {
            ex.submit(
                backtest_symbol,
                s,
                args.days,
                args.vol_mult,
                args.green_only,
                args.time_bars,
                args.target_r,
                args.overlap,
            ): s
            for s in symbols
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym = futs[fut]
            try:
                ts, n_hit = fut.result()
                trades.extend(ts)
                raw_hits += n_hit
            except Exception as e:
                errors += 1
                print("err", sym, e, flush=True)
            if done % 40 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)}  訊號 {raw_hits}  進場 {len(trades)}", flush=True)

    trades.sort(key=lambda t: (int(t.d["t"][t.entry_idx]), t.symbol))
    stats = summarize(trades)
    print(
        f"done errors={errors} raw={raw_hits} trades={stats['count']} "
        f"closedWR={stats['closed_win_rate']:.1f}% avg={stats['avg_pct']*100:+.2f}% "
        f"fwd1h={_fmt_fwd(stats['fwd_1h'])} fwd24h={_fmt_fwd(stats['fwd_24h'])}",
        flush=True,
    )
    for i, t in enumerate(trades, 1):
        print(
            f"  [{i:3d}] {t.symbol:16s} {hm(int(t.d['t'][t.entry_idx]))}  "
            f"{t.reason:6s} {t.pnl_pct*100:+6.2f}%  {t.vol_ratio:.2f}×",
            flush=True,
        )

    extra = {
        "days": args.days,
        "vol_mult": args.vol_mult,
        "green_only": args.green_only,
        "overlap": args.overlap,
        "time_bars": args.time_bars,
        "target_r": args.target_r,
        "scanned": len(symbols),
        "raw_hits": raw_hits,
        "generated": datetime.now(TZ).isoformat(timespec="seconds"),
    }
    html_path = Path(args.html) if args.html else (PAGES if args.pages else None)
    if html_path:
        out = write_html(html_path, trades, stats, extra, max_charts=max(0, args.max_charts))
        write_view_html(out)
        print(f"html={out}", flush=True)
    json_path = Path(args.json_path) if args.json_path else (html_path.with_name("hits.json") if html_path else None)
    if json_path:
        dump_hits(json_path, trades, stats, extra)
        print(f"json={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
