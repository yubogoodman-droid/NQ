#!/usr/bin/env python3
"""回測：15 分 K MA7>14>25 多頭排列，且收盤站上日線 200 日均線。

    python3 examples/backtest_15m_bull.py --demo
    python3 examples/backtest_15m_bull.py --days 7 --pages
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.binance import fetch_klines, universe
from nq.ma15_bull import (
    HORIZONS,
    SignalRow,
    add_15m_mas,
    apply_filter,
    attach_daily_ma200,
    detect_15m_ma200_cross,
    detect_combo,
    fail_rate,
    forward_moves,
    sma,
    summarize_rows,
)

TZ = timezone(timedelta(hours=8))
DISPLAY = {"HK1810USDT": "小米"}
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}
HORIZON_LABEL = {1: "15m", 2: "30m", 4: "1h", 8: "2h", 16: "4h", 32: "8h"}
FILTERS = (
    ("原始：7>14>25 且收盤站上 200 日（組合剛成立）", {"crossed": None, "formed": None, "min_vol": None, "max_ext": None}),
    ("本根剛站上 200 日", {"crossed": True, "formed": None, "min_vol": None, "max_ext": None}),
    ("已在 200 日上，本根才多頭排列", {"crossed": None, "formed": True, "min_vol": None, "max_ext": None}),
    ("放量 ≥ 1.5×", {"crossed": None, "formed": None, "min_vol": 1.5, "max_ext": None}),
    ("剛站上 200 日 + 放量 ≥ 1.5×", {"crossed": True, "formed": None, "min_vol": 1.5, "max_ext": None}),
    ("距 200 日 ≤ 1.0%", {"crossed": None, "formed": None, "min_vol": None, "max_ext": 1.0}),
)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or f"s{abs(hash(symbol)) % 10_000_000_000}"


def sym_label(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    name = DISPLAY.get(symbol)
    return f"{name} {base}" if name else base


def pct(v: float | None) -> str:
    if v is None:
        return "—"
    cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    return f'<span class="{cls}">{v:+.2f}%</span>'


def scan_symbol(sym: str, days: int) -> tuple[str, dict | None, list[SignalRow], list[SignalRow]]:
    daily = fetch_klines(sym, interval="1d", limit=260, extra_bars=0)
    if daily is None or len(daily["c"]) < 200:
        return sym, None, [], []
    raw = fetch_klines(sym, interval="15m", days=days, extra_bars=48)
    if raw is None or len(raw["c"]) < 40:
        return sym, None, [], []
    d = attach_daily_ma200(add_15m_mas(raw), daily)
    cutoff = int(d["t"][-1]) - days * 24 * 60 * 60 * 1000

    def pack(hits) -> list[SignalRow]:
        rows = []
        for sig in hits:
            ts = int(d["t"][sig.idx])
            if ts < cutoff:
                continue
            entry, moves = forward_moves(d, sig)
            if np.isnan(entry):
                continue
            rows.append(SignalRow(symbol=sym, sig=sig, time_ms=ts, entry=entry, moves=moves))
        return rows

    return sym, d, pack(detect_combo(d)), pack(detect_15m_ma200_cross(d))


def filter_stats(rows: list[SignalRow]) -> list[dict]:
    out = []
    for name, kw in FILTERS:
        subset = apply_filter(rows, **kw)
        item = {"name": name, "count": len(subset), "fail15_pct": fail_rate(subset)}
        for h in HORIZONS:
            item[f"h{h}"] = summarize_rows(subset, h)
        out.append(item)
    return out


def draw_chart(sym: str, d: dict, row: SignalRow, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    i = row.sig.idx
    a0 = max(0, i - 48)
    a1 = min(len(d["c"]), i + 20)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
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
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, v, width=0.8, color=colors_v, linewidth=0)
    for n, col in PAL.items():
        if n == 200:
            continue
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.15, label=f"MA{n}")
    ax.plot(xs, d["m200d"][sl], color="#ffffff", lw=1.35, label="200d")
    x = i - a0
    ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
    ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    r4 = row.moves.get(16)
    rtxt = f"  4h {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else ""
    title_sym = file_base(sym) if any(ord(ch) >= 128 for ch in sym) else sym
    kind = "reclaim 200d" if row.crossed_200d else "7>14>25"
    ax.set_title(f"{title_sym}  15m  {hm(row.time_ms)}  {kind}{rtxt}", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def pick_gallery(rows: list[SignalRow], limit: int = 18) -> list[SignalRow]:
    pool = [r for r in rows if r.crossed_200d] or list(rows)
    if not pool:
        return []

    def score(r: SignalRow) -> float:
        mv = r.moves.get(16)
        return mv.ret_pct if mv and mv.ret_pct is not None else 0.0

    ranked = sorted(pool, key=score, reverse=True)
    winners = ranked[: max(1, limit * 2 // 3)]
    losers = list(reversed(ranked[len(winners) :]))[: limit - len(winners)]
    mixed = winners + losers
    seen: set[str] = set()
    out: list[SignalRow] = []
    for r in mixed:
        if r.symbol in seen and len(out) >= 8:
            continue
        seen.add(r.symbol)
        out.append(r)
        if len(out) >= limit:
            break
    if len(out) < min(limit, len(ranked)):
        for r in ranked:
            if r in out:
                continue
            out.append(r)
            if len(out) >= limit:
                break
    return out


def kpi_block(stats: list[dict]) -> str:
    raw = stats[0]
    cross = stats[1]
    h = cross["h16"]
    return "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in (
            ("通知／突破", cross["count"]),
            ("4h 勝率", f"{h['wr']:.1f}%"),
            ("4h 平均", f"{h['avg']:+.3f}%"),
            ("15m 假突破", f"{cross['fail15_pct']:.1f}%"),
            ("全部組合剛成立", raw["count"]),
            ("才形成多頭", stats[2]["count"]),
        )
    )


def stats_table(stats: list[dict]) -> str:
    heads = "".join(f"<th>{HORIZON_LABEL[h]}勝率</th><th>{HORIZON_LABEL[h]}均</th>" for h in (1, 4, 16, 32))
    rows = []
    for s in stats:
        cells = [f"<td>{html.escape(s['name'])}</td>", f'<td class="mono">{s["count"]}</td>', f'<td class="mono">{s["fail15_pct"]:.1f}%</td>']
        for h in (1, 4, 16, 32):
            st = s[f"h{h}"]
            cells.append(f'<td class="mono">{st["wr"]:.1f}%</td>')
            cells.append(f"<td>{pct(st['avg'])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr><th>條件</th><th>筆數</th><th>15m假突破</th>"
        + heads
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def signal_table(rows: list[SignalRow], limit: int = 80) -> str:
    focus = [r for r in rows if r.crossed_200d] or rows
    ranked = sorted(
        focus,
        key=lambda r: -(r.moves.get(16).ret_pct or -999) if r.moves.get(16) and r.moves[16].ret_pct is not None else 999,
    )[:limit]
    body = []
    for r in ranked:
        kind = "站上200日" if r.crossed_200d else "多頭排列"
        cells = [
            f'<td class="mono">{hm(r.time_ms)}</td>',
            f"<td>{html.escape(sym_label(r.symbol))}</td>",
            f"<td>{html.escape(kind)}</td>",
            f'<td class="mono">{r.entry:g}</td>',
            f'<td class="mono">{r.vol_ratio:.2f}</td>',
            f'<td class="mono">{r.ext_pct:+.2f}%</td>',
        ]
        for h in (1, 4, 16, 32):
            mv = r.moves.get(h)
            cells.append(f"<td>{pct(mv.ret_pct if mv else None)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr><th>時間</th><th>標的</th><th>種類</th><th>進場</th><th>量比</th><th>距200日</th>"
        "<th>15m</th><th>1h</th><th>4h</th><th>8h</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


IMG_BASE = (
    "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
    "cursor/15m-bull-ma200-e2b2/docs/binance/"
)
PUBLIC_PAGE = (
    "https://raw.githack.com/yubogoodman-droid/NQ/"
    "cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull.html"
)


def public_img_src(rel: str) -> str:
    """外網預覽要用絕對網址；GitHub raw 的 HTML 是 text/plain，不能拿來開頁。"""
    if rel.startswith(("data:", "http://", "https://")):
        return rel
    return IMG_BASE.rstrip("/") + "/" + rel.lstrip("./")


def gallery_html(gallery: list[tuple[SignalRow, str]]) -> str:
    if not gallery:
        return '<p class="note">這段期間沒有圖例。</p>'
    cards = []
    for row, rel in gallery:
        kind = "本根剛站上 200 日" if row.crossed_200d else "已在 200 日上，本根才 7>14>25"
        r4 = row.moves.get(16)
        rtxt = f"4h {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else ""
        src = public_img_src(rel)
        cards.append(
            f'<div class="card"><div class="cap">{html.escape(sym_label(row.symbol))} · {hm(row.time_ms)} · '
            f"{html.escape(kind)} · 量比 {row.vol_ratio:.2f} · {html.escape(rtxt)}</div>"
            f'<img src="{html.escape(src)}" alt="{html.escape(row.symbol)}"/></div>'
        )
    return "\n".join(cards)


def write_html(
    *,
    path: Path,
    days: int,
    universe_n: int,
    rows: list[SignalRow],
    stats: list[dict],
    gallery: list[tuple[SignalRow, str]],
    cross15_n: int,
    cross15_stats: dict,
) -> None:
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    first = datetime.fromtimestamp(min(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    last = datetime.fromtimestamp(max(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    c15 = (
        f"對照組（15m MA7&gt;14&gt;25 且剛站上 15m MA200，不是日線 200 日）"
        f"：{cross15_n} 筆，4h 勝率 {cross15_stats['wr']:.1f}%，4h 均 {cross15_stats['avg']:+.3f}%。"
    )
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>15分K 7/14/25 多頭 · 站上 200 日</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 14px 56px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:16px;margin:0 0 10px}}
.sub{{color:#8aa193;line-height:1.65;margin:0 0 16px}}
.card{{background:#14201b;border:1px solid rgba(232,240,234,.12);border-radius:12px;padding:14px;margin-bottom:16px}}
img{{width:100%;height:auto;display:block;border-radius:8px;background:#101814;margin:6px 0 10px}}
.note{{color:#8aa193;font-size:13px;margin:8px 0 0;line-height:1.5}}
.cap{{color:#c9a227;font-size:12px;margin:0 0 6px}}
.pos{{color:#3dba7a}}.neg{{color:#e35d5d}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0 18px}}
.kpi{{border:1px solid rgba(232,240,234,.12);border-radius:10px;padding:10px}}
.kpi .k{{color:#8aa193;font-size:12px}} .kpi .v{{font-size:18px;margin-top:4px}}
.table-wrap{{overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 8px;border-bottom:1px solid rgba(232,240,234,.12);white-space:nowrap;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
th{{color:#8aa193;font-size:11px;letter-spacing:.03em}}
@media(max-width:720px){{.kpis{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body>
<div class="wrap">
  <h1>15 分 K：7 / 14 / 25 多頭排列，收盤站上 200 日</h1>
  <p class="sub">
    規則：15 分 SMA7 &gt; SMA14 &gt; SMA25，且收盤高於<strong>日線 SMA200</strong>（用已收盤日 K，不偷看當天）。
    Telegram 通知只用「本根剛站上 200 日」：前收還在 200 日下，這一根收盤站上，同時短均已多頭排列。
    已在 200 日上、只是 7/14/25 又排一次的，會列入表內但不推播（近 7 天噪音太多）。
    進場用訊號下一根開盤。假突破 = 進場那根 15 分收盤又跌回 200 日下方。
    掃描幣安 U 本位流動永續近 {days} 天、{universe_n} 個合約。訊號區間 {first} → {last}（GMT+8）。產生於 {now}。
    {c15}
    外網請開 <a href="{PUBLIC_PAGE}" style="color:#c9a227">這頁（githack）</a>；GitHub raw / jsDelivr 會把 HTML 當純文字。
    僅供型態對照，不是進出場建議。
  </p>
  <div class="kpis">{kpi_block(stats)}</div>
  <h1>圖例</h1>
  <p class="sub">黃虛線是訊號 K。白線是日線 200 日均線對齊到 15 分圖。</p>
  {gallery_html(gallery)}
  <div class="card">
    <h2>過濾對照</h2>
    <div class="table-wrap">{stats_table(stats)}</div>
    <p class="note">距 200 日 =（收盤 / 日線 MA200 − 1）× 100%。量比 = 當根量 / 20 根均量。</p>
  </div>
  <div class="card">
    <h2>訊號表（剛站上 200 日，依 4h 報酬排序）</h2>
    <div class="table-wrap">{signal_table(rows)}</div>
  </div>
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def make_demo_bars() -> tuple[dict, dict]:
    n_d = 220
    daily_t = np.arange(n_d, dtype=np.int64) * 86_400_000
    daily_c = np.full(n_d, 100.0)
    daily = {
        "t": daily_t,
        "o": daily_c.copy(),
        "h": daily_c + 0.2,
        "l": daily_c - 0.2,
        "c": daily_c,
        "v": np.full(n_d, 1_000.0),
    }
    start = int(daily_t[210] + 86_400_000)
    n = 90
    t = start + np.arange(n, dtype=np.int64) * 15 * 60_000
    px = np.full(n, 99.15)
    for i in range(45, 78):
        px[i] = 99.15 + (i - 45) * 0.04
    px[78:] = 100.55
    o = np.roll(px, 1)
    o[0] = 99.15
    return {
        "t": t,
        "o": o,
        "h": np.maximum(o, px) + 0.05,
        "l": np.minimum(o, px) - 0.05,
        "c": px,
        "v": np.full(n, 1200.0),
    }, daily


def run_demo() -> int:
    raw, daily = make_demo_bars()
    d = attach_daily_ma200(add_15m_mas(raw), daily)
    hits = detect_combo(d)
    print(f"demo 偵測到 {len(hits)} 筆（7>14>25 且站上 200 日）")
    for sig in hits:
        print(
            f"  idx={sig.idx} close={sig.close:.3f} ma200d={sig.ma200d:.3f} "
            f"7>14>25={sig.m7:.3f}>{sig.m14:.3f}>{sig.m25:.3f} "
            f"crossed={sig.crossed_200d} formed={sig.formed_align}"
        )
    if not hits:
        print("demo 失敗：應該至少有一根組合成立")
        return 1
    if not any(s.crossed_200d or s.formed_align for s in hits):
        print("demo 失敗：訊號種類不對")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="15 分 K 7/14/25 多頭且收盤站上 200 日")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance/ma15-bull.html")
    p.add_argument("-o", "--output", help="HTML 輸出路徑")
    p.add_argument("--limit-symbols", type=int, default=0)
    args = p.parse_args()
    if args.demo:
        return run_demo()

    print("載入標的…", flush=True)
    symbols = universe()
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    print(f"掃描 {len(symbols)} 個 15m，近 {args.days} 天（需有 200 根日 K）", flush=True)

    rows: list[SignalRow] = []
    rows15: list[SignalRow] = []
    data: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(scan_symbol, s, args.days): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, hits, hits15 = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            if d is not None:
                data[sym] = d
            rows.extend(hits)
            rows15.extend(hits15)
            if done % 40 == 0 or done == len(symbols):
                print(
                    f"  {done}/{len(symbols)}  主規則 {len(rows)}  15mMA200對照 {len(rows15)}  {time.time()-t0:.1f}s",
                    flush=True,
                )
    rows.sort(key=lambda r: r.time_ms)
    stats = filter_stats(rows)
    print("\n=== 15 分 7/14/25 + 200 日 ===")
    for s in stats:
        h = s["h16"]
        print(
            f"{s['name']}: n={s['count']}  4h勝率 {h['wr']:.1f}%  4h均 {h['avg']:+.3f}%  "
            f"假突破 {s['fail15_pct']:.1f}%"
        )
    c15 = summarize_rows(rows15, 16)
    print(f"對照 15m MA200 剛站上: n={len(rows15)}  4h勝率 {c15['wr']:.1f}%  4h均 {c15['avg']:+.3f}%")

    img_dir = Path("docs/binance/img/ma15-bull")
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()
    gallery: list[tuple[SignalRow, str]] = []
    for row in pick_gallery(rows):
        d = data.get(row.symbol)
        if d is None:
            continue
        stamp = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m%d%H%M")
        fname = f"{file_base(row.symbol)}_{stamp}.png"
        out = img_dir / fname
        if draw_chart(row.symbol, d, row, out):
            gallery.append((row, f"./img/ma15-bull/{fname}"))

    out_html = Path(args.output) if args.output else Path("docs/binance/ma15-bull.html")
    if args.pages:
        out_html = Path("docs/binance/ma15-bull.html")
    write_html(
        path=out_html,
        days=args.days,
        universe_n=len(symbols),
        rows=rows,
        stats=stats,
        gallery=gallery,
        cross15_n=len(rows15),
        cross15_stats=c15,
    )
    Path("output").mkdir(exist_ok=True)
    Path("output/ma15_bull_summary.json").write_text(
        json.dumps(
            {
                "days": args.days,
                "universe": len(symbols),
                "signals": len(rows),
                "cross15m_ma200": len(rows15),
                "filters": stats,
                "html": str(out_html),
                "charts": len(gallery),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"\n已寫入 {out_html}  圖 {len(gallery)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
