#!/usr/bin/env python3
"""回測：15 分 K 同時突破 MA7 / 14 / 25 / 99 / 120 / 200，圖例底下附同一時間 1 小時圖。

    python3 examples/backtest_15m_ribbon.py --demo
    python3 examples/backtest_15m_ribbon.py --days 7 --pages
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ribbon15 import (
    H1_FILTERS,
    H1_RECOMMENDED,
    HORIZONS,
    MA_PERIODS,
    SignalRow,
    add_mas,
    apply_filter,
    apply_h1_filter,
    detect_long_breaks,
    fail_rate,
    forward_moves,
    hourly_context,
    last_closed_h1_index,
    sma,
    summarize_rows,
)

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})

PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}
FILTERS = (
    ("原始：一根K從帶下收到帶上", {"body": None, "max_width": None, "min_width": None, "min_vol": None}),
    ("實體穿越（開在帶下、收在帶上）", {"body": True, "max_width": None, "min_width": None, "min_vol": None}),
    ("帶子寬度 ≤ 1.0%", {"body": None, "max_width": 1.0, "min_width": None, "min_vol": None}),
    ("帶子寬度 ≤ 0.4%", {"body": None, "max_width": 0.4, "min_width": None, "min_vol": None}),
    ("帶子寬度 > 1.0%", {"body": None, "max_width": None, "min_width": 1.0, "min_vol": None}),
    ("放量 ≥ 1.5×", {"body": None, "max_width": None, "min_width": None, "min_vol": 1.5}),
    ("實體 + 寬度≤1% + 放量≥1.5×", {"body": True, "max_width": 1.0, "min_width": None, "min_vol": 1.5}),
)
HORIZON_LABEL = {1: "15m", 2: "30m", 4: "1h", 8: "2h", 16: "4h", 32: "8h"}
BARS_PER_DAY = {"15m": 96, "5m": 288, "1m": 1440, "1h": 24}
INTERVAL_MS = {"15m": 15 * 60_000, "5m": 5 * 60_000, "1m": 60_000, "1h": 60 * 60_000}


def get_json(path: str, params=None, retries: int = 6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(1.4 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.45 * (i + 1))
    raise last


def universe() -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < 5_000_000 and sym not in KEEP:
            continue
        out.append(sym)
    return out


def fetch_klines(sym: str, *, days: int, interval: str = "15m") -> dict | None:
    bars_per_day = BARS_PER_DAY[interval]
    need = days * bars_per_day + max(MA_PERIODS) + 8
    chunks: list[list] = []
    end_time = None
    while sum(len(c) for c in chunks) < need:
        params = {"symbol": sym, "interval": interval, "limit": min(1500, need)}
        if end_time is not None:
            params["endTime"] = end_time
        raw = get_json("/fapi/v1/klines", params)
        if not raw:
            break
        chunks.append(raw)
        if len(raw) < params["limit"]:
            break
        end_time = int(raw[0][0]) - 1
    rows = []
    seen = set()
    for chunk in reversed(chunks):
        for x in chunk:
            t = int(x[0])
            if t in seen:
                continue
            seen.add(t)
            rows.append(x)
    if len(rows) < max(MA_PERIODS) + 5:
        return None
    now_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MS[interval]
    if int(rows[-1][0]) + interval_ms > now_ms:
        rows = rows[:-1]
    rows = rows[-need:]
    if len(rows) < max(MA_PERIODS) + 5:
        return None
    return {
        "t": np.array([int(x[0]) for x in rows], np.int64),
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
        "v": np.array([float(x[5]) for x in rows]),
    }


def scan_symbol(sym: str, days: int, interval: str = "15m") -> tuple[str, dict | None, list[SignalRow]]:
    raw = fetch_klines(sym, days=days, interval=interval)
    if raw is None:
        return sym, None, []
    d = add_mas(raw)
    cutoff = int(d["t"][-1]) - days * 24 * 60 * 60 * 1000
    rows = []
    for br in detect_long_breaks(d):
        ts = int(d["t"][br.idx])
        if ts < cutoff:
            continue
        entry, moves = forward_moves(d, br)
        if np.isnan(entry):
            continue
        rows.append(SignalRow(symbol=sym, break_=br, time_ms=ts, entry=entry, moves=moves))
    return sym, d, rows


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def pct(v: float | None) -> str:
    if v is None:
        return "—"
    cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    return f'<span class="{cls}">{v:+.2f}%</span>'


def draw_marked_chart(
    sym: str,
    d: dict,
    mark_idx: int,
    path: Path,
    *,
    title: str,
    before: int = 48,
    after: int = 20,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    i = mark_idx
    if i < 0 or i >= len(d["c"]):
        return None
    a0 = max(0, i - before)
    a1 = min(len(d["c"]), i + after)
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
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    x = i - a0
    ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
    ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    ax.set_title(title, color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_chart(
    sym: str,
    d: dict,
    row: SignalRow,
    path: Path,
    *,
    interval: str,
    summary_h: int,
    summary_label: str,
) -> Path | None:
    r4 = row.moves.get(summary_h)
    rtxt = f"  {summary_label} {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else ""
    return draw_marked_chart(
        sym,
        d,
        row.break_.idx,
        path,
        title=f"{sym}  {interval}  {hm(row.time_ms)}{rtxt}",
    )


def h1_index(d: dict, ts_ms: int) -> int | None:
    t = d["t"]
    i = int(np.searchsorted(t, ts_ms, side="right") - 1)
    if i < 0 or i >= len(t):
        return None
    return i


def filter_stats(rows: list[SignalRow]) -> list[dict]:
    out = []
    for name, kw in FILTERS:
        subset = apply_filter(rows, **kw)
        out.append(_subset_stats(name, subset))
    return out


def h1_filter_stats(rows: list[SignalRow]) -> list[dict]:
    out = [_subset_stats("原始（有 1h 狀態）", [r for r in rows if r.hourly is not None])]
    out[0]["key"] = ""
    for name, key in H1_FILTERS:
        item = _subset_stats(name, apply_h1_filter(rows, key))
        item["key"] = key
        out.append(item)
    return out


def _subset_stats(name: str, subset: list[SignalRow]) -> dict:
    item = {
        "name": name,
        "count": len(subset),
        "symbols": len({r.symbol for r in subset}),
        "fail15_pct": round(fail_rate(subset), 1),
        "key": "",
    }
    for h in HORIZONS:
        item[f"h{h}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize_rows(subset, h).items()}
    return item


def pick_h1_rule(h1_stats: list[dict], *, min_n: int) -> dict | None:
    """預先指定的濾網；沒資料時才退回門檻挑選。"""
    by_key = {s.get("key"): s for s in h1_stats if s.get("key")}
    for _name, key in H1_RECOMMENDED:
        s = by_key.get(key)
        if s and s["count"] >= max(8, min_n // 2):
            return s
    if not h1_stats:
        return None
    raw = h1_stats[0]
    raw_wr = raw["h16"]["wr"]
    raw_avg = raw["h16"]["avg"]
    ranked = []
    for s in h1_stats[1:]:
        if s["name"].startswith("（對照）"):
            continue
        if s["count"] < min_n:
            continue
        wr, avg = s["h16"]["wr"], s["h16"]["avg"]
        if wr < raw_wr + 3.0:
            continue
        if avg < raw_avg - 0.15:
            continue
        ranked.append(s)
    if not ranked:
        return None
    ranked.sort(key=lambda s: (s["h16"]["wr"], s["count"], s["h16"]["avg"]), reverse=True)
    return ranked[0]


def h1_recs_from_stats(h1_stats: list[dict]) -> list[dict]:
    by_key = {s.get("key"): s for s in h1_stats if s.get("key")}
    out = []
    for _name, key in H1_RECOMMENDED:
        s = by_key.get(key)
        if s:
            out.append(s)
    return out


def pick_gallery(rows: list[SignalRow], limit: int = 24, *, summary_h: int = 16) -> list[SignalRow]:
    body = apply_filter(rows, body=True) or rows

    def rsum(r: SignalRow) -> float:
        m = r.moves.get(summary_h)
        return m.ret_pct if m and m.ret_pct is not None else -1e9

    ranked = sorted(body, key=rsum, reverse=True)
    tight = sorted(apply_filter(body, max_width=0.4), key=rsum, reverse=True)
    n_best = min(8, len(ranked))
    n_worst = min(8, max(0, len(ranked) - n_best))
    chosen = ranked[:n_best] + ranked[-n_worst:] if n_worst else ranked[:n_best]
    for r in tight[:8]:
        chosen.append(r)
    seen = set()
    out = []
    for r in chosen:
        key = (r.symbol, r.time_ms)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def h1_caption(row: SignalRow) -> str:
    h = row.hourly
    if h is None:
        return ""
    loc = "帶下" if h.below_ribbon else ("帶上" if h.above_ribbon else "帶內")
    hour = "綠小時" if h.forming_green else "紅小時"
    ma = "MA200上" if h.px_above_ma200 else "MA200下"
    return f"1h {loc} · {hour} · 15m收在1h {ma}（{h.dist_ma200_pct:+.2f}%）"


def kpi_block(stats: list[dict], *, summary_h: int = 16, h1_recs: list[dict] | None = None) -> str:
    if not stats:
        return ""
    by = {s["name"]: s for s in stats}
    raw = stats[0]
    tight = by.get("帶子寬度 ≤ 1.0%", stats[min(2, len(stats) - 1)])
    wide = by.get("帶子寬度 > 1.0%", raw)
    h = raw[f"h{summary_h}"]
    cards = [
        ("原始筆數", str(raw["count"])),
        ("4h 勝率", f"{h['wr']:.1f}%"),
        ("4h 平均", f"{h['avg']:+.2f}%"),
        ("4h 中位", f"{h['med']:+.2f}%"),
        ("寬度≤1%", str(tight["count"])),
        ("寬度>1% 4h均", f"{wide[f'h{summary_h}']['avg']:+.2f}%"),
    ]
    if h1_recs:
        for s in h1_recs[:2]:
            bh = s[f"h{summary_h}"]
            label = "嚴：1h已站上帶4h勝率" if s.get("key") == "above_ribbon" else "寬：價上MA200帶緊4h勝率"
            cards.append((label, f"{bh['wr']:.1f}%"))
            cards.append((f"{label}筆數".replace("4h勝率", ""), str(s["count"])))
    html_cards = []
    for k, v in cards:
        cls = ""
        if k in ("4h 平均", "4h 中位", "寬度>1% 4h均"):
            try:
                cls = "pos" if float(v.strip("%").replace("+", "")) > 0 else "neg"
            except ValueError:
                cls = ""
        if "勝率" in k:
            try:
                cls = "pos" if float(v.strip("%")) >= 50 else "neg"
            except ValueError:
                cls = ""
        html_cards.append(f'<div class="kpi"><div class="k">{html.escape(k)}</div><div class="v {cls}">{html.escape(v)}</div></div>')
    return "\n".join(html_cards)


def stats_table(stats: list[dict], *, horizon_label: dict[int, str] | None = None) -> str:
    labels = horizon_label or HORIZON_LABEL
    heads = ["條件", "筆數", "幣數", "下一根假突破"]
    for h in HORIZONS:
        heads += [f"{labels[h]}勝率", f"{labels[h]}均"]
    thead = "".join(f"<th>{x}</th>" for x in heads)
    body = []
    for s in stats:
        tds = [
            html.escape(s["name"]),
            str(s["count"]),
            str(s["symbols"]),
            f"{s['fail15_pct']:.1f}%",
        ]
        for h in HORIZONS:
            block = s[f"h{h}"]
            tds.append(f"{block['wr']:.1f}%" if block["n"] else "—")
            tds.append(f"{block['avg']:+.3f}%" if block["n"] else "—")
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in tds) + "</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def signal_table(
    rows: list[SignalRow],
    limit: int = 80,
    *,
    summary_h: int = 16,
    table_hs: tuple[int, ...] = (1, 4, 16, 32),
    horizon_label: dict[int, str] | None = None,
) -> str:
    labels = horizon_label or HORIZON_LABEL
    ranked = sorted(
        rows,
        key=lambda r: (r.moves.get(summary_h).ret_pct if r.moves.get(summary_h) and r.moves[summary_h].ret_pct is not None else -1e9),
        reverse=True,
    )[:limit]
    heads = ["時間", "標的", "進場", "寬度", "量比", "實體"] + [labels[h] for h in table_hs]
    thead = "".join(f"<th>{x}</th>" for x in heads)
    body = []
    for r in ranked:
        body_tag = "是" if r.body_through else ""
        tds = [
            hm(r.time_ms),
            r.symbol.replace("USDT", ""),
            f"{r.entry:g}",
            f"{r.width_pct:.2f}%",
            f"{r.vol_ratio:.2f}×",
            body_tag,
        ]
        for h in table_hs:
            tds.append(pct(r.moves.get(h).ret_pct if r.moves.get(h) else None))
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in tds) + "</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def public_img_src(rel: str) -> str:
    """GitHub Pages 用相對路徑；和 docs/binance/img/ 放一起。"""
    if rel.startswith(("data:", "http://", "https://")):
        return rel
    return rel


def gallery_html(
    items: list[tuple[SignalRow, str, str | None]],
    *,
    summary_h: int = 16,
    table_hs: tuple[int, ...] = (1, 4, 16, 32),
    horizon_label: dict[int, str] | None = None,
) -> str:
    labels = horizon_label or HORIZON_LABEL
    note_keys = " / ".join(labels[h] for h in table_hs)
    cards = []
    for row, rel15, rel1h in items:
        r4 = row.moves.get(summary_h)
        ret = r4.ret_pct if r4 else None
        cls = "pos" if ret and ret > 0 else "neg"
        ret_s = f"{ret:+.2f}%" if ret is not None else "—"
        note = (
            f"寬度 {row.width_pct:.3f}% · 量比 {row.vol_ratio:.2f}× · "
            f"{note_keys}："
            + " / ".join(_plain(row, h) for h in table_hs)
        )
        if row.body_through:
            note += " · 實體穿越"
        cap = h1_caption(row)
        if cap:
            note += " · " + cap
        src15 = public_img_src(rel15)
        h1 = ""
        if rel1h:
            src1h = public_img_src(rel1h)
            h1 = f"""
  <p class="cap">1 小時圖 · 黃虛線是上面那根 15 分所在的小時 K</p>
  <img src="{html.escape(src1h)}" alt="{html.escape(row.symbol)} 1h {hm(row.time_ms)}"/>"""
        cards.append(
            f"""<div class="card">
  <h2>{html.escape(row.symbol.replace("USDT",""))} {hm(row.time_ms)} · 4h <span class="{cls}">{ret_s}</span></h2>
  <p class="cap">15 分圖 · 黃虛線是同時穿過六條均線的那根</p>
  <img src="{html.escape(src15)}" alt="{html.escape(row.symbol)} {hm(row.time_ms)}"/>
  {h1}
  <p class="note">{html.escape(note)}</p>
</div>"""
        )
    return "\n".join(cards)


def _plain(row: SignalRow, h: int) -> str:
    m = row.moves.get(h)
    if not m or m.ret_pct is None:
        return "—"
    return f"{m.ret_pct:+.2f}%"


def write_html(
    *,
    path: Path,
    days: int,
    universe_n: int,
    rows: list[SignalRow],
    stats: list[dict],
    h1_stats: list[dict],
    h1_recs: list[dict],
    h1_week_stats: list[dict] | None,
    gallery: list[tuple[SignalRow, str, str | None]],
) -> None:
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    first = datetime.fromtimestamp(min(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    last = datetime.fromtimestamp(max(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    rec_bits = []
    for s in h1_recs:
        rec_bits.append(
            f"{html.escape(s['name'])}：{s['count']} 筆、4h 勝率 {s['h16']['wr']:.1f}%、均 {s['h16']['avg']:+.2f}%"
        )
    rec_txt = "；".join(rec_bits) if rec_bits else "沒有預先指定的 1h 濾網可顯示。"
    week_block = ""
    if h1_week_stats:
        week_block = f"""
  <div class="card">
    <h2>最近 7 天同一組 1h 濾網（容易過擬合，只拿來對照）</h2>
    <div class="table-wrap">
      {stats_table(h1_week_stats)}
    </div>
    <p class="note">這週「1h 收在 MA99 上」看起來很好，拉長到 {days} 天後差距會收掉。不要只看一週就定規則。</p>
  </div>"""
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>15分K 同時突破 7/14/25/99/120/200</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 14px 56px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:16px;margin:0 0 10px}}
.sub{{color:#8aa193;line-height:1.65;margin:0 0 16px}}
.card{{background:#14201b;border:1px solid rgba(232,240,234,.12);border-radius:12px;padding:14px;margin-bottom:16px}}
img{{width:100%;height:auto;display:block;border-radius:8px;background:#101814;margin:6px 0 10px}}
.note{{color:#8aa193;font-size:13px;margin:8px 0 0;line-height:1.5}}
.cap{{color:#c9a227;font-size:12px;margin:10px 0 0}}
.pos{{color:#3dba7a}}.neg{{color:#e35d5d}}.mark{{color:#c9a227}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 18px}}
.kpi{{border:1px solid rgba(232,240,234,.12);border-radius:10px;padding:10px}}
.kpi .k{{color:#8aa193;font-size:12px}} .kpi .v{{font-size:18px;margin-top:4px}}
.table-wrap{{overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 8px;border-bottom:1px solid rgba(232,240,234,.12);white-space:nowrap;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{color:#8aa193;font-size:11px;letter-spacing:.03em}}
@media(max-width:720px){{.kpis{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body>
<div class="wrap">
  <h1>一根 15 分 K 同時突破 7 / 14 / 25 / 99 / 120 / 200</h1>
  <p class="sub">
    前一根收盤完全在六條均線下方，這一根收盤完全站上六條均線。進場用訊號下一根開盤。
    掃描幣安 U 本位永續近 {days} 天、{universe_n} 個流動合約。訊號區間 {first} → {last}（GMT+8）。產生於 {now}。
    圖例每筆上面是 15 分，底下是同一時間的 1 小時圖。僅供型態對照，不是進出場建議。
  </p>
  <div class="kpis">
    {kpi_block(stats, summary_h=16, h1_recs=h1_recs)}
  </div>
  <p class="sub">
    1 小時濾網只用訊號當下已收完的小時 K，不偷看還在走的那根收盤。
    能把 4h 勝率拉開、而且 7 天與 {days} 天方向一致的：{rec_txt}。
    代價是成交變少。1h 還在整條帶下的 15 分穿刺，短線勝率不一定差，平均常被少數大漲拉高，但不穩。
  </p>
  <h1>圖例</h1>
  <p class="sub">黃虛線：15 分圖是穿越六條均線的那根；1 小時圖是這根 15 分所在的小時 K。圖例優先抽「1h 已站上帶」和「價上 1h MA200 且帶子不寬」。</p>
  {gallery_html(gallery)}
  <div class="card">
    <h2>用 1 小時當濾網（{days} 天）</h2>
    <p class="note" style="margin:0 0 10px">只看訊號當下已收完的 1 小時 K。當根小時「已翻綠」= 這根 15 分收盤已經高於該小時開盤，幾乎每筆都會過，所以沒有區分度。</p>
    <div class="table-wrap">
      {stats_table(h1_stats)}
    </div>
  </div>
  {week_block}
  <div class="card">
    <h2>15 分自己的過濾對照</h2>
    <div class="table-wrap">
      {stats_table(stats)}
    </div>
    <p class="note">寬度 =（最高均線 / 最低均線 − 1）× 100%，用突破前一根的均線。假突破 = 進場那根 15 分收盤又跌回最高均線下方。</p>
  </div>
  <div class="card">
    <h2>訊號表（依 4h 報酬排序，最多 80 筆）</h2>
    <div class="table-wrap">
      {signal_table(rows)}
    </div>
  </div>
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def make_demo_bars() -> dict:
    """先把六條均線黏在 100，再一根大陽線從帶下收到帶上。"""
    n = 240
    px = np.full(n, 100.0)
    o = np.full(n, 100.0)
    h = np.full(n, 100.05)
    l = np.full(n, 99.95)
    v = np.full(n, 1000.0)
    # 200：從均線堆跌到下方
    o[200], l[200], h[200], px[200] = 100.0, 98.90, 100.02, 99.00
    v[200] = 1800
    # 201：同一根打穿整條帶子
    o[201], l[201], h[201], px[201] = 99.05, 98.95, 101.60, 101.40
    v[201] = 3200
    o[202], l[202], h[202], px[202] = 101.40, 101.20, 101.70, 101.55
    return {
        "t": np.arange(n, dtype=np.int64) * 15 * 60_000,
        "o": o,
        "h": h,
        "l": l,
        "c": px,
        "v": v,
    }


def run_demo() -> int:
    d = add_mas(make_demo_bars())
    hits = detect_long_breaks(d)
    print(f"demo 偵測到 {len(hits)} 筆")
    for br in hits:
        print(
            f"  idx={br.idx} close={br.close:.3f} width={br.width_pct:.3f}% "
            f"body={br.body_through} vol={br.vol_ratio:.2f}"
        )
    if not hits:
        print("demo 失敗：應該至少有一根穿過黏帶")
        return 1
    t = np.array([0, 3_600_000, 7_200_000], np.int64)
    # 16:30 這根 15m 收在 16:45，最後收完的小時是 15:00。
    if last_closed_h1_index(t, 5_400_000) != 0:
        print("demo 失敗：last_closed_h1_index 16:30")
        return 1
    # 16:45 這根 15m 收在 17:00，16:00 那根小時剛好收完。
    if last_closed_h1_index(t, 6_300_000) != 1:
        print("demo 失敗：last_closed_h1_index 16:45")
        return 1
    return 0


def scan_interval(symbols: list[str], days: int, interval: str, workers: int) -> tuple[list[SignalRow], dict[str, dict]]:
    rows: list[SignalRow] = []
    data: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(scan_symbol, s, days, interval): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, hits = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            if d is not None:
                data[sym] = d
            rows.extend(hits)
            if done % 40 == 0 or done == len(symbols):
                print(f"  {interval} {done}/{len(symbols)}  訊號 {len(rows)}  {time.time()-t0:.1f}s", flush=True)
    rows.sort(key=lambda r: r.time_ms)
    return rows, data


def fetch_h1_map(symbols: list[str], days: int, workers: int) -> dict[str, dict]:
    data_1h: dict[str, dict] = {}
    if not symbols:
        return data_1h
    print(f"補抓 {len(symbols)} 個 1h K…", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_klines, s, days=days, interval="1h"): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym = futs[fut]
            try:
                raw = fut.result()
            except Exception as e:
                print("err 1h", sym, e, flush=True)
                continue
            if raw is not None:
                data_1h[sym] = add_mas(raw)
            if done % 40 == 0 or done == len(symbols):
                print(f"  1h {done}/{len(symbols)}", flush=True)
    return data_1h


def attach_hourly(rows: list[SignalRow], data_1h: dict[str, dict]) -> int:
    n = 0
    for r in rows:
        d1 = data_1h.get(r.symbol)
        if d1 is None:
            continue
        r.hourly = hourly_context(d1, r.time_ms, r.break_.close)
        if r.hourly is not None:
            n += 1
    return n


def print_filter_table(title: str, stats: list[dict]) -> None:
    print(f"\n=== {title} ===")
    for s in stats:
        h = s["h16"]
        print(
            f"{s['name']}: n={s['count']}  4h勝率 {h['wr']:.1f}%  4h均 {h['avg']:+.3f}%  "
            f"中位 {h['med']:+.3f}%  假突破 {s['fail15_pct']:.1f}%"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="15 分 K 同時突破六條均線（圖例附 1 小時）")
    p.add_argument("--demo", action="store_true", help="只用合成資料驗證偵測")
    p.add_argument("--days", type=int, default=7, help="回測天數（訊號窗口；前面另留 MA200 熱身）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance/ma-ribbon-15m.html")
    p.add_argument("-o", "--output", help="HTML 輸出路徑")
    p.add_argument("--limit-symbols", type=int, default=0, help="除錯用，只掃前 N 個幣")
    p.add_argument("--skip-charts", action="store_true", help="只算統計、不畫圖")
    args = p.parse_args()
    if args.demo:
        return run_demo()

    print("載入標的…", flush=True)
    symbols = universe()
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    print(f"掃描 {len(symbols)} 個 15m 合約，近 {args.days} 天", flush=True)
    rows, data = scan_interval(symbols, args.days, "15m", args.workers)
    stats = filter_stats(rows)
    print_filter_table("15 分 過濾對照", stats)

    need_1h = sorted({r.symbol for r in rows})
    data_1h = fetch_h1_map(need_1h, args.days, args.workers)
    n_h1 = attach_hourly(rows, data_1h)
    print(f"對上 1h 狀態 {n_h1}/{len(rows)} 筆", flush=True)
    h1_stats = h1_filter_stats(rows)
    print_filter_table("1 小時濾網", h1_stats)
    h1_recs = h1_recs_from_stats(h1_stats)
    h1_week_stats = None
    if args.days >= 14 and data:
        end_ms = max(int(d["t"][-1]) for d in data.values())
        week_cut = end_ms - 7 * 24 * 60 * 60 * 1000
        week_rows = [r for r in rows if r.time_ms >= week_cut]
        h1_week_stats = h1_filter_stats(week_rows)
        print_filter_table("最近 7 天 1 小時濾網", h1_week_stats)
    for s in h1_recs:
        print(
            f"預設濾網：{s['name']}  n={s['count']}  "
            f"4h勝率 {s['h16']['wr']:.1f}%  均 {s['h16']['avg']:+.3f}%"
        )
    gallery_src: list[SignalRow] = []
    seen_g = set()
    for _name, key in H1_RECOMMENDED:
        for r in apply_h1_filter(rows, key):
            k = (r.symbol, r.time_ms)
            if k in seen_g:
                continue
            seen_g.add(k)
            gallery_src.append(r)
    if not gallery_src:
        gallery_src = rows

    img15 = Path("docs/binance/img/ribbon15")
    img1h = Path("docs/binance/img/ribbon1h")
    gallery: list[tuple[SignalRow, str, str | None]] = []
    if not args.skip_charts:
        img15.mkdir(parents=True, exist_ok=True)
        img1h.mkdir(parents=True, exist_ok=True)
        for old in list(img15.glob("*.png")) + list(img1h.glob("*.png")):
            old.unlink()
        gallery_rows = pick_gallery(gallery_src, summary_h=16)
        extra = pick_gallery(rows, summary_h=16)
        seen = {(r.symbol, r.time_ms) for r in gallery_rows}
        for r in extra:
            if (r.symbol, r.time_ms) in seen:
                continue
            gallery_rows.append(r)
            seen.add((r.symbol, r.time_ms))
            if len(gallery_rows) >= 24:
                break
        for row in gallery_rows:
            d = data.get(row.symbol)
            if d is None:
                continue
            stamp = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m%d%H%M")
            base = row.symbol.replace("USDT", "")
            fname15 = f"{base}_{stamp}.png"
            out15 = img15 / fname15
            if not draw_chart(
                row.symbol,
                d,
                row,
                out15,
                interval="15m",
                summary_h=16,
                summary_label="4h",
            ):
                continue
            rel1h = None
            d1 = data_1h.get(row.symbol)
            if d1 is not None:
                hi = h1_index(d1, row.time_ms)
                if hi is not None:
                    fname1h = f"{base}_{stamp}_1h.png"
                    out1h = img1h / fname1h
                    if draw_marked_chart(
                        row.symbol,
                        d1,
                        hi,
                        out1h,
                        title=f"{row.symbol}  1h  {hm(row.time_ms)}",
                        before=36,
                        after=12,
                    ):
                        rel1h = f"./img/ribbon1h/{fname1h}"
            gallery.append((row, f"./img/ribbon15/{fname15}", rel1h))

    out_html = Path(args.output) if args.output else Path("docs/binance/ma-ribbon-15m.html")
    if args.pages:
        out_html = Path("docs/binance/ma-ribbon-15m.html")
    write_html_ok = args.pages or args.output or not args.skip_charts
    if write_html_ok:
        write_html(
            path=out_html,
            days=args.days,
            universe_n=len(symbols),
            rows=rows,
            stats=stats,
            h1_stats=h1_stats,
            h1_recs=h1_recs,
            h1_week_stats=h1_week_stats,
            gallery=gallery,
        )
        print(f"\n已寫入 {out_html}  圖 {len(gallery)} 組（15m + 1h）")
    else:
        out_html = Path("output/ribbon15_summary.json")
    Path("output").mkdir(exist_ok=True)
    Path("output/ribbon15_summary.json").write_text(
        json.dumps(
            {
                "days": args.days,
                "universe": len(symbols),
                "signals": len(rows),
                "hourly_attached": n_h1,
                "filters": stats,
                "h1_filters": h1_stats,
                "h1_recommended": h1_recs,
                "h1_week": h1_week_stats,
                "html": str(out_html),
                "charts": len(gallery),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not write_html_ok:
        print(f"\n統計已寫入 output/ribbon15_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
