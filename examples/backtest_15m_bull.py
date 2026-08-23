#!/usr/bin/env python3
"""回測：收盤高於 MA7/25/200，且 7>25。MA14 只畫圖、不擋單。

通知只推本根剛站上該週期 MA200，且 1h MA25 未下彎。大週期 SMA200 只對照、不擋單。

    python3 examples/backtest_15m_bull.py --demo
    python3 examples/backtest_15m_bull.py --days 7 --stocks --pages
    python3 examples/backtest_15m_bull.py --days 30 --tf 1h --pages
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.binance import INTERVAL_MS, fetch_klines, universe
from nq.ma15_bull import (
    HORIZONS,
    SignalRow,
    add_15m_mas,
    above_htf_ma200,
    apply_filter,
    bar_above_ma200,
    detect_combo,
    fail_rate,
    forward_moves,
    htf_ma200_at,
    htf_ma25_now_prev,
    sma,
    summarize_rows,
)

TZ = timezone(timedelta(hours=8))
DISPLAY = {
    "HK1810USDT": "小米",
    "CRCLUSDT": "CRCL",
    "HK0700USDT": "騰訊",
    "TENCENTUSDT": "騰訊",
    "MEITUANUSDT": "美團",
    "KUAISHOUUSDT": "快手",
    "POPMARTUSDT": "泡泡瑪特",
}
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}


@dataclass(frozen=True)
class TfSpec:
    signal: str
    htf: str
    htf_min_days: int
    hold4: int
    table_horizons: tuple[int, ...]
    labels: dict[int, str]
    fail_label: str
    img: str
    html: str
    html_stocks: str
    public: str
    public_stocks: str
    require_htf: bool = True
    min_below: int | None = None
    min_vol: float | None = None
    max_ext: float | None = None
    max_rng24: float | None = None
    max_bars_above: int | None = None
    require_btc_1h: bool = False
    require_h1_ma25_up: bool = False
    lookback: int = 48

    @property
    def htf_ms(self) -> int:
        return INTERVAL_MS[self.htf]


TF_SPECS = {
    "15m": TfSpec(
        signal="15m",
        htf="1h",
        htf_min_days=14,
        hold4=16,
        table_horizons=(1, 4, 16, 32),
        labels={1: "15m", 2: "30m", 4: "1h", 8: "2h", 16: "4h", 32: "8h"},
        fail_label="15m假突破",
        img="ma15-bull",
        html="docs/binance/ma15-bull.html",
        html_stocks="docs/binance/ma15-bull-stocks.html",
        public=(
            "https://htmlpreview.github.io/?"
            "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
            "cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull.html"
        ),
        public_stocks=(
            "https://htmlpreview.github.io/?"
            "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
            "cursor/15m-bull-ma200-e2b2/docs/binance/ma15-bull-stocks.html"
        ),
        require_htf=False,
        require_h1_ma25_up=True,
    ),
    "1h": TfSpec(
        signal="1h",
        htf="4h",
        htf_min_days=40,
        hold4=4,
        table_horizons=(1, 2, 4, 8),
        labels={1: "1h", 2: "2h", 4: "4h", 8: "8h", 16: "16h", 32: "32h"},
        fail_label="1h假突破",
        img="ma1h-bull",
        html="docs/binance/ma1h-bull.html",
        html_stocks="docs/binance/ma1h-bull-stocks.html",
        public=(
            "https://htmlpreview.github.io/?"
            "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
            "cursor/15m-bull-ma200-e2b2/docs/binance/ma1h-bull.html"
        ),
        public_stocks=(
            "https://htmlpreview.github.io/?"
            "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
            "cursor/15m-bull-ma200-e2b2/docs/binance/ma1h-bull-stocks.html"
        ),
        require_htf=False,
        min_below=None,
        min_vol=None,
        max_ext=None,
        max_rng24=None,
        require_btc_1h=False,
        require_h1_ma25_up=True,
        lookback=80,
    ),
}


def notify_kwargs(spec: TfSpec) -> dict:
    kw: dict = {}
    if spec.max_bars_above is not None:
        kw["max_bars_above"] = spec.max_bars_above
    else:
        kw["crossed"] = True
    if spec.min_below is not None:
        kw["min_below"] = spec.min_below
    if spec.min_vol is not None:
        kw["min_vol"] = spec.min_vol
    if spec.max_ext is not None:
        kw["max_ext"] = spec.max_ext
    if spec.max_rng24 is not None:
        kw["max_rng24"] = spec.max_rng24
    if spec.require_btc_1h:
        kw["require_btc_1h"] = True
    if spec.require_h1_ma25_up:
        kw["require_h1_ma25_up"] = True
    return kw


def notify_label(spec: TfSpec) -> str:
    if spec.max_bars_above is not None:
        base = (
            f"通知：剛站上 {spec.signal} MA200，或站上後 {spec.max_bars_above} 根內收出 7>25"
        )
    else:
        base = f"通知：本根剛站上 {spec.signal} MA200"
    if spec.require_h1_ma25_up:
        base += "，且 1h MA25 未下彎"
    return base


def filter_defs(spec: TfSpec) -> tuple:
    tf = spec.signal
    rows = [
        (f"原始：收盤 > MA7>25 且 > {tf} MA200（組合剛成立）", {"crossed": None, "formed": None, "min_vol": None, "max_ext": None}),
        (f"本根剛站上 {tf} MA200", {"crossed": True, "formed": None, "min_vol": None, "max_ext": None}),
        ("已在 MA200 上，本根才收上 7>25", {"crossed": None, "formed": True, "min_vol": None, "max_ext": None}),
        ("放量 ≥ 1.5×", {"crossed": None, "formed": None, "min_vol": 1.5, "max_ext": None}),
        ("剛站上 MA200 + 放量 ≥ 1.5×", {"crossed": True, "formed": None, "min_vol": 1.5, "max_ext": None}),
        ("距 MA200 ≤ 1.0%", {"crossed": None, "formed": None, "min_vol": None, "max_ext": 1.0}),
        (notify_label(spec), notify_kwargs(spec)),
    ]
    if spec.max_bars_above is not None:
        rows.insert(
            -1,
            (
                f"已在 MA200 上，且站上後 ≤ {spec.max_bars_above} 根才排好",
                {"formed": True, "max_bars_above": spec.max_bars_above},
            ),
        )
    if spec.min_below is not None:
        rows.extend(
            [
                (
                    f"剛站上且底下 ≥ {spec.min_below} 根",
                    {"crossed": True, "min_below": spec.min_below},
                ),
                (
                    notify_label(spec),
                    notify_kwargs(spec),
                ),
                (
                    "同上但不看 BTC",
                    {k: v for k, v in notify_kwargs(spec).items() if k != "require_btc_1h"},
                ),
            ]
        )
    return tuple(rows)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or f"s{abs(hash(symbol)) % 10_000_000_000}"


def sym_label(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    name = DISPLAY.get(symbol)
    if name and name != base:
        return f"{name} {base}"
    return base


def pct(v: float | None) -> str:
    if v is None:
        return "—"
    cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    return f'<span class="{cls}">{v:+.2f}%</span>'


def scan_symbol(sym: str, days: int, spec: TfSpec) -> tuple[str, dict | None, list[SignalRow], int, int]:
    raw = fetch_klines(sym, interval=spec.signal, days=days, extra_bars=220)
    if raw is None or len(raw["c"]) < 220:
        return sym, None, [], 0, 0
    d = add_15m_mas(raw)
    raw_h = fetch_klines(sym, interval=spec.htf, days=max(days, spec.htf_min_days), extra_bars=220)
    d_htf = add_15m_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
    cutoff = int(d["t"][-1]) - days * 24 * 60 * 60 * 1000
    rows = []
    n_raw = 0
    n_drop = 0
    for sig in detect_combo(d):
        ts = int(d["t"][sig.idx])
        if ts < cutoff:
            continue
        n_raw += 1
        if spec.require_htf and not above_htf_ma200(d_htf, ts, sig.close, spec.htf_ms):
            n_drop += 1
            continue
        entry, moves = forward_moves(d, sig)
        if np.isnan(entry):
            continue
        ma_h = htf_ma200_at(d_htf, ts, sig.close, spec.htf_ms) if d_htf is not None else None
        ext_h = (sig.close / ma_h - 1.0) * 100.0 if ma_h else None
        src_1h = d if spec.signal == "1h" else d_htf
        bar_1h = INTERVAL_MS["1h"]
        ma25_now, ma25_prev = htf_ma25_now_prev(src_1h, ts, sig.close, bar_1h)
        rows.append(
            SignalRow(
                symbol=sym,
                sig=sig,
                time_ms=ts,
                entry=entry,
                moves=moves,
                h1_ma200=ma_h,
                h1_ext_pct=ext_h,
                h1_ma25=ma25_now,
                h1_ma25_prev=ma25_prev,
                h1_ma25_up=bool(
                    ma25_now is not None and ma25_prev is not None and ma25_now >= ma25_prev
                ),
            )
        )
    return sym, d, rows, n_raw, n_drop


def filter_stats(rows: list[SignalRow], spec: TfSpec) -> list[dict]:
    out = []
    for name, kw in filter_defs(spec):
        subset = apply_filter(rows, **kw)
        item = {"name": name, "count": len(subset), "fail15_pct": fail_rate(subset)}
        for h in HORIZONS:
            item[f"h{h}"] = summarize_rows(subset, h)
        out.append(item)
    return out


def _style_ax(ax) -> None:
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")


def _paint_ohlcv(ax, axv, d: dict, a0: int, a1: int, mark_i: int | None) -> None:
    from matplotlib.patches import Rectangle

    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
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
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.35 if n == 200 else 1.15, label=f"MA{n}")
    if mark_i is not None and a0 <= mark_i < a1:
        x = mark_i - a0
        ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
        ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)


def tf_bar_idx(d: dict, time_ms: int, bar_ms: int) -> int | None:
    t = d["t"]
    open_ms = int(time_ms) - (int(time_ms) % bar_ms)
    w = np.where(t == open_ms)[0]
    if len(w):
        return int(w[0])
    w = np.where(t <= time_ms)[0]
    return int(w[-1]) if len(w) else None


def draw_chart(
    sym: str,
    d: dict,
    row: SignalRow,
    path: Path,
    spec: TfSpec,
    d_htf: dict | None = None,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    i = row.sig.idx
    a0 = max(0, i - spec.lookback)
    a1 = min(len(d["c"]), i + 20)
    r4 = row.moves.get(spec.hold4)
    rtxt = f"  4h {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else ""
    title_sym = file_base(sym) if any(ord(ch) >= 128 for ch in sym) else sym
    kind = "reclaim MA200" if row.crossed_200d else f"stack within {row.bars_above} bars of MA200"
    extra = f"  below={row.bars_below}  above={row.bars_above}  vol={row.vol_ratio:.2f}x  ext={row.ext_pct:+.2f}%"
    if row.h1_ma25 is not None and row.h1_ma25_prev is not None:
        extra += "  1hMA25 " + ("up" if row.h1_ma25_up else "down")

    hi = tf_bar_idx(d_htf, row.time_ms, spec.htf_ms) if d_htf is not None and len(d_htf.get("c", [])) else None
    stacked = hi is not None
    if stacked:
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(10.6, 10.6),
            sharex=False,
            gridspec_kw={"height_ratios": [3.1, 0.9, 3.1, 0.9]},
            facecolor="#0c1210",
        )
        ax, axv, axh, axhv = axes
    else:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(10.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
        )
        axh = axhv = None
    for a in (ax, axv, axh, axhv):
        if a is not None:
            _style_ax(a)
    _paint_ohlcv(ax, axv, d, a0, a1, i)
    ax.set_title(f"{title_sym}  {spec.signal}  {hm(row.time_ms)}  {kind}{extra}{rtxt}", color="#e8f0ea", fontsize=11)
    if stacked and axh is not None and axhv is not None and d_htf is not None and hi is not None:
        b0 = max(0, hi - 48)
        b1 = min(len(d_htf["c"]), hi + 16)
        _paint_ohlcv(axh, axhv, d_htf, b0, b1, hi)
        h_close = float(d_htf["c"][hi])
        h_ma = float(d_htf["m200"][hi]) if not np.isnan(d_htf["m200"][hi]) else None
        vs = ""
        if h_ma:
            vs = f"  close {h_close:g} vs {spec.htf} MA200 {h_ma:g} ({(h_close / h_ma - 1) * 100:+.2f}%)"
        axh.set_title(f"{title_sym}  {spec.htf}  {hm(int(d_htf['t'][hi]))}{vs}", color="#e8f0ea", fontsize=12)
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def pick_gallery(rows: list[SignalRow], spec: TfSpec, limit: int = 18) -> list[SignalRow]:
    pool = apply_filter(rows, **notify_kwargs(spec)) or [r for r in rows if r.crossed_200d] or list(rows)
    if not pool:
        return []
    pinned = []
    for sym in ("TUTUSDT", "BTCUSDT", "TRUMPUSDT", "CRCLUSDT"):
        pinned.extend(sorted([r for r in pool if r.symbol == sym], key=lambda r: r.time_ms, reverse=True)[:4])

    def score(r: SignalRow) -> float:
        mv = r.moves.get(spec.hold4)
        return mv.ret_pct if mv and mv.ret_pct is not None else 0.0

    ranked = sorted(pool, key=score, reverse=True)
    winners = ranked[: max(1, limit * 2 // 3)]
    losers = list(reversed(ranked[len(winners) :]))[: limit - len(winners)]
    mixed = pinned + [r for r in (winners + losers) if r not in pinned]
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


def kpi_block(stats: list[dict], spec: TfSpec) -> str:
    raw = stats[0]
    cross = stats[1]
    notify = next((s for s in stats if s["name"].startswith("通知：")), cross)
    h = notify[f"h{spec.hold4}"]
    return "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in (
            ("通知", notify["count"]),
            ("4h 勝率", f"{h['wr']:.1f}%"),
            ("4h 平均", f"{h['avg']:+.3f}%"),
            (spec.fail_label, f"{notify['fail15_pct']:.1f}%"),
            ("全部剛站上", cross["count"]),
            ("全部組合剛成立", raw["count"]),
        )
    )


def stats_table(stats: list[dict], spec: TfSpec) -> str:
    hs = spec.table_horizons
    heads = "".join(f"<th>{spec.labels[h]}勝率</th><th>{spec.labels[h]}均</th>" for h in hs)
    rows = []
    for s in stats:
        cells = [f"<td>{html.escape(s['name'])}</td>", f'<td class="mono">{s["count"]}</td>', f'<td class="mono">{s["fail15_pct"]:.1f}%</td>']
        for h in hs:
            st = s[f"h{h}"]
            cells.append(f'<td class="mono">{st["wr"]:.1f}%</td>')
            cells.append(f"<td>{pct(st['avg'])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr><th>條件</th><th>筆數</th><th>{html.escape(spec.fail_label)}</th>"
        + heads
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def signal_table(rows: list[SignalRow], spec: TfSpec, limit: int = 80) -> str:
    hs = spec.table_horizons
    focus = list(rows)
    ranked = sorted(
        focus,
        key=lambda r: -(r.moves.get(spec.hold4).ret_pct or -999)
        if r.moves.get(spec.hold4) and r.moves[spec.hold4].ret_pct is not None
        else 999,
    )[:limit]
    body = []
    for r in ranked:
        kind = "站上MA200" if r.crossed_200d else f"站上後{r.bars_above}根內"
        cells = [
            f'<td class="mono">{hm(r.time_ms)}</td>',
            f"<td>{html.escape(sym_label(r.symbol))}</td>",
            f"<td>{html.escape(kind)}</td>",
            f'<td class="mono">{r.entry:g}</td>',
            f'<td class="mono">{r.bars_below}</td>',
            f'<td class="mono">{r.vol_ratio:.2f}</td>',
            f'<td class="mono">{r.ext_pct:+.2f}%</td>',
            f'<td class="mono">{r.h1_ext_pct:+.2f}%</td>' if r.h1_ext_pct is not None else "<td>—</td>",
        ]
        for h in hs:
            mv = r.moves.get(h)
            cells.append(f"<td>{pct(mv.ret_pct if mv else None)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    ret_heads = "".join(f"<th>{spec.labels[h]}</th>" for h in hs)
    return (
        "<table><thead><tr><th>時間</th><th>標的</th><th>種類</th><th>進場</th><th>底下</th><th>量比</th>"
        f"<th>距{spec.signal}MA200</th><th>距{spec.htf}MA200</th>"
        f"{ret_heads}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def pin_card(rows: list[SignalRow], spec: TfSpec, symbol: str, title: str) -> str:
    mine = [r for r in rows if r.symbol == symbol]
    if not mine:
        return (
            f'<div class="card"><h2>{html.escape(title)}</h2>'
            f'<p class="note">這段期間沒有符合通知條件的訊號。</p></div>'
        )
    st = summarize_rows(mine, spec.hold4)
    return f"""<div class="card">
    <h2>{html.escape(title)}</h2>
    <p class="note">符合通知 {len(mine)} 筆。4h 勝率 {st['wr']:.1f}%　4h 均 {st['avg']:+.3f}%　4h 中位 {st['med']:+.3f}%。</p>
    <div class="table-wrap">{signal_table(mine, spec, limit=40)}</div>
  </div>"""


def public_img_src(rel: str) -> str:
    """內嵌 PNG，htmlpreview / GitHub raw 才看得到圖。"""
    if rel.startswith("data:"):
        return rel
    local = Path("docs/binance") / rel.lstrip("./")
    if not local.exists() and ("/" in rel):
        name = rel.rsplit("/", 1)[-1]
        for folder in ("ma15-bull", "ma15-bull-stocks", "ma1h-bull", "ma1h-bull-stocks"):
            cand = Path("docs/binance/img") / folder / name
            if cand.exists():
                local = cand
                break
    if local.exists():
        b64 = base64.b64encode(local.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return rel


def gallery_html(gallery: list[tuple[SignalRow, str]], spec: TfSpec) -> str:
    if not gallery:
        return '<p class="note">這段期間沒有圖例。</p>'
    cards = []
    for row, rel in gallery:
        kind = (
            f"本根剛站上 {spec.signal} MA200"
            if row.crossed_200d
            else f"站上後 {row.bars_above} 根內才收出 7>25"
        )
        r4 = row.moves.get(spec.hold4)
        rtxt = f"4h {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else ""
        src = public_img_src(rel)
        slope = ""
        if row.h1_ma25 is not None and row.h1_ma25_prev is not None:
            slope = " · 1hMA25 未下彎" if row.h1_ma25_up else " · 1hMA25 下彎"
        cards.append(
            f'<div class="card"><div class="cap">{html.escape(sym_label(row.symbol))} · {hm(row.time_ms)} · '
            f"{html.escape(kind)} · 底下 {row.bars_below} 根 · 量比 {row.vol_ratio:.2f} · "
            f"{html.escape(rtxt)}{html.escape(slope)}</div>"
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
    stocks_only: bool = False,
    spec: TfSpec,
    notify_rows: list[SignalRow] | None = None,
) -> None:
    notify_rows = notify_rows if notify_rows is not None else apply_filter(rows, **notify_kwargs(spec))
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    first = datetime.fromtimestamp(min(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    last = datetime.fromtimestamp(max(r.time_ms for r in rows) / 1000, TZ).strftime("%m-%d") if rows else "—"
    public = spec.public_stocks if stocks_only else spec.public
    other = TF_SPECS["1h" if spec.signal == "15m" else "15m"]
    other_url = other.public_stocks if stocks_only else other.public
    other_name = "小時圖" if spec.signal == "15m" else "15 分圖"
    sister_link = (
        f'另有 {other_name} 回測：<a href="{html.escape(other_url)}" style="color:#c9a227">{html.escape(other_name)}</a>。'
    )
    tf = spec.signal
    htf = spec.htf
    title = f"{tf} 收盤在 7/25/200 之上" + (" · 幣安股票" if stocks_only else "")
    heading = f"{tf} K：收盤在 7 / 25 / 200 之上" + ("（只掃幣安股票）" if stocks_only else "")
    if spec.min_below is not None:
        heading += " · 底下夠久、波動不大、BTC 先站上"
    universe_txt = (
        f"只掃幣安 TradFi 股票永續（美／港／韓／中股、股票 ETF、Pre-IPO；不含黃金原油等商品）近 {days} 天、{universe_n} 個合約。"
        if stocks_only
        else f"掃描幣安 U 本位成交額前 {universe_n} 永續近 {days} 天。"
    )
    if spec.require_htf:
        htf_rule = f"還要當下價格在<strong>{html.escape(htf)} 圖 SMA200</strong>之上。"
        notify_rule = (
            f"Telegram 通知只用「本根剛站上 {html.escape(tf)} MA200」：前收還在 {html.escape(tf)} MA200 下，"
            f"這一根收盤站上，同時收盤也在 7>25 之上，"
            f"<strong>且當下價格已在 {html.escape(htf)} MA200 上方</strong>"
            f"（未收完的大週期 K 用當下收盤，不看未來）。"
        )
    else:
        htf_rule = f"{html.escape(htf)} 圖 SMA200 只放在底下對照，<strong>不當作過濾</strong>。"
        if spec.require_h1_ma25_up:
            htf_rule += (
                "另要求當時 <strong>1h SMA25 未下彎</strong>"
                "（當下 1h MA25 ≥ 前一根已收完；未收完的 1h 用當下收盤，不看未來）。"
            )
        if spec.min_below is not None:
            extra = ""
            if spec.min_vol is not None:
                extra += f"量比 ≥ <strong>{spec.min_vol:g}×</strong>、"
            extra += f"收盤距 MA200 ≤ <strong>{spec.max_ext:g}%</strong>"
            if spec.max_rng24 is not None:
                extra += f"、近 24 根高低差 ≤ <strong>{spec.max_rng24:g}%</strong>"
            if spec.require_btc_1h:
                extra += "，且當時 <strong>BTC 已在 1h MA200 上</strong>"
            extra += "。"
            notify_rule = (
                f"Telegram 通知只推這種：<strong>剛站上 {html.escape(tf)} MA200</strong>，"
                f"且站上前連續至少 <strong>{spec.min_below} 根</strong>收盤在 MA200 下、"
                f"{extra}"
            )
        else:
            if spec.max_bars_above is not None:
                notify_rule = (
                    f"Telegram 通知：<strong>剛站上 {html.escape(tf)} MA200</strong>"
                    f"（前收還在 MA200 下、本根收盤站上且 7&gt;25），"
                    f"或<strong>站上後 {spec.max_bars_above} 根內</strong>才收出 7&gt;25。"
                    f"不會把已經在 MA200 上很久才排好的訊號都打進去。"
                )
            else:
                notify_rule = (
                    f"Telegram 通知只用「本根剛站上 {html.escape(tf)} MA200」：前收還在 {html.escape(tf)} MA200 下，"
                    f"這一根收盤站上，同時收盤也在 7>25 之上"
                    + ("，且 <strong>1h MA25 未下彎</strong>。" if spec.require_h1_ma25_up else "。")
                )
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
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
  <h1>{html.escape(heading)}</h1>
  <p class="sub">
    規則：收盤同時高於 <strong>MA7、MA25、MA200</strong>（同一張 {html.escape(tf)} 圖），且 SMA7 &gt; SMA25。MA14 只畫圖、不擋單。
    {htf_rule}
    {notify_rule}
    進場用訊號下一根開盤。假突破 = 進場那根 {html.escape(tf)} 收盤又跌回 MA200 下方。
    {universe_txt}訊號區間 {first} → {last}（GMT+8）。產生於 {now}。
    外網請開 <a href="{public}" style="color:#c9a227">這頁（htmlpreview）</a>；圖已內嵌。
    {sister_link}
    僅供型態對照，不是進出場建議。
  </p>
  <div class="kpis">{kpi_block(stats, spec)}</div>
  {pin_card(notify_rows, spec, "TUTUSDT", "TUT")}
  {pin_card(notify_rows, spec, "TRUMPUSDT", "TRUMP")}
  {pin_card(notify_rows, spec, "CRCLUSDT", "CRCL")}
  <h1>圖例</h1>
  <p class="sub">上面 {html.escape(tf)} K，下面同一檔 {html.escape(htf)} K（只對照，不擋單）。黃虛線是訊號時間。白線是各週期自己的 MA200。圖例只畫通知條件（TUT、TRUMP、CRCL 若有會釘在最上面）。</p>
  {gallery_html(gallery, spec)}
  <div class="card">
    <h2>過濾對照</h2>
    <div class="table-wrap">{stats_table(stats, spec)}</div>
    <p class="note">底下 = 站上前連續幾根收盤在 MA200 下。距{html.escape(tf)}MA200 =（收盤 / {html.escape(tf)} SMA200 − 1）× 100%。距{html.escape(htf)}MA200 用訊號當下價 vs 當時 {html.escape(htf)} SMA200。量比 = 當根量 / 20 根均量。1h MA25 未下彎 = 當下 1h SMA25 ≥ 前一根已收完（未收完的 1h 用當下收盤）。</p>
  </div>
  <div class="card">
    <h2>訊號表（通知條件，依 4h 報酬排序）</h2>
    <div class="table-wrap">{signal_table(notify_rows, spec)}</div>
  </div>
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def make_demo_bars() -> dict:
    n = 240
    px = np.full(n, 99.00)
    px[220:] = 100.20
    o = np.roll(px, 1)
    o[0] = 99.00
    o[220] = 99.00
    return {
        "t": np.arange(n, dtype=np.int64) * 15 * 60_000,
        "o": o,
        "h": np.maximum(o, px) + 0.05,
        "l": np.minimum(o, px) - 0.02,
        "c": px,
        "v": np.full(n, 1200.0),
    }


def run_demo() -> int:
    d = add_15m_mas(make_demo_bars())
    hits = detect_combo(d)
    print(f"demo 偵測到 {len(hits)} 筆（收盤 > 7>25 且 > MA200）")
    for sig in hits:
        print(
            f"  idx={sig.idx} close={sig.close:.3f} ma200={sig.ma200:.3f} "
            f"close>7>25={sig.close:.3f}>{sig.m7:.3f}>{sig.m25:.3f} "
            f"crossed={sig.crossed_200} formed={sig.formed_align} below={sig.bars_below}"
        )
    if not hits:
        print("demo 失敗：應該至少有一根組合成立")
        return 1
    if not any(s.crossed_200 for s in hits):
        print("demo 失敗：應該有剛站上 MA200")
        return 1
    if any(s.close <= s.m7 or s.close <= s.m25 or s.close <= s.ma200 or s.m7 <= s.m25 for s in hits):
        print("demo 失敗：收盤必須在 7>25 且 > MA200")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="收盤在 7>25 且 > MA200 之上（15m 或 1h）")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tf", choices=("15m", "1h"), default="15m", help="訊號週期；大週期圖只對照、不擋單")
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance/ma15-bull.html 或 ma1h-bull.html")
    p.add_argument("-o", "--output", help="HTML 輸出路徑")
    p.add_argument("--limit-symbols", type=int, default=0)
    p.add_argument("--stocks", action="store_true", help="只掃幣安 TradFi 股票永續（不含商品）")
    args = p.parse_args()
    if args.demo:
        return run_demo()
    spec = TF_SPECS[args.tf]

    print("載入標的…", flush=True)
    symbols = universe(stocks_only=args.stocks)
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    scope = "幣安股票永續" if args.stocks else "成交額前100永續"
    htf_scan = (
        f"{spec.signal} MA200 + {spec.htf} MA200"
        if spec.require_htf
        else f"{spec.signal} MA200（{spec.htf} 只畫圖、不擋單）"
    )
    print(f"掃描 {len(symbols)} 個 {spec.signal} {scope}，近 {args.days} 天（{htf_scan}）", flush=True)

    rows: list[SignalRow] = []
    data: dict[str, dict] = {}
    n_raw = n_drop = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(scan_symbol, s, args.days, spec): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, hits, a, b = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            n_raw += a
            n_drop += b
            if d is not None:
                data[sym] = d
            rows.extend(hits)
            if done % 40 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)}  訊號 {len(rows)}  {time.time()-t0:.1f}s", flush=True)
    rows.sort(key=lambda r: r.time_ms)
    if spec.require_btc_1h:
        raw_btc = fetch_klines("BTCUSDT", interval="1h", days=max(args.days, 14), extra_bars=220)
        btc_d = add_15m_mas(raw_btc) if raw_btc is not None and len(raw_btc["c"]) >= 200 else None
        if btc_d is None:
            print("警告：抓不到 BTC 1h，BTC 大盤過濾全判不過", flush=True)
        for r in rows:
            r.btc_1h_ok = bar_above_ma200(btc_d, r.time_ms, INTERVAL_MS["1h"])
    stats = filter_stats(rows, spec)
    notify_rows = apply_filter(rows, **notify_kwargs(spec))
    if spec.require_htf:
        print(f"\n=== {spec.signal} 7>25 + MA200，且當下在 {spec.htf} MA200 上 ===")
        print(f"{spec.signal} 組合 {n_raw} 筆，未站上 {spec.htf} MA200 去掉 {n_drop} 筆，留下 {n_raw - n_drop}")
    else:
        print(f"\n=== {spec.signal} 7>25 + MA200（不擋 {spec.htf} MA200）===")
        print(f"{spec.signal} 組合 {n_raw} 筆，留下 {n_raw - n_drop}")
        print(f"{notify_label(spec)} → {len(notify_rows)} 筆")
    for s in stats:
        h = s[f"h{spec.hold4}"]
        print(
            f"{s['name']}: n={s['count']}  4h勝率 {h['wr']:.1f}%  4h均 {h['avg']:+.3f}%  "
            f"假突破 {s['fail15_pct']:.1f}%"
        )
    crcl = [r for r in notify_rows if r.symbol == "CRCLUSDT"]
    print(f"CRCL 通知：{len(crcl)} 筆")
    for r in crcl:
        mv = r.moves.get(spec.hold4)
        print(f"  {hm(r.time_ms)}  close={r.sig.close:g}  below={r.bars_below}  above={r.bars_above}  vol={r.vol_ratio:.2f}  4h={mv.ret_pct if mv else None}")
    trump = [r for r in notify_rows if r.symbol == "TRUMPUSDT"]
    print(f"TRUMP 通知：{len(trump)} 筆")
    for r in trump:
        mv = r.moves.get(spec.hold4)
        print(f"  {hm(r.time_ms)}  close={r.sig.close:g}  below={r.bars_below}  above={r.bars_above}  vol={r.vol_ratio:.2f}  4h={mv.ret_pct if mv else None}")
    tut = [r for r in notify_rows if r.symbol == "TUTUSDT"]
    print(f"TUT 通知：{len(tut)} 筆")
    for r in tut:
        mv = r.moves.get(spec.hold4)
        print(f"  {hm(r.time_ms)}  close={r.sig.close:g}  crossed={r.crossed_200d}  above={r.bars_above}  vol={r.vol_ratio:.2f}  1hMA25={'up' if r.h1_ma25_up else 'down'}  4h={mv.ret_pct if mv else None}")
    btc = [r for r in notify_rows if r.symbol == "BTCUSDT"]
    print(f"BTC 通知：{len(btc)} 筆")
    for r in btc:
        mv = r.moves.get(spec.hold4)
        print(f"  {hm(r.time_ms)}  close={r.sig.close:g}  below={r.bars_below}  vol={r.vol_ratio:.2f}  4h={mv.ret_pct if mv else None}")

    img_name = spec.img + ("-stocks" if args.stocks else "")
    img_dir = Path("docs/binance/img") / img_name
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()
    htf_cache: dict[str, dict | None] = {}
    gallery: list[tuple[SignalRow, str]] = []
    for row in pick_gallery(rows, spec, limit=24):
        d = data.get(row.symbol)
        if d is None:
            continue
        if row.symbol not in htf_cache:
            raw_h = fetch_klines(row.symbol, interval=spec.htf, days=max(args.days, spec.htf_min_days), extra_bars=220)
            htf_cache[row.symbol] = add_15m_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
        stamp = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m%d%H%M")
        fname = f"{file_base(row.symbol)}_{stamp}.png"
        out = img_dir / fname
        if draw_chart(row.symbol, d, row, out, spec, d_htf=htf_cache[row.symbol]):
            gallery.append((row, f"./img/{img_name}/{fname}"))

    default_html = spec.html_stocks if args.stocks else spec.html
    out_html = Path(args.output) if args.output else Path(default_html)
    if args.pages:
        out_html = Path(default_html)
    write_html(
        path=out_html,
        days=args.days,
        universe_n=len(symbols),
        rows=rows,
        stats=stats,
        gallery=gallery,
        stocks_only=args.stocks,
        spec=spec,
        notify_rows=notify_rows,
    )
    Path("output").mkdir(exist_ok=True)
    if args.tf == "15m":
        summary_path = Path("output/ma15_bull_stocks_summary.json" if args.stocks else "output/ma15_bull_summary.json")
    else:
        summary_path = Path("output/ma1h_bull_stocks_summary.json" if args.stocks else "output/ma1h_bull_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "days": args.days,
                "tf": args.tf,
                "htf": spec.htf,
                "require_htf": spec.require_htf,
                "stocks_only": args.stocks,
                "universe": len(symbols),
                "signals": len(rows),
                "combo": n_raw,
                "min_below": spec.min_below,
                "min_vol": spec.min_vol,
                "max_ext": spec.max_ext,
                "max_rng24": spec.max_rng24,
                "require_btc_1h": spec.require_btc_1h,
                "notify": len(notify_rows),
                "crcl": [
                    {"time": hm(r.time_ms), "close": r.sig.close, "crossed": r.crossed_200d}
                    for r in crcl
                ],
                "trump": [
                    {
                        "time": hm(r.time_ms),
                        "close": r.sig.close,
                        "below": r.bars_below,
                        "vol": r.vol_ratio,
                        "ext": r.ext_pct,
                    }
                    for r in trump
                ],
                "btc": [
                    {
                        "time": hm(r.time_ms),
                        "close": r.sig.close,
                        "below": r.bars_below,
                        "vol": r.vol_ratio,
                        "ext": r.ext_pct,
                    }
                    for r in btc
                ],
                "tut": [
                    {
                        "time": hm(r.time_ms),
                        "close": r.sig.close,
                        "below": r.bars_below,
                        "vol": r.vol_ratio,
                        "ext": r.ext_pct,
                    }
                    for r in tut
                ],
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
