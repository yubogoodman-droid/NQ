#!/usr/bin/env python3
"""指定日期窗的幣安 MA200 通知回測 → 單一自包含 HTML（每筆訊號一張圖）。

只讀 K 線、不算下單。台股不在此策略範圍。
"""
from __future__ import annotations

import base64
import html
import importlib.util
import io
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nq.binance import INTERVAL_MS, fetch_klines, universe
from nq.ma15_bull import (
    add_15m_mas,
    confirm_reclaim_hold,
    detect_combo,
    summarize_rows,
)

_BT = importlib.util.spec_from_file_location("bt15", ROOT / "examples" / "backtest_15m_bull.py")
bt = importlib.util.module_from_spec(_BT)
sys.modules["bt15"] = bt
_BT.loader.exec_module(bt)

TZ = timezone(timedelta(hours=8))
START = datetime(2026, 10, 17, 0, 0, tzinfo=TZ)
END = datetime(2026, 10, 30, 23, 59, 59, tzinfo=TZ)
START_MS = int(START.timestamp() * 1000)
END_MS = int(END.timestamp() * 1000)
OUT_HTML = ROOT / "docs" / "binance" / "ma-bull-20261017-1030.html"
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m-%d %H:%M")


def scan_notify(sym: str, spec: bt.TfSpec) -> tuple[str, dict | None, list]:
    days = 22
    raw = fetch_klines(sym, interval=spec.signal, days=days, extra_bars=220)
    if raw is None or len(raw["c"]) < 220:
        return sym, None, []
    d = add_15m_mas(raw)
    raw_h = fetch_klines(sym, interval=spec.htf, days=max(days, spec.htf_min_days), extra_bars=220)
    d_htf = add_15m_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
    rows = []
    for sig in detect_combo(d):
        if spec.min_hold_bars:
            if not sig.crossed_200:
                continue
            held = confirm_reclaim_hold(d, sig, spec.min_hold_bars)
            if held is None:
                continue
            ts = int(d["t"][held.idx])
            if ts < START_MS or ts > END_MS:
                continue
            row = bt.row_from_sig(sym, d, d_htf, held, spec, hold_confirmed=True)
        else:
            if not sig.crossed_200:
                continue
            ts = int(d["t"][sig.idx])
            if ts < START_MS or ts > END_MS:
                continue
            row = bt.row_from_sig(sym, d, d_htf, sig, spec, hold_confirmed=False)
        if row is None:
            continue
        if spec.require_h1_ma25_up and not row.h1_ma25_up:
            continue
        if spec.require_h1_stack and not row.h1_stack_ok:
            continue
        rows.append(row)
    return sym, d, rows


def _setup_font() -> None:
    from matplotlib import font_manager, rcParams

    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return


def draw_one(d: dict, row, spec: bt.TfSpec) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        _setup_font()
    except Exception:
        return None
    i = row.sig.idx
    entry_i = i + 1
    exit_i = entry_i + spec.hold4 - 1
    a0 = max(0, i - spec.lookback)
    a1 = min(len(d["c"]), max(i + 6, (exit_i + 3) if exit_i < len(d["c"]) else i + 6))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(9.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 0.85]}, facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        axv.bar(xs[k], v[k], width=0.8, color="#3dba7a99" if up else "#e35d5d99", linewidth=0)
    for n, col in PAL.items():
        ax.plot(xs, bt.sma(d["c"], n)[sl], color=col, lw=1.35 if n == 200 else 1.1, label=f"MA{n}")
    if a0 <= i < a1:
        x = i - a0
        ax.axvline(x, color="#c9a227", ls="--", lw=0.9, label="訊號K")
        ax.scatter([x], [c[x]], s=28, color="#c9a227", zorder=5)
    if a0 <= entry_i < a1:
        x = entry_i - a0
        ax.scatter([x], [row.entry], s=46, marker="^", color="#7CFFB2", zorder=6, label="進場(次K開)")
    if exit_i < len(d["c"]) and a0 <= exit_i < a1:
        x = exit_i - a0
        ax.axvline(x, color="#6ec6ff", ls=":", lw=0.9)
        ax.scatter([x], [float(d["c"][exit_i])], s=42, marker="v", color="#6ec6ff", zorder=6, label="4h時間出場")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, labelcolor="#c8d5cc", ncol=4)
    r4 = row.moves.get(spec.hold4)
    rtxt = f"  4h {r4.ret_pct:+.2f}%" if r4 and r4.ret_pct is not None else "  4h —"
    ax.set_title(
        f"{bt.sym_label(row.symbol)}  {spec.signal}  {hm(row.time_ms)}  進場 {row.entry:g}{rtxt}",
        color="#e8f0ea", fontsize=11,
    )
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=72, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 68})
    plt.close(fig)
    return buf.getvalue()


def extrems(rows, horizon: int):
    scored = [
        (r, r.moves[horizon].ret_pct)
        for r in rows
        if r.moves.get(horizon) and r.moves[horizon].ret_pct is not None
    ]
    if not scored:
        return None, None
    best = max(scored, key=lambda x: x[1])
    worst = min(scored, key=lambda x: x[1])
    return best, worst


def stats_block(name: str, spec: bt.TfSpec, rows) -> tuple[dict, str]:
    st = summarize_rows(rows, spec.hold4)
    best, worst = extrems(rows, spec.hold4)
    n_incomplete = sum(
        1 for r in rows if not (r.moves.get(spec.hold4) and r.moves[spec.hold4].ret_pct is not None)
    )
    def one(pair):
        if pair is None:
            return "—"
        r, v = pair
        return f"{bt.sym_label(r.symbol)} {hm(r.time_ms)}　{v:+.2f}%　進場 {r.entry:g}"
    data = {
        "name": name,
        "n": len(rows),
        "n_fwd": st["n"],
        "wr": st["wr"],
        "avg": st["avg"],
        "med": st["med"],
        "best": one(best),
        "worst": one(worst),
        "incomplete": n_incomplete,
        "horizon": spec.labels[spec.hold4],
    }
    extra = []
    for h in spec.table_horizons:
        s = summarize_rows(rows, h)
        extra.append(f"{spec.labels[h]} 勝率 {s['wr']:.1f}%　均 {s['avg']:+.3f}%　中位 {s['med']:+.3f}%　n={s['n']}")
    data["horizons"] = extra
    return data, extra


def section_html(title: str, spec: bt.TfSpec, rows, frames: dict[str, dict], rules: str) -> str:
    st, extra = stats_block(title, spec, rows)
    parts = [
        f"<h2>{html.escape(title)}</h2>",
        f"<div class='rules'>{rules}</div>",
        "<div class='kpis'>",
        f"<div class='kpi'><div class='k'>訊號數</div><div class='v'>{st['n']}</div></div>",
        f"<div class='kpi'><div class='k'>{html.escape(st['horizon'])} 勝率</div><div class='v'>{st['wr']:.1f}%</div></div>",
        f"<div class='kpi'><div class='k'>{html.escape(st['horizon'])} 平均</div><div class='v'>{st['avg']:+.3f}%</div></div>",
        f"<div class='kpi'><div class='k'>{html.escape(st['horizon'])} 中位</div><div class='v'>{st['med']:+.3f}%</div></div>",
        "</div>",
        f"<p class='note'>前瞻報酬 = 進場後固定持有 {html.escape(st['horizon'])}（訊號下一根開盤買進，該段最後一根收盤賣出）。沒有停損／停利。未扣手續費與資金費／稅。"
        f"有完整 {html.escape(st['horizon'])} 出路的 {st['n_fwd']} 筆；樣本末端來不及走出的 {st['incomplete']} 筆不計入勝率。</p>",
        f"<p>最好：{html.escape(st['best'])}<br/>最差：{html.escape(st['worst'])}</p>",
        "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in extra) + "</ul>",
    ]
    heads = "".join(f"<th>{html.escape(spec.labels[h])}</th>" for h in spec.table_horizons)
    body = []
    ranked = sorted(
        rows,
        key=lambda r: r.time_ms,
    )
    for r in ranked:
        cells = [
            f"<td class='mono'>{html.escape(hm(r.time_ms))}</td>",
            f"<td>{html.escape(bt.sym_label(r.symbol))}</td>",
            f"<td class='mono'>{r.entry:g}</td>",
            f"<td class='mono'>{r.sig.close:g}</td>",
            f"<td class='mono'>{r.vol_ratio:.2f}</td>",
            f"<td class='mono'>{r.ext_pct:+.2f}%</td>",
        ]
        for h in spec.table_horizons:
            mv = r.moves.get(h)
            val = mv.ret_pct if mv else None
            if val is None:
                cells.append("<td>—</td>")
            else:
                cls = "pos" if val > 0 else ("neg" if val < 0 else "")
                cells.append(f"<td class='{cls}'>{val:+.2f}%</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    parts.append(
        "<div class='table-wrap'><table><thead><tr>"
        "<th>訊號時間(GMT+8)</th><th>標的</th><th>進場價</th><th>訊號收</th><th>量比</th>"
        f"<th>距{html.escape(spec.signal)}MA200</th>{heads}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    parts.append("<h3>每筆訊號 K 線</h3>")
    parts.append("<p class='note'>黃虛線＝訊號 K（收盤確認）。綠三角＝進場（次一根開盤）。藍倒三角／點線＝固定 4h 時間出場（若已走出）。白線＝該週期 SMA200；黃／橘／紫＝MA7／14／25。</p>")
    for r in ranked:
        d = frames.get(r.symbol)
        jpg = draw_one(d, r, spec) if d is not None else None
        cap = (
            f"{bt.sym_label(r.symbol)} · {hm(r.time_ms)} · 進場 {r.entry:g}"
        )
        mv = r.moves.get(spec.hold4)
        if mv and mv.ret_pct is not None:
            cap += f" · 4h {mv.ret_pct:+.2f}%"
        parts.append(f"<div class='card'><div class='cap'>{html.escape(cap)}</div>")
        if jpg:
            b64 = base64.b64encode(jpg).decode("ascii")
            parts.append(f"<img src='data:image/jpeg;base64,{b64}' alt='{html.escape(r.symbol)}'/>")
        else:
            parts.append("<p class='note'>圖產生失敗。</p>")
        parts.append("</div>")
    return "\n".join(parts), st


def run_tf(spec: bt.TfSpec, symbols: list[str]) -> tuple[list, dict[str, dict]]:
    rows = []
    frames: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_notify, s, spec): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, hits = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            if d is not None:
                frames[sym] = d
            rows.extend(hits)
            if done % 40 == 0 or done == len(symbols):
                print(f"  {spec.signal} {done}/{len(symbols)}  訊號 {len(rows)}  {time.time()-t0:.1f}s", flush=True)
    rows.sort(key=lambda r: r.time_ms)
    return rows, frames


def main() -> int:
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    print(f"窗 {START} → {END}  GMT+8", flush=True)
    print("載入標的…", flush=True)
    symbols = universe(top_n=100)
    print(f"{len(symbols)} 個成交額前100 USDT 永續（回測當下排名，非 10/17 當時）", flush=True)

    spec15 = bt.TF_SPECS["15m"]
    spec1h = bt.TF_SPECS["1h"]
    rows15, frames15 = run_tf(spec15, symbols)
    rows1h, frames1h = run_tf(spec1h, symbols)

    rules15 = """
    <ol>
      <li>週期：15 分已收盤 K（未收完那根丟掉）。</li>
      <li>標的：幣安 USDT 永續，24h quoteVolume 前 100（含 PERPETUAL / TRADIFI_PERPETUAL，排除 INDEX）。</li>
      <li>訊號 K 本身：收盤 &gt; SMA7 &gt; SMA25，且收盤 &gt; 15m SMA200。SMA14 只畫圖。</li>
      <li>剛站上：前一根收盤 ≤ 15m SMA200，本根收盤 &gt; SMA200。</li>
      <li>站上後連續 3 根收盤都在 SMA200 上（含剛站上那根）。中間根只要收盤 &gt; SMA200；第 3 根仍要 7&gt;25。訊號時間＝第 3 根。</li>
      <li>第 3 根當下，1h SMA25 未下彎：當下 1h SMA25 ≥ 前一根已收 1h 的 SMA25；未收完的 1h 用訊號收盤代入。</li>
      <li>同一時刻 1h：收盤 &gt; SMA7 &gt; SMA25（未收完同樣代入訊號收盤）。</li>
      <li>進場價：訊號下一根 15m 開盤。沒有停損、沒有停利。</li>
      <li>時間出場（回測用）：進場後持有 16 根 15m（4 小時）以該根收盤計報酬；另報 15m／1h／8h。假突破標記＝進場那根收盤又跌破當時 SMA200。</li>
    </ol>
    """
    rules1h = """
    <ol>
      <li>週期：1 小時已收盤 K。</li>
      <li>標的：同上，成交額前 100 USDT 永續。</li>
      <li>本根剛站上 1h SMA200，且收盤 &gt; SMA7 &gt; SMA25。不要求連 3 根。SMA14 只畫圖。</li>
      <li>當時 1h SMA25 未下彎（當下 ≥ 前一根已收）。1h 本身已是 7&gt;25，不再另要一次 1h 7&gt;25。4h SMA200 只對照、不擋單。</li>
      <li>進場價：訊號下一根 1h 開盤。沒有停損、沒有停利。</li>
      <li>時間出場：進場後持有 4 根 1h（4 小時）收盤；另報 1h／2h／8h。</li>
    </ol>
    """

    html15, st15 = section_html("15 分通知", spec15, rows15, frames15, rules15)
    html1h, st1h = section_html("1 小時通知", spec1h, rows1h, frames1h, rules1h)
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>幣安 MA200 通知回測 2026-10-17～10-30</title>
<style>
body{{background:#0c1210;color:#d5e0d8;font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px;line-height:1.5}}
h1,h2,h3{{color:#e8f0ea}}
a{{color:#c9a227}}
.note{{color:#8aa193}}
.rules ol{{padding-left:1.3rem}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.kpi{{background:#16201c;border:1px solid #2a3a33;border-radius:10px;padding:12px 16px;min-width:120px}}
.kpi .k{{font-size:12px;color:#8aa193}}
.kpi .v{{font-size:22px;color:#e8f0ea}}
.card{{background:#101814;border:1px solid #2a3a33;border-radius:12px;padding:10px;margin:14px 0}}
.cap{{font-size:13px;color:#c8d5cc;margin-bottom:8px}}
img{{width:100%;height:auto;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #2a3a33;padding:6px 8px;text-align:left}}
.mono{{font-variant-numeric:tabular-nums}}
.pos{{color:#3dba7a}} .neg{{color:#e35d5d}}
.table-wrap{{overflow:auto;margin:12px 0 24px}}
</style>
</head>
<body>
<h1>幣安 MA200 多頭通知　2026-10-17 → 2026-10-30（GMT+8）</h1>
<p class="note">產生於 {html.escape(now)} GMT+8。這份策略是幣安 U 本位永續，<strong>不是台股</strong>，因此沒有跑 09:00–13:30 正規盤／颱風休市過濾。只回測、不下單。報酬未扣手續費、資金費與稅。</p>
<p>標的範圍：回測當下 24h 成交額前 <strong>{len(symbols)}</strong> 個 USDT 永續。均線皆為簡單移動平均 SMA。</p>
{html15}
{html1h}
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"\n15m n={st15['n']}  4h WR {st15['wr']:.1f}% avg {st15['avg']:+.3f}% med {st15['med']:+.3f}%")
    print(f"  best {st15['best']}")
    print(f"  worst {st15['worst']}")
    print(f"1h  n={st1h['n']}  4h WR {st1h['wr']:.1f}% avg {st1h['avg']:+.3f}% med {st1h['med']:+.3f}%")
    print(f"  best {st1h['best']}")
    print(f"  worst {st1h['worst']}")
    print(f"已寫入 {OUT_HTML}  size={OUT_HTML.stat().st_size/1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
