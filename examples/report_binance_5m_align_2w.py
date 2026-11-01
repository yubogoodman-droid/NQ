#!/usr/bin/env python3
"""幣安 5m 多頭排列：指定區間自包含 HTML 回測報告（圖內嵌、繁中）。"""
from __future__ import annotations

import base64
import io
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_5m_align import (  # noqa: E402
    TZ,
    UNIVERSE_LIMIT,
    _fmt,
    _pnl_cls,
    backtest_all,
    draw_chart,
    hm,
    parse_day_ms,
    summarize_hits,
    universe,
)


OUT = Path(__file__).resolve().parents[1] / "docs" / "binance-5m-align-2w" / "index.html"

RULES_HTML = """
<ol>
<li><b>週期</b>：進場用已收盤的 <b>5 分鐘 K</b>。15 分鐘與 1 小時用「形成中」的那根：把當下 5m 收盤價當成該週期當根收盤，均線不算這根之後的收盤（無未來函數）。</li>
<li><b>標的範圍</b>：幣安 USDT 計價永續（<code>PERPETUAL</code> / <code>TRADIFI_PERPETUAL</code>）、狀態 <code>TRADING</code>、排除 <code>INDEX</code>。依當下 24h <code>quoteVolume</code> 由高到低取前 <b>100</b> 檔（與實盤 <code>universe()</code> 相同，是回測當下的成交額排名，不是歷史當日排名）。</li>
<li><b>均線</b>：收盤價簡單移動平均（SMA）。5m 用 7 / 14 / 25 / 200；形成中 15m 用 MA200；形成中 1h 用 MA99 與 MA200。</li>
<li><b>進場條件（須同時成立）</b>
  <ol>
    <li>5m：<code>SMA7 &gt; SMA14 &gt; SMA25</code>，且收盤 &gt; SMA200。</li>
    <li>5m：前一根收盤 &lt; 前一根 SMA200，這一根收盤 &gt; 這一根 SMA200（從下站上；已在上面只是短均排好的不算）。</li>
    <li>15m：前一根已收盤 15m 收在其 SMA200 下，當下價站上形成中 15m 的 SMA200。</li>
    <li>1h：當下價 &gt; 形成中 1h 的 SMA99 與 SMA200。</li>
  </ol>
</li>
<li><b>進場價</b>：觸發那根 5m 的<b>收盤價</b>。同一標的同一根 5m 不重發（鍵為 <code>symbol:開盤時間</code>）。</li>
<li><b>停損 / 停利</b>：程式<b>沒有</b>停損、停利、移動停利或時間強制平倉單。</li>
<li><b>時間出場（僅回測前瞻）</b>：從進場 5m 收盤再往後 3 / 6 / 12 / 24 根 5m（15 / 30 / 60 / 120 分鐘）的收盤，算報酬％。圖上的出場點標的是 <b>60 分鐘</b>那根收盤。數字<b>未扣手續費與資金費</b>。</li>
</ol>
<p>此策略是<b>幣安 USDT 永續</b>，24 小時交易，沒有台股 09:00–13:30 或颱風休市。台股規則不適用。</p>
"""


def png_b64(sym: str, d5: dict, i: int) -> str:
    buf = io.BytesIO()
    exit_i = i + 12 if i + 12 < len(d5["c"]) else None
    drawn = draw_chart(sym, d5, i, buf, ahead=24, exit_i=exit_i)
    if not drawn:
        return ""
    if hasattr(buf, "getvalue"):
        raw = buf.getvalue()
        if raw:
            return base64.b64encode(raw).decode("ascii")
    return ""


def _ext_stats(hits: list[dict], key: str) -> dict:
    rows = [(h, float(h[key])) for h in hits if h.get(key) is not None]
    if not rows:
        return {"n": 0, "win_rate": 0.0, "avg": 0.0, "med": 0.0, "best": None, "worst": None}
    vals = [v for _, v in rows]
    best_h, best_v = max(rows, key=lambda x: x[1])
    worst_h, worst_v = min(rows, key=lambda x: x[1])
    wins = sum(1 for v in vals if v > 0)

    def slim(h: dict, v: float) -> dict:
        return {"symbol": h["symbol"], "t": h["t"], "close": h["close"], "_v": v}

    return {
        "n": len(vals),
        "win_rate": 100.0 * wins / len(vals),
        "avg": float(np.mean(vals)),
        "med": float(np.median(vals)),
        "best": slim(best_h, best_v),
        "worst": slim(worst_h, worst_v),
    }


def write_embedded(path: Path, hits: list[dict], stats: dict, funnel: dict, period: str) -> Path:
    s60 = _ext_stats(hits, "60m")
    cards = []
    for n, h in enumerate(hits, 1):
        b64 = png_b64(h["symbol"], h["d5"], h["i"])
        img = (
            f"<img alt='{escape(h['symbol'])}' src='data:image/png;base64,{b64}' "
            "style='width:100%;display:block;border-radius:10px'/>"
            if b64
            else "<p class='muted'>圖未產出</p>"
        )
        ext = (h["close"] / h["m200"] - 1) * 100
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{n} · {escape(h['symbol'])}</span>"
            f"<span class='trade-time'>{escape(hm(h['t']))}</span></div>"
            f"<div class='card-pnl {_pnl_cls(h.get('60m'))}'>60m {_fmt(h.get('60m'))}</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag'>進場收盤 {h['close']:g}</span>"
            f"<span class='tag'>15m {_fmt(h.get('15m'))}</span>"
            f"<span class='tag'>30m {_fmt(h.get('30m'))}</span>"
            f"<span class='tag'>120m {_fmt(h.get('120m'))}</span></div>"
            "<pre class='trade-detail'>"
            f"5m MA7 {h['m7']:g} &gt; MA14 {h['m14']:g} &gt; MA25 {h['m25']:g}\n"
            f"5m 從 MA200 下站上 {h['m200']:g}（{ext:+.2f}%）\n"
            f"15m 同步從 MA200 下站上\n"
            f"1h {h['h_close']:g} &gt; MA99 {h['h_m99']:g} / MA200 {h['h_m200']:g}\n"
            f"綠虛線＝進場 5m 收盤；藍虛線＝60 分鐘後收盤（無停損停利）"
            "</pre>"
            f"{img}"
            "</article>"
        )

    def bw(label: str, row: dict | None) -> str:
        if not row:
            return f"<p class='muted'>{label}：無</p>"
        return (
            f"<p>{label}：<b>{escape(row['symbol'])}</b> {escape(hm(row['t']))}　"
            f"<span class='{_pnl_cls(row['_v'])}'>{_fmt(row['_v'])}</span>　進場 {row['close']:g}</p>"
        )

    rows = []
    for i, h in enumerate(hits, 1):
        rows.append(
            "<tr>"
            f"<td>{i}</td><td>{escape(h['symbol'])}</td><td>{escape(hm(h['t']))}</td>"
            f"<td>{h['close']:g}</td>"
            f"<td class='{_pnl_cls(h.get('15m'))}'>{_fmt(h.get('15m'))}</td>"
            f"<td class='{_pnl_cls(h.get('30m'))}'>{_fmt(h.get('30m'))}</td>"
            f"<td class='{_pnl_cls(h.get('60m'))}'>{_fmt(h.get('60m'))}</td>"
            f"<td class='{_pnl_cls(h.get('120m'))}'>{_fmt(h.get('120m'))}</td>"
            "</tr>"
        )

    def line(name: str, st: dict) -> str:
        return (
            f"{name}　n={st['n']}　勝率 {st['win_rate']:.1f}%　"
            f"平均 {_fmt(st['avg'])}　中位 {_fmt(st['med'])}"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 5m 多頭排列回測 · {escape(period)}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:16px 14px 40px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 8px}} h2{{font-size:16px;margin:16px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6}}
ol,li{{line-height:1.55;font-size:13px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:110px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}}
th,td{{padding:6px 4px;border-bottom:1px solid #21262d;text-align:right}}
th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px}}
code{{background:#0d1117;padding:1px 5px;border-radius:4px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 5m 多頭排列 · {escape(period)}</h1>
<p class="muted">回測區間台北時間 {escape(period)}。標的為回測當下 24h 成交額前 {funnel.get('symbols', 0)} 檔 USDT 永續。未扣手續費與資金費。</p>
<h2>觸發規則（依 examples/watch_binance_5m_align.py）</h2>
{RULES_HTML}
<h2>統計（主看出場＝進場後 60 分鐘收盤）</h2>
<div class="cards">
<div class="card">訊號數<b>{stats['count']}</b></div>
<div class="card">標的數<b>{stats['symbols']}</b></div>
<div class="card">60m 勝率<b>{s60['win_rate']:.1f}%</b></div>
<div class="card">60m 平均<b class="{_pnl_cls(s60['avg'])}">{_fmt(s60['avg'])}</b></div>
<div class="card">60m 中位<b class="{_pnl_cls(s60['med'])}">{_fmt(s60['med'])}</b></div>
</div>
<p class="muted">漏斗：5m 從下站上且多排 {funnel.get('five_new', 0)} → 15m 也從下站上 {funnel.get('fifteen', 0)} → 小時過濾 {funnel.get('hits', 0)} · 讀檔失敗 {funnel.get('errors', 0)}</p>
<p class="muted">{escape(line('15m', stats['15m']))}<br/>
{escape(line('30m', stats['30m']))}<br/>
{escape(line('60m', stats['60m']))}<br/>
{escape(line('120m', stats['120m']))}</p>
{bw('60m 最好一筆', s60['best'])}
{bw('60m 最差一筆', s60['worst'])}
</section>
<h2>每筆訊號（綠＝進場，藍＝60m 出場）</h2>
{''.join(cards) or "<div class='empty'>這段期間沒有符合的訊號</div>"}
<section class="summary">
<h2>全部訊號表</h2>
<table>
<thead><tr><th>#</th><th>標的</th><th>進場時間</th><th>進場價</th><th>15m</th><th>30m</th><th>60m</th><th>120m</th></tr></thead>
<tbody>
{''.join(rows) or "<tr><td colspan='8'>無</td></tr>"}
</tbody>
</table>
</section>
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    start = "2026-10-17"
    end = "2026-10-30"
    start_ms = parse_day_ms(start)
    now_ms = int(datetime.now(TZ).timestamp() * 1000)
    end_ms = min(parse_day_ms(end, end_of_day=True), now_ms)
    period = f"{start} → {end}"
    print("載入標的…", flush=True)
    symbols = universe(UNIVERSE_LIMIT)
    print(f"回測成交額前 {len(symbols)} 檔 · {period}", flush=True)
    hits, funnel = backtest_all(symbols, start_ms, end_ms)
    stats = summarize_hits(hits)
    print(
        f"訊號 {stats['count']} / {stats['symbols']} 檔　"
        f"60m 勝率 {stats['60m']['win_rate']:.1f}% 均 {stats['60m']['avg']:+.2f}% 中位 {stats['60m']['med']:+.2f}%",
        flush=True,
    )
    out = write_embedded(OUT, hits, stats, funnel, period)
    print(f"報告 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
