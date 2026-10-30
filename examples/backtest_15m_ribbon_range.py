#!/usr/bin/env python3
"""固定日期回測：15 分同時站上 MA7/14/25/99/120，每一筆內嵌成單一 HTML。

只讀幣安 K 線，不下單。

    python3 examples/backtest_15m_ribbon_range.py
"""

from __future__ import annotations

import base64
import html
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.backtest_15m_ribbon import (  # noqa: E402
    INTERVAL_MS,
    KEEP,
    PAL,
    file_base,
    get_json,
    sym_label,
)
from nq.ribbon15 import (  # noqa: E402
    HORIZONS,
    MA_PERIODS,
    SIGNAL_MA_PERIODS,
    SignalRow,
    add_mas,
    detect_long_breaks,
    fail_rate,
    forward_moves,
    sma,
)

TZ = timezone(timedelta(hours=8))
START = datetime(2026, 10, 17, 0, 0, tzinfo=TZ)
END = datetime(2026, 10, 31, 0, 0, tzinfo=TZ)  # 含 10/30 全天，不含 10/31
OUT = Path("docs/binance/ma-ribbon-15m-20261017-20261030.html")
HORIZON_LABEL = {1: "15m", 2: "30m", 4: "1h", 8: "2h", 16: "4h", 32: "8h"}
HEADLINE_H = 16  # 程式沒有停損停利；摘要用持有 16 根（4 小時）收盤


def fetch_klines_through_now(sym: str) -> dict | None:
    """從訊號窗口前再抓 MA200 熱身，一直抓到最新已收盤 15 分。"""
    interval_ms = INTERVAL_MS["15m"]
    start_ms = int(START.timestamp() * 1000) - (max(MA_PERIODS) + 8) * interval_ms
    end_ms = int(time.time() * 1000)
    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        raw = get_json(
            "/fapi/v1/klines",
            {"symbol": sym, "interval": "15m", "startTime": cursor, "limit": 1500},
        )
        if not raw:
            break
        rows.extend(raw)
        last_open = int(raw[-1][0])
        nxt = last_open + interval_ms
        if nxt <= cursor or len(raw) < 1500:
            break
        cursor = nxt
    dedup = []
    seen = set()
    for x in rows:
        t = int(x[0])
        if t in seen or t < start_ms:
            continue
        seen.add(t)
        dedup.append(x)
    dedup.sort(key=lambda x: int(x[0]))
    if int(dedup[-1][0]) + interval_ms > end_ms if dedup else True:
        if dedup and int(dedup[-1][0]) + interval_ms > end_ms:
            dedup = dedup[:-1]
    if len(dedup) < max(MA_PERIODS) + 5:
        return None
    return {
        "t": np.array([int(x[0]) for x in dedup], np.int64),
        "o": np.array([float(x[1]) for x in dedup]),
        "h": np.array([float(x[2]) for x in dedup]),
        "l": np.array([float(x[3]) for x in dedup]),
        "c": np.array([float(x[4]) for x in dedup]),
        "v": np.array([float(x[5]) for x in dedup]),
    }


def scan_symbol(sym: str) -> tuple[str, dict | None, list[SignalRow]]:
    raw = fetch_klines_through_now(sym)
    if raw is None:
        return sym, None, []
    d = add_mas(raw)
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows = []
    for br in detect_long_breaks(d, periods=SIGNAL_MA_PERIODS):
        ts = int(d["t"][br.idx])
        if ts < start_ms or ts >= end_ms:
            continue
        entry, moves = forward_moves(d, br)
        if np.isnan(entry):
            continue
        rows.append(SignalRow(symbol=sym, break_=br, time_ms=ts, entry=entry, moves=moves))
    return sym, d, rows


def horizon_stats(rows: list[SignalRow], h: int) -> dict:
    vals = [
        (r, r.moves[h].ret_pct)
        for r in rows
        if r.moves.get(h) and r.moves[h].ret_pct is not None
    ]
    if not vals:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "med": 0.0, "best": None, "worst": None}
    arr = np.array([v for _, v in vals], dtype=float)
    best = max(vals, key=lambda x: x[1])
    worst = min(vals, key=lambda x: x[1])
    return {
        "n": int(len(arr)),
        "wr": float((arr > 0).mean() * 100.0),
        "avg": float(arr.mean()),
        "med": float(np.median(arr)),
        "best": best,
        "worst": worst,
    }


def chart_png(row: SignalRow, d: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    i = row.break_.idx
    exit_i = i + HEADLINE_H  # forward_moves: j = (i+1) + 16 - 1
    entry_i = i + 1
    a0 = max(0, i - 36)
    a1 = min(len(d["c"]), max(i + 24, exit_i + 3))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl]
    fig, ax = plt.subplots(figsize=(8.8, 4.2), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.6)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.2))
    for n, col in PAL.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.0, label=f"MA{n}")
    sx = i - a0
    ax.axvline(sx, color="#c9a227", ls="--", lw=0.9, label="signal")
    ex = entry_i - a0
    if 0 <= ex < len(xs):
        ax.scatter([ex], [d["o"][entry_i]], marker="^", s=42, color="#7CFFB2", zorder=6, label="entry")
    ret = row.moves.get(HEADLINE_H)
    ret_pct = ret.ret_pct if ret else None
    if ret_pct is not None and 0 <= exit_i - a0 < len(xs):
        ax.scatter([exit_i - a0], [d["c"][exit_i]], marker="v", s=42, color="#ffb4a2", zorder=6, label="exit 4h")
    when = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m-%d %H:%M")
    ret_s = "n/a" if ret_pct is None else f"{ret_pct:+.2f}%"
    ax.set_title(
        f"{file_base(row.symbol)} {when} entry {row.entry:g} 4h {ret_s}",
        color="#e8f0ea",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor="#c8d5cc", ncol=8)
    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fmt_trade(pair) -> str:
    if pair is None:
        return "—"
    row, ret = pair
    when = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m-%d %H:%M")
    return f"{html.escape(sym_label(row.symbol))} {when} 進場 {row.entry:g}　{ret:+.2f}%"


def main() -> int:
    from examples.backtest_15m_ribbon import universe

    print("載入標的…", flush=True)
    symbols = universe(asset="all")
    print(f"掃描 {len(symbols)} 個流動 USDT 永續，訊號 {START:%Y-%m-%d} → {END - timedelta(days=1):%Y-%m-%d} GMT+8", flush=True)
    rows: list[SignalRow] = []
    data: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s): s for s in symbols}
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
                print(f"  {done}/{len(symbols)}  訊號 {len(rows)}  {time.time()-t0:.1f}s", flush=True)
    rows.sort(key=lambda r: r.time_ms)
    stats = {h: horizon_stats(rows, h) for h in HORIZONS}
    h4 = stats[HEADLINE_H]
    print(
        f"訊號 {len(rows)}  4h n={h4['n']} 勝率 {h4['wr']:.1f}% 均 {h4['avg']:+.3f}% 中位 {h4['med']:+.3f}%",
        flush=True,
    )
    print("畫圖…", flush=True)
    images = []
    for n, row in enumerate(rows, 1):
        images.append(chart_png(row, data[row.symbol]))
        if n % 50 == 0 or n == len(rows):
            print(f"  圖 {n}/{len(rows)}", flush=True)

    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    cards = []
    for row, b64 in zip(rows, images):
        ret = row.moves.get(HEADLINE_H)
        ret_pct = ret.ret_pct if ret else None
        cls = "pos" if ret_pct and ret_pct > 0 else "neg"
        ret_s = f"{ret_pct:+.2f}%" if ret_pct is not None else "尚未滿 4h"
        bits = []
        for h in HORIZONS:
            m = row.moves.get(h)
            v = m.ret_pct if m else None
            bits.append(f"{HORIZON_LABEL[h]} {'—' if v is None else f'{v:+.2f}%'}")
        when = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m-%d %H:%M")
        exit_px = "—"
        if ret_pct is not None:
            j = row.break_.idx + HEADLINE_H
            exit_px = f"{float(data[row.symbol]['c'][j]):g}"
        cards.append(
            f"""<section class="card">
  <h2>{html.escape(sym_label(row.symbol))} {when} · 4h <span class="{cls}">{ret_s}</span></h2>
  <img alt="{html.escape(row.symbol)} {when}" src="data:image/png;base64,{b64}"/>
  <p class="note">進場 {row.entry:g}（下一根開盤） · 4h 出場價 {exit_px}（持有 16 根的收盤） · 寬度 {row.width_pct:.3f}% · 量比 {row.vol_ratio:.2f}× · {'實體穿越' if row.body_through else '影線穿過、開盤已在帶內'}<br/>{' · '.join(bits)}</p>
</section>"""
        )

    def kpi(k, v, cls=""):
        return f'<div class="kpi"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'

    wr_cls = "pos" if h4["wr"] >= 50 else "neg"
    avg_cls = "pos" if h4["avg"] > 0 else "neg"
    kpis = "".join(
        [
            kpi("訊號數", str(len(rows))),
            kpi("4h 已結算", str(h4["n"])),
            kpi("4h 勝率", f"{h4['wr']:.1f}%", wr_cls),
            kpi("4h 平均", f"{h4['avg']:+.3f}%", avg_cls),
            kpi("4h 中位", f"{h4['med']:+.3f}%", "pos" if h4["med"] > 0 else "neg"),
            kpi("下一根假突破", f"{fail_rate(rows):.1f}%"),
        ]
    )
    heads = "".join(f"<th>{HORIZON_LABEL[h]}</th>" for h in HORIZONS)
    body_rows = []
    for name, getter in (
        ("已結算筆數", lambda s: str(s["n"])),
        ("勝率", lambda s: f"{s['wr']:.1f}%"),
        ("平均", lambda s: f"{s['avg']:+.3f}%"),
        ("中位數", lambda s: f"{s['med']:+.3f}%"),
        ("最好", lambda s: fmt_trade(s["best"])),
        ("最差", lambda s: fmt_trade(s["worst"])),
    ):
        tds = "".join(f"<td>{getter(stats[h])}</td>" for h in HORIZONS)
        body_rows.append(f"<tr><th>{name}</th>{tds}</tr>")

    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>近兩週 15分同時站上 7/14/25/99/120 · 2026-10-17 至 2026-10-30</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:20px 14px 64px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:16px;margin:0 0 8px}}
.sub,.note{{color:#8aa193;line-height:1.65}}
.card{{background:#14201b;border:1px solid rgba(232,240,234,.12);border-radius:12px;padding:14px;margin:14px 0}}
img{{width:100%;height:auto;border-radius:8px;background:#101814}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.kpi{{border:1px solid rgba(232,240,234,.12);border-radius:10px;padding:10px}}
.kpi .k{{color:#8aa193;font-size:12px}} .kpi .v{{font-size:18px;margin-top:4px}}
.pos{{color:#3dba7a}}.neg{{color:#e35d5d}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 8px;border-bottom:1px solid rgba(232,240,234,.12);text-align:right;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}}
ol{{margin:8px 0 0;padding-left:1.2em}}
li{{margin:6px 0}}
code{{color:#e8d48b}}
@media(max-width:720px){{.kpis{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <h1>近兩週 · 15 分同時站上 7 / 14 / 25 / 99 / 120</h1>
  <p class="sub">幣安 U 本位永續。訊號 K 的開盤時間在 2026-10-17 00:00 至 2026-10-30 23:45（GMT+8）。產生於 {now}。只回測，沒有下單。未扣手續費、資金費率、滑價與稅。</p>
  <div class="card">
    <h2>觸發規則（照程式，不是另做一套）</h2>
    <ol>
      <li><b>週期。</b>只做多，只用 15 分 K（<code>interval="15m"</code>）。未完成的最後一根先丟掉。</li>
      <li><b>標的。</b><code>universe(asset="all")</code>：USDT 計價、狀態 TRADING、合約類型 PERPETUAL 或 TRADIFI_PERPETUAL、標的類型不是 INDEX。24 小時成交額 ≥ 5,000,000 USDT，另外保留 {html.escape(", ".join(sorted(KEEP)))}。成交額用這次回測當下的 24h ticker，不是 10/17 當時的成交額。股票永續若當下成交額達標也會在裡面；這不是台股現股。</li>
      <li><b>均線。</b>收盤價簡單移動平均。訊號用 MA{html.escape(" / ".join(str(n) for n in SIGNAL_MA_PERIODS))}。圖上另畫 MA200，條件不算它。</li>
      <li><b>進場條件。</b>對已收盤的第 i 根：前一根收盤 &lt; 前一根這五條均線的最小值；這一根收盤 &gt; 這一根這五條均線的最大值；這一根最低價 ≤ 前一根均線最小值（跳空飛過帶子的不算）。寬度、量比、實體穿越只做註記，不過濾。</li>
      <li><b>進場價。</b>下一根 15 分的開盤。沒有下一根的訊號不計。</li>
      <li><b>停損、停利。</b>程式沒有。假突破只是統計：進場那根收盤 &lt; 訊號那根五條均線的最大值，並不是出場單。</li>
      <li><b>時間出場。</b>從進場那根算起持有 1、2、4、8、16、32 根，對應 15 分、30 分、1 小時、2 小時、4 小時、8 小時，用該根收盤相對進場價。摘要的勝率與報酬用 4 小時（16 根）。還沒走完的不計入該檔。</li>
      <li><b>台股。</b>這支策略沒有台股現股、沒有 09:00–13:30、也不跳過休市日。10/23 幣安照常有 K 線。沒有用 Shioaji，也沒有送出任何委託。</li>
    </ol>
  </div>
  <div class="kpis">{kpis}</div>
  <div class="card">
    <h2>固定時間出場（未扣成本）</h2>
    <div style="overflow:auto">
      <table>
        <thead><tr><th></th>{heads}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <p class="note">勝率 = 報酬 &gt; 0 的比例。最好、最差是該持有時間已結算的單筆。</p>
  </div>
  <h2>每一筆訊號</h2>
  <p class="sub">黃虛線是訊號 K。綠色上三角是進場（下一根開盤）。橘色下三角是 4 小時出場（進場後第 16 根收盤）。均線：黃 7、橙 14、紫 25、藍 99、青 120、白 200。</p>
  {''.join(cards)}
</div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"已寫入 {OUT}  {OUT.stat().st_size/1_048_576:.1f} MB  圖 {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
