"""
Generate per-symbol HTML charts for all shadow-neckline signal coins,
plus an index page. Publishes under docs/charts/ (and optionally gh-pages).
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pandas_ta as ta

CACHE = Path("/tmp/binance_um_klines")
DAY = "2026-08-09"
HIST = "2026-08-08"
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144}
DATA_NOTE = ""


STEM_ALIAS = {
    "龙虾USDT": "LONGXIAUSDT",
}


def file_stem(symbol: str) -> str:
    # BICO/USDT -> BICOUSDT
    stem = symbol.replace("/", "")
    return STEM_ALIAS.get(stem, stem)


def load_ohlcv(sym: str) -> pd.DataFrame:
    # accept chart stem or alias target
    candidates = [sym]
    for src, dst in STEM_ALIAS.items():
        if sym == src:
            candidates.append(dst)
        if sym == dst:
            candidates.append(src)
    paths = []
    for stem in candidates:
        for d in (HIST, DAY):
            p = CACHE / f"{stem}-5m-{d}.csv"
            if p.exists():
                paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No klines for {sym} in {CACHE} ({HIST}/{DAY})")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def short_pnl_table(df: pd.DataFrame, signal_rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, s in signal_rows.iterrows():
        entry_ts = int(pd.Timestamp(s["time_utc"], tz="UTC").timestamp() * 1000)
        idxs = df.index[df["timestamp"] == entry_ts].tolist()
        if not idxs or idxs[0] + 1 >= len(df):
            continue
        i = idxs[0]
        entry = float(df.loc[i + 1, "open"])
        pnl = {}
        for name, n in HORIZONS.items():
            j = i + n
            if j >= len(df):
                pnl[name] = None
                continue
            exit_px = float(df.loc[j, "close"])
            pnl[name] = round((entry - exit_px) / entry * 100, 2)
        row = {
            "time_utc": s["time_utc"],
            "price": float(s["price"]),
            "entry": entry,
            "bias": float(s["bias"]),
            "line_val": float(s["line_val"]),
            "sma14": float(s["sma14"]),
            "pnl": pnl,
            "time": entry_ts // 1000,
            "entry_time": (entry_ts // 1000) + 300,  # next 5m bar = actual short entry
            "dist_ma99_pct": None,
            "dist_ma200_pct": None,
            "close_break_pct": None,
            "vol_ratio": None,
        }
        if "dist_ma99_pct" in s and pd.notna(s["dist_ma99_pct"]):
            row["dist_ma99_pct"] = float(s["dist_ma99_pct"])
        if "dist_ma200_pct" in s and pd.notna(s["dist_ma200_pct"]):
            row["dist_ma200_pct"] = float(s["dist_ma200_pct"])
        if "close_break_pct" in s and pd.notna(s["close_break_pct"]):
            row["close_break_pct"] = float(s["close_break_pct"])
        if "vol_ratio" in s and pd.notna(s["vol_ratio"]):
            row["vol_ratio"] = float(s["vol_ratio"])
        out.append(row)
    return out


def chart_payload(symbol: str, signal_rows: pd.DataFrame) -> dict:
    sym = file_stem(symbol)
    df = load_ohlcv(sym)
    for n in (7, 14, 25, 99, 200):
        df[f"sma{n}"] = ta.sma(df["close"], length=n)

    start_ts = int(pd.Timestamp(f"{HIST} 18:00:00", tz="UTC").timestamp() * 1000)
    plot = df[df["timestamp"] >= start_ts].copy()

    def series(col: str):
        return [
            {"time": int(r["timestamp"] // 1000), "value": float(r[col])}
            for _, r in plot.iterrows()
            if pd.notna(r[col])
        ]

    candles = [
        {
            "time": int(r["timestamp"] // 1000),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        for _, r in plot.iterrows()
    ]
    signals = short_pnl_table(df, signal_rows)
    return {
        "symbol": symbol,
        "day": DAY,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "candles": candles,
        "sma7": series("sma7"),
        "sma14": series("sma14"),
        "sma25": series("sma25"),
        "sma99": series("sma99"),
        "sma200": series("sma200"),
        "signals": signals,
    }


def render_symbol_html(
    data: dict,
    index_href: str = "./index.html",
    badge: str = "",
    filter_note: str = "",
) -> str:
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    note_html = (
        f'<div class="filter-note">{filter_note}</div>' if filter_note else ""
    )
    show_vol = any(s.get("vol_ratio") is not None for s in data["signals"])
    show_dist = any(s.get("dist_ma99_pct") is not None for s in data["signals"])
    show_dist200 = any(s.get("dist_ma200_pct") is not None for s in data["signals"])
    dist_th = ("<th>爆量</th>" if show_vol else "") + (
        "<th>距SMA99</th>" if show_dist else ""
    ) + ("<th>距SMA200</th>" if show_dist200 else "")
    dist_td_js = ""
    if show_vol:
        dist_td_js += """
          <td class="mono">${s.vol_ratio === null || s.vol_ratio === undefined ? '—' : s.vol_ratio.toFixed(2) + '×'}</td>
"""
    if show_dist:
        dist_td_js += """
          <td class="mono">${s.dist_ma99_pct === null || s.dist_ma99_pct === undefined ? '—' : (s.dist_ma99_pct>=0?'+':'') + s.dist_ma99_pct.toFixed(2) + '%'}</td>
"""
    if show_dist200:
        dist_td_js += """
          <td class="mono">${s.dist_ma200_pct === null || s.dist_ma200_pct === undefined ? '—' : (s.dist_ma200_pct>=0?'+':'') + s.dist_ma200_pct.toFixed(2) + '%'}</td>
"""
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{data['symbol']} 影線頸線 · {data['day']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {{
    --bg0:#0c1210; --bg1:#14201b; --ink:#e8f0ea; --muted:#8aa193;
    --line:rgba(232,240,234,0.12); --long:#3dba7a; --short:#e35d5d;
    --accent:#c9a227; --panel:rgba(20,32,27,0.72);
    --ma7:#f0c14a; --ma14:#7eb6ff; --ma25:#d28cff; --ma99:#5fd2c2; --ma200:#c9a227;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; min-height:100%; }}
  body {{
    font-family:"IBM Plex Sans",sans-serif; color:var(--ink);
    background:
      radial-gradient(1100px 600px at 12% -10%, rgba(201,162,39,.16), transparent 55%),
      radial-gradient(900px 500px at 90% 10%, rgba(61,186,122,.10), transparent 50%),
      linear-gradient(165deg, var(--bg0), var(--bg1) 45%, #0a0f0d);
  }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 20px 48px; }}
  .nav a {{ color:var(--muted); text-decoration:none; font-size:.86rem; }}
  .nav a:hover {{ color:var(--ink); }}
  .badge {{ display:inline-block; margin-top:10px; font-family:"JetBrains Mono",monospace; font-size:.72rem; color:var(--accent); border:1px solid rgba(201,162,39,.35); padding:4px 8px; }}
  .filter-note {{ margin-top:8px; color:var(--muted); font-size:.86rem; line-height:1.45; max-width:42rem; }}
  header {{ display:grid; gap:8px; margin:14px 0 22px; }}
  .brand {{ font-family:"IBM Plex Serif",serif; font-size:clamp(1.8rem,4vw,2.6rem); font-weight:600; letter-spacing:-.02em; }}
  .brand span {{ color:var(--accent); }}
  .sub {{ color:var(--muted); line-height:1.5; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:10px 18px; font-family:"JetBrains Mono",monospace; font-size:.78rem; color:var(--muted); }}
  .meta b {{ color:var(--ink); font-weight:500; }}
  .chart-shell {{ position:relative; border:1px solid var(--line); background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 40%), var(--panel); overflow:hidden; }}
  #chart {{ width:100%; height:min(62vh,560px); }}
  .legend {{ position:absolute; top:12px; left:14px; right:14px; z-index:2; display:flex; flex-wrap:wrap; gap:10px 14px; font-size:.75rem; color:var(--muted); pointer-events:none; }}
  .legend i {{ display:inline-block; width:18px; height:2px; vertical-align:middle; margin-right:6px; }}
  .ma7 i{{background:var(--ma7)}} .ma14 i{{background:var(--ma14)}} .ma25 i{{background:var(--ma25)}}
  .ma99 i{{background:var(--ma99)}} .ma200 i{{background:var(--ma200)}}
  .sig i{{width:8px;height:8px;border-radius:50%;background:var(--short)}}
  section {{ margin-top:22px; }}
  h2 {{ font-family:"IBM Plex Serif",serif; font-size:1.2rem; margin:0 0 6px; }}
  section p {{ color:var(--muted); margin:0 0 14px; font-size:.92rem; }}
  .table-wrap {{ overflow-x:auto; border:1px solid var(--line); }}
  table {{ width:100%; border-collapse:collapse; font-size:.86rem; min-width:720px; }}
  th,td {{ padding:11px 12px; text-align:right; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
  th {{ font-family:"JetBrains Mono",monospace; font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); background:rgba(0,0,0,.22); font-weight:500; }}
  td.mono {{ font-family:"JetBrains Mono",monospace; font-size:.8rem; }}
  .pos {{ color:var(--long); }} .neg {{ color:var(--short); }} .na {{ color:var(--muted); }}
  .note {{ margin-top:14px; color:var(--muted); font-size:.8rem; }}
  @media (max-width:640px) {{ #chart{{height:420px}} .wrap{{padding:18px 12px 36px}} }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="nav"><a href="{index_href}">← 全部訊號幣種</a></div>
    {badge_html}
    {note_html}
    <header>
      <div class="brand">{data['symbol'].split('/')[0]}<span>/{data['symbol'].split('/')[-1]}</span></div>
      <div class="sub">影線頸線破位訊號與做空報酬（{data['day']} UTC，Binance USDT-M 5m）</div>
      <div class="meta">
        <span>進場：<b>訊號下一根開盤</b></span>
        <span>均線：<b>SMA 7 / 14 / 25 / 99 / 200</b></span>
        <span>訊號數：<b>{len(data['signals'])}</b></span>
        <span>更新：<b>{data.get('updated','')}</b></span>
      </div>
    </header>
    <div class="chart-shell">
      <div class="legend">
        <span class="ma7"><i></i>SMA7</span>
        <span class="ma14"><i></i>SMA14</span>
        <span class="ma25"><i></i>SMA25</span>
        <span class="ma99"><i></i>SMA99</span>
        <span class="ma200"><i></i>SMA200</span>
        <span class="sig"><i></i>做空訊號</span>
      </div>
      <div id="chart"></div>
      <div id="jumps" class="jumps"></div>
    </div>
    <section>
      <h2>做空報酬（%）</h2>
      <p>正值表示空單獲利。缺資料以 — 表示（接近日終、持有期超出可用 K 線）。</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>訊號時間</th><th>進場價</th><th>乖離</th>{dist_th}
              <th>15m</th><th>30m</th><th>1h</th><th>2h</th><th>4h</th><th>8h</th><th>12h</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
      <p class="note">資料：Binance Vision USDT-M 日檔。紅色箭頭 = 進場 K；點下方按鈕可跳到該訊號。</p>
    </section>
  </div>
  <style>
    .jumps {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    .jumps button {{
      font-family:"JetBrains Mono",monospace; font-size:.72rem;
      color:#e8f0ea; background:rgba(227,93,93,.18); border:1px solid rgba(227,93,93,.45);
      padding:6px 10px; cursor:pointer;
    }}
    .jumps button:hover {{ background:rgba(227,93,93,.32); }}
  </style>
  <script>
    const DATA = {json.dumps(data, ensure_ascii=False)};
    function fmtPct(v) {{
      if (v === null || v === undefined) return '<span class="na">—</span>';
      const cls = v >= 0 ? 'pos' : 'neg';
      return `<span class="${{cls}}">${{(v>=0?'+':'') + v.toFixed(2)}}%</span>`;
    }}
    const tbody = document.getElementById('tbody');
    for (const s of DATA.signals) {{
      const p = s.pnl;
      tbody.insertAdjacentHTML('beforeend', `
        <tr>
          <td class="mono">${{s.time_utc}}</td>
          <td class="mono">${{Number(s.entry).toPrecision(6)}}</td>
          <td class="mono">${{s.bias.toFixed(2)}}%</td>
          {dist_td_js}
          <td>${{fmtPct(p['15m'])}}</td><td>${{fmtPct(p['30m'])}}</td>
          <td>${{fmtPct(p['1h'])}}</td><td>${{fmtPct(p['2h'])}}</td>
          <td>${{fmtPct(p['4h'])}}</td><td>${{fmtPct(p['8h'])}}</td>
          <td>${{fmtPct(p['12h'])}}</td>
        </tr>`);
    }}
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
      autoSize: true,
      layout: {{ background: {{ type:'solid', color:'transparent' }}, textColor:'#8aa193', fontFamily:'JetBrains Mono, monospace' }},
      grid: {{ vertLines: {{ color:'rgba(232,240,234,0.06)' }}, horzLines: {{ color:'rgba(232,240,234,0.06)' }} }},
      crosshair: {{
        vertLine: {{ color:'rgba(201,162,39,0.45)', labelBackgroundColor:'#c9a227' }},
        horzLine: {{ color:'rgba(201,162,39,0.45)', labelBackgroundColor:'#c9a227' }},
      }},
      rightPriceScale: {{ borderColor:'rgba(232,240,234,0.12)' }},
      timeScale: {{ borderColor:'rgba(232,240,234,0.12)', timeVisible:true, secondsVisible:false }},
    }});
    const candleSeries = chart.addCandlestickSeries({{
      upColor:'#3dba7a', downColor:'#e35d5d',
      borderUpColor:'#3dba7a', borderDownColor:'#e35d5d',
      wickUpColor:'#3dba7a', wickDownColor:'#e35d5d',
    }});
    candleSeries.setData(DATA.candles);
    function addMa(key, color) {{
      const s = chart.addLineSeries({{ color, lineWidth:2, priceLineVisible:false, lastValueVisible:false }});
      s.setData(DATA[key]);
    }}
    addMa('sma7','#f0c14a'); addMa('sma14','#7eb6ff'); addMa('sma25','#d28cff');
    addMa('sma99','#5fd2c2'); addMa('sma200','#c9a227');

    // Visible markers: arrow on signal bar, circle on next-bar entry
    const candleTimes = new Set(DATA.candles.map(c => c.time));
    const markers = [];
    DATA.signals.forEach((s, idx) => {{
      markers.push({{
        time: s.time,
        position: 'aboveBar',
        color: '#ff4d4f',
        shape: 'arrowDown',
        text: `訊號#${{idx+1}}`,
        size: 2,
      }});
      const et = (s.entry_time && candleTimes.has(s.entry_time)) ? s.entry_time : s.time;
      markers.push({{
        time: et,
        position: 'belowBar',
        color: '#ffd666',
        shape: 'circle',
        text: `進場 ${{Number(s.entry).toPrecision(5)}}`,
        size: 3,
      }});
    }});
    candleSeries.setMarkers(markers);

    DATA.signals.forEach((s, idx) => {{
      candleSeries.createPriceLine({{
        price: s.entry,
        color: 'rgba(255,77,79,0.85)',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: `進場#${{idx+1}}`,
      }});
      candleSeries.createPriceLine({{
        price: s.line_val,
        color: 'rgba(227,93,93,0.35)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false,
        title: `頸線#${{idx+1}}`,
      }});
    }});

    function focusSignal(s) {{
      const pad = 12 * 3600; // ±12h
      chart.timeScale().setVisibleRange({{
        from: s.time - pad,
        to: s.time + pad,
      }});
    }}
    // Default view: zoom around signals (not whole 10-day fit, or markers look tiny)
    if (DATA.signals.length) {{
      const times = DATA.signals.map(s => s.time);
      const lo = Math.min(...times) - 24 * 3600;
      const hi = Math.max(...times) + 24 * 3600;
      chart.timeScale().setVisibleRange({{ from: lo, to: hi }});
    }} else {{
      chart.timeScale().fitContent();
    }}

    const jumps = document.getElementById('jumps');
    DATA.signals.forEach((s, idx) => {{
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = `跳到空#${{idx+1}} ${{s.time_utc.slice(5)}}`;
      btn.onclick = () => focusSignal(s);
      jumps.appendChild(btn);
    }});
  </script>
</body>
</html>
"""


def render_index(cards: list[dict], title: str, subtitle: str, extra_nav: str = "") -> str:
    cards_sorted = sorted(cards, key=lambda x: (-x["n"], x["symbol"]))
    items = []
    for c in cards_sorted:
        avg = c["avg_1h"]
        avg_cls = "pos" if avg is not None and avg >= 0 else "neg"
        avg_txt = "—" if avg is None else f"{avg:+.2f}%"
        items.append(
            f"""
        <a class="card" href="./{c['href']}">
          <div class="name">{c['symbol']}</div>
          <div class="row"><span>訊號</span><b>{c['n']}</b></div>
          <div class="row"><span>1h 空均報酬</span><b class="{avg_cls}">{avg_txt}</b></div>
          <div class="times">{c['times']}</div>
        </a>"""
        )
    body = "\n".join(items)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {{
    --bg0:#0c1210; --bg1:#14201b; --ink:#e8f0ea; --muted:#8aa193;
    --line:rgba(232,240,234,0.12); --long:#3dba7a; --short:#e35d5d; --accent:#c9a227;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; font-family:"IBM Plex Sans",sans-serif; color:var(--ink);
    background:
      radial-gradient(1000px 520px at 8% -8%, rgba(201,162,39,.18), transparent 55%),
      radial-gradient(800px 480px at 100% 0%, rgba(61,186,122,.10), transparent 50%),
      linear-gradient(165deg, var(--bg0), var(--bg1) 50%, #0a0f0d);
    min-height:100vh;
  }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:36px 20px 56px; }}
  h1 {{ font-family:"IBM Plex Serif",serif; font-size:clamp(2rem,4vw,2.8rem); margin:0 0 8px; letter-spacing:-.02em; }}
  .sub {{ color:var(--muted); max-width:40rem; line-height:1.55; margin-bottom:28px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px 22px; margin-bottom:26px; font-family:"JetBrains Mono",monospace; font-size:.8rem; color:var(--muted); }}
  .stats b {{ color:var(--ink); }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; }}
  a.card {{
    display:block; text-decoration:none; color:inherit;
    border:1px solid var(--line); padding:16px 16px 14px;
    background:linear-gradient(180deg, rgba(255,255,255,.04), transparent 55%), rgba(20,32,27,.65);
    transition: border-color .15s ease, transform .15s ease;
  }}
  a.card:hover {{ border-color: rgba(201,162,39,.55); transform: translateY(-2px); }}
  .name {{ font-family:"IBM Plex Serif",serif; font-size:1.25rem; margin-bottom:10px; }}
  .row {{ display:flex; justify-content:space-between; gap:10px; font-size:.86rem; color:var(--muted); margin:4px 0; }}
  .row b {{ color:var(--ink); font-family:"JetBrains Mono",monospace; font-weight:500; }}
  .pos {{ color:var(--long)!important; }} .neg {{ color:var(--short)!important; }}
  .times {{ margin-top:10px; color:var(--muted); font-size:.72rem; font-family:"JetBrains Mono",monospace; line-height:1.45; }}
</style>
</head>
<body>
  <div class="wrap">
    {extra_nav}
    <h1>{title}</h1>
    <p class="sub">{subtitle}</p>
    <div class="stats">
      <span>幣種 <b>{len(cards_sorted)}</b></span>
      <span>訊號總數 <b>{sum(c['n'] for c in cards_sorted)}</b></span>
      <span>均線 <b>7 / 14 / 25 / 99 / 200</b></span>
    </div>
    <div class="grid">
      {body}
    </div>
  </div>
</body>
</html>
"""


def render_hub(baseline_n: int, balanced_n: int, strict_n: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>影線頸線回測圖表 · {DAY}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {{ --bg0:#0c1210; --bg1:#14201b; --ink:#e8f0ea; --muted:#8aa193; --line:rgba(232,240,234,.12); --accent:#c9a227; --long:#3dba7a; }}
  body {{ margin:0; font-family:"IBM Plex Sans",sans-serif; color:var(--ink);
    background: radial-gradient(900px 500px at 10% -10%, rgba(201,162,39,.18), transparent 55%), linear-gradient(165deg,var(--bg0),var(--bg1)); min-height:100vh; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:48px 20px; }}
  h1 {{ font-family:"IBM Plex Serif",serif; font-size:clamp(2rem,4vw,2.8rem); margin:0 0 10px; }}
  .sub {{ color:var(--muted); line-height:1.55; margin-bottom:28px; }}
  .grid {{ display:grid; gap:14px; }}
  a.card {{ display:block; text-decoration:none; color:inherit; border:1px solid var(--line); padding:20px;
    background:rgba(20,32,27,.65); transition:border-color .15s, transform .15s; }}
  a.card:hover {{ border-color:rgba(201,162,39,.55); transform:translateY(-2px); }}
  a.card.recommend {{ border-color: rgba(201,162,39,.45); }}
  .name {{ font-family:"IBM Plex Serif",serif; font-size:1.35rem; margin-bottom:8px; }}
  .desc {{ color:var(--muted); font-size:.92rem; line-height:1.45; }}
  .meta {{ margin-top:12px; font-family:"JetBrains Mono",monospace; font-size:.78rem; color:var(--accent); }}
  .tag {{ display:inline-block; margin-bottom:8px; font-family:"JetBrains Mono",monospace; font-size:.68rem; color:var(--accent); border:1px solid rgba(201,162,39,.35); padding:2px 6px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>影線頸線圖表</h1>
    <p class="sub">{DAY} UTC · {DATA_NOTE or "Binance USDT-M 5m"}。原版影線頸線（Low 刺破）+ 爆量 + 拒絕上升頸線。</p>
    <div class="grid">
      <a class="card recommend" href="./balanced/index.html">
        <div class="tag">RECOMMENDED</div>
        <div class="name">原版 + 爆量（≥1.5×）</div>
        <div class="desc">破位量能 ≥ 1.5×，並拒絕上升頸線（右肩抬高）。</div>
        <div class="meta">{balanced_n} 筆訊號</div>
      </a>
      <a class="card" href="./strict/index.html">
        <div class="name">原版 + 強爆量（≥2.0×）</div>
        <div class="desc">量能 ≥ 2.0×，同樣拒絕上升頸線。</div>
        <div class="meta">{strict_n} 筆訊號</div>
      </a>
      <a class="card" href="./raw/index.html">
        <div class="name">原版訊號</div>
        <div class="desc">Low 刺破頸線 + SMA14 即報，無爆量過濾。</div>
        <div class="meta">{baseline_n} 筆訊號</div>
      </a>
    </div>
  </div>
</body>
</html>
"""


def generate_set(
    sig_csv: Path,
    out_dir: Path,
    title: str,
    subtitle: str,
    badge: str,
    index_href: str,
    extra_nav: str,
    filter_note: str = "",
):
    sig = pd.read_csv(sig_csv)
    # normalize columns for strict csv which already has pnl_* optional
    need_cols = ["symbol", "time_utc", "price", "bias", "line_val", "sma14"]
    for opt in ("dist_ma99_pct", "dist_ma200_pct", "close_break_pct", "vol_ratio"):
        if opt in sig.columns:
            need_cols.append(opt)
    missing = set(need_cols) - set(sig.columns)
    if missing:
        raise SystemExit(f"{sig_csv} missing {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for symbol, g in sig.groupby("symbol"):
        g = g.sort_values("time_utc")
        data = chart_payload(symbol, g[need_cols])
        # If csv already has pnl columns, override table values
        if "pnl_1h" in g.columns:
            by_time = {r["time_utc"]: r for _, r in g.iterrows()}
            for s in data["signals"]:
                src = by_time.get(s["time_utc"])
                if src is None:
                    continue
                for h in HORIZONS:
                    col = f"pnl_{h}"
                    if col in src and pd.notna(src[col]):
                        s["pnl"][h] = round(float(src[col]), 2)
                if "dist_ma99_pct" in src and pd.notna(src["dist_ma99_pct"]):
                    s["dist_ma99_pct"] = float(src["dist_ma99_pct"])
                if "dist_ma200_pct" in src and pd.notna(src["dist_ma200_pct"]):
                    s["dist_ma200_pct"] = float(src["dist_ma200_pct"])
                if "vol_ratio" in src and pd.notna(src["vol_ratio"]):
                    s["vol_ratio"] = float(src["vol_ratio"])
        stem = file_stem(symbol)
        href = f"{stem}.html"
        html = render_symbol_html(
            data, index_href=index_href, badge=badge, filter_note=filter_note
        )
        (out_dir / href).write_text(html, encoding="utf-8")
        pnls_1h = [s["pnl"].get("1h") for s in data["signals"] if s["pnl"].get("1h") is not None]
        avg_1h = round(sum(pnls_1h) / len(pnls_1h), 2) if pnls_1h else None
        times = " · ".join(t[11:] for t in g["time_utc"].tolist())
        cards.append({"symbol": symbol, "href": href, "n": len(data["signals"]), "avg_1h": avg_1h, "times": times})
        print(f"[{out_dir.name}] {href} signals={len(data['signals'])}")

    index = render_index(cards, title=title, subtitle=subtitle, extra_nav=extra_nav)
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    return len(sig), len(cards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "raw", "balanced", "strict"], default="all")
    parser.add_argument("--day", default=None, help="UTC day YYYY-MM-DD")
    parser.add_argument("--hist", default=None, help="UTC warmup day YYYY-MM-DD")
    parser.add_argument(
        "--cache",
        default=None,
        help="klines cache dir (default /tmp/binance_um_klines)",
    )
    parser.add_argument(
        "--data-note",
        default="",
        help="Short data-source note shown on hub",
    )
    args = parser.parse_args()

    global DAY, HIST, CACHE, DATA_NOTE
    if args.day:
        DAY = args.day
    if args.hist:
        HIST = args.hist
    elif args.day:
        HIST = (datetime.fromisoformat(args.day).date() - timedelta(days=1)).isoformat()
    if args.cache:
        CACHE = Path(args.cache)
    DATA_NOTE = args.data_note

    root = Path("/workspace/docs/charts")
    root.mkdir(parents=True, exist_ok=True)

    nav = (
        '<div class="stats" style="margin-bottom:18px">'
        '<a href="../index.html" style="color:#c9a227;text-decoration:none">← 回總覽</a> · '
        '<a href="../balanced/index.html" style="color:#8aa193;text-decoration:none">爆量1.5×</a> · '
        '<a href="../strict/index.html" style="color:#8aa193;text-decoration:none">強爆量2.0×</a> · '
        '<a href="../raw/index.html" style="color:#8aa193;text-decoration:none">原版</a>'
        "</div>"
    )

    baseline_n = balanced_n = strict_n = 0
    if args.mode in ("all", "raw"):
        baseline_n, _ = generate_set(
            Path("/workspace/output/shadow_neckline_backtest_1d.csv"),
            root / "raw",
            title=f"原版影線頸線訊號 · {DAY}",
            subtitle=f"{DAY} UTC · Low 刺破頸線 + SMA14 即報（無爆量過濾）。",
            badge="RAW",
            index_href="./index.html",
            extra_nav=nav,
        )
    if args.mode in ("all", "balanced"):
        balanced_n, _ = generate_set(
            Path("/workspace/output/shadow_neckline_balanced_1d.csv"),
            root / "balanced",
            title=f"原版 + 爆量（≥1.5×）· {DAY}",
            subtitle=f"{DAY} UTC · 原版 + 量能≥1.5× + 拒絕上升頸線。",
            badge="VOL≥1.5×",
            index_href="./index.html",
            extra_nav=nav,
            filter_note="Low 破頸線+SMA14；volume ≥ 1.5×20均量；右肩高於左肩（上升頸線）不空。",
        )
    if args.mode in ("all", "strict"):
        strict_n, _ = generate_set(
            Path("/workspace/output/shadow_neckline_strict_1d.csv"),
            root / "strict",
            title=f"原版 + 強爆量（≥2.0×）· {DAY}",
            subtitle=f"{DAY} UTC · 原版 + 量能≥2.0× + 拒絕上升頸線。",
            badge="VOL≥2.0×",
            index_href="./index.html",
            extra_nav=nav,
            filter_note="Low 破頸線+SMA14；volume ≥ 2.0×20均量；上升頸線不空。",
        )

    if args.mode == "all":
        if not baseline_n:
            baseline_n = len(pd.read_csv("/workspace/output/shadow_neckline_backtest_1d.csv"))
        if not balanced_n:
            balanced_n = len(pd.read_csv("/workspace/output/shadow_neckline_balanced_1d.csv"))
        if not strict_n:
            strict_n = len(pd.read_csv("/workspace/output/shadow_neckline_strict_1d.csv"))
        (root / "index.html").write_text(
            render_hub(baseline_n, balanced_n, strict_n), encoding="utf-8"
        )
        print("hub ->", root / "index.html")


if __name__ == "__main__":
    main()


