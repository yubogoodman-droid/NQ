#!/usr/bin/env python3
"""掃描幣安 USDT 永續 5 分 K，找近似 UAI 的 W 底，輸出手機版 HTML 圖。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TPE = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-w-bottom" / "index.html"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
FAPI = "https://www.binance.com/fapi/v1"
MA_PERIODS = (7, 14, 25, 99, 120, 200)
MA_COLORS = {
    7: "#f0c14b",
    14: "#79c0ff",
    25: "#f778ba",
    99: "#b392f0",
    120: "#3ddc97",
    200: "#ff7b72",
}

# 對齊 UAI 5m：兩底約 0.4885 / 0.4963，間隔 13 根，頸線深度約 8.7%
MIN_SEP = 8
MAX_SEP = 48
MAX_SYM_PCT = 0.04
MIN_DEPTH = 0.07
MAX_DEPTH = 0.18
PRIOR_DROP = 0.07
BREAKOUT_PAD = 0.003
BREAKOUT_WINDOW = 48
MIN_QUOTE_VOL = 1_000_000.0


@dataclass
class WHit:
    symbol: str
    status: str
    score: float
    bottom1: float
    bottom2: float
    neck: float
    depth_pct: float
    sym_pct: float
    sep_min: int
    target: float
    current: float
    vsneck_pct: float
    vs_target_pct: float
    volume24: float
    t1: int
    t2: int
    t_neck: int
    t_break: int | None
    i: int
    j: int
    neck_idx: int
    b: int | None
    volx: float | None
    reference: bool = False


def get_json(path: str, params: dict[str, Any] | None = None, retries: int = 4) -> Any:
    url = FAPI + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last: Exception | None = None
    for n in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                body = resp.read()
                if body:
                    return json.loads(body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (2**n))
    raise RuntimeError(f"GET failed {url}: {last}")


def swing_lows(lows: list[float], lookback: int = 3) -> list[int]:
    n = len(lows)
    out: list[int] = []
    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] != min(window):
            continue
        left = min(lows[i - lookback : i])
        right = min(lows[i + 1 : i + lookback + 1])
        if lows[i] < left * 0.998 and lows[i] < right * 0.998:
            out.append(i)
    return out


def rolling_mean(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    total = 0.0
    for i, val in enumerate(values):
        total += val
        if i >= period:
            total -= values[i - period]
        if i + 1 >= period:
            out[i] = total / period
    return out


def detect_w_bottoms(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    quote_vol: list[float],
    times: list[int],
    *,
    symbol: str = "",
    volume24: float = 0.0,
) -> list[WHit]:
    n = len(closes)
    if n < 60:
        return []
    mins = swing_lows(lows)
    hits: list[WHit] = []
    last_t = times[-1]
    for ai, i in enumerate(mins):
        for j in mins[ai + 1 :]:
            sep = j - i
            if sep < MIN_SEP:
                continue
            if sep > MAX_SEP:
                break
            l1, l2 = lows[i], lows[j]
            avg = (l1 + l2) / 2
            if avg <= 0:
                continue
            sym = abs(l2 / l1 - 1)
            if sym > MAX_SYM_PCT:
                continue
            if min(lows[i : j + 1]) < min(l1, l2) * 0.985:
                continue
            if j - i < 6:
                continue
            neck_idx = max(range(i + 3, j - 2), key=lambda x: highs[x])
            neck = highs[neck_idx]
            depth = neck / avg - 1
            if not (MIN_DEPTH <= depth <= MAX_DEPTH):
                continue
            prior = max(closes[max(0, i - 24) : i])
            if prior / l1 - 1 < PRIOR_DROP:
                continue
            end = min(n, j + BREAKOUT_WINDOW + 1)
            b = next((x for x in range(j + 2, end) if closes[x] > neck * (1 + BREAKOUT_PAD)), None)
            current = closes[-1]
            target = neck + (neck - avg)
            age2_h = (last_t - times[j]) / 3_600_000
            volx = None
            t_break = None
            if b is None:
                if age2_h > 12 or min(lows[j:]) < l2 * 0.98:
                    continue
                if not (avg * 1.015 < current < neck * 1.01):
                    continue
                status = "待突破"
                age_h = age2_h
            else:
                if min(lows[j : b + 1]) < l2 * 0.98:
                    continue
                age_h = (last_t - times[b]) / 3_600_000
                if age_h > 24:
                    continue
                base = quote_vol[max(0, b - 20) : b] or [1.0]
                volx = quote_vol[b] / (statistics.median(base) or 1.0)
                t_break = times[b]
                if current >= neck:
                    status = "已延伸" if current >= target * 1.02 else "突破仍有效"
                elif current >= avg:
                    status = "跌回頸線下"
                else:
                    status = "形態失敗"
            score = (
                100
                - 600 * sym
                - 220 * abs(depth - 0.087)
                - min(age_h, 12)
                + (min(volx, 4) * 2 if volx else 0)
            )
            hits.append(
                WHit(
                    symbol=symbol,
                    status=status,
                    score=score,
                    bottom1=l1,
                    bottom2=l2,
                    neck=neck,
                    depth_pct=depth * 100,
                    sym_pct=sym * 100,
                    sep_min=sep * 5,
                    target=target,
                    current=current,
                    vsneck_pct=(current / neck - 1) * 100,
                    vs_target_pct=(current / target - 1) * 100,
                    volume24=volume24,
                    t1=times[i],
                    t2=times[j],
                    t_neck=times[neck_idx],
                    t_break=t_break,
                    i=i,
                    j=j,
                    neck_idx=neck_idx,
                    b=b,
                    volx=volx,
                )
            )
    return hits


def best_hit(hits: list[WHit]) -> WHit | None:
    if not hits:
        return None
    rank = {"待突破": 0, "突破仍有效": 1, "已延伸": 2, "跌回頸線下": 3, "形態失敗": 4}
    return max(hits, key=lambda h: (-rank.get(h.status, 9), h.score))


def parse_klines(raw: list[list[Any]]) -> tuple[list[float], ...]:
    opens = [float(x[1]) for x in raw]
    highs = [float(x[2]) for x in raw]
    lows = [float(x[3]) for x in raw]
    closes = [float(x[4]) for x in raw]
    quote = [float(x[7]) for x in raw]
    times = [int(x[0]) for x in raw]
    return opens, highs, lows, closes, quote, times


def fmt_price(value: float) -> str:
    if value >= 100:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.01:
        return f"{value:.5f}"
    return f"{value:.6f}"


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TPE).strftime("%m-%d %H:%M")


def fmt_vol(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value / 1e3:.0f}K"


def chart_payload(hit: WHit, ohlcv: tuple[list[float], ...]) -> dict[str, Any]:
    opens, highs, lows, closes, quote, times = ohlcv
    n = len(closes)
    left = max(0, hit.i - 24)
    right = min(n, (hit.b or hit.j) + 56)
    window_o = opens[left:right]
    window_h = highs[left:right]
    window_l = lows[left:right]
    window_c = closes[left:right]
    window_v = quote[left:right]
    window_t = [fmt_time(t) for t in times[left:right]]
    mas = {f"ma{p}": rolling_mean(closes, p)[left:right] for p in MA_PERIODS}
    return {
        **asdict(hit),
        "times": window_t,
        "open": window_o,
        "high": window_h,
        "low": window_l,
        "close": window_c,
        "volume": window_v,
        "t1_label": fmt_time(hit.t1),
        "t2_label": fmt_time(hit.t2),
        "t_neck_label": fmt_time(hit.t_neck),
        "t_break_label": fmt_time(hit.t_break) if hit.t_break else None,
        "i_rel": hit.i - left,
        "j_rel": hit.j - left,
        "neck_rel": hit.neck_idx - left,
        "b_rel": (hit.b - left) if hit.b is not None else None,
        **mas,
    }


STATUS_CLASS = {
    "待突破": "tag-wait",
    "突破仍有效": "tag-ok",
    "已延伸": "tag-hot",
    "跌回頸線下": "tag-warn",
    "形態失敗": "tag-sl",
    "基準": "tag-info",
}


def build_html(payload: dict[str, Any]) -> str:
    hits = payload["hits"]
    cards = []
    for idx, hit in enumerate(hits, 1):
        status = "基準" if hit.get("reference") else hit["status"]
        tag = STATUS_CLASS.get(status, "tag-info")
        vs = hit["vsneck_pct"]
        vs_cls = "pnl-win" if vs >= 0 else "pnl-loss"
        detail = (
            f"L1 {fmt_price(hit['bottom1'])} @ {hit['t1_label']}\n"
            f"L2 {fmt_price(hit['bottom2'])} @ {hit['t2_label']}\n"
            f"頸線 {fmt_price(hit['neck'])} @ {hit['t_neck_label']}\n"
            f"量度目標 {fmt_price(hit['target'])}\n"
            f"兩底價差 {hit['sym_pct']:.2f}% · 間隔 {hit['sep_min']} 分 · 深度 {hit['depth_pct']:.1f}%\n"
            f"現價 {fmt_price(hit['current'])}  距頸線 {vs:+.2f}%"
        )
        if hit.get("t_break_label"):
            extra = f"突破 {hit['t_break_label']}"
            if hit.get("volx"):
                extra += f" · 量能 {hit['volx']:.1f}x"
            detail += f"\n{extra}"
        cards.append(
            f"""
    <article class="trade-card" data-status="{escape(hit['status'])}">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{idx} · {escape(hit['symbol'])}</span>
          <span class="trade-time">L1 {escape(hit['t1_label'])} → L2 {escape(hit['t2_label'])} 台北</span>
        </div>
        <div class="card-pnl {vs_cls}">{vs:+.1f}%</div>
      </header>
      <div class="tags">
        <span class="tag {tag}">{escape(status)}</span>
        <span class="tag tag-info">5m</span>
        <span class="tag tag-info">24h {escape(fmt_vol(hit['volume24']))}</span>
      </div>
      <pre class="trade-detail">{escape(detail)}</pre>
      <div class="mini-chart" id="chart-{idx}"></div>
    </article>"""
        )
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>幣安 5m W底 · 近兩天</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #0b0e11;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
    }}
    .page {{ max-width: 720px; margin: 0 auto; padding: 12px 12px 32px; }}
    .summary {{
      background: #161b22; border: 1px solid #30363d; border-radius: 14px;
      padding: 14px 16px; margin-bottom: 14px;
    }}
    h1 {{ font-size: 18px; margin: 0 0 6px; }}
    .muted {{ color: #8b949e; font-size: 13px; line-height: 1.55; margin: 0; }}
    .cards {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 0; }}
    .stat {{ background: #0d1117; padding: 10px 12px; border-radius: 10px; min-width: 92px; border: 1px solid #21262d; }}
    .stat b {{ display: block; font-size: 20px; margin-top: 4px; }}
    .trade-card {{
      background: #161b22; border: 1px solid #30363d; border-radius: 14px;
      padding: 14px 14px 8px; margin-bottom: 14px;
    }}
    .card-header {{ display: flex; justify-content: space-between; gap: 10px; }}
    .trade-no {{ font-weight: 700; }}
    .trade-time {{ font-size: 12px; color: #8b949e; }}
    .card-pnl {{ font-weight: 700; white-space: nowrap; }}
    .pnl-win {{ color: #00c805; }} .pnl-loss {{ color: #ff5252; }}
    .tags {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
    .tag {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 999px; border: 1px solid #30363d; }}
    .tag-info {{ background: rgba(88,166,255,0.12); color: #79c0ff; border-color: rgba(88,166,255,0.28); }}
    .tag-ok {{ background: rgba(0,200,5,0.15); color: #3ddc68; border-color: rgba(0,200,5,0.35); }}
    .tag-wait {{ background: rgba(255,193,7,0.12); color: #f0c14b; border-color: rgba(255,193,7,0.3); }}
    .tag-hot {{ background: rgba(255,122,69,0.14); color: #ff9b6a; border-color: rgba(255,122,69,0.35); }}
    .tag-warn {{ background: rgba(240,193,75,0.12); color: #f0c14b; }}
    .tag-sl {{ background: rgba(255,82,82,0.15); color: #ff7b72; border-color: rgba(255,82,82,0.35); }}
    .trade-detail {{
      margin: 0 0 8px; padding: 10px 12px; background: #0d1117; border-radius: 10px;
      border: 1px solid #21262d; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px; line-height: 1.55; color: #c9d1d9; white-space: pre-wrap;
    }}
    .mini-chart {{ height: 420px; margin: 0 -6px -4px; }}
    .note {{ color: #8b949e; font-size: 12px; margin-top: 8px; line-height: 1.5; }}
  </style>
</head>
<body>
<div class="page">
  <section class="summary">
    <h1>幣安 USDT 永續 · 5m W底 · 近兩天</h1>
    <p class="muted">對齊 UAI：兩底價差 ≤4%、間隔 40–240 分、頸線深度 7–18%，先有一腳下跌再做出雙底。時間為台北。資料截至 {escape(payload['asof'])}。僅供型態對照，不是進出場建議。</p>
    <div class="cards">
      <div class="stat">掃描<b>{payload['universe']}</b></div>
      <div class="stat">命中<b>{payload['matched']}</b></div>
      <div class="stat">待突破<b>{payload['pending']}</b></div>
      <div class="stat">仍有效<b>{payload['valid']}</b></div>
    </div>
    <p class="note">黃虛線頸線、綠虛線量度目標。UAI 是基準圖；MAGMA / BULLA 較接近尚未確認突破。</p>
  </section>
  {"".join(cards)}
</div>
<script id="payload" type="application/json">{data_json}</script>
<script>
const MA_COLORS = {json.dumps(MA_COLORS)};
const payload = JSON.parse(document.getElementById("payload").textContent);
function draw(hit, idx) {{
  const traces = [{{
    type: "candlestick",
    x: hit.times, open: hit.open, high: hit.high, low: hit.low, close: hit.close,
    name: hit.symbol,
    increasing: {{line: {{color: "#26a69a"}}}},
    decreasing: {{line: {{color: "#ef5350"}}}},
    xaxis: "x", yaxis: "y"
  }}];
  for (const p of [7,14,25,99,120,200]) {{
    traces.push({{
      type: "scatter", mode: "lines", x: hit.times, y: hit["ma"+p],
      name: "MA"+p, line: {{color: MA_COLORS[p], width: p<=25 ? 1.4 : 1.05}},
      hoverinfo: "skip", xaxis: "x", yaxis: "y"
    }});
  }}
  traces.push({{
    type: "scatter", mode: "markers+text",
    x: [hit.times[hit.i_rel], hit.times[hit.j_rel]],
    y: [hit.bottom1, hit.bottom2],
    text: ["L1","L2"], textposition: "bottom center",
    marker: {{size: 10, color: ["#79c0ff","#f778ba"], line: {{color: "#fff", width: 1}}}},
    name: "W底", xaxis: "x", yaxis: "y"
  }});
  traces.push({{
    type: "scatter", mode: "lines",
    x: [hit.times[hit.neck_rel], hit.times[hit.b_rel != null ? hit.b_rel : hit.times.length-1]],
    y: [hit.neck, hit.neck],
    line: {{color: "#f0c14b", width: 1.4, dash: "dash"}},
    name: "頸線", hoverinfo: "skip", xaxis: "x", yaxis: "y"
  }});
  if (hit.b_rel != null) {{
    traces.push({{
      type: "scatter", mode: "markers+text",
      x: [hit.times[hit.b_rel]], y: [hit.close[hit.b_rel]],
      text: ["突破"], textposition: "top center",
      marker: {{symbol: "triangle-up", size: 13, color: "#00e676"}},
      name: "突破", xaxis: "x", yaxis: "y"
    }});
  }}
  const volColor = hit.close.map((c,i) => c >= hit.open[i] ? "rgba(38,166,154,0.45)" : "rgba(239,83,80,0.45)");
  traces.push({{
    type: "bar", x: hit.times, y: hit.volume, marker: {{color: volColor}},
    name: "Volume", showlegend: false, xaxis: "x2", yaxis: "y2"
  }});
  Plotly.newPlot("chart-"+idx, traces, {{
    template: "plotly_dark",
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#0d1117",
    margin: {{l: 48, r: 12, t: 8, b: 36}},
    height: 420,
    showlegend: false,
    hovermode: "x unified",
    xaxis: {{matches: "x2", showticklabels: false, rangeslider: {{visible: false}}, gridcolor: "rgba(255,255,255,0.06)"}},
    yaxis: {{domain: [0.28, 1], gridcolor: "rgba(255,255,255,0.06)"}},
    xaxis2: {{anchor: "y2", gridcolor: "rgba(255,255,255,0.06)"}},
    yaxis2: {{domain: [0, 0.22], gridcolor: "rgba(255,255,255,0.06)"}},
    shapes: [
      {{type:"line", xref:"x domain", x0:0, x1:1, y0:hit.neck, y1:hit.neck, line:{{color:"#f0c14b", width:1, dash:"dot"}}, opacity:0.55}},
      {{type:"line", xref:"x domain", x0:0, x1:1, y0:hit.target, y1:hit.target, line:{{color:"#3ddc68", width:1, dash:"dot"}}, opacity:0.45}}
    ]
  }}, {{responsive: true, displayModeBar: false}});
}}
payload.hits.forEach((hit, i) => draw(hit, i+1));
</script>
</body>
</html>
"""


def universe_symbols(min_quote_vol: float = MIN_QUOTE_VOL) -> tuple[list[str], dict[str, float]]:
    info = get_json("/exchangeInfo")
    ticks = get_json("/ticker/24hr")
    tv = {x["symbol"]: float(x["quoteVolume"]) for x in ticks}
    symbols = [
        s["symbol"]
        for s in info["symbols"]
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and tv.get(s["symbol"], 0) >= min_quote_vol
    ]
    return symbols, tv


def fetch_klines(symbol: str) -> tuple[str, list[list[Any]] | None]:
    try:
        return symbol, get_json("/klines", {"symbol": symbol, "interval": "5m", "limit": 576})
    except Exception:  # noqa: BLE001
        return symbol, None


def status_rank(hit: WHit) -> tuple[int, int, float]:
    order = {
        "待突破": 0,
        "突破仍有效": 1,
        "已延伸": 2,
        "跌回頸線下": 3,
        "形態失敗": 4,
    }
    return (0 if hit.reference else 1, order.get(hit.status, 9), -hit.score)


def scan(min_quote_vol: float = MIN_QUOTE_VOL) -> dict[str, Any]:
    symbols, tv = universe_symbols(min_quote_vol)
    with ThreadPoolExecutor(max_workers=16) as pool:
        raw = dict(pool.map(fetch_klines, symbols))
    hits: list[WHit] = []
    series: dict[str, tuple[list[float], ...]] = {}
    for symbol, klines in raw.items():
        if not klines or len(klines) < 120:
            continue
        closed = klines[:-1]
        ohlcv = parse_klines(closed)
        series[symbol] = ohlcv
        found = detect_w_bottoms(*ohlcv, symbol=symbol, volume24=tv.get(symbol, 0.0))
        best = best_hit(found)
        if best:
            if symbol == "UAIUSDT":
                best.reference = True
            hits.append(best)
    if "UAIUSDT" in series and not any(h.symbol == "UAIUSDT" for h in hits):
        found = detect_w_bottoms(*series["UAIUSDT"], symbol="UAIUSDT", volume24=tv.get("UAIUSDT", 0.0))
        if found:
            best = max(found, key=lambda h: h.score)
            best.reference = True
            hits.append(best)
    hits.sort(key=status_rank)
    asof = datetime.now(TPE).strftime("%Y-%m-%d %H:%M")
    if series:
        last_ms = max(v[5][-1] for v in series.values() if v[5])
        asof = datetime.fromtimestamp(last_ms / 1000, TPE).strftime("%Y-%m-%d %H:%M")
    payload_hits = [chart_payload(h, series[h.symbol]) for h in hits if h.symbol in series]
    return {
        "asof": asof,
        "universe": len(symbols),
        "matched": len(hits),
        "pending": sum(1 for h in hits if h.status == "待突破"),
        "valid": sum(1 for h in hits if h.status in {"突破仍有效", "已延伸"}),
        "hits": payload_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="掃描幣安 5m W底並輸出 HTML")
    parser.add_argument("--output", "-o", default=str(PAGES))
    parser.add_argument("--min-quote-vol", type=float, default=MIN_QUOTE_VOL)
    args = parser.parse_args()
    payload = scan(args.min_quote_vol)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload), encoding="utf-8")
    json_path = out.with_name("hits.json")
    slim = {**payload, "hits": [{k: v for k, v in h.items() if k not in {"open", "high", "low", "close", "volume", "times"} and not str(k).startswith("ma")} for h in payload["hits"]]}
    json_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已產生: {out}")
    print(f"掃描 {payload['universe']} · 命中 {payload['matched']} · 截至 {payload['asof']}")
    for hit in payload["hits"]:
        print(f"  {hit['symbol']:16} {hit['status']:8} 頸線{fmt_price(hit['neck'])} 現價{fmt_price(hit['current'])} {hit['vsneck_pct']:+.1f}%")


if __name__ == "__main__":
    main()
