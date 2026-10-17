#!/usr/bin/env python3
"""幣安 15 分 K：爆量，並且同一根一次突破 7/14/25/99/120/200。

條件就這兩句：

  • 一次突破：開盤在 MA7、MA14、MA25、MA99、MA120、MA200 每一條之下，收盤在每一條之上。
  • 爆量：這根成交量至少是前一根的 10 倍。
  • 收盤在小時 K 的 MA200 之上。用的是這根 15 分 K 收盤時已經走完的那根小時 K。

    python3 examples/binance_ma_burst.py --symbol AINUSDT
    python3 examples/binance_ma_burst.py --days 2
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

MA_PERIODS = (7, 14, 25, 99, 120, 200)
VOL_MULT = 10.0  # 至少是前一根的 10 倍
H1_MA = 200
INTERVAL_MS = 15 * 60_000
HOUR_MS = 60 * 60_000

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def burst_at(
    o: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    ma: dict[int, np.ndarray],
    i: int,
) -> dict | None:
    """同一根 15 分 K：一次穿六條均線，而且量至少是前一根的 10 倍。"""
    need = max(MA_PERIODS) - 1
    if i < need or i >= len(c):
        return None
    vals = np.array([ma[n][i] for n in MA_PERIODS], dtype=float)
    if np.isnan(vals).any():
        return None
    if not (c[i] > o[i] and np.all(o[i] <= vals) and np.all(c[i] > vals)):
        return None
    prior = float(v[i - 1])
    if prior <= 0:
        return None
    ratio = float(v[i] / prior)
    if ratio < VOL_MULT:
        return None
    return {
        "i": i,
        "open": float(o[i]),
        "close": float(c[i]),
        "body": float(c[i] / o[i] - 1.0),
        "volume": float(v[i]),
        "prior_volume": prior,
        "vol_ratio": ratio,
        "mas": {n: float(ma[n][i]) for n in MA_PERIODS},
    }


def find_bursts(o: np.ndarray, c: np.ndarray, v: np.ndarray) -> list[dict]:
    ma = {n: sma(c, n) for n in MA_PERIODS}
    return [hit for i in range(len(c)) if (hit := burst_at(o, c, v, ma, i))]


def hour_ma200_for_bars(bar_open_ms: np.ndarray, hour_open_ms: np.ndarray, hour_close: np.ndarray) -> np.ndarray:
    """每根 15 分 K 收盤時，已經走完的那根小時 K 的 MA200。"""
    ma = sma(np.asarray(hour_close, dtype=float), H1_MA)
    close_ms = np.asarray(bar_open_ms, dtype=np.int64) + INTERVAL_MS
    last_open = ((close_ms - HOUR_MS) // HOUR_MS) * HOUR_MS
    hour_open_ms = np.asarray(hour_open_ms, dtype=np.int64)
    idx = np.searchsorted(hour_open_ms, last_open, side="left")
    out = np.full(len(bar_open_ms), np.nan)
    valid = idx < len(hour_open_ms)
    if not valid.any():
        return out
    idx_safe = np.where(valid, idx, 0)
    match = valid & (hour_open_ms[idx_safe] == last_open)
    out[match] = ma[idx_safe[match]]
    return out


def above_hour_ma200(close: float, h1_ma: float) -> bool:
    return bool(np.isfinite(h1_ma) and close > h1_ma)


def signal_hour_index(signal_open_ms: int, hour_open_ms: np.ndarray) -> int:
    """小時圖上要標的那一根：15 分訊號落在哪一根小時 K。還沒收到就退回已收盤的那根。"""
    hour_open_ms = np.asarray(hour_open_ms, dtype=np.int64)
    contain = (int(signal_open_ms) // HOUR_MS) * HOUR_MS
    idx = int(np.searchsorted(hour_open_ms, contain, side="left"))
    if idx < len(hour_open_ms) and int(hour_open_ms[idx]) == contain:
        return idx
    close_ms = int(signal_open_ms) + INTERVAL_MS
    last_open = ((close_ms - HOUR_MS) // HOUR_MS) * HOUR_MS
    idx = int(np.searchsorted(hour_open_ms, last_open, side="left"))
    if idx < len(hour_open_ms) and int(hour_open_ms[idx]) == last_open:
        return idx
    return -1


def hour_view(hour_open_ms: np.ndarray, signal_open_ms: int, before: int = 48, after: int = 12) -> tuple[int, int, int] | None:
    """小時圖視窗：訊號那根前面 before 根、後面 after 根。回傳 [start, end) 與標記位置。"""
    mark = signal_hour_index(signal_open_ms, hour_open_ms)
    if mark < 0:
        return None
    n = len(hour_open_ms)
    return max(0, mark - before), min(n, mark + after + 1), mark


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))
    raise last


def fetch_klines(sym: str, limit: int = 1500, interval: str = "15m", bar_ms: int = INTERVAL_MS, min_bars: int | None = None) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
    need = max(MA_PERIODS) + 1 if min_bars is None else min_bars
    if not raw or len(raw) < need:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    if len(raw) < need:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def path_after(d: dict, i: int, bars: int) -> dict | None:
    """訊號收盤之後 bars 根：報酬、最高、最低。資料不夠就收到最後一根。"""
    c, h, l = d["c"], d["h"], d["l"]
    if i + 1 >= len(c):
        return None
    j = min(len(c) - 1, i + bars)
    entry = float(c[i])
    if entry <= 0:
        return None
    hh = float(h[i + 1 : j + 1].max())
    ll = float(l[i + 1 : j + 1].min())
    return {
        "bars": j - i,
        "ret": float(c[j] / entry - 1.0),
        "mfe": float(hh / entry - 1.0),
        "mae": float(ll / entry - 1.0),
    }


def universe(min_quote_volume: float = 5_000_000) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_quote_volume:
            continue
        out.append(sym)
    return out


def scan_since(symbols: list[str], since_ms: int, limit: int = 500) -> list[dict]:
    rows = []

    def one(sym: str) -> list[dict]:
        d = fetch_klines(sym, limit=limit)
        h1_limit = min(1500, H1_MA + limit // 4 + 3)
        h1 = fetch_klines(sym, limit=h1_limit, interval="1h", bar_ms=HOUR_MS, min_bars=H1_MA)
        if d is None or h1 is None:
            return []
        d["h1_ma200"] = hour_ma200_for_bars(d["t"], h1["t"], h1["c"])
        found = []
        for hit in find_bursts(d["o"], d["c"], d["v"]):
            ts = int(d["t"][hit["i"]])
            if ts < since_ms:
                continue
            h1_ma = float(d["h1_ma200"][hit["i"]])
            if not above_hour_ma200(hit["close"], h1_ma):
                continue
            hit = dict(hit)
            hit["h1_ma200"] = h1_ma
            hit["symbol"] = sym
            hit["time"] = ts
            hit["after_1h"] = path_after(d, hit["i"], 4)
            hit["after_4h"] = path_after(d, hit["i"], 16)
            hit["d"] = d
            hit["h1"] = h1
            found.append(hit)
        return found

    with ThreadPoolExecutor(12) as ex:
        futs = [ex.submit(one, sym) for sym in symbols]
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print("err", e, flush=True)
    rows.sort(key=lambda r: (r["time"], r["symbol"]))
    return rows


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.1f}%"


def fmt_vol(x: float) -> str:
    if x >= 1_000_000:
        return f"{x / 1e6:.2f}M"
    if x >= 1_000:
        return f"{x / 1e3:.1f}K"
    return f"{x:.0f}"


def fmt_span(after: dict | None, bars: int) -> str:
    if after is None:
        return "—"
    if after["bars"] >= bars:
        return fmt_pct(after["ret"])
    hours = after["bars"] * 15 / 60
    return f"{fmt_pct(after['ret'])}（{hours:.1f}h）"


def fmt_row(row: dict) -> str:
    ts = datetime.fromtimestamp(row["time"] / 1000, TZ).strftime("%m-%d %H:%M")
    a1, a4 = row["after_1h"], row["after_4h"]
    return (
        f"{row['symbol']:<16} {ts}  實體{row['body'] * 100:+6.1f}%  "
        f"量 {fmt_vol(row['volume']):>8} / 前一根 {fmt_vol(row['prior_volume']):>8} ={row['vol_ratio']:5.1f}x  "
        f"1h {fmt_span(a1, 4):<14}  "
        f"4h {fmt_span(a4, 16):<16}  "
        f"之後高 {fmt_pct(None if a4 is None else a4['mfe'])}  "
        f"之後低 {fmt_pct(None if a4 is None else a4['mae'])}  "
        f"時MA200 {row.get('h1_ma200', float('nan')):.6g}"
    )


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, Rectangle


def _style_axes(axes) -> None:
    for a in axes:
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")


def _draw_candles(ax, axv, o, h, l, c, v, rectangle) -> np.ndarray:
    xs = np.arange(len(c))
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], c[k] * 1e-6) * 0.02
        ax.add_patch(rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        axv.bar(xs[k], v[k], width=0.72, color="#3dba7a99" if up else "#e35d5d99", linewidth=0)
    return xs


def _time_ticks(ax, opens_ms: np.ndarray, n_labels: int = 6) -> None:
    if len(opens_ms) == 0:
        return
    idx = np.unique(np.linspace(0, len(opens_ms) - 1, n_labels, dtype=int))
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [datetime.fromtimestamp(int(opens_ms[i]) / 1000, TZ).strftime("%m-%d %H:%M") for i in idx],
        fontsize=7,
    )


def draw_burst(row: dict, path: str) -> str:
    plt, rectangle = _mpl()
    d = row["d"]
    i = row["i"]
    a0 = max(0, i - 48)
    a1 = min(len(d["c"]), i + 17)
    sl = slice(a0, a1)
    h1 = row.get("h1")
    view = None if h1 is None else hour_view(h1["t"], int(row["time"]))
    if view is None:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(10.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}, facecolor="#0c1210"
        )
        panels = (ax, axv)
    else:
        fig, (ax, axv, axh, axhv) = plt.subplots(
            4,
            1,
            figsize=(10.4, 10.8),
            gridspec_kw={"height_ratios": [3.15, 0.85, 3.15, 0.85]},
            facecolor="#0c1210",
        )
        panels = (ax, axv, axh, axhv)
    _style_axes(panels)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    xs = _draw_candles(ax, axv, o, h, l, c, v, rectangle)
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}
    full = {n: sma(d["c"], n) for n in MA_PERIODS}
    for n, col in pal.items():
        ax.plot(xs, full[n][sl], color=col, lw=1.05, label=f"MA{n}")
    h1_line = d.get("h1_ma200")
    if h1_line is not None:
        ax.plot(xs, h1_line[sl], color="#ff6b9a", lw=1.15, ls="--", label="1h MA200")
    x = i - a0
    ax.axvline(x, color="#f0c14a", ls="--", lw=0.9)
    ax.scatter([x], [c[x]], s=28, color="#f0c14a", zorder=5)
    ts = datetime.fromtimestamp(row["time"] / 1000, TZ).strftime("%m-%d %H:%M")
    a1r, a4 = row.get("after_1h"), row.get("after_4h")
    ax.set_title(
        f"{row['symbol']}  15分  {ts}    實體 {row['body'] * 100:+.1f}%    量 {row['vol_ratio']:.1f}×前一根\n"
        f"收 {row['close']:.6g} > 時MA200 {row.get('h1_ma200', float('nan')):.6g}    "
        f"之後 1h {fmt_span(a1r, 4)}    4h {fmt_span(a4, 16)}",
        color="#e8f0ea",
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=7)
    ax.tick_params(axis="x", labelbottom=False)
    _time_ticks(axv, d["t"][sl])
    if view is not None:
        h0, h1_end, mark = view
        hsl = slice(h0, h1_end)
        ho, hh, hl, hc, hv = h1["o"][hsl], h1["h"][hsl], h1["l"][hsl], h1["c"][hsl], h1["v"][hsl]
        hxs = _draw_candles(axh, axhv, ho, hh, hl, hc, hv, rectangle)
        ma200 = sma(h1["c"], H1_MA)
        axh.plot(hxs, ma200[hsl], color="#ff6b9a", lw=1.35, label="MA200")
        hx = mark - h0
        axh.axvline(hx, color="#f0c14a", ls="--", lw=0.9)
        axh.scatter([hx], [row["close"]], s=28, color="#f0c14a", zorder=5)
        hour_ts = datetime.fromtimestamp(int(h1["t"][mark]) / 1000, TZ).strftime("%m-%d %H:%M")
        axh.set_title(
            f"小時 K    黃虛線 {hour_ts}    MA200 {row.get('h1_ma200', float('nan')):.6g}",
            color="#e8f0ea",
            fontsize=11,
            loc="left",
        )
        axh.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc")
        axh.tick_params(axis="x", labelbottom=False)
        _time_ticks(axhv, h1["t"][hsl])
    fig.tight_layout(pad=0.45)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_charts(rows: list[dict], directory: str) -> list[str]:
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, row in enumerate(rows, start=1):
        ts = datetime.fromtimestamp(row["time"] / 1000, TZ).strftime("%m%d_%H%M")
        path = out / f"{n:02d}_{row['symbol']}_{ts}.png"
        paths.append(draw_burst(row, str(path)))
    return paths


def write_gallery_html(rows: list[dict], paths: list[str], dest: str, *, since_label: str, scanned: int) -> str:
    """單檔 HTML，圖內嵌，手機開 htmlpreview 不會漏圖。每張上面是 15 分、下面是小時。"""
    import base64
    from pathlib import Path

    def cls(x: float | None) -> str:
        if x is None:
            return ""
        return "up" if x >= 0 else "dn"

    cards = []
    for row, path in zip(rows, paths):
        ts = datetime.fromtimestamp(row["time"] / 1000, TZ).strftime("%m-%d %H:%M")
        a1, a4 = row.get("after_1h"), row.get("after_4h")
        raw = Path(path).read_bytes()
        b64 = base64.b64encode(raw).decode()
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    '  <div class="head">',
                    "    <div>",
                    f'      <div class="sym">{row["symbol"]}</div>',
                    f'      <div class="when">{ts} 台北</div>',
                    "    </div>",
                    '    <div class="nums">',
                    f'      <span>實體 <b class="{cls(row["body"])}">{row["body"] * 100:+.1f}%</b></span>',
                    f'      <span>量 <b>{row["vol_ratio"]:.1f}×</b></span>',
                    f'      <span>1h <b class="{cls(None if a1 is None else a1["ret"])}">{fmt_span(a1, 4)}</b></span>',
                    f'      <span>4h <b class="{cls(None if a4 is None else a4["ret"])}">{fmt_span(a4, 16)}</b></span>',
                    "    </div>",
                    "  </div>",
                    f'  <img alt="{row["symbol"]} {ts} 15分與小時" src="data:image/png;base64,{b64}" />',
                    "</article>",
                ]
            )
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>15 分爆量穿六均線 · 近三天</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #0b0e11; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
  }}
  .page {{ max-width: 920px; margin: 0 auto; padding: 16px 12px 40px; }}
  h1 {{ margin: 0 0 8px; font-size: 22px; }}
  .sub {{ margin: 0 0 16px; color: #8b949e; font-size: 14px; line-height: 1.55; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 12px; margin-bottom: 14px; }}
  .head {{ display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
  .sym {{ font-size: 18px; font-weight: 700; }}
  .when {{ color: #8b949e; font-size: 13px; margin-top: 2px; }}
  .nums {{ display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 13px; color: #8b949e; align-items: center; }}
  .nums b {{ font-weight: 650; }}
  .up {{ color: #3dba7a; }}
  .dn {{ color: #e35d5d; }}
  img {{ width: 100%; height: auto; border-radius: 8px; display: block; background: #0c1210; }}
</style>
</head>
<body>
<div class="page">
  <h1>15 分爆量穿六均線 · 近三天</h1>
  <p class="sub">{since_label} 起，24 小時成交額 500 萬 USDT 以上的永續 {scanned} 檔，{len(rows)} 根。同一根 15 分 K：開盤在 MA7、14、25、99、120、200 每一條下面，收盤在每一條上面，量至少是前一根的 10 倍，而且收盤在小時 K 的 MA200 之上。每張圖上面是 15 分，下面是小時 K。粉紅是小時 MA200，黃虛線是訊號。1h、4h 用訊號收盤當進場。</p>
  {"".join(cards)}
</div>
</body>
</html>
"""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)


def fmt_hit(sym: str, d: dict, hit: dict) -> str:
    ts = datetime.fromtimestamp(int(d["t"][hit["i"]]) / 1000, TZ).strftime("%Y-%m-%d %H:%M")
    mas = "  ".join(f"MA{n} {hit['mas'][n]:.6g}" for n in MA_PERIODS)
    return (
        f"{sym}  15m  {ts}\n"
        f"開 {hit['open']:.6g} → 收 {hit['close']:.6g}  ({hit['body'] * 100:+.2f}%)\n"
        f"量 {fmt_vol(hit['volume'])}，前一根 {fmt_vol(hit['prior_volume'])} 的 {hit['vol_ratio']:.1f} 倍\n"
        f"收 {hit['close']:.6g} > 小時MA200 {hit.get('h1_ma200', float('nan')):.6g}\n"
        f"{mas}"
    )


def check_symbol(sym: str, limit: int = 1500) -> list[str]:
    d = fetch_klines(sym, limit=limit)
    h1_limit = min(1500, H1_MA + limit // 4 + 3)
    h1 = fetch_klines(sym, limit=h1_limit, interval="1h", bar_ms=HOUR_MS, min_bars=H1_MA) if d is not None else None
    if d is None or h1 is None:
        return []
    d["h1_ma200"] = hour_ma200_for_bars(d["t"], h1["t"], h1["c"])
    lines = []
    for hit in find_bursts(d["o"], d["c"], d["v"]):
        h1_ma = float(d["h1_ma200"][hit["i"]])
        if not above_hour_ma200(hit["close"], h1_ma):
            continue
        hit = dict(hit)
        hit["h1_ma200"] = h1_ma
        lines.append(fmt_hit(sym, d, hit))
    return lines


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="15 分 K 爆量，一根穿 7/14/25/99/120/200")
    p.add_argument("--symbol", default="AINUSDT", help="永續代號，例如 AINUSDT")
    p.add_argument("--limit", type=int, default=1500, help="單幣往回看幾根 15 分 K，最多 1500")
    p.add_argument("--days", type=float, default=0, help="掃成交額夠的永續，只留最近幾天的訊號")
    p.add_argument("--charts", default="", help="把訊號圖存到這個資料夾")
    p.add_argument("--page", default="", help="把 15 分圖和下面的小時圖寫成一頁 HTML")
    args = p.parse_args()
    if args.days > 0:
        since = datetime.now(TZ) - timedelta(days=args.days)
        since_ms = int(since.timestamp() * 1000)
        print(f"載入標的… 自 {since.strftime('%m-%d %H:%M')} 起", flush=True)
        symbols = universe()
        limit = min(1500, int(args.days * 24 * 60 / 15) + max(MA_PERIODS) + 1)
        t0 = time.time()
        rows = scan_since(symbols, since_ms, limit=limit)
        print(f"掃 {len(symbols)} 檔，{time.time() - t0:.0f}s，{len(rows)} 根\n")
        if not rows:
            print("這段沒有。")
            return 0
        for row in rows:
            print(fmt_row(row))
        paths: list[str] = []
        if args.charts or args.page:
            chart_dir = args.charts or "output/ma_burst_charts"
            paths = write_charts(rows, chart_dir)
            print(f"\n圖 {len(paths)} 張 → {chart_dir}")
        if args.page:
            page = write_gallery_html(
                rows,
                paths,
                args.page,
                since_label=since.strftime("%Y-%m-%d %H:%M"),
                scanned=len(symbols),
            )
            print(f"頁面 → {page}")
        return 0
    sym = args.symbol.upper()
    lines = check_symbol(sym, limit=args.limit)
    if not lines:
        print(f"{sym} 這段 15 分 K 沒有「爆量一根穿六條均線」")
        return 0
    print(f"{sym} 抓到 {len(lines)} 根\n")
    print("\n\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
