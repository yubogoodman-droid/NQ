#!/usr/bin/env python3
"""幣安 15m 黏帶擠壓：均線纏在 200MA 附近，放量打出箱頂且收盤仍靠近 200 才進。

用法:
  python3 examples/scan_binance_15m_ma200.py --symbol ETHUSDT --days 14
  python3 examples/scan_binance_15m_ma200.py --backtest --days 7 --pages
  python3 examples/scan_binance_15m_ma200.py --once
  python3 examples/scan_binance_15m_ma200.py --test
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma200_squeeze import (  # noqa: E402
    HOLD_BARS,
    MA_PERIODS,
    add_indicators,
    detect_signals,
    simulate_trades,
    sma,
    summarize_trades,
)

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_15m_ma200_seen.json"
CONFIG_ENV = REPO / "tg_config.env"
PAGES_HTML = REPO / "docs" / "binance" / "ma200-squeeze-15m" / "index.html"
KEEP = {"ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"}
MIN_BARS = 220
KLINE_LIMIT = 1000
BAR_MS = 900_000
ALERT_BUCKET_MS = 8 * 3600 * 1000
PRE_BARS = 28
FWD_BARS = 8

MA_COLORS = {7: "#f0c14a", 14: "#64b5f6", 25: "#d28cff", 99: "#7e57c2", 120: "#26a69a", 200: "#ef5350"}
CJK_FONTS = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def apply_keys() -> None:
    if CONFIG_ENV.exists():
        for line in CONFIG_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def get_json(path: str, params=None, retries: int = 6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.5 * (i + 1))
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


def bars_from_raw(raw: list) -> dict:
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def fetch_klines(
    sym: str,
    *,
    limit: int = KLINE_LIMIT,
    start_ms: int | None = None,
    end_ms: int | None = None,
    drop_unclosed: bool = True,
    min_bars: int = MIN_BARS,
) -> dict | None:
    params: dict = {"symbol": sym, "interval": "15m", "limit": min(limit, 1500)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    raw = get_json("/fapi/v1/klines", params=params)
    if not raw or len(raw) < min_bars:
        return None
    now_ms = int(time.time() * 1000)
    if drop_unclosed and int(raw[-1][0]) + BAR_MS > now_ms:
        raw = raw[:-1]
    if len(raw) < min_bars:
        return None
    return bars_from_raw(raw)


def fetch_range(sym: str, start: datetime, end: datetime) -> dict | None:
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=10), end)
        part = fetch_klines(
            sym,
            limit=1500,
            start_ms=int(cur.timestamp() * 1000),
            end_ms=int(nxt.timestamp() * 1000),
            drop_unclosed=False,
            min_bars=2,
        )
        if part is not None:
            chunks.append(part)
        time.sleep(0.12)
        cur = nxt
    if not chunks:
        return None
    t = np.concatenate([p["t"] for p in chunks])
    _, idx = np.unique(t, return_index=True)
    idx.sort()
    out = {k: np.concatenate([p[k] for p in chunks])[idx] for k in ("t", "o", "h", "l", "c", "v")}
    return out if len(out["c"]) >= MIN_BARS else None


def use_cjk_font(plt) -> None:
    import matplotlib.font_manager as fm

    for path in CJK_FONTS:
        if not Path(path).exists():
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return


def draw_chart(sym: str, d: dict, sig, path: Path, *, trade=None) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    use_cjk_font(plt)
    i = sig.idx
    a0 = max(0, i - PRE_BARS)
    a1 = min(len(d["c"]), (trade.exit_idx if trade else i) + FWD_BARS + 1)
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
    for n, col in MA_COLORS.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.35 if n == 200 else 1.05, label=f"MA{n}")
    ax.axhline(sig.box_high, color="#c9a227", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(sig.box_low, color="#8aa193", ls=":", lw=0.8, alpha=0.55)
    ax.axhline(sig.stop, color="#e35d5d", ls=":", lw=0.9, alpha=0.75)
    ax.axhline(sig.target, color="#3dba7a", ls=":", lw=0.9, alpha=0.7)
    x = i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([x], [c[x]], s=36, color="#00e676", marker="^", zorder=6)
    if trade is not None:
        xe = trade.exit_idx - a0
        if 0 <= xe < len(c):
            ax.scatter([xe], [trade.exit_price], s=36, color="#00c805" if trade.pnl_pct > 0 else "#ff5252", marker="x", zorder=6)
    q = sig.quality
    extra = ""
    if trade is not None:
        sign = "+" if trade.pnl_pct >= 0 else ""
        extra = f"  {trade.exit_reason}  {sign}{trade.pnl_pct:.2f}%"
    ax.set_title(f"{sym}  15m  Q{q}  黏帶擠壓 · 200MA 附近{extra}", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)))


def telegram_send(text: str, photo: str | None = None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                    files={"photo": f},
                    timeout=25,
                )
            if r.ok:
                return True
        r = SESSION.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


def format_hit(sym: str, d: dict, sig) -> str:
    ts = hm(int(d["t"][sig.idx]))
    return (
        f"<b>黏帶擠壓 · 200MA 附近</b>  {sym}  15m  Q{sig.quality}\n"
        f"{ts}　現價 {sig.close:g}　離 200 {sig.ext*100:+.2f}%\n"
        f"MA200 {sig.ma200:g}　黏度 {sig.ribbon*100:.2f}%　箱體 {sig.box_pct*100:.2f}%\n"
        f"量比 {sig.vol_ratio:.1f}×　擴張 {sig.expand:.1f}×　靠近 200 的 K {sig.near_frac*100:.0f}%\n"
        f"進 {sig.entry:g}　停 {sig.stop:g}　目標 {sig.target:g}（{2:.0f}R）\n"
        f"<i>均線黏在 200 附近橫盤，放量打出箱頂且收盤仍靠近 200 才進，不追直豎。</i>"
    )


def key_of(sym: str, d: dict, sig) -> str:
    return f"{sym}:{int(d['t'][sig.idx]) // ALERT_BUCKET_MS}"


def scan_symbol(sym: str, *, last_bars: int | None = 2) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = add_indicators(raw)
    if last_bars is None:
        sigs = detect_signals(d)
    else:
        sigs = []
        for i in range(max(220, len(d["c"]) - last_bars), len(d["c"])):
            from nq.ma200_squeeze import signal_at

            s = signal_at(d, i)
            if s:
                sigs.append(s)
    return [{"symbol": sym, "d": d, "sig": s} for s in sigs]


def scan_all(symbols: list[str], *, last_bars: int | None = 2) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, last_bars=last_bars): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (e["sig"].quality, -e["sig"].vol_ratio, e["symbol"]))
    return events


def backtest_symbol(sym: str, start: datetime, end: datetime) -> list[dict]:
    raw = fetch_range(sym, start - timedelta(days=3), end)
    if raw is None:
        return []
    d = add_indicators(raw)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    sigs = [s for s in detect_signals(d) if start_ms <= int(d["t"][s.idx]) < end_ms]
    trades = simulate_trades(d, sigs)
    return [{"symbol": sym, "d": d, "sig": t.signal, "trade": t} for t in trades]


def _b64_img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _equity_svg(pnls: list[float], width: int = 720, height: int = 160) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys.min()), float(ys.max())
    pad = max(0.2, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys[i]):.1f}" for i in range(len(ys)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def write_html(rows: list[dict], path: Path, *, title: str, subtitle: str, max_cards: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_dir = path.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) > max_cards:
        def _rank(r):
            s = r["sig"]
            q = {"A": 0, "B": 1, "C": 2}.get(s.quality, 3)
            eth = 0 if r["symbol"] == "ETHUSDT" else 1
            return (eth, q, -s.vol_ratio, s.ribbon)

        rows = sorted(rows, key=_rank)[:max_cards]
        rows = sorted(rows, key=lambda r: int(r["d"]["t"][r["sig"].idx]))
    trades = [r["trade"] for r in rows if r.get("trade")]
    stats = summarize_trades(trades) if trades else {"count": len(rows), "wins": 0, "losses": 0, "win_rate": 0.0, "pnl_pct": 0.0, "avg_pct": 0.0, "by_quality": {}}
    cards = []
    for n, row in enumerate(rows, 1):
        sig, d, sym = row["sig"], row["d"], row["symbol"]
        trade = row.get("trade")
        img_name = f"{n:03d}_{sym}_{hm(int(d['t'][sig.idx])).replace(' ', '_').replace(':', '')}.png"
        png = draw_chart(sym, d, sig, img_dir / img_name, trade=trade)
        src = _b64_img(png) if png and png.exists() else ""
        if trade:
            cls = "pnl-win" if trade.pnl_pct > 0 else ("pnl-flat" if trade.pnl_pct == 0 else "pnl-loss")
            sign = "+" if trade.pnl_pct >= 0 else ""
            pnl = f"{sign}{trade.pnl_pct:.2f}%"
            tag = trade.exit_reason
            t2 = hm(int(d["t"][trade.exit_idx]))
        else:
            cls, pnl, tag, t2 = "pnl-flat", "—", "signal", "—"
        detail = (
            f"進場 {sig.entry:g}　停損 {sig.stop:g}　目標 {sig.target:g}\n"
            f"離 200 {sig.ext*100:+.2f}%　黏度 {sig.ribbon*100:.2f}%　箱體 {sig.box_pct*100:.2f}%\n"
            f"量比 {sig.vol_ratio:.1f}×　擴張 {sig.expand:.1f}×　靠近比例 {sig.near_frac*100:.0f}%\n"
            f"MA7 {sig.ma7:g} / MA25 {sig.ma25:g} / MA200 {sig.ma200:g}"
        )
        img_html = f"<img src='{src}' alt='{escape(sym)}' style='width:100%;display:block;border-radius:10px'/>" if src else ""
        cards.append(
            "<article class='trade-card'>"
            f"<header class='card-header'><div class='card-title'><span class='trade-no'>#{n} · {escape(sym)} · Q{sig.quality}</span>"
            f"<span class='trade-time'>{hm(int(d['t'][sig.idx]))} → {t2}</span></div>"
            f"<div class='card-pnl {cls}'>{pnl}</div></header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(tag)}</span>"
            f"<span class='tag tag-info'>15m</span><span class='tag tag-info'>Q{sig.quality}</span></div>"
            f"<pre class='trade-detail'>{escape(detail)}</pre>"
            f"<div class='mini-chart'>{img_html}</div></article>"
        )
    qline = " · ".join(
        f"Q{q} {v['n']}筆 {v['pnl']:+.2f}%"
        for q, v in (stats.get("by_quality") or {}).items()
        if v.get("n")
    )
    pnl_cls = "pnl-win" if stats.get("pnl_pct", 0) >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid rgba(88,166,255,0.28);background:rgba(88,166,255,0.12);color:#79c0ff}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(title)}</h1>
<p class="muted">{escape(subtitle)}</p>
<div class="cards">
<div class="card">筆數<b>{stats.get('count', 0)}</b></div>
<div class="card">勝率<b>{stats.get('win_rate', 0):.1f}%</b></div>
<div class="card">總報酬<b class="{pnl_cls}">{stats.get('pnl_pct', 0):+.2f}%</b></div>
<div class="card">均筆<b>{stats.get('avg_pct', 0):+.2f}%</b></div>
</div>
<p class="muted">{escape(qline) if qline else "均線黏在 200 附近、放量打出箱頂、收盤離 200 ≤ 1.5% 才進。"}</p>
<div class="equity">{_equity_svg([t.pnl_pct for t in trades])}</div>
</section>
{''.join(cards) if cards else "<div class='empty'>這段期間沒抓到黏帶擠壓。</div>"}
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")
    view = path.parent / "view.html"
    view.write_text(html, encoding="utf-8")
    return path


def print_row(sym: str, d: dict, sig, trade=None) -> None:
    extra = ""
    if trade:
        sign = "+" if trade.pnl_pct >= 0 else ""
        extra = f"  {trade.exit_reason} {sign}{trade.pnl_pct:.2f}%"
    print(
        f"{sym:12s} {hm(int(d['t'][sig.idx]))}  Q{sig.quality}  "
        f"px={sig.close:g}  200={sig.ma200:g}  ext={sig.ext*100:+.2f}%  "
        f"ribbon={sig.ribbon*100:.2f}%  vol={sig.vol_ratio:.1f}x{extra}"
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    days = int(args.days)
    end = datetime.now(TZ)
    start = end - timedelta(days=days)
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"掃描 {len(symbols)} 個流動永續，{days} 日", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(backtest_symbol, s, start, end): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                part = fut.result()
                rows.extend(part)
            except Exception as e:
                print("err", futs[fut], e, flush=True)
            if done % 40 == 0:
                print(f"  {done}/{len(symbols)}  已找到 {len(rows)}", flush=True)
    rows.sort(key=lambda r: int(r["d"]["t"][r["sig"].idx]))
    trades = [r["trade"] for r in rows if r.get("trade")]
    stats = summarize_trades(trades)
    print(f"\n=== {days} 日黏帶擠壓 ===")
    print(f"筆數 {stats['count']}  勝率 {stats['win_rate']:.1f}%  總 {stats['pnl_pct']:+.2f}%  均 {stats['avg_pct']:+.2f}%")
    for r in rows:
        print_row(r["symbol"], r["d"], r["sig"], r.get("trade"))
    if args.pages or args.html:
        out = Path(args.html) if args.html else PAGES_HTML
        write_html(
            rows,
            out,
            title="15m 黏帶擠壓 · 200MA 附近進場",
            subtitle=f"{days}d · {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} · {len(symbols)} 檔 · 抱最多 {HOLD_BARS} 根",
        )
        print(f"HTML {out}")
    return 0


def cmd_symbol(args: argparse.Namespace) -> int:
    days = int(args.days)
    end = datetime.now(TZ)
    start = end - timedelta(days=days)
    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        print(f"\n=== {sym} {days}d ===")
        rows = backtest_symbol(sym, start, end)
        if not rows:
            # 沒成交也列出訊號
            raw = fetch_range(sym, start - timedelta(days=3), end)
            if raw is None:
                print("沒有 K 線")
                continue
            d = add_indicators(raw)
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            sigs = [s for s in detect_signals(d) if start_ms <= int(d["t"][s.idx]) < end_ms]
            if not sigs:
                print("沒抓到黏帶擠壓")
                continue
            for s in sigs:
                print_row(sym, d, s)
            continue
        for r in rows:
            print_row(sym, r["d"], r["sig"], r["trade"])
        stats = summarize_trades([r["trade"] for r in rows])
        print(f"筆數 {stats['count']}  勝率 {stats['win_rate']:.1f}%  總 {stats['pnl_pct']:+.2f}%")
        if args.html or args.pages:
            out = Path(args.html) if args.html else PAGES_HTML
            write_html(rows, out, title=f"{sym} 15m 黏帶擠壓", subtitle=f"{days}d")
            print(f"HTML {out}")
    return 0


def notify(ev: dict) -> None:
    text = format_hit(ev["symbol"], ev["d"], ev["sig"])
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("&gt;", ">"))
    tmp = Path("/tmp") / f"ma200_{ev['symbol']}_{ev['sig'].idx}.png"
    photo = draw_chart(ev["symbol"], ev["d"], ev["sig"], tmp)
    ok = telegram_send(text, photo=str(photo) if photo else None)
    if ok:
        print("  → Telegram 已送")
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        print("  → 還沒填 Telegram" if not token else "  → Telegram 送出失敗")


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 900 + 1) * 900 + 3
    time.sleep(max(1, nxt - now))


def cmd_watch(args: argparse.Namespace) -> int:
    apply_keys()
    if args.test:
        ok = telegram_send("15m 黏帶擠壓測試\n如果你看到這則，Telegram 已通。")
        print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
        return 0 if ok else 1
    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe()
    print(f"監看 {len(symbols)} 個流動永續。黏帶打出箱頂且靠近 200 會推。", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, last_bars=2)
        new = [e for e in events if key_of(e["symbol"], e["d"], e["sig"]) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] 掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev["symbol"], ev["d"], ev["sig"]))
            notify(ev)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中，每根 15m 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="幣安 15m 黏帶擠壓 · 200MA 附近進場")
    p.add_argument("--symbol", help="只掃這些合約，逗號分隔，例如 ETHUSDT")
    p.add_argument("--days", type=int, default=7, help="回看天數")
    p.add_argument("--backtest", action="store_true", help="全市場回測")
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance/ma200-squeeze-15m/")
    p.add_argument("--html", help="HTML 輸出路徑")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的 1～2 根")
    p.add_argument("--test", action="store_true", help="測 Telegram")
    p.add_argument("--watch", action="store_true", help="持續監看")
    args = p.parse_args()
    apply_keys()
    if args.test or args.once or args.watch:
        return cmd_watch(args)
    if args.backtest:
        return cmd_backtest(args)
    if args.symbol:
        return cmd_symbol(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
