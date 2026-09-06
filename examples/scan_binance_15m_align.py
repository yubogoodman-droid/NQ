#!/usr/bin/env python3
"""幣安 15m：MA5 > MA20 > MA99 多頭排列且站上 MA200，成交額前 100 跳通知。

用法:
  python3 examples/scan_binance_15m_align.py --once
  python3 examples/scan_binance_15m_align.py --watch
  python3 examples/scan_binance_15m_align.py --backtest --days 5 --pages
  python3 examples/scan_binance_15m_align.py --symbol ETHUSDT --days 7
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

from nq.ma_align import (  # noqa: E402
    MIN_BARS,
    AlignSignal,
    add_indicators,
    detect_signals,
    signal_at,
    sma,
)

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_15m_align_seen.json"
CONFIG_ENV = REPO / "tg_config.env"
PAGES_HTML = REPO / "docs" / "binance" / "ma-align-15m" / "index.html"
TOP_N = 100
KLINE_LIMIT = 1000
BAR_MS = 900_000
PRE_BARS = 36
FWD_BARS = 8

MA_COLORS = {5: "#f0c14a", 20: "#64b5f6", 99: "#7e57c2", 200: "#ef5350"}
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


def universe(top_n: int = TOP_N) -> list[tuple[str, float]]:
    """成交額（24h quoteVolume）前 N 的 USDT 永續，含幣與股票。"""
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = get_json("/fapi/v1/ticker/24hr")
    trading = set()
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        trading.add(s["symbol"])
    ranked: list[tuple[float, str]] = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in trading:
            continue
        qv = float(t.get("quoteVolume") or 0)
        ranked.append((qv, sym))
    ranked.sort(reverse=True)
    return [(sym, qv) for qv, sym in ranked[:top_n]]


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


def draw_chart(sym: str, d: dict, sig: AlignSignal, path: Path) -> Path | None:
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
    a1 = min(len(d["c"]), i + FWD_BARS + 1)
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
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.6 if n == 200 else 1.1, label=f"MA{n}")
    x = i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([x], [c[x]], s=40, color="#00e676", marker="^", zorder=6)
    ax.set_title(
        f"{sym}  15m  MA5>20>99>200  站上200  離200 {sig.ext*100:+.2f}%",
        color="#e8f0ea",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc", ncol=4)
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


def format_hit(sym: str, d: dict, sig: AlignSignal, *, rank: int | None = None, qv: float | None = None) -> str:
    ts = hm(int(d["t"][sig.idx]))
    rank_s = f"成交額 #{rank}" if rank else "成交額前100"
    qv_s = f"　24h {qv/1e6:.0f}M USDT" if qv else ""
    return (
        f"<b>15m 多頭排列 · 站上 MA200</b>  {sym}\n"
        f"{ts}　現價 {sig.close:g}　離 200 {sig.ext*100:+.2f}%\n"
        f"MA5 {sig.ma5:g} &gt; MA20 {sig.ma20:g} &gt; MA99 {sig.ma99:g} &gt; MA200 {sig.ma200:g}\n"
        f"{rank_s}{qv_s}\n"
        f"<i>剛排好 5&gt;20&gt;99&gt;200，收盤站上 200。</i>"
    )


def key_of(sym: str, d: dict, sig: AlignSignal) -> str:
    return f"{sym}:{int(d['t'][sig.idx])}"


def scan_symbol(sym: str, *, last_bars: int | None = 2) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = add_indicators(raw)
    if last_bars is None:
        sigs = detect_signals(d)
    else:
        sigs = []
        for i in range(max(MIN_BARS, len(d["c"]) - last_bars), len(d["c"])):
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
    events.sort(key=lambda e: int(e["d"]["t"][e["sig"].idx]))
    return events


def backtest_symbol(sym: str, start: datetime, end: datetime) -> list[dict]:
    raw = fetch_range(sym, start - timedelta(days=3), end)
    if raw is None:
        return []
    d = add_indicators(raw)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    sigs = [s for s in detect_signals(d) if start_ms <= int(d["t"][s.idx]) < end_ms]
    return [{"symbol": sym, "d": d, "sig": s} for s in sigs]


def _b64_img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_html(rows: list[dict], path: Path, *, title: str, subtitle: str, max_cards: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_dir = path.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) > max_cards:
        rows = sorted(rows, key=lambda r: -r["sig"].ext)[:max_cards]
        rows = sorted(rows, key=lambda r: int(r["d"]["t"][r["sig"].idx]))
    cards = []
    for n, row in enumerate(rows, 1):
        sig, d, sym = row["sig"], row["d"], row["symbol"]
        img_name = f"{n:03d}_{sym}_{hm(int(d['t'][sig.idx])).replace(' ', '_').replace(':', '')}.png"
        png = draw_chart(sym, d, sig, img_dir / img_name)
        src = _b64_img(png) if png and png.exists() else ""
        detail = (
            f"現價 {sig.close:g}　離 200 {sig.ext*100:+.2f}%　張開 {sig.stack_pct*100:.2f}%\n"
            f"MA5 {sig.ma5:g} > MA20 {sig.ma20:g} > MA99 {sig.ma99:g} > MA200 {sig.ma200:g}"
        )
        img_html = f"<img src='{src}' alt='{escape(sym)}' style='width:100%;display:block;border-radius:10px'/>" if src else ""
        cards.append(
            "<article class='trade-card'>"
            f"<header class='card-header'><div class='card-title'><span class='trade-no'>#{n} · {escape(sym)}</span>"
            f"<span class='trade-time'>{hm(int(d['t'][sig.idx]))}</span></div>"
            f"<div class='card-pnl pnl-win'>{sig.ext*100:+.2f}%</div></header>"
            f"<div class='tags'><span class='tag tag-info'>剛排好</span>"
            f"<span class='tag tag-info'>15m</span><span class='tag tag-info'>站上200</span></div>"
            f"<pre class='trade-detail'>{escape(detail)}</pre>"
            f"<div class='mini-chart'>{img_html}</div></article>"
        )
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
.pnl-win{{color:#00c805}}
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
<div class="card">訊號<b>{len(rows)}</b></div>
<div class="card">週期<b>15m</b></div>
<div class="card">標的<b>成交額前{TOP_N}</b></div>
</div>
<p class="muted">MA5 &gt; MA20 &gt; MA99 &gt; MA200，收盤站上 200。只在剛排好的第一根通知。</p>
</section>
{''.join(cards) if cards else "<div class='empty'>這段期間沒有剛排好的多頭排列。</div>"}
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")
    (path.parent / "view.html").write_text(html, encoding="utf-8")
    return path


def print_row(sym: str, d: dict, sig: AlignSignal, *, rank: int | None = None) -> None:
    r = f"  #{rank}" if rank else ""
    print(
        f"{sym:12s}{r:5s} {hm(int(d['t'][sig.idx]))}  "
        f"px={sig.close:g}  5={sig.ma5:g} > 20={sig.ma20:g} > 99={sig.ma99:g} > 200={sig.ma200:g}  "
        f"ext={sig.ext*100:+.2f}%"
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    days = int(args.days)
    end = datetime.now(TZ)
    start = end - timedelta(days=days)
    ranks: dict[str, int] = {}
    qvs: dict[str, float] = {}
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    else:
        print("載入成交額前 100…", flush=True)
        uni = universe(int(args.top))
        symbols = [s for s, _ in uni]
        ranks = {s: i for i, (s, _) in enumerate(uni, 1)}
        qvs = {s: q for s, q in uni}
        print(f"掃描 {len(symbols)} 檔，{days} 日", flush=True)
        print(f"  #1 {uni[0][0]}  {uni[0][1]/1e6:.0f}M  ·  #{len(uni)} {uni[-1][0]}  {uni[-1][1]/1e6:.0f}M", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(backtest_symbol, s, start, end): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                rows.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
            if done % 20 == 0:
                print(f"  {done}/{len(symbols)}  已找到 {len(rows)}", flush=True)
    rows.sort(key=lambda r: int(r["d"]["t"][r["sig"].idx]))
    print(f"\n=== {days} 日 5>20>99>200 ===")
    print(f"剛排好 {len(rows)} 筆")
    for r in rows:
        print_row(r["symbol"], r["d"], r["sig"], rank=ranks.get(r["symbol"]))
    if args.pages or args.html:
        out = Path(args.html) if args.html else PAGES_HTML
        write_html(
            rows,
            out,
            title="15m 多頭排列 · 站上 MA200",
            subtitle=(
                f"{days}d · {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} · "
                f"成交額前 {len(symbols)} · 剛排好才記"
            ),
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
            print("沒有剛排好的多頭排列")
            continue
        for r in rows:
            print_row(sym, r["d"], r["sig"])
        if args.html or args.pages:
            out = Path(args.html) if args.html else PAGES_HTML
            write_html(rows, out, title=f"{sym} 15m 多頭排列", subtitle=f"{days}d")
            print(f"HTML {out}")
    return 0


def notify(ev: dict, *, rank: int | None = None, qv: float | None = None) -> None:
    text = format_hit(ev["symbol"], ev["d"], ev["sig"], rank=rank, qv=qv)
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("&gt;", ">"))
    tmp = Path("/tmp") / f"align_{ev['symbol']}_{ev['sig'].idx}.png"
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
        ok = telegram_send("15m 多頭排列測試\n如果你看到這則，Telegram 已通。")
        print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
        return 0 if ok else 1
    seen = load_seen()
    print("載入成交額前 100…", flush=True)
    uni = universe(int(args.top))
    symbols = [s for s, _ in uni]
    ranks = {s: i for i, (s, _) in enumerate(uni, 1)}
    qvs = {s: q for s, q in uni}
    print(f"監看 {len(symbols)} 檔。5>20>99>200 剛排好且站上 200 會推。", flush=True)
    print(f"  #1 {uni[0][0]}  {uni[0][1]/1e6:.0f}M  ·  #{len(uni)} {uni[-1][0]}  {uni[-1][1]/1e6:.0f}M", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal uni, symbols, ranks, qvs, uni_ts
        if time.time() - uni_ts > 1800:
            uni = universe(int(args.top))
            symbols = [s for s, _ in uni]
            ranks = {s: i for i, (s, _) in enumerate(uni, 1)}
            qvs = {s: q for s, q in uni}
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}  #1 {uni[0][0]}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, last_bars=2)
        new = [e for e in events if key_of(e["symbol"], e["d"], e["sig"]) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] 掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev["symbol"], ev["d"], ev["sig"]))
            notify(ev, rank=ranks.get(ev["symbol"]), qv=qvs.get(ev["symbol"]))
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
    p = argparse.ArgumentParser(description="幣安 15m MA5/20/99 多頭排列 · 站上 MA200 · 成交額前100")
    p.add_argument("--symbol", help="只掃這些合約，逗號分隔，例如 ETHUSDT")
    p.add_argument("--days", type=int, default=5, help="回看天數")
    p.add_argument("--top", type=int, default=TOP_N, help="成交額前 N")
    p.add_argument("--backtest", action="store_true", help="前 N 名回測剛排好的時點")
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance/ma-align-15m/")
    p.add_argument("--html", help="HTML 輸出路徑")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的 1～2 根並推通知")
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
