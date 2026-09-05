#!/usr/bin/env python3
"""幣安 5m 多頭排列 Telegram 監看。

五分 K：MA7 > MA14 > MA25，且前一根收在 MA200 下、這一根收盤才站上。
同時小時 K 收盤要在 MA99 與 MA200 之上。
同一根不重發。

    python3 examples/watch_binance_5m_align.py --test
    python3 examples/watch_binance_5m_align.py --once --dry-run
    python3 examples/watch_binance_5m_align.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import requests

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})

ROOT = Path(__file__).resolve().parents[1]
SEEN_PATH = ROOT / "output" / "binance_5m_align_seen.json"
CONFIG_ENV = ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"

KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}
MS_5M = 5 * 60_000
MS_1H = 60 * 60_000
HORIZONS = (("15m", 3), ("30m", 6), ("60m", 12), ("120m", 24))
PAGES = ROOT / "docs" / "binance-5m-align" / "index.html"


def load_dotenv(path: Path = CONFIG_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def apply_keys() -> None:
    load_dotenv()
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def parse_klines(raw: list) -> dict:
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def drop_unclosed(raw: list, interval_ms: int, now_ms: int | None = None) -> list:
    if not raw:
        return raw
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if int(raw[-1][0]) + interval_ms > now_ms:
        return raw[:-1]
    return raw


def add_mas(d: dict, periods: tuple[int, ...] = (7, 14, 25, 99, 200)) -> dict:
    out = dict(d)
    for n in periods:
        out[f"m{n}"] = sma(out["c"], n)
    return out


def short_align_ok(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    vals = [d["m7"][i], d["m14"][i], d["m25"][i]]
    if np.isnan(vals).any():
        return False
    return vals[0] > vals[1] > vals[2]


def five_align_ok(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    m200 = d["m200"][i]
    if np.isnan(m200):
        return False
    return short_align_ok(d, i) and d["c"][i] > m200


def reclaim_ma200(d: dict, i: int) -> bool:
    """前一根收在 MA200 下，這一根收盤站上。"""
    if i < 1 or i >= len(d["c"]):
        return False
    prev, now = d["m200"][i - 1], d["m200"][i]
    if np.isnan([prev, now]).any():
        return False
    return d["c"][i - 1] < prev and d["c"][i] > now


def hour_above_ok(h: dict, i: int) -> bool:
    if i < 0 or i >= len(h["c"]):
        return False
    m99, m200 = h["m99"][i], h["m200"][i]
    if np.isnan([m99, m200]).any():
        return False
    return h["c"][i] > m99 and h["c"][i] > m200


def hour_index_at(h: dict, t_ms: int) -> int:
    """已開出、且開盤時間 ≤ t_ms 的最後一根小時 K。"""
    if len(h["t"]) == 0:
        return -1
    return int(np.searchsorted(h["t"], t_ms, side="right") - 1)


def hour_mas_at(h: dict, t_ms: int, px: float) -> tuple[float, float] | None:
    """用當下價格當形成中小時 K 的收盤，算 MA99/200，不看這根之後的收盤。"""
    hi = hour_index_at(h, t_ms)
    if hi < 199:
        return None
    m99 = float(np.mean(np.append(h["c"][hi - 98 : hi], px)))
    m200 = float(np.mean(np.append(h["c"][hi - 199 : hi], px)))
    return m99, m200


def hour_above_at(h: dict, t_ms: int, px: float) -> bool:
    mas = hour_mas_at(h, t_ms, px)
    return bool(mas and px > mas[0] and px > mas[1])


def detect_new_align(d5: dict, i: int, h1: dict, hi: int | None = None) -> dict | None:
    """5m 7>14>25，且這一根才從 MA200 下收盤站上；小時也在 99/200 上。"""
    if not five_align_ok(d5, i) or not reclaim_ma200(d5, i):
        return None
    px = float(d5["c"][i])
    t = int(d5["t"][i])
    if hi is None:
        mas = hour_mas_at(h1, t, px)
        if mas is None or not (px > mas[0] and px > mas[1]):
            return None
        h_close, h_m99, h_m200 = px, mas[0], mas[1]
        hi_used = hour_index_at(h1, t)
    else:
        if not hour_above_ok(h1, hi):
            return None
        h_close = float(h1["c"][hi])
        h_m99 = float(h1["m99"][hi])
        h_m200 = float(h1["m200"][hi])
        hi_used = hi
    return {
        "i": i,
        "hi": hi_used,
        "close": px,
        "m7": float(d5["m7"][i]),
        "m14": float(d5["m14"][i]),
        "m25": float(d5["m25"][i]),
        "m200": float(d5["m200"][i]),
        "h_close": h_close,
        "h_m99": h_m99,
        "h_m200": h_m200,
        "t": t,
    }


def collect_signals(d5: dict, h1: dict, start_ms: int, end_ms: int) -> list[dict]:
    out = []
    for i in range(len(d5["c"])):
        t = int(d5["t"][i])
        if t < start_ms or t > end_ms:
            continue
        sig = detect_new_align(d5, i, h1)
        if sig:
            out.append(sig)
    return out


def forward_pct(d5: dict, i: int, bars: int) -> float | None:
    j = i + bars
    if j >= len(d5["c"]) or d5["c"][i] == 0:
        return None
    return (float(d5["c"][j]) / float(d5["c"][i]) - 1) * 100


def attach_forwards(d5: dict, sig: dict) -> dict:
    row = dict(sig)
    for name, bars in HORIZONS:
        row[name] = forward_pct(d5, sig["i"], bars)
    return row


def summarize_hits(hits: list[dict]) -> dict:
    stats: dict = {
        "count": len(hits),
        "symbols": len({h["symbol"] for h in hits}),
        "five_only": sum(int(h.get("five_only", 0)) for h in hits),
    }
    for name, _bars in HORIZONS:
        vals = [float(h[name]) for h in hits if h.get(name) is not None]
        n = len(vals)
        wins = sum(1 for v in vals if v > 0)
        stats[name] = {
            "n": n,
            "wins": wins,
            "win_rate": (100.0 * wins / n) if n else 0.0,
            "avg": float(np.mean(vals)) if vals else 0.0,
            "med": float(np.median(vals)) if vals else 0.0,
            "sum": float(np.sum(vals)) if vals else 0.0,
        }
    return stats


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


def fetch_klines(sym: str, interval: str, limit: int, interval_ms: int, *, keep_forming: bool = False) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
    if not raw:
        return None
    if not keep_forming:
        raw = drop_unclosed(raw, interval_ms)
    if len(raw) < 210:
        return None
    return parse_klines(raw)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


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
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


def draw_chart(sym: str, d: dict, i: int, path: str, *, ahead: int = 4) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print("chart skip", sym, e, flush=True)
        return None
    a0 = max(0, i - 80)
    a1 = min(len(d["c"]), i + max(4, ahead))
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
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}
    for n, col in pal.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.1, label=f"MA{n}")
    x = i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([x], [c[x]], s=36, color="#3dba7a", zorder=5)
    ax.set_title(f"{sym}  5m align", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def scan_symbol(sym: str) -> list[dict]:
    raw5 = fetch_klines(sym, "5m", 260, MS_5M, keep_forming=False)
    raw1 = fetch_klines(sym, "1h", 260, MS_1H, keep_forming=True)
    if raw5 is None or raw1 is None:
        return []
    d5 = add_mas(raw5, (7, 14, 25, 200))
    h1 = add_mas(raw1, (99, 200))
    last5 = len(d5["c"]) - 1
    events = []
    for closed in (last5, last5 - 1):
        sig = detect_new_align(d5, closed, h1)
        if not sig:
            continue
        events.append({"symbol": sym, "sig": sig, "d5": d5, "h1": h1})
    return events


def format_alert(ev: dict) -> str:
    sig, sym = ev["sig"], ev["symbol"]
    ext = (sig["close"] / sig["m200"] - 1) * 100
    return (
        f"📈 <b>5m 多頭排列</b>  {sym}\n"
        f"時間 {hm(sig['t'])}\n"
        f"收盤 {sig['close']:g}\n"
        f"5m MA7 {sig['m7']:g} &gt; MA14 {sig['m14']:g} &gt; MA25 {sig['m25']:g}\n"
        f"前一根在 MA200 下，這根站上 {sig['m200']:g}（{ext:+.2f}%）\n"
        f"1h 收盤 {sig['h_close']:g} &gt; MA99 {sig['h_m99']:g} / MA200 {sig['h_m200']:g}"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{ev['sig']['t']}"


def scan_all(symbols: list[str]) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: e["symbol"])
    return events


def notify(ev: dict, *, dry_run: bool = False) -> None:
    text = format_alert(ev)
    plain = text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">")
    print("\n" + plain, flush=True)
    if dry_run:
        print("  → dry-run，不送 Telegram", flush=True)
        return
    tmp = Path("/tmp") / f"align5m_{ev['symbol']}_{ev['sig']['t']}.png"
    photo = draw_chart(ev["symbol"], ev["d5"], ev["sig"]["i"], str(tmp))
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送", flush=True)
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗，檢查 token 與 chat id", flush=True)


def scan_history_symbol(sym: str, start_ms: int, end_ms: int) -> tuple[list[dict], dict]:
    meta = {"symbol": sym, "five_new": 0, "hits": 0, "error": ""}
    raw5 = fetch_klines(sym, "5m", 1500, MS_5M, keep_forming=False)
    raw1 = fetch_klines(sym, "1h", 500, MS_1H, keep_forming=True)
    if raw5 is None or raw1 is None:
        meta["error"] = "too_few_bars"
        return [], meta
    d5 = add_mas(raw5, (7, 14, 25, 200))
    h1 = add_mas(raw1, (99, 200))
    hits = []
    for i in range(len(d5["c"])):
        t = int(d5["t"][i])
        if t < start_ms or t > end_ms:
            continue
        if five_align_ok(d5, i) and reclaim_ma200(d5, i):
            meta["five_new"] += 1
            sig = detect_new_align(d5, i, h1)
            if sig:
                row = attach_forwards(d5, sig)
                row["symbol"] = sym
                row["d5"] = d5
                hits.append(row)
    meta["hits"] = len(hits)
    return hits, meta


def backtest_all(symbols: list[str], start_ms: int, end_ms: int) -> tuple[list[dict], dict]:
    hits: list[dict] = []
    funnel = {"symbols": len(symbols), "ok": 0, "five_new": 0, "hits": 0, "errors": 0}
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_history_symbol, s, start_ms, end_ms): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                rows, meta = fut.result()
            except Exception as e:
                funnel["errors"] += 1
                print("err", futs[fut], e, flush=True)
                continue
            funnel["ok"] += 1
            funnel["five_new"] += meta["five_new"]
            funnel["hits"] += meta["hits"]
            hits.extend(rows)
            if done % 40 == 0 or done == len(symbols):
                print(f"  回測進度 {done}/{len(symbols)}　已中 {funnel['hits']}", flush=True)
    hits.sort(key=lambda h: (h["t"], h["symbol"]))
    return hits, funnel


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip() or "main"
    except Exception:
        return "main"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def _pnl_cls(v: float | None) -> str:
    if v is None or v == 0:
        return "pnl-flat"
    return "pnl-win" if v > 0 else "pnl-loss"


def _equity_svg(pcts: list[float], width: int = 720, height: int = 160) -> str:
    if not pcts:
        return ""
    eq = np.cumsum(pcts)
    lo, hi = float(min(0.0, eq.min())), float(max(0.0, eq.max()))
    span = hi - lo or 1.0
    pad = 8
    ys = [pad + (1 - (v - lo) / span) * (height - 2 * pad) for v in eq]
    xs = [i * width / max(1, len(eq) - 1) for i in range(len(eq))]
    zero_y = pad + (1 - (0 - lo) / span) * (height - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    color = "#16a34a" if eq[-1] >= 0 else "#e35d5d"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0d1117;border-radius:8px">'
        f'<line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def pick_chart_hits(hits: list[dict], limit: int) -> list[dict]:
    scored = [h for h in hits if h.get("60m") is not None]
    scored.sort(key=lambda h: float(h["60m"]), reverse=True)
    if len(scored) <= limit:
        return scored
    half = max(1, limit // 2)
    chosen = scored[:half] + scored[-half:]
    seen = {(h["symbol"], h["t"]) for h in chosen}
    # 不夠時用時間補
    for h in reversed(hits):
        if len(chosen) >= limit:
            break
        key = (h["symbol"], h["t"])
        if key not in seen and h.get("60m") is not None:
            chosen.append(h)
            seen.add(key)
    return chosen


def write_report(path: Path, hits: list[dict], stats: dict, funnel: dict, period: str, chart_limit: int) -> Path:
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    chart_hits = pick_chart_hits(hits, chart_limit)
    cards = []
    for n, h in enumerate(chart_hits, 1):
        img_name = f"t{n:02d}_{h['symbol']}_{hm(h['t']).replace(' ', '_').replace(':', '')}.png"
        drawn = draw_chart(h["symbol"], h["d5"], h["i"], str(img_dir / img_name), ahead=24)
        ext = (h["close"] / h["m200"] - 1) * 100
        img_html = (
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(h['symbol'])}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            if drawn
            else ""
        )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{n} · {escape(h['symbol'])}</span>"
            f"<span class='trade-time'>{escape(hm(h['t']))}</span></div>"
            f"<div class='card-pnl {_pnl_cls(h.get('60m'))}'>60m {_fmt(h.get('60m'))}</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag'>15m {_fmt(h.get('15m'))}</span>"
            f"<span class='tag'>30m {_fmt(h.get('30m'))}</span>"
            f"<span class='tag'>120m {_fmt(h.get('120m'))}</span></div>"
            "<pre class='trade-detail'>"
            f"收盤 {h['close']:g}  MA7 {h['m7']:g} > MA14 {h['m14']:g} > MA25 {h['m25']:g}\n"
            f"前一根在 MA200 下，這根站上 {h['m200']:g}（{ext:+.2f}%）\n"
            f"1h {h['h_close']:g} > MA99 {h['h_m99']:g} / MA200 {h['h_m200']:g}"
            "</pre>"
            f"{img_html}"
            "</article>"
        )

    rows = []
    for i, h in enumerate(hits, 1):
        rows.append(
            "<tr>"
            f"<td>{i}</td><td>{escape(h['symbol'])}</td><td>{escape(hm(h['t']))}</td>"
            f"<td class='{_pnl_cls(h.get('15m'))}'>{_fmt(h.get('15m'))}</td>"
            f"<td class='{_pnl_cls(h.get('30m'))}'>{_fmt(h.get('30m'))}</td>"
            f"<td class='{_pnl_cls(h.get('60m'))}'>{_fmt(h.get('60m'))}</td>"
            f"<td class='{_pnl_cls(h.get('120m'))}'>{_fmt(h.get('120m'))}</td>"
            "</tr>"
        )

    s60 = stats["60m"]
    eq = _equity_svg([float(h["60m"]) for h in hits if h.get("60m") is not None])
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 5m 多頭排列 · {escape(period)}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:8px 0 4px}}
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
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 5m 多頭排列 · {escape(period)}</h1>
<p class="muted">流動 USDT 永續 {funnel.get('symbols', 0)} 檔。
5m <b>MA7&gt;MA14&gt;MA25</b>，且前一根收在 MA200 下、這一根收盤才站上。
同時 1h 收盤在 MA99 / MA200 之上。已在 MA200 上只是短均排好的不算。
報酬從訊號收盤算到之後 15/30/60/120 分鐘收盤，不是進出場建議。</p>
<p class="muted">漏斗：5m 從 MA200 下站上且多排 {funnel.get('five_new', 0)} → 加上小時過濾 {funnel.get('hits', 0)}
· 讀檔失敗 {funnel.get('errors', 0)}</p>
<p class="muted">15m 勝率 {stats['15m']['win_rate']:.1f}% 均 {_fmt(stats['15m']['avg'])}
· 30m {stats['30m']['win_rate']:.1f}% 均 {_fmt(stats['30m']['avg'])}
· 60m {stats['60m']['win_rate']:.1f}% 均 {_fmt(stats['60m']['avg'])}
· 120m {stats['120m']['win_rate']:.1f}% 均 {_fmt(stats['120m']['avg'])}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">標的<b>{stats['symbols']}</b></div>
<div class="card">60m 勝率<b>{s60['win_rate']:.1f}%</b></div>
<div class="card">60m 平均<b class="{_pnl_cls(s60['avg'])}">{_fmt(s60['avg'])}</b></div>
</div>
<div class="equity">{eq}</div>
<p class="muted">下圖累積的是各筆 60 分鐘報酬相加，不是組合複利。圖卡只放 60m 最好/最差各一部分。</p>
</section>
{''.join(cards) or "<div class='empty'>這三天沒有符合的訊號</div>"}
<section class="summary">
<h1>全部訊號</h1>
<table>
<thead><tr><th>#</th><th>標的</th><th>時間</th><th>15m</th><th>30m</th><th>60m</th><th>120m</th></tr></thead>
<tbody>
{''.join(rows) or "<tr><td colspan='7'>無</td></tr>"}
</tbody>
</table>
</section>
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def dump_hits_json(path: Path, hits: list[dict], stats: dict, funnel: dict, period: str) -> Path:
    slim = []
    for h in hits:
        slim.append(
            {
                "symbol": h["symbol"],
                "t": h["t"],
                "time": hm(h["t"]),
                "close": h["close"],
                "m7": h["m7"],
                "m14": h["m14"],
                "m25": h["m25"],
                "m200": h["m200"],
                "h_m99": h["h_m99"],
                "h_m200": h["h_m200"],
                "15m": h.get("15m"),
                "30m": h.get("30m"),
                "60m": h.get("60m"),
                "120m": h.get("120m"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"period": period, "stats": stats, "funnel": funnel, "hits": slim}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def cmd_backtest(args) -> int:
    days = max(1, int(args.days))
    now = datetime.now(TZ)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    period = f"{datetime.fromtimestamp(start_ms / 1000, TZ).strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} ({days}d)"
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol]
        print(f"回測指定 {len(symbols)} 個：{', '.join(symbols)}", flush=True)
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"回測 {len(symbols)} 個流動永續 · {period}", flush=True)
    t0 = time.time()
    hits, funnel = backtest_all(symbols, start_ms, end_ms)
    stats = summarize_hits(hits)
    print(
        f"完成 {time.time()-t0:.1f}s　5m 從 MA200 下站上 {funnel['five_new']} → 訊號 {stats['count']} / {stats['symbols']} 檔\n"
        f"15m {stats['15m']['win_rate']:.1f}% 均 {stats['15m']['avg']:+.2f}%　"
        f"30m {stats['30m']['win_rate']:.1f}% 均 {stats['30m']['avg']:+.2f}%　"
        f"60m {stats['60m']['win_rate']:.1f}% 均 {stats['60m']['avg']:+.2f}%　"
        f"120m {stats['120m']['win_rate']:.1f}% 均 {stats['120m']['avg']:+.2f}%",
        flush=True,
    )
    html_path = Path(args.html) if args.html else (PAGES if args.pages else ROOT / "output" / "binance_5m_align.html")
    write_report(html_path, hits, stats, funnel, period, chart_limit=args.chart_limit)
    dump_hits_json(html_path.parent / "hits.json", hits, stats, funnel, period)
    if args.pages or html_path.parent == PAGES.parent:
        write_view_html(html_path)
    print(f"報告 {html_path}", flush=True)
    return 0


def wait_next_5m_close() -> None:
    now = time.time()
    nxt = (int(now) // 300 + 1) * 300 + 2
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("5m 多頭排列監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="幣安 5m 多頭排列 Telegram 監看")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的五分 K，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--dry-run", action="store_true", help="有訊號只印、不送 Telegram")
    p.add_argument("--symbol", action="append", default=[], help="只掃這些合約，可重複")
    p.add_argument("--backtest", action="store_true", help="回測近 N 天並出報告")
    p.add_argument("--days", type=int, default=3, help="回測天數（預設 3）")
    p.add_argument("--pages", action="store_true", help="寫入 docs/binance-5m-align/")
    p.add_argument("--html", default="", help="回測 HTML 路徑")
    p.add_argument("--chart-limit", type=int, default=30, help="報告圖卡最多幾張")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.backtest:
        return cmd_backtest(args)

    seen = load_seen()
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol]
        print(f"監看指定 {len(symbols)} 個：{', '.join(symbols)}", flush=True)
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"監看 {len(symbols)} 個流動永續。5m 7>14>25，且從 MA200 下那根收盤站上，1h 在 MA99/200 上才推。", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if not args.symbol and time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols)
        new = [e for e in events if key_of(e) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev, dry_run=args.dry_run)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中，每根 5m 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next_5m_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
