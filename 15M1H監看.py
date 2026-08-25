#!/usr/bin/env python3
"""15 分 + 1 小時 MA200 監看 → Telegram。這一個檔就能跑，不必 nq 資料夾。

PyCharm：開新專案，把本檔貼進去，按 Run，視窗不要關。

    pip install numpy requests matplotlib
    python 15M1H監看.py --test
    python 15M1H監看.py

15m：剛站上 15m MA200 且收盤 > MA7 > MA25，再連 3 根都在年線上，
     且 1h MA25 未下彎、1h 收盤 > MA7 > MA25。第 3 根才推。
1h ：本根剛站上 1h MA200 且收盤 > MA7 > MA25，且 1h MA25 未下彎。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字

BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"}
)
INTERVAL_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}
TZ = timezone(timedelta(hours=8))
HERE = Path(__file__).resolve().parent
SEEN_PATH = HERE / "ma_bull_tg_seen.json"
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}

TF_WATCH = {
    "15m": {
        "signal": "15m",
        "htf": "1h",
        "htf_ms": INTERVAL_MS["1h"],
        "sig_limit": 280,
        "htf_limit": 250,
        "require_h1_ma25_up": True,
        "require_h1_stack": True,
        "lookback": 48,
        "min_hold_bars": 3,
        "title": "15m MA200 上連 3 根",
    },
    "1h": {
        "signal": "1h",
        "htf": "4h",
        "htf_ms": INTERVAL_MS["4h"],
        "sig_limit": 420,
        "htf_limit": 250,
        "require_h1_ma25_up": True,
        "require_h1_stack": False,
        "lookback": 80,
        "min_hold_bars": None,
        "title": "1h 剛站上 MA200",
    },
}


def get_json(path: str, params=None, retries: int = 6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(1.4 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.45 * (i + 1))
    raise last


def universe(top_n: int = 100) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    ranked = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        ranked.append((qv, sym))
    ranked.sort(reverse=True)
    return [sym for _, sym in ranked[:top_n]]


def fetch_klines(sym: str, interval: str, limit: int = 500, extra_bars: int = 8) -> dict | None:
    need = limit
    chunks: list[list] = []
    end_time = None
    while sum(len(c) for c in chunks) < need:
        params = {"symbol": sym, "interval": interval, "limit": min(1500, need)}
        if end_time is not None:
            params["endTime"] = end_time
        raw = get_json("/fapi/v1/klines", params)
        if not raw:
            break
        chunks.append(raw)
        if len(raw) < params["limit"]:
            break
        end_time = int(raw[0][0]) - 1
    rows = []
    seen: set[int] = set()
    for chunk in reversed(chunks):
        for x in chunk:
            t = int(x[0])
            if t in seen:
                continue
            seen.add(t)
            rows.append(x)
    if not rows:
        return None
    now_ms = int(time.time() * 1000)
    if int(rows[-1][0]) + INTERVAL_MS[interval] > now_ms:
        rows = rows[:-1]
    rows = rows[-need:]
    if not rows:
        return None
    return {
        "t": np.array([int(x[0]) for x in rows], np.int64),
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
        "v": np.array([float(x[5]) for x in rows]),
    }


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def add_mas(d: dict) -> dict:
    out = dict(d)
    c, v = d["c"], d["v"]
    out["m7"] = sma(c, 7)
    out["m14"] = sma(c, 14)
    out["m25"] = sma(c, 25)
    out["m200"] = sma(c, 200)
    out["v20"] = sma(v, 20)
    return out


def stack_ok(c, m7, m25, m200, i: int) -> bool:
    px, ma7, ma25, ma200 = c[i], m7[i], m25[i], m200[i]
    if np.isnan([px, ma7, ma25, ma200]).any():
        return False
    return bool(px > ma7 > ma25 and px > ma200)


@dataclass(frozen=True)
class Sig:
    idx: int
    close: float
    m7: float
    m25: float
    ma200: float
    vol_ratio: float
    ext_pct: float
    crossed_200: bool
    bars_below: int
    bars_above: int


def bars_below(c, m200, i: int) -> int:
    n, j = 0, i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] <= m200[j]:
        n += 1
        j -= 1
    return n


def bars_above(c, m200, i: int) -> int:
    n, j = 0, i - 1
    while j >= 0 and not np.isnan(m200[j]) and c[j] > m200[j]:
        n += 1
        j -= 1
    return n


def signal_at(d: dict, i: int) -> Sig | None:
    if i < 200 or i >= len(d["c"]):
        return None
    c, m7, m25, m200, v, v20 = d["c"], d["m7"], d["m25"], d["m200"], d["v"], d["v20"]
    if np.isnan([m200[i], m200[i - 1]]).any() or not stack_ok(c, m7, m25, m200, i):
        return None
    vr = float(v[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
    ext = (c[i] / m200[i] - 1.0) * 100.0 if m200[i] else 0.0
    return Sig(
        idx=i,
        close=float(c[i]),
        m7=float(m7[i]),
        m25=float(m25[i]),
        ma200=float(m200[i]),
        vol_ratio=vr,
        ext_pct=float(ext),
        crossed_200=bool(c[i - 1] <= m200[i - 1] and c[i] > m200[i]),
        bars_below=bars_below(c, m200, i),
        bars_above=bars_above(c, m200, i),
    )


def detect_combo(d: dict) -> list[Sig]:
    c, m7, m25, m200 = d["c"], d["m7"], d["m25"], d["m200"]
    out = []
    for i in range(200, len(c)):
        if np.isnan([m200[i], m200[i - 1]]).any():
            continue
        if not stack_ok(c, m7, m25, m200, i) or stack_ok(c, m7, m25, m200, i - 1):
            continue
        sig = signal_at(d, i)
        if sig:
            out.append(sig)
    return out


def ma200_held(d: dict, start: int, n: int) -> bool:
    end = start + n - 1
    if start < 1 or end >= len(d["c"]):
        return False
    c, m200 = d["c"], d["m200"]
    for k in range(start, end + 1):
        if np.isnan(m200[k]) or float(c[k]) <= float(m200[k]):
            return False
    return True


def confirm_hold(d: dict, sig: Sig, hold_bars: int) -> Sig | None:
    if not sig.crossed_200:
        return None
    if hold_bars <= 1:
        return sig
    if not ma200_held(d, sig.idx, hold_bars):
        return None
    return signal_at(d, sig.idx + hold_bars - 1)


def htf_sma_at(d_htf: dict, time_ms: int, last_price: float, bar_ms: int, n: int) -> float | None:
    t = d_htf["t"]
    opened = t <= time_ms
    if not opened.any():
        return None
    last_i = int(np.where(opened)[0][-1])
    if last_i + 1 < n:
        return None
    c = np.array(d_htf["c"][: last_i + 1], dtype=float)
    if int(t[last_i]) + bar_ms > time_ms:
        c[-1] = float(last_price)
    window = c[-n:]
    if np.isnan(window).any():
        return None
    return float(window.mean())


def htf_ma25_now_prev(d_htf, time_ms, last_price, bar_ms):
    if d_htf is None or len(d_htf.get("c", [])) < 26:
        return None, None
    now = htf_sma_at(d_htf, time_ms, last_price, bar_ms, 25)
    t = d_htf["t"]
    opened = t <= time_ms
    if not opened.any():
        return now, None
    last_i = int(np.where(opened)[0][-1])
    if last_i < 25:
        return now, None
    prev_win = np.array(d_htf["c"][last_i - 25 : last_i], dtype=float)
    if len(prev_win) < 25 or np.isnan(prev_win).any():
        return now, None
    return now, float(prev_win.mean())


def htf_ma25_not_down(d_htf, time_ms, last_price, bar_ms) -> bool:
    now, prev = htf_ma25_now_prev(d_htf, time_ms, last_price, bar_ms)
    return now is not None and prev is not None and now >= prev


def htf_ma7_25_at(d_htf, time_ms, last_price, bar_ms):
    if d_htf is None:
        return None, None
    return htf_sma_at(d_htf, time_ms, last_price, bar_ms, 7), htf_sma_at(
        d_htf, time_ms, last_price, bar_ms, 25
    )


def htf_ma7_25_stack(d_htf, time_ms, last_price, bar_ms) -> bool:
    m7, m25 = htf_ma7_25_at(d_htf, time_ms, last_price, bar_ms)
    return m7 is not None and m25 is not None and float(last_price) > m7 > m25


def htf_ma200_at(d_htf, time_ms, last_price, bar_ms):
    if d_htf is None:
        return None
    return htf_sma_at(d_htf, time_ms, last_price, bar_ms, 200)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "sym"


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.write_text(json.dumps(sorted(seen)), encoding="utf-8")


def telegram_send(text: str, photo: str | None = None) -> bool:
    token = TELEGRAM_BOT_TOKEN.strip()
    chat_id = TELEGRAM_CHAT_ID.strip()
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
    except Exception:
        return False


def tf_bar_idx(d: dict, time_ms: int, bar_ms: int) -> int | None:
    t = d["t"]
    open_ms = int(time_ms) - (int(time_ms) % bar_ms)
    w = np.where(t == open_ms)[0]
    if len(w):
        return int(w[0])
    w = np.where(t <= time_ms)[0]
    return int(w[-1]) if len(w) else None


def _style_ax(ax) -> None:
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")


def _paint(ax, axv, d: dict, a0: int, a1: int, mark_i: int | None) -> None:
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


def draw_chart(sym: str, d: dict, sig: Sig, spec: dict, path: str, d_htf, d_4h=None) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    i, ts = sig.idx, int(d["t"][sig.idx])
    panels = [(d, spec["signal"], max(0, i - spec["lookback"]), min(len(d["c"]), i + 4), i)]
    hi = tf_bar_idx(d_htf, ts, spec["htf_ms"]) if d_htf is not None and len(d_htf.get("c", [])) else None
    if d_htf is not None and hi is not None:
        panels.append((d_htf, spec["htf"], max(0, hi - 48), min(len(d_htf["c"]), hi + 2), hi))
    if spec["signal"] == "15m" and d_4h is not None and len(d_4h.get("c", [])):
        i4 = tf_bar_idx(d_4h, ts, INTERVAL_MS["4h"])
        if i4 is not None:
            panels.append((d_4h, "4h", max(0, i4 - 48), min(len(d_4h["c"]), i4 + 2), i4))
    n = len(panels)
    ratios = []
    for k in range(n):
        ratios.extend([3.1, 0.9] if k == 0 else [2.4, 0.75])
    fig, axes = plt.subplots(
        n * 2, 1, figsize=(10.6, 5.4 + 4.6 * (n - 1)), sharex=False,
        gridspec_kw={"height_ratios": ratios}, facecolor="#0c1210",
    )
    if n == 1:
        axes = [axes[0], axes[1]]
    for a in axes:
        _style_ax(a)
    title_sym = file_base(sym)
    for k, (frame, tf, a0, a1, mark) in enumerate(panels):
        _paint(axes[2 * k], axes[2 * k + 1], frame, a0, a1, mark)
        extra = ""
        if k == 0:
            extra = f"  vol={sig.vol_ratio:.2f}x  ext={sig.ext_pct:+.2f}%"
        else:
            h_close = float(frame["c"][mark])
            h_ma = float(frame["m200"][mark]) if not np.isnan(frame["m200"][mark]) else None
            if h_ma:
                extra = f"  close {h_close:g} vs {tf} MA200 {h_ma:g}"
            if tf == "4h" or spec["htf"] == "4h":
                extra += "  compare only"
        axes[2 * k].set_title(
            f"{title_sym}  {tf}  {hm(int(frame['t'][mark]))}{extra}",
            color="#e8f0ea", fontsize=11,
        )
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def passes(sig: Sig, spec: dict, d, d_htf, ts: int) -> bool:
    src = d if spec["signal"] == "1h" else d_htf
    if spec.get("require_h1_ma25_up") and not htf_ma25_not_down(src, ts, sig.close, INTERVAL_MS["1h"]):
        return False
    if spec.get("require_h1_stack") and not htf_ma7_25_stack(src, ts, sig.close, INTERVAL_MS["1h"]):
        return False
    if spec.get("min_hold_bars"):
        return True
    return bool(sig.crossed_200)


def scan_symbol(sym: str, spec: dict) -> list[dict]:
    raw = fetch_klines(sym, spec["signal"], limit=spec["sig_limit"])
    if raw is None or len(raw["c"]) < 220:
        return []
    d = add_mas(raw)
    raw_h = fetch_klines(sym, spec["htf"], limit=spec["htf_limit"])
    d_htf = add_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
    n = len(d["c"])
    by_idx = {s.idx: s for s in detect_combo(d)}
    hold = spec.get("min_hold_bars")
    events = []
    for closed in (n - 1, n - 2):
        if closed < 200:
            continue
        if hold:
            start = closed - (hold - 1)
            sig0 = by_idx.get(start)
            if sig0 is None or not sig0.crossed_200:
                continue
            held = confirm_hold(d, sig0, hold)
            if held is None or held.idx != closed:
                continue
            ts = int(d["t"][held.idx])
            if not passes(held, spec, d, d_htf, ts):
                continue
            events.append({"symbol": sym, "sig": held, "d": d, "d_htf": d_htf, "spec": spec})
            continue
        sig = by_idx.get(closed)
        if sig is None:
            continue
        ts = int(d["t"][sig.idx])
        if not passes(sig, spec, d, d_htf, ts):
            continue
        events.append({"symbol": sym, "sig": sig, "d": d, "d_htf": d_htf, "spec": spec})
    return events


def format_ev(ev: dict) -> str:
    d, sig, spec = ev["d"], ev["sig"], ev["spec"]
    sym, ts = ev["symbol"], hm(int(d["t"][sig.idx]))
    tf, htf = spec["signal"], spec["htf"]
    ma_h = htf_ma200_at(ev.get("d_htf"), int(d["t"][sig.idx]), sig.close, spec["htf_ms"])
    htxt = f"{htf} MA200 {ma_h:g}　距 {(sig.close / ma_h - 1) * 100:+.2f}%" if ma_h else f"{htf} MA200 —"
    extra = f"距 {tf} MA200 {sig.ext_pct:+.2f}%　量比 {sig.vol_ratio:.2f}×\n"
    if spec.get("min_hold_bars"):
        extra = f"{tf} MA200 上已連 {spec['min_hold_bars']} 根（剛站上後維持）\n" + extra
    src = d if spec["signal"] == "1h" else ev.get("d_htf")
    now, prev = htf_ma25_now_prev(src, int(d["t"][sig.idx]), sig.close, INTERVAL_MS["1h"])
    if now is not None and prev is not None:
        extra += f"1h MA25 {now:g} ≥ 前一根 {prev:g}（未下彎）\n"
    if spec.get("require_h1_stack"):
        m7, m25 = htf_ma7_25_at(src, int(d["t"][sig.idx]), sig.close, INTERVAL_MS["1h"])
        if m7 is not None and m25 is not None:
            extra += f"1h MA7 {m7:g} &gt; MA25 {m25:g}（多頭排列）\n"
    if spec["signal"] == "15m":
        extra += "圖附 4h 對照（不擋單）\n"
    return (
        f"<b>{spec['title']}</b>\n"
        f"<b>{sym.replace('USDT', '')}</b>  {sym}\n"
        f"{ts}  收 {sig.close:g}\n"
        f"收盤 &gt; MA7 {sig.m7:g} &gt; MA25 {sig.m25:g}\n"
        f"且 &gt; {tf} MA200 {sig.ma200:g}\n"
        f"{extra}"
        f"{htxt}（參考，不擋單）"
    )


def key_of(ev: dict) -> str:
    return f"{ev['spec']['signal']}:{ev['symbol']}:{int(ev['d']['t'][ev['sig'].idx])}"


def notify(ev: dict) -> None:
    text = format_ev(ev)
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">"))
    spec = ev["spec"]
    tmp = Path(tempfile.gettempdir()) / f"ma_{spec['signal']}_{ev['symbol']}_{ev['sig'].idx}.png"
    d_4h = None
    if spec["signal"] == "15m":
        raw4 = fetch_klines(ev["symbol"], "4h", limit=460)
        d_4h = add_mas(raw4) if raw4 is not None and len(raw4["c"]) >= 200 else None
    photo = draw_chart(ev["symbol"], ev["d"], ev["sig"], spec, str(tmp), ev.get("d_htf"), d_4h)
    ok = telegram_send(text, photo=photo)
    print("  → Telegram 已送" if ok else "  → Telegram 送出失敗，檢查 token / chat id")


def next_close_unix(interval_ms: int) -> float:
    now = time.time()
    step = interval_ms / 1000
    return (int(now) // int(step) + 1) * step + 3


def due_to_scan(spec: dict, *, force: bool) -> bool:
    if force or spec["signal"] == "15m":
        return True
    now = time.time()
    hour_close = (int(now) // 3600) * 3600
    return (now - hour_close) < 14 * 60


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    if not TELEGRAM_BOT_TOKEN.strip() or not TELEGRAM_CHAT_ID.strip():
        print("請在檔案最上面填 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID")
        if args.test:
            return 1
    if args.test:
        ok = telegram_send("MA200 監看測試\n15m + 1h 同一個腳本\n如果你看到這則，Telegram 已通。")
        print("Telegram 測試", "成功" if ok else "失敗")
        return 0 if ok else 1

    specs = [TF_WATCH["15m"], TF_WATCH["1h"]]
    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe()
    print(f"同一個腳本監看 {len(symbols)} 個成交額前100：15m + 1h", flush=True)
    print("  · 15m 剛站上後連 3 根，且 1h MA25 未下彎、1h 7>25", flush=True)
    print("  · 1h 本根剛站上 MA200，且 1h MA25 未下彎", flush=True)
    if not args.once:
        telegram_send(
            "<b>監看已啟動</b>\n"
            "同一個腳本盯 15m 與 1h，收盤掃一次，符合才推圖。\n"
            "15m：剛站上年線後連 3 根，且 1h MA25 未下彎、1h 7&gt;25\n"
            "1h：本根剛站上 1h MA200，且 1h MA25 未下彎"
        )
    uni_ts = time.time()
    first = True

    def round_once() -> None:
        nonlocal symbols, uni_ts, first
        if time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        force = first or args.once
        first = False
        t0 = time.time()
        jobs = [(s, spec) for spec in specs if due_to_scan(spec, force=force) for s in symbols]
        if not jobs:
            return
        events = []
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(scan_symbol, sym, spec): (sym, spec) for sym, spec in jobs}
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as e:
                    sym, spec = futs[fut]
                    print("err", spec["signal"], sym, e, flush=True)
        new = [e for e in events if key_of(e) not in seen]
        scanned = sorted({spec["signal"] for _, spec in jobs})
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} × {'+'.join(scanned)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中（15m + 1h），收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            nxt = min(next_close_unix(INTERVAL_MS[s["signal"]]) for s in specs)
            time.sleep(max(1, nxt - time.time()))
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
