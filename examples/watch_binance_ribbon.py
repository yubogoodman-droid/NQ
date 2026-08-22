#!/usr/bin/env python3
"""幣安 Telegram：15 分同時站上 MA7/14/25/99/120；可順便跑 1m 黏帶三幕。

在下面填 Telegram 後執行：

    python3 examples/watch_binance_ribbon.py                 # 流動盤全掃
    python3 examples/watch_binance_ribbon.py --asset stocks  # 只要股票

Ctrl+C 結束。同一根 K 不會重發。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ribbon15 import SIGNAL_MA_PERIODS, add_mas, detect_long_breaks

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "binance_ribbon_seen.json"
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT", "HK1810USDT"}
DISPLAY = {"HK1810USDT": "小米"}
STOCK_UNDERLYING = {"EQUITY", "KR_EQUITY", "HK_EQUITY", "CN_EQUITY"}
INTERVAL_MS = {"1m": 60_000, "15m": 15 * 60_000}
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}


def apply_keys() -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


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


def universe(*, asset: str = "all") -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if asset == "stocks":
            if s.get("contractType") != "TRADIFI_PERPETUAL":
                continue
            if s.get("underlyingType") not in STOCK_UNDERLYING:
                continue
            out.append(s["symbol"])
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


def fetch_klines(sym: str, *, interval: str = "1m", limit: int = 400) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
    min_bars = 230 if interval == "1m" else 140
    if not raw or len(raw) < min_bars:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + INTERVAL_MS[interval] > now_ms:
        raw = raw[:-1]
    if len(raw) < min_bars:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def indicators(d: dict) -> dict:
    c, h, l, v = d["c"], d["h"], d["l"], d["v"]
    d = dict(d)
    d["m7"], d["m14"], d["m25"] = sma(c, 7), sma(c, 14), sma(c, 25)
    d["m99"], d["m120"], d["m200"] = sma(c, 99), sma(c, 120), sma(c, 200)
    d["v20"] = sma(v, 20)
    return d


def kiss_at(d: dict, i: int) -> dict | None:
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    c, l = d["c"], d["l"]
    if i < 210 or i >= len(c):
        return None
    vals = [m7[i], m14[i], m25[i], m99[i], m120[i], m200[i], m200[i - 30]]
    if np.isnan(vals).any():
        return None
    if not (c[i] > m200[i] and c[i - 1] <= m200[i - 1]):
        return None
    cl = (max(m99[i], m120[i], m200[i]) / min(m99[i], m120[i], m200[i]) - 1) * 100
    if cl > 0.18:
        return None
    s200 = (m200[i] / m200[i - 30] - 1) * 100
    if not (-0.35 <= s200 <= 0.05):
        return None
    ext = (c[i] / m200[i] - 1) * 100
    if not (0.02 <= ext <= 0.18):
        return None
    if not (m7[i] > m14[i] > m25[i] and m25[i] < m200[i]):
        return None
    below = 0
    for j in range(i - 1, max(0, i - 120), -1):
        if np.isnan(m200[j]) or c[j] >= m200[j]:
            break
        below += 1
    if below < 40:
        return None
    bounce = (c[i] / l[i - 40 : i + 1].min() - 1) * 100
    if not (0.70 <= bounce <= 1.35):
        return None
    return {"i": i, "cl": float(cl), "s200": float(s200), "ext": float(ext), "bounce": float(bounce), "below": below}


def lift_at(d: dict, kiss_i: int, j: int) -> bool:
    if j - kiss_i < 4 or j - kiss_i > 15:
        return False
    m99, m120, m200, v20 = d["m99"], d["m120"], d["m200"], d["v20"]
    if np.isnan([m99[j], m120[j], m200[j], v20[j]]).any():
        return False
    cl2 = (max(m99[j], m120[j], m200[j]) / min(m99[j], m120[j], m200[j]) - 1) * 100
    ext2 = (d["c"][j] / m200[j] - 1) * 100
    vr = d["v"][j] / v20[j]
    return cl2 <= 0.20 and ext2 >= 0.40 and vr >= 1.8


def look_at(d: dict, kiss_i: int) -> dict | None:
    j = kiss_i + 29
    if j >= len(d["c"]):
        return None
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    if np.isnan([m99[j], m120[j], m200[j], m7[j], m14[j], m25[j]]).any():
        return None
    cl29 = (max(m99[j], m120[j], m200[j]) / min(m99[j], m120[j], m200[j]) - 1) * 100
    ext29 = (d["c"][j] / m200[j] - 1) * 100
    short29 = (max(m7[j], m14[j], m25[j]) / min(m7[j], m14[j], m25[j]) - 1) * 100
    if cl29 <= 0.10 and ext29 >= 1.2 and short29 >= 0.50:
        return {"cl29": float(cl29), "ext29": float(ext29), "short29": float(short29)}
    return None


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def sym_label(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    name = DISPLAY.get(symbol)
    return f"{name} {base}" if name else base


def html_label(symbol: str) -> str:
    return (
        sym_label(symbol)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or f"s{abs(hash(symbol)) % 10_000_000_000}"


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


def telegram_send(text: str, *, strong: bool = False, photo: str | None = None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": text[:1024],
                        "parse_mode": "HTML",
                        "disable_notification": "false" if strong else "false",
                    },
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
                "disable_notification": False,
            },
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


def draw_chart(sym: str, d: dict, kiss_i: int, lift_i: int, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    a0 = max(0, kiss_i - 70)
    a1 = min(len(d["c"]), kiss_i + 45)
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
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}
    for n, col in pal.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    for idx, label, color in ((kiss_i, "吻", "#c9a227"), (lift_i, "離開", "#3dba7a")):
        x = idx - a0
        if 0 <= x < len(c):
            ax.axvline(x, color=color, ls="--", lw=0.9)
            ax.scatter([x], [c[x]], s=32, color=color, zorder=5)
    ax.set_title(f"{sym}  1m", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_15m_chart(sym: str, d: dict, mark_idx: int, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    if mark_idx < 0 or mark_idx >= len(d["c"]):
        return None
    a0 = max(0, mark_idx - 48)
    a1 = min(len(d["c"]), mark_idx + 8)
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
    for n, col in PAL.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    x = mark_idx - a0
    ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
    ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    title_sym = file_base(sym) if any(ord(ch) >= 128 for ch in sym) else sym
    ax.set_title(f"{title_sym}  15m  {hm(int(d['t'][mark_idx]))}", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def scan_symbol(sym: str) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    n = len(d["c"])
    last = n - 1
    events = []
    # 只看剛收盤、以及前一根（怕整點掃晚了）
    for closed in (last, last - 1):
        if closed < 220:
            continue
        # 這根是不是某次吻的離開
        for kiss_i in range(closed - 15, closed - 3):
            kiss = kiss_at(d, kiss_i)
            if not kiss:
                continue
            if not lift_at(d, kiss_i, closed):
                continue
            # 確認這是第一次離開（吻後 4~15 根裡這根最早）
            first = None
            for j in range(kiss_i + 4, closed + 1):
                if lift_at(d, kiss_i, j):
                    first = j
                    break
            if first != closed:
                continue
            events.append(
                {
                    "kind": "lift",
                    "symbol": sym,
                    "kiss": kiss,
                    "kiss_i": kiss_i,
                    "lift_i": closed,
                    "d": d,
                }
            )
        # 這根是不是吻後第 29 根，完成樣貌
        kiss_i = closed - 29
        kiss = kiss_at(d, kiss_i)
        if kiss:
            lifted = any(lift_at(d, kiss_i, j) for j in range(kiss_i + 4, min(kiss_i + 16, closed + 1)))
            look = look_at(d, kiss_i) if lifted else None
            if look:
                lift_i = next(j for j in range(kiss_i + 4, kiss_i + 16) if lift_at(d, kiss_i, j))
                events.append(
                    {
                        "kind": "look",
                        "symbol": sym,
                        "kiss": kiss,
                        "kiss_i": kiss_i,
                        "lift_i": lift_i,
                        "look": look,
                        "d": d,
                    }
                )
    return events


def scan_symbol_15m(sym: str) -> list[dict]:
    raw = fetch_klines(sym, interval="15m", limit=250)
    if raw is None:
        return []
    d = add_mas(raw)
    hits = detect_long_breaks(d, periods=SIGNAL_MA_PERIODS)
    n = len(d["c"])
    events = []
    for br in hits:
        if br.idx not in (n - 1, n - 2):
            continue
        events.append({"kind": "m15", "symbol": sym, "break": br, "d": d})
    return events


def format_lift(ev: dict) -> str:
    d, kiss, sym = ev["d"], ev["kiss"], ev["symbol"]
    ks = hm(int(d["t"][ev["kiss_i"]]))
    ls = hm(int(d["t"][ev["lift_i"]]))
    px = d["c"][ev["lift_i"]]
    return (
        f"<b>離開</b>  {html_label(sym)}\n"
        f"吻 {ks} → 離開 {ls}\n"
        f"現價 {px:g}\n"
        f"長均黏度 {kiss['cl']:.3f}%　回彈 {kiss['bounce']:.2f}%\n"
        f"短均 7&gt;14&gt;25，剛從帶子放量走。\n"
        f"<i>還不是完成樣貌。約 15–25 分鐘後若長均更黏、價帶走短均，會再推一則強訊號。</i>"
    )


def format_look(ev: dict) -> str:
    d, kiss, look, sym = ev["d"], ev["kiss"], ev["look"], ev["symbol"]
    ks = hm(int(d["t"][ev["kiss_i"]]))
    ls = hm(int(d["t"][ev["lift_i"]]))
    px = d["c"][ev["kiss_i"] + 29]
    return (
        f"🔥🔥🔥 <b>完全符合（強）</b>\n"
        f"<b>{html_label(sym)}</b>  1m\n"
        f"吻 {ks} → 離開 {ls}\n"
        f"現價 {px:g}\n"
        f"長均黏度 {kiss['cl']:.3f}% → {look['cl29']:.3f}%\n"
        f"價離開帶子 {look['ext29']:+.2f}%　短均散開 {look['short29']:.2f}%\n"
        f"這就是 NBIS 那張圖：帶子還在下面黏著，短均被帶走。"
    )


def format_m15(ev: dict) -> str:
    d, br, sym = ev["d"], ev["break"], ev["symbol"]
    ts = hm(int(d["t"][br.idx]))
    body = "是" if br.body_through else "否"
    return (
        f"<b>15分 同時站上 7/14/25/99/120</b>\n"
        f"<b>{html_label(sym)}</b>\n"
        f"時間 {ts}（GMT+8）\n"
        f"收 {br.close:g}　開 {br.open:g}　低 {br.low:g}\n"
        f"帶子寬度 {br.width_pct:.2f}%　量比 {br.vol_ratio:.2f}×　實體穿越 {body}\n"
        f"<i>前一根收在五條均線下方，這一根收盤同時站上。不必過 MA200。</i>"
    )


def key_of(ev: dict) -> str:
    if ev["kind"] == "m15":
        return f"15m:{ev['symbol']}:{int(ev['d']['t'][ev['break'].idx])}"
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['kiss_i']])}:{ev['kind']}"


def scan_all(symbols: list[str], fn) -> list[dict]:
    events = []
    if not symbols:
        return events
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fn, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    return events


def should_scan_15m() -> bool:
    return datetime.now(TZ).minute % 15 <= 1


def strip_html(text: str) -> str:
    return (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )


def notify(ev: dict) -> None:
    if ev["kind"] == "m15":
        text = format_m15(ev)
        print("\n" + strip_html(text))
        tmp = Path("/tmp") / f"ribbon15_{file_base(ev['symbol'])}_{ev['break'].idx}.png"
        photo = draw_15m_chart(ev["symbol"], ev["d"], ev["break"].idx, str(tmp))
        telegram_send("15分 同時站上 " + sym_label(ev["symbol"]), strong=True)
        ok = telegram_send(text, strong=True, photo=photo)
    else:
        strong = ev["kind"] == "look"
        text = format_look(ev) if strong else format_lift(ev)
        print("\n" + strip_html(text))
        photo = None
        if strong:
            tmp = Path("/tmp") / f"ribbon_{ev['symbol']}_{ev['kiss_i']}.png"
            photo = draw_chart(ev["symbol"], ev["d"], ev["kiss_i"], ev["lift_i"], str(tmp))
            telegram_send("🔥🔥🔥 完全符合（強） " + ev["symbol"], strong=True)
        ok = telegram_send(text, strong=strong, photo=photo)
    if ok:
        print("  → Telegram 已送")
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
        else:
            print("  → Telegram 送出失敗，檢查 token 與 chat id")


def wait_next_close() -> None:
    now = time.time()
    # 等到下一分鐘 + 2 秒，讓 K 收盤
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("15分同時站上監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def make_demo_15m_bars() -> dict:
    n = 180
    px = np.full(n, 100.0)
    o = np.full(n, 100.0)
    h = np.full(n, 100.05)
    l = np.full(n, 99.95)
    v = np.full(n, 1000.0)
    o[140], l[140], h[140], px[140] = 100.0, 98.90, 100.02, 99.00
    v[140] = 1800
    o[141], l[141], h[141], px[141] = 99.05, 98.95, 101.60, 101.40
    v[141] = 3200
    return {"t": np.arange(n, dtype=np.int64) * 15 * 60_000, "o": o, "h": h, "l": l, "c": px, "v": v}


def run_demo() -> int:
    d = add_mas(make_demo_15m_bars())
    hits = detect_long_breaks(d, periods=SIGNAL_MA_PERIODS)
    print(f"demo 15m 偵測到 {len(hits)} 筆")
    for br in hits:
        print(f"  idx={br.idx} close={br.close:.3f} width={br.width_pct:.3f}% body={br.body_through}")
    if not hits:
        print("demo 失敗：應偵測到同時站上 7/14/25/99/120")
        return 1
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="15 分同時站上 7/14/25/99/120 Telegram 監看")
    p.add_argument("--once", action="store_true", help="掃一輪就結束（15 分看剛收的兩根）")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--demo", action="store_true", help="只用合成 15 分 K 驗證偵測")
    p.add_argument("--asset", choices=("stocks", "all"), default="all", help="15 分掃描範圍，預設全掃流動盤")
    p.add_argument("--also-1m", action="store_true", help="順便跑原本的 1m 黏帶離開/強訊號")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.demo:
        return run_demo()

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols_15m = universe(asset=args.asset)
    symbols_1m = universe(asset="all") if args.also_1m else []
    label = "股票永續" if args.asset == "stocks" else "流動永續（加密+股票）"
    print(
        f"15 分同時站上 7/14/25/99/120：監看 {len(symbols_15m)} 個{label}。",
        flush=True,
    )
    if args.also_1m:
        print(f"另外監看 {len(symbols_1m)} 個 1m 黏帶。", flush=True)
    uni_ts = time.time()

    def round_once(*, force_15m: bool = False) -> None:
        nonlocal symbols_15m, symbols_1m, uni_ts
        if time.time() - uni_ts > 1800:
            symbols_15m = universe(asset=args.asset)
            if args.also_1m:
                symbols_1m = universe(asset="all")
            uni_ts = time.time()
            print(f"更新標的 15m {len(symbols_15m)}" + (f"  1m {len(symbols_1m)}" if args.also_1m else ""), flush=True)
        t0 = time.time()
        events: list[dict] = []
        scanned = []
        if args.also_1m:
            one = scan_all(symbols_1m, scan_symbol)
            one.sort(key=lambda e: (0 if e["kind"] == "look" else 1, e["symbol"]))
            events.extend(one)
            scanned.append("1m")
        if force_15m or should_scan_15m():
            events.extend(scan_all(symbols_15m, scan_symbol_15m))
            scanned.append("15m")
        new = [e for e in events if key_of(e) not in seen]
        tag = "+".join(scanned) if scanned else "idle"
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"{tag} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev)
        if new:
            save_seen(seen)

    round_once(force_15m=True)
    if args.once:
        return 0
    print("watch 中：每根 15 分收盤掃一次條件（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
