#!/usr/bin/env python3
"""幣安 1m 黏帶三幕：離開推 Telegram；完全符合再推強訊號（帶圖）。

在下面填 Telegram 後執行：

    python3 examples/watch_binance_ribbon.py

Ctrl+C 結束。同一根 K 不會重發。
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "binance_ribbon_seen.json"
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}


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


def fetch_klines(sym: str, limit: int = 400) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": limit})
    if not raw or len(raw) < 230:
        return None
    now_ms = int(time.time() * 1000)
    # 丟掉還沒收盤的當根
    if int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
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


def format_lift(ev: dict) -> str:
    d, kiss, sym = ev["d"], ev["kiss"], ev["symbol"]
    ks = hm(int(d["t"][ev["kiss_i"]]))
    ls = hm(int(d["t"][ev["lift_i"]]))
    px = d["c"][ev["lift_i"]]
    return (
        f"<b>離開</b>  {sym}\n"
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
        f"<b>{sym}</b>  1m\n"
        f"吻 {ks} → 離開 {ls}\n"
        f"現價 {px:g}\n"
        f"長均黏度 {kiss['cl']:.3f}% → {look['cl29']:.3f}%\n"
        f"價離開帶子 {look['ext29']:+.2f}%　短均散開 {look['short29']:.2f}%\n"
        f"這就是 NBIS 那張圖：帶子還在下面黏著，短均被帶走。"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['kiss_i']])}:{ev['kind']}"


def scan_all(symbols: list[str]) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    # look 優先
    events.sort(key=lambda e: (0 if e["kind"] == "look" else 1, e["symbol"]))
    return events


def notify(ev: dict) -> None:
    strong = ev["kind"] == "look"
    text = format_look(ev) if strong else format_lift(ev)
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("&gt;", ">"))
    photo = None
    if strong:
        tmp = Path("/tmp") / f"ribbon_{ev['symbol']}_{ev['kiss_i']}.png"
        photo = draw_chart(ev["symbol"], ev["d"], ev["kiss_i"], ev["lift_i"], str(tmp))
        # 強訊號連發文字 + 圖，比較不容易被滑過
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
    ok = telegram_send("黏帶監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安黏帶三幕 Telegram 監看")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe()
    print(f"監看 {len(symbols)} 個流動永續。離開會推；完全符合再推強訊號。", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
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
            notify(ev)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中，每根 1m 收盤掃一次（Ctrl+C 停）", flush=True)
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
