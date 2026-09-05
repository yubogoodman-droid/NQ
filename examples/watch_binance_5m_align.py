#!/usr/bin/env python3
"""幣安 5m 多頭排列 Telegram 監看。

五分 K：MA7 > MA14 > MA25，且收盤站上 MA200。
同時小時 K 收盤要在 MA99 與 MA200 之上。
條件剛成立的那根 5m 收盤推一次，同一根不重發。

    python3 examples/watch_binance_5m_align.py --test
    python3 examples/watch_binance_5m_align.py --once --dry-run
    python3 examples/watch_binance_5m_align.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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


def five_align_ok(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    vals = [d["m7"][i], d["m14"][i], d["m25"][i], d["m200"][i]]
    if np.isnan(vals).any():
        return False
    return vals[0] > vals[1] > vals[2] and d["c"][i] > vals[3]


def hour_above_ok(h: dict, i: int) -> bool:
    if i < 0 or i >= len(h["c"]):
        return False
    m99, m200 = h["m99"][i], h["m200"][i]
    if np.isnan([m99, m200]).any():
        return False
    return h["c"][i] > m99 and h["c"][i] > m200


def detect_new_align(d5: dict, i: int, h1: dict, hi: int) -> dict | None:
    """剛收的 5m 第一次同時滿足多頭排列 + 小時站上 99/200。"""
    if not five_align_ok(d5, i) or not hour_above_ok(h1, hi):
        return None
    if five_align_ok(d5, i - 1):
        return None
    return {
        "i": i,
        "hi": hi,
        "close": float(d5["c"][i]),
        "m7": float(d5["m7"][i]),
        "m14": float(d5["m14"][i]),
        "m25": float(d5["m25"][i]),
        "m200": float(d5["m200"][i]),
        "h_close": float(h1["c"][hi]),
        "h_m99": float(h1["m99"][hi]),
        "h_m200": float(h1["m200"][hi]),
        "t": int(d5["t"][i]),
    }


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


def draw_chart(sym: str, d: dict, i: int, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    a0 = max(0, i - 80)
    a1 = min(len(d["c"]), i + 4)
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
    ax.set_title(f"{sym}  5m  多頭排列", color="#e8f0ea", fontsize=12)
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
    last5, last1 = len(d5["c"]) - 1, len(h1["c"]) - 1
    events = []
    for closed in (last5, last5 - 1):
        sig = detect_new_align(d5, closed, h1, last1)
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
        f"收盤站上 MA200 {sig['m200']:g}（{ext:+.2f}%）\n"
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
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    seen = load_seen()
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol]
        print(f"監看指定 {len(symbols)} 個：{', '.join(symbols)}", flush=True)
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"監看 {len(symbols)} 個流動永續。5m 7>14>25 且收盤站上 MA200，1h 在 MA99/200 上才推。", flush=True)
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
