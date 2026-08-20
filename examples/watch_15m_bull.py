#!/usr/bin/env python3
"""幣安 15 分 K：收盤在 MA7/14/25/200 之上且剛站上 MA200 → Telegram。

在下面填 Telegram 後執行：

    python3 examples/watch_15m_bull.py --test
    python3 examples/watch_15m_bull.py
    python3 examples/watch_15m_bull.py --stocks   # 只掃幣安股票永續

Ctrl+C 結束。同一根 K 不會重發。
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.binance import INTERVAL_MS, SESSION, fetch_klines, universe
from nq.ma15_bull import add_15m_mas, above_htf_ma200, detect_combo, htf_ma200_at, sma

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "ma15_bull_seen.json"
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}


def apply_keys() -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


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
    except Exception:
        return False


def draw_chart(sym: str, d: dict, idx: int, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    a0 = max(0, idx - 48)
    a1 = min(len(d["c"]), idx + 4)
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
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.35 if n == 200 else 1.15, label=f"MA{n}")
    x = idx - a0
    ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
    ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    ax.set_title(f"{sym}  15m", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


WATCH = {"signal": "15m", "htf": "1h", "htf_ms": INTERVAL_MS["1h"], "sig_limit": 280, "htf_limit": 250}


def scan_symbol(sym: str) -> list[dict]:
    raw = fetch_klines(sym, interval=WATCH["signal"], limit=WATCH["sig_limit"], extra_bars=8)
    if raw is None or len(raw["c"]) < 220:
        return []
    d = add_15m_mas(raw)
    raw_h = fetch_klines(sym, interval=WATCH["htf"], limit=WATCH["htf_limit"], extra_bars=8)
    d_htf = add_15m_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
    n = len(d["c"])
    events = []
    for closed in (n - 1, n - 2):
        if closed < 200:
            continue
        hits = [s for s in detect_combo(d, min_gap_bars=0) if s.idx == closed]
        for sig in hits:
            if not sig.crossed_200:
                continue
            ts = int(d["t"][sig.idx])
            if not above_htf_ma200(d_htf, ts, sig.close, WATCH["htf_ms"]):
                continue
            events.append({"symbol": sym, "sig": sig, "d": d, "d_htf": d_htf})
    return events


def format_ev(ev: dict) -> str:
    d, sig, sym = ev["d"], ev["sig"], ev["symbol"]
    ts = hm(int(d["t"][sig.idx]))
    kind = "剛站上 MA200" if sig.crossed_200 else "多頭排列剛成立"
    tf, htf = WATCH["signal"], WATCH["htf"]
    ma_h = htf_ma200_at(ev.get("d_htf"), int(d["t"][sig.idx]), sig.close, WATCH["htf_ms"]) if ev.get("d_htf") is not None else None
    htxt = f"{htf} MA200 {ma_h:g}　距 {(sig.close / ma_h - 1) * 100:+.2f}%" if ma_h else f"{htf} MA200 —"
    return (
        f"<b>{tf} 收盤在 7/14/25/200 上 · {kind}</b>\n"
        f"<b>{sym}</b>\n"
        f"{ts}  收 {sig.close:g}\n"
        f"收盤 &gt; MA7 {sig.m7:g} &gt; MA14 {sig.m14:g} &gt; MA25 {sig.m25:g}\n"
        f"且 &gt; {tf} MA200 {sig.ma200:g}　距 {sig.ext_pct:+.2f}%　量比 {sig.vol_ratio:.2f}×\n"
        f"且 &gt; {htxt}"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['sig'].idx])}"


def notify(ev: dict) -> None:
    text = format_ev(ev)
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">"))
    tmp = Path("/tmp") / f"ma15_{ev['symbol']}_{ev['sig'].idx}.png"
    photo = draw_chart(ev["symbol"], ev["d"], ev["sig"].idx, str(tmp))
    ok = telegram_send(text, photo=photo)
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
    step = INTERVAL_MS[WATCH["signal"]] / 1000
    nxt = (int(now) // int(step) + 1) * step + 3
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("15m 7/14/25 多頭 × MA200 監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="15 分 K 7/14/25 多頭且站上 15m MA200 Telegram")
    p.add_argument("--once", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--stocks", action="store_true", help="只掃幣安 TradFi 股票永續（不含商品）")
    p.add_argument("--tf", choices=("15m", "1h"), default="15m", help="訊號週期；1h 時大週期改用 4h MA200")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    if args.tf == "1h":
        WATCH.update({"signal": "1h", "htf": "4h", "htf_ms": INTERVAL_MS["4h"], "sig_limit": 250, "htf_limit": 250})

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe(stocks_only=args.stocks)
    scope = "幣安股票永續" if args.stocks else "流動永續"
    print(
        f"監看 {len(symbols)} 個{scope}。{WATCH['signal']} 剛站上 MA200、收在 7/14/25/200 上、"
        f"且當下在 {WATCH['htf']} MA200 上會推 Telegram。",
        flush=True,
    )
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
            symbols = universe(stocks_only=args.stocks)
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = []
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(scan_symbol, s): s for s in symbols}
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as e:
                    print("err", futs[fut], e, flush=True)
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
    print(f"watch 中，每根 {WATCH['signal']} 收盤掃一次（Ctrl+C 停）", flush=True)
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
