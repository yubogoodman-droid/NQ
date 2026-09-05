#!/usr/bin/env python3
"""幣安 1 小時 K 多頭爆發：均線多頭排列 + 收盤量大於前一根一倍以上，推 Telegram。

條件（剛收盤的 1h K）：
  MA7 > MA14 > MA25 > MA99 > MA120 > MA200
  收盤成交量 > 前一根 × 2

用法:
  python3 examples/watch_binance_1h_burst.py --test     # 測 Telegram
  python3 examples/watch_binance_1h_burst.py --once     # 只掃剛收盤那根
  python3 examples/watch_binance_1h_burst.py            # 每根 1h 收盤掃一次

Telegram 憑證放 tg_config.env（勿提交），或在下面填：
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
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

# —— 也可直接填這裡 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
INTERVAL = "1h"
INTERVAL_MS = 3_600_000
MA_PERIODS = (7, 14, 25, 99, 120, 200)
VOL_MULT = 2.0
MIN_QUOTE_VOL = 5_000_000
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}

REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_1h_burst_seen.json"
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})


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
        if qv < MIN_QUOTE_VOL and sym not in KEEP:
            continue
        out.append(sym)
    return out


def drop_unclosed(raw: list, now_ms: int | None = None) -> list:
    if not raw:
        return raw
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if int(raw[-1][0]) + INTERVAL_MS > now_ms:
        return raw[:-1]
    return raw


def bars_from_raw(raw: list) -> dict | None:
    if not raw or len(raw) < MA_PERIODS[-1] + 2:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def fetch_klines(sym: str, limit: int = 260) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": INTERVAL, "limit": limit})
    return bars_from_raw(drop_unclosed(raw))


def indicators(d: dict) -> dict:
    c = d["c"]
    out = dict(d)
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    return out


def ma_stack(d: dict, i: int) -> tuple[float, ...] | None:
    vals = tuple(float(d[f"m{n}"][i]) for n in MA_PERIODS)
    if np.isnan(vals).any():
        return None
    return vals


def is_bull_align(mas: tuple[float, ...]) -> bool:
    return all(mas[k] > mas[k + 1] for k in range(len(mas) - 1))


def burst_at(d: dict, i: int, vol_mult: float = VOL_MULT) -> dict | None:
    """剛收盤的第 i 根是否符合多頭爆發。"""
    if i < 1 or i >= len(d["c"]):
        return None
    mas = ma_stack(d, i)
    if mas is None or not is_bull_align(mas):
        return None
    prev_v = float(d["v"][i - 1])
    cur_v = float(d["v"][i])
    if prev_v <= 0 or not (cur_v > prev_v * vol_mult):
        return None
    return {
        "i": i,
        "mas": mas,
        "vol": cur_v,
        "prev_vol": prev_v,
        "vol_ratio": cur_v / prev_v,
        "close": float(d["c"][i]),
        "open": float(d["o"][i]),
        "high": float(d["h"][i]),
        "low": float(d["l"][i]),
    }


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
    a0 = max(0, i - 72)
    a1 = min(len(d["c"]), i + 2)
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
    x = i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([x], [c[x]], s=36, color="#3dba7a", zorder=5)
        axv.axvline(x, color="#3dba7a", ls="--", lw=0.9)
    ax.set_title(f"{sym}  1h  多頭爆發", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def scan_symbol(sym: str, lookback: int = 2, vol_mult: float = VOL_MULT) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    last = len(d["c"]) - 1
    events = []
    start = max(MA_PERIODS[-1], last - lookback + 1)
    for i in range(start, last + 1):
        hit = burst_at(d, i, vol_mult=vol_mult)
        if not hit:
            continue
        events.append({"symbol": sym, "d": d, **hit})
    return events


def format_burst(ev: dict) -> str:
    d = ev["d"]
    ts = hm(int(d["t"][ev["i"]]))
    mas = ev["mas"]
    ma_txt = " &gt; ".join(f"MA{n} {mas[k]:g}" for k, n in enumerate(MA_PERIODS))
    side = "陽線" if ev["close"] >= ev["open"] else "陰線"
    return (
        f"🚀 <b>1h 多頭爆發</b>  {ev['symbol']}\n"
        f"收盤 {ts}（台北）  {side}\n"
        f"現價 {ev['close']:g}　OHLC {ev['open']:g} / {ev['high']:g} / {ev['low']:g} / {ev['close']:g}\n"
        f"成交量 {ev['vol']:.4g}　前一根 {ev['prev_vol']:.4g}　放大 {ev['vol_ratio']:.2f}×\n"
        f"多頭排列  {ma_txt}"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['i']])}"


def scan_all(symbols: list[str], lookback: int, vol_mult: float) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, lookback, vol_mult): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (-e["vol_ratio"], e["symbol"]))
    return events


def notify(ev: dict, dry_run: bool = False) -> None:
    text = format_burst(ev)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("&gt;", ">")
        .replace("🚀 ", "")
    )
    print("\n" + plain, flush=True)
    if dry_run:
        print("  → dry-run，不送 Telegram", flush=True)
        return
    tmp = Path("/tmp") / f"burst1h_{ev['symbol']}_{ev['i']}.png"
    photo = draw_chart(ev["symbol"], ev["d"], ev["i"], str(tmp))
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送", flush=True)
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗，檢查 token 與 chat id", flush=True)


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 3600 + 1) * 3600 + 3
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("1h 多頭爆發監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安 1h 多頭爆發 Telegram 監看")
    p.add_argument("--once", action="store_true", help="只掃一次然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--dry-run", action="store_true", help="掃到也不送 Telegram")
    p.add_argument("--lookback", type=int, default=2, help="往回看幾根已收盤 1h（預設 2）")
    p.add_argument("--vol-mult", type=float, default=VOL_MULT, help="成交量倍數門檻（預設 2 = 大於前一根一倍）")
    p.add_argument("--symbols", nargs="*", help="只掃這些代號，例如 BTCUSDT ETHUSDT")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = list(args.symbols) if args.symbols else universe()
    print(
        f"監看 {len(symbols)} 個 1h：MA7>14>25>99>120>200 且量 > 前一根 × {args.vol_mult:g}",
        flush=True,
    )
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if not args.symbols and time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, lookback=max(1, args.lookback), vol_mult=args.vol_mult)
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
    print("watch 中，每根 1h 收盤掃一次（Ctrl+C 停）", flush=True)
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
