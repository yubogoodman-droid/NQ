#!/usr/bin/env python3
"""幣安股票永續 5 分起跌：衝高吐回、回抽失敗，收盤跌破長均線。

對得上 AMDUSDT 2026-11-19 23:10（台北）那根：收 536.07，停損在回抽高 547.24。
後面 23:35 的大陰線是跌勢已經走出來，不在這裡抓。

    python3 examples/watch_binance_breakdown.py --symbol AMDUSDT --backtest 5
    python3 examples/watch_binance_breakdown.py --once
    python3 examples/watch_binance_breakdown.py
    python3 examples/test_watch_binance_breakdown.py

Telegram 選填，沒填就只印在終端。同一根 K 不重發。
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# —— 選填 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "binance_breakdown_seen.json"
BAR_MS = 5 * 60_000

# 衝高：近 3 小時新高，收在上緣，而且真的離開前一段平台
SPIKE_LOOKBACK = 36
SPIKE_CLOSE_POS = 0.70
THRUST_BARS = 12
MIN_THRUST = 0.015
MIN_SPIKE_ATR = 1.2
# 吐回：接下來 2 根內，影線測到高點、收在下緣
REJECT_CLOSE_POS = 0.25
MIN_GIVEBACK = 0.006
# 回抽失敗後 30 分鐘內，第一根收盤跌破 MA99/120/200，收在下緣
BOUNCE_CLOSE_POS = 0.60
MAX_BARS_AFTER_REJECT = 6
TRIGGER_CLOSE_POS = 0.25
MAX_BREAK_PCT = 0.018
MAX_DROP_FROM_HIGH = 0.05
# 回抽高離起跌收盤太近，就不是這種圖上的失敗回抽
MIN_RISK_PCT = 1.2
COOLDOWN_BARS = 12
FORWARD_BARS = 24

_TLS = threading.local()
_RATE_LOCK = threading.Lock()
_PAUSE_UNTIL = 0.0


def get_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
        _TLS.session = s
    return s


def apply_keys() -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        cs = np.cumsum(np.insert(np.asarray(a, dtype=np.float64), 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return sma(tr, n)


def close_pos(h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int) -> float:
    rng = float(h[i] - l[i])
    if rng <= 0:
        return 0.5
    return float((c[i] - l[i]) / rng)


def get_json(path: str, params=None, retries: int = 6):
    global _PAUSE_UNTIL
    last: BaseException = RuntimeError(f"request failed: {path}")
    for i in range(retries):
        wait = _PAUSE_UNTIL - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = get_session().get(BASE + path, params=params, timeout=20)
            used = int(r.headers.get("x-mbx-used-weight-1m") or 0)
            if used > 2150:
                with _RATE_LOCK:
                    nxt = (int(time.time()) // 60 + 1) * 60 + 1.2
                    _PAUSE_UNTIL = max(_PAUSE_UNTIL, nxt)
            if r.status_code in (429, 418):
                ra = float(r.headers.get("Retry-After") or (1.6 * (i + 1)))
                with _RATE_LOCK:
                    _PAUSE_UNTIL = max(_PAUSE_UNTIL, time.time() + ra)
                last = RuntimeError(f"HTTP {r.status_code} on {path}")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.45 * (i + 1))
    raise last


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def bars_from_klines(raw: list) -> dict | None:
    if not raw or len(raw) < 230:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw], np.float64),
        "h": np.array([float(x[2]) for x in raw], np.float64),
        "l": np.array([float(x[3]) for x in raw], np.float64),
        "c": np.array([float(x[4]) for x in raw], np.float64),
    }


def drop_forming(raw: list, now_ms: int | None = None) -> list:
    if not raw:
        return raw
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if int(raw[-1][0]) + BAR_MS > now:
        return raw[:-1]
    return raw


def detect_breakdowns(d: dict) -> list[dict]:
    """5 分起跌。沒有未來 K：信號落在跌破那根的收盤。"""
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    m25 = sma(c, 25)
    m99, m120, m200 = sma(c, 99), sma(c, 120), sma(c, 200)
    a14 = atr(h, l, c, 14)
    out: list[dict] = []
    cool = -1
    for i in range(210, len(c)):
        if i < cool:
            continue
        rib = (m99[i], m120[i], m200[i])
        prev = (m99[i - 1], m120[i - 1], m200[i - 1])
        if np.isnan(rib).any() or np.isnan(prev).any() or np.isnan(m25[i]):
            continue
        rib_lo = float(min(rib))
        prev_lo = float(min(prev))
        # 第一根收盤跌到長均線整條帶的下面，而且收在這根的下緣
        if not (c[i] < rib_lo and c[i - 1] >= prev_lo):
            continue
        if c[i] >= o[i] or c[i] >= m25[i] or close_pos(h, l, c, i) > TRIGGER_CLOSE_POS:
            continue
        break_pct = rib_lo / float(c[i]) - 1.0
        if break_pct > MAX_BREAK_PCT:
            continue

        spike_i = _find_spike(h, l, c, m99, m120, m200, a14, i)
        if spike_i is None:
            continue
        rej_i = _find_rejection(o, h, l, c, spike_i, i)
        if rej_i is None or i - rej_i > MAX_BARS_AFTER_REJECT:
            continue
        if (h[rej_i] - c[rej_i]) / h[rej_i] < MIN_GIVEBACK:
            continue
        # 吐回之後又創高，這次衝高沒有失敗
        if np.any(h[rej_i + 1 : i + 1] > h[rej_i]):
            continue
        bounce_i = _find_bounce(h, l, c, spike_i, rej_i, i)
        if bounce_i is None:
            continue
        peak = float(h[spike_i : i + 1].max())
        if (peak - c[i]) / peak > MAX_DROP_FROM_HIGH:
            continue

        stop = float(h[bounce_i : i + 1].max())
        risk_pct = float((stop / c[i] - 1.0) * 100.0)
        if stop <= c[i] or risk_pct < MIN_RISK_PCT:
            continue
        out.append(
            {
                "i": i,
                "t": int(t[i]),
                "close": float(c[i]),
                "stop": stop,
                "risk_pct": risk_pct,
                "ribbon_lo": rib_lo,
                "break_pct": float(break_pct * 100.0),
                "drop_pct": float((peak - c[i]) / peak * 100.0),
                "spike_i": spike_i,
                "spike_t": int(t[spike_i]),
                "reject_i": rej_i,
                "reject_t": int(t[rej_i]),
                "bounce_i": bounce_i,
                "bounce_t": int(t[bounce_i]),
            }
        )
        cool = i + COOLDOWN_BARS
    return out


def _find_spike(h, l, c, m99, m120, m200, a14, i: int) -> int | None:
    """往前 8 根裡，最近一次收在上緣的 3 小時新高。"""
    start = max(SPIKE_LOOKBACK, i - 8)
    for j in range(i - 1, start - 1, -1):
        ribbon_hi = max(float(m99[j]), float(m120[j]), float(m200[j]))
        if np.isnan(ribbon_hi) or not np.isfinite(a14[j]) or a14[j] <= 0:
            continue
        if close_pos(h, l, c, j) < SPIKE_CLOSE_POS:
            continue
        if h[j] < h[j - SPIKE_LOOKBACK : j].max():
            continue
        if c[j] <= ribbon_hi:
            continue
        if (h[j] - l[j]) / a14[j] < MIN_SPIKE_ATR:
            continue
        base = float(c[j - THRUST_BARS : j].min())
        if base <= 0 or h[j] / base - 1.0 < MIN_THRUST:
            continue
        return j
    return None


def _find_rejection(o, h, l, c, spike_i: int, i: int) -> int | None:
    """衝高後 2 根內，測到高點並收在下緣。"""
    last = min(spike_i + 3, i)
    for j in range(spike_i + 1, last):
        tests_high = h[j] >= h[spike_i] * 0.998
        if tests_high and c[j] < o[j] and c[j] < c[spike_i] and close_pos(h, l, c, j) <= REJECT_CLOSE_POS:
            return j
    return None


def _find_bounce(h, l, c, spike_i: int, rej_i: int, i: int) -> int | None:
    """吐回之後有一根收在上緣的回抽，但沒有回到衝高收盤之上。"""
    for j in range(rej_i + 1, i):
        if close_pos(h, l, c, j) < BOUNCE_CLOSE_POS or c[j] <= c[j - 1]:
            continue
        if c[j] >= c[spike_i]:
            return None
        return j
    return None


def path_after(d: dict, sig: dict, bars: int = FORWARD_BARS) -> dict:
    """信號之後的路徑。做空：先碰到停損高點就記停損，否則看這段的最低價。"""
    i = sig["i"]
    h, l, c = d["h"], d["l"], d["c"]
    end = min(len(c) - 1, i + bars)
    entry = sig["close"]
    stop = sig["stop"]
    for k in range(i + 1, end + 1):
        if h[k] >= stop:
            return {
                "reason": "stop",
                "bars": k - i,
                "mfe_pct": float((entry - l[i + 1 : k + 1].min()) / entry * 100.0),
            }
    if end <= i:
        return {"reason": "open", "bars": 0, "mfe_pct": 0.0}
    low = float(l[i + 1 : end + 1].min())
    return {"reason": "open", "bars": end - i, "mfe_pct": float((entry - low) / entry * 100.0)}


def format_signal(sym: str, sig: dict) -> str:
    return (
        f"{sym} 5分起跌 {hm(sig['t'])}\n"
        f"收盤 {sig['close']:.6g} 跌破長均線 {sig['ribbon_lo']:.6g}（帶下 {sig['break_pct']:.2f}%）\n"
        f"停損 {sig['stop']:.6g}（回抽高）風險 {sig['risk_pct']:.2f}%\n"
        f"衝高 {hm(sig['spike_t'])}  吐回 {hm(sig['reject_t'])}  回抽 {hm(sig['bounce_t'])}"
    )


def universe(equity_only: bool = True) -> list[str]:
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
        if equity_only and s.get("underlyingType") != "EQUITY":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < 5_000_000:
            continue
        out.append(sym)
    return sorted(set(out))


def fetch_klines(sym: str, limit: int = 320) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "5m", "limit": limit})
    return bars_from_klines(drop_forming(raw or []))


def fetch_range(sym: str, start_ms: int, end_ms: int) -> dict | None:
    cur = start_ms - 240 * BAR_MS
    rows: list = []
    while cur < end_ms:
        raw = get_json(
            "/fapi/v1/klines",
            params={"symbol": sym, "interval": "5m", "startTime": cur, "endTime": end_ms - 1, "limit": 1500},
        )
        if not raw:
            break
        rows.extend(raw)
        last = int(raw[-1][0])
        if len(raw) < 1500:
            break
        cur = last + BAR_MS
    seen = {int(x[0]): x for x in rows if int(x[0]) + BAR_MS <= end_ms}
    if len(seen) < 230:
        return None
    kl = [seen[k] for k in sorted(seen)]
    return bars_from_klines(kl)


def latest_signals(sym: str, d: dict) -> list[dict]:
    last = int(d["t"][-1])
    return [s for s in detect_breakdowns(d) if s["t"] == last]


def telegram_send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = get_session().post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


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


def key_of(sym: str, sig: dict) -> str:
    return f"{sym}:{sig['t']}"


def announce(sym: str, sig: dict) -> None:
    text = format_signal(sym, sig)
    print(text, flush=True)
    if telegram_send(text):
        print("  已推 Telegram", flush=True)


def scan_all(symbols: list[str]) -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []

    def one(sym: str):
        d = fetch_klines(sym)
        if d is None:
            return []
        return [(sym, s) for s in latest_signals(sym, d)]

    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, s) for s in symbols]
        for fut in as_completed(futs):
            try:
                found.extend(fut.result())
            except Exception as e:
                print(f"  err {e}", flush=True)
    found.sort(key=lambda x: x[1]["t"])
    return found


def run_backtest(symbols: list[str], days: int) -> int:
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    t0, t1 = int(start.timestamp() * 1000), int(now.timestamp() * 1000)
    print(f"回測 {start.strftime('%Y-%m-%d %H:%M')} ~ {now.strftime('%Y-%m-%d %H:%M')} 台北 | {len(symbols)} 檔", flush=True)
    hits: list[tuple[str, dict, dict]] = []
    done = 0
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fetch_range, s, t0, t1): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                d = fut.result()
            except Exception as e:
                print(f"  err {sym}: {e}", flush=True)
                d = None
            if d is not None:
                for sig in detect_breakdowns(d):
                    if t0 <= sig["t"] < t1:
                        hits.append((sym, sig, path_after(d, sig)))
            if done % 20 == 0 or done == len(symbols):
                print(f"  進度 {done}/{len(symbols)} 信號 {len(hits)}", flush=True)
    hits.sort(key=lambda x: x[1]["t"])
    for sym, sig, path in hits:
        if path["reason"] == "stop":
            tail = f"第 {path['bars']} 根碰到停損，途中往下 {path['mfe_pct']:.2f}%"
        else:
            tail = f"之後 {path['bars']} 根最多往下 {path['mfe_pct']:.2f}%（未碰停損）"
        print(f"{format_signal(sym, sig)}  | {tail}\n", flush=True)
    print(f"共 {len(hits)} 筆", flush=True)
    return 0


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 300 + 1) * 300 + 3
    time.sleep(max(1.0, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("5分起跌監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安 5 分起跌監看")
    p.add_argument("--symbol", action="append", default=[], help="只看這檔，可重複。預設是有量的股票永續")
    p.add_argument("--all", action="store_true", help="股票以外的流動 U 本位永續也掃")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的這根 5 分")
    p.add_argument("--backtest", type=int, default=0, metavar="DAYS", help="回放近 DAYS 天")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    symbols = list(args.symbol) or universe(equity_only=not args.all)
    if args.backtest > 0:
        return run_backtest(symbols, args.backtest)

    seen = load_seen()
    print(f"監看 {len(symbols)} 檔 5 分起跌", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if not args.symbol and time.time() - uni_ts > 1800:
            symbols = universe(equity_only=not args.all)
            uni_ts = time.time()
        t0 = time.time()
        found = scan_all(symbols)
        new = [(sym, sig) for sym, sig in found if key_of(sym, sig) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] 掃完 {len(symbols)} 用 {time.time() - t0:.1f}s  新信號 {len(new)}",
            flush=True,
        )
        for sym, sig in new:
            seen.add(key_of(sym, sig))
            announce(sym, sig)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("每根 5 分收盤掃一次（Ctrl+C 停）", flush=True)
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
