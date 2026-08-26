#!/usr/bin/env python3
"""幣安 15m 壓縮後放量擴張：FIL / SNDK / CRCL 那種連陽噴，以及 PIPPIN 那種墊高。

四張圖的共同骨架（不是 RSI 頂到 96 才算）：
  前面窄幅盤整 → 1.5～7 小時內從波段低點漲 ≥7% → 放量、多數陽線、短均往上、收盤還靠近高點。

用法:
  python3 examples/scan_binance_15m_expansion.py --verify   # 回放四張圖，確認都抓得到
  python3 examples/scan_binance_15m_expansion.py --once     # 掃剛收盤的 15m
  python3 examples/scan_binance_15m_expansion.py            # 每根 15m 收盤掃，可推 Telegram
  python3 examples/scan_binance_15m_expansion.py --test     # 測 Telegram

Telegram 可在檔案最上面填，或放 tg_config.env。
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

# —— 填這裡（也可改放 repo 根目錄 tg_config.env）——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_15m_expansion_seen.json"
CONFIG_ENV = REPO / "tg_config.env"

# 這四檔一定進宇宙，即使 24h 成交額暫時不夠
KEEP = {"FILUSDT", "PIPPINUSDT", "SNDKUSDT", "CRCLUSDT"}

PRE_BARS = 16
IMPULSE_LENS = (6, 8, 10, 12, 16, 20, 24, 28)
MIN_MOVE = 0.07
NEAR_HIGH = 0.965
MAX_PRE_RANGE = 0.085
MIN_EXPAND_RATIO = 2.2
MIN_VOL_RATIO = 1.50
MIN_GREEN_RATIO = 0.55
MIN_MA7_SLOPE = 0.012
MIN_RSI6 = 70.0
MIN_FROM_START = 0.045
MIN_BARS = 80
KLINE_LIMIT = 320
ALERT_BUCKET_MS = 3 * 3600 * 1000  # 同一檔 3 小時內只推一次

# --verify 對齊那四張截圖的時間窗（台北時間）
VERIFY_CASES = [
    {
        "symbol": "FILUSDT",
        "title": "FIL 壓縮後連陽",
        "fetch_start": "2025-12-31 12:00",
        "fetch_end": "2026-01-02 08:00",
        "expect_start": "2026-01-01 21:30",
        "expect_end": "2026-01-02 01:30",
    },
    {
        "symbol": "PIPPINUSDT",
        "title": "PIPPIN 墊高",
        "fetch_start": "2026-01-19 12:00",
        "fetch_end": "2026-01-22 02:00",
        "expect_start": "2026-01-21 17:00",
        "expect_end": "2026-01-21 23:30",
    },
    {
        "symbol": "SNDKUSDT",
        "title": "SNDK 壓縮後噴出",
        "fetch_start": "2026-07-28 12:00",
        "fetch_end": "2026-07-31 08:00",
        "expect_start": "2026-07-30 19:00",
        "expect_end": "2026-07-30 22:30",
    },
    {
        "symbol": "CRCLUSDT",
        "title": "CRCL 壓縮後噴出",
        "fetch_start": "2026-08-17 12:00",
        "fetch_end": "2026-08-20 08:00",
        "expect_start": "2026-08-19 21:00",
        "expect_end": "2026-08-20 00:30",
    },
]


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


def parse_tw(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def rsi_sma(close: np.ndarray, n: int = 6) -> np.ndarray:
    """Binance 圖上那種 SMA RSI（不是 Wilder）。"""
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    for i in range(n, len(close)):
        ag = float(gain[i - n + 1 : i + 1].mean())
        al = float(loss[i - n + 1 : i + 1].mean())
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


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


def fetch_klines(
    sym: str,
    *,
    limit: int = KLINE_LIMIT,
    start_ms: int | None = None,
    end_ms: int | None = None,
    drop_unclosed: bool = True,
    interval: str = "15m",
) -> dict | None:
    params: dict = {"symbol": sym, "interval": interval, "limit": min(limit, 1500)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    raw = get_json("/fapi/v1/klines", params=params)
    if not raw or len(raw) < MIN_BARS:
        return None
    bar_ms = 15 * 60 * 1000 if interval == "15m" else 60_000
    now_ms = int(time.time() * 1000)
    if drop_unclosed and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    if len(raw) < MIN_BARS:
        return None
    return bars_from_raw(raw)


def fetch_klines_range(sym: str, start: datetime, end: datetime) -> dict | None:
    """歷史回放：分段抓，不丟未收盤（那段早已收盤）。"""
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=8), end)
        part = fetch_klines(
            sym,
            limit=1500,
            start_ms=int(cur.timestamp() * 1000),
            end_ms=int(nxt.timestamp() * 1000),
            drop_unclosed=False,
        )
        if part is not None:
            chunks.append(part)
        time.sleep(0.25)
        cur = nxt
    if not chunks:
        return None
    t = np.concatenate([p["t"] for p in chunks])
    _, idx = np.unique(t, return_index=True)
    idx.sort()
    out = {}
    for k in ("t", "o", "h", "l", "c", "v"):
        out[k] = np.concatenate([p[k] for p in chunks])[idx]
    return out if len(out["c"]) >= MIN_BARS else None


def bars_from_raw(raw: list) -> dict:
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def indicators(d: dict) -> dict:
    d = dict(d)
    c = d["c"]
    d["m7"], d["m14"], d["m25"] = sma(c, 7), sma(c, 14), sma(c, 25)
    d["m99"], d["m120"], d["m200"] = sma(c, 99), sma(c, 120), sma(c, 200)
    d["rsi6"] = rsi_sma(c, 6)
    return d


def _hit_at(d: dict, i: int) -> dict | None:
    """單根收盤是否走出壓縮後擴張。垂直連陽與 PIPPIN 墊高共用這組門檻。"""
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
    m7, m25, r6 = d["m7"], d["m25"], d["rsi6"]
    if i < PRE_BARS + IMPULSE_LENS[-1] + 2 or i >= len(c):
        return None
    if np.isnan([m7[i], m25[i], r6[i]]).any():
        return None
    if not (c[i] > m25[i] and m7[i] > m25[i]):
        return None
    if r6[i] < MIN_RSI6:
        return None

    best = None
    for L in IMPULSE_LENS:
        s = i - L
        if s < PRE_BARS + 2:
            continue
        ilow = float(l[s : i + 1].min())
        ihigh = float(h[s : i + 1].max())
        if ilow <= 0:
            continue
        move = float(c[i] / ilow - 1.0)
        if move < MIN_MOVE:
            continue
        if c[i] < ihigh * NEAR_HIGH:
            continue
        if c[s] <= 0 or float(c[i] / c[s] - 1.0) < MIN_FROM_START:
            continue

        pre_h = float(h[s - PRE_BARS : s].max())
        pre_l = float(l[s - PRE_BARS : s].min())
        pre_mid = (pre_h + pre_l) / 2.0
        if pre_mid <= 0:
            continue
        pre_rng = (pre_h - pre_l) / pre_mid
        imp_rng = (ihigh - ilow) / ilow
        if pre_rng > MAX_PRE_RANGE and (pre_rng <= 0 or imp_rng / pre_rng < MIN_EXPAND_RATIO):
            continue

        v_pre = float(v[s - PRE_BARS : s].mean())
        v_imp = float(v[s : i + 1].mean())
        if v_pre <= 0 or v_imp / v_pre < MIN_VOL_RATIO:
            continue

        greens = int(np.sum(c[s : i + 1] >= o[s : i + 1]))
        if greens / L < MIN_GREEN_RATIO:
            continue

        cons = 0
        for k in range(i, s - 1, -1):
            if c[k] >= o[k]:
                cons += 1
            elif k == i:
                continue
            else:
                break

        back = min(L, 8)
        m_back = m7[i - back]
        if np.isnan(m_back) or m_back <= 0 or float(m7[i] / m_back - 1.0) < MIN_MA7_SLOPE:
            continue

        kind = "vertical" if cons >= 5 else "stair"
        score = move * 100.0 + v_imp / v_pre + cons + (2.0 if kind == "vertical" else 0.0)
        rec = {
            "i": i,
            "start_i": s,
            "L": L,
            "kind": kind,
            "move": move,
            "pre_rng": pre_rng,
            "imp_rng": imp_rng,
            "vol_ratio": v_imp / v_pre,
            "green_ratio": greens / L,
            "cons": cons,
            "rsi6": float(r6[i]),
            "ma7": float(m7[i]),
            "ma25": float(m25[i]),
            "close": float(c[i]),
            "score": float(score),
        }
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def detect_expansion(d: dict, *, last_bars: int | None = None) -> list[dict]:
    """回傳所有符合的收盤 K。last_bars=2 時只看剛收的 1～2 根（監看用）。"""
    n = len(d["c"])
    if last_bars is None:
        lo = PRE_BARS + IMPULSE_LENS[-1] + 2
        indices = range(lo, n)
    else:
        indices = [j for j in range(n - last_bars, n) if j >= 0]
    hits = []
    for i in indices:
        hit = _hit_at(d, i)
        if hit:
            hits.append(hit)
    return hits


def collapse_hits(hits: list[dict]) -> list[dict]:
    """連續命中合成一段，留分數最高的那根。"""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h["i"])
    out = [ordered[0]]
    for h in ordered[1:]:
        if h["i"] == out[-1]["i"] + 1:
            if h["score"] > out[-1]["score"]:
                out[-1] = h
        else:
            out.append(h)
    return out


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


def draw_chart(sym: str, d: dict, hit: dict, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    i = hit["i"]
    a0 = max(0, hit["start_i"] - PRE_BARS - 8)
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
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}
    for n, col in pal.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    x = i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([x], [c[x]], s=32, color="#3dba7a", zorder=5)
    x0 = hit["start_i"] - a0
    if 0 <= x0 < len(c):
        ax.axvline(x0, color="#c9a227", ls=":", lw=0.8)
    tag = "連陽噴出" if hit["kind"] == "vertical" else "墊高擴張"
    ax.set_title(f"{sym}  15m  {tag}", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_hit(sym: str, d: dict, hit: dict) -> str:
    ts = hm(int(d["t"][hit["i"]]))
    tag = "連陽噴出" if hit["kind"] == "vertical" else "墊高擴張"
    return (
        f"<b>{tag}</b>  {sym}  15m\n"
        f"時間 {ts}\n"
        f"現價 {hit['close']:g}　波段 {hit['move']*100:.1f}%　{hit['L']} 根\n"
        f"量比 {hit['vol_ratio']:.1f}×　陽線 {hit['green_ratio']*100:.0f}%　連陽 {hit['cons']}\n"
        f"壓縮振幅 {hit['pre_rng']*100:.1f}%　RSI6 {hit['rsi6']:.1f}\n"
        f"MA7 {hit['ma7']:g}　MA25 {hit['ma25']:g}\n"
        f"<i>對齊 FIL / SNDK / CRCL 的壓縮噴出，以及 PIPPIN 那種墊高。</i>"
    )


def key_of(sym: str, d: dict, hit: dict) -> str:
    t = int(d["t"][hit["i"]])
    return f"{sym}:{t // ALERT_BUCKET_MS}"


def scan_symbol(sym: str, *, last_bars: int | None = 2) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    hits = detect_expansion(d, last_bars=last_bars)
    if last_bars is None:
        hits = collapse_hits(hits)
        if hits:
            hits = [max(hits, key=lambda h: h["score"])]
    return [{"symbol": sym, "hit": hit, "d": d} for hit in hits]


def scan_all(symbols: list[str], *, last_bars: int | None = 2) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, last_bars=last_bars): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (-e["hit"]["score"], e["symbol"]))
    return events


def notify(ev: dict) -> None:
    text = format_hit(ev["symbol"], ev["d"], ev["hit"])
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("&gt;", ">")
    )
    print("\n" + plain, flush=True)
    tmp = Path("/tmp") / f"exp15_{ev['symbol']}_{ev['hit']['i']}.png"
    photo = draw_chart(ev["symbol"], ev["d"], ev["hit"], str(tmp))
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
    nxt = (int(now) // 900 + 1) * 900 + 3
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("15m 壓縮擴張監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def verify_four() -> int:
    """回放使用者那四張 15m 圖，四檔都要在對應時間窗內命中。"""
    print("回放四張 15m 圖…", flush=True)
    failed = []
    for case in VERIFY_CASES:
        sym = case["symbol"]
        d0 = fetch_klines_range(sym, parse_tw(case["fetch_start"]), parse_tw(case["fetch_end"]))
        if d0 is None:
            print(f"FAIL  {sym}  抓不到 K 線", flush=True)
            failed.append(sym)
            continue
        d = indicators(d0)
        hits = collapse_hits(detect_expansion(d))
        a = int(parse_tw(case["expect_start"]).timestamp() * 1000)
        b = int(parse_tw(case["expect_end"]).timestamp() * 1000)
        matched = [h for h in hits if a <= int(d["t"][h["i"]]) <= b]
        if matched:
            h = max(matched, key=lambda x: x["score"])
            print(
                f"PASS  {sym}  {case['title']}  "
                f"{hm(int(d['t'][h['i']]))}  {h['kind']}  "
                f"+{h['move']*100:.1f}%  {h['L']}根  量比{h['vol_ratio']:.1f}×  RSI6={h['rsi6']:.0f}",
                flush=True,
            )
        else:
            print(f"FAIL  {sym}  {case['title']}  時間窗內沒抓到", flush=True)
            if hits:
                last = hits[-1]
                print(f"      最近一次 {hm(int(d['t'][last['i']]))} +{last['move']*100:.1f}%", flush=True)
            failed.append(sym)
        time.sleep(0.2)
    if failed:
        print("沒過：", ", ".join(failed))
        return 1
    print("四張都抓到了。")
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安 15m 壓縮後放量擴張（FIL/PIPPIN/SNDK/CRCL）")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的 15m，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--verify", action="store_true", help="回放四張截圖，確認都抓得到")
    p.add_argument("--symbols", default="", help="逗號分隔，例如 FILUSDT,PIPPINUSDT")
    p.add_argument("--full", action="store_true", help="掃整段 K 而不只最後 2 根（找正在噴的）")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.verify:
        return verify_four()

    seen = load_seen()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"指定 {len(symbols)} 檔：{', '.join(symbols)}", flush=True)
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"監看 {len(symbols)} 個流動永續（含 TRADIFI）。", flush=True)
    uni_ts = time.time()
    last_bars = None if args.full else 2

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if not args.symbols.strip() and time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, last_bars=last_bars)
        new = [e for e in events if key_of(e["symbol"], e["d"], e["hit"]) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev["symbol"], ev["d"], ev["hit"]))
            notify(ev)
        if new:
            save_seen(seen)

    round_once()
    if args.once or args.full:
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


if __name__ == "__main__":
    raise SystemExit(main())
