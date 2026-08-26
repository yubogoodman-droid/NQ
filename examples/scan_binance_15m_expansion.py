#!/usr/bin/env python3
"""幣安 15m 壓縮後放量擴張：FIL / SNDK / CRCL 那種連陽噴，以及 PIPPIN 那種墊高。

四張圖的共同骨架（不是 RSI 頂到 96 才算）：
  前面窄幅盤整 → 1.5～7 小時內從波段低點漲 ≥7% → 放量、多數陽線、短均往上、收盤還靠近高點。

用法:
  python3 examples/scan_binance_15m_expansion.py --verify   # 回放四張圖，確認都抓得到
  python3 examples/scan_binance_15m_expansion.py --backtest --days 7 --pages
  python3 examples/scan_binance_15m_expansion.py --once     # 掃剛收盤的 15m
  python3 examples/scan_binance_15m_expansion.py            # 每根 15m 收盤掃，可推 Telegram
  python3 examples/scan_binance_15m_expansion.py --test     # 測 Telegram

Telegram 可在檔案最上面填，或放 tg_config.env。
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
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
HOLD_BARS = 8          # 最多抱 2 小時
TARGET_R = 1.5
MAX_RISK = 0.03        # 單筆風險上限 3%
MIN_RISK = 0.004
FWD_BARS = (1, 4, 8, 16)  # 15m / 1h / 2h / 4h
PAGES_HTML = REPO / "docs" / "binance" / "expansion-15m-7d" / "index.html"

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


def select_alerts(d: dict, start_ms: int, end_ms: int) -> list[dict]:
    """跟 Telegram 監看一樣：時間序第一根命中，同一檔 3 小時內只留一筆。"""
    seen: set[int] = set()
    out: list[dict] = []
    for h in sorted(detect_expansion(d), key=lambda x: x["i"]):
        t = int(d["t"][h["i"]])
        if t < start_ms or t > end_ms:
            continue
        bucket = t // ALERT_BUCKET_MS
        if bucket in seen:
            continue
        seen.add(bucket)
        out.append(h)
    return out


def _fwd_pct(d: dict, entry_i: int, entry: float, bars: int) -> float | None:
    j = entry_i + bars
    if j >= len(d["c"]) or entry <= 0:
        return None
    return float(d["c"][j] / entry - 1.0)


def simulate_trade(d: dict, hit: dict) -> dict | None:
    """訊號收盤後下一根開盤做多。停損在訊號 K 低點（風險夾在 0.4%～3%），1.5R 或 8 根時間出場。"""
    i = hit["i"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    if i + 1 >= len(c):
        return None
    entry_i = i + 1
    entry = float(o[entry_i])
    if entry <= 0:
        return None
    raw_stop = float(l[i])
    if raw_stop >= entry:
        raw_stop = entry * (1.0 - MIN_RISK)
    risk_pct = (entry - raw_stop) / entry
    if risk_pct > MAX_RISK:
        stop = entry * (1.0 - MAX_RISK)
    elif risk_pct < MIN_RISK:
        stop = entry * (1.0 - MIN_RISK)
    else:
        stop = raw_stop
    risk = entry - stop
    if risk <= 0:
        return None
    target = entry + TARGET_R * risk
    last = min(entry_i + HOLD_BARS, len(c) - 1)
    exit_i = last
    exit_px = float(c[last])
    reason = "eod" if last < entry_i + HOLD_BARS else "time"
    mfe = mae = 0.0
    for k in range(entry_i, last + 1):
        mfe = max(mfe, float(h[k]) / entry - 1.0)
        mae = min(mae, float(l[k]) / entry - 1.0)
        if float(l[k]) <= stop:
            exit_i, exit_px, reason = k, stop, "stop"
            break
        if float(h[k]) >= target:
            exit_i, exit_px, reason = k, target, "target"
            break
    pnl_pct = (exit_px / entry - 1.0) * 100.0
    fwd = {n: _fwd_pct(d, entry_i, entry, n) for n in FWD_BARS}
    return {
        "signal_i": i,
        "entry_i": entry_i,
        "exit_i": exit_i,
        "entry": entry,
        "stop": stop,
        "target": target,
        "exit": exit_px,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "r_mult": (exit_px - entry) / risk,
        "mfe_pct": mfe * 100.0,
        "mae_pct": mae * 100.0,
        "fwd": fwd,
        "kind": hit["kind"],
        "move": hit["move"],
        "vol_ratio": hit["vol_ratio"],
        "rsi6": hit["rsi6"],
        "L": hit["L"],
        "score": hit["score"],
        "t_signal": int(d["t"][i]),
        "t_entry": int(d["t"][entry_i]),
        "t_exit": int(d["t"][exit_i]),
    }


def summarize_trades(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "count": 0,
            "wins": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
            "by_kind": {},
            "by_reason": {},
            "fwd": {},
        }
    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    by_kind: dict[str, dict] = {}
    for t in trades:
        k = t["kind"]
        slot = by_kind.setdefault(k, {"n": 0, "wins": 0, "pnl": 0.0})
        slot["n"] += 1
        slot["pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            slot["wins"] += 1
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1
    fwd_stats = {}
    for nbar in FWD_BARS:
        vals = [t["fwd"][nbar] * 100.0 for t in trades if t["fwd"].get(nbar) is not None]
        if not vals:
            fwd_stats[nbar] = {"n": 0, "win_rate": 0.0, "avg": 0.0}
        else:
            fwd_stats[nbar] = {
                "n": len(vals),
                "win_rate": 100.0 * sum(1 for v in vals if v > 0) / len(vals),
                "avg": float(np.mean(vals)),
            }
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n,
        "pnl": float(sum(pnls)),
        "avg": float(np.mean(pnls)),
        "mfe": float(np.mean([t["mfe_pct"] for t in trades])),
        "mae": float(np.mean([t["mae_pct"] for t in trades])),
        "by_kind": by_kind,
        "by_reason": by_reason,
        "fwd": fwd_stats,
    }


def _equity_svg(pnls: list[float], width: int = 720, height: int = 160) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys.min()), float(ys.max())
    pad = max(0.4, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys[i]):.1f}" for i in range(len(ys)))
    zero = yv(0.0)
    color = "#3dba7a" if eq[-1] >= 0 else "#e35d5d"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f1714;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#2a3a33" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def draw_trade_b64(sym: str, d: dict, tr: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    i = tr["signal_i"]
    a0 = max(0, i - 36)
    a1 = min(len(d["c"]), tr["exit_i"] + 6)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
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
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.0, label=f"MA{n}")
    ax.axhline(tr["entry"], color="#e8f0ea", ls=":", lw=0.7)
    ax.axhline(tr["stop"], color="#e35d5d", ls="--", lw=0.7)
    ax.axhline(tr["target"], color="#3dba7a", ls="--", lw=0.7)
    for idx, color, mark in (
        (tr["signal_i"], "#c9a227", "o"),
        (tr["entry_i"], "#3dba7a", "^"),
        (tr["exit_i"], "#e35d5d" if tr["pnl_pct"] < 0 else "#3dba7a", "x"),
    ):
        x = idx - a0
        if 0 <= x < len(c):
            ax.axvline(x, color=color, ls="--", lw=0.7)
            ax.scatter([x], [c[x] if mark != "^" else tr["entry"]], s=28, color=color, marker=mark, zorder=6)
    tag = "連陽" if tr["kind"] == "vertical" else "墊高"
    ax.set_title(
        f"{sym}  15m  {tag}  {tr['reason']}  {tr['pnl_pct']:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.45)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def backtest_symbol(sym: str, start_ms: int, end_ms: int, warmup_days: int = 3) -> tuple[str, dict | None, list[dict]]:
    fetch_start = datetime.fromtimestamp(start_ms / 1000, TZ) - timedelta(days=warmup_days)
    d0 = fetch_klines(
        sym,
        limit=1500,
        start_ms=int(fetch_start.timestamp() * 1000),
        end_ms=end_ms + 15 * 60 * 1000,
        drop_unclosed=True,
    )
    if d0 is None:
        return sym, None, []
    d = indicators(d0)
    trades = []
    for hit in select_alerts(d, start_ms, end_ms):
        tr = simulate_trade(d, hit)
        if tr is None:
            continue
        tr["symbol"] = sym
        trades.append(tr)
    return sym, d, trades


def run_backtest(symbols: list[str], days: int = 7) -> tuple[list[dict], dict[str, dict]]:
    end = datetime.now(TZ)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    print(
        f"回測 {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}  "
        f"{len(symbols)} 檔 15m",
        flush=True,
    )
    trades: list[dict] = []
    data: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(backtest_symbol, s, start_ms, end_ms): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, rows = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            if d is not None:
                data[sym] = d
            trades.extend(rows)
            if done % 25 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)}  訊號 {len(trades)}", flush=True)
    trades.sort(key=lambda t: t["t_signal"])
    return trades, data


def write_backtest_html(
    path: Path,
    trades: list[dict],
    data: dict[str, dict],
    *,
    days: int,
    n_symbols: int,
    chart_limit: int = 60,
) -> Path:
    stats = summarize_trades(trades)
    start = datetime.fromtimestamp(trades[0]["t_signal"] / 1000, TZ).strftime("%Y-%m-%d %H:%M") if trades else ""
    end = datetime.fromtimestamp(trades[-1]["t_signal"] / 1000, TZ).strftime("%Y-%m-%d %H:%M") if trades else ""
    total_cls = "pnl-win" if stats["pnl"] >= 0 else "pnl-loss"
    kind_line = " · ".join(
        f"{('連陽' if k == 'vertical' else '墊高')} {v['n']}筆 勝率 {100*v['wins']/v['n']:.0f}% {v['pnl']:+.1f}%"
        for k, v in stats["by_kind"].items()
        if v["n"]
    )
    reason_line = " · ".join(f"{k} {n}" for k, n in stats["by_reason"].items())
    fwd_bits = []
    labels = {1: "15m", 4: "1h", 8: "2h", 16: "4h"}
    for nbar, lab in labels.items():
        fs = stats["fwd"].get(nbar) or {}
        if fs.get("n"):
            fwd_bits.append(f"{lab} 勝率 {fs['win_rate']:.0f}% 均 {fs['avg']:+.2f}%")
    fwd_line = " · ".join(fwd_bits)

    ranked = sorted(trades, key=lambda t: abs(t["pnl_pct"]), reverse=True)
    chart_set = {id(t) for t in ranked[:chart_limit]}
    cards = []
    for i, t in enumerate(trades, 1):
        cls = "pnl-win" if t["pnl_pct"] > 0 else ("pnl-flat" if t["pnl_pct"] == 0 else "pnl-loss")
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t["reason"], "tag-time")
        tag = "連陽噴出" if t["kind"] == "vertical" else "墊高擴張"
        img = ""
        if id(t) in chart_set:
            d = data.get(t["symbol"])
            b64 = draw_trade_b64(t["symbol"], d, t) if d else None
            if b64:
                img = (
                    f"<div class='mini-chart'><img src='data:image/png;base64,{b64}' "
                    f"alt='#{i} {escape(t['symbol'])}' style='width:100%;display:block;border-radius:10px'/></div>"
                )
        h1 = (t["fwd"].get(4) or 0) * 100
        h2 = (t["fwd"].get(8) or 0) * 100
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(t['symbol'])}</span>"
            f"<span class='trade-time'>{escape(hm(t['t_entry']))} → {escape(hm(t['t_exit']))}</span></div>"
            f"<div class='card-pnl {cls}'>{t['pnl_pct']:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t['reason'])}</span>"
            f"<span class='tag tag-info'>{escape(tag)}</span>"
            f"<span class='tag tag-info'>{t['L']}根</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t['entry']:g}  stop {t['stop']:g}  target {t['target']:g}\n"
            f"exit  {t['exit']:g}  {t['reason']}  {t['r_mult']:+.2f}R\n"
            f"波段已走 {t['move']*100:.1f}%  量比 {t['vol_ratio']:.1f}×  RSI6 {t['rsi6']:.0f}\n"
            f"MFE {t['mfe_pct']:+.2f}%  MAE {t['mae_pct']:+.2f}%\n"
            f"無停損 1h {h1:+.2f}%  2h {h2:+.2f}%"
            "</pre>"
            f"{img}"
            "</article>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 15m 壓縮擴張 · 近 {days} 天</title>
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
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15m 壓縮後放量擴張</h1>
<p class="muted">近 {days} 天 · {escape(start)} → {escape(end)} · 掃 {n_symbols} 檔永續。訊號出現時波段已走 ≥7%，這是<strong>追價做多</strong>：下一根開盤進，停損在訊號 K 低點（風險上限 3%），1.5R 或 2 小時平。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總報酬<b class="{total_cls}">{stats['pnl']:+.1f}%</b></div>
<div class="card">均筆<b class="{total_cls}">{stats['avg']:+.2f}%</b></div>
</div>
<div class="equity">{_equity_svg([t['pnl_pct'] for t in trades])}</div>
<p class="muted">MFE 均 {stats['mfe']:+.2f}% · MAE 均 {stats['mae']:+.2f}%</p>
<p class="muted">{escape(kind_line) if kind_line else '無分組'}</p>
<p class="muted">出場 {escape(reason_line) if reason_line else '—'}</p>
<p class="muted">無停損續走 {escape(fwd_line) if fwd_line else '—'}</p>
<p class="muted">等權重每筆 1 單位，不含手續費／資金費。圖只畫 |報酬| 最大的 {min(chart_limit, stats['count'])} 筆。</p>
</section>
{''.join(cards) if cards else "<div class='empty'>這週沒有訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    view = path.parent / "view.html"
    if path.name == "index.html":
        view.write_text(html, encoding="utf-8")
    return path


def cmd_backtest(args) -> int:
    days = int(args.days)
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"指定 {len(symbols)} 檔", flush=True)
    else:
        print("載入標的…", flush=True)
        symbols = universe()
        print(f"{len(symbols)} 個流動永續", flush=True)
    trades, data = run_backtest(symbols, days=days)
    stats = summarize_trades(trades)
    print(
        f"trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['pnl']:+.1f}% avg={stats['avg']:+.2f}%"
    )
    for k, v in stats["by_kind"].items():
        wr = 100 * v["wins"] / v["n"] if v["n"] else 0
        print(f"  {k}: n={v['n']} WR={wr:.1f}% pnl={v['pnl']:+.1f}%")
    for nbar, lab in ((1, "15m"), (4, "1h"), (8, "2h"), (16, "4h")):
        fs = stats["fwd"].get(nbar) or {}
        if fs.get("n"):
            print(f"  fwd {lab}: n={fs['n']} WR={fs['win_rate']:.1f}% avg={fs['avg']:+.2f}%")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] {t['symbol']} {hm(t['t_entry'])} {t['kind']} "
            f"{t['reason']} {t['pnl_pct']:+.2f}%"
        )
    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_backtest_html(Path(html_path), trades, data, days=days, n_symbols=len(symbols))
        print(f"html={out}")
    return 0


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
    p.add_argument("--backtest", action="store_true", help="回測近 N 天（預設 7）")
    p.add_argument("--days", type=int, default=7, help="回測天數")
    p.add_argument("--html", default="", help="回測 HTML 路徑")
    p.add_argument("--pages", action="store_true", help="回測寫入 docs/binance/expansion-15m-7d/")
    p.add_argument("--symbols", default="", help="逗號分隔，例如 FILUSDT,PIPPINUSDT")
    p.add_argument("--full", action="store_true", help="掃整段 K 而不只最後 2 根（找正在噴的）")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.verify:
        return verify_four()
    if args.backtest:
        return cmd_backtest(args)

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
