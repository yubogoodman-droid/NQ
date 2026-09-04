#!/usr/bin/env python3
"""幣安 15m 從 MA200 爆量擴張：ETH 2026-09-03 22:45 那根要進。

  記號：價格先在 MA200 附近盤整，然後一根放量長陽（ETH 22:45 那種）。
  進場：擴張棒收完，下一根開盤做多。
  出場：收盤跌破 MA200，或 4 小時到期。

用法:
  python3 examples/scan_binance_15m_expansion.py --verify
  python3 examples/scan_binance_15m_expansion.py --backtest --days 7 --pages
  python3 examples/scan_binance_15m_expansion.py --once
  python3 examples/scan_binance_15m_expansion.py
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

# 基準圖一定進宇宙
KEEP = {"ETHUSDT", "FILUSDT", "PIPPINUSDT", "SNDKUSDT", "CRCLUSDT"}

PRE_BARS = 12
CONFIRM_BARS = 0         # 擴張棒本身就是訊號，下一根開盤進
MIN_VOL_RATIO = 2.50     # ETH 22:45 約 3.7×；21:30 那根較早但不是基準圖
MIN_MARK_VOL_VS_PREV = 1.0
MIN_BODY = 0.007         # ETH 22:45 +1.32%；FIL 19:45 +0.77%；ETH 21:30 +0.62% 不夠
MIN_RANGE_ATR = 2.50     # ETH 22:45 約 3.5×；21:30 約 2.38× 不夠
MIN_CLOSE_POS = 0.78     # ETH 22:45 0.98；FIL 0.79
LOOKBACK_BREAK = 20      # 收盤創近 20 根新高
MAX_MARK_EXT = 0.05      # 允許 ETH 22:45 那種離開 200 約 +3.45% 的長陽
NEAR_200_LOOKBACK = 16   # 近 4 小時內必須曾經貼過／跌破 200（從 200 起漲，不是天上掉下來）
NEAR_200_BAND = 0.01
MAX_PRIOR_ATR_PCT = 0.012
SQUEEZE_LOOKBACK = 24
MAX_SQUEEZE_RIBBON = 0.05
MIN_STACK_7_14 = 0.001
MIN_STACK_7_25 = 0.004
SESSION_HOURS = range(0, 24)
# ETH 22:45 當晚會有一堆標的一起噴，不能整批丟掉。
MAX_SAME_MARK = 99
CLUSTER_COOLDOWN_MS = 3 * 3600 * 1000
MIN_BARS = 220
KLINE_LIMIT = 500
BAR_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
ALERT_BUCKET_MS = 8 * 3600 * 1000
HOLD_BARS = 16         # 最多抱 4 小時
TARGET_R = 2.0
MAX_RISK = 0.02
MIN_RISK = 0.004
FWD_BARS = (1, 4, 8, 16)
PAGES_HTML = REPO / "docs" / "binance" / "expansion-15m-7d" / "index.html"

# --verify：ETH 那張 22:45 長陽一定要進
VERIFY_CASES = [
    {
        "symbol": "ETHUSDT",
        "title": "ETH 22:45 爆量擴張（要進）",
        "fetch_start": "2026-08-28 00:00",
        "fetch_end": "2026-09-04 08:00",
        "expect_start": "2026-09-03 22:30",
        "expect_end": "2026-09-03 23:15",
    },
    {
        "symbol": "FILUSDT",
        "title": "FIL 站上 MA200",
        "fetch_start": "2025-12-28 00:00",
        "fetch_end": "2026-01-02 08:00",
        "expect_start": "2026-01-01 19:30",
        "expect_end": "2026-01-01 20:15",
    },
    {
        "symbol": "CRCLUSDT",
        "title": "CRCL 站上 MA200",
        "fetch_start": "2026-08-15 00:00",
        "fetch_end": "2026-08-20 08:00",
        "expect_start": "2026-08-19 20:15",
        "expect_end": "2026-08-19 21:00",
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
    min_bars: int | None = None,
) -> dict | None:
    params: dict = {"symbol": sym, "interval": interval, "limit": min(limit, 1500)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    raw = get_json("/fapi/v1/klines", params=params)
    need = MIN_BARS if min_bars is None else int(min_bars)
    if not raw or len(raw) < need:
        return None
    bar_ms = BAR_MS.get(interval, 900_000)
    now_ms = int(time.time() * 1000)
    if drop_unclosed and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    if len(raw) < need:
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


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev = np.empty_like(c)
    prev[0] = c[0]
    prev[1:] = c[:-1]
    return np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))


def indicators(d: dict) -> dict:
    d = dict(d)
    c, v = d["c"], d["v"]
    d["m7"], d["m14"], d["m25"] = sma(c, 7), sma(c, 14), sma(c, 25)
    d["m99"], d["m120"], d["m200"] = sma(c, 99), sma(c, 120), sma(c, 200)
    d["rsi6"] = rsi_sma(c, 6)
    d["v20"] = sma(v, 20)
    d["tr"] = true_range(d["h"], d["l"], c)
    d["atr20"] = sma(d["tr"], 20)
    return d


def _hit_at(d: dict, i: int) -> dict | None:
    """i 是從 MA200 附近打出來的擴張棒（ETH 22:45 那種）。"""
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    v20, r6, atr20 = d["v20"], d["rsi6"], d["atr20"]
    need = max(200, LOOKBACK_BREAK, SQUEEZE_LOOKBACK, 21)
    if i < need or i >= len(c):
        return None
    vals = [m7[i], m14[i], m25[i], m99[i], m120[i], m200[i], v20[i], atr20[i - 1]]
    if np.isnan(vals).any():
        return None
    hour = datetime.fromtimestamp(int(d["t"][i]) / 1000, TZ).hour
    if hour not in SESSION_HOURS:
        return None
    if c[i] <= o[i]:
        return None
    bar_rng = float(h[i] - l[i])
    if bar_rng <= 0:
        return None
    close_pos = float((c[i] - l[i]) / bar_rng)
    if close_pos < MIN_CLOSE_POS:
        return None
    body = float(c[i] / o[i] - 1.0)
    if body < MIN_BODY:
        return None
    atr_prev = float(atr20[i - 1])
    if atr_prev <= 0:
        return None
    range_atr = bar_rng / atr_prev
    if range_atr < MIN_RANGE_ATR:
        return None
    if float(c[i - 1]) <= 0 or atr_prev / float(c[i - 1]) > MAX_PRIOR_ATR_PCT:
        return None
    vr = float(v[i] / v20[i]) if v20[i] > 0 else 0.0
    if vr < MIN_VOL_RATIO:
        return None
    prev_vol = float(v[i - 1])
    if prev_vol > 0 and float(v[i]) < prev_vol * MIN_MARK_VOL_VS_PREV:
        return None
    if float(c[i]) <= float(np.max(h[i - LOOKBACK_BREAK : i])):
        return None
    if not (m7[i] > m14[i] > m25[i]):
        return None
    if float(m7[i] / m14[i] - 1.0) < MIN_STACK_7_14:
        return None
    if float(m7[i] / m25[i] - 1.0) < MIN_STACK_7_25:
        return None
    stack_hi = max(
        float(m7[i]), float(m14[i]), float(m25[i]), float(m99[i]), float(m120[i]), float(m200[i])
    )
    if float(c[i]) <= stack_hi:
        return None
    ribbons = []
    for j in range(i - SQUEEZE_LOOKBACK, i):
        xs = [m7[j], m14[j], m25[j], m99[j], m120[j], m200[j]]
        if np.isnan(xs).any() or min(xs) <= 0:
            continue
        ribbons.append(float(max(xs) / min(xs) - 1.0))
    if not ribbons or min(ribbons) > MAX_SQUEEZE_RIBBON:
        return None
    if float(c[i]) <= float(m200[i]):
        return None
    ext = float(c[i] / m200[i] - 1.0)
    if ext > MAX_MARK_EXT:
        return None
    near_200 = False
    lo = max(0, i - NEAR_200_LOOKBACK)
    for j in range(lo, i):
        if np.isnan(m200[j]) or float(m200[j]) <= 0:
            continue
        if float(c[j]) <= float(m200[j]) * (1.0 + NEAR_200_BAND):
            near_200 = True
            break
    if not near_200:
        return None
    ribbon = float(max(m99[i], m120[i], m200[i]) / min(m99[i], m120[i], m200[i]) - 1.0)
    return {
        "i": i,
        "mark_i": i,
        "start_i": i,
        "L": 1,
        "kind": "expand",
        "move": body,
        "pre_rng": min(ribbons),
        "imp_rng": body,
        "vol_ratio": vr,
        "green_ratio": 1.0,
        "cons": 0,
        "rsi6": float(r6[i]) if not np.isnan(r6[i]) else 0.0,
        "ma7": float(m7[i]),
        "ma14": float(m14[i]),
        "ma25": float(m25[i]),
        "ma200": float(m200[i]),
        "close": float(c[i]),
        "ext": ext,
        "mark_ext": ext,
        "ribbon": ribbon,
        "hour": hour,
        "stop_low": float(m200[i]),
        "body": body,
        "range_atr": float(range_atr),
        "close_pos": close_pos,
        "score": float(vr + max(0.0, MAX_MARK_EXT - ext) * 200.0),
    }


def detect_expansion(d: dict, *, last_bars: int | None = None) -> list[dict]:
    """回傳剛收完的擴張棒。last_bars=2 時只看剛收的 1～2 根。"""
    n = len(d["c"])
    lo = 200 + max(CONFIRM_BARS, 0)
    if last_bars is None:
        indices = range(lo, n)
    else:
        indices = [j for j in range(n - last_bars, n) if j >= lo]
    hits = []
    for i in indices:
        hit = _hit_at(d, i)
        if hit:
            hits.append(hit)
    return hits


def item_mark_ms(item: dict) -> int:
    if "t_mark" in item:
        return int(item["t_mark"])
    hit = item["hit"]
    mark_i = int(hit.get("mark_i", hit["i"] - CONFIRM_BARS))
    return int(item["d"]["t"][mark_i])


def crowded_mark_times(items: list[dict], extra: list[int] | None = None) -> list[int]:
    counts: dict[int, int] = {}
    for x in items:
        t = item_mark_ms(x)
        counts[t] = counts.get(t, 0) + 1
    crowded = [t for t, n in counts.items() if n >= MAX_SAME_MARK]
    if extra:
        crowded.extend(int(t) for t in extra)
    return sorted(set(crowded))


def drop_market_cluster(items: list[dict], extra_crowded: list[int] | None = None) -> list[dict]:
    """同一根 15m 記號太多檔就整批拿掉，之後 CLUSTER_COOLDOWN_MS 內的跟風也拿掉。"""
    crowded = crowded_mark_times(items, extra_crowded)
    if not crowded:
        return items
    kept = []
    for x in items:
        t = item_mark_ms(x)
        if any(c <= t <= c + CLUSTER_COOLDOWN_MS for c in crowded):
            continue
        kept.append(x)
    return kept


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


CJK_FONTS = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def use_cjk_font(plt) -> None:
    """幣安有中文名的合約（幣安人生、龍蝦），標題要能畫出來。"""
    import matplotlib.font_manager as fm

    for path in CJK_FONTS:
        if not Path(path).exists():
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return


def draw_chart(sym: str, d: dict, hit: dict, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    use_cjk_font(plt)
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
    x0 = hit.get("mark_i", hit["start_i"]) - a0
    if 0 <= x0 < len(c):
        ax.axvline(x0, color="#c9a227", ls=":", lw=0.9)
        ax.scatter([x0], [c[x0]], s=28, color="#c9a227", marker="D", zorder=5)
    tag = "MA200 附近擴張"
    ax.set_title(f"{sym}  15m  {tag}", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_hit(sym: str, d: dict, hit: dict) -> str:
    mark_ts = hm(int(d["t"][hit.get("mark_i", hit["i"])]))
    return (
        f"<b>15m MA200 附近擴張</b>  {sym}\n"
        f"記號 {mark_ts}　離 200 {hit.get('mark_ext', 0)*100:+.2f}%　"
        f"實體 {hit.get('body', hit.get('move', 0))*100:+.2f}%　量比 {hit['vol_ratio']:.1f}×\n"
        f"現價 {hit['close']:g}　MA200 {hit.get('ma200', 0):g}　{hit.get('hour', 0):02d} 點\n"
        f"MA7 {hit['ma7']:g} &gt; MA14 {hit.get('ma14', 0):g} &gt; MA25 {hit['ma25']:g}\n"
        f"<i>從 MA200 附近盤整後放量長陽（ETH 22:45 那種要進）；下一根開盤做多，收盤破 200 出場。</i>"
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


def scan_all(
    symbols: list[str],
    *,
    last_bars: int | None = 2,
    extra_crowded: list[int] | None = None,
    crowded_out: list[int] | None = None,
) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, last_bars=last_bars): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (-e["hit"]["score"], e["symbol"]))
    if crowded_out is not None:
        crowded_out.extend(crowded_mark_times(events))
    return drop_market_cluster(events, extra_crowded=extra_crowded)


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
    ok = telegram_send("15m MA200 附近擴張監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def select_alerts(d: dict, start_ms: int, end_ms: int) -> list[dict]:
    """跟 Telegram 監看一樣：時間序第一根命中，同一檔 8 小時內只留一筆。"""
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
    """記號收完下一根開盤做多。收盤跌破 MA200 出場；最多 HOLD_BARS 根。"""
    i = hit["i"]
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    m200 = d["m200"]
    if i + 1 >= len(c):
        return None
    entry_i = i + 1
    entry = float(o[entry_i])
    if entry <= 0:
        return None
    stop = float(hit.get("stop_low", m200[i]))
    last = min(entry_i + HOLD_BARS, len(c) - 1)
    exit_i = last
    exit_px = float(c[last])
    reason = "eod" if last < entry_i + HOLD_BARS else "time"
    mfe = mae = 0.0
    for k in range(entry_i, last + 1):
        mfe = max(mfe, float(h[k]) / entry - 1.0)
        mae = min(mae, float(l[k]) / entry - 1.0)
        if not np.isnan(m200[k]) and float(c[k]) < float(m200[k]):
            exit_i, exit_px, reason = k, float(c[k]), "ma_break"
            break
    pnl_pct = (exit_px / entry - 1.0) * 100.0
    risk = max(entry - stop, entry * MIN_RISK, 1e-12)
    fwd = {n: _fwd_pct(d, entry_i, entry, n) for n in FWD_BARS}
    return {
        "signal_i": i,
        "entry_i": entry_i,
        "exit_i": exit_i,
        "entry": entry,
        "stop": stop,
        "target": entry * 1.0,
        "exit": exit_px,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "r_mult": (exit_px - entry) / risk,
        "mfe_pct": mfe * 100.0,
        "mae_pct": mae * 100.0,
        "fwd": fwd,
        "kind": hit["kind"],
        "move": hit["move"],
        "ext": hit.get("ext", hit["move"]),
        "mark_ext": hit.get("mark_ext", 0.0),
        "vol_ratio": hit["vol_ratio"],
        "rsi6": hit["rsi6"],
        "L": hit["L"],
        "score": hit["score"],
        "pre_rng": hit.get("pre_rng", hit.get("ribbon", 0.0)),
        "ribbon": hit.get("ribbon", hit.get("pre_rng", 0.0)),
        "ma200": hit.get("ma200", 0.0),
        "hour": hit.get("hour", 0),
        "body": hit.get("body", hit.get("move", 0.0)),
        "range_atr": hit.get("range_atr", 0.0),
        "close_pos": hit.get("close_pos", 0.0),
        "mark_i": hit.get("mark_i", i),
        "t_mark": int(d["t"][hit.get("mark_i", i)]),
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


def bar_index_at(d: dict, ts_ms: int) -> int | None:
    """時間戳落在哪一根 K（開盤時間 ≤ ts 的最後一根）。"""
    t = d["t"]
    if len(t) == 0:
        return None
    i = int(np.searchsorted(t, int(ts_ms), side="right") - 1)
    if i < 0 or i >= len(t):
        return None
    return i


def _render_ohlc_b64(
    sym: str,
    d: dict,
    tr: dict,
    *,
    title: str,
    a0: int,
    a1: int,
    mark_i: int,
    signal_i: int,
    entry_i: int,
    exit_i: int,
) -> str | None:
    try:
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    if a1 - a0 < 4:
        return None
    use_cjk_font(plt)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(9.4, 5.0), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
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
    marks = [(mark_i, "#c9a227", "D")]
    if signal_i != mark_i:
        marks.append((signal_i, "#8ab4f8", "o"))
    marks.append((entry_i, "#3dba7a", "^"))
    marks.append((exit_i, "#e35d5d" if tr["pnl_pct"] < 0 else "#3dba7a", "x"))
    for idx, color, mark in marks:
        x = idx - a0
        if 0 <= x < len(c):
            ax.axvline(x, color=color, ls="--", lw=0.7)
            ax.scatter([x], [c[x] if mark != "^" else tr["entry"]], s=28, color=color, marker=mark, zorder=6)
    ax.set_title(title, color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.45)
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_trade_b64(sym: str, d: dict, tr: dict) -> str | None:
    i = tr["signal_i"]
    a0 = max(0, i - 36)
    a1 = min(len(d["c"]), tr["exit_i"] + 6)
    return _render_ohlc_b64(
        sym,
        d,
        tr,
        title=f"{sym}  15m  MA200附近擴張  {tr['reason']}  {tr['pnl_pct']:+.2f}%",
        a0=a0,
        a1=a1,
        mark_i=int(tr.get("mark_i", tr["signal_i"])),
        signal_i=int(tr["signal_i"]),
        entry_i=int(tr["entry_i"]),
        exit_i=int(tr["exit_i"]),
    )


def draw_1h_b64(sym: str, d1h: dict, tr: dict) -> str | None:
    mark_i = bar_index_at(d1h, tr["t_mark"])
    signal_i = bar_index_at(d1h, tr["t_signal"])
    entry_i = bar_index_at(d1h, tr["t_entry"])
    exit_i = bar_index_at(d1h, tr["t_exit"])
    if mark_i is None or exit_i is None:
        return None
    if signal_i is None:
        signal_i = mark_i
    if entry_i is None:
        entry_i = signal_i
    a0 = max(0, mark_i - 48)
    a1 = min(len(d1h["c"]), exit_i + 10)
    return _render_ohlc_b64(
        sym,
        d1h,
        tr,
        title=f"{sym}  1h  對照  {tr['reason']}  {tr['pnl_pct']:+.2f}%",
        a0=a0,
        a1=a1,
        mark_i=mark_i,
        signal_i=signal_i,
        entry_i=entry_i,
        exit_i=exit_i,
    )


def fetch_1h_for_trades(trades: list[dict]) -> dict[str, dict]:
    """只抓有成交的標的小時 K，給報告對照圖。"""
    if not trades:
        return {}
    by_sym: dict[str, list[dict]] = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)

    def one(sym: str) -> tuple[str, dict | None]:
        rows = by_sym[sym]
        t0 = min(int(r["t_mark"]) for r in rows) - 16 * 24 * 3600 * 1000
        t1 = max(int(r["t_exit"]) for r in rows) + 18 * 3600 * 1000
        d0 = fetch_klines(
            sym,
            limit=1500,
            start_ms=t0,
            end_ms=t1,
            interval="1h",
            drop_unclosed=True,
            min_bars=80,
        )
        if d0 is None:
            return sym, None
        return sym, indicators(d0)

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(one, s) for s in by_sym]
        for fut in as_completed(futs):
            try:
                sym, d = fut.result()
            except Exception as e:
                print("1h err", e, flush=True)
                continue
            if d is not None:
                out[sym] = d
    print(f"小時 K {len(out)}/{len(by_sym)} 檔", flush=True)
    return out


def backtest_symbol(sym: str, start_ms: int, end_ms: int, warmup_days: int = 4) -> tuple[str, dict | None, list[dict]]:
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
    before = len(trades)
    trades = drop_market_cluster(trades)
    if before != len(trades):
        print(f"大盤集群濾掉 {before - len(trades)} 筆（同一根記號 ≥{MAX_SAME_MARK} 檔）", flush=True)
    return trades, data


def write_backtest_html(
    path: Path,
    trades: list[dict],
    data: dict[str, dict],
    *,
    days: int,
    n_symbols: int,
    chart_limit: int | None = None,
) -> Path:
    stats = summarize_trades(trades)
    start = datetime.fromtimestamp(trades[0]["t_signal"] / 1000, TZ).strftime("%Y-%m-%d %H:%M") if trades else ""
    end = datetime.fromtimestamp(trades[-1]["t_signal"] / 1000, TZ).strftime("%Y-%m-%d %H:%M") if trades else ""
    total_cls = "pnl-win" if stats["pnl"] >= 0 else "pnl-loss"
    kind_line = " · ".join(
        f"{'擴張' if k == 'expand' else k} {v['n']}筆 勝率 {100*v['wins']/v['n']:.0f}% {v['pnl']:+.1f}%"
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

    if chart_limit is None:
        chart_set = {id(t) for t in trades}
    else:
        ranked = sorted(trades, key=lambda t: abs(t["pnl_pct"]), reverse=True)
        chart_set = {id(t) for t in ranked[:chart_limit]}
    print(f"畫 {len(chart_set)} 張圖…", flush=True)
    data_1h = fetch_1h_for_trades([t for t in trades if id(t) in chart_set])
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, t in enumerate(trades, 1):
        cls = "pnl-win" if t["pnl_pct"] > 0 else ("pnl-flat" if t["pnl_pct"] == 0 else "pnl-loss")
        reason_cls = {"broke_low": "tag-sl", "ma_break": "tag-sl", "time": "tag-time", "eod": "tag-time"}.get(t["reason"], "tag-info")
        tag = "MA200 附近擴張"
        img = ""
        if id(t) in chart_set:
            d = data.get(t["symbol"])
            b64 = draw_trade_b64(t["symbol"], d, t) if d else None
            if b64:
                png_name = f"{i:03d}.png"
                (img_dir / png_name).write_bytes(base64.b64decode(b64))
                img += (
                    f"<div class='mini-chart'><div class='chart-label'>15m</div>"
                    f"<img src='img/{png_name}' loading='lazy' "
                    f"alt='#{i} {escape(t['symbol'])} 15m' style='width:100%;display:block'/></div>"
                )
            d1h = data_1h.get(t["symbol"])
            b64h = draw_1h_b64(t["symbol"], d1h, t) if d1h else None
            if b64h:
                png_1h = f"{i:03d}-1h.png"
                (img_dir / png_1h).write_bytes(base64.b64decode(b64h))
                img += (
                    f"<div class='mini-chart'><div class='chart-label'>1h 對照</div>"
                    f"<img src='img/{png_1h}' loading='lazy' "
                    f"alt='#{i} {escape(t['symbol'])} 1h' style='width:100%;display:block'/></div>"
                )
            if i % 40 == 0:
                print(f"  圖 {i}/{len(trades)}", flush=True)
            if i == 1 and not img:
                print("畫圖失敗（檢查有沒有裝 matplotlib）", flush=True)
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
            f"<span class='tag tag-info'>記號 {escape(hm(t.get('t_mark', t['t_signal'])))}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t['entry']:g}  MA200 {t['stop']:g}\n"
            f"exit  {t['exit']:g}  {t['reason']}\n"
            f"離200 {t.get('mark_ext', 0)*100:+.2f}%  實體 {t.get('body', t.get('move', 0))*100:+.2f}%  {t.get('range_atr', 0):.1f}×ATR  量比 {t['vol_ratio']:.1f}×\n"
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
<title>幣安 15m MA200 附近擴張 · 近 {days} 天</title>
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
.mini-chart{{margin:0 -6px 8px;border-radius:10px;overflow:hidden;background:#0c1210}}
.mini-chart:last-child{{margin-bottom:-4px}}
.chart-label{{font-size:11px;font-weight:600;color:#8b949e;padding:8px 12px 0}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15m MA200 附近擴張</h1>
<p class="muted">近 {days} 天 · {escape(start)} → {escape(end)} · 掃 {n_symbols} 檔永續。基準是 ETH 09-03 <strong>22:45 那根長陽要進</strong>：先在 MA200 附近盤整，再放量長陽（實體 ≥{MIN_BODY:.1%}、振幅 ≥{MIN_RANGE_ATR:.1f}×ATR、量比 ≥{MIN_VOL_RATIO:.1f}×、收在棒子上方 {MIN_CLOSE_POS:.0%}、創 20 根新高）。離 200 可以到 {MAX_MARK_EXT:.0%}（ETH 那根約 +3.5%）。下一根開盤做多，<strong>收盤跌破 MA200</strong> 出場，最多 4 小時。</p>
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
<p class="muted">等權重每筆 1 單位，不含手續費／資金費。下面 {stats['count']} 筆都有圖：上面 15m、下面 1h 對照。黃菱形＝在 200 附近的擴張棒、綠三角＝進場、× ＝出場。</p>
</section>
{''.join(cards) if cards else "<div class='empty'>這週沒有訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    view = path.with_name("view.html")
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
    if getattr(args, "pages", False) or not html_path:
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_backtest_html(Path(html_path), trades, data, days=days, n_symbols=len(symbols))
        print(f"html={out}")
    return 0


def verify_four() -> int:
    """回放 ETH 那張 15m 擴張圖，時間窗內要命中。"""
    print("回放 MA200 附近擴張圖…", flush=True)
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
            entry_i = h["i"] + 1
            entry_s = hm(int(d["t"][entry_i])) if entry_i < len(d["t"]) else "—"
            print(
                f"PASS  {sym}  {case['title']}  "
                f"擴張 {hm(int(d['t'][h['mark_i']]))} 實體 {h.get('body', 0)*100:+.2f}%  "
                f"{h.get('range_atr', 0):.1f}×ATR  量 {h['vol_ratio']:.1f}×  "
                f"進場 {entry_s}",
                flush=True,
            )
        else:
            print(f"FAIL  {sym}  {case['title']}  時間窗內沒抓到", flush=True)
            if hits:
                last = hits[-1]
                print(
                    f"      最近一次 擴張 {hm(int(d['t'][last['mark_i']]))} "
                    f"實體 {last.get('body', 0)*100:+.2f}%  量 {last['vol_ratio']:.1f}×",
                    flush=True,
                )
            failed.append(sym)
        time.sleep(0.2)
    if failed:
        print("沒過：", ", ".join(failed))
        return 1
    print("都在 MA200 附近抓到了。")
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安 15m 在 MA200 附近的爆量擴張")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的 15m，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--verify", action="store_true", help="回放 ETH 22:45，確認那根會進")
    p.add_argument("--backtest", action="store_true", help="回測近 N 天（預設 7）")
    p.add_argument("--days", type=int, default=7, help="回測天數")
    p.add_argument("--html", default="", help="回測 HTML 路徑")
    p.add_argument("--pages", action="store_true", help="回測寫入 docs/binance/expansion-15m-7d/")
    p.add_argument("--symbols", default="", help="逗號分隔，例如 ETHUSDT,BTCUSDT")
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
    known_crowded: list[int] = []
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
        found: list[int] = []
        events = scan_all(
            symbols,
            last_bars=last_bars,
            extra_crowded=known_crowded,
            crowded_out=found,
        )
        now_ms = int(time.time() * 1000)
        known_crowded[:] = [
            t for t in known_crowded + found if t >= now_ms - CLUSTER_COOLDOWN_MS
        ]
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
