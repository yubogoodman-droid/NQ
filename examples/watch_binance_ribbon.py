#!/usr/bin/env python3
"""幣安 1m 黏帶三幕（優化版）：過濾假突破離開推 Telegram（附圖與進出場價）；完全符合再推強訊號。

在下面填 Telegram 後執行：

    python3 examples/watch_binance_ribbon.py          # 每根 1m 收盤監看（預設優化版濾網）
    python3 examples/watch_binance_ribbon.py --raw    # 舊版未過濾離開全推
    python3 examples/watch_binance_ribbon.py --once   # 只掃剛收盤這一分
    python3 examples/watch_binance_ribbon.py --test   # 測試 Telegram 通不通
    python3 examples/watch_binance_ribbon.py --backtest 7   # 回測近 N 天（原始 vs 優化版）

Ctrl+C 結束。同一根 K 不會重發。
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

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "binance_ribbon_seen.json"
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}
FEE_PCT = 0.08  # 幣安合約來回手續費估 0.08%

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
    return sorted(set(out))


def fetch_klines(sym: str, limit: int = 420) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": limit})
    if not raw or len(raw) < 250:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if len(raw) < 250:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw], np.float64),
        "h": np.array([float(x[2]) for x in raw], np.float64),
        "l": np.array([float(x[3]) for x in raw], np.float64),
        "c": np.array([float(x[4]) for x in raw], np.float64),
        "v": np.array([float(x[5]) for x in raw], np.float64),
    }


def indicators(d: dict) -> dict:
    c, v = d["c"], d["v"]
    d = dict(d)
    d["m7"], d["m14"], d["m25"] = sma(c, 7), sma(c, 14), sma(c, 25)
    d["m99"], d["m120"], d["m200"] = sma(c, 99), sma(c, 120), sma(c, 200)
    d["v20"] = sma(v, 20)
    return d


def btc_market_state() -> dict:
    """抓取 BTCUSDT 1m 狀態，確認大盤不在短線急殺（close > MA25）。"""
    try:
        raw = fetch_klines("BTCUSDT", limit=80)
        if raw is None or len(raw["c"]) < 30:
            r2 = get_json("/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "1m", "limit": 60})
            now_ms = int(time.time() * 1000)
            if r2 and int(r2[-1][0]) + 60_000 > now_ms:
                r2 = r2[:-1]
            c = np.array([float(x[4]) for x in r2], np.float64)
            t = np.array([int(x[0]) for x in r2], np.int64)
        else:
            c, t = raw["c"], raw["t"]
        m25 = sma(c, 25)
        return {"t": t, "c": c, "m25": m25, "idx": {int(ts): i for i, ts in enumerate(t)}}
    except Exception:
        return {"t": np.array([], np.int64), "c": np.array([]), "m25": np.array([]), "idx": {}}


def btc_above_ma25_at(btc_state: dict, ts_ms: int) -> bool:
    if len(btc_state.get("t", [])) == 0:
        return True
    idx = btc_state["idx"].get(int(ts_ms))
    if idx is None:
        pos = int(np.searchsorted(btc_state["t"], ts_ms, side="right") - 1)
        if pos < 0 or pos >= len(btc_state["t"]):
            return True
        idx = pos
    if np.isnan(btc_state["m25"][idx]):
        return True
    return bool(btc_state["c"][idx] > btc_state["m25"][idx])


def kiss_at(d: dict, i: int) -> dict | None:
    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    m99, m120, m200 = d["m99"], d["m120"], d["m200"]
    c, h, l = d["c"], d["h"], d["l"]
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
    for j in range(i - 1, max(0, i - 160), -1):
        if np.isnan(m200[j]) or c[j] >= m200[j]:
            break
        below += 1
    if below < 40:
        return None
    lo_40 = float(l[i - 40 : i + 1].min())
    bounce = (c[i] / lo_40 - 1) * 100
    if not (0.70 <= bounce <= 1.35):
        return None
    hi_120 = float(h[max(0, i - 120) : i + 1].max())
    drop_120 = (hi_120 - lo_40) / hi_120 * 100
    return {
        "i": i,
        "cl": float(cl),
        "s200": float(s200),
        "ext": float(ext),
        "bounce": float(bounce),
        "below": below,
        "lo_40": lo_40,
        "drop_120": float(drop_120),
    }


def valid_kisses(d: dict, start_i: int = 210, end_i: int | None = None) -> dict[int, dict]:
    """帶 50 根冷卻期的吻索引表，與回測一致。"""
    stop = len(d["c"]) if end_i is None else min(end_i, len(d["c"]))
    out: dict[int, dict] = {}
    last = -999
    for i in range(max(210, start_i), stop):
        if i - last < 50:
            continue
        k = kiss_at(d, i)
        if k is not None:
            out[i] = k
            last = i
    return out


def lift_at(d: dict, kiss_i: int, j: int) -> bool:
    if j - kiss_i < 4 or j - kiss_i > 15:
        return False
    m99, m120, m200, v20 = d["m99"], d["m120"], d["m200"], d["v20"]
    if np.isnan([m99[j], m120[j], m200[j], v20[j]]).any() or v20[j] <= 0:
        return False
    cl2 = (max(m99[j], m120[j], m200[j]) / min(m99[j], m120[j], m200[j]) - 1) * 100
    ext2 = (d["c"][j] / m200[j] - 1) * 100
    vr = d["v"][j] / v20[j]
    return bool(cl2 <= 0.20 and ext2 >= 0.40 and vr >= 1.8)


def opt_lift_plan(d: dict, kiss: dict, j: int, *, btc_ok: bool = True) -> dict | None:
    """零未來函數優化濾網 + 過高觸價／圓底停損／60m 出場計畫。"""
    kiss_i = kiss["i"]
    if not lift_at(d, kiss_i, j):
        return None
    h, l, c, v = d["h"], d["l"], d["c"], d["v"]
    m99, m120, m200, v20 = d["m99"], d["m120"], d["m200"], d["v20"]

    rng = max(float(h[j] - l[j]), 1e-12)
    close_pos = float((c[j] - l[j]) / rng)
    brk_kiss_h = bool(c[j] >= float(np.max(h[kiss_i:j])))
    vr = float(v[j] / v20[j])
    s200_lift = float((m200[j] / m200[j - 30] - 1) * 100)
    ds200 = s200_lift - kiss["s200"]

    if close_pos < 0.70 or not brk_kiss_h:
        return None
    if kiss["drop_120"] < 1.2:
        return None
    if ds200 <= 0 or vr > 6.0:
        return None
    if not btc_ok:
        return None

    trig_px = float(h[j]) * 1.0001
    rib_floor = float(min(m99[j], m120[j], m200[j]))
    raw_risk = (trig_px - kiss["lo_40"]) / trig_px * 100.0 + 0.05
    risk_pct = min(max(raw_risk, 0.50), 1.60)
    sl_px = trig_px * (1.0 - risk_pct / 100.0)
    cl2 = float((max(m99[j], m120[j], m200[j]) / min(m99[j], m120[j], m200[j]) - 1) * 100)
    ext2 = float((c[j] / m200[j] - 1) * 100)
    return {
        "close_pos": close_pos,
        "vr": vr,
        "cl2": cl2,
        "ext2": ext2,
        "s200_lift": s200_lift,
        "ds200": ds200,
        "trig_px": trig_px,
        "rib_floor": rib_floor,
        "sl_px": sl_px,
        "risk_pct": risk_pct,
        "hold_min": 60,
    }


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
    sess = get_session()
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = sess.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": text[:1024],
                        "parse_mode": "HTML",
                        "disable_notification": "false" if strong else "true",
                    },
                    files={"photo": f},
                    timeout=25,
                )
            if r.ok:
                return True
        r = sess.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
                "disable_notification": not strong,
            },
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


def _cjk_font():
    try:
        from matplotlib.font_manager import FontProperties

        for fp in (
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ):
            if os.path.exists(fp):
                return FontProperties(fname=fp)
    except Exception:
        pass
    return None


def draw_chart(sym: str, d: dict, kiss_i: int, lift_i: int, path: str, plan: dict | None = None) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    font = _cjk_font()
    a0 = max(0, kiss_i - 65)
    a1 = min(len(d["c"]), max(lift_i + 15, kiss_i + 35))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    times = [datetime.fromtimestamp(int(ts) / 1000, TZ) for ts in d["t"][sl]]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.8, 5.9), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
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
            ax.scatter([x], [c[x]], s=34, color=color, zorder=5)
            kw = {"fontproperties": font} if font else {}
            ax.annotate(label, xy=(x, c[x]), xytext=(5, 8), textcoords="offset points", color=color, fontsize=8.5, **kw)
    if plan is not None:
        lx = max(0, lift_i - a0)
        rx = len(c) - 1
        ax.hlines(plan["trig_px"], lx, rx, colors="#f59e0b", linestyles="--", lw=0.9, label="過高進場")
        ax.hlines(plan["sl_px"], lx, rx, colors="#e35d5d", linestyles=":", lw=1.0, label="圓底停損")
    ax.set_title(f"{sym}  1m", color="#e8f0ea", fontsize=12)
    leg_kw = {"prop": font} if font else {"fontsize": 7.5}
    ax.legend(loc="upper left", frameon=False, labelcolor="#c8d5cc", ncol=8, **leg_kw)
    step = max(1, len(xs) // 7)
    ticks = list(range(0, len(xs), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([times[k].strftime("%H:%M") for k in ticks], color="#8aa193", fontsize=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=115, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def scan_symbol(sym: str, *, btc_state: dict | None = None, raw_mode: bool = False) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    n = len(d["c"])
    last = n - 1
    kisses = valid_kisses(d, start_i=max(210, last - 80), end_i=n)
    events = []
    for closed in (last, last - 1):
        if closed < 220:
            continue
        for kiss_i in range(closed - 15, closed - 3):
            kiss = kisses.get(kiss_i)
            if not kiss:
                continue
            if not lift_at(d, kiss_i, closed):
                continue
            first = next((j for j in range(kiss_i + 4, closed + 1) if lift_at(d, kiss_i, j)), None)
            if first != closed:
                continue
            btc_ok = btc_above_ma25_at(btc_state or {}, int(d["t"][closed]))
            plan = opt_lift_plan(d, kiss, closed, btc_ok=btc_ok)
            if not raw_mode and plan is None:
                continue
            events.append(
                {
                    "kind": "opt_lift" if plan is not None else "lift",
                    "symbol": sym,
                    "kiss": kiss,
                    "kiss_i": kiss_i,
                    "lift_i": closed,
                    "plan": plan,
                    "d": d,
                }
            )
        kiss_i = closed - 29
        kiss = kisses.get(kiss_i)
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
    d, kiss, sym, plan = ev["d"], ev["kiss"], ev["symbol"], ev.get("plan")
    ks = hm(int(d["t"][ev["kiss_i"]]))
    ls = hm(int(d["t"][ev["lift_i"]]))
    px = d["c"][ev["lift_i"]]
    if plan is not None:
        return (
            f"✅ <b>優化離開</b>  <b>{sym}</b>  1m\n"
            f"吻 {ks} → 離開 {ls}（量 {plan['vr']:.1f}x · 收高 {plan['close_pos']*100:.0f}%）\n"
            f"收盤 {px:g}　前2h跌幅 {kiss['drop_120']:.2f}%　BTC&gt;MA25\n"
            f"• 觸價進場：突破 <b>{plan['trig_px']:.6g}</b>（4分鐘內有效，先跌破黏帶 {plan['rib_floor']:.6g} 取消）\n"
            f"• 圓底停損：<b>{plan['sl_px']:.6g}</b>（-{plan['risk_pct']:.2f}%）\n"
            f"• 出場規則：不設緊停利，成交後抱滿 <b>{plan['hold_min']}m</b> 收盤出場"
        )
    return (
        f"<b>離開（未過濾）</b>  {sym}\n"
        f"吻 {ks} → 離開 {ls}\n"
        f"現價 {px:g}\n"
        f"長均黏度 {kiss['cl']:.3f}%　回彈 {kiss['bounce']:.2f}%"
    )


def format_look(ev: dict) -> str:
    d, kiss, look, sym = ev["d"], ev["kiss"], ev["look"], ev["symbol"]
    ks = hm(int(d["t"][ev["kiss_i"]]))
    ls = hm(int(d["t"][ev["lift_i"]]))
    px = d["c"][ev["kiss_i"] + 29]
    return (
        f"🔥🔥🔥 <b>完全符合（樣貌確認）</b>\n"
        f"<b>{sym}</b>  1m\n"
        f"吻 {ks} → 離開 {ls}\n"
        f"現價 {px:g}\n"
        f"長均黏度 {kiss['cl']:.3f}% → {look['cl29']:.3f}%\n"
        f"價離開帶子 {look['ext29']:+.2f}%　短均散開 {look['short29']:.2f}%\n"
        f"<i>註：吻後29m樣貌已走開，適合觀摩或追蹤，勿在此追高。</i>"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['kiss_i']])}:{ev['kind']}"


def scan_all(symbols: list[str], *, raw_mode: bool = False) -> list[dict]:
    btc_state = btc_market_state()
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, btc_state=btc_state, raw_mode=raw_mode): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    order = {"look": 0, "opt_lift": 1, "lift": 2}
    events.sort(key=lambda e: (order.get(e["kind"], 9), e["symbol"]))
    return events


def notify(ev: dict) -> None:
    is_look = ev["kind"] == "look"
    is_opt = ev["kind"] == "opt_lift"
    strong = is_look or is_opt
    text = format_look(ev) if is_look else format_lift(ev)
    clean = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("&gt;", ">")
    )
    print("\n" + clean, flush=True)
    photo = None
    if strong:
        tmp = Path("/tmp") / f"ribbon_{ev['symbol']}_{ev['kiss_i']}_{ev['kind']}.png"
        photo = draw_chart(ev["symbol"], ev["d"], ev["kiss_i"], ev["lift_i"], str(tmp), plan=ev.get("plan"))
    if is_look:
        telegram_send("🔥🔥🔥 完全符合（樣貌確認） " + ev["symbol"], strong=True)
    ok = telegram_send(text, strong=strong, photo=photo)
    if ok:
        print("  → Telegram 已送")
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
        else:
            print("  → Telegram 送出失敗，檢查 token 與 chat id")


def simulate_opt_series(d: dict, btc_state: dict, t0_ms: int, t1_ms: int) -> tuple[list[float], list[dict]]:
    """回測單一標的：回傳 (原始離開30m扣費報酬, 優化版交易清單)。"""
    d = indicators(d)
    t, o, h, l, c = d["t"], d["o"], d["h"], d["l"], d["c"]
    n = len(c)
    kisses = valid_kisses(d, start_i=250, end_i=n - 65)
    raw_r30: list[float] = []
    opt_trades: list[dict] = []
    for kiss_i, kiss in kisses.items():
        if t[kiss_i] < t0_ms or t[kiss_i] >= t1_ms:
            continue
        lift_j = next((j for j in range(kiss_i + 4, min(kiss_i + 16, n - 61)) if lift_at(d, kiss_i, j)), None)
        if lift_j is None or t[lift_j] >= t1_ms:
            continue
        raw_r30.append(float((c[lift_j + 30] / c[lift_j] - 1.0) * 100.0 - FEE_PCT))
        btc_ok = btc_above_ma25_at(btc_state, int(t[lift_j]))
        plan = opt_lift_plan(d, kiss, lift_j, btc_ok=btc_ok)
        if plan is None:
            continue
        # 4 分鐘內過高觸價進場（先跌破黏帶則取消）
        fill_i = None
        epx = None
        for w in range(1, 5):
            if lift_j + w >= n - 60:
                break
            if l[lift_j + w] < plan["rib_floor"]:
                break
            if h[lift_j + w] >= plan["trig_px"]:
                fill_i = lift_j + w
                epx = float(max(o[lift_j + w], plan["trig_px"]))
                break
        if fill_i is None or epx is None:
            continue
        raw_risk = (epx - kiss["lo_40"]) / epx * 100.0 + 0.05
        risk_pct = min(max(raw_risk, 0.50), 1.60)
        sl_px = epx * (1.0 - risk_pct / 100.0)
        end_i = min(fill_i + 60, n - 1)
        gross = float((c[end_i] / epx - 1.0) * 100.0)
        reason = "60m"
        for k in range(fill_i, end_i + 1):
            if l[k] <= sl_px:
                gross = -risk_pct
                reason = "SL"
                break
        opt_trades.append(
            {
                "kiss_ms": int(t[kiss_i]),
                "lift_ms": int(t[lift_j]),
                "gross": gross,
                "net": gross - FEE_PCT,
                "reason": reason,
            }
        )
    return raw_r30, opt_trades


def fetch_range(sym: str, start_ms: int, end_ms: int) -> dict | None:
    cur = start_ms - 300 * 60_000
    rows = []
    while cur < end_ms:
        raw = get_json(
            "/fapi/v1/klines",
            params={"symbol": sym, "interval": "1m", "startTime": cur, "endTime": end_ms - 1, "limit": 1500},
        )
        if not raw:
            break
        rows.extend(raw)
        last = int(raw[-1][0])
        if len(raw) < 1500:
            break
        cur = last + 60_000
    seen = {int(x[0]): x for x in rows if int(x[0]) + 60_000 <= end_ms}
    if len(seen) < 320:
        return None
    kl = [seen[k] for k in sorted(seen)]
    return {
        "t": np.array([int(x[0]) for x in kl], np.int64),
        "o": np.array([float(x[1]) for x in kl], np.float64),
        "h": np.array([float(x[2]) for x in kl], np.float64),
        "l": np.array([float(x[3]) for x in kl], np.float64),
        "c": np.array([float(x[4]) for x in kl], np.float64),
        "v": np.array([float(x[5]) for x in kl], np.float64),
    }


def run_backtest_cli(days: int) -> int:
    now_dt = datetime.now(TZ).replace(second=0, microsecond=0)
    start_dt = (now_dt - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    t0_ms, t1_ms = int(start_dt.timestamp() * 1000), int(now_dt.timestamp() * 1000)
    symbols = universe()
    print(f"回測區間 {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {now_dt.strftime('%Y-%m-%d %H:%M')} GMT+8 | 標的 {len(symbols)}", flush=True)
    btc_raw = fetch_range("BTCUSDT", t0_ms, t1_ms)
    btc_state = (
        {"t": btc_raw["t"], "c": btc_raw["c"], "m25": sma(btc_raw["c"], 25), "idx": {int(ts): i for i, ts in enumerate(btc_raw["t"])}}
        if btc_raw is not None
        else {}
    )
    raw_all: list[float] = []
    opt_all: list[dict] = []
    done = 0
    with ThreadPoolExecutor(10) as ex:
        futs = {ex.submit(fetch_range, s, t0_ms, t1_ms): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                d = fut.result()
                if d is not None:
                    r30, otr = simulate_opt_series(d, btc_state, t0_ms, t1_ms)
                    raw_all.extend(r30)
                    for x in otr:
                        x["symbol"] = sym
                        opt_all.append(x)
            except Exception as e:
                print(f"  err {sym}: {e}", flush=True)
            if done % 40 == 0 or done == len(symbols):
                print(f"  進度 {done}/{len(symbols)} | 原始離開={len(raw_all)} 優化成交={len(opt_all)}", flush=True)

    opt_all.sort(key=lambda x: x["lift_ms"])
    if raw_all:
        ra = np.array(raw_all)
        print(f"\n[原始離開 30m 扣費] n={len(ra)} 勝率={100*np.mean(ra>0):.1f}% 平均={np.mean(ra):+.3f}% 累計={np.sum(ra):+.2f}%")
    if opt_all:
        oa = np.array([x["net"] for x in opt_all])
        pf = float(np.sum(oa[oa > 0]) / max(-np.sum(oa[oa <= 0]), 1e-9))
        print(f"[優化過高進 60m 扣費] n={len(oa)} 勝率={100*np.mean(oa>0):.1f}% 平均={np.mean(oa):+.3f}% PF={pf:.2f} 累計={np.sum(oa):+.2f}%")
        for x in opt_all:
            print(f"  {hm(x['lift_ms'])} {x['symbol']:14s} {x['reason']:3s} 扣費 {x['net']:+6.2f}%")
    return 0


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("黏帶監看測試（優化版）\n如果你看到這則，Telegram 已通。", strong=True)
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安黏帶三幕 Telegram 監看（優化版）")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--raw", action="store_true", help="關閉優化濾網，恢復舊版所有離開都推")
    p.add_argument("--backtest", type=int, default=0, metavar="DAYS", help="回測近 DAYS 天原始 vs 優化版")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.backtest > 0:
        return run_backtest_cli(args.backtest)

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe()
    mode_txt = "舊版全離開" if args.raw else "優化版（收高過高+前段跌幅+MA200拐頭+BTC順風）"
    print(f"監看 {len(symbols)} 個流動永續｜模式：{mode_txt}", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, raw_mode=args.raw)
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
