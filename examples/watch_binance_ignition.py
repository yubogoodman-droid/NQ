#!/usr/bin/env python3
"""幣安 1m 橫盤壓縮爆量起漲點監看（Breakout Ignition）：專抓死魚平台第一根真突破，附圖推 Telegram。

針對像 EVAAUSDT 22:10 這類「前段數小時窄幅橫盤 → 單根 1m 實體爆量收穿平台高點」的起漲點設計，
排除長上影假針（如 21:41）、高位寬幅震盪追高（如已漲 80% 末端）與極端單根消息倒貨針。

在下面填 Telegram 後執行：

    python3 examples/watch_binance_ignition.py                         # 每根 1m 收盤全市場監看
    python3 examples/watch_binance_ignition.py --once                  # 只掃剛收盤這一分
    python3 examples/watch_binance_ignition.py --test                  # 測試 Telegram 通不通
    python3 examples/watch_binance_ignition.py --symbol EVAAUSDT --replay 240  # 回放單幣抓起漲點並出圖
    python3 examples/watch_binance_ignition.py --backtest 14           # 回測近 N 天全市場表現
    python3 examples/watch_binance_ignition.py --raw                   # 放寬濾網（觀察所有帶量過高）

Ctrl+C 結束。同一波起漲冷卻期內不會重複推播。
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from numpy.lib.stride_tricks import sliding_window_view

# —— 填這裡（或設環境變數 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
VISION_DAILY = "https://data.binance.vision/data/futures/um/daily/klines"
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "binance_ignition_seen.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "output"

FEE_PCT = 0.10          # 突破觸價吃單來回手續費與滑價保守估 0.10%
COOLDOWN_BARS = 120     # 同一檔觸發後冷卻 120 分鐘，不重複追同一波中段
LIVE_BARS = 390         # 即時監看保留最近 390 根已收盤 1m K（涵蓋 6h 箱體 + MA200 斜率）
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"}

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


def sma_prev(a: np.ndarray, n: int) -> np.ndarray:
    """不含當根 i 的前 n 根簡單平均：out[i] = mean(a[i-n : i])。"""
    out = np.full(len(a), np.nan)
    if len(a) > n:
        cs = np.cumsum(np.insert(np.asarray(a, dtype=np.float64), 0, 0.0))
        out[n:] = (cs[n:-1] - cs[: -n - 1]) / n
    return out


def rolling_max_prev(a: np.ndarray, n: int) -> np.ndarray:
    """不含當根 i 的前 n 根最大值：out[i] = max(a[i-n : i])。"""
    out = np.full(len(a), np.nan)
    arr = np.asarray(a, dtype=np.float64)
    if len(arr) > n:
        out[n:] = sliding_window_view(arr[:-1], n).max(axis=-1)
    return out


def rolling_min_prev(a: np.ndarray, n: int) -> np.ndarray:
    """不含當根 i 的前 n 根最小值：out[i] = min(a[i-n : i])。"""
    out = np.full(len(a), np.nan)
    arr = np.asarray(a, dtype=np.float64)
    if len(arr) > n:
        out[n:] = sliding_window_view(arr[:-1], n).min(axis=-1)
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
            if used > 2100:
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


def universe(min_quote_vol: float = 1_500_000.0) -> list[str]:
    """取得純加密貨幣 USDT 永續合約清單（排除美股 TRADIFI 與主流權值幣）。"""
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("underlyingType") != "COIN":
            continue
        sym = s["symbol"]
        if sym in MAJORS:
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0.0)
        if qv < min_quote_vol:
            continue
        out.append(sym)
    return sorted(set(out))


def _rows_to_dict(rows: list) -> dict:
    return {
        "t": np.array([int(x[0]) for x in rows], dtype=np.int64),
        "o": np.array([float(x[1]) for x in rows], dtype=np.float64),
        "h": np.array([float(x[2]) for x in rows], dtype=np.float64),
        "l": np.array([float(x[3]) for x in rows], dtype=np.float64),
        "c": np.array([float(x[4]) for x in rows], dtype=np.float64),
        "qv": np.array([float(x[7]) for x in rows], dtype=np.float64),
        "tbq": np.array([float(x[10]) for x in rows], dtype=np.float64),
    }


def fetch_klines(sym: str, limit: int = LIVE_BARS) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": limit})
    if not raw:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if len(raw) < min(limit - 5, 365):
        return None
    return _rows_to_dict(raw)


def refresh_klines_cache(sym: str, cached: dict | None) -> dict | None:
    """增量更新快取：已載入過標的時只抓最新 6 根（weight=1），大幅縮短收盤掃描延遲。"""
    if cached is None or len(cached.get("t", [])) < 365:
        return fetch_klines(sym, limit=LIVE_BARS)
    last_ts = int(cached["t"][-1])
    now_ms = int(time.time() * 1000)
    gap_bars = (now_ms - last_ts) // 60_000
    if gap_bars > 15:
        return fetch_klines(sym, limit=LIVE_BARS)
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": 8})
    if not raw:
        return cached
    if int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if not raw:
        return cached
    tail = _rows_to_dict(raw)
    merged: dict[str, np.ndarray] = {}
    all_t = np.concatenate([cached["t"], tail["t"]])
    _, uniq_idx = np.unique(all_t, return_index=True)
    keep_idx = uniq_idx[-LIVE_BARS:]
    for k in ("t", "o", "h", "l", "c", "qv", "tbq"):
        arr = np.concatenate([cached[k], tail[k]])[keep_idx]
        merged[k] = arr
    return merged


def indicators(d: dict) -> dict:
    c, h, l, qv = d["c"], d["h"], d["l"], d["qv"]
    out = dict(d)
    out["m7"] = sma(c, 7)
    out["m25"] = sma(c, 25)
    out["m99"] = sma(c, 99)
    out["m200"] = sma(c, 200)
    out["hi120"] = rolling_max_prev(h, 120)
    out["lo120"] = rolling_min_prev(l, 120)
    out["hi360"] = rolling_max_prev(h, 360)
    out["lo360"] = rolling_min_prev(l, 360)
    out["v20"] = sma_prev(qv, 20)
    out["v120"] = sma_prev(qv, 120)
    return out


def raw_breakout_at(d: dict, i: int) -> dict | None:
    """基礎帶量過高偵測（--raw 模式用，只看收穿前 2h 高點與基本量價）。"""
    if i < 360 or i >= len(d["c"]):
        return None
    hi120, lo120 = float(d["hi120"][i]), float(d["lo120"][i])
    hi360, lo360 = float(d["hi360"][i]), float(d["lo360"][i])
    v20, v120 = float(d["v20"][i]), float(d["v120"][i])
    if np.isnan([hi120, lo120, hi360, lo360, v20, v120]).any() or lo120 <= 0 or lo360 <= 0:
        return None
    o, h, l, c = float(d["o"][i]), float(d["h"][i]), float(d["l"][i]), float(d["c"][i])
    qv, tbq = float(d["qv"][i]), float(d["tbq"][i])
    if c <= hi120 or o <= 0:
        return None
    rng = max(h - l, 1e-12)
    close_pos = (c - l) / rng
    bar_pct = (c / o - 1.0) * 100.0
    clr_pct = (c / hi120 - 1.0) * 100.0
    box120 = (hi120 / lo120 - 1.0) * 100.0
    box360 = (hi360 / lo360 - 1.0) * 100.0
    vr20 = qv / max(v20, 1.0)
    vr120 = qv / max(v120, 1.0)
    tbr = tbq / max(qv, 1e-9)
    if box120 > 8.0 or vr20 < 6.0 or qv < 100_000.0 or close_pos < 0.65 or bar_pct < 1.2:
        return None
    return {
        "i": i,
        "hi120": hi120,
        "lo120": lo120,
        "box120": box120,
        "box360": box360,
        "clr_pct": clr_pct,
        "bar_pct": bar_pct,
        "close_pos": close_pos,
        "vr20": vr20,
        "vr120": vr120,
        "qv": qv,
        "v120": v120,
        "tbr": tbr,
    }


def ignition_plan_at(d: dict, i: int) -> dict | None:
    """零未來函數實證優化版起漲點濾網 + 過高觸價／保本吊燈出場計畫。"""
    base = raw_breakout_at(d, i)
    if base is None:
        return None
    m25, m99, m200 = float(d["m25"][i]), float(d["m99"][i]), float(d["m200"][i])
    m200_prev = float(d["m200"][i - 60])
    if np.isnan([m25, m99, m200, m200_prev]).any() or m200_prev <= 0:
        return None
    c, h, l = float(d["c"][i]), float(d["h"][i]), float(d["l"][i])
    s200 = (m200 / m200_prev - 1.0) * 100.0

    # 1. 前段必須是低波壓縮死魚盤（2h 振幅 2.0%~6.5%、6h 振幅 <=12%、前 2h 均量 <=8萬U/m）
    if not (2.0 <= base["box120"] <= 6.5) or base["box360"] > 12.0 or base["v120"] > 80_000.0:
        return None
    # 2. 當根實體明確收穿箱頂（過高 >=0.60%、收在當根振幅 >=72%、單根漲幅 1.5%~7.0% 不追失控棒）
    if base["clr_pct"] < 0.60 or base["close_pos"] < 0.72 or not (1.5 <= base["bar_pct"] <= 7.0):
        return None
    # 3. 黃金點火量能帶（排除 >55x 單根消息面極端倒貨針）+ 主動吃單買盤 >=56%
    if not (7.5 <= base["vr20"] <= 55.0) or base["vr120"] < 8.0 or base["qv"] < 150_000.0 or base["tbr"] < 0.56:
        return None
    # 4. 均線結構墊高（收在 MA200 之上、MA25 > MA99、MA200 近 1h 斜率不走空）
    if not (c > m200 and m25 > m99 and s200 >= 0.0):
        return None

    trig_px = h * 1.0001
    cancel_px = max(l, base["hi120"] * 0.995)
    raw_risk = (trig_px - l) / trig_px * 100.0 + 0.10
    risk_pct = min(max(raw_risk, 1.20), 3.00)
    sl_px = trig_px * (1.0 - risk_pct / 100.0)

    plan = dict(base)
    plan.update(
        {
            "s200": s200,
            "trig_px": trig_px,
            "cancel_px": cancel_px,
            "sl_px": sl_px,
            "risk_pct": risk_pct,
            "tp5_px": trig_px * 1.05,
            "tp8_px": trig_px * 1.08,
            "be_activate_pct": 4.5,
            "be_lock_pct": 1.0,
            "trail_activate_pct": 10.0,
            "trail_dd_pct": 7.5,
            "hold_min": 60,
        }
    )
    return plan


def valid_ignitions(
    d: dict,
    *,
    start_i: int = 360,
    end_i: int | None = None,
    raw_mode: bool = False,
    cooldown: int = COOLDOWN_BARS,
) -> dict[int, dict]:
    """帶冷卻期的起漲點索引表，確保同一段主升浪只抓第一根起漲點。"""
    stop = len(d["c"]) if end_i is None else min(end_i, len(d["c"]))
    out: dict[int, dict] = {}
    last = -999_999
    detector = raw_breakout_at if raw_mode else ignition_plan_at
    for i in range(max(360, start_i), stop):
        if i - last < cooldown:
            continue
        hit = detector(d, i)
        if hit is not None:
            out[i] = hit
            last = i
    return out


def evaluate_trade_from(d: dict, sig_i: int, plan: dict) -> dict | None:
    """從起漲棒 sig_i 往後模擬 4 分鐘過高觸價成交與保本＋吊燈移動停利（同根 K 先判停損）。"""
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    n = len(c)
    trig_px = float(plan["trig_px"])
    cancel_px = float(plan["cancel_px"])
    fill_i = None
    epx = None
    for w in range(1, 5):
        if sig_i + w >= n:
            break
        if float(l[sig_i + w]) <= cancel_px:
            break
        if float(h[sig_i + w]) >= trig_px:
            fill_i = sig_i + w
            epx = float(max(o[sig_i + w], trig_px))
            break
    if fill_i is None or epx is None:
        return None

    raw_sl = (epx - float(l[sig_i])) / epx * 100.0 + 0.10
    risk_pct = min(max(raw_sl, 1.20), 3.00)
    sl_px = epx * (1.0 - risk_pct / 100.0)
    hold_m = int(plan.get("hold_min", 60))
    end_i = min(fill_i + hold_m, n - 1)

    cur_sl = sl_px
    peak = epx
    exit_i = end_i
    exit_px = float(c[end_i])
    gross = (exit_px / epx - 1.0) * 100.0
    reason = f"{hold_m}m"
    hit_5_ms = None
    hit_8_ms = None

    for k in range(fill_i, end_i + 1):
        # 保守原則：先檢查當根低點是否觸及目前停損，再更新當根最高價
        if float(l[k]) <= cur_sl:
            exit_i = k
            exit_px = cur_sl
            gross = (cur_sl / epx - 1.0) * 100.0
            reason = "SL" if cur_sl <= epx else "TRL"
            break
        hk = float(h[k])
        if hit_5_ms is None and hk >= epx * 1.05:
            hit_5_ms = int(t[k])
        if hit_8_ms is None and hk >= epx * 1.08:
            hit_8_ms = int(t[k])
        if hk > peak:
            peak = hk
        mfe_pct = (peak / epx - 1.0) * 100.0
        if mfe_pct >= float(plan.get("be_activate_pct", 4.5)):
            cur_sl = max(cur_sl, epx * (1.0 + float(plan.get("be_lock_pct", 1.0)) / 100.0))
        if mfe_pct >= float(plan.get("trail_activate_pct", 10.0)):
            cur_sl = max(cur_sl, peak * (1.0 - float(plan.get("trail_dd_pct", 7.5)) / 100.0))

    mfe_total = (peak / epx - 1.0) * 100.0
    return {
        "sig_i": sig_i,
        "sig_ms": int(t[sig_i]),
        "fill_i": fill_i,
        "fill_ms": int(t[fill_i]),
        "exit_i": exit_i,
        "exit_ms": int(t[exit_i]),
        "epx": epx,
        "sl_px": sl_px,
        "exit_px": exit_px,
        "mfe_pct": mfe_total,
        "gross": gross,
        "net": gross - FEE_PCT,
        "reason": reason,
        "hit_5_ms": hit_5_ms,
        "hit_8_ms": hit_8_ms,
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


def telegram_send(text: str, *, strong: bool = True, photo: str | None = None) -> bool:
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


def draw_chart(
    sym: str,
    d: dict,
    sig_i: int,
    path: str,
    plan: dict,
    trade: dict | None = None,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    font = _cjk_font()
    a0 = max(0, sig_i - 95)
    after = 65 if trade is not None else 15
    a1 = min(len(d["c"]), max(sig_i + after, (trade["exit_i"] + 6) if trade else sig_i + 15))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, qv = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["qv"][sl] / 1e3
    times = [datetime.fromtimestamp(int(ts) / 1000, TZ) for ts in d["t"][sl]]

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}, facecolor="#0c1210"
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    # 畫前段 2h 橫盤箱體陰影
    bx0 = max(0, sig_i - 120 - a0)
    bx1 = max(0, sig_i - a0)
    if bx1 > bx0:
        ax.add_patch(
            Rectangle(
                (bx0, plan["lo120"]),
                bx1 - bx0,
                plan["hi120"] - plan["lo120"],
                facecolor="#38bdf818",
                edgecolor="#38bdf866",
                linestyle="--",
                lw=0.8,
            )
        )

    colors_v = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.75)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors_v.append("#3dba7acc" if up else "#e35d5d99")
    axv.bar(xs, qv, width=0.8, color=colors_v, linewidth=0)
    axv.set_ylabel("k USDT", color="#8aa193", fontsize=8)

    pal = {7: "#f0c14a", 25: "#d28cff", 99: "#42a5f5", 200: "#ffffff"}
    for n_ma, col in pal.items():
        ax.plot(xs, sma(d["c"], n_ma)[sl], color=col, lw=1.05, label=f"MA{n_ma}")

    sx = sig_i - a0
    kw = {"fontproperties": font} if font else {}
    if 0 <= sx < len(c):
        ax.axvline(sx, color="#3dba7a", ls="--", lw=0.95)
        ax.scatter([sx], [c[sx]], s=42, color="#3dba7a", zorder=6)
        ax.annotate("起漲突破", xy=(sx, c[sx]), xytext=(5, 8), textcoords="offset points", color="#3dba7a", fontsize=9, **kw)

    if "trig_px" in plan:
        rx = len(c) - 1
        ax.hlines(plan["trig_px"], max(0, sx), rx, colors="#f59e0b", linestyles="--", lw=1.0, label="過高進場")
        ax.hlines(plan["sl_px"], max(0, sx), rx, colors="#e35d5d", linestyles=":", lw=1.0, label="初始停損")
        ax.hlines(plan["tp5_px"], max(0, sx), rx, colors="#22d3ee", linestyles="-.", lw=0.85, label="+5% 減倉位")
        ax.hlines(plan["tp8_px"], max(0, sx), rx, colors="#a3e635", linestyles="-.", lw=0.85, label="+8% 目標位")

    if trade is not None:
        fx = trade["fill_i"] - a0
        ex_x = trade["exit_i"] - a0
        if 0 <= fx < len(c):
            ax.scatter([fx], [trade["epx"]], s=55, facecolors="none", edgecolors="#f59e0b", lw=1.8, zorder=7)
        if 0 <= ex_x < len(c):
            ecol = "#3dba7a" if trade["net"] > 0 else "#e35d5d"
            ax.scatter([ex_x], [trade["exit_px"]], s=55, facecolors="none", edgecolors=ecol, lw=1.8, zorder=7)
            ax.annotate(
                f"{trade['reason']} {trade['net']:+.1f}%",
                xy=(ex_x, trade["exit_px"]),
                xytext=(-48, 10),
                textcoords="offset points",
                color=ecol,
                fontsize=8.5,
                **kw,
            )

    ax.set_title(
        f"{sym}  1m 起漲點  (2h箱體 {plan['box120']:.1f}% · 量 {plan['vr20']:.1f}x · 主動買 {plan['tbr']*100:.0f}%)",
        color="#e8f0ea",
        fontsize=11,
        **kw,
    )
    leg_kw = {"prop": font} if font else {"fontsize": 7.5}
    ax.legend(loc="upper left", frameon=False, labelcolor="#c8d5cc", ncol=7, **leg_kw)
    step = max(1, len(xs) // 8)
    ticks = list(range(0, len(xs), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([times[k].strftime("%H:%M") for k in ticks], color="#8aa193", fontsize=8)
    fig.tight_layout(pad=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_event(ev: dict) -> str:
    sym, d, p = ev["symbol"], ev["d"], ev["plan"]
    sig_i = ev["sig_i"]
    ts_str = hm(int(d["t"][sig_i]))
    px = float(d["c"][sig_i])
    if ev["kind"] == "raw":
        return (
            f"⚡ <b>帶量過高（未過濾）</b>  <b>{sym}</b>  1m\n"
            f"時間 {ts_str}　收盤 {px:g}（+{p['bar_pct']:.2f}%）\n"
            f"2h箱體 {p['box120']:.2f}%　量 {p['vr20']:.1f}x（{p['qv']/1e3:.0f}k U）　主動買 {p['tbr']*100:.0f}%"
        )
    return (
        f"🚀 <b>橫盤爆量起漲點</b>  <b>{sym}</b>  1m\n"
        f"時間 {ts_str}　收盤 <b>{px:g}</b>（當根 +{p['bar_pct']:.2f}% · 收高 {p['close_pos']*100:.0f}%）\n"
        f"前2h箱體 {p['lo120']:.5g}–{p['hi120']:.5g}（寬 {p['box120']:.2f}% · 站上 +{p['clr_pct']:.2f}%）\n"
        f"爆量 <b>{p['vr20']:.1f}x</b>（{p['qv']/1e3:.0f}k U · 主動買 {p['tbr']*100:.0f}%）　MA200斜率 {p['s200']:+.2f}%\n"
        f"• 觸價進場：突破 <b>{p['trig_px']:.6g}</b>（4分鐘內有效，先跌破 {p['cancel_px']:.6g} 取消）\n"
        f"• 初始停損：<b>{p['sl_px']:.6g}</b>（-{p['risk_pct']:.2f}%）\n"
        f"• 短打停利參考：<b>+5%</b> → {p['tp5_px']:.6g} ｜ <b>+8%</b> → {p['tp8_px']:.6g}\n"
        f"• 波段移動停利：漲滿 +4.5% 停損推保本 +1%；漲滿 +10% 改守高點回撤 7.5%（最長 60m）"
    )


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['sig_i']])}:{ev['kind']}"


def scan_symbol(sym: str, *, cache: dict[str, dict] | None = None, raw_mode: bool = False) -> list[dict]:
    prev = cache.get(sym) if cache is not None else None
    raw = refresh_klines_cache(sym, prev)
    if raw is None:
        return []
    if cache is not None:
        cache[sym] = raw
    d = indicators(raw)
    n = len(d["c"])
    last = n - 1
    hits = valid_ignitions(d, start_i=max(360, last - COOLDOWN_BARS - 5), end_i=n, raw_mode=raw_mode)
    events = []
    for closed in (last, last - 1):
        if closed in hits:
            events.append(
                {
                    "kind": "raw" if raw_mode else "ignition",
                    "symbol": sym,
                    "sig_i": closed,
                    "plan": hits[closed],
                    "d": d,
                }
            )
    return events


def scan_all(symbols: list[str], *, cache: dict[str, dict] | None = None, raw_mode: bool = False) -> list[dict]:
    events: list[dict] = []
    with ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(scan_symbol, s, cache=cache, raw_mode=raw_mode): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (-e["plan"]["vr20"], e["symbol"]))
    return events


def notify(ev: dict) -> None:
    strong = ev["kind"] == "ignition"
    text = format_event(ev)
    clean = text.replace("<b>", "").replace("</b>", "")
    print("\n" + clean, flush=True)
    photo = None
    if strong:
        tmp = Path("/tmp") / f"ignition_{ev['symbol']}_{ev['sig_i']}.png"
        photo = draw_chart(ev["symbol"], ev["d"], ev["sig_i"], str(tmp), ev["plan"])
    ok = telegram_send(text, strong=strong, photo=photo)
    if ok:
        print("  → Telegram 已送")
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
        else:
            print("  → Telegram 送出失敗，檢查 token 與 chat id")


def _parse_vision_zip(content: bytes) -> list[list[float]]:
    zf = zipfile.ZipFile(io.BytesIO(content))
    lines = zf.read(zf.namelist()[0]).decode().splitlines()
    if lines and lines[0][0].isalpha():
        lines = lines[1:]
    rows: list[list[float]] = []
    for ln in lines:
        if not ln:
            continue
        p = ln.split(",")
        rows.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), 0.0, 0, float(p[7]), 0, 0.0, float(p[10])])
    return rows


def fetch_history_fast(sym: str, start_ms: int, end_ms: int) -> dict | None:
    """優先從 data.binance.vision 下載日檔 zip（零 API 權重），缺漏或當日尾段再用 REST 補齊。"""
    warmup_ms = start_ms - 400 * 60_000
    d0 = datetime.fromtimestamp(warmup_ms / 1000, timezone.utc).date()
    d1 = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    today_utc = datetime.now(timezone.utc).date()

    all_rows: list = []
    missing_from_ms: int | None = None
    sess = get_session()
    cur_d = d0
    while cur_d <= d1:
        day_start_ms = int(datetime(cur_d.year, cur_d.month, cur_d.day, tzinfo=timezone.utc).timestamp() * 1000)
        if cur_d < today_utc:
            url = f"{VISION_DAILY}/{sym}/1m/{sym}-1m-{cur_d.strftime('%Y-%m-%d')}.zip"
            try:
                r = sess.get(url, timeout=15)
                if r.status_code == 200:
                    all_rows.extend(_parse_vision_zip(r.content))
                    cur_d += timedelta(days=1)
                    continue
            except requests.RequestException:
                pass
        if missing_from_ms is None:
            missing_from_ms = max(warmup_ms, day_start_ms)
        cur_d += timedelta(days=1)

    if missing_from_ms is not None:
        cur = missing_from_ms
        while cur < end_ms:
            # 使用 limit=499（單次 weight 僅 2，比 limit=1500 的 weight=10 省 40%+ 配額）
            raw = get_json(
                "/fapi/v1/klines",
                params={"symbol": sym, "interval": "1m", "startTime": cur, "endTime": end_ms - 1, "limit": 499},
            )
            if not raw:
                break
            all_rows.extend(raw)
            last_open = int(raw[-1][0])
            if len(raw) < 499:
                break
            cur = last_open + 60_000

    dedup = {int(x[0]): x for x in all_rows if warmup_ms <= int(x[0]) and int(x[0]) + 60_000 <= end_ms}
    if len(dedup) < 420:
        return None
    ordered = [dedup[k] for k in sorted(dedup)]
    return _rows_to_dict(ordered)


def simulate_symbol(raw: dict, t0_ms: int, t1_ms: int) -> list[dict]:
    """對單一標的執行零未來函數回測。"""
    d = indicators(raw)
    hits = valid_ignitions(d, start_i=360, end_i=len(d["c"]) - 5, raw_mode=False)
    trades: list[dict] = []
    for sig_i, plan in hits.items():
        ts = int(d["t"][sig_i])
        if ts < t0_ms or ts >= t1_ms:
            continue
        tr = evaluate_trade_from(d, sig_i, plan)
        if tr is not None:
            tr["vr20"] = plan["vr20"]
            tr["box120"] = plan["box120"]
            trades.append(tr)
    return trades


def run_replay_cli(sym: str, lookback_min: int, raw_mode: bool = False) -> int:
    sym = sym.upper().strip()
    need = min(max(lookback_min + 380, 420), 1500)
    now_ms = int(time.time() * 1000)
    if need <= 499:
        raw_kl = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": need})
        if raw_kl and int(raw_kl[-1][0]) + 60_000 > now_ms:
            raw_kl = raw_kl[:-1]
        raw = _rows_to_dict(raw_kl) if raw_kl and len(raw_kl) >= 380 else None
    else:
        raw = fetch_history_fast(sym, now_ms - lookback_min * 60_000, now_ms)
    if raw is None:
        print(f"無法取得足夠 {sym} 1m K 線資料。")
        return 1

    d = indicators(raw)
    start_i = max(360, len(d["c"]) - lookback_min)
    hits = valid_ignitions(d, start_i=start_i, end_i=len(d["c"]), raw_mode=raw_mode)
    print(
        f"回放 {sym}｜區間 {hm(int(d['t'][start_i]))} ~ {hm(int(d['t'][-1]))} (UTC+8)｜命中起漲點 {len(hits)} 筆",
        flush=True,
    )
    if not hits:
        return 0

    for sig_i, plan in hits.items():
        ev = {"kind": "raw" if raw_mode else "ignition", "symbol": sym, "sig_i": sig_i, "plan": plan, "d": d}
        print("\n" + format_event(ev).replace("<b>", "").replace("</b>", ""))
        tr = evaluate_trade_from(d, sig_i, plan) if not raw_mode else None
        if tr is not None:
            t5 = hm(tr["hit_5_ms"]) if tr["hit_5_ms"] else "未達"
            t8 = hm(tr["hit_8_ms"]) if tr["hit_8_ms"] else "未達"
            print(
                f"  ↳ 模擬成交：{hm(tr['fill_ms'])} @ {tr['epx']:.6g}｜"
                f"+5% 時間：{t5}｜+8% 時間：{t8}｜"
                f"最高浮盈：+{tr['mfe_pct']:.1f}%｜出場：{hm(tr['exit_ms'])} ({tr['reason']}) 扣費淨利 {tr['net']:+.2f}%"
            )
        out_png = OUT_DIR / f"ignition_{sym}_{datetime.fromtimestamp(int(d['t'][sig_i])/1000, TZ).strftime('%m%d_%H%M')}.png"
        saved = draw_chart(sym, d, sig_i, str(out_png), plan, trade=tr)
        if saved:
            print(f"  ↳ 圖表已存：{saved}")
    return 0


def run_backtest_cli(days: int) -> int:
    now_dt = datetime.now(TZ).replace(second=0, microsecond=0)
    start_dt = (now_dt - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    t0_ms, t1_ms = int(start_dt.timestamp() * 1000), int(now_dt.timestamp() * 1000)
    symbols = universe()
    print(
        f"回測區間 {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {now_dt.strftime('%Y-%m-%d %H:%M')} (UTC+8)｜純幣永續 {len(symbols)} 檔",
        flush=True,
    )

    all_trades: list[dict] = []
    done = 0
    with ThreadPoolExecutor(10) as ex:
        futs = {ex.submit(fetch_history_fast, s, t0_ms, t1_ms): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                raw = fut.result()
                if raw is not None:
                    for tr in simulate_symbol(raw, t0_ms, t1_ms):
                        tr["symbol"] = sym
                        all_trades.append(tr)
            except Exception as e:
                print(f"  err {sym}: {e}", flush=True)
            if done % 60 == 0 or done == len(symbols):
                print(f"  進度 {done}/{len(symbols)}｜累積成交 {len(all_trades)} 筆", flush=True)

    all_trades.sort(key=lambda x: x["sig_ms"])
    if not all_trades:
        print("區間內無符合條件成交。")
        return 0

    nets = np.array([x["net"] for x in all_trades], dtype=np.float64)
    mfes = np.array([x["mfe_pct"] for x in all_trades], dtype=np.float64)
    pf = float(np.sum(nets[nets > 0]) / max(-np.sum(nets[nets <= 0]), 1e-9))
    print(
        f"\n[起漲點過高進 + 保本吊燈移動停利（扣費 {FEE_PCT:.2f}%）]\n"
        f"筆數={len(nets)}｜勝率={100*np.mean(nets>0):.1f}%｜觸及+5%率={100*np.mean(mfes>=5.0):.1f}%｜"
        f"平均={np.mean(nets):+.3f}%｜PF={pf:.2f}｜累計={np.sum(nets):+.2f}%"
    )
    for x in all_trades:
        print(
            f"  {hm(x['sig_ms'])} {x['symbol']:15s} 進 {x['epx']:<9.5g} 量 {x['vr20']:4.1f}x "
            f"箱體 {x['box120']:4.1f}% 最高 +{x['mfe_pct']:5.1f}% → {x['reason']:3s} {x['net']:+6.2f}%"
        )
    return 0


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1.0, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("🚀 幣安起漲點監看測試\n如果你看到這則，Telegram 推播已通。", strong=True)
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="幣安 1m 橫盤壓縮爆量起漲點 Telegram 監看")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--raw", action="store_true", help="放寬濾網，列出所有帶量收過 2h 高點的訊號")
    p.add_argument("--symbol", type=str, default="", help="指定單一幣別（搭配 --replay 回放驗證）")
    p.add_argument("--replay", type=int, default=0, metavar="MINUTES", help="回放單一幣別近 MINUTES 分鐘抓起漲點並存圖")
    p.add_argument("--backtest", type=int, default=0, metavar="DAYS", help="回測近 DAYS 天全市場表現")
    p.add_argument("--min-qv", type=float, default=1_500_000.0, help="24h 最低成交額門檻（預設 150 萬 USDT）")
    args = p.parse_args()

    apply_keys()
    if args.test:
        return test_telegram()
    if args.symbol and args.replay > 0:
        return run_replay_cli(args.symbol, args.replay, raw_mode=args.raw)
    if args.backtest > 0:
        return run_backtest_cli(args.backtest)

    seen = load_seen()
    cache: dict[str, dict] = {}
    print("載入純幣永續標的…", flush=True)
    symbols = universe(min_quote_vol=args.min_qv)
    mode_txt = "未過濾帶量過高" if args.raw else "實證優化版（窄箱壓縮 + 7.5~55x 爆量收高 + 主動買盤 + 保本吊燈）"
    print(f"監看 {len(symbols)} 個永續合約｜模式：{mode_txt}", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
            symbols = universe(min_quote_vol=args.min_qv)
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)} 檔", flush=True)
        t0 = time.time()
        events = scan_all(symbols, cache=cache, raw_mode=args.raw)
        new = [e for e in events if key_of(e) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 檔用 {time.time()-t0:.1f}s　新起漲訊號 {len(new)}",
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
