#!/usr/bin/env python3
"""幣安 5m 均線糾結爆量起漲監控與回測掃描（AKEUSDT 型起漲點）。

抓「長時極窄橫盤 + 八均線麻花糾結 + 爆量實體紅 K 破箱體 + 合約持倉量（OI）同步放大」的第一根點火棒，
並在後續 2～6 根（10～30 分鐘）縮量守住 MA25 / 箱體上緣、短均多頭散開時推播「回踩續攻強訊號（附圖）」。

用法：
  # 1. 即時監控（每 5 分鐘收盤掃全市場流動永續，推 Telegram）
  python3 examples/watch_binance_breakout.py

  # 2. 只掃剛收盤這一輪 / 測試 Telegram
  python3 examples/watch_binance_breakout.py --once
  python3 examples/watch_binance_breakout.py --test

  # 3. 回測掃描指定幣種或全市場近 N 天，並輸出 HTML 圖表報告
  python3 examples/watch_binance_breakout.py --scan --symbols AKEUSDT,SHELLUSDT --days 4 --html output/ake_breakout.html
  python3 examples/watch_binance_breakout.py --scan --days 3 --html output/binance_breakout_3d.html

Telegram 憑證可填在下方常數，或放在專案根目錄 tg_config.env：
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Callable

import numpy as np
import requests

# —— 可直接填這裡，或放在專案根目錄 tg_config.env ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO / "tg_config.env"
SEEN_PATH = REPO / "output" / "binance_breakout_seen.json"
PAGES_PATH = REPO / "docs" / "binance-5m-breakout" / "index.html"

INTERVAL = "5m"
INTERVAL_MS = 300_000
MA_PERIODS = (7, 14, 25, 30, 60, 99, 120, 200)

# 對齊幣安手機版八條均線配色
MA_COLORS = {
    7: "#f0b90b",    # 黃
    14: "#4dd0e1",   # 青
    25: "#d28cff",   # 粉
    30: "#5c6bc0",   # 藍
    60: "#26a69a",   # 綠
    99: "#ab47bc",   # 紫
    120: "#66bb6a",  # 淺綠
    200: "#ef5350",  # 酒紅
}

# 排除大盤權值幣與反向槓桿標的，專注抓獨立行情起漲標的
MAJOR_SYMBOLS = frozenset({
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
})
INVERSE_SYMBOLS = frozenset({
    "SQQQUSDT",
    "SOXSUSDT",
    "SKDDUSDT",
    "TBTUSDT",
    "TZAUSDT",
    "UVXYUSDT",
})

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0",
        "Clienttype": "web",
        "Accept": "application/json",
    }
)


@dataclass(frozen=True)
class BreakoutParams:
    box_bars: int = 48               # 過去 48 根 5m（4 小時）橫盤箱體
    max_box_range_pct: float = 10.0  # 箱體高低振幅上限（%）
    max_ma_spread_pct: float = 5.0   # 起漲前八條均線最大糾結帶寬（%）
    min_bar_gain_pct: float = 2.8    # 點火棒單根實體漲幅下限（%）
    max_bar_gain_pct: float = 16.0   # 單根漲幅上限（排除一步到位插針棒）
    min_btc_alpha_pct: float = 2.5   # 相對同根 BTCUSDT 超額漲幅下限（排除宏觀數據齊漲假突破）
    min_vol_ratio: float = 7.0       # 當根成交額相對前 20 根均量倍數
    min_bar_quote_vol: float = 200_000.0  # 點火棒最低成交額（USDT）
    min_close_pos: float = 0.70      # 收盤位於當根振幅比例 (C-L)/(H-L)，排除長上影倒貨
    min_oi_change_pct: float = 0.5   # 當根 5m 持倉量（OI）增幅下限（%）；設 None 代表不檢查
    confirm_min_bars: int = 2        # 點火後最早第幾根檢查回踩續攻確認
    confirm_max_bars: int = 6        # 點火後最晚第幾根檢查回踩續攻確認
    cooldown_bars: int = 36          # 同幣種訊號冷卻根數（36 根 = 3 小時）
    min_24h_quote_vol: float = 5_000_000.0  # 全市場掃描 24h 最低成交額（USDT）


# ---------------------------------------------------------------------------
# Environment & HTTP helpers
# ---------------------------------------------------------------------------


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


def get_json(path: str, params: dict | None = None, retries: int = 5):
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code in (418, 429):
                time.sleep(2.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.6 * (attempt + 1))
    assert last_err is not None
    raise last_err


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Indicators & Pure Signal Logic
# ---------------------------------------------------------------------------


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if len(a) >= n and n > 0:
        cs = np.cumsum(np.insert(a.astype(float), 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def indicators(d: dict) -> dict:
    c = np.asarray(d["c"], dtype=float)
    qv = np.asarray(d["qv"], dtype=float)
    out = dict(d)
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    out["qv20"] = sma(qv, 20)
    return out


def _ma_spread_at(d: dict, idx: int) -> float | None:
    if idx < 0 or idx >= len(d["c"]):
        return None
    vals = [float(d[f"m{n}"][idx]) for n in MA_PERIODS]
    if np.isnan(vals).any():
        return None
    lo = min(vals)
    if lo <= 0:
        return None
    return (max(vals) / lo - 1.0) * 100.0


def ignition_at(
    d: dict,
    i: int,
    params: BreakoutParams = BreakoutParams(),
    btc_ret_map: dict[int, float] | None = None,
) -> dict | None:
    """檢查索引 i 是否為「均線糾結 + 窄箱體爆量突破」的第一根點火棒。"""
    c, o, h, l, qv = d["c"], d["o"], d["h"], d["l"], d["qv"]
    warmup = max(max(MA_PERIODS) + 5, params.box_bars + 5)
    if i < warmup or i >= len(c):
        return None

    # 1. 起漲前均線必須極窄糾結（看 i-1 與 i-3 取較小值，容許點火前 1~2 根微幅試盤）
    sp1 = _ma_spread_at(d, i - 1)
    sp3 = _ma_spread_at(d, i - 3)
    if sp1 is None:
        return None
    ma_spread_pct = min(sp1, sp3) if sp3 is not None else sp1
    if ma_spread_pct > params.max_ma_spread_pct:
        return None

    # 2. 過去 box_bars 根（預設 4 小時）高低振幅壓縮
    box_hi = float(np.max(h[i - params.box_bars : i]))
    box_lo = float(np.min(l[i - params.box_bars : i]))
    if box_lo <= 0:
        return None
    box_range_pct = (box_hi / box_lo - 1.0) * 100.0
    if box_range_pct > params.max_box_range_pct:
        return None

    # 3. 當根收盤同時站上全部八條均線，且收盤突破箱體上緣
    close_i = float(c[i])
    if close_i <= box_hi:
        return None
    cur_mas = [float(d[f"m{n}"][i]) for n in MA_PERIODS]
    if np.isnan(cur_mas).any() or not all(close_i > m for m in cur_mas):
        return None

    # 4. 點火實體漲幅與相對 BTC 超額報酬（濾掉宏觀數據瞬間齊漲假突破）
    open_i = float(o[i])
    if open_i <= 0:
        return None
    bar_gain_pct = (close_i / open_i - 1.0) * 100.0
    if not (params.min_bar_gain_pct <= bar_gain_pct <= params.max_bar_gain_pct):
        return None

    ts = int(d["t"][i])
    btc_gain = float(btc_ret_map.get(ts, 0.0)) if btc_ret_map else 0.0
    btc_alpha_pct = bar_gain_pct - btc_gain
    if btc_ret_map is not None and btc_alpha_pct < params.min_btc_alpha_pct:
        return None

    # 5. 成交額爆量倍數與絕對流動性門檻
    base_idx = max(0, i - 3)
    base_qv = float(d["qv20"][base_idx])
    if np.isnan(base_qv) or base_qv <= 0:
        base_qv = float(d["qv20"][i - 1])
    if np.isnan(base_qv) or base_qv <= 0:
        return None
    quote_vol = float(qv[i])
    vol_ratio = quote_vol / base_qv
    if vol_ratio < params.min_vol_ratio or quote_vol < params.min_bar_quote_vol:
        return None

    # 6. 實體收在當根振幅上緣，排除長上影衝高倒貨棒
    rng = float(h[i] - l[i])
    if rng <= 0:
        return None
    close_pos = (close_i - float(l[i])) / rng
    if close_pos < params.min_close_pos:
        return None

    return {
        "idx": i,
        "ts": ts,
        "entry_price": close_i,
        "stop_price": float(l[i]),
        "box_hi": box_hi,
        "box_lo": box_lo,
        "box_range_pct": float(box_range_pct),
        "ma_spread_pct": float(ma_spread_pct),
        "bar_gain_pct": float(bar_gain_pct),
        "btc_alpha_pct": float(btc_alpha_pct),
        "vol_ratio": float(vol_ratio),
        "quote_vol": quote_vol,
        "close_pos": float(close_pos),
    }


def confirm_at(
    d: dict,
    ign_i: int,
    j: int,
    ign: dict,
    params: BreakoutParams = BreakoutParams(),
) -> dict | None:
    """檢查點火後第 j 根是否完成「縮量不破 MA25 與點火低點、短均多頭散開收紅確認」。"""
    if j - ign_i < params.confirm_min_bars or j - ign_i > params.confirm_max_bars:
        return None
    c, o, l, qv = d["c"], d["o"], d["l"], d["qv"]
    if j >= len(c):
        return None

    m7, m14, m25 = d["m7"], d["m14"], d["m25"]
    if np.isnan([m7[j], m14[j], m25[j]]).any():
        return None

    # 點火後到確認棒之間：收盤不可跌破 MA25，且最低點不可跌破點火棒低點（停損位）
    if np.any(c[ign_i + 1 : j + 1] < m25[ign_i + 1 : j + 1]):
        return None
    pullback_low = float(np.min(l[ign_i + 1 : j + 1]))
    if pullback_low < float(ign["stop_price"]):
        return None

    # 確認當根需短均多頭排列 MA7 > MA14 > MA25、收紅 K 且站穩箱體頂之上
    if not (m7[j] > m14[j] > m25[j]):
        return None
    if not (c[j] >= o[j] and c[j] >= float(ign["box_hi"])):
        return None

    min_pb_qv = float(np.min(qv[ign_i + 1 : j + 1]))
    pb_vol_ratio = min_pb_qv / max(float(ign["quote_vol"]), 1e-9)
    short_fan_pct = (float(m7[j]) / float(m25[j]) - 1.0) * 100.0
    return {
        "confirm_idx": j,
        "confirm_ts": int(d["t"][j]),
        "confirm_price": float(c[j]),
        "pullback_low": pullback_low,
        "pullback_vol_ratio": float(pb_vol_ratio),
        "short_fan_pct": float(short_fan_pct),
    }


def first_confirm(
    d: dict,
    ign_i: int,
    ign: dict,
    params: BreakoutParams = BreakoutParams(),
    upto_idx: int | None = None,
) -> dict | None:
    hi = len(d["c"]) - 1 if upto_idx is None else min(len(d["c"]) - 1, upto_idx)
    for j in range(ign_i + params.confirm_min_bars, min(hi, ign_i + params.confirm_max_bars) + 1):
        conf = confirm_at(d, ign_i, j, ign, params)
        if conf is not None:
            return conf
    return None


def detect_signals(
    d: dict,
    params: BreakoutParams = BreakoutParams(),
    btc_ret_map: dict[int, float] | None = None,
    oi_change_fn: Callable[[int], float | None] | None = None,
    eval_bars: int = 144,
) -> list[dict]:
    """在單一標的完整 K 線序列上找出所有點火訊號與後續確認、12h 最大漲幅（回測/掃描用）。"""
    if "m200" not in d:
        d = indicators(d)
    c, h, l = d["c"], d["h"], d["l"]
    n = len(c)
    results: list[dict] = []
    last_ign = -10_000

    for i in range(max(MA_PERIODS) + 5, n):
        if i - last_ign < params.cooldown_bars:
            continue
        ign = ignition_at(d, i, params, btc_ret_map=btc_ret_map)
        if ign is None:
            continue

        oi_chg: float | None = None
        if oi_change_fn is not None:
            oi_chg = oi_change_fn(ign["ts"])
            if (
                params.min_oi_change_pct is not None
                and oi_chg is not None
                and oi_chg < params.min_oi_change_pct
            ):
                continue

        last_ign = i
        conf = first_confirm(d, i, ign, params)

        end_j = min(n, i + 1 + eval_bars)
        if i + 1 < end_j:
            fut_hi = float(np.max(h[i + 1 : end_j]))
            mfe_pct = (fut_hi / ign["entry_price"] - 1.0) * 100.0
            peak_j = i + 1 + int(np.argmax(h[i + 1 : end_j]))
            mae_to_peak_pct = (float(np.min(l[i + 1 : peak_j + 1])) / ign["entry_price"] - 1.0) * 100.0
        else:
            mfe_pct = 0.0
            mae_to_peak_pct = 0.0

        rec = dict(ign)
        rec["oi_change_pct"] = oi_chg
        rec["confirm"] = conf
        rec["mfe_12h_pct"] = float(mfe_pct)
        rec["mae_to_peak_pct"] = float(mae_to_peak_pct)
        results.append(rec)
    return results


# 回測出場：停損放在點火低點，但最多離進場 8%；進場 3 根後收盤跌破 MA99 出場；最長持有 24h。
STOP_CAP = 0.08
TRAIL_ARM_BARS = 3
MAX_HOLD_BARS = 288


def simulate_trade(d: dict, rec: dict) -> dict | None:
    """回踩確認收盤進場。停損先看點火低，離進場超過 8% 就改停在進場下方 8%。

    進場後每一根先看最低價是否打到停損；沒有，且已過 3 根，收盤跌破 MA99 就用該收盤出場。
    24 小時（288 根 5m）都沒出場，用該根收盤時間出場。資料在期限前結束則標未平。
    """
    conf = rec.get("confirm")
    if not conf:
        return None
    c = d["c"]
    h = d["h"]
    l = d["l"]
    m99 = d["m99"]
    e_idx = int(conf["confirm_idx"])
    entry = float(conf["confirm_price"])
    if e_idx >= len(c) - 1 or entry <= 0:
        return None
    stop = max(float(rec["stop_price"]), entry * (1.0 - STOP_CAP))
    if stop >= entry:
        stop = entry * (1.0 - STOP_CAP)
    last_i = min(len(c) - 1, e_idx + MAX_HOLD_BARS)
    data_ended = e_idx + MAX_HOLD_BARS > len(c) - 1
    ex_i = last_i
    ex_p = float(c[ex_i])
    reason = "open" if data_ended else "time"
    for j in range(e_idx + 1, last_i + 1):
        if float(l[j]) <= stop:
            ex_i, ex_p, reason = j, stop, "stop"
            break
        if j >= e_idx + TRAIL_ARM_BARS and float(c[j]) < float(m99[j]):
            ex_i, ex_p, reason = j, float(c[j]), "ma99"
            break
    return {
        "entry_idx": e_idx,
        "exit_idx": int(ex_i),
        "entry_price": entry,
        "exit_price": float(ex_p),
        "stop_price": float(stop),
        "pnl_pct": (float(ex_p) / entry - 1.0) * 100.0,
        "reason": reason,
        "hold_bars": int(ex_i - e_idx),
    }


# ---------------------------------------------------------------------------
# Binance Data Fetching
# ---------------------------------------------------------------------------


def universe(min_quote_vol: float = 5_000_000.0) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    pairs: list[tuple[float, str]] = []
    for s in info.get("symbols", []):
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        if sym in MAJOR_SYMBOLS or sym in INVERSE_SYMBOLS:
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0.0)
        if qv < min_quote_vol:
            continue
        pairs.append((qv, sym))
    pairs.sort(reverse=True)
    return [sym for _, sym in pairs]


def fetch_klines(sym: str, limit: int = 320) -> dict | None:
    """抓取已收盤 5m K 線（自動支援超過 1500 根的分頁回溯）。"""
    need = max(260, int(limit) + 1)
    chunks: list[list] = []
    end_ms: int | None = None
    rem = need
    while rem > 0:
        batch = min(1500, rem)
        q: dict[str, int | str] = {"symbol": sym, "interval": INTERVAL, "limit": batch}
        if end_ms is not None:
            q["endTime"] = end_ms
        raw = get_json("/fapi/v1/klines", params=q)
        if not raw:
            break
        chunks.append(raw)
        rem -= len(raw)
        if len(raw) < batch:
            break
        end_ms = int(raw[0][0]) - 1
    if not chunks:
        return None
    merged = [row for part in reversed(chunks) for row in part]
    now_ms = int(time.time() * 1000)
    if merged and int(merged[-1][0]) + INTERVAL_MS > now_ms:
        merged = merged[:-1]
    if len(merged) < 220:
        return None
    return {
        "t": np.array([int(x[0]) for x in merged], dtype=np.int64),
        "o": np.array([float(x[1]) for x in merged], dtype=float),
        "h": np.array([float(x[2]) for x in merged], dtype=float),
        "l": np.array([float(x[3]) for x in merged], dtype=float),
        "c": np.array([float(x[4]) for x in merged], dtype=float),
        "v": np.array([float(x[5]) for x in merged], dtype=float),
        "qv": np.array([float(x[7]) for x in merged], dtype=float),
    }


def fetch_btc_ret_map(limit: int = 320) -> dict[int, float]:
    d = fetch_klines("BTCUSDT", limit=limit)
    if d is None:
        return {}
    return {
        int(ts): float((c / o - 1.0) * 100.0) if o > 0 else 0.0
        for ts, o, c in zip(d["t"], d["o"], d["c"])
    }


def fetch_oi_change_5m(sym: str, bar_open_ms: int) -> float | None:
    """查詢點火當根 5m K 線期間 [bar_open_ms, bar_open_ms + 5m] 的持倉量（OI）變化率（%）。"""
    try:
        rows = get_json(
            "/futures/data/openInterestHist",
            params={
                "symbol": sym,
                "period": "5m",
                "startTime": bar_open_ms - INTERVAL_MS,
                "endTime": bar_open_ms + 2 * INTERVAL_MS,
                "limit": 10,
            },
            retries=2,
        )
    except Exception:  # noqa: BLE001
        return None
    mp = {int(x["timestamp"]): float(x["sumOpenInterest"]) for x in (rows or []) if "timestamp" in x}
    oi0 = mp.get(bar_open_ms)
    oi1 = mp.get(bar_open_ms + INTERVAL_MS)
    # 剛收盤幾秒內若歷史快照尚未寫入，改讀即時未平倉量
    if oi0 and oi1 is None and abs(int(time.time() * 1000) - (bar_open_ms + INTERVAL_MS)) < 120_000:
        try:
            live = get_json("/fapi/v1/openInterest", params={"symbol": sym}, retries=2)
            oi1 = float(live["openInterest"])
        except Exception:  # noqa: BLE001
            oi1 = None
    if oi0 and oi1 and oi0 > 0:
        return (oi1 / oi0 - 1.0) * 100.0
    return None


# ---------------------------------------------------------------------------
# Charting & HTML Report
# ---------------------------------------------------------------------------


def render_chart_bytes(
    sym: str,
    d: dict,
    ign_i: int,
    box_hi: float,
    conf_i: int | None = None,
    exit_i: int | None = None,
    stop_price: float | None = None,
    entry_price: float | None = None,
    exit_price: float | None = None,
    title: str | None = None,
) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        plt.rcParams["font.sans-serif"] = [
            "PingFang TC",
            "Microsoft JhengHei",
            "Noto Sans CJK TC",
            "WenQuanYi Micro Hei",
            "Droid Sans Fallback",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        return None

    end_i = exit_i if exit_i is not None else ign_i
    a0 = max(0, ign_i - 36)
    a1 = min(len(d["c"]), max(ign_i + 40, end_i + 8))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, qv = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["qv"][sl] / 1e6
    ts_arr = d["t"][sl]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    vol_cols = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.75)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(float(h[k] - l[k]), 1e-12) * 0.03
        ax.add_patch(Rectangle((xs[k] - 0.34, y0), 0.68, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        vol_cols.append("#3dba7a99" if up else "#e35d5d99")

    axv.bar(xs, qv, width=0.78, color=vol_cols, linewidth=0)
    axv.set_ylabel("USDT (M)", color="#8aa193", fontsize=8)

    for n in MA_PERIODS:
        key = f"m{n}"
        arr = d[key][sl] if key in d else sma(d["c"], n)[sl]
        ax.plot(xs, arr, color=MA_COLORS[n], lw=1.05, label=f"MA{n}")

    ax.axhline(box_hi, color="#8aa193", ls=":", lw=0.9, alpha=0.8)
    if stop_price is not None:
        ax.axhline(stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.9)

    def _mark(idx: int | None, price: float | None, color: str, text: str) -> None:
        if idx is None:
            return
        x = idx - a0
        if not (0 <= x < len(c)):
            return
        y = float(c[x] if price is None else price)
        ax.scatter([x], [y], s=220, facecolors="none", edgecolors=color, linewidths=1.8, zorder=6)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 9), ha="center", color=color, fontsize=8)

    _mark(ign_i, None, "#ff9800", "點火")
    _mark(conf_i, entry_price, "#ff9f1a", "進")
    exit_col = "#00c805" if (exit_price is not None and entry_price is not None and exit_price >= entry_price) else "#ff5252"
    _mark(exit_i, exit_price, exit_col, "出")

    tick_step = max(1, len(xs) // 7)
    tick_idx = list(range(0, len(xs), tick_step))
    axv.set_xticks(tick_idx)
    axv.set_xticklabels([hm(int(ts_arr[k])) for k in tick_idx], color="#8aa193", fontsize=8)

    ax.set_title(title or f"{sym}  5m  ({hm(int(d['t'][ign_i]))} UTC+8)", color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def draw_chart(sym: str, d: dict, ign_i: int, box_hi: float, path: str, conf_i: int | None = None) -> str | None:
    raw = render_chart_bytes(sym, d, ign_i, box_hi, conf_i=conf_i)
    if raw is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return path


def _equity_svg(pnls: list[float]) -> str:
    if not pnls:
        return ""
    cum = np.cumsum(np.asarray(pnls, dtype=float))
    width, height = 720, 160
    lo = float(min(0.0, cum.min()))
    hi = float(max(0.0, cum.max()))
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(cum):
        x = 0.0 if len(cum) == 1 else i * width / (len(cum) - 1)
        y = height - 10 - (float(v) - lo) / span * (height - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    zero_y = height - 10 - (0.0 - lo) / span * (height - 20)
    color = "#00c805" if float(cum[-1]) >= 0 else "#ff5252"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0d1117;border-radius:10px">'
        f'<line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/></svg>'
    )


def write_backtest_html(records: list[dict], out_path: Path, days: float, n_symbols: int) -> Path:
    """回踩確認進場的回測頁。圖嵌在同一個 HTML，手機直接開。"""
    reason_label = {"stop": "停損", "ma99": "收盤跌破 MA99", "time": "持有 24h", "open": "未平"}
    trades: list[dict] = []
    for rec in records:
        trade = simulate_trade(rec["d"], rec)
        if trade is None:
            continue
        row = dict(rec)
        row["trade"] = trade
        trades.append(row)
    trades.sort(key=lambda r: r["confirm"]["confirm_ts"])

    pnls = [t["trade"]["pnl_pct"] for t in trades]
    closed = [t for t in trades if t["trade"]["reason"] != "open"]
    wins = [p for p in pnls if p > 0]
    closed_pnls = [t["trade"]["pnl_pct"] for t in closed]
    closed_wins = [p for p in closed_pnls if p > 0]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["trade"]["reason"]] = reasons.get(t["trade"]["reason"], 0) + 1

    def _avg(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _med(xs: list[float]) -> float:
        return float(np.median(xs)) if xs else 0.0

    cards: list[str] = []
    for i, rec in enumerate(trades, 1):
        tr = rec["trade"]
        conf = rec["confirm"]
        pnl = tr["pnl_pct"]
        cls = "pnl-win" if pnl > 0 else ("pnl-flat" if pnl == 0 else "pnl-loss")
        title = (
            f"#{i}  {rec['symbol']}  {hm(conf['confirm_ts'])} → {hm(int(rec['d']['t'][tr['exit_idx']]))}  "
            f"{reason_label.get(tr['reason'], tr['reason'])}  {pnl:+.2f}%"
        )
        png = render_chart_bytes(
            rec["symbol"],
            rec["d"],
            rec["idx"],
            rec["box_hi"],
            conf_i=conf["confirm_idx"],
            exit_i=tr["exit_idx"],
            stop_price=tr["stop_price"],
            entry_price=tr["entry_price"],
            exit_price=tr["exit_price"],
            title=title,
        )
        img = ""
        if png:
            img = (
                "<img src='data:image/png;base64,"
                + base64.b64encode(png).decode("ascii")
                + f"' alt='{escape(rec['symbol'])}'/>"
            )
        oi_txt = f"{rec['oi_change_pct']:+.2f}%" if rec.get("oi_change_pct") is not None else "n/a"
        risk = (tr["entry_price"] - tr["stop_price"]) / tr["entry_price"] * 100.0
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div><span class='trade-no'>#{i} · {escape(rec['symbol'])}</span>"
            f"<span class='trade-time'>{hm(conf['confirm_ts'])} → {hm(int(rec['d']['t'][tr['exit_idx']]))} UTC+8</span></div>"
            f"<div class='card-pnl {cls}'>{pnl:+.2f}%</div></header>"
            f"<div class='tags'><span class='tag'>{escape(reason_label.get(tr['reason'], tr['reason']))}</span>"
            f"<span class='tag'>爆量 {rec['vol_ratio']:.1f}x</span>"
            f"<span class='tag'>OI {oi_txt}</span>"
            f"<span class='tag'>持有 {tr['hold_bars'] * 5} 分</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {tr['entry_price']:.6g}  @ {hm(conf['confirm_ts'])}\n"
            f"停損 {tr['stop_price']:.6g}  （-{risk:.2f}%）\n"
            f"出場 {tr['exit_price']:.6g}  {reason_label.get(tr['reason'], tr['reason'])}\n"
            f"點火 {hm(rec['ts'])} {rec['entry_price']:.6g}（+{rec['bar_gain_pct']:.1f}%）"
            f"  箱體 {rec['box_range_pct']:.2f}%  均線糾結 {rec['ma_spread_pct']:.2f}%"
            "</pre>"
            f"<div class='mini-chart'>{img}</div></article>"
        )

    ign_only = len(records) - len(trades)
    gen_ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M UTC+8")
    total = float(sum(pnls))
    avg = _avg(pnls)
    med = _med(pnls)
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    closed_rate = (len(closed_wins) / len(closed_pnls) * 100.0) if closed_pnls else 0.0
    avg_cls = "pnl-win" if avg >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 5m 起漲回測 · 近 {days:g} 天</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:14px 12px 40px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:16px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:92px;border:1px solid #21262d}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e;display:block}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.rules{{margin:0;padding-left:1.2em;color:#c9d1d9;font-size:13px;line-height:1.65}}
.note{{margin-top:8px;padding:8px 10px;background:#0d1117;border-radius:8px;color:#f0c14b;font-size:12px}}
img{{width:100%;display:block;border-radius:10px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 5m 均線糾結爆量起漲 · 回測</h1>
<p class="muted">近 {days:g} 天 · 24h 成交額 ≥ 1,000 萬 USDT 的永續 · {n_symbols} 檔 · 產出 {escape(gen_ts)}</p>
<h2>進出場規則</h2>
<ol class="rules">
<li>標的：幣安 USDT 永續，排除 BTC／ETH／SOL／BNB／XRP／DOGE／ADA／TRX 與反向 ETF。24h 成交額要到 1,000 萬 USDT。</li>
<li>點火：前一根八條均線（MA7／14／25／30／60／99／120／200）糾結 ≤ 5%，過去 4 小時箱體振幅 ≤ 10%。當根收盤站上全部均線並突破箱體，單根漲 2.8%～16%，比同一根 BTC 多漲至少 2.5%，成交額 ≥ 7 倍均量且 ≥ 20 萬 USDT，收在當根振幅上緣 ≥ 70%，5 分鐘持倉量增加 ≥ 0.5%。</li>
<li>進場：點火後第 2～6 根，收盤沒有跌破 MA25、最低沒有跌破點火低點，而且 MA7 &gt; MA14 &gt; MA25、收紅站回箱體上緣。進場價＝這根收盤。沒有確認的點火不進場。</li>
<li>停損：點火那根最低價。若離進場超過 8%，改停在進場價下方 8%。先看最低價打到停損。</li>
<li>出場：進場 3 根之後，5 分鐘收盤跌破 MA99，用該收盤價出場。都沒有就持有滿 24 小時，用收盤價出場。資料還沒走完的標未平。</li>
<li>報酬：出場價 / 進場價 − 1。未扣手續費與滑價。下面的加總是每一筆報酬相加，不是複利。</li>
</ol>
<p class="note">橘字「點火」＝爆量破箱體。橘圈「進」＝回踩確認收盤進場。綠圈或紅圈「出」＝出場。紅虛線＝停損。黃／青／粉／藍／綠／紫／淺綠／紅線＝MA7／14／25／30／60／99／120／200。</p>
<h2>回測統計</h2>
<p class="muted">點火 {len(records)} 筆，通過回踩確認進場 {len(trades)} 筆，點火後沒確認 {ign_only} 筆（不進場）。
<br/>出場：收盤跌破 MA99 {reasons.get('ma99', 0)} · 停損 {reasons.get('stop', 0)} · 持有 24h {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}</p>
<div class="cards">
<div class="card">筆數<b>{len(trades)}</b></div>
<div class="card">勝率<b>{win_rate:.1f}%</b></div>
<div class="card">已平勝率<b>{closed_rate:.1f}%</b></div>
<div class="card">加總<b class="{avg_cls}">{total:+.1f}%</b></div>
<div class="card">平均<b class="{avg_cls}">{avg:+.2f}%</b></div>
<div class="card">中位數<b class="{'pnl-win' if med>=0 else 'pnl-loss'}">{med:+.2f}%</b></div>
<div class="card">最好<b class="pnl-win">{best:+.1f}%</b></div>
<div class="card">最差<b class="pnl-loss">{worst:+.1f}%</b></div>
</div>
{_equity_svg(pnls)}
</section>
{''.join(cards) or "<p class='muted'>這段期間沒有通過回踩確認的進場。</p>"}
</div></body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_html_report(records: list[dict], out_path: Path, days: float, n_symbols: int) -> Path:
    cards: list[str] = []
    confirmed_cnt = sum(1 for r in records if r.get("confirm") is not None)
    avg_mfe = float(np.mean([r["mfe_12h_pct"] for r in records])) if records else 0.0

    for r in records:
        sym = r["symbol"]
        d = r["d"]
        conf = r.get("confirm")
        conf_idx = conf["confirm_idx"] if conf else None
        png = render_chart_bytes(sym, d, r["idx"], r["box_hi"], conf_i=conf_idx)
        img_html = ""
        if png:
            b64 = base64.b64encode(png).decode("ascii")
            img_html = f'<img src="data:image/png;base64,{b64}" alt="{escape(sym)}" />'
        oi_txt = f"{r['oi_change_pct']:+.2f}%" if r.get("oi_change_pct") is not None else "n/a"
        if conf:
            badge = f'<span class="badge strong">回踩確認 @ {hm(conf["confirm_ts"])} ({conf["confirm_price"]:.6g})</span>'
        else:
            badge = '<span class="badge ign">僅點火（未過回踩確認）</span>'
        cards.append(
            f"""
            <section class="card">
              <div class="head">
                <div>
                  <h2>{escape(sym)}</h2>
                  <div class="meta">點火時間：{hm(r["ts"])} (UTC+8) &nbsp;|&nbsp; {badge}</div>
                </div>
                <div class="mfe">12h 最高 +{r["mfe_12h_pct"]:.1f}%</div>
              </div>
              <div class="grid">
                <div><span>點火收盤</span><b>{r["entry_price"]:.6g} (+{r["bar_gain_pct"]:.1f}%)</b></div>
                <div><span>箱體上緣 / 停損低</span><b>{r["box_hi"]:.6g} / {r["stop_price"]:.6g}</b></div>
                <div><span>爆量倍數 / 成交額</span><b>{r["vol_ratio"]:.1f}x ({r["quote_vol"]/1e6:.2f}M U)</b></div>
                <div><span>均線糾結 / 4h 箱體</span><b>{r["ma_spread_pct"]:.2f}% / {r["box_range_pct"]:.2f}%</b></div>
                <div><span>5m OI 增幅</span><b>{oi_txt}</b></div>
                <div><span>達最高前最大回撤</span><b>{r["mae_to_peak_pct"]:+.1f}%</b></div>
              </div>
              {img_html}
            </section>
            """
        )

    gen_ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M UTC+8")
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>幣安 5m 均線糾結爆量起漲掃描報告</title>
<style>
  body {{ margin:0; background:#0c1210; color:#e8f0ea; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:20px 14px 48px; }}
  h1 {{ margin:0 0 6px; font-size:1.45rem; }}
  .sub {{ color:#8aa193; font-size:0.9rem; margin-bottom:18px; line-height:1.5; }}
  .card {{ background:#131d19; border:1px solid rgba(232,240,234,0.12); border-radius:10px; padding:14px; margin-bottom:16px; }}
  .head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
  h2 {{ margin:0; font-size:1.2rem; }}
  .meta {{ color:#8aa193; font-size:0.83rem; margin-top:4px; }}
  .mfe {{ color:#3dba7a; font-weight:700; font-size:1.1rem; font-family:monospace; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.76rem; }}
  .badge.strong {{ background:rgba(61,186,122,0.18); color:#5ce09b; border:1px solid rgba(61,186,122,0.45); }}
  .badge.ign {{ background:rgba(255,152,0,0.16); color:#ffb74d; border:1px solid rgba(255,152,0,0.4); }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:12px; }}
  .grid > div {{ background:rgba(0,0,0,0.22); border:1px solid rgba(232,240,234,0.08); border-radius:6px; padding:7px 9px; }}
  .grid span {{ display:block; color:#8aa193; font-size:0.74rem; }}
  .grid b {{ font-family:monospace; font-size:0.9rem; margin-top:2px; display:block; }}
  img {{ width:100%; height:auto; border-radius:6px; display:block; }}
  @media (max-width:680px) {{ .grid {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>幣安 5m 均線糾結爆量起漲掃描報告</h1>
  <div class="sub">
    掃描範圍：{n_symbols} 檔流動永續 · 近 {days:g} 天 · 產出時間 {gen_ts}<br/>
    共點火 <b>{len(records)}</b> 筆（其中通過回踩續攻確認 <b>{confirmed_cnt}</b> 筆），12h 平均最大漲幅 <b>+{avg_mfe:.1f}%</b>。橘圈＝點火突破、綠圈＝縮量回踩續攻確認。
  </div>
  {''.join(cards) if cards else '<p class="sub">此區間無符合條件訊號。</p>'}
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Live Scanning & Telegram Notification
# ---------------------------------------------------------------------------


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8")


def telegram_send(text: str, *, photo: str | None = None) -> bool:
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
                        "disable_notification": "false",
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


def scan_live_symbol(
    sym: str,
    params: BreakoutParams,
    btc_ret_map: dict[int, float],
    check_oi: bool = True,
) -> list[dict]:
    raw = fetch_klines(sym, limit=280)
    if raw is None:
        return []
    d = indicators(raw)
    last = len(d["c"]) - 1
    events: list[dict] = []

    # 只檢查剛收盤與前一根（避免掃描跨整點漏單）
    for closed in (last, last - 1):
        if closed < max(MA_PERIODS) + 5:
            continue

        # 1) 這根是否本身就是第一根點火起漲棒
        ign = ignition_at(d, closed, params, btc_ret_map=btc_ret_map)
        if ign is not None:
            # 確認前 cooldown_bars 內沒有重複點火
            recent_dup = any(
                ignition_at(d, k, params, btc_ret_map=btc_ret_map) is not None
                for k in range(max(max(MA_PERIODS) + 5, closed - params.cooldown_bars), closed)
            )
            if not recent_dup:
                oi_chg = fetch_oi_change_5m(sym, ign["ts"]) if check_oi else None
                if not (
                    check_oi
                    and params.min_oi_change_pct is not None
                    and oi_chg is not None
                    and oi_chg < params.min_oi_change_pct
                ):
                    events.append(
                        {
                            "kind": "ignition",
                            "symbol": sym,
                            "ign": ign,
                            "oi_change_pct": oi_chg,
                            "d": d,
                        }
                    )

        # 2) 這根是否為先前某根點火棒的「第一次回踩續攻確認」
        for ign_i in range(closed - params.confirm_max_bars, closed - params.confirm_min_bars + 1):
            ign_prev = ignition_at(d, ign_i, params, btc_ret_map=btc_ret_map)
            if ign_prev is None:
                continue
            conf = first_confirm(d, ign_i, ign_prev, params, upto_idx=closed)
            if conf is None or conf["confirm_idx"] != closed:
                continue
            oi_chg = fetch_oi_change_5m(sym, ign_prev["ts"]) if check_oi else None
            if (
                check_oi
                and params.min_oi_change_pct is not None
                and oi_chg is not None
                and oi_chg < params.min_oi_change_pct
            ):
                continue
            events.append(
                {
                    "kind": "confirm",
                    "symbol": sym,
                    "ign": ign_prev,
                    "confirm": conf,
                    "oi_change_pct": oi_chg,
                    "d": d,
                }
            )
    return events


def format_ignition(ev: dict) -> str:
    ign = ev["ign"]
    sym = ev["symbol"]
    oi_txt = f"{ev['oi_change_pct']:+.2f}%" if ev.get("oi_change_pct") is not None else "n/a"
    risk_pct = (ign["entry_price"] - ign["stop_price"]) / ign["entry_price"] * 100.0
    return (
        f"<b>[點火起漲 · 5m 破箱體]</b>  <b>{sym}</b>\n"
        f"時間：{hm(ign['ts'])} (UTC+8)\n"
        f"點火收盤：<b>{ign['entry_price']:.6g}</b>（+{ign['bar_gain_pct']:.1f}%）\n"
        f"建議觀察回踩區：{ign['box_hi']:.6g} ～ {ign['entry_price']:.6g}\n"
        f"參考停損（點火低）：{ign['stop_price']:.6g}（風險 {risk_pct:.1f}%）\n"
        f"爆量：{ign['vol_ratio']:.1f}x（{ign['quote_vol']/1e6:.2f}M U）　5m OI：{oi_txt}\n"
        f"前均線糾結：{ign['ma_spread_pct']:.2f}%　4h 箱體：{ign['box_range_pct']:.2f}%\n"
        f"<i>若接下來 10–30 分鐘縮量守住 MA25 並收紅散開，會再推強訊號附圖。</i>"
    )


def format_confirm(ev: dict) -> str:
    ign = ev["ign"]
    conf = ev["confirm"]
    sym = ev["symbol"]
    oi_txt = f"{ev['oi_change_pct']:+.2f}%" if ev.get("oi_change_pct") is not None else "n/a"
    risk_pct = (conf["confirm_price"] - ign["stop_price"]) / conf["confirm_price"] * 100.0
    return (
        f"<b>[強訊號 · 回踩續攻確認]</b>\n"
        f"<b>{sym}</b>  5m\n"
        f"點火 {hm(ign['ts'])} ({ign['entry_price']:.6g}) → 確認 {hm(conf['confirm_ts'])} (<b>{conf['confirm_price']:.6g}</b>)\n"
        f"回踩最低：{conf['pullback_low']:.6g}（未破箱體頂 {ign['box_hi']:.6g} / MA25）\n"
        f"回踩縮量比：{conf['pullback_vol_ratio']*100:.0f}%　點火 OI：{oi_txt}\n"
        f"參考停損：{ign['stop_price']:.6g}（距離 {risk_pct:.1f}%）\n"
        f"短均 MA7&gt;MA14&gt;MA25 已散開（+{conf['short_fan_pct']:.2f}%），AKE 型起漲確認。"
    )


def event_key(ev: dict) -> str:
    return f"{ev['symbol']}:{ev['ign']['ts']}:{ev['kind']}"


def notify(ev: dict) -> None:
    strong = ev["kind"] == "confirm"
    text = format_confirm(ev) if strong else format_ignition(ev)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("&gt;", ">")
    )
    print("\n" + plain, flush=True)
    photo = None
    if strong:
        tmp = Path("/tmp") / f"breakout_{ev['symbol']}_{ev['ign']['ts']}.png"
        photo = draw_chart(
            ev["symbol"],
            ev["d"],
            ev["ign"]["idx"],
            ev["ign"]["box_hi"],
            str(tmp),
            conf_i=ev["confirm"]["confirm_idx"],
        )
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送出", flush=True)
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("  → 未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，僅印在終端機", flush=True)
        else:
            print("  → Telegram 送出失敗，請檢查 token 與 chat id", flush=True)


def wait_next_5m_close() -> None:
    now = time.time()
    nxt = (int(now) // 300 + 1) * 300 + 3
    time.sleep(max(1.0, nxt - now))


# ---------------------------------------------------------------------------
# Historical Scan Mode (--scan)
# ---------------------------------------------------------------------------


def run_scan(
    symbols: list[str],
    days: float,
    params: BreakoutParams,
    check_oi: bool = True,
    html_out: Path | None = None,
) -> list[dict]:
    bars = int(days * 288) + max(MA_PERIODS) + 20
    print(f"載入 BTCUSDT 5m 基準（近 {days:g} 天，約 {bars} 根）…", flush=True)
    btc_ret_map = fetch_btc_ret_map(limit=bars)
    cutoff_ms = int((time.time() - days * 86400) * 1000)

    def _worker(sym: str) -> list[dict]:
        raw = fetch_klines(sym, limit=bars)
        if raw is None:
            return []
        d = indicators(raw)
        oi_fn = (lambda ts: fetch_oi_change_5m(sym, ts)) if check_oi else None
        sigs = detect_signals(d, params=params, btc_ret_map=btc_ret_map, oi_change_fn=oi_fn)
        out = []
        for s in sigs:
            if s["ts"] >= cutoff_ms:
                s["symbol"] = sym
                s["d"] = d
                out.append(s)
        return out

    print(f"掃描 {len(symbols)} 檔標的…", flush=True)
    t0 = time.time()
    records: list[dict] = []
    done = 0
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            try:
                records.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  [略過] {futs[fut]}: {exc}", flush=True)
            if done % 25 == 0 or done == len(symbols):
                print(f"  進度 {done}/{len(symbols)}", flush=True)

    records.sort(key=lambda r: r["ts"])
    elapsed = time.time() - t0
    conf_records = [r for r in records if r.get("confirm") is not None]
    print(
        f"\n掃描完成（{elapsed:.1f}s）：點火訊號 {len(records)} 筆，"
        f"通過回踩續攻確認 {len(conf_records)} 筆（全部時間為台北 UTC+8）\n",
        flush=True,
    )
    if records:
        header = (
            f"{'點火時間':11s}  {'標的':13s}  {'點火價':>10s}  {'單根':>6s}  "
            f"{'爆量':>6s}  {'成交額':>8s}  {'糾結':>6s}  {'箱體':>6s}  "
            f"{'5m OI':>7s}  {'回踩確認':18s}  {'12h最高漲幅':>10s}"
        )
        print(header)
        print("-" * len(header))
        for r in records:
            oi_s = f"{r['oi_change_pct']:+6.2f}%" if r.get("oi_change_pct") is not None else "   n/a "
            conf = r.get("confirm")
            if conf:
                conf_s = f"YES {hm(conf['confirm_ts'])[-5:]} @{conf['confirm_price']:.5g}"
            else:
                conf_s = "NO"
            print(
                f"{hm(r['ts']):11s}  {r['symbol']:13s}  {r['entry_price']:10.6g}  "
                f"+{r['bar_gain_pct']:4.1f}%  {r['vol_ratio']:5.1f}x  "
                f"{r['quote_vol']/1e6:6.2f}M  {r['ma_spread_pct']:5.2f}%  "
                f"{r['box_range_pct']:5.2f}%  {oi_s}  {conf_s:18s}  +{r['mfe_12h_pct']:8.1f}%"
            )

    if html_out is not None:
        saved = write_backtest_html(records, html_out, days=days, n_symbols=len(symbols))
        print(f"\nHTML 圖表報告已輸出：{saved}", flush=True)
    return records


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="幣安 5m 均線糾結爆量起漲監控與掃描")
    p.add_argument("--once", action="store_true", help="只掃剛收盤這一輪即結束")
    p.add_argument("--test", action="store_true", help="發送測試訊息驗證 Telegram 設定")
    p.add_argument("--scan", action="store_true", help="回測掃描近 N 天歷史訊號")
    p.add_argument("--days", type=float, default=3.5, help="--scan 回看天數（預設 3.5 天）")
    p.add_argument("--symbols", type=str, default="", help="指定標的（逗號分隔，如 AKEUSDT,SHELLUSDT）")
    p.add_argument("--min-qv", type=float, default=5_000_000.0, help="全市場掃描最低 24h 成交額 USDT（預設 5,000,000）")
    p.add_argument("--no-oi", action="store_true", help="關閉 5m 持倉量（OI）增幅過濾")
    p.add_argument("--html", type=str, default="", help="--scan 時輸出 HTML 報告路徑")
    p.add_argument("--pages", action="store_true", help="--scan 時輸出至 docs/binance-5m-breakout/index.html")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_keys()

    if args.test:
        ok = telegram_send("幣安 5m 起漲監控測試\n如果你看到這則訊息，Telegram 推播已接通。")
        print("Telegram 測試：", "成功" if ok else "失敗（請檢查 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）")
        return 0 if ok else 1

    params = BreakoutParams(
        min_24h_quote_vol=args.min_qv,
        min_oi_change_pct=None if args.no_oi else 0.5,
    )
    check_oi = not args.no_oi

    if args.symbols.strip():
        symbols = [
            s.strip().upper() if s.strip().upper().endswith("USDT") else s.strip().upper() + "USDT"
            for s in args.symbols.split(",")
            if s.strip()
        ]
    else:
        print("載入幣安流動 USDT 永續清單…", flush=True)
        symbols = universe(params.min_24h_quote_vol)

    if args.scan:
        html_out: Path | None = None
        if args.html.strip():
            html_out = Path(args.html.strip())
        elif args.pages:
            html_out = PAGES_PATH
        run_scan(symbols, days=args.days, params=params, check_oi=check_oi, html_out=html_out)
        return 0

    seen = load_seen()
    print(f"監看 {len(symbols)} 檔流動永續（5m 均線糾結爆量破箱體 + 回踩續攻確認）。", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, uni_ts
        if not args.symbols.strip() and time.time() - uni_ts > 1800:
            symbols = universe(params.min_24h_quote_vol)
            uni_ts = time.time()
            print(f"更新標的清單：{len(symbols)} 檔", flush=True)

        t0 = time.time()
        btc_ret_map = fetch_btc_ret_map(limit=30)
        events: list[dict] = []
        with ThreadPoolExecutor(8) as ex:
            futs = {
                ex.submit(scan_live_symbol, sym, params, btc_ret_map, check_oi): sym
                for sym in symbols
            }
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"  [錯誤] {futs[fut]}: {exc}", flush=True)

        events.sort(key=lambda e: (0 if e["kind"] == "confirm" else 1, e["symbol"]))
        new_events = [e for e in events if event_key(e) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')} UTC+8] "
            f"掃完 {len(symbols)} 檔（{time.time() - t0:.1f}s）　新訊號 {len(new_events)}",
            flush=True,
        )
        for ev in new_events:
            seen.add(event_key(ev))
            notify(ev)
        if new_events:
            save_seen(seen)

    round_once()
    if args.once:
        return 0

    print("持續監控中，每根 5m 收盤自動掃描一次（Ctrl+C 停止）…", flush=True)
    try:
        while True:
            wait_next_5m_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止監控。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
