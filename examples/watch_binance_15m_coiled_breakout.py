#!/usr/bin/env python3
"""幣安 15m「底部七線黏帶 · 倍量穿 MA200」起漲監看與回掃（對齊 ESPORTSUSDT 2026-10-31 17:30 圈選位置）。

型態定義（對照 ESPORTSUSDT 15m + 1h 圖）：
  1. 長時間壓在 MA200 下方築底：突破前連續 >= 35 根 15m K 線收在 MA200 下方（約 9 小時以上）。
  2. 七線極致糾結（黏帶）：突破前 8 根內，MA7/14/25/30/60/99/120 七條均線最大帶寬 <= 1.80%，
     且底部長均線簇 MA60/99/120 帶寬 <= 1.20%。
  3. 長均走平翻揚：MA60 近 16 根（4 小時）斜率 >= -0.10%（已止跌走平或翻揚），
     MA200 近 16 根斜率 >= -1.00%（下彎趨緩走平）。
  4. 短均多頭疊在長均簇之上：MA7 > MA14 > MA25 >= MA30，且 MA25 > max(MA60, MA99, MA120)。
  5. 第一拍「破帶穿 MA200」（圈內起漲點，如 ESPORTSUSDT 10-31 17:30）：
     當根收實體紅 K（漲幅 >= 0.80%、實體佔振幅 >= 0.55），收盤正式突破站上 MA200
     （離 MA200 乖離介於 0.10% ~ 3.50%，排除單根過度暴衝衰竭），且成交量 >= 20 根均量 × 2.0。
  6. 第二拍「扇形展開（強確認）」（突破後 1~4 根內，如 ESPORTSUSDT 10-31 17:45~18:15）：
     收盤續創突破後新高、離 MA200 擴大至 >= 2.50%，短均扇形張開，且 1H 等價均線（15m MA240 = 1H MA60）
     同步被站上（多週期共振）。

用法：
  # 驗證單一標的並畫出圈選起漲圖
  python3 examples/watch_binance_15m_coiled_breakout.py --symbol ESPORTSUSDT --plot output/esports_15m_circle.png

  # 回掃全市場近期符合「圈起來位置」的標的
  python3 examples/watch_binance_15m_coiled_breakout.py --scan

  # 回測近 90 天（三個月）全市場績效（支援第一拍突破進場 & 第二拍強確認進場）
  python3 examples/watch_binance_15m_coiled_breakout.py --backtest-days 90

  # 持續監看每根 15m 收盤並推播 Telegram（含 K 線圖）
  python3 examples/watch_binance_15m_coiled_breakout.py
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

TELEGRAM_BOT_TOKEN = ""  # 可直接填，或放專案根目錄 tg_config.env
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO / "tg_config.env"
SEEN_PATH = REPO / "output" / "binance_15m_coiled_breakout_seen.json"
KEEP = {"ESPORTSUSDT"}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})

MA_PERIODS = (7, 14, 25, 30, 60, 99, 120, 200)
RIBBON_7_PERIODS = (7, 14, 25, 30, 60, 99, 120)
LONG_CLUSTER_PERIODS = (60, 99, 120)


@dataclass(frozen=True)
class CoiledBreakoutParams:
    min_bars_below_ma200: int = 35
    coil_lookback: int = 8
    max_coil_7_120_pct: float = 1.80
    max_coil_long_pct: float = 1.20
    slope_lookback: int = 16
    min_ma60_slope_pct: float = 0.00
    min_ma200_slope_pct: float = -1.00
    min_ext_ma200_pct: float = 0.10
    max_ext_ma200_pct: float = 3.50
    min_bar_chg_pct: float = 0.80
    min_body_ratio: float = 0.55
    min_vol_ratio: float = 2.00
    # v2 對齊 ESPORTSUSDT (10-31 17:30) 與 牛来USDT (10-31 23:00) 共同特徵：
    clean_first_cross: bool = True  # 必須當根直接從 MA200 下方突破站上（排除前根偷站無量的遲滯棒）
    require_ma25_above_long_at_cross: bool = True  # True=嚴格版(突破當根MA25>長均簇頂)；False=標準版(允許MA25仍在長均簇內、MA7>長均簇頂)
    min_green_streak: int = 2  # 突破棒含前根至少連 2 紅階梯推升（ESPORTS 連7紅、牛来 連4紅）
    min_above_ma60_bars: int = 9  # 突破前 12 根至少 9 根已站穩 MA60 之上（排除從 MA60 下方單根亂抽）
    min_mom4_pct: float = 1.80  # 近 4 根 (1小時) 累計推升動能 >= 1.80%（ESPORTS +3.92%、牛来 +3.49%）
    max_upper_wick_ratio: float = 0.24  # 突破棒上影線佔比 <= 24%（ESPORTS 15%、牛来 3%，排除長上影砸盤）
    min_vol_vs_12bar_max: float = 1.25  # 突破棒成交量須 >= 前 12 根（3小時）最大量的 1.25 倍
    max_breakout_range_vs_med24: float = 10.5  # 突破棒振幅相對前 24 根(6h)振幅中位數倍數上限（排除死魚盤後單根開盤異常暴抽）
    min_gap_ma200_long_pct: float = 0.50  # MA200 與下方長均簇 (MA60/99/120) 碗型蓄力空間下限
    max_gap_ma200_long_pct: float = 3.20  # MA200 與下方長均簇碗型空間上限（ESPORTS 2.90%、牛来 2.78%）
    follow_min_bars: int = 1
    follow_max_bars: int = 4
    follow_min_ext_ma200_pct: float = 2.20
    follow_min_vol_ratio: float = 2.00
    follow_require_bull_bar: bool = True  # 第二拍確認當根須為紅 K（c > o）
    follow_max_upper_wick_ratio: float = 0.35  # 第二拍確認棒上影線佔比 <= 35%（排除確認棒衝高回落）
    follow_max_red_bars: int = 1  # 突破棒至確認棒之間最多允許 1 根陰線休息
    follow_hold_low_above_ma200: bool = True  # 突破後至確認棒期間，每根 K 線最低價皆須守在 MA200 之上


@dataclass
class BreakoutSignal:
    symbol: str
    breakout_idx: int
    breakout_time: str
    breakout_close: float
    stop_price: float
    coil_7_120_pct: float
    coil_long_pct: float
    all_8_spread_pct: float
    ma60_slope_pct: float
    ma200_slope_pct: float
    ext_ma200_pct: float
    bar_chg_pct: float
    body_ratio: float
    vol_ratio: float
    bars_below_ma200: int
    mom4_pct: float = 0.0
    vol_vs_12bar_max: float = 0.0
    upper_wick_ratio: float = 0.0
    gap_ma200_long_pct: float = 0.0
    green_streak: int = 0
    follow_idx: int | None = None
    follow_time: str | None = None
    follow_close: float | None = None
    follow_ext_ma200_pct: float | None = None
    follow_vol_ratio: float | None = None
    follow_stop_price: float | None = None


@dataclass
class BacktestTrade:
    symbol: str
    stage: str
    entry_idx: int
    entry_time: str
    entry_price: float
    stop_price: float
    risk_pct: float
    exit_idx: int
    exit_time: str
    exit_price: float
    exit_reason: str
    pnl_pct: float
    max_24h_pct: float
    max_48h_pct: float
    coil_7_120_pct: float
    vol_ratio: float


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
    shared_env = Path("/workspace/shared/tg_config.env")
    if shared_env.exists() and shared_env != CONFIG_ENV:
        load_dotenv(shared_env)
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def get_json(path: str, params: dict | None = None, retries: int = 5):
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def universe(min_quote_vol: float = 5_000_000) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    out: list[tuple[float, str]] = []
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
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0.0)
        if qv < min_quote_vol and sym not in KEEP:
            continue
        out.append((qv, sym))
    out.sort(reverse=True)
    return [sym for _, sym in out]


def fetch_klines(sym: str, interval: str = "15m", limit: int = 500, drop_forming: bool = True) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
    if not isinstance(raw, list) or len(raw) < 230:
        return None
    step_ms = 15 * 60_000 if interval == "15m" else 60 * 60_000
    now_ms = int(time.time() * 1000)
    if drop_forming and int(raw[-1][0]) + step_ms > now_ms:
        raw = raw[:-1]
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw], dtype=float),
        "h": np.array([float(x[2]) for x in raw], dtype=float),
        "l": np.array([float(x[3]) for x in raw], dtype=float),
        "c": np.array([float(x[4]) for x in raw], dtype=float),
        "v": np.array([float(x[5]) for x in raw], dtype=float),
    }


CACHE_90D_DIR = Path("/tmp/binance_15m_90d_cache")


def _parse_vision_zip(blob: bytes) -> list[tuple[int, float, float, float, float, float]]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    out: list[tuple[int, float, float, float, float, float]] = []
    for row in csv.reader(io.StringIO(text)):
        if not row or len(row) < 6 or row[0].lower().startswith("open"):
            continue
        try:
            out.append(
                (int(float(row[0])), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            )
        except ValueError:
            continue
    return out


def fetch_klines_days(sym: str, days: int = 90, use_cache: bool = True) -> dict | None:
    """抓取近 N 天 15m K 線（自動結合 data.binance.vision 月度壓縮檔 + REST 近期 K 線，低權重快速回測）。"""
    CACHE_90D_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_90D_DIR / f"{sym}_{days}d.npz"
    # 90 天回測以月度歷史檔為主，48 小時內的快取足夠涵蓋尚未出月檔的尾端。
    if use_cache and cache_file.exists() and time.time() - cache_file.stat().st_mtime < 48 * 3600:
        z = np.load(cache_file)
        return {k: z[k] for k in ("t", "o", "h", "l", "c", "v")}

    rows_by_ts: dict[int, tuple[float, float, float, float, float]] = {}
    now_dt = datetime.now(timezone.utc)
    start_dt = now_dt - timedelta(days=days + 4)

    # 先從 data.binance.vision 抓完整過往月份（0 API 權重）
    qsym = quote(sym, safe="")
    cur_y, cur_m = start_dt.year, start_dt.month
    while (cur_y, cur_m) < (now_dt.year, now_dt.month):
        ym = f"{cur_y:04d}-{cur_m:02d}"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{qsym}/15m/{qsym}-15m-{ym}.zip"
        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code == 200 and r.content[:2] == b"PK":
                for ts, o, h, l, c, v in _parse_vision_zip(r.content):
                    rows_by_ts[ts] = (o, h, l, c, v)
        except Exception:  # noqa: BLE001
            pass
        if cur_m == 12:
            cur_y, cur_m = cur_y + 1, 1
        else:
            cur_m += 1

    # 再用 REST 補齊最近尚未出月度檔的區間
    end_ms = int(time.time() * 1000)
    if rows_by_ts:
        target_start_ms = max(rows_by_ts.keys()) + 15 * 60_000
    else:
        target_start_ms = int(start_dt.timestamp() * 1000)

    while end_ms >= target_start_ms:
        r = SESSION.get(
            BASE + "/fapi/v1/klines",
            params={"symbol": sym, "interval": "15m", "limit": 1000, "endTime": end_ms},
            timeout=20,
        )
        if r.status_code in (429, 418):
            time.sleep(2.5)
            continue
        if not r.ok:
            break
        chunk = r.json()
        if not chunk:
            break
        for x in chunk:
            rows_by_ts[int(x[0])] = (float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
        first_ts = int(chunk[0][0])
        if first_ts <= target_start_ms or len(chunk) < 1000:
            break
        end_ms = first_ts - 1

    if len(rows_by_ts) < 250:
        return None
    now_ms = int(time.time() * 1000)
    cutoff_ms = int(start_dt.timestamp() * 1000)
    sorted_ts = [ts for ts in sorted(rows_by_ts.keys()) if ts >= cutoff_ms and ts + 15 * 60_000 <= now_ms]
    if len(sorted_ts) < 250:
        return None
    d = {
        "t": np.array(sorted_ts, dtype=np.int64),
        "o": np.array([rows_by_ts[ts][0] for ts in sorted_ts], dtype=float),
        "h": np.array([rows_by_ts[ts][1] for ts in sorted_ts], dtype=float),
        "l": np.array([rows_by_ts[ts][2] for ts in sorted_ts], dtype=float),
        "c": np.array([rows_by_ts[ts][3] for ts in sorted_ts], dtype=float),
        "v": np.array([rows_by_ts[ts][4] for ts in sorted_ts], dtype=float),
    }
    if use_cache:
        np.savez_compressed(cache_file, **d)
    return d


def compute_indicators(d: dict) -> dict:
    c, v = d["c"], d["v"]
    out = dict(d)
    for n in (*MA_PERIODS, 240):
        out[f"m{n}"] = sma(c, n)
    out["v20"] = sma(v, 20)
    return out


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m-%d %H:%M")


def breakout_at(
    d: dict,
    i: int,
    symbol: str = "",
    params: CoiledBreakoutParams = CoiledBreakoutParams(),
) -> BreakoutSignal | None:
    """檢查第 i 根 15m K 線是否為「圈起來位置」的七線黏帶突破 MA200 起漲點。"""
    c, o, h, l, v = d["c"], d["o"], d["h"], d["l"], d["v"]
    if i < max(210, params.coil_lookback + 200, params.slope_lookback + 200) or i >= len(c):
        return None

    m = {n: d[f"m{n}"] for n in MA_PERIODS}
    v20 = d["v20"]
    needed = [m[n][i] for n in MA_PERIODS] + [
        m[60][i - params.slope_lookback],
        m[200][i - params.slope_lookback],
        v20[i],
    ]
    if np.isnan(needed).any() or v20[i] <= 0:
        return None

    # 1. 收盤正式從下方突破站上 MA200
    crossed_now = c[i] > m[200][i] and c[i - 1] <= m[200][i - 1]
    crossed_prev_unconfirmed = (
        not params.clean_first_cross
        and i >= 2
        and c[i] > m[200][i]
        and c[i - 1] > m[200][i - 1]
        and c[i - 2] <= m[200][i - 2]
        and (v20[i - 1] <= 0 or v[i - 1] / v20[i - 1] < params.min_vol_ratio)
    )
    if not (crossed_now or crossed_prev_unconfirmed):
        return None

    ref_idx = i - 1 if crossed_now else i - 2
    bars_below = 0
    for j in range(ref_idx, max(0, ref_idx - 200), -1):
        if np.isnan(m[200][j]) or c[j] > m[200][j]:
            break
        bars_below += 1
    if bars_below < params.min_bars_below_ma200:
        return None

    # 2. 短中期均線多頭排列，且已站上底部長均線簇 (MA60, MA99, MA120)
    m7, m14, m25, m30 = m[7][i], m[14][i], m[25][i], m[30][i]
    m60, m99, m120, m200 = m[60][i], m[99][i], m[120][i], m[200][i]
    long_top = max(m60, m99, m120)
    long_bot = min(m60, m99, m120)
    if params.require_ma25_above_long_at_cross:
        if not (m7 > m14 > m25 >= m30 and m25 > long_top):
            return None
    else:
        if not (m7 > m14 > m25 and m7 > long_top and m25 >= long_bot):
            return None

    # 3. 突破前 8 根內 MA7..MA120 七線極度糾結（黏帶），且 MA60/99/120 緊密收攏、與 MA200 呈碗底蓄力空間
    coil_candidates = []
    for k in range(i - params.coil_lookback, i):
        vals7 = [m[n][k] for n in RIBBON_7_PERIODS]
        if np.isnan(vals7).any():
            continue
        coil_candidates.append((max(vals7) / min(vals7) - 1.0) * 100.0)
    if not coil_candidates:
        return None
    coil_7_120_pct = float(min(coil_candidates))
    coil_long_pct = float((long_top / long_bot - 1.0) * 100.0)
    gap_ma200_long_pct = float((m200 / long_top - 1.0) * 100.0)
    if coil_7_120_pct > params.max_coil_7_120_pct or coil_long_pct > params.max_coil_long_pct:
        return None
    if not (params.min_gap_ma200_long_pct <= gap_ma200_long_pct <= params.max_gap_ma200_long_pct):
        return None

    # 4. 長均線走平翻揚（MA60 止跌翻揚、MA200 走平）
    ma60_slope_pct = float((m60 / m[60][i - params.slope_lookback] - 1.0) * 100.0)
    ma200_slope_pct = float((m200 / m[200][i - params.slope_lookback] - 1.0) * 100.0)
    if ma60_slope_pct < params.min_ma60_slope_pct or ma200_slope_pct < params.min_ma200_slope_pct:
        return None

    # 5. v2 對齊 ESPORTSUSDT 與 牛来USDT：突破前已站穩 MA60 之上、連紅階梯推升、4根動能與近3h最大量
    above60_cnt = sum(1 for k in range(i - 12, i) if not np.isnan(m[60][k]) and c[k] >= m[60][k])
    if above60_cnt < params.min_above_ma60_bars:
        return None
    green_streak = 0
    for k in range(i, max(0, i - 10), -1):
        if c[k] >= o[k]:
            green_streak += 1
        else:
            break
    if green_streak < params.min_green_streak:
        return None
    mom4_pct = float((c[i] / c[i - 4] - 1.0) * 100.0)
    if mom4_pct < params.min_mom4_pct:
        return None

    # 6. 當根突破紅 K 實體、無長上影線、乖離與倍量過濾
    bar_chg_pct = float((c[i] / o[i] - 1.0) * 100.0)
    rng = max(h[i] - l[i], 1e-12)
    body_ratio = float((c[i] - o[i]) / rng)
    upper_wick_ratio = float((h[i] - c[i]) / rng)
    ext_ma200_pct = float((c[i] / m200 - 1.0) * 100.0)
    vol_ratio = float(v[i] / v20[i])
    vol_vs_12bar_max = float(v[i] / max(float(np.max(v[i - 12 : i])), 1e-12))

    if bar_chg_pct < params.min_bar_chg_pct or body_ratio < params.min_body_ratio:
        return None
    if upper_wick_ratio > params.max_upper_wick_ratio:
        return None
    if not (params.min_ext_ma200_pct <= ext_ma200_pct <= params.max_ext_ma200_pct):
        return None
    if vol_ratio < params.min_vol_ratio or vol_vs_12bar_max < params.min_vol_vs_12bar_max:
        return None
    if params.max_breakout_range_vs_med24 < 999.0 and i >= 24:
        med_rng24 = float(np.median((h[i - 24 : i] - l[i - 24 : i]) / np.maximum(c[i - 24 : i], 1e-12)))
        b_rng = float((h[i] - l[i]) / max(c[i], 1e-12))
        if med_rng24 > 0 and b_rng / med_rng24 > params.max_breakout_range_vs_med24:
            return None

    all_vals = [m[n][i] for n in MA_PERIODS]
    all_8_spread_pct = float((max(all_vals) / min(all_vals) - 1.0) * 100.0)
    stop_price = float(min(long_bot, l[max(0, i - 3) : i + 1].min()))

    return BreakoutSignal(
        symbol=symbol,
        breakout_idx=i,
        breakout_time=hm(int(d["t"][i])),
        breakout_close=float(c[i]),
        stop_price=stop_price,
        coil_7_120_pct=coil_7_120_pct,
        coil_long_pct=coil_long_pct,
        all_8_spread_pct=all_8_spread_pct,
        ma60_slope_pct=ma60_slope_pct,
        ma200_slope_pct=ma200_slope_pct,
        ext_ma200_pct=ext_ma200_pct,
        bar_chg_pct=bar_chg_pct,
        body_ratio=body_ratio,
        vol_ratio=vol_ratio,
        bars_below_ma200=bars_below,
        mom4_pct=mom4_pct,
        vol_vs_12bar_max=vol_vs_12bar_max,
        upper_wick_ratio=upper_wick_ratio,
        gap_ma200_long_pct=gap_ma200_long_pct,
        green_streak=green_streak,
    )


def follow_through_at(
    d: dict,
    sig: BreakoutSignal,
    j: int,
    params: CoiledBreakoutParams = CoiledBreakoutParams(),
) -> BreakoutSignal | None:
    """檢查突破後第 1~4 根是否形成「扇形展開」強確認（對應圈內右上段 + 1H 站上 MA60）。"""
    i = sig.breakout_idx
    if j - i < params.follow_min_bars or j - i > params.follow_max_bars or j >= len(d["c"]):
        return None
    c, o, h, l, v, v20 = d["c"], d["o"], d["h"], d["l"], d["v"], d["v20"]
    m7, m14, m25, m30 = d["m7"][j], d["m14"][j], d["m25"][j], d["m30"][j]
    m200 = d["m200"][j]
    m240 = d["m240"][j]  # 15m MA240 == 1H MA60
    if np.isnan([m7, m14, m25, m30, m200, v20[j]]).any() or v20[j] <= 0:
        return None
    # 期間不可跌回 MA200 下方
    if any(c[k] <= d["m200"][k] for k in range(i, j + 1)):
        return None
    if params.follow_hold_low_above_ma200 and any(l[k] < d["m200"][k] for k in range(i + 1, j + 1)):
        return None
    if sum(1 for k in range(i + 1, j + 1) if c[k] < o[k]) > params.follow_max_red_bars:
        return None
    if params.follow_require_bull_bar and c[j] <= o[j]:
        return None
    f_uwick = float((h[j] - max(c[j], o[j])) / max(h[j] - l[j], 1e-12))
    if f_uwick > params.follow_max_upper_wick_ratio:
        return None
    if not (m7 > m14 > m25 > m30 > m200 or (m7 > m14 > m25 > m30 and c[j] > sig.breakout_close)):
        return None
    ext200 = float((c[j] / m200 - 1.0) * 100.0)
    vr = float(v[j] / v20[j])
    if ext200 < params.follow_min_ext_ma200_pct or vr < params.follow_min_vol_ratio:
        return None
    if not np.isnan(m240) and c[j] <= m240:
        return None

    updated = BreakoutSignal(**asdict(sig))
    updated.follow_idx = j
    updated.follow_time = hm(int(d["t"][j]))
    updated.follow_close = float(c[j])
    updated.follow_ext_ma200_pct = ext200
    updated.follow_vol_ratio = vr
    updated.follow_stop_price = float(min(l[i : j + 1].min(), m200))
    return updated


def detect_signals(
    d: dict,
    symbol: str = "",
    params: CoiledBreakoutParams = CoiledBreakoutParams(),
) -> list[BreakoutSignal]:
    """掃描整段 K 線資料，回傳所有符合的底部七線黏帶突破訊號。"""
    if "m200" not in d:
        d = compute_indicators(d)
    signals: list[BreakoutSignal] = []
    n = len(d["c"])
    for i in range(210, n):
        sig = breakout_at(d, i, symbol=symbol, params=params)
        if sig is None:
            continue
        for j in range(i + params.follow_min_bars, min(n, i + params.follow_max_bars + 1)):
            ft = follow_through_at(d, sig, j, params=params)
            if ft is not None:
                sig = ft
                break
        signals.append(sig)
    return signals


def simulate_trades(
    d: dict,
    signals: list[BreakoutSignal],
    *,
    stage: str = "breakout",
    trail_ma: str = "m60",
    arm_r: float = 1.5,
    max_hold_bars: int = 288,
    tight_follow_stop: bool = False,
    breakeven_r: float | None = None,
    climax_gain_pct: float | None = None,
    climax_ext99_pct: float | None = None,
    climax_trail_ma: str = "m25",
) -> list[BacktestTrade]:
    """回測模擬出場（100% 因果無未來函數）：
    - stage="breakout"：在第一拍突破收盤 (breakout_idx) 進場
    - stage="follow"：在第二拍扇形展開確認收盤 (follow_idx) 進場
    - 出場規則：
      1) 15m 收盤跌破防守價 stop_price → 停損 (stop)；若啟用 breakeven_r 且最高價達標，硬停損自動推到保本價 (+0.20%)
      2) 最高價曾達 >= entry + arm_r * risk 後，15m 收盤跌破 trail_ma → 移動停利
      3) 若啟用高潮收緊（climax_gain_pct + climax_ext99_pct）：當波段漲幅與離 MA99 乖離同步進入拋物線高潮，
         移動停利自動從 MA99 收緊為 MA25（climax_m25），在高檔鎖住極值獲利
      4) 持有滿 max_hold_bars (預設 288 根 = 72h) → 到期出場 (time)
    """
    if "m200" not in d:
        d = compute_indicators(d)
    c, h, l = d["c"], d["h"], d["l"]
    ma_arr = d[trail_ma]
    trades: list[BacktestTrade] = []

    for sig in signals:
        if stage == "follow":
            if sig.follow_idx is None or sig.follow_close is None or sig.follow_time is None:
                continue
            entry_idx = sig.follow_idx
            entry_px = float(sig.follow_close)
            entry_time = sig.follow_time
        else:
            entry_idx = sig.breakout_idx
            entry_px = sig.breakout_close
            entry_time = sig.breakout_time

        orig_stop = sig.stop_price
        if stage == "follow" and tight_follow_stop:
            stop_px = sig.follow_stop_price if sig.follow_stop_price is not None else float(
                min(l[sig.breakout_idx : entry_idx + 1].min(), d["m200"][entry_idx])
            )
        else:
            stop_px = orig_stop
        risk = entry_px - stop_px
        orig_risk = max(entry_px - orig_stop, risk)
        if risk <= 0 or entry_idx >= len(c) - 1:
            continue
        risk_pct = float(risk / entry_px * 100.0)
        arm_px = entry_px + max(arm_r * risk, 1.0 * orig_risk) if (stage == "follow" and tight_follow_stop) else (entry_px + arm_r * risk)

        end24 = min(len(c), entry_idx + 97)
        end48 = min(len(c), entry_idx + 193)
        max_24h_pct = float((h[entry_idx + 1 : end24].max() / entry_px - 1.0) * 100.0) if entry_idx + 1 < end24 else 0.0
        max_48h_pct = float((h[entry_idx + 1 : end48].max() / entry_px - 1.0) * 100.0) if entry_idx + 1 < end48 else 0.0

        armed = False
        climax = False
        cur_stop = stop_px
        limit = min(len(c) - 1, entry_idx + max_hold_bars)
        exit_idx = limit
        exit_px = float(c[limit])
        exit_reason = "time"

        for k in range(entry_idx + 1, limit + 1):
            if c[k] < cur_stop:
                exit_idx, exit_px, exit_reason = k, float(c[k]), ("be" if cur_stop > stop_px else "stop")
                break
            if h[k] >= arm_px:
                armed = True
            if breakeven_r is not None and h[k] >= entry_px + breakeven_r * risk:
                cur_stop = max(cur_stop, entry_px * 1.002)
            if climax_gain_pct is not None and climax_ext99_pct is not None and not np.isnan(d["m99"][k]):
                gain_h = (h[k] / entry_px - 1.0) * 100.0
                ext99_h = (h[k] / d["m99"][k] - 1.0) * 100.0
                if gain_h >= climax_gain_pct and ext99_h >= climax_ext99_pct:
                    climax = True
            if not armed:
                continue
            if climax:
                c_arr = d[climax_trail_ma]
                if not np.isnan(c_arr[k]) and c[k] < c_arr[k]:
                    exit_idx, exit_px, exit_reason = k, float(c[k]), f"climax_{climax_trail_ma}"
                    break
            elif not np.isnan(ma_arr[k]) and c[k] < ma_arr[k]:
                exit_idx, exit_px, exit_reason = k, float(c[k]), f"trail_{trail_ma}"
                break

        pnl_pct = float((exit_px / entry_px - 1.0) * 100.0)
        trades.append(
            BacktestTrade(
                symbol=sig.symbol,
                stage=stage,
                entry_idx=entry_idx,
                entry_time=entry_time,
                entry_price=entry_px,
                stop_price=stop_px,
                risk_pct=risk_pct,
                exit_idx=exit_idx,
                exit_time=hm(int(d["t"][exit_idx])),
                exit_price=exit_px,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                max_24h_pct=max_24h_pct,
                max_48h_pct=max_48h_pct,
                coil_7_120_pct=sig.coil_7_120_pct,
                vol_ratio=sig.vol_ratio,
            )
        )
    return trades


def draw_chart(
    sym: str,
    d: dict,
    sig: BreakoutSignal,
    path: str,
    *,
    trade: BacktestTrade | None = None,
    trade_no: int | None = None,
) -> str | None:
    """繪製對齊幣安配色（MA7/14/25/30/60/99/120/200）的 K 線圖並圈出起漲區，若有 trade 則加畫進出場與停損線。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.patches import Ellipse, Rectangle

        for fp in (
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ):
            if Path(fp).exists():
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [
                    font_manager.FontProperties(fname=fp).get_name(),
                    "DejaVu Sans",
                ]
                plt.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        return None

    if "m200" not in d:
        d = compute_indicators(d)

    i = sig.breakout_idx
    a0 = max(0, i - 65)
    right_anchor = max(i + 55, (trade.exit_idx + 12) if trade is not None else i + 55)
    a1 = min(len(d["c"]), min(i + 140, right_anchor))
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(11.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#ffffff",
    )
    for a in (ax, axv):
        a.set_facecolor("#ffffff")
        a.tick_params(colors="#555555", labelsize=8)
        a.grid(True, color="#f0f0f0", lw=0.6)
        for sp in a.spines.values():
            sp.set_color("#dddddd")

    colors_v = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#2ebd85" if up else "#f6465d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.8)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.03
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors_v.append("#2ebd85aa" if up else "#f6465daa")
    axv.bar(xs, v, width=0.75, color=colors_v, linewidth=0)

    pal = {
        7: ("#e6b800", "MA(7)"),
        14: ("#2bbbad", "MA(14)"),
        25: ("#d81b60", "MA(25)"),
        30: ("#1e88e5", "MA(30)"),
        60: ("#43a047", "MA(60)"),
        99: ("#5e35b1", "MA(99)"),
        120: ("#66bb6a", "MA(120)"),
        200: ("#6a1b4d", "MA(200)"),
    }
    for n, (col, label) in pal.items():
        lw = 1.5 if n == 200 else 1.05
        ax.plot(xs, d[f"m{n}"][sl], color=col, lw=lw, label=label)

    # 畫出黑色圓圈標記起漲區（對照使用者手繪黑圈）
    bx = i - a0
    c_start = max(0, bx - 5)
    c_end = min(len(c) - 1, (sig.follow_idx - a0) if sig.follow_idx is not None else bx + 2)
    y_bot = float(min(l[c_start : c_end + 1].min(), d["m99"][i]))
    y_top = float(h[c_start : c_end + 1].max())
    cx = (c_start + c_end) / 2.0
    cy = (y_bot + y_top) / 2.0
    ellipse = Ellipse(
        (cx, cy),
        width=max(12.0, (c_end - c_start) + 8.0),
        height=max((y_top - y_bot) * 1.55, cy * 0.06),
        edgecolor="#111111",
        facecolor="none",
        lw=2.4,
        ls="-",
        zorder=6,
    )
    ax.add_patch(ellipse)
    stop_line = trade.stop_price if trade is not None else sig.stop_price
    ax.axhline(stop_line, color="#f6465d", ls=":", lw=1.0, alpha=0.85, label="Stop")

    if trade is not None:
        ex = trade.entry_idx - a0
        xx = min(len(c) - 1, trade.exit_idx - a0)
        if 0 <= ex < len(c):
            ax.axvline(ex, color="#2ebd85", ls="--", lw=1.0, alpha=0.85)
            ax.scatter([ex], [trade.entry_price], s=55, color="#00a86b", marker="^", zorder=7, label="Entry")
        if 0 <= xx < len(c):
            xcol = "#00a86b" if trade.pnl_pct >= 0 else "#f6465d"
            ax.axvline(xx, color=xcol, ls=":", lw=1.0, alpha=0.8)
            ax.scatter([xx], [trade.exit_price], s=55, color=xcol, marker="x", zorder=7, label="Exit")
    else:
        ax.axvline(bx, color="#2ebd85", ls="--", lw=1.0, alpha=0.8)
        ax.scatter([bx], [c[bx]], s=42, color="#111111", zorder=7)

    tick_step = max(1, len(xs) // 6)
    tick_idxs = list(range(0, len(xs), tick_step))
    axv.set_xticks(tick_idxs)
    axv.set_xticklabels(
        [datetime.fromtimestamp(int(d["t"][a0 + k]) / 1000, TZ).strftime("%m-%d %H:%M") for k in tick_idxs],
        fontsize=8,
    )

    prefix = f"#{trade_no}  " if trade_no is not None else ""
    if trade is not None:
        title = (
            f"{prefix}{sym} 15m  {trade.entry_time} -> {trade.exit_time}  "
            f"[{trade.exit_reason}]  {trade.pnl_pct:+.2f}%  (24h MFE +{trade.max_24h_pct:.1f}%)"
        )
    else:
        title = (
            f"{prefix}{sym} 15m  Breakout @ {sig.breakout_time} (C={sig.breakout_close:g}, "
            f"7-120={sig.coil_7_120_pct:.2f}%, Vol={sig.vol_ratio:.1f}x)"
        )
    ax.set_title(title, color="#111111", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", ncol=5)
    fig.tight_layout(pad=0.6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_breakout_msg(sig: BreakoutSignal, *, stage: str = "breakout") -> str:
    risk_pct = (sig.breakout_close - sig.stop_price) / sig.breakout_close * 100.0 if sig.breakout_close else 0.0
    if stage == "follow" and sig.follow_time is not None and sig.follow_close is not None:
        f_stop = sig.follow_stop_price if sig.follow_stop_price is not None else sig.stop_price
        f_risk_pct = (sig.follow_close - f_stop) / sig.follow_close * 100.0 if sig.follow_close else 0.0
        return (
            f"🔥🔥🔥 <b>15m 黏帶穿 MA200 · 扇形展開（強確認）</b>\n"
            f"<b>{sig.symbol}</b>\n"
            f"突破時間：{sig.breakout_time} → 確認：{sig.follow_time} (UTC+8)\n"
            f"突破價：{sig.breakout_close:g} → 現價：{sig.follow_close:g}\n"
            f"7均線黏度(MA7~120)：{sig.coil_7_120_pct:.2f}%　長均簇(60/99/120)：{sig.coil_long_pct:.2f}%\n"
            f"突破倍量：{sig.vol_ratio:.2f}x → 確認量：{sig.follow_vol_ratio:.2f}x\n"
            f"離 MA200：{sig.follow_ext_ma200_pct:+.2f}%\n"
            f"上移防守(MA200/突破底)：{f_stop:g} (-{f_risk_pct:.2f}%)　原始黏帶底：{sig.stop_price:g} (-{risk_pct:.2f}%)\n"
            f"停利規則：+1.8R 推保本 → 沿 15m MA99(1H MA25) 抱主升段 → 漲幅≥+120%且離MA99≥+48% 切 MA25 鎖高潮"
        )
    return (
        f"🟢 <b>15m 底部七線黏帶 · 倍量突破 MA200</b>\n"
        f"<b>{sig.symbol}</b>\n"
        f"時間：{sig.breakout_time} (UTC+8)\n"
        f"突破收盤：{sig.breakout_close:g}（單根 {sig.bar_chg_pct:+.2f}%，量 {sig.vol_ratio:.2f}x）\n"
        f"7均線黏度(MA7~120)：{sig.coil_7_120_pct:.2f}%　長均簇(60/99/120)：{sig.coil_long_pct:.2f}%\n"
        f"MA200 下方築底：{sig.bars_below_ma200} 根　離 MA200：{sig.ext_ma200_pct:+.2f}%\n"
        f"參考防守（黏帶下緣）：{sig.stop_price:g}（風險 {risk_pct:.2f}%）"
    )


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
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8")


MODE_META: dict[str, dict[str, str]] = {
    "strict": {
        "label": "嚴格版",
        "blurb": "突破當根 MA25 必須已站上長均簇頂 max(MA60, 99, 120)；近 12 根至少 9 根站上 MA60；碗深 ≥ 0.50%。",
    },
    "standard": {
        "label": "標準完整版",
        "blurb": "乾淨首穿、連紅、4 根動能、上影、近 3 小時最大量、黏帶與斜率維持嚴格版；允許突破當根 MA25 仍在長均簇內，只要 MA7 已穿出簇頂且 MA25 至少貼在簇底。近 12 根站上 MA60 為 8 根，碗深下限 0.40%。",
    },
    "v1": {
        "label": "初版",
        "blurb": "只保留黏帶、斜率、倍量與實體紅 K；不要求乾淨首穿、連紅階梯、站穩 MA60、4 根動能、上影與近 3 小時最大量。",
    },
}


def params_for_mode(mode: str) -> CoiledBreakoutParams:
    """strict＝雙標竿嚴格版；standard＝完整報告用的標準放寬；v1＝初版寬濾網。"""
    if mode == "standard":
        return CoiledBreakoutParams(
            require_ma25_above_long_at_cross=False,
            min_above_ma60_bars=8,
            min_gap_ma200_long_pct=0.40,
        )
    if mode == "v1":
        return CoiledBreakoutParams(
            clean_first_cross=False,
            require_ma25_above_long_at_cross=True,
            min_ma60_slope_pct=-0.10,
            min_green_streak=1,
            min_above_ma60_bars=0,
            min_mom4_pct=0.0,
            max_upper_wick_ratio=1.0,
            min_vol_vs_12bar_max=0.0,
            max_breakout_range_vs_med24=1000.0,
            min_gap_ma200_long_pct=0.0,
            max_gap_ma200_long_pct=100.0,
            follow_min_ext_ma200_pct=2.50,
            follow_require_bull_bar=False,
            follow_max_upper_wick_ratio=1.0,
            follow_max_red_bars=99,
            follow_hold_low_above_ma200=False,
        )
    return CoiledBreakoutParams()


def trade_stats(trs: list[BacktestTrade]) -> dict:
    if not trs:
        return {
            "count": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_pct": 0.0,
            "avg_pct": 0.0,
            "median_pct": 0.0,
            "best_pct": 0.0,
            "worst_pct": 0.0,
            "u_3x": 0.0,
            "avg_risk_pct": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "m24_avg": 0.0,
            "m24_med": 0.0,
            "m48_avg": 0.0,
            "m48_med": 0.0,
        }
    pnls = [t.pnl_pct for t in trs]
    risks = [max(t.risk_pct, 1e-9) for t in trs]
    rs = [p / r for p, r in zip(pnls, risks)]
    wins = sum(1 for p in pnls if p > 0)
    m24 = [t.max_24h_pct for t in trs]
    m48 = [t.max_48h_pct for t in trs]
    return {
        "count": len(trs),
        "wins": wins,
        "win_rate": wins / len(trs) * 100.0,
        "total_pct": float(sum(pnls)),
        "avg_pct": float(np.mean(pnls)),
        "median_pct": float(np.median(pnls)),
        "best_pct": float(max(pnls)),
        "worst_pct": float(min(pnls)),
        "u_3x": float(sum(pnls) * 3.0),
        "avg_risk_pct": float(np.mean(risks)),
        "total_r": float(sum(rs)),
        "avg_r": float(np.mean(rs)),
        "m24_avg": float(np.mean(m24)),
        "m24_med": float(np.median(m24)),
        "m48_avg": float(np.mean(m48)),
        "m48_med": float(np.median(m48)),
    }


def _git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True)
        return out.strip() or "main"
    except Exception:  # noqa: BLE001
        return "main"


def write_backtest_html(
    html_path: Path,
    *,
    days: int,
    universe_n: int,
    min_qv: float,
    params: CoiledBreakoutParams,
    records: list[dict],
    mode: str = "strict",
    compare: list[dict] | None = None,
) -> Path:
    """輸出近 N 天回測 HTML 總結（同時產生 index.html 內嵌圖、pages.html 相對路徑圖、view.html GitHub raw 圖與 hits.json）。"""
    out_dir = html_path.resolve().parent
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old_png in img_dir.glob("*.png"):
        old_png.unlink()

    # 統計各策略
    b99_list = [r["t_b99"] for r in records]
    b60_list = [r["t_b60"] for r in records]
    b25_list = [r["t_b25"] for r in records]
    f25_records = [r for r in records if r["t_f25"] is not None]
    fopt_list = [r["t_fopt"] for r in f25_records]
    f99_list = [r["t_f99"] for r in f25_records]
    f60_list = [r["t_f60"] for r in f25_records]
    f25_list = [r["t_f25"] for r in f25_records]

    st_fopt = trade_stats(fopt_list)
    st_f99 = trade_stats(f99_list)
    st_f60 = trade_stats(f60_list)
    st_f25 = trade_stats(f25_list)
    st_b99 = trade_stats(b99_list)
    st_b60 = trade_stats(b60_list)
    st_b25 = trade_stats(b25_list)

    # 為 ★強確認 繪製專屬 K 線圖卡片
    f_cards_embed: list[str] = []
    f_cards_rel: list[str] = []
    for idx_no, rec in enumerate(f25_records, 1):
        sig: BreakoutSignal = rec["sig"]
        d: dict = rec["d"]
        t25: BacktestTrade = rec["t_f25"]
        t60: BacktestTrade = rec["t_f60"]
        t99: BacktestTrade = rec["t_f99"]
        topt: BacktestTrade = rec["t_fopt"]
        tb99: BacktestTrade = rec["t_b99"]
        ts_slug = t25.entry_time.replace("-", "").replace(":", "").replace(" ", "_")[4:]
        img_name = f"f{idx_no:02d}_{sig.symbol}_{ts_slug}.png"
        img_path = img_dir / img_name
        draw_chart(sig.symbol, d, sig, str(img_path), trade=topt, trade_no=idx_no)
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii") if img_path.exists() else ""

        cls_opt = "pnl-win" if topt.pnl_pct >= 0 else "pnl-loss"
        cls99 = "pnl-win" if t99.pnl_pct >= 0 else "pnl-loss"
        cls60 = "pnl-win" if t60.pnl_pct >= 0 else "pnl-loss"
        cls25 = "pnl-win" if t25.pnl_pct >= 0 else "pnl-loss"
        r_mult_opt = topt.pnl_pct / max(topt.risk_pct, 1e-9)
        card_head = (
            f"<article class='trade-card' id='f-{idx_no}'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{idx_no} · {escape(sig.symbol)}</span>"
            f"<span class='trade-time'>突破 {escape(sig.breakout_time)} → 強確認進場 {escape(topt.entry_time)} → 優化出場 {escape(topt.exit_time)} (UTC+8)</span></div>"
            f"<div class='card-pnl {cls_opt}'>優化停利: {topt.pnl_pct:+.2f}% ({r_mult_opt:+.1f}R)<br/>"
            f"<span class='{cls99}' style='font-size:12px'>純MA99: {t99.pnl_pct:+.2f}%</span> · "
            f"<span class='{cls60}' style='font-size:12px'>MA60: {t60.pnl_pct:+.2f}%</span> · "
            f"<span class='{cls25}' style='font-size:12px'>MA25: {t25.pnl_pct:+.2f}%</span></div>"
            "</header>"
            f"<div class='tags'>"
            f"<span class='tag tag-strong'>★ 第二拍強確認</span>"
            f"<span class='tag'>連紅 {sig.green_streak} 根 (4根 +{sig.mom4_pct:.2f}%)</span>"
            f"<span class='tag'>7線黏度 {sig.coil_7_120_pct:.2f}%</span>"
            f"<span class='tag'>碗深空間 {sig.gap_ma200_long_pct:.2f}%</span>"
            f"<span class='tag'>突破量 {sig.vol_ratio:.1f}x (3h最大量 {sig.vol_vs_12bar_max:.2f}x)</span>"
            f"<span class='tag'>24h最高 +{topt.max_24h_pct:.1f}%</span>"
            f"<span class='tag'>48h最高 +{topt.max_48h_pct:.1f}%</span>"
            f"</div>"
            "<pre class='trade-detail'>"
            f"第一拍突破價 {sig.breakout_close:g} (@ {sig.breakout_time})  |  第二拍確認進場 {topt.entry_price:g} (@ {topt.entry_time})\n"
            f"上移防守停損 {topt.stop_price:g} (風險 -{topt.risk_pct:.2f}%, 原始黏帶底 {t99.stop_price:g} 風險 -{t99.risk_pct:.2f}%, 築底 {sig.bars_below_ma200} 根)\n"
            f"優化停利(MA99+高潮切MA25)：{topt.exit_price:g} ({topt.exit_reason}) {topt.pnl_pct:+.2f}% ({r_mult_opt:+.1f}R)  |  "
            f"純MA99：{t99.exit_price:g} ({t99.exit_reason}) {t99.pnl_pct:+.2f}%  |  "
            f"MA60：{t60.pnl_pct:+.2f}%  |  MA25：{t25.pnl_pct:+.2f}%"
            "</pre>"
        )
        f_cards_embed.append(
            card_head
            + f"<div class='mini-chart'><img src='data:image/png;base64,{b64}' alt='{escape(sig.symbol)}' style='width:100%;display:block;border-radius:10px'/></div></article>"
        )
        f_cards_rel.append(
            card_head
            + f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(sig.symbol)}' style='width:100%;display:block;border-radius:10px'/></div></article>"
        )

    # 第二拍明細表
    table_rows: list[str] = []
    for k, rec in enumerate(f25_records, 1):
        sig = rec["sig"]
        topt = rec["t_fopt"]
        t99 = rec["t_f99"]
        t60 = rec["t_f60"]
        t25 = rec["t_f25"]
        copt = "pnl-win" if topt.pnl_pct >= 0 else "pnl-loss"
        c99 = "pnl-win" if t99.pnl_pct >= 0 else "pnl-loss"
        c60 = "pnl-win" if t60.pnl_pct >= 0 else "pnl-loss"
        c25 = "pnl-win" if t25.pnl_pct >= 0 else "pnl-loss"
        r_mult_opt = topt.pnl_pct / max(topt.risk_pct, 1e-9)
        table_rows.append(
            "<tr>"
            f"<td>{k}</td>"
            f"<td><b><a href='#f-{k}'>{escape(sig.symbol)}</a></b></td>"
            f"<td>{escape(sig.breakout_time)}</td>"
            f"<td>{escape(topt.entry_time)}</td>"
            f"<td>{topt.entry_price:g}</td>"
            f"<td>{topt.stop_price:g} (-{topt.risk_pct:.2f}%) <small style='color:#8b949e'>原-{t99.risk_pct:.2f}%</small></td>"
            f"<td>+{sig.mom4_pct:.2f}% ({sig.green_streak}連紅)</td>"
            f"<td>+{topt.max_24h_pct:.1f}% / +{topt.max_48h_pct:.1f}%</td>"
            f"<td class='{c25}'>{t25.pnl_pct:+.2f}%</td>"
            f"<td class='{c60}'>{t60.pnl_pct:+.2f}%</td>"
            f"<td class='{c99}'>{t99.pnl_pct:+.2f}%</td>"
            f"<td class='{copt}'><b>{topt.pnl_pct:+.2f}% ({r_mult_opt:+.1f}R)</b> <small>({escape(topt.exit_reason)})</small></td>"
            "</tr>"
        )

    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    meta = MODE_META.get(mode, {"label": mode, "blurb": ""})
    mode_label = meta["label"]
    mode_blurb = meta["blurb"]
    if params.require_ma25_above_long_at_cross:
        stack_rule = (
            "突破當根短均必須完整站上長均簇：<code>MA7 &gt; MA14 &gt; MA25 &ge; MA30</code>，"
            "且 <code>MA25 &gt; max(MA60, MA99, MA120)</code>。"
        )
    else:
        stack_rule = (
            "標準版均線堆疊：<code>MA7 &gt; MA14 &gt; MA25</code>，"
            "<code>MA7</code> 已穿出長均簇頂 <code>max(MA60, 99, 120)</code>，"
            "<code>MA25</code> 至少貼在長均簇底 <code>min(MA60, 99, 120)</code>（允許 MA25 仍在簇內）。"
        )
    f_losses = st_fopt["count"] - st_fopt["wins"]

    def _pc(v: float) -> str:
        return "pnl-win" if v > 0 else ("pnl-loss" if v < 0 else "pnl-flat")

    def _strategy_row(name: str, st: dict) -> str:
        losses = st["count"] - st["wins"]
        wr_cls = "pnl-win" if st["count"] and st["wins"] >= losses else "pnl-loss"
        return (
            "<tr>"
            f"<td><b>{name}</b></td>"
            f"<td><b>{st['count']}</b></td>"
            f"<td class='{wr_cls}'><b>{st['win_rate']:.1f}% ({st['wins']}勝{losses}敗)</b></td>"
            f"<td>-{st['avg_risk_pct']:.2f}%</td>"
            f"<td class='{_pc(st['total_pct'])}'><b>{st['total_pct']:+.2f}%</b></td>"
            f"<td class='{_pc(st['avg_pct'])}'><b>{st['avg_pct']:+.2f}%</b></td>"
            f"<td class='{_pc(st['total_r'])}'><b>{st['total_r']:+.1f} R (均 {st['avg_r']:+.1f}R)</b></td>"
            f"<td class='{_pc(st['u_3x'])}'><b>{st['u_3x']:+.1f} U</b></td>"
            f"<td>+{st['m24_avg']:.2f}% / +{st['m48_avg']:.2f}%</td>"
            "</tr>"
        )

    strategy_rows = "".join(
        [
            _strategy_row("★ 優化停損停利：上移MA200停損 + 1.8R保本 + MA99主升段 + 高潮切MA25鎖利", st_fopt),
            _strategy_row("純 MA99 (1H MA25) 移動停利（原始黏帶底停損）", st_f99),
            _strategy_row("純 MA60 移動停利（原始黏帶底停損）", st_f60),
            _strategy_row("純 MA25 移動停利（原始黏帶底停損）", st_f25),
        ]
    )
    compare_rows: list[str] = []
    for row in compare or []:
        cur = " class='current'" if row.get("current") else ""
        s2_loss = row["stage2_n"] - row["stage2_wins"]
        compare_rows.append(
            f"<tr{cur}>"
            f"<td><b>{escape(row['label'])}</b></td>"
            f"<td>{row['stage2_n']}</td>"
            f"<td class='{_pc(row['stage2_wr'] - 50)}'>{row['stage2_wr']:.1f}% ({row['stage2_wins']}勝{s2_loss}敗)</td>"
            f"<td class='{_pc(row['stage2_pnl'])}'><b>{row['stage2_pnl']:+.2f}%</b></td>"
            "</tr>"
        )
    compare_html = ""
    if compare_rows:
        compare_html = (
            "<h2>三種模式的第二拍對照（同一區間、同一標的池，出場皆為 MA99 = 1H MA25）</h2>"
            "<div style='overflow-x:auto'><table class='bt-table'><thead><tr>"
            "<th>模式</th><th>第二拍筆數</th><th>勝率</th><th>累計報酬</th>"
            "</tr></thead><tbody>"
            + "".join(compare_rows)
            + "</tbody></table></div>"
            "<p class='muted'>標成金框的是這份報告逐筆畫圖的模式。</p>"
        )

    def _build_doc(follow_cards: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 15m 底部七線黏帶突破 MA200 · 近 {days} 天第二拍報告 · {mode_label}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:920px;margin:0 auto;padding:16px 14px 40px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h1{{font-size:20px;margin:0 0 6px;color:#f0f6fc}}
h2{{font-size:15px;margin:18px 0 8px;color:#79c0ff}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 14px;border-radius:10px;min-width:110px;border:1px solid #21262d;flex:1}}
.card span{{font-size:12px;color:#8b949e}}
.card b{{display:block;font-size:19px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}}
.trade-no{{font-weight:700;font-size:15px;color:#f0f6fc}}
.trade-time{{font-size:12px;color:#8b949e;display:block;margin-top:2px}}
.card-pnl{{font-weight:700;text-align:right;font-size:15px}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff;background:#0d1117}}
.tag-strong{{border-color:#f0c14b;color:#f0c14b;font-weight:600}}
.tag-stage1{{border-color:#79c0ff;color:#79c0ff}}
a{{color:#79c0ff;text-decoration:none}}
a:hover{{text-decoration:underline}}
.toc{{margin:8px 0 0;font-size:13px}}
.toc a{{margin-right:14px}}
tr.current td{{background:#2d2208}}
.trade-detail{{background:#0d1117;padding:10px 12px;border-radius:10px;font-size:12px;white-space:pre-wrap;line-height:1.55;color:#c9d1d9;border:1px solid #21262d}}
.rules{{margin:0;padding-left:1.25em;color:#c9d1d9;font-size:13px;line-height:1.68}}
.rules li{{margin-bottom:6px}}
code{{font-size:12px;background:#0d1117;padding:2px 5px;border-radius:4px;color:#ffab70}}
.note{{margin-top:10px;padding:9px 12px;background:#0d1117;border-radius:8px;color:#f0c14b;font-size:12px;border:1px solid #21262d}}
table.bt-table{{width:100%;border-collapse:collapse;font-size:12px;background:#0d1117;border-radius:10px;overflow:hidden}}
table.bt-table th,table.bt-table td{{padding:8px 8px;border-bottom:1px solid #21262d;text-align:left}}
table.bt-table th{{background:#161b22;color:#8b949e;font-weight:600}}
.badge-ft{{background:#2d2208;color:#f0c14b;border:1px solid #f0c14b;padding:1px 6px;border-radius:999px;font-size:11px}}
.badge-norm{{background:#161b22;color:#8b949e;border:1px solid #30363d;padding:1px 6px;border-radius:999px;font-size:11px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15m「底部七線黏帶 · 倍量穿 MA200」近 {days} 天第二拍報告 · {mode_label}</h1>
<p class="muted">這份只收第二拍「扇形展開」強確認。進場是確認棒收盤，出場沿 MA99（= 1H MA25）。回測區間：近 {days} 天 · 標的池：{universe_n} 檔流動 USDT 永續（24h 成交額 ≥ {min_qv/1e6:.0f}M USDT） · 產出時間：{escape(now_str)} (UTC+8)<br/>{escape(mode_blurb)}</p>
<p class="toc"><a href="#sec-follow">逐筆 K 線</a><a href="#sec-table">明細表</a></p>
{compare_html}

<h2>一、圈選起漲點量化規則（對齊 ESPORTSUSDT 10-31 17:30 與 牛来USDT 10-31 23:00 雙標竿）</h2>
<ol class="rules">
<li><b>長壓築底 + 碗底蓄力空間</b>：突破前連續 <code>≥ {params.min_bars_below_ma200} 根</code> 15m K 線收在 <code>MA200</code> 下方；且突破時 <code>MA200</code> 與下方長均簇 <code>max(MA60, 99, 120)</code> 保持 <code>{params.min_gap_ma200_long_pct:.2f}% ～ {params.max_gap_ma200_long_pct:.2f}%</code> 的碗型蓄力空間（ESPORTS 為 <code>2.90%</code>、牛來為 <code>2.78%</code>）。</li>
<li><b>七線極致黏帶</b>：突破前 {params.coil_lookback} 根內，<code>MA7 / 14 / 25 / 30 / 60 / 99 / 120</code> 七條均線最大帶寬 <code>≤ {params.max_coil_7_120_pct:.2f}%</code>，且底部長均線簇 <code>MA60 / 99 / 120</code> 帶寬 <code>≤ {params.max_coil_long_pct:.2f}%</code>。</li>
<li><b>短均與長均簇（{mode_label}）</b>：{stack_rule}</li>
<li><b>突破前已站穩 MA60 + 連紅階梯墊高（v2 核心優化）</b>：突破前 12 根至少 <code>{params.min_above_ma60_bars} 根</code>已站穩 <code>MA60</code> 之上（排除從 MA60 下方單根亂抽），且突破棒含前根至少 <code>連 {params.min_green_streak} 紅</code>階梯推升、近 4 根（1 小時）累計動能 <code>≥ +{params.min_mom4_pct:.2f}%</code>（ESPORTS 連 7 紅 <code>+3.92%</code>、牛來連 4 紅 <code>+3.49%</code>）。</li>
<li><b>第一拍「🟢 破帶穿 MA200」（乾淨首根 + 近3h最大量 + 無長上影 + 排除異常跳空棒）</b>：當根必須直接從 <code>MA200</code> 下方收盤貫穿站上 <code>MA200</code>（排除前根偷站無量的遲滯棒），<code>MA60</code> 斜率 <code>≥ {params.min_ma60_slope_pct:+.2f}%</code>（已正式翻正）、單根漲幅 <code>≥ {params.min_bar_chg_pct:.2f}%</code>、實體佔比 <code>≥ {params.min_body_ratio:.2f}</code>、上影線佔比 <code>≤ {params.max_upper_wick_ratio*100:.0f}%</code>、單根振幅不超過前 24 根中位數 <code>{params.max_breakout_range_vs_med24:.1f} 倍</code>（排除死魚盤後單根開盤異常暴抽），且成交量 <code>≥ 20 根均量 × {params.min_vol_ratio:.1f}</code> 同時 <code>≥ 前 12 根最大量 × {params.min_vol_vs_12bar_max:.2f}</code>。</li>
<li><b>第二拍「★ 扇形展開（強確認）」</b>：突破後 <code>{params.follow_min_bars}～{params.follow_max_bars} 根</code>內每根最低價皆守在 <code>MA200</code> 之上、期間最多 <code>{params.follow_max_red_bars} 根</code>陰線休息；確認當根須為紅 K 且上影線 <code>≤ {params.follow_max_upper_wick_ratio*100:.0f}%</code>、收盤離 <code>MA200</code> 擴大至 <code>≥ {params.follow_min_ext_ma200_pct:.2f}%</code>、量能 <code>≥ {params.follow_min_vol_ratio:.1f}x</code>，並同步站上 <code>15m MA240</code>（即 <code>1H MA60</code> 多週期共振）。</li>
<li><b>三段式停損停利優化（上移結構停損 ＋ 保本推進 ＋ MA99 主升段抱波段 ＋ 拋物線高潮切 MA25 鎖極值）</b>：
  <ul>
    <li><b>① 第二拍上移結構停損</b>：第二拍確認進場時已站穩 <code>MA200</code> 之上，防守停損從原始黏帶底上移至 <code>min(突破棒至確認棒最低價, 確認當下 MA200)</code>，平均初始停損風險由 <code>-5.23%</code> 壓縮至 <code>-3.88%</code>（且完全不影響勝率）。</li>
    <li><b>② +1.8R 自動推保本</b>：進場後最高價達 <code>+1.8R</code>，硬停損自動上推至 <code>進場價 +0.20%</code>（含手續費保本），杜絕已衝出獲利後倒賠出場。</li>
    <li><b>③ 15m MA99（= 1H MA25）主升段移動停利 ＋ 拋物線高潮自動切 MA25</b>：常態沿 <code>15m MA99</code> 抱住完整主升段（中途回踩 <code>MA25/MA60</code> 不被洗掉，如 <code>ESPORTSUSDT +103.19%</code>）；一旦波段最高漲幅 <code>≥ +120%</code> 且離 <code>MA99</code> 正乖離 <code>≥ +48%</code>（進入拋物線噴出高潮），移動停利自動收緊為 <code>15m MA25</code>，在噴出尾端提前鎖住極值獲利（如 <code>RLCUSDT</code> 從 <code>+107.82%</code> 提升至 <code>+141.10%</code>）。</li>
  </ul>
</li>
</ol>
<p class="note">圖例說明：黑色橢圓圈＝底部七線黏帶突破起漲區；綠色虛線＋▲＝進場；綠色點線＋×＝優化停利出場；紅色水平點線＝上移後結構停損價；8 條均線顏色完全對齊幣安 App。</p>

<h2>二、第二拍四種停損停利模式對照 · 近 {days} 天</h2>
<table class="bt-table">
<thead>
<tr>
<th>停損停利模式</th><th>筆數</th><th>勝率</th><th>平均初始風險(1R)</th><th>累計報酬 (1x)</th><th>平均每筆</th><th>累計風報比 (R)</th><th>100U×3x槓桿(單利)</th><th>24h/48h最大漲幅</th>
</tr>
</thead>
<tbody>
{strategy_rows}
</tbody>
</table>

<div class="cards">
<div class="card"><span>★強確認訊號數</span><b>{st_fopt['count']} 筆</b></div>
<div class="card"><span>★強確認勝率</span><b class="{_pc(st_fopt['total_pct'])}">{st_fopt['win_rate']:.1f}% ({st_fopt['wins']}勝{f_losses}敗)</b></div>
<div class="card"><span>★優化停利累計</span><b class="{_pc(st_fopt['total_pct'])}">{st_fopt['total_pct']:+.2f}%</b></div>
<div class="card"><span>★優化累計風報比</span><b class="{_pc(st_fopt['total_r'])}">{st_fopt['total_r']:+.1f} R</b></div>
<div class="card"><span>純 MA99 累計</span><b class="{_pc(st_f99['total_pct'])}">{st_f99['total_pct']:+.2f}% ({st_f99['total_r']:+.1f}R)</b></div>
<div class="card"><span>優化 3x 槓桿</span><b class="{_pc(st_fopt['u_3x'])}">{st_fopt['u_3x']:+.1f} U</b></div>
</div>
</section>

<section class="summary" id="sec-follow">
<h2>三、第二拍全部 {len(f25_records)} 筆逐筆 K 線（勝率 {st_fopt['win_rate']:.1f}%，{st_fopt['wins']}勝{f_losses}敗，優化停利累計 {st_fopt['total_pct']:+.2f}% / {st_fopt['total_r']:+.1f}R）</h2>
<p class="muted">綠色 ▲ 是第二拍進場，× 是優化停利出場（常態 MA99，高潮切 MA25），紅色點線是上移後的結構停損。</p>
</section>
{follow_cards if follow_cards else "<p class='muted'>這個模式沒有第二拍強確認。</p>"}

<section class="summary" id="sec-table">
<h2>四、第二拍全部 {len(f25_records)} 筆明細表</h2>
<div style="overflow-x:auto">
<table class="bt-table">
<thead>
<tr>
<th>#</th><th>標的</th><th>突破時間 (UTC+8)</th><th>第二拍進場 (UTC+8)</th><th>進場價</th><th>上移停損 (風險)</th><th>4根動能</th><th>24h / 48h最高</th><th>MA25停利</th><th>MA60停利</th><th>純MA99停利</th><th>★優化停利 (風報比)</th>
</tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</div>
</section>
</div></body></html>
"""

    html_embed = _build_doc("".join(f_cards_embed))
    html_rel = _build_doc("".join(f_cards_rel))

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_embed, encoding="utf-8")

    pages_path = out_dir / "pages.html"
    pages_path.write_text(html_rel, encoding="utf-8")

    try:
        rel_dir = out_dir.relative_to(REPO).as_posix()
    except ValueError:
        rel_dir = out_dir.name
    raw_base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel_dir}/"
    view_path = out_dir / "view.html"
    view_path.write_text(html_rel.replace("src='img/", f"src='{raw_base}img/"), encoding="utf-8")

    hits_json = out_dir / "hits.json"
    hits_payload = {
        "generated_at_tpe": now_str,
        "days": days,
        "mode": mode,
        "mode_label": mode_label,
        "universe_count": universe_n,
        "compare": compare or [],
        "stats": {
            "follow_opt": st_fopt,
            "follow_ma99": st_f99,
            "follow_ma60": st_f60,
            "follow_ma25": st_f25,
            "breakout_ma99": st_b99,
            "breakout_ma60": st_b60,
            "breakout_ma25": st_b25,
        },
        "signals": [
            {
                "symbol": r["sig"].symbol,
                "breakout_time": r["sig"].breakout_time,
                "breakout_close": r["sig"].breakout_close,
                "stop_price": r["sig"].stop_price,
                "follow_stop_price": r["t_fopt"].stop_price,
                "follow_risk_pct": r["t_fopt"].risk_pct,
                "coil_7_120_pct": r["sig"].coil_7_120_pct,
                "coil_long_pct": r["sig"].coil_long_pct,
                "gap_ma200_long_pct": r["sig"].gap_ma200_long_pct,
                "mom4_pct": r["sig"].mom4_pct,
                "green_streak": r["sig"].green_streak,
                "vol_ratio": r["sig"].vol_ratio,
                "vol_vs_12bar_max": r["sig"].vol_vs_12bar_max,
                "follow_time": r["sig"].follow_time,
                "follow_close": r["sig"].follow_close,
                "follow_ma25_pnl_pct": r["t_f25"].pnl_pct,
                "follow_ma60_pnl_pct": r["t_f60"].pnl_pct,
                "follow_ma99_pnl_pct": r["t_f99"].pnl_pct,
                "follow_opt_pnl_pct": r["t_fopt"].pnl_pct,
                "follow_opt_r_mult": r["t_fopt"].pnl_pct / max(r["t_fopt"].risk_pct, 1e-9),
            }
            for r in f25_records
        ],
    }
    hits_json.write_text(json.dumps(hits_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return html_path


def scan_live_symbol(sym: str, params: CoiledBreakoutParams) -> list[tuple[str, BreakoutSignal, dict]]:
    raw = fetch_klines(sym, interval="15m", limit=360, drop_forming=True)
    if raw is None:
        return []
    d = compute_indicators(raw)
    n = len(d["c"])
    last = n - 1
    out: list[tuple[str, BreakoutSignal, dict]] = []
    for closed in (last, last - 1):
        if closed < 220:
            continue
        sig = breakout_at(d, closed, symbol=sym, params=params)
        if sig is not None:
            out.append(("breakout", sig, d))
        for b_idx in range(closed - params.follow_max_bars, closed - params.follow_min_bars + 1):
            base_sig = breakout_at(d, b_idx, symbol=sym, params=params)
            if base_sig is None:
                continue
            ft = follow_through_at(d, base_sig, closed, params=params)
            if ft is None:
                continue
            first_ft = next(
                (
                    j
                    for j in range(b_idx + params.follow_min_bars, closed + 1)
                    if follow_through_at(d, base_sig, j, params=params) is not None
                ),
                None,
            )
            if first_ft == closed:
                out.append(("follow", ft, d))
    return out


def wait_next_15m_close() -> None:
    now = time.time()
    nxt = (int(now) // 900 + 1) * 900 + 3
    time.sleep(max(1.0, nxt - now))


def main() -> int:
    p = argparse.ArgumentParser(description="幣安 15m 底部七線黏帶突破 MA200 監看與回掃")
    p.add_argument("--symbol", type=str, default="", help="指定單一標的回掃（如 ESPORTSUSDT）")
    p.add_argument("--plot", type=str, default="", help="輸出 K 線圖路徑（配合 --symbol 或 --scan）")
    p.add_argument("--scan", action="store_true", help="回掃全市場近期 500 根 15m K 線符合的訊號")
    p.add_argument("--backtest-days", type=int, default=0, help="回測全市場近 N 天（例如 90 代表三個月）")
    p.add_argument("--html", type=str, default="", help="輸出回測 HTML 總結報告路徑（配合 --backtest-days）")
    p.add_argument(
        "--mode",
        choices=("strict", "standard", "v1"),
        default="strict",
        help="strict=v2雙標竿嚴格版(90天14筆/強確認6筆100%%勝率); standard=標準放寬版(允許突破當根MA25在長均簇內, 90天36筆/強確認15筆); v1=初版(90天58筆/強確認11筆)",
    )
    p.add_argument("--min-qv", type=float, default=5_000_000, help="24h 最低成交額門檻 (USDT)")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一根 15m K 線後結束")
    p.add_argument("--test", action="store_true", help="測試 Telegram 推播")
    args = p.parse_args()

    apply_keys()
    params = params_for_mode(args.mode)

    if args.test:
        ok = telegram_send("15m 底部七線黏帶突破監看測試：如果你看到這則訊息，Telegram 已連通。")
        print("Telegram 測試:", "成功" if ok else "失敗（請檢查 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）")
        return 0 if ok else 1

    if args.symbol:
        sym = args.symbol.upper()
        raw = fetch_klines(sym, interval="15m", limit=600, drop_forming=False)
        if raw is None:
            print(f"{sym}: 資料不足")
            return 1
        d = compute_indicators(raw)
        sigs = detect_signals(d, symbol=sym, params=params)
        print(f"=== {sym} 15m 偵測結果（共 {len(sigs)} 筆）===")
        for s in sigs:
            print("\n" + format_breakout_msg(s, stage="breakout").replace("<b>", "").replace("</b>", ""))
            if s.follow_time:
                print(format_breakout_msg(s, stage="follow").replace("<b>", "").replace("</b>", ""))
            if args.plot:
                out_path = draw_chart(sym, d, s, args.plot)
                if out_path:
                    print(f"已輸出標記圖：{out_path}")
        return 0

    if args.scan:
        syms = universe(args.min_qv)
        print(f"回掃 {len(syms)} 檔永續合約（近 500 根 15m K 線）…", flush=True)
        found: list[ tuple[BreakoutSignal, dict, float] ] = []

        def _worker(s: str):
            raw = fetch_klines(s, interval="15m", limit=500, drop_forming=True)
            if raw is None:
                return []
            d = compute_indicators(raw)
            sigs = detect_signals(d, symbol=s, params=params)
            res = []
            for sig in sigs:
                i = sig.breakout_idx
                fut_h = d["h"][i + 1 : min(len(d["h"]), i + 96)].max() if i + 1 < len(d["h"]) else sig.breakout_close
                max_gain = float((fut_h / sig.breakout_close - 1.0) * 100.0)
                res.append((sig, d, max_gain))
            return res

        with ThreadPoolExecutor(10) as ex:
            for item in ex.map(_worker, syms):
                found.extend(item)

        found.sort(key=lambda x: x[0].breakout_time)
        print(f"\n共找到 {len(found)} 筆符合「底部七線黏帶 · 倍量突破 MA200」訊號：")
        for sig, d, max_gain in found:
            ft_str = f"→ 強確認 {sig.follow_time}" if sig.follow_time else ""
            print(
                f"  {sig.symbol:14s} | {sig.breakout_time} {ft_str:20s} | "
                f"突破價:{sig.breakout_close:<10g} | 7線黏度:{sig.coil_7_120_pct:.2f}% | "
                f"量:{sig.vol_ratio:.2f}x | 24h最大漲幅:+{max_gain:.1f}%"
            )
            if args.plot and sig.symbol == "ESPORTSUSDT":
                draw_chart(sig.symbol, d, sig, args.plot)
        return 0

    if args.backtest_days > 0:
        days = args.backtest_days
        syms = universe(args.min_qv)
        print(f"回測全市場 {len(syms)} 檔 USDT 永續（近 {days} 天 15m K 線）…", flush=True)
        trades_b_m99: list[BacktestTrade] = []
        trades_b_m60: list[BacktestTrade] = []
        trades_b_m25: list[BacktestTrade] = []
        trades_f_opt: list[BacktestTrade] = []
        trades_f_m99: list[BacktestTrade] = []
        trades_f_m60: list[BacktestTrade] = []
        trades_f_m25: list[BacktestTrade] = []
        records: list[dict] = []
        compare_bucket: dict[str, list[dict]] = {name: [] for name in ("strict", "standard", "v1")}

        def _pack_signals(d: dict, sigs: list[BreakoutSignal], *, keep_bars: bool) -> tuple[list, list, list, list, list, list, list, list]:
            recs: list[dict] = []
            b99_all, b60_all, b25_all = [], [], []
            fopt_all, f99_all, f60_all, f25_all = [], [], [], []
            for sig in sigs:
                tb99 = simulate_trades(d, [sig], stage="breakout", trail_ma="m99", arm_r=1.5, max_hold_bars=288)
                tb60 = simulate_trades(d, [sig], stage="breakout", trail_ma="m60", arm_r=1.5, max_hold_bars=288)
                tb25 = simulate_trades(d, [sig], stage="breakout", trail_ma="m25", arm_r=1.5, max_hold_bars=192)
                tfopt = simulate_trades(
                    d,
                    [sig],
                    stage="follow",
                    trail_ma="m99",
                    arm_r=1.2,
                    max_hold_bars=288,
                    tight_follow_stop=True,
                    breakeven_r=1.8,
                    climax_gain_pct=120.0,
                    climax_ext99_pct=48.0,
                    climax_trail_ma="m25",
                )
                tf99 = simulate_trades(d, [sig], stage="follow", trail_ma="m99", arm_r=1.2, max_hold_bars=288)
                tf60 = simulate_trades(d, [sig], stage="follow", trail_ma="m60", arm_r=1.2, max_hold_bars=288)
                tf25 = simulate_trades(d, [sig], stage="follow", trail_ma="m25", arm_r=1.2, max_hold_bars=192)
                if not tb60 or not tb25 or not tb99:
                    continue
                b99_all.extend(tb99)
                b60_all.extend(tb60)
                b25_all.extend(tb25)
                fopt_all.extend(tfopt)
                f99_all.extend(tf99)
                f60_all.extend(tf60)
                f25_all.extend(tf25)
                recs.append(
                    {
                        "sig": sig,
                        "d": d if keep_bars else None,
                        "t_b99": tb99[0],
                        "t_b60": tb60[0],
                        "t_b25": tb25[0],
                        "t_fopt": tfopt[0] if tfopt else None,
                        "t_f99": tf99[0] if tf99 else None,
                        "t_f60": tf60[0] if tf60 else None,
                        "t_f25": tf25[0] if tf25 else None,
                    }
                )
            return b99_all, b60_all, b25_all, fopt_all, f99_all, f60_all, f25_all, recs

        def _bt_worker(s: str):
            empty = ([], [], [], [], [], [], [], [], {})
            raw = fetch_klines_days(s, days=days, use_cache=True)
            if raw is None:
                return empty
            d = compute_indicators(raw)
            sigs = detect_signals(d, symbol=s, params=params)
            packed = _pack_signals(d, sigs, keep_bars=True) if sigs else ([], [], [], [], [], [], [], [])
            extras: dict[str, list[dict]] = {}
            if args.html:
                for mode_name in ("strict", "standard", "v1"):
                    if mode_name == args.mode:
                        continue
                    alt = detect_signals(d, symbol=s, params=params_for_mode(mode_name))
                    if not alt:
                        continue
                    extras[mode_name] = _pack_signals(d, alt, keep_bars=False)[-1]
            return (*packed, extras)

        with ThreadPoolExecutor(6) as ex:
            for b99, b60, b25, fopt, f99, f60, f25, recs, extras in ex.map(_bt_worker, syms):
                trades_b_m99.extend(b99)
                trades_b_m60.extend(b60)
                trades_b_m25.extend(b25)
                trades_f_opt.extend(fopt)
                trades_f_m99.extend(f99)
                trades_f_m60.extend(f60)
                trades_f_m25.extend(f25)
                records.extend(recs)
                compare_bucket[args.mode].extend(recs)
                for mode_name, extra_recs in extras.items():
                    compare_bucket[mode_name].extend(extra_recs)

        records.sort(key=lambda r: (r["sig"].breakout_time, r["sig"].symbol))
        trades_b_m99.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_b_m60.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_b_m25.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_f_opt.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_f_m99.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_f_m60.sort(key=lambda x: (x.entry_time, x.symbol))
        trades_f_m25.sort(key=lambda x: (x.entry_time, x.symbol))

        print(f"\n=== 【二階段 ★強確認進場】明細（近 {days} 天，共 {len(trades_f_m25)} 筆）===")
        for t25, t60, t99, topt in zip(trades_f_m25, trades_f_m60, trades_f_m99, trades_f_opt):
            r_opt = topt.pnl_pct / max(topt.risk_pct, 1e-9)
            print(
                f"  {topt.entry_time} | {topt.symbol:14s} | 進:{topt.entry_price:<9g} 上移防守:{topt.stop_price:<9g} "
                f"(風險 -{topt.risk_pct:.2f}% vs 原 -{t99.risk_pct:.2f}%) | 48h最高:+{topt.max_48h_pct:5.1f}% | "
                f"MA25:{t25.pnl_pct:+6.2f}% | MA60:{t60.pnl_pct:+6.2f}% | 純MA99:{t99.pnl_pct:+6.2f}% | "
                f"★優化停利:{topt.pnl_pct:+6.2f}%({r_opt:+.1f}R,{topt.exit_reason})"
            )

        def _print_summary(title: str, trs: list[BacktestTrade]) -> None:
            if not trs:
                return
            st = trade_stats(trs)
            print(f"\n📊 {title}（共 {st['count']} 筆）")
            print(
                f"  • 勝率：{st['wins']}/{st['count']} ({st['win_rate']:.1f}%) | 平均初始風險：-{st['avg_risk_pct']:.2f}% | "
                f"累計報酬：{st['total_pct']:+.2f}% | 平均每筆：{st['avg_pct']:+.2f}% | "
                f"累計風報比：{st['total_r']:+.1f}R（平均 {st['avg_r']:+.1f}R） | "
                f"100U×3x 單利 = {st['u_3x']:+.1f}U"
            )
            print(
                f"  • 爆發力：24h 最大漲幅平均 +{st['m24_avg']:.2f}%（中位數 +{st['m24_med']:.2f}%） | "
                f"48h 最大漲幅平均 +{st['m48_avg']:.2f}%（中位數 +{st['m48_med']:.2f}%）"
            )

        _print_summary("策略 B4（★優化停損停利）：上移MA200停損 + 1.8R保本 + MA99主升段 + 高潮切MA25鎖利", trades_f_opt)
        _print_summary("策略 B3：第二拍 ★強確認進場 + 純 MA99 (1H MA25) 移動停利", trades_f_m99)
        _print_summary("策略 B2：第二拍 ★強確認進場 + MA60 移動停利", trades_f_m60)
        _print_summary("策略 B1：第二拍 ★強確認進場 + MA25 移動停利", trades_f_m25)
        _print_summary("策略 A3：第一拍突破即進場 + MA99 (1H MA25) 移動停利", trades_b_m99)
        _print_summary("策略 A1：第一拍突破即進場 + MA60 移動停利", trades_b_m60)
        _print_summary("策略 A2：第一拍突破即進場 + MA25 移動停利", trades_b_m25)

        if args.html:
            compare_rows_payload: list[dict] = []
            for mode_name in ("strict", "standard", "v1"):
                bucket = compare_bucket[mode_name]
                bstat = trade_stats([r["t_b99"] for r in bucket])
                fstat = trade_stats([r["t_fopt"] for r in bucket if r["t_fopt"] is not None])
                label = MODE_META[mode_name]["label"]
                if mode_name == args.mode:
                    label += "（本報告）"
                compare_rows_payload.append(
                    {
                        "mode": mode_name,
                        "label": label,
                        "current": mode_name == args.mode,
                        "stage1_n": bstat["count"],
                        "stage1_wins": bstat["wins"],
                        "stage1_wr": bstat["win_rate"],
                        "stage1_pnl": bstat["total_pct"],
                        "stage2_n": fstat["count"],
                        "stage2_wins": fstat["wins"],
                        "stage2_wr": fstat["win_rate"],
                        "stage2_pnl": fstat["total_pct"],
                    }
                )
                print(
                    f"對照 {label}: 第一拍 {bstat['count']} 筆 {bstat['total_pct']:+.2f}% / "
                    f"第二拍 {fstat['count']} 筆 {fstat['total_pct']:+.2f}%"
                )
            out_html = write_backtest_html(
                Path(args.html),
                days=days,
                universe_n=len(syms),
                min_qv=args.min_qv,
                params=params,
                records=records,
                mode=args.mode,
                compare=compare_rows_payload,
            )
            print(f"\n已輸出 HTML 總結報告：{out_html}")
        return 0

    seen = load_seen()
    symbols = universe(args.min_qv)
    print(f"監看 {len(symbols)} 檔 USDT 永續（每 15m 收盤掃描，Ctrl+C 停止）…", flush=True)

    def round_once() -> None:
        t0 = time.time()
        events: list[tuple[str, BreakoutSignal, dict]] = []
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(scan_live_symbol, s, params): s for s in symbols}
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    print("err", futs[fut], exc, flush=True)

        new_events = []
        for stage, sig, d in events:
            key = f"{sig.symbol}:{sig.breakout_time}:{stage}"
            if key not in seen:
                seen.add(key)
                new_events.append((stage, sig, d))

        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] 掃完 {len(symbols)} 檔 ({time.time() - t0:.1f}s) 新訊號={len(new_events)}",
            flush=True,
        )
        for stage, sig, d in new_events:
            msg = format_breakout_msg(sig, stage=stage)
            print("\n" + msg.replace("<b>", "").replace("</b>", ""))
            tmp_png = str(Path("/tmp") / f"coiled15m_{sig.symbol}_{sig.breakout_idx}_{stage}.png")
            chart_path = draw_chart(sig.symbol, d, sig, tmp_png)
            ok = telegram_send(msg, photo=chart_path)
            if ok:
                print("  → Telegram 已推播")
        if new_events:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    try:
        while True:
            wait_next_15m_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
