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

  # 持續監看每根 15m 收盤並推播 Telegram（含 K 線圖）
  python3 examples/watch_binance_15m_coiled_breakout.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    min_ma60_slope_pct: float = -0.10
    min_ma200_slope_pct: float = -1.00
    min_ext_ma200_pct: float = 0.10
    max_ext_ma200_pct: float = 3.50
    min_bar_chg_pct: float = 0.80
    min_body_ratio: float = 0.55
    min_vol_ratio: float = 2.00
    follow_min_bars: int = 1
    follow_max_bars: int = 4
    follow_min_ext_ma200_pct: float = 2.50
    follow_min_vol_ratio: float = 2.00


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
    follow_idx: int | None = None
    follow_time: str | None = None
    follow_close: float | None = None
    follow_ext_ma200_pct: float | None = None
    follow_vol_ratio: float | None = None


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

    # 1. 收盤正式從下方突破站上 MA200（允許前一根微幅探頭但未放量，當根放量正式突破）
    crossed_now = c[i] > m[200][i] and c[i - 1] <= m[200][i - 1]
    crossed_prev_unconfirmed = (
        i >= 2
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
    if not (m7 > m14 > m25 >= m30 and m25 > long_top):
        return None

    # 3. 突破前 8 根內 MA7..MA120 七線極度糾結（黏帶），且 MA60/99/120 緊密收攏
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
    if coil_7_120_pct > params.max_coil_7_120_pct or coil_long_pct > params.max_coil_long_pct:
        return None

    # 4. 長均線走平翻揚（MA60 止跌翻揚、MA200 走平）
    ma60_slope_pct = float((m60 / m[60][i - params.slope_lookback] - 1.0) * 100.0)
    ma200_slope_pct = float((m200 / m[200][i - params.slope_lookback] - 1.0) * 100.0)
    if ma60_slope_pct < params.min_ma60_slope_pct or ma200_slope_pct < params.min_ma200_slope_pct:
        return None

    # 5. 當根突破紅 K 實體、乖離與倍量過濾
    bar_chg_pct = float((c[i] / o[i] - 1.0) * 100.0)
    rng = max(h[i] - l[i], 1e-12)
    body_ratio = float((c[i] - o[i]) / rng)
    ext_ma200_pct = float((c[i] / m200 - 1.0) * 100.0)
    vol_ratio = float(v[i] / v20[i])

    if bar_chg_pct < params.min_bar_chg_pct or body_ratio < params.min_body_ratio:
        return None
    if not (params.min_ext_ma200_pct <= ext_ma200_pct <= params.max_ext_ma200_pct):
        return None
    if vol_ratio < params.min_vol_ratio:
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
    c, v, v20 = d["c"], d["v"], d["v20"]
    m7, m14, m25, m30 = d["m7"][j], d["m14"][j], d["m25"][j], d["m30"][j]
    m200 = d["m200"][j]
    m240 = d["m240"][j]  # 15m MA240 == 1H MA60
    if np.isnan([m7, m14, m25, m30, m200, v20[j]]).any() or v20[j] <= 0:
        return None
    # 期間不可跌回 MA200 下方
    if any(c[k] <= d["m200"][k] for k in range(i, j + 1)):
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


def draw_chart(sym: str, d: dict, sig: BreakoutSignal, path: str) -> str | None:
    """繪製對齊幣安配色（MA7/14/25/30/60/99/120/200）的 K 線圖並圈出起漲區。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle
    except Exception:
        return None

    if "m200" not in d:
        d = compute_indicators(d)

    i = sig.breakout_idx
    a0 = max(0, i - 65)
    a1 = min(len(d["c"]), i + 55)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(11.2, 6.4),
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

    # 對齊幣安 App 均線顏色
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
    ax.axvline(bx, color="#2ebd85", ls="--", lw=1.0, alpha=0.8)
    ax.scatter([bx], [c[bx]], s=42, color="#111111", zorder=7)

    # X 軸時間刻度
    tick_step = max(1, len(xs) // 6)
    tick_idxs = list(range(0, len(xs), tick_step))
    axv.set_xticks(tick_idxs)
    axv.set_xticklabels(
        [datetime.fromtimestamp(int(d["t"][a0 + k]) / 1000, TZ).strftime("%m-%d %H:%M") for k in tick_idxs],
        fontsize=8,
    )

    title = (
        f"{sym} 15m  Breakout @ {sig.breakout_time} (C={sig.breakout_close:g}, "
        f"7-120={sig.coil_7_120_pct:.2f}%, Vol={sig.vol_ratio:.1f}x)"
    )
    ax.set_title(title, color="#111111", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", ncol=4)
    fig.tight_layout(pad=0.6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_breakout_msg(sig: BreakoutSignal, *, stage: str = "breakout") -> str:
    risk_pct = (sig.breakout_close - sig.stop_price) / sig.breakout_close * 100.0 if sig.breakout_close else 0.0
    if stage == "follow" and sig.follow_time is not None:
        return (
            f"🔥🔥🔥 <b>15m 黏帶穿 MA200 · 扇形展開（強確認）</b>\n"
            f"<b>{sig.symbol}</b>\n"
            f"突破時間：{sig.breakout_time} → 確認：{sig.follow_time} (UTC+8)\n"
            f"突破價：{sig.breakout_close:g} → 現價：{sig.follow_close:g}\n"
            f"7均線黏度(MA7~120)：{sig.coil_7_120_pct:.2f}%　長均簇(60/99/120)：{sig.coil_long_pct:.2f}%\n"
            f"突破倍量：{sig.vol_ratio:.2f}x → 確認量：{sig.follow_vol_ratio:.2f}x\n"
            f"離 MA200：{sig.follow_ext_ma200_pct:+.2f}%　參考防守：{sig.stop_price:g} (-{risk_pct:.2f}%)"
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
    p.add_argument("--min-qv", type=float, default=5_000_000, help="24h 最低成交額門檻 (USDT)")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一根 15m K 線後結束")
    p.add_argument("--test", action="store_true", help="測試 Telegram 推播")
    args = p.parse_args()

    apply_keys()
    params = CoiledBreakoutParams()

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
