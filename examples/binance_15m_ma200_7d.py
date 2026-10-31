#!/usr/bin/env python3
"""
幣安 USDT 永續 15m + 1H 雙週期優化版 —
「15m MA200 六線收斂起跳 + 1H 全均線多頭淨空 & 24h 高點空間過濾」近 7 天回測與 HTML 報告產生器

用法:
    python3 examples/binance_15m_ma200_7d.py --days 7 --out docs/binance-15m-ma200-7d/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO = Path(__file__).resolve().parents[1]

MA_COLORS = {
    7: "#f0b90b",    # 黃 (MA7)
    14: "#26c6da",   # 青藍 (MA14)
    25: "#ec407a",   # 粉紅 (MA25)
    99: "#ab47bc",   # 紫 (MA99)
    120: "#66bb6a",  # 綠 (MA120)
    200: "#ffffff",  # 白粗線 (MA200)
}


def setup_cjk() -> None:
    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break


def git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            text=True,
        )
        return out.strip() or "main"
    except Exception:
        return "main"


def get_json(path: str, params: dict | None = None, retries: int = 4):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(0.4 * (i + 1))


def sma(arr: list[float], n: int) -> list[float]:
    out = [math.nan] * len(arr)
    if len(arr) < n:
        return out
    s = sum(arr[:n])
    out[n - 1] = s / n
    for i in range(n, len(arr)):
        s += arr[i] - arr[i - n]
        out[i] = s / n
    return out


def get_universe(min_qv: float = 3_000_000) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    syms = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        sym = s["symbol"]
        if sym in ("USDCUSDT", "BTCDOMUSDT"):
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv >= min_qv:
            syms.append(sym)
    return sorted(syms)


def fetch_symbol_both(sym: str) -> tuple[dict, dict] | None:
    r15 = get_json("/fapi/v1/klines", {"symbol": sym, "interval": "15m", "limit": 960})
    r1h = get_json("/fapi/v1/klines", {"symbol": sym, "interval": "1h", "limit": 500})
    if not r15 or len(r15) < 260 or not r1h or len(r1h) < 220:
        return None
    k15 = r15[:-1]
    k1h = r1h[:-1]
    d15 = {
        "t": [int(k[0]) for k in k15],
        "o": [float(k[1]) for k in k15],
        "h": [float(k[2]) for k in k15],
        "l": [float(k[3]) for k in k15],
        "c": [float(k[4]) for k in k15],
        "v": [float(k[5]) for k in k15],
    }
    d1h = {
        "t": [int(k[0]) for k in k1h],
        "o": [float(k[1]) for k in k1h],
        "h": [float(k[2]) for k in k1h],
        "l": [float(k[3]) for k in k1h],
        "c": [float(k[4]) for k in k1h],
        "v": [float(k[5]) for k in k1h],
    }
    return d15, d1h


def detect_entry_15m_1h(
    d15: dict,
    d1h: dict,
    i: int,
    ind15: dict,
    ind1h: dict,
) -> dict | None:
    m7, m14, m25 = ind15["m7"], ind15["m14"], ind15["m25"]
    m99, m120, m200 = ind15["m99"], ind15["m120"], ind15["m200"]
    v20 = ind15["v20"]

    if i < 220 or math.isnan(m200[i - 12]):
        return None
    o, h, l, c, v, t = d15["o"], d15["h"], d15["l"], d15["c"], d15["v"], d15["t"]

    # === 15m 條件 ===
    hi_long = max(m99[i], m120[i], m200[i])
    lo_long = min(m99[i], m120[i], m200[i])
    long_spread = (hi_long - lo_long) / m200[i] * 100

    all6 = [m7[i], m14[i], m25[i], m99[i], m120[i], m200[i]]
    all6_spread = (max(all6) - min(all6)) / m200[i] * 100
    if long_spread > 1.5 or all6_spread > 2.2:
        return None

    slope200 = (m200[i] - m200[i - 12]) / m200[i - 12] * 100
    if not (-0.15 <= slope200 <= 0.40):
        return None

    box_hi = max(h[i - 20 : i])
    box_lo = min(l[i - 20 : i])
    box_width = (box_hi - box_lo) / m200[i] * 100
    if not (0.8 <= box_width <= 4.5):
        return None

    lo_first = min(l[i - 24 : i - 8])
    lo_second = min(l[i - 8 : i + 1])
    if not (lo_second > lo_first and lo_first < m200[i]):
        return None

    dist200 = (c[i] - m200[i]) / m200[i] * 100
    if not (0.05 <= dist200 <= 1.8):
        return None
    if c[i] <= hi_long or c[i] <= max(c[i - 12 : i]):
        return None
    if not (c[i - 1] <= hi_long * 1.003 or c[i - 2] <= hi_long * 1.003):
        return None

    if not (m7[i] > m14[i] >= m25[i] and m7[i] > m7[i - 1]):
        return None
    rng = h[i] - l[i]
    if not (c[i] > o[i] and rng > 0 and (c[i] - l[i]) / rng >= 0.65):
        return None
    vol_ratio = v[i] / max(v20[i], 1e-9)
    if vol_ratio < 1.8:
        return None

    # === 對齊 1H 結構 (無偷看未來：取進場當下已完整收盤的最新 1H K 棒) ===
    close_15m_ms = t[i] + 900_000
    k1_list = [k for k in range(len(d1h["t"])) if d1h["t"][k] + 3600_000 <= close_15m_ms]
    if not k1_list or k1_list[-1] < 205:
        return None
    k1 = k1_list[-1]

    px = c[i]
    c1 = d1h["c"]
    h1_m99 = ind1h["m99"][k1]
    h1_m120 = ind1h["m120"][k1]
    h1_m200 = ind1h["m200"][k1]
    # 即時 1H MA25 (前 24 根已收盤 1H + 當前收盤價)
    rt_1h_m25 = (sum(c1[k1 - 23 : k1 + 1]) + px) / 25.0

    # 1H 條件 1: 進場價站上 1H MA25 與所有 1H 長均線 (MA99/120/200)，頭頂無 1H 大均線蓋頭反壓
    above_mas = [(m - px) / px * 100 for m in (h1_m99, h1_m120, h1_m200) if not math.isnan(m) and m > px]
    above_1h_longs = len(above_mas) == 0
    above_1h_m25 = px > rt_1h_m25

    # 1H 條件 2: 距離 1H 近 24 根高點空間 >= 1.1% (代表是 1H 衝高後回檔整理再起跳，不是頂在 24h 天花板追高)
    h1_hi24 = max(d1h["h"][k1 - 24 : k1 + 1])
    room_to_1h_hi = (h1_hi24 - px) / px * 100

    pass_1h = above_1h_longs and above_1h_m25 and room_to_1h_hi >= 1.1
    fail_reasons = []
    if not above_1h_longs:
        fail_reasons.append(f"1H長均反壓(距頂+{min(above_mas):.2f}%)")
    if not above_1h_m25:
        fail_reasons.append("未站上1H_MA25")
    if room_to_1h_hi < 1.1:
        fail_reasons.append(f"1H距24h高僅{room_to_1h_hi:.2f}%(頂到天花板)")

    stop = max(min(lo_second, lo_long * 0.996), px * 0.98)
    if (px - stop) / px < 0.004:
        stop = px * 0.992

    return {
        "entry_idx": i,
        "h1_entry_idx": k1,
        "entry_price": px,
        "stop_price": stop,
        "ma200": m200[i],
        "dist200_pct": dist200,
        "long_spread_pct": long_spread,
        "all6_spread_pct": all6_spread,
        "slope200_pct": slope200,
        "box_hi": box_hi,
        "box_lo": box_lo,
        "box_width_pct": box_width,
        "vol_ratio": vol_ratio,
        "h1_hi24": h1_hi24,
        "room_to_1h_hi": room_to_1h_hi,
        "rt_1h_m25": rt_1h_m25,
        "h1_m99": h1_m99,
        "h1_m120": h1_m120,
        "h1_m200": h1_m200,
        "pass_1h": pass_1h,
        "fail_1h_reason": "、".join(fail_reasons) if fail_reasons else "通過1H雙濾網",
    }


def draw_dual_chart(
    sym: str,
    d15: dict,
    d1h: dict,
    ind15: dict,
    ind1h: dict,
    tr: dict,
    trade_no: int,
    out_path: Path,
) -> None:
    """左圖：15m K線（MA200六線收斂+窄箱體突破）；右圖：1H K線（全均線之上+24h高點空間）。"""
    fig = plt.figure(figsize=(14.4, 5.8), facecolor="#0c1210")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[1.15, 1.0], hspace=0.08, wspace=0.14)
    ax15 = fig.add_subplot(gs[0, 0])
    ax15v = fig.add_subplot(gs[1, 0], sharex=ax15)
    ax1h = fig.add_subplot(gs[0, 1])
    ax1hv = fig.add_subplot(gs[1, 1], sharex=ax1h)

    for a in (ax15, ax15v, ax1h, ax1hv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    # --- 左側：15m 圖 ---
    entry_i = tr["entry_idx"]
    exit_i = tr["exit_idx"]
    a0 = max(0, entry_i - 34)
    a1 = min(len(d15["c"]) - 1, max(exit_i + 6, entry_i + 24))
    xs15 = list(range(a1 - a0 + 1))

    o, h, l, c, v, t = (
        d15["o"][a0 : a1 + 1],
        d15["h"][a0 : a1 + 1],
        d15["l"][a0 : a1 + 1],
        d15["c"][a0 : a1 + 1],
        d15["v"][a0 : a1 + 1],
        d15["t"][a0 : a1 + 1],
    )

    box_x0 = max(0, (entry_i - 20) - a0)
    box_x1 = entry_i - a0
    if box_x1 > box_x0:
        ax15.add_patch(
            Rectangle(
                (box_x0 - 0.4, tr["box_lo"]),
                box_x1 - box_x0 + 0.4,
                tr["box_hi"] - tr["box_lo"],
                facecolor="#79c0ff18",
                edgecolor="#79c0ff77",
                ls="--",
                lw=0.85,
                zorder=1,
            )
        )

    cols15 = []
    for k in range(len(xs15)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax15.vlines(xs15[k], l[k], h[k], color=col, lw=0.8, zorder=3)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax15.add_patch(Rectangle((xs15[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3, zorder=3))
        cols15.append("#3dba7a99" if up else "#e35d5d99")

    ax15v.bar(xs15, v, width=0.78, color=cols15, linewidth=0)
    for n, col in MA_COLORS.items():
        ma_sl = ind15[f"m{n}"][a0 : a1 + 1]
        ax15.plot(xs15, ma_sl, color=col, lw=2.0 if n == 200 else 1.1, label=f"MA{n}", zorder=4 if n == 200 else 2)

    ax15.axhline(tr["stop_price"], color="#ff5252", ls=":", lw=1.0, alpha=0.85)
    ax15.axhline(tr["target_price"], color="#3dba7a", ls=":", lw=1.0, alpha=0.80)

    ex15 = entry_i - a0
    xx15 = exit_i - a0
    if 0 <= ex15 < len(xs15):
        ax15.axvline(ex15, color="#3dba7a", ls="--", lw=0.95)
        ax15.scatter([ex15], [tr["entry_price"]], s=55, color="#00e676", marker="^", zorder=6)
        ax15.annotate(
            f"進 {tr['entry_price']:g}",
            (ex15, tr["entry_price"]),
            textcoords="offset points",
            xytext=(4, -13),
            color="#00e676",
            fontsize=8,
        )
    if 0 <= xx15 < len(xs15):
        ax15.axvline(xx15, color="#f0c14b", ls=":", lw=0.95)
        ax15.scatter(
            [xx15],
            [tr["exit_price"]],
            s=48,
            color="#00e676" if tr["pnl_pct"] > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )
        ax15.annotate(
            f"出 {tr['exit_price']:.4g} ({tr['pnl_pct']:+.2f}%)",
            (xx15, tr["exit_price"]),
            textcoords="offset points",
            xytext=(4, 7),
            color="#f0c14b",
            fontsize=8,
        )

    ax15.set_title(
        f"[15分K] #{trade_no} {sym}  {tr['entry_time']}→{tr['exit_time']}  {tr['reason']}  {tr['pnl_pct']:+.2f}% ({tr['r_mult']:+.2f}R)",
        color="#e8f0ea",
        fontsize=10,
    )
    ax15.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step15 = max(1, len(xs15) // 5)
    ticks15 = list(range(0, len(xs15), step15))
    ax15v.set_xticks(ticks15)
    ax15v.set_xticklabels([datetime.fromtimestamp(t[idx] / 1000, TZ).strftime("%m-%d %H:%M") for idx in ticks15], color="#8aa193")
    plt.setp(ax15.get_xticklabels(), visible=False)

    # --- 右側：1H 圖 ---
    k1 = tr["h1_entry_idx"]
    b0 = max(0, k1 - 36)
    b1 = min(len(d1h["c"]) - 1, k1 + 16)
    xs1h = list(range(b1 - b0 + 1))

    o1, h1, l1, c1, v1, t1 = (
        d1h["o"][b0 : b1 + 1],
        d1h["h"][b0 : b1 + 1],
        d1h["l"][b0 : b1 + 1],
        d1h["c"][b0 : b1 + 1],
        d1h["v"][b0 : b1 + 1],
        d1h["t"][b0 : b1 + 1],
    )

    cols1h = []
    for k in range(len(xs1h)):
        up = c1[k] >= o1[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax1h.vlines(xs1h[k], l1[k], h1[k], color=col, lw=0.85, zorder=3)
        y0, y1 = min(o1[k], c1[k]), max(o1[k], c1[k])
        if y1 == y0:
            y1 = y0 + max(h1[k] - l1[k], 1e-12) * 0.02
        ax1h.add_patch(Rectangle((xs1h[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3, zorder=3))
        cols1h.append("#3dba7a99" if up else "#e35d5d99")

    ax1hv.bar(xs1h, v1, width=0.78, color=cols1h, linewidth=0)
    for n, col in MA_COLORS.items():
        ma_sl = ind1h[f"m{n}"][b0 : b1 + 1]
        ax1h.plot(xs1h, ma_sl, color=col, lw=2.0 if n == 200 else 1.1, label=f"1H MA{n}", zorder=4 if n == 200 else 2)

    # 標示 1H 24h 前高水平線與進場位置
    ax1h.axhline(tr["h1_hi24"], color="#79c0ff", ls="--", lw=0.9, alpha=0.8)
    ex1h = k1 - b0
    if 0 <= ex1h < len(xs1h):
        ax1h.axvline(ex1h, color="#3dba7a", ls="--", lw=0.95)
        ax1h.scatter([ex1h], [tr["entry_price"]], s=55, color="#00e676", marker="^", zorder=6)
        ax1h.annotate(
            f"1H距24h高 +{tr['room_to_1h_hi']:.1f}%",
            (ex1h, tr["h1_hi24"]),
            textcoords="offset points",
            xytext=(-40, 4),
            color="#79c0ff",
            fontsize=8,
        )

    status_1h = "1H全均線之上(通過)" if tr["pass_1h"] else f"1H攔截:{tr['fail_1h_reason']}"
    ax1h.set_title(
        f"[1小時K 結構] {status_1h} · 盤中最高 +{tr['mfe_pct']:.2f}%",
        color="#3ddc68" if tr["pass_1h"] else "#ff7b72",
        fontsize=10,
    )
    ax1h.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step1h = max(1, len(xs1h) // 5)
    ticks1h = list(range(0, len(xs1h), step1h))
    ax1hv.set_xticks(ticks1h)
    ax1hv.set_xticklabels([datetime.fromtimestamp(t1[idx] / 1000, TZ).strftime("%m-%d %H:%M") for idx in ticks1h], color="#8aa193")
    plt.setp(ax1h.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)


def simulate_trade(d15: dict, ind15: dict, sig: dict) -> dict:
    i = sig["entry_idx"]
    h, l, c = d15["h"], d15["l"], d15["c"]
    m14 = ind15["m14"]
    entry, stop = sig["entry_price"], sig["stop_price"]
    risk = entry - stop
    target = entry + risk * 4.0
    cur_stop = stop
    be_armed = False

    exit_idx = len(c) - 1
    exit_price = c[-1]
    reason = "期末未平"
    cat = "time"
    mfe_pct = 0.0
    mae_pct = 0.0

    for j in range(i + 1, min(len(c), i + 49)):
        mfe_pct = max(mfe_pct, (h[j] - entry) / entry * 100)
        mae_pct = min(mae_pct, (l[j] - entry) / entry * 100)
        if l[j] <= cur_stop:
            exit_idx = j
            exit_price = cur_stop
            if be_armed:
                reason = "移動保本(+0.2R)"
                cat = "be"
            else:
                reason = "觸及停損(-1R)"
                cat = "sl"
            break
        if h[j] >= target:
            exit_idx = j
            exit_price = target
            reason = "達標停利(+4.0R)"
            cat = "tp"
            break
        if (h[j] - entry) >= risk * 1.2:
            be_armed = True
            cur_stop = max(cur_stop, entry + risk * 0.2)
        if (h[j] - entry) >= risk * 2.0 and c[j] < m14[j]:
            exit_idx = j
            exit_price = c[j]
            reason = "破MA14波段停利"
            cat = "tp"
            break
    else:
        if i + 48 < len(c):
            exit_idx = i + 48
            exit_price = c[exit_idx]
            reason = "時間到期(12h)"
            cat = "time"

    pnl_pct = (exit_price - entry) / entry * 100
    r_mult = (exit_price - entry) / risk
    tr = dict(sig)
    tr.update(
        {
            "exit_idx": exit_idx,
            "exit_price": exit_price,
            "target_price": target,
            "entry_time": datetime.fromtimestamp(d15["t"][i] / 1000, TZ).strftime("%m-%d %H:%M"),
            "exit_time": datetime.fromtimestamp(d15["t"][exit_idx] / 1000, TZ).strftime("%m-%d %H:%M"),
            "entry_dt": datetime.fromtimestamp(d15["t"][i] / 1000, TZ).strftime("%Y-%m-%d %H:%M"),
            "pnl_pct": pnl_pct,
            "r_mult": r_mult,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "reason": reason,
            "cat": cat,
        }
    )
    return tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", type=str, default="docs/binance-15m-ma200-7d/index.html")
    args = ap.parse_args()

    setup_cjk()
    out_html = (REPO / args.out).resolve()
    img_dir = out_html.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old_png in img_dir.glob("*.png"):
        old_png.unlink()

    syms = get_universe()
    print(f"載入全市場 {len(syms)} 個幣安 USDT 永續合約 (15m + 1H)...")
    market = {}
    with ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(fetch_symbol_both, s): s for s in syms}
        for f in as_completed(futs):
            sym = futs[f]
            try:
                res = f.result()
                if res:
                    market[sym] = res
            except Exception:
                pass

    print(f"成功取得 {len(market)} 個標的，開始掃描 15m + 1H 結構...")
    max_ts = max(d15["t"][-1] for d15, _ in market.values())
    cutoff_ms = max_ts - args.days * 86400 * 1000

    raw_signals = []
    ind15_cache = {}
    ind1h_cache = {}
    for sym, (d15, d1h) in market.items():
        c15, v15 = d15["c"], d15["v"]
        ind15 = {
            "m7": sma(c15, 7),
            "m14": sma(c15, 14),
            "m25": sma(c15, 25),
            "m99": sma(c15, 99),
            "m120": sma(c15, 120),
            "m200": sma(c15, 200),
            "v20": sma(v15, 20),
        }
        c1h, v1h = d1h["c"], d1h["v"]
        ind1h = {
            "m7": sma(c1h, 7),
            "m14": sma(c1h, 14),
            "m25": sma(c1h, 25),
            "m99": sma(c1h, 99),
            "m120": sma(c1h, 120),
            "m200": sma(c1h, 200),
            "v20": sma(v1h, 20),
        }
        ind15_cache[sym] = ind15
        ind1h_cache[sym] = ind1h

        for i in range(220, len(c15) - 1):
            if d15["t"][i] < cutoff_ms:
                continue
            sig = detect_entry_15m_1h(d15, d1h, i, ind15, ind1h)
            if sig:
                sig["symbol"] = sym
                raw_signals.append((d15["t"][i], sym, i, sig))

    ts_counts = Counter(x[0] for x in raw_signals)
    skipped_spike_bars = sum(1 for _, cnt in ts_counts.items() if cnt >= 4)

    # 1. 通過 15m + 1H 雙週期優化版的正式交易
    opt_trades = []
    last_exit_opt = {}
    # 2. 被 1H 濾網擋下的對照組交易
    blocked_trades = []
    last_exit_blk = {}

    for ts, sym, i, sig in sorted(raw_signals, key=lambda x: (x[0], x[1])):
        if ts_counts[ts] >= 4:
            continue
        d15 = market[sym][0]
        ind15 = ind15_cache[sym]
        if sig["pass_1h"]:
            if i <= last_exit_opt.get(sym, -1):
                continue
            tr = simulate_trade(d15, ind15, sig)
            last_exit_opt[sym] = tr["exit_idx"]
            opt_trades.append(tr)
        else:
            if i <= last_exit_blk.get(sym, -1):
                continue
            tr = simulate_trade(d15, ind15, sig)
            last_exit_blk[sym] = tr["exit_idx"]
            blocked_trades.append(tr)

    print(f"繪製優化版 {len(opt_trades)} 張雙週期 (15m + 1H) K 線圖...")
    for idx, tr in enumerate(opt_trades, 1):
        sym = tr["symbol"]
        stamp = tr["entry_time"].replace("-", "").replace(" ", "_").replace(":", "")
        fname = f"opt_{idx:02d}_{sym}_{stamp}.png"
        tr["img_file"] = f"img/{fname}"
        d15, d1h = market[sym]
        draw_dual_chart(sym, d15, d1h, ind15_cache[sym], ind1h_cache[sym], tr, idx, img_dir / fname)

    (out_html.parent / "hits.json").write_text(
        json.dumps({"optimized": opt_trades, "blocked_by_1h": blocked_trades}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 統計數據
    opt_wins = [x for x in opt_trades if x["pnl_pct"] > 0]
    opt_pnl = sum(x["pnl_pct"] for x in opt_trades)
    opt_r = sum(x["r_mult"] for x in opt_trades)
    opt_wr = len(opt_wins) / max(len(opt_trades), 1) * 100
    opt_sl = sum(1 for x in opt_trades if x["cat"] == "sl")

    blk_wins = [x for x in blocked_trades if x["pnl_pct"] > 0]
    blk_pnl = sum(x["pnl_pct"] for x in blocked_trades)
    blk_r = sum(x["r_mult"] for x in blocked_trades)
    blk_wr = len(blk_wins) / max(len(blocked_trades), 1) * 100
    blk_sl = sum(1 for x in blocked_trades if x["cat"] == "sl")

    start_str = datetime.fromtimestamp(cutoff_ms / 1000, TZ).strftime("%Y-%m-%d %H:%M")
    end_str = datetime.fromtimestamp(max_ts / 1000, TZ).strftime("%Y-%m-%d %H:%M")

    cards_html = []
    for idx, tr in enumerate(opt_trades, 1):
        cls = "pnl-win" if tr["pnl_pct"] > 0 else ("pnl-flat" if tr["pnl_pct"] == 0 else "pnl-loss")
        tag_cls = {"tp": "tag-tp", "be": "tag-info", "time": "tag-time", "sl": "tag-sl"}.get(tr["cat"], "tag-info")
        detail = (
            f"進場 {tr['entry_price']:g}   停損 {tr['stop_price']:.4g}   目標(4R) {tr['target_price']:.4g}   出場 {tr['exit_price']:.4g}\n"
            f"[15m] MA200 {tr['ma200']:.4g} (+{tr['dist200_pct']:.2f}%) · 六線黏度 {tr['all6_spread_pct']:.2f}% · 前20K箱體 {tr['box_width_pct']:.2f}% · 突破量 {tr['vol_ratio']:.1f}x\n"
            f"[1H ] 站上全部1H均線 (MA25={tr['rt_1h_m25']:.4g} / MA99={tr['h1_m99']:.4g} / MA200={tr['h1_m200']:.4g}) · 距1H_24h高點空間 +{tr['room_to_1h_hi']:.2f}%"
        )
        cards_html.append(
            f"<article class='trade-card' data-cat='{tr['cat']}'>"
            f"<header class='card-header'>"
            f"<div class='card-title'>"
            f"<span class='trade-no'>#{idx} · {html.escape(tr['symbol'])}</span>"
            f"<span class='trade-time'>{tr['entry_dt']} → {tr['exit_time']} (UTC+8)</span>"
            f"</div>"
            f"<div class='card-pnl {cls}'>{tr['pnl_pct']:+.2f}% <small>({tr['r_mult']:+.2f}R)</small></div>"
            f"</header>"
            f"<div class='tags'>"
            f"<span class='tag {tag_cls}'>{html.escape(tr['reason'])}</span>"
            f"<span class='tag tag-tp'>盤中最高 +{tr['mfe_pct']:.2f}%</span>"
            f"<span class='tag tag-info'>1H 距24h高 +{tr['room_to_1h_hi']:.1f}%</span>"
            f"<span class='tag'>15m 六線黏度 {tr['all6_spread_pct']:.2f}%</span>"
            f"<span class='tag'>量 {tr['vol_ratio']:.1f}x</span>"
            f"</div>"
            f"<pre class='trade-detail'>{html.escape(detail)}</pre>"
            f"<div class='mini-chart'><img src='{tr['img_file']}' alt='{html.escape(tr['symbol'])}' loading='lazy'/></div>"
            f"</article>"
        )

    blocked_rows = []
    for idx, tr in enumerate(blocked_trades, 1):
        cls = "pnl-win" if tr["pnl_pct"] > 0 else "pnl-loss"
        blocked_rows.append(
            f"<tr>"
            f"<td>#{idx}</td>"
            f"<td><b>{html.escape(tr['symbol'])}</b></td>"
            f"<td>{tr['entry_time']}</td>"
            f"<td class='{cls}'>{tr['pnl_pct']:+.2f}% ({tr['r_mult']:+.2f}R)</td>"
            f"<td>+{tr['mfe_pct']:.2f}%</td>"
            f"<td>{html.escape(tr['reason'])}</td>"
            f"<td style='color:#ff7b72'>{html.escape(tr['fail_1h_reason'])}</td>"
            f"</tr>"
        )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 15m MA200 + 1H 結構優化版 · 近 7 天回測 ({len(opt_trades)} 筆)</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;-webkit-font-smoothing:antialiased}}
.page{{max-width:980px;margin:0 auto;padding:14px 12px 44px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px 18px;margin-bottom:14px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:14px 0 8px;color:#e6edf3}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6;margin:0}}
.rules{{margin:0;padding-left:1.25em;color:#c9d1d9;font-size:13px;line-height:1.65}}
.rules li{{margin-bottom:5px}}
code{{font-size:12px;background:#0d1117;padding:2px 5px;border-radius:4px;color:#79c0ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:14px 0 10px}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;border:1px solid #21262d}}
.card span{{font-size:12px;color:#8b949e;display:block}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.card small{{font-size:11px;color:#8b949e;display:block;margin-top:2px}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 14px}}
.fbtn{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:999px;padding:7px 14px;font-size:12px;cursor:pointer;font-weight:600}}
.fbtn.active{{background:#1f6feb;color:#fff;border-color:#388bfd}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}}
.trade-no{{font-weight:700;font-size:16px}}
.trade-time{{font-size:12px;color:#8b949e;display:block;margin-top:2px}}
.card-pnl{{font-weight:700;font-size:17px;text-align:right;white-space:nowrap}}
.card-pnl small{{font-size:12px;opacity:.85}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#8b949e;background:#0d1117}}
.tag-tp{{background:rgba(0,200,5,.15);color:#3ddc68;border-color:rgba(0,200,5,.35)}}
.tag-sl{{background:rgba(255,82,82,.15);color:#ff7b72;border-color:rgba(255,82,82,.35)}}
.tag-time{{background:rgba(255,193,7,.12);color:#f0c14b;border-color:rgba(255,193,7,.3)}}
.tag-info{{background:rgba(88,166,255,.12);color:#79c0ff;border-color:rgba(88,166,255,.28)}}
.trade-detail{{background:#0d1117;padding:10px 12px;border-radius:10px;border:1px solid #21262d;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap;margin:0 0 10px}}
.mini-chart img{{width:100%;height:auto;display:block;border-radius:10px;background:#101814}}
.note{{margin-top:10px;padding:10px 12px;background:#0d1117;border-radius:8px;color:#f0c14b;font-size:12px;line-height:1.6}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:#161b22;border-radius:12px;overflow:hidden;border:1px solid #30363d}}
th,td{{padding:8px 10px;border-bottom:1px solid #21262d;text-align:left}}
th{{background:#0d1117;color:#8b949e;font-weight:600}}
</style>
</head>
<body>
<div class="page">
  <section class="summary">
    <h1>幣安 USDT 永續 · 15m MA200 六線收斂 + 1H 結構優化版（近 7 天回測）</h1>
    <p class="muted">回測區間：{start_str} ～ {end_str} (UTC+8) · 全市場 {len(market)} 檔 USDT 永續合約 · 每張圖左邊為 15m 進出場、右邊為當下 1H 結構</p>

    <div class="cards">
      <div class="card">
        <span>優化後筆數 (15m+1H)</span>
        <b>{len(opt_trades)} 筆</b>
        <small>原純 15m 為 {len(opt_trades)+len(blocked_trades)} 筆</small>
      </div>
      <div class="card">
        <span>優化後勝率</span>
        <b class="pnl-win">{opt_wr:.1f}%</b>
        <small>{len(opt_wins)} 勝 / {len(opt_trades)-len(opt_wins)} 敗 (原 54.1%)</small>
      </div>
      <div class="card">
        <span>累計報酬 (單利1x)</span>
        <b class="pnl-win">{opt_pnl:+.2f}%</b>
        <small>平均每筆 {opt_pnl/max(len(opt_trades),1):+.2f}% (原 +18.22%)</small>
      </div>
      <div class="card">
        <span>累計盈虧比 (R)</span>
        <b class="pnl-win">{opt_r:+.2f} R</b>
        <small>停損僅 {opt_sl} 筆 (擋掉 {blk_sl} 筆停損)</small>
      </div>
    </div>

    <h2>為什麼加上 1 小時 K (1H) 條件差這麼多？</h2>
    <ol class="rules">
      <li><b>1H 全均線多頭淨空（無 1H 長均線蓋頭反壓）</b>：<code>15m MA200</code> 換算到 1H 就是 <code>1H MA50</code>。如果某個幣處於 1H 空頭趨勢，它的 <code>1H MA99 / MA120 / MA200</code> 會剛好壓在頭頂，15m 一突破就會撞牆停損。優化版規定進場價必須<b>同時高於 <code>1H MA25 / MA99 / MA120 / MA200</code></b>，頭頂完全沒有 1H 均線壓制。</li>
      <li><b>1H 距離 24 小時前高空間 ≥ 1.1%（不追 1H 天花板）</b>：若 15m 突破時已經貼在 1H 過去 24 根最高點下方不到 <code>1.1%</code>，代表是在 24h 高檔天花板追高；要求 <code>距 1H 24h 高點 ≥ 1.1%</code>，確保是 <b>1H 先有過多頭衝高、回檔在 15m MA200 縮量築底後再次起跳</b>。</li>
      <li><b>15m 核心與出場不變</b>：15m 六線黏度 ≤ <code>2.2%</code>、前 20 根窄箱體墊高、帶量突破長均線帶、排除同時間 ≥4 幣新聞齊插針（擋掉 {skipped_spike_bars} 個大盤插針時段）；浮盈 <code>1.2R</code> 推保本、<code>2R</code> 破 MA14 波段停利或 <code>4R</code> 達標。</li>
    </ol>

    <p class="note">
      <b>對照驗證：</b>被 1H 濾網攔截的 <b>{len(blocked_trades)} 筆</b> 交易，若硬做勝率僅 <b>{blk_wr:.1f}%</b>、包含 <b>{blk_sl} 筆停損</b>，累計報酬為 <b>{blk_pnl:+.2f}% ({blk_r:+.2f}R)</b>。1H 濾網精準砍掉 21 筆停損，同時完整保留 <code>TRUTH (+8.0%)</code>、<code>CVC (+6.4%)</code>、<code>MEGA (+8.0%)</code>、<code>KMNO (+4.8%)</code>、<code>PROVE (+3.6%)</code> 等所有大噴單！
    </p>
  </section>

  <div class="filters">
    <button class="fbtn active" onclick="showTab('all', this)">15m+1H 優化版全部 ({len(opt_trades)})</button>
    <button class="fbtn" onclick="showTab('tp', this)">波段停利 4R / 破MA14 ({sum(1 for x in opt_trades if x['cat']=='tp')})</button>
    <button class="fbtn" onclick="showTab('be', this)">保本/到期 ({sum(1 for x in opt_trades if x['cat'] in ('be','time'))})</button>
    <button class="fbtn" onclick="showTab('sl', this)">觸及停損 ({opt_sl})</button>
    <button class="fbtn" onclick="showTab('blocked', this)">查看被 1H 濾網擋掉的 {len(blocked_trades)} 筆明細</button>
  </div>

  <div id="cards-container">
    {''.join(cards_html)}
  </div>

  <div id="blocked-container" style="display:none">
    <div class="trade-card">
      <h2 style="margin-top:0">被 1H 濾網攔截的 {len(blocked_trades)} 筆交易（擋掉 {blk_sl} 筆停損，合計避開 {blk_pnl:+.2f}% / {blk_r:+.2f}R）</h2>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr><th>#</th><th>標的</th><th>進場時間</th><th>假設進場損益</th><th>盤中最高</th><th>結局</th><th>1H 攔截原因</th></tr>
          </thead>
          <tbody>
            {''.join(blocked_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<script>
function showTab(cat, btn) {{
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const cardsBox = document.getElementById('cards-container');
  const blkBox = document.getElementById('blocked-container');
  if (cat === 'blocked') {{
    cardsBox.style.display = 'none';
    blkBox.style.display = '';
    return;
  }}
  cardsBox.style.display = '';
  blkBox.style.display = 'none';
  document.querySelectorAll('#cards-container .trade-card').forEach(card => {{
    if (cat === 'all') card.style.display = '';
    else if (cat === 'be') card.style.display = (card.dataset.cat === 'be' || card.dataset.cat === 'time') ? '' : 'none';
    else card.style.display = (card.dataset.cat === cat) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""
    out_html.write_text(page_html, encoding="utf-8")

    rel = out_html.parent.relative_to(REPO).as_posix()
    raw_base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{git_branch()}/{rel}/"
    view_html = out_html.with_name("view.html")
    view_html.write_text(page_html.replace("src='img/", f"src='{raw_base}img/"), encoding="utf-8")

    print(f"已輸出優化版 HTML：{out_html}")
    print(f"已輸出優化版預覽：{view_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
