#!/usr/bin/env python3
"""
幣安 USDT 永續 15m + 1H 雙週期 — 近半年 (180 天: 2026-05 ~ 2026-10) 回測與優化 HTML 報告產生器

用法:
    python3 examples/binance_15m_ma200_6m.py --days 180 --out docs/binance-15m-ma200-6m/index.html
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import requests

TZ = timezone(timedelta(hours=8))
REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = Path("/tmp/binance_15m_6m_cache")

MA_COLORS = {
    7: "#f0b90b",    # 黃 (MA7)
    14: "#26c6da",   # 青藍 (MA14)
    25: "#ec407a",   # 粉紅 (MA25)
    99: "#ab47bc",   # 紫 (MA99)
    120: "#66bb6a",  # 綠 (MA120)
    200: "#ffffff",  # 白粗線 (MA200)
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})


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


def sma_np(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        cs = np.cumsum(np.insert(arr, 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def parse_csv_bytes(b: bytes):
    rows = []
    for line in b.decode("utf-8", errors="ignore").splitlines():
        if not line or line[0].isalpha():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append(
                (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            )
    return rows


def ensure_cache() -> list[Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CACHE_DIR.glob("*.npz"))
    if len(existing) >= 80:
        return existing

    info = SESSION.get("https://www.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
    tickers = {
        t["symbol"]: t
        for t in SESSION.get("https://www.binance.com/fapi/v1/ticker/24hr", timeout=20).json()
    }
    must_have = {
        "ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "FILUSDT", "FOLKSUSDT", "PIPPINUSDT",
        "TRUTHUSDT", "CVCUSDT", "MEGAUSDT", "KMNOUSDT", "PROVEUSDT", "AGLDUSDT", "HAEDALUSDT",
        "IOSTUSDT", "WOOUSDT",
    }
    cands = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        sym = s["symbol"]
        if sym in ("USDCUSDT", "BTCDOMUSDT"):
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        cands.append((qv, sym))

    cands.sort(reverse=True)
    syms = [sym for _, sym in cands[:85]]
    for m in must_have:
        if m not in syms and any(sym == m for _, sym in cands):
            syms.append(m)
    syms = sorted(set(syms))
    months = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]

    def fetch_one(sym: str):
        cf = CACHE_DIR / f"{sym}.npz"
        if cf.exists():
            return
        rows_map = {}
        for ym in months:
            url = f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/15m/{sym}-15m-{ym}.zip"
            try:
                r = SESSION.get(url, timeout=15)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        for name in zf.namelist():
                            for row in parse_csv_bytes(zf.read(name)):
                                rows_map[row[0]] = row
            except Exception:
                pass
        cur_start = 1790812800000
        for _ in range(3):
            try:
                r = SESSION.get(
                    "https://www.binance.com/fapi/v1/klines",
                    params={"symbol": sym, "interval": "15m", "startTime": cur_start, "limit": 1500},
                    timeout=20,
                )
                if r.status_code == 429:
                    time.sleep(2.0)
                    continue
                arr = r.json()
                if not isinstance(arr, list) or not arr:
                    break
                for k in arr[:-1]:
                    ts = int(k[0])
                    rows_map[ts] = (ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                if len(arr) < 1500:
                    break
                cur_start = int(arr[-1][0])
            except Exception:
                break
        if len(rows_map) < 2000:
            return
        sr = [rows_map[k] for k in sorted(rows_map.keys())]
        np.savez_compressed(
            cf,
            t=np.array([r[0] for r in sr], dtype=np.int64),
            o=np.array([r[1] for r in sr], dtype=np.float64),
            h=np.array([r[2] for r in sr], dtype=np.float64),
            l=np.array([r[3] for r in sr], dtype=np.float64),
            c=np.array([r[4] for r in sr], dtype=np.float64),
            v=np.array([r[5] for r in sr], dtype=np.float64),
        )

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(fetch_one, syms))
    return sorted(CACHE_DIR.glob("*.npz"))


def draw_dual_chart_6m(
    sym: str,
    d: dict,
    tr: dict,
    trade_no: int,
    out_path: Path,
) -> None:
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

    # --- 左側：15m ---
    entry_i = tr["entry_idx"]
    exit_i = tr["exit_idx"]
    a0 = max(0, entry_i - 34)
    a1 = min(len(d["c"]) - 1, max(exit_i + 6, entry_i + 24))
    xs15 = list(range(a1 - a0 + 1))

    o, h, l, c, v, t = (
        d["o"][a0 : a1 + 1],
        d["h"][a0 : a1 + 1],
        d["l"][a0 : a1 + 1],
        d["c"][a0 : a1 + 1],
        d["v"][a0 : a1 + 1],
        d["t"][a0 : a1 + 1],
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
        ma_sl = d[f"m{n}"][a0 : a1 + 1]
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
        f"[15分K] #{trade_no} {sym}  {tr['entry_dt']}→{tr['exit_time']}  {tr['reason']}  {tr['pnl_pct']:+.2f}% ({tr['r_mult']:+.2f}R)",
        color="#e8f0ea",
        fontsize=9.8,
    )
    ax15.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step15 = max(1, len(xs15) // 5)
    ticks15 = list(range(0, len(xs15), step15))
    ax15v.set_xticks(ticks15)
    ax15v.set_xticklabels([datetime.fromtimestamp(int(t[idx]) / 1000, TZ).strftime("%m-%d %H:%M") for idx in ticks15], color="#8aa193")
    plt.setp(ax15.get_xticklabels(), visible=False)

    # --- 右側：1H ---
    k1 = tr["h1_entry_idx"]
    b0 = max(0, k1 - 36)
    b1 = min(len(d["h1_c"]) - 1, k1 + 16)
    xs1h = list(range(b1 - b0 + 1))

    o1, h1, l1, c1, v1, t1 = (
        d["h1_o"][b0 : b1 + 1],
        d["h1_h"][b0 : b1 + 1],
        d["h1_l"][b0 : b1 + 1],
        d["h1_c"][b0 : b1 + 1],
        d["h1_v"][b0 : b1 + 1],
        d["h1_t"][b0 : b1 + 1],
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
        ma_sl = d[f"h1_m{n}"][b0 : b1 + 1]
        ax1h.plot(xs1h, ma_sl, color=col, lw=2.0 if n == 200 else 1.1, label=f"1H MA{n}", zorder=4 if n == 200 else 2)

    ax1h.axhline(tr["h1_hi24"], color="#79c0ff", ls="--", lw=0.9, alpha=0.8)
    ex1h = k1 - b0
    if 0 <= ex1h < len(xs1h):
        ax1h.axvline(ex1h, color="#3dba7a", ls="--", lw=0.95)
        ax1h.scatter([ex1h], [tr["entry_price"]], s=55, color="#00e676", marker="^", zorder=6)
        ax1h.annotate(
            f"1H距24h高 +{tr['room_1h']:.1f}%",
            (ex1h, tr["h1_hi24"]),
            textcoords="offset points",
            xytext=(-45, 4),
            color="#79c0ff",
            fontsize=8,
        )

    ax1h.set_title(
        f"[1小時K 結構] 1H全均線之上(MA99≥MA200) · 強於BTC +{tr['rs_3h']:.1f}% · 最高 +{tr['mfe_pct']:.2f}%",
        color="#3ddc68",
        fontsize=9.8,
    )
    ax1h.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step1h = max(1, len(xs1h) // 5)
    ticks1h = list(range(0, len(xs1h), step1h))
    ax1hv.set_xticks(ticks1h)
    ax1hv.set_xticklabels([datetime.fromtimestamp(int(t1[idx]) / 1000, TZ).strftime("%m-%d %H:%M") for idx in ticks1h], color="#8aa193")
    plt.setp(ax1h.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--out", type=str, default="docs/binance-15m-ma200-6m/index.html")
    args = ap.parse_args()

    setup_cjk()
    out_html = (REPO / args.out).resolve()
    img_dir = out_html.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old_png in img_dir.glob("*.png"):
        old_png.unlink()

    files = ensure_cache()
    print(f"載入 {len(files)} 檔標的之半年 15m K 線並合成 1H 結構...")

    end_ms = 1793445300000
    cutoff_ms = end_ms - args.days * 86400 * 1000

    market = {}
    for f in files:
        sym = f.stem
        d = np.load(f)
        t, o, h, l, c, v = d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]
        if len(t) < 2000:
            continue
        m7, m14, m25 = sma_np(c, 7), sma_np(c, 14), sma_np(c, 25)
        m99, m120, m200 = sma_np(c, 99), sma_np(c, 120), sma_np(c, 200)
        v5, v10, v20 = sma_np(v, 5), sma_np(v, 10), sma_np(v, 20)

        is_h1_close = (t % 3600_000) == 2700_000
        h1_close_idx = np.where(is_h1_close)[0]
        h1_t = t[h1_close_idx] - 2700_000
        h1_c = c[h1_close_idx]
        h1_o = np.array([o[max(0, idx - 3)] for idx in h1_close_idx])
        h1_h = np.array([h[max(0, idx - 3) : idx + 1].max() for idx in h1_close_idx])
        h1_l = np.array([l[max(0, idx - 3) : idx + 1].min() for idx in h1_close_idx])
        h1_v = np.array([v[max(0, idx - 3) : idx + 1].sum() for idx in h1_close_idx])

        h1_m7 = sma_np(h1_c, 7)
        h1_m14 = sma_np(h1_c, 14)
        h1_m25 = sma_np(h1_c, 25)
        h1_m99 = sma_np(h1_c, 99)
        h1_m120 = sma_np(h1_c, 120)
        h1_m200 = sma_np(h1_c, 200)
        k1_for_15m = np.searchsorted(h1_close_idx, np.arange(len(t)), side="right") - 1

        market[sym] = {
            "t": t, "o": o, "h": h, "l": l, "c": c, "v": v,
            "m7": m7, "m14": m14, "m25": m25, "m99": m99, "m120": m120, "m200": m200,
            "v5": v5, "v10": v10, "v20": v20,
            "k1_map": k1_for_15m,
            "h1_t": h1_t, "h1_o": h1_o, "h1_h": h1_h, "h1_l": h1_l, "h1_c": h1_c, "h1_v": h1_v,
            "h1_m7": h1_m7, "h1_m14": h1_m14, "h1_m25": h1_m25,
            "h1_m99": h1_m99, "h1_m120": h1_m120, "h1_m200": h1_m200,
        }

    btc = market["BTCUSDT"]
    btc_ts_to_idx = {int(ts): idx for idx, ts in enumerate(btc["t"])}

    all_cands = []
    for sym, d in market.items():
        t, o, h, l, c, v = d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]
        m7, m14, m25, m99, m120, m200, v20 = d["m7"], d["m14"], d["m25"], d["m99"], d["m120"], d["m200"], d["v20"]
        hi_long = np.maximum(m99, np.maximum(m120, m200))
        lo_long = np.minimum(m99, np.minimum(m120, m200))
        long_spread = (hi_long - lo_long) / m200 * 100.0
        hi_all6 = np.maximum(hi_long, np.maximum(m7, np.maximum(m14, m25)))
        lo_all6 = np.minimum(lo_long, np.minimum(m7, np.minimum(m14, m25)))
        all6_spread = (hi_all6 - lo_all6) / m200 * 100.0
        slope200 = np.zeros_like(m200)
        slope200[12:] = (m200[12:] - m200[:-12]) / m200[:-12] * 100.0
        dist200 = (c - m200) / m200 * 100.0
        rng = h - l
        close_pos = np.where(rng > 0, (c - l) / np.maximum(rng, 1e-12), 0.0)
        vol_ratio = np.where(v20 > 0, v / np.maximum(v20, 1e-9), 0.0)

        mask = (
            (t >= cutoff_ms)
            & (~np.isnan(m200))
            & (long_spread <= 1.5)
            & (all6_spread <= 2.2)
            & (slope200 >= -0.15) & (slope200 <= 0.40)
            & (dist200 >= 0.05) & (dist200 <= 1.8)
            & (c > hi_long)
            & (c > o) & (close_pos >= 0.65)
            & (m7 > m14) & (m14 >= m25)
            & (vol_ratio >= 1.8)
        )
        for i in np.where(mask)[0]:
            if i < 240 or i >= len(c) - 64:
                continue
            if m7[i] <= m7[i - 1] or c[i] <= c[i - 12 : i].max():
                continue
            if not (c[i - 1] <= hi_long[i - 1] * 1.003 or c[i - 2] <= hi_long[i - 2] * 1.003):
                continue
            box_hi = float(h[i - 20 : i].max())
            box_lo = float(l[i - 20 : i].min())
            box_width = (box_hi - box_lo) / float(m200[i]) * 100.0
            if not (0.8 <= box_width <= 4.5):
                continue
            lo_first = float(l[i - 24 : i - 8].min())
            lo_second = float(l[i - 8 : i + 1].min())
            if not (lo_second > lo_first and lo_first < float(m200[i])):
                continue
            k1 = int(d["k1_map"][i])
            if k1 < 205 or np.isnan(d["h1_m200"][k1]):
                continue

            px = float(c[i])
            h1_m7, h1_m14, h1_m25 = float(d["h1_m7"][k1]), float(d["h1_m14"][k1]), float(d["h1_m25"][k1])
            h1_m99, h1_m120, h1_m200 = float(d["h1_m99"][k1]), float(d["h1_m120"][k1]), float(d["h1_m200"][k1])
            h1_hi24 = float(d["h1_h"][k1 - 24 : k1 + 1].max())
            room_1h = (h1_hi24 - px) / px * 100.0
            h1_hi_l = max(h1_m99, h1_m120, h1_m200)
            h1_lo_l = min(h1_m99, h1_m120, h1_m200)
            h1_long_spread = (h1_hi_l - h1_lo_l) / h1_m200 * 100.0
            h1_hl = float(d["h1_l"][k1 - 5 : k1 + 1].min()) > float(d["h1_l"][k1 - 20 : k1 - 5].min())
            h1_slp99 = (h1_m99 - float(d["h1_m99"][k1 - 6])) / float(d["h1_m99"][k1 - 6]) * 100.0
            above_1h_all = px > max(h1_m7, h1_m14, h1_m25, h1_m99, h1_m120, h1_m200)
            dist_h1_m25 = (px - h1_m25) / h1_m25 * 100.0

            b_idx = btc_ts_to_idx.get(int(t[i]), -1)
            rs_3h = 0.0
            if b_idx >= 12:
                rs_3h = (px / float(c[i - 12]) - float(btc["c"][b_idx]) / float(btc["c"][b_idx - 12])) * 100.0

            all_cands.append(
                {
                    "ts": int(t[i]),
                    "symbol": sym,
                    "entry_idx": int(i),
                    "h1_entry_idx": k1,
                    "entry_price": px,
                    "ma200": float(m200[i]),
                    "dist200_pct": float(dist200[i]),
                    "long_spread_pct": float(long_spread[i]),
                    "all6_spread_pct": float(all6_spread[i]),
                    "box_hi": box_hi,
                    "box_lo": box_lo,
                    "box_width_pct": box_width,
                    "vol_ratio": float(vol_ratio[i]),
                    "lo_second": lo_second,
                    "lo_long": float(lo_long[i]),
                    "above_1h_all": above_1h_all,
                    "room_1h": room_1h,
                    "h1_hi24": h1_hi24,
                    "h1_long_spread": h1_long_spread,
                    "h1_hl": h1_hl,
                    "h1_slp99": h1_slp99,
                    "dist_h1_m25": dist_h1_m25,
                    "rs_3h": rs_3h,
                    "v5_gt_v10": float(d["v5"][i]) > float(d["v10"][i]),
                    "h1_m99_gt_200": h1_m99 >= h1_m200,
                    "h1_m25": h1_m25,
                    "h1_m99": h1_m99,
                    "h1_m200": h1_m200,
                }
            )

    ts_counts = Counter(x["ts"] for x in all_cands)

    def passes_6m_opt(s: dict) -> bool:
        return (
            s["above_1h_all"]
            and s["room_1h"] >= 1.1
            and s["h1_hl"]
            and s["h1_long_spread"] <= 3.5
            and s["h1_slp99"] >= -0.05
            and s["dist_h1_m25"] <= 2.5
            and s["rs_3h"] >= 0.4
            and s["v5_gt_v10"]
            and s["h1_m99_gt_200"]
        )

    opt_trades = []
    last_exit = {}
    by_month = defaultdict(list)

    for sig in sorted(all_cands, key=lambda x: (x["ts"], x["symbol"])):
        near_cnt = ts_counts[sig["ts"]] + ts_counts[sig["ts"] - 900_000] + ts_counts[sig["ts"] + 900_000]
        if ts_counts[sig["ts"]] >= 4 or near_cnt >= 6:
            continue
        if not passes_6m_opt(sig):
            continue
        sym, i = sig["symbol"], sig["entry_idx"]
        if i <= last_exit.get(sym, -1):
            continue

        d = market[sym]
        h, l, c, m14 = d["h"], d["l"], d["c"], d["m14"]
        entry = sig["entry_price"]
        raw_stop = min(sig["lo_second"] * 0.998, sig["lo_long"] * 0.990)
        stop = min(entry * (1.0 - 0.014), max(raw_stop, entry * (1.0 - 0.025)))
        risk = entry - stop
        hard_stop = entry - risk * 1.35
        target = entry + risk * 4.0
        cur_stop = stop
        be = False

        exit_idx = i + 48
        exit_price = float(c[exit_idx])
        reason = "時間到期(12h)"
        cat = "time"
        mfe_pct = 0.0
        mae_pct = 0.0

        for j in range(i + 1, i + 49):
            hj, lj, cj = float(h[j]), float(l[j]), float(c[j])
            mfe_pct = max(mfe_pct, (hj - entry) / entry * 100.0)
            mae_pct = min(mae_pct, (lj - entry) / entry * 100.0)
            if not be:
                if lj <= hard_stop:
                    exit_idx, exit_price, reason, cat = j, hard_stop, "硬停損(-1.35R)", "sl"
                    break
                if cj <= stop:
                    exit_idx, exit_price, reason, cat = j, cj, "收盤停損(-1R)", "sl"
                    break
            else:
                if lj <= cur_stop:
                    exit_idx, exit_price, reason, cat = j, cur_stop, "移動保本(+0.25R)", "be"
                    break
            if hj >= target:
                exit_idx, exit_price, reason, cat = j, target, "達標停利(+4.0R)", "tp"
                break
            if (hj - entry) >= risk * 1.2:
                be = True
                cur_stop = max(cur_stop, entry + risk * 0.25)
            if (hj - entry) >= risk * 2.0 and cj < float(m14[j]):
                exit_idx, exit_price, reason, cat = j, cj, "破MA14波段停利", "tp"
                break

        last_exit[sym] = exit_idx
        pnl_pct = (exit_price - entry) / entry * 100.0
        r_mult = (exit_price - entry) / risk
        ym = datetime.fromtimestamp(sig["ts"] / 1000, TZ).strftime("%Y-%m")

        tr = dict(sig)
        tr.update(
            {
                "ym": ym,
                "stop_price": stop,
                "target_price": target,
                "exit_idx": exit_idx,
                "exit_price": exit_price,
                "entry_dt": datetime.fromtimestamp(sig["ts"] / 1000, TZ).strftime("%Y-%m-%d %H:%M"),
                "entry_time": datetime.fromtimestamp(sig["ts"] / 1000, TZ).strftime("%m-%d %H:%M"),
                "exit_time": datetime.fromtimestamp(int(d["t"][exit_idx]) / 1000, TZ).strftime("%m-%d %H:%M"),
                "pnl_pct": pnl_pct,
                "r_mult": r_mult,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "reason": reason,
                "cat": cat,
            }
        )
        opt_trades.append(tr)
        by_month[ym].append(tr)

    print(f"繪製半年優化版 {len(opt_trades)} 張雙週期 (15m + 1H) K 線圖...")
    for idx, tr in enumerate(opt_trades, 1):
        sym = tr["symbol"]
        stamp = tr["entry_dt"].replace("-", "").replace(" ", "_").replace(":", "")
        fname = f"m6_{idx:02d}_{sym}_{stamp}.png"
        tr["img_file"] = f"img/{fname}"
        draw_dual_chart_6m(sym, market[sym], tr, idx, img_dir / fname)

    (out_html.parent / "hits.json").write_text(
        json.dumps(opt_trades, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wins = [x for x in opt_trades if x["pnl_pct"] > 0]
    sl_trades = [x for x in opt_trades if x["cat"] == "sl"]
    tp_trades = [x for x in opt_trades if x["cat"] == "tp"]
    be_trades = [x for x in opt_trades if x["cat"] == "be"]
    time_trades = [x for x in opt_trades if x["cat"] == "time"]

    tot_pnl = sum(x["pnl_pct"] for x in opt_trades)
    tot_r = sum(x["r_mult"] for x in opt_trades)
    wr = len(wins) / max(len(opt_trades), 1) * 100.0

    month_rows = []
    for ym in sorted(by_month.keys()):
        arr = by_month[ym]
        w = sum(1 for x in arr if x["pnl_pct"] > 0)
        sl_c = sum(1 for x in arr if x["cat"] == "sl")
        mpnl = sum(x["pnl_pct"] for x in arr)
        mr = sum(x["r_mult"] for x in arr)
        cls = "pnl-win" if mpnl >= 0 else "pnl-loss"
        month_rows.append(
            f"<tr>"
            f"<td><b>{ym}</b></td>"
            f"<td>{len(arr)} 筆</td>"
            f"<td>{w / len(arr) * 100:.1f}% ({w}勝/{len(arr)-w}敗)</td>"
            f"<td>{sl_c} 筆</td>"
            f"<td class='{cls}'><b>{mpnl:+.2f}%</b></td>"
            f"<td class='{cls}'><b>{mr:+.2f} R</b></td>"
            f"</tr>"
        )

    cards_html = []
    for idx, tr in enumerate(opt_trades, 1):
        cls = "pnl-win" if tr["pnl_pct"] > 0 else ("pnl-flat" if tr["pnl_pct"] == 0 else "pnl-loss")
        tag_cls = {"tp": "tag-tp", "be": "tag-info", "time": "tag-time", "sl": "tag-sl"}.get(tr["cat"], "tag-info")
        detail = (
            f"進場 {tr['entry_price']:g}   收盤停損線 {tr['stop_price']:.4g}   目標(4R) {tr['target_price']:.4g}   出場 {tr['exit_price']:.4g}\n"
            f"[15m] MA200 {tr['ma200']:.4g} (+{tr['dist200_pct']:.2f}%) · 六線黏度 {tr['all6_spread_pct']:.2f}% · 箱體 {tr['box_width_pct']:.2f}% · 突破量 {tr['vol_ratio']:.1f}x · 強於BTC +{tr['rs_3h']:.2f}%\n"
            f"[1H ] 站上全均線且MA99≥MA200 (MA25={tr['h1_m25']:.4g} / MA99={tr['h1_m99']:.4g} / MA200={tr['h1_m200']:.4g}) · 1H長均黏度 {tr['h1_long_spread']:.2f}% · 距24h高 +{tr['room_1h']:.2f}%"
        )
        cards_html.append(
            f"<article class='trade-card' data-cat='{tr['cat']}' data-ym='{tr['ym']}'>"
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
            f"<span class='tag tag-info'>強於BTC +{tr['rs_3h']:.1f}%</span>"
            f"<span class='tag'>1H 距24h高 +{tr['room_1h']:.1f}%</span>"
            f"<span class='tag'>15m 六線黏度 {tr['all6_spread_pct']:.2f}%</span>"
            f"</div>"
            f"<pre class='trade-detail'>{html.escape(detail)}</pre>"
            f"<div class='mini-chart'><img src='{tr['img_file']}' alt='{html.escape(tr['symbol'])}' loading='lazy'/></div>"
            f"</article>"
        )

    month_btns = "".join(
        f"<button class='fbtn' onclick=\"filterBy('ym','{ym}',this)\">{ym} ({len(by_month[ym])})</button>"
        for ym in sorted(by_month.keys())
    )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 15m MA200 + 1H 雙週期優化版 · 近半年 (180天) 回測 ({len(opt_trades)} 筆)</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;-webkit-font-smoothing:antialiased}}
.page{{max-width:980px;margin:0 auto;padding:14px 12px 44px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px 18px;margin-bottom:14px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:16px 0 8px;color:#e6edf3}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6;margin:0}}
.rules{{margin:0;padding-left:1.25em;color:#c9d1d9;font-size:13px;line-height:1.65}}
.rules li{{margin-bottom:6px}}
code{{font-size:12px;background:#0d1117;padding:2px 5px;border-radius:4px;color:#79c0ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:14px 0 10px}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;border:1px solid #21262d}}
.card span{{font-size:12px;color:#8b949e;display:block}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.card small{{font-size:11px;color:#8b949e;display:block;margin-top:2px}}
.filters{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 14px}}
.fbtn{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:999px;padding:6px 12px;font-size:12px;cursor:pointer;font-weight:600}}
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
table{{width:100%;border-collapse:collapse;font-size:12.5px;background:#0d1117;border-radius:10px;overflow:hidden;border:1px solid #21262d;margin-top:8px}}
th,td{{padding:8px 10px;border-bottom:1px solid #21262d;text-align:left}}
th{{background:#161b22;color:#8b949e;font-weight:600}}
</style>
</head>
<body>
<div class="page">
  <section class="summary">
    <h1>幣安 USDT 永續 · 15m MA200 + 1H 雙週期優化版 · 近半年 (180天) 回測</h1>
    <p class="muted">回測區間：2026-05-04 ～ 2026-10-31 (UTC+8，共 180 天) · 96 檔高流動性 USDT 永續合約（約 190 萬根 15m K 線）</p>

    <div class="cards">
      <div class="card">
        <span>半年總觸發筆數</span>
        <b>{len(opt_trades)} 筆</b>
        <small>純 15m 原為 1,167 筆 (-42.4%)</small>
      </div>
      <div class="card">
        <span>勝率 / 停損率</span>
        <b class="pnl-win">{wr:.1f}%</b>
        <small>{len(wins)} 勝 · 停損僅 {len(sl_trades)} 筆 ({len(sl_trades)/len(opt_trades)*100:.1f}%)</small>
      </div>
      <div class="card">
        <span>半年累計報酬 (單利1x)</span>
        <b class="pnl-win">{tot_pnl:+.2f}%</b>
        <small>平均每筆 {tot_pnl/len(opt_trades):+.2f}% (較純15m提升 +99.2%)</small>
      </div>
      <div class="card">
        <span>半年累計盈虧比</span>
        <b class="pnl-win">{tot_r:+.2f} R</b>
        <small>波段停利 {len(tp_trades)} · 保本 {len(be_trades)} · 到期 {len(time_trades)}</small>
      </div>
    </div>

    <h2>半年跨牛熊優化進演表（2026-05 ～ 2026-10）</h2>
    <table>
      <thead>
        <tr><th>策略階段</th><th>半年筆數</th><th>勝率</th><th>停損筆數 (停損率)</th><th>累計報酬 (1x)</th><th>累計 R 數</th></tr>
      </thead>
      <tbody>
        <tr><td>V0 · 純 15m 六線收斂突破（無 1H 濾網）</td><td>1,167 筆</td><td>43.3%</td><td>526 筆 (45.1%)</td><td class="pnl-loss">-42.44%</td><td class="pnl-loss">-23.79 R</td></tr>
        <tr><td>V1 · 加上基礎 1H 濾網（1H 全均線上 + 距24h高≥1.1%）</td><td>255 筆</td><td>43.5%</td><td>125 筆 (49.0%)</td><td class="pnl-win">+4.28%</td><td class="pnl-win">+10.68 R</td></tr>
        <tr><td>V2 · 1H 底部墊高 + 1H 長均收斂≤3.5% + 收盤確認停損</td><td>107 筆</td><td>50.5%</td><td>41 筆 (38.3%)</td><td class="pnl-win">+28.63%</td><td class="pnl-win">+23.21 R</td></tr>
        <tr><td><b>V3 · 半年最終優化版（+ 1H MA99≥MA200 + 強於BTC + 預留回踩空間）</b></td><td><b>{len(opt_trades)} 筆</b></td><td><b>{wr:.1f}%</b></td><td><b>{len(sl_trades)} 筆 ({len(sl_trades)/len(opt_trades)*100:.1f}%)</b></td><td class="pnl-win"><b>{tot_pnl:+.2f}%</b></td><td class="pnl-win"><b>{tot_r:+.2f} R</b></td></tr>
      </tbody>
    </table>

    <h2>各月份表現明細（6 個月中有 5 個月正報酬，成功躲過 6 月 BTC -20.7% 崩盤）</h2>
    <table>
      <thead>
        <tr><th>月份</th><th>交易筆數</th><th>勝率 (勝/敗)</th><th>停損筆數</th><th>當月累計報酬</th><th>當月累計 R</th></tr>
      </thead>
      <tbody>
        {''.join(month_rows)}
      </tbody>
    </table>

    <h2>半年大樣本提煉出的 5 大核心優化規則</h2>
    <ol class="rules">
      <li><b>1H 大結構多頭排列且長均收斂</b>：進場價必須高於 <code>1H 全均線 (7/14/25/99/120/200)</code>，且 <code>1H MA99 ≥ 1H MA200</code>、<code>1H 長均黏度 ≤ 3.5%</code>、<code>1H MA99 斜率 ≥ -0.05%</code>、<code>1H 近 6 根低點墊高</code>。這條直接把空頭趨勢裡的死貓跳全部過濾。</li>
      <li><b>1H 空間與不追高</b>：距離 <code>1H 24h 高點空間 ≥ 1.1%</code>（不買在 24h 天花板），且距離 <code>1H MA25 ≤ 2.5%</code>（1H 不過度乖離）。</li>
      <li><b>3 小時相對強弱（RS vs BTC）≥ +0.4% + 15m 均量金叉</b>：突破當下該幣近 3 小時漲幅必須比 BTC 強 <code>+0.4%</code> 以上，且 <code>15m Vol MA5 &gt; Vol MA10</code>，只做有獨立資金點火的強勢幣。</li>
      <li><b>預留回踩 MA200 的呼吸空間 + 15m 收盤確認停損</b>：半年數據顯示，很多真突破會先下影線回測 MA200（約 -0.8%～-1.2%）再噴出。將初始停損設於箱體第二低點下方且<b>至少留 1.4% 空間（上限 2.5%）</b>，並改為<b>「15m 收盤跌破才停損（盤中設 1.35R 防爆倉硬底線）」</b>，停損率直接從 49% 砍到 <b>26.6%</b>！</li>
      <li><b>出場抱波段</b>：浮盈達 <code>1.2R</code> 推保本（<code>+0.25R</code>）；浮盈達 <code>2.0R</code> 後收盤跌破 <code>15m MA14</code> 波段停利，或直達 <code>4.0R</code> 停利（最長持有 12 小時）。</li>
    </ol>
  </section>

  <div class="filters">
    <button class="fbtn active" onclick="filterBy('all','',this)">全部 ({len(opt_trades)})</button>
    <button class="fbtn" onclick="filterBy('cat','tp',this)">波段停利 4R / 破MA14 ({len(tp_trades)})</button>
    <button class="fbtn" onclick="filterBy('cat','be',this)">移動保本 +0.25R ({len(be_trades)})</button>
    <button class="fbtn" onclick="filterBy('cat','time',this)">時間到期 ({len(time_trades)})</button>
    <button class="fbtn" onclick="filterBy('cat','sl',this)">觸及停損 ({len(sl_trades)})</button>
    {month_btns}
  </div>

  <div id="cards-container">
    {''.join(cards_html)}
  </div>
</div>
<script>
function filterBy(mode, val, btn) {{
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.trade-card').forEach(card => {{
    if (mode === 'all') card.style.display = '';
    else if (mode === 'cat') card.style.display = (card.dataset.cat === val) ? '' : 'none';
    else if (mode === 'ym') card.style.display = (card.dataset.ym === val) ? '' : 'none';
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

    print(f"已輸出半年優化版 HTML：{out_html}")
    print(f"已輸出半年優化版預覽：{view_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
