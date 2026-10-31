#!/usr/bin/env python3
"""
幣安 USDT 永續 15m — 「MA200 六線收斂 + 窄箱體墊高突破」近 7 天回測與 HTML 報告產生器

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
import numpy as np

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


def fetch_klines(sym: str, limit: int = 960) -> dict | None:
    raw = get_json("/fapi/v1/klines", {"symbol": sym, "interval": "15m", "limit": limit})
    if not raw or len(raw) < 260:
        return None
    klines = raw[:-1]
    return {
        "t": [int(k[0]) for k in klines],
        "o": [float(k[1]) for k in klines],
        "h": [float(k[2]) for k in klines],
        "l": [float(k[3]) for k in klines],
        "c": [float(k[4]) for k in klines],
        "v": [float(k[5]) for k in klines],
    }


def detect_entry(d: dict, i: int, m7, m14, m25, m99, m120, m200, v20) -> dict | None:
    if i < 220 or math.isnan(m200[i - 12]):
        return None
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]

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

    stop = max(min(lo_second, lo_long * 0.996), c[i] * 0.98)
    if (c[i] - stop) / c[i] < 0.004:
        stop = c[i] * 0.992

    return {
        "entry_idx": i,
        "entry_price": c[i],
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
    }


def draw_trade_chart(
    sym: str,
    d: dict,
    inds: dict,
    tr: dict,
    trade_no: int,
    out_path: Path,
) -> None:
    entry_i = tr["entry_idx"]
    exit_i = tr["exit_idx"]
    a0 = max(0, entry_i - 36)
    a1 = min(len(d["c"]) - 1, max(exit_i + 8, entry_i + 24))
    xs = list(range(a1 - a0 + 1))

    o = d["o"][a0 : a1 + 1]
    h = d["h"][a0 : a1 + 1]
    l = d["l"][a0 : a1 + 1]
    c = d["c"][a0 : a1 + 1]
    v = d["v"][a0 : a1 + 1]
    t = d["t"][a0 : a1 + 1]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.6, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    # 畫前 20 根窄幅盤整箱體陰影
    box_x0 = max(0, (entry_i - 20) - a0)
    box_x1 = entry_i - a0
    if box_x1 > box_x0:
        ax.add_patch(
            Rectangle(
                (box_x0 - 0.4, tr["box_lo"]),
                box_x1 - box_x0 + 0.4,
                tr["box_hi"] - tr["box_lo"],
                facecolor="#79c0ff18",
                edgecolor="#79c0ff66",
                ls="--",
                lw=0.8,
                zorder=1,
            )
        )

    colors_v = []
    for k in range(len(xs)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.85, zorder=3)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(
            Rectangle(
                (xs[k] - 0.35, y0),
                0.7,
                y1 - y0,
                facecolor=col,
                edgecolor=col,
                lw=0.3,
                zorder=3,
            )
        )
        colors_v.append("#3dba7a99" if up else "#e35d5d99")

    axv.bar(xs, v, width=0.78, color=colors_v, linewidth=0)
    v5_slice = inds["v5"][a0 : a1 + 1]
    v10_slice = inds["v10"][a0 : a1 + 1]
    axv.plot(xs, v5_slice, color="#f0b90b", lw=0.9, label="Vol MA5")
    axv.plot(xs, v10_slice, color="#ab47bc", lw=0.9, label="Vol MA10")

    for n, col in MA_COLORS.items():
        ma_slice = inds[f"m{n}"][a0 : a1 + 1]
        lw = 2.0 if n == 200 else 1.15
        ax.plot(xs, ma_slice, color=col, lw=lw, label=f"MA{n}", zorder=4 if n == 200 else 2)

    # 停損與停利水平線
    ax.axhline(tr["stop_price"], color="#ff5252", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(tr["target_price"], color="#3dba7a", ls=":", lw=1.0, alpha=0.80)

    ex = entry_i - a0
    xx = exit_i - a0
    if 0 <= ex < len(xs):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=1.0, alpha=0.9)
        ax.scatter([ex], [tr["entry_price"]], s=55, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            f"進 {tr['entry_price']:g}",
            (ex, tr["entry_price"]),
            textcoords="offset points",
            xytext=(4, -14),
            color="#00e676",
            fontsize=8,
            weight="bold",
        )
    if 0 <= xx < len(xs):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=1.0, alpha=0.9)
        ax.scatter(
            [xx],
            [tr["exit_price"]],
            s=50,
            color="#00e676" if tr["pnl_pct"] > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )
        ax.annotate(
            f"出 {tr['exit_price']:.4g} ({tr['pnl_pct']:+.2f}%)",
            (xx, tr["exit_price"]),
            textcoords="offset points",
            xytext=(4, 8),
            color="#f0c14b",
            fontsize=8,
        )

    ax.set_title(
        f"#{trade_no}  {sym} (15m)  {tr['entry_time']} → {tr['exit_time']}  "
        f"{tr['reason']}  {tr['pnl_pct']:+.2f}% ({tr['r_mult']:+.2f}R)  最高曾漲 +{tr['mfe_pct']:.2f}%",
        color="#e8f0ea",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor="#c8d5cc", ncol=6)

    step = max(1, len(xs) // 6)
    ticks = list(range(0, len(xs), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [datetime.fromtimestamp(t[idx] / 1000, TZ).strftime("%m-%d %H:%M") for idx in ticks],
        color="#8aa193",
    )

    fig.tight_layout(pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115, facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", type=str, default="docs/binance-15m-ma200-7d/index.html")
    args = ap.parse_args()

    setup_cjk()
    out_html = (REPO / args.out).resolve()
    img_dir = out_html.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    syms = get_universe()
    print(f"載入全市場 {len(syms)} 個幣安 USDT 永續合約 (15m)...")
    data = {}
    with ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(fetch_klines, s, 960): s for s in syms}
        for f in as_completed(futs):
            sym = futs[f]
            try:
                res = f.result()
                if res:
                    data[sym] = res
            except Exception:
                pass

    print(f"成功取得 {len(data)} 個標的，開始掃描與回測過去 {args.days} 天...")
    max_ts = max(d["t"][-1] for d in data.values())
    cutoff_ms = max_ts - args.days * 86400 * 1000

    raw_signals = []
    ind_cache = {}
    for sym, d in data.items():
        c, v = d["c"], d["v"]
        inds = {
            "m7": sma(c, 7),
            "m14": sma(c, 14),
            "m25": sma(c, 25),
            "m99": sma(c, 99),
            "m120": sma(c, 120),
            "m200": sma(c, 200),
            "v5": sma(v, 5),
            "v10": sma(v, 10),
            "v20": sma(v, 20),
        }
        ind_cache[sym] = inds
        for i in range(220, len(c) - 1):
            if d["t"][i] < cutoff_ms:
                continue
            sig = detect_entry(
                d,
                i,
                inds["m7"],
                inds["m14"],
                inds["m25"],
                inds["m99"],
                inds["m120"],
                inds["m200"],
                inds["v20"],
            )
            if sig:
                raw_signals.append((d["t"][i], sym, i, sig))

    ts_counts = Counter(x[0] for x in raw_signals)
    skipped_spike_bars = sum(1 for k, cnt in ts_counts.items() if cnt >= 4)
    skipped_spike_signals = sum(cnt for k, cnt in ts_counts.items() if cnt >= 4)

    trades = []
    last_exit = {}
    for ts, sym, i, sig in sorted(raw_signals):
        if ts_counts[ts] >= 4:
            continue
        if i <= last_exit.get(sym, -1):
            continue

        d = data[sym]
        h, l, c = d["h"], d["l"], d["c"]
        m14 = ind_cache[sym]["m14"]
        entry = sig["entry_price"]
        stop = sig["stop_price"]
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

        last_exit[sym] = exit_idx
        pnl_pct = (exit_price - entry) / entry * 100
        r_mult = (exit_price - entry) / risk

        tr = dict(sig)
        tr.update(
            {
                "symbol": sym,
                "exit_idx": exit_idx,
                "exit_price": exit_price,
                "target_price": target,
                "entry_time": datetime.fromtimestamp(d["t"][i] / 1000, TZ).strftime("%m-%d %H:%M"),
                "exit_time": datetime.fromtimestamp(d["t"][exit_idx] / 1000, TZ).strftime("%m-%d %H:%M"),
                "entry_dt": datetime.fromtimestamp(d["t"][i] / 1000, TZ).strftime("%Y-%m-%d %H:%M"),
                "pnl_pct": pnl_pct,
                "r_mult": r_mult,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "reason": reason,
                "cat": cat,
            }
        )
        trades.append(tr)

    # 繪製每筆交易的 15m K 線圖
    print(f"繪製 {len(trades)} 張 15m K 線圖...")
    for idx, tr in enumerate(trades, 1):
        sym = tr["symbol"]
        stamp = tr["entry_time"].replace("-", "").replace(" ", "_").replace(":", "")
        fname = f"t{idx:02d}_{sym}_{stamp}.png"
        tr["img_file"] = f"img/{fname}"
        draw_trade_chart(sym, data[sym], ind_cache[sym], tr, idx, img_dir / fname)

    # 輸出 hits.json
    (out_html.parent / "hits.json").write_text(
        json.dumps(trades, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wins = [x for x in trades if x["pnl_pct"] > 0]
    tp_trades = [x for x in trades if x["cat"] == "tp"]
    be_trades = [x for x in trades if x["cat"] == "be"]
    time_trades = [x for x in trades if x["cat"] == "time"]
    sl_trades = [x for x in trades if x["cat"] == "sl"]

    total_pnl = sum(x["pnl_pct"] for x in trades)
    total_r = sum(x["r_mult"] for x in trades)
    win_rate = len(wins) / max(len(trades), 1) * 100
    best_tr = max(trades, key=lambda x: x["pnl_pct"]) if trades else None

    start_str = datetime.fromtimestamp(cutoff_ms / 1000, TZ).strftime("%Y-%m-%d %H:%M")
    end_str = datetime.fromtimestamp(max_ts / 1000, TZ).strftime("%Y-%m-%d %H:%M")

    # 將波段大賺單排在最上方展示區，再列出全部時序明細
    cards_html = []
    for idx, tr in enumerate(trades, 1):
        cls = "pnl-win" if tr["pnl_pct"] > 0 else ("pnl-flat" if tr["pnl_pct"] == 0 else "pnl-loss")
        tag_cls = {
            "tp": "tag-tp",
            "be": "tag-info",
            "time": "tag-time",
            "sl": "tag-sl",
        }.get(tr["cat"], "tag-info")
        detail = (
            f"進場 {tr['entry_price']:g}   停損 {tr['stop_price']:.4g}   目標(4R) {tr['target_price']:.4g}   出場 {tr['exit_price']:.4g}\n"
            f"MA200 {tr['ma200']:.4g} (離MA200 +{tr['dist200_pct']:.2f}%)  ·  長均黏度 {tr['long_spread_pct']:.2f}%  ·  六線黏度 {tr['all6_spread_pct']:.2f}%\n"
            f"前20根箱體 {tr['box_width_pct']:.2f}% ({tr['box_lo']:.4g}～{tr['box_hi']:.4g})  ·  突破量比 {tr['vol_ratio']:.1f}x  ·  盤中最高 +{tr['mfe_pct']:.2f}%"
        )
        card = (
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
            f"<span class='tag'>最高 +{tr['mfe_pct']:.2f}%</span>"
            f"<span class='tag'>六線黏度 {tr['all6_spread_pct']:.2f}%</span>"
            f"<span class='tag'>箱體 {tr['box_width_pct']:.1f}%</span>"
            f"<span class='tag'>量 {tr['vol_ratio']:.1f}x</span>"
            f"</div>"
            f"<pre class='trade-detail'>{html.escape(detail)}</pre>"
            f"<div class='mini-chart'><img src='{tr['img_file']}' alt='{html.escape(tr['symbol'])}' loading='lazy'/></div>"
            f"</article>"
        )
        cards_html.append(card)

    best_desc = (
        f"{best_tr['symbol']} ({best_tr['entry_time']}) {best_tr['pnl_pct']:+.2f}% (盤中最高 +{best_tr['mfe_pct']:.2f}%)"
        if best_tr
        else "—"
    )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 15m MA200 六線收斂起跳 · 近 7 天回測 ({len(trades)} 筆)</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;-webkit-font-smoothing:antialiased}}
.page{{max-width:780px;margin:0 auto;padding:14px 12px 40px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px 18px;margin-bottom:14px}}
h1{{font-size:19px;margin:0 0 6px}}
h2{{font-size:15px;margin:14px 0 8px;color:#e6edf3}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6;margin:0}}
.rules{{margin:0;padding-left:1.25em;color:#c9d1d9;font-size:13px;line-height:1.65}}
.rules li{{margin-bottom:5px}}
code{{font-size:12px;background:#0d1117;padding:2px 5px;border-radius:4px;color:#79c0ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0 10px}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;border:1px solid #21262d}}
.card span{{font-size:12px;color:#8b949e;display:block}}
.card b{{display:block;font-size:19px;margin-top:4px}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 14px}}
.fbtn{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:999px;padding:6px 12px;font-size:12px;cursor:pointer;font-weight:600}}
.fbtn.active{{background:#1f6feb;color:#fff;border-color:#388bfd}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}}
.trade-no{{font-weight:700;font-size:15px}}
.trade-time{{font-size:12px;color:#8b949e;display:block;margin-top:2px}}
.card-pnl{{font-weight:700;font-size:16px;text-align:right;white-space:nowrap}}
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
.note{{margin-top:10px;padding:8px 10px;background:#0d1117;border-radius:8px;color:#f0c14b;font-size:12px;line-height:1.5}}
</style>
</head>
<body>
<div class="page">
  <section class="summary">
    <h1>幣安 USDT 永續 15m · MA200 六線收斂起跳 · 近 7 天全市場回測</h1>
    <p class="muted">回測區間：{start_str} ～ {end_str} (UTC+8) · 全市場 24h 成交額 ≥ 300 萬 USDT 共 {len(data)} 檔永續合約（不指定幣種，只認線型與 K 棒結構）</p>

    <div class="cards">
      <div class="card"><span>獨立觸發筆數</span><b>{len(trades)} 筆</b></div>
      <div class="card"><span>勝率</span><b class="pnl-win">{win_rate:.1f}%</b></div>
      <div class="card"><span>累計報酬 (單利1x)</span><b class="{'pnl-win' if total_pnl >= 0 else 'pnl-loss'}">{total_pnl:+.2f}%</b></div>
      <div class="card"><span>累計盈虧比</span><b class="{'pnl-win' if total_r >= 0 else 'pnl-loss'}">{total_r:+.2f} R</b></div>
    </div>

    <h2>進出場規則（對齊 ETH / FOLKS / FIL / PIPPIN 起漲前結構）</h2>
    <ol class="rules">
      <li><b>六線合一與長均走平</b>：<code>MA99 / MA120 / MA200</code> 最大間距 ≤ <code>1.5%</code>，且全體六條均線 <code>MA7 / 14 / 25 / 99 / 120 / 200</code> 最大間距 ≤ <code>2.2%</code>；<code>MA200</code> 近 12 根斜率介於 <code>-0.15% ～ +0.40%</code>。</li>
      <li><b>窄箱體橫盤墊高</b>：前 20 根 K 棒高低振幅 <code>0.8% ～ 4.5%</code>（藍框處），且近 8 根最低點 &gt; 前段最低點（底部墊高 Higher Low）。</li>
      <li><b>MA200 邊緣帶量起跳</b>：收盤剛突破三條長均線與前 12 根收盤高點，離 <code>MA200</code> 在 <code>+0.05% ～ +1.8%</code> 內（不追高）；<code>MA7 &gt; MA14 ≥ MA25</code>，收紅 K（收在整根上 35% 內）且成交量 ≥ 20 均量 × <code>1.8</code>。</li>
      <li><b>排除大盤新聞齊插針</b>：同一根 15m 若全市場 ≥ 4 個幣同時突破（本週共擋掉 {skipped_spike_bars} 個時間點、{skipped_spike_signals} 筆集體假突破），視為宏觀數據連動針跳過，只做個別幣獨立蓄勢突破。</li>
      <li><b>出場與波段抱單</b>：初始停損設於近期小低點與長均線下沿（最多 <code>-2.0%</code>）；浮盈達 <code>1.2R</code> 推保本（<code>+0.2R</code>）；浮盈達 <code>2.0R</code> 後若收盤跌破 <code>MA14</code> 波段停利，或直達 <code>4.0R</code> 目標停利，最長持有 48 根（12 小時）。</li>
    </ol>

    <p class="note">
      出場分佈：波段停利（4R／破MA14）<b>{len(tp_trades)}</b> 筆 · 移動保本（+0.2R）<b>{len(be_trades)}</b> 筆 · 時間到期 <b>{len(time_trades)}</b> 筆 · 停損 <b>{len(sl_trades)}</b> 筆<br/>
      本週最佳：{html.escape(best_desc)}<br/>
      圖例：白粗線＝MA200；黃/青/粉/紫/綠＝MA7/14/25/99/120；藍虛線框＝前 20 根蓄勢箱體；綠 ▲＝進場；黃 ×＝出場。
    </p>
  </section>

  <div class="filters">
    <button class="fbtn active" onclick="filterCards('all', this)">全部 ({len(trades)})</button>
    <button class="fbtn" onclick="filterCards('tp', this)">波段大賺 4R / 破MA14 ({len(tp_trades)})</button>
    <button class="fbtn" onclick="filterCards('be', this)">移動保本 +0.2R ({len(be_trades)})</button>
    <button class="fbtn" onclick="filterCards('time', this)">時間到期 ({len(time_trades)})</button>
    <button class="fbtn" onclick="filterCards('sl', this)">觸及停損 ({len(sl_trades)})</button>
  </div>

  <div id="cards-container">
    {''.join(cards_html)}
  </div>
</div>
<script>
function filterCards(cat, btn) {{
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.trade-card').forEach(card => {{
    card.style.display = (cat === 'all' || card.dataset.cat === cat) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""
    out_html.write_text(page_html, encoding="utf-8")

    # 同步產生供 htmlpreview.github.io 預覽的 view.html
    rel = out_html.parent.relative_to(REPO).as_posix()
    raw_base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{git_branch()}/{rel}/"
    view_html = out_html.with_name("view.html")
    view_html.write_text(page_html.replace("src='img/", f"src='{raw_base}img/"), encoding="utf-8")

    print(f"已輸出 HTML 報告：{out_html}")
    print(f"已輸出預覽 HTML：{view_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
