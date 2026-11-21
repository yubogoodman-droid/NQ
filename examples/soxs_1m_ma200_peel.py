#!/usr/bin/env python3
"""SOXS 一分 K：短均線從還平的 MA200 那一束裡剛撥開，做多。

長相（對齊 2026-11-04 19:32 那根，收盤 51.12、MA200 51.14）：
- MA200 近 40 根還平（允許緩跌，末端 10 根沒有上弧）
- MA7 / MA14 / MA25 近 5 根斜率由陡到緩，而且都往上
- 三條剛分開，MA25 仍在 MA200 下面
- 收盤貼在 MA200 上、收在 MA7 之上，低點踩到 MA200
- 前 8 根裡短均線曾經纏在一起

出場：停損放前 40 根最低。先拿 1 點，剩下的一半抱到一分 K 收盤跌回 MA200。
另一欄是整筆碰到 1 點、跌回 MA200 或停損就全出。

    python3 examples/soxs_1m_ma200_peel.py --days 30
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
FEE_PCT = 0.08  # 來回，跟黏帶回測同一個假設

# 斜率、價差都是百分比。
S7_MIN = 0.30
GAP_MIN = 0.30
GAP_MAX = 0.75
M200_BODY_MIN = -0.85
M200_BODY_MAX = 0.10
M200_TIP_MAX = 0.03
BUNDLE_MAX = 1.35
TANGLE_GAP = 0.20
TANGLE_BARS = 8
REARM_UNDER = 15
BASE_BARS = 40
TARGET_PTS = 1.0


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        cs = np.cumsum(np.insert(np.asarray(a, dtype=np.float64), 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def slope_pct(arr: np.ndarray, i: int, n: int) -> float | None:
    if i < n:
        return None
    prev = float(arr[i - n])
    cur = float(arr[i])
    if not np.isfinite(prev) or not np.isfinite(cur) or prev == 0.0:
        return None
    return (cur - prev) / prev * 100.0


def is_peel(
    i: int,
    c: np.ndarray,
    l: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    m99: np.ndarray,
    m120: np.ndarray,
    m200: np.ndarray,
) -> bool:
    """這一根是不是「剛從平的 MA200 束裡撥開」。"""
    if i < max(BASE_BARS, TANGLE_BARS, 40):
        return False
    vals = (
        c[i],
        l[i],
        m7[i],
        m14[i],
        m25[i],
        m99[i],
        m120[i],
        m200[i],
        m7[i - 5],
        m25[i - 5],
        m200[i - 10],
        m200[i - 40],
    )
    if not np.isfinite(vals).all() or c[i] <= 0 or m200[i] <= 0:
        return False

    s7 = slope_pct(m7, i, 5)
    s14 = slope_pct(m14, i, 5)
    s25 = slope_pct(m25, i, 5)
    body = slope_pct(m200, i, 40)
    tip = slope_pct(m200, i, 10)
    if None in (s7, s14, s25, body, tip):
        return False
    if not (s7 > s14 > s25 > 0 and s7 >= S7_MIN):
        return False
    if not (M200_BODY_MIN <= body <= M200_BODY_MAX and tip <= M200_TIP_MAX):
        return False
    if not (m7[i] > m14[i] > m25[i] and m25[i] < m200[i] and c[i] > m7[i]):
        return False

    gap = (m7[i] - m25[i]) / c[i] * 100.0
    if not (GAP_MIN <= gap <= GAP_MAX):
        return False
    if not (m200[i] * 0.9995 <= c[i] <= m200[i] * 1.0035):
        return False
    if not (m200[i] * 0.9970 <= l[i] <= m200[i] * 1.0010):
        return False

    bhi = max(m99[i], m120[i], m200[i])
    blo = min(m99[i], m120[i], m200[i])
    if (bhi - blo) / m200[i] * 100.0 > BUNDLE_MAX:
        return False

    tangled = False
    for j in range(i - TANGLE_BARS, i):
        if c[j] <= 0 or not np.isfinite([m7[j], m25[j], m200[j], c[j]]).all():
            continue
        gj = (m7[j] - m25[j]) / c[j] * 100.0
        on_line = c[j] <= m200[j] * 1.001 or m7[j] <= m200[j]
        if gj < TANGLE_GAP and on_line:
            tangled = True
            break
    return tangled


def entry_indices(
    c: np.ndarray,
    l: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    m99: np.ndarray,
    m120: np.ndarray,
    m200: np.ndarray,
    t: np.ndarray,
    t_min: int,
) -> list[int]:
    """每個撥開只進第一根。之後要先有 15 根收盤回到 MA200 下，才看下一次。"""
    armed = False
    under = 0
    out: list[int] = []
    prev = False
    for i in range(len(c)):
        below = bool(np.isfinite(m200[i]) and c[i] < m200[i])
        under = under + 1 if below else 0
        if under >= REARM_UNDER:
            armed = True
        flag = is_peel(i, c, l, m7, m14, m25, m99, m120, m200)
        if int(t[i]) >= t_min and armed and flag and not prev:
            out.append(i)
            armed = False
            under = 0
        prev = flag
    return out


def _fee(entry: float) -> float:
    return entry * FEE_PCT / 100.0


def simulate(i: int, h: np.ndarray, l: np.ndarray, c: np.ndarray, m200: np.ndarray) -> dict:
    """停損＝前 40 根最低。scalp 整筆先碰到 1 點、跌回或停損就出。

    scale 是先拿一半 1 點，另一半抱到收盤跌回 MA200（或停損）。
    同一根先碰到停損就算出停損，1 點不算數。
    """
    entry = float(c[i])
    stop = float(np.min(l[i - BASE_BARS : i]))
    risk = entry - stop
    fee = _fee(entry)

    scalp_px = None
    scalp_reason = "未平"
    scalp_hold = 0
    scale_pnl = 0.0
    half = False
    scale_reason = "未平"
    hold = 0
    mfe = 0.0

    for k in range(i + 1, len(c)):
        hold = k - i
        mfe = max(mfe, float(h[k]) - entry)
        hit_stop = l[k] <= stop
        hit_tgt = h[k] >= entry + TARGET_PTS
        lost = bool(np.isfinite(m200[k]) and c[k] < m200[k])

        if scalp_px is None:
            if hit_stop:
                scalp_px, scalp_reason, scalp_hold = stop, "停損", hold
            elif hit_tgt:
                scalp_px, scalp_reason, scalp_hold = entry + TARGET_PTS, "+1", hold
            elif lost:
                scalp_px, scalp_reason, scalp_hold = float(c[k]), "跌回", hold

        if hit_stop:
            weight = 0.5 if half else 1.0
            scale_pnl += weight * (stop - entry)
            scale_reason = "停損" if not half else "一半後停損"
            break
        if not half and hit_tgt:
            scale_pnl += 0.5 * TARGET_PTS
            half = True
        if lost:
            weight = 0.5 if half else 1.0
            scale_pnl += weight * (float(c[k]) - entry)
            scale_reason = "一半後跌回" if half else "跌回"
            break
    else:
        last = float(c[-1])
        if scalp_px is None:
            scalp_px, scalp_reason, scalp_hold = last, "未平", hold
        weight = 0.5 if half else 1.0
        scale_pnl += weight * (last - entry)
        scale_reason = "未平"

    return {
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "hold": scalp_hold,
        "scale_hold": hold,
        "mfe": mfe,
        "scalp_pts": float(scalp_px) - entry,
        "scalp_net": float(scalp_px) - entry - fee,
        "scalp_reason": scalp_reason,
        "scale_pts": scale_pnl,
        "scale_net": scale_pnl - fee,
        "scale_reason": scale_reason,
    }


def summarize(rows: list[dict], key_pts: str, key_net: str) -> str:
    if not rows:
        return "沒有進場"
    pts = np.array([r[key_pts] for r in rows], dtype=float)
    net = np.array([r[key_net] for r in rows], dtype=float)
    return (
        f"n={len(rows)}  勝率={100 * np.mean(pts > 0):.0f}%  "
        f"毛點={pts.sum():+.2f}  扣費={net.sum():+.2f}  平均扣費={net.mean():+.3f}"
    )


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> dict:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web"})
    cur = start_ms
    rows: list = []
    while cur < end_ms:
        raw = None
        last_err: Exception | None = None
        for attempt in range(6):
            r = sess.get(
                BASE + "/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": cur,
                    "endTime": end_ms - 1,
                    "limit": 1500,
                },
                timeout=20,
            )
            if r.status_code in (429, 418):
                time.sleep(float(r.headers.get("Retry-After") or 1.2 * (attempt + 1)))
                continue
            try:
                r.raise_for_status()
                raw = r.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
        if raw is None:
            raise RuntimeError(f"klines failed: {last_err}")
        if not raw:
            break
        rows.extend(raw)
        last = int(raw[-1][0])
        if len(raw) < 1500:
            break
        cur = last + 60_000
        time.sleep(0.05)
    seen = {int(x[0]): x for x in rows if int(x[0]) + 60_000 <= end_ms}
    kl = [seen[k] for k in sorted(seen)]
    return {
        "t": np.array([int(x[0]) for x in kl], np.int64),
        "o": np.array([float(x[1]) for x in kl], np.float64),
        "h": np.array([float(x[2]) for x in kl], np.float64),
        "l": np.array([float(x[3]) for x in kl], np.float64),
        "c": np.array([float(x[4]) for x in kl], np.float64),
    }


MA_COLORS = {
    7: "#f0b90b",
    14: "#4dd0e1",
    25: "#d28cff",
    99: "#7e57c2",
    120: "#66bb6a",
    200: "#c62828",
}


def _cjk_font():
    from matplotlib import font_manager

    path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path)
    return None


def draw_trade(
    path: Path,
    d: dict,
    mas: dict[int, np.ndarray],
    trade: dict,
    trade_no: int,
) -> None:
    """進場前後的一分 K。黃 MA7、青 MA14、粉 MA25、紫 MA99、綠 MA120、酒紅 MA200。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    font = _cjk_font()
    if font is not None:
        plt.rcParams["font.family"] = font.get_name()
    i = int(trade["i"])
    n = len(d["c"])
    a0 = max(0, i - 80)
    a1 = min(n - 1, i + max(50, min(int(trade["scale_hold"]) + 12, 150)))
    sl = slice(a0, a1 + 1)
    o, h, l, c = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl]
    xs = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(11.6, 6.2), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#eceff1", lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#cfd8dc")
    up = c >= o
    for k in range(len(c)):
        col = "#00c853" if up[k] else "#ff5252"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7, zorder=2)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(float(h[k] - l[k]), 1e-6) * 0.15
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.4, zorder=3))
    for nlen, col in MA_COLORS.items():
        ax.plot(xs, mas[nlen][sl], color=col, lw=1.35 if nlen == 200 else 1.05, label=f"MA{nlen}", zorder=4)

    ex = i - a0
    entry = float(trade["entry"])
    stop = float(trade["stop"])
    y_lo = float(np.nanmin(l))
    y_hi = float(np.nanmax(h))
    for nlen in MA_COLORS:
        seg = mas[nlen][sl]
        if np.isfinite(seg).any():
            y_lo = min(y_lo, float(np.nanmin(seg)))
            y_hi = max(y_hi, float(np.nanmax(seg)))
    pad = max((y_hi - y_lo) * 0.06, 0.05)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    if y_lo - pad <= stop <= y_hi + pad:
        ax.axhline(stop, color="#e35d5d", ls=":", lw=1.0, alpha=0.85, label="停損")
    tgt = entry + TARGET_PTS
    if y_lo - pad <= tgt <= y_hi + pad:
        ax.axhline(tgt, color="#00c853", ls=":", lw=1.0, alpha=0.8, label="+1")
    ax.axvline(ex, color="#fb8c00", ls="--", lw=0.8, alpha=0.9)
    ax.scatter([ex], [entry], s=54, color="#00c853", marker="^", zorder=6)
    ax.scatter([ex], [entry], s=220, facecolors="none", edgecolors="#fb8c00", linewidths=2.0, zorder=7)
    kw = {"fontproperties": font} if font else {}
    ax.annotate("進", (ex, entry), textcoords="offset points", xytext=(8, 10), color="#e65100", fontsize=11, fontweight="bold", **kw)

    sx = ex + int(trade["hold"])
    if 0 <= sx < len(c) and trade["scalp_reason"] != "+1":
        ax.scatter([sx], [c[sx]], s=36, color="#ff5252", marker="x", zorder=6)
        ax.annotate("跌回", (sx, c[sx]), textcoords="offset points", xytext=(6, -12), color="#c62828", fontsize=8, **kw)
    hx = ex + int(trade["scale_hold"])
    if trade["scale_reason"].startswith("一半") and 0 <= hx < len(c) and hx != sx:
        ax.scatter([hx], [c[hx]], s=36, color="#1565c0", marker="x", zorder=6)
        ax.annotate("半出", (hx, c[hx]), textcoords="offset points", xytext=(6, 8), color="#1565c0", fontsize=8, **kw)

    title = (
        f"#{trade_no}  {trade['time']}  進 {entry:.2f}  "
        f"整筆 {trade['scalp_reason']} {trade['scalp_net']:+.2f}  "
        f"一半 {trade['scale_net']:+.2f}  最大 {trade['mfe']:.2f}"
    )
    ax.set_title(title, color="#263238", fontsize=12, loc="left", **kw)
    ax.legend(loc="upper left", frameon=False, ncol=8, fontsize=8)
    step = max(1, len(xs) // 6)
    ticks = list(range(0, len(xs), step))
    labels = [datetime.fromtimestamp(int(d["t"][a0 + k]) / 1000, TZ).strftime("%m-%d %H:%M") for k in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, color="#546e7a", fontsize=8)
    ax.tick_params(colors="#546e7a")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_gallery(rows: list[dict], img_dir: Path, page: Path) -> None:
    cards = []
    for n, r in enumerate(rows, 1):
        name = f"t{n:02d}_{r['time'].replace(' ', '_').replace('-', '')}.png"
        border = "#00c853" if r["scale_net"] > 0 else "#ff5252"
        cards.append(
            f'<a href="img/{name}"><figure style="border-color:{border}">'
            f'<img src="img/{name}" alt="{r["time"]}">'
            f'<figcaption>#{n} {r["time"]}  整筆 {r["scalp_net"]:+.2f}　一半 {r["scale_net"]:+.2f}</figcaption>'
            f"</figure></a>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>SOXS 一分 K MA200 撥開</title>
<style>
body {{ font-family: "WenQuanYi Micro Hei", sans-serif; background:#f7f7f5; color:#263238; margin:24px; }}
h1 {{ font-size:20px; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
figure {{ margin:0; background:#fff; border:3px solid #ccc; border-radius:10px; padding:8px; }}
img {{ width:100%; height:auto; display:block; }}
figcaption {{ padding:8px 4px 2px; font-size:14px; }}
</style>
<h1>SOXSUSDT 一分 K，MA200 束剛撥開，近 30 天 {len(rows)} 筆</h1>
<p>綠框是一半抱到跌回之後還賺錢的。進場圈在撥開那根。紅叉是收盤跌回 MA200。</p>
<div class="grid">{''.join(cards)}</div>
"""
    page.write_text(html, encoding="utf-8")


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def run(symbol: str = "SOXSUSDT", days: int = 30, charts: Path | None = None) -> list[dict]:
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    win_start = now - timedelta(days=days)
    fetch_start = win_start - timedelta(days=2)
    d = fetch_klines(symbol, int(fetch_start.timestamp() * 1000), end_ms)
    c = d["c"]
    mas = {7: sma(c, 7), 14: sma(c, 14), 25: sma(c, 25), 99: sma(c, 99), 120: sma(c, 120), 200: sma(c, 200)}
    m7, m14, m25, m99, m120, m200 = (mas[n] for n in (7, 14, 25, 99, 120, 200))
    t_min = int(win_start.timestamp() * 1000)
    idxs = entry_indices(c, d["l"], m7, m14, m25, m99, m120, m200, d["t"], t_min)
    rows = []
    for i in idxs:
        tr = simulate(i, d["h"], d["l"], c, m200)
        tr["i"] = i
        tr["time"] = hm(int(d["t"][i]))
        tr["ms"] = int(d["t"][i])
        rows.append(tr)
    print(
        f"{symbol} 1 分 K  {hm(int(d['t'][0]))} 暖機 → 統計 {win_start.strftime('%m-%d %H:%M')} ～ {hm(int(d['t'][-1]))}（台北）"
    )
    print(f"K 線 {len(c)} 根")
    print("整筆碰到 1 點或跌回就出：", summarize(rows, "scalp_pts", "scalp_net"))
    print("一半 1 點、一半抱到跌回：", summarize(rows, "scale_pts", "scale_net"))
    print("時間                進場    停損   風險  整筆      扣費   一半          扣費  持有     最大")
    for r in rows:
        print(
            f"{r['time']:16} {r['entry']:7.2f} {r['stop']:7.2f} {r['risk']:5.2f}  "
            f"{r['scalp_reason']:4} {r['scalp_pts']:+6.2f} {r['scalp_net']:+6.2f}  "
            f"{r['scale_reason']:8} {r['scale_pts']:+6.2f} {r['scale_net']:+6.2f}  "
            f"{r['hold']:>3}/{r['scale_hold']:<3} {r['mfe']:5.2f}"
        )
    if charts is not None:
        img = charts / "img"
        img.mkdir(parents=True, exist_ok=True)
        for n, r in enumerate(rows, 1):
            name = f"t{n:02d}_{r['time'].replace(' ', '_').replace('-', '')}.png"
            draw_trade(img / name, d, mas, r, n)
        write_gallery(rows, img, charts / "index.html")
        print(f"圖：{charts / 'index.html'}")
    return rows


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SOXS 一分 K，MA200 束剛撥開，回測")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--symbol", default="SOXSUSDT")
    p.add_argument("--charts", type=Path, default=None, help="每筆一張圖，寫到這個資料夾")
    args = p.parse_args()
    run(args.symbol, args.days, args.charts)
