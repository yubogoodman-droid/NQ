#!/usr/bin/env python3
"""SOXS 一分 K ＋ 五分 K ＋ 十五分 K：MA200 束撥開站上，做多。

三週期條件（對齊 2026-11-04 19:38 起漲；5m/15m 只用已收完的 K 線，不偷看未來）：
- 1 分 K 長均線空頭收平：MA200 > MA120 > MA99，整束緊，MA200 走平或緩跌、末端沒上弧
- 1 分 K 先前乖乖待在 MA200 下方（前段至少 32/41 根收在 MA200 下）
- 1 分 K 短中均線撥開扇形往上：MA7 > MA14 > MA25 > MA30，5 根斜率由陡到緩且都往上，MA14 已跨過 MA120
- 1 分 K MA7 近 3 根內剛上穿 MA200，收盤 > MA7 > MA200，連續至少 4 根收在 MA200 上方，離 MA200 不超過 1%
- 5 分 K 短均線已翻多：已收線的 5m MA7 > MA14
- 15 分 K 大趨勢不在急跌：已收線的 15m MA200 近 10 根斜率 >= -0.20%

出場：停損放前 40 根最低；收盤跌破 1m MA200 超過 0.3% 算跌回（留一點正常回踩空間）。
兩種一起算：整筆碰到 +1 或跌回/停損就全出；另一種先出一半 +1，剩下一半抱到跌回或停損。

    python3 examples/soxs_1m_ma200_peel.py --days 30 --charts docs/soxs-1m-ma200-peel
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

MA_PERIODS = (7, 14, 25, 30, 60, 99, 120, 200)
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}

# 斜率、價差都是百分比。
S7_MIN = 0.25
M200_BODY_MIN = -0.85
M200_BODY_MAX = 0.05
M200_TIP_MAX = 0.05
BUNDLE_MAX = 1.35
EXT_MAX = 1.00
HOLD_ABOVE = 4
CROSS_LOOKBACK = 3
PRIOR_WINDOW = 45
PRIOR_UNDER_MIN = 32
REARM_UNDER = 25
BASE_BARS = 40
TARGET_PTS = 1.0
EXIT_BUF_PCT = 0.30
M15_M200_SLOPE_MIN = -0.20


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


def last_closed_idx(t_htf: np.ndarray, step_ms: int, t1_open_ms: int) -> int:
    """回傳在 1m 收盤當下（t1_open_ms + 60_000）已經收完的最新高週期 K 線索引。"""
    return int(np.searchsorted(t_htf + step_ms, t1_open_ms + 60_000, side="right")) - 1


def is_peel(
    i: int,
    c: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    m30: np.ndarray,
    m99: np.ndarray,
    m120: np.ndarray,
    m200: np.ndarray,
) -> bool:
    """這一根 1m 是不是「長均線收平、MA7 剛撥開站上 MA200」。"""
    if i < max(BASE_BARS, PRIOR_WINDOW, 40):
        return False
    vals = (
        c[i],
        m7[i],
        m14[i],
        m25[i],
        m30[i],
        m99[i],
        m120[i],
        m200[i],
        m7[i - 5],
        m14[i - 5],
        m25[i - 5],
        m30[i - 5],
        m200[i - 10],
        m200[i - 40],
    )
    if not np.isfinite(vals).all() or c[i] <= 0 or m200[i] <= 0:
        return False

    if not (c[i] > m7[i] > m200[i]):
        return False
    if not any(
        np.isfinite(m7[i - k]) and np.isfinite(m200[i - k]) and m7[i - k] <= m200[i - k]
        for k in range(1, CROSS_LOOKBACK + 1)
    ):
        return False

    if not (m7[i] > m14[i] > m25[i] > m30[i] and m14[i] > m120[i]):
        return False

    s7 = slope_pct(m7, i, 5)
    s14 = slope_pct(m14, i, 5)
    s25 = slope_pct(m25, i, 5)
    s30 = slope_pct(m30, i, 5)
    body = slope_pct(m200, i, 40)
    tip = slope_pct(m200, i, 10)
    if None in (s7, s14, s25, s30, body, tip):
        return False
    if not (s7 > s14 > s25 > 0 and s7 >= S7_MIN and s30 >= 0):
        return False
    if not (M200_BODY_MIN <= body <= M200_BODY_MAX and tip <= M200_TIP_MAX):
        return False

    if not (m200[i] > m120[i] > m99[i]):
        return False
    if (m200[i] - m99[i]) / m200[i] * 100.0 > BUNDLE_MAX:
        return False

    seg_hold = slice(i - HOLD_ABOVE + 1, i + 1)
    if not np.isfinite(m200[seg_hold]).all() or np.any(c[seg_hold] < m200[seg_hold]):
        return False
    ext = (c[i] - m200[i]) / m200[i] * 100.0
    if not (0.0 <= ext <= EXT_MAX):
        return False

    seg_prior = slice(i - PRIOR_WINDOW, i - HOLD_ABOVE)
    if not np.isfinite(m200[seg_prior]).all():
        return False
    if int(np.sum(c[seg_prior] < m200[seg_prior])) < PRIOR_UNDER_MIN:
        return False
    return True


def htf_aligned(
    t1_open_ms: int,
    t5: np.ndarray,
    m5_7: np.ndarray,
    m5_14: np.ndarray,
    t15: np.ndarray,
    m15_200: np.ndarray,
) -> tuple[bool, float | None, float | None]:
    """檢查 5m MA7 > MA14 且 15m MA200 沒有急跌（近 10 根斜率 >= -0.20%）。"""
    j5 = last_closed_idx(t5, INTERVAL_MS["5m"], t1_open_ms)
    j15 = last_closed_idx(t15, INTERVAL_MS["15m"], t1_open_ms)
    if j5 < 14 or j15 < 209:
        return False, None, None
    v5_7, v5_14 = float(m5_7[j5]), float(m5_14[j5])
    s15_200 = slope_pct(m15_200, j15, 10)
    if not np.isfinite([v5_7, v5_14]).all() or s15_200 is None or v5_14 <= 0:
        return False, None, s15_200
    gap5 = (v5_7 - v5_14) / v5_14 * 100.0
    ok = (v5_7 > v5_14) and (s15_200 >= M15_M200_SLOPE_MIN)
    return ok, gap5, s15_200


def entry_indices(
    c: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    m30: np.ndarray,
    m99: np.ndarray,
    m120: np.ndarray,
    m200: np.ndarray,
    t: np.ndarray,
    t_min: int,
    htf: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[int]:
    """每個撥開只進第一根；有傳 htf 時同步驗證 5m/15m。之後要 25 根收在 MA200 下才重裝。"""
    armed = False
    under = 0
    out: list[int] = []
    for i in range(len(c)):
        below = bool(np.isfinite(m200[i]) and c[i] < m200[i])
        under = under + 1 if below else 0
        if under >= REARM_UNDER:
            armed = True
        if int(t[i]) >= t_min and armed and is_peel(i, c, m7, m14, m25, m30, m99, m120, m200):
            if htf is not None:
                t5, m5_7, m5_14, t15, m15_200 = htf
                ok, _, _ = htf_aligned(int(t[i]), t5, m5_7, m5_14, t15, m15_200)
                if not ok:
                    continue
            out.append(i)
            armed = False
            under = 0
    return out


def _fee(entry: float) -> float:
    return entry * FEE_PCT / 100.0


def simulate(i: int, h: np.ndarray, l: np.ndarray, c: np.ndarray, m200: np.ndarray) -> dict:
    """停損＝前 40 根最低。跌回＝收盤跌破 MA200 超過 0.3%。

    scalp 整筆先碰到 +1、跌回或停損就出。
    scale 先拿一半 +1，另一半抱到收盤跌回 MA200（超過 0.3%）或停損。
    同一根先碰到停損就算停損，+1 不算數。
    """
    entry = float(c[i])
    stop = float(np.min(l[i - BASE_BARS : i]))
    risk = entry - stop
    fee = _fee(entry)
    buf_mult = 1.0 - EXIT_BUF_PCT / 100.0

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
        lost = bool(np.isfinite(m200[k]) and c[k] < m200[k] * buf_mult)

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


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1m") -> dict:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web"})
    step = INTERVAL_MS[interval]
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
                    "interval": interval,
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
            raise RuntimeError(f"klines ({interval}) failed: {last_err}")
        if not raw:
            break
        rows.extend(raw)
        last = int(raw[-1][0])
        if len(raw) < 1500:
            break
        cur = last + step
        time.sleep(0.05)
    seen = {int(x[0]): x for x in rows if int(x[0]) + step <= end_ms}
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
    25: "#e066c3",
    30: "#4285f4",
    60: "#2e7d32",
    99: "#7e57c2",
    120: "#66bb6a",
    200: "#ad1457",
}


def _cjk_font():
    from matplotlib import font_manager

    path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path)
    return None


def _idx_at(t: np.ndarray, ms: int) -> int:
    return int(np.searchsorted(t, ms, side="right")) - 1


def draw_trade(
    path: Path,
    d: dict,
    mas: dict[int, np.ndarray],
    trade: dict,
    trade_no: int,
    *,
    tf_label: str = "1分K",
    bar_ms: int = 60_000,
) -> None:
    """進場前後的 K 線，8 條均線跟幣安原圖同色。1 分 K、5 分 K 都走這支。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    font = _cjk_font()
    if font is not None:
        plt.rcParams["font.family"] = font.get_name()
    entry_ms = int(trade["ms"])
    i = _idx_at(d["t"], entry_ms) if bar_ms != 60_000 else int(trade["i"])
    if i < 0:
        i = 0
    n = len(d["c"])
    hold_min = int(trade["hold"])
    scale_min = int(trade["scale_hold"])
    if bar_ms == 60_000:
        before, after = 45, max(60, min(hold_min + 25, 115))
    else:
        # 5 分 K 看得比 1 分 K 寬：進場前約 4 小時，並蓋住這筆持有。
        need = max(hold_min, scale_min) // (bar_ms // 60_000) + 12
        before, after = 48, max(36, min(need, 90))
    a0 = max(0, i - before)
    a1 = min(n - 1, i + after)
    sl = slice(a0, a1 + 1)
    o, h, l, c = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl]
    xs = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#eceff1", lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#cfd8dc")
    up = c >= o
    for k in range(len(c)):
        col = "#00c853" if up[k] else "#ff5252"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.8, zorder=2)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(float(h[k] - l[k]), 1e-6) * 0.15
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.4, zorder=3))
    for nlen, col in MA_COLORS.items():
        ax.plot(xs, mas[nlen][sl], color=col, lw=1.5 if nlen == 200 else 1.0, label=f"MA{nlen}", zorder=4)

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

    sx = _idx_at(d["t"], entry_ms + hold_min * 60_000) - a0
    if 0 <= sx < len(c):
        if trade["scalp_reason"] == "+1":
            ax.scatter([sx], [tgt], s=70, color="#00c853", marker="*", zorder=7)
            ax.annotate("+1", (sx, tgt), textcoords="offset points", xytext=(6, 6), color="#00a152", fontsize=9, fontweight="bold", **kw)
        elif trade["scalp_reason"] in ("跌回", "停損"):
            mark = stop if trade["scalp_reason"] == "停損" else float(c[sx])
            ax.scatter([sx], [mark], s=36, color="#ff5252", marker="x", zorder=6)
            ax.annotate(trade["scalp_reason"], (sx, mark), textcoords="offset points", xytext=(6, -12), color="#c62828", fontsize=8, **kw)
    hx = _idx_at(d["t"], entry_ms + scale_min * 60_000) - a0
    if trade["scale_reason"].startswith("一半") and 0 <= hx < len(c) and hx != sx:
        ax.scatter([hx], [c[hx]], s=36, color="#1565c0", marker="x", zorder=6)
        ax.annotate("半出", (hx, c[hx]), textcoords="offset points", xytext=(6, 8), color="#1565c0", fontsize=8, **kw)

    title = (
        f"{tf_label}  #{trade_no}  {trade['time']}  進 {entry:.2f}  "
        f"整筆 {trade['scalp_reason']} {trade['scalp_net']:+.2f}  "
        f"一半 {trade['scale_net']:+.2f}  最大 {trade['mfe']:.2f}"
    )
    ax.set_title(title, color="#263238", fontsize=12, loc="left", **kw)
    ax.legend(loc="upper left", frameon=False, ncol=10, fontsize=7.5)
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


def _png_name(n: int, when: str, suffix: str = "") -> str:
    stamp = when.replace("-", "").replace(" ", "_").replace(":", "")
    extra = f"_{suffix}" if suffix else ""
    return f"t{n:02d}_{stamp}{extra}.png"


def write_gallery(rows: list[dict], img_dir: Path, page: Path, window_label: str = "") -> None:
    """手機寬度的卡片頁，跟其他回測 view.html 同一套。"""
    del img_dir
    hit1 = sum(1 for r in rows if r["scalp_reason"] == "+1")
    net = sum(r["scale_net"] for r in rows)
    scalp_net = sum(r["scalp_net"] for r in rows)
    net_cls = "pnl-win" if net > 0 else "pnl-loss" if net < 0 else "pnl-flat"
    sc_cls = "pnl-win" if scalp_net > 0 else "pnl-loss" if scalp_net < 0 else "pnl-flat"
    cards = []
    for n, r in enumerate(rows, 1):
        name = _png_name(n, r["time"])
        name5 = _png_name(n, r["time"], "5m")
        cls = "pnl-win" if r["scale_net"] > 0 else "pnl-loss" if r["scale_net"] < 0 else "pnl-flat"
        tag = "tag-tp" if r["scalp_reason"] == "+1" else "tag-sl"
        htf_line = ""
        if r.get("gap5") is not None and r.get("s15_200") is not None:
            htf_line = f"\n5m 7-14 {r['gap5']:+.2f}%  15m MA200斜率 {r['s15_200']:+.2f}%"
        cards.append(
            f"""<article class='trade-card'><header class='card-header'><div class='card-title'><span class='trade-no'>#{n}</span><span class='trade-time'>{r['time']} 台北</span></div><div class='card-pnl {cls}'>{r['scale_net']:+.2f}</div></header><div class='tags'><span class='tag {tag}'>{r['scalp_reason']}</span><span class='tag tag-info'>一半 {r['scale_reason']}</span></div><pre class='trade-detail'>進場 {r['entry']:.2f}
停損 {r['stop']:.2f}  風險 {r['risk']:.2f}
整筆 {r['scalp_pts']:+.2f}  扣費 {r['scalp_net']:+.2f}  {r['hold']} 分
一半 {r['scale_pts']:+.2f}  扣費 {r['scale_net']:+.2f}  {r['scale_hold']} 分
最大浮盈 {r['mfe']:.2f}{htf_line}</pre><div class='mini-chart'><div class='chart-label'>1 分 K</div><img src='img/{name}' alt='#{n} {r['time']} 1分K' style='width:100%;display:block;border-radius:10px'/><div class='chart-label'>5 分 K</div><img src='img/{name5}' alt='#{n} {r['time']} 5分K' style='width:100%;display:block;border-radius:10px'/></div></article>"""
        )
    win_str = f"台北 {window_label}。" if window_label else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>SOXS 1m＋5m＋15m MA200 撥開</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
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
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.chart-label{{font-size:12px;font-weight:700;color:#8b949e;padding:8px 8px 4px}}
.chart-label + img{{margin-bottom:4px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>SOXSUSDT 1m＋5m＋15m，MA200 束撥開</h1>
<p class="muted">{win_str}1 分 K：MA200 &gt; MA120 &gt; MA99 收平，MA7 剛上穿 MA200 且站穩 4 根；同步確認已收線 5 分 K MA7 &gt; MA14、15 分 K MA200 沒急跌（10 根斜率 ≥ -0.20%）。每筆下面先是當時的 1 分 K，再是同一時間的 5 分 K。收盤跌破 1m MA200 超過 0.3% 算跌回。橘圈是進場，綠星是 +1，紅叉是跌回。</p>
<div class="cards">
<div class="card">筆數<b>{len(rows)}</b></div>
<div class="card">拿到 +1<b>{hit1}/{len(rows)}</b></div>
<div class="card">一半扣費<b class="{net_cls}">{net:+.2f}</b></div>
<div class="card">整筆扣費<b class="{sc_cls}">{scalp_net:+.2f}</b></div>
</div>
</section>
{''.join(cards)}
</div></body></html>
"""
    page.write_text(html, encoding="utf-8")
    if page.name == "index.html":
        (page.parent / "view.html").write_text(html, encoding="utf-8")


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def run(symbol: str = "SOXSUSDT", days: int = 30, charts: Path | None = None) -> list[dict]:
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    win_start = now - timedelta(days=days)
    d = fetch_klines(symbol, int((win_start - timedelta(days=2)).timestamp() * 1000), end_ms, "1m")
    d5 = fetch_klines(symbol, int((win_start - timedelta(days=5)).timestamp() * 1000), end_ms, "5m")
    d15 = fetch_klines(symbol, int((win_start - timedelta(days=5)).timestamp() * 1000), end_ms, "15m")

    c = d["c"]
    mas = {n: sma(c, n) for n in MA_PERIODS}
    m7, m14, m25, m30, m99, m120, m200 = (mas[n] for n in (7, 14, 25, 30, 99, 120, 200))
    mas5 = {n: sma(d5["c"], n) for n in MA_PERIODS}
    m5_7, m5_14 = mas5[7], mas5[14]
    m15_200 = sma(d15["c"], 200)
    htf = (d5["t"], m5_7, m5_14, d15["t"], m15_200)

    t_min = int(win_start.timestamp() * 1000)
    idxs = entry_indices(c, m7, m14, m25, m30, m99, m120, m200, d["t"], t_min, htf=htf)
    rows = []
    for i in idxs:
        tr = simulate(i, d["h"], d["l"], c, m200)
        _, gap5, s15_200 = htf_aligned(int(d["t"][i]), *htf)
        tr["i"] = i
        tr["time"] = hm(int(d["t"][i]))
        tr["ms"] = int(d["t"][i])
        tr["gap5"] = gap5
        tr["s15_200"] = s15_200
        rows.append(tr)
    window_label = f"{win_start.strftime('%Y-%m-%d %H:%M')} → {hm(int(d['t'][-1]))}"
    print(
        f"{symbol} 1m+5m+15m  統計 {win_start.strftime('%m-%d %H:%M')} ～ {hm(int(d['t'][-1]))}（台北）"
    )
    print(f"K 線 1m={len(c)}  5m={len(d5['c'])}  15m={len(d15['c'])}")
    print("整筆碰到 1 點或跌回就出：", summarize(rows, "scalp_pts", "scalp_net"))
    print("一半 1 點、一半抱到跌回：", summarize(rows, "scale_pts", "scale_net"))
    print("時間                進場    停損   風險  整筆      扣費   一半          扣費  持有     最大   5m7-14  15m200")
    for r in rows:
        print(
            f"{r['time']:16} {r['entry']:7.2f} {r['stop']:7.2f} {r['risk']:5.2f}  "
            f"{r['scalp_reason']:4} {r['scalp_pts']:+6.2f} {r['scalp_net']:+6.2f}  "
            f"{r['scale_reason']:8} {r['scale_pts']:+6.2f} {r['scale_net']:+6.2f}  "
            f"{r['hold']:>3}/{r['scale_hold']:<3} {r['mfe']:5.2f}  "
            f"{r['gap5']:+6.2f}% {r['s15_200']:+6.2f}%"
        )
    if charts is not None:
        img = charts / "img"
        img.mkdir(parents=True, exist_ok=True)
        for old in img.glob("*.png"):
            old.unlink()
        for n, r in enumerate(rows, 1):
            draw_trade(img / _png_name(n, r["time"]), d, mas, r, n)
            draw_trade(
                img / _png_name(n, r["time"], "5m"),
                d5,
                mas5,
                r,
                n,
                tf_label="5分K",
                bar_ms=INTERVAL_MS["5m"],
            )
        write_gallery(rows, img, charts / "index.html", window_label=window_label)
        print(f"圖：{charts / 'index.html'}")
    return rows


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SOXS 1m+5m+15m，MA200 束撥開站上，回測")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--symbol", default="SOXSUSDT")
    p.add_argument("--charts", type=Path, default=None, help="每筆一張圖，寫到這個資料夾")
    args = p.parse_args()
    run(args.symbol, args.days, args.charts)
