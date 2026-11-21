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
        "h": np.array([float(x[2]) for x in kl], np.float64),
        "l": np.array([float(x[3]) for x in kl], np.float64),
        "c": np.array([float(x[4]) for x in kl], np.float64),
    }


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def run(symbol: str = "SOXSUSDT", days: int = 30) -> list[dict]:
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    win_start = now - timedelta(days=days)
    fetch_start = win_start - timedelta(days=2)
    d = fetch_klines(symbol, int(fetch_start.timestamp() * 1000), end_ms)
    c = d["c"]
    m7, m14, m25 = sma(c, 7), sma(c, 14), sma(c, 25)
    m99, m120, m200 = sma(c, 99), sma(c, 120), sma(c, 200)
    t_min = int(win_start.timestamp() * 1000)
    idxs = entry_indices(c, d["l"], m7, m14, m25, m99, m120, m200, d["t"], t_min)
    rows = []
    for i in idxs:
        tr = simulate(i, d["h"], d["l"], c, m200)
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
    return rows


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SOXS 一分 K，MA200 束剛撥開，回測")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--symbol", default="SOXSUSDT")
    args = p.parse_args()
    run(args.symbol, args.days)
