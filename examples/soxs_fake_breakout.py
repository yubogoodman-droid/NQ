#!/usr/bin/env python3
"""SOXSUSDT 1 分 K 假突破做空回測。

對齊 2026-11-12 22:14（台北時間）那波：價格刺過近兩小時前高後，
超漲停在 3.5% 以內，接著收盤跌破 MA25 與 MA30，再掛空在
前一根 MA14／MA25 的回測價。停損放在假突破高點上方，停利 2.5R。

用法:
  python3 examples/soxs_fake_breakout.py
  python3 examples/soxs_fake_breakout.py --days 30 --html docs/soxs-fake-breakout-30d/index.html
  python3 examples/soxs_fake_breakout.py --mode breakdown --session all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
NY = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "soxs-fake-breakout-30d" / "index.html"
BASE = "https://www.binance.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Clienttype": "web", "Accept": "application/json"})

MA_COLORS = {
    7: "#f0b90b",
    14: "#4dd0e1",
    25: "#d28cff",
    30: "#42a5f5",
    60: "#ff8a4c",
    99: "#7e57c2",
    120: "#66bb6a",
    200: "#c62828",
}
MA_WIDTH = {7: 1.15, 14: 1.05, 25: 2.05, 30: 1.15, 60: 1.05, 99: 1.15, 120: 1.05, 200: 1.15}


@dataclass(frozen=True)
class Params:
    """假突破做空參數。預設對齊 2026-11-12 22:26 那筆回測空單。"""

    lookback: int = 120
    min_sep: int = 15
    min_overshoot: float = 0.002
    max_overshoot: float = 0.035
    break_window: int = 25
    retest_window: int = 10
    target_r: float = 2.5
    max_hold: int = 60
    max_risk: float = 0.035
    min_risk: float = 0.002
    stop_buffer: float = 0.001
    # limit：回測掛單 maker 0.02% + 出場 taker 0.04%。breakdown 在 run 裡改 0.08%。
    fee_pct: float = 0.0006
    mode: str = "limit"  # limit | breakdown
    session: str = "tpe_us"  # tpe_us | ny | all
    days: int = 30


@dataclass
class Trade:
    prev_idx: int
    prev_high: float
    peak_idx: int
    peak_high: float
    break_idx: int
    bd_idx: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str
    gross_pct: float
    net_pct: float
    r_multiple: float
    ma7: float
    ma14: float
    ma25: float
    ma30: float


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().to_numpy(float)


def in_session(ts: pd.Timestamp, session: str) -> bool:
    """美股相關時段。tpe_us 是這次回測用的台北 20:00–05:00 窗。"""
    if session == "all":
        return True
    if ts.tzinfo is None:
        ts = ts.tz_localize(TPE)
    if session == "tpe_us":
        local = ts.tz_convert(TPE)
        wd, hr = local.weekday(), local.hour
        return (wd in (0, 1, 2, 3, 4) and hr >= 20) or (wd in (1, 2, 3, 4, 5) and hr < 5)
    if session == "ny":
        local = ts.tz_convert(NY)
        if local.weekday() >= 5:
            return False
        minutes = local.hour * 60 + local.minute
        return 7 * 60 <= minutes < 16 * 60
    raise ValueError(f"unknown session: {session}")


def _limit_price(prev_close: float, ma14: float, ma25: float) -> float:
    """前一根已收盤才能知道的掛單價：MA14／MA25 中間帶的下緣，且要在市價之上。"""
    mid = min(ma25, 0.5 * (ma14 + ma25))
    return max(prev_close * 1.0002, mid)


def _stop_price(peak_high: float, entry: float, params: Params) -> Optional[float]:
    stop = peak_high * (1.0 + params.stop_buffer)
    if stop <= entry * (1.0 + params.min_risk):
        stop = entry * (1.0 + params.min_risk * 2.0)
    risk = (stop - entry) / entry
    if risk <= params.min_risk or risk > params.max_risk:
        return None
    return stop


def simulate_exit(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    max_hold: int,
) -> Tuple[int, float, str]:
    """進場那根只認停損；停利從下一根開始。同一根先算停損。"""
    last = len(close) - 1
    end = min(last, entry_idx + max_hold)
    for m in range(entry_idx, end + 1):
        if high[m] >= stop_price:
            return m, stop_price, "stop"
        if m > entry_idx and low[m] <= target_price:
            return m, target_price, "target"
    reason = "time" if end == entry_idx + max_hold else "open"
    return end, float(close[end]), reason


def run_backtest(
    df: pd.DataFrame,
    params: Params = Params(),
) -> Tuple[List[Trade], Dict[str, int]]:
    """在已收盤的 1 分 K 上跑假突破空單。均線只用當根以前的收盤。"""
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    ma7 = sma(c, 7)
    ma14 = sma(c, 14)
    ma25 = sma(c, 25)
    ma30 = sma(c, 30)
    n = len(df)
    funnel = {
        "breakout": 0,
        "runaway": 0,
        "breakdown": 0,
        "risk_skip": 0,
        "entered": 0,
    }
    if params.days > 0 and n:
        cutoff = df.index[-1] - pd.Timedelta(days=params.days)
    else:
        cutoff = df.index[0] if n else pd.Timestamp("1970-01-01", tz=TPE)

    trades: List[Trade] = []
    i = max(params.lookback + 1, 30)
    busy_until = -1
    while i < n:
        if i <= busy_until or df.index[i] < cutoff or not in_session(df.index[i], params.session):
            i += 1
            continue
        if np.isnan(ma30[i]) or np.isnan(ma25[i]):
            i += 1
            continue

        win_lo = i - params.lookback
        win_hi = i - params.min_sep
        if win_hi < win_lo:
            i += 1
            continue
        window = h[win_lo : win_hi + 1]
        prev_rel = int(np.argmax(window))
        prev_idx = win_lo + prev_rel
        prev_high = float(window[prev_rel])
        if prev_high <= 0 or np.max(h[prev_idx + 1 : i]) > prev_high or h[i] <= prev_high:
            i += 1
            continue
        dipped = False
        for j in range(prev_idx + 1, i):
            if not np.isnan(ma25[j]) and c[j] < ma25[j]:
                dipped = True
                break
        if not dipped:
            i += 1
            continue

        funnel["breakout"] += 1
        peak_h = float(h[i])
        peak_idx = i
        bd_idx: Optional[int] = None
        aborted = False
        last_j = min(n - 1, i + params.break_window)
        for j in range(i, last_j + 1):
            if h[j] > peak_h:
                peak_h = float(h[j])
                peak_idx = j
            overshoot = peak_h / prev_high - 1.0
            if overshoot > params.max_overshoot:
                aborted = True
                break
            if j <= peak_idx or overshoot < params.min_overshoot:
                continue
            if np.isnan(ma7[j]) or np.isnan(ma14[j]) or np.isnan(ma25[j]) or np.isnan(ma30[j]):
                continue
            if c[j] < ma25[j] and c[j] < ma30[j] and c[j] < prev_high and ma7[j] < ma14[j]:
                bd_idx = j
                break
        if aborted:
            funnel["runaway"] += 1
        if bd_idx is None:
            i += 1
            continue
        funnel["breakdown"] += 1

        entry_idx: Optional[int] = None
        entry_px = 0.0
        stop_px = 0.0
        if params.mode == "breakdown":
            fill_i = bd_idx + 1
            if fill_i < n and in_session(df.index[fill_i], params.session):
                px = float(o[fill_i])
                if (ma25[bd_idx] - c[bd_idx]) / ma25[bd_idx] <= 0.025:
                    stop = _stop_price(peak_h, px, params)
                    if stop is None:
                        funnel["risk_skip"] += 1
                    elif h[fill_i] < stop:
                        entry_idx, entry_px, stop_px = fill_i, px, stop
                    else:
                        # 開盤那根已經打到停損，視為沒做成。
                        funnel["risk_skip"] += 1
        elif params.mode == "limit":
            last_k = min(n - 1, bd_idx + params.retest_window)
            for k in range(bd_idx + 1, last_k + 1):
                prev_c = float(c[k - 1])
                if np.isnan(ma25[k - 1]) or np.isnan(ma14[k - 1]):
                    continue
                if prev_c >= ma25[k - 1]:
                    if prev_c > prev_high:
                        break
                    continue
                lim = _limit_price(prev_c, float(ma14[k - 1]), float(ma25[k - 1]))
                stop = _stop_price(peak_h, lim, params)
                if stop is None:
                    funnel["risk_skip"] += 1
                    continue
                if h[k] < lim:
                    continue
                if not in_session(df.index[k], params.session):
                    continue
                fill = float(o[k]) if o[k] > lim else lim
                if fill >= stop:
                    # 開盤已經站上停損，空單沒有放得進去。
                    funnel["risk_skip"] += 1
                    break
                entry_idx, entry_px, stop_px = k, fill, stop
                break
        else:
            raise ValueError(f"unknown mode: {params.mode}")

        if entry_idx is None:
            i = bd_idx + 1
            continue

        risk = stop_px - entry_px
        target = entry_px - params.target_r * risk
        exit_idx, exit_px, reason = simulate_exit(
            h, l, c, entry_idx, entry_px, stop_px, target, params.max_hold
        )
        gross = (entry_px - exit_px) / entry_px
        fee = params.fee_pct
        if params.mode == "breakdown" and params.fee_pct == Params().fee_pct:
            fee = 0.0008
        trades.append(
            Trade(
                prev_idx=prev_idx,
                prev_high=prev_high,
                peak_idx=peak_idx,
                peak_high=peak_h,
                break_idx=i,
                bd_idx=bd_idx,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry_px,
                stop_price=stop_px,
                target_price=target,
                exit_price=exit_px,
                exit_reason=reason,
                gross_pct=gross,
                net_pct=gross - fee,
                r_multiple=(entry_px - exit_px) / risk,
                ma7=float(ma7[entry_idx]),
                ma14=float(ma14[entry_idx]),
                ma25=float(ma25[entry_idx]),
                ma30=float(ma30[entry_idx]),
            )
        )
        funnel["entered"] += 1
        busy_until = exit_idx
        i = exit_idx + 1

    return trades, funnel


def summarize(trades: Sequence[Trade]) -> dict:
    n = len(trades)
    nets = [t.net_pct for t in trades]
    wins = sum(1 for p in nets if p > 0)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {
        "count": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "net_pct": float(sum(nets) * 100.0),
        "avg_pct": float(np.mean(nets) * 100.0) if n else 0.0,
        "median_pct": float(np.median(nets) * 100.0) if n else 0.0,
        "sum_r": float(sum(t.r_multiple for t in trades)),
        "best_pct": float(max(nets) * 100.0) if n else 0.0,
        "worst_pct": float(min(nets) * 100.0) if n else 0.0,
        "reasons": reasons,
    }


def score_window(df: pd.DataFrame, trades: Sequence[Trade], minutes: int, fee_pct: float) -> dict:
    """同一批進場，只看到進場後 minutes 根。先碰到停損或停利就用那個價。"""
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    nets: List[float] = []
    reasons: Dict[str, int] = {}
    for trade in trades:
        _, exit_px, reason = simulate_exit(
            high,
            low,
            close,
            trade.entry_idx,
            trade.entry_price,
            trade.stop_price,
            trade.target_price,
            minutes,
        )
        nets.append((trade.entry_price - exit_px) / trade.entry_price - fee_pct)
        reasons[reason] = reasons.get(reason, 0) + 1
    n = len(nets)
    wins = sum(1 for pnl in nets if pnl > 0)
    return {
        "minutes": minutes,
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "net_pct": float(sum(nets) * 100.0),
        "reasons": reasons,
    }


def fetch_klines(symbol: str, days: int) -> pd.DataFrame:
    """拉幣安 U 本位 1 分 K。多抓 2 天給 MA200 熱身，未收盤的當根丟掉。"""
    bar_ms = 60_000
    need = int((days + 2) * 86_400_000 / bar_ms) + 50
    raw: list = []
    end_ms: Optional[int] = None
    now_ms = int(time.time() * 1000)
    while len(raw) < need:
        params: dict = {"symbol": symbol, "interval": "1m", "limit": min(1500, need - len(raw))}
        if end_ms is not None:
            params["endTime"] = end_ms
        last_err: Optional[Exception] = None
        chunk = None
        for attempt in range(5):
            try:
                resp = SESSION.get(BASE + "/fapi/v1/klines", params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                chunk = resp.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
        if chunk is None:
            raise RuntimeError(f"klines failed: {last_err}")
        if not chunk:
            break
        raw = list(chunk) + raw
        end_ms = int(chunk[0][0]) - 1
        if len(chunk) < int(params["limit"]):
            break
        time.sleep(0.05)
    if raw and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def _fmt_ts(ts: pd.Timestamp, with_date: bool = True) -> str:
    local = ts.tz_convert(TPE) if ts.tzinfo else ts
    return local.strftime("%Y-%m-%d %H:%M" if with_date else "%m-%d %H:%M")


def _equity_svg(pnls_pct: Sequence[float], width: int = 520, height: int = 120) -> str:
    if not pnls_pct:
        return ""
    eq = np.cumsum(np.asarray(pnls_pct, dtype=float) * 100.0)
    xs = np.linspace(0, width, len(eq) + 1)
    ys = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys.min()), float(ys.max())
    pad = max(0.4, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys[i]):.1f}" for i in range(len(ys)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def _use_cjk_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def draw_trade_png(df: pd.DataFrame, trade: Trade, path: Path, trade_no: int, symbol: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _use_cjk_font()
    n = len(df)
    end = min(n - 1, trade.exit_idx + 20)
    start = max(0, min(trade.prev_idx, trade.entry_idx) - 15)
    if end - start > 160:
        start = max(0, end - 160)
    window = df.iloc[start : end + 1]
    xs = np.arange(len(window))
    o = window["Open"].to_numpy(float)
    h = window["High"].to_numpy(float)
    l = window["Low"].to_numpy(float)
    c = window["Close"].to_numpy(float)
    vol = window["Volume"].to_numpy(float) if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}, facecolor="#0c1210"
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    colors = []
    for k in range(len(window)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.6)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.2))
        colors.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(xs, vol, width=0.8, color=colors, linewidth=0)
    for period, col in MA_COLORS.items():
        ma = close_full.rolling(period, min_periods=period).mean().iloc[start : end + 1]
        ax.plot(xs, ma, color=col, lw=MA_WIDTH[period], label=f"MA{period}")

    ax.axhline(trade.prev_high, color="#8aa193", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.9)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.85)

    def mark(idx: int, price: float, color: str, text: str, marker: str, yoff: int) -> None:
        x = idx - start
        if 0 <= x < len(window):
            ax.scatter([x], [price], s=36, color=color, marker=marker, zorder=6)
            ax.annotate(text, (x, price), textcoords="offset points", xytext=(0, yoff), ha="center", color=color, fontsize=8)

    mark(trade.peak_idx, trade.peak_high, "#f0c14b", "假突破", "o", 8)
    mark(trade.entry_idx, trade.entry_price, "#ff5252", "空", "v", -14)
    mark(trade.exit_idx, trade.exit_price, "#00c805" if trade.net_pct > 0 else "#ff8a80", "出", "x", 10)

    et = df.index[trade.entry_idx].tz_convert(TPE)
    xt = df.index[trade.exit_idx].tz_convert(TPE)
    sign = "+" if trade.net_pct >= 0 else ""
    ax.set_title(
        f"#{trade_no}  {symbol}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.net_pct * 100:.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [window.index[k].tz_convert(TPE).strftime("%m-%d %H:%M") for k in ticks],
        color="#8aa193",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _rules_html(params: Params, symbol: str) -> str:
    session = {
        "tpe_us": "只在台北時間週一至週五 20:00–隔日 05:00 尋找與進場（約美東正規盤加開盤前兩小時；冬令時對得上 07:00–16:00 ET，夏令時會晚一小時）。",
        "ny": "只在美東週一至週五 07:00–16:00 尋找與進場。",
        "all": "全時段都做，不限美股開盤。",
    }[params.session]
    entry = (
        "跌破後 1～10 根內，前一根收盤仍在 MA25 下方，才掛空單。"
        "掛單價＝前一根 MA14 與 MA25 中價、以及前一根 MA25，取較低者；"
        "若這個價低於前一根收盤，就改掛在前一根收盤上方 0.02%，避免變成追空。"
        "當根最高價碰到掛單才成交；開盤價已高於掛單時，以開盤價賣出。"
        if params.mode == "limit"
        else "跌破那根的下一根，以開盤價直接空。跌破當根收盤離 MA25 超過 2.5% 就不追。"
    )
    fee = "0.06%（掛單 maker 0.02% + 出場 taker 0.04%）" if params.mode == "limit" else "0.08%（進出都算 taker）"
    return f"""
<ol class="rules">
<li><b>資料</b>：幣安 U 本位 <code>{escape(symbol)}</code> 1 分 K，時區 Asia/Taipei。未收盤的當根先丟掉。</li>
<li><b>回測窗</b>：只統計最後一根往前 {params.days} 天內進場的單。MA 用更早的 K 熱身。</li>
<li><b>時段</b>：{escape(session)}</li>
<li><b>前高</b>：當根往前第 {params.min_sep}～{params.lookback} 根的最高價，而且這段之後到當根之前都沒再摸到該高點。前高之後、突破之前，收盤曾經低於 MA25。</li>
<li><b>假突破</b>：高點刺過前高，往後 {params.break_window} 根內的最高超漲要落在 {params.min_overshoot * 100:.1f}%～{params.max_overshoot * 100:.1f}%。超過上限當成真突破，整段放棄。</li>
<li><b>跌破</b>：超漲達標之後，第一根同時收盤 &lt; MA25、收盤 &lt; MA30、收盤 &lt; 前高，而且 MA7 &lt; MA14。</li>
<li><b>進場</b>：{escape(entry)}進場那根若最高價已經碰到停損，這筆不算。</li>
<li><b>停損</b>：假突破最高價 × {1 + params.stop_buffer:.3f}。停損距離不到 {params.min_risk * 100:.1f}% 或超過 {params.max_risk * 100:.1f}% 就不做。</li>
<li><b>出場</b>：停利 {params.target_r:g}R。進場那根只允許被停損，停利從下一根才算；同一根先算停損。最長 {params.max_hold} 根，否則用該根收盤出場。</li>
<li><b>成本與持倉</b>：來回扣 {escape(fee)}，未扣資金費率。同一時間只留一筆。</li>
</ol>
<p class="note">價格報酬是永續合約本身的漲跌，不是再乘保證金槓桿。SOXS 現貨本來就是 3 倍反向半導體；這裡沒有再疊一層槓桿，也沒有模擬強平。</p>
"""


def write_html(
    path: Path,
    df: pd.DataFrame,
    trades: List[Trade],
    funnel: Dict[str, int],
    params: Params,
    symbol: str,
    comparisons: Sequence[Tuple[str, dict]],
    window_stats: Optional[dict] = None,
) -> Path:
    stats = summarize(trades)
    cards: List[str] = []
    img_dir = path.parent / "img"
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx].tz_convert(TPE)
        xt = df.index[t.exit_idx].tz_convert(TPE)
        img_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, t, img_dir / img_name, i, symbol)
        cls = "pnl-win" if t.net_pct > 0 else ("pnl-flat" if t.net_pct == 0 else "pnl-loss")
        tag_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        risk = t.stop_price - t.entry_price
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div><span class='trade-no'>#{i} · {escape(symbol)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.net_pct * 100:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {tag_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag'>{t.r_multiple:+.2f}R</span>"
            f"<span class='tag'>前高 {t.prev_high:.2f}</span>"
            f"<span class='tag'>假高 {t.peak_high:.2f}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  stop {t.stop_price:.2f}  (+{risk:.2f})\n"
            f"target {t.target_price:.2f}  exit {t.exit_price:.2f}  {t.exit_reason}\n"
            f"前高 {_fmt_ts(df.index[t.prev_idx], False)} {t.prev_high:.2f}  "
            f"假突破 {_fmt_ts(df.index[t.peak_idx], False)} {t.peak_high:.2f}\n"
            f"跌破 {_fmt_ts(df.index[t.bd_idx], False)}  "
            f"MA7 {t.ma7:.2f} / MA14 {t.ma14:.2f} / MA25 {t.ma25:.2f} / MA30 {t.ma30:.2f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{img_name}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    start = df.index[0].tz_convert(TPE).strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].tz_convert(TPE).strftime("%Y-%m-%d %H:%M")
    entry_from = (df.index[-1] - pd.Timedelta(days=params.days)).tz_convert(TPE).strftime("%Y-%m-%d %H:%M") if params.days else start
    total_cls = "pnl-win" if stats["net_pct"] >= 0 else "pnl-loss"
    cmp_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{info['count']}</td>"
        f"<td>{info['win_rate']:.1f}%</td>"
        f"<td class='{'pnl-win' if info['net_pct']>=0 else 'pnl-loss'}'>{info['net_pct']:+.2f}%</td>"
        f"<td>{info['sum_r']:+.2f}R</td></tr>"
        for name, info in comparisons
    )
    reasons = stats["reasons"]
    window_line = ""
    if window_stats and window_stats.get("count"):
        wreasons = window_stats.get("reasons") or {}
        window_line = (
            f"<h2>半小時勝率</h2>"
            f"<p class='muted'>同一 {window_stats['count']} 筆進場，只看到第 {window_stats['minutes']} 根。"
            f"先碰到停損 {wreasons.get('stop', 0)} 筆、先碰到停利 {wreasons.get('target', 0)} 筆、"
            f"時間到用收盤 {wreasons.get('time', 0)} 筆。</p>"
            f"<div class='cards'>"
            f"<div class='card'>半小時筆數<b>{window_stats['count']}</b></div>"
            f"<div class='card'>半小時勝率<b>{window_stats['win_rate']:.1f}%</b></div>"
            f"<div class='card'>半小時淨報酬<b class='{'pnl-win' if window_stats['net_pct']>=0 else 'pnl-loss'}'>{window_stats['net_pct']:+.2f}%</b></div>"
            f"<div class='card'>半小時勝/負<b>{window_stats['wins']}/{window_stats['count']-window_stats['wins']}</b></div>"
            f"</div>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(symbol)} 假突破做空 · {params.days} 天</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:16px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e;display:block}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.rules{{margin:0;padding-left:1.2em;color:#c9d1d9;font-size:13px;line-height:1.65}}
.rules li{{margin-bottom:6px}}
code{{font-size:12px;background:#0d1117;padding:1px 4px;border-radius:4px}}
.note{{margin-top:8px;padding:8px 10px;background:#0d1117;border-radius:8px;color:#f0c14b;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{border-bottom:1px solid #21262d;padding:6px 4px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 1 分 K 假突破做空</h1>
<p class="muted">進場 {escape(entry_from)} → {escape(end)} 台北時間 · K 線熱身從 {escape(start)} · {len(df)} 根 · 進場模式 {escape(params.mode)} · 時段 {escape(params.session)}</p>
<h2>觸發規則（對齊程式）</h2>
{_rules_html(params, symbol)}
<h2>回測統計</h2>
<p class="muted">漏斗：刺破前高 {funnel.get('breakout', 0)} → 超漲過大放棄 {funnel.get('runaway', 0)} → 跌破 MA25/30 {funnel.get('breakdown', 0)} → 進場 {funnel.get('entered', 0)}
（風險不合 {funnel.get('risk_skip', 0)}）
<br/>出場：停利 {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未走完 {reasons.get('open', 0)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">淨報酬<b class="{total_cls}">{stats['net_pct']:+.2f}%</b></div>
<div class="card">合計 R<b class="{total_cls}">{stats['sum_r']:+.2f}R</b></div>
<div class="card">平均<b class="{total_cls}">{stats['avg_pct']:+.2f}%</b></div>
<div class="card">中位數<b>{stats['median_pct']:+.2f}%</b></div>
<div class="card">最好<b class="pnl-win">{stats['best_pct']:+.2f}%</b></div>
<div class="card">最差<b class="pnl-loss">{stats['worst_pct']:+.2f}%</b></div>
</div>
<div class="equity">{_equity_svg([t.net_pct for t in trades])}</div>
<h2>同資料對照</h2>
<table>
<tr><th>版本</th><th>筆數</th><th>勝率</th><th>淨報酬</th><th>合計 R</th></tr>
{cmp_rows}
</table>
<p class="note">上面每一張圖是主策略，出場可以拿到 {params.max_hold} 分鐘。半小時勝率用同一批進場，30 分鐘內先碰到停損或停利就結算，否則用第 30 根收盤。紅虛線是停損，綠虛線是停利，灰虛線是被刺破的前高。</p>
{window_line}
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有假突破空單</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def dump_hits(path: Path, df: pd.DataFrame, trades: Sequence[Trade], stats: dict, funnel: dict, params: Params) -> None:
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_time": str(df.index[t.entry_idx]),
                "exit_time": str(df.index[t.exit_idx]),
                "entry": t.entry_price,
                "exit": t.exit_price,
                "stop": t.stop_price,
                "target": t.target_price,
                "net_pct": t.net_pct,
                "r": t.r_multiple,
                "reason": t.exit_reason,
                "prev_high": t.prev_high,
                "peak_high": t.peak_high,
                "peak_time": str(df.index[t.peak_idx]),
            }
        )
    path.write_text(
        json.dumps(
            {
                "generated": datetime.now(TPE).isoformat(timespec="seconds"),
                "params": params.__dict__,
                "stats": stats,
                "funnel": funnel,
                "trades": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _print_stats(label: str, trades: Sequence[Trade], funnel: Optional[dict] = None) -> dict:
    stats = summarize(trades)
    extra = f" funnel={funnel}" if funnel else ""
    print(
        f"{label} trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"net={stats['net_pct']:+.2f}% avg={stats['avg_pct']:+.3f}% "
        f"sumR={stats['sum_r']:+.2f}R reasons={stats['reasons']}{extra}"
    )
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SOXSUSDT 1 分 K 假突破做空回測")
    p.add_argument("--symbol", default="SOXSUSDT")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--max-hold", type=int, default=60, help="圖上與主統計的最長持有分鐘")
    p.add_argument("--score-minutes", type=int, default=30, help="另算這個時間內的勝率")
    p.add_argument("--mode", choices=("limit", "breakdown"), default="limit")
    p.add_argument("--session", choices=("tpe_us", "ny", "all"), default="tpe_us")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/soxs-fake-breakout-30d/index.html")
    p.add_argument("--branch", default="cursor/soxs-fake-breakout-backtest-274a")
    args = p.parse_args(argv)

    params = Params(days=args.days, mode=args.mode, session=args.session, max_hold=args.max_hold)
    print(f"fetch {args.symbol} 1m days={args.days + 2}")
    df = fetch_klines(args.symbol, args.days)
    if len(df) < 400:
        print(f"bars too few: {len(df)}", flush=True)
        return 1
    print(f"bars={len(df)} {df.index[0]} -> {df.index[-1]}")

    trades, funnel = run_backtest(df, params)
    stats = _print_stats(f"{args.symbol} {args.mode}/{args.session}", trades, funnel)
    window = score_window(df, trades, args.score_minutes, params.fee_pct)
    print(
        f"within {window['minutes']}m WR={window['win_rate']:.1f}% "
        f"({window['wins']}/{window['count']}) net={window['net_pct']:+.2f}% reasons={window['reasons']}"
    )
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx].tz_convert(TPE)
        xt = df.index[t.exit_idx].tz_convert(TPE)
        print(
            f"[{i:02d}] {et.strftime('%m-%d %H:%M')} -> {xt.strftime('%m-%d %H:%M')} "
            f"in={t.entry_price:.2f} sl={t.stop_price:.2f} out={t.exit_price:.2f} "
            f"{t.exit_reason} {t.net_pct * 100:+.2f}% ({t.r_multiple:+.2f}R) peak={t.peak_high:.2f}"
        )

    comparisons = [(f"主策略 {args.mode} / {args.session}", stats)]
    for mode, session, name in (
        ("limit", "ny", "回測掛單 / 美東 07:00–16:00"),
        ("limit", "all", "回測掛單 / 全時段"),
        ("breakdown", args.session, f"跌破下一根開盤 / {args.session}"),
    ):
        if mode == args.mode and session == args.session:
            continue
        alt, _ = run_backtest(df, Params(days=args.days, mode=mode, session=session, max_hold=args.max_hold))
        info = _print_stats(name, alt)
        comparisons.append((name, info))

    html_path = Path(args.html).resolve() if args.html else None
    if html_path is None and args.pages:
        html_path = PAGES
    if html_path:
        out = write_html(
            html_path, df, trades, funnel, params, args.symbol, comparisons, window_stats=window
        )
        view = write_view_html(out, args.branch)
        dump_hits(out.with_name("hits.json"), df, trades, stats, funnel, params)
        print(f"html={out}")
        print(f"view={view}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
