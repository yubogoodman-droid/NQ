#!/usr/bin/env python3
"""台股 5 分 K：破底反彈後出現 5/10/20 多頭排列就推 Telegram。

對齊力成 6239 那種圖：先急殺破近期低點，再 V 彈，等 5MA > 10MA > 20MA 第一次排好才通知。

用法:
  python3 examples/watch_tw_5m_bounce.py scan --symbols 6239 --range 5d --pages
  python3 examples/watch_tw_5m_bounce.py scan --limit 80 --range 5d --pages
  python3 examples/watch_tw_5m_bounce.py alert --test
  python3 examples/watch_tw_5m_bounce.py alert --dry-run --once
  python3 examples/watch_tw_5m_bounce.py alert

Telegram 憑證放 tg_config.env（勿提交）:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    TPE,
    UA,
    _chart_payload_to_df,
    _get_json,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "tw-5m-bounce" / "index.html"
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO / "output" / "tw_5m_bounce_seen.json"
STATE_PATH = Path(__file__).resolve().parent / "tw_5m_bounce_state.json"

# 5 藍、10 綠、20 橘、60 青、120 紫、200 白、240 粉
MA_PERIODS = (5, 10, 20, 60, 120, 200, 240)
MA_COLORS = {
    5: "#3b82f6",
    10: "#22c55e",
    20: "#f59e0b",
    60: "#14b8a6",
    120: "#a855f7",
    200: "#e5e7eb",
    240: "#f472b6",
}


@dataclass
class BounceSignal:
    break_idx: int
    entry_idx: int
    entry_price: float
    break_low: float
    prior_low: float
    window_high: float
    drop_pct: float
    bounce_pct: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    ma120: float
    ma200: float
    volume_ratio: float
    climax_ratio: float = float("nan")
    bounce_vol_ratio: float = float("nan")
    lid_pct: float = float("nan")


@dataclass
class BounceTrade:
    signal: BounceSignal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_pct: float
    exit_reason: str


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def drop_incomplete_5m(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """丟掉非整點 5 分 K，以及還沒收盤的當根。"""
    if df is None or df.empty:
        return df
    idx = df.index
    aligned = (idx.minute % 5 == 0) & (idx.second == 0)
    out = df.loc[aligned].copy()
    if out.empty:
        return out
    cur = now or datetime.now(TPE)
    last = out.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize(TPE)
    if cur.tzinfo is None:
        cur = cur.replace(tzinfo=TPE)
    if cur < last.tz_convert(cur.tzinfo) + pd.Timedelta(minutes=5):
        out = out.iloc[:-1]
    return out


def fetch_yahoo_5m(symbol: str, range_: str = "5d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=5m&range={range_}&includePrePost=false"
    )
    return drop_incomplete_5m(_chart_payload_to_df(_get_json(url)))


def parse_symbols(text: str) -> list[dict]:
    rows: list[dict] = []
    for i, raw in enumerate(text.split(","), 1):
        token = raw.strip().upper()
        if not token:
            continue
        if token.endswith(".TW") or token.endswith(".TWO"):
            code, market = token.split(".", 1)
            mkt = "tse" if market == "TW" else "otc"
            symbol = token
        else:
            code = token
            mkt = "tse"
            symbol = yahoo_symbol(code, "tse")
        rows.append(
            {
                "rank": i,
                "code": code,
                "name": "",
                "market": mkt,
                "amount": 0,
                "close": None,
                "symbol": symbol,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def sma(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().to_numpy(float)


def _finite(*vals: float) -> bool:
    return all(v is not None and not np.isnan(v) for v in vals)


def _fmt_ma(name: str, val: float) -> str:
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return f"{name} —"
    return f"{name} {float(val):.2f}"


def _fmt_x(name: str, val: float) -> str:
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return f"{name} —"
    return f"{name} {float(val):.1f}x"


def _fmt_lid(val: float) -> str:
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "蓋子 —"
    return f"蓋子 {float(val)*100:.2f}%"


def nearest_overhead_ma_pct(px: float, *ma_vals: float) -> float:
    """進場價上方最近一條均線的距離（佔價百分比）。沒有蓋子回 NaN。"""
    if px is None or px <= 0 or np.isnan(px):
        return float("nan")
    best = float("inf")
    for v in ma_vals:
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            continue
        if float(v) >= px:
            gap = (float(v) - px) / px
            if gap < best:
                best = gap
    return best if best < float("inf") else float("nan")


def lid_ok(lid: float, drop_pct: float, *, min_lid_pct: float, punch_drop_pct: float) -> bool:
    """頭上沒蓋子、或蓋子夠遠、或當日急殺夠深可以穿蓋。min_lid_pct<=0 不檢查。"""
    if min_lid_pct <= 0:
        return True
    if lid is None or (isinstance(lid, (float, np.floating)) and np.isnan(lid)):
        return True
    if lid >= min_lid_pct:
        return True
    return drop_pct >= punch_drop_pct


def trough_clear_of_mas(low_px: float, *ma_vals: float) -> bool:
    """破底那根的下方不能有任何均線。還沒算出來的長均（NaN）不算。"""
    if low_px <= 0:
        return False
    for v in ma_vals:
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            continue
        if float(v) < low_px:
            return False
    return True


def climax_volume_ratio(volume: np.ndarray, trough_idx: int, *, span: int = 3, lookback: int = 20) -> float:
    """急殺段（破底那根含前 span-1 根）最大量 / 再往前 lookback 根均量。富喬那種爆量殺盤會 >> 2。"""
    lo = max(0, trough_idx - span + 1)
    pre = volume[max(0, lo - lookback) : lo]
    pre_avg = float(np.mean(pre)) if len(pre) else 0.0
    if pre_avg <= 0:
        return float("nan")
    return float(np.max(volume[lo : trough_idx + 1])) / pre_avg


def bounce_volume_ratio(volume: np.ndarray, trough_idx: int, entry_idx: int, *, lookback: int = 12) -> float:
    """破底後到進場這段的均量 / 破底前 lookback 根均量。反彈要有量進來，不能縮量乾拉。"""
    pre = volume[max(0, trough_idx - lookback) : trough_idx]
    up = volume[trough_idx + 1 : entry_idx + 1]
    pre_avg = float(np.mean(pre)) if len(pre) else 0.0
    if pre_avg <= 0 or not len(up):
        return float("nan")
    return float(np.mean(up)) / pre_avg


def _ratio_ok(ratio: float, minimum: float) -> bool:
    """門檻 <= 0 代表不檢查；算不出來（NaN）也放行。"""
    if minimum <= 0 or ratio is None or np.isnan(ratio):
        return True
    return ratio >= minimum


def ma_flip_count(fast: np.ndarray, slow: np.ndarray, i: int, lookback: int) -> int:
    """近 lookback 根裡 fast/slow 上下穿越的次數（糾結帶會很高）。"""
    flips = 0
    start = max(1, i - lookback + 1)
    for k in range(start, i + 1):
        if not _finite(fast[k], slow[k], fast[k - 1], slow[k - 1]):
            continue
        if (fast[k] > slow[k]) != (fast[k - 1] > slow[k - 1]):
            flips += 1
    return flips


def stack_pretty(
    ma5: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    close: np.ndarray,
    i: int,
    *,
    slope_bars: int = 3,
    min_gap_5_10: float = 0.0015,
    min_gap_10_20: float = 0.0012,
    min_gap_5_20: float = 0.0030,
    min_ma5_slope: float = 0.0020,
    min_ma10_slope: float = 0.0015,
    min_ma20_slope: float = 0.0005,
    tangle_lookback: int = 8,
    max_ma5_ma10_flips: int = 2,
) -> bool:
    """5/10/20 明顯分開、往上張開。黏在一起或橫盤穿越的糾結帶不算。"""
    if i < max(slope_bars, 1):
        return False
    a5, a10, a20, px = float(ma5[i]), float(ma10[i]), float(ma20[i]), float(close[i])
    if not _finite(a5, a10, a20, px) or px <= 0:
        return False
    if not (a5 > a10 > a20):
        return False
    gap5 = (a5 - a10) / px
    gap10 = (a10 - a20) / px
    gap20 = (a5 - a20) / px
    if gap5 < min_gap_5_10 or gap10 < min_gap_10_20 or gap20 < min_gap_5_20:
        return False
    p5, p10, p20 = float(ma5[i - slope_bars]), float(ma10[i - slope_bars]), float(ma20[i - slope_bars])
    if not _finite(p5, p10, p20):
        return False
    if (a5 - p5) / px < min_ma5_slope:
        return False
    if (a10 - p10) / px < min_ma10_slope:
        return False
    if (a20 - p20) / px < min_ma20_slope:
        return False
    if gap20 <= (p5 - p20) / px:
        return False
    if ma_flip_count(ma5, ma10, i, tangle_lookback) > max_ma5_ma10_flips:
        return False
    if px < a5:
        return False
    return True


def detect_signals(
    df: pd.DataFrame,
    *,
    lookback: int = 48,
    min_drop_pct: float = 0.02,
    rebound_bars: int = 24,
    min_bounce_pct: float = 0.01,
    slope_bars: int = 3,
    min_entry_gap: int = 12,
    vol_lookback: int = 20,
    skip_before: tuple[int, int] = (9, 30),
    require_pretty: bool = True,
    min_climax_vol: float = 2.0,
    min_bounce_vol: float = 1.0,
    require_above_ma60: bool = True,
    min_lid_pct: float = 0.005,
    punch_drop_pct: float = 0.05,
) -> list[BounceSignal]:
    """急殺破近期低點後，等 5>10>20 排漂亮（分開、上彎）才出訊號。

    富喬 1815 08-28 那種標準：破底那段要爆量（min_climax_vol 倍前 20 根均量）、
    反彈段要帶量（min_bounce_vol 倍破底前均量）、進場價要站回 60MA 之上。
    卡片上的當日跌幅也要 ≥ min_drop_pct（跨日前慢跌不算急殺）。
    頭上 0.5% 內有均線蓋子的，除非當日急殺 ≥ punch_drop_pct（預設 5%）否則不算；
    彈到蓋子就結束這次破底，不追著穿。
    門檻設 0 / False 就不檢查。
    """
    if df is None or len(df) < lookback + 20:
        return []
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.zeros(len(df))
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    ma120 = sma(close, 120)
    ma200 = sma(close, 200)
    ma240 = sma(close, 240)
    n = len(close)
    warmup = max(lookback, 20)
    dates = np.array([ts.date() for ts in df.index])
    sess0 = np.zeros(n, dtype=int)
    start = 0
    for i in range(n):
        if i and dates[i] != dates[i - 1]:
            start = i
        sess0[i] = start
    signals: list[BounceSignal] = []
    dump_from: int | None = None
    trough_idx = 0
    trough_low = 0.0
    prior_support = 0.0
    dump_high = 0.0
    last_entry = -(10**9)

    def stacked_at(i: int) -> bool:
        if i < 0 or np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            return False
        return bool(ma5[i] > ma10[i] > ma20[i])

    def mark_dump(i: int, support: float, peak: float) -> None:
        nonlocal dump_from, trough_idx, trough_low, prior_support, dump_high
        px = float(low[i])
        if dump_from is None:
            dump_from = i
            trough_idx = i
            trough_low = px
            prior_support = support
            dump_high = peak
        elif px < trough_low:
            trough_idx = i
            trough_low = px
            dump_high = max(dump_high, peak)

    for i in range(warmup, n):
        if dump_from is not None and i - trough_idx > rebound_bars:
            dump_from = None

        prior_low = float(np.min(low[i - lookback : i]))
        win_high = float(np.max(high[i - lookback : i]))
        if prior_low > 0 and win_high > 0 and low[i] < prior_low:
            drop_pct = (win_high - float(low[i])) / win_high
            if drop_pct >= min_drop_pct:
                mark_dump(i, prior_low, win_high)

        # 今日急殺：即使沒破到昨天低點，盤中高點回檔夠深也算破底
        s0 = int(sess0[i])
        if i - s0 >= 8:
            sess_low = float(np.min(low[s0:i]))
            sess_high = float(np.max(high[s0:i]))
            if sess_high > 0 and low[i] < sess_low:
                sess_drop = (sess_high - float(low[i])) / sess_high
                if sess_drop >= min_drop_pct:
                    mark_dump(i, sess_low, sess_high)

        if dump_from is None or i <= trough_idx:
            continue
        ts = df.index[i]
        if skip_before and (ts.hour, ts.minute) < skip_before:
            continue
        if i - last_entry < min_entry_gap:
            continue
        if require_pretty:
            ready = stack_pretty(ma5, ma10, ma20, close, i, slope_bars=slope_bars)
            was_ready = stack_pretty(ma5, ma10, ma20, close, i - 1, slope_bars=slope_bars)
        else:
            ready = stacked_at(i)
            was_ready = stacked_at(i - 1)
        if not ready or was_ready:
            continue
        if trough_low <= 0:
            continue
        if not trough_clear_of_mas(
            trough_low,
            ma5[trough_idx],
            ma10[trough_idx],
            ma20[trough_idx],
            ma60[trough_idx],
            ma120[trough_idx],
            ma200[trough_idx],
            ma240[trough_idx],
        ):
            continue
        bounce = (float(close[i]) - trough_low) / trough_low
        if bounce < min_bounce_pct:
            continue
        if i >= slope_bars and not np.isnan(ma5[i - slope_bars]) and ma5[i] <= ma5[i - slope_bars]:
            continue
        if require_above_ma60 and not np.isnan(ma60[i]) and float(close[i]) <= float(ma60[i]):
            continue
        climax = climax_volume_ratio(volume, trough_idx, lookback=vol_lookback)
        if not _ratio_ok(climax, min_climax_vol):
            continue
        bounce_vol = bounce_volume_ratio(volume, trough_idx, i)
        if not _ratio_ok(bounce_vol, min_bounce_vol):
            continue
        s0 = int(sess0[trough_idx])
        sess_peak = float(np.max(high[s0 : trough_idx + 1])) if trough_idx >= s0 else dump_high
        peak = sess_peak if sess_peak > trough_low else dump_high
        drop_pct = (peak - trough_low) / peak if peak > 0 else 0.0
        if drop_pct < min_drop_pct:
            continue
        lid = nearest_overhead_ma_pct(float(close[i]), ma60[i], ma120[i], ma200[i], ma240[i])
        if not lid_ok(lid, drop_pct, min_lid_pct=min_lid_pct, punch_drop_pct=punch_drop_pct):
            dump_from = None
            continue
        vol_avg = float(np.mean(volume[max(0, i - vol_lookback) : i]) or 0.0)
        vol_ratio = float(volume[i] / vol_avg) if vol_avg > 0 else 0.0
        dump_high = peak
        signals.append(
            BounceSignal(
                break_idx=trough_idx,
                entry_idx=i,
                entry_price=float(close[i]),
                break_low=trough_low,
                prior_low=prior_support,
                window_high=dump_high,
                drop_pct=drop_pct,
                bounce_pct=bounce,
                ma5=float(ma5[i]),
                ma10=float(ma10[i]),
                ma20=float(ma20[i]),
                ma60=float(ma60[i]),
                ma120=float(ma120[i]),
                ma200=float(ma200[i]),
                volume_ratio=vol_ratio,
                climax_ratio=climax,
                bounce_vol_ratio=bounce_vol,
                lid_pct=lid,
            )
        )
        last_entry = i
        dump_from = None
    return signals


def simulate(df: pd.DataFrame, sigs: Sequence[BounceSignal]) -> list[BounceTrade]:
    if df.empty or not sigs:
        return []
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    idx = df.index
    trades: list[BounceTrade] = []
    for sig in sigs:
        entry = sig.entry_price
        stop = sig.break_low * 0.997
        risk = entry - stop
        if risk <= 0:
            continue
        target = entry + 2.0 * risk
        exit_idx = sig.entry_idx
        exit_px = entry
        reason = "eod"
        for k in range(sig.entry_idx + 1, len(df)):
            if float(low[k]) <= stop:
                exit_idx, exit_px, reason = k, stop, "stop"
                break
            if float(high[k]) >= target:
                exit_idx, exit_px, reason = k, target, "target"
                break
            last_of_day = k == len(df) - 1 or idx[k].date() != idx[k + 1].date()
            if last_of_day:
                exit_idx, exit_px, reason = k, float(close[k]), "eod"
                break
        trades.append(
            BounceTrade(
                signal=sig,
                entry_idx=sig.entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_px,
                stop_price=stop,
                target_price=target,
                pnl_pct=(exit_px - entry) / entry,
                exit_reason=reason,
            )
        )
    return trades


def summarize_trades(trades: Sequence[BounceTrade]) -> dict:
    pnls = [t.pnl_pct for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_pct": float(sum(pnls) * 100.0),
    }


# ---------------------------------------------------------------------------
# Chart / HTML
# ---------------------------------------------------------------------------


def _use_cjk_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break


def draw_signal_png(
    df: pd.DataFrame,
    sig: BounceSignal,
    path: Path,
    title: str,
    trade: BounceTrade | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _use_cjk_font()
    end = trade.exit_idx if trade is not None else sig.entry_idx
    start = max(0, sig.break_idx - 30)
    stop = min(len(df) - 1, end + 12)
    window = df.iloc[start : stop + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#ef4444" if up else "#22c55e"  # 台股：紅漲綠跌
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#ef444499" if up else "#22c55e99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : stop + 1]
        if n <= 20:
            lw = 1.45
        elif n in (60, 120, 200):
            lw = 1.55
        else:
            lw = 1.05
        ax.plot(list(xs), ma, color=col, lw=lw, label=f"{n}MA")

    bx, ex = sig.break_idx - start, sig.entry_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.break_low], s=42, color="#facc15", zorder=6)
        ax.annotate(
            f"破底 {sig.break_low:.1f}",
            (bx, sig.break_low),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            color="#fde68a",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#f87171", ls="--", lw=0.9)
        ax.scatter([ex], [sig.entry_price], s=44, color="#f87171", marker="^", zorder=6)

    ts = df.index[sig.entry_idx]
    ax.set_title(
        f"{title}  {ts.strftime('%m-%d %H:%M')}  "
        f"跌 {sig.drop_pct*100:.1f}% → 彈 {sig.bounce_pct*100:.1f}%  5>10>20",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, labelcolor="#c8d5cc", ncol=7)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_html_report(
    path: Path,
    hits: list[tuple[dict, BounceSignal, BounceTrade | None, pd.DataFrame]],
    universe: list[dict],
    period: str,
) -> Path:
    stats = summarize_trades([h[2] for h in hits if h[2] is not None])
    cards = []
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        et = df.index[sig.entry_idx]
        bt = df.index[sig.break_idx]
        label = f"{row['code']} {row.get('name') or ''}".strip()
        img_name = f"t{i:02d}_{row['code']}_{et.strftime('%m%d_%H%M')}.png"
        draw_signal_png(df, sig, path.parent / "img" / img_name, label, trade=trade)
        pnl = ""
        if trade is not None:
            cls = "pnl-win" if trade.pnl_pct > 0 else ("pnl-flat" if trade.pnl_pct == 0 else "pnl-loss")
            pnl = f"<div class='card-pnl {cls}'>{trade.pnl_pct*100:+.2f}%</div>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))}</span></div>"
            f"{pnl}"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(row['symbol'])}</span>"
            f"<span class='tag'>5分K</span><span class='tag'>5>10>20</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {sig.entry_price:.2f}  破底 {sig.break_low:.2f} @ {bt.strftime('%H:%M')}\n"
            f"跌幅 {sig.drop_pct*100:.1f}%  反彈 {sig.bounce_pct*100:.1f}%\n"
            f"MA5 {sig.ma5:.2f}  MA10 {sig.ma10:.2f}  MA20 {sig.ma20:.2f}"
            f"  間隔 {(sig.ma5-sig.ma20)/sig.entry_price*100:.2f}%\n"
            f"{_fmt_ma('MA60', sig.ma60)}  {_fmt_ma('MA120', sig.ma120)}  {_fmt_ma('MA200', sig.ma200)}\n"
            f"{_fmt_x('破底量', sig.climax_ratio)}  {_fmt_x('反彈量', sig.bounce_vol_ratio)}  {_fmt_lid(sig.lid_pct)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 5分K 破底反彈 · 5/10/20 多頭排列</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#ef4444}} .pnl-loss{{color:#22c55e}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 5分K 破底反彈</h1>
<p class="muted">{escape(period)} · {len(universe)} 檔
<br/>急殺破近 4 小時低點或今日低點（跌幅 ≥ 2%），且破底那根下方不能有任何均線。24 根內 5MA &gt; 10MA &gt; 20MA 要明顯分開、往上張開才算；糾結黏帶不算。
<br/>富喬標準：破底那段要爆量（≥ 2 倍前 20 根均量）、反彈段要帶量（≥ 破底前均量）、進場價站回 60MA 之上。卡片跌幅用當日高點，也要 ≥ 2%。頭上 0.5% 內有均線蓋子的，除非當日急殺 ≥ 5% 否則不算。</p>
<div class="cards">
<div class="card">筆數<b>{len(hits)}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總報酬<b class="{'pnl-win' if stats['total_pct']>=0 else 'pnl-loss'}">{stats['total_pct']:+.2f}%</b></div>
<div class="card">標的<b>{len({h[0]['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有破底後形成 5/10/20 多頭排列</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = "cursor/tw-5m-bounce-alert-c176") -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Telegram
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


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def tg_send(token: str, chat_id: str, text: str, photo: Path | None = None, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    if photo is not None and photo.exists():
        with photo.open("rb") as fh:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                files={"photo": fh},
                timeout=30,
            )
        if r.ok:
            return True
        print(f"[tg] photo HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text[:3900],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        headers={"User-Agent": UA},
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    return bool(r.json().get("ok"))


def load_state() -> dict[str, Any]:
    path = STATE_PATH if STATE_PATH.exists() else SEEN_PATH
    if not path.exists():
        return {"alerted": [], "initialized": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted": [], "initialized": False}
    if isinstance(raw, list):
        return {"alerted": raw, "initialized": True}
    return raw


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(state, ensure_ascii=False, indent=2)
    STATE_PATH.write_text(text, encoding="utf-8")
    SEEN_PATH.write_text(text, encoding="utf-8")


def signal_key(row: dict, df: pd.DataFrame, sig: BounceSignal) -> str:
    ts = df.index[sig.entry_idx]
    return f"{row['symbol']}|{ts.isoformat()}|{sig.entry_price:.2f}"


def fmt_alert(row: dict, df: pd.DataFrame, sig: BounceSignal) -> str:
    et = df.index[sig.entry_idx]
    bt = df.index[sig.break_idx]
    last = float(df["Close"].iloc[-1])
    name = row.get("name") or row["code"]
    return (
        f"🔴 <b>破底反彈 · 5/10/20 多頭排列</b>\n"
        f"{escape(str(name))} <code>{escape(row['code'])}</code>\n"
        f"時間: <code>{et.strftime('%Y-%m-%d %H:%M')} 台北</code>\n"
        f"現價: <code>{sig.entry_price:.2f}</code>（最新 {last:.2f}）\n"
        f"破底: <code>{bt.strftime('%H:%M')}</code> low={sig.break_low:.2f}\n"
        f"跌幅: <b>{sig.drop_pct*100:.1f}%</b> → 反彈 <b>{sig.bounce_pct*100:.1f}%</b>\n"
        f"MA5 {sig.ma5:.2f} &gt; MA10 {sig.ma10:.2f} &gt; MA20 {sig.ma20:.2f}"
        f"（間隔 {(sig.ma5-sig.ma20)/sig.entry_price*100:.2f}%）\n"
        f"{_fmt_ma('MA60', sig.ma60)}  {_fmt_ma('MA120', sig.ma120)}  {_fmt_ma('MA200', sig.ma200)}\n"
        f"{_fmt_x('破底量', sig.climax_ratio)}  {_fmt_x('反彈量', sig.bounce_vol_ratio)}  {_fmt_lid(sig.lid_pct)}\n"
        f"#台股 #五分K #破底反彈 #{row['code']}"
    )


def in_tw_session(now: datetime | None = None, pad_min: int = 8) -> bool:
    cur = now or datetime.now(TPE)
    if cur.weekday() >= 5:
        return False
    minutes = cur.hour * 60 + cur.minute
    return (9 * 60) <= minutes <= (13 * 60 + 30 + pad_min)


def wait_next_5m_close() -> None:
    now = datetime.now(TPE)
    # 5 分 K 收在 :00/:05/... 再多等 12 秒讓 Yahoo 寫入
    elapsed = now.minute % 5
    extra = 12 - now.second
    wait = (5 - elapsed) * 60 + extra
    if wait < 5:
        wait += 5 * 60
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Scan / alert loops
# ---------------------------------------------------------------------------


def merge_universe(base: list[dict], extra: list[dict]) -> list[dict]:
    seen = {r["code"] for r in base}
    out = list(base)
    for row in extra:
        if row["code"] in seen:
            continue
        seen.add(row["code"])
        out.append(row)
    return out


def hit_on_day(df: pd.DataFrame, sig: BounceSignal, day) -> bool:
    return df.index[sig.entry_idx].date() == day


def hit_prices(row: dict, sig: BounceSignal, df: pd.DataFrame) -> list[float]:
    out: list[float] = [float(sig.entry_price), float(sig.break_low)]
    if row.get("close") is not None:
        out.append(float(row["close"]))
    if df is None or not len(df):
        return out
    out.append(float(df["Close"].iloc[-1]))
    ts = df.index[sig.entry_idx]
    same_day = df.index.normalize() == ts.normalize()
    if same_day.any():
        out.append(float(df.loc[same_day, "High"].max()))
    return out


def hit_within_max_price(row: dict, sig: BounceSignal, df: pd.DataFrame, max_price: float | None) -> bool:
    if max_price is None:
        return True
    return all(px <= max_price for px in hit_prices(row, sig, df))


def resolve_on_day(args) -> object | None:
    if getattr(args, "today", False):
        return datetime.now(TPE).date()
    text = getattr(args, "on", "") or ""
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def resolve_universe(args) -> list[dict]:
    extra = parse_symbols(getattr(args, "also", "") or "")
    if getattr(args, "symbols", ""):
        return merge_universe(parse_symbols(args.symbols), extra)
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    pool = max(args.limit, args.pool if args.max_price else args.limit)
    print(f"universe date={date} limit={args.limit} pool={pool} max_price={args.max_price}")
    raw = fetch_top_turnover(date, pool)
    universe, dropped = filter_by_max_price(raw, args.max_price, args.limit)
    if dropped:
        print(
            "drop price>"
            + str(args.max_price)
            + ": "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:10])
            + (" …" if len(dropped) > 10 else "")
        )
    if universe:
        print(
            f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
            f"{universe[0]['amount']/1e8:.1f}億 / {universe[0]['close']}"
        )
    return merge_universe(universe, extra)


def detect_kwargs_from_args(args) -> dict[str, Any]:
    """CLI 旗標 → detect_signals 的關鍵字參數。"""
    return {
        "require_pretty": not getattr(args, "loose", False),
        "min_climax_vol": float(getattr(args, "min_climax_vol", 2.0)),
        "min_bounce_vol": float(getattr(args, "min_bounce_vol", 1.0)),
        "require_above_ma60": not getattr(args, "no_ma60", False),
        "min_lid_pct": float(getattr(args, "min_lid_pct", 0.5)) / 100.0,
        "punch_drop_pct": float(getattr(args, "punch_drop", 5.0)) / 100.0,
    }


def scan_symbol(
    row: dict,
    range_: str,
    *,
    require_pretty: bool = True,
    **detect_kwargs: Any,
) -> tuple[list[tuple[BounceSignal, pd.DataFrame]], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0}
    try:
        df = fetch_yahoo_5m(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 70:
        meta["error"] = "too_few_bars"
        return [], meta
    if row.get("close") is None and len(df):
        row["close"] = float(df["Close"].iloc[-1])
    sigs = detect_signals(df, require_pretty=require_pretty, **detect_kwargs)
    meta["n_sig"] = len(sigs)
    return [(s, df) for s in sigs], meta


def cmd_scan(args) -> int:
    universe = resolve_universe(args)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    hits: list[tuple[dict, BounceSignal, BounceTrade | None, pd.DataFrame]] = []
    errors = 0
    on_day = resolve_on_day(args)
    if on_day is not None:
        print(f"filter day={on_day}")
    detect_kw = detect_kwargs_from_args(args)
    pretty = detect_kw["require_pretty"]
    for i, row in enumerate(universe, 1):
        pairs, meta = scan_symbol(row, args.range_, **detect_kw)
        if meta["error"]:
            errors += 1
        trades_by_entry = {}
        if pairs:
            df0 = pairs[0][1]
            for t in simulate(df0, [s for s, _ in pairs]):
                trades_by_entry[t.entry_idx] = t
        for sig, df in pairs:
            if on_day is not None and not hit_on_day(df, sig, on_day):
                continue
            if not hit_within_max_price(row, sig, df, getattr(args, "max_price", None)):
                continue
            hits.append((row, sig, trades_by_entry.get(sig.entry_idx), df))
        flag = f" sigs={meta['n_sig']}" if meta["n_sig"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row.get('name','')} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h[3].index[h[1].entry_idx])
    stats = summarize_trades([h[2] for h in hits if h[2] is not None])
    print(
        f"done errors={errors} signals={len(hits)} "
        f"WR={stats['win_rate']:.1f}% pnl={stats['total_pct']:+.2f}%"
    )
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        ts = df.index[sig.entry_idx]
        extra = f" {trade.exit_reason} {trade.pnl_pct*100:+.2f}%" if trade else ""
        print(
            f"  [{i}] {row['code']} {row.get('name','')} {ts.strftime('%m-%d %H:%M')} "
            f"drop {sig.drop_pct*100:.1f}% bounce {sig.bounce_pct*100:.1f}%{extra}"
        )

    html_path = Path(args.html) if args.html else (PAGES if args.pages else None)
    if html_path:
        period = args.range_
        on_day = resolve_on_day(args)
        if on_day is not None:
            period = f"{on_day.isoformat()} · {args.range_}資料"
        if args.max_price is not None:
            period += f" · 股價≤{args.max_price:g}"
        if pretty:
            period += " · 均線不糾結"
        if detect_kw["min_climax_vol"] > 0 or detect_kw["min_bounce_vol"] > 0 or detect_kw["require_above_ma60"]:
            period += " · 富喬標準"
        out = write_html_report(html_path, hits, universe, period)
        write_view_html(out)
        print(f"html={out}")
    return 0


def scan_once(
    universe: list[dict],
    token: str,
    chat_id: str,
    *,
    range_: str,
    dry_run: bool,
    seed_alert: bool,
    sleep_s: float,
    require_pretty: bool = True,
    **detect_kwargs: Any,
) -> None:
    state = load_state()
    alerted = set(state.get("alerted") or [])
    first_run = not state.get("initialized")
    new_items: list[tuple[str, dict, BounceSignal, pd.DataFrame]] = []
    for row in universe:
        pairs, meta = scan_symbol(row, range_, require_pretty=require_pretty, **detect_kwargs)
        if meta["error"]:
            print(f"  skip {row['symbol']} {meta['error']}", file=sys.stderr)
        for sig, df in pairs:
            key = signal_key(row, df, sig)
            if key in alerted:
                continue
            new_items.append((key, row, sig, df))
        time.sleep(max(0.05, sleep_s))

    now = datetime.now(TPE)
    if first_run and not seed_alert:
        for key, _, _, _ in new_items:
            alerted.add(key)
        state["alerted"] = sorted(alerted)[-400:]
        state["initialized"] = True
        state["last_scan"] = now.isoformat()
        save_state(state)
        print(f"[{now.strftime('%H:%M:%S')}] init: marked {len(new_items)} recent signals")
        return

    sent = 0
    for key, row, sig, df in new_items:
        tmp = Path("/tmp") / f"tw5m_{row['code']}_{sig.entry_idx}.png"
        try:
            draw_signal_png(df, sig, tmp, f"{row['code']} {row.get('name') or ''}")
        except Exception as exc:  # noqa: BLE001
            print(f"[chart] {exc}", file=sys.stderr)
            tmp = None
        ok = tg_send(token, chat_id, fmt_alert(row, df, sig), photo=tmp, dry_run=dry_run)
        if ok:
            alerted.add(key)
            sent += 1
            ts = df.index[sig.entry_idx]
            print(f"[alert] {row['code']} {ts} @ {sig.entry_price:.2f}")
    state["alerted"] = sorted(alerted)[-400:]
    state["initialized"] = True
    state["last_scan"] = now.isoformat()
    save_state(state)
    print(f"[{now.strftime('%H:%M:%S')}] scan ok new_sent={sent} pending={len(new_items)}")


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN") or ""
    chat_id = env("TELEGRAM_CHAT_ID") or ""
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2
    if args.test:
        ok = tg_send(
            token,
            chat_id,
            f"✅ 台股 5分K 破底反彈 bot 測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    universe = resolve_universe(args)
    if not universe:
        return 1
    detect_kw = detect_kwargs_from_args(args)
    print(
        f"TW 5m bounce TG | n={len(universe)} | dry_run={args.dry_run} | "
        f"range={args.range_} | pretty={detect_kw['require_pretty']} | "
        f"climax>={detect_kw['min_climax_vol']:g} bounce_vol>={detect_kw['min_bounce_vol']:g} "
        f"ma60={detect_kw['require_above_ma60']} lid>={detect_kw['min_lid_pct']*100:g}% "
        f"punch={detect_kw['punch_drop_pct']*100:g}% | session_only={not args.all_hours}"
    )
    while True:
        try:
            if args.all_hours or in_tw_session():
                scan_once(
                    universe,
                    token,
                    chat_id,
                    range_=args.range_,
                    dry_run=args.dry_run,
                    seed_alert=args.seed_alert,
                    sleep_s=args.sleep,
                    **detect_kw,
                )
            else:
                print(f"[{datetime.now(TPE).strftime('%H:%M:%S')}] outside session, skip")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        wait_next_5m_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股 5分K 破底反彈 → 5/10/20 多頭排列通知")
    sub = p.add_subparsers(dest="cmd")

    def add_universe(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--date", default="", help="成交額基準日 YYYYMMDD")
        sp.add_argument("--limit", type=int, default=80)
        sp.add_argument("--pool", type=int, default=160)
        sp.add_argument("--max-price", type=float, default=None)
        sp.add_argument("--symbols", default="", help="逗號分隔代號，例如 6239,2330")
        sp.add_argument("--also", default="", help="額外併入掃描的代號，例如 6239")
        sp.add_argument("--range", dest="range_", default="5d")
        sp.add_argument("--sleep", type=float, default=0.2)
        sp.add_argument(
            "--loose",
            action="store_true",
            help="不擋均線糾結，第一次 5>10>20 就算",
        )
        sp.add_argument(
            "--min-climax-vol",
            type=float,
            default=2.0,
            help="破底那段最大量至少要是前 20 根均量的幾倍（0 = 不檢查）",
        )
        sp.add_argument(
            "--min-bounce-vol",
            type=float,
            default=1.0,
            help="破底後到進場的均量至少要是破底前均量的幾倍（0 = 不檢查）",
        )
        sp.add_argument("--no-ma60", action="store_true", help="不要求進場價站上 60MA")
        sp.add_argument(
            "--min-lid-pct",
            type=float,
            default=0.5,
            help="進場價上方最近均線至少要空出多少 %（0 = 不檢查蓋子）",
        )
        sp.add_argument(
            "--punch-drop",
            type=float,
            default=5.0,
            help="當日急殺達此 % 時允許穿蓋（頭上均線可以貼著）",
        )

    s = sub.add_parser("scan", help="回看近幾日並可出 HTML")
    add_universe(s)
    s.add_argument("--today", action="store_true", help="只留台北今天的訊號")
    s.add_argument("--on", default="", help="只留這一天 YYYY-MM-DD")
    s.add_argument("--pages", action="store_true")
    s.add_argument("--html", default="")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("alert", help="Telegram 輪詢（每根 5 分 K 收盤掃一次）")
    add_universe(a)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--seed-alert", action="store_true", help="第一次也把近期訊號推出去")
    a.add_argument("--all-hours", action="store_true")
    a.set_defaults(func=cmd_alert)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
