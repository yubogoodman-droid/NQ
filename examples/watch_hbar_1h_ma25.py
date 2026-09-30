#!/usr/bin/env python3
"""HBAR 1 小時 K：假突破後收盤跌破 MA25，推 Telegram。

對齊 2026-09-29 14:00（台北）那波：先拉到 0.13099（相對當時 MA25 伸 ~23%），
12 根後收盤從 MA25 上方跌到下方（0.11766 < 0.11837）。

條件（剛收盤的 1h K，對齊 HBAR 箱體假突破）：
  1. 上一根收盤 ≥ MA25，這一根收盤 < MA25（第一次跌破），且是收跌陰線
  2. 往回 48 根內有假突破高點：創先前 24 根新高、當根收盤仍在 MA25 上
  3. 高點相對當根 MA25 伸 15%～40%（HBAR 約 23%；不要 8% 輕觸，也不要 100% 妖幣）
  4. 從高點回落 8%～18%（HBAR 約 10%；剛失守，不是已經崩完）
  5. 隔 6～18 根才跌破（HBAR 12 根；要有一段派發）
  6. 相對拉升前箱體（高點往前 48 根、扣掉最後 10 根尖峰）再高出 ≥ 18%
  7. 收盤離 MA25 不超過 2.5%（第一根跌破，不要追低）
  8. MA25 仍在上（近 6 根至少 +1%），尖峰附近量能 ≥ 均量 1.8 倍

用法:
  python3 examples/watch_hbar_1h_ma25.py --test
  python3 examples/watch_hbar_1h_ma25.py --once --dry-run
  python3 examples/watch_hbar_1h_ma25.py
  python3 examples/watch_hbar_1h_ma25.py --backtest --days 60 --pages
  python3 examples/watch_hbar_1h_ma25.py --backtest --universe --days 7 --pages
  python3 examples/test_watch_hbar_1h_ma25.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

# —— 也可直接填這裡 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
INTERVAL = "1h"
INTERVAL_MS = 3_600_000
MA_PERIODS = (7, 14, 25, 99, 120, 200)
DEFAULT_SYMBOL = "HBARUSDT"
MIN_QUOTE_VOL = 5_000_000
KEEP = {DEFAULT_SYMBOL}
MAX_CHARTS = 36

# 對齊 HBAR 09-29 14:00 箱體假突破（伸 ~23%、回落 ~10%、隔 12 根）
LOOKBACK = 48
PRIOR_RANGE = 24
MIN_EXT = 0.15
MAX_EXT = 0.40
MIN_FAIL = 0.08
MAX_FAIL = 0.18
MIN_BARS_AFTER = 6
MAX_BARS_AFTER = 18
MIN_RANGE_BREAK = 0.18
RANGE_BASE_SKIP = 10
MAX_CLOSE_BELOW = 0.025
MIN_MA25_SLOPE = 0.01
MA25_SLOPE_BARS = 6
MIN_VOL_RATIO = 1.8
VOL_LOOKBACK = 12
TARGET_R = 2.0
TIME_BARS = 24

REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "hbar_1h_ma25_seen.json"
PAGES = REPO / "docs" / "hbar-1h-ma25" / "index.html"
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})

MA_COLORS = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#42a5f5", 120: "#26c6da", 200: "#ffffff"}


@dataclass(frozen=True)
class Params:
    lookback: int = LOOKBACK
    prior_range: int = PRIOR_RANGE
    min_ext: float = MIN_EXT
    max_ext: float = MAX_EXT
    min_fail: float = MIN_FAIL
    max_fail: float = MAX_FAIL
    min_bars_after: int = MIN_BARS_AFTER
    max_bars_after: int = MAX_BARS_AFTER
    min_range_break: float = MIN_RANGE_BREAK
    range_base_skip: int = RANGE_BASE_SKIP
    max_close_below: float = MAX_CLOSE_BELOW
    min_ma25_slope: float = MIN_MA25_SLOPE
    ma25_slope_bars: int = MA25_SLOPE_BARS
    require_red: bool = True
    min_vol_ratio: float = MIN_VOL_RATIO
    vol_lookback: int = VOL_LOOKBACK
    target_r: float = TARGET_R
    time_bars: int = TIME_BARS


@dataclass
class Signal:
    i: int
    peak_i: int
    close: float
    ma25: float
    peak_high: float
    peak_ma25: float
    ext: float
    fail: float
    bars_after: int
    range_break: float = 0.0
    close_below: float = 0.0
    vol_ratio: float = 0.0


@dataclass
class Trade:
    symbol: str
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry: float
    exit: float
    stop: float
    target: float
    pnl_pct: float
    reason: str
    d: dict
    fwd_4h: Optional[float] = None
    fwd_8h: Optional[float] = None
    fwd_24h: Optional[float] = None


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
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))
    raise last


def drop_unclosed(raw: list, now_ms: int | None = None) -> list:
    if not raw:
        return raw
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if int(raw[-1][0]) + INTERVAL_MS > now_ms:
        return raw[:-1]
    return raw


def bars_from_raw(raw: list) -> dict | None:
    if not raw or len(raw) < MA_PERIODS[-1] + 2:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def fetch_klines(sym: str, limit: int = 1500) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": INTERVAL, "limit": limit})
    return bars_from_raw(drop_unclosed(raw))


def filter_universe(
    info: dict,
    tickers: list,
    min_quote_vol: float = MIN_QUOTE_VOL,
    keep: set[str] | None = None,
) -> list[str]:
    keep = keep or set(KEEP)
    tmap = {t["symbol"]: t for t in tickers}
    out = []
    for s in info.get("symbols") or []:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tmap.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_quote_vol and sym not in keep:
            continue
        out.append(sym)
    return out


def universe(min_quote_vol: float = MIN_QUOTE_VOL) -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = get_json("/fapi/v1/ticker/24hr")
    return filter_universe(info, tickers, min_quote_vol=min_quote_vol)


def skip_overlap(trades: list[Trade]) -> list[Trade]:
    """同一標的持倉重疊時，只留先進場那筆。"""
    kept: list[Trade] = []
    last_exit: dict[str, int] = {}
    for t in sorted(trades, key=lambda x: (x.symbol, x.entry_idx, x.exit_idx)):
        prev = last_exit.get(t.symbol)
        if prev is not None and t.entry_idx <= prev:
            continue
        kept.append(t)
        last_exit[t.symbol] = t.exit_idx
    return sorted(kept, key=lambda t: (int(t.d["t"][t.entry_idx]), t.symbol))


def pick_chart_indices(
    trades: list[Trade],
    max_charts: int = MAX_CHARTS,
    pin: tuple[str, ...] = (DEFAULT_SYMBOL,),
) -> set[int]:
    n = len(trades)
    if n <= max_charts:
        return set(range(n))
    chosen: list[int] = []
    for i, t in enumerate(trades):
        if t.symbol in pin:
            chosen.append(i)
    ranked = sorted(range(n), key=lambda i: trades[i].pnl_pct, reverse=True)
    half = max(1, (max_charts - len(chosen)) // 2)
    for i in ranked:
        if i in chosen:
            continue
        chosen.append(i)
        if sum(1 for j in chosen if trades[j].symbol not in pin) >= half:
            break
    for i in reversed(ranked):
        if i in chosen:
            continue
        chosen.append(i)
        if len(chosen) >= max_charts:
            break
    return set(chosen[:max_charts])


def safe_name(sym: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sym)
    return out or "SYM"


def pages_path(*, universe: bool, days: int, symbol: str) -> Path:
    if universe:
        return REPO / "docs" / f"binance-1h-fake-ma25-{days}d" / "index.html"
    if symbol == DEFAULT_SYMBOL:
        return PAGES
    return REPO / "docs" / f"{symbol.lower()}-1h-ma25" / "index.html"


def indicators(d: dict) -> dict:
    out = dict(d)
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(d["c"], n)
    out["v20"] = sma(d["v"], 20)
    return out


def crossed_below_ma25(d: dict, i: int) -> bool:
    if i < 1 or i >= len(d["c"]):
        return False
    m = d["m25"]
    if np.isnan(m[i]) or np.isnan(m[i - 1]):
        return False
    return float(d["c"][i - 1]) >= float(m[i - 1]) and float(d["c"][i]) < float(m[i])


def find_fake_peak(d: dict, i: int, p: Params) -> int | None:
    """跌破這根之前，往回找符合箱體假突破的最高點。"""
    lo = max(p.prior_range, i - p.lookback)
    if i - lo < p.min_bars_after:
        return None
    hi = i - p.min_bars_after
    if hi < lo:
        return None
    peak = int(lo + np.argmax(d["h"][lo : hi + 1]))
    bars_after = i - peak
    if bars_after < p.min_bars_after or bars_after > p.max_bars_after:
        return None
    m_peak = float(d["m25"][peak])
    if np.isnan(m_peak) or m_peak <= 0:
        return None
    peak_high = float(d["h"][peak])
    if float(d["c"][peak]) < m_peak:
        return None
    prior_from = max(0, peak - p.prior_range)
    if peak <= prior_from:
        return None
    prior_high = float(np.max(d["h"][prior_from:peak]))
    if peak_high <= prior_high:
        return None
    ext = peak_high / m_peak - 1.0
    if ext < p.min_ext or ext > p.max_ext:
        return None
    fail = 1.0 - float(d["c"][i]) / peak_high
    if fail < p.min_fail or fail > p.max_fail:
        return None
    return peak


def _range_break(d: dict, peak: int, p: Params) -> float | None:
    base_to = peak - p.range_base_skip
    base_from = max(0, peak - p.lookback)
    if base_to <= base_from:
        return None
    base_high = float(np.max(d["h"][base_from:base_to]))
    if base_high <= 0:
        return None
    return float(d["h"][peak]) / base_high - 1.0


def _vol_ratio(d: dict, peak: int, p: Params) -> float | None:
    v20 = float(d["v20"][peak]) if "v20" in d else float("nan")
    if np.isnan(v20) or v20 <= 0:
        return None
    lo = max(0, peak - p.vol_lookback)
    return float(np.max(d["v"][lo : peak + 1])) / v20


def detect_at(d: dict, i: int, p: Params | None = None) -> Signal | None:
    p = p or Params()
    if not crossed_below_ma25(d, i):
        return None
    close = float(d["c"][i])
    ma25 = float(d["m25"][i])
    if ma25 <= 0 or np.isnan(ma25):
        return None
    if p.require_red and close >= float(d["o"][i]):
        return None
    close_below = 1.0 - close / ma25
    if close_below > p.max_close_below:
        return None
    slope_i = i - p.ma25_slope_bars
    if slope_i < 0 or np.isnan(d["m25"][slope_i]) or float(d["m25"][slope_i]) <= 0:
        return None
    if ma25 / float(d["m25"][slope_i]) - 1.0 < p.min_ma25_slope:
        return None
    peak = find_fake_peak(d, i, p)
    if peak is None:
        return None
    rb = _range_break(d, peak, p)
    if rb is None or rb < p.min_range_break:
        return None
    vr = _vol_ratio(d, peak, p)
    if vr is None or vr < p.min_vol_ratio:
        return None
    peak_high = float(d["h"][peak])
    peak_ma25 = float(d["m25"][peak])
    return Signal(
        i=i,
        peak_i=peak,
        close=close,
        ma25=ma25,
        peak_high=peak_high,
        peak_ma25=peak_ma25,
        ext=peak_high / peak_ma25 - 1.0,
        fail=1.0 - close / peak_high,
        bars_after=i - peak,
        range_break=rb,
        close_below=close_below,
        vol_ratio=vr,
    )


def detect_signals(d: dict, start: int = 0, end: int | None = None, p: Params | None = None) -> list[Signal]:
    p = p or Params()
    end = len(d["c"]) - 1 if end is None else end
    lo = max(MA_PERIODS[2] + p.prior_range + 1, start)
    hits: list[Signal] = []
    for i in range(lo, min(end, len(d["c"]) - 1) + 1):
        sig = detect_at(d, i, p)
        if sig:
            hits.append(sig)
    return hits


def short_fwd(d: dict, i: int, bars: int) -> Optional[float]:
    j = i + bars
    if j >= len(d["c"]):
        return None
    entry = float(d["c"][i])
    if entry <= 0:
        return None
    return (entry - float(d["c"][j])) / entry


def simulate(d: dict, sigs: list[Signal], p: Params | None = None, symbol: str = DEFAULT_SYMBOL) -> list[Trade]:
    p = p or Params()
    trades: list[Trade] = []
    for sig in sigs:
        i = sig.i
        entry = float(d["c"][i])
        stop = float(sig.peak_high)
        risk = stop - entry
        if risk <= 0 or entry <= 0:
            continue
        target = entry - p.target_r * risk
        reason = "open"
        exit_idx = len(d["c"]) - 1
        exit_px = float(d["c"][exit_idx])
        last = min(len(d["c"]) - 1, i + p.time_bars)
        for j in range(i + 1, last + 1):
            if float(d["h"][j]) >= stop:
                reason = "stop"
                exit_idx = j
                exit_px = stop
                break
            if float(d["l"][j]) <= target:
                reason = "target"
                exit_idx = j
                exit_px = target
                break
        else:
            if last > i and last < len(d["c"]):
                if last == i + p.time_bars or last == len(d["c"]) - 1:
                    reason = "time" if last == i + p.time_bars else "open"
                    exit_idx = last
                    exit_px = float(d["c"][last])
        pnl = (entry - exit_px) / entry
        trades.append(
            Trade(
                symbol=symbol,
                signal=sig,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit=exit_px,
                stop=stop,
                target=target,
                pnl_pct=pnl,
                reason=reason,
                d=d,
                fwd_4h=short_fwd(d, i, 4),
                fwd_8h=short_fwd(d, i, 8),
                fwd_24h=short_fwd(d, i, 24),
            )
        )
    return trades


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.reason != "open"]
    wins = [t for t in closed if t.pnl_pct > 0]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    avg = float(np.mean([t.pnl_pct for t in closed])) if closed else 0.0
    total = float(np.sum([t.pnl_pct for t in closed])) if closed else 0.0
    wr = 100.0 * len(wins) / len(closed) if closed else 0.0

    def avg_fwd(attr: str) -> Optional[float]:
        vals = [getattr(t, attr) for t in trades if getattr(t, attr) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "count": len(trades),
        "closed": len(closed),
        "wins": len(wins),
        "closed_win_rate": wr,
        "avg_pct": avg,
        "total_pct": total,
        "symbols": len({t.symbol for t in trades}),
        "reasons": reasons,
        "fwd_4h": avg_fwd("fwd_4h"),
        "fwd_8h": avg_fwd("fwd_8h"),
        "fwd_24h": avg_fwd("fwd_24h"),
    }


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)))


def telegram_send(text: str, photo: str | None = None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
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


def _draw_candles(ax, axv, o, h, l, c, v, xs) -> None:
    colors_v = []
    from matplotlib.patches import Rectangle

    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, v, width=0.8, color=colors_v, linewidth=0)


def draw_chart(sym: str, d: dict, sig: Signal, path: str, trade: Trade | None = None) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        return None
    i = sig.i
    a0 = max(0, min(sig.peak_i, i) - 36)
    a1 = min(len(d["c"]), max(i, trade.exit_idx if trade else i) + 6)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#0c1210"
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    _draw_candles(ax, axv, o, h, l, c, v, xs)
    for n, col in MA_COLORS.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.05, label=f"MA{n}")
    px = sig.peak_i - a0
    sx = i - a0
    if 0 <= px < len(c):
        ax.axvline(px, color="#c9a227", ls="--", lw=0.9)
        ax.scatter([px], [sig.peak_high], s=42, color="#c9a227", marker="o", zorder=6, label="假突破")
    if 0 <= sx < len(c):
        ax.axvline(sx, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([sx], [sig.close], s=42, color="#e35d5d", marker="v", zorder=6, label="跌破MA25")
        axv.axvline(sx, color="#e35d5d", ls="--", lw=0.9)
    if trade is not None:
        ax.axhline(trade.stop, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
        ax.axhline(trade.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
        xx = trade.exit_idx - a0
        if 0 <= xx < len(c):
            ax.scatter(
                [xx],
                [trade.exit],
                s=40,
                color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
                marker="x",
                zorder=6,
            )
    title = f"{sym}  1h  假突破跌破 MA25"
    if trade is not None:
        title += f"  {trade.reason}  {trade.pnl_pct * 100:+.2f}%"
    ax.set_title(title, color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_alert(sym: str, d: dict, sig: Signal) -> str:
    ts = hm(int(d["t"][sig.i]))
    pk = hm(int(d["t"][sig.peak_i]))
    o, h, l, c = float(d["o"][sig.i]), float(d["h"][sig.i]), float(d["l"][sig.i]), sig.close
    return (
        f"🔻 <b>1h 箱體假突破跌破 MA25</b>\n"
        f"<b>{sym}</b>  收盤 {ts}（台北）\n"
        f"現價 {c:g}　OHLC {o:g} / {h:g} / {l:g} / {c:g}\n"
        f"MA25 {sig.ma25:g}　剛跌破 {sig.close_below * 100:.2f}%　陰線\n"
        f"假突破高 {pk}  {sig.peak_high:g}　伸 {sig.ext * 100:.1f}%　箱體假突破 {sig.range_break * 100:.1f}%\n"
        f"從高點回落 {sig.fail * 100:.1f}%　隔 {sig.bars_after} 根　量 {sig.vol_ratio:.1f}×\n"
        f"空：停在假突破高、目標 {TARGET_R:g}R（對照用）"
    )


def key_of(sym: str, d: dict, sig: Signal) -> str:
    return f"{sym}:{int(d['t'][sig.i])}"


def scan_symbol(sym: str, lookback: int, p: Params) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    last = len(d["c"]) - 1
    start = max(0, last - lookback + 1)
    events = []
    for sig in detect_signals(d, start=start, end=last, p=p):
        events.append({"symbol": sym, "d": d, "sig": sig})
    return events


def notify(ev: dict, dry_run: bool = False) -> None:
    sig: Signal = ev["sig"]
    text = format_alert(ev["symbol"], ev["d"], sig)
    plain = text.replace("<b>", "").replace("</b>", "").replace("🔻 ", "")
    print("\n" + plain, flush=True)
    if dry_run:
        print("  → dry-run，不送 Telegram", flush=True)
        return
    tmp = Path("/tmp") / f"hbar1h_{ev['symbol']}_{sig.i}.png"
    photo = draw_chart(ev["symbol"], ev["d"], sig, str(tmp))
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送", flush=True)
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗，檢查 token 與 chat id", flush=True)


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 3600 + 1) * 3600 + 3
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("HBAR 1h 假突破跌破 MA25 監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "main"


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _fmt_fwd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


def write_html(
    path: Path,
    trades: list[Trade],
    stats: dict,
    extra: dict,
    max_charts: int = MAX_CHARTS,
) -> Path:
    cards = []
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    chart_ix = pick_chart_indices(trades, max_charts=max_charts)
    pad = 3 if len(trades) >= 100 else 2
    for i, t in enumerate(trades, 1):
        et = hm(int(t.d["t"][t.entry_idx]))
        xt = hm(int(t.d["t"][t.exit_idx]))
        pk = hm(int(t.d["t"][t.signal.peak_i]))
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        img_html = ""
        if (i - 1) in chart_ix:
            img_name = f"t{i:0{pad}d}_{safe_name(t.symbol)}_{et.replace(' ', '_').replace(':', '')}.png"
            draw_chart(t.symbol, t.d, t.signal, str(img_dir / img_name), trade=t)
            img_html = (
                f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(t.symbol)}' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(t.symbol)}</span>"
            f"<span class='trade-time'>{escape(et)} → {escape(xt)}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct * 100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(t.reason)}</span>"
            f"<span class='tag'>假突破 {escape(pk)}</span>"
            f"<span class='tag'>伸 {t.signal.ext * 100:.1f}%</span>"
            f"<span class='tag'>箱破 {t.signal.range_break * 100:.1f}%</span>"
            f"<span class='tag'>回落 {t.signal.fail * 100:.1f}%</span>"
            f"<span class='tag'>{t.signal.bars_after} 根</span>"
            f"<span class='tag'>量 {t.signal.vol_ratio:.1f}×</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry:g}  stop {t.stop:g}  target {t.target:g}\n"
            f"exit {t.exit:g} {t.reason}  {t.pnl_pct * 100:+.2f}%\n"
            f"fwd +4h {_fmt_fwd(t.fwd_4h)}  +8h {_fmt_fwd(t.fwd_8h)}  +24h {_fmt_fwd(t.fwd_24h)}"
            "</pre>"
            f"{img_html}"
            "</article>"
        )
    reasons = stats.get("reasons") or {}
    avg = stats["avg_pct"]
    uni = bool(extra.get("universe"))
    title_scope = "幣安流動永續" if uni else extra.get("symbol", DEFAULT_SYMBOL)
    scanned_line = ""
    if extra.get("scanned"):
        scanned_line = f"掃 {extra['scanned']} 檔 · 原始訊號 {extra.get('raw_hits', stats['count'])} · "
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(str(title_scope))} 1h 假突破跌破 MA25 · {extra['days']} 天</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.tag-info{{color:#f0c14b;border-color:#3d3418}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(str(title_scope))} 1h 假突破跌破 MA25 · 近 {extra['days']} 天</h1>
<p class="muted">對齊 2026-09-29 HBAR 箱體假突破：高點離 MA25 {extra['min_ext']*100:g}～{extra.get('max_ext', 0.4)*100:g}%，
從箱體再高出 ≥ {extra.get('min_range_break', 0.18)*100:g}%，回落 {extra['min_fail']*100:g}～{extra.get('max_fail', 0.18)*100:g}% 後，
隔 6～18 根、陰線剛跌破 MA25（離均線 ≤ 2.5%）才算。MA25 仍上、尖峰放量。
<br/>收盤空；停在假突破高、目標 {extra['target_r']:g}R、或 {extra['time_bars']} 根時間停。同標的重疊不重做。加總％是各筆相加。
<br/>{scanned_line}進場 {stats['count']} · 標的 {stats.get('symbols', extra.get('symbol', ''))}
· 出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)}
· 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}
· 收盤後 +4h {_fmt_fwd(stats.get('fwd_4h'))} · +8h {_fmt_fwd(stats.get('fwd_8h'))}
· +24h {_fmt_fwd(stats.get('fwd_24h'))}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">已平勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{'pnl-win' if avg >= 0 else 'pnl-loss'}">{avg * 100:+.2f}%</b></div>
<div class="card">加總<b class="{'pnl-win' if stats['total_pct'] >= 0 else 'pnl-loss'}">{stats['total_pct'] * 100:+.2f}%</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有假突破跌破 MA25</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def dump_hits(path: Path, trades: list[Trade], stats: dict, extra: dict) -> Path:
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_time": hm(int(t.d["t"][t.entry_idx])),
                "exit_time": hm(int(t.d["t"][t.exit_idx])),
                "peak_time": hm(int(t.d["t"][t.signal.peak_i])),
                "entry": t.entry,
                "exit": t.exit,
                "stop": t.stop,
                "target": t.target,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
                "ext": t.signal.ext,
                "fail": t.signal.fail,
                "bars_after": t.signal.bars_after,
                "range_break": t.signal.range_break,
                "close_below": t.signal.close_below,
                "vol_ratio": t.signal.vol_ratio,
                "fwd_4h": t.fwd_4h,
                "fwd_8h": t.fwd_8h,
                "fwd_24h": t.fwd_24h,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "extra": extra, "hits": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def backtest_one(sym: str, days: int, p: Params, limit: int) -> tuple[str, list[Trade], int, str]:
    raw = fetch_klines(sym, limit=limit)
    if raw is None:
        return sym, [], 0, "no data"
    d = indicators(raw)
    cutoff = int(d["t"][-1]) - days * 24 * INTERVAL_MS
    start = next((i for i, ts in enumerate(d["t"]) if int(ts) >= cutoff), 0)
    sigs = detect_signals(d, start=start, p=p)
    trades = simulate(d, sigs, p=p, symbol=sym)
    return sym, trades, len(sigs), hm(int(d["t"][start])) + " → " + hm(int(d["t"][-1]))


def cmd_backtest(args) -> int:
    p = Params(
        min_ext=args.min_ext,
        max_ext=args.max_ext,
        min_fail=args.min_fail,
        max_fail=args.max_fail,
        min_range_break=args.min_range_break,
        target_r=args.target_r,
        time_bars=args.time_bars,
    )
    limit = min(1500, max(260, args.days * 24 + 220))
    if args.universe:
        print("載入標的…", flush=True)
        symbols = universe(min_quote_vol=args.min_quote_vol)
    else:
        symbols = [s.upper() for s in (args.symbols or [args.symbol])]
    print(f"回測 {len(symbols)} 檔 1h × {limit}，近 {args.days} 天…", flush=True)
    trades: list[Trade] = []
    raw_hits = 0
    span = ""
    t0 = time.time()
    workers = 1 if len(symbols) == 1 else 8
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(backtest_one, s, args.days, p, limit): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, ts, nsig, sp = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            trades.extend(ts)
            raw_hits += nsig
            if sp:
                span = sp
            if done % 40 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)}  訊號 {raw_hits}  {time.time()-t0:.0f}s", flush=True)
    before = len(trades)
    trades = skip_overlap(trades)
    stats = summarize(trades)
    extra = {
        "symbol": symbols[0] if len(symbols) == 1 else "UNIVERSE",
        "universe": bool(args.universe) or len(symbols) > 1,
        "days": args.days,
        "min_ext": p.min_ext,
        "max_ext": p.max_ext,
        "min_fail": p.min_fail,
        "max_fail": p.max_fail,
        "min_range_break": p.min_range_break,
        "target_r": p.target_r,
        "time_bars": p.time_bars,
        "scanned": len(symbols),
        "raw_hits": raw_hits,
        "overlap_dropped": before - len(trades),
        "start": span.split(" → ")[0] if " → " in span else "",
        "end": span.split(" → ")[-1] if span else "",
    }
    print(
        f"{extra['symbol']} {args.days}d scanned={len(symbols)} raw={raw_hits} "
        f"taken={stats['count']} (drop overlap {extra['overlap_dropped']}) "
        f"{extra['start']} → {extra['end']}",
        flush=True,
    )
    print(
        f"signals={stats['count']} symbols={stats['symbols']} WR={stats['closed_win_rate']:.1f}% "
        f"avg={stats['avg_pct']*100:+.2f}% total={stats['total_pct']*100:+.2f}% "
        f"{stats['reasons']}",
        flush=True,
    )
    show = trades[:40]
    for i, t in enumerate(show, 1):
        print(
            f"[{i}] {t.symbol} {hm(int(t.d['t'][t.entry_idx]))} peak {hm(int(t.d['t'][t.signal.peak_i]))} "
            f"{t.reason} {t.pnl_pct*100:+.2f}% ext={t.signal.ext*100:.1f}% fail={t.signal.fail*100:.1f}%",
            flush=True,
        )
    if len(trades) > 40:
        print(f"  … 還有 {len(trades) - 40} 筆", flush=True)
    html_path = Path(args.html) if args.html else None
    if args.pages:
        html_path = pages_path(universe=extra["universe"], days=args.days, symbol=symbols[0])
    if html_path:
        out = write_html(html_path, trades, stats, extra, max_charts=args.max_charts)
        write_view_html(out)
        dump_hits(out.with_name("hits.json"), trades, stats, extra)
        print(f"html={out}", flush=True)
        print(f"view={out.with_name('view.html')}", flush=True)
    return 0


def cmd_watch(args) -> int:
    apply_keys()
    if args.test:
        return test_telegram()
    p = Params(min_ext=args.min_ext, max_ext=args.max_ext, min_fail=args.min_fail, max_fail=args.max_fail, min_range_break=args.min_range_break)
    seen = load_seen()
    if args.universe:
        print("載入標的…", flush=True)
        symbols = universe(min_quote_vol=args.min_quote_vol)
    else:
        symbols = [s.upper() for s in (args.symbols or [DEFAULT_SYMBOL])]
    print(
        f"監看 {len(symbols)} 檔 1h：HBAR 型箱體假突破後收盤跌破 MA25"
        f"（伸 {p.min_ext*100:g}～{p.max_ext*100:g}% 回落 {p.min_fail*100:g}～{p.max_fail*100:g}%）",
        flush=True,
    )

    def round_once() -> None:
        t0 = time.time()
        events: list[dict] = []
        for sym in symbols:
            try:
                events.extend(scan_symbol(sym, lookback=max(1, args.lookback), p=p))
            except Exception as e:
                print("err", sym, e, flush=True)
        new = [e for e in events if key_of(e["symbol"], e["d"], e["sig"]) not in seen]
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            notify(ev, dry_run=args.dry_run)
            if not args.dry_run:
                seen.add(key_of(ev["symbol"], ev["d"], ev["sig"]))
        if new and not args.dry_run:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中，每根 1h 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HBAR 1h 假突破跌破 MA25 Telegram 監看")
    p.add_argument("--once", action="store_true", help="只掃一次然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--dry-run", action="store_true", help="掃到也不送 Telegram")
    p.add_argument("--lookback", type=int, default=2, help="往回看幾根已收盤 1h（預設 2）")
    p.add_argument("--symbols", nargs="*", help="預設 HBARUSDT")
    p.add_argument("--min-ext", type=float, default=MIN_EXT, help="假突破高點相對 MA25 最少伸出去")
    p.add_argument("--max-ext", type=float, default=MAX_EXT, help="最多伸出去，避開妖幣直線")
    p.add_argument("--min-fail", type=float, default=MIN_FAIL, help="從高點回到收盤最少回落")
    p.add_argument("--max-fail", type=float, default=MAX_FAIL, help="最多回落，不要已經崩完才進")
    p.add_argument("--min-range-break", type=float, default=MIN_RANGE_BREAK, help="相對拉升前箱體最少再高出多少")
    p.add_argument("--backtest", action="store_true", help="回測最近 N 天並可出 HTML")
    p.add_argument("--universe", action="store_true", help="掃幣安流動 USDT 永續（全幣種）")
    p.add_argument("--min-quote-vol", type=float, default=MIN_QUOTE_VOL, help="24h 成交額門檻，預設 500 萬")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/…/index.html")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="回測代號（沒開 --universe 時）")
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--time-bars", type=int, default=TIME_BARS)
    p.add_argument("--max-charts", type=int, default=MAX_CHARTS)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backtest:
        if args.universe:
            args.symbols = args.symbols or []
        elif not args.symbols:
            args.symbols = [args.symbol]
        else:
            args.symbol = args.symbols[0]
        return cmd_backtest(args)
    return cmd_watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
