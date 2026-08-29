#!/usr/bin/env python3
"""幣安 15 分 K：同時跌破 MA99/120/200，且 7/14/25 空頭排列，做空回測一週。

對齊截圖 MUBARAK/USDT：一根大陰線從長均黏帶上沿打穿到三條之下，
同時短均 MA7 < MA14 < MA25。

    python3 examples/binance_15m_ma_break_short.py
    python3 examples/binance_15m_ma_break_short.py --symbol MUBARAKUSDT
    python3 examples/binance_15m_ma_break_short.py --pages
"""

from __future__ import annotations

import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TZ = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-15m-ma-short" / "index.html"
BRANCH_VIEW = "cursor/15m-ma-break-short-9d44"

INTERVAL = "15m"
BAR_MS = 15 * 60 * 1000
MA_SHORT = (7, 14, 25)
MA_LONG = (99, 120, 200)
WARMUP = 200
EVAL_DAYS = 7
MAX_HOLD = 32  # 8 小時
RR = 2.0
STOP_BUFFER = 0.001  # 停損在破位 K 高點上方 0.1%
MIN_QV = 5_000_000
KEEP = ("MUBARAKUSDT",)
MAX_CHARTS = 80

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0",
        "Clienttype": "web",
        "Accept": "application/json",
    }
)
FAPI = ("https://fapi.binance.com", "https://www.binance.com")
SPOT = ("https://api.binance.com", "https://data-api.binance.vision")

MA_COLORS = {
    7: "#f0c14a",
    14: "#26c6da",
    25: "#d28cff",
    99: "#5c6bc0",
    120: "#43a047",
    200: "#8d6e63",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def get_json(url: str, params: dict | None = None, retries: int = 5) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.35 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _klines_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    idx = pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TZ)
    df = pd.DataFrame(
        {
            "open": [float(x[1]) for x in raw],
            "high": [float(x[2]) for x in raw],
            "low": [float(x[3]) for x in raw],
            "close": [float(x[4]) for x in raw],
            "volume": [float(x[5]) for x in raw],
        },
        index=idx,
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_klines(symbol: str, limit: int = 1000, *, futures: bool = True) -> pd.DataFrame:
    hosts = FAPI if futures else SPOT
    path = "/fapi/v1/klines" if futures else "/api/v3/klines"
    last: Exception | None = None
    for host in hosts:
        try:
            raw = get_json(
                host + path,
                {"symbol": symbol, "interval": INTERVAL, "limit": limit},
            )
            df = _klines_to_df(raw)
            if df.empty:
                continue
            now_ms = int(time.time() * 1000)
            last_open = int(df.index[-1].tz_convert("UTC").timestamp() * 1000)
            if last_open + BAR_MS > now_ms:
                df = df.iloc[:-1]
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
    if futures:
        return fetch_klines(symbol, limit=limit, futures=False)
    raise RuntimeError(f"klines {symbol}: {last}")


def universe(min_qv: float = MIN_QV) -> list[str]:
    info = None
    tickers = None
    last: Exception | None = None
    for host in FAPI:
        try:
            info = get_json(host + "/fapi/v1/exchangeInfo")
            tickers = {t["symbol"]: t for t in get_json(host + "/fapi/v1/ticker/24hr")}
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
    if info is None or tickers is None:
        raise RuntimeError(f"universe: {last}")
    out: list[str] = []
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
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_qv and sym not in KEEP:
            continue
        out.append(sym)
    for k in KEEP:
        if k not in out:
            out.append(k)
    return out


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    for n in MA_SHORT + MA_LONG:
        out[f"ma{n}"] = close.rolling(n, min_periods=n).mean()
    if "volume" in out.columns:
        out["vol20"] = out["volume"].rolling(20, min_periods=20).mean()
    return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    symbol: str
    bar_idx: int
    timestamp: pd.Timestamp
    entry: float
    stop: float
    target: float
    ma7: float
    ma14: float
    ma25: float
    ma99: float
    ma120: float
    ma200: float
    cluster_hi: float
    cluster_lo: float
    vol_ratio: float
    break_pct: float


@dataclass
class TradeResult:
    signal: Signal
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_pct: float


def detect_signals(
    df: pd.DataFrame,
    symbol: str = "",
    *,
    eval_start: pd.Timestamp | None = None,
    funnel: dict[str, int] | None = None,
) -> list[Signal]:
    """一根 15m 同時打穿 MA99/120/200，且 MA7<MA14<MA25。"""
    if len(df) < WARMUP + 2:
        return []
    work = add_mas(df) if "ma200" not in df.columns else df
    c = work["close"].to_numpy(float)
    h = work["high"].to_numpy(float)
    m7 = work["ma7"].to_numpy(float)
    m14 = work["ma14"].to_numpy(float)
    m25 = work["ma25"].to_numpy(float)
    m99 = work["ma99"].to_numpy(float)
    m120 = work["ma120"].to_numpy(float)
    m200 = work["ma200"].to_numpy(float)
    vol = work["volume"].to_numpy(float) if "volume" in work.columns else np.ones(len(work))
    v20 = work["vol20"].to_numpy(float) if "vol20" in work.columns else np.full(len(work), np.nan)

    def bump(key: str) -> None:
        if funnel is not None:
            funnel[key] = funnel.get(key, 0) + 1

    signals: list[Signal] = []
    for i in range(WARMUP, len(work)):
        vals = [m7[i], m14[i], m25[i], m99[i], m120[i], m200[i], m99[i - 1], m120[i - 1], m200[i - 1]]
        if np.isnan(vals).any():
            continue
        if eval_start is not None and work.index[i] < eval_start:
            continue

        cluster_hi = max(m99[i], m120[i], m200[i])
        cluster_lo = min(m99[i], m120[i], m200[i])
        prev_lo = min(m99[i - 1], m120[i - 1], m200[i - 1])
        now_below = c[i] < m99[i] and c[i] < m120[i] and c[i] < m200[i]
        prev_not_below = c[i - 1] >= prev_lo
        # 這根有碰到長均帶（含跳空：前收還在帶上）
        pierced = h[i] >= cluster_lo or c[i - 1] >= prev_lo
        if not (now_below and prev_not_below and pierced):
            continue
        bump("break")

        if not (m7[i] < m14[i] < m25[i]):
            bump("skip_stack")
            continue
        bump("taken")

        entry = float(c[i])
        stop = float(h[i]) * (1.0 + STOP_BUFFER)
        risk = stop - entry
        if risk <= 0 or risk / entry > 0.18:
            bump("skip_risk")
            continue
        target = entry - RR * risk
        vr = float(vol[i] / v20[i]) if v20[i] and not np.isnan(v20[i]) and v20[i] > 0 else 0.0
        signals.append(
            Signal(
                symbol=symbol,
                bar_idx=i,
                timestamp=work.index[i],
                entry=entry,
                stop=stop,
                target=target,
                ma7=float(m7[i]),
                ma14=float(m14[i]),
                ma25=float(m25[i]),
                ma99=float(m99[i]),
                ma120=float(m120[i]),
                ma200=float(m200[i]),
                cluster_hi=float(cluster_hi),
                cluster_lo=float(cluster_lo),
                vol_ratio=vr,
                break_pct=(cluster_hi - entry) / cluster_hi * 100.0,
            )
        )
    return signals


def simulate(df: pd.DataFrame, signals: list[Signal], *, max_hold: int = MAX_HOLD) -> list[TradeResult]:
    if df.empty or not signals:
        return []
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    ma99 = (
        df["ma99"].to_numpy(float)
        if "ma99" in df.columns
        else df["close"].rolling(99, min_periods=99).mean().to_numpy(float)
    )
    results: list[TradeResult] = []
    busy_until = -1
    for sig in sorted(signals, key=lambda s: s.bar_idx):
        if sig.bar_idx <= busy_until:
            continue
        end = min(sig.bar_idx + max_hold, len(df) - 1)
        exit_idx = end
        exit_price = float(close[end])
        reason = "time_stop"
        for i in range(sig.bar_idx + 1, end + 1):
            if high[i] >= sig.stop:
                exit_idx = i
                exit_price = sig.stop
                reason = "stop_loss"
                break
            if low[i] <= sig.target:
                exit_idx = i
                exit_price = sig.target
                reason = "take_profit"
                break
            if not np.isnan(ma99[i]) and close[i] > ma99[i]:
                exit_idx = i
                exit_price = float(close[i])
                reason = "reclaim_ma99"
                break
        busy_until = exit_idx
        pnl = (sig.entry - exit_price) / sig.entry * 100.0
        results.append(
            TradeResult(
                signal=sig,
                exit_idx=exit_idx,
                exit_time=df.index[exit_idx],
                exit_price=exit_price,
                exit_reason=reason,
                pnl_pct=pnl,
            )
        )
    return results


def summarize(trades: list[TradeResult]) -> dict[str, float | int]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    total = sum(t.pnl_pct for t in trades)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": 100.0 * wins / len(trades),
        "total_pnl": total,
        "avg_pnl": total / len(trades),
    }


# ---------------------------------------------------------------------------
# Charts + HTML
# ---------------------------------------------------------------------------


def _setup_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _setup_font()
    sig = trade.signal
    start = max(0, sig.bar_idx - 36)
    end = min(len(df) - 1, max(trade.exit_idx + 10, sig.bar_idx + 16))
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    close_full = df["close"].astype(float)

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
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 25 else 1.05, label=f"MA{n}")

    ax.axhline(trade.signal.stop, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.signal.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    ex, xx = sig.bar_idx - start, trade.exit_idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([ex], [sig.entry], s=42, color="#ff5252", marker="v", zorder=6)
        ax.annotate("做空", (ex, sig.entry), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#ff8a80", fontsize=8)
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    et = sig.timestamp
    xt = trade.exit_time
    if hasattr(et, "tz_convert"):
        et = et.tz_convert(TZ)
        xt = xt.tz_convert(TZ)
    sign = "+" if trade.pnl_pct >= 0 else ""
    ax.set_title(
        f"#{trade_no}  {sig.symbol}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_pct:.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _equity_svg(pnls: list[float]) -> str:
    if not pnls:
        return ""
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    w, h = 720, 180
    lo, hi = min(eq), max(eq)
    span = hi - lo or 1.0
    pts = []
    for i, v in enumerate(eq):
        x = i / (len(eq) - 1) * w if len(eq) > 1 else 0
        y = h - 16 - (v - lo) / span * (h - 32)
        pts.append(f"{x:.1f},{y:.1f}")
    zero_y = h - 16 - (0 - lo) / span * (h - 32)
    color = "#16a34a" if eq[-1] >= 0 else "#ef4444"
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero_y:.1f}" x2="{w}" y2="{zero_y:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>'
        f"</svg>"
    )


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fmt_px(px: float) -> str:
    if px >= 100:
        return f"{px:.2f}"
    if px >= 1:
        return f"{px:.4f}"
    return f"{px:.6f}".rstrip("0").rstrip(".")


def _render_cards(
    trades: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    html_path: Path,
    *,
    embed: bool,
    prefix: str = "t",
) -> str:
    cards: list[str] = []
    img_dir = html_path.parent / "img"
    for i, t in enumerate(trades, 1):
        sig = t.signal
        df = frames[sig.symbol]
        et = sig.timestamp.tz_convert(TZ) if getattr(sig.timestamp, "tzinfo", None) else sig.timestamp
        xt = t.exit_time.tz_convert(TZ) if getattr(t.exit_time, "tzinfo", None) else t.exit_time
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        reason_cls = {
            "take_profit": "tag-tp",
            "stop_loss": "tag-sl",
            "reclaim_ma99": "tag-time",
            "time_stop": "tag-time",
        }.get(t.exit_reason, "tag-time")
        img_name = f"{prefix}{i:02d}_{sig.symbol}_{et.strftime('%m%d_%H%M')}.png"
        chart_html = ""
        if i <= MAX_CHARTS:
            png = draw_trade_png(df, t, img_dir / img_name, i)
            src = _img_data_uri(png) if embed else f"img/{img_name}"
            chart_html = (
                f"<div class='mini-chart'><img src='{escape(src)}' alt='#{i} {escape(sig.symbol)}' "
                "style='width:100%;display:block;border-radius:10px'/></div>"
            )
        risk_pct = (sig.stop - sig.entry) / sig.entry * 100.0
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(sig.symbol)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → "
            f"{escape(xt.strftime('%m-%d %H:%M'))} 台北</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            "<span class='tag tag-info'>15m 做空</span>"
            f"<span class='tag tag-info'>量比 {sig.vol_ratio:.1f}x</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {_fmt_px(sig.entry)}\n"
            f"stop  {_fmt_px(sig.stop)}  (風險 {risk_pct:.2f}%)\n"
            f"target {_fmt_px(sig.target)}  ({RR:.1f}R)\n"
            f"exit  {_fmt_px(t.exit_price)}  {t.exit_reason}\n"
            f"打穿長均 {sig.break_pct:.2f}%  cluster {_fmt_px(sig.cluster_lo)}–{_fmt_px(sig.cluster_hi)}\n"
            f"MA7 {_fmt_px(sig.ma7)} < MA14 {_fmt_px(sig.ma14)} < MA25 {_fmt_px(sig.ma25)}\n"
            f"MA99 {_fmt_px(sig.ma99)} / MA120 {_fmt_px(sig.ma120)} / MA200 {_fmt_px(sig.ma200)}"
            "</pre>"
            f"{chart_html}"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: Path,
    trades: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    funnel: dict[str, int] | None = None,
    embed: bool = False,
    mubarak_trades: list[TradeResult] | None = None,
) -> Path:
    stats = summarize(trades)
    total_cls = "pnl-win" if float(stats["total_pnl"]) >= 0 else "pnl-loss"
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：同時跌破 99/120/200　{funnel.get('break', 0)} → "
            f"7&lt;14&lt;25 進場 {funnel.get('taken', 0)}"
            f"（短均未排列 {funnel.get('skip_stack', 0)} · 風險過大 {funnel.get('skip_risk', 0)}）</p>"
        )
    extra = ""
    if mubarak_trades is not None:
        ms = summarize(mubarak_trades)
        mcls = "pnl-win" if float(ms["total_pnl"]) >= 0 else "pnl-loss"
        extra = (
            "<section class='summary'><h1>MUBARAKUSDT（截圖標的）</h1>"
            f"<p class='muted'>同一套規則，只看圖上那檔。</p>"
            f"<div class='cards'><div class='card'>筆數<b>{ms['trades']}</b></div>"
            f"<div class='card'>勝率<b>{ms['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總報酬<b class='{mcls}'>{ms['total_pnl']:+.2f}%</b></div>"
            f"<div class='card'>勝/負<b>{ms['wins']}/{ms['losses']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_pct for t in mubarak_trades])}</div></section>"
            + (
                _render_cards(mubarak_trades, frames, path, embed=embed, prefix="m")
                or "<div class='empty'>這一週 MUBARAK 沒打出同時跌破＋空頭排列</div>"
            )
        )
    cards = _render_cards(trades, frames, path, embed=embed, prefix="t")
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
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
.equity{{margin:10px 0 4px}}
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
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(title)}</h1>
<p class="muted">{escape(subtitle)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['trades']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總報酬<b class="{total_cls}">{stats['total_pnl']:+.2f}%</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['losses']}</b></div>
</div>
<p class="muted">停損＝破位 K 高點 +0.1% · 停利 2R · 收復 MA99 或持倉 {MAX_HOLD} 根（8h）平倉。報酬是單筆價格百分比，未計資金費。</p>
{funnel_line}
<div class="equity">{_equity_svg([t.pnl_pct for t in trades])}</div>
</section>
{extra}
{cards or "<div class='empty'>這一週沒有同時跌破＋空頭排列的做空訊號</div>"}
</div>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = BRANCH_VIEW) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    trades: list[TradeResult] = field(default_factory=list)
    funnel: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    errors: int = 0


def eval_start_ts(days: int, now: datetime | None = None) -> pd.Timestamp:
    now = now or datetime.now(TZ)
    return pd.Timestamp(now - timedelta(days=days))


def scan_symbol(sym: str, *, days: int, limit: int) -> tuple[str, pd.DataFrame, list[TradeResult], dict[str, int]]:
    df = fetch_klines(sym, limit=limit)
    df = add_mas(df)
    funnel: dict[str, int] = {}
    start = eval_start_ts(days)
    sigs = detect_signals(df, sym, eval_start=start, funnel=funnel)
    trades = simulate(df, sigs)
    return sym, df, trades, funnel


def scan_many(symbols: list[str], *, days: int, limit: int, workers: int = 10) -> ScanResult:
    out = ScanResult()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(scan_symbol, s, days=days, limit=limit): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                name, df, trades, funnel = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"err {sym} {exc}", flush=True)
                out.errors += 1
                continue
            out.symbols += 1
            out.frames[name] = df
            out.trades.extend(trades)
            for k, v in funnel.items():
                out.funnel[k] = out.funnel.get(k, 0) + v
            print(f"  {name} bars={len(df)} trades={len(trades)}", flush=True)
    out.trades.sort(key=lambda t: (t.signal.timestamp, t.signal.symbol))
    return out


def print_summary(label: str, trades: list[TradeResult], funnel: dict[str, int] | None = None) -> None:
    stats = summarize(trades)
    print(
        f"{label}: trades={stats['trades']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_pnl']:+.2f}% avg={stats['avg_pnl']:+.2f}%"
    )
    if funnel:
        print(
            f"  funnel break={funnel.get('break', 0)} taken={funnel.get('taken', 0)} "
            f"skip_stack={funnel.get('skip_stack', 0)} skip_risk={funnel.get('skip_risk', 0)}"
        )
    for i, t in enumerate(trades, 1):
        et = t.signal.timestamp.tz_convert(TZ) if getattr(t.signal.timestamp, "tzinfo", None) else t.signal.timestamp
        xt = t.exit_time.tz_convert(TZ) if getattr(t.exit_time, "tzinfo", None) else t.exit_time
        print(
            f"  [{i}] {t.signal.symbol} {et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_pct:+.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="幣安 15m 同時跌破 99/120/200 + 短均空頭排列 做空回測")
    p.add_argument("--symbol", default="", help="只回測單一標的，例如 MUBARAKUSDT")
    p.add_argument("--days", type=int, default=EVAL_DAYS)
    p.add_argument("--limit", type=int, default=1000, help="15m K 根數（含均線暖身）")
    p.add_argument("--min-qv", type=float, default=MIN_QV)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/binance-15m-ma-short/")
    p.add_argument("--embed", action="store_true", help="圖用 base64 嵌進 HTML")
    args = p.parse_args(argv)

    if args.symbol:
        symbols = [args.symbol.upper().replace("/", "")]
        if not symbols[0].endswith("USDT"):
            symbols[0] += "USDT"
    else:
        print("載入標的…", flush=True)
        symbols = universe(args.min_qv)
        print(f"掃描 {len(symbols)} 個 USDT 永續（含 {', '.join(KEEP)}）", flush=True)

    t0 = time.time()
    result = scan_many(symbols, days=args.days, limit=args.limit, workers=args.workers)
    print(f"掃完 {result.symbols} 檔 用 {time.time() - t0:.1f}s  失敗 {result.errors}", flush=True)
    print_summary("全市場", result.trades, result.funnel)

    mubarak = [t for t in result.trades if t.signal.symbol == "MUBARAKUSDT"]
    if not args.symbol or args.symbol.upper().replace("/", "") in {"MUBARAK", "MUBARAKUSDT"}:
        if "MUBARAKUSDT" in result.frames and not mubarak:
            print_summary("MUBARAKUSDT", [], result.funnel if args.symbol else None)
        elif mubarak:
            print_summary("MUBARAKUSDT", mubarak)

    html_path = Path(args.html) if args.html else None
    if args.pages:
        html_path = PAGES
    if html_path:
        now = datetime.now(TZ)
        start = eval_start_ts(args.days, now)
        title = "15m 同時跌破 99/120/200 做空"
        subtitle = (
            f"{args.days} 日 · {start.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} 台北 · "
            f"{result.symbols} 檔 15m · 進場＝一根 K 打穿 MA99/120/200 且 7<14<25"
        )
        show_mubarak = not args.symbol or "MUBARAK" in args.symbol.upper()
        write_html_report(
            html_path,
            result.trades,
            result.frames,
            title=title,
            subtitle=subtitle,
            funnel=result.funnel,
            embed=args.embed,
            mubarak_trades=mubarak if show_mubarak and not args.symbol else None,
        )
        view = write_view_html(html_path)
        print(f"html={html_path}")
        print(f"view={view}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
