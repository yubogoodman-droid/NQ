#!/usr/bin/env python3
"""幣安 15 分 K：破底 → 站上 NB 圖上的六條均線 → 等回測 MA25 做多。

圖上的均線：MA7、MA14、MA25、MA99、MA120、MA200。

進場
  1. 破底：最低價跌破前 32 根，而且收盤同時低於這六條。
  2. 站回：48 根內，第一根收盤同時高於這六條，且 MA7 > MA14 > MA25。
  3. 先離開：之後至少一根 K 的最低價也在 MA7 之上。
  4. 回測進場：再下來第一根最低價觸到 MA25，收盤站回 MA25，
     同時收盤仍高於 MA99、MA120、MA200，短均維持 7 > 14 > 25。
     先收盤跌破 MA25、MA99、MA120、MA200 任一條，這波作廢。

出場（這次沒指定，回測用）
  停損＝回測那根的低點。目標＝2R。32 根（8 小時）都沒碰到就用收盤時間停。
  進場這根不再用它的高低點出場。同一根同時碰到停損和目標，算停損。
  停損距離小於 0.2% 或大於 4% 不做。

用法
  python3 examples/test_binance_15m_ma25_retest.py
  python3 examples/binance_15m_ma25_retest.py --days 3 --html docs/binance/ma25-retest-3d/index.html
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import requests

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
ROOT = Path(__file__).resolve().parents[1]
KEEP = {"NBISUSDT", "UBUSDT", "STXXUSDT", "SNDKUSDT"}


@dataclass(frozen=True)
class Params:
    ma: int = 25
    break_lookback: int = 32
    reclaim_window: int = 48
    retest_window: int = 48
    target_r: float = 2.0
    time_stop_bars: int = 32
    min_risk_pct: float = 0.002
    max_risk_pct: float = 0.04


@dataclass
class Trade:
    symbol: str
    trough_i: int
    reclaim_i: int
    entry_i: int
    exit_i: int
    trough_low: float
    entry: float
    stop: float
    target: float
    exit: float
    reason: str
    pnl_pct: float
    depth_pct: float
    quote_volume: float = 0.0

    @property
    def risk_pct(self) -> float:
        if self.entry <= 0:
            return 0.0
        return (self.entry - self.stop) / self.entry * 100.0


def sma(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= n:
        out[n - 1 :] = np.convolve(values, np.ones(n) / n, mode="valid")
    return out


def _bump(funnel: dict, key: str) -> None:
    funnel[key] = funnel.get(key, 0) + 1


def manage_exit(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    entry_i: int,
    entry: float,
    stop: float,
    target: float,
    time_stop_bars: int,
) -> tuple[int, float, str]:
    """從下一根開始看出場。同一根先算停損。資料在時間停之前結束 → 未平倉。"""
    last = len(c) - 1
    deadline = entry_i + time_stop_bars
    end = min(last, deadline)
    for j in range(entry_i + 1, end + 1):
        hit_stop = l[j] <= stop
        hit_target = h[j] >= target
        if hit_stop:
            px = float(o[j] if o[j] < stop else stop)
            return j, px, "停損"
        if hit_target:
            px = float(o[j] if o[j] > target else target)
            return j, px, "2R"
    if last < deadline:
        return last, float(c[last]), "未平倉"
    return end, float(c[end]), "時間停"


def _mas(c: np.ndarray) -> dict[int, np.ndarray]:
    return {n: sma(c, n) for n in (7, 14, 25, 99, 120, 200)}


def _ready(mas: dict[int, np.ndarray], i: int) -> bool:
    return not any(np.isnan(mas[n][i]) for n in mas)


def _under_all(c: np.ndarray, mas: dict[int, np.ndarray], i: int) -> bool:
    return all(c[i] < mas[n][i] for n in mas)


def _above_all(c: np.ndarray, mas: dict[int, np.ndarray], i: int) -> bool:
    return all(c[i] > mas[n][i] for n in mas)


def _fan(mas: dict[int, np.ndarray], i: int) -> bool:
    return mas[7][i] > mas[14][i] > mas[25][i]


def _holds_long(c: np.ndarray, mas: dict[int, np.ndarray], i: int) -> bool:
    return c[i] > mas[25][i] and c[i] > mas[99][i] and c[i] > mas[120][i] and c[i] > mas[200][i]


def detect_trades(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    params: Params | None = None,
    min_entry_i: int = 0,
    funnel: dict | None = None,
) -> list[Trade]:
    """回傳進場索引 >= min_entry_i 的交易。有倉時不重疊進下一筆。"""
    p = params or Params()
    fun = funnel if funnel is not None else {}
    n = len(c)
    mas = _mas(c)
    ma = mas[25]
    trades: list[Trade] = []
    i = max(200, p.break_lookback)
    while i < n:
        if not _ready(mas, i) or not (l[i] < np.min(l[i - p.break_lookback : i]) and _under_all(c, mas, i)):
            i += 1
            continue
        if i >= min_entry_i:
            _bump(fun, "break")
        trough_i = i
        trough_low = float(l[i])
        reclaim_i = None
        j = i + 1
        j_end = min(n, i + 1 + p.reclaim_window)
        while j < j_end:
            if l[j] < trough_low:
                trough_low = float(l[j])
                trough_i = j
            if j > trough_i and _ready(mas, j) and _above_all(c, mas, j) and _fan(mas, j):
                reclaim_i = j
                break
            j += 1
        if reclaim_i is None:
            if i >= min_entry_i:
                _bump(fun, "no_reclaim")
            i = max(j, i + 1)
            continue
        if reclaim_i >= min_entry_i:
            _bump(fun, "reclaim")

        held = False
        entry_i = None
        fail_i = None
        k = reclaim_i + 1
        k_end = min(n, reclaim_i + 1 + p.retest_window)
        while k < k_end:
            if not _ready(mas, k):
                k += 1
                continue
            if c[k] < ma[k] or c[k] < mas[99][k] or c[k] < mas[120][k] or c[k] < mas[200][k]:
                fail_i = k
                break
            if not held:
                if l[k] > mas[7][k]:
                    held = True
                k += 1
                continue
            if l[k] <= ma[k] and _holds_long(c, mas, k) and _fan(mas, k):
                entry_i = k
                break
            k += 1

        if entry_i is None:
            if reclaim_i >= min_entry_i:
                if not held:
                    _bump(fun, "no_stand")
                elif fail_i is not None:
                    _bump(fun, "lost_before_retest")
                else:
                    _bump(fun, "no_retest")
            i = max((fail_i if fail_i is not None else k), reclaim_i + 1)
            continue

        entry = float(c[entry_i])
        stop = float(l[entry_i])
        risk = entry - stop
        risk_pct = risk / entry if entry > 0 else 0.0
        if entry_i >= min_entry_i:
            _bump(fun, "retest")
        if risk <= 0 or not (p.min_risk_pct <= risk_pct <= p.max_risk_pct):
            if entry_i >= min_entry_i:
                _bump(fun, "risk_skip")
            i = entry_i + 1
            continue

        target = entry + p.target_r * risk
        exit_i, exit_px, reason = manage_exit(o, h, l, c, entry_i, entry, stop, target, p.time_stop_bars)
        if entry_i >= min_entry_i:
            ma_at = float(ma[trough_i]) if not np.isnan(ma[trough_i]) else entry
            depth = (ma_at - trough_low) / ma_at * 100.0 if ma_at else 0.0
            pnl = (exit_px - entry) / entry * 100.0 if entry else 0.0
            trades.append(
                Trade(
                    symbol="",
                    trough_i=trough_i,
                    reclaim_i=reclaim_i,
                    entry_i=entry_i,
                    exit_i=exit_i,
                    trough_low=trough_low,
                    entry=entry,
                    stop=stop,
                    target=target,
                    exit=exit_px,
                    reason=reason,
                    pnl_pct=pnl,
                    depth_pct=depth,
                )
            )
            _bump(fun, "trades")
        i = exit_i + 1
    return trades


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.reason != "未平倉"]
    n = len(closed)
    wins = sum(1 for t in closed if t.pnl_pct > 0)
    by: dict[str, dict] = {}
    for t in trades:
        row = by.setdefault(t.reason, {"n": 0, "pnl": 0.0, "wins": 0})
        row["n"] += 1
        row["pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            row["wins"] += 1
    return {
        "count": n,
        "open": len(trades) - n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "avg_pct": float(np.mean([t.pnl_pct for t in closed])) if n else 0.0,
        "sum_pct": float(sum(t.pnl_pct for t in closed)),
        "sum_all_pct": float(sum(t.pnl_pct for t in trades)),
        "by_reason": by,
    }


def fmt_px(price: float) -> str:
    ax = abs(price)
    if ax >= 100:
        return f"{price:.2f}"
    if ax >= 1:
        return f"{price:.4f}"
    if ax >= 0.01:
        return f"{price:.5f}"
    return f"{price:.8f}"


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def draw_trade_png(bars: dict, trade: Trade, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    for font_path in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    o, h, l, c = bars["o"], bars["h"], bars["l"], bars["c"]
    a0 = max(0, trade.trough_i - 16)
    a1 = min(len(c), trade.exit_i + 6)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    fig, ax = plt.subplots(figsize=(10.4, 5.2), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")
    oo, hh, ll, cc = o[sl], h[sl], l[sl], c[sl]
    for k in range(len(cc)):
        up = cc[k] >= oo[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], ll[k], hh[k], color=col, lw=0.7)
        y0, y1 = min(oo[k], cc[k]), max(oo[k], cc[k])
        if y1 == y0:
            y1 = y0 + max(hh[k] - ll[k], 1e-12) * 0.04
        ax.add_patch(Rectangle((xs[k] - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
    full_c = bars["c"]
    mas = _mas(full_c)
    palette = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 99: "#5fd2c2", 120: "#42a5f5", 200: "#ffffff"}
    for n, col in palette.items():
        ax.plot(xs, mas[n][sl], color=col, lw=1.05, label=f"MA{n}")
    marks = (
        (trade.trough_i, "破底", "#8ab4ff", trade.trough_low),
        (trade.reclaim_i, "站回", "#f0c14a", c[trade.reclaim_i]),
        (trade.entry_i, "回測進", "#3dba7a", trade.entry),
        (trade.exit_i, trade.reason, "#ff8a80", trade.exit),
    )
    for idx, label, col, y in marks:
        x = idx - a0
        if 0 <= x < len(cc):
            ax.scatter([x], [y], s=28, color=col, zorder=5)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), color=col, fontsize=8, ha="center")
    ax.axhline(trade.stop, color="#e35d5d", ls="--", lw=0.7)
    ax.axhline(trade.target, color="#3dba7a", ls="--", lw=0.7)
    ax.set_title(title, color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_html(
    path: Path,
    rows: list[tuple[Trade, dict, list[int]]],
    stats: dict,
    funnel: dict,
    meta: dict,
) -> Path:
    cards: list[str] = []
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    for i, (trade, bars, times) in enumerate(rows, 1):
        et = fmt_ts(times[trade.entry_i])
        xt = fmt_ts(times[trade.exit_i])
        bt = fmt_ts(times[trade.trough_i])
        rt = fmt_ts(times[trade.reclaim_i])
        cls = "pnl-win" if trade.pnl_pct > 0 else ("pnl-flat" if trade.pnl_pct == 0 else "pnl-loss")
        img_name = f"t{i:03d}_{trade.symbol}_{et.replace(' ', '_').replace(':', '')}.png"
        title = f"{trade.symbol}  {et}  {trade.pnl_pct:+.2f}%  {trade.reason}"
        draw_trade_png(bars, trade, img_dir / img_name, title)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(trade.symbol)}</span>"
            f"<span class='trade-time'>{escape(et)} → {escape(xt)}</span></div>"
            f"<div class='card-pnl {cls}'>{trade.pnl_pct:+.2f}%</div>"
            "</header>"
            f"<div class='tags'><span class='tag'>{escape(trade.reason)}</span>"
            f"<span class='tag tag-info'>風險 {trade.risk_pct:.2f}%</span>"
            f"<span class='tag tag-info'>破底深 {trade.depth_pct:.2f}%</span></div>"
            "<pre class='trade-detail'>"
            f"破底 {escape(bt)}  low {fmt_px(trade.trough_low)}\n"
            f"站回 {escape(rt)}\n"
            f"進場 {fmt_px(trade.entry)}  停損 {fmt_px(trade.stop)}  目標 {fmt_px(trade.target)}\n"
            f"出場 {fmt_px(trade.exit)}  {escape(trade.reason)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(trade.symbol)}'/></div>"
            "</article>"
        )

    reason_bits = []
    for name, row in sorted(stats["by_reason"].items(), key=lambda kv: -kv[1]["n"]):
        reason_bits.append(f"{escape(name)} {row['n']} 筆 {row['pnl']:+.2f}%")
    reason_line = " · ".join(reason_bits) if reason_bits else "沒有交易"
    sum_cls = "pnl-win" if stats["sum_pct"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 15 分 破底站回 MA25 回測</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{display:block;font-size:12px;color:#8b949e;margin-top:2px}}
.card-pnl{{font-weight:700;white-space:nowrap}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#c9d1d9}}
.tag-info{{color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap;margin:0}}
.mini-chart img{{width:100%;display:block;border-radius:10px;margin-top:8px}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15 分 · 六條均線站回後，等回測 MA25</h1>
<p class="muted">{escape(meta['period'])} · 掃描 {meta['symbols']} 檔 U 本位永續（24h 成交額 ≥ 500 萬 USDT）· 進場 {meta['entries']} 筆
<br/>均線用 NB 圖上那六條：MA7、MA14、MA25、MA99、MA120、MA200。破底＝收盤同時低於這六條，且低點跌破前 32 根。48 根內收盤同時站上六條，且 MA7 &gt; MA14 &gt; MA25。先有一根低點也在 MA7 上，之後觸到 MA25、收盤仍站上 MA25 / MA99 / MA120 / MA200，短均排列還在，才做多。
<br/>出場：停損在回測低點，目標 2R，或 32 根時間停。停損距離 0.2%–4%。加總％是各筆報酬相加，不是組合複利。未平倉不計勝率，用最後一根收盤估。沒扣手續費與滑價。
<br/>漏斗：破底 {funnel.get('break', 0)} → 站上六條 {funnel.get('reclaim', 0)} → 回測 {funnel.get('retest', 0)} → 風險過濾掉 {funnel.get('risk_skip', 0)} → 成交 {funnel.get('trades', 0)}
<br/>沒站上六條 {funnel.get('no_reclaim', 0)} · 沒離開 MA7 {funnel.get('no_stand', 0)} · 回測前失守 {funnel.get('lost_before_retest', 0)} · 等到超時 {funnel.get('no_retest', 0)}
<br/>{reason_line}</p>
<div class="cards">
<div class="card">已平<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{sum_cls}">{stats['avg_pct']:+.2f}%</b></div>
<div class="card">加總<b class="{sum_cls}">{stats['sum_pct']:+.2f}%</b></div>
<div class="card">未平<b>{stats['open']}</b></div>
</div>
<p class="muted">含未平倉的加總 {stats['sum_all_pct']:+.2f}%。這是規則回測，不是進出場建議。</p>
</section>
{''.join(cards) or "<div class='empty'>這三天沒有走完回測進場的交易</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})
    return s


def get_json(session: requests.Session, path: str, params: dict | None = None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = session.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (i + 1))
    raise last


def universe(session: requests.Session) -> list[tuple[str, float]]:
    info = get_json(session, "/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json(session, "/fapi/v1/ticker/24hr")}
    out: list[tuple[str, float]] = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < 5_000_000 and sym not in KEEP:
            continue
        out.append((sym, qv))
    return out


BAR_MS = 15 * 60 * 1000


def fetch_klines(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> dict | None:
    """15 分 K 分頁抓。幣安單次最多 1500 根，一個月要連抓幾次。"""
    merged: dict[int, list] = {}
    cursor = start_ms
    for _ in range(8):
        if cursor >= end_ms:
            break
        batch = get_json(
            session,
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": "15m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not batch:
            break
        for row in batch:
            merged[int(row[0])] = row
        last_open = int(batch[-1][0])
        nxt = last_open + BAR_MS
        if nxt <= cursor or len(batch) < 1500:
            break
        cursor = nxt
    if not merged:
        return None
    rows = [merged[k] for k in sorted(merged)]
    now_ms = int(time.time() * 1000)
    if int(rows[-1][0]) + BAR_MS > now_ms:
        rows = rows[:-1]
    if len(rows) < 220:
        return None
    return {
        "t": [int(x[0]) for x in rows],
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
    }


def window_start_ms(days: int, last_ms: int) -> int:
    last = datetime.fromtimestamp(last_ms / 1000, TZ)
    start = datetime(last.year, last.month, last.day, tzinfo=TZ) - timedelta(days=days - 1)
    return int(start.timestamp() * 1000)


def scan_symbol(symbol: str, qv: float, days: int, params: Params) -> tuple[list[tuple[Trade, dict, list[int]]], dict, str]:
    session = _session()
    end_ms = int(time.time() * 1000)
    start_ms = window_start_ms(days, end_ms) - 10 * 24 * 60 * 60 * 1000
    try:
        bars = fetch_klines(session, symbol, start_ms, end_ms)
    except Exception as exc:  # noqa: BLE001
        return [], {}, str(exc)[:80]
    if bars is None:
        return [], {}, "too_few_bars"
    times = bars["t"]
    start_ms = window_start_ms(days, times[-1])
    min_entry_i = next((i for i, ts in enumerate(times) if ts >= start_ms), len(times))
    funnel: dict = {}
    trades = detect_trades(bars["o"], bars["h"], bars["l"], bars["c"], params, min_entry_i, funnel)
    rows = []
    for trade in trades:
        trade.symbol = symbol
        trade.quote_volume = qv
        rows.append((trade, bars, times))
    return rows, funnel, ""


def merge_funnel(parts: list[dict]) -> dict:
    out: dict[str, int] = {}
    for part in parts:
        for k, v in part.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def run_scan(days: int = 3, workers: int = 12, params: Params | None = None) -> tuple[list, dict, dict]:
    params = params or Params()
    session = _session()
    symbols = universe(session)
    rows: list[tuple[Trade, dict, list[int]]] = []
    funnels: list[dict] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scan_symbol, sym, qv, days, params): sym for sym, qv in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            got, funnel, err = fut.result()
            if err:
                errors += 1
            rows.extend(got)
            funnels.append(funnel)
            if done % 40 == 0 or done == len(symbols):
                print(f"[scan] {done}/{len(symbols)} trades={len(rows)} errors={errors}", flush=True)
    rows.sort(key=lambda r: (r[2][r[0].entry_i], r[0].symbol))
    last_ms = max((r[2][-1] for r in rows), default=int(time.time() * 1000))
    start_ms = window_start_ms(days, last_ms)
    meta = {
        "symbols": len(symbols),
        "errors": errors,
        "entries": len(rows),
        "period": (
            f"{datetime.fromtimestamp(start_ms / 1000, TZ):%Y-%m-%d} → "
            f"{datetime.fromtimestamp(last_ms / 1000, TZ):%Y-%m-%d %H:%M} 台北"
        ),
    }
    return rows, merge_funnel(funnels), meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="幣安 15 分破底站回 MA25 回測")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--html", default=str(ROOT / "docs" / "binance" / "ma25-retest-3d" / "index.html"))
    args = parser.parse_args(argv)
    rows, funnel, meta = run_scan(args.days, args.workers)
    trades = [r[0] for r in rows]
    stats = summarize(trades)
    path = Path(args.html)
    payload = {
        "meta": meta,
        "funnel": funnel,
        "stats": stats,
        "trades": [
            {
                "symbol": t.symbol,
                "entry": t.entry,
                "stop": t.stop,
                "target": t.target,
                "exit": t.exit,
                "reason": t.reason,
                "pnl_pct": round(t.pnl_pct, 4),
                "depth_pct": round(t.depth_pct, 4),
                "risk_pct": round(t.risk_pct, 4),
            }
            for t in trades
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.parent / "hits.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] {meta['period']} symbols={meta['symbols']} entries={meta['entries']} "
        f"closed={stats['count']} win={stats['win_rate']:.1f}% avg={stats['avg_pct']:+.2f}% "
        f"sum={stats['sum_pct']:+.2f}% open={stats['open']}"
    )
    print("[reasons]", stats["by_reason"])
    write_html(path, rows, stats, funnel, meta)
    preview = write_preview(path)
    print(f"[html] {path}")
    print(f"[web] {preview}")
    return 0


def write_preview(index: Path) -> Path:
    index = index.resolve()
    rel = index.parent.relative_to(ROOT).as_posix()
    base = (
        "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
        f"cursor/binance-15m-ma25-retest-431e/{rel}/"
    )
    text = index.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    text = text.replace(
        "<title>幣安 15 分 破底站回 MA25 回測</title>",
        "<title>一個月 · 六條均線回測 MA25</title>",
    )
    out = index.with_name("all.html")
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
