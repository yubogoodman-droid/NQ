#!/usr/bin/env python3
"""掃描幣安 USDT 永續 5 分 K，找近似 UAI 的 W 底，輸出手機版 HTML 圖。"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TPE = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "binance-w-bottom" / "index.html"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
FAPI = "https://www.binance.com/fapi/v1"
MA_PERIODS = (7, 14, 25, 99, 120, 200)
MA_COLORS = {
    7: "#f0c14b",
    14: "#79c0ff",
    25: "#f778ba",
    99: "#b392f0",
    120: "#3ddc97",
    200: "#ff7b72",
}

# 對齊 UAI 5m：急殺約 21%、較高第二底 +1.6%、間隔 13 根、頸線深度約 8.7%、兩根內突破
MIN_SEP = 8
MAX_SEP = 20
MAX_SYM_PCT = 0.035
MIN_DEPTH = 0.06
MAX_DEPTH = 0.14
PRIOR_DROP = 0.10
BREAKOUT_PAD = 0.003
BREAKOUT_WINDOW = 16
MIN_QUOTE_VOL = 1_000_000.0
MIN_LIKE_PCT = 58.0
UAI_REF_SCORE = 118.0
LOOKBACK_BARS = 1000
KEEP_HOURS = 48


@dataclass
class WHit:
    symbol: str
    status: str
    score: float
    bottom1: float
    bottom2: float
    neck: float
    depth_pct: float
    sym_pct: float
    sep_min: int
    target: float
    current: float
    vsneck_pct: float
    vs_target_pct: float
    volume24: float
    t1: int
    t2: int
    t_neck: int
    t_break: int | None
    i: int
    j: int
    neck_idx: int
    b: int | None
    volx: float | None
    dump_pct: float = 0.0
    breakout_bars: int | None = None
    ext_pct: float = 0.0
    like_pct: float = 0.0
    reference: bool = False


def get_json(path: str, params: dict[str, Any] | None = None, retries: int = 4) -> Any:
    url = FAPI + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last: Exception | None = None
    for n in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                body = resp.read()
                if body:
                    return json.loads(body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (2**n))
    raise RuntimeError(f"GET failed {url}: {last}")


def swing_lows(lows: list[float], lookback: int = 3) -> list[int]:
    n = len(lows)
    out: list[int] = []
    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] != min(window):
            continue
        left = min(lows[i - lookback : i])
        right = min(lows[i + 1 : i + lookback + 1])
        if lows[i] < left * 0.998 and lows[i] < right * 0.998:
            out.append(i)
    return out


def uai_like_score(
    *,
    sep: int,
    depth: float,
    hl: float,
    dump: float,
    breakout_bars: int | None,
    ext: float,
    now: float,
    volx: float | None,
    age_h: float,
) -> float:
    """UAI 模板相似度，約 118 分 = 100%。"""
    hl_pen = 0.0 if hl >= 0 else 40.0 * min(abs(hl) / 0.03, 1.0)
    bo_pen = 8.0 * abs((breakout_bars or 8) - 2) if breakout_bars is not None else 18.0
    return (
        100.0
        - 3.5 * abs(sep - 13)
        - 350.0 * abs(depth - 0.087)
        - 250.0 * abs(max(hl, 0.0) - 0.016)
        - hl_pen
        - 70.0 * abs(min(dump, 0.30) - 0.21)
        - bo_pen
        + (min(ext, 0.28) * 70.0 if breakout_bars is not None else 0.0)
        + min(volx or 0.0, 4.0) * 2.0
        - (0.0 if now > 0 else 12.0)
        - min(age_h, 18.0) * 0.6
    )


def like_pct_from_score(score: float) -> float:
    return max(0.0, min(100.0, 100.0 * score / UAI_REF_SCORE))


def rolling_mean(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    total = 0.0
    for i, val in enumerate(values):
        total += val
        if i >= period:
            total -= values[i - period]
        if i + 1 >= period:
            out[i] = total / period
    return out


def detect_w_bottoms(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    quote_vol: list[float],
    times: list[int],
    *,
    symbol: str = "",
    volume24: float = 0.0,
) -> list[WHit]:
    n = len(closes)
    if n < 60:
        return []
    mins = swing_lows(lows)
    hits: list[WHit] = []
    last_t = times[-1]
    for ai, i in enumerate(mins):
        for j in mins[ai + 1 :]:
            sep = j - i
            if sep < MIN_SEP:
                continue
            if sep > MAX_SEP:
                break
            l1, l2 = lows[i], lows[j]
            avg = (l1 + l2) / 2
            if avg <= 0:
                continue
            hl = l2 / l1 - 1
            if hl < -0.005:
                continue
            if abs(hl) > MAX_SYM_PCT:
                continue
            if min(lows[i : j + 1]) < min(l1, l2) * 0.985:
                continue
            neck_idx = max(range(i + 3, j - 2), key=lambda x: highs[x])
            neck = highs[neck_idx]
            depth = neck / avg - 1
            if not (MIN_DEPTH <= depth <= MAX_DEPTH):
                continue
            dump = max(closes[max(0, i - 18) : i]) / l1 - 1
            if dump < PRIOR_DROP:
                continue
            end = min(n, j + BREAKOUT_WINDOW + 1)
            b = next((x for x in range(j + 1, end) if closes[x] > neck * (1 + BREAKOUT_PAD)), None)
            current = closes[-1]
            now = current / neck - 1
            target = neck + (neck - avg)
            volx = None
            t_break = None
            ext = 0.0
            bo_bars = None
            if b is None:
                age_h = (last_t - times[j]) / 3_600_000
                if age_h > 12 or min(lows[j:]) < l2 * 0.98:
                    continue
                status = "待突破"
            else:
                if min(lows[j : b + 1]) < l2 * 0.98:
                    continue
                age_h = (last_t - times[b]) / 3_600_000
                if age_h > KEEP_HOURS:
                    continue
                base = quote_vol[max(0, b - 20) : b] or [1.0]
                volx = quote_vol[b] / (statistics.median(base) or 1.0)
                t_break = times[b]
                bo_bars = b - j
                ext = max(closes[b:]) / neck - 1
                if current >= neck:
                    status = "已延伸" if current >= target * 1.02 else "突破仍有效"
                elif current >= avg:
                    status = "跌回頸線下"
                else:
                    status = "形態失敗"
            raw = uai_like_score(
                sep=sep,
                depth=depth,
                hl=hl,
                dump=dump,
                breakout_bars=bo_bars,
                ext=ext,
                now=now,
                volx=volx,
                age_h=age_h,
            )
            like = like_pct_from_score(raw)
            hits.append(
                WHit(
                    symbol=symbol,
                    status=status,
                    score=raw,
                    bottom1=l1,
                    bottom2=l2,
                    neck=neck,
                    depth_pct=depth * 100,
                    sym_pct=abs(hl) * 100,
                    sep_min=sep * 5,
                    target=target,
                    current=current,
                    vsneck_pct=now * 100,
                    vs_target_pct=(current / target - 1) * 100,
                    volume24=volume24,
                    t1=times[i],
                    t2=times[j],
                    t_neck=times[neck_idx],
                    t_break=t_break,
                    i=i,
                    j=j,
                    neck_idx=neck_idx,
                    b=b,
                    volx=volx,
                    dump_pct=dump * 100,
                    breakout_bars=bo_bars,
                    ext_pct=ext * 100,
                    like_pct=like,
                )
            )
    return hits


def best_hit(hits: list[WHit]) -> WHit | None:
    if not hits:
        return None
    return max(hits, key=lambda h: (h.like_pct, h.score))


def parse_klines(raw: list[list[Any]]) -> tuple[list[float], ...]:
    opens = [float(x[1]) for x in raw]
    highs = [float(x[2]) for x in raw]
    lows = [float(x[3]) for x in raw]
    closes = [float(x[4]) for x in raw]
    quote = [float(x[7]) for x in raw]
    times = [int(x[0]) for x in raw]
    return opens, highs, lows, closes, quote, times


def fmt_price(value: float) -> str:
    if value >= 100:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.01:
        return f"{value:.5f}"
    return f"{value:.6f}"


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TPE).strftime("%m-%d %H:%M")


def fmt_vol(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value / 1e3:.0f}K"


def _setup_cjk_font() -> None:
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
            return


def chart_window(hit: WHit, n: int) -> tuple[int, int]:
    left = max(0, hit.i - 24)
    right = min(n, max((hit.b or hit.j) + 48, hit.j + 28))
    return left, right


def draw_hit_png(hit: WHit, ohlcv: tuple[list[float], ...], path: Path, title: str) -> Path:
    """靜態 K 線，不依賴 Plotly CDN，預覽頁也能直接看到圖。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _setup_cjk_font()
    opens, highs, lows, closes, quote, times = ohlcv
    left, right = chart_window(hit, len(closes))
    w_o, w_h, w_l, w_c = opens[left:right], highs[left:right], lows[left:right], closes[left:right]
    w_v, w_t = quote[left:right], times[left:right]
    xs = list(range(len(w_c)))

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
    for k in xs:
        up = w_c[k] >= w_o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(k, w_l[k], w_h[k], color=col, lw=0.65)
        y0, y1 = min(w_o[k], w_c[k]), max(w_o[k], w_c[k])
        if y1 == y0:
            y1 = y0 + max(w_h[k] - w_l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((k - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, w_v, width=0.8, color=colors_v, linewidth=0)

    for period in MA_PERIODS:
        ma = rolling_mean(closes, period)[left:right]
        ys = [v if v is not None else float("nan") for v in ma]
        if all(v is None for v in ma):
            continue
        ax.plot(xs, ys, color=MA_COLORS[period], lw=1.35 if period <= 25 else 1.05, label=f"MA{period}")

    i_rel, j_rel = hit.i - left, hit.j - left
    neck_rel = hit.neck_idx - left
    b_rel = (hit.b - left) if hit.b is not None else None
    neck_end = b_rel if b_rel is not None else len(xs) - 1
    if 0 <= neck_rel < len(xs) and 0 <= neck_end < len(xs):
        ax.hlines(hit.neck, neck_rel, neck_end, colors="#ffa726", linestyles="--", lw=1.15, alpha=0.95)
    ax.axhline(hit.neck, color="#ffa726", ls=":", lw=0.8, alpha=0.45)
    ax.axhline(hit.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(min(hit.bottom1, hit.bottom2), color="#e35d5d", ls=":", lw=0.8, alpha=0.55)

    if 0 <= i_rel < len(xs):
        ax.scatter([i_rel], [hit.bottom1], s=42, color="#42a5f5", zorder=5)
        ax.annotate("L1", (i_rel, hit.bottom1), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= j_rel < len(xs):
        ax.scatter([j_rel], [hit.bottom2], s=42, color="#ec407a", zorder=5)
        ax.annotate("L2", (j_rel, hit.bottom2), textcoords="offset points", xytext=(0, -13),
                    ha="center", color="#f9a8d4", fontsize=8)
    if b_rel is not None and 0 <= b_rel < len(xs):
        ax.axvline(b_rel, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([b_rel], [w_c[b_rel]], s=48, color="#00e676", marker="^", zorder=6)
        ax.annotate("突破", (b_rel, w_c[b_rel]), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#3ddc68", fontsize=8)

    y_min, y_max = min(w_l), max(w_h)
    pad = max((y_max - y_min) * 0.08, y_min * 0.01)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(title, color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(xs) // 6)
    ticks = list(range(0, len(xs), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([fmt_time(w_t[i]) for i in ticks], color="#8aa193", rotation=20, ha="right")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def png_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def chart_payload(hit: WHit, ohlcv: tuple[list[float], ...]) -> dict[str, Any]:
    n = len(ohlcv[3])
    left, right = chart_window(hit, n)
    return {
        **asdict(hit),
        "t1_label": fmt_time(hit.t1),
        "t2_label": fmt_time(hit.t2),
        "t_neck_label": fmt_time(hit.t_neck),
        "t_break_label": fmt_time(hit.t_break) if hit.t_break else None,
        "i_rel": hit.i - left,
        "j_rel": hit.j - left,
        "neck_rel": hit.neck_idx - left,
        "b_rel": (hit.b - left) if hit.b is not None else None,
        "left": left,
        "right": right,
    }


STATUS_CLASS = {
    "待突破": "tag-wait",
    "突破仍有效": "tag-ok",
    "已延伸": "tag-hot",
    "跌回頸線下": "tag-warn",
    "形態失敗": "tag-sl",
    "基準": "tag-info",
}


def build_html(payload: dict[str, Any]) -> str:
    hits = payload["hits"]
    cards = []
    for idx, hit in enumerate(hits, 1):
        status = "基準" if hit.get("reference") else hit["status"]
        tag = STATUS_CLASS.get(status, "tag-info")
        vs = hit["vsneck_pct"]
        detail = (
            f"像 UAI {hit['like_pct']:.0f}%\n"
            f"L1 {fmt_price(hit['bottom1'])} @ {hit['t1_label']}\n"
            f"L2 {fmt_price(hit['bottom2'])} @ {hit['t2_label']}\n"
            f"頸線 {fmt_price(hit['neck'])} @ {hit['t_neck_label']}\n"
            f"量度目標 {fmt_price(hit['target'])}\n"
            f"急殺 {hit.get('dump_pct', 0):.1f}% · 較高低 {hit['sym_pct']:.2f}% · 間隔 {hit['sep_min']} 分 · 深度 {hit['depth_pct']:.1f}%\n"
            f"現價 {fmt_price(hit['current'])}  距頸線 {vs:+.2f}%"
        )
        if hit.get("t_break_label"):
            extra = f"突破 {hit['t_break_label']}"
            if hit.get("volx"):
                extra += f" · 量能 {hit['volx']:.1f}x"
            detail += f"\n{extra}"
        img = hit.get("img_href") or hit.get("img_src") or ""
        like = hit.get("like_pct", 0.0)
        like_cls = "pnl-win" if like >= 70 else ("pnl-loss" if like < 50 else "")
        cards.append(
            f"""
    <article class="trade-card" data-status="{escape(hit['status'])}">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{idx} · {escape(hit['symbol'])}</span>
          <span class="trade-time">L1 {escape(hit['t1_label'])} → L2 {escape(hit['t2_label'])} 台北</span>
        </div>
        <div class="card-pnl {like_cls}">像UAI {like:.0f}%</div>
      </header>
      <div class="tags">
        <span class="tag {tag}">{escape(status)}</span>
        <span class="tag tag-info">5m</span>
        <span class="tag tag-info">24h {escape(fmt_vol(hit['volume24']))}</span>
      </div>
      <pre class="trade-detail">{escape(detail)}</pre>
      <div class="mini-chart"><img src="{escape(img)}" alt="{escape(hit['symbol'])} W底" loading="lazy" /></div>
    </article>"""
        )
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>幣安 5m W底 · 近兩天</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #0b0e11;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
    }}
    .page {{ max-width: 720px; margin: 0 auto; padding: 12px 12px 32px; }}
    .summary {{
      background: #161b22; border: 1px solid #30363d; border-radius: 14px;
      padding: 14px 16px; margin-bottom: 14px;
    }}
    h1 {{ font-size: 18px; margin: 0 0 6px; }}
    .muted {{ color: #8b949e; font-size: 13px; line-height: 1.55; margin: 0; }}
    .cards {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 0; }}
    .stat {{ background: #0d1117; padding: 10px 12px; border-radius: 10px; min-width: 92px; border: 1px solid #21262d; }}
    .stat b {{ display: block; font-size: 20px; margin-top: 4px; }}
    .trade-card {{
      background: #161b22; border: 1px solid #30363d; border-radius: 14px;
      padding: 14px 14px 8px; margin-bottom: 14px;
    }}
    .card-header {{ display: flex; justify-content: space-between; gap: 10px; }}
    .trade-no {{ font-weight: 700; }}
    .trade-time {{ font-size: 12px; color: #8b949e; }}
    .card-pnl {{ font-weight: 700; white-space: nowrap; }}
    .pnl-win {{ color: #00c805; }} .pnl-loss {{ color: #ff5252; }}
    .tags {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
    .tag {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 999px; border: 1px solid #30363d; }}
    .tag-info {{ background: rgba(88,166,255,0.12); color: #79c0ff; border-color: rgba(88,166,255,0.28); }}
    .tag-ok {{ background: rgba(0,200,5,0.15); color: #3ddc68; border-color: rgba(0,200,5,0.35); }}
    .tag-wait {{ background: rgba(255,193,7,0.12); color: #f0c14b; border-color: rgba(255,193,7,0.3); }}
    .tag-hot {{ background: rgba(255,122,69,0.14); color: #ff9b6a; border-color: rgba(255,122,69,0.35); }}
    .tag-warn {{ background: rgba(240,193,75,0.12); color: #f0c14b; }}
    .tag-sl {{ background: rgba(255,82,82,0.15); color: #ff7b72; border-color: rgba(255,82,82,0.35); }}
    .trade-detail {{
      margin: 0 0 8px; padding: 10px 12px; background: #0d1117; border-radius: 10px;
      border: 1px solid #21262d; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px; line-height: 1.55; color: #c9d1d9; white-space: pre-wrap;
    }}
    .mini-chart {{ margin: 0 -6px -4px; }}
    .mini-chart img {{ display: block; width: 100%; height: auto; border-radius: 8px; }}
    .note {{ color: #8b949e; font-size: 12px; margin-top: 8px; line-height: 1.5; }}
  </style>
</head>
<body>
<div class="page">
  <section class="summary">
    <h1>幣安 USDT 永續 · 5m W底 · 近兩天</h1>
    <p class="muted">只留長得像 UAI 的：急殺 ≥10%、較高第二底、間隔 40–100 分、頸線深度 6–14%，最好兩根內放量突破。相似度用 UAI 當 100%。時間台北。截至 {escape(payload['asof'])}。僅供型態對照，不是進出場建議。</p>
    <div class="cards">
      <div class="stat">掃描<b>{payload['universe']}</b></div>
      <div class="stat">像UAI<b>{payload['matched']}</b></div>
      <div class="stat">待突破<b>{payload['pending']}</b></div>
      <div class="stat">已突破<b>{payload['valid']}</b></div>
    </div>
    <p class="note">黃虛線頸線、綠虛線量度目標。4USDT / BULLA 那種寬底或還在殺的，相似度不夠，已拿掉。</p>
  </section>
  {"".join(cards)}
</div>
</body>
</html>
"""


def universe_symbols(min_quote_vol: float = MIN_QUOTE_VOL) -> tuple[list[str], dict[str, float]]:
    info = get_json("/exchangeInfo")
    ticks = get_json("/ticker/24hr")
    tv = {x["symbol"]: float(x["quoteVolume"]) for x in ticks}
    symbols = [
        s["symbol"]
        for s in info["symbols"]
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and tv.get(s["symbol"], 0) >= min_quote_vol
    ]
    return symbols, tv


def fetch_klines(symbol: str) -> tuple[str, list[list[Any]] | None]:
    try:
        return symbol, get_json("/klines", {"symbol": symbol, "interval": "5m", "limit": LOOKBACK_BARS})
    except Exception:  # noqa: BLE001
        return symbol, None


def status_rank(hit: WHit) -> tuple[int, float]:
    return (0 if hit.reference else 1, -hit.like_pct)


def scan(min_quote_vol: float = MIN_QUOTE_VOL) -> tuple[dict[str, Any], dict[str, tuple[list[float], ...]]]:
    symbols, tv = universe_symbols(min_quote_vol)
    with ThreadPoolExecutor(max_workers=16) as pool:
        raw = dict(pool.map(fetch_klines, symbols))
    hits: list[WHit] = []
    series: dict[str, tuple[list[float], ...]] = {}
    for symbol, klines in raw.items():
        if not klines or len(klines) < 120:
            continue
        closed = klines[:-1]
        ohlcv = parse_klines(closed)
        series[symbol] = ohlcv
        found = detect_w_bottoms(*ohlcv, symbol=symbol, volume24=tv.get(symbol, 0.0))
        best = best_hit(found)
        if best:
            if symbol == "UAIUSDT":
                best.reference = True
            if not best.reference and best.like_pct < MIN_LIKE_PCT:
                continue
            if not best.reference and best.status in {"跌回頸線下", "形態失敗"}:
                continue
            hits.append(best)
    if "UAIUSDT" in series and not any(h.symbol == "UAIUSDT" for h in hits):
        found = detect_w_bottoms(*series["UAIUSDT"], symbol="UAIUSDT", volume24=tv.get("UAIUSDT", 0.0))
        if found:
            best = max(found, key=lambda h: h.score)
            best.reference = True
            hits.append(best)
    hits.sort(key=status_rank)
    asof = datetime.now(TPE).strftime("%Y-%m-%d %H:%M")
    if series:
        last_ms = max(v[5][-1] for v in series.values() if v[5])
        asof = datetime.fromtimestamp(last_ms / 1000, TPE).strftime("%Y-%m-%d %H:%M")
    payload_hits = [chart_payload(h, series[h.symbol]) for h in hits if h.symbol in series]
    payload = {
        "asof": asof,
        "universe": len(symbols),
        "matched": len(hits),
        "pending": sum(1 for h in hits if h.status == "待突破"),
        "valid": sum(1 for h in hits if h.status in {"突破仍有效", "已延伸"}),
        "hits": payload_hits,
    }
    return payload, series


def write_report(payload: dict[str, Any], series: dict[str, tuple[list[float], ...]], out: Path) -> None:
    img_dir = out.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    for idx, hit in enumerate(payload["hits"], 1):
        symbol = hit["symbol"]
        wh = WHit(**{k: hit[k] for k in WHit.__dataclass_fields__})
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol)
        png = img_dir / f"w{idx:02d}_{safe}.png"
        status = "基準" if hit.get("reference") else hit["status"]
        title = f"#{idx}  {symbol}  5m  像UAI {hit.get('like_pct', 0):.0f}%  {status}"
        draw_hit_png(wh, series[symbol], png, title)
        hit["img_src"] = f"img/{png.name}"
        hit["img_href"] = png_data_uri(png)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload), encoding="utf-8")
    json_path = out.with_name("hits.json")
    slim = {
        **payload,
        "hits": [
            {k: v for k, v in h.items() if k not in {"img_href"}}
            for h in payload["hits"]
        ],
    }
    json_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="掃描幣安 5m W底並輸出 HTML")
    parser.add_argument("--output", "-o", default=str(PAGES))
    parser.add_argument("--min-quote-vol", type=float, default=MIN_QUOTE_VOL)
    args = parser.parse_args()
    payload, series = scan(args.min_quote_vol)
    out = Path(args.output)
    write_report(payload, series, out)
    print(f"已產生: {out}")
    print(f"掃描 {payload['universe']} · 命中 {payload['matched']} · 截至 {payload['asof']}")
    for hit in payload["hits"]:
        print(f"  {hit['symbol']:16} {hit['status']:8} 像UAI {hit.get('like_pct', 0):5.1f}% 頸線{fmt_price(hit['neck'])} 現價{fmt_price(hit['current'])} {hit['vsneck_pct']:+.1f}%")


if __name__ == "__main__":
    main()
