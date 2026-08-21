"""五分 K 回測報告：PNG 圖 + HTML。"""

from __future__ import annotations

import html
import io
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from tw.backtest_5m import BacktestHit, BacktestResult
from tw.signals import add_moving_averages

WEEKDAY_ZH = "一二三四五六日"
MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    200: "#ce93d8",
}
CHART_BARS = (36, 8)
UP = "#ef5350"
DOWN = "#26a69a"
BG = "#161b22"
FG = "#e6edf3"
GRID = "#30363d"

_FONT = None
for _path in (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(_path).exists():
        _FONT = fm.FontProperties(fname=_path)
        break


def weekday_zh(d) -> str:
    return f"週{WEEKDAY_ZH[d.weekday()]}"


def save_backtest_html(result: BacktestResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    chart_rel = Path("charts") / out.stem
    chart_dir = out.parent / chart_rel
    if chart_dir.exists():
        shutil.rmtree(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    image_base = _github_raw_base(out.parent)
    out.write_text(
        _render(result, chart_rel=chart_rel, chart_dir=chart_dir, image_base=image_base),
        encoding="utf-8",
    )
    return out


def _render(
    result: BacktestResult,
    *,
    chart_rel: Path,
    chart_dir: Path,
    image_base: str | None,
) -> str:
    scanned = result.scanned_at.strftime("%Y-%m-%d %H:%M:%S")
    start, end = result.days[0], result.days[-1]
    title = f"台股五分K回測 {start.isoformat()}～{end.isoformat()}"
    day_chips = "".join(
        f'<span class="chip">{weekday_zh(day)} {day.isoformat()} · {len(result.hits_on(day))} 則</span>'
        for day in result.days
    )
    sections = []
    card_no = 0
    for day in result.days:
        uni = result.universes[day]
        day_hits = result.hits_on(day)
        cards = []
        for hit in day_hits:
            card_no += 1
            cards.append(
                _hit_card(
                    card_no,
                    hit,
                    chart_rel=chart_rel,
                    chart_dir=chart_dir,
                    image_base=image_base,
                )
            )
        body = "\n".join(cards) or '<p class="empty">這天沒有符合條件的通知。</p>'
        sections.append(
            f"""
    <section class="day">
      <h2>{weekday_zh(day)} {day.isoformat()}</h2>
      <p class="lead">成交額前 {len(uni.universe)} → 股價濾掉 {uni.price_dropped}、ETF {uni.etf_dropped}、金融 {uni.financial_dropped} → 掃描 {len(uni.candidates)} → 通知 {len(day_hits)} 則</p>
      {body}
    </section>"""
        )
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0e11;
      --card: #161b22;
      --line: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --up: #ff5c7a;
      --ok: #7ee787;
      --chip: #21262d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{ max-width: 760px; margin: 0 auto; padding: 16px 12px 40px; }}
    h1 {{ font-size: 1.2rem; margin: 0 0 6px; }}
    h2 {{ font-size: 1.05rem; margin: 22px 0 8px; }}
    .lead {{ color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0 0 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
    .chip {{
      background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
      padding: 4px 10px; font-size: 12px; color: var(--muted);
    }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted);
      margin: 0 0 12px;
    }}
    .swatch {{ display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }}
    .summary {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; margin-bottom: 14px; font-size: .9rem; line-height: 1.65;
      color: var(--muted);
    }}
    .summary .ok {{ color: var(--ok); font-weight: 700; font-size: 1.05rem; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 14px 10px 10px; margin: 0 0 14px; color: inherit;
    }}
    .top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding: 0 6px; }}
    .name {{ font-weight: 700; font-size: 1.05rem; }}
    .sym a {{ color: var(--muted); font-weight: 500; margin-left: 6px; font-size: .9rem; text-decoration: none; }}
    .price {{ color: var(--up); font-weight: 700; white-space: nowrap; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: .9rem; color: var(--muted); padding: 0 6px; }}
    .row b {{ color: var(--text); font-weight: 600; }}
    .chart {{ margin-top: 10px; }}
    .chart img {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
    .empty {{ color: var(--muted); }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p class="lead">
      同一套台股掃描池：每天上市＋上櫃成交額前 100，濾掉 ETF、金融股與收盤價 650 以上。
      五分K MA5 &gt; MA10 &gt; MA20 多頭排列，<strong>當根收盤剛站上五分 MA200</strong>（前一根尚未站上），
      且這根收盤必須高於 MA5／10／20／200（收在所有均線上面）。開盤第一根因隔夜跳空不算。K 棒漲紅跌綠。
    </p>
    <div class="chips">
      <span class="chip">五分K</span>
      <span class="chip">不含 ETF</span>
      <span class="chip">不含金融股</span>
      <span class="chip">股價 &lt; 650</span>
      <span class="chip">MA5 &gt; 10 &gt; 20</span>
      <span class="chip">當根收盤站上 MA200</span>
      <span class="chip">收盤 &gt; 所有均線</span>
      {day_chips}
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#ffa726"></i>MA5</span>
      <span><i class="swatch" style="background:#ffeb3b"></i>MA10</span>
      <span><i class="swatch" style="background:#66bb6a"></i>MA20</span>
      <span><i class="swatch" style="background:#ce93d8"></i>MA200</span>
    </div>
    <div class="summary">
      五天共通知 <span class="ok">{len(result.hits)}</span> 則<br/>
      掃描時間 {html.escape(scanned)}（台北）<br/>
      資料：證交所／櫃買盤後成交額 ＋ Yahoo 五分K
    </div>
    {''.join(sections)}
    <footer>僅供研究，不構成投資建議。代號可開 Yahoo 報價。</footer>
  </div>
</body>
</html>
"""


def _hit_card(
    index: int,
    hit: BacktestHit,
    *,
    chart_rel: Path,
    chart_dir: Path,
    image_base: str | None,
) -> str:
    s = hit.stock
    snap = hit.snapshot
    chg = ""
    if s.change_percent is not None:
        chg = f" {s.change_percent:+.2f}%"
    ts = snap.timestamp.strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{html.escape(s.symbol)}"
    chart = _chart_img(hit, chart_rel=chart_rel, chart_dir=chart_dir, image_base=image_base)
    return f"""
    <article class="card">
      <div class="top">
        <div class="name">{index}. {html.escape(s.name)}<span class="sym"><a href="{url}" target="_blank" rel="noopener">{html.escape(s.symbol)}</a></span></div>
        <div class="price">{s.price:.2f}{html.escape(chg)}</div>
      </div>
      <div class="row"><span>五分站上時間</span><b>{ts}</b></div>
      <div class="row"><span>收盤 / MA200</span><b>{snap.close:.2f} &gt; {snap.ma200:.2f}</b></div>
      <div class="row"><span>收盤 vs 均線</span><b>{snap.close:.2f} &gt; MA5 {snap.ma5:.2f} / 10 {snap.ma10:.2f} / 20 {snap.ma20:.2f}</b></div>
      <div class="row"><span>前收 / 前MA200</span><b>{snap.prev_close:.2f} ≤ {snap.prev_ma200:.2f}</b></div>
      <div class="row"><span>成交額排名</span><b>#{s.rank} · {s.turnover/1e8:.2f} 億</b></div>
      <div class="chart">{chart}</div>
    </article>
"""


def _chart_img(
    hit: BacktestHit,
    *,
    chart_rel: Path,
    chart_dir: Path,
    image_base: str | None,
) -> str:
    png = render_k_chart_png(hit)
    if not png:
        return '<p class="empty">無五分 K 線資料</p>'
    stamp = hit.snapshot.timestamp.strftime("%H%M")
    fname = f"{hit.day.isoformat()}-{_safe_symbol(hit.stock.symbol)}-{stamp}.png"
    (chart_dir / fname).write_bytes(png)
    rel = f"{chart_rel.as_posix()}/{fname}"
    src = html.escape(f"{image_base}{rel}" if image_base else rel)
    alt = html.escape(f"{hit.stock.name} {hit.stock.symbol} 五分K")
    return f'<img alt="{alt}" src="{src}"/>'


def render_k_chart_png(hit: BacktestHit) -> bytes | None:
    if hit.frame is None or hit.frame.empty:
        return None
    work = add_moving_averages(hit.frame)
    mark_ts = pd.Timestamp(hit.snapshot.timestamp).floor("5min")
    if work.index.tz is not None:
        mark_ts = (
            mark_ts.tz_convert(work.index.tz)
            if mark_ts.tzinfo
            else mark_ts.tz_localize(work.index.tz)
        )
    work = work[work.index < mark_ts.normalize() + pd.Timedelta(days=1)]
    window = _window_around(work, mark_ts, before=CHART_BARS[0], after=CHART_BARS[1])
    if window.empty:
        return None

    n = len(window)
    xs = list(range(n))
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    body_w = 0.7
    for i, (_, row) in enumerate(window.iterrows()):
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = UP if c >= o else DOWN
        ax.vlines(i, l, h, color=color, linewidth=0.9, zorder=2)
        body = abs(c - o)
        if body < 1e-9:
            body = max((h - l) * 0.04, 1e-6)
        ax.add_patch(
            Rectangle(
                (i - body_w / 2, min(o, c)),
                body_w,
                body,
                facecolor=color,
                edgecolor=color,
                linewidth=0.4,
                zorder=3,
            )
        )

    for period, color in MA_COLORS.items():
        col = f"ma{period}"
        if col not in window.columns:
            continue
        vals = window[col].astype(float)
        if vals.notna().sum() == 0:
            continue
        ax.plot(
            xs,
            vals,
            color=color,
            linewidth=2.0 if period == 200 else 1.35,
            label=f"MA{period}",
            zorder=4,
            solid_capstyle="round",
        )

    loc = int(window.index.get_indexer([mark_ts], method="nearest")[0])
    if 0 <= loc < n:
        ax.scatter(
            [loc],
            [float(window["close"].iloc[loc])],
            marker="^",
            s=70,
            color="#7ee787",
            zorder=5,
            label="站上MA200",
        )

    ref = float(hit.snapshot.close) if hit.snapshot.close else float(window["close"].iloc[-1])
    ax.set_ylim(*_axis_ylim(window, ref, min_span_pct=0.03))
    ax.set_xlim(-1, n)
    bar_time = (
        pd.Timestamp(window.index[loc]).strftime("%H:%M")
        if 0 <= loc < n
        else hit.snapshot.timestamp.strftime("%H:%M")
    )
    title = f"{hit.stock.name} {hit.stock.symbol}  五分K  {bar_time}"
    ax.set_title(title, color=FG, fontsize=11, pad=8, fontproperties=_FONT)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=FG,
        ncol=5,
        prop=_FONT,
    )
    step = max(1, n // 6)
    ticks = list(range(0, n, step))
    if n - 1 not in ticks:
        ticks.append(n - 1)
    labels = [pd.Timestamp(window.index[i]).strftime("%H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def _axis_ylim(
    window: pd.DataFrame,
    ref_price: float,
    min_span_pct: float = 0.03,
) -> tuple[float, float]:
    lows: list[float] = [float(window["low"].min())]
    highs: list[float] = [float(window["high"].max())]
    for col in ("ma5", "ma10", "ma20", "ma200"):
        if col not in window.columns:
            continue
        series = window[col].astype(float)
        if series.notna().any():
            lows.append(float(series.min()))
            highs.append(float(series.max()))
    lo, hi = min(lows), max(highs)
    if not (lo < hi):
        pad = abs(ref_price) * 0.01 or 1.0
        return lo - pad, hi + pad
    span = hi - lo
    min_span = abs(ref_price) * min_span_pct if ref_price else 0.0
    if span < min_span:
        mid = (lo + hi) / 2.0
        lo = mid - min_span / 2.0
        hi = mid + min_span / 2.0
        span = min_span
    pad = span * 0.04
    return lo - pad, hi + pad


def _window_around(df: pd.DataFrame, ts: pd.Timestamp, before: int = 36, after: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    loc = df.index.get_indexer([ts], method="nearest")[0]
    if loc < 0:
        return df.iloc[-44:]
    start = max(0, int(loc) - before)
    end = min(len(df), int(loc) + after + 1)
    return df.iloc[start:end]


def _safe_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in symbol)


def _github_raw_base(page_dir: Path) -> str | None:
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=page_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    parsed = _parse_github_owner_repo(remote)
    if parsed is None or not branch or branch == "HEAD":
        return None
    owner, repo = parsed
    rel = page_dir.resolve().relative_to(root).as_posix().strip("/")
    prefix = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
    return f"{prefix}{rel}/" if rel != "." else prefix


def _parse_github_owner_repo(remote: str) -> tuple[str, str] | None:
    text = re.sub(r"^git@", "https://", remote)
    text = text.replace("github.com:", "github.com/")
    match = re.search(r"github.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\\.git)?$", text)
    if not match:
        return None
    return match.group("owner"), match.group("repo")
