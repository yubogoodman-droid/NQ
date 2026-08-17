"""掃描結果 HTML 報告（K 棒圖以 PNG 內嵌，免靠 JavaScript）。"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from tw.screener import ScanHit, ScanResult
from tw.signals import add_moving_averages, ma200_at

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    200: "#ce93d8",
}
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


def save_scan_html(result: ScanResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(result), encoding="utf-8")
    return out


def _render(result: ScanResult) -> str:
    scanned = result.scanned_at.strftime("%Y-%m-%d %H:%M:%S")
    rank_time = html.escape(result.rank_time or "—")
    hit_rows = "\n".join(_hit_card(i, h) for i, h in enumerate(result.hits, 1)) or (
        '<p class="empty">目前沒有符合條件的個股。</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>台股一分K · 多頭排列站上 MA200</title>
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
    .chart-label {{
      font-size: 12px; color: var(--ok); font-weight: 700;
      padding: 8px 6px 4px;
    }}
    .empty {{ color: var(--muted); }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>台股一分K · 剛站上 MA200</h1>
    <p class="lead">成交額前 100、濾掉 ETF 與股價 650 以上。一分K MA5&gt;MA10&gt;MA20，且這根收盤剛站上 MA200（前一根還沒）。含該金叉的五分K收盤也必須高於五分 MA200。K 棒漲紅跌綠。</p>
    <div class="chips">
      <span class="chip">不含 ETF</span>
      <span class="chip">股價 &lt; 650</span>
      <span class="chip">MA5 &gt; 10 &gt; 20</span>
      <span class="chip">金叉 MA200</span>
      <span class="chip">五分收盤 &gt; MA200</span>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#ffa726"></i>MA5</span>
      <span><i class="swatch" style="background:#ffeb3b"></i>MA10</span>
      <span><i class="swatch" style="background:#66bb6a"></i>MA20</span>
      <span><i class="swatch" style="background:#ce93d8"></i>MA200</span>
    </div>
    <div class="summary">
      命中 <span class="ok">{len(result.hits)}</span> 檔<br/>
      掃描時間 {html.escape(scanned)}（台北）<br/>
      排行時間 {rank_time}<br/>
      前 100 名 → 濾掉股價 {result.price_dropped}、ETF {result.etf_dropped} → 掃描 {len(result.candidates)} 檔 → 五分MA200底下 {result.below_5m_dropped}
    </div>
    {hit_rows}
    <footer>僅供研究，不構成投資建議。代號可開 Yahoo 報價。</footer>
  </div>
</body>
</html>
"""


def _hit_card(index: int, hit: ScanHit) -> str:
    s = hit.stock
    snap = hit.snapshot
    chg = ""
    if s.change_percent is not None:
        chg = f" {s.change_percent:+.2f}%"
    ts = snap.timestamp.strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{html.escape(s.symbol)}"
    chart_1m = build_k_chart(hit, timeframe="1m")
    chart_5m = build_k_chart(hit, timeframe="5m")
    return f"""
    <article class="card">
      <div class="top">
        <div class="name">{index}. {html.escape(s.name)}<span class="sym"><a href="{url}" target="_blank" rel="noopener">{html.escape(s.symbol)}</a></span></div>
        <div class="price">{s.price:.2f}{html.escape(chg)}</div>
      </div>
      <div class="row"><span>1分金叉時間</span><b>{ts}</b></div>
      <div class="row"><span>1分收盤 / MA200</span><b>{snap.close:.2f} &gt; {snap.ma200:.2f}</b></div>
      <div class="row"><span>五分收盤 / MA200</span><b>{html.escape(_five_min_ma200_text(hit))}</b></div>
      <div class="row"><span>1分 MA5 / 10 / 20</span><b>{snap.ma5:.2f} &gt; {snap.ma10:.2f} &gt; {snap.ma20:.2f}</b></div>
      <div class="row"><span>成交額排名</span><b>#{s.rank} · {s.turnover/1e8:.2f} 億</b></div>
      <div class="chart-label">一分 K</div>
      <div class="chart">{chart_1m}</div>
      <div class="chart-label">五分 K（對照）</div>
      <div class="chart">{chart_5m}</div>
    </article>
"""


def _five_min_ma200_text(hit: ScanHit) -> str:
    pair = ma200_at(hit.frame_5m, hit.snapshot.timestamp, floor="5min")
    if pair is None:
        return "—"
    close, ma200 = pair
    return f"{close:.2f} > {ma200:.2f}"


def build_k_chart(hit: ScanHit, timeframe: str = "1m") -> str:
    """K 棒圖 + MA5/10/20/200，輸出成 PNG。timeframe: 1m 或 5m。"""
    png = render_k_chart_png(hit, timeframe=timeframe)
    if not png:
        label = "五分" if timeframe == "5m" else "一分"
        return f'<p class="empty">無{label} K 線資料</p>'
    b64 = base64.b64encode(png).decode("ascii")
    alt = html.escape(f"{hit.stock.name} {hit.stock.symbol} {'五分K' if timeframe == '5m' else '一分K'}")
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


def render_k_chart_png(hit: ScanHit, timeframe: str = "1m") -> bytes | None:
    frame = hit.frame_5m if timeframe == "5m" else hit.frame
    if frame is None or frame.empty:
        return None
    work = add_moving_averages(frame)
    before, after = (70, 12) if timeframe == "5m" else (90, 20)
    mark_ts = pd.Timestamp(hit.snapshot.timestamp)
    if timeframe == "5m":
        mark_ts = mark_ts.floor("5min")
    window = _window_around(work, mark_ts, before=before, after=after)
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
    tf_name = "五分K" if timeframe == "5m" else "一分K"
    marker_label = "1分金叉" if timeframe == "5m" else "金叉"
    if 0 <= loc < n:
        ax.scatter(
            [loc],
            [float(window["close"].iloc[loc])],
            marker="^",
            s=70,
            color="#7ee787",
            zorder=5,
            label=marker_label,
        )
        ma200_val = window["ma200"].iloc[loc] if "ma200" in window.columns else hit.snapshot.ma200
        if pd.notna(ma200_val):
            ax.axhline(float(ma200_val), color="#ce93d8", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1)

    ax.set_xlim(-1, n)
    bar_time = pd.Timestamp(window.index[loc]).strftime("%H:%M") if 0 <= loc < n else hit.snapshot.timestamp.strftime("%H:%M")
    title = f"{hit.stock.name} {hit.stock.symbol}  {tf_name}  {bar_time}"
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


def _window_around(df: pd.DataFrame, ts: pd.Timestamp, before: int = 90, after: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    loc = df.index.get_indexer([ts], method="nearest")[0]
    if loc < 0:
        return df.iloc[-110:]
    start = max(0, int(loc) - before)
    end = min(len(df), int(loc) + after + 1)
    return df.iloc[start:end]
