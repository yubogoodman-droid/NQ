"""掃描結果 HTML 報告（含一分 K 棒圖，可放到 GitHub Pages）。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from tw.screener import ScanHit, ScanResult
from tw.signals import add_moving_averages

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    200: "#ce93d8",
}


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
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
    .page {{ max-width: 720px; margin: 0 auto; padding: 16px 12px 40px; }}
    h1 {{ font-size: 1.2rem; margin: 0 0 6px; }}
    .lead {{ color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0 0 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
    .chip {{
      background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
      padding: 4px 10px; font-size: 12px; color: var(--muted);
    }}
    .summary {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; margin-bottom: 14px; font-size: .9rem; line-height: 1.65;
      color: var(--muted);
    }}
    .summary .ok {{ color: var(--ok); font-weight: 700; font-size: 1.05rem; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 14px 10px 8px; margin: 0 0 14px; color: inherit;
    }}
    .top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding: 0 6px; }}
    .name {{ font-weight: 700; font-size: 1.05rem; }}
    .sym a {{ color: var(--muted); font-weight: 500; margin-left: 6px; font-size: .9rem; text-decoration: none; }}
    .price {{ color: var(--up); font-weight: 700; white-space: nowrap; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: .9rem; color: var(--muted); padding: 0 6px; }}
    .row b {{ color: var(--text); font-weight: 600; }}
    .chart {{ margin-top: 8px; }}
    .empty {{ color: var(--muted); }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>台股一分K · 剛站上 MA200</h1>
    <p class="lead">成交額前 100、濾掉 ETF 與股價 650 以上。一分K MA5&gt;MA10&gt;MA20，且這根收盤剛站上 MA200（前一根還沒）。K 棒為台股慣例：漲紅跌綠。</p>
    <div class="chips">
      <span class="chip">不含 ETF</span>
      <span class="chip">股價 &lt; 650</span>
      <span class="chip">MA5 &gt; 10 &gt; 20</span>
      <span class="chip">金叉 MA200</span>
    </div>
    <div class="summary">
      命中 <span class="ok">{len(result.hits)}</span> 檔<br/>
      掃描時間 {html.escape(scanned)}（台北）<br/>
      排行時間 {rank_time}<br/>
      前 100 名 → 濾掉股價 {result.price_dropped}、ETF {result.etf_dropped} → 掃描 {len(result.candidates)} 檔
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
    chart = build_k_chart(hit, index)
    return f"""
    <article class="card">
      <div class="top">
        <div class="name">{index}. {html.escape(s.name)}<span class="sym"><a href="{url}" target="_blank" rel="noopener">{html.escape(s.symbol)}</a></span></div>
        <div class="price">{s.price:.2f}{html.escape(chg)}</div>
      </div>
      <div class="row"><span>金叉時間</span><b>{ts}</b></div>
      <div class="row"><span>收盤 / MA200</span><b>{snap.close:.2f} &gt; {snap.ma200:.2f}</b></div>
      <div class="row"><span>MA5 / 10 / 20</span><b>{snap.ma5:.2f} &gt; {snap.ma10:.2f} &gt; {snap.ma20:.2f}</b></div>
      <div class="row"><span>成交額排名</span><b>#{s.rank} · {s.turnover/1e8:.2f} 億</b></div>
      <div class="chart">{chart}</div>
    </article>
"""


def build_k_chart(hit: ScanHit, index: int = 1) -> str:
    """一分 K 棒圖 + MA5/10/20/200，標記剛站上 MA200 的那根。"""
    if hit.frame is None or hit.frame.empty:
        return '<p class="empty">無 K 線資料</p>'
    work = add_moving_averages(hit.frame)
    window = _window_around(work, hit.snapshot.timestamp)
    if window.empty:
        return '<p class="empty">無 K 線資料</p>'

    times = [_naive(t) for t in window.index]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color="#ef5350",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
            name="K",
            showlegend=False,
        )
    )
    for period, color in MA_COLORS.items():
        col = f"ma{period}"
        if col not in window.columns or window[col].notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=window[col],
                mode="lines",
                name=f"MA{period}",
                line=dict(color=color, width=1.6 if period == 200 else 1.2),
                connectgaps=False,
            )
        )

    loc = window.index.get_indexer([hit.snapshot.timestamp], method="nearest")[0]
    if loc >= 0:
        fig.add_trace(
            go.Scatter(
                x=[times[loc]],
                y=[hit.snapshot.close],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=12, color="#7ee787"),
                text=["金叉"],
                textposition="top center",
                textfont=dict(size=10, color="#7ee787"),
                name="金叉",
                showlegend=False,
            )
        )
        fig.add_hline(
            y=hit.snapshot.ma200,
            line_dash="dot",
            line_color="#ce93d8",
            opacity=0.7,
        )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=44, r=8, t=28, b=28),
        title=dict(text=f"一分K  {hit.stock.name} {hit.snapshot.timestamp.strftime('%H:%M')}", x=0.01, font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat="%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _window_around(df: pd.DataFrame, ts: pd.Timestamp, before: int = 80, after: int = 25) -> pd.DataFrame:
    if df.empty:
        return df
    loc = df.index.get_indexer([ts], method="nearest")[0]
    if loc < 0:
        return df.iloc[-90:]
    start = max(0, int(loc) - before)
    end = min(len(df), int(loc) + after + 1)
    return df.iloc[start:end]


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo else t
