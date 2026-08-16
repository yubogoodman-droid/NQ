"""上週五假跌破訊號的 1 分 K 報告（卡片 + 圖）。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nq.backtest import TradeResult
from nq.spring import FakeBreakdownPattern
from nq.strategy import Signal

MA_PERIODS = (5, 10, 20, 60, 120, 200)
MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    60: "#42a5f5",
    120: "#26c6da",
    200: "#ab47bc",
}


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period in MA_PERIODS:
        out[f"ma{period}"] = out["close"].rolling(period, min_periods=period).mean()
    return out


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    t = ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts
    return t.replace(tzinfo=None)


def _fmt(ts: pd.Timestamp) -> str:
    return _naive(ts).strftime("%m-%d %H:%M")


def chart_window(df: pd.DataFrame, pattern: FakeBreakdownPattern, extra: int = 35) -> pd.DataFrame:
    pivot = pattern.breakout_idx or pattern.reclaim_idx
    day = _naive(df.index[pivot]).date()
    session = [i for i, t in enumerate(df.index) if _naive(t).date() == day]
    sess0, sess1 = session[0], session[-1]
    start = sess0 if pattern.range_start_idx < sess0 else max(sess0, pattern.range_start_idx - 8)
    end = min(sess1, pivot + extra)
    return df.iloc[start : end + 1]


def build_signal_figure(
    df: pd.DataFrame,
    signal: Signal,
    trade: TradeResult | None,
    *,
    title: str,
    height: int = 460,
) -> go.Figure:
    p = signal.pattern
    assert isinstance(p, FakeBreakdownPattern)
    window = chart_window(df, p)
    times = [_naive(t) for t in window.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
            name="K",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for period in MA_PERIODS:
        col = f"ma{period}"
        if col in window.columns and window[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=window[col],
                    mode="lines",
                    name=f"MA{period}",
                    line=dict(color=MA_COLORS[period], width=3 if period <= 20 else 2.4),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    box_start, box_end = _naive(df.index[p.range_start_idx]), _naive(df.index[p.range_end_idx])
    fig.add_trace(
        go.Scatter(
            x=[box_start, box_end, box_end, box_start],
            y=[p.support, p.support, p.resistance, p.resistance],
            fill="toself",
            fillcolor="rgba(66,165,245,0.14)",
            line=dict(color="#42a5f5", width=1, dash="dash"),
            name="盤整箱",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[_naive(df.index[p.spring_idx])],
            y=[p.spring_low],
            mode="markers+text",
            marker=dict(symbol="circle", size=10, color="#ff5252"),
            text=["假跌破"],
            textposition="bottom center",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[_naive(signal.timestamp)],
            y=[signal.entry],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=13, color="#00e676"),
            text=[f"進 {signal.entry:.2f}"],
            textposition="top center",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if trade is not None:
        fig.add_trace(
            go.Scatter(
                x=[_naive(trade.exit_time)],
                y=[trade.exit_price],
                mode="markers+text",
                marker=dict(
                    symbol="x",
                    size=11,
                    color="#69f0ae" if trade.pnl_points > 0 else "#ff5252",
                ),
                text=[f"{trade.exit_reason}"],
                textposition="bottom center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_hline(y=signal.stop_loss, line_dash="dot", line_color="#ff5252", opacity=0.55, row=1, col=1)
    fig.add_hline(y=signal.target, line_dash="dot", line_color="#00c805", opacity=0.55, row=1, col=1)

    vol_colors = ["#ef5350" if c >= o else "#26a69a" for o, c in zip(window["open"], window["close"])]
    fig.add_trace(
        go.Bar(x=times, y=window["volume"], marker_color=vol_colors, showlegend=False, name="量"),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=48, r=12, t=48, b=28),
        title=dict(text=title, x=0.02, font=dict(size=14)),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat="%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    return fig


def render_report_html(
    cards: list[tuple[str, pd.DataFrame, Signal, TradeResult | None]],
    *,
    title: str,
    summary: str,
) -> str:
    sections = []
    for i, (label, df, sig, trade) in enumerate(cards, 1):
        p = sig.pattern
        assert isinstance(p, FakeBreakdownPattern)
        fig = build_signal_figure(df, sig, trade, title=f"#{i} {label} · 1分K")
        chart = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
        pnl = ""
        tag = '<span class="tag tag-info">訊號</span>'
        if trade is not None:
            cls = "pnl-win" if trade.pnl_points > 0 else "pnl-loss"
            pnl = f'<div class="card-pnl {cls}">{trade.pnl_points:+.2f}</div>'
            reason = {"take_profit": ("TP", "tag-tp"), "stop_loss": ("SL", "tag-sl")}.get(
                trade.exit_reason, ("TIME", "tag-time")
            )
            tag = f'<span class="tag {reason[1]}">{reason[0]}</span>'
        detail = (
            f"進場 {_fmt(sig.timestamp)} @ {sig.entry:.2f}\n"
            f"停損 {sig.stop_loss:.2f} / 目標 {sig.target:.2f}\n"
            f"箱 {p.support:.2f}–{p.resistance:.2f} / 假跌破 {p.spring_low:.2f} ({p.break_pct * 100:.2f}%)\n"
            f"放量 {p.volume_ratio:.2f}x"
        )
        sections.append(
            f"""
    <article class="trade-card">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{i} {html.escape(label)}</span>
          <span class="trade-time">{_fmt(sig.timestamp)}</span>
        </div>
        {pnl}
      </header>
      <div class="tags">{tag}<span class="tag tag-info">1分K</span><span class="tag tag-info">假跌破</span></div>
      <pre class="trade-detail">{html.escape(detail)}</pre>
      <div class="mini-chart">{chart}</div>
    </article>"""
        )

    body = "\n".join(sections) or '<div class="empty">沒有訊號</div>'
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#0b0e11; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif; }}
    .page {{ max-width: 640px; margin: 0 auto; padding: 12px 12px 28px; }}
    .summary {{ background:#161b22; border:1px solid #30363d; border-radius:14px; padding:14px 16px; margin-bottom:14px; }}
    .summary h1 {{ margin:0 0 6px; font-size:18px; }}
    .summary p {{ margin:0; color:#8b949e; font-size:13px; line-height:1.55; }}
    .trade-card {{ background:#161b22; border:1px solid #30363d; border-radius:14px; padding:14px 14px 10px; margin-bottom:14px; overflow:hidden; }}
    .card-header {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:8px; }}
    .trade-no {{ font-size:15px; font-weight:700; }}
    .trade-time {{ font-size:12px; color:#8b949e; }}
    .card-pnl {{ font-size:16px; font-weight:700; }}
    .pnl-win {{ color:#00c805; }} .pnl-loss {{ color:#ff5252; }}
    .tags {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }}
    .tag {{ font-size:11px; font-weight:600; padding:3px 8px; border-radius:999px; border:1px solid transparent; }}
    .tag-tp {{ background:rgba(0,200,5,.15); color:#3ddc68; border-color:rgba(0,200,5,.35); }}
    .tag-sl {{ background:rgba(255,82,82,.15); color:#ff7b72; border-color:rgba(255,82,82,.35); }}
    .tag-time {{ background:rgba(255,193,7,.12); color:#f0c14b; border-color:rgba(255,193,7,.3); }}
    .tag-info {{ background:rgba(88,166,255,.12); color:#79c0ff; border-color:rgba(88,166,255,.28); }}
    .trade-detail {{ margin:0 0 10px; padding:10px 12px; background:#0d1117; border-radius:10px; border:1px solid #21262d; font-family:ui-monospace,Menlo,monospace; font-size:12px; line-height:1.55; color:#c9d1d9; white-space:pre-wrap; }}
    .mini-chart {{ margin:0 -6px -4px; border-radius:10px; overflow:hidden; }}
    .empty {{ text-align:center; color:#8b949e; padding:40px 16px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="summary">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(summary)}</p>
      <p style="margin-top:8px"><a href="https://raw.githack.com/yubogoodman-droid/NQ/cursor/fake-breakdown-spring-b99b/docs/spring_top50_20260814.html" style="color:#79c0ff">外網開啟</a></p>
    </section>
    {body}
  </div>
</body>
</html>
"""


def save_report(path: str | Path, html_text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out
