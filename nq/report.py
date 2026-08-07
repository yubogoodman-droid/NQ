"""手機版交易卡片 HTML 報告。"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go

from nq.backtest import TradeResult, run_backtest, summarize
from nq.strategy import NQWBottomStrategy

MA_PERIODS = (5, 10, 20, 60, 120, 200)
MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    60: "#42a5f5",
    120: "#26c6da",
    200: "#ab47bc",
}


def _fmt_time(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m-%d %H:%M")


def _exit_tag(reason: str, pnl: float) -> tuple[str, str]:
    if reason == "take_profit":
        return "TP", "tag-tp"
    if reason == "stop_loss":
        return "SL", "tag-sl"
    return "TIME", "tag-time"


def _filter_today_results(df: pd.DataFrame, results: list[TradeResult]) -> list[TradeResult]:
    if not len(df):
        return results
    today = df.index[-1]
    if hasattr(today, "tz_convert"):
        today = today.tz_convert("America/New_York")
    day = today.date()
    return [r for r in results if r.signal.timestamp.date() == day]


def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period in MA_PERIODS:
        out[f"ma{period}"] = out["close"].rolling(period, min_periods=period).mean()
    return out


def _ma_snapshot(row: pd.Series) -> str:
    short = []
    long_ = []
    for period in MA_PERIODS:
        val = row.get(f"ma{period}")
        if pd.notna(val):
            text = f"MA{period} {val:.2f}"
            if period <= 20:
                short.append(text)
            else:
                long_.append(text)
    lines = []
    if short:
        lines.append(" / ".join(short))
    if long_:
        lines.append(" / ".join(long_))
    return "\n".join(lines)


def _chart_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    p = trade.signal.pattern
    start = max(0, p.first_low_idx - 18)
    end = trade.signal.bar_idx + 28
    for i in range(trade.signal.bar_idx + 1, len(df)):
        if df.index[i] == trade.exit_time:
            end = min(len(df) - 1, i + 10)
            break
    end = min(len(df) - 1, end)
    return start, end


def _build_trade_chart(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    p = trade.signal.pattern
    start, end = _chart_window(df, trade)
    window = df.iloc[start : end + 1].copy()
    times = [t.tz_localize(None) if t.tzinfo else t for t in window.index]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="K",
            showlegend=False,
        )
    )

    for period in MA_PERIODS:
        col = f"ma{period}"
        if col not in window.columns:
            continue
        ma_vals = window[col]
        if ma_vals.notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ma_vals,
                mode="lines",
                name=f"MA{period}",
                line=dict(color=MA_COLORS[period], width=1.3 if period <= 20 else 1.1),
                connectgaps=False,
            )
        )

    if df.index[p.first_low_idx] in window.index:
        l1_t = window.index.get_loc(df.index[p.first_low_idx])
        fig.add_trace(
            go.Scatter(
                x=[times[l1_t]],
                y=[p.first_low],
                mode="markers+text",
                marker=dict(symbol="circle", size=8, color="#42a5f5"),
                text=["L1"],
                textposition="bottom center",
                showlegend=False,
            )
        )
    if df.index[p.second_low_idx] in window.index:
        l2_t = window.index.get_loc(df.index[p.second_low_idx])
        fig.add_trace(
            go.Scatter(
                x=[times[l2_t]],
                y=[p.second_low],
                mode="markers+text",
                marker=dict(symbol="circle", size=8, color="#ec407a"),
                text=["L2"],
                textposition="bottom center",
                showlegend=False,
            )
        )

    entry_t = _fmt_time(trade.signal.timestamp)
    fig.add_hline(y=p.neckline, line_dash="dash", line_color="#ffa726", opacity=0.75)
    fig.add_hline(y=trade.signal.stop_loss, line_dash="dot", line_color="#ff5252", opacity=0.65)
    fig.add_hline(y=trade.signal.target, line_dash="dot", line_color="#00c805", opacity=0.65)

    entry_idx = window.index.get_indexer([trade.signal.timestamp], method="nearest")[0]
    fig.add_trace(
        go.Scatter(
            x=[times[entry_idx]],
            y=[trade.signal.entry],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#00e676"),
            name="進場",
            showlegend=False,
        )
    )

    exit_idx = window.index.get_indexer([trade.exit_time], method="nearest")[0]
    exit_color = "#00c805" if trade.pnl_points > 0 else "#ff5252"
    fig.add_trace(
        go.Scatter(
            x=[times[exit_idx]],
            y=[trade.exit_price],
            mode="markers",
            marker=dict(symbol="x", size=10, color=exit_color),
            name="出場",
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=42, r=10, t=54, b=24),
        title=dict(text=f"#{trade_no} W底 | {entry_t}", x=0.02, font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat="%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _render_trade_card(df: pd.DataFrame, trade: TradeResult, trade_no: int, contracts: int = 1) -> str:
    sig = trade.signal
    p = sig.pattern
    pnl_class = "pnl-win" if trade.pnl_points > 0 else "pnl-loss"
    tag_text, tag_class = _exit_tag(trade.exit_reason, trade.pnl_points)
    depth = p.neckline - min(p.first_low, p.second_low)
    low_gap = abs(p.first_low - p.second_low)
    avg_low = (p.first_low + p.second_low) / 2
    gap_pct = low_gap / avg_low * 100 if avg_low else 0
    entry_row = df.iloc[sig.bar_idx]
    ma_line = _ma_snapshot(entry_row)

    chart_html = _build_trade_chart(df, trade, trade_no)

    return f"""
    <article class="trade-card">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{trade_no}</span>
          <span class="trade-time">{_fmt_time(sig.timestamp)} → {_fmt_time(trade.exit_time)}</span>
        </div>
        <div class="card-pnl {pnl_class}">{trade.pnl_points:+.1f} pts</div>
      </header>
      <div class="tags">
        <span class="tag {tag_class}">{tag_text}</span>
        <span class="tag tag-info">W底</span>
        <span class="tag tag-info">5m</span>
      </div>
      <pre class="trade-detail">entry(頸線突破) {sig.entry:.2f}
stop L2 {sig.stop_loss:.2f}
TP 量度漲幅 = {sig.target:.2f}
exit {trade.exit_price:.2f}
W底 L1 {p.first_low:.2f} / L2 {p.second_low:.2f}
頸線 {p.neckline:.2f} / 深度 {depth:.2f}
雙底價差 {gap_pct:.2f}% (≤0.10%)
{ma_line}
$ {trade.pnl_dollars:+,.2f} NQ×{contracts}</pre>
      <div class="tf-badge">🕐 5分 K</div>
      <div class="mini-chart">{chart_html}</div>
    </article>
    """


def build_report_html(
    df: pd.DataFrame,
    results: list[TradeResult],
    *,
    title: str,
    symbol: str = "NQ=F",
) -> str:
    df = _add_mas(df)
    stats = summarize(results)
    cards = "".join(_render_trade_card(df, r, i + 1) for i, r in enumerate(results))
    empty = '<div class="empty">今日未偵測到 W 底突破訊號</div>' if not results else ""

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #0b0e11;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      max-width: 520px;
      margin: 0 auto;
      padding: 12px 12px 28px;
    }}
    .summary {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    .summary h1 {{
      margin: 0 0 6px;
      font-size: 18px;
      font-weight: 700;
    }}
    .summary p {{
      margin: 0;
      color: #8b949e;
      font-size: 13px;
      line-height: 1.5;
    }}
    .summary .total {{
      margin-top: 8px;
      font-size: 15px;
      font-weight: 600;
      color: #00c805;
    }}
    .trade-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 14px;
      padding: 14px 14px 10px;
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .card-title {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .trade-no {{
      font-size: 15px;
      font-weight: 700;
    }}
    .trade-time {{
      font-size: 12px;
      color: #8b949e;
    }}
    .card-pnl {{
      font-size: 16px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .pnl-win {{ color: #00c805; }}
    .pnl-loss {{ color: #ff5252; }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .tag {{
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid transparent;
    }}
    .tag-tp {{ background: rgba(0,200,5,0.15); color: #3ddc68; border-color: rgba(0,200,5,0.35); }}
    .tag-sl {{ background: rgba(255,82,82,0.15); color: #ff7b72; border-color: rgba(255,82,82,0.35); }}
    .tag-time {{ background: rgba(255,193,7,0.12); color: #f0c14b; border-color: rgba(255,193,7,0.3); }}
    .tag-info {{ background: rgba(88,166,255,0.12); color: #79c0ff; border-color: rgba(88,166,255,0.28); }}
    .trade-detail {{
      margin: 0 0 10px;
      padding: 10px 12px;
      background: #0d1117;
      border-radius: 10px;
      border: 1px solid #21262d;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      color: #c9d1d9;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .tf-badge {{
      font-size: 12px;
      color: #8b949e;
      margin-bottom: 6px;
    }}
    .mini-chart {{
      margin: 0 -6px -4px;
      border-radius: 10px;
      overflow: hidden;
    }}
    .empty {{
      text-align: center;
      color: #8b949e;
      padding: 40px 16px;
      background: #161b22;
      border-radius: 14px;
      border: 1px solid #30363d;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="summary">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(symbol)} · 五分 K W底做多 · {html.escape(_report_date_range(df))}</p>
      <div class="total">
        {stats.get("trades", 0)} 筆 · 勝率 {stats.get("win_rate", 0) * 100:.0f}% ·
        總計 {stats.get("total_pnl_points", 0):+.1f} 點 (${stats.get("total_pnl_dollars", 0):+,.0f})
      </div>
    </section>
    {cards}{empty}
  </div>
</body>
</html>
"""


def _report_date_range(df: pd.DataFrame) -> str:
    start = df.index[0]
    end = df.index[-1]
    if hasattr(start, "tz_convert"):
        start = start.tz_convert("America/New_York")
        end = end.tz_convert("America/New_York")
    if start.date() == end.date():
        return start.strftime("%Y-%m-%d")
    return f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"


def save_report_html(
    df: pd.DataFrame,
    output: str | Path,
    *,
    strategy: NQWBottomStrategy | None = None,
    title: str | None = None,
    symbol: str = "NQ=F",
    today_only: bool = True,
) -> Path:
    strategy = strategy or NQWBottomStrategy()
    results = run_backtest(df, strategy)
    if today_only:
        results = _filter_today_results(df, results)
    if title is None:
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        title = f"NQ W底回測 — {today}"

    content = build_report_html(df, results, title=title, symbol=symbol)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
