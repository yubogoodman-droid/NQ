"""台股 MA 做空回測 HTML 報告。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from tw.backtest import TradeResult, summarize
from tw.universe import TwStock, latest_weekly_top

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    200: "#ab47bc",
}


def _fmt_day(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _exit_tag(reason: str) -> tuple[str, str]:
    mapping = {
        "take_profit": ("TP", "tag-tp"),
        "stop_loss": ("SL", "tag-sl"),
        "ma200_reclaim": ("站回200", "tag-time"),
        "time_stop": ("到期", "tag-time"),
    }
    return mapping.get(reason, (reason, "tag-info"))


def _equity_chart(results: list[TradeResult]) -> str:
    if not results:
        return ""
    ordered = sorted(results, key=lambda r: r.exit_time)
    dates = [_fmt_day(r.exit_time) for r in ordered]
    equity = pd.Series([r.pnl_twd for r in ordered]).cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            line=dict(color="#00c805", width=2),
            name="累計損益",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=48, r=16, t=36, b=40),
        title=dict(text="累計損益（每筆 10 萬名義本金）", x=0.02, font=dict(size=13)),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", ticksuffix=" ")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _trade_chart(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    sig = trade.signal
    start = max(0, sig.bar_idx - 40)
    end = min(len(df) - 1, sig.bar_idx + max(trade.hold_days, 5) + 8)
    window = df.iloc[start : end + 1].copy()
    close = window["close"]
    for p in (5, 10, 20, 200):
        window[f"ma{p}"] = df["close"].rolling(p, min_periods=p).mean().iloc[start : end + 1]
    times = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in window.index]

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
    for period, color in MA_COLORS.items():
        col = f"ma{period}"
        if window[col].notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=window[col],
                mode="lines",
                name=f"MA{period}",
                line=dict(color=color, width=1.4 if period <= 20 else 1.6),
                connectgaps=False,
            )
        )
    entry_x = pd.Timestamp(sig.timestamp).strftime("%Y-%m-%d")
    exit_x = pd.Timestamp(trade.exit_time).strftime("%Y-%m-%d")
    fig.add_trace(
        go.Scatter(
            x=[entry_x],
            y=[sig.entry],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ff5252"),
            name="進場做空",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[exit_x],
            y=[trade.exit_price],
            mode="markers",
            marker=dict(
                symbol="x",
                size=10,
                color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
            ),
            name="回補",
            showlegend=False,
        )
    )
    title = f"#{trade_no} {sig.ticker} {sig.name} 做空"
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=42, r=10, t=54, b=24),
        title=dict(text=title, x=0.02, font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _render_trade_card(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    sig = trade.signal
    pnl_class = "pnl-win" if trade.pnl_pct > 0 else "pnl-loss"
    tag_text, tag_class = _exit_tag(trade.exit_reason)
    chart = _trade_chart(df, trade, trade_no)
    return f"""
    <article class="trade-card">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{trade_no} {html.escape(sig.ticker)} {html.escape(sig.name)}</span>
          <span class="trade-time">{_fmt_day(sig.timestamp)} → {_fmt_day(trade.exit_time)} · 持有 {trade.hold_days} 日</span>
        </div>
        <div class="card-pnl {pnl_class}">{trade.pnl_pct * 100:+.2f}%</div>
      </header>
      <div class="tags">
        <span class="tag {tag_class}">{html.escape(tag_text)}</span>
        <span class="tag tag-info">空頭排列</span>
        <span class="tag tag-info">跌破MA200</span>
      </div>
      <pre class="trade-detail">進場(收盤做空) {sig.entry:.2f}
回補 {trade.exit_price:.2f}
MA5 {sig.ma5:.2f} / MA10 {sig.ma10:.2f} / MA20 {sig.ma20:.2f}
MA200 {sig.ma200:.2f}
損益 {trade.pnl_twd:+,.0f}（10萬名義）</pre>
      <div class="mini-chart">{chart}</div>
    </article>
    """


def _universe_table(top: pd.DataFrame) -> str:
    if top is None or top.empty:
        return ""
    week = pd.Timestamp(top.attrs.get("week_end")).strftime("%Y-%m-%d") if top.attrs.get("week_end") is not None else ""
    rows = []
    for _, r in top.iterrows():
        rows.append(
            f"<tr><td>{int(r['rank'])}</td><td>{html.escape(str(r['ticker']))}</td>"
            f"<td>{html.escape(str(r['name']))}</td>"
            f"<td>{float(r['close']):.2f}</td>"
            f"<td>{float(r['turnover']) / 1e8:.2f}</td></tr>"
        )
    return f"""
    <section class="summary">
      <h2>最近一週成交額前 100（已排除 ETF、收盤 &gt; 600）</h2>
      <p>週截止 {html.escape(week)} · 成交額單位：億元</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>代號</th><th>名稱</th><th>收盤</th><th>成交額</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def build_report_html(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    universe_top: pd.DataFrame | None = None,
    chart_trades: int = 20,
) -> str:
    stats = summarize(results)
    pf = stats["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
    equity_html = _equity_chart(results)
    cards = []
    show = list(reversed(results[-chart_trades:])) if results else []
    for i, trade in enumerate(show, start=max(1, len(results) - len(show) + 1)):
        df = frames.get(trade.signal.ticker)
        if df is None:
            continue
        cards.append(_render_trade_card(df, trade, i))
    empty = '<div class="empty">期間內沒有符合條件的進場訊號</div>' if not results else ""
    uni = _universe_table(universe_top) if universe_top is not None else ""
    rows = []
    for i, r in enumerate(results, start=1):
        cls = "win" if r.pnl_pct > 0 else "loss"
        rows.append(
            f"<tr class='{cls}'><td>{i}</td><td>{html.escape(r.signal.ticker)}</td>"
            f"<td>{html.escape(r.signal.name)}</td>"
            f"<td>{_fmt_day(r.signal.timestamp)}</td>"
            f"<td>{_fmt_day(r.exit_time)}</td>"
            f"<td>{html.escape(_exit_tag(r.exit_reason)[0])}</td>"
            f"<td>{r.pnl_pct * 100:+.2f}%</td>"
            f"<td>{r.pnl_twd:+,.0f}</td></tr>"
        )

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
    }}
    .page {{ max-width: 720px; margin: 0 auto; padding: 12px 12px 28px; }}
    .summary, .trade-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 18px; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    .summary p {{ margin: 0; color: #8b949e; font-size: 13px; line-height: 1.5; }}
    .total {{ margin-top: 8px; font-size: 15px; font-weight: 600; color: #00c805; }}
    .rules {{ margin-top: 10px; color: #8b949e; font-size: 12px; line-height: 1.55; }}
    .card-header {{ display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    .trade-no {{ font-size: 15px; font-weight: 700; }}
    .trade-time {{ font-size: 12px; color: #8b949e; }}
    .card-pnl {{ font-size: 16px; font-weight: 700; white-space: nowrap; }}
    .pnl-win {{ color: #00c805; }}
    .pnl-loss {{ color: #ff5252; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }}
    .tag {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 999px; border: 1px solid transparent; }}
    .tag-tp {{ background: rgba(0,200,5,0.15); color: #3ddc68; border-color: rgba(0,200,5,0.35); }}
    .tag-sl {{ background: rgba(255,82,82,0.15); color: #ff7b72; border-color: rgba(255,82,82,0.35); }}
    .tag-time {{ background: rgba(255,193,7,0.12); color: #f0c14b; border-color: rgba(255,193,7,0.3); }}
    .tag-info {{ background: rgba(88,166,255,0.12); color: #79c0ff; border-color: rgba(88,166,255,0.28); }}
    .trade-detail {{
      margin: 0 0 10px; padding: 10px 12px; background: #0d1117; border-radius: 10px;
      border: 1px solid #21262d; font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 12px; line-height: 1.55; color: #c9d1d9; white-space: pre-wrap;
    }}
    .empty {{ text-align: center; color: #8b949e; padding: 40px 16px; }}
    .table-wrap {{ overflow-x: auto; margin-top: 10px; max-height: 420px; overflow-y: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #21262d; white-space: nowrap; }}
    th {{ color: #8b949e; position: sticky; top: 0; background: #161b22; }}
    tr.win td:nth-last-child(-n+2) {{ color: #3ddc68; }}
    tr.loss td:nth-last-child(-n+2) {{ color: #ff7b72; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="summary">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
      <div class="total">
        {stats.get("trades", 0)} 筆 · 勝率 {stats.get("win_rate", 0) * 100:.1f}% ·
        平均 {stats.get("avg_pnl_pct", 0) * 100:+.2f}% · 獲利因子 {html.escape(pf_txt)} ·
        總計 {stats.get("total_pnl_twd", 0):+,.0f} · MDD {stats.get("max_drawdown_twd", 0):,.0f}
      </div>
      <div class="rules">
        進場：上一週成交額前 100（已排除 ETF、股價 &gt; 600），且 MA5 &lt; MA10 &lt; MA20，當日收盤跌破 MA200 做空。<br/>
        出場：停利 12% / 停損 8% / 收盤站回 MA200 / 持有滿 20 個交易日。含手續費 0.1425%×2 + 證交稅 0.3%。
      </div>
    </section>
    <section class="summary">{equity_html}</section>
    {uni}
    {''.join(cards)}{empty}
    <section class="summary">
      <h2>全部交易</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>代號</th><th>名稱</th><th>進場</th><th>出場</th><th>原因</th><th>報酬</th><th>損益</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
"""


def save_report_html(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    output: str | Path,
    *,
    title: str,
    subtitle: str,
    stocks: list[TwStock] | None = None,
) -> Path:
    universe_top = None
    if stocks and frames:
        from tw.data import to_panels

        _o, _h, _l, closes, volumes = to_panels(frames)
        universe_top = latest_weekly_top(closes, volumes, stocks)
    content = build_report_html(
        results,
        frames,
        title=title,
        subtitle=subtitle,
        universe_top=universe_top,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
