"""假跌破後上拉 HTML 圖表。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nq.backtest import TradeResult, run_backtest, summarize
from nq.spring import FakeBreakdownPattern
from nq.strategy import FakeBreakdownStrategy, Signal


def _ts(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(None) if ts.tzinfo else ts


def build_spring_chart(
    df: pd.DataFrame,
    signals: list[Signal],
    results: list[TradeResult],
    *,
    title: str = "假跌破後上拉",
) -> go.Figure:
    times = [_ts(t) for t in df.index]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(title, "成交量"),
    )
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1,
        col=1,
    )

    colors = ["#42a5f5", "#ab47bc", "#ffa726", "#66bb6a"]
    for i, sig in enumerate(signals):
        p = sig.pattern
        if not isinstance(p, FakeBreakdownPattern):
            continue
        color = colors[i % len(colors)]
        box_x = [_ts(df.index[p.range_start_idx]), _ts(df.index[p.range_end_idx])]
        fig.add_trace(
            go.Scatter(
                x=box_x + box_x[::-1],
                y=[p.support, p.support, p.resistance, p.resistance],
                fill="toself",
                fillcolor="rgba(66,165,245,0.12)",
                line=dict(color=color, width=1, dash="dash"),
                name=f"盤整 #{i + 1}",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[_ts(df.index[p.spring_idx])],
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
                x=[_ts(sig.timestamp)],
                y=[sig.entry],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color="#00e676"),
                text=[f"進場 {sig.entry:.2f}"],
                textposition="top center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        if i < len(results):
            trade = results[i]
            fig.add_trace(
                go.Scatter(
                    x=[_ts(trade.exit_time)],
                    y=[trade.exit_price],
                    mode="markers+text",
                    marker=dict(
                        symbol="x",
                        size=11,
                        color="#69f0ae" if trade.pnl_points > 0 else "#ff5252",
                    ),
                    text=[f"{trade.exit_reason} {trade.pnl_points:+.1f}"],
                    textposition="bottom center",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(x=times, y=df["volume"], marker_color="rgba(120,144,156,0.5)", showlegend=False),
        row=2,
        col=1,
    )
    stats = summarize(results)
    subtitle = (
        f"交易 {stats.get('trades', 0)} 筆 | 勝率 {stats.get('win_rate', 0) * 100:.0f}% | "
        f"總損益 {stats.get('total_pnl_points', 0):+.1f}"
    )
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.01, xanchor="left"),
        template="plotly_dark",
        height=820,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=60, r=30, t=90, b=40),
    )
    return fig


def save_spring_html_chart(
    df: pd.DataFrame,
    output: str | Path,
    *,
    strategy: FakeBreakdownStrategy | None = None,
    title: str = "假跌破後上拉",
    max_bars_hold: int = 60,
) -> Path:
    strategy = strategy or FakeBreakdownStrategy()
    signals = strategy.generate_signals(df)
    results = run_backtest(df, strategy, max_bars_hold=max_bars_hold)
    fig = build_spring_chart(df, signals, results, title=title)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out
