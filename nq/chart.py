"""產生 NQ W 底回測 HTML 圖表。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nq.backtest import TradeResult, run_backtest, summarize
from nq.strategy import NQWBottomStrategy, Signal


def _marker_time(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(None) if ts.tzinfo else ts


def build_chart(
    df: pd.DataFrame,
    signals: list[Signal],
    results: list[TradeResult],
    *,
    title: str = "NQ 五分K W底進場",
) -> go.Figure:
    times = [_marker_time(t) for t in df.index]

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
            name="NQ",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    colors = ["#42a5f5", "#ab47bc", "#ffa726", "#66bb6a", "#ec407a", "#29b6f6", "#8d6e63"]
    for i, sig in enumerate(signals):
        color = colors[i % len(colors)]
        p = sig.pattern
        trade = results[i] if i < len(results) else None

        low_points_x = [
            _marker_time(df.index[p.l1_idx]),
            _marker_time(df.index[p.l2_idx]),
            _marker_time(df.index[p.l3_idx]),
        ]
        low_points_y = [p.l1, p.l2, p.l3]
        fig.add_trace(
            go.Scatter(
                x=low_points_x,
                y=low_points_y,
                mode="markers+text",
                marker=dict(symbol="circle", size=9, color=color, line=dict(width=1, color="white")),
                text=["L1", "L2破底", "L3"],
                textposition="bottom center",
                name=f"破底W #{i + 1}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        neck_x = [_marker_time(df.index[p.neckline_idx]), _marker_time(df.index[p.l3_idx])]
        fig.add_trace(
            go.Scatter(
                x=neck_x,
                y=[p.neckline, p.neckline],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        entry_x = _marker_time(sig.timestamp)
        fig.add_trace(
            go.Scatter(
                x=[entry_x],
                y=[sig.entry],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color="#00e676", line=dict(width=1, color="white")),
                text=[f"進場 {sig.entry:.2f}"],
                textposition="top center",
                name=f"進場 #{i + 1}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_hline(y=sig.stop_loss, line_dash="dot", line_color="#ff5252", opacity=0.35, row=1, col=1)
        fig.add_hline(y=sig.target, line_dash="dot", line_color="#69f0ae", opacity=0.35, row=1, col=1)

        if trade:
            exit_x = _marker_time(trade.exit_time)
            exit_color = "#69f0ae" if trade.pnl_points > 0 else "#ff5252"
            fig.add_trace(
                go.Scatter(
                    x=[exit_x],
                    y=[trade.exit_price],
                    mode="markers+text",
                    marker=dict(symbol="x", size=11, color=exit_color, line=dict(width=2, color="white")),
                    text=[f"{trade.exit_reason} {trade.pnl_points:+.1f}pt"],
                    textposition="bottom center",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(
            x=times,
            y=df["volume"],
            name="Volume",
            marker_color="rgba(120,144,156,0.5)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    stats = summarize(results)
    subtitle = (
        f"交易 {stats.get('trades', 0)} 筆 | 勝率 {stats.get('win_rate', 0) * 100:.0f}% | "
        f"總損益 {stats.get('total_pnl_points', 0):+.1f} 點 "
        f"(${stats.get('total_pnl_dollars', 0):+,.0f})"
    )

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.01, xanchor="left"),
        template="plotly_dark",
        height=820,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def save_html_chart(
    df: pd.DataFrame,
    output: str | Path,
    *,
    strategy: NQWBottomStrategy | None = None,
    title: str = "NQ 五分K W底進場",
) -> Path:
    strategy = strategy or NQWBottomStrategy()
    signals = strategy.generate_signals(df)
    results = run_backtest(df, strategy)
    fig = build_chart(df, signals, results, title=title)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out
