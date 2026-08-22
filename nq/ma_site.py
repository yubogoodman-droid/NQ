"""南亞科均線回測 HTML 站：每筆一分 K + 六條均線。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nq.nanya_ma import (
    MA_PERIODS,
    NanyaMaTrade,
    add_nanya_features,
    summarize_ma_trades,
)

MA_COLORS = {
    5: "#42a5f5",
    10: "#66bb6a",
    20: "#ffa726",
    60: "#26c6da",
    120: "#ab47bc",
    200: "#ef5350",
}

EXIT_ZH = {
    "take_profit": "停利",
    "stop_loss": "停損",
    "lost_ma20": "跌破MA20",
    "time_stop": "時間停",
    "session_flat": "收盤平",
}


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts


def _fmt(ts: pd.Timestamp) -> str:
    return _naive(ts).strftime("%m-%d %H:%M")


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _chart_html(df: pd.DataFrame, trade: NanyaMaTrade, trade_no: int) -> str:
    work = add_nanya_features(df)
    start = max(0, trade.signal.bar_idx - 55)
    end = min(len(work) - 1, trade.signal.bar_idx + 35)
    for i in range(trade.signal.bar_idx, len(work)):
        if work.index[i] == trade.exit_time:
            end = min(len(work) - 1, i + 12)
            break
    window = work.iloc[start : end + 1]
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
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
            name="K",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for period in MA_PERIODS:
        col = f"ma{period}"
        if col not in window.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=window[col],
                mode="lines",
                name=f"MA{period}",
                line=dict(color=MA_COLORS[period], width=1.6 if period <= 20 else 1.2),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    fig.add_hline(y=trade.signal.range_high, line_dash="dash", line_color="#8b949e", opacity=0.45, row=1, col=1)
    fig.add_hline(y=trade.signal.stop_loss, line_dash="dot", line_color="#ff5252", opacity=0.7, row=1, col=1)
    fig.add_hline(y=trade.signal.target, line_dash="dot", line_color="#69f0ae", opacity=0.7, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=[_naive(trade.signal.timestamp)],
            y=[trade.signal.entry],
            mode="markers",
            marker=dict(symbol="triangle-up", size=13, color="#00e676", line=dict(width=1, color="white")),
            name="進場",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[_naive(trade.exit_time)],
            y=[trade.exit_price],
            mode="markers",
            marker=dict(
                symbol="x",
                size=11,
                color="#69f0ae" if trade.pnl_pct_net > 0 else "#ff5252",
                line=dict(width=2, color="white"),
            ),
            name="出場",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    colors = ["#ef5350" if c >= o else "#26a69a" for o, c in zip(window["open"], window["close"])]
    fig.add_trace(
        go.Bar(x=times, y=window["volume"], marker_color=colors, opacity=0.55, showlegend=False, name="量"),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=48, r=12, t=36, b=28),
        title=dict(text=f"#{trade_no} {trade.symbol} · {_fmt(trade.signal.timestamp)}", x=0.01, font=dict(size=13)),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat="%m-%d %H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _equity_html(trades: list[NanyaMaTrade]) -> str:
    if not trades:
        return ""
    xs = [_naive(t.exit_time) for t in trades]
    ys = []
    acc = 0.0
    for t in trades:
        acc += t.pnl_pct_net * 100
        ys.append(acc)
    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#79c0ff", width=2), marker=dict(size=6))
    )
    fig.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=48, r=12, t=28, b=28),
        title=dict(text="累計淨損益（%）", x=0.01, font=dict(size=13)),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_backtest_site(
    *,
    title: str,
    trades: list[NanyaMaTrade],
    frames: dict[str, pd.DataFrame],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> str:
    overall = summarize_ma_trades(trades)

    def wr(x: float) -> str:
        return f"{x * 100:.0f}%"

    chips = ['<button class="chip active" data-sym="ALL">全部</button>']
    for sym, stats, _ in symbol_stats:
        cls = "pos" if stats["total_pnl_pct_net"] >= 0 else "neg"
        chips.append(
            f'<button class="chip" data-sym="{html.escape(sym)}">{html.escape(sym)} '
            f'<span class="{cls}">{_pct(stats["total_pnl_pct_net"])}</span></button>'
        )

    sym_rows = "".join(
        f"<tr><td>{html.escape(sym)}</td><td>{bars}</td><td>{s['trades']}</td>"
        f"<td>{wr(s['win_rate'])}</td>"
        f"<td class=\"{'pos' if s['total_pnl_pct_net']>=0 else 'neg'}\">{_pct(s['total_pnl_pct_net'])}</td></tr>"
        for sym, s, bars in symbol_stats
    )

    cards = []
    for i, trade in enumerate(trades, start=1):
        df = frames.get(trade.symbol)
        chart = _chart_html(df, trade, i) if df is not None and len(df) else "<p class='muted'>沒有K線</p>"
        pnl_cls = "pos" if trade.pnl_pct_net >= 0 else "neg"
        tag = EXIT_ZH.get(trade.exit_reason, trade.exit_reason)
        cards.append(
            f"""
<article class="trade" data-sym="{html.escape(trade.symbol)}">
  <header>
    <div>
      <div class="t-title">#{i} {html.escape(trade.symbol)}</div>
      <div class="t-sub">{_fmt(trade.signal.timestamp)} → {_fmt(trade.exit_time)}</div>
    </div>
    <div class="t-pnl {pnl_cls}">{_pct(trade.pnl_pct_net)}</div>
  </header>
  <div class="tags">
    <span class="tag">{html.escape(tag)}</span>
    <span class="tag">1分K</span>
    <span class="tag">5&gt;10&gt;20</span>
  </div>
  <pre>進場 {trade.signal.entry:.2f}
停損 {trade.signal.stop_loss:.2f}　停利 {trade.signal.target:.2f}　出場 {trade.exit_price:.2f}
MA5 {trade.signal.ma5:.2f}　MA10 {trade.signal.ma10:.2f}　MA20 {trade.signal.ma20:.2f}
MA60 {trade.signal.ma60:.2f}　MA120 {trade.signal.ma120:.2f}　MA200 {trade.signal.ma200:.2f}
5–20 扇開 {trade.signal.short_span_pct*100:.2f}%　離MA200 {trade.signal.ext_200_pct*100:.2f}%　量比 {trade.signal.vol_ratio:.1f}x</pre>
  <div class="chart">{chart}</div>
</article>
"""
        )

    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    empty = "" if trades else '<div class="empty">沒有成交</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg:#0b0e11; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e;
      --line:#30363d; --pos:#3ddc68; --neg:#ff7b72; --acc:#79c0ff;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:radial-gradient(900px 500px at 10% -10%, rgba(121,192,255,.12), transparent 50%), var(--bg);
      color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;
    }}
    .wrap {{ max-width:1080px; margin:0 auto; padding:22px 16px 56px; }}
    h1 {{ font-size:clamp(1.5rem,3vw,2rem); margin:0 0 8px; }}
    h2 {{ font-size:16px; margin:26px 0 10px; }}
    .sub, li {{ color:var(--muted); line-height:1.65; font-size:14px; }}
    .nav {{ display:flex; gap:10px; flex-wrap:wrap; margin:0 0 18px; }}
    .nav a {{ color:var(--acc); text-decoration:none; font-size:13px; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0; }}
    .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
    .stat .k {{ color:var(--muted); font-size:12px; }}
    .stat .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 16px; }}
    .chip {{
      background:var(--panel); border:1px solid var(--line); color:var(--ink);
      border-radius:999px; padding:7px 12px; cursor:pointer; font-size:13px;
    }}
    .chip.active {{ border-color:var(--acc); color:var(--acc); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid #21262d; padding:8px 6px; text-align:left; }}
    th {{ color:var(--muted); }}
    .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
    .trade {{
      background:var(--panel); border:1px solid var(--line); border-radius:14px;
      padding:14px 14px 8px; margin-bottom:14px;
    }}
    .trade header {{ display:flex; justify-content:space-between; gap:10px; }}
    .t-title {{ font-weight:700; }}
    .t-sub {{ color:var(--muted); font-size:12px; margin-top:2px; }}
    .t-pnl {{ font-size:18px; font-weight:700; }}
    .tags {{ display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }}
    .tag {{
      font-size:11px; padding:3px 8px; border-radius:999px;
      border:1px solid rgba(121,192,255,.3); color:var(--acc); background:rgba(121,192,255,.08);
    }}
    pre {{
      margin:0 0 10px; padding:10px 12px; background:#0d1117; border-radius:10px;
      border:1px solid #21262d; font-size:12px; line-height:1.55; color:#c9d1d9;
      white-space:pre-wrap;
    }}
    .chart {{ margin:0 -6px; }}
    .empty {{ text-align:center; color:var(--muted); padding:40px 12px; border:1px solid var(--line); border-radius:14px; }}
    .eq {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:8px; margin-bottom:16px; }}
    @media (max-width:720px) {{ .stats {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="nav">
      <a href="../nanya-ma/">表格版</a>
      <a href="../1m-candles/">K棒回測</a>
      <a href="../">NQ W底</a>
    </div>
    <h1>{html.escape(title)}</h1>
    <p class="sub">
      南亞科一分圖同款均線：MA5 / 10 / 20 / 60 / 120 / 200。
      盤整短均要黏，進場是 5&gt;10&gt;20 剛扇開、價剛離開 MA200。
      436 那種末端多頭不追。K 線紅漲綠跌（台股配色）。近 7 日 Yahoo 一分 K，學習用。
    </p>
    <div class="stats">
      <div class="stat"><div class="k">成交</div><div class="v">{overall['trades']}</div></div>
      <div class="stat"><div class="k">勝率</div><div class="v">{wr(overall['win_rate'])}</div></div>
      <div class="stat"><div class="k">累計淨損益</div><div class="v {'pos' if overall['total_pnl_pct_net']>=0 else 'neg'}">{_pct(overall['total_pnl_pct_net'])}</div></div>
      <div class="stat"><div class="k">單筆期望</div><div class="v {'pos' if overall['expectancy_net']>=0 else 'neg'}">{_pct(overall['expectancy_net'])}</div></div>
    </div>
    <div class="chips">{''.join(chips)}</div>
    <div class="eq">{_equity_html(trades)}</div>
    <h2>標的</h2>
    <table>
      <thead><tr><th>代號</th><th>K 數</th><th>筆數</th><th>勝率</th><th>淨損益</th></tr></thead>
      <tbody>{sym_rows}</tbody>
    </table>
    <h2>每筆一分圖</h2>
    {''.join(cards)}{empty}
    <h2>規則</h2>
    <ul>{note_html}</ul>
  </div>
  <script>
    const chips = document.querySelectorAll('.chip');
    const cards = document.querySelectorAll('.trade');
    chips.forEach(btn => btn.addEventListener('click', () => {{
      chips.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const sym = btn.dataset.sym;
      cards.forEach(c => {{
        c.style.display = (sym === 'ALL' || c.dataset.sym === sym) ? '' : 'none';
      }});
    }}));
  </script>
</body>
</html>
"""


def save_backtest_site(
    output: str | Path,
    *,
    title: str,
    trades: list[NanyaMaTrade],
    frames: dict[str, pd.DataFrame],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_backtest_site(title=title, trades=trades, frames=frames, notes=notes, symbol_stats=symbol_stats),
        encoding="utf-8",
    )
    return out
