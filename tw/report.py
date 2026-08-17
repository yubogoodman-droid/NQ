"""台股一分 K 做空回測報告：PNG 圖檔 + HTML / Markdown（跟掃描頁同一套畫法）。"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tw.backtest import TradeResult, summarize
from tw.chart import BG, FG, FONT, GRID, MA_COLORS, save_trade_charts
from tw.universe import TwStock

CHART_TRADES = 20


def _fmt_ts(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.hour or t.minute or t.second:
        return t.strftime("%m-%d %H:%M")
    return t.strftime("%Y-%m-%d")


def _exit_tag(reason: str) -> tuple[str, str]:
    mapping = {
        "take_profit": ("TP", "tag-tp"),
        "stop_loss": ("SL", "tag-sl"),
        "ma200_reclaim": ("站回200", "tag-time"),
        "time_stop": ("到期", "tag-time"),
        "session_close": ("收盤", "tag-time"),
    }
    return mapping.get(reason, (reason, "tag-info"))


def _safe_stem(ticker: str, trade_no: int) -> str:
    return f"{trade_no:03d}-{ticker.replace('.', '_')}"


def _equity_png(results: list[TradeResult]) -> bytes | None:
    if not results:
        return None
    ordered = sorted(results, key=lambda r: r.exit_time)
    equity = pd.Series([r.pnl_twd for r in ordered]).cumsum()
    fig, ax = plt.subplots(figsize=(8.4, 2.8), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot(range(len(equity)), equity, color="#7ee787", linewidth=1.8)
    ax.set_title("累計損益（每筆 10 萬名義本金）", color=FG, fontsize=11, pad=8, fontproperties=FONT)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_xticks([])
    fig.tight_layout(pad=0.35)
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


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


def _chart_block(rel: str | None, label: str) -> str:
    if not rel:
        return f'<p class="empty">無{html.escape(label)}資料</p>'
    return f'<div class="chart-label">{html.escape(label)}</div><div class="chart"><img src="{html.escape(rel)}" alt="{html.escape(label)}"/></div>'


def build_report_html(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    universe_top: pd.DataFrame | None = None,
    chart_rel: Path,
    chart_dir: Path,
    chart_trades: int = CHART_TRADES,
) -> str:
    stats = summarize(results)
    pf = stats["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"

    equity_rel = ""
    png = _equity_png(results)
    if png:
        equity_path = chart_dir / "equity.png"
        equity_path.write_bytes(png)
        equity_rel = f"{chart_rel.as_posix()}/equity.png"

    show = list(reversed(results[-chart_trades:])) if results else []
    start_no = max(1, len(results) - len(show) + 1)
    cards = []
    for i, trade in enumerate(show):
        trade_no = start_no + i
        df = frames.get(trade.signal.ticker)
        rels: dict[str, str] = {}
        if df is not None:
            stem = _safe_stem(trade.signal.ticker, trade_no)
            saved = save_trade_charts(df, trade, chart_dir, stem)
            rels = {tf: f"{chart_rel.as_posix()}/{path.name}" for tf, path in saved.items()}
        cards.append(_render_trade_card(trade, trade_no, rels))

    empty = '<p class="empty">期間內沒有符合條件的進場訊號</p>' if not results else ""
    uni = _universe_table(universe_top) if universe_top is not None else ""
    rows = []
    for i, r in enumerate(results, start=1):
        cls = "win" if r.pnl_pct > 0 else "loss"
        rows.append(
            f"<tr class='{cls}'><td>{i}</td><td>{html.escape(r.signal.ticker)}</td>"
            f"<td>{html.escape(r.signal.name)}</td>"
            f"<td>{_fmt_ts(r.signal.timestamp)}</td>"
            f"<td>{_fmt_ts(r.exit_time)}</td>"
            f"<td>{html.escape(_exit_tag(r.exit_reason)[0])}</td>"
            f"<td>{r.pnl_pct * 100:+.2f}%</td>"
            f"<td>{r.pnl_twd:+,.0f}</td></tr>"
        )
    legend = "".join(
        f'<span><i class="swatch" style="background:{color}"></i>MA{p}</span>'
        for p, color in MA_COLORS.items()
    )
    equity_html = (
        f'<section class="summary"><img class="equity" src="{html.escape(equity_rel)}" alt="累計損益"/></section>'
        if equity_rel
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>{html.escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #0b0e11;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
    }}
    .page {{ max-width: 760px; margin: 0 auto; padding: 12px 12px 28px; }}
    .summary, .trade-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 18px; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    .summary p, .lead {{ margin: 0; color: #8b949e; font-size: 13px; line-height: 1.5; }}
    .total {{ margin-top: 8px; font-size: 15px; font-weight: 600; color: #7ee787; }}
    .rules {{ margin-top: 10px; color: #8b949e; font-size: 12px; line-height: 1.55; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #8b949e; margin: 10px 0 0; }}
    .swatch {{ display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }}
    .card-header {{ display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    .trade-no {{ font-size: 15px; font-weight: 700; }}
    .trade-time {{ font-size: 12px; color: #8b949e; }}
    .card-pnl {{ font-size: 16px; font-weight: 700; white-space: nowrap; }}
    .pnl-win {{ color: #7ee787; }}
    .pnl-loss {{ color: #ff7b72; }}
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
    .chart-label {{ font-size: 12px; color: #7ee787; font-weight: 700; padding: 8px 0 4px; }}
    .chart img, img.equity {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
    .empty {{ text-align: center; color: #8b949e; padding: 20px 16px; }}
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
      <p class="lead">{html.escape(subtitle)}</p>
      <div class="total">
        {stats.get("trades", 0)} 筆 · 勝率 {stats.get("win_rate", 0) * 100:.1f}% ·
        平均 {stats.get("avg_pnl_pct", 0) * 100:+.2f}% · 獲利因子 {html.escape(pf_txt)} ·
        總計 {stats.get("total_pnl_twd", 0):+,.0f} · MDD {stats.get("max_drawdown_twd", 0):,.0f}
      </div>
      <div class="rules">
        進場：上一週成交額前 100（已排除 ETF、股價 &gt; 600），一分 K 的 MA5 &lt; MA10 &lt; MA20，當根收盤跌破 MA200 做空。13:00 後不再進場。<br/>
        出場：停利 1.2% / 停損 0.8% / 收盤站回 MA200 / 持有滿 30 根一分 K / 當日收盤強制回補。費用採當沖：手續費 0.1425%×2 + 證交稅 0.15%。<br/>
        圖：漲紅跌綠；一分 K 只切跌破附近，縱軸對準 MA20/MA200（不要被旁邊回檔撐開）。
      </div>
      <div class="legend">{legend}<span>K棒 漲紅跌綠</span></div>
    </section>
    {equity_html}
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


def _render_trade_card(trade: TradeResult, trade_no: int, rels: dict[str, str]) -> str:
    sig = trade.signal
    pnl_class = "pnl-win" if trade.pnl_pct > 0 else "pnl-loss"
    tag_text, tag_class = _exit_tag(trade.exit_reason)
    return f"""
    <article class="trade-card">
      <header class="card-header">
        <div class="card-title">
          <span class="trade-no">#{trade_no} {html.escape(sig.ticker)} {html.escape(sig.name)}</span>
          <span class="trade-time">{_fmt_ts(sig.timestamp)} → {_fmt_ts(trade.exit_time)} · 持有 {trade.hold_bars} 分</span>
        </div>
        <div class="card-pnl {pnl_class}">{trade.pnl_pct * 100:+.2f}%</div>
      </header>
      <div class="tags">
        <span class="tag {tag_class}">{html.escape(tag_text)}</span>
        <span class="tag tag-info">一分K</span>
        <span class="tag tag-info">空頭排列</span>
        <span class="tag tag-info">跌破MA200</span>
      </div>
      <pre class="trade-detail">進場(一分K收盤做空) {sig.entry:.2f}
回補 {trade.exit_price:.2f}
MA5 {sig.ma5:.2f} / MA10 {sig.ma10:.2f} / MA20 {sig.ma20:.2f}
MA200 {sig.ma200:.2f}
損益 {trade.pnl_twd:+,.0f}（10萬名義）</pre>
      {_chart_block(rels.get("1m"), "一分 K")}
      {_chart_block(rels.get("5m"), "五分 K（對照）")}
    </article>
    """


def _write_markdown(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    path: Path,
    *,
    title: str,
    subtitle: str,
    chart_rel: Path,
    chart_dir: Path,
    chart_trades: int,
) -> None:
    stats = summarize(results)
    lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        f"- {stats.get('trades', 0)} 筆 · 勝率 {stats.get('win_rate', 0) * 100:.1f}% · 平均 {stats.get('avg_pnl_pct', 0) * 100:+.2f}%",
        "- 圖：漲紅跌綠；一分 K 只切跌破附近，縱軸對準 MA20/MA200。",
        "",
    ]
    if (chart_dir / "equity.png").exists():
        lines.extend([f"![累計損益]({chart_rel.as_posix()}/equity.png)", ""])
    show = list(reversed(results[-chart_trades:])) if results else []
    start_no = max(1, len(results) - len(show) + 1)
    for i, trade in enumerate(show):
        trade_no = start_no + i
        sig = trade.signal
        stem = _safe_stem(sig.ticker, trade_no)
        rel = chart_rel.as_posix()
        lines.extend(
            [
                f"## {trade_no}. {sig.name} [{sig.ticker}](https://tw.stock.yahoo.com/quote/{sig.ticker})",
                "",
                f"- {_fmt_ts(sig.timestamp)} → {_fmt_ts(trade.exit_time)} · 持有 {trade.hold_bars} 分 · {trade.pnl_pct * 100:+.2f}%",
                f"- 進場 {sig.entry:.2f} / 回補 {trade.exit_price:.2f} / MA200 {sig.ma200:.2f}",
                f"- MA5 {sig.ma5:.2f} < MA10 {sig.ma10:.2f} < MA20 {sig.ma20:.2f}",
                "",
                "**一分 K**",
                "",
                f"![{sig.name} 一分K]({rel}/{stem}-1m.png)",
                "",
                "**五分 K（對照）**",
                "",
                f"![{sig.name} 五分K]({rel}/{stem}-5m.png)",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report_html(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    output: str | Path,
    *,
    title: str,
    subtitle: str,
    stocks: list[TwStock] | None = None,
    universe_top: pd.DataFrame | None = None,
    chart_trades: int = CHART_TRADES,
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    chart_rel = Path("charts") / out.stem
    chart_dir = out.parent / chart_rel
    if chart_dir.exists():
        shutil.rmtree(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    content = build_report_html(
        results,
        frames,
        title=title,
        subtitle=subtitle,
        universe_top=universe_top,
        chart_rel=chart_rel,
        chart_dir=chart_dir,
        chart_trades=chart_trades,
    )
    out.write_text(content, encoding="utf-8")
    _write_markdown(
        results,
        frames,
        out.with_suffix(".md"),
        title=title,
        subtitle=subtitle,
        chart_rel=chart_rel,
        chart_dir=chart_dir,
        chart_trades=chart_trades,
    )
    if out.stem in {"index", "today"}:
        today_md = out.parent / "today.md"
        if today_md.resolve() != out.with_suffix(".md").resolve():
            shutil.copyfile(out.with_suffix(".md"), today_md)
    return out
