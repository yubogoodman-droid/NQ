"""掃描頁同一套模板：chips + 卡片列 + 一分/五分 PNG。"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from tw.backtest import TradeResult, summarize
from tw.chart import add_moving_averages, resample_5m, save_trade_charts
from tw.universe import TwStock

CHART_TRADES = 20

PAGE_CSS = """
    :root {
      color-scheme: dark;
      --bg: #0b0e11;
      --card: #161b22;
      --line: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --up: #ff5c7a;
      --ok: #7ee787;
      --chip: #21262d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    .page { max-width: 760px; margin: 0 auto; padding: 16px 12px 40px; }
    h1 { font-size: 1.2rem; margin: 0 0 6px; }
    .lead { color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0 0 12px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .chip {
      background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
      padding: 4px 10px; font-size: 12px; color: var(--muted);
    }
    .legend {
      display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted);
      margin: 0 0 12px;
    }
    .swatch { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }
    .summary {
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; margin-bottom: 14px; font-size: .9rem; line-height: 1.65;
      color: var(--muted);
    }
    .summary .ok { color: var(--ok); font-weight: 700; font-size: 1.05rem; }
    .card {
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 14px 10px 10px; margin: 0 0 14px; color: inherit;
    }
    .top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding: 0 6px; }
    .name { font-weight: 700; font-size: 1.05rem; }
    .sym a { color: var(--muted); font-weight: 500; margin-left: 6px; font-size: .9rem; text-decoration: none; }
    .price { color: var(--up); font-weight: 700; white-space: nowrap; }
    .price.ok { color: var(--ok); }
    .row { display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: .9rem; color: var(--muted); padding: 0 6px; }
    .row b { color: var(--text); font-weight: 600; }
    .chart { margin-top: 10px; }
    .chart img { width: 100%; height: auto; display: block; border-radius: 8px; }
    .chart-label {
      font-size: 12px; color: var(--ok); font-weight: 700;
      padding: 8px 6px 4px;
    }
    .empty { color: var(--muted); }
    footer { color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.5; }
    .note {
      background: var(--chip); border: 1px solid var(--line); border-radius: 10px;
      padding: 10px 12px; margin: 0 0 12px; font-size: 13px; color: var(--muted); line-height: 1.5;
    }
    .note a { color: #58a6ff; }
"""


def _safe_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in symbol)


def _exit_label(reason: str) -> str:
    return {
        "take_profit": "停利",
        "stop_loss": "停損",
        "ma200_reclaim": "站回200",
        "time_stop": "到期",
        "session_close": "收盤",
    }.get(reason, reason)


def _five_min_ma200_text(df: pd.DataFrame | None, ts: pd.Timestamp) -> str:
    if df is None or df.empty:
        return "—"
    work = add_moving_averages(resample_5m(df))
    if work.empty or "ma200" not in work.columns:
        return "—"
    mark = pd.Timestamp(ts).floor("5min")
    loc = work.index.get_indexer([mark], method="nearest")[0]
    if loc < 0:
        return "—"
    close = float(work["close"].iloc[loc])
    ma200 = work["ma200"].iloc[loc]
    if pd.isna(ma200):
        return f"{close:.2f} / —"
    ma200 = float(ma200)
    gap = (close - ma200) / ma200 if ma200 else 0.0
    cmp_ = ">" if close > ma200 else "<"
    return f"{close:.2f} {cmp_} {ma200:.2f}（{gap:+.1%}）"


def _rank_text(ticker: str, universe_top: pd.DataFrame | None) -> str:
    if universe_top is None or universe_top.empty:
        return "—"
    rows = universe_top[universe_top["ticker"] == ticker]
    if rows.empty:
        return "—"
    r = rows.iloc[0]
    return f"#{int(r['rank'])} · {float(r['turnover']) / 1e8:.2f} 億"


def _github_blob_url(file_path: Path) -> str | None:
    """GitHub markdown preview URL. Uses refs/heads so slash branch names work."""
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=file_path.parent,
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
    text = re.sub(r"^git@", "https://", remote).replace("github.com:", "github.com/")
    match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", text)
    if match is None or not branch or branch == "HEAD":
        return None
    rel = file_path.resolve().relative_to(root).as_posix()
    return (
        f"https://github.com/{match.group('owner')}/{match.group('repo')}"
        f"/blob/refs/heads/{branch}/{rel}"
    )


def _img(rel: str | None, alt: str, image_base: str | None) -> str:
    if not rel:
        return '<p class="empty">無 K 線資料</p>'
    # Relative paths work on GitHub Pages and github.com markdown preview.
    # raw.githubusercontent.com is often blocked in TW.
    src = html.escape(f"{image_base.rstrip('/')}/{rel}" if image_base else rel)
    return f'<img alt="{html.escape(alt)}" src="{src}"/>'


def _trade_card(
    index: int,
    trade: TradeResult,
    df: pd.DataFrame | None,
    *,
    chart_rel: Path,
    chart_dir: Path,
    image_base: str | None,
    universe_top: pd.DataFrame | None,
) -> tuple[str, str]:
    sig = trade.signal
    ts = pd.Timestamp(sig.timestamp).strftime("%H:%M")
    exit_t = pd.Timestamp(trade.exit_time).strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{html.escape(sig.ticker)}"
    chg = f" {trade.pnl_pct * 100:+.2f}%"
    price_cls = "price ok" if trade.pnl_pct > 0 else "price"
    gap = (sig.ma20 - sig.ma200) / sig.ma200 if sig.ma200 else 0.0
    five = _five_min_ma200_text(df, sig.timestamp)
    stem = f"{_safe_symbol(sig.ticker)}-{ts.replace(':', '')}"
    rels: dict[str, str] = {}
    if df is not None:
        saved = save_trade_charts(df, trade, chart_dir, stem)
        rels = {tf: f"{chart_rel.as_posix()}/{path.name}" for tf, path in saved.items()}
    chart_1m = _img(rels.get("1m"), f"{sig.name} {sig.ticker} 一分K", image_base)
    chart_5m = _img(rels.get("5m"), f"{sig.name} {sig.ticker} 五分K", image_base)
    card = f"""
    <article class="card">
      <div class="top">
        <div class="name">{index}. {html.escape(sig.name)}<span class="sym"><a href="{url}" target="_blank" rel="noopener">{html.escape(sig.ticker)}</a></span></div>
        <div class="{price_cls}">{sig.entry:.2f}{html.escape(chg)}</div>
      </div>
      <div class="row"><span>1分跌破時間</span><b>{ts}</b></div>
      <div class="row"><span>1分收盤 / MA200</span><b>{sig.entry:.2f} &lt; {sig.ma200:.2f}</b></div>
      <div class="row"><span>五分收盤 / MA200</span><b>{html.escape(five)}</b></div>
      <div class="row"><span>1分 MA5 / 10 / 20</span><b>{sig.ma5:.2f} &lt; {sig.ma10:.2f} &lt; {sig.ma20:.2f}</b></div>
      <div class="row"><span>MA20 與 MA200</span><b>{sig.ma20:.2f} / {sig.ma200:.2f}（差 {gap:.1%}）</b></div>
      <div class="row"><span>成交額排名</span><b>{html.escape(_rank_text(sig.ticker, universe_top))}</b></div>
      <div class="row"><span>回補</span><b>{exit_t} {trade.exit_price:.2f} · {_exit_label(trade.exit_reason)} · {trade.hold_bars} 分</b></div>
      <div class="chart-label">一分 K</div>
      <div class="chart">{chart_1m}</div>
      <div class="chart-label">五分 K（對照）</div>
      <div class="chart">{chart_5m}</div>
    </article>
"""
    md = "\n".join(
        [
            f"## {index}. {sig.name} [{sig.ticker}](https://tw.stock.yahoo.com/quote/{sig.ticker})",
            "",
            f"- 價格 {sig.entry:.2f}{chg}　成交額排名 {_rank_text(sig.ticker, universe_top)}",
            f"- 1分跌破 {ts}　收 {sig.entry:.2f} < MA200 {sig.ma200:.2f}",
            f"- 1分 MA5 {sig.ma5:.2f} < MA10 {sig.ma10:.2f} < MA20 {sig.ma20:.2f}　MA20/MA200 差 {gap:.1%}",
            f"- 五分收盤 / MA200　{five}",
            f"- 回補 {exit_t} {trade.exit_price:.2f}（{_exit_label(trade.exit_reason)} · {trade.hold_bars} 分）",
            "",
            "**一分 K**",
            "",
            f"![{sig.name} 一分K]({chart_rel.as_posix()}/{stem}-1m.png)",
            "",
            "**五分 K（對照）**",
            "",
            f"![{sig.name} 五分K]({chart_rel.as_posix()}/{stem}-5m.png)",
            "",
        ]
    )
    return card, md


def build_report_html(
    results: list[TradeResult],
    frames: dict[str, pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    universe_top: pd.DataFrame | None = None,
    chart_rel: Path,
    chart_dir: Path,
    image_base: str | None = None,
    chart_trades: int = CHART_TRADES,
    github_md_url: str | None = None,
) -> tuple[str, str]:
    stats = summarize(results)
    pf = stats["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
    show = list(reversed(results[-chart_trades:])) if results else []
    cards: list[str] = []
    md_cards: list[str] = []
    for i, trade in enumerate(show, 1):
        df = frames.get(trade.signal.ticker)
        card, md = _trade_card(
            i,
            trade,
            df,
            chart_rel=chart_rel,
            chart_dir=chart_dir,
            image_base=image_base,
            universe_top=universe_top,
        )
        cards.append(card)
        md_cards.append(md)
    empty = '<p class="empty">目前沒有符合條件的個股。</p>' if not results else ""
    heading = "台股一分K · 跌破 MA200"
    lead = (
        "成交額前 100、濾掉 ETF 與股價 600 以上。一分K MA5&lt;MA10&lt;MA20 空頭排列，"
        "當根收盤跌破 MA200 做空；13:00 後不再進場。"
        "出場：停利 1.2% / 停損 0.8% / 收盤站回 MA200 / 持有滿 30 分 / 當日收盤回補。"
        "K 棒漲紅跌綠。"
    )
    note = ""
    if github_md_url:
        note = (
            '<div class="note">網頁打不開時，請改開 GitHub 上這份報告'
            "（跟掃描頁同一種，K 線圖會顯示）："
            f'<a href="{html.escape(github_md_url)}">docs/tw/today.md</a></div>'
        )
    html_page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>{html.escape(title)}</title>
  <style>
{PAGE_CSS}
  </style>
</head>
<body>
  <div class="page">
    {note}
    <h1>{html.escape(heading)}</h1>
    <p class="lead">{lead}</p>
    <div class="chips">
      <span class="chip">不含 ETF</span>
      <span class="chip">股價 ≤ 600</span>
      <span class="chip">週成交額前 100</span>
      <span class="chip">MA5 &lt; 10 &lt; 20 空頭</span>
      <span class="chip">當根跌破 MA200</span>
      <span class="chip">13:00 後不進</span>
      <span class="chip">當沖回補</span>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#ffa726"></i>MA5</span>
      <span><i class="swatch" style="background:#ffeb3b"></i>MA10</span>
      <span><i class="swatch" style="background:#66bb6a"></i>MA20</span>
      <span><i class="swatch" style="background:#ce93d8"></i>MA200</span>
    </div>
    <div class="summary">
      回測 <span class="ok">{stats.get("trades", 0)}</span> 筆 · 下圖最近 {len(show)} 筆<br/>
      勝率 {stats.get("win_rate", 0) * 100:.1f}% · 平均 {stats.get("avg_pnl_pct", 0) * 100:+.2f}% · 獲利因子 {html.escape(pf_txt)}<br/>
      總計 {stats.get("total_pnl_twd", 0):+,.0f} · MDD {stats.get("max_drawdown_twd", 0):,.0f}（每筆 10 萬名義）<br/>
      {html.escape(subtitle)}
    </div>
    {''.join(cards)}{empty}
    <footer>僅供研究，不構成投資建議。代號可開 Yahoo 報價。</footer>
  </div>
</body>
</html>
"""
    md_lines = [
        f"# {heading}",
        "",
        "成交額前 100、濾掉 ETF 與股價 600 以上。一分K MA5<MA10<MA20 空頭排列，當根收盤跌破 MA200 做空；13:00 後不再進場。K 棒漲紅跌綠。",
        "",
        f"- 回測 **{stats.get('trades', 0)}** 筆 · 下圖最近 {len(show)} 筆",
        f"- 勝率 {stats.get('win_rate', 0) * 100:.1f}% · 平均 {stats.get('avg_pnl_pct', 0) * 100:+.2f}%",
        f"- {subtitle}",
        "",
        *md_cards,
    ]
    if not results:
        md_lines.append("目前沒有符合條件的個股。")
    return html_page, "\n".join(md_lines) + "\n"


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
    today_md = out.parent / "today.md"
    blob_target = (
        out.parent.parent / "tw" / "today.md"
        if out.parent.name == "tw-ma-short"
        else today_md
    )
    github_md_url = _github_blob_url(blob_target)
    html_page, markdown = build_report_html(
        results,
        frames,
        title=title,
        subtitle=subtitle,
        universe_top=universe_top,
        chart_rel=chart_rel,
        chart_dir=chart_dir,
        image_base=None,
        chart_trades=chart_trades,
        github_md_url=github_md_url,
    )
    out.write_text(html_page, encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    if out.stem in {"index", "today"}:
        if today_md.resolve() != md_path.resolve():
            shutil.copyfile(md_path, today_md)
    _mirror_screener_dir(out)
    return out


def _mirror_screener_dir(out: Path) -> None:
    """Copy the report to docs/tw/ so the GitHub URL matches the screener path."""
    if out.parent.name != "tw-ma-short":
        return
    dest = out.parent.parent / "tw"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "index.md", "today.md"):
        src = out.parent / name
        if src.exists():
            shutil.copyfile(src, dest / name)
    src_charts = out.parent / "charts"
    dest_charts = dest / "charts"
    if not src_charts.exists():
        return
    if dest_charts.exists():
        shutil.rmtree(dest_charts)
    shutil.copytree(src_charts, dest_charts)
