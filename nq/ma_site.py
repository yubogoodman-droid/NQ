"""南亞科均線回測 HTML 站：每筆一分 K + 六條均線（靜態 PNG）。"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

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


def _save_png(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def _style_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor("#10141a")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(True, color="#ffffff10", lw=0.6)


def _window(df: pd.DataFrame, trade: NanyaMaTrade) -> pd.DataFrame:
    work = add_nanya_features(df)
    start = max(0, trade.signal.bar_idx - 55)
    end = min(len(work) - 1, trade.signal.bar_idx + 35)
    for i in range(trade.signal.bar_idx, len(work)):
        if work.index[i] == trade.exit_time:
            end = min(len(work) - 1, i + 12)
            break
    return work.iloc[start : end + 1]


def _draw_trade_chart(df: pd.DataFrame, trade: NanyaMaTrade, trade_no: int, path: Path) -> Path:
    window = _window(df, trade)
    xs = np.arange(len(window))
    o = window["open"].to_numpy()
    h = window["high"].to_numpy()
    l = window["low"].to_numpy()
    c = window["close"].to_numpy()
    v = window["volume"].to_numpy()

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.15, 1]},
        facecolor="#161b22",
    )
    _style_axes(ax, axv)
    vol_colors = []
    for i in range(len(c)):
        up = c[i] >= o[i]
        col = "#ef5350" if up else "#26a69a"
        ax.vlines(xs[i], l[i], h[i], color=col, lw=0.8)
        y0, y1 = min(o[i], c[i]), max(o[i], c[i])
        if y1 == y0:
            y1 = y0 + max(h[i] - l[i], 1e-9) * 0.04
        ax.add_patch(Rectangle((xs[i] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        vol_colors.append("#ef535099" if up else "#26a69a99")
    axv.bar(xs, v, width=0.8, color=vol_colors, linewidth=0)

    for period in MA_PERIODS:
        col = f"ma{period}"
        if col not in window.columns:
            continue
        ax.plot(xs, window[col], color=MA_COLORS[period], lw=1.35 if period <= 20 else 1.1, label=f"MA{period}")

    ax.axhline(trade.signal.range_high, color="#8b949e", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(trade.signal.stop_loss, color="#ff5252", ls=":", lw=0.9)
    ax.axhline(trade.signal.target, color="#69f0ae", ls=":", lw=0.9)

    entry_mask = window.index == trade.signal.timestamp
    exit_mask = window.index == trade.exit_time
    entry_x = int(entry_mask.argmax()) if bool(entry_mask.any()) else None
    exit_x = int(exit_mask.argmax()) if bool(exit_mask.any()) else None
    if entry_x is not None:
        ax.scatter([entry_x], [trade.signal.entry], marker="^", s=46, color="#00e676", zorder=5, label="IN")
    if exit_x is not None:
        ax.scatter(
            [exit_x],
            [trade.exit_price],
            marker="x",
            s=42,
            color="#69f0ae" if trade.pnl_pct_net > 0 else "#ff5252",
            zorder=5,
            label="OUT",
        )

    ticks = list(range(0, len(window), max(1, len(window) // 6)))
    axv.set_xticks(ticks)
    axv.set_xticklabels([_naive(window.index[i]).strftime("%m-%d %H:%M") for i in ticks], rotation=0)
    ax.set_title(f"#{trade_no} {trade.symbol}  1m  {_fmt(trade.signal.timestamp)}", color="#e6edf3", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c9d1d9", ncol=5)
    fig.tight_layout(pad=0.45)
    return _save_png(fig, path)


def _draw_equity(trades: list[NanyaMaTrade], path: Path) -> Path | None:
    if not trades:
        return None
    ys = []
    acc = 0.0
    for t in trades:
        acc += t.pnl_pct_net * 100
        ys.append(acc)
    fig, ax = plt.subplots(figsize=(10.4, 2.6), facecolor="#161b22")
    _style_axes(ax)
    ax.plot(range(1, len(ys) + 1), ys, color="#79c0ff", lw=2, marker="o", ms=4)
    ax.axhline(0, color="#ffffff33", lw=0.8)
    ax.set_title("Net PnL %", color="#e6edf3", fontsize=11, loc="left")
    fig.tight_layout(pad=0.45)
    return _save_png(fig, path)


def _img_tag(rel: str, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="{html.escape(rel)}" />'


def build_backtest_site(
    *,
    title: str,
    trades: list[NanyaMaTrade],
    frames: dict[str, pd.DataFrame],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
    img_dir: Path,
) -> tuple[str, str]:
    overall = summarize_ma_trades(trades)
    img_dir.mkdir(parents=True, exist_ok=True)

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
    md_parts = [
        f"# {title}",
        "",
        "請直接開這個 Markdown（GitHub 會顯示圖）。htmlpreview 常常把圖擋掉。",
        "",
        f"- 成交 **{overall['trades']}**　勝率 **{overall['win_rate']*100:.0f}%**　淨損益 **{_pct(overall['total_pnl_pct_net'])}**　期望 **{_pct(overall['expectancy_net'])}**",
        "",
        "南亞科一分圖同款均線 MA5/10/20/60/120/200。進場是短均剛扇開，不是 436 末端。",
        "",
    ]
    eq_path = _draw_equity(trades, img_dir / "equity.png")
    eq_html = _img_tag("img/equity.png", "累計淨損益") if eq_path else ""
    if eq_path:
        md_parts += ["## 累計淨損益", "", "![equity](img/equity.png)", ""]

    for i, trade in enumerate(trades, start=1):
        df = frames.get(trade.symbol)
        slug = trade.symbol.replace("=", "").replace(".", "-")
        png_name = f"{i:02d}_{slug}.png"
        if df is not None and len(df):
            _draw_trade_chart(df, trade, i, img_dir / png_name)
            chart = _img_tag(f"img/{png_name}", f"{trade.symbol} {_fmt(trade.signal.timestamp)}")
            md_parts += [
                f"## #{i} {trade.symbol}　{_fmt(trade.signal.timestamp)}　{_pct(trade.pnl_pct_net)}",
                "",
                f"進場 `{trade.signal.entry:.2f}`　出場 `{trade.exit_price:.2f}`　{EXIT_ZH.get(trade.exit_reason, trade.exit_reason)}",
                "",
                f"MA5/10/20 `{trade.signal.ma5:.2f}` / `{trade.signal.ma10:.2f}` / `{trade.signal.ma20:.2f}`　離200 `{trade.signal.ext_200_pct*100:.2f}%`",
                "",
                f"![trade {i}](img/{png_name})",
                "",
            ]
        else:
            chart = "<p class='muted'>沒有K線</p>"
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

    html_page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
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
    .chart img, .eq img {{ width:100%; height:auto; display:block; border-radius:8px; background:#10141a; }}
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
    <div class="eq">{eq_html}</div>
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
    return html_page, "\n".join(md_parts)


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
    html_page, markdown = build_backtest_site(
        title=title,
        trades=trades,
        frames=frames,
        notes=notes,
        symbol_stats=symbol_stats,
        img_dir=out.parent / "img",
    )
    out.write_text(html_page, encoding="utf-8")
    (out.parent / "README.md").write_text(markdown, encoding="utf-8")
    return out
