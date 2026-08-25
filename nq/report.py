"""手機版交易卡片 HTML 報告。"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nq.backtest import TradeResult, run_backtest, summarize
from nq.strategy import NQWBottomStrategy

_IMG_PREFIX = "wbt"

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


def _trade_img_name(trade: TradeResult, trade_no: int) -> str:
    ts = trade.signal.timestamp
    if hasattr(ts, "tz_convert"):
        ts = ts.tz_convert("America/New_York")
    return f"{_IMG_PREFIX}{trade_no:02d}_{ts.strftime('%m%d_%H%M')}.png"


def _draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
) -> Path:
    """以整數 x 軸繪製靜態 K 線圖，避免盤中缺口造成連線扭曲。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    p = sig.pattern
    start, end = _chart_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for period in MA_PERIODS:
        col_name = f"ma{period}"
        if col_name not in window.columns:
            continue
        ma = window[col_name]
        if ma.notna().sum() == 0:
            continue
        ax.plot(list(xs), ma, color=MA_COLORS[period], lw=1.35 if period <= 20 else 1.05, label=f"MA{period}")

    ax.axhline(p.neckline, color="#ffa726", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(sig.stop_loss, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(sig.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    l1_rel = p.first_low_idx - start
    l2_rel = p.second_low_idx - start
    if 0 <= l1_rel < len(window):
        ax.scatter([l1_rel], [p.first_low], s=42, color="#42a5f5", zorder=5)
        ax.annotate("L1", (l1_rel, p.first_low), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= l2_rel < len(window):
        ax.scatter([l2_rel], [p.second_low], s=42, color="#ec407a", zorder=5)
        ax.annotate("L2", (l2_rel, p.second_low), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#f9a8d4", fontsize=8)

    entry_rel = window.index.get_indexer([sig.timestamp], method="nearest")[0]
    exit_rel = window.index.get_indexer([trade.exit_time], method="nearest")[0]
    if 0 <= entry_rel < len(window):
        ax.axvline(entry_rel, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([entry_rel], [sig.entry], s=48, color="#00e676", marker="^", zorder=6)
    if 0 <= exit_rel < len(window):
        ax.axvline(exit_rel, color="#f0c14b", ls=":", lw=0.9)
        exit_color = "#00c805" if trade.pnl_points > 0 else "#ff5252"
        ax.scatter([exit_rel], [trade.exit_price], s=44, color=exit_color, marker="x", zorder=6)

    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  W底  {_fmt_time(sig.timestamp)} → {_fmt_time(trade.exit_time)}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [_fmt_time(window.index[i]) for i in ticks],
        color="#8aa193",
        rotation=20,
        ha="right",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _render_trade_card(
    df: pd.DataFrame,
    trade: TradeResult,
    trade_no: int,
    *,
    img_href: str,
    contracts: int = 1,
) -> str:
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
      <div class="mini-chart"><img src="{html.escape(img_href)}" alt="trade #{trade_no}" loading="lazy" /></div>
    </article>
    """


def build_report_html(
    df: pd.DataFrame,
    results: list[TradeResult],
    *,
    title: str,
    symbol: str = "NQ=F",
    img_dir: Path | None = None,
    img_href_prefix: str = "img/",
) -> str:
    df = _add_mas(df)
    stats = summarize(results)
    cards_parts: list[str] = []
    for i, trade in enumerate(results, 1):
        img_name = _trade_img_name(trade, i)
        if img_dir is not None:
            _draw_trade_png(df, trade, img_dir / img_name, i)
        cards_parts.append(
            _render_trade_card(df, trade, i, img_href=f"{img_href_prefix}{img_name}")
        )
    cards = "".join(cards_parts)
    empty = '<div class="empty">今日未偵測到 W 底突破訊號</div>' if not results else ""

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
    .mini-chart img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
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

    out = Path(output)
    img_dir = out.parent / "img"
    content = build_report_html(
        df,
        results,
        title=title,
        symbol=symbol,
        img_dir=img_dir,
        img_href_prefix="img/",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
