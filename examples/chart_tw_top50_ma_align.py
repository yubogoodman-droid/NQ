#!/usr/bin/env python3
"""把成交額前 50 的日線多頭排列通知畫成報告。"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

for _fp in (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_fp).get_name(), "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.scan_tw_top50_ma_align import fetch_yahoo_daily, scan_stock
from examples.scan_tw_top50_spring import fetch_top_turnover, tw_tick_size
from nq.backtest import run_backtest
from nq.ma_align import MaAlignPattern, add_daily_mas
from nq.strategy import MaAlignStrategy

MA_COLORS = {5: "#ffa726", 10: "#ffeb3b", 20: "#66bb6a", 200: "#ab47bc"}


def _naive(ts):
    t = ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts
    return pd.Timestamp(t).tz_localize(None) if getattr(t, "tzinfo", None) else pd.Timestamp(t)


def save_png(df: pd.DataFrame, signal, trade, path: Path, title: str) -> Path:
    p = signal.pattern
    assert isinstance(p, MaAlignPattern)
    work = add_daily_mas(df)
    start = max(0, signal.bar_idx - 80)
    end = min(len(work) - 1, signal.bar_idx + 20)
    w = work.iloc[start : end + 1]
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor="#0b0e11")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#ffffff", alpha=0.06)

    for i, (_, row) in enumerate(w.iterrows()):
        up = row["close"] >= row["open"]
        color = "#ef5350" if up else "#26a69a"
        ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
        bottom = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), 0.01)
        ax.add_patch(plt.Rectangle((i - 0.35, bottom), 0.7, height, facecolor=color, edgecolor=color, linewidth=0))

    for period, color in MA_COLORS.items():
        ax.plot(range(len(w)), w[f"ma{period}"], color=color, lw=2.2 if period <= 20 else 2.6, label=f"MA{period}")

    loc = int(w.index.get_indexer([df.index[signal.bar_idx]], method="nearest")[0])
    ax.scatter([loc], [signal.entry], marker="^", s=80, c="#00e676", zorder=6, label="通知進場")
    ax.axhline(signal.stop_loss, color="#ff5252", ls=":", lw=1, alpha=0.7)
    ax.axhline(signal.target, color="#00c805", ls=":", lw=1, alpha=0.7)
    if trade is not None:
        eloc = int(w.index.get_indexer([trade.exit_time], method="nearest")[0])
        ax.scatter([eloc], [trade.exit_price], marker="x", s=50, c="#69f0ae" if trade.pnl_points > 0 else "#ff5252")
    ax.set_title(title, color="#e6edf3", loc="left", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8, ncol=5, loc="upper left")
    ticks = list(range(0, len(w), max(1, len(w) // 6)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_naive(w.index[i]).strftime("%m-%d") for i in ticks], color="#c9d1d9")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_html(cards: list, title: str, summary: str) -> str:
    sections = []
    for i, (label, df, sig, trade, png_name) in enumerate(cards, 1):
        p = sig.pattern
        pnl = ""
        tag = '<span class="tag tag-info">通知</span>'
        if trade is not None:
            cls = "pnl-win" if trade.pnl_points > 0 else "pnl-loss"
            pnl = f'<div class="card-pnl {cls}">{trade.pnl_points:+.2f}</div>'
            reason = {"take_profit": ("TP", "tag-tp"), "stop_loss": ("SL", "tag-sl")}.get(
                trade.exit_reason, ("TIME", "tag-time")
            )
            tag = f'<span class="tag {reason[1]}">{reason[0]}</span>'
        detail = (
            f"進場 {_naive(sig.timestamp).strftime('%Y-%m-%d')} @ {sig.entry:.2f}\n"
            f"停損 {sig.stop_loss:.2f} / 目標 {sig.target:.2f}\n"
            f"MA5 {p.ma5:.2f} > MA10 {p.ma10:.2f} > MA20 {p.ma20:.2f}\n"
            f"收盤 {p.close:.2f} > MA200 {p.ma200:.2f}"
        )
        png = html.escape(png_name)
        img = f'<img src="{png}" alt="{html.escape(label)}" />'
        sections.append(
            f"""
    <article class="card">
      <header class="card-header">
        <div><span class="trade-no">#{i} {html.escape(label)}</span>
        <span class="trade-time">{_naive(sig.timestamp).strftime('%Y-%m-%d')}</span></div>
        {pnl}
      </header>
      <div class="tags">{tag}<span class="tag tag-info">日線</span><span class="tag tag-info">5/10/20+200</span></div>
      <pre class="trade-detail">{html.escape(detail)}</pre>
      {img}
    </article>"""
        )
    body = "\n".join(sections) or '<div class="empty">這天沒有新通知</div>'
    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
body {{ margin:0; background:#0b0e11; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif; }}
.page {{ max-width:720px; margin:0 auto; padding:16px 12px 32px; }}
.summary {{ background:#161b22; border:1px solid #30363d; border-radius:14px; padding:14px 16px; margin-bottom:14px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:14px; padding:12px; margin-bottom:14px; }}
.card-header {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:8px; }}
.trade-no {{ font-weight:700; }} .trade-time {{ font-size:12px; color:#8b949e; margin-left:8px; }}
.pnl-win {{ color:#00c805; font-weight:700; }} .pnl-loss {{ color:#ff5252; font-weight:700; }}
.tag {{ font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; margin-right:6px; }}
.tag-tp {{ background:rgba(0,200,5,.15); color:#3ddc68; }} .tag-sl {{ background:rgba(255,82,82,.15); color:#ff7b72; }}
.tag-time {{ background:rgba(255,193,7,.12); color:#f0c14b; }} .tag-info {{ background:rgba(88,166,255,.12); color:#79c0ff; }}
.trade-detail {{ margin:0 0 10px; padding:10px 12px; background:#0d1117; border-radius:10px; font-family:ui-monospace,Menlo,monospace; font-size:12px; color:#c9d1d9; white-space:pre-wrap; }}
.card img {{ width:100%; height:auto; border-radius:10px; }}
.empty {{ text-align:center; color:#8b949e; padding:40px 16px; }}
</style></head><body><div class="page">
<section class="summary"><h1>{html.escape(title)}</h1><p>{html.escape(summary)}</p></section>
{body}
</div></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="日線多頭排列通知圖")
    parser.add_argument("--date", default="2026-08-14")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--html", default="docs/ma-align/index.html")
    parser.add_argument("--png-dir", default="docs/ma-align")
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    ymd = args.date.replace("-", "")
    universe = fetch_top_turnover(ymd, args.limit)
    cards = []
    aligned_names = []
    for row in universe:
        result = scan_stock(row, args.date)
        time.sleep(0.15)
        if result.get("aligned"):
            aligned_names.append(f"{row['code']}{row['name']}")
        if not result.get("alert"):
            continue
        df = fetch_yahoo_daily(row["symbol"])
        strategy = MaAlignStrategy(tick_size=tw_tick_size(float(df["close"].iloc[-1])))
        signals = [s for s in strategy.generate_signals(df) if _naive(s.timestamp).strftime("%Y-%m-%d") == args.date]
        trades = {t.signal.bar_idx: t for t in run_backtest(df, strategy, max_bars_hold=40)}
        for sig in signals:
            trade = trades.get(sig.bar_idx)
            png_name = f"{row['code']}_{_naive(sig.timestamp).strftime('%Y%m%d')}.png"
            png = save_png(df, sig, trade, png_dir / png_name, f"{row['code']} {row['name']}  日線")
            cards.append((f"{row['code']} {row['name']}", df, sig, trade, png_name))
            print(f"圖: {png}")

    summary = (
        f"條件：MA5>MA10>MA20 且收盤站上 MA200（當日才成立才跳通知）。"
        f" 已排列站上200：{'、'.join(aligned_names) if aligned_names else '無'}。"
    )
    text = render_html(cards, f"{args.date} 成交額前50 · 日線多頭排列通知", summary)
    out = Path(args.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"HTML: {out.resolve()}  通知 {len(cards)} 張")


if __name__ == "__main__":
    main()
