#!/usr/bin/env python3
"""把上週五成交額前 50 的假跌破訊號畫成 1 分 K 報告。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

from examples.scan_tw_top50_spring import fetch_top_turnover, fetch_yahoo_1m, tw_tick_size
from nq.backtest import run_backtest
from nq.spring import FakeBreakdownPattern
from nq.spring_report import (
    MA_COLORS,
    MA_PERIODS,
    add_mas,
    chart_window,
    render_report_html,
    resample_to_5m,
    save_report,
    session_window,
)
from nq.strategy import FakeBreakdownStrategy

def _naive_ts(ts):
    t = ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts
    return t.replace(tzinfo=None)


def save_png(df: pd.DataFrame, signal, trade, path: Path, title: str, *, window=None, source_df=None) -> Path:
    p = signal.pattern
    assert isinstance(p, FakeBreakdownPattern)
    src = source_df if source_df is not None else df
    w = chart_window(df, p) if window is None else window
    if w.empty:
        raise ValueError("empty chart window")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.04},
        facecolor="#0b0e11",
    )
    ax, axv = axes
    for a in axes:
        a.set_facecolor("#161b22")
        a.tick_params(colors="#c9d1d9")
        for spine in a.spines.values():
            spine.set_color("#30363d")
        a.grid(True, color="#ffffff", alpha=0.06)

    for i, (_, row) in enumerate(w.iterrows()):
        up = row["close"] >= row["open"]
        color = "#ef5350" if up else "#26a69a"
        ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
        bottom = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), 0.01)
        ax.add_patch(plt.Rectangle((i - 0.35, bottom), 0.7, height, facecolor=color, edgecolor=color, linewidth=0))
        axv.bar(i, row["volume"], width=0.7, color=color, alpha=0.85)

    xs = list(range(len(w)))
    ma_lows = []
    ma_highs = []
    for period in MA_PERIODS:
        col = f"ma{period}"
        if col not in w.columns or not w[col].notna().any():
            continue
        series = w[col]
        ma_lows.append(float(series.min()))
        ma_highs.append(float(series.max()))
        ax.plot(xs, series, color="#0b0e11", lw=5.0 if period <= 20 else 4.2, zorder=18, solid_capstyle="round")
        ax.plot(
            xs,
            series,
            color=MA_COLORS[period],
            lw=2.8 if period <= 20 else 2.4,
            label=f"MA{period}",
            zorder=19,
            solid_capstyle="round",
        )
    y0 = float(w["low"].min())
    y1 = float(w["high"].max())
    if ma_lows:
        y0 = min(y0, min(ma_lows))
        y1 = max(y1, max(ma_highs))
    pad = (y1 - y0) * 0.04 or 1
    ax.set_ylim(y0 - pad, y1 + pad)

    def loc(ts: pd.Timestamp) -> int:
        return int(w.index.get_indexer([ts], method="nearest")[0])

    if src.index[p.range_start_idx] is not None:
        x0 = loc(src.index[p.range_start_idx])
        x1 = loc(src.index[p.range_end_idx])
        ax.plot([x0, x1], [p.support, p.support], color="#42a5f5", ls="--", lw=1)
        ax.plot([x0, x1], [p.resistance, p.resistance], color="#ffa726", ls="--", lw=1)
    ax.axhspan(p.support, p.resistance, color="#42a5f5", alpha=0.08)
    spring_ts = src.index[p.spring_idx]
    if w.index.get_indexer([spring_ts], method="nearest")[0] >= 0:
        ax.scatter([loc(spring_ts)], [p.spring_low], s=36, c="#ff5252", zorder=5, label="假跌破")
    ax.scatter([loc(signal.timestamp)], [signal.entry], marker="^", s=70, c="#00e676", zorder=6, label="進場")
    if trade is not None:
        ax.scatter(
            [loc(trade.exit_time)],
            [trade.exit_price],
            marker="x",
            s=50,
            c="#69f0ae" if trade.pnl_points > 0 else "#ff5252",
            zorder=6,
        )
    ax.axhline(signal.stop_loss, color="#ff5252", ls=":", lw=1, alpha=0.7)
    ax.axhline(signal.target, color="#00c805", ls=":", lw=1, alpha=0.7)
    ax.set_title(title, color="#e6edf3", loc="left", fontsize=12)
    ax.legend(
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        fontsize=7.5,
        ncol=4,
        loc="upper left",
        framealpha=0.85,
    )
    ticks = list(range(0, len(w), max(1, len(w) // 6)))
    axv.set_xticks(ticks)
    axv.set_xticklabels([_naive_ts(w.index[i]).strftime("%H:%M") for i in ticks], color="#c9d1d9")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_pages_html(gallery: list[dict], *, title: str, summary: str) -> str:
    cards = []
    for item in gallery:
        feature = ' feature' if item.get("feature") else ""
        extra = ""
        if item.get("png5"):
            extra = (
                f'<div class="tf">5分K</div>'
                f'<img src="{item["png5"]}" alt="{item["label"]} 5m" />'
            )
        cards.append(
            f"""
    <article class="card{feature}">
      <h2>{item["label"]} · {item["time"]} {item["tag"]}</h2>
      <p>{item["detail"]}</p>
      <div class="tf">1分K</div>
      <img src="{item["png1"]}" alt="{item["label"]} 1m" />
      {extra}
    </article>"""
        )
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0b0e11; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif; }}
    .page {{ max-width: 720px; margin: 0 auto; padding: 16px 12px 32px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    .sub {{ color: #8b949e; font-size: 13px; line-height: 1.55; margin: 0 0 10px; }}
    .mas {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; font-size: 12px; font-weight: 700; }}
    .mas span {{ padding: 4px 8px; border-radius: 8px; background: #161b22; border: 1px solid #30363d; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 12px; margin-bottom: 14px; }}
    .card.feature {{ border-color: #ef5350; }}
    .card h2 {{ font-size: 15px; margin: 0 0 8px; }}
    .card p {{ font-size: 12px; color: #8b949e; margin: 0 0 8px; line-height: 1.5; }}
    .tf {{ font-size: 11px; font-weight: 700; color: #79c0ff; margin: 10px 0 6px; }}
    .card img {{ width: 100%; height: auto; border-radius: 10px; display: block; background: #0d1117; }}
    .tag {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-right: 6px; }}
    .time {{ background: rgba(255,193,7,.12); color: #f0c14b; }}
    .tp {{ background: rgba(0,200,5,.15); color: #3ddc68; }}
    .sl {{ background: rgba(255,82,82,.15); color: #ff7b72; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{title}</h1>
    <p class="sub">{summary}</p>
    <div class="mas">
      <span style="color:#ffa726">MA5</span>
      <span style="color:#ffeb3b">MA10</span>
      <span style="color:#66bb6a">MA20</span>
      <span style="color:#42a5f5">MA60</span>
      <span style="color:#26c6da">MA120</span>
      <span style="color:#ab47bc">MA200</span>
    </div>
    {body}
  </div>
</body>
</html>
"""


def _copy(src: Path, *dests: Path) -> None:
    data = src.read_bytes()
    for dest in dests:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="假跌破訊號 1 分 K 圖（底下附 5 分 K）")
    parser.add_argument("--date", default="2026-08-14")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--html", default="docs/spring_top50_20260814.html")
    parser.add_argument("--pages-html", default="docs/spring/index.html")
    parser.add_argument("--png-dir", default="output/spring_charts")
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/opt/cursor/artifacts/screenshots")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = Path(args.pages_html).parent
    pages_dir.mkdir(parents=True, exist_ok=True)

    ymd = args.date.replace("-", "")
    universe = fetch_top_turnover(ymd, args.limit)
    cards = []
    gallery = []
    for row in universe:
        code, name = row["code"], row["name"]
        df1 = fetch_yahoo_1m(row["symbol"])
        time.sleep(0.2)
        if df1.empty:
            print(f"{code} {name}: 無 1 分 K")
            continue
        df = add_mas(df1)
        df5 = add_mas(resample_to_5m(df1))
        strategy = FakeBreakdownStrategy(tick_size=tw_tick_size(float(df["close"].iloc[-1])))
        signals = [
            s
            for s in strategy.generate_signals(df)
            if (s.timestamp.tz_convert("Asia/Taipei") if s.timestamp.tzinfo else s.timestamp).strftime("%Y-%m-%d")
            == args.date
        ]
        if not signals:
            print(f"{code} {name}: 無訊號")
            continue
        trades = run_backtest(df, strategy, max_bars_hold=60)
        by_bar = {t.signal.bar_idx: t for t in trades}
        for sig in signals:
            trade = by_bar.get(sig.bar_idx)
            label = f"{code} {name}"
            cards.append((label, df, sig, trade))
            local = sig.timestamp.tz_convert("Asia/Taipei") if sig.timestamp.tzinfo else sig.timestamp
            hhmm = local.strftime("%H%M")
            png_name = f"{code}_{hhmm}_v2.png"
            png5_name = f"{code}_{hhmm}_5m.png"
            png = save_png(df, sig, trade, png_dir / png_name, f"{code} {name}  {args.date} 1m")
            win5 = session_window(df5, sig.timestamp)
            png5 = None
            if not win5.empty:
                png5 = save_png(
                    df5,
                    sig,
                    trade,
                    png_dir / png5_name,
                    f"{code} {name}  {args.date} 5m",
                    window=win5,
                    source_df=df,
                )
            _copy(png, artifact_dir / png_name, pages_dir / png_name)
            if png5 is not None:
                _copy(png5, artifact_dir / png5_name, pages_dir / png5_name)
            print(f"圖: {png}" + (f"  5m: {png5}" if png5 else ""))
            if trade is None:
                tag = '<span class="tag time">訊號</span>'
            elif trade.exit_reason == "take_profit":
                tag = f'<span class="tag tp">TP {trade.pnl_points:+.2f}</span>'
            elif trade.exit_reason == "stop_loss":
                tag = f'<span class="tag sl">SL {trade.pnl_points:+.2f}</span>'
            else:
                tag = f'<span class="tag time">TIME {trade.pnl_points:+.2f}</span>'
            p = sig.pattern
            gallery.append(
                {
                    "label": label,
                    "time": local.strftime("%H:%M"),
                    "tag": tag,
                    "detail": (
                        f"進 {sig.entry:g} / 停 {sig.stop_loss:g} / 目標 {sig.target:g}"
                        f" · 跌破 {p.break_pct * 100:.2f}% · 量 {p.volume_ratio:.2f}x"
                    ),
                    "png1": png_name,
                    "png5": png5_name if png5 is not None else "",
                    "feature": code == "8358",
                }
            )

    html_text = render_report_html(
        cards,
        title="2026-08-14 成交額前50 · 假跌破 1分K",
        summary="訊號用 1 分 K。每張底下附同一段 5 分 K。已濾開盤雜訊；站回後最多等 24 根才突破（對齊金居 8/14）。",
    )
    out = save_report(args.html, html_text)
    print(f"HTML: {out.resolve()}  共 {len(cards)} 張")
    pages = Path(args.pages_html)
    pages.write_text(
        render_pages_html(
            gallery,
            title="假跌破 · 1分K（底下 5分K）",
            summary="2026-08-14 成交額前 50。上面 1 分 K 是進場訊號，下面 5 分 K 看同一段結構。站回後最多等 24 根才突破（金居 09:35 站回、09:53 放量站上箱頂）。",
        ),
        encoding="utf-8",
    )
    print(f"Pages: {pages.resolve()}")


if __name__ == "__main__":
    main()
