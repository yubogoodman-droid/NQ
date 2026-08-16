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
from nq.spring_report import MA_COLORS, MA_PERIODS, add_mas, chart_window, render_report_html, save_report
from nq.strategy import FakeBreakdownStrategy

def _naive_ts(ts):
    t = ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts
    return t.replace(tzinfo=None)


def save_png(df: pd.DataFrame, signal, trade, path: Path, title: str) -> Path:
    p = signal.pattern
    assert isinstance(p, FakeBreakdownPattern)
    w = chart_window(df, p)
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

    if df.index[p.range_start_idx] in w.index and df.index[p.range_end_idx] in w.index:
        x0 = int(w.index.get_loc(df.index[p.range_start_idx]))
        x1 = int(w.index.get_loc(df.index[p.range_end_idx]))
        ax.plot([x0, x1], [p.support, p.support], color="#42a5f5", ls="--", lw=1, label="支撐")
        ax.plot([x0, x1], [p.resistance, p.resistance], color="#ffa726", ls="--", lw=1.6, label="頸線")
        if p.box_high > p.resistance + 1e-6:
            ax.plot([x0, x1], [p.box_high, p.box_high], color="#ffa726", ls=":", lw=0.9, alpha=0.55, label="箱頂影線")
    ax.axhspan(p.support, p.resistance, color="#42a5f5", alpha=0.08)
    if df.index[p.spring_idx] in w.index:
        ax.scatter([int(w.index.get_loc(df.index[p.spring_idx]))], [p.spring_low], s=36, c="#ff5252", zorder=5, label="假跌破")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="假跌破訊號 1 分 K 圖")
    parser.add_argument("--date", default="2026-08-14")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--html", default="docs/spring_top50_20260814.html")
    parser.add_argument("--png-dir", default="output/spring_charts")
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/opt/cursor/artifacts/screenshots")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ymd = args.date.replace("-", "")
    universe = fetch_top_turnover(ymd, args.limit)
    cards = []
    for row in universe:
        code, name = row["code"], row["name"]
        df = add_mas(fetch_yahoo_1m(row["symbol"]))
        time.sleep(0.2)
        if df.empty:
            print(f"{code} {name}: 無 1 分 K")
            continue
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
            hhmm = (sig.timestamp.tz_convert("Asia/Taipei") if sig.timestamp.tzinfo else sig.timestamp).strftime("%H%M")
            png_name = f"{code}_{hhmm}_v2.png"
            png = save_png(df, sig, trade, png_dir / png_name, f"{code} {name}  {args.date} 1m")
            try:
                (artifact_dir / png_name).write_bytes(png.read_bytes())
            except OSError:
                pass
            print(f"圖: {png}")

    html_text = render_report_html(
        cards,
        title="2026-08-14 成交額前50 · 假跌破 1分K",
        summary="頸線改吃盤整收盤高（不吃上影）。已濾開盤雜訊。均線 MA5/10/20/60/120/200。",
    )
    out = save_report(args.html, html_text)
    print(f"HTML: {out.resolve()}  共 {len(cards)} 張")


if __name__ == "__main__":
    main()
