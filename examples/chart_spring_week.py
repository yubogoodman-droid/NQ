#!/usr/bin/env python3
"""把近一週假跌破回測訊號畫成 1 分 K（底下 5 分 K）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.chart_spring_top50 import _copy, render_pages_html, save_png
from examples.scan_tw_top50_spring import fetch_yahoo_1m, tw_tick_size
from nq.backtest import run_backtest
from nq.spring_report import add_mas, resample_to_5m, session_window
from nq.strategy import FakeBreakdownStrategy


def _local(ts):
    return ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts


def main() -> None:
    parser = argparse.ArgumentParser(description="近一週假跌破訊號圖")
    parser.add_argument("--json", default="output/spring_scan_week_20260831_0904.json")
    parser.add_argument("--pages-html", default="docs/spring/week/index.html")
    parser.add_argument("--png-dir", default="output/spring_charts/week")
    parser.add_argument("--yahoo-range", default="8d")
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--title", default="假跌破 · 近一週 1分K（底下 5分K）")
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    wanted = {
        (h["code"], s["date"], s["time"])
        for h in data["hits"]
        for s in h["signals"]
    }
    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/opt/cursor/artifacts/screenshots")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = Path(args.pages_html).parent
    pages_dir.mkdir(parents=True, exist_ok=True)

    gallery = []
    for hit in data["hits"]:
        code, name, symbol = hit["code"], hit["name"], hit["symbol"]
        df1 = fetch_yahoo_1m(symbol, range=args.yahoo_range)
        time.sleep(args.sleep)
        if df1.empty:
            print(f"{code} {name}: 無 1 分 K")
            continue
        df = add_mas(df1)
        df5 = add_mas(resample_to_5m(df1))
        strategy = FakeBreakdownStrategy(tick_size=tw_tick_size(float(df["close"].iloc[-1])))
        trades = run_backtest(df, strategy, max_bars_hold=60)
        by_bar = {t.signal.bar_idx: t for t in trades}
        for sig in strategy.generate_signals(df):
            local = _local(sig.timestamp)
            key = (code, local.strftime("%Y-%m-%d"), local.strftime("%H:%M"))
            if key not in wanted:
                continue
            trade = by_bar.get(sig.bar_idx)
            ymd = local.strftime("%Y%m%d")
            hhmm = local.strftime("%H%M")
            png_name = f"{code}_{ymd}_{hhmm}_v2.png"
            png5_name = f"{code}_{ymd}_{hhmm}_5m.png"
            day = local.strftime("%Y-%m-%d")
            png = save_png(df, sig, trade, png_dir / png_name, f"{code} {name}  {day} 1m")
            win5 = session_window(df5, sig.timestamp)
            png5 = None
            if not win5.empty:
                png5 = save_png(
                    df5,
                    sig,
                    trade,
                    png_dir / png5_name,
                    f"{code} {name}  {day} 5m",
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
                    "label": f"{code} {name}",
                    "time": f"{local.strftime('%m-%d')} {local.strftime('%H:%M')}",
                    "sort": local.isoformat(),
                    "tag": tag,
                    "detail": (
                        f"進 {sig.entry:g} / 停 {sig.stop_loss:g} / 目標 {sig.target:g}"
                        f" · 跌破 {p.break_pct * 100:.2f}% · 量 {p.volume_ratio:.2f}x"
                    ),
                    "png1": png_name,
                    "png5": png5_name if png5 is not None else "",
                    "feature": trade is not None and trade.exit_reason == "take_profit",
                }
            )

    gallery.sort(key=lambda item: (not item["feature"], item["sort"]))
    dates = data.get("dates") or []
    stats = data.get("summary") or {}
    span = f"{dates[0]}～{dates[-1]}" if dates else ""
    summary = (
        f"{span} 每日成交額前 100、收盤 > 700 剔除。"
        f"{stats.get('signals', len(gallery))} 筆：TP {stats.get('tp', 0)} / SL {stats.get('sl', 0)} / TIME {stats.get('time', 0)}"
        f"，合計 {stats.get('pnl_sum', 0):+.2f} 點、{stats.get('r_sum', 0):+.2f}R。"
        "上面 1 分 K，下面同一段 5 分 K。"
    )
    pages = Path(args.pages_html)
    pages.write_text(
        render_pages_html(
            gallery,
            title=args.title,
            summary=summary,
        ),
        encoding="utf-8",
    )
    print(f"Pages: {pages.resolve()}  共 {len(gallery)} 張")


if __name__ == "__main__":
    main()
