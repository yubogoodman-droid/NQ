#!/usr/bin/env python3
"""Generate per-symbol charts for the 10-day volume-tier backtest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_ta as ta

from generate_signal_charts import (
    HORIZONS,
    STEM_ALIAS,
    file_stem,
    render_index,
    render_symbol_html,
    short_pnl_table,
)

CACHE = Path("/tmp/binance_um_klines")
SIG_CSV = Path("/workspace/output/shadow_neckline_volume_10d.csv")
OUT_DIR = Path("/workspace/docs/charts/ten_day")
START = "2026-07-31"
END = "2026-08-10"
PLOT_FROM = "2026-08-01 00:00:00"


def load_range(stem: str) -> pd.DataFrame:
    candidates = [stem]
    for src, dst in STEM_ALIAS.items():
        if stem == src:
            candidates.append(dst)
        if stem == dst:
            candidates.append(src)
    paths = []
    for s in candidates:
        paths.extend(sorted(CACHE.glob(f"{s}-5m-*.csv")))
    # keep only START..END
    keep = []
    for p in paths:
        day = p.name.split("-5m-")[-1].replace(".csv", "")
        if START <= day <= END:
            keep.append(p)
    if not keep:
        raise FileNotFoundError(stem)
    df = pd.concat([pd.read_csv(p) for p in keep], ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def chart_payload(symbol: str, signal_rows: pd.DataFrame) -> dict:
    stem = file_stem(symbol)
    df = load_range(stem)
    for n in (7, 14, 25, 99, 200):
        df[f"sma{n}"] = ta.sma(df["close"], length=n)

    start_ts = int(pd.Timestamp(PLOT_FROM, tz="UTC").timestamp() * 1000)
    plot = df[df["timestamp"] >= start_ts].copy()

    def series(col: str):
        return [
            {"time": int(r["timestamp"] // 1000), "value": float(r[col])}
            for _, r in plot.iterrows()
            if pd.notna(r[col])
        ]

    candles = [
        {
            "time": int(r["timestamp"] // 1000),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        for _, r in plot.iterrows()
    ]
    signals = short_pnl_table(df, signal_rows)
    # prefer csv pnl if present
    if "pnl_1h" in signal_rows.columns:
        by_time = {r["time_utc"]: r for _, r in signal_rows.iterrows()}
        for s in signals:
            src = by_time.get(s["time_utc"])
            if src is None:
                continue
            for h in HORIZONS:
                col = f"pnl_{h}"
                if col in src and pd.notna(src[col]):
                    s["pnl"][h] = round(float(src[col]), 2)
            if "vol_ratio" in src and pd.notna(src["vol_ratio"]):
                s["vol_ratio"] = float(src["vol_ratio"])
            if "neck_chg_pct" in src and pd.notna(src["neck_chg_pct"]):
                s["neck_chg_pct"] = float(src["neck_chg_pct"])
    return {
        "symbol": symbol,
        "day": f"{START[5:]}→{END[5:]}",
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "candles": candles,
        "sma7": series("sma7"),
        "sma14": series("sma14"),
        "sma25": series("sma25"),
        "sma99": series("sma99"),
        "sma200": series("sma200"),
        "signals": signals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(SIG_CSV))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    sig = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    need = ["symbol", "time_utc", "price", "bias", "line_val", "sma14"]
    for opt in ("vol_ratio", "close_break_pct", "neck_chg_pct", "dist_ma99_pct", "dist_ma200_pct"):
        if opt in sig.columns:
            need.append(opt)

    nav = (
        '<div class="stats" style="margin-bottom:18px">'
        '<a href="../index.html" style="color:#c9a227;text-decoration:none">← 回總覽</a> · '
        '<a href="../ten_day.html" style="color:#8aa193;text-decoration:none">10日摘要</a> · '
        '<a href="./index.html" style="color:#8aa193;text-decoration:none">10日圖表</a>'
        "</div>"
    )

    cards = []
    for symbol, g in sig.groupby("symbol"):
        g = g.sort_values("time_utc")
        data = chart_payload(symbol, g[need])
        stem = file_stem(symbol)
        # use ASCII-safe filename for pages when needed
        href = f"{stem}.html"
        html = render_symbol_html(
            data,
            index_href="./index.html",
            badge="10D · VOL≥1.5×",
            filter_note="10日回測：原版 + 爆量≥1.5× + 拒絕上升頸線。區間 2026-08-01→08-10（Vision）。",
        )
        # patch plot note dates inside html (render uses global HIST) — replace if present
        html = html.replace("前置 K 線自 2026-08-08 18:00 UTC", "K 線自 2026-08-01 00:00 UTC")
        html = html.replace("前置 K 線自 2026-08-10 18:00 UTC", "K 線自 2026-08-01 00:00 UTC")
        (out / href).write_text(html, encoding="utf-8")
        pnls = [s["pnl"].get("1h") for s in data["signals"] if s["pnl"].get("1h") is not None]
        avg_1h = round(sum(pnls) / len(pnls), 2) if pnls else None
        times = " · ".join(t[5:16] for t in g["time_utc"].tolist())  # MM-DD HH:MM
        cards.append(
            {
                "symbol": symbol,
                "href": href,
                "n": len(data["signals"]),
                "avg_1h": avg_1h,
                "times": times,
            }
        )
        print(f"[ten_day] {href} signals={len(data['signals'])}")

    index = render_index(
        cards,
        title="10日爆量訊號圖表 · 2026-08-01→08-10",
        subtitle="Binance Vision · 原版 + 量能≥1.5× + 拒絕上升頸線。點幣種看 K 線。",
        extra_nav=nav,
    )
    (out / "index.html").write_text(index, encoding="utf-8")

    # refresh summary page link to charts
    summary = Path("/workspace/docs/charts/ten_day.html")
    if summary.exists():
        t = summary.read_text(encoding="utf-8")
        if 'href="./ten_day/index.html"' not in t:
            t = t.replace(
                "<h1>10 日回測</h1>",
                '<h1>10 日回測</h1>\n<p class="sub"><a href="./ten_day/index.html">查看 39 幣 K 線圖表 →</a></p>',
            )
            summary.write_text(t, encoding="utf-8")
    print("done ->", out, "symbols", len(cards), "signals", len(sig))


if __name__ == "__main__":
    main()
