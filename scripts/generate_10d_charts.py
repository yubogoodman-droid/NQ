#!/usr/bin/env python3
"""Generate one chart page per signal for the 10-day volume backtest."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_ta as ta

from generate_signal_charts import (
    HORIZONS,
    STEM_ALIAS,
    add_sma_cols,
    file_stem,
    plot_bundle,
    render_symbol_html,
    short_pnl_table,
    with_15m_bundle,
)

CACHE = Path("/tmp/binance_um_klines")
# candles kept around each signal (± hours)
PAD_HOURS = 18


def load_range(stem: str, start: str, end: str) -> pd.DataFrame:
    candidates = [stem]
    for src, dst in STEM_ALIAS.items():
        if stem == src:
            candidates.append(dst)
        if stem == dst:
            candidates.append(src)
    paths = []
    for s in candidates:
        paths.extend(sorted(CACHE.glob(f"{s}-5m-*.csv")))
    keep = []
    for p in paths:
        day = p.name.split("-5m-")[-1].replace(".csv", "")
        if start <= day <= end:
            keep.append(p)
    if not keep:
        raise FileNotFoundError(stem)
    df = pd.concat([pd.read_csv(p) for p in keep], ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def signal_filename(symbol: str, time_utc: str) -> str:
    stem = file_stem(symbol)
    ts = time_utc.replace("-", "").replace(":", "").replace(" ", "_")
    return f"{stem}_{ts}.html"


def chart_payload_one(symbol: str, row: pd.Series, df: pd.DataFrame) -> dict:
    signal_ts = int(pd.Timestamp(row["time_utc"], tz="UTC").timestamp() * 1000)
    pad_ms = PAD_HOURS * 3600 * 1000
    lo, hi = signal_ts - pad_ms, signal_ts + pad_ms
    plot = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].copy()
    if plot.empty:
        plot = df.copy()

    b5 = plot_bundle(plot)
    b15 = with_15m_bundle(df, lo, hi)

    one = pd.DataFrame([row])
    signals = short_pnl_table(df, one)
    if signals and "pnl_1h" in row.index and pd.notna(row.get("pnl_1h")):
        for h in HORIZONS:
            col = f"pnl_{h}"
            if col in row.index and pd.notna(row[col]):
                signals[0]["pnl"][h] = round(float(row[col]), 2)
        if "vol_ratio" in row.index and pd.notna(row.get("vol_ratio")):
            signals[0]["vol_ratio"] = float(row["vol_ratio"])
        if "neck_chg_pct" in row.index and pd.notna(row.get("neck_chg_pct")):
            signals[0]["neck_chg_pct"] = float(row["neck_chg_pct"])

    day_label = str(row["time_utc"])[:16]
    return {
        "symbol": f"{symbol} · {day_label}",
        "day": day_label,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        **b5,
        "tf15": b15,
        "signals": signals,
    }


def _pct_cell(v) -> tuple[str, str]:
    if v is None:
        return "na", "—"
    cls = "pos" if v >= 0 else "neg"
    return cls, f"{v:+.2f}%"


def render_signal_index(cards: list[dict], extra_nav: str) -> str:
    cards_sorted = sorted(cards, key=lambda x: x["time_utc"], reverse=True)
    items = []
    for c in cards_sorted:
        c1, t1 = _pct_cell(c.get("pnl_1h"))
        c4, t4 = _pct_cell(c.get("pnl_4h"))
        c8, t8 = _pct_cell(c.get("pnl_8h"))
        vol = "—" if c.get("vol_ratio") is None else f"{c['vol_ratio']:.2f}×"
        items.append(
            f"""
        <a class="card" href="./{c['href']}">
          <div class="name">{c['symbol']}</div>
          <div class="row"><span>時間</span><b class="mono">{c['time_utc'][5:]}</b></div>
          <div class="row"><span>爆量</span><b>{vol}</b></div>
          <div class="pills">
            <span>1h <b class="{c1}">{t1}</b></span>
            <span>4h <b class="{c4}">{t4}</b></span>
            <span>8h <b class="{c8}">{t8}</b></span>
          </div>
        </a>"""
        )
    body = "\n".join(items)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>10日爆量訊號圖 · 一訊一圖</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {{ --bg0:#0c1210; --bg1:#14201b; --ink:#e8f0ea; --muted:#8aa193; --line:rgba(232,240,234,0.12);
    --long:#3dba7a; --short:#e35d5d; --accent:#c9a227; }}
  body {{ margin:0; font-family:"IBM Plex Sans",sans-serif; color:var(--ink);
    background: radial-gradient(900px 500px at 10% -10%, rgba(201,162,39,.18), transparent 55%), linear-gradient(165deg,var(--bg0),var(--bg1)); min-height:100vh; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:40px 20px 64px; }}
  h1 {{ font-family:"IBM Plex Serif",serif; font-size:clamp(1.8rem,3.5vw,2.4rem); margin:0 0 8px; }}
  .sub {{ color:var(--muted); max-width:44rem; line-height:1.55; margin-bottom:22px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px 22px; margin-bottom:22px; color:var(--muted); font-size:.9rem; }}
  .stats b {{ color:var(--ink); font-family:"JetBrains Mono",monospace; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }}
  a.card {{ display:block; text-decoration:none; color:inherit; border:1px solid var(--line); padding:16px;
    background:linear-gradient(180deg, rgba(255,255,255,.04), transparent 55%), rgba(20,32,27,.65);
    transition: border-color .15s, transform .15s; }}
  a.card:hover {{ border-color:rgba(201,162,39,.55); transform:translateY(-2px); }}
  .name {{ font-family:"IBM Plex Serif",serif; font-size:1.15rem; margin-bottom:10px; }}
  .row {{ display:flex; justify-content:space-between; gap:10px; font-size:.86rem; color:var(--muted); margin-top:6px; }}
  .row b {{ color:var(--ink); font-family:"JetBrains Mono",monospace; font-size:.8rem; }}
  .pills {{ display:flex; justify-content:space-between; gap:6px; margin-top:12px; font-size:.72rem; color:var(--muted); }}
  .pills b {{ font-family:"JetBrains Mono",monospace; font-size:.78rem; }}
  .pos {{ color:var(--long); }} .neg {{ color:var(--short); }} .na {{ color:var(--muted); }}
  .mono {{ font-family:"JetBrains Mono",monospace; }}
</style>
</head>
<body>
  <div class="wrap">
    {extra_nav}
    <h1>10日爆量訊號 · 一訊一圖</h1>
    <p class="sub">滾動24h Top10 · 爆量≥2.5×（破位棒或4h峰值）+ 結構過濾 · 一訊一圖（±{PAD_HOURS}h）。</p>
    <div class="stats">
      <span>訊號總數 <b>{len(cards)}</b></span>
      <span>幣種 <b>{len({c['symbol'] for c in cards})}</b></span>
    </div>
    <div class="grid">
      {body}
    </div>
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/workspace/output/shadow_neckline_volume_10d.csv",
    )
    parser.add_argument("--out", default="/workspace/docs/charts/ten_day")
    parser.add_argument("--start", default=None, help="kline day start YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="kline day end YYYY-MM-DD")
    parser.add_argument("--label", default="10日", help="index title label")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="rebuild index.html from CSV without rewriting chart pages",
    )
    args = parser.parse_args()

    sig = pd.read_csv(args.csv).sort_values(["time_utc", "symbol"]).reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        days = sorted(sig["day"].astype(str).unique()) if "day" in sig.columns else []
        if days:
            start = (pd.Timestamp(days[0]) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            end = days[-1]
        else:
            start, end = "2026-07-31", "2026-08-10"

    # clear old multi-signal pages (STEM.html without timestamp)
    for p in out.glob("*.html"):
        if p.name == "index.html":
            continue
        if re.match(r"^.+_\d{8}_\d{4}\.html$", p.name):
            continue
        p.unlink()

    need_cols = ["symbol", "time_utc", "price", "bias", "line_val", "sma14"]
    for opt in ("vol_ratio", "close_break_pct", "neck_chg_pct", "dist_ma99_pct", "dist_ma200_pct"):
        if opt in sig.columns:
            need_cols.append(opt)
    for h in HORIZONS:
        col = f"pnl_{h}"
        if col in sig.columns:
            need_cols.append(col)

    nav = (
        '<div class="stats" style="margin-bottom:18px">'
        '<a href="../index.html" style="color:#c9a227;text-decoration:none">← 回總覽</a> · '
        '<a href="../ten_day.html" style="color:#8aa193;text-decoration:none">10日摘要</a> · '
        '<a href="../thirty_day.html" style="color:#8aa193;text-decoration:none">30日摘要</a> · '
        '<a href="./index.html" style="color:#8aa193;text-decoration:none">一訊一圖</a>'
        "</div>"
    )

    cards = []
    df_cache: dict[str, pd.DataFrame] = {}

    for _, row in sig.iterrows():
        symbol = row["symbol"]
        href = signal_filename(symbol, row["time_utc"])
        if not args.index_only:
            stem = file_stem(symbol)
            if stem not in df_cache:
                df = load_range(stem, start, end)
                for n in (7, 14, 25, 99, 200):
                    df[f"sma{n}"] = ta.sma(df["close"], length=n)
                df_cache[stem] = df
            df = df_cache[stem]
            data = chart_payload_one(symbol, row[need_cols], df)
            html = render_symbol_html(
                data,
                index_href="./index.html",
                badge="一訊一圖 · VOL≥2.5×",
                filter_note=f"單一訊號：{row['time_utc']} UTC · 圖面 ±{PAD_HOURS}h · 紅箭=訊號 / 黃點=進場。",
            )
            (out / href).write_text(html, encoding="utf-8")
            print(f"[{args.label}] {href}")

        pnl_1h = float(row["pnl_1h"]) if "pnl_1h" in row and pd.notna(row["pnl_1h"]) else None
        pnl_4h = float(row["pnl_4h"]) if "pnl_4h" in row and pd.notna(row["pnl_4h"]) else None
        pnl_8h = float(row["pnl_8h"]) if "pnl_8h" in row and pd.notna(row["pnl_8h"]) else None
        vol = float(row["vol_ratio"]) if "vol_ratio" in row and pd.notna(row["vol_ratio"]) else None
        cards.append(
            {
                "symbol": symbol,
                "href": href,
                "time_utc": row["time_utc"],
                "pnl_1h": pnl_1h,
                "pnl_4h": pnl_4h,
                "pnl_8h": pnl_8h,
                "vol_ratio": vol,
            }
        )

    if not args.index_only:
        keep = {c["href"] for c in cards}
        for p in out.glob("*.html"):
            if p.name == "index.html":
                continue
            if p.name not in keep:
                p.unlink()

    index_html = render_signal_index(cards, nav)
    index_html = index_html.replace("10日爆量訊號圖", f"{args.label}爆量訊號圖").replace(
        "10日爆量訊號", f"{args.label}爆量訊號"
    )
    # inject actual window into subtitle
    day0 = str(sig["day"].min()) if "day" in sig.columns and len(sig) else start
    day1 = str(sig["day"].max()) if "day" in sig.columns and len(sig) else end
    index_html = index_html.replace(
        "滾動24h Top10 · 爆量≥2.5×（破位棒或4h峰值）+ 結構過濾 · 一訊一圖",
        f"{day0} → {day1} UTC · 滾動24h Top10 · 爆量≥2.5×（破位棒或4h峰值）+ 結構過濾 · 一訊一圖",
    )
    (out / "index.html").write_text(index_html, encoding="utf-8")
    print("done", len(cards), "charts ->", out)


if __name__ == "__main__":
    main()
