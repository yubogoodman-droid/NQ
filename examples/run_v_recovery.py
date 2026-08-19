#!/usr/bin/env python3
"""NQ 一分 K 均線多頭排列：近一週有多少訊號。"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_stack import (  # noqa: E402
    MA_PERIODS,
    StackSignal,
    add_indicators,
    count_stack_events,
    ladder_counts,
)

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    30: "#26a69a",
    60: "#42a5f5",
    100: "#7e57c2",
    120: "#26c6da",
    200: "#ffffff",
}


def fetch_nq_1m(symbol: str = "NQ=F", period: str = "7d") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(symbol).history(period=period, interval="1m", auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"無法取得 {symbol} 一分 K")
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _fmt(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m-%d %H:%M")


def _stem(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m%d_%H%M")


def _fwd(df: pd.DataFrame, idx: int, minutes: int) -> float | None:
    j = idx + minutes
    if j >= len(df):
        return None
    return float(df["close"].iloc[j] - df["close"].iloc[idx])


def draw_stack(df: pd.DataFrame, sig: StackSignal, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    start = max(0, sig.idx - 50)
    end = min(len(df) - 1, sig.idx + 25)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c, v = window["open"], window["high"], window["low"], window["close"], window["volume"]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.15, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = c.iloc[k] >= o.iloc[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l.iloc[k], h.iloc[k], color=col, lw=0.65)
        y0, y1 = min(o.iloc[k], c.iloc[k]), max(o.iloc[k], c.iloc[k])
        if y1 == y0:
            y1 = y0 + max(h.iloc[k] - l.iloc[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(list(xs), v, width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        lw = 1.35 if n <= 20 else 1.05
        ax.plot(list(xs), window[f"ma{n}"], color=col, lw=lw, label=f"MA{n}")

    sx = sig.idx - start
    ax.axvline(sx, color="#3dba7a", ls="--", lw=1.0)
    ax.scatter([sx], [c.iloc[sx]], s=38, color="#3dba7a", zorder=5)
    ax.set_title(
        f"NQ 1m  {sig.order_text}   {_fmt(sig.timestamp)}   fan {sig.fan_pct:+.3f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_report(df: pd.DataFrame, signals: list[StackSignal], counts: dict[str, int], out_dir: Path, symbol: str) -> Path:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    # drop leftover dump charts from the previous report
    for old in img_dir.glob("*.png"):
        old.unlink()

    cards = []
    fwds = {15: [], 30: [], 60: []}
    for sig in signals:
        png = img_dir / f"stack_{_stem(sig.timestamp)}.png"
        draw_stack(df, sig, png)
        pts = {m: _fwd(df, sig.idx, m) for m in (15, 30, 60)}
        for m, val in pts.items():
            if val is not None:
                fwds[m].append(val)

        def _pt(x):
            if x is None:
                return "n/a"
            cls = "pos" if x >= 0 else "neg"
            return f'<span class="{cls}">{x:+.1f}pt</span>'

        cards.append(
            f"""
  <div class="card">
    <h2>{_fmt(sig.timestamp)} · {html.escape(sig.order_text)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="stack {_fmt(sig.timestamp)}"/>
    <p class="note">
      進場 {sig.entry:.2f} · 短均相對 MA200 {sig.fan_pct:+.3f}%<br/>
      進場後 15/30/60m：{_pt(pts[15])} / {_pt(pts[30])} / {_pt(pts[60])}
    </p>
  </div>"""
        )

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _win(xs):
        return (sum(1 for x in xs if x > 0) / len(xs) * 100) if xs else 0.0

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    days = sorted({t.date() for t in df.index})
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 1m 均線多頭排列 · 近一週</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 14px 56px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:16px;margin:0 0 10px}}
.sub{{color:#8aa193;line-height:1.65;margin:0 0 16px}}
.card{{background:#14201b;border:1px solid rgba(232,240,234,.12);border-radius:12px;padding:14px;margin-bottom:16px}}
img{{width:100%;height:auto;display:block;border-radius:8px;background:#101814}}
.note{{color:#8aa193;font-size:13px;margin:8px 0 0;line-height:1.5}}
.pos{{color:#3dba7a}}.neg{{color:#e35d5d}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0 18px}}
.kpi{{border:1px solid rgba(232,240,234,.12);border-radius:10px;padding:10px}}
.kpi .k{{color:#8aa193;font-size:12px}} .kpi .v{{font-size:18px;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid rgba(232,240,234,.08)}}
th{{color:#8aa193;font-weight:600}}
@media(max-width:720px){{.kpis{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body>
<div class="wrap">
  <h1>NQ 一分 K 均線多頭排列 · 近一週 {len(signals)} 筆</h1>
  <p class="sub">
    {html.escape(symbol)} · {days[0]} ~ {days[-1]} · {len(df)} 根 1m
    （{html.escape(start)} ~ {html.escape(end)} ET）。<br/>
    關注的是均線打開，不是急跌那一根。訊號 = 八條均線首次排成
    <b>MA5&gt;10&gt;20&gt;30&gt;60&gt;100&gt;120&gt;200</b>，且收盤站在全部均線之上。
    同一段行情 30 分鐘只記一次。綠虛線是排列形成的那一分。
    截圖 20:30 只是短均開始排；完整八條打開是 <b>08-18 21:30</b>。
  </p>
  <div class="kpis">
    <div class="kpi"><div class="k">短均 5&gt;10&gt;20</div><div class="v">{counts['short']}</div></div>
    <div class="kpi"><div class="k">中段接到 MA60</div><div class="v">{counts['mid']}</div></div>
    <div class="kpi"><div class="k">完整八條排列</div><div class="v pos">{counts['full']}</div></div>
  </div>
  <div class="card">
    <h2>排列梯子（都要求收盤站上全部均線，30 分鐘去重）</h2>
    <table>
      <tr><th>均線條件</th><th>近一週訊號</th></tr>
      <tr><td>短均多頭 MA5&gt;MA10&gt;MA20</td><td>{counts['short']}</td></tr>
      <tr><td>再加 MA20&gt;MA30&gt;MA60</td><td>{counts['mid']}</td></tr>
      <tr><td>完整 MA5&gt;10&gt;20&gt;30&gt;60&gt;100&gt;120&gt;200</td><td class="pos">{counts['full']}</td></tr>
    </table>
    <p class="note">
      完整排列進場後 15/30/60m 平均 {_avg(fwds[15]):+.1f} / {_avg(fwds[30]):+.1f} / {_avg(fwds[60]):+.1f} 點，
      勝率 {_win(fwds[15]):.0f}% / {_win(fwds[30]):.0f}% / {_win(fwds[60]):.0f}%。
    </p>
  </div>
  {''.join(cards)}
</div>
</body>
</html>
"""
    out = out_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="NQ 1m 均線多頭排列近一週回測")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="7d")
    p.add_argument("--csv")
    p.add_argument("--out", default="docs/nq-1m-v")
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
        df.index = df.index.tz_localize("America/New_York") if df.index.tz is None else df.index.tz_convert("America/New_York")
        df = df.rename(columns=str.lower)
    else:
        print("抓 NQ 一分 K…", flush=True)
        df = fetch_nq_1m(args.symbol, args.period)

    df = add_indicators(df)
    counts = ladder_counts(df)
    signals = count_stack_events(df, level="full")
    print(f"K 線 {len(df)} 根 | {df.index[0]} ~ {df.index[-1]} ET")
    print(f"短均 5>10>20：{counts['short']}")
    print(f"中段接到 MA60：{counts['mid']}")
    print(f"完整八條多頭排列：{counts['full']}")
    for sig in signals:
        print(f"  {_fmt(sig.timestamp)}  {sig.order_text}  @ {sig.entry:.2f}  fan {sig.fan_pct:+.3f}%")

    out = write_report(df, signals, counts, Path(args.out), args.symbol)
    print(f"\n報告 {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
