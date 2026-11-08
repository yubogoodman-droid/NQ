#!/usr/bin/env python3
"""NQ 一分 K：急跌 + 均線多頭排列，近一週有多少訊號。"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_stack import (  # noqa: E402
    LOOSE_DUMP,
    MID_DUMP,
    STRICT_DUMP,
    ComboSignal,
    add_indicators,
    dump_align_ladder,
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
    """Yahoo 1m 單次最多約 7–8 天；超過改用 7 日切片（約可回看 30 天）。"""
    import time
    from datetime import datetime, timedelta, timezone

    import yfinance as yf

    def _clean(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = idx.tz_convert("America/New_York")
        return df

    p = (period or "7d").strip().lower()
    days: int | None = None
    if p.endswith("d") and p[:-1].isdigit():
        days = int(p[:-1])
    elif p.endswith("w") and p[:-1].isdigit():
        days = int(p[:-1]) * 7
    elif p.endswith("mo") and p[:-2].isdigit():
        days = int(p[:-2]) * 30

    ticker = yf.Ticker(symbol)
    parts: list[pd.DataFrame] = []
    if days is not None and days > 8:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        cur = start
        delta = timedelta(days=7)
        while cur < end:
            nxt = min(cur + delta, end)
            raw = ticker.history(start=cur, end=nxt, interval="1m", auto_adjust=False)
            chunk = _clean(raw)
            if chunk.empty:
                print(f"  {cur.date()} → {nxt.date()} 空", flush=True)
            else:
                parts.append(chunk)
                print(f"  {cur.date()} → {nxt.date()} {len(chunk)} 根", flush=True)
            cur = nxt
            time.sleep(0.35)
        if not parts:
            raise RuntimeError(f"無法取得 {symbol} 一分 K（{period}）")
        df = pd.concat(parts)
    else:
        raw = ticker.history(period=period, interval="1m", auto_adjust=False)
        df = _clean(raw)
        if df.empty:
            raise RuntimeError(f"無法取得 {symbol} 一分 K")

    return df[~df.index.duplicated(keep="last")].sort_index()


def _fmt(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m-%d %H:%M")


def _stem(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m%d_%H%M")


EXIT_LABEL = {
    "stop": "停損",
    "ma20": "跌破 MA20",
    "time": "持滿 90 分",
}


def _use_cjk_font() -> None:
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _ring(ax, x: float, y: float, color: str, label: str, dy: int = 12) -> None:
    ax.scatter(
        [x],
        [y],
        s=720,
        facecolors="none",
        edgecolors=color,
        linewidths=2.4,
        zorder=8,
        marker="o",
    )
    ax.scatter([x], [y], s=28, color=color, zorder=9)
    ax.annotate(
        label,
        (x, y),
        xytext=(10, dy),
        textcoords="offset points",
        color=color,
        fontsize=10,
        fontweight="normal",
        zorder=10,
    )


def draw_combo(df: pd.DataFrame, combo: ComboSignal, path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    _use_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dump = combo.dump
    last = dump.idx
    if combo.short:
        last = max(last, combo.short.idx)
    if combo.exit_event:
        last = max(last, combo.exit_event.idx)
    start = max(0, dump.idx - 18)
    end = min(len(df) - 1, last + 18)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c, v = window["open"], window["high"], window["low"], window["close"], window["volume"]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.8),
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
        ax.plot(list(xs), window[f"ma{n}"], color=col, lw=1.3 if n <= 20 else 1.05, label=f"MA{n}")

    dx = dump.idx - start
    ax.axvline(dx, color="#e35d5d", ls="--", lw=0.9, alpha=0.85)
    ax.axhline(dump.low, color="#e35d5d", ls=":", lw=0.9, alpha=0.7)
    if combo.short:
        sx = combo.short.idx - start
        if 0 <= sx < len(window):
            _ring(ax, sx, combo.short.entry, "#3dba7a", "進場", dy=14)
    if combo.exit_event:
        ex = combo.exit_event
        xx = ex.idx - start
        if 0 <= xx < len(window):
            out_color = "#c9a227" if ex.pnl >= 0 else "#e35d5d"
            dy = -18 if combo.short and abs(ex.idx - combo.short.idx) <= 8 else 14
            _ring(ax, xx, ex.price, out_color, "出場", dy=dy)

    ax.set_title(title, color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _pt(x: float | None) -> str:
    if x is None:
        return "n/a"
    cls = "pos" if x >= 0 else "neg"
    return f'<span class="{cls}">{x:+.1f}pt</span>'


def write_report(
    df: pd.DataFrame,
    *,
    strict: dict,
    mid: dict,
    loose: dict,
    out_dir: Path,
    symbol: str,
) -> Path:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    seen: set[int] = set()
    ranked: list[tuple[str, ComboSignal]] = []
    for tag, bucket in (("嚴格", strict), ("中等", mid), ("寬鬆", loose)):
        for combo in bucket["signals"]:
            idx = combo.dump.idx
            if idx in seen:
                continue
            seen.add(idx)
            ranked.append((tag, combo))
    ranked.sort(key=lambda item: item[1].dump.idx)

    cards = []
    for tag, combo in ranked:
        dump = combo.dump
        png = img_dir / f"hit_{_stem(dump.timestamp)}.png"
        short_t = _fmt(combo.short.timestamp) if combo.short else "—"
        ex = combo.exit_event
        exit_t = _fmt(ex.timestamp) if ex else "—"
        reason = EXIT_LABEL.get(ex.reason, ex.reason) if ex else "—"
        title = f"NQ 1m  進場 {short_t}  →  出場 {exit_t}  {reason}"
        draw_combo(df, combo, png, title=title)
        entry = combo.short
        cards.append(
            f"""
  <div class="card">
    <h2>{html.escape(tag)}　進場 {html.escape(short_t)} → 出場 {html.escape(exit_t)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="combo {_fmt(dump.timestamp)}"/>
    <p class="note">
      急跌 {_fmt(dump.timestamp)} {dump.range_pts:.1f} 點 · 停損 {dump.low:.2f}<br/>
      進場綠圈 {(entry.entry if entry else 0):.2f} · 出場圈 {ex.price if ex else 0:.2f} · {html.escape(reason)} · {_pt(ex.pnl if ex else None)}
    </p>
  </div>"""
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    days = sorted({t.date() for t in df.index})
    span = (days[-1] - days[0]).days + 1
    pnls = [c.exit_event.pnl for _, c in ranked if c.exit_event]
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    total = sum(pnls) if pnls else 0.0
    wr = f"{100.0 * wins / len(pnls):.0f}%" if pnls else "—"
    rows = []
    for tag, combo in ranked:
        dump = combo.dump
        entry = combo.short
        ex = combo.exit_event
        reason = EXIT_LABEL.get(ex.reason, ex.reason) if ex else "—"
        rows.append(
            f"<tr><td>{html.escape(tag)}</td><td>{_fmt(dump.timestamp)}</td>"
            f"<td>{_fmt(entry.timestamp) if entry else '—'}</td>"
            f"<td>{entry.entry:.2f}</td>"
            f"<td>{_fmt(ex.timestamp) if ex else '—'}</td>"
            f"<td>{ex.price:.2f}</td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{_pt(ex.pnl if ex else None)}</td></tr>"
        )
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 1m 急跌 + 均線排列 · {span} 天</title>
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
  <h1>NQ 一分 K 急跌 + 均線排列 · {span} 天</h1>
  <p class="sub">
    {html.escape(symbol)} · {days[0]} ~ {days[-1]} · {len(df)} 根 1m
    （{html.escape(start)} ~ {html.escape(end)} ET）。<br/>
    只畫急跌後 90 分鐘內不破低、再走出 MA5&gt;10&gt;20 的命中。綠圈進場、另一圈出場（停損急跌低、收盤跌破 MA20、或持滿 90 分）。
  </p>
  <div class="kpis">
    <div class="kpi"><div class="k">命中</div><div class="v pos">{len(cards)}</div></div>
    <div class="kpi"><div class="k">勝率</div><div class="v">{wr} · {wins}/{len(pnls) if pnls else 0}</div></div>
    <div class="kpi"><div class="k">總點數</div><div class="v {'pos' if total >= 0 else 'neg'}">{total:+.1f}</div></div>
  </div>
  <div class="card">
    <h2>急跌 × 均線梯子</h2>
    <table>
      <tr><th>條件</th><th>急跌</th><th>未破低</th><th>短均 5&gt;10&gt;20</th><th>接到 MA60</th><th>完整八條</th></tr>
      <tr><td>嚴格（5×ATR 或 50 點、5×量、跌破全部均線）</td><td>{strict['dumps']}</td><td>{strict['v']}</td><td class="pos">{strict['short']}</td><td>{strict['mid']}</td><td>{strict['full']}</td></tr>
      <tr><td>中等</td><td>{mid['dumps']}</td><td>{mid['v']}</td><td>{mid['short']}</td><td>{mid['mid']}</td><td>{mid['full']}</td></tr>
      <tr><td>寬鬆</td><td>{loose['dumps']}</td><td>{loose['v']}</td><td>{loose['short']}</td><td>{loose['mid']}</td><td>{loose['full']}</td></tr>
    </table>
    <p class="note">圖只列急跌+排列命中，去重後標最嚴那一檔。</p>
  </div>
  <div class="card">
    <h2>進出場</h2>
    <table>
      <tr><th>檔</th><th>急跌</th><th>進場</th><th>進場價</th><th>出場</th><th>出場價</th><th>原因</th><th>點數</th></tr>
      {''.join(rows) if rows else '<tr><td colspan="8">沒有命中</td></tr>'}
    </table>
  </div>
  <h1 id="hits" style="margin-top:8px">急跌 + 排列</h1>
  {''.join(cards) if cards else '<div class="card"><p class="note">這一窗沒有急跌後走出排列的訊號。</p></div>'}
</div>
</body>
</html>
"""
    out = out_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def _print(name: str, ladder: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"急跌 {ladder['dumps']} | 未破低 {ladder['v']} | "
        f"短均 {ladder['short']} | 中段 {ladder['mid']} | 完整 {ladder['full']}"
    )
    for c in ladder["signals"]:
        d = c.dump
        st = _fmt(c.short.timestamp) if c.short else "—"
        if c.exit_event:
            et = _fmt(c.exit_event.timestamp)
            why = EXIT_LABEL.get(c.exit_event.reason, c.exit_event.reason)
            pnl = f"{c.exit_event.pnl:+.1f}pt"
        else:
            et, why, pnl = "—", "—", ""
        print(f"  急跌 {_fmt(d.timestamp)} {d.range_pts:.1f}pt → 進場 {st} → 出場 {et} {why} {pnl}")


def main() -> int:
    p = argparse.ArgumentParser(description="NQ 1m 急跌+均線排列回測")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="30d")
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
    print(f"K 線 {len(df)} 根 | {df.index[0]} ~ {df.index[-1]} ET")
    strict = dump_align_ladder(df, STRICT_DUMP)
    mid = dump_align_ladder(df, MID_DUMP)
    loose = dump_align_ladder(df, LOOSE_DUMP)
    _print("嚴格急跌 + 排列", strict)
    _print("中等急跌 + 排列", mid)
    _print("寬鬆急跌 + 排列", loose)

    out = write_report(
        df,
        strict=strict,
        mid=mid,
        loose=loose,
        out_dir=Path(args.out),
        symbol=args.symbol,
    )
    uniq = []
    seen: set[int] = set()
    for c in strict["signals"] + mid["signals"] + loose["signals"]:
        if c.dump.idx in seen:
            continue
        seen.add(c.dump.idx)
        uniq.append(c)
    pnls = [c.exit_event.pnl for c in uniq if c.exit_event]
    print(f"\n報告 {out.resolve()}")
    print(f"急跌+短均排列（去重）：{len(uniq)} 筆")
    if pnls:
        wins = sum(1 for x in pnls if x > 0)
        print(f"勝 {wins} / 負 {sum(1 for x in pnls if x < 0)} · 勝率 {100*wins/len(pnls):.0f}% · 總點 {sum(pnls):+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
