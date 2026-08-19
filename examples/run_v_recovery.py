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
    StackSignal,
    add_indicators,
    count_stack_events,
    dump_align_ladder,
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


def draw_combo(df: pd.DataFrame, combo: ComboSignal, path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dump = combo.dump
    last = dump.idx
    if combo.full:
        last = combo.full.idx
    elif combo.short:
        last = combo.short.idx
    start = max(0, dump.idx - 18)
    end = min(len(df) - 1, last + 25)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c, v = window["open"], window["high"], window["low"], window["close"], window["volume"]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
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
    ax.axvline(dx, color="#e35d5d", ls="--", lw=1.0)
    ax.scatter([dx], [c.iloc[dx]], s=36, color="#e35d5d", zorder=5)
    if combo.short:
        sx = combo.short.idx - start
        if 0 <= sx < len(window):
            ax.axvline(sx, color="#3dba7a", ls="--", lw=1.0)
            ax.scatter([sx], [c.iloc[sx]], s=36, color="#3dba7a", zorder=5)
    if combo.full and combo.short and combo.full.idx != combo.short.idx:
        fx = combo.full.idx - start
        if 0 <= fx < len(window):
            ax.axvline(fx, color="#c9a227", ls=":", lw=1.1)
            ax.scatter([fx], [c.iloc[fx]], s=32, color="#c9a227", zorder=5)

    ax.set_title(title, color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


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
        ax.plot(list(xs), window[f"ma{n}"], color=col, lw=1.3 if n <= 20 else 1.05, label=f"MA{n}")
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
    stacks: list[StackSignal],
    align_only: dict[str, int],
    out_dir: Path,
    symbol: str,
) -> Path:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    strict_ids = {c.dump.idx for c in strict["signals"]}
    mid_ids = {c.dump.idx for c in mid["signals"]}
    cards = []
    for combo in loose["signals"]:
        dump = combo.dump
        if dump.idx in strict_ids:
            tag = "嚴格"
        elif dump.idx in mid_ids:
            tag = "中等"
        else:
            tag = "寬鬆"
        png = img_dir / f"hit_{_stem(dump.timestamp)}.png"
        short_t = _fmt(combo.short.timestamp) if combo.short else "—"
        full_t = _fmt(combo.full.timestamp) if combo.full else "未完成"
        draw_combo(
            df,
            combo,
            png,
            title=f"NQ 1m  dump {_fmt(dump.timestamp)}  ->  stack {short_t}  full {full_t}",
        )
        entry = combo.short
        pts = {m: _fwd(df, entry.idx, m) for m in (15, 30, 60)} if entry else {15: None, 30: None, 60: None}
        cards.append(
            f"""
  <div class="card">
    <h2>{html.escape(tag)}　急跌 {_fmt(dump.timestamp)} → 短均 {html.escape(short_t)} → 完整 {html.escape(full_t)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="combo {_fmt(dump.timestamp)}"/>
    <p class="note">
      急跌 {dump.range_pts:.1f} 點 · 量比 {dump.vol_ratio:.1f}× · ATR {dump.range_atr:.1f}×<br/>
      短均進場 {(entry.entry if entry else 0):.2f} · {html.escape(entry.order_text if entry else '')}<br/>
      進場後 15/30/60m：{_pt(pts[15])} / {_pt(pts[30])} / {_pt(pts[60])}
    </p>
  </div>"""
        )

    failed = [c for c in strict["combos"] if not c.aligned]
    fails = []
    for combo in failed:
        dump = combo.dump
        png = img_dir / f"fail_{_stem(dump.timestamp)}.png"
        draw_combo(df, combo, png, title=f"NQ 1m  dump (no stack)  {_fmt(dump.timestamp)}")
        fails.append(
            f"""
  <div class="card">
    <h2>急跌未走出排列　{_fmt(dump.timestamp)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="dump {_fmt(dump.timestamp)}"/>
    <p class="note">急跌 {dump.range_pts:.1f} 點 · 量比 {dump.vol_ratio:.1f}× · ATR {dump.range_atr:.1f}× · 90 分鐘內破低或均線沒排成多頭</p>
  </div>"""
        )

    stack_cards = []
    for sig in stacks:
        png = img_dir / f"stack_{_stem(sig.timestamp)}.png"
        draw_stack(df, sig, png)
        pts = {m: _fwd(df, sig.idx, m) for m in (15, 30, 60)}
        stack_cards.append(
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

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    days = sorted({t.date() for t in df.index})
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 1m 急跌 + 均線排列 · 近一週</title>
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
  <h1>NQ 一分 K 急跌 + 均線排列 · 全部圖</h1>
  <p class="sub">
    {html.escape(symbol)} · {days[0]} ~ {days[-1]} · {len(df)} 根 1m
    （{html.escape(start)} ~ {html.escape(end)} ET）。<br/>
    紅虛線急跌、綠虛線短均排列、金虛線完整八條打開。下面三區：急跌+排列全部命中、急跌失敗、以及只看均線的 40 張完整打開。
  </p>
  <p class="sub">
    <a href="#hits" style="color:#c9a227">急跌+排列 {len(cards)}</a> ·
    <a href="#fails" style="color:#c9a227">急跌失敗 {len(fails)}</a> ·
    <a href="#stacks" style="color:#c9a227">完整排列 {len(stack_cards)}</a>
  </p>
  <div class="kpis">
    <div class="kpi"><div class="k">嚴格急跌</div><div class="v">{strict['dumps']}</div></div>
    <div class="kpi"><div class="k">未破低</div><div class="v">{strict['v']}</div></div>
    <div class="kpi"><div class="k">急跌+短均排列</div><div class="v pos">{strict['short']}</div></div>
  </div>
  <div class="card">
    <h2>急跌 × 均線梯子</h2>
    <table>
      <tr><th>條件</th><th>急跌</th><th>未破低</th><th>短均 5&gt;10&gt;20</th><th>接到 MA60</th><th>完整八條</th></tr>
      <tr><td>嚴格急跌（5×ATR 或 50 點、5×量、跌破全部均線）</td><td>{strict['dumps']}</td><td>{strict['v']}</td><td class="pos">{strict['short']}</td><td>{strict['mid']}</td><td>{strict['full']}</td></tr>
      <tr><td>中等急跌</td><td>{mid['dumps']}</td><td>{mid['v']}</td><td>{mid['short']}</td><td>{mid['mid']}</td><td>{mid['full']}</td></tr>
      <tr><td>寬鬆急跌</td><td>{loose['dumps']}</td><td>{loose['v']}</td><td>{loose['short']}</td><td>{loose['mid']}</td><td>{loose['full']}</td></tr>
      <tr><td>只看均線、不看急跌（30 分鐘去重）</td><td>—</td><td>—</td><td>{align_only['short']}</td><td>{align_only['mid']}</td><td>{align_only['full']}</td></tr>
    </table>
    <p class="note">只看排列一週有 40 筆完整打開；加上急跌之後，嚴格條件只剩截圖那一筆。</p>
  </div>
  <h1 id="hits" style="margin-top:8px">急跌 + 排列（全部命中）</h1>
  <p class="sub">寬鬆門檻下的全部命中；標籤標嚴／中／寬。</p>
  {''.join(cards) if cards else '<div class="card"><p class="note">這一週沒有急跌後走出排列的訊號。</p></div>'}
  <h1 id="fails" style="margin-top:28px">嚴格急跌但沒走出排列</h1>
  <p class="sub">有砸、有量，但 90 分鐘內破低或均線沒排成多頭。</p>
  {''.join(fails)}
  <h1 id="stacks" style="margin-top:28px">只看均線：完整八條打開（{len(stack_cards)} 張）</h1>
  <p class="sub">不管有沒有急跌。綠虛線是 MA5&gt;10&gt;20&gt;30&gt;60&gt;100&gt;120&gt;200 第一次排好、價站上。</p>
  {''.join(stack_cards)}
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
        ft = _fmt(c.full.timestamp) if c.full else "—"
        print(f"  急跌 {_fmt(d.timestamp)} {d.range_pts:.1f}pt → 短均 {st} → 完整 {ft}")


def main() -> int:
    p = argparse.ArgumentParser(description="NQ 1m 急跌+均線排列近一週回測")
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
    print(f"K 線 {len(df)} 根 | {df.index[0]} ~ {df.index[-1]} ET")
    strict = dump_align_ladder(df, STRICT_DUMP)
    mid = dump_align_ladder(df, MID_DUMP)
    loose = dump_align_ladder(df, LOOSE_DUMP)
    align_only = ladder_counts(df)
    stacks = count_stack_events(df, level="full")
    _print("嚴格急跌 + 排列", strict)
    _print("中等急跌 + 排列", mid)
    _print("寬鬆急跌 + 排列", loose)
    print(f"\n只看排列：短 {align_only['short']} / 中 {align_only['mid']} / 完整 {align_only['full']}")

    out = write_report(
        df,
        strict=strict,
        mid=mid,
        loose=loose,
        stacks=stacks,
        align_only=align_only,
        out_dir=Path(args.out),
        symbol=args.symbol,
    )
    print(f"\n報告 {out.resolve()}")
    print(f"嚴格急跌+短均排列：{strict['short']} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
