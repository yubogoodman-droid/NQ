#!/usr/bin/env python3
"""NQ 一分 K 急跌 V 反回測：近一週有多少訊號。"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.v_recovery import (  # noqa: E402
    LOOSE,
    MID,
    STRICT,
    DumpEvent,
    VRecoverySignal,
    add_indicators,
    count_ladder,
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
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fmt(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m-%d %H:%M")


def _stem(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    return t.strftime("%m%d_%H%M")


def draw_window(
    df: pd.DataFrame,
    dump: DumpEvent,
    path: Path,
    *,
    signal: VRecoverySignal | None = None,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    start = max(0, dump.idx - 20)
    end = min(len(df) - 1, (signal.entry_idx if signal else dump.idx) + 50)
    sl = slice(start, end + 1)
    window = df.iloc[sl]
    xs = range(len(window))
    o, h, l, c, v = window["open"], window["high"], window["low"], window["close"], window["volume"]

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.6, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1]},
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
        ax.vlines(xs[k], l.iloc[k], h.iloc[k], color=col, lw=0.7)
        y0, y1 = min(o.iloc[k], c.iloc[k]), max(o.iloc[k], c.iloc[k])
        if y1 == y0:
            y1 = y0 + max(h.iloc[k] - l.iloc[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(list(xs), v, width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ax.plot(list(xs), window[f"ma{n}"], color=col, lw=1.05, label=f"MA{n}")

    dump_x = dump.idx - start
    ax.axvline(dump_x, color="#e35d5d", ls="--", lw=0.9)
    ax.scatter([dump_x], [c.iloc[dump_x]], s=36, color="#e35d5d", zorder=5)
    if signal is not None:
        sx = signal.entry_idx - start
        if 0 <= sx < len(window):
            ax.axvline(sx, color="#3dba7a", ls="--", lw=0.9)
            ax.scatter([sx], [c.iloc[sx]], s=36, color="#3dba7a", zorder=5)
        if signal.prev_close is not None:
            ax.axhline(signal.prev_close, color="#c9a227", ls=":", lw=0.9, alpha=0.8)

    ax.set_title(title, color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _fwd(df: pd.DataFrame, idx: int, minutes: int) -> float | None:
    j = idx + minutes
    if j >= len(df):
        return None
    return float(df["close"].iloc[j] - df["close"].iloc[idx])


def write_report(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    symbol: str,
    strict,
    mid,
    loose,
) -> Path:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for sig in strict.signals:
        png = img_dir / f"signal_{_stem(sig.dump.timestamp)}.png"
        draw_window(
            df,
            sig.dump,
            png,
            signal=sig,
            title=f"{symbol} 1m  V-reclaim  {_fmt(sig.dump.timestamp)} -> {_fmt(sig.entry_time)}",
        )
        f15, f30, f60 = (_fwd(df, sig.entry_idx, m) for m in (15, 30, 60))

        def _pt(x):
            return "n/a" if x is None else f"{x:+.1f}pt"

        pc_txt = "—"
        if sig.prev_close is not None:
            pc_txt = f"{sig.prev_close:.2f}"
            if sig.prev_close_idx is not None:
                pc_txt += f" · 突破 {_fmt(df.index[sig.prev_close_idx])}"
        cards.append(
            f"""
  <div class="card">
    <h2>訊號 {_fmt(sig.entry_time)} · 急跌 {_fmt(sig.dump.timestamp)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="signal {_fmt(sig.dump.timestamp)}"/>
    <p class="note">
      急跌 {sig.dump.range_pts:.1f} 點 · 量比 {sig.dump.vol_ratio:.1f}× · ATR {sig.dump.range_atr:.1f}×<br/>
      進場 {sig.entry:.2f} · 停損 {sig.stop_loss:.2f} · 前收 {html.escape(pc_txt)}<br/>
      進場後 15/30/60m：{_pt(f15)} / {_pt(f30)} / {_pt(f60)}
    </p>
  </div>"""
        )

    failed = []
    signal_dumps = {s.dump.idx for s in strict.signals}
    for dump in strict.dumps_list:
        if dump.idx in signal_dumps:
            continue
        png = img_dir / f"fail_{_stem(dump.timestamp)}.png"
        draw_window(
            df,
            dump,
            png,
            title=f"{symbol} 1m  dump (no V)  {_fmt(dump.timestamp)}",
        )
        failed.append(
            f"""
  <div class="card">
    <h2>急跌未完成 V　{_fmt(dump.timestamp)}</h2>
    <img src="./img/{html.escape(png.name)}" alt="dump {_fmt(dump.timestamp)}"/>
    <p class="note">急跌 {dump.range_pts:.1f} 點 · 量比 {dump.vol_ratio:.1f}× · ATR {dump.range_atr:.1f}× · 90 分鐘內破了急跌低點</p>
  </div>"""
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    days = sorted({t.date() for t in df.index})
    day_txt = f"{days[0]} ~ {days[-1]}（{len(days)} 個交易日曆日）"

    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 1m 急跌 V 反 · 近一週</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 14px 56px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:16px;margin:0 0 10px}}
.sub{{color:#8aa193;line-height:1.65;margin:0 0 16px}}
.card{{background:#14201b;border:1px solid rgba(232,240,234,.12);border-radius:12px;padding:14px;margin-bottom:16px}}
img{{width:100%;height:auto;display:block;border-radius:8px;background:#101814}}
.note{{color:#8aa193;font-size:13px;margin:8px 0 0;line-height:1.5}}
.pos{{color:#3dba7a}}.neg{{color:#e35d5d}}.mark{{color:#c9a227}}
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
  <h1>NQ 一分 K 急跌 V 反 · 近一週 {strict.reclaim_fan} 筆訊號</h1>
  <p class="sub">
    {html.escape(symbol)} · {html.escape(day_txt)} · {len(df)} 根 1m
    （{html.escape(start)} ~ {html.escape(end)} ET）。<br/>
    規則對齊截圖那根：爆量長陰打穿全部均線 → 90 分鐘內不破低、收回 70% → 收盤站回 MA5/10/20/30/60/100/120/200 且 MA5&gt;MA10&gt;MA20。
    紅虛線是急跌，綠虛線是進場，金線是前收（16:00 ET）。
  </p>
  <div class="kpis">
    <div class="kpi"><div class="k">嚴格急跌</div><div class="v">{strict.dumps}</div></div>
    <div class="kpi"><div class="k">未破低收回 70%</div><div class="v">{strict.v70}</div></div>
    <div class="kpi"><div class="k">站回均線+短均開花</div><div class="v pos">{strict.reclaim_fan}</div></div>
  </div>
  <div class="card">
    <h2>條件梯子（同一週、同一份 1m）</h2>
    <table>
      <tr><th>急跌定義</th><th>急跌事件</th><th>V 收回 70%</th><th>站回全部均線</th><th>再加短均開花＝訊號</th><th>再過前收</th></tr>
      <tr><td>嚴格（截圖：5×ATR 或 50 點、5×量、跌破全部均線）</td><td>{strict.dumps}</td><td>{strict.v70}</td><td>{strict.reclaim}</td><td class="pos">{strict.reclaim_fan}</td><td>{strict.prev_close}</td></tr>
      <tr><td>中等（3×ATR 或 25 點、3×量、跌破全部均線）</td><td>{mid.dumps}</td><td>{mid.v70}</td><td>{mid.reclaim}</td><td>{mid.reclaim_fan}</td><td>{mid.prev_close}</td></tr>
      <tr><td>寬鬆（2.5×ATR、2.5×量、僅跌破 MA20）</td><td>{loose.dumps}</td><td>{loose.v70}</td><td>{loose.reclaim}</td><td>{loose.reclaim_fan}</td><td>{loose.prev_close}</td></tr>
    </table>
    <p class="note">寬鬆那 2 筆多出來的，多半是 08:30 美股開盤的寬幅 K，不是截圖那種 Globex 夜盤掃蕩。</p>
  </div>
  {''.join(cards) if cards else '<div class="card"><p class="note">這一週沒有完成樣貌的訊號。</p></div>'}
  <h1 style="margin-top:28px">嚴格急跌但沒走出 V</h1>
  <p class="sub">同一套急跌門檻，90 分鐘內又破了低點，不當訊號。</p>
  {''.join(failed)}
</div>
</body>
</html>
"""
    out = out_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def print_ladder(name: str, ladder) -> None:
    print(f"\n=== {name} ===")
    print(
        f"急跌 {ladder.dumps} | V70 {ladder.v70} | 站回均線 {ladder.reclaim} | "
        f"訊號(站回+開花) {ladder.reclaim_fan} | 再過前收 {ladder.prev_close}"
    )
    for sig in ladder.signals:
        print(
            f"  {_fmt(sig.dump.timestamp)} 急跌 {sig.dump.range_pts:.1f}pt "
            f"vr={sig.dump.vol_ratio:.1f} → 進場 {_fmt(sig.entry_time)} @ {sig.entry:.2f} "
            f"SL {sig.stop_loss:.2f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="NQ 1m 急跌 V 反近一週回測")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="7d")
    p.add_argument("--csv", help="既有 1m CSV（datetime,open,high,low,close,volume）")
    p.add_argument("--out", default="docs/nq-1m-v", help="HTML 報告目錄")
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        df = df.rename(columns=str.lower)
    else:
        print("抓 NQ 一分 K…", flush=True)
        df = fetch_nq_1m(args.symbol, args.period)

    df = add_indicators(df)
    print(f"K 線 {len(df)} 根 | {df.index[0]} ~ {df.index[-1]} ET")

    strict = count_ladder(df, STRICT)
    mid = count_ladder(df, MID)
    loose = count_ladder(df, LOOSE)
    print_ladder("嚴格（截圖邏輯）", strict)
    print_ladder("中等", mid)
    print_ladder("寬鬆", loose)

    out = write_report(df, out_dir=Path(args.out), symbol=args.symbol, strict=strict, mid=mid, loose=loose)
    print(f"\n報告 {out.resolve()}")
    print(f"近一週完整訊號：{strict.reclaim_fan} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
