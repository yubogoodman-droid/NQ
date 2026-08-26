#!/usr/bin/env python3
"""NQ 五分 K：收盤同時跌破 MA5/10/20/30/60/120 做空。

用法:
  python3 examples/nq_ma_breakdown_short.py --demo
  python3 examples/nq_ma_breakdown_short.py backtest --period 60d --pages
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_breakdown_short import (  # noqa: E402
    TradeResult,
    detect_signals,
    simulate,
    summarize_trades,
)

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma-breakdown-short" / "index.html"
VIEW_BRANCH = "cursor/nq-5m-ma10-retest-59b8"

MA_COLORS = {
    5: "#ffa726",
    10: "#4dd0e1",
    20: "#ec407a",
    30: "#42a5f5",
    60: "#66bb6a",
    120: "#ef5350",
}


def load_yfinance(symbol: str = "NQ=F", interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


def to_et(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(ET)
    else:
        out.index = out.index.tz_convert(ET)
    return out


def make_demo_bars(n: int = 180) -> pd.DataFrame:
    """合成：均線上方盤整 → 一根大陰線同時跌破六條均線 → 續跌。"""
    close = np.full(n, 20000.0)
    high = close + 3.0
    low = close - 3.0
    open_ = close.copy()
    dump = 130
    close[dump] = 19910.0
    open_[dump] = 20002.0
    high[dump] = 20008.0
    low[dump] = 19898.0
    for i in range(dump + 1, n):
        close[i] = close[i - 1] - 8.0
        open_[i] = close[i - 1]
        high[i] = open_[i] + 1.5
        low[i] = close[i] - 2.0
    idx = pd.date_range("2026-07-28 04:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )


def _macd(close: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2.0
    return dif.to_numpy(float), dea.to_numpy(float), hist.to_numpy(float)


def _equity_svg(pnls: List[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(1.0, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys_src[i]):.1f}" for i in range(len(ys_src)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, trade.entry_idx - 24)
    end = min(len(df) - 1, trade.exit_idx + 10)
    return start, end


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break

    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    close_full = df["Close"].astype(float)
    dif, dea, hist = _macd(close_full)
    dif_w, dea_w, hist_w = dif[start : end + 1], dea[start : end + 1], hist[start : end + 1]

    fig, (ax, axm) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axm):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 30 else 1.05, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=48, color="#ff5252", marker="v", zorder=6)
        ax.annotate(
            "跌破做空",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(8, 10),
            ha="left",
            color="#ff8a80",
            fontsize=8,
        )
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    colors_h = ["#3dba7a" if v >= 0 else "#e35d5d" for v in hist_w]
    axm.bar(list(xs), hist_w, width=0.8, color=colors_h, linewidth=0, alpha=0.85)
    axm.plot(list(xs), dif_w, color="#42a5f5", lw=1.05, label="DIF")
    axm.plot(list(xs), dea_w, color="#ffa726", lw=1.05, label="DEA")
    axm.axhline(0, color="#334155", lw=0.7)
    axm.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=2)

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  Q{trade.quality}  short  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axm.set_xticks(ticks)
    axm.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int) -> str:
    et = df.index[trade.entry_idx]
    return f"t{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(df: pd.DataFrame, trades: List[TradeResult], html_path: Path) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.stop_price - t.entry_price
        r_mult = (t.entry_price - t.target_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i)
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        s = t.signal
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>5m 空</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"short {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"MA5 {s.ma5:.1f} / MA10 {s.ma10:.1f} / MA20 {s.ma20:.1f}\n"
            f"MA30 {s.ma30:.1f} / MA60 {s.ma60:.1f} / MA120 {s.ma120:.1f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[dict] = None,
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.1f}" for q, info in stats.get("by_quality", {}).items()]
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades, out)
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：同時低於六均 {funnel.get('below_all', 0)} → "
            f"第一根跌破 {funnel.get('fresh_break', 0)} → 進場 {funnel.get('taken', 0)}"
            f"（已在下方 {funnel.get('already_below', 0)} · 非陰線 {funnel.get('skip_not_bear', 0)} · "
            f"太淺 {funnel.get('skip_thin', 0)} · "
            f"時段 {funnel.get('skip_session', 0)} · 風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分K 跌破均線叢做空</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 五分K 跌破均線叢做空</h1>
<p class="muted">收盤同時跌破 MA5/10/20/30/60/120 · {escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">{escape(q_line)}</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{cards or "<div class='empty'>無交易</div>"}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def cmd_backtest(args) -> int:
    if args.demo:
        df = make_demo_bars()
        period = "demo"
    else:
        df = to_et(load_yfinance(args.symbol, "5m", args.period))
        period = args.period
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    trades = simulate(df, sigs, max_hold=args.max_hold)
    stats = summarize_trades(trades)
    print(f"{args.symbol} 5m short {period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if funnel:
        print(
            "funnel "
            f"below_all={funnel.get('below_all', 0)} fresh={funnel.get('fresh_break', 0)} "
            f"taken={funnel.get('taken', 0)} already={funnel.get('already_below', 0)} "
            f"thin={funnel.get('skip_thin', 0)} bear={funnel.get('skip_not_bear', 0)} "
            f"session={funnel.get('skip_session', 0)} risk={funnel.get('skip_max_risk', 0)}"
        )
    for q, info in stats.get("by_quality", {}).items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"short {t.entry_price:.1f} stop {t.stop_price:.1f}"
        )
    html_path = args.html
    if args.pages:
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(html_path, df, trades, args.symbol, period, funnel=funnel)
        view = write_view_html(out)
        print(f"html={out}")
        print(f"view={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 五分K 同時跌破 MA5/10/20/30/60/120 做空")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="60d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--max-hold", type=int, default=24)
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="60d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true")
    b.add_argument("--demo", action="store_true")
    b.add_argument("--max-hold", type=int, default=24)
    b.set_defaults(func=cmd_backtest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
