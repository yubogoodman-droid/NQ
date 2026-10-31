#!/usr/bin/env python3
"""NQ 五分 K：破兩小時低點後，1 小時內站回且 5/10/20 多排做多。

用法:
  python3 examples/nq_5m_reclaim.py backtest --period 7d --pages
  python3 examples/nq_5m_reclaim.py backtest --period 30d --html output/nq_5m_reclaim.html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_5m_base_stack import (  # noqa: E402
    ET,
    REPO_ROOT,
    VIEW_BRANCH,
    _equity_svg,
    draw_trade_png,
    load_yfinance,
    simulate,
    sma,
    summarize_trades,
    to_et,
    write_view_html,
)

PAGES_HTML = REPO_ROOT / "docs" / "nq-5m-reclaim" / "index.html"
BLURB = (
    "五分 K 跌破近 2 小時低點後，1 小時內（12 根）收盤站回且 MA5>MA10>MA20 多排才做多。"
    "停損在破底低下方，目標 2R。"
)


@dataclass
class Signal:
    dump_idx: int
    base_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    dump_high: float
    base_low: float
    drop_pts: float
    recover: float
    ribbon: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float = 0.0
    quality: str = "B"
    quality_score: int = 1
    two_hr_low: float = 0.0
    mark_label: str = "破底"


def rolling_min_prev(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.shift(1).rolling(n, min_periods=n).min().to_numpy(float)


def detect_signals(
    df,
    two_hour_bars: int = 24,
    reclaim_window: int = 12,
    min_break_depth: float = 12.0,
    stop_buffer: float = 8.0,
    target_r: float = 2.0,
    max_risk: float = 150.0,
    min_entry_gap: int = 6,
    require_stack: bool = True,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """破近 2 小時低點後，reclaim_window 根內收盤站上 MA5/10/20 且多排。"""
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    two_hr_low = rolling_min_prev(low, two_hour_bars)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(30, two_hour_bars) + 1
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    armed = False
    break_idx = -1
    break_low = 0.0
    support = 0.0
    i = warmup
    while i < n - 1:
        if np.isnan(two_hr_low[i]) or np.isnan(ma20[i]):
            i += 1
            continue
        lvl = float(two_hr_low[i])
        bar_low = float(low[i])
        if bar_low < lvl and (not armed or bar_low < break_low - 1e-9):
            depth0 = lvl - bar_low
            if depth0 >= min_break_depth:
                if not armed:
                    bump("break")
                    bump("deep_break")
                else:
                    bump("new_low")
                armed = True
                break_idx = i
                break_low = bar_low
                support = lvl
            elif not armed:
                bump("break")
                bump("skip_shallow")

        if not armed:
            i += 1
            continue

        if i - break_idx >= reclaim_window:
            bump("skip_no_reclaim")
            armed = False
            i += 1
            continue

        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            i += 1
            continue
        above = float(close[i]) > float(ma5[i]) and float(close[i]) > float(ma10[i]) and float(close[i]) > float(ma20[i])
        stacked = float(ma5[i]) > float(ma10[i]) > float(ma20[i])
        if not above:
            i += 1
            continue
        if require_stack and not stacked:
            bump("skip_no_stack")
            i += 1
            continue
        bump("reclaim")
        if i - last_entry < min_entry_gap:
            bump("skip_gap")
            armed = False
            i += 1
            continue
        entry = float(close[i])
        stop = break_low - stop_buffer
        risk = entry - stop
        depth = support - break_low
        if risk <= 0:
            bump("skip_bad_risk")
            armed = False
            i += 1
            continue
        if max_risk > 0 and risk > max_risk:
            bump("skip_max_risk")
            i += 1
            continue
        recover = (entry - break_low) / depth if depth else 1.0
        q = "A" if depth >= 25.0 and (i - break_idx) <= 3 else "B"
        bump("taken")
        signals.append(
            Signal(
                dump_idx=break_idx,
                base_idx=break_idx,
                entry_idx=i,
                entry_price=entry,
                stop_price=stop,
                target_price=entry + risk * target_r,
                dump_high=support,
                base_low=break_low,
                drop_pts=depth,
                recover=recover,
                ribbon=float(ma5[i] - ma20[i]),
                ma5=float(ma5[i]),
                ma10=float(ma10[i]),
                ma20=float(ma20[i]),
                ma30=float(ma30[i]) if not np.isnan(ma30[i]) else 0.0,
                quality=q,
                quality_score=2 if q == "A" else 1,
                two_hr_low=support,
            )
        )
        last_entry = i
        armed = False
        i += 1

    return signals


def _trade_img_name(df: pd.DataFrame, trade, trade_no: int, prefix: str = "t") -> str:
    et = df.index[trade.entry_idx]
    return f"{prefix}{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(df, trades, html_path: Path, prefix: str = "t") -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "be": "tag-time", "trail": "tag-tp"}.get(
            t.exit_reason, "tag-time"
        )
        img_name = _trade_img_name(df, t, i, prefix=prefix)
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        wait = t.entry_idx - t.signal.dump_idx
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>5m</span>"
            f"<span class='tag tag-info'>破底+{wait * 5}分</span>"
            f"<span class='tag tag-info'>深 {t.signal.drop_pts:.0f}pt</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破底 {t.signal.base_low:.2f} ← 兩小時低 {t.signal.two_hr_low:.2f}  (−{t.signal.drop_pts:.1f})\n"
            f"{wait} 根後站回且 5/10/20 多排\n"
            f"MA5 {t.signal.ma5:.1f} > MA10 {t.signal.ma10:.1f} > MA20 {t.signal.ma20:.1f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "<p class='muted' style='margin:8px 2px 0'>上：進場放大 · 下：五分 K 參考</p>"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: str | Path,
    df,
    trades,
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    verdict: str = "",
    blurb: str = "",
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    out = Path(path)
    cards = _render_trade_cards(df, trades, out)
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破底 {funnel.get('break', 0)} → "
            f"夠深 {funnel.get('deep_break', 0)} → "
            f"站回 {funnel.get('reclaim', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（太淺 {funnel.get('skip_shallow', 0)} · 1小時內沒站回 {funnel.get('skip_no_reclaim', 0)} · "
            f"風險 {funnel.get('skip_max_risk', 0)}）</p>"
        )
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    verdict_html = f"<p class='muted'><b>{escape(verdict)}</b></p>" if verdict else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 五分破底站回 5/10/20</title>
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
<h1>{escape(symbol)} 五分破底後 1 小時內 5/10/20 多排</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)} · 五分 K</p>
<p class="muted">{escape(blurb or BLURB)}</p>
{verdict_html}
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">均筆<b>{stats['avg']:+.1f}</b></div>
</div>
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


def _print_trades(df, trades) -> None:
    for i, t in enumerate(trades, 1):
        wait = t.entry_idx - t.signal.dump_idx
        print(
            f"  [{i}] {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"depth={t.signal.drop_pts:.0f} wait={wait}bars risk={t.entry_price - t.stop_price:.0f}"
        )


def _verdict(stats: dict) -> str:
    if stats["count"] == 0:
        return "這段沒打到「破兩小時低、1 小時內站回且 5/10/20 多排」。"
    if stats["total_points"] > 0:
        return "有抓到破底後多排，筆數少，單筆停損仍在破底低下方。"
    return "這段是虧的。破底後要在 1 小時內站回且 5/10/20 多排才進。"


def cmd_backtest(args) -> int:
    print(f"load {args.symbol} 5m {args.period}", file=sys.stderr)
    df = to_et(load_yfinance(args.symbol, "5m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    print(f"bars={len(df)} {df.index[0]} → {df.index[-1]}", file=sys.stderr)
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, require_stack=not bool(getattr(args, "no_stack", False)))
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(
        f"trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"pnl={stats['total_points']:+.1f} avg={stats['avg']:+.1f}"
    )
    print("funnel", funnel)
    _print_trades(df, trades)
    verdict = _verdict(stats)
    print(f"verdict: {verdict}")

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(html_path, df, trades, args.symbol, args.period, funnel=funnel, verdict=verdict)
        print(f"html={out}")
        if getattr(args, "pages", False):
            period = str(getattr(args, "period", ""))
            dest = "view.html"
            if period.startswith("7"):
                dest = "one-week-1h.html"
            elif period.startswith("30"):
                dest = "one-month-1h.html"
            view = write_view_html(out, branch=VIEW_BRANCH, dest_name=dest)
            print(f"preview={view}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 五分破底後 1 小時內 5/10/20 多排")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="7d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true")
    b.add_argument("--no-stack", action="store_true", help="不要求 MA5>MA10>MA20（只站上三條）")
    b.set_defaults(func=cmd_backtest)
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="7d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--no-stack", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
