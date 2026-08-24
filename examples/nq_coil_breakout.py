#!/usr/bin/env python3
"""NQ 起漲點：均線糾結盤整後放量突破（1 分 / 5 分對照）。

用法:
  python3 examples/nq_coil_breakout.py --demo
  python3 examples/nq_coil_breakout.py backtest --period 8d --html output/nq_coil.html
  python3 examples/nq_coil_breakout.py backtest --period 8d --pages
  python3 examples/nq_coil_breakout.py alert --dry-run --once
  python3 examples/nq_coil_breakout.py alert --test

Telegram 憑證放 tg_config.env（勿提交）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from nq.coil import (  # noqa: E402
    ET,
    CoilSignal,
    CoilTrade,
    detect_coil_breakouts,
    make_coil_demo_bars,
    resample_ohlcv,
    simulate,
    summarize_trades,
)
from nq_ma_reclaim import (  # noqa: E402
    env,
    load_bars,
    load_dotenv,
    load_yfinance,
    tg_send,
    to_et,
)

STATE_PATH = ROOT / "tg_coil_alert_state.json"
PAGES_HTML = REPO_ROOT / "docs" / "nq-coil-breakout" / "index.html"
PAGES_BRANCH = "cursor/nq-1m-coil-breakout-36d9"

TF_LABELS = {"1m": "1分K", "5m": "5分K"}
TF_PREFIX = {"1m": "m1", "5m": "m5"}

MA_COLORS = {
    5: "#ffa726",
    10: "#42a5f5",
    20: "#ec407a",
    30: "#26c6da",
    60: "#66bb6a",
    100: "#ffeb3b",
    120: "#ef5350",
    200: "#26a69a",
}


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


def _trade_window(df: pd.DataFrame, trade: CoilTrade) -> tuple[int, int]:
    start = max(0, trade.signal.coil_start_idx - 20)
    end = min(len(df) - 1, trade.exit_idx + 16)
    return start, end


def draw_trade_png(
    df: pd.DataFrame,
    trade: CoilTrade,
    path: Path,
    trade_no: int,
    interval: str = "1m",
) -> Path:
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

    sig = trade.signal
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.2 if n <= 20 else 0.95, label=f"MA{n}")

    ax.axhline(sig.coil_high, color="#f0c14b", ls="--", lw=0.9, alpha=0.8)
    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)

    cx = sig.coil_start_idx - start
    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= cx < len(window):
        ax.axvline(cx, color="#f0c14b", ls=":", lw=0.7, alpha=0.6)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=46, color="#00e676", marker="^", zorder=6)
        ax.annotate(
            "起漲",
            (ex, trade.entry_price),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#86efac",
            fontsize=8,
        )
    if 0 <= xx < len(window):
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    tf = TF_LABELS.get(interval, interval)
    ax.set_title(
        f"{tf} #{trade_no}  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor="#c8d5cc", ncol=8)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _trade_img_name(
    df: pd.DataFrame, trade: CoilTrade, trade_no: int, interval: str = "1m"
) -> str:
    et = df.index[trade.entry_idx]
    prefix = TF_PREFIX.get(interval, interval.replace("m", ""))
    return f"{prefix}_t{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _funnel_html(funnel: Optional[Dict[str, int]]) -> str:
    if not funnel:
        return ""
    return (
        f"<p class='muted'>漏斗：檢查 {funnel.get('checked', 0)} → "
        f"盤整箱 {funnel.get('coil', 0)} → "
        f"等待突破 {funnel.get('sticky', 0)} → "
        f"長實體 {funnel.get('body', 0)} → "
        f"站上盤整 {funnel.get('above_coil', 0)} → "
        f"5/10/20/30排列 {funnel.get('stack', 0)} → "
        f"站上MA200 {funnel.get('above_200', 0)} → "
        f"60與120在200下 {funnel.get('long_below', 0)} → "
        f"放量 {funnel.get('volume', 0)} → "
        f"進場 {funnel.get('taken', 0)}</p>"
    )


def _quality_line(stats: Dict[str, Any]) -> str:
    q_bits = [f"Q{q} {info['n']}筆 {info['pnl']:+.1f}" for q, info in stats.get("by_quality", {}).items()]
    return " · ".join(q_bits) if q_bits else "無品質分組"


def _day_hits(df: pd.DataFrame, trades: Sequence[CoilTrade]) -> Dict[str, List[str]]:
    days: Dict[str, List[str]] = {}
    for i, t in enumerate(trades, 1):
        ts = df.index[t.entry_idx]
        days.setdefault(ts.strftime("%m-%d"), []).append(f"#{i} {ts.strftime('%H:%M')}")
    return days


def _compare_table(
    rows: Sequence[Tuple[str, pd.DataFrame, List[CoilTrade]]],
) -> str:
    if len(rows) < 2:
        return ""
    labels = [TF_LABELS.get(iv, iv) for iv, _df, _tr in rows]
    stats = [summarize_trades(tr) for _iv, _df, tr in rows]
    keys = [
        ("K 數", [str(len(df)) for _iv, df, _tr in rows]),
        ("筆數", [str(s["count"]) for s in stats]),
        ("勝率", [f"{s['win_rate']:.1f}%" for s in stats]),
        ("淨點數", [f"{s['total_points']:+.1f}" for s in stats]),
        ("勝/負", [f"{s['wins']}/{s['count'] - s['wins']}" for s in stats]),
    ]
    head = "".join(f"<th>{escape(lab)}</th>" for lab in labels)
    body = []
    for name, vals in keys:
        tds = "".join(f"<td>{escape(v)}</td>" for v in vals)
        body.append(f"<tr><th>{escape(name)}</th>{tds}</tr>")

    day_maps = [_day_hits(df, tr) for _iv, df, tr in rows]
    all_days = sorted(set().union(*[set(m) for m in day_maps]))
    day_rows = []
    for d in all_days:
        tds = []
        for m in day_maps:
            tds.append(f"<td>{escape(' · '.join(m.get(d, [])) or '—')}</td>")
        day_rows.append(f"<tr><th>{escape(d)}</th>{''.join(tds)}</tr>")
    day_html = (
        "<p class='muted' style='margin-top:12px'>同日進場對照（日期為美東）</p>"
        f"<table class='compare'><thead><tr><th>日期</th>{head}</tr></thead>"
        f"<tbody>{''.join(day_rows)}</tbody></table>"
        if day_rows
        else ""
    )
    return (
        "<table class='compare'><thead><tr><th></th>"
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        + day_html
    )


def _trade_cards(
    df: pd.DataFrame,
    trades: List[CoilTrade],
    img_dir: Path,
    interval: str,
    keep: Set[str],
) -> str:
    cards: List[str] = []
    tf = TF_LABELS.get(interval, interval)
    tf_cls = "tag-tf1" if interval == "1m" else "tag-tf5"
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i, interval)
        keep.add(img_name)
        draw_trade_png(df, t, img_dir / img_name, i, interval=interval)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {tf_cls}'>{escape(tf)}</span>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>起漲</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"盤整 {t.signal.coil_low:.2f}–{t.signal.coil_high:.2f}  "
            f"區間 {t.signal.coil_range:.1f}  帶寬 {t.signal.ribbon_width:.1f}\n"
            f"量能 {t.signal.vol_ratio:.2f}x  實體 {t.signal.body:.1f}  前回檔 {t.signal.prior_drop:.1f}\n"
            f"MA {t.signal.ma5:.1f}>{t.signal.ma10:.1f}>{t.signal.ma20:.1f}>{t.signal.ma30:.1f}  "
            f"60 {t.signal.ma60:.1f} / 120 {t.signal.ma120:.1f} < 200 {t.signal.ma200:.1f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(tf)} #{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    return "".join(cards) or "<div class='empty'>無起漲點</div>"


def _tf_summary(
    interval: str,
    df: pd.DataFrame,
    trades: List[CoilTrade],
    period: str,
    funnel: Optional[Dict[str, int]],
) -> str:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    start = df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else "—"
    end = df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else "—"
    tf = TF_LABELS.get(interval, interval)
    return (
        f"<section class='summary' id='tf-{escape(interval)}'>"
        f"<h2>{escape(tf)}</h2>"
        f"<p class='muted'>{escape(period)} · {escape(start)} → {escape(end)} · bars={len(df)}</p>"
        "<div class='cards'>"
        f"<div class='card'>筆數<b>{stats['count']}</b></div>"
        f"<div class='card'>勝率<b>{stats['win_rate']:.1f}%</b></div>"
        f"<div class='card'>總點數<b class='{total_cls}'>{stats['total_points']:+.1f}</b></div>"
        f"<div class='card'>勝/負<b>{stats['wins']}/{stats['count'] - stats['wins']}</b></div>"
        "</div>"
        f"<p class='muted'>{escape(_quality_line(stats))}</p>"
        f"{_funnel_html(funnel)}"
        f"<div class='equity'>{_equity_svg(pnls)}</div>"
        "</section>"
    )


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{PAGES_BRANCH}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[CoilTrade],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    *,
    interval: str = "1m",
    others: Optional[Sequence[Tuple[str, pd.DataFrame, List[CoilTrade], Optional[Dict[str, int]]]]] = None,
    write_view: bool = False,
) -> Path:
    out = Path(path)
    img_dir = out.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    keep: Set[str] = set()

    frames: List[Tuple[str, pd.DataFrame, List[CoilTrade], Optional[Dict[str, int]]]] = [
        (interval, df, trades, funnel)
    ]
    if others:
        frames.extend(list(others))

    compare_rows = [(iv, frame, tr) for iv, frame, tr, _fun in frames]
    compare_html = _compare_table(compare_rows) if len(frames) > 1 else ""
    jump = ""
    if len(frames) > 1:
        links = " · ".join(
            f"<a href='#sec-{escape(iv)}'>{escape(TF_LABELS.get(iv, iv))}</a>" for iv, *_rest in frames
        )
        jump = f"<p class='muted'>跳到交易：{links}</p>"

    sections: List[str] = []
    for iv, frame, tr, fun in frames:
        tf = TF_LABELS.get(iv, iv)
        sections.append(f"<h2 class='tf-h' id='sec-{escape(iv)}'>{escape(tf)} 交易</h2>")
        sections.append(_trade_cards(frame, tr, img_dir, iv, keep))

    for old in img_dir.glob("*.png"):
        if old.name not in keep:
            old.unlink()

    summaries = "".join(_tf_summary(iv, frame, tr, period, fun) for iv, frame, tr, fun in frames)
    note = ""
    if len(frames) > 1:
        note = (
            "<p class='muted'>同一套規則對照：盤整 10–18 根、區間 ≤36 點、第一次收盤站上 MA200，"
            "且 5&gt;10&gt;20&gt;30、MA60/120 在 200 下方。5分K 的 10 根盤整約 50 分鐘，MA200 約 16.7 小時。"
            "5分K 由 1分K 重採樣，時段對齊。</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 起漲點（1分 / 5分對照）</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 6px}}
.tf-h{{font-size:16px;margin:22px 0 10px;padding-top:4px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.compare{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0 4px}}
.compare th,.compare td{{padding:7px 8px;border-bottom:1px solid #30363d;text-align:right;vertical-align:top}}
.compare th:first-child,.compare td:first-child{{text-align:left;color:#8b949e;font-weight:600}}
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
.tag-tf1{{background:rgba(88,166,255,0.18);color:#79c0ff;border-color:rgba(88,166,255,0.4)}}
.tag-tf5{{background:rgba(163,113,247,0.18);color:#d2a8ff;border-color:rgba(163,113,247,0.4)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
a{{color:#79c0ff}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 起漲點（均線糾結突破）</h1>
<p class="muted">圖是靜態 K 線。手機請往下捲。</p>
{note}
{compare_html}
{jump}
</section>
{summaries}
{"".join(sections)}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if write_view:
        write_view_html(out)
    return out


def print_signals(
    df: pd.DataFrame,
    sigs: Sequence[CoilSignal],
    trades: Sequence[CoilTrade],
    *,
    label: str = "",
) -> None:
    prefix = f"{label} " if label else ""
    by_entry = {t.entry_idx: t for t in trades}
    if not sigs:
        print(f"{prefix}未抓到起漲點")
        return
    for i, sig in enumerate(sigs, 1):
        ts = df.index[sig.entry_idx]
        t = by_entry.get(sig.entry_idx)
        extra = ""
        if t is not None:
            extra = f" | {t.exit_reason} {t.pnl_points:+.1f}"
        print(
            f"{prefix}[{i}] Q{sig.quality} {ts.strftime('%Y-%m-%d %H:%M')}  "
            f"起漲 {sig.entry_price:.2f}  盤整 {sig.coil_low:.2f}-{sig.coil_high:.2f}  "
            f"帶寬 {sig.ribbon_width:.1f}  量 {sig.vol_ratio:.2f}x  "
            f"停損 {sig.stop_price:.2f}  目標 {sig.target_price:.2f}{extra}"
        )


def cmd_demo(args) -> int:
    df = make_coil_demo_bars()
    funnel: Dict[str, int] = {}
    sigs = detect_coil_breakouts(df, funnel=funnel)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(f"demo bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    print_signals(df, sigs, trades)
    html_path = args.html or str(REPO_ROOT / "output" / "nq_coil_demo.html")
    out = write_html_report(html_path, df, trades, "NQ demo", "demo", funnel=funnel)
    print(f"html={out}")
    return 0 if sigs else 1


def _print_funnel(funnel: Dict[str, int]) -> None:
    if not funnel:
        return
    print(
        "funnel "
        f"checked={funnel.get('checked', 0)} coil={funnel.get('coil', 0)} "
        f"sticky={funnel.get('sticky', 0)} body={funnel.get('body', 0)} "
        f"above={funnel.get('above_coil', 0)} stack={funnel.get('stack', 0)} "
        f"ma200={funnel.get('above_200', 0)} below={funnel.get('long_below', 0)} "
        f"vol={funnel.get('volume', 0)} taken={funnel.get('taken', 0)}"
    )


def _run_tf(df: pd.DataFrame) -> Tuple[List[CoilSignal], List[CoilTrade], Dict[str, int]]:
    funnel: Dict[str, int] = {}
    sigs = detect_coil_breakouts(df, funnel=funnel)
    trades = simulate(df, sigs)
    return sigs, trades, funnel


def cmd_backtest(args) -> int:
    df1 = to_et(load_bars(args.symbol, "1m", args.period))
    if df1.empty:
        print("no data", file=sys.stderr)
        return 1
    df5 = resample_ohlcv(df1, "5min")
    sigs1, trades1, funnel1 = _run_tf(df1)
    sigs5, trades5, funnel5 = _run_tf(df5)
    stats1 = summarize_trades(trades1)
    stats5 = summarize_trades(trades5)
    print(f"{args.symbol} {args.period} 1m bars={len(df1)} {df1.index[0]} -> {df1.index[-1]}")
    print(f"1m trades={stats1['count']} WR={stats1['win_rate']:.1f}% pnl={stats1['total_points']:+.1f}")
    _print_funnel(funnel1)
    print_signals(df1, sigs1, trades1, label="1m")
    print(f"{args.symbol} {args.period} 5m bars={len(df5)} {df5.index[0]} -> {df5.index[-1]}")
    print(f"5m trades={stats5['count']} WR={stats5['win_rate']:.1f}% pnl={stats5['total_points']:+.1f}")
    _print_funnel(funnel5)
    print_signals(df5, sigs5, trades5, label="5m")
    html_path = args.html
    pages = getattr(args, "pages", False)
    if pages:
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(
            html_path,
            df1,
            trades1,
            args.symbol,
            args.period,
            funnel=funnel1,
            interval="1m",
            others=[("5m", df5, trades5, funnel5)],
            write_view=pages,
        )
        print(f"html={out}")
        if pages:
            print(f"view={out.with_name('view.html')}")
    return 0


def _ts_et(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize("UTC").tz_convert(ET)
    return ts.tz_convert(ET)


def entry_key(df, sig: CoilSignal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    return f"coil|{ts.isoformat()}|{sig.entry_price:.2f}"


def exit_key(df, tr: CoilTrade) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    return f"coil|{et.isoformat()}->{xt.isoformat()}|{tr.exit_reason}|{tr.pnl_points:.2f}"


def fmt_entry(df, sig: CoilSignal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    risk = sig.entry_price - sig.stop_price
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>起漲點（均線糾結突破）</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"品質: <b>Q{sig.quality}</b> ({sig.quality_score}/4)\n"
        f"進場: <code>{sig.entry_price:.2f}</code>\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> ({r_mult:.1f}R)\n"
        f"盤整: <code>{sig.coil_low:.1f}–{sig.coil_high:.1f}</code> "
        f"區間 {sig.coil_range:.1f} / 帶寬 {sig.ribbon_width:.1f}\n"
        f"量能: {sig.vol_ratio:.2f}x · 現價 <code>{last:.2f}</code>\n"
        f"排列: 5>10>20>30 · 站上MA200 · 60/120&lt;200\n"
        f"#起漲點 #NQ #Q{sig.quality}"
    )


def fmt_exit(df, tr: CoilTrade) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_points > 0 else ("⚪" if tr.pnl_points == 0 else "🔴")
    return (
        f"{emoji} <b>起漲點出場</b>\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.2f}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.2f}\n"
        f"原因: <b>{tr.exit_reason}</b>\n"
        f"盈虧: <b>{tr.pnl_points:+.1f} pts</b> · Q{tr.quality}\n"
        f"#起漲點 #出場"
    )


def scan_once(
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    alert_exits: bool,
    seed_alert: bool,
    lookback_hours: float,
    period: str = "5d",
) -> None:
    df = to_et(load_yfinance("NQ=F", "1m", period))
    sigs = detect_coil_breakouts(df)
    trades = simulate(df, sigs)
    state = _load_coil_state()
    alerted_e: Set[str] = set(state.get("alerted_entries") or [])
    alerted_x: Set[str] = set(state.get("alerted_exits") or [])
    now = datetime.now(ET)
    cutoff = now.timestamp() - lookback_hours * 3600
    first_run = not STATE_PATH.exists() or (not alerted_e and not state.get("initialized"))

    new_entries = []
    for sig in sigs:
        k = entry_key(df, sig)
        ts = _ts_et(df.index[sig.entry_idx])
        if ts.timestamp() < cutoff:
            alerted_e.add(k)
            continue
        if k in alerted_e:
            continue
        new_entries.append((k, sig, ts))

    if first_run and not seed_alert:
        for k, _, _ in new_entries:
            alerted_e.add(k)
        for tr in trades:
            alerted_x.add(exit_key(df, tr))
        _save_coil_state(alerted_e, alerted_x, now)
        print(
            f"[{now.strftime('%H:%M:%S')} ET] init: marked {len(new_entries)} recent signals, "
            f"bars={len(df)} last={df['Close'].iloc[-1]:.2f}"
        )
        return

    sent = 0
    for k, sig, ts in new_entries:
        ok = tg_send(token, chat_id, fmt_entry(df, sig), dry_run=dry_run)
        if ok:
            alerted_e.add(k)
            sent += 1
            print(f"[entry] {ts} Q{sig.quality} @ {sig.entry_price:.2f}")

    if alert_exits:
        for tr in trades:
            k = exit_key(df, tr)
            xt = _ts_et(df.index[tr.exit_idx])
            if xt.timestamp() < cutoff:
                alerted_x.add(k)
                continue
            if k in alerted_x:
                continue
            ok = tg_send(token, chat_id, fmt_exit(df, tr), dry_run=dry_run)
            if ok:
                alerted_x.add(k)
                sent += 1
                print(f"[exit] {xt} {tr.exit_reason} {tr.pnl_points:+.1f}")

    _save_coil_state(alerted_e, alerted_x, now)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] scan ok bars={len(df)} "
        f"sigs={len(sigs)} new_sent={sent} last={df['Close'].iloc[-1]:.2f}"
    )


def _load_coil_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": [], "alerted_exits": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": []}


def _save_coil_state(alerted_e: Set[str], alerted_x: Set[str], now: datetime) -> None:
    STATE_PATH.write_text(
        json.dumps(
            {
                "alerted_entries": sorted(alerted_e)[-200:],
                "alerted_exits": sorted(alerted_x)[-200:],
                "initialized": True,
                "last_scan": now.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2

    if args.test:
        ok = tg_send(
            token or "",
            chat_id or "",
            f"✅ 起漲點 bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    print(
        f"Coil breakout TG | interval={args.interval}s | exits={not args.no_exits} | "
        f"dry_run={args.dry_run} | lookback={args.lookback_hours}h"
    )
    while True:
        try:
            scan_once(
                token or "",
                chat_id or "",
                dry_run=args.dry_run,
                alert_exits=not args.no_exits,
                seed_alert=args.seed_alert,
                lookback_hours=args.lookback_hours,
                period=args.period,
            )
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(max(15, args.interval))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 起漲點（均線糾結突破，1分/5分對照）")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="用模擬那張圖的走勢示範")
    d.add_argument("--html", default="")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("backtest", help="Yahoo 1m 回測，並重採樣 5m 對照")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-coil-breakout/index.html")
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("alert", help="Telegram 輪詢")
    a.add_argument("--interval", type=int, default=None)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--no-exits", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=None)
    a.add_argument("--period", default="5d")
    a.set_defaults(func=cmd_alert)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--demo", action="store_true", help="用模擬資料（無子命令時）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "alert":
        if args.interval is None:
            args.interval = int(env("POLL_SECONDS", "60") or 60)
        if args.lookback_hours is None:
            args.lookback_hours = float(env("LOOKBACK_HOURS", "36") or 36)
        return cmd_alert(args)
    if args.cmd == "demo" or (args.cmd is None and getattr(args, "demo", False)):
        if args.cmd is None:
            args.html = args.html
        return cmd_demo(args)
    if args.cmd is None:
        args.cmd = "backtest"
        return cmd_backtest(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
