"""5>10>20>60 站上 MA200 回測圖（靜態 PNG + Markdown）。"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from nq.align200 import Align200Trade, add_align_features, summarize_align

MA_COLORS = {5: "#42a5f5", 10: "#66bb6a", 20: "#ffa726", 60: "#26c6da", 200: "#ef5350"}


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts


def _fmt(ts: pd.Timestamp) -> str:
    return _naive(ts).strftime("%m-%d %H:%M")


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _style(ax) -> None:
    ax.set_facecolor("#10141a")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.grid(True, color="#ffffff10", lw=0.6)


def draw_trade(df: pd.DataFrame, trade: Align200Trade, path: Path, trade_no: int) -> Path:
    work = add_align_features(df)
    start = max(0, trade.signal.bar_idx - 50)
    end = min(len(work) - 1, trade.signal.bar_idx + 30)
    for i in range(trade.signal.bar_idx, len(work)):
        if work.index[i] == trade.exit_time:
            end = min(len(work) - 1, i + 8)
            break
    window = work.iloc[start : end + 1]
    xs = np.arange(len(window))
    o, h, l, c = (window[k].to_numpy() for k in ("open", "high", "low", "close"))
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.4, 5.4), sharex=True, gridspec_kw={"height_ratios": [3.1, 1]}, facecolor="#161b22"
    )
    _style(ax)
    _style(axv)
    vols = []
    for i in range(len(c)):
        up = c[i] >= o[i]
        col = "#ef5350" if up else "#26a69a"
        ax.vlines(xs[i], l[i], h[i], color=col, lw=0.8)
        y0, y1 = min(o[i], c[i]), max(o[i], c[i])
        if y1 == y0:
            y1 = y0 + max(h[i] - l[i], 1e-9) * 0.04
        ax.add_patch(Rectangle((xs[i] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        vols.append("#ef535099" if up else "#26a69a99")
    axv.bar(xs, window["volume"], width=0.8, color=vols, linewidth=0)
    for n, color in MA_COLORS.items():
        ax.plot(xs, window[f"ma{n}"], color=color, lw=1.3, label=f"MA{n}")
    in_mask = window.index == trade.signal.timestamp
    out_mask = window.index == trade.exit_time
    if bool(in_mask.any()):
        ax.scatter([int(in_mask.argmax())], [trade.signal.entry], marker="^", s=46, color="#00e676", zorder=5, label="IN")
    if bool(out_mask.any()):
        ax.scatter([int(out_mask.argmax())], [trade.exit_price], marker="x", s=42, color="#ff5252", zorder=5, label="OUT")
    ticks = list(range(0, len(window), max(1, len(window) // 6)))
    axv.set_xticks(ticks)
    axv.set_xticklabels([_naive(window.index[i]).strftime("%m-%d %H:%M") for i in ticks])
    ax.set_title(
        f"#{trade_no} {trade.signal.symbol}  1m  {_fmt(trade.signal.timestamp)}",
        color="#e6edf3",
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c9d1d9", ncol=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_gallery(trades: list[Align200Trade], limit: int = 24) -> list[Align200Trade]:
    """每天先抽幾筆，再補最大賺／最大虧，避免圖全擠在第一天。"""
    chosen: list[Align200Trade] = []
    seen: set[tuple] = set()

    def add(trade: Align200Trade) -> None:
        key = (trade.signal.symbol, trade.signal.timestamp)
        if key in seen:
            return
        seen.add(key)
        chosen.append(trade)

    by_day: dict = {}
    for trade in trades:
        by_day.setdefault(trade.signal.day, []).append(trade)
    for day_trades in by_day.values():
        for trade in day_trades[:3]:
            add(trade)
    ranked = sorted(trades, key=lambda t: t.pnl_pct_net, reverse=True)
    for trade in ranked[:5] + ranked[-5:]:
        add(trade)
    for trade in trades:
        if len(chosen) >= limit:
            break
        add(trade)
    chosen.sort(key=lambda t: (t.signal.day, t.signal.timestamp, t.signal.symbol))
    return chosen[:limit]


def write_report(
    output_dir: Path,
    *,
    title: str,
    trades: list[Align200Trade],
    frames: dict[str, pd.DataFrame],
    notes: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()
    stats = summarize_align(trades)
    lines = [
        f"# {title}",
        "",
        "請直接開這個 Markdown，GitHub 會顯示圖。",
        "",
        f"- 成交 **{stats['trades']}**　勝 **{stats['wins']}**　勝率 **{stats['win_rate']*100:.0f}%**　單筆期望 **{_pct(stats['expectancy_net'])}**",
        f"- 把每筆淨報酬加總 **{_pct(stats['total_pnl_pct_net'])}**（不是資金曲線；每天最多同時一百檔）",
        "",
        "規則：一分K **MA5>MA10>MA20>MA60** 且收盤**站上 MA200**，條件剛成立才通知／進場。",
        "",
        "### 每日訊號",
        "",
        "| 日期 | 筆數 |",
        "| --- | ---: |",
    ]
    for day in sorted(stats["by_day"]):
        lines.append(f"| {day.isoformat()} | {stats['by_day'][day]} |")
    lines += [
        "",
        "### 出場",
        "",
        "| 原因 | 筆數 |",
        "| --- | ---: |",
    ]
    for reason, n in sorted(stats["by_exit"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{reason}` | {n} |")
    lines += ["", "## 抽樣圖", ""]
    cards = []
    gallery = _pick_gallery(trades)
    for i, trade in enumerate(gallery, start=1):
        df = frames.get(trade.signal.symbol)
        png = img_dir / f"{i:02d}_{trade.signal.symbol.replace('.', '-')}.png"
        if df is not None and len(df):
            draw_trade(df, trade, png, i)
            img_rel = f"img/{png.name}"
        else:
            img_rel = ""
        lines += [
            f"### #{i} {trade.signal.symbol} {trade.signal.name}　{_fmt(trade.signal.timestamp)}　{_pct(trade.pnl_pct_net)}",
            "",
            f"進場 `{trade.signal.entry:.2f}`　出場 `{trade.exit_price:.2f}`　{trade.exit_reason}",
            "",
            f"5/10/20/60 `{trade.signal.ma5:.2f}` / `{trade.signal.ma10:.2f}` / `{trade.signal.ma20:.2f}` / `{trade.signal.ma60:.2f}`　MA200 `{trade.signal.ma200:.2f}`",
            "",
        ]
        if img_rel:
            lines += [f"![t{i}]({img_rel})", ""]
        cards.append(
            f'<article class="trade"><h3>#{i} {html.escape(trade.signal.symbol)} {html.escape(trade.signal.name)} '
            f'<span class="{"pos" if trade.pnl_pct_net>=0 else "neg"}">{_pct(trade.pnl_pct_net)}</span></h3>'
            f"<p>{_fmt(trade.signal.timestamp)} → {_fmt(trade.exit_time)}　{html.escape(trade.exit_reason)}</p>"
            + (f'<img src="{img_rel}" alt="{i}"/>' if img_rel else "")
            + "</article>"
        )
    lines += [
        "## 全部成交",
        "",
        "| # | 日期 | 代號 | 名稱 | 進場 | 出場 | 淨% | 原因 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for i, trade in enumerate(trades, start=1):
        lines.append(
            f"| {i} | {trade.signal.day.isoformat()} | {trade.signal.symbol} | {trade.signal.name} | "
            f"{trade.signal.entry:.2f} | {trade.exit_price:.2f} | {_pct(trade.pnl_pct_net)} | {trade.exit_reason} |"
        )
    lines.append("")
    md = output_dir / "README.md"
    md.write_text("\n".join(lines + ["## 規則"] + [f"- {n}" for n in notes] + [""]), encoding="utf-8")
    html_page = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:20px 14px 48px}}
.pos{{color:#3ddc68}} .neg{{color:#ff7b72}}
img{{width:100%;border-radius:8px;background:#10141a}}
.trade{{margin:0 0 18px;padding:12px;border:1px solid #30363d;border-radius:12px;background:#161b22}}
p,li{{color:#8b949e}}
</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p>成交 {stats['trades']}　勝率 {stats['win_rate']*100:.0f}%　淨 {_pct(stats['total_pnl_pct_net'])}</p>
<p>請改開 <a href="./README.md">README.md</a> 看圖（GitHub 會顯示）。</p>
{''.join(cards)}
</div></body></html>
"""
    (output_dir / "index.html").write_text(html_page, encoding="utf-8")
    return md
