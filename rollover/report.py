"""一個月回測的摘要和每筆 5 分 K 圖。"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from rollover.strategy import ShortTrade, qty_for_risk

TPE = "Asia/Taipei"
KIND = {"break": "跌破", "retest": "回測"}
REASON = {"stop": "停損", "target": "停利", "time": "時間到"}


def write_report(
    path: str | Path,
    df5: pd.DataFrame,
    df1: pd.DataFrame,
    trades: list[ShortTrade],
    *,
    symbol: str,
    risk_usdt: float,
    fee_rate: float,
    cutoff: pd.Timestamp,
) -> dict:
    """寫出摘要、權益曲線，以及每筆交易一張 5 分 K。回傳統計。"""

    out = Path(path)
    img_dir = out.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    frame5 = _ohlcv(df5)
    month5 = frame5.loc[frame5.index >= cutoff]
    rows = [_trade_row(trade, risk_usdt, fee_rate) for trade in trades]
    stats = _stats(rows, month5 if len(month5) else frame5)
    _draw_equity(rows, img_dir / "equity.png", symbol)
    _draw_overview(month5 if len(month5) else frame5, trades, img_dir / "overview.png", symbol)
    cards: list[str] = []
    for i, (trade, row) in enumerate(zip(trades, rows), start=1):
        name = f"t{i:02d}_{_stamp(trade.signal.timestamp).strftime('%m%d_%H%M')}.png"
        _draw_trade(frame5, trade, row, img_dir / name, i)
        cards.append(_card(i, row, f"img/{name}"))
    page = _page(symbol, stats, cards, risk_usdt, fee_rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return stats


def _trade_row(trade: ShortTrade, risk_usdt: float, fee_rate: float) -> dict:
    sig = trade.signal
    risk = sig.risk_per_unit
    qty = qty_for_risk(sig.entry, sig.stop, risk_usdt)
    r_mult = trade.pnl_per_unit / risk if risk > 0 else 0.0
    gross = qty * trade.pnl_per_unit
    fee = qty * fee_rate * (sig.entry + trade.exit_price)
    return {
        "trade": trade,
        "kind": KIND.get(sig.kind, sig.kind),
        "reason": REASON.get(trade.exit_reason, trade.exit_reason),
        "reason_key": trade.exit_reason,
        "entry": sig.entry,
        "stop": sig.stop,
        "target": sig.target,
        "exit": trade.exit_price,
        "extreme": sig.extreme,
        "neckline": sig.neckline,
        "lower_high": sig.lower_high,
        "four_hour_high": sig.four_hour_high,
        "r": r_mult,
        "qty": qty,
        "gross": gross,
        "fee": fee,
        "net": gross - fee,
        "off_session": sig.off_session,
        "entry_time": sig.timestamp,
        "exit_time": trade.exit_time,
    }


def _stats(rows: list[dict], month5: pd.DataFrame) -> dict:
    count = len(rows)
    wins = sum(1 for row in rows if row["r"] > 0)
    gross = sum(row["gross"] for row in rows)
    net = sum(row["net"] for row in rows)
    equity = np.cumsum([row["net"] for row in rows]) if rows else np.array([0.0])
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min()) if len(equity) else 0.0
    by_reason: dict[str, dict] = {}
    for key in ("target", "stop", "time"):
        picked = [row for row in rows if row["reason_key"] == key]
        by_reason[key] = {
            "n": len(picked),
            "r": sum(row["r"] for row in picked),
        }
    def bucket(off: bool) -> dict:
        picked = [row for row in rows if row["off_session"] is off]
        n = len(picked)
        w = sum(1 for row in picked if row["r"] > 0)
        return {
            "n": n,
            "win_rate": (100.0 * w / n) if n else 0.0,
            "r": sum(row["r"] for row in picked),
            "net": sum(row["net"] for row in picked),
        }

    start = month5.index[0] if len(month5) else pd.Timestamp.now(tz="UTC")
    end = month5.index[-1] if len(month5) else start
    return {
        "count": count,
        "wins": wins,
        "losses": count - wins,
        "win_rate": (100.0 * wins / count) if count else 0.0,
        "total_r": sum(row["r"] for row in rows),
        "gross_usdt": gross,
        "fees_usdt": sum(row["fee"] for row in rows),
        "net_usdt": net,
        "max_dd_usdt": max_dd,
        "by_reason": by_reason,
        "cash": bucket(False),
        "off": bucket(True),
        "start": _fmt(start),
        "end": _fmt(end),
    }


def _card(number: int, row: dict, src: str) -> str:
    pnl_cls = "pnl-win" if row["net"] >= 0 else "pnl-loss"
    reason_cls = {"target": "tag-tp", "stop": "tag-sl"}.get(row["reason_key"], "tag-time")
    session = "現貨休市" if row["off_session"] else "現貨時段"
    session_cls = "tag-off" if row["off_session"] else "tag-info"
    detail = (
        f"進場 {row['entry']:.4f}  停損 {row['stop']:.4f}  目標 {row['target']:.4f}\n"
        f"出場 {row['exit']:.4f}  {row['reason']}  {row['r']:+.2f}R\n"
        f"極點 {row['extreme']:.4f}  頸線 {row['neckline']:.4f}  較低高點 {row['lower_high']:.4f}  "
        f"四小時高點 {row['four_hour_high']:.4f}\n"
        f"張數 {row['qty']}  毛損益 {row['gross']:+.2f}  手續費 {row['fee']:.2f}  淨損益 {row['net']:+.2f} USDT"
    )
    return f"""
    <article class="trade-card">
      <header class="card-header">
        <div>
          <div class="trade-no">#{number} · {html.escape(row['kind'])}</div>
          <div class="trade-time">{html.escape(_fmt(row['entry_time']))} → {html.escape(_fmt(row['exit_time']))}</div>
        </div>
        <div class="card-pnl {pnl_cls}">{row['net']:+.0f} USDT</div>
      </header>
      <div class="tags">
        <span class="tag {reason_cls}">{html.escape(row['reason'])}</span>
        <span class="tag {session_cls}">{session}</span>
        <span class="tag tag-info">{row['r']:+.2f}R</span>
      </div>
      <pre class="trade-detail">{html.escape(detail)}</pre>
      <img src="{html.escape(src)}" alt="第 {number} 筆 5 分 K"/>
    </article>
    """


def _page(symbol: str, stats: dict, cards: list[str], risk_usdt: float, fee_rate: float) -> str:
    net_cls = "pnl-win" if stats["net_usdt"] >= 0 else "pnl-loss"
    reasons = " · ".join(
        f"{REASON[key]} {stats['by_reason'][key]['n']} 筆 {stats['by_reason'][key]['r']:+.1f}R"
        for key in ("target", "stop", "time")
        if stats["by_reason"][key]["n"]
    )
    cash, off = stats["cash"], stats["off"]
    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(symbol)} 起跌放空 一個月</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif}}
.page{{max-width:760px;margin:0 auto;padding:16px 12px 40px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:16px;margin:18px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.summary,.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:108px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}}
.trade-time{{color:#8b949e;font-size:12px;margin-top:3px}}
.card-pnl{{font-weight:700;white-space:nowrap}}
.pnl-win{{color:#3ddc68}} .pnl-loss{{color:#ff7b72}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px}}
.tag-tp{{background:rgba(0,200,5,.15);color:#3ddc68}}
.tag-sl{{background:rgba(255,82,82,.15);color:#ff7b72}}
.tag-time{{background:rgba(255,193,7,.12);color:#f0c14b}}
.tag-info{{background:rgba(88,166,255,.12);color:#79c0ff}}
.tag-off{{background:rgba(180,180,180,.12);color:#c9d1d9}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;white-space:pre-wrap}}
img{{width:100%;display:block;border-radius:10px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{html.escape(symbol)} 起跌放空 · 近一個月</h1>
<p class="muted">{html.escape(stats['start'])} → {html.escape(stats['end'])}<br/>
5 分定極點、較低高點、頸線。1 分收盤跌破頸線、收在 MA25 下、量能回到前 10 根均量才放空。每筆風險預算 {risk_usdt:.0f} USDT。單邊費率 {fee_rate * 100:.3f}%，進出各計一次。張數是整數。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">合計<b>{stats['total_r']:+.2f}R</b></div>
<div class="card">淨損益<b class="{net_cls}">{stats['net_usdt']:+.0f}</b></div>
</div>
<p class="muted">毛損益 {stats['gross_usdt']:+.0f} USDT，手續費 {stats['fees_usdt']:.0f}，淨值最大回撤 {stats['max_dd_usdt']:.0f} USDT。<br/>
{html.escape(reasons) or "這段沒有進場"}<br/>
現貨時段 {cash['n']} 筆、勝率 {cash['win_rate']:.0f}%、{cash['net']:+.0f} USDT。休市 {off['n']} 筆、勝率 {off['win_rate']:.0f}%、{off['net']:+.0f} USDT。</p>
<h2>淨損益曲線</h2>
<img src="img/equity.png" alt="每筆累計淨損益"/>
<h2>這一個月的 5 分收盤和進場點</h2>
<img src="img/overview.png" alt="5 分收盤與放空進場"/>
</section>
{''.join(cards) if cards else '<p class="muted">這一個月沒有進場。</p>'}
</div></body></html>
"""


def _draw_equity(rows: list[dict], path: Path, symbol: str) -> None:
    plt = _plt()
    equity = np.cumsum([0.0] + [row["net"] for row in rows])
    fig, ax = plt.subplots(figsize=(10.2, 3.4), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    xs = list(range(len(equity)))
    color = "#3dba7a" if equity[-1] >= 0 else "#e35d5d"
    ax.plot(xs, equity, color=color, lw=1.8)
    ax.axhline(0, color="#2a3a33", lw=0.8)
    ax.set_title(f"{symbol}  每筆累計淨損益（USDT）", color="#e8f0ea", fontsize=12)
    ax.set_xlabel("筆數", color="#8aa193")
    ax.set_ylabel("累計 USDT", color="#8aa193")
    _style_ax(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_overview(df5: pd.DataFrame, trades: list[ShortTrade], path: Path, symbol: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.2, 4.2), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    close = df5["close"].astype(float)
    ax.plot(range(len(close)), close.to_numpy(), color="#8aa193", lw=0.8)
    for trade in trades:
        idx = int(df5.index.searchsorted(trade.signal.timestamp, side="right")) - 1
        if idx < 0 or idx >= len(df5):
            continue
        win = trade.pnl_per_unit > 0
        ax.scatter([idx], [trade.signal.entry], s=18, color="#3dba7a" if win else "#e35d5d", marker="v", zorder=4)
    step = max(1, len(df5) // 6)
    ticks = list(range(0, len(df5), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt(df5.index[i])[5:16] for i in ticks], rotation=20, ha="right")
    ax.set_title(f"{symbol}  5 分收盤  ·  倒三角是放空進場", color="#e8f0ea", fontsize=12)
    ax.set_ylabel("價格", color="#8aa193")
    _style_ax(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_trade(df5: pd.DataFrame, trade: ShortTrade, row: dict, path: Path, number: int) -> None:
    plt = _plt()
    from matplotlib.patches import Rectangle

    sig = trade.signal
    exit_i = int(df5.index.searchsorted(trade.exit_time, side="right")) - 1
    exit_i = min(max(exit_i, 0), len(df5) - 1)
    start = max(0, min(sig.extreme_idx, sig.lower_high_idx) - 6)
    end = min(len(df5) - 1, max(exit_i + 4, sig.lower_high_idx + 4))
    if end - start > 90:
        start = max(0, end - 90)
    window = df5.iloc[start : end + 1]
    xs = list(range(len(window)))
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.2, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1]}, facecolor="#0c1210",
    )
    for artist in (ax, axv):
        artist.set_facecolor("#101814")
        _style_ax(artist)
    opens, highs, lows, closes = (window[name].astype(float) for name in ("open", "high", "low", "close"))
    for k in xs:
        up = float(closes.iloc[k]) >= float(opens.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(k, float(lows.iloc[k]), float(highs.iloc[k]), color=col, lw=0.7)
        y0, y1 = sorted((float(opens.iloc[k]), float(closes.iloc[k])))
        if y1 == y0:
            y1 = y0 + 1e-6
        ax.add_patch(Rectangle((k - 0.32, y0), 0.64, y1 - y0, facecolor=col, edgecolor=col, lw=0.2))
        axv.bar(k, float(window["volume"].iloc[k]), width=0.7, color=col, alpha=0.85)
    full_close = df5["close"].astype(float)
    for period, color in ((7, "#ffa726"), (25, "#42a5f5"), (200, "#ab47bc")):
        ma = full_close.rolling(period, min_periods=period).mean().iloc[start : end + 1]
        ax.plot(xs, ma, color=color, lw=1.0, label=f"MA{period}")
    if sig.four_hour_high > 0:
        ax.axhline(sig.four_hour_high, color="#8aa193", ls="-.", lw=0.8, label="4小時高")
    ax.axhline(sig.neckline, color="#ffa726", ls="--", lw=0.9, label="頸線")
    ax.axhline(sig.stop, color="#e35d5d", ls=":", lw=1.0, label="停損")
    ax.axhline(sig.target, color="#3dba7a", ls=":", lw=1.0, label="目標")
    _mark(ax, sig.extreme_idx - start, sig.extreme, len(window), "極點", "#c9d1d9")
    _mark(ax, sig.lower_high_idx - start, sig.lower_high, len(window), "次高", "#f0c14b")
    entry_i = int(df5.index.searchsorted(sig.timestamp, side="right")) - 1 - start
    if 0 <= entry_i < len(window):
        ax.scatter([entry_i], [sig.entry], s=46, marker="v", color="#ff7b72", zorder=6)
        ax.annotate("空", (entry_i, sig.entry), textcoords="offset points", xytext=(0, 8), ha="center", color="#ff7b72", fontsize=8)
    exit_rel = exit_i - start
    if 0 <= exit_rel < len(window):
        exit_color = "#3dba7a" if trade.pnl_per_unit > 0 else "#ff7b72"
        ax.scatter([exit_rel], [trade.exit_price], s=36, marker="x", color=exit_color, zorder=6)
    lows_min = float(min(lows.min(), sig.target, sig.entry))
    highs_max = float(max(highs.max(), sig.stop, sig.entry))
    if start <= sig.extreme_idx <= end:
        highs_max = max(highs_max, sig.extreme)
    pad = max((highs_max - lows_min) * 0.08, 0.05)
    ax.set_ylim(lows_min - pad, highs_max + pad)
    sign = "+" if row["net"] >= 0 else ""
    session = "休市" if row["off_session"] else "現貨"
    ax.set_title(
        f"#{number}  {_fmt(sig.timestamp)}  {row['kind']}  {row['reason']}  {row['r']:+.2f}R  {sign}{row['net']:.0f} USDT  {session}",
        color="#e8f0ea", fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([_fmt(window.index[i])[5:16] for i in ticks], rotation=20, ha="right")
    axv.set_ylabel("量", color="#8aa193")
    fig.tight_layout(pad=0.45)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _mark(ax, rel: int, price: float, length: int, label: str, color: str) -> None:
    if 0 <= rel < length:
        ax.scatter([rel], [price], s=28, color=color, zorder=5)
        ax.annotate(label, (rel, price), textcoords="offset points", xytext=(0, 7), ha="center", color=color, fontsize=8)


def _style_ax(ax) -> None:
    ax.tick_params(colors="#8aa193", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3a33")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
    return plt


def _ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).lower() for col in df.columns}
    frame = df.rename(columns=renamed).loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
    frame.index = pd.DatetimeIndex(df.index)
    return frame.sort_index()


def _stamp(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(TPE)


def _fmt(ts: pd.Timestamp) -> str:
    return _stamp(ts).strftime("%Y-%m-%d %H:%M") + " 台北"
