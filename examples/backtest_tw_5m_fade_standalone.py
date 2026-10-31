#!/usr/bin/env python3
"""台股 5m MA60 空頭：指定區間回測，產出單一自包含 HTML（圖片內嵌）。只回測、不下單。"""

from __future__ import annotations

import argparse
import base64
import sys
from datetime import date, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import TPE  # noqa: E402
from watch_tw_5m_fade import (  # noqa: E402
    DEFAULT_MA_N,
    MAX_BREAK_PCT,
    MIN_BREAK_PCT,
    MIN_PRIOR_OVER_PCT,
    PRIOR_LOOKBACK,
    PRIOR_MIN_BARS,
    REPO,
    detect_signals,
    draw_signal_png,
    fetch_yahoo_5m,
    hit_since,
    hit_until,
    hit_within_max_price,
    resolve_universe,
    simulate,
)

SKIP_HOLIDAYS = {date(2026, 10, 23)}  # 颱風休市
SESSION_OPEN = 9 * 60
SESSION_CLOSE = 13 * 60 + 30  # 含 13:25 那根（涵蓋到 13:30）


def clip_regular_session(df: pd.DataFrame, holidays: set[date] | None = None) -> pd.DataFrame:
    """只留 09:00–13:30 正規盤 5 分 K；丟掉盤前／試搓，休市日整日略過。"""
    if df is None or df.empty:
        return df
    holidays = holidays or SKIP_HOLIDAYS
    idx = df.index
    minutes = idx.hour * 60 + idx.minute
    keep = (minutes >= SESSION_OPEN) & (minutes <= SESSION_CLOSE)
    days = np.array([ts.date() for ts in idx])
    if holidays:
        keep &= ~np.isin(days, list(holidays))
    return df.loc[keep].copy()


def extra_stats(pnls: list[float]) -> dict:
    n = len(pnls)
    if not n:
        return {
            "count": 0,
            "wins": 0,
            "win_rate": 0.0,
            "mean_pct": 0.0,
            "median_pct": 0.0,
            "best_pct": 0.0,
            "worst_pct": 0.0,
            "total_pct": 0.0,
        }
    arr = np.array(pnls, dtype=float)
    wins = int(np.sum(arr > 0))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n,
        "mean_pct": float(arr.mean() * 100.0),
        "median_pct": float(np.median(arr) * 100.0),
        "best_pct": float(arr.max() * 100.0),
        "worst_pct": float(arr.min() * 100.0),
        "total_pct": float(arr.sum() * 100.0),
    }


def draw_to_bytes(df, sig, title: str, trade) -> bytes:
    buf = Path("/tmp") / f"tw5m_fade_{title.replace(' ', '_')}_{sig.entry_idx}.png"
    draw_signal_png(df, sig, buf, title, trade=trade)
    data = buf.read_bytes()
    try:
        buf.unlink()
    except OSError:
        pass
    return data


def write_standalone_html(
    path: Path,
    hits: list,
    universe: list[dict],
    *,
    since: date,
    until: date,
    holidays: set[date],
) -> Path:
    pnls = [h[2].pnl_pct for h in hits if h[2] is not None]
    stats = extra_stats(pnls)
    holiday_txt = "、".join(d.isoformat() for d in sorted(holidays)) or "無"
    reason_map = {"stop": "停損", "target": "停利 2R", "eod": "當日收盤"}

    rules = f"""
<section class="rules">
<h2>觸發規則（對齊 examples/watch_tw_5m_fade.py）</h2>
<ul>
<li><b>週期</b>：Yahoo 5 分 K（interval=5m，includePrePost=false），非整點 5 分 K 丟掉。</li>
<li><b>時段</b>：只留台北 09:00–13:30 正規盤（K 線時間戳 09:00 至 13:30 含）；不納入試搓／盤前集合競價。</li>
<li><b>休市</b>：{escape(holiday_txt)} 整日略過（例如 10/23 颱風）。</li>
<li><b>標的</b>：TWSE／TPEX 成交額前 80，收盤價 &gt; 700 剔除；本檔實掃 {len(universe)} 檔。</li>
<li><b>均線</b>：SMA 5／10／20／60（預設跌破 MA60）。</li>
<li><b>空排</b>：MA5 &lt; MA10 &lt; MA20，且三條都比前一根低。</li>
<li><b>跌破</b>：前一根收盤 ≥ MA60，當根收盤 &lt; MA60，且收盤低於 MA5／MA10／MA20／MA60。</li>
<li><b>距線</b>：(MA60 − 收盤)／MA60 介於 0.7%–3%（MIN_BREAK_PCT={MIN_BREAK_PCT}，MAX_BREAK_PCT={MAX_BREAK_PCT}）。</li>
<li><b>前一盤</b>：前一個日曆交易日最後 {PRIOR_LOOKBACK} 根裡，至少 {PRIOR_MIN_BARS} 根有效，(收盤−MA60)／MA60 的最小值 ≥ {MIN_PRIOR_OVER_PCT*100:.1f}%；不含當天。</li>
<li><b>次數</b>：同一標的同一天只取第一筆。</li>
<li><b>進場價</b>：跌破那根的收盤價（進場根＝破線根）。</li>
<li><b>停損</b>：max(破線根最高價, MA60) × 1.003；後續 K 的最高價觸及即出場（同根先檢查停損再檢查停利）。</li>
<li><b>停利</b>：進場價 − 2 ×（停損 − 進場價），後續 K 的最低價觸及即出場。</li>
<li><b>時間出場</b>：當日最後一根 5 分 K 收盤（eod）。</li>
<li><b>報酬</b>：做空 (進場 − 出場)／進場。未扣手續費、證交稅、融券費。</li>
</ul>
</section>
"""

    cards = []
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        et = df.index[sig.entry_idx]
        bt = df.index[sig.break_idx]
        label = f"{row['code']} {row.get('name') or ''}".strip()
        png = draw_to_bytes(df, sig, label, trade)
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        pnl_html = ""
        exit_txt = "無模擬成交"
        if trade is not None:
            cls = "pnl-win" if trade.pnl_pct > 0 else ("pnl-flat" if trade.pnl_pct == 0 else "pnl-loss")
            pnl_html = f"<div class='card-pnl {cls}'>{trade.pnl_pct*100:+.2f}%</div>"
            xt = df.index[trade.exit_idx]
            exit_txt = (
                f"出場 {trade.exit_price:.2f}  {reason_map.get(trade.exit_reason, trade.exit_reason)} "
                f"@ {xt.strftime('%H:%M')}\n"
                f"停損 {trade.stop_price:.2f}  停利 {trade.target_price:.2f}"
            )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))}</span></div>"
            f"{pnl_html}"
            "</header>"
            f"<div class='tags'><span class='tag'>{escape(row['symbol'])}</span>"
            f"<span class='tag'>5分K</span><span class='tag'>5&lt;10&lt;20</span>"
            f"<span class='tag'>破MA{sig.ma_n}</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {sig.entry_price:.2f}  破 MA{sig.ma_n} {sig.ma240:.2f} @ {bt.strftime('%H:%M')}\n"
            f"前收 {sig.prev_close:.2f}  距線 {sig.dist_pct*100:.2f}%  前一盤 {sig.prior_over*100:+.2f}%\n"
            f"MA5 {sig.ma5:.2f}  MA10 {sig.ma10:.2f}  MA20 {sig.ma20:.2f}  MA{sig.ma_n} {sig.ma240:.2f}\n"
            f"{exit_txt}"
            "</pre>"
            f"<div class='mini-chart'><img src='{uri}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    tot_cls = "pnl-win" if stats["total_pct"] >= 0 else "pnl-loss"
    mean_cls = "pnl-win" if stats["mean_pct"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 5分K 空排跌破 MA60 · {since.isoformat()}–{until.isoformat()}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:640px;margin:0 auto;padding:14px 12px 32px}}
.summary,.rules{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:0 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.rules ul{{margin:0;padding-left:1.2em;color:#c9d1d9;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#ef4444}} .pnl-loss{{color:#22c55e}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
.note{{color:#8b949e;font-size:12px;margin-top:8px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 5分K 空頭排列跌破 MA60</h1>
<p class="muted">{since.isoformat()}–{until.isoformat()} · 正規盤 09:00–13:30 · 休市略過 {escape(holiday_txt)} · 股價≤700 · {len(universe)} 檔
<br/>報酬未扣手續費與稅。只回測、不下單。</p>
<div class="cards">
<div class="card">訊號數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">平均<b class="{mean_cls}">{stats['mean_pct']:+.2f}%</b></div>
<div class="card">中位數<b>{stats['median_pct']:+.2f}%</b></div>
<div class="card">最好<b class="pnl-win">{stats['best_pct']:+.2f}%</b></div>
<div class="card">最差<b class="pnl-loss">{stats['worst_pct']:+.2f}%</b></div>
<div class="card">合計<b class="{tot_cls}">{stats['total_pct']:+.2f}%</b></div>
<div class="card">標的<b>{len({h[0]['code'] for h in hits}) if hits else 0}</b></div>
</div>
<p class="note">勝：報酬 &gt; 0。綠跌紅漲。圖上綠三角＝進場，黃叉＝出場。</p>
</section>
{rules}
{''.join(cards) or "<div class='empty'>這段期間沒有訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="台股 5m MA60 空頭區間回測 → 自包含 HTML")
    p.add_argument("--since", default="2026-10-17")
    p.add_argument("--until", default="2026-10-30")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--pool", type=int, default=80)
    p.add_argument("--max-price", type=float, default=700)
    p.add_argument("--range", dest="range_", default="60d")
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--date", default="")
    p.add_argument("--symbols", default="")
    p.add_argument("--also", default="")
    p.add_argument("--html", default=str(REPO / "docs" / "tw-5m-fade-2w-1017-1030" / "index.html"))
    args = p.parse_args()
    since = datetime.strptime(args.since, "%Y-%m-%d").date()
    until = datetime.strptime(args.until, "%Y-%m-%d").date()

    universe = resolve_universe(args)
    hits = []
    errors = 0
    import time

    for i, row in enumerate(universe, 1):
        try:
            df = clip_regular_session(fetch_yahoo_5m(row["symbol"], args.range_))
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:3d}/{len(universe)}] {row['symbol']} ERR {exc}")
            errors += 1
            time.sleep(max(0.05, args.sleep))
            continue
        if row.get("close") is None and len(df):
            row["close"] = float(df["Close"].iloc[-1])
        sigs = detect_signals(df, ma_n=DEFAULT_MA_N)
        trades = {t.entry_idx: t for t in simulate(df, sigs)}
        kept = 0
        for sig in sigs:
            if not hit_since(df, sig, since) or not hit_until(df, sig, until):
                continue
            ts = df.index[sig.entry_idx]
            if ts.date() in SKIP_HOLIDAYS:
                continue
            minutes = ts.hour * 60 + ts.minute
            if minutes < SESSION_OPEN or minutes > SESSION_CLOSE:
                continue
            if not hit_within_max_price(row, sig, df, args.max_price):
                continue
            hits.append((row, sig, trades.get(sig.entry_idx), df))
            kept += 1
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row.get('name','')} bars={len(df)} sigs={kept}")
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h[3].index[h[1].entry_idx])
    pnls = [h[2].pnl_pct for h in hits if h[2] is not None]
    stats = extra_stats(pnls)
    print(
        f"done errors={errors} n={stats['count']} WR={stats['win_rate']:.1f}% "
        f"mean={stats['mean_pct']:+.2f}% med={stats['median_pct']:+.2f}% "
        f"best={stats['best_pct']:+.2f}% worst={stats['worst_pct']:+.2f}% "
        f"sum={stats['total_pct']:+.2f}%"
    )
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        ts = df.index[sig.entry_idx]
        extra = f" {trade.exit_reason} {trade.pnl_pct*100:+.2f}%" if trade else ""
        print(f"  [{i}] {row['code']} {row.get('name','')} {ts.strftime('%m-%d %H:%M')}{extra}")

    out = write_standalone_html(Path(args.html), hits, universe, since=since, until=until, holidays=SKIP_HOLIDAYS)
    print(f"html={out} size={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
