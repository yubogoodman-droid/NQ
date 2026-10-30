#!/usr/bin/env python3
"""台股 1h 破底翻：指定進場視窗回測，輸出單一自包含 HTML（圖內嵌）。

只回測、不下單。參數完全沿用 tw_1h_reclaim.loose_params()。
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import tempfile
import time
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_reclaim import (  # noqa: E402
    REPO,
    TPE,
    TwHit,
    detect_signals,
    draw_trade_png,
    fetch_top_turnover,
    fetch_yahoo_1h,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    loose_params,
    resolve_twse_date,
    simulate,
    strict_params,
    summarize_trades,
)

WINDOW_START = pd.Timestamp("2026-10-17 09:00", tz=TPE)
WINDOW_END = pd.Timestamp("2026-10-30 13:30", tz=TPE)
SKIP_DATES: Set[date] = {date(2026, 10, 23), date(2026, 10, 26)}  # TWSE 無成交（10/23 颱風等）
DEFAULT_OUT = REPO / "docs" / "tw-1h-reclaim-20261017-1030" / "index.html"


def _ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(TPE)
    return t.tz_convert(TPE)


def entry_in_window(ts) -> bool:
    t = _ts(ts)
    d = t.date()
    if d in SKIP_DATES or t.weekday() >= 5:
        return False
    minutes = int(t.hour) * 60 + int(t.minute)
    if not (9 * 60 <= minutes <= 13 * 60 + 30):
        return False
    return WINDOW_START <= t <= WINDOW_END


def extra_stats(trades: Sequence) -> dict:
    pcts = [float(t.pnl_pct) for t in trades]
    if not pcts:
        return {
            "median_pct": None,
            "best_pct": None,
            "worst_pct": None,
            "best_label": "—",
            "worst_label": "—",
        }
    best_i = max(range(len(pcts)), key=lambda i: pcts[i])
    worst_i = min(range(len(pcts)), key=lambda i: pcts[i])
    return {
        "median_pct": float(statistics.median(pcts)),
        "best_pct": pcts[best_i],
        "worst_pct": pcts[worst_i],
        "best_i": best_i,
        "worst_i": worst_i,
    }


def png_data_uri(df, trade, trade_no: int, label: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.png"
        draw_trade_png(df, trade, path, trade_no, title_extra=label)
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


def _reason_zh(reason: str) -> str:
    return {
        "stop": "停損",
        "target": "停利 2R",
        "time": "時間出場（20 根）",
        "open": "未平（資料尚未滿 20 根）",
    }.get(reason, reason)


RULES_HTML = """
<ol class="rules">
<li><b>週期</b>：Yahoo Finance 60 分鐘 K（<code>interval=60m</code>、<code>includePrePost=false</code>）。時間戳為 K 開盤。下載後只留台北時間 09:00–13:30 的根（<code>session_mask</code>），08:30–09:00 盤前試搓／集合競價不納入。</li>
<li><b>標的範圍</b>：上市櫃普通股（四位數字代號，排除 ETF／債／權證）。依指定基準日成交金額排序，先取前 400 再刪收盤價 ≥ 1000，保留成交額前 200。上市用 <code>.TW</code>、上櫃用 <code>.TWO</code>。</li>
<li><b>均線</b>：收盤價 SMA 5 / 10 / 20（程式 <code>sma</code>）。</li>
<li><b>開始一波</b>：第 i 根滿足「前一根收盤 ≥ MA20 且本根收盤 &lt; MA20」。</li>
<li><b>在下面</b>：從跌破那根起算，收盤 ≤ MA20 的根數須 6～36（<code>min_below=6</code>、<code>max_below=36</code>）。中間不准假站上（<code>max_fakes=0</code>：尚未成波前若收盤 &gt; MA20 即失敗）。</li>
<li><b>深度</b>：相對過程中最高 MA20，最低價（Low）深度 ≥ 3.5%（<code>min_depth=0.035</code>）。</li>
<li><b>破底</b>：最低點必須低於其前 16 根 Low 的最低（<code>lookback=16</code>）。更高低的 W 不算。</li>
<li><b>成波</b>：上述成立後，再出現一根收盤 &gt; MA20，這一根為站回（<code>reclaim_idx</code>）。破底與站回不能同一根（<code>separate_trough=True</code>）。</li>
<li><b>進場</b>：破底起 36 根內（<code>entry_window=36</code>），第一根同時收盤 &gt; MA5、MA10、MA20，且進場收盤相對破底低點彈回 ≥ 3%（<code>min_bounce=0.03</code>）。本報告用寬鬆版，<b>不要求</b> MA5&gt;MA10&gt;MA20（<code>require_stack=False</code>）。進場價 = 該根收盤價。</li>
<li><b>停損</b>：破底低點（<code>wave.trough_low</code>）。若進場價不高於停損，改用進場價 × 0.99。</li>
<li><b>停利</b>：進場價 + 2×（進場價 − 停損）（<code>target_r=2.0</code>）。後續每根先看低點是否打到停損，再看高點是否打到目標。</li>
<li><b>時間出場</b>：進場後第 20 根收盤（<code>time_bars=20</code>）。資料在 20 根前結束則標為未平，出場價用最後一根收盤。</li>
<li><b>本視窗</b>：只統計進場時間落在 2026-10-17 09:00 至 2026-10-30 13:30（台北）的訊號。週末與 TWSE 無成交日（2026-10-23、2026-10-26）略過。報酬未扣手續費與稅。</li>
</ol>
"""


def write_embedded_html(
    path: Path,
    hits: List[TwHit],
    universe: List[dict],
    date: str,
    funnel: Dict[str, int],
    extra: dict,
) -> Path:
    trades = [h.trade for h in hits]
    stats = summarize_trades(trades)
    xstats = extra_stats(trades)
    reasons = stats.get("reasons") or {}
    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    cards: List[str] = []
    best_i = xstats.get("best_i")
    worst_i = xstats.get("worst_i")

    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        label = f"{hit.row['code']} {hit.row['name']}"
        uri = png_data_uri(df, t, i, label)
        w = t.signal.wave
        mark = ""
        if best_i is not None and i - 1 == best_i:
            mark = "<span class='tag tag-best'>最好</span>"
        if worst_i is not None and i - 1 == worst_i:
            mark = "<span class='tag tag-worst'>最差</span>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            f"<div class='tags'>{mark}<span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>{escape(_reason_zh(t.exit_reason))}</span>"
            f"<span class='tag'>深度 {w.depth_pct*100:.1f}%</span>"
            f"<span class='tag'>下面 {w.bars_below} 根</span>"
            f"<span class='tag'>彈 {t.entry_price / w.trough_low * 100 - 100:.1f}%</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {t.entry_price:.2f}  停損 {t.stop_price:.2f}（−{risk:.2f}）\n"
            f"目標 {t.target_price:.2f}  出場 {t.exit_price:.2f}  {_reason_zh(t.exit_reason)}  {t.pnl_points:+.2f}\n"
            f"破底 {w.trough_low:.2f} @ {df.index[w.trough_idx].strftime('%m-%d %H:%M')}  "
            f"站回 {df.index[w.reclaim_idx].strftime('%m-%d %H:%M')}\n"
            f"MA5 {t.signal.ma5:.2f} / MA10 {t.signal.ma10:.2f} / MA20 {t.signal.ma20:.2f}\n"
            f"前瞻收盤 +1d {_fmt_pct(t.fwd_1d)}  +3d {_fmt_pct(t.fwd_3d)}  +5d {_fmt_pct(t.fwd_5d)}"
            "</pre>"
            f"<div class='mini-chart'><img src='{uri}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    fun = funnel or {}
    med = _fmt_pct(xstats["median_pct"])
    best = _fmt_pct(xstats["best_pct"])
    worst = _fmt_pct(xstats["worst_pct"])
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · 2026-10-17～10-30</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:640px;margin:0 auto;padding:14px 12px 40px}}
.summary,.rules-box{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 8px}} h2{{font-size:15px;margin:0 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.rules{{margin:0;padding-left:1.2em;font-size:13px;line-height:1.55;color:#c9d1d9}}
.rules li{{margin:0 0 8px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:104px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.tag-best{{color:#00c805;border-color:#00c805}} .tag-worst{{color:#ff5252;border-color:#ff5252}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
code{{font-size:12px;color:#79c0ff}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 1h 破底翻 · 2026-10-17～10-30</h1>
<p class="muted">進場視窗 2026-10-17 09:00 至 2026-10-30 13:30（台北）· 只算 09:00–13:30 正規盤 · 略過 10/23 颱風休市與週末
<br/>宇宙：基準日 {escape(date)} · 成交額前 {len(universe)} · 股價 &lt; 1000 · 末名約 {cutoff:.1f} 億
<br/>Yahoo 1h · 元晶型寬鬆參數（6～36 根、不准假站、深度 ≥ 3.5%、彈 ≥ 3%）
<br/><b>只回測、不下單。報酬未扣手續費與稅。</b> 加總％是各筆相加，不是組合複利。產生時間 {escape(extra.get("generated", ""))}</p>
<div class="cards">
<div class="card">訊號數<b>{stats["count"]}</b></div>
<div class="card">勝率<b>{stats["win_rate"]:.1f}%</b></div>
<div class="card">已平勝率<b>{stats["closed_win_rate"]:.1f}%</b></div>
<div class="card">平均<b class="{"pnl-win" if (stats["avg_pct"] or 0) >= 0 else "pnl-loss"}">{_fmt_pct(stats["avg_pct"])}</b></div>
<div class="card">中位數<b>{med}</b></div>
<div class="card">最好<b class="pnl-win">{best}</b></div>
<div class="card">最差<b class="pnl-loss">{worst}</b></div>
<div class="card">標的<b>{len({h.row["code"] for h in hits})}</b></div>
</div>
<p class="muted">已平 {stats["closed"]} · 未平 {stats["open"]}
· 停利 {reasons.get("target", 0)} · 停損 {reasons.get("stop", 0)} · 時間 {reasons.get("time", 0)} · 未平 {reasons.get("open", 0)}
<br/>漏斗：跌破 {fun.get("cross_below", 0)} → 成波 {fun.get("wave_ok", 0)} → 進場（全歷史）{fun.get("entry", 0)} → 視窗內 {stats["count"]}
· 太短 {fun.get("too_short", 0)} · 不夠深 {fun.get("shallow", 0)} · 非破底 {fun.get("not_16h_low", 0)}
· 假站 {fun.get("too_many_fakes", 0)} · 逾時 {fun.get("timeout", 0)} · 沒站上 {fun.get("entry_timeout", 0)}
· 同根錘 {fun.get("same_bar", 0)} · 彈太弱 {fun.get("weak_bounce", 0)} · 進場價過高 {fun.get("price_cap", 0)}
<br/>收盤前瞻平均 +1d {_fmt_pct(stats.get("fwd_1d"))} · +3d {_fmt_pct(stats.get("fwd_3d"))} · +5d {_fmt_pct(stats.get("fwd_5d"))}</p>
</section>
<section class="rules-box">
<h2>觸發規則（對齊 examples/tw_1h_reclaim.py 實際程式）</h2>
{RULES_HTML}
</section>
{"".join(cards) or "<div class='empty'>這個視窗沒有元晶型 1h 破底翻訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def scan_symbol_window(row: dict, range_: str, params, funnel: Dict[str, int], max_price: Optional[float]):
    meta = {**row, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_yahoo_1h(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 40:
        meta["error"] = "too_few_bars"
        return [], meta
    local_fun: Dict[str, int] = {}
    sigs = detect_signals(df, params, funnel=local_fun)
    if funnel is not None:
        for k, v in local_fun.items():
            funnel[k] = funnel.get(k, 0) + v
    n_all = len(sigs)
    sigs = [s for s in sigs if entry_in_window(df.index[s.entry_idx])]
    if max_price is not None:
        n_hi = sum(1 for s in sigs if s.entry_price >= max_price)
        if n_hi and funnel is not None:
            funnel["price_cap"] = funnel.get("price_cap", 0) + n_hi
        sigs = [s for s in sigs if s.entry_price < max_price]
    trades = simulate(df, sigs, params)
    meta["n_sig"] = n_all
    meta["n_trade"] = len(trades)
    return [TwHit(row, t, df) for t in trades], meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股 1h 破底翻指定視窗回測（只回測）")
    p.add_argument("--date", default="20261030", help="宇宙成交額基準日 YYYYMMDD")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--max-price", type=float, default=1000)
    p.add_argument("--range", dest="range_", default="2mo")
    p.add_argument("--sleep", type=float, default=0.18)
    p.add_argument("--html", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    params = loose_params()
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    pool = 0 if args.limit <= 0 else max(args.limit, args.pool if args.max_price else args.limit)
    print(
        f"window {WINDOW_START} → {WINDOW_END} skip={sorted(SKIP_DATES)} "
        f"universe date={date} limit={args.limit} range={args.range_}",
        flush=True,
    )
    raw = fetch_top_turnover(date, pool)
    price_cap = None if args.max_price is None or args.max_price <= 0 else float(args.max_price)
    helper_cap = None if price_cap is None else price_cap - 1e-9
    universe, dropped = filter_by_max_price(raw, helper_cap, args.limit)
    if dropped:
        print("drop price>=" + str(price_cap) + f" n={len(dropped)}", flush=True)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
        f"{universe[0]['amount']/1e8:.1f}億 / {universe[0]['close']}",
        flush=True,
    )

    hits: List[TwHit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol_window(row, args.range_, params, funnel, price_cap)
        if meta["error"]:
            errors += 1
        hits.extend(stock_hits)
        flag = f" trades={meta['n_trade']}" if meta["n_trade"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row['name']} bars={meta['bars']}{flag}{err}", flush=True)
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    stats = summarize_trades([h.trade for h in hits])
    xstats = extra_stats([h.trade for h in hits])
    print(
        f"done errors={errors} n={stats['count']} WR={stats['win_rate']:.1f}% "
        f"avg={_fmt_pct(stats['avg_pct'])} med={_fmt_pct(xstats['median_pct'])} "
        f"best={_fmt_pct(xstats['best_pct'])} worst={_fmt_pct(xstats['worst_pct'])} funnel={funnel}",
        flush=True,
    )
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        ts = hit.df.index[t.entry_idx]
        print(
            f"  [{i}] {hit.row['code']} {hit.row['name']} {ts.strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_pct*100:+.2f}%",
            flush=True,
        )

    extra = {
        "date": date,
        "window": "2026-10-17..2026-10-30",
        "skip_dates": [d.isoformat() for d in sorted(SKIP_DATES)],
        "strict": False,
        "range": args.range_,
        "limit": args.limit,
        "max_price": price_cap,
        "generated": datetime.now(TPE).isoformat(timespec="seconds"),
        "note": "alerts/backtest only, no orders, no fees",
    }
    out = write_embedded_html(Path(args.html).resolve(), hits, universe, date, funnel, extra)
    print(f"html={out} bytes={out.stat().st_size}", flush=True)
    summary_path = out.with_name("summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "stats": stats,
                "extra_stats": {k: v for k, v in xstats.items() if k not in {"best_i", "worst_i"}},
                "funnel": funnel,
                "extra": extra,
                "n_hits": len(hits),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"json={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
