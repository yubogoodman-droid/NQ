#!/usr/bin/env python3
"""1m vs 5m MA Reclaim：同一個 30d 窗口對打，另跑 Yahoo 原生 5m 60d。

不改 1m 預設。五分有兩種譯法：
  naive     根數照搬（2h=120 根=10 小時，收復 15 根=75 分鐘）
  clock     時間對齊（2h=24 根，收復 15 分鐘=3 根，持有 60 分鐘=12 根）
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    PAGES_HTML,
    REPO_ROOT,
    detect_signals,
    load_bars,
    load_yfinance,
    simulate,
    summarize_trades,
    to_et,
    write_html_report,
    write_view_html,
)

PAGES = REPO_ROOT / "docs" / "nq-ma-reclaim" / "m5.html"


def resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample("5min", label="right", closed="right").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def funnel_line(fun: Dict[str, int]) -> str:
    return (
        f"break={fun.get('break', 0)} deep={fun.get('deep_break', 0)} "
        f"reclaim={fun.get('reclaim_stack', 0)} taken={fun.get('taken', 0)} "
        f"hug={fun.get('skip_hug_ma20', 0)} ma60={fun.get('skip_ma60', 0)} "
        f"hour={fun.get('skip_open_hour', 0)} vol={fun.get('skip_vol', 0)} "
        f"risk={fun.get('skip_max_risk', 0)} wide={fun.get('skip_wide_risk', 0)}"
    )


def run_one(
    df: pd.DataFrame,
    name: str,
    detect_kw: Dict[str, Any],
    sim_kw: Dict[str, Any],
) -> dict:
    fun: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=fun, **detect_kw)
    trades = simulate(df, sigs, **sim_kw)
    stats = summarize_trades(trades)
    rows = []
    for i, t in enumerate(trades, 1):
        rows.append(
            {
                "n": i,
                "q": t.quality,
                "entry": df.index[t.entry_idx].strftime("%m-%d %H:%M"),
                "exit": df.index[t.exit_idx].strftime("%m-%d %H:%M"),
                "reason": t.exit_reason,
                "pnl": round(float(t.pnl_points), 1),
                "risk": round(float(t.entry_price - t.stop_price), 1),
            }
        )
    return {
        "name": name,
        "bars": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "n": stats["count"],
        "wins": stats["wins"],
        "wr": round(stats["win_rate"], 1),
        "pnl": round(stats["total_points"], 1),
        "by_q": stats["by_quality"],
        "funnel": dict(fun),
        "funnel_line": funnel_line(fun),
        "trades": rows,
        "_trades": trades,
        "_df": df,
        "_detect": detect_kw,
        "_sim": sim_kw,
    }


CLOCK_DETECT = dict(
    two_hour_bars=24,
    reclaim_window=3,
    min_entry_gap=3,
    ma20_slope_bars=1,
    ma60_slope_bars=1,
    hug_ma20_max_slope=0.5,
)
CLOCK_SIM = dict(max_hold=12, hard_cap_bars=78)


def variants_30d(m1: pd.DataFrame, m5: pd.DataFrame) -> List[dict]:
    out = [
        run_one(m1, "1m 預設（對照）", {}, {}),
        run_one(
            m5,
            "5m 根數照搬（2h=10h、收復 75 分、持有 5h）",
            {},
            {},
        ),
        run_one(
            m5,
            "5m 時間對齊（2h=24、收復 15 分=3 根、持有 60 分）",
            dict(CLOCK_DETECT),
            dict(CLOCK_SIM),
        ),
        run_one(
            m5,
            "5m 時間對齊 · 收復 30 分（6 根）",
            dict(CLOCK_DETECT, reclaim_window=6),
            dict(CLOCK_SIM),
        ),
        run_one(
            m5,
            "5m 時間對齊 · 收復 75 分（15 根）",
            dict(CLOCK_DETECT, reclaim_window=15),
            dict(CLOCK_SIM),
        ),
        run_one(
            m5,
            "5m 時間對齊 MA（MA4/6≈1m MA20/30）",
            dict(
                CLOCK_DETECT,
                ma_lens=(1, 2, 4, 6, 12, 40),
            ),
            dict(CLOCK_SIM, ma_exit_period=4),
        ),
        run_one(
            m5,
            "5m 根數照搬 · 關 hug",
            dict(hug_ma20_pts=0.0),
            {},
        ),
        run_one(
            m5,
            "5m 時間對齊 · 關 hug",
            dict(CLOCK_DETECT, hug_ma20_pts=0.0),
            dict(CLOCK_SIM),
        ),
    ]
    return out


def print_block(row: dict) -> None:
    print(f"\n== {row['name']} ==")
    print(f"bars={row['bars']} {row['start']} -> {row['end']}")
    print(f"trades={row['n']} WR={row['wr']:.1f}% pnl={row['pnl']:+.1f}")
    print("funnel", row["funnel_line"])
    for q, info in row["by_q"].items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for t in row["trades"]:
        print(
            f"  [{t['n']}] Q{t['q']} {t['entry']} -> {t['exit']} "
            f"{t['reason']} {t['pnl']:+.1f} risk={t['risk']:.0f}"
        )


HTML_SKIP = {
    "5m 根數照搬 · 關 hug",
    "5m 時間對齊 · 關 hug",
}


def write_compare_html(path: Path, window: str, rows: Sequence[dict]) -> Path:
    summary_rows = []
    for r in rows:
        if r["name"] in HTML_SKIP:
            continue
        cls = "pnl-win" if r["pnl"] > 0 else ("pnl-flat" if r["pnl"] == 0 else "pnl-loss")
        summary_rows.append(
            f"<tr><td>{escape(r['name'])}</td><td>{r['n']}</td>"
            f"<td>{r['wr']:.1f}%</td><td class='{cls}'>{r['pnl']:+.1f}</td></tr>"
        )
    cards = []
    for r in rows:
        if r["name"] in HTML_SKIP:
            continue
        cls = "pnl-win" if r["pnl"] >= 0 else "pnl-loss"
        trade_bits = []
        for t in r["trades"]:
            tcls = "pnl-win" if t["pnl"] > 0 else "pnl-loss"
            trade_bits.append(
                f"<tr><td>{t['n']}</td><td>Q{escape(str(t['q']))}</td>"
                f"<td>{escape(t['entry'])}</td><td>{escape(t['exit'])}</td>"
                f"<td>{escape(t['reason'])}</td>"
                f"<td class='{tcls}'>{t['pnl']:+.1f}</td></tr>"
            )
        body = (
            "<table><thead><tr><th>#</th><th>Q</th><th>進</th><th>出</th><th>原因</th><th>點</th></tr></thead>"
            f"<tbody>{''.join(trade_bits)}</tbody></table>"
            if trade_bits
            else "<p class='muted'>無交易</p>"
        )
        cards.append(
            "<section class='summary'>"
            f"<h1>{escape(r['name'])}</h1>"
            f"<p class='muted'>{escape(r['funnel_line'])}</p>"
            "<div class='cards'>"
            f"<div class='card'>筆數<b>{r['n']}</b></div>"
            f"<div class='card'>勝率<b>{r['wr']:.1f}%</b></div>"
            f"<div class='card'>總點數<b class='{cls}'>{r['pnl']:+.1f}</b></div>"
            f"<div class='card'>勝/負<b>{r['wins']}/{r['n']-r['wins']}</b></div>"
            "</div>"
            f"{body}"
            "</section>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 1m vs 5m MA Reclaim</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:16px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}}
th,td{{text-align:left;padding:4px 6px;border-bottom:1px solid #21262d;vertical-align:top}}
th{{color:#8b949e;font-weight:600}}
.verdict{{font-size:18px;font-weight:700;margin:0 0 8px}}
</style></head><body>
<div class="page">
<section class="summary">
<p class="verdict">沒料，預設仍用 1m。</p>
<p class="muted">{escape(window)}</p>
<p class="muted">同一套破底翻丟到五分 K：根數照搬會把 2 小時變成 10 小時、15 根收復變成 75 分鐘；時間對齊後 15 分鐘內排不出 5m 的 MA5&gt;MA10&gt;MA20（那是 25/50/100 分鐘均線），30 天 0 筆。</p>
<table>
<thead><tr><th>版本</th><th>筆</th><th>勝率</th><th>點</th></tr></thead>
<tbody>
{''.join(summary_rows)}
</tbody>
</table>
</section>
{''.join(cards)}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="30d")
    p.add_argument("--m5-period", default="60d")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--skip-native60", action="store_true")
    args = p.parse_args()

    print("[data] 1m", args.period, file=sys.stderr)
    m1 = to_et(load_bars("NQ=F", "1m", args.period))
    if m1.empty:
        print("no 1m data", file=sys.stderr)
        return 1
    print(f"[data] 1m bars={len(m1)} {m1.index[0]} -> {m1.index[-1]}", file=sys.stderr)
    m5 = resample_m5(m1)
    print(f"[data] 5m resample bars={len(m5)} {m5.index[0]} -> {m5.index[-1]}", file=sys.stderr)

    rows = variants_30d(m1, m5)
    for row in rows:
        print_block(row)

    native_rows: List[dict] = []
    if not args.skip_native60:
        print("[data] native 5m", args.m5_period, file=sys.stderr)
        raw5 = to_et(load_yfinance("NQ=F", "5m", args.m5_period))
        if raw5.empty:
            print("[data] native 5m empty, skip", file=sys.stderr)
        else:
            print(f"[data] native 5m bars={len(raw5)} {raw5.index[0]} -> {raw5.index[-1]}", file=sys.stderr)
            native_rows = [
                run_one(raw5, f"Yahoo 5m {args.m5_period} · 根數照搬", {}, {}),
                run_one(
                    raw5,
                    f"Yahoo 5m {args.m5_period} · 時間對齊 15 分收復",
                    dict(CLOCK_DETECT),
                    dict(CLOCK_SIM),
                ),
                run_one(
                    raw5,
                    f"Yahoo 5m {args.m5_period} · 時間對齊 75 分收復",
                    dict(CLOCK_DETECT, reclaim_window=15),
                    dict(CLOCK_SIM),
                ),
            ]
            for row in native_rows:
                print_block(row)

    all_rows = list(rows) + native_rows
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows]
    print("\nJSON")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.pages:
        window = f"1m {m1.index[0].strftime('%m-%d %H:%M')} → {m1.index[-1].strftime('%m-%d %H:%M')} ET"
        write_compare_html(PAGES, window, all_rows)
        write_view_html(PAGES, dest_name="m5-view.html")
        chart_src = rows[1]
        if chart_src["n"]:
            gal = PAGES.with_name("m5-trades.html")
            write_html_report(
                gal,
                chart_src["_df"],
                chart_src["_trades"],
                "NQ=F",
                chart_src["name"],
                funnel=chart_src["funnel"],
                note="五分K根數照搬實驗，不是 live 預設。1m 圖廊沒改。",
                timeframe="5m",
                prefix="m5",
            )
            write_view_html(gal, dest_name="m5-trades-view.html")
            print(f"html={PAGES}")
            print(f"trades_html={gal}")
        else:
            print(f"html={PAGES}")
        write_view_html(PAGES_HTML)
        print(f"view={PAGES_HTML.with_name('view.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
