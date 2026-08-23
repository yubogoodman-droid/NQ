#!/usr/bin/env python3
"""NQ 破底翻：同一個 30d 資料，拆過濾器 / 時段 / 品質。不改進場規則。"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    CORE_DETECT,
    REPO_ROOT,
    detect_signals,
    load_bars,
    simulate,
    summarize_trades,
    to_et,
)

PAGES = REPO_ROOT / "docs" / "nq-ma-reclaim" / "research.html"

VARIANTS: List[Tuple[str, dict]] = [
    ("嚴格（預設）", {}),
    ("核心（關 hug / MA60 / 寬停損QA）", dict(CORE_DETECT)),
    ("只關 hug", dict(hug_ma20_pts=0.0)),
    ("只關 MA60 特例", dict(use_ma60_skip=False)),
    ("只關 09–10 檔", dict(skip_hour_start=None, skip_hour_end=None)),
    ("只關 100 點風險上限", dict(max_risk=0.0)),
    ("只關 寬停損 QA 門檻", dict(max_risk_non_qa=0.0)),
]


def session_bucket(hour: int) -> str:
    if hour >= 18 or hour < 3:
        return "亞盤 18–03"
    if hour < 9:
        return "倫敦 03–09"
    if hour < 10:
        return "美開 09–10"
    if hour < 16:
        return "美股 10–16"
    return "尾盤 16–18"


def _stat_line(trades: Sequence[Any]) -> dict:
    s = summarize_trades(trades)
    return {
        "n": s["count"],
        "wins": s["wins"],
        "wr": s["win_rate"],
        "pnl": s["total_points"],
        "by_q": s["by_quality"],
    }


def extra_vs_base(base: Sequence[Any], other: Sequence[Any]) -> List[Any]:
    keys = {int(t.entry_idx) for t in base}
    return [t for t in other if int(t.entry_idx) not in keys]


def run_research(df) -> dict:
    funnel: Dict[str, int] = {}
    strict_sigs = detect_signals(df, funnel=funnel)
    strict = simulate(df, strict_sigs)
    variants: List[dict] = []
    cached: Dict[str, List[Any]] = {"嚴格（預設）": strict}
    for name, kwargs in VARIANTS:
        if name == "嚴格（預設）":
            trades = strict
        else:
            trades = simulate(df, detect_signals(df, **kwargs))
            cached[name] = trades
        row = _stat_line(trades)
        row["name"] = name
        variants.append(row)

    core = cached["核心（關 hug / MA60 / 寬停損QA）"]
    extras = extra_vs_base(strict, core)
    qa_only = simulate(df, [s for s in strict_sigs if s.quality == "A"])

    sessions = []
    for label in ("亞盤 18–03", "倫敦 03–09", "美開 09–10", "美股 10–16", "尾盤 16–18"):
        for mode, trades in (("嚴格", strict), ("核心", core)):
            bucket = [t for t in trades if session_bucket(int(df.index[t.entry_idx].hour)) == label]
            row = _stat_line(bucket)
            row["session"] = label
            row["mode"] = mode
            sessions.append(row)

    extra_rows = []
    for t in extras:
        ts = df.index[t.entry_idx]
        extra_rows.append(
            {
                "when": ts.strftime("%m-%d %H:%M"),
                "session": session_bucket(int(ts.hour)),
                "q": t.quality,
                "reason": t.exit_reason,
                "pnl": float(t.pnl_points),
            }
        )

    return {
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "bars": int(len(df)),
        "funnel": funnel,
        "variants": variants,
        "sessions": sessions,
        "qa_only": _stat_line(qa_only),
        "extras": extra_rows,
        "extras_stat": _stat_line(extras),
        "strict": _stat_line(strict),
        "core": _stat_line(core),
    }


def _fmt_pnl(v: float) -> str:
    cls = "win" if v > 0 else ("loss" if v < 0 else "flat")
    return f"<b class='{cls}'>{v:+.1f}</b>"


def write_research_html(path: Path, report: dict) -> Path:
    funnel = report["funnel"]
    v_rows = "".join(
        "<tr>"
        f"<td>{escape(r['name'])}</td><td>{r['n']}</td>"
        f"<td>{r['wr']:.1f}%</td><td>{_fmt_pnl(r['pnl'])}</td>"
        "</tr>"
        for r in report["variants"]
    )
    qa = report["qa_only"]
    ex = report["extras_stat"]
    s_rows = "".join(
        "<tr>"
        f"<td>{escape(r['session'])}</td><td>{escape(r['mode'])}</td>"
        f"<td>{r['n']}</td><td>{r['wr']:.1f}%</td><td>{_fmt_pnl(r['pnl'])}</td>"
        "</tr>"
        for r in report["sessions"]
    )
    extra_rows = "".join(
        "<tr>"
        f"<td>{escape(r['when'])}</td><td>{escape(r['session'])}</td>"
        f"<td>Q{escape(r['q'])}</td><td>{escape(r['reason'])}</td>"
        f"<td>{_fmt_pnl(r['pnl'])}</td>"
        "</tr>"
        for r in report["extras"]
    ) or "<tr><td colspan='5'>沒有多出來的核心單</td></tr>"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NQ 破底翻 · 30d 拆解</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:16px 14px 36px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:22px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin:12px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 6px;border-bottom:1px solid #21262d}}
th{{color:#8b949e;font-weight:600}}
.win{{color:#3ddc68}} .loss{{color:#ff7b72}} .flat{{color:#8b949e}}
</style></head><body>
<div class="page">
<h1>NQ 破底翻 · 30 天拆解</h1>
<p class="muted">{escape(report['start'])} → {escape(report['end'])} · bars={report['bars']}
<br/>同一套 Yahoo 1m，只開關既有過濾器，不改核心：破 2h 低、15 根收復 MA20/30、5/10/20 多頭。</p>
<div class="box">
<h2>漏斗（嚴格）</h2>
<p class="muted">破底 {funnel.get('break', 0)} → 深度夠 {funnel.get('deep_break', 0)} →
收復+排列 {funnel.get('reclaim_stack', 0)} → 進場 {funnel.get('taken', 0)}
<br/>擋掉：hug {funnel.get('skip_hug_ma20', 0)} · MA60 {funnel.get('skip_ma60', 0)} ·
09–10 {funnel.get('skip_open_hour', 0)} · 量能 {funnel.get('skip_vol', 0)} ·
風險 {funnel.get('skip_max_risk', 0)} · 寬停損 {funnel.get('skip_wide_risk', 0)}</p>
<p>嚴格 {_fmt_pnl(report['strict']['pnl'])} / {report['strict']['n']} 筆 ·
核心 {_fmt_pnl(report['core']['pnl'])} / {report['core']['n']} 筆 ·
多出來的核心單 {_fmt_pnl(ex['pnl'])} / {ex['n']} 筆 ·
嚴格只做 QA {_fmt_pnl(qa['pnl'])} / {qa['n']} 筆（勝率 {qa['wr']:.1f}%）</p>
</div>
<div class="box">
<h2>一次只關一個過濾器</h2>
<table><thead><tr><th>設定</th><th>筆</th><th>勝率</th><th>點數</th></tr></thead>
<tbody>{v_rows}</tbody></table>
</div>
<div class="box">
<h2>時段（進場小時，美東）</h2>
<table><thead><tr><th>時段</th><th>模式</th><th>筆</th><th>勝率</th><th>點數</th></tr></thead>
<tbody>{s_rows}</tbody></table>
</div>
<div class="box">
<h2>核心比嚴格多出來的單</h2>
<table><thead><tr><th>進場</th><th>時段</th><th>Q</th><th>出場</th><th>點數</th></tr></thead>
<tbody>{extra_rows}</tbody></table>
</div>
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = (
        "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
        f"cursor/nq-1m-ma-reclaim-2484/{rel}/"
    )
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("research-view.html")
    out.write_text(text, encoding="utf-8")
    return out


def print_report(report: dict) -> None:
    print(f"bars={report['bars']} {report['start']} -> {report['end']}")
    print("variants:")
    for r in report["variants"]:
        print(f"  {r['name']}: n={r['n']} WR={r['wr']:.1f}% pnl={r['pnl']:+.1f}")
    qa = report["qa_only"]
    print(f"strict QA-only: n={qa['n']} WR={qa['wr']:.1f}% pnl={qa['pnl']:+.1f}")
    print("sessions:")
    for r in report["sessions"]:
        if r["n"] == 0:
            continue
        print(
            f"  {r['session']} {r['mode']}: n={r['n']} WR={r['wr']:.1f}% pnl={r['pnl']:+.1f}"
        )
    ex = report["extras_stat"]
    print(f"core extras: n={ex['n']} WR={ex['wr']:.1f}% pnl={ex['pnl']:+.1f}")
    for r in report["extras"]:
        print(f"  {r['when']} {r['session']} Q{r['q']} {r['reason']} {r['pnl']:+.1f}")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NQ 破底翻 30d 拆解")
    p.add_argument("--period", default="30d")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    args = p.parse_args(list(argv) if argv is not None else None)

    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    report = run_research(df)
    print_report(report)
    html_path = Path(args.html) if args.html else (PAGES if args.pages else None)
    if html_path:
        out = write_research_html(html_path, report)
        write_view_html(out)
        print(f"html={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
