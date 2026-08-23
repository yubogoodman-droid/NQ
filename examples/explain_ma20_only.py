#!/usr/bin/env python3
"""Why MA20-only reclaim still yields the same 8 NQ trades."""

from __future__ import annotations

import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    REPO_ROOT,
    detect_signals,
    load_bars,
    to_et,
)

REASON_ZH = {
    "taken": "進場",
    "skip_hug_ma20": "hug（貼著走平/下彎的 MA20，整波放棄）",
    "skip_open_hour": "09–10 不進",
    "skip_max_risk": "風險 > 100",
    "skip_ma60": "MA60 特例",
    "skip_vol": "量能太大",
    "skip_ma20_slope": "MA20 斜率太負",
    "skip_ma200_hug": "貼著下方 MA200",
    "skip_wide_risk": "寬停損但非 QA",
    "skip_entry_gap": "距上一筆不到 15 根",
    "skip_bad_risk": "風險無效",
}


def main() -> int:
    df = to_et(load_bars("NQ=F", "1m", "30d"))
    strict: list = []
    loose: list = []
    detect_signals(df, trace=strict)
    detect_signals(df, require_ma30=False, trace=loose)

    strict_keys = {(r["idx"], r["reason"]) for r in strict}
    extra = [r for r in loose if (r["idx"], r["reason"]) not in strict_keys]
    extra_only_under_ma30 = [r for r in extra if not r["above_ma30"]]
    counts = Counter(r["reason"] for r in extra_only_under_ma30)
    taken_strict = {r["idx"] for r in strict if r["reason"] == "taken"}
    taken_loose = {r["idx"] for r in loose if r["reason"] == "taken"}

    rows = []
    for r in extra_only_under_ma30:
        ts = df.index[r["idx"]]
        gap20 = r["close"] - r["ma20"]
        gap30 = r["close"] - r["ma30"]
        rows.append(
            "<tr>"
            f"<td>{escape(ts.strftime('%m-%d %H:%M'))}</td>"
            f"<td>{escape(REASON_ZH.get(r['reason'], r['reason']))}</td>"
            f"<td>{r['close']:.2f}</td>"
            f"<td>{gap20:+.1f}</td>"
            f"<td>{gap30:+.1f}</td>"
            "</tr>"
        )
    count_line = " · ".join(
        f"{REASON_ZH.get(k, k)} {v}" for k, v in counts.most_common()
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>為什麼只要收復 MA20 還是 8 筆</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:16px 14px 36px}}
h1{{font-size:18px;margin:0 0 8px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin:12px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 6px;border-bottom:1px solid #21262d}}
th{{color:#8b949e}}
</style></head><body>
<div class="page">
<h1>為什麼只要收復 MA20，這個月還是 8 筆</h1>
<p class="muted">嚴格：收復次數 {len(strict)} · 進場 {len(taken_strict)} 根<br/>
只要 MA20：收復次數 {len(loose)} · 進場 {len(taken_loose)} 根（同一 {len(taken_strict & taken_loose)} 根）</p>
<div class="box">
<p class="muted">已經 MA5&gt;MA10&gt;MA20 且站上 MA20 時，收盤通常也已經站上 MA30。
真正「站上 MA20、還沒站上 MA30」的多出來收復有 <b>{len(extra_only_under_ma30)}</b> 次，全部被後面的過濾擋掉，沒有任何一筆變成進場：</p>
<p class="muted">{escape(count_line or "無")}</p>
</div>
<div class="box">
<table><thead><tr><th>時間 ET</th><th>為什麼沒進</th><th>收盤</th><th>距 MA20</th><th>距 MA30</th></tr></thead>
<tbody>{''.join(rows) or "<tr><td colspan='5'>沒有這種收復</td></tr>"}</tbody></table>
</div>
</div></body></html>
"""
    out = REPO_ROOT / "docs" / "nq-ma-reclaim-ma20" / "why.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    view = out.with_name("why-view.html")
    view.write_text(html, encoding="utf-8")
    print(f"strict_hits={len(strict)} loose_hits={len(loose)}")
    print(f"taken_same={taken_strict == taken_loose} n={len(taken_strict)}")
    print(f"extra_under_ma30={len(extra_only_under_ma30)} {dict(counts)}")
    print(f"html={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
