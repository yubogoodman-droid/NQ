#!/usr/bin/env python3
"""Why MA20-only reclaim still yields the same 8 NQ trades — with charts."""

from __future__ import annotations

import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    REPO_ROOT,
    detect_signals,
    draw_event_png,
    load_bars,
    to_et,
)

REASON_ZH = {
    "taken": "進場",
    "skip_hug_ma20": "hug（貼著走平/下彎的 MA20）",
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

PAGES = REPO_ROOT / "docs" / "nq-ma-reclaim-ma20" / "why.html"
BRANCH = "cursor/nq-30d-ablation-2484"


def extra_under_ma30(df) -> tuple[list, dict]:
    strict: list = []
    loose: list = []
    detect_signals(df, trace=strict)
    detect_signals(df, require_ma30=False, trace=loose)
    strict_keys = {(r["idx"], r["reason"]) for r in strict}
    extra = [
        r
        for r in loose
        if (r["idx"], r["reason"]) not in strict_keys and not r["above_ma30"]
    ]
    meta = {
        "strict_n": len(strict),
        "loose_n": len(loose),
        "taken_strict": sum(1 for r in strict if r["reason"] == "taken"),
        "taken_loose": sum(1 for r in loose if r["reason"] == "taken"),
        "counts": Counter(r["reason"] for r in extra),
    }
    return extra, meta


def write_why_gallery(df, extra: list, meta: dict, path: Path) -> Path:
    img_dir = path.parent / "img"
    cards = []
    for i, r in enumerate(extra, 1):
        ts = df.index[r["idx"]]
        gap20 = r["close"] - r["ma20"]
        gap30 = r["close"] - r["ma30"]
        reason = REASON_ZH.get(r["reason"], r["reason"])
        img_name = f"w{i:02d}_{ts.strftime('%m%d_%H%M')}_{r['reason'].replace('skip_', '')}.png"
        title = f"#{i}  {ts.strftime('%m-%d %H:%M')}  {reason}  距MA20 {gap20:+.1f}  距MA30 {gap30:+.1f}"
        draw_event_png(df, r["idx"], img_dir / img_name, title, break_idx=r.get("break_idx"))
        cards.append(
            "<article class='trade-card'>"
            f"<header class='card-header'><div class='card-title'><span class='trade-no'>#{i}</span>"
            f"<span class='trade-time'>{escape(ts.strftime('%Y-%m-%d %H:%M'))} ET</span></div>"
            f"<div class='card-pnl'>{escape(reason)}</div></header>"
            f"<pre class='trade-detail'>close {r['close']:.2f}  MA20 {r['ma20']:.2f} ({gap20:+.1f})"
            f"  MA30 {r['ma30']:.2f} ({gap30:+.1f})</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    count_line = " · ".join(
        f"{REASON_ZH.get(k, k)} {v}" for k, v in meta["counts"].most_common()
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>站上 MA20 還沒站上 MA30 的 {len(extra)} 次</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.box,.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:13px;color:#f0c14b}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px}}
</style></head><body>
<div class="page">
<h1>站上 MA20、還沒站上 MA30</h1>
<p class="muted">嚴格收復 {meta['strict_n']} → 只要 MA20 {meta['loose_n']}。進場仍是 {meta['taken_strict']} 筆。
下面 {len(extra)} 次都沒進：{escape(count_line)}。粉點是破底，黃線是這根收復。</p>
{''.join(cards)}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view(src: Path) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{BRANCH}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("why-view.html")
    out.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    df = to_et(load_bars("NQ=F", "1m", "30d"))
    extra, meta = extra_under_ma30(df)
    out = write_why_gallery(df, extra, meta, PAGES)
    view = write_view(out)
    print(f"extra={len(extra)} {dict(meta['counts'])}")
    print(f"html={out}")
    print(f"view={view}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
