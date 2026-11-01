#!/usr/bin/env python3
"""半年回測：2025-11～2026-04 挑規則，2026-05～10-30 才打開。只回測、不下單。"""

from __future__ import annotations

import base64
import json
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_tw_1h_reclaim import CLOSED_DAYS, build_rows, daily_regime, load_cache, metrics  # noqa: E402
from tw_1h_reclaim import detect_signals, draw_trade_png, loose_params, simulate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "tw-1h-reclaim-halfyear-202605-1030"
SEL = (date(2025, 11, 1), date(2026, 4, 30))
EV = (date(2026, 5, 1), date(2026, 10, 30))
REASON = {"stop": "停損", "target": "停利", "time": "時間出場", "open": "未平"}


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def cls(x: float) -> str:
    return "pnl-win" if x > 0 else ("pnl-loss" if x < 0 else "")


def window(rows, bounds):
    lo, hi = bounds
    return [r for r in rows if lo <= r.entry_ts.date() <= hi]


def chosen(row) -> bool:
    return row.mkt_above is True and row.depth >= 0.10


def mrow(title: str, rows) -> str:
    m = metrics(rows)
    if not m["n"]:
        return f"<tr><td>{escape(title)}</td><td colspan='6'>0</td></tr>"
    return (
        f"<tr><td>{escape(title)}</td><td>{m['n']}</td><td>{m['win']:.1f}%</td>"
        f"<td class='{cls(m['avg'])}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td>"
        f"<td>{pct(m['best'])} / {pct(m['worst'])}</td><td>{m['pf']:.2f}</td></tr>"
    )


def draw(path: Path, base, picked) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    months = [date(2026, m, 1) for m in range(5, 11)]

    def win_rate(rows, month: int) -> float:
        sel = [r for r in rows if r.entry_ts.month == month]
        if not sel:
            return float("nan")
        return 100.0 * sum(1 for r in sel if r.pnl_pct > 0) / len(sel)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.2, 7.0), gridspec_kw={"height_ratios": [1.1, 1]}, facecolor="#0c1210"
    )
    for a in (ax, ax2):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    x = list(range(6))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [win_rate(base, m) for m in range(5, 11)], width=w, color="#8aa193", label="原規則 2R")
    ax.bar(
        [i + w / 2 for i in x],
        [win_rate(picked, m) for m in range(5, 11)],
        width=w,
        color="#f0c14b",
        label="優化：1R · 1h>MA20 · 深度≥10%",
    )
    ax.axhline(50, color="#2a3a33", lw=0.8)
    ax.axvspan(4.5, 5.5, color="#ff525222", lw=0)
    ax.set_ylabel("勝率 %", color="#8aa193")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.month}月" for m in months], color="#8aa193")
    ax.set_title("2026-05～10 月勝率 · 紅底 10 月優化版沒有比較高", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#c8d5cc")

    def curve(rows):
        total, xs, ys = 0.0, [], []
        for r in sorted(rows, key=lambda z: z.entry_ts):
            total += r.pnl_pct * 100
            xs.append(r.entry_ts)
            ys.append(total)
        return xs, ys

    for rows, color, name in (
        (base, "#8aa193", f"原規則（{len(base)} 筆）"),
        (picked, "#f0c14b", f"優化版（{len(picked)} 筆）"),
    ):
        xs, ys = curve(rows)
        ax2.plot(xs, ys, color=color, lw=1.6, label=name)
    ax2.axvspan(pd.Timestamp("2026-10-01", tz="Asia/Taipei"), pd.Timestamp("2026-10-30 23:59", tz="Asia/Taipei"), color="#ff525233", lw=0)
    ax2.axhline(0, color="#2a3a33", lw=0.8)
    ax2.set_ylabel("各筆 % 相加", color="#8aa193")
    ax2.set_title("半年累積（各筆相加，未扣費稅）· 黃線總和較低是少做；單筆平均較高，9–10 月沒有", color="#e8f0ea", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    fig.tight_layout(pad=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def trade_cards(cache: dict, rows, img_dir: Path) -> str:
    import tw_1h_reclaim as tw

    setup, ready = tw._setup_cjk, {"ok": False}

    def once() -> None:
        if not ready["ok"]:
            setup()
            ready["ok"] = True

    tw._setup_cjk = once  # type: ignore[method-assign]
    wanted: dict = {}
    for row in rows:
        wanted.setdefault(row.symbol, set()).add(row.entry_ts)
    params = loose_params(target_r=1.0, time_bars=20)
    found = {}
    for sym, stamps in wanted.items():
        df = cache["data"][sym]
        df = df.loc[[ts.date() not in CLOSED_DAYS for ts in df.index]]
        for trade in simulate(df, detect_signals(df, params), params):
            ts = df.index[trade.entry_idx]
            if ts in stamps:
                found[(sym, ts)] = (df, trade)
    missing = [r for r in rows if (r.symbol, r.entry_ts) not in found]
    if missing:
        raise SystemExit(f"missing {missing[0].code} {missing[0].entry_ts} ({len(missing)})")
    cards = [
        "<section><h2>優化版這半年的每一筆 K 線</h2>"
        f"<p class='muted'>{len(rows)} 筆都是 1R、當時加權 1h 收盤站上 1h MA20、破底深度至少 10%。"
        "綠三角進場、黃叉出場、黃／藍／粉是 MA5／10／20、粉點破底、紅虛線停損、綠虛線 1R。"
        "原規則另外一千多筆沒有逐張畫。</p></section>"
    ]
    for i, row in enumerate(rows, 1):
        df, trade = found[(row.symbol, row.entry_ts)]
        name = f"t{i:03d}_{row.code}_{row.entry_ts.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, trade, img_dir / name, i, title_extra=f"{row.code} {row.name}")
        reason = REASON.get(trade.exit_reason, trade.exit_reason)
        klass = "pnl-win" if row.pnl_pct > 0 else "pnl-loss"
        cards.append(
            "<article class='trade-card'><header class='card-header'>"
            f"<span>#{i} · {escape(row.code)} {escape(row.name)}</span>"
            f"<span class='{klass}'>{row.pnl_pct * 100:+.2f}%</span></header>"
            f"<p class='muted'>{row.entry_ts.strftime('%m-%d %H:%M')} 進 {row.entry:.2f} → "
            f"{row.exit_ts.strftime('%m-%d %H:%M')} {escape(reason)} {row.exit_px:.2f} · 深度 {row.depth * 100:.1f}%</p>"
            f"<img src='img/{name}' alt='{escape(row.code)} {escape(row.name)}'/></article>"
        )
        if i % 50 == 0:
            print(f"charts {i}/{len(rows)}", flush=True)
    return "".join(cards)


def main() -> int:
    cache = load_cache()
    base_all = build_rows(cache, loose_params())
    one_r = build_rows(cache, loose_params(target_r=1.0, time_bars=20))
    base = window(base_all, EV)
    picked = [r for r in window(one_r, EV) if chosen(r)]
    same_2r = [r for r in base if chosen(r)]
    long = [r for r in base if r.bars_below >= 21]
    sel_base = window(base_all, SEL)
    sel_pick = [r for r in window(one_r, SEL) if chosen(r)]
    oct_base = [r for r in base if r.entry_ts.month == 10]
    oct_pick = [r for r in picked if r.entry_ts.month == 10]

    out = OUT
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    draw(img / "halfyear.png", base, picked)
    uri = "data:image/png;base64," + base64.b64encode((img / "halfyear.png").read_bytes()).decode("ascii")
    cards = trade_cards(cache, picked, img)

    def block(title, pairs):
        body = "".join(mrow(name, rows) for name, rows in pairs)
        return (
            f"<h3>{title}</h3><table><tr><th></th><th>筆數</th><th>勝率</th><th>平均</th>"
            f"<th>中位數</th><th>最好 / 最差</th><th>獲利因子</th></tr>{body}</table>"
        )

    summary = {
        "select": {"base": metrics(sel_base), "chosen": metrics(sel_pick)},
        "half": {"base": metrics(base), "chosen": metrics(picked), "filter_2r": metrics(same_2r), "below21": metrics(long)},
        "october": {"base": metrics(oct_base), "chosen": metrics(oct_pick)},
    }
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · 半年回測</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:820px;margin:0 auto;padding:14px 12px 48px}}
section{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:19px;margin:0 0 8px}} h2{{font-size:16px;margin:0 0 8px}} h3{{font-size:14px;margin:12px 0 6px;color:#c9d1d9}}
.muted{{color:#8b949e;font-size:13px;line-height:1.65}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{border-bottom:1px solid #21262d;padding:6px 4px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:#8b949e;font-weight:500}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
img{{width:100%;border-radius:10px;display:block}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:12px;margin-bottom:12px}}
.card-header{{display:flex;justify-content:space-between;gap:8px;font-weight:700}}
.verdict{{border-left:4px solid #f0c14b;padding-left:10px}}
</style></head><body><div class="page">
<section>
<h1>台股 1h 破底翻 · 半年回測（2026-05-01～10-30）</h1>
<p class="muted">只回測、不下單。報酬未扣手續費與稅。宇宙是 2026-10-30 成交額前 200、股價 &lt; 1000，同一批股票回推，有倖存者偏差。偵測仍是元晶型寬鬆規則。優化只改兩件事：停利從 2R 改成 1R，而且進場當下加權 1h 收盤要站上 1h MA20、破底深度至少 10%。</p>
</section>
<section class="verdict">
<h2>結論</h2>
<p class="muted">規則是用前一段（2025-11-01～2026-04-30）的勝率挑的，這半年沒有拿來選。挑完套過來：1817 筆、勝率 51.7%、平均 +1.58%，變成 557 筆、勝率 64.5%、平均 +2.70%。5～8 月勝率比較高；5 月平均從 +4.57% 降到 +3.81%，因為停利改成 1R，大贏被截斷。9 月勝率 52.6% → 46.7%，10 月 27.8% / −2.30% → 25.6% / −3.60%。半年總和被 6～8 月撐住。監看預設不改。</p>
</section>
<section>
<h2>這半年</h2>
<img src='{uri}' alt='半年月勝率與累積報酬'/>
</section>
<section>
<h2>數字</h2>
{block("用來挑規則的前半年 2025-11～2026-04", (("原規則 2R", sel_base), ("選定 1R · 1h>MA20 · 深度≥10%", sel_pick)))}
{block("考卷：2026-05～10", (("原規則 2R", base), ("選定 1R · 1h>MA20 · 深度≥10%", picked), ("同一濾網但停利仍 2R", same_2r), ("只留下面≥21 根（2R，這次沒選到）", long)))}
{block("其中 10 月", (("原規則 2R", oct_base), ("選定", oct_pick)))}
<p class="muted">同一濾網如果停利留在 2R，這半年勝率 61.0%、平均 +3.61%，比 1R 少幾個勝率、多一點平均。10 月兩種停利都沒有救。下面待滿 21 根是上一輪的勝率規則，這次在前半年排名較後面，所以不是這次的選定。</p>
</section>
{cards}
</div></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    from tw_1h_reclaim import write_view_html

    view = write_view_html(out / "index.html")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("view", view, "picked", len(picked), "base", len(base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
