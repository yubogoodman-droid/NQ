#!/usr/bin/env python3
"""勝率優化：下面待滿 21 根才進場。訓練期挑、驗證期確認，測試期只看一次。

只回測、不下單。沒有改動監看預設。
"""

from __future__ import annotations

import base64
import json
import sys
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_tw_1h_reclaim import PERIODS, build_rows, in_period, load_cache, metrics  # noqa: E402
from tw_1h_reclaim import loose_params  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "tw-1h-reclaim-winrate-20261031"
MIN_BELOW = 21


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def cls(x: float) -> str:
    return "pnl-win" if x > 0 else ("pnl-loss" if x < 0 else "")


def row_html(title: str, rows) -> str:
    m = metrics(rows)
    if not m["n"]:
        return f"<tr><td>{escape(title)}</td><td colspan='6' class='muted'>沒有訊號</td></tr>"
    return (
        f"<tr><td>{escape(title)}</td><td>{m['n']}</td><td>{m['win']:.1f}%</td>"
        f"<td class='{cls(m['avg'])}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td>"
        f"<td>{pct(m['best'])} / {pct(m['worst'])}</td><td>{m['pf']:.2f}</td></tr>"
    )


def month_key(ts) -> str:
    if ts.year == 2026 and ts.month == 10:
        return "2026-10a" if ts.day <= 16 else "2026-10b"
    return f"{ts.year}-{ts.month:02d}"


def draw(path: Path, base, kept) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    months = []
    cur = pd.Timestamp("2025-11-01")
    end = pd.Timestamp("2026-09-01")
    while cur <= end:
        months.append(f"{cur.year}-{cur.month:02d}")
        cur += pd.DateOffset(months=1)
    months += ["2026-10a", "2026-10b"]

    def win_rate(rows, month: str) -> float:
        sel = [r for r in rows if month_key(r.entry_ts) == month]
        if not sel:
            return float("nan")
        return 100.0 * sum(1 for r in sel if r.pnl_pct > 0) / len(sel)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.2, 7.2), gridspec_kw={"height_ratios": [1.15, 1]}, facecolor="#0c1210"
    )
    for a in (ax, ax2):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    x = list(range(len(months)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [win_rate(base, m) for m in months], width=w, color="#8aa193", label="原規則")
    ax.bar([i + w / 2 for i in x], [win_rate(kept, m) for m in months], width=w, color="#f0c14b", label="下面 ≥ 21 根")
    ax.axhline(50, color="#2a3a33", lw=0.8)
    ax.set_ylabel("勝率 %", color="#8aa193")
    ax.set_title("月勝率 · 10上=驗證 · 10下=測試（黃柱沒有比較高）", color="#e8f0ea", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    # October is both valid (early) and test (late); shade the whole month lightly and mark test on equity.
    ax.axvspan(len(months) - 1.5, len(months) - 0.5, color="#ff525222", lw=0)

    def curve(rows):
        ordered = sorted(rows, key=lambda r: r.entry_ts)
        ys, xs = [], []
        total = 0.0
        for r in ordered:
            total += r.pnl_pct * 100
            xs.append(r.entry_ts)
            ys.append(total)
        return xs, ys

    for rows, color, name in (
        (base, "#8aa193", f"原規則（{len(base)} 筆）"),
        (kept, "#f0c14b", f"下面 ≥ 21 根（{len(kept)} 筆）"),
    ):
        xs, ys = curve(rows)
        ax2.plot(xs, ys, color=color, lw=1.5, label=name)
    for key, shade in (("valid", "#79c0ff22"), ("test", "#ff525233")):
        lo, hi = PERIODS[key]
        ax2.axvspan(lo, hi, color=shade, lw=0)
    ax2.axhline(0, color="#2a3a33", lw=0.8)
    ax2.set_ylabel("各筆 % 相加", color="#8aa193")
    ax2.set_title("累積報酬（未扣費稅）· 測試期黃線跌比較少，是因為少做，不是勝率變高", color="#e8f0ea", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    labels = ["10上" if m.endswith("10a") else "10下" if m.endswith("10b") else m[2:] for m in months]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#8aa193")
    fig.tight_layout(pad=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    rows = build_rows(load_cache(), loose_params())
    kept = [r for r in rows if r.bars_below >= MIN_BELOW]
    out = OUT
    (out / "img").mkdir(parents=True, exist_ok=True)
    draw(out / "img" / "winrate.png", rows, kept)
    png = (out / "img" / "winrate.png").read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    blocks = []
    summary = {}
    labels = (
        ("train", "訓練 2025-11～2026-09（用來挑規則）"),
        ("valid", "驗證 10/01～10/16（用來確認）"),
        ("test", "測試 10/17～10/30（挑完才看一次）"),
    )
    for key, title in labels:
        base_p = in_period(rows, key)
        kept_p = in_period(kept, key)
        summary[key] = {"base": metrics(base_p), "below21": metrics(kept_p)}
        blocks.append(
            f"<h3>{title}</h3><table><tr><th></th><th>筆數</th><th>勝率</th><th>平均</th>"
            f"<th>中位數</th><th>最好 / 最差</th><th>獲利因子</th></tr>"
            f"{row_html('原規則', base_p)}{row_html('下面 ≥ 21 根', kept_p)}</table>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · 勝率優化</title>
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
.verdict{{border-left:4px solid #f0c14b;padding-left:10px}}
</style></head><body><div class="page">
<section>
<h1>台股 1h 破底翻 · 能不能把勝率做高</h1>
<p class="muted">只回測、不下單。報酬未扣手續費與稅。偵測、2R、20 根出場都沒改，只在訊號出來後多一道：收盤在 MA20 下面要先待滿 21 根（大約 4～7 個交易日）才進場。宇宙仍是 2026-10-30 成交額前 200、股價 &lt; 1000，回推有倖存者偏差。</p>
</section>
<section class="verdict">
<h2>結論</h2>
<p class="muted">訓練期勝率 53.8% → 61.4%，驗證期 41.9% → 57.9%（38 筆），平均報酬也一起變好。但挑完才打開的 10/17～10/30，勝率 17.1% → 16.2%，沒有提高。那兩週幾乎什麼濾網都會輸。監看預設不改。</p>
<p class="muted">另外兩條路也試過，沒有採用：把停利從 2R 收到 1R，三個時期勝率都會高幾個點，但平均報酬變差。加權 1h 站上 MA20 在訓練和驗證看起來勝率很高（驗證 65.7%），測試期掉到 8.8%，跟上一份優化報告一樣。</p>
</section>
<section>
<h2>勝率與累積</h2>
<img src='{uri}' alt='月勝率與累積報酬'/>
</section>
<section><h2>數字</h2>{''.join(blocks)}
<p class="muted">下面只待 6～20 根的那一批，訓練勝率 49.6%、驗證 32.8%。訓練期前半（到 2026-04）待滿 21 根是 58.2%，未滿是 48.8%；後半是 64.1% 對 50.4%。198 檔都有出現，沒有集中在少數股票。2026-01 即使待滿 21 根，勝率仍只有 38.8%。</p>
</section>
</div></body></html>
"""
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    (out / "view.html").write_text(html, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("html", index, "png", len(png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
