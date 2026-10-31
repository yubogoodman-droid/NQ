#!/usr/bin/env python3
"""2026-08-03～10-30：原規則 vs 前一日加權日線收盤 > 日 MA20。只回測、不下單。"""

from __future__ import annotations

import json
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_tw_1h_reclaim import CACHE, build_rows, load_cache, metrics  # noqa: E402
from tw_1h_reclaim import loose_params, write_view_html  # noqa: E402
from watch_tw_1h_reclaim import daily_market_regime, regular_session_bars  # noqa: E402

WINDOW_LO = date(2026, 8, 3)
WINDOW_HI = date(2026, 10, 30)
# 券商滿額手續費 0.1425% × 買賣兩邊 + 證交稅 0.3%。有折扣會更低，這裡只做粗估。
ROUND_TRIP_FEE = 0.001425 * 2 + 0.003
OUT = Path(__file__).resolve().parents[1] / "docs" / "tw-1h-reclaim-3m-20260803-1030"


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def cls_avg(x: float) -> str:
    if x > 0:
        return "pnl-win"
    if x < 0:
        return "pnl-loss"
    return ""


def equity(rows, days) -> pd.Series:
    if not rows:
        return pd.Series(0.0, index=days)
    daily = pd.Series([r.pnl_pct * 100 for r in rows], index=[r.entry_ts.date() for r in rows])
    return daily.groupby(level=0).sum().reindex(days, fill_value=0.0).cumsum()


def max_dd(curve: pd.Series) -> float:
    return float((curve - curve.cummax()).min()) if len(curve) else 0.0


def stat_row(title: str, rows) -> str:
    m = metrics(rows)
    if not m["n"]:
        return f"<tr><td>{escape(title)}</td><td colspan='8' class='muted'>沒有訊號</td></tr>"
    return (
        f"<tr><td>{escape(title)}</td><td>{m['n']}</td><td>{m['win']:.1f}%</td>"
        f"<td class='{cls_avg(m['avg'])}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td>"
        f"<td>{pct(m['best'])}</td><td>{pct(m['worst'])}</td>"
        f"<td>{m['sum'] * 100:+.0f}</td><td>{pct(m['avg'] - ROUND_TRIP_FEE)}</td></tr>"
    )


def draw(path: Path, days, base_rows, filt_rows, on: set, daily_close, daily_ma) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]}, facecolor="#0c1210"
    )
    for a in (ax, ax2):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    x = list(range(len(days)))
    ax.plot(x, equity(base_rows, days).values, color="#8aa193", lw=1.6, label=f"原規則（{len(base_rows)} 筆）")
    ax.plot(x, equity(filt_rows, days).values, color="#f0c14b", lw=1.9, label=f"加大盤日線濾網（{len(filt_rows)} 筆）")
    ax.axhline(0, color="#2a3a33", lw=0.8)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor="#c8d5cc")
    ax.set_ylabel("各筆 % 相加", color="#8aa193", fontsize=8)
    ax.set_title(
        "2026-08-03～10-30 累積報酬（未扣費稅）· 灰底 = 濾網擋下、當天不進場",
        color="#e8f0ea",
        fontsize=12,
    )
    ax2.plot(x, [float(daily_close[d]) for d in days], color="#e8f0ea", lw=1.3, label="加權日收盤")
    ax2.plot(x, [float(daily_ma[d]) for d in days], color="#f472b6", lw=1.3, label="日 MA20")
    ax2.legend(loc="lower left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    ax2.set_ylabel("加權指數", color="#8aa193", fontsize=8)
    for i, d in enumerate(days):
        if d not in on:
            ax.axvspan(i - 0.5, i + 0.5, color="#ffffff18", lw=0)
            ax2.axvspan(i - 0.5, i + 0.5, color="#ffffff18", lw=0)
    step = max(1, len(days) // 10)
    ticks = list(range(0, len(days), step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([days[i].strftime("%m/%d") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    if not CACHE.exists():
        print(f"missing cache {CACHE}; run: python3 examples/optimize_tw_1h_reclaim.py fetch", file=sys.stderr)
        return 1
    cache = load_cache()
    twii = cache["twii"]
    session = regular_session_bars(twii)
    days = sorted(d for d in {ts.date() for ts in session.index} if WINDOW_LO <= d <= WINDOW_HI)
    on = set()
    for d in days:
        regime = daily_market_regime(twii, d)
        if regime.ok:
            on.add(d)
    rows = [r for r in build_rows(cache, loose_params()) if WINDOW_LO <= r.entry_ts.date() <= WINDOW_HI]
    base = rows
    filt = [r for r in rows if r.entry_ts.date() in on]
    daily_close = session["Close"].groupby([ts.date() for ts in session.index]).last()
    daily_ma = daily_close.rolling(20).mean()

    out = OUT
    (out / "img").mkdir(parents=True, exist_ok=True)
    draw(out / "img" / "equity.png", days, base, filt, on, daily_close, daily_ma)

    blocked_spans = []
    span = []
    for d in days:
        if d in on:
            if span:
                blocked_spans.append(span)
                span = []
        else:
            span.append(d)
    if span:
        blocked_spans.append(span)

    month_rows = []
    summary = {"fee_round_trip": ROUND_TRIP_FEE, "days": len(days), "filter_on_days": len(on)}
    for title, sel, key in (("原規則", base, "base"), ("前日加權日線 > 日 MA20", filt, "filter")):
        summary[key] = {"all": metrics(sel)}
        summary[key]["all"]["max_dd_pct_pts"] = max_dd(equity(sel, days))
        summary[key]["all"]["open"] = sum(1 for r in sel if r.reason == "open")
        summary[key]["months"] = {}
    for month, label in ((8, "8 月"), (9, "9 月"), (10, "10 月")):
        b = [r for r in base if r.entry_ts.month == month]
        f = [r for r in filt if r.entry_ts.month == month]
        month_rows.append(stat_row(f"{label} · 原規則", b))
        month_rows.append(stat_row(f"{label} · 濾網", f))
        summary["base"]["months"][label] = metrics(b)
        summary["filter"]["months"][label] = metrics(f)

    block_rows = []
    summary["blocked"] = []
    for span in blocked_spans:
        sel = [r for r in base if span[0] <= r.entry_ts.date() <= span[-1]]
        m = metrics(sel)
        summary["blocked"].append({"start": span[0].isoformat(), "end": span[-1].isoformat(), "days": len(span), **m})
        block_rows.append(
            f"<tr><td>{span[0]:%m/%d}–{span[-1]:%m/%d}</td><td>{len(span)}</td>"
            f"<td>{m['n']}</td><td>{m['win']:.1f}%</td>"
            f"<td class='{cls_avg(m['avg'])}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td></tr>"
        )

    b_m, f_m = metrics(base), metrics(filt)
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · 三個月大盤日線濾網</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:820px;margin:0 auto;padding:14px 12px 48px}}
section{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:19px;margin:0 0 8px}} h2{{font-size:16px;margin:0 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.65}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{border-bottom:1px solid #21262d;padding:6px 4px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:#8b949e;font-weight:500}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
img{{width:100%;border-radius:10px;display:block}}
code{{font-size:12px}}
</style></head><body><div class="page">
<section>
<h1>台股 1h 破底翻 · 三個月大盤日線濾網</h1>
<p class="muted">2026-08-03～10-30，60 個有成交的交易日。只回測、不下單。報酬<b>未扣</b>手續費與稅；最右欄是每筆再扣滿額成本 0.585%（手續費 0.1425%×2 + 證交稅 0.3%）的粗估。累積是各筆報酬相加，不是實際資金曲線。</p>
</section>
<section>
<h2>累積報酬與加權日線</h2>
<img src='img/equity.png' alt='三個月累積報酬與加權日線'/>
<p class="muted">上圖黃線是濾網版。灰底三段濾網不進場：8/3–8/5、9/15–9/17 是反彈剛起步，原規則賺最多的地方被擋掉；10/16–10/30 大盤跌破日 MA20，原規則從高點回吐，濾網整段沒有新單。</p>
</section>
<section>
<h2>三個月合計</h2>
<table>
<tr><th></th><th>筆數</th><th>勝率</th><th>平均</th><th>中位數</th><th>最好</th><th>最差</th><th>累積</th><th>扣 0.585%</th></tr>
{stat_row("原規則", base)}
{stat_row("前日加權 > 日 MA20", filt)}
</table>
<p class="muted">原規則最大回撤 {max_dd(equity(base, days)):.0f} 個百分點（高點 10/06），濾網 {max_dd(equity(filt, days)):.0f}。
原規則有 {sum(1 for r in base if r.reason == 'open')} 筆到 10/30 13:30 還沒滿 20 根，用最後收盤計價；濾網這段沒有未平倉。
放行 45 天、擋下 15 天。濾網少做 {len(base) - len(filt)} 筆，平均從 {pct(b_m['avg'])} 升到 {pct(f_m['avg'])}，累積從 {b_m['sum']*100:+.0f} 降到 {f_m['sum']*100:+.0f}。</p>
</section>
<section>
<h2>逐月</h2>
<table>
<tr><th></th><th>筆數</th><th>勝率</th><th>平均</th><th>中位數</th><th>最好</th><th>最差</th><th>累積</th><th>扣 0.585%</th></tr>
{''.join(month_rows)}
</table>
<p class="muted">濾網的改善幾乎都在 10 月。8 月、9 月平均比原規則差，因為兩段反彈的頭幾天還在日 MA20 下方。</p>
</section>
<section>
<h2>濾網擋下的日子（這些筆數只在原規則）</h2>
<table>
<tr><th>區間</th><th>交易日</th><th>筆數</th><th>勝率</th><th>平均</th><th>中位數</th></tr>
{''.join(block_rows)}
</table>
</section>
<section>
<h2>規則</h2>
<p class="muted">偵測、進場、停損、2R、20 根時間出場沿用 <code>examples/tw_1h_reclaim.py</code> 預設（寬鬆版，不要求均線多頭排列）。標的是 2026-10-30 成交額前 200、股價 &lt; 1000，同一批股票回推，有倖存者偏差。只算 09:00–13:30 的 1h K，週末與無成交日略過。</p>
<p class="muted">濾網與監看 <code>--market-filter daily-ma20</code> 相同：當天開盤前，看<b>前一個完整交易日</b>加權（^TWII）日線收盤是否高於該日的 20 日均線。日線收盤用當天最後一根盤中 1h。當天的 K 不納入。不成立的日子，進場不計入濾網版（實盤則寫狀態、不推播）。</p>
<p class="muted">這條濾網是看過 10 月之後才提出的，8–9 月又落在當初挑規則的訓練期裡，所以這份三個月結果不是樣本外驗證。</p>
</section>
</div></body></html>
"""
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    view = write_view_html(index)
    print("html", index)
    print("view", view)
    print("base", len(base), "filter", len(filt), "days", len(days), "on", len(on))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
