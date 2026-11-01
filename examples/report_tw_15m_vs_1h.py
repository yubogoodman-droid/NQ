#!/usr/bin/env python3
"""15 分 K vs 1 小時 K：Yahoo 15m 只有最近 60 天（2026-08-04～10-30），沒辦法拉半年。只回測、不下單。"""

from __future__ import annotations

import base64
import json
import pickle
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_tw_1h_reclaim import CLOSED_DAYS, build_rows, daily_regime, load_cache, metrics  # noqa: E402
from tw_1h_reclaim import detect_signals, draw_trade_png, loose_params, simulate, write_view_html  # noqa: E402

CACHE_15M = Path("/tmp/tw1h_cache/cache_15m_60d.pkl")
OUT = Path(__file__).resolve().parents[1] / "docs" / "tw-15m-vs-1h-20260804-1030"
ROUND_TRIP_FEE = 0.001425 * 2 + 0.003
SEL = (date(2026, 8, 4), date(2026, 9, 30))
OCT = (date(2026, 10, 1), date(2026, 10, 30))
ALL = (date(2026, 8, 4), date(2026, 10, 30))
REASON = {"stop": "停損", "target": "停利 2R", "time": "時間出場", "open": "未平"}


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def cls(x: float) -> str:
    return "pnl-win" if x > 0 else ("pnl-loss" if x < 0 else "")


def win(rows, bounds):
    lo, hi = bounds
    return [r for r in rows if lo <= r.entry_ts.date() <= hi]


def clean_15m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # 15m 盤中開盤是 09:00..13:15 共 18 根；最後一天 Yahoo 會多一根 13:30 零量收盤撮合，拿掉。
    keep = [
        ts.weekday() < 5 and ts.date() not in CLOSED_DAYS and 9 * 60 <= ts.hour * 60 + ts.minute <= 13 * 60 + 15
        for ts in df.index
    ]
    return df.loc[keep]


def mrow(title: str, rows) -> str:
    m = metrics(rows)
    if not m["n"]:
        return f"<tr><td>{escape(title)}</td><td colspan='7'>0</td></tr>"
    net = m["avg"] - ROUND_TRIP_FEE
    return (
        f"<tr><td>{escape(title)}</td><td>{m['n']}</td><td>{m['win']:.1f}%</td>"
        f"<td class='{cls(m['avg'])}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td>"
        f"<td>{pct(m['best'])} / {pct(m['worst'])}</td><td>{m['pf']:.2f}</td>"
        f"<td class='{cls(net)}'>{pct(net)}</td></tr>"
    )


def draw_summary(path: Path, series: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.2, 7.0), sharex=True, facecolor="#0c1210")
    for a in (ax, ax2):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    colors = ["#8aa193", "#79c0ff", "#f472b6", "#f0c14b"]
    for (name, rows), col in zip(series.items(), colors):
        ordered = sorted(rows, key=lambda r: r.entry_ts)
        xs = [r.entry_ts for r in ordered]
        raw, net = [], []
        s1 = s2 = 0.0
        for r in ordered:
            s1 += r.pnl_pct * 100
            s2 += (r.pnl_pct - ROUND_TRIP_FEE) * 100
            raw.append(s1)
            net.append(s2)
        ax.plot(xs, raw, color=col, lw=1.5, label=f"{name}（{len(rows)} 筆）")
        ax2.plot(xs, net, color=col, lw=1.5, label=name)
    oct_lo = pd.Timestamp("2026-10-01", tz="Asia/Taipei")
    oct_hi = pd.Timestamp("2026-10-30 23:59", tz="Asia/Taipei")
    for a in (ax, ax2):
        a.axvspan(oct_lo, oct_hi, color="#ff525222", lw=0)
        a.axhline(0, color="#2a3a33", lw=0.8)
    ax.set_ylabel("各筆 % 相加", color="#8aa193")
    ax.set_title("2026-08-04～10-30 未扣費稅 · 紅底 = 10 月考卷", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    ax2.set_ylabel("每筆再扣 0.585%", color="#8aa193")
    ax2.set_title("扣滿額費稅 0.585% 後 · 15 分 K 單筆空間太小，扣完幾乎不賺", color="#e8f0ea", fontsize=12)
    fig.tight_layout(pad=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def trade_cards(c15: dict, rows, img_dir: Path) -> str:
    import tw_1h_reclaim as tw

    setup, ready = tw._setup_cjk, {"ok": False}

    def once() -> None:
        if not ready["ok"]:
            setup()
            ready["ok"] = True

    tw._setup_cjk = once  # type: ignore[method-assign]
    wanted: dict = {}
    for r in rows:
        wanted.setdefault(r.symbol, set()).add(r.entry_ts)
    params = loose_params()
    found = {}
    for sym, stamps in wanted.items():
        df = c15["data"][sym]
        for trade in simulate(df, detect_signals(df, params), params):
            ts = df.index[trade.entry_idx]
            if ts in stamps:
                found[(sym, ts)] = (df, trade)
    missing = [r for r in rows if (r.symbol, r.entry_ts) not in found]
    if missing:
        raise SystemExit(f"missing {missing[0].code} {missing[0].entry_ts}")
    cards = [
        "<section><h2>15 分 K 選定版這 60 天的每一筆 K 線</h2>"
        f"<p class='muted'>{len(rows)} 筆都是 15m、原參數（6～36 根在 MA20 下、20 根出場、2R）、破底深度至少 8%。"
        "綠三角進場、黃叉出場、黃／藍／粉是 MA5／10／20、粉點破底、紅虛線停損、綠虛線 2R。</p></section>"
    ]
    for i, r in enumerate(rows, 1):
        df, trade = found[(r.symbol, r.entry_ts)]
        name = f"t{i:03d}_{r.code}_{r.entry_ts.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, trade, img_dir / name, i, title_extra=f"{r.code} {r.name} 15m")
        reason = REASON.get(trade.exit_reason, trade.exit_reason)
        klass = "pnl-win" if r.pnl_pct > 0 else "pnl-loss"
        cards.append(
            "<article class='trade-card'><header class='card-header'>"
            f"<span>#{i} · {escape(r.code)} {escape(r.name)}</span>"
            f"<span class='{klass}'>{r.pnl_pct * 100:+.2f}%</span></header>"
            f"<p class='muted'>{r.entry_ts.strftime('%m-%d %H:%M')} 進 {r.entry:.2f} → "
            f"{r.exit_ts.strftime('%m-%d %H:%M')} {escape(reason)} {r.exit_px:.2f} · 深度 {r.depth * 100:.1f}%</p>"
            f"<img src='img/{name}' alt='{escape(r.code)} {escape(r.name)}'/></article>"
        )
        if i % 50 == 0:
            print(f"charts {i}/{len(rows)}", flush=True)
    return "".join(cards)


def main() -> int:
    c1h = load_cache()
    raw15 = pickle.loads(CACHE_15M.read_bytes())
    c15 = {
        "universe": raw15["universe"],
        "twii": clean_15m(raw15["twii"]),
        "data": {k: clean_15m(v) for k, v in raw15["data"].items()},
    }
    reg = daily_regime(c1h["twii"])

    def tag(rows):
        for r in rows:
            v = reg.get(r.entry_ts.date())
            r.daily_up = (not pd.isna(v)) and bool(v)
        return rows

    r1h = win(tag(build_rows(c1h, loose_params())), ALL)
    r1h_daily = [r for r in r1h if r.daily_up]
    r15_raw = win(tag(build_rows(c15, loose_params())), ALL)
    p15_scaled = loose_params(min_below=20, max_below=120, lookback=60, entry_window=120, time_bars=72)
    r15_scaled = win(tag(build_rows(c15, p15_scaled)), ALL)
    r15_pick = [r for r in r15_raw if r.depth >= 0.08]
    r15_scaled_daily = [r for r in r15_scaled if r.daily_up]

    out = OUT
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    draw_summary(
        img / "compare.png",
        {
            "1h 原規則": r1h,
            "1h + 前日日線>MA20": r1h_daily,
            "15m 無濾網（6–36 根）": r15_raw,
            "15m 選定（深度≥8%）": r15_pick,
        },
    )
    uri = "data:image/png;base64," + base64.b64encode((img / "compare.png").read_bytes()).decode("ascii")
    cards = trade_cards(c15, r15_pick, img)

    variants = (
        ("1h 原規則（2R / 20 根）", r1h),
        ("1h + 前日加權 > 日 MA20", r1h_daily),
        ("15m 原參數（6–36 根 / 20 根出場）", r15_raw),
        ("15m 時間換算（20–120 根 / 72 根出場）", r15_scaled),
        ("15m 選定：原參數 + 深度 ≥ 8%", r15_pick),
        ("15m 時間換算 + 前日加權 > 日 MA20", r15_scaled_daily),
    )

    def block(title, bounds):
        body = "".join(mrow(name, win(rows, bounds)) for name, rows in variants)
        return (
            f"<h3>{title}</h3><table><tr><th></th><th>筆數</th><th>勝率</th><th>平均</th>"
            f"<th>中位數</th><th>最好 / 最差</th><th>獲利因子</th><th>扣 0.585%</th></tr>{body}</table>"
        )

    summary = {
        name: {k: metrics(win(rows, b)) for k, b in (("aug_sep", SEL), ("oct", OCT), ("all_60d", ALL))}
        for name, rows in variants
    }
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股破底翻 · 15 分 K 對照 1 小時 K</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:840px;margin:0 auto;padding:14px 12px 48px}}
section{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:19px;margin:0 0 8px}} h2{{font-size:16px;margin:0 0 8px}} h3{{font-size:14px;margin:12px 0 6px;color:#c9d1d9}}
.muted{{color:#8b949e;font-size:13px;line-height:1.65}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{border-bottom:1px solid #21262d;padding:6px 4px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:#8b949e;font-weight:500}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
img{{width:100%;border-radius:10px;display:block}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:12px;margin-bottom:12px}}
.card-header{{display:flex;justify-content:space-between;gap:8px;font-weight:700}}
.verdict{{border-left:4px solid #ff5252;padding-left:10px}}
</style></head><body><div class="page">
<section>
<h1>台股破底翻 · 15 分 K 對照 1 小時 K</h1>
<p class="muted">先講資料限制：Yahoo Finance 的 15 分 K 最多只給最近 60 天（這批是 <b>2026-08-04～10-30</b>，59 個交易日），再往前抓會直接回 HTTP 422，所以 15 分 K <b>拿不到半年</b>。下面用同一段 60 天把 15m 和 1h 放在一起比。8–9 月用來挑 15m 規則，10 月當考卷。只回測、不下單。</p>
</section>
<section class="verdict">
<h2>結論</h2>
<p class="muted">15 分 K 不如 1 小時 K。直接套 6～36 根，60 天出 1540 筆、勝率 43.3%、平均 +0.20%；把根數乘以 4 換成同樣的天數（20～120 根在 MA20 下、72 根出場），775 筆、勝率 44.4%、平均 +0.29%。8–9 月挑出最好的 15m 條件是「原參數 + 破底深度 ≥ 8%」，8–9 月勝率 55.6%、平均 +1.37%，到了 10 月掉成 33.8%、平均 −0.46%，60 天合計平均 +0.46%。</p>
<p class="muted">關鍵在最後一欄：15 分 K 的停損距離短，2R 和時間出場拿到的振幅也小，單筆毛利只有 +0.2%～+0.5%，一扣台股滿額手續費加證交稅 0.585%，全部變成負數。同一段時間 1h 原規則平均 +0.49%，加上前日加權日線 &gt; 日 MA20 是 +1.34%（扣費後 +0.75%）。監看維持 1h，不改。</p>
</section>
<section>
<h2>未扣費稅 vs 扣 0.585%</h2>
<img src='{uri}' alt='15m 與 1h 累積報酬對照'/>
</section>
<section>
<h2>數字（同一段 2026-08-04～10-30）</h2>
{block("全部 60 天（08/04～10/30）", ALL)}
{block("用來挑規則：8–9 月（08/04～09/30）", SEL)}
{block("考卷：10 月（10/01～10/30）", OCT)}
<p class="muted">15m 每天盤中是 09:00～13:15 共 18 根（最後一天 Yahoo 多給的一根 13:30 零量收盤撮合已剔除）。宇宙仍是 2026-10-30 成交額前 200、股價 &lt; 1000，有倖存者偏差。</p>
</section>
{cards}
</div></body></html>
"""
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    view = write_view_html(index)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("view", view, "pick", len(r15_pick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
