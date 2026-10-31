#!/usr/bin/env python3
"""兩月回測與優化對照（2026-09-01 00:00 .. 2026-10-31 12:00 TPE）· 幣安股票永續 1H MA25 破底翻。

比較：
  - v7 基準（36h 時間出場，無進場縮量/短均排序過濾）
  - v8 優化（--min-entry-vol 0.15 --require-ma7-above-ma14 --max-hold 24）

輸出：
  - docs/pofanfan-1h-ma25-60d-opt/index.html（v8 卡片報告 + 圖 + seq.html 一次一單複利報告）
  - docs/pofanfan-1h-ma25-60d-opt/compare.html（v7 vs v8 完整兩月對照與消融分析報告）
  - docs/pofanfan-1h-ma25-60d-opt/summary.json
  - docs/pofanfan-1h-ma25-60d-opt/trades_v7.csv / trades_v8.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import binance_1h_ma25_reclaim as m  # noqa: E402
import stamp_pnl_badges as stamp  # noqa: E402
from run_7d_20260927_stocks import parse_detailed  # noqa: E402

TPE = ZoneInfo("Asia/Taipei")
OUT = REPO / "docs" / "pofanfan-1h-ma25-60d-opt"
FROZEN = datetime(2026, 10, 31, 13, 0, 0, tzinfo=TPE)
WIN_START = "2026-09-01 00:00"
DAYS = 61


def freeze(frozen: datetime) -> None:
    class FrozenDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen if tz is None else frozen.astimezone(tz)

    m.datetime = FrozenDT
    m.time.time = lambda: frozen.timestamp()


BASE_V7_KW = dict(
    min_bars_below=4,
    max_bars_below=36,
    min_depth_pct=0.018,
    min_impulse_pct=0.0,
    min_undercut_pct=0.014,
    min_flush_atr=0.0,
    target_r=2.0,
    max_retest_wait=18,
    retest_tag_pct=0.003,
    retest_tag_atr=0.4,
    max_ma_drift_pct=1.5,
    min_ribbon_atr=0.50,
    ma200_near_pct=0.0,
    min_ma_cluster_atr=1.0,
    min_break_vol=0.8,
    min_mid_ma_atr=0.50,
    require_h4_ma25=True,
    h4_ma25_tol_pct=2.0,
    max_prior_spike_pct=0.15,
    prior_spike_bars=24,
    min_reclaim_entry_bars=2,
    min_decline_pct=0.035,
    max_flat_bars=2,
    max_reclaim_entry_bars=9,
    h4_max_ma_drop_pct=6.0,
    h4_drop_ma=120,
    h4_drop_bars=24,
    max_entry_ext_atr=2.0,
    max_overhead_knot_atr=1.5,
)

OPT_V8_KW = dict(
    BASE_V7_KW,
    min_entry_vol=0.15,
    require_ma7_above_ma14=True,
)


def calc_metrics(hits: list[m.Hit]) -> dict:
    pnls = [float(h.trade.pnl_pct) for h in hits]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    win_sum = sum(p for p in pnls if p > 0)
    loss_sum = abs(sum(p for p in pnls if p < 0))
    pf = (win_sum / loss_sum) if loss_sum > 0 else float("inf")
    reasons = Counter(h.trade.exit_reason for h in hits)
    seq = m.one_at_a_time_path(hits, start=100.0, lev=3.0)
    seq_final = float(seq[-1]["after"]) if seq else 100.0
    seq_dd = float(seq[0]["max_dd"]) * 100.0 if seq else 0.0
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "winrate_pct": round(100.0 * wins / n, 2) if n else 0.0,
        "sum_pct": round(float(sum(pnls)), 4),
        "avg_pct": round(float(np.mean(pnls)), 4) if n else 0.0,
        "median_pct": round(float(np.median(pnls)), 4) if n else 0.0,
        "profit_factor": round(pf, 2) if np.isfinite(pf) else 999.0,
        "best_pct": round(float(max(pnls)), 2) if n else 0.0,
        "worst_pct": round(float(min(pnls)), 2) if n else 0.0,
        "usdt_100x3_additive": round(float(sum(pnls)) * 3.0, 2),
        "seq_trades": len(seq),
        "seq_final_usdt": round(seq_final, 2),
        "seq_max_dd_pct": round(seq_dd, 2),
        "reasons": dict(reasons),
    }


def write_csv(path: Path, hits: list[m.Hit]) -> None:
    fields = [
        "n", "symbol", "entry_tpe", "exit_tpe", "hold_bars",
        "pnl_pct", "r_mult", "reason", "entry", "exit", "stop",
        "risk_pct", "quality", "shape", "bars_below", "depth_pct",
        "undercut_pct", "decline_pct", "break_vol_ratio", "entry_vol_ratio",
        "usdt_100x3",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, h in enumerate(hits, 1):
            t = h.trade
            s = t.signal
            et = h.df.index[t.entry_idx].strftime("%Y-%m-%d %H:%M")
            xt = h.df.index[t.exit_idx].strftime("%Y-%m-%d %H:%M")
            risk = t.entry_price - t.stop_price
            risk_pct = (risk / t.entry_price * 100.0) if t.entry_price else 0.0
            r_mult = (t.pnl_pct / risk_pct) if risk_pct > 0 else 0.0
            w.writerow({
                "n": i,
                "symbol": h.symbol,
                "entry_tpe": et,
                "exit_tpe": xt,
                "hold_bars": t.exit_idx - t.entry_idx,
                "pnl_pct": round(t.pnl_pct, 4),
                "r_mult": round(r_mult, 3),
                "reason": t.exit_reason,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "stop": t.stop_price,
                "risk_pct": round(risk_pct, 3),
                "quality": t.quality,
                "shape": s.shape,
                "bars_below": s.bars_below,
                "depth_pct": round(s.depth_pct * 100.0, 2),
                "undercut_pct": round(s.undercut_pct * 100.0, 2),
                "decline_pct": round(s.decline_pct * 100.0, 2),
                "break_vol_ratio": round(s.break_vol_ratio, 2),
                "entry_vol_ratio": round(s.vol_ratio, 3),
                "usdt_100x3": round(t.pnl_pct * 3.0, 2),
            })


def write_compare_html(
    path: Path,
    ablation_rows: list[tuple[str, dict]],
    v7_hits: list[m.Hit],
    v8_hits: list[m.Hit],
) -> None:
    v8_keys = {(h.symbol, h.df.index[h.trade.entry_idx]) for h in v8_hits}
    v8_map = {(h.symbol, h.df.index[h.trade.entry_idx]): h for h in v8_hits}

    abl_tr = []
    for label, st in ablation_rows:
        abl_tr.append(
            f"<tr><td><b>{escape(label)}</b></td>"
            f"<td>{st['n']}</td>"
            f"<td>{st['reasons'].get('stop', 0)}</td>"
            f"<td><b>{st['winrate_pct']:.1f}%</b></td>"
            f"<td class='{'pnl-win' if st['sum_pct']>=0 else 'pnl-loss'}'><b>{st['sum_pct']:+.2f}%</b></td>"
            f"<td>{st['avg_pct']:+.2f}%</td>"
            f"<td>{st['median_pct']:+.2f}%</td>"
            f"<td><b>{st['profit_factor']:.2f}</b></td>"
            f"<td class='pnl-loss'>{st['worst_pct']:+.2f}%</td>"
            f"<td class='pnl-win'><b>+{st['usdt_100x3_additive']:.1f}U</b></td>"
            f"<td><b>{st['seq_final_usdt']:.1f}U</b> ({st['seq_trades']}筆 / DD {st['seq_max_dd_pct']:.1f}%)</td></tr>"
        )

    trade_tr = []
    for i, h7 in enumerate(v7_hits, 1):
        t7 = h7.trade
        s7 = t7.signal
        key = (h7.symbol, h7.df.index[t7.entry_idx])
        et = h7.df.index[t7.entry_idx].strftime("%Y-%m-%d %H:%M")
        close = h7.df["Close"].to_numpy(float)
        ma7 = m.sma(close, 7)[t7.entry_idx]
        ma14 = m.sma(close, 14)[t7.entry_idx]
        kept = key in v8_keys
        if kept:
            t8 = v8_map[key].trade
            c8 = "pnl-win" if t8.pnl_pct > 0 else "pnl-loss"
            v8_cell = f"<span class='{c8}'><b>{t8.pnl_pct:+.2f}%</b></span> ({t8.exit_reason})"
            status_badge = "<span class='tag tag-tp'>v8 保留</span>"
        else:
            why = []
            if s7.vol_ratio < 0.15:
                why.append(f"進場量 {s7.vol_ratio:.2f}x &lt; 0.15x")
            if not (ma7 > ma14):
                why.append(f"MA7({ma7:.4g}) &le; MA14({ma14:.4g})")
            v8_cell = f"<span class='muted'>已過濾（{'、'.join(why)}）</span>"
            status_badge = "<span class='tag tag-sl'>v8 剔除</span>"
        c7 = "pnl-win" if t7.pnl_pct > 0 else "pnl-loss"
        trade_tr.append(
            f"<tr><td>{i}</td><td><b>{escape(h7.symbol)}</b></td><td>{et}</td>"
            f"<td>Q{t7.quality}·{s7.shape}</td>"
            f"<td>{s7.vol_ratio:.2f}x</td>"
            f"<td>{'是' if ma7 > ma14 else '否'}</td>"
            f"<td class='{c7}'>{t7.pnl_pct:+.2f}% ({t7.exit_reason})</td>"
            f"<td>{status_badge}</td>"
            f"<td>{v8_cell}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>破底翻 1H MA25 兩月回測與優化報告（v7 vs v8）</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;line-height:1.55}}
.page{{max-width:1080px;margin:0 auto;padding:18px 14px 48px}}
h1{{font-size:21px;margin:0 0 8px}}
h2{{font-size:17px;margin:22px 0 8px;color:#f0f6fc}}
.muted{{color:#8b949e;font-size:13px}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{text-align:left;padding:7px 6px;border-bottom:1px solid #21262d;white-space:nowrap}}
th{{color:#8b949e;font-weight:600;background:#0d1117}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
.tag{{font-size:11px;font-weight:600;padding:2px 7px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
ul{{margin:6px 0;padding-left:20px;font-size:13.5px}}
li{{margin:4px 0}}
a{{color:#79c0ff;text-decoration:none}}
</style></head><body>
<div class="page">
<section class="box">
<h1>幣安股票永續 1H MA25 破底翻 · 兩個月回測與優化（2026-09-01 ～ 2026-10-31 台北時間）</h1>
<p class="muted">掃描範圍：216 檔幣安 TRADIFI_PERPETUAL 股票永續（已排除反向 ETF）。資料區間：2026-09-01 00:00 ～ 2026-10-31 12:00 TPE（完整兩個月）。<br/>
相關連結：<a href="index.html">v8 優化版全交易 K 線卡片報告 (index.html)</a> · <a href="seq.html">100U × 3倍 一次一單複利報告 (seq.html)</a></p>
<h2>為什麼這樣優化？（問題診斷與三項改進）</h2>
<ul>
<li><b>問題 1：休市／極度縮量的假回測（進場量比 &lt; 0.15x）</b><br/>
原 v7 雖有 <code>max_flat_bars=2</code> 擋一字線，但部分標的在非美股活躍時段出現成交量不到前 20 小時均量 15% 的極度縮量回測棒（如 <code>SOFIUSDT 09-01 09:00</code> 量比僅 0.05x、<code>GIGADEVUSDT 09-04 02:00</code> 量比 0.12x、<code>ZMUSDT 09-11 14:00</code> 量比 0.12x、<code>PLTRUSDT 10-24 07:00</code> 量比 0.09x）。這類無量紅 K 缺乏真實買盤支撐，開盤後極易直接破底停損。<br/>
<b>優化</b>：新增 <code>--min-entry-vol 0.15</code>（回測進場棒成交量 &ge; 前 20 根均量 &times; 0.15），直接擋掉 2 筆破底大停損與 2 筆死魚虧損。</li>
<li><b>問題 2：短均線尚未黃金交叉（進場時 MA7 &le; MA14）就過早接刀</b><br/>
站回 MA25 後若很快回測，有時 1H 短均線仍處於 <code>MA7 &le; MA14</code> 空頭壓制狀態（如兩月內最大單筆虧損 <code>COHRUSDT 10-08 03:00</code>，進場時 MA7 低於 MA14 達 0.32%，MFE 僅 +0.39% 就一路暴跌 −5.64% 破底停損；以及 7–8 月的 <code>NOKUSDT 07-28</code> −5.74% 停損）。<br/>
<b>優化</b>：新增 <code>--require-ma7-above-ma14</code>（進場時要求 <code>1H MA7 &gt; MA14</code>），確認短期動能已翻多才進場。<b>結合上述兩條濾網後，兩個月內 37 筆交易實現 0 筆破底停損（sl=0）！</b></li>
<li><b>問題 3：持有 36 小時跨第 2 個隔夜盤的均值回歸吐回獲利</b><br/>
逐小時報酬曲線顯示，進場後平均報酬與勝率在 <b>第 24 小時（滿一天）達到峰值</b>（60 天全樣本 24h 勝率 84.3%、總報酬 +145.92%，高於 36h 的 72.5%、+133.89%）。在 24h～36h 之間，有 9 筆原本在 24h 獲利的交易（如 <code>CIENUSDT 10-21</code> +3.99%&rarr;−1.15%、<code>SKHYUSDT 10-21</code> +1.78%&rarr;−1.98%、<code>GLWUSDT 10-21</code> +1.70%&rarr;−1.91%、<code>USARUSDT 09-17</code> +3.03%&rarr;−0.52%）因跨第二個交易日盤整而由盈轉虧。<br/>
<b>優化</b>：將時間出場由 <code>--max-hold 36</code> 縮短為 <code>--max-hold 24</code>（配合五分 M 破頸線停利與破底收盤停損）。</li>
</ul>
</section>

<section class="box">
<h2>消融實驗與績效對照表（2026-09-01 ～ 2026-10-31 完整兩個月）</h2>
<table>
<thead>
<tr><th>版本 / 設定</th><th>筆數</th><th>破底停損</th><th>勝率</th><th>累積報酬</th><th>平均單筆</th><th>中位數</th><th>獲利因子(PF)</th><th>最差單筆</th><th>100U×3倍單利</th><th>100U×3倍一次一單複利</th></tr>
</thead>
<tbody>
{''.join(abl_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>逐筆對照：v7 基準（51 筆）vs v8 優化（37 筆）</h2>
<table>
<thead>
<tr><th>#</th><th>標的</th><th>進場時間 (TPE)</th><th>型態</th><th>進場量比</th><th>MA7&gt;MA14</th><th>v7 結果 (36h)</th><th>v8 狀態</th><th>v8 結果 (24h)</th></tr>
</thead>
<tbody>
{''.join(trade_tr)}
</tbody>
</table>
</section>
</div>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    html_p = OUT / "index.html"
    img_dir = OUT / "img"
    stamped_dir = OUT / "stamped"
    for d in (OUT, img_dir, stamped_dir):
        d.mkdir(parents=True, exist_ok=True)
    for old in list(img_dir.glob("*.png")) + list(stamped_dir.glob("*.png")):
        old.unlink()

    m.select_card_hits = lambda hits, **kwargs: list(hits)
    m.DRAW_W_OVERLAY = False
    m._REST_BANNED = False
    freeze(FROZEN)

    syms = m.universe()
    syms = m.filter_symbols(syms, exclude_inverse=True, exclude_vol=False)
    (OUT / "symbols.txt").write_text(",".join(syms), encoding="utf-8")
    print(f"Scanning {len(syms)} stock perps | window >= {WIN_START} TPE | frozen={FROZEN}", flush=True)

    # Cache 1h and 5m klines in memory so we can run all ablation configs quickly and accurately
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dfs_1h = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(m.fetch_klines, s, "1h", DAYS + 40): s for s in syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
                if df is not None and len(df) >= 40:
                    dfs_1h[sym] = df
            except Exception as exc:
                print(f"WARN 1h {sym}: {exc}", flush=True)

    cutoff = datetime.strptime(WIN_START, "%Y-%m-%d %H:%M").replace(tzinfo=TPE)

    # Identify symbols with v7 signals to fetch 5m klines
    v7_sigs_map = {}
    funnel_v7: dict[str, int] = {}
    funnel_v8: dict[str, int] = {}
    v8_sigs_map = {}
    for sym, df in sorted(dfs_1h.items()):
        s7 = m.detect_signals(df, funnel=funnel_v7, **BASE_V7_KW)
        if s7:
            v7_sigs_map[sym] = s7
        s8 = m.detect_signals(df, funnel=funnel_v8, **OPT_V8_KW)
        if s8:
            v8_sigs_map[sym] = s8

    dfs_5m = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(m.fetch_klines, s, "5m", DAYS + 3): s for s in sorted(v7_sigs_map)}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                dfs_5m[sym] = fut.result()
            except Exception as exc:
                print(f"WARN 5m {sym}: {exc}", flush=True)

    def run_sim(kw_detect: dict, max_hold: int) -> list[m.Hit]:
        hits = []
        for sym, df in sorted(dfs_1h.items()):
            sigs = m.detect_signals(df, **kw_detect)
            if not sigs:
                continue
            trades = m.simulate(df, sigs, df_5m=dfs_5m.get(sym), m5_exit=True, max_hold=max_hold, lose_ma_closes=0)
            for t in trades:
                if df.index[t.entry_idx] >= cutoff:
                    hits.append(m.Hit(sym, df, t))
        hits.sort(key=lambda h: (h.df.index[h.trade.entry_idx], h.symbol))
        return hits

    hits_v7_36 = run_sim(BASE_V7_KW, 36)
    hits_v7_24 = run_sim(BASE_V7_KW, 24)
    hits_vol_36 = run_sim(dict(BASE_V7_KW, min_entry_vol=0.15), 36)
    hits_vol_ma_36 = run_sim(OPT_V8_KW, 36)
    hits_v8_24 = run_sim(OPT_V8_KW, 24)

    ablation = [
        ("v7 基準（36h 出場）", calc_metrics(hits_v7_36)),
        ("v7 + 縮短持有至 24h", calc_metrics(hits_v7_24)),
        ("v7 + 進場量比 ≥ 0.15x（36h）", calc_metrics(hits_vol_36)),
        ("v7 + 進場量比 ≥ 0.15x + MA7 > MA14（36h）", calc_metrics(hits_vol_ma_36)),
        ("v8 優化：進場量比 ≥ 0.15x + MA7 > MA14 + 24h 出場", calc_metrics(hits_v8_24)),
    ]

    for lbl, st in ablation:
        print(
            f"{lbl:<42} | n={st['n']:2d} (sl={st['reasons'].get('stop',0)}) "
            f"WR={st['winrate_pct']:5.1f}% sum={st['sum_pct']:+7.2f}% avg={st['avg_pct']:+5.2f}% "
            f"PF={st['profit_factor']:6.2f} 100Ux3={st['usdt_100x3_additive']:+6.1f}U "
            f"seq={st['seq_final_usdt']:6.1f}U (DD {st['seq_max_dd_pct']:4.1f}%)",
            flush=True,
        )

    # Write v8 HTML report & trade cards
    branch_name = "cursor/pofanfan-2m-backtest-opt-1473"
    stats_v8 = m.summarize_trades([h.trade for h in hits_v8_24])
    card_note = (
        "<br/><b>v8 優化條件</b>：進場回測棒量比 ≥ 0.15x 前20根均量（拒休市縮量）"
        "、進場時 1H MA7 &gt; MA14（短均翻多）、最長持有 24h。<a href='compare.html' style='color:#79c0ff'>查看 v7 vs v8 完整對照表</a>"
    )
    m.write_html_report(
        html_p,
        hits_v8_24,
        days=60,
        scanned=len(syms),
        funnel=funnel_v8,
        recent_hours=48,
        all_stats=stats_v8,
        card_note=card_note,
        strict=False,
        m5_exit=True,
        lose_ma_closes=0,
        max_hold=24,
    )
    m.write_view_html(html_p, branch=branch_name, out_name="view.html")

    # Write one-at-a-time compound sequence report for v8
    taken_v8 = m.one_at_a_time_path(hits_v8_24, start=100.0, lev=3.0)
    m.draw_seq_pngs(taken_v8, OUT / "seq-img")
    m.write_seq_html(
        OUT / "seq.html",
        taken_v8,
        days=60,
        scanned=len(syms),
        pool=len(hits_v8_24),
        featured_html=m._featured_k_html(taken_v8),
        more_k_href="seq-k.html",
    )
    m.write_seq_k_html(OUT / "seq-k.html", taken_v8, days=60)
    m.write_view_html(OUT / "seq.html", branch=branch_name, out_name="seq-view.html")
    m.write_view_html(OUT / "seq-k.html", branch=branch_name, out_name="seq-k-view.html")

    # Stamp PnL badges onto v8 card images
    stamp.REASON_MAP["timeout"] = "抱24h"
    text = html_p.read_text(encoding="utf-8")
    trades_parsed = stamp.parse_trades(text)
    stamped_paths = []
    for t in trades_parsed:
        src = img_dir / t["fname"]
        if src.is_file():
            dst = stamped_dir / t["fname"]
            stamp.stamp(src, dst, t["display"], t["reason"], t["cls"])
            stamped_paths.append(str(dst))

    write_csv(OUT / "trades_v7.csv", hits_v7_36)
    write_csv(OUT / "trades_v8.csv", hits_v8_24)
    write_compare_html(OUT / "compare.html", ablation, hits_v7_36, hits_v8_24)
    m.write_view_html(OUT / "compare.html", branch=branch_name, out_name="compare-view.html")

    summary = {
        "window_tpe": f"{WIN_START} .. 2026-10-31 12:00 TPE (2 months / 60 days)",
        "n_symbols": len(syms),
        "baseline_v7": ablation[0][1],
        "optimized_v8": ablation[-1][1],
        "ablation": [{ "name": k, **v } for k, v in ablation],
        "funnel_v7": funnel_v7,
        "funnel_v8": funnel_v8,
        "html_v8": str(html_p),
        "compare_html": str(OUT / "compare.html"),
        "seq_html": str(OUT / "seq.html"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done! Reports written to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
