#!/usr/bin/env python3
"""全歷史回測與優化對照（2026-01-28 22:00 .. 2026-10-31 12:00 TPE，共 276 天 / 9 個月）· 幣安股票永續 1H MA25 破底翻。

幣安 TRADIFI_PERPETUAL 股票永續合約最早自 2026-01-28 22:00 TPE（TSLAUSDT）起陸續掛牌上線，
截至 2026-10-31 共 216 檔（排除 5 檔反向 ETF 後），合計 586,737 根 1H K 線。
本腳本回測「從第一檔股票永續上線以來」的完整歷史極限區間（2026-01-28 ～ 2026-10-31）。

輸出：
  - docs/pofanfan-1h-ma25-full-opt/index.html / view.html（v9 全歷史 52 筆交易卡片報告 + 含進出場圓圈標記 K 線圖）
  - docs/pofanfan-1h-ma25-full-opt/compare.html / compare-view.html（全歷史 v7 vs v8 vs v9 完整消融、逐月拆解、掛牌分布與 100 筆逐筆對照報告）
  - docs/pofanfan-1h-ma25-full-opt/seq.html / seq-view.html（100U × 3倍 一次一單複利報告）
  - docs/pofanfan-1h-ma25-full-opt/summary.json & trades_v7.csv / trades_v8.csv / trades_v9.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import binance_1h_ma25_reclaim as m  # noqa: E402
import stamp_pnl_badges as stamp  # noqa: E402

TPE = ZoneInfo("Asia/Taipei")
OUT = REPO / "docs" / "pofanfan-1h-ma25-full-opt"
FROZEN = datetime(2026, 10, 31, 13, 0, 0, tzinfo=TPE)
WIN_START = "2026-01-28 00:00"
DAYS = 280
BRANCH = "cursor/pofanfan-2m-backtest-opt-1473"
CACHE_DIR = Path("/tmp/pofanfan_cache_max")


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

OPT_V9_KW = dict(
    OPT_V8_KW,
    min_break_vol=2.2,
    min_undercut_pct=0.016,
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


def monthly_breakdown(hits: list[m.Hit]) -> dict[str, dict]:
    months = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
    by_m: dict[str, list[m.Hit]] = {ym: [] for ym in months}
    for h in hits:
        ym = h.df.index[h.trade.entry_idx].strftime("%Y-%m")
        by_m.setdefault(ym, []).append(h)
    return {ym: calc_metrics(hs) for ym, hs in by_m.items()}


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
    ablation_rows: list[tuple[str, dict, dict, dict, dict]],
    m_v7: dict[str, dict],
    m_v8: dict[str, dict],
    m_v9: dict[str, dict],
    v7_hits: list[m.Hit],
    v8_hits: list[m.Hit],
    v9_hits: list[m.Hit],
) -> None:
    v8_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v8_hits, 1)}
    v9_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v9_hits, 1)}

    abl_tr = []
    for label, st_all, st_early, st_mid, st_late in ablation_rows:
        abl_tr.append(
            f"<tr><td><b>{escape(label)}</b></td>"
            f"<td>{st_all['n']}</td>"
            f"<td><b>{st_all['reasons'].get('stop', 0)}</b></td>"
            f"<td><b>{st_all['winrate_pct']:.1f}%</b></td>"
            f"<td class='{'pnl-win' if st_all['sum_pct']>=0 else 'pnl-loss'}'><b>{st_all['sum_pct']:+.2f}%</b></td>"
            f"<td>{st_all['avg_pct']:+.2f}%</td>"
            f"<td><b>{st_all['profit_factor']:.2f}</b></td>"
            f"<td class='pnl-loss'>{st_all['worst_pct']:+.2f}%</td>"
            f"<td class='pnl-win'><b>+{st_all['usdt_100x3_additive']:.1f}U</b></td>"
            f"<td><b>{st_all['seq_final_usdt']:.1f}U</b> (DD {st_all['seq_max_dd_pct']:.1f}%)</td>"
            f"<td>{st_early['n']}筆(sl={st_early['reasons'].get('stop',0)}) · {st_early['sum_pct']:+.2f}%</td>"
            f"<td>{st_mid['n']}筆(sl={st_mid['reasons'].get('stop',0)}) · {st_mid['sum_pct']:+.2f}%</td>"
            f"<td>{st_late['n']}筆(sl={st_late['reasons'].get('stop',0)}) · {st_late['sum_pct']:+.2f}%</td></tr>"
        )

    month_tr = []
    for ym in ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]:
        s7, s8, s9 = m_v7[ym], m_v8[ym], m_v9[ym]
        c7 = "pnl-win" if s7["sum_pct"] >= 0 else "pnl-loss"
        c8 = "pnl-win" if s8["sum_pct"] >= 0 else "pnl-loss"
        c9 = "pnl-win" if s9["sum_pct"] >= 0 else "pnl-loss"
        month_tr.append(
            f"<tr><td><b>{ym}</b></td>"
            f"<td>{s7['n']}筆 (停損 {s7['reasons'].get('stop',0)})</td>"
            f"<td>{s7['winrate_pct']:.1f}%</td>"
            f"<td class='{c7}'><b>{s7['sum_pct']:+.2f}%</b></td>"
            f"<td>{s8['n']}筆 (停損 {s8['reasons'].get('stop',0)})</td>"
            f"<td>{s8['winrate_pct']:.1f}%</td>"
            f"<td class='{c8}'><b>{s8['sum_pct']:+.2f}%</b></td>"
            f"<td>{s9['n']}筆 (停損 {s9['reasons'].get('stop',0)})</td>"
            f"<td><b>{s9['winrate_pct']:.1f}%</b></td>"
            f"<td class='{c9}'><b>{s9['sum_pct']:+.2f}%</b></td></tr>"
        )

    trade_tr = []
    for i, h7 in enumerate(v7_hits, 1):
        t7 = h7.trade
        s7 = t7.signal
        key = (h7.symbol, h7.df.index[t7.entry_idx])
        et = h7.df.index[t7.entry_idx].strftime("%Y-%m-%d %H:%M")
        xt7 = h7.df.index[t7.exit_idx].strftime("%m-%d %H:%M")
        close = h7.df["Close"].to_numpy(float)
        ma7 = m.sma(close, 7)[t7.entry_idx]
        ma14 = m.sma(close, 14)[t7.entry_idx]
        risk = t7.entry_price - t7.stop_price
        risk_pct = (risk / t7.entry_price * 100.0) if t7.entry_price else 0.0
        arm_15r = t7.entry_price + 1.5 * risk

        if key in v8_map:
            _, h8 = v8_map[key]
            t8 = h8.trade
            c8 = "pnl-win" if t8.pnl_pct > 0 else "pnl-loss"
            v8_cell = f"<span class='{c8}'>{t8.pnl_pct:+.2f}%</span> ({t8.exit_reason})"
        else:
            v8_cell = "<span class='muted'>v8剔除</span>"

        if key in v9_map:
            idx9, h9 = v9_map[key]
            t9 = h9.trade
            xt9 = h9.df.index[t9.exit_idx].strftime("%Y-%m-%d %H:%M")
            if getattr(t9, "exit_ts_5m", None) is not None:
                xt9_5m = t9.exit_ts_5m.tz_convert(TPE).strftime("%m-%d %H:%M")
                xt9_disp = f"{xt9} (5m:{xt9_5m})"
            else:
                xt9_disp = xt9
            c9 = "pnl-win" if t9.pnl_pct > 0 else "pnl-loss"
            exit_pt_cell = f"<b>{t9.exit_price:.6g}</b><br/><span class='muted'>{xt9_disp} · {t9.exit_reason}</span>"
            v9_cell = f"<span class='{c9}'><b>{t9.pnl_pct:+.2f}%</b></span><br/><a href='view.html#t{idx9}' style='font-size:11px'>看K線圖 #{idx9}</a>"
            status_badge = f"<span class='tag tag-tp'>v9 #{idx9}</span>"
        else:
            why = []
            if s7.vol_ratio < 0.15:
                why.append(f"進場量 {s7.vol_ratio:.2f}x&lt;0.15x")
            if not (ma7 > ma14):
                why.append("MA7&le;MA14")
            if s7.break_vol_ratio < 2.2:
                why.append(f"破底量 {s7.break_vol_ratio:.2f}x&lt;2.2x")
            if s7.undercut_pct < 0.016:
                why.append(f"破前低 {s7.undercut_pct*100:.1f}%&lt;1.6%")
            exit_pt_cell = "<span class='muted'>—</span>"
            v9_cell = f"<span class='muted'>已過濾（{'、'.join(why)}）</span>"
            status_badge = "<span class='tag tag-sl'>v9 剔除</span>"

        c7 = "pnl-win" if t7.pnl_pct > 0 else "pnl-loss"
        trade_tr.append(
            f"<tr><td>{i}</td><td><b>{escape(h7.symbol)}</b><br/><span class='muted'>Q{t7.quality}·{s7.shape}</span></td>"
            f"<td><b>{t7.entry_price:.6g}</b><br/><span class='muted'>{et}</span></td>"
            f"<td>{t7.stop_price:.6g} (−{risk_pct:.2f}%)<br/><span class='muted'>1.5R啟動: {arm_15r:.6g}</span></td>"
            f"<td>破底量 {s7.break_vol_ratio:.1f}x · 破前低 {s7.undercut_pct*100:.1f}%<br/><span class='muted'>進場量 {s7.vol_ratio:.2f}x · MA7&gt;14:{'是' if ma7>ma14 else '否'}</span></td>"
            f"<td class='{c7}'>{t7.pnl_pct:+.2f}% ({t7.exit_reason})<br/><span class='muted'>出 {t7.exit_price:.6g} @ {xt7}</span></td>"
            f"<td>{v8_cell}</td>"
            f"<td>{status_badge}</td>"
            f"<td>{exit_pt_cell}</td>"
            f"<td>{v9_cell}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>破底翻 1H MA25 全歷史極限回測與優化報告（2026-01-28 ～ 2026-10-31）</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;line-height:1.55}}
.page{{max-width:1220px;margin:0 auto;padding:18px 14px 48px}}
h1{{font-size:21px;margin:0 0 8px}}
h2{{font-size:17px;margin:22px 0 8px;color:#f0f6fc}}
.muted{{color:#8b949e;font-size:13px}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
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
<h1>幣安股票永續 1H MA25 破底翻 · 全歷史極限回測（2026-01-28 ～ 2026-10-31 台北時間，共 276 天）</h1>
<p class="muted"><b>最多能回測多久？</b> 幣安 USDT-M 股票永續合約（<code>TRADIFI_PERPETUAL</code>）最早自 <b>2026-01-28 22:00 TPE</b>（<code>TSLAUSDT</code>）起上線交易，隨後逐月擴充標的池（1月 1 檔、2月 +7 檔、3月 +6 檔、4月 +9 檔、5月 +27 檔、6月 +49 檔、7月 +36 檔、8月 +29 檔、9月 +30 檔、10月 +22 檔，排除 5 檔反向 ETF 後共 <b>216 檔</b>）。由於策略需先累積滿 200 根 1H K 線計算 <code>MA200</code>，全市場第一筆訊號出現在 2026-02-20。本報告完整抓取全部 216 檔自掛牌第一根 K 線起算的 <b>586,737 根 1H K 線</b>，達到幣安歷史資料的 100% 極限深度。<br/>
相關連結：<a href="view.html">v9 全歷史 52 筆交易 K 線卡片報告（含進出場圓圈標記）</a> · <a href="seq-view.html">v9 全歷史 100U × 3倍 一次一單複利報告（100U → 674.6U）</a></p>
</section>

<section class="box">
<h2>一、全歷史（2026-01-28 ～ 2026-10-31，約 9 個月）消融與三階段穩健性對照表</h2>
<table>
<thead>
<tr><th>版本 / 設定</th><th>全期筆數</th><th>破底停損</th><th>全期勝率</th><th>全期總報酬</th><th>平均單筆</th><th>獲利因子(PF)</th><th>最差單筆</th><th>100U×3單利</th><th>100U×3複利(MaxDD)</th><th>早期 (2～4月)</th><th>中期 (5～8月)</th><th>近期 (9～10月)</th></tr>
</thead>
<tbody>
{''.join(abl_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>二、全歷史逐月績效拆解（2026 年 2 月 ～ 10 月，共 9 個月）</h2>
<table>
<thead>
<tr><th>月份</th><th>v7 筆數 (停損)</th><th>v7 勝率</th><th>v7 單月報酬</th><th>v8 筆數 (停損)</th><th>v8 勝率</th><th>v8 單月報酬</th><th>v9 筆數 (停損)</th><th>v9 勝率</th><th>v9 單月報酬</th></tr>
</thead>
<tbody>
{''.join(month_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>三、全歷史全部 100 筆訊號逐筆對照（v7 vs v8 vs v9 · 含完整進出場點與 K 線圖連結）</h2>
<table>
<thead>
<tr><th>#</th><th>標的 / 型態</th><th>進場點 (價格 / TPE)</th><th>停損價 / 1.5R啟動價</th><th>破底量·破前低 / 進場量·MA7&gt;14</th><th>v7 結果 (36h)</th><th>v8 結果 (24h)</th><th>v9 狀態</th><th>v9 出場點 (價格 / TPE / 原因)</th><th>v9 報酬</th></tr>
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
    print(f"Scanning {len(syms)} stock perps | full history >= {WIN_START} TPE", flush=True)

    dfs_1h: dict[str, pd.DataFrame] = {}
    for s in syms:
        p = CACHE_DIR / f"{s}_1h.pkl"
        if p.exists():
            df = pd.read_pickle(p)
            if df is not None and len(df) >= 40:
                dfs_1h[s] = df
    if len(dfs_1h) < len(syms) - 10:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(m.fetch_klines, s, "1h", 300): s for s in syms if s not in dfs_1h}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    df = fut.result()
                    if df is not None and len(df) >= 40:
                        dfs_1h[sym] = df
                except Exception as exc:
                    print(f"WARN 1h {sym}: {exc}", flush=True)

    dfs_5m: dict[str, pd.DataFrame] = {}
    for p in sorted(CACHE_DIR.glob("*_5m.pkl")):
        dfs_5m[p.name.replace("_5m.pkl", "")] = pd.read_pickle(p)

    cutoff_early = datetime(2026, 5, 1, 0, 0, tzinfo=TPE)
    cutoff_late = datetime(2026, 9, 1, 0, 0, tzinfo=TPE)

    funnel_v7: dict[str, int] = {}
    funnel_v8: dict[str, int] = {}
    funnel_v9: dict[str, int] = {}
    for sym, df in sorted(dfs_1h.items()):
        m.detect_signals(df, funnel=funnel_v7, **BASE_V7_KW)
        m.detect_signals(df, funnel=funnel_v8, **OPT_V8_KW)
        m.detect_signals(df, funnel=funnel_v9, **OPT_V9_KW)

    def run_sim(kw_detect: dict, max_hold: int) -> tuple[list[m.Hit], list[m.Hit], list[m.Hit], list[m.Hit]]:
        h_all, h_early, h_mid, h_late = [], [], [], []
        for sym, df in sorted(dfs_1h.items()):
            sigs = m.detect_signals(df, **kw_detect)
            if not sigs:
                continue
            trades = m.simulate(df, sigs, df_5m=dfs_5m.get(sym), m5_exit=True, max_hold=max_hold, lose_ma_closes=0)
            for t in trades:
                et = df.index[t.entry_idx]
                h = m.Hit(sym, df, t)
                h_all.append(h)
                if et < cutoff_early:
                    h_early.append(h)
                elif et < cutoff_late:
                    h_mid.append(h)
                else:
                    h_late.append(h)
        for hs in (h_all, h_early, h_mid, h_late):
            hs.sort(key=lambda h: (h.df.index[h.trade.entry_idx], h.symbol))
        return h_all, h_early, h_mid, h_late

    hits_v7, v7_e, v7_m, v7_l = run_sim(BASE_V7_KW, 36)
    hits_v7_24, v7_24_e, v7_24_m, v7_24_l = run_sim(BASE_V7_KW, 24)
    hits_v8, v8_e, v8_m, v8_l = run_sim(OPT_V8_KW, 24)
    hits_v8_bvol, v8_bvol_e, v8_bvol_m, v8_bvol_l = run_sim(dict(OPT_V8_KW, min_break_vol=2.2), 24)
    hits_v9, v9_e, v9_m, v9_l = run_sim(OPT_V9_KW, 24)

    ablation = [
        ("v7 基準（36h 出場）", calc_metrics(hits_v7), calc_metrics(v7_e), calc_metrics(v7_m), calc_metrics(v7_l)),
        ("v7 + 縮短持有至 24h", calc_metrics(hits_v7_24), calc_metrics(v7_24_e), calc_metrics(v7_24_m), calc_metrics(v7_24_l)),
        ("v8 兩月版（進場量≥0.15x + MA7>14 + 24h）", calc_metrics(hits_v8), calc_metrics(v8_e), calc_metrics(v8_m), calc_metrics(v8_l)),
        ("v8 + 破底量 ≥ 2.2x（24h）", calc_metrics(hits_v8_bvol), calc_metrics(v8_bvol_e), calc_metrics(v8_bvol_m), calc_metrics(v8_bvol_l)),
        ("v9 全歷史優化版（v8 + 破底量≥2.2x + 破前低≥1.6% + 24h）", calc_metrics(hits_v9), calc_metrics(v9_e), calc_metrics(v9_m), calc_metrics(v9_l)),
    ]

    for lbl, sta, ste, stm, stl in ablation:
        print(
            f"{lbl:<52} | All: n={sta['n']:3d}(sl={sta['reasons'].get('stop',0):2d}) "
            f"WR={sta['winrate_pct']:5.1f}% sum={sta['sum_pct']:+7.2f}% PF={sta['profit_factor']:6.2f} "
            f"seq={sta['seq_final_usdt']:6.1f}U(DD {sta['seq_max_dd_pct']:4.1f}%) | "
            f"2-4M:{ste['sum_pct']:+6.2f}% | 5-8M:{stm['sum_pct']:+6.2f}% | 9-10M:{stl['sum_pct']:+6.2f}%",
            flush=True,
        )

    m_v7 = monthly_breakdown(hits_v7)
    m_v8 = monthly_breakdown(hits_v8)
    m_v9 = monthly_breakdown(hits_v9)

    stats_v9 = m.summarize_trades([h.trade for h in hits_v9])
    card_note = (
        "<br/><b>v9 全歷史優化條件（2026-01-28 上線以來 276 天）</b>：破底量 ≥ 2.2x 均量、破前低 ≥ 1.6%、進場量 ≥ 0.15x 均量"
        "、進場時 1H MA7 &gt; MA14、最長持有 24h。<a href='compare-view.html' style='color:#79c0ff'>查看全歷史 v7 vs v8 vs v9 完整對照表</a>"
    )
    m.write_html_report(
        html_p,
        hits_v9,
        days=276,
        scanned=len(syms),
        funnel=funnel_v9,
        recent_hours=48,
        all_stats=stats_v9,
        card_note=card_note,
        strict=False,
        m5_exit=True,
        lose_ma_closes=0,
        max_hold=24,
    )
    m.write_view_html(html_p, branch=BRANCH, out_name="view.html")

    taken_v9 = m.one_at_a_time_path(hits_v9, start=100.0, lev=3.0)
    m.draw_seq_pngs(taken_v9, OUT / "seq-img")
    m.write_seq_html(
        OUT / "seq.html",
        taken_v9,
        days=276,
        scanned=len(syms),
        pool=len(hits_v9),
        featured_html=m._featured_k_html(taken_v9),
        more_k_href="seq-k.html",
    )
    m.write_seq_k_html(OUT / "seq-k.html", taken_v9, days=276)
    m.write_view_html(OUT / "seq.html", branch=BRANCH, out_name="seq-view.html")
    m.write_view_html(OUT / "seq-k.html", branch=BRANCH, out_name="seq-k-view.html")

    stamp.REASON_MAP["timeout"] = "抱24h"
    text = html_p.read_text(encoding="utf-8")
    trades_parsed = stamp.parse_trades(text)
    stamped_paths = []
    for t in trades_parsed:
        src = img_dir / t["fname"]
        if src.is_file():
            dst = stamped_dir / t["fname"]
            stamp.stamp(src, dst, t["display"], t["reason"], t["cls"])
            src.write_bytes(dst.read_bytes())
            stamped_paths.append(str(dst))

    write_csv(OUT / "trades_v7.csv", hits_v7)
    write_csv(OUT / "trades_v8.csv", hits_v8)
    write_csv(OUT / "trades_v9.csv", hits_v9)
    write_compare_html(OUT / "compare.html", ablation, m_v7, m_v8, m_v9, hits_v7, hits_v8, hits_v9)
    m.write_view_html(OUT / "compare.html", branch=BRANCH, out_name="compare-view.html")

    summary = {
        "window_tpe": "2026-01-28 22:00 .. 2026-10-31 12:00 TPE (Full History since TRADIFI_PERPETUAL inception: 276 days / 9 months)",
        "n_symbols": len(syms),
        "total_1h_bars": int(sum(len(df) for df in dfs_1h.values())),
        "baseline_v7": ablation[0][1],
        "two_month_v8": ablation[2][1],
        "full_history_v9": ablation[4][1],
        "monthly_v7": m_v7,
        "monthly_v8": m_v8,
        "monthly_v9": m_v9,
        "funnel_v7": funnel_v7,
        "funnel_v8": funnel_v8,
        "funnel_v9": funnel_v9,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done! Full-history reports written to {OUT} (stamped {len(stamped_paths)} images)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
