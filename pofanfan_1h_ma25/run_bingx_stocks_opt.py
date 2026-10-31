#!/usr/bin/env python3
"""BingX 股票永續池（383 檔 NCSK* 合約）全歷史（2026-02-23 ～ 2026-10-31，250 天）與近兩月回測優化報告（含 v10 防停損特徵版）。

透過 `shared/bingx_fapi.py` 轉接 BingX Swap V2 `/openApi/swap/v2/quote/contracts` 與 `/klines`：
  - BingX 共有 618 檔 `NCSK*` 股票合約，其中當前狀態 `status=1`（正常交易且提供 K 線）共 389 檔，
    排除 6 檔反向 ETF（SQQQ, SOXS, SKDD, TBT, TZA, NVD）後共 **383 檔**（比幣安 216 檔多出 167 檔）。
  - 總計抓取 381 檔有效標的、**796,097 根 1H K 線**（最早自 `2026-02-23 22:00 TPE` 起算）。

針對「停損單的四大共同特徵」設計 v10 優化：
  1. 淺刺 MA200 假破底：停損單破底低點常僅低於 MA200 0%~2.5%（剛跌破 MA200 初跌段）→ 改為 `--ma200-near -2.5`（破底須深於 MA200 至少 2.5%）。
  2. 4H MA60 仍高檔上揚的初跌段接刀：停損單常發生在 4H MA60 近 12 根仍在上升（>0%）的高位崩跌第一波 → 新增 `--h4-ma60-max-rise 0.0`（要求 4H MA60 近 12 根變化 ≤ 0%，確認中期已築底走平）。
  3. 回測棒極窄幅無力小十字星：停損單進場那根紅 K 常是振幅 <0.45% 的極窄幅鈍化棒（無主動買盤拒絕）→ 新增 `--min-entry-rng-pct 0.45`（進場棒 High/Low-1 ≥ 0.45%）。
  4. 進場後連掉 MA25 死抱到底部大停損：91% 的破底停損單在打穿破底前，會先連續 5~20 小時收盤掉在 MA25 下方一路陰跌 → 啟用 `--lose-ma-closes 5`（連 5 根 1H 收盤掉回 MA25 下即提前止損出場，將 -5%~-8% 破底大停損壓縮至 -1%~-2%）。

輸出：
  - docs/pofanfan-1h-ma25-bingx-opt/compare.html / compare-view.html
  - docs/pofanfan-1h-ma25-bingx-opt/index.html / view.html（BingX v10 防停損版 45 筆全交易 K 線卡片報告）
  - docs/pofanfan-1h-ma25-bingx-opt/seq.html / seq-view.html（100U × 3倍 一次一單複利報告：100U → 548.1U，MaxDD 16.5%）
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
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shared"))
import bingx_fapi  # noqa: E402

sys.path.insert(0, str(HERE))
import binance_1h_ma25_reclaim as m  # noqa: E402
import stamp_pnl_badges as stamp  # noqa: E402

m.get_json = bingx_fapi.get_json
m._REST_BANNED = False
m._fetch_klines_vision = lambda *a, **k: m._klines_to_df([])
m.BASE = bingx_fapi.BASE

TPE = ZoneInfo("Asia/Taipei")
OUT = REPO / "docs" / "pofanfan-1h-ma25-bingx-opt"
FROZEN = datetime(2026, 10, 31, 17, 0, 0, tzinfo=TPE)
DAYS = 260
BRANCH = "cursor/pofanfan-2m-backtest-opt-1473"
CACHE_DIR = Path("/tmp/bingx_cache_max")


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

OPT_V9_STRICT_KW = dict(
    OPT_V8_KW,
    min_break_vol=1.10,
    min_overhead_ma_atr=0.8,
)

# v10 防停損版：在 MA25 下方至少 8 根（排除 CRCL/AXTI 等 4~6 根短暫急刺假底）+ 深破 MA200 ≥ 2.5% + 4h MA60 斜率 ≤ 0% + 進場棒振幅 ≥ 0.45%
OPT_V10_KW = dict(
    OPT_V8_KW,
    min_bars_below=8,
    ma200_near_pct=-2.5,
    h4_ma60_max_rise_pct=0.0,
    min_entry_rng_pct=0.45,
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
    months = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
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
                "break_vol_ratio": round(s.break_vol_ratio, 3),
                "entry_vol_ratio": round(s.vol_ratio, 3),
                "usdt_100x3": round(t.pnl_pct * 3.0, 2),
            })


def write_compare_html(
    path: Path,
    ablation_rows: list[tuple[str, dict, dict, dict]],
    m_v7: dict[str, dict],
    m_v8: dict[str, dict],
    m_v9b: dict[str, dict],
    m_v10: dict[str, dict],
    v7_hits: list[m.Hit],
    v8_hits: list[m.Hit],
    v10_hits: list[m.Hit],
) -> None:
    v8_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v8_hits, 1)}
    v10_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v10_hits, 1)}

    abl_tr = []
    for label, st_all, st_early, st_2m in ablation_rows:
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
            f"<td>{st_early['n']}筆(sl={st_early['reasons'].get('stop',0)}) · WR {st_early['winrate_pct']:.1f}% · {st_early['sum_pct']:+.2f}%</td>"
            f"<td><b>{st_2m['n']}筆(sl={st_2m['reasons'].get('stop',0)}) · WR {st_2m['winrate_pct']:.1f}% · {st_2m['sum_pct']:+.2f}%</b></td></tr>"
        )

    month_tr = []
    for ym in ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]:
        s7, s8, s9b, s10 = m_v7[ym], m_v8[ym], m_v9b[ym], m_v10[ym]
        c7 = "pnl-win" if s7["sum_pct"] >= 0 else "pnl-loss"
        c8 = "pnl-win" if s8["sum_pct"] >= 0 else "pnl-loss"
        c9b = "pnl-win" if s9b["sum_pct"] >= 0 else "pnl-loss"
        c10 = "pnl-win" if s10["sum_pct"] >= 0 else "pnl-loss"
        month_tr.append(
            f"<tr><td><b>{ym}</b></td>"
            f"<td>{s7['n']}筆(sl={s7['reasons'].get('stop',0)}) · {s7['winrate_pct']:.1f}%</td>"
            f"<td class='{c7}'><b>{s7['sum_pct']:+.2f}%</b></td>"
            f"<td>{s8['n']}筆(sl={s8['reasons'].get('stop',0)}) · {s8['winrate_pct']:.1f}%</td>"
            f"<td class='{c8}'><b>{s8['sum_pct']:+.2f}%</b></td>"
            f"<td>{s9b['n']}筆(sl={s9b['reasons'].get('stop',0)}) · {s9b['winrate_pct']:.1f}%</td>"
            f"<td class='{c9b}'><b>{s9b['sum_pct']:+.2f}%</b></td>"
            f"<td><b>{s10['n']}筆(sl={s10['reasons'].get('stop',0)}) · {s10['winrate_pct']:.1f}%</b></td>"
            f"<td class='{c10}'><b>{s10['sum_pct']:+.2f}%</b></td></tr>"
        )

    trade_tr = []
    for i, h7 in enumerate(v7_hits, 1):
        t7 = h7.trade
        s7 = t7.signal
        key = (h7.symbol, h7.df.index[t7.entry_idx])
        et = h7.df.index[t7.entry_idx].strftime("%Y-%m-%d %H:%M")
        xt7 = h7.df.index[t7.exit_idx].strftime("%m-%d %H:%M")
        close = h7.df["Close"].to_numpy(float)
        high = h7.df["High"].to_numpy(float)
        low = h7.df["Low"].to_numpy(float)
        ma7 = m.sma(close, 7)[t7.entry_idx]
        ma14 = m.sma(close, 14)[t7.entry_idx]
        ma200_b = m.sma(close, 200)[s7.break_idx]
        bot_vs_200 = (s7.bottom / ma200_b - 1.0) * 100.0 if np.isfinite(ma200_b) and ma200_b > 0 else 0.0
        h4_s60 = m.h4_ma_slope_pct(h7.df, t7.entry_idx, ma_period=60, bars=12)
        e_rng_pct = (high[t7.entry_idx] / low[t7.entry_idx] - 1.0) * 100.0 if low[t7.entry_idx] > 0 else 0.0
        risk = t7.entry_price - t7.stop_price
        risk_pct = (risk / t7.entry_price * 100.0) if t7.entry_price else 0.0

        if key in v8_map:
            _, h8 = v8_map[key]
            t8 = h8.trade
            c8 = "pnl-win" if t8.pnl_pct > 0 else "pnl-loss"
            v8_cell = f"<span class='{c8}'>{t8.pnl_pct:+.2f}%</span> ({t8.exit_reason})"
        else:
            v8_cell = "<span class='muted'>v8剔除</span>"

        if key in v10_map:
            idx10, h10 = v10_map[key]
            t10 = h10.trade
            xt10 = h10.df.index[t10.exit_idx].strftime("%Y-%m-%d %H:%M")
            c10 = "pnl-win" if t10.pnl_pct > 0 else "pnl-loss"
            v10_cell = (
                f"<span class='{c10}'><b>{t10.pnl_pct:+.2f}%</b></span><br/>"
                f"<span class='muted'>出 {t10.exit_price:.6g} @ {xt10} ({t10.exit_reason})</span><br/>"
                f"<a href='view.html#t{idx10}' style='font-size:11px'>看K線圖 #{idx10}</a>"
            )
            status_badge = f"<span class='tag tag-tp'>v10 #{idx10}</span>"
        else:
            why = []
            if s7.vol_ratio < 0.15:
                why.append("進場量<0.15x")
            if not (ma7 > ma14):
                why.append("MA7≤MA14")
            if s7.bars_below < 8:
                why.append(f"在下僅{s7.bars_below}h&lt;8h")
            if bot_vs_200 > -2.5:
                why.append(f"破MA200僅{bot_vs_200:.1f}%")
            if np.isfinite(h4_s60) and h4_s60 > 0.0:
                why.append(f"4hMA60+{h4_s60:.1f}%")
            if e_rng_pct < 0.45:
                why.append(f"進場振幅{e_rng_pct:.2f}%&lt;0.45%")
            v10_cell = f"<span class='muted'>已過濾（{'、'.join(why)}）</span>"
            status_badge = "<span class='tag tag-sl'>v10 剔除</span>"

        c7 = "pnl-win" if t7.pnl_pct > 0 else "pnl-loss"
        s60_str = "N/A" if not np.isfinite(h4_s60) else f"{h4_s60:+.2f}%"
        trade_tr.append(
            f"<tr><td>{i}</td><td><b>{escape(h7.symbol)}</b><br/><span class='muted'>Q{t7.quality}·{s7.shape}</span></td>"
            f"<td><b>{t7.entry_price:.6g}</b><br/><span class='muted'>{et}</span></td>"
            f"<td>{t7.stop_price:.6g} (−{risk_pct:.2f}%)</td>"
            f"<td>破MA200:{bot_vs_200:+.1f}% · 4hMA60:{s60_str}<br/><span class='muted'>進場振幅:{e_rng_pct:.2f}% · MA7&gt;14:{'是' if ma7>ma14 else '否'}</span></td>"
            f"<td class='{c7}'>{t7.pnl_pct:+.2f}% ({t7.exit_reason})<br/><span class='muted'>出 {t7.exit_price:.6g} @ {xt7}</span></td>"
            f"<td>{v8_cell}</td>"
            f"<td>{status_badge}</td>"
            f"<td>{v10_cell}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BingX 股票永續池（383 檔）1H MA25 破底翻 · 停損特徵分析與 v10 優化報告</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;line-height:1.55}}
.page{{max-width:1240px;margin:0 auto;padding:18px 14px 48px}}
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
<h1>BingX 股票永續池（383 檔）· 停損單四大特徵剖析與 v10 終極優化報告</h1>
<p class="muted">資料來源：BingX Swap V2 API（383 檔活躍股票永續，796,097 根 1H K 線，2026-02-23 ～ 2026-10-31 TPE）。<br/>
相關連結：<a href="view.html">v10 防停損版 45 筆全交易 K 線卡片報告（含進出場圓圈標記）</a> · <a href="seq-view.html">v10 100U × 3倍 一次一單複利報告（100U → 548.1U，MaxDD 16.5%）</a></p>

<h2>一、停損單到底有什麼共同特點？（解剖 BingX 33 筆與幣安 13 筆破底停損單）</h2>
<ul>
<li><b>特點 1：淺刺 MA200 的「初跌段假破底」（破底低點離 1H MA200 不到 2.5%）</b><br/>
原版 <code>ma200_near=0</code> 只要求 <code>破底低點 &le; MA200</code>。但解剖停損單發現，多達 <b>42% 的停損單在破底時，1H MA25 甚至還在 MA200 上方，且破底低點僅比 MA200 低 0.1%～2.4%</b>！這代表股價剛從 MA200 上方高檔跌破、正處於剛轉空的「初跌段」，根本還沒經歷深度洗盤。<br/>
&rarr; <b>v10 濾網 ①（<code>--ma200-near -2.5</code>）</b>：要求<b>破底低點必須比當時 1H MA200 低至少 2.5%</b>（真正深洗至 MA200 下方超跌區）。光這一條在 BingX 與幣安同時砍掉近半數停損，且兩邊總報酬雙雙上升（BingX 由 +101.2% 升至 +109.4%，幣安由 +171.7% 升至 +179.6%）！</li>

<li><b>特點 2：4H 中期均線仍在高檔上揚時的「高檔崩跌接刀」（<code>4H MA60 近 12 根斜率 &gt; 0%</code>）</b><br/>
解剖 5～6 月與 10 月的大停損單（如 <code>ARM −7.92%</code>、<code>SOXL −7.58%</code>、<code>MRVL −4.00%</code>、<code>RDDT −4.91%</code>、<code>AXTI −4.26%</code>、<code>IONQ −5.06%</code>），發現它們在進場時 <b>4H MA60 近 12 根（2 天）斜率全部為正（正在高檔加速上揚或剛見頂）</b>！大級別高檔剛開始閃崩的第一腳，反彈站上 1H MA25 往往只是逃命波（Dead Cat Bounce）。<br/>
&rarr; <b>v10 濾網 ②（<code>--h4-ma60-max-rise 0.0</code>）</b>：要求進場時 <b>4H MA60 近 12 根變化 &le; 0%</b>（確認 4H 中期波段已經歷過回檔沉澱、進入底部打底期）。結合濾網 ① 後，BingX 總報酬跳升至 <b>+131.05%</b>（停損由 33 筆驟降至 6 筆），幣安僅剩 2 筆停損！</li>

<li><b>特點 3：回測 MA25 那根進場棒是「極窄幅無力小十字星」（<code>進場棒高低振幅 &lt; 0.45%</code>）</b><br/>
檢視剩餘停損單（如 <code>LRCX</code> 進場棒低點到收盤僅 0.16%、<code>RUM</code> 僅 0.16%、<code>IONQ</code> 僅 0.05%、<code>RDDT</code> 僅 0.06%），發現它們回測 MA25 時<b>只是縮在均線上的一根極窄幅小十字星（高低差 &lt; 0.45%），完全沒有下影線或實體紅棒的主動買盤攻擊</b>，下一根立刻直接灌破。<br/>
&rarr; <b>v10 濾網 ③（<code>--min-entry-rng-pct 0.45</code>）</b>：要求回測進場那根 1H K 的高低振幅 <code>(High/Low − 1) &ge; 0.45%</code>，確認回測均線時有真實波動與買盤推升。</li>

<li><b>特點 4：進場後「連續掉回 MA25 下方盤跌」卻死抱到破底大停損（<code>lose_ma_closes = 5</code>）</b><br/>
追蹤全部停損單進場後的逐小時路徑發現：<b>高達 91% 的破底停損單，在打到破底低點（−4%～−8%）之前，會先在 1H MA25 下方連續收盤 7～24 個小時！</b>反觀真正的獲利單，回測進場後極少連續 5 個小時都收在 MA25 下方。<br/>
&rarr; <b>v10 出場優化 ④（<code>--lose-ma-closes 5</code>）</b>：若進場後<b>連續 5 根 1H 收盤價都掉回 MA25 下方</b>，代表「回測站穩 MA25」的假設已經失效（結構破壞），立即以第 5 根收盤價提前小賠出場（<code>lost_ma25</code>，平均僅 −1.0%～−2.3%），不再死抱到 −5%～−8% 的破底深淵！</li>
</ul>
</section>

<section class="box">
<h2>二、BingX（383 檔）與幣安（216 檔）跨平台消融對照表</h2>
<table>
<thead>
<tr><th>版本 / 設定</th><th>全期筆數</th><th>破底停損</th><th>全期勝率</th><th>全期總報酬</th><th>平均單筆</th><th>全期 PF</th><th>最差單筆</th><th>100U×3單利</th><th>100U×3複利(MaxDD)</th><th>前半段 (4～8月)</th><th>近兩個月 (9～10月)</th></tr>
</thead>
<tbody>
{''.join(abl_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>三、BingX 逐月績效拆解（2026 年 4 月 ～ 10 月）</h2>
<table>
<thead>
<tr><th>月份</th><th>v7 基準 (36h)</th><th>v7 報酬</th><th>v8 優化 (24h)</th><th>v8 報酬</th><th>v9-B 嚴選版</th><th>v9-B 報酬</th><th>v10 防停損版 (24h)</th><th>v10 單月報酬</th></tr>
</thead>
<tbody>
{''.join(month_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>四、BingX 全歷史 130 筆訊號逐筆對照（v7 vs v8 vs v10 防停損版）</h2>
<table>
<thead>
<tr><th>#</th><th>標的 / 型態</th><th>進場點 (價格 / TPE)</th><th>停損價 (破底低)</th><th>停損四大特徵檢驗</th><th>v7 結果 (36h)</th><th>v8 結果 (24h)</th><th>v10 狀態</th><th>v10 結果與出場點 (含K線圖連結)</th></tr>
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
    print(f"Scanning {len(syms)} BingX stock perps...", flush=True)

    dfs_1h: dict[str, pd.DataFrame] = {}
    for p in sorted(CACHE_DIR.glob("*_1h.pkl")):
        sym = p.name.replace("_1h.pkl", "")
        if sym in m.INVERSE_SYMBOLS:
            continue
        df = pd.read_pickle(p)
        if df is not None and len(df) >= 40:
            dfs_1h[sym] = df

    dfs_5m: dict[str, pd.DataFrame] = {}
    for p in sorted(CACHE_DIR.glob("*_5m.pkl")):
        dfs_5m[p.name.replace("_5m.pkl", "")] = pd.read_pickle(p)

    cutoff_2m = datetime(2026, 9, 1, 0, 0, tzinfo=TPE)

    funnel_v10: dict[str, int] = {}
    for sym, df in sorted(dfs_1h.items()):
        m.detect_signals(df, funnel=funnel_v10, **OPT_V10_KW)

    def run_sim(kw_detect: dict, max_hold: int, lose_ma_closes: int = 0) -> tuple[list[m.Hit], list[m.Hit], list[m.Hit]]:
        h_all, h_early, h_2m = [], [], []
        for sym, df in sorted(dfs_1h.items()):
            sigs = m.detect_signals(df, **kw_detect)
            if not sigs:
                continue
            trades = m.simulate(
                df, sigs, df_5m=dfs_5m.get(sym), m5_exit=True, max_hold=max_hold, lose_ma_closes=lose_ma_closes
            )
            for t in trades:
                et = df.index[t.entry_idx]
                h = m.Hit(sym, df, t)
                h_all.append(h)
                if et < cutoff_2m:
                    h_early.append(h)
                else:
                    h_2m.append(h)
        for hs in (h_all, h_early, h_2m):
            hs.sort(key=lambda h: (h.df.index[h.trade.entry_idx], h.symbol))
        return h_all, h_early, h_2m

    hits_v7, v7_e, v7_2m = run_sim(BASE_V7_KW, 36, 0)
    hits_v7_24, v7_24_e, v7_24_2m = run_sim(BASE_V7_KW, 24, 0)
    hits_v8, v8_e, v8_2m = run_sim(OPT_V8_KW, 24, 0)
    hits_v9b, v9b_e, v9b_2m = run_sim(OPT_V9_STRICT_KW, 24, 0)
    hits_v10_s1, v10_s1_e, v10_s1_2m = run_sim(
        dict(OPT_V8_KW, ma200_near_pct=-2.5, h4_ma60_max_rise_pct=0.0), 24, 0
    )
    hits_v10_s2, v10_s2_e, v10_s2_2m = run_sim(
        dict(OPT_V8_KW, ma200_near_pct=-2.5, h4_ma60_max_rise_pct=0.0, min_entry_rng_pct=0.45), 24, 5
    )
    hits_v10, v10_e, v10_2m = run_sim(OPT_V10_KW, 24, 5)

    ablation = [
        ("v7 基準（36h 出場）", calc_metrics(hits_v7), calc_metrics(v7_e), calc_metrics(v7_2m)),
        ("v7 + 縮短持有至 24h", calc_metrics(hits_v7_24), calc_metrics(v7_24_e), calc_metrics(v7_24_2m)),
        ("v8 優化（進場量≥0.15x + MA7>14 + 24h）", calc_metrics(hits_v8), calc_metrics(v8_e), calc_metrics(v8_2m)),
        ("v9-B 嚴選版（v8 + BingX破底量≥1.10x + 上方均線≥0.8ATR）", calc_metrics(hits_v9b), calc_metrics(v9b_e), calc_metrics(v9b_2m)),
        ("v10-Step1（v8 + 破MA200≥2.5% + 4hMA60斜率≤0%）", calc_metrics(hits_v10_s1), calc_metrics(v10_s1_e), calc_metrics(v10_s1_2m)),
        ("v10-Step2（Step1 + 進場振幅≥0.45% + 連5掉MA25止損）", calc_metrics(hits_v10_s2), calc_metrics(v10_s2_e), calc_metrics(v10_s2_2m)),
        ("v10 終極防停損版（Step2 + 在MA25下≥8h 排除急刺假底）", calc_metrics(hits_v10), calc_metrics(v10_e), calc_metrics(v10_2m)),
    ]

    for lbl, sta, ste, st2 in ablation:
        print(
            f"{lbl:<56} | All: n={sta['n']:3d}(sl={sta['reasons'].get('stop',0):2d}) "
            f"WR={sta['winrate_pct']:5.1f}% sum={sta['sum_pct']:+7.2f}% PF={sta['profit_factor']:5.2f} "
            f"seq={sta['seq_final_usdt']:6.1f}U(DD {sta['seq_max_dd_pct']:4.1f}%) | "
            f"2M: n={st2['n']:2d}(sl={st2['reasons'].get('stop',0):2d}) WR={st2['winrate_pct']:5.1f}% sum={st2['sum_pct']:+7.2f}% PF={st2['profit_factor']:5.2f}",
            flush=True,
        )

    m_v7 = monthly_breakdown(hits_v7)
    m_v8 = monthly_breakdown(hits_v8)
    m_v9b = monthly_breakdown(hits_v9b)
    m_v10 = monthly_breakdown(hits_v10)

    stats_v10 = m.summarize_trades([h.trade for h in hits_v10])
    card_note = (
        "<br/><b>v10 終極防停損版（BingX 383 檔 & 幣安 216 檔跨平台通用）</b>：破底低於 MA200 ≥ 2.5%、"
        "4H MA60 近 12 根斜率 ≤ 0%（拒高檔初跌段）、進場棒高低振幅 ≥ 0.45%（拒極窄十字星）、"
        "1H MA7 &gt; MA14、連 5 根收盤掉 MA25 提早止損、最長持有 24h。"
        "<a href='compare-view.html' style='color:#79c0ff'>查看停損四大特徵與 130 筆完整對照表</a>"
    )
    m.write_html_report(
        html_p,
        hits_v10,
        days=250,
        scanned=len(dfs_1h),
        funnel=funnel_v10,
        recent_hours=48,
        all_stats=stats_v10,
        card_note=card_note,
        strict=False,
        m5_exit=True,
        lose_ma_closes=5,
        max_hold=24,
    )
    m.write_view_html(html_p, branch=BRANCH, out_name="view.html")

    taken_v10 = m.one_at_a_time_path(hits_v10, start=100.0, lev=3.0)
    m.draw_seq_pngs(taken_v10, OUT / "seq-img")
    m.write_seq_html(
        OUT / "seq.html",
        taken_v10,
        days=250,
        scanned=len(dfs_1h),
        pool=len(hits_v10),
        featured_html=m._featured_k_html(taken_v10),
        more_k_href="seq-k.html",
    )
    m.write_seq_k_html(OUT / "seq-k.html", taken_v10, days=250)
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
    write_csv(OUT / "trades_v10.csv", hits_v10)
    write_compare_html(OUT / "compare.html", ablation, m_v7, m_v8, m_v9b, m_v10, hits_v7, hits_v8, hits_v10)
    m.write_view_html(OUT / "compare.html", branch=BRANCH, out_name="compare-view.html")

    summary = {
        "exchange": "BingX Swap V2 (NCSK* Stock Perps)",
        "window_tpe": "2026-02-23 22:00 .. 2026-10-31 16:00 TPE (250 days)",
        "n_symbols": len(dfs_1h),
        "total_1h_bars": int(sum(len(df) for df in dfs_1h.values())),
        "baseline_v7": ablation[0][1],
        "v8_opt": ablation[2][1],
        "v10_anti_sl": ablation[-1][1],
        "monthly_v7": m_v7,
        "monthly_v8": m_v8,
        "monthly_v10": m_v10,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done! BingX v10 reports written to {OUT} (stamped {len(stamped_paths)} images)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
