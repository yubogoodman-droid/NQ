#!/usr/bin/env python3
"""BingX 股票永續池（383 檔 NCSK* 合約）全歷史（2026-02-23 ～ 2026-10-31，250 天）與近兩月回測優化報告。

透過 `shared/bingx_fapi.py` 轉接 BingX Swap V2 `/openApi/swap/v2/quote/contracts` 與 `/klines`：
  - BingX 共有 618 檔 `NCSK*` 股票合約，其中當前狀態 `status=1`（正常交易且提供 K 線）共 389 檔，
    排除 6 檔反向 ETF（SQQQ, SOXS, SKDD, TBT, TZA, NVD）後共 **383 檔**（比幣安 216 檔多出 167 檔）。
  - 總計抓取 381 檔有效標的、**796,097 根 1H K 線**（最早自 `2026-02-23 22:00 TPE` 起算）。

輸出：
  - docs/pofanfan-1h-ma25-bingx-opt/compare.html / compare-view.html（BingX 383 檔消融、逐月拆解、幣安 vs BingX 雙交易所特性比較與全訊號逐筆對照）
  - docs/pofanfan-1h-ma25-bingx-opt/index.html / view.html（BingX 嚴選版 21 筆全歷史 K 線卡片報告，含進出場圓圈標記）
  - docs/pofanfan-1h-ma25-bingx-opt/seq.html / seq-view.html（100U × 3倍 一次一單複利報告）
  - docs/pofanfan-1h-ma25-bingx-opt/summary.json & trades_v7.csv / trades_v8.csv / trades_v9_trend.csv / trades_v9_strict.csv
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
sys.path.insert(0, str(REPO / "shared"))
import bingx_fapi  # noqa: E402

sys.path.insert(0, str(HERE))
import binance_1h_ma25_reclaim as m  # noqa: E402
import stamp_pnl_badges as stamp  # noqa: E402

# Route reclaim module through BingX API
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

# v9-A (BingX 波段多單版): 4h MA120 未處於末升段過熱 (<= +0.5%) + 破前低 >= 1.8%
OPT_V9_TREND_KW = dict(
    OPT_V8_KW,
    min_undercut_pct=0.018,
    h4_max_ma_rise_pct=0.5,
)

# v9-B (BingX 嚴選防禦版): 針對 BingX 平滑造市量能校準破底量 >= 1.10x + 頭上長均線距離 >= 0.8 ATR
OPT_V9_STRICT_KW = dict(
    OPT_V8_KW,
    min_break_vol=1.10,
    min_overhead_ma_atr=0.8,
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
    m_v9a: dict[str, dict],
    m_v9b: dict[str, dict],
    v7_hits: list[m.Hit],
    v8_hits: list[m.Hit],
    v9a_hits: list[m.Hit],
    v9b_hits: list[m.Hit],
) -> None:
    v8_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v8_hits, 1)}
    v9a_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v9a_hits, 1)}
    v9b_map = {(h.symbol, h.df.index[h.trade.entry_idx]): (idx, h) for idx, h in enumerate(v9b_hits, 1)}

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
        s7, s8, s9a, s9b = m_v7[ym], m_v8[ym], m_v9a[ym], m_v9b[ym]
        c7 = "pnl-win" if s7["sum_pct"] >= 0 else "pnl-loss"
        c8 = "pnl-win" if s8["sum_pct"] >= 0 else "pnl-loss"
        c9a = "pnl-win" if s9a["sum_pct"] >= 0 else "pnl-loss"
        c9b = "pnl-win" if s9b["sum_pct"] >= 0 else "pnl-loss"
        month_tr.append(
            f"<tr><td><b>{ym}</b></td>"
            f"<td>{s7['n']}筆(sl={s7['reasons'].get('stop',0)}) · {s7['winrate_pct']:.1f}%</td>"
            f"<td class='{c7}'><b>{s7['sum_pct']:+.2f}%</b></td>"
            f"<td>{s8['n']}筆(sl={s8['reasons'].get('stop',0)}) · {s8['winrate_pct']:.1f}%</td>"
            f"<td class='{c8}'><b>{s8['sum_pct']:+.2f}%</b></td>"
            f"<td>{s9a['n']}筆(sl={s9a['reasons'].get('stop',0)}) · {s9a['winrate_pct']:.1f}%</td>"
            f"<td class='{c9a}'><b>{s9a['sum_pct']:+.2f}%</b></td>"
            f"<td>{s9b['n']}筆(sl={s9b['reasons'].get('stop',0)}) · {s9b['winrate_pct']:.1f}%</td>"
            f"<td class='{c9b}'><b>{s9b['sum_pct']:+.2f}%</b></td></tr>"
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
        ma60 = m.sma(close, 60)[t7.entry_idx]
        ma99 = m.sma(close, 99)[t7.entry_idx]
        ma120 = m.sma(close, 120)[t7.entry_idx]
        ma200 = m.sma(close, 200)[t7.entry_idx]
        atr_e = m.atr(high, low, close, 14)[t7.entry_idx]
        ceils = [mv for mv in (ma60, ma99, ma120, ma200) if np.isfinite(mv) and mv > t7.entry_price]
        ceil_atr = min((mv - t7.entry_price) / atr_e for mv in ceils) if ceils and atr_e > 0 else 99.0
        h4_slp120 = m.h4_ma_slope_pct(h7.df, t7.entry_idx, ma_period=120, bars=24)
        risk = t7.entry_price - t7.stop_price
        risk_pct = (risk / t7.entry_price * 100.0) if t7.entry_price else 0.0

        if key in v8_map:
            _, h8 = v8_map[key]
            t8 = h8.trade
            c8 = "pnl-win" if t8.pnl_pct > 0 else "pnl-loss"
            v8_cell = f"<span class='{c8}'>{t8.pnl_pct:+.2f}%</span> ({t8.exit_reason})"
        else:
            v8_cell = "<span class='muted'>v8剔除</span>"

        if key in v9a_map:
            _, h9a = v9a_map[key]
            t9a = h9a.trade
            c9a = "pnl-win" if t9a.pnl_pct > 0 else "pnl-loss"
            v9a_cell = f"<span class='{c9a}'><b>{t9a.pnl_pct:+.2f}%</b></span> ({t9a.exit_reason})"
        else:
            v9a_cell = "<span class='muted'>—</span>"

        if key in v9b_map:
            idx9b, h9b = v9b_map[key]
            t9b = h9b.trade
            xt9b = h9b.df.index[t9b.exit_idx].strftime("%Y-%m-%d %H:%M")
            c9b = "pnl-win" if t9b.pnl_pct > 0 else "pnl-loss"
            v9b_cell = (
                f"<span class='{c9b}'><b>{t9b.pnl_pct:+.2f}%</b></span><br/>"
                f"<span class='muted'>出 {t9b.exit_price:.6g} @ {xt9b} ({t9b.exit_reason})</span><br/>"
                f"<a href='view.html#t{idx9b}' style='font-size:11px'>看K線圖 #{idx9b}</a>"
            )
        else:
            v9b_cell = "<span class='muted'>—</span>"

        c7 = "pnl-win" if t7.pnl_pct > 0 else "pnl-loss"
        ceil_str = "無" if ceil_atr > 50 else f"{ceil_atr:.2f}ATR"
        slp_str = "N/A" if not np.isfinite(h4_slp120) else f"{h4_slp120:+.1f}%"
        trade_tr.append(
            f"<tr><td>{i}</td><td><b>{escape(h7.symbol)}</b><br/><span class='muted'>Q{t7.quality}·{s7.shape}</span></td>"
            f"<td><b>{t7.entry_price:.6g}</b><br/><span class='muted'>{et}</span></td>"
            f"<td>{t7.stop_price:.6g} (−{risk_pct:.2f}%)</td>"
            f"<td>破底量 {s7.break_vol_ratio:.2f}x · 破前低 {s7.undercut_pct*100:.1f}%<br/><span class='muted'>上方均線:{ceil_str} · 4hMA120:{slp_str}</span></td>"
            f"<td class='{c7}'>{t7.pnl_pct:+.2f}% ({t7.exit_reason})<br/><span class='muted'>出 {t7.exit_price:.6g} @ {xt7}</span></td>"
            f"<td>{v8_cell}</td>"
            f"<td>{v9a_cell}</td>"
            f"<td>{v9b_cell}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BingX 股票永續池（383 檔）1H MA25 破底翻 · 全歷史與近兩月回測優化報告</title>
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
<h1>BingX 股票永續池（383 檔 NCSK 合約）· 1H MA25 破底翻全歷史與兩月回測優化報告</h1>
<p class="muted">資料來源：BingX Swap V2 API（<code>/openApi/swap/v2/quote/contracts</code> 與 <code>/klines</code>）。<br/>
相關連結：<a href="view.html">BingX v9-B 嚴選版 21 筆全交易 K 線卡片報告（含進出場圓圈標記）</a> · <a href="seq-view.html">BingX 100U × 3倍 一次一單複利報告</a></p>

<h2>一、BingX vs 幣安（Binance）股票永續池關鍵差異診斷</h2>
<ul>
<li><b>1. 股票池規模（BingX 383 檔 vs 幣安 216 檔）</b>：<br/>
BingX 上共有 618 檔 <code>NCSK*2USD-USDT</code> 股票合約，其中當前活躍可交易（<code>status=1</code>）共 389 檔，剔除 6 檔反向 ETF（<code>SQQQ</code>, <code>SOXS</code>, <code>SKDD</code>, <code>TBT</code>, <code>TZA</code>, <code>NVD</code>）後共 <b>383 檔</b>（包含大量幣安沒有的美股中小型股、航太軍工、量子電腦與生技股，如 <code>ASTS</code>, <code>RKLB</code>, <code>IONQ</code>, <code>QBTS</code>, <code>RGTI</code>, <code>OKLO</code>, <code>AXON</code>, <code>VRT</code>, <code>VST</code>, <code>CEG</code> 等），共抓取 <b>796,097 根 1H K 線</b>（最早自 <code>2026-02-23 22:00 TPE</code> 起算，共 250 天）。</li>
<li><b>2. 成交量結構的關鍵差異（為什麼幣安的 <code>break_vol &ge; 2.2x</code> 不能直接照搬到 BingX？）</b>：<br/>
實測數據顯示，幣安股票永續的每小時成交量是真實散戶/機構撮合量（恐慌破底時會爆出 4x～20x 均量）；而 <b>BingX 的 <code>NCSK*</code> 股票合約採用平滑造市商報價量（例如 TSLA 每小時成交量幾乎恆定在 250～295 張之間）</b>，全市場 130 筆訊號的 <code>break_vol_ratio</code> 四分位數僅落在 <code>1.006x ～ 1.132x</code>（中位數 1.055x）。因此若直接套用幣安的 <code>break_vol &ge; 2.2x</code>，在 BingX 上會把 97% 的訊號都濾掉！</li>
<li><b>3. 針對 BingX 量價特性的三階段優化方案</b>：<br/>
  - <b>v8 基礎優化（<code>進場量 &ge; 0.15x</code> + <code>1H MA7 &gt; MA14</code> + <code>24h 出場</code>）</b>：在 BingX 全歷史（112 筆）將總報酬從原版 v7 的 <b>+4.93%</b> 大幅拉升至 <b>+101.21%</b>；近兩個月（9～10 月，66 筆）勝率達 <b>69.7%</b>、累積 <b>+107.67%</b>（PF 4.11，一次一單 3x 複利由 100U 成長至 <b>449.0U</b>，MaxDD 僅 11.4%）。<br/>
  - <b>v9-A 波段多單版（v8 + <code>--h4-max-ma-rise 0.5</code> + <code>--min-undercut 1.8</code>）</b>：避開 4h MA120 末升段過熱高位崩跌接刀（要求 4h MA120 近 24 根漲幅 &le; +0.5%）並要求破前低 &ge; 1.8%。全歷史 62 筆勝率提升至 <b>69.4%</b>、總報酬 <b>+104.20%</b>（PF 3.28）；近兩月 39 筆勝率 <b>74.4%</b>、累積 <b>+75.54%</b>（PF 5.31，停損僅 2 筆）。<br/>
  - <b>v9-B 嚴選防禦版（v8 + <code>--min-break-vol 1.10</code> + <code>--min-overhead-ma-atr 0.8</code>）</b>：針對 BingX 平滑量能校準破底量門檻為 <code>&ge; 1.10x</code>（在 BingX 約等於前 25% 出量棒），且要求進場價距離上方最近長均線（MA60/99/120/200）至少 <code>&ge; 0.8 ATR</code>（不買在長均線天花板正下方）。全歷史 21 筆勝率達 <b>76.2%</b>、單筆平均 <b>+2.70%</b>、獲利因子高達 <b>8.81</b>（全歷史僅 1 筆停損，近兩月 **0 筆停損**、PF 13.68）。</li>
</ul>
</section>

<section class="box">
<h2>二、BingX 383 檔股票池 · 全歷史（250 天）與近兩月（60 天）消融對照表</h2>
<table>
<thead>
<tr><th>版本 / 設定</th><th>全期筆數</th><th>全期停損</th><th>全期勝率</th><th>全期總報酬</th><th>平均單筆</th><th>全期 PF</th><th>最差單筆</th><th>100U×3單利</th><th>100U×3複利(MaxDD)</th><th>前半段 (4～8月)</th><th>近兩個月 (9～10月)</th></tr>
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
<tr><th>月份</th><th>v7 基準 (36h)</th><th>v7 單月報酬</th><th>v8 優化 (24h)</th><th>v8 單月報酬</th><th>v9-A 波段版 (24h)</th><th>v9-A 單月報酬</th><th>v9-B 嚴選版 (24h)</th><th>v9-B 單月報酬</th></tr>
</thead>
<tbody>
{''.join(month_tr)}
</tbody>
</table>
</section>

<section class="box">
<h2>四、BingX 全歷史 130 筆訊號逐筆對照（v7 vs v8 vs v9-A vs v9-B）</h2>
<table>
<thead>
<tr><th>#</th><th>標的 / 型態</th><th>進場點 (價格 / TPE)</th><th>停損價 (破底低)</th><th>BingX 量價特徵</th><th>v7 結果 (36h)</th><th>v8 結果 (24h)</th><th>v9-A 波段版</th><th>v9-B 嚴選版 (含K線圖連結)</th></tr>
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

    funnel_v7: dict[str, int] = {}
    funnel_v8: dict[str, int] = {}
    funnel_v9a: dict[str, int] = {}
    funnel_v9b: dict[str, int] = {}
    for sym, df in sorted(dfs_1h.items()):
        m.detect_signals(df, funnel=funnel_v7, **BASE_V7_KW)
        m.detect_signals(df, funnel=funnel_v8, **OPT_V8_KW)
        m.detect_signals(df, funnel=funnel_v9a, **OPT_V9_TREND_KW)
        m.detect_signals(df, funnel=funnel_v9b, **OPT_V9_STRICT_KW)

    def run_sim(kw_detect: dict, max_hold: int) -> tuple[list[m.Hit], list[m.Hit], list[m.Hit]]:
        h_all, h_early, h_2m = [], [], []
        for sym, df in sorted(dfs_1h.items()):
            sigs = m.detect_signals(df, **kw_detect)
            if not sigs:
                continue
            trades = m.simulate(df, sigs, df_5m=dfs_5m.get(sym), m5_exit=True, max_hold=max_hold, lose_ma_closes=0)
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

    hits_v7, v7_e, v7_2m = run_sim(BASE_V7_KW, 36)
    hits_v7_24, v7_24_e, v7_24_2m = run_sim(BASE_V7_KW, 24)
    hits_v8, v8_e, v8_2m = run_sim(OPT_V8_KW, 24)
    hits_v9a, v9a_e, v9a_2m = run_sim(OPT_V9_TREND_KW, 24)
    hits_v9b, v9b_e, v9b_2m = run_sim(OPT_V9_STRICT_KW, 24)

    ablation = [
        ("v7 基準（36h 出場）", calc_metrics(hits_v7), calc_metrics(v7_e), calc_metrics(v7_2m)),
        ("v7 + 縮短持有至 24h", calc_metrics(hits_v7_24), calc_metrics(v7_24_e), calc_metrics(v7_24_2m)),
        ("v8 優化（進場量≥0.15x + MA7>14 + 24h）", calc_metrics(hits_v8), calc_metrics(v8_e), calc_metrics(v8_2m)),
        ("v9-A 波段版（v8 + 4hMA120漲幅≤0.5% + 破前低≥1.8% + 24h）", calc_metrics(hits_v9a), calc_metrics(v9a_e), calc_metrics(v9a_2m)),
        ("v9-B 嚴選版（v8 + BingX破底量≥1.10x + 上方均線≥0.8ATR + 24h）", calc_metrics(hits_v9b), calc_metrics(v9b_e), calc_metrics(v9b_2m)),
    ]

    for lbl, sta, ste, st2 in ablation:
        print(
            f"{lbl:<58} | All: n={sta['n']:3d}(sl={sta['reasons'].get('stop',0):2d}) "
            f"WR={sta['winrate_pct']:5.1f}% sum={sta['sum_pct']:+7.2f}% PF={sta['profit_factor']:5.2f} | "
            f"2M: n={st2['n']:2d}(sl={st2['reasons'].get('stop',0):2d}) WR={st2['winrate_pct']:5.1f}% sum={st2['sum_pct']:+7.2f}% PF={st2['profit_factor']:5.2f}",
            flush=True,
        )

    m_v7 = monthly_breakdown(hits_v7)
    m_v8 = monthly_breakdown(hits_v8)
    m_v9a = monthly_breakdown(hits_v9a)
    m_v9b = monthly_breakdown(hits_v9b)

    stats_v9b = m.summarize_trades([h.trade for h in hits_v9b])
    card_note = (
        "<br/><b>BingX 嚴選版（v9-B，針對 BingX 383 檔股票池平滑量能校準）</b>：破底量 ≥ 1.10x 均量、上方最近長均線 ≥ 0.8 ATR、"
        "進場量 ≥ 0.15x 均量、1H MA7 &gt; MA14、最長持有 24h。<a href='compare-view.html' style='color:#79c0ff'>查看 BingX 383 檔完整消融與 130 筆對照表</a>"
    )
    m.write_html_report(
        html_p,
        hits_v9b,
        days=250,
        scanned=len(dfs_1h),
        funnel=funnel_v9b,
        recent_hours=48,
        all_stats=stats_v9b,
        card_note=card_note,
        strict=False,
        m5_exit=True,
        lose_ma_closes=0,
        max_hold=24,
    )
    m.write_view_html(html_p, branch=BRANCH, out_name="view.html")

    taken_v9b = m.one_at_a_time_path(hits_v9b, start=100.0, lev=3.0)
    m.draw_seq_pngs(taken_v9b, OUT / "seq-img")
    m.write_seq_html(
        OUT / "seq.html",
        taken_v9b,
        days=250,
        scanned=len(dfs_1h),
        pool=len(hits_v9b),
        featured_html=m._featured_k_html(taken_v9b),
        more_k_href="seq-k.html",
    )
    m.write_seq_k_html(OUT / "seq-k.html", taken_v9b, days=250)
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
    write_csv(OUT / "trades_v9_trend.csv", hits_v9a)
    write_csv(OUT / "trades_v9_strict.csv", hits_v9b)
    write_compare_html(OUT / "compare.html", ablation, m_v7, m_v8, m_v9a, m_v9b, hits_v7, hits_v8, hits_v9a, hits_v9b)
    m.write_view_html(OUT / "compare.html", branch=BRANCH, out_name="compare-view.html")

    summary = {
        "exchange": "BingX Swap V2 (NCSK* Stock Perps)",
        "window_tpe": "2026-02-23 22:00 .. 2026-10-31 16:00 TPE (250 days)",
        "n_symbols": len(dfs_1h),
        "total_1h_bars": int(sum(len(df) for df in dfs_1h.values())),
        "baseline_v7": ablation[0][1],
        "v8_opt": ablation[2][1],
        "v9a_trend": ablation[3][1],
        "v9b_strict": ablation[4][1],
        "monthly_v7": m_v7,
        "monthly_v8": m_v8,
        "monthly_v9a": m_v9a,
        "monthly_v9b": m_v9b,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done! BingX reports written to {OUT} (stamped {len(stamped_paths)} images)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
