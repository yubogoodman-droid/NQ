#!/usr/bin/env python3
"""台股 1h 破底翻：參數／濾網搜尋，訓練期挑、驗證期確認、測試期只看一次。

只回測、不下單。偵測規則仍是 tw_1h_reclaim.detect_signals / simulate；
濾網都是在訊號產生後套用，不改原本的回測行為。

    python3 examples/optimize_tw_1h_reclaim.py fetch      # 下載 1y 1h 快取
    python3 examples/optimize_tw_1h_reclaim.py explore    # 訓練期分桶
    python3 examples/optimize_tw_1h_reclaim.py search     # 網格搜尋
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_reclaim import (  # noqa: E402
    TPE,
    ReclaimParams,
    detect_signals,
    fetch_top_turnover,
    fetch_yahoo_1h,
    filter_by_max_price,
    loose_params,
    simulate,
    sma,
)

CACHE = Path("/tmp/tw1h_cache/cache_1y.pkl")
CLOSED_DAYS = {date(2026, 10, 23), date(2026, 10, 26)}

PERIODS = {
    "train": (pd.Timestamp("2025-11-01", tz=TPE), pd.Timestamp("2026-09-30 23:59", tz=TPE)),
    "valid": (pd.Timestamp("2026-10-01", tz=TPE), pd.Timestamp("2026-10-16 23:59", tz=TPE)),
    "test": (pd.Timestamp("2026-10-17", tz=TPE), pd.Timestamp("2026-10-30 23:59", tz=TPE)),
}


@dataclass(frozen=True)
class Filters:
    max_risk: float = 1.0
    skip_open_bar: bool = False
    market_above_ma20: bool = False
    max_bounce: float = 1.0
    require_stack: bool = False
    min_depth: float = 0.0


@dataclass
class Row:
    symbol: str
    code: str
    name: str
    entry_ts: pd.Timestamp
    entry: float
    stop: float
    target: float
    exit_ts: pd.Timestamp
    exit_px: float
    reason: str
    pnl_pct: float
    risk_pct: float
    bounce: float
    depth: float
    bars_below: int
    hour: int
    stacked: bool
    mkt_above: Optional[bool]
    mkt_ma20_up: Optional[bool]
    entry_idx: int
    exit_idx: int


def load_cache() -> dict:
    return pickle.loads(CACHE.read_bytes())


def cmd_fetch(args) -> int:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_top_turnover(args.date, 400)
    uni, _ = filter_by_max_price(raw, 1000 - 1e-9, 200)
    data = {}
    for row in uni:
        try:
            data[row["symbol"]] = fetch_yahoo_1h(row["symbol"], "1y")
        except Exception as exc:  # noqa: BLE001
            print("err", row["symbol"], exc, file=sys.stderr)
        time.sleep(0.15)
    twii = fetch_yahoo_1h("^TWII", "1y")
    CACHE.write_bytes(pickle.dumps({"universe": uni, "data": data, "twii": twii}))
    print("cached", len(data), "twii", len(twii))
    return 0


def market_frame(twii: pd.DataFrame) -> pd.DataFrame:
    c = twii["Close"].to_numpy(float)
    ma20 = sma(c, 20)
    out = pd.DataFrame({"close": c, "ma20": ma20}, index=twii.index)
    out["above"] = out["close"] > out["ma20"]
    out["ma20_up"] = out["ma20"] > out["ma20"].shift(5)
    return out


def build_rows(cache: dict, params: ReclaimParams) -> List[Row]:
    mkt = market_frame(cache["twii"])
    meta = {r["symbol"]: r for r in cache["universe"]}
    rows: List[Row] = []
    for sym, df in cache["data"].items():
        if df is None or len(df) < 40:
            continue
        df = df.loc[[ts.date() not in CLOSED_DAYS for ts in df.index]]
        sigs = detect_signals(df, params)
        trades = simulate(df, sigs, params)
        close = df["Close"].to_numpy(float)
        ma5, ma10, ma20 = sma(close, 5), sma(close, 10), sma(close, 20)
        for t in trades:
            s = t.signal
            ts = df.index[t.entry_idx]
            m = mkt.loc[mkt.index <= ts].tail(1)
            above = bool(m["above"].iloc[0]) if len(m) and not np.isnan(m["ma20"].iloc[0]) else None
            up = bool(m["ma20_up"].iloc[0]) if len(m) else None
            e = t.entry_idx
            rows.append(
                Row(
                    symbol=sym,
                    code=meta[sym]["code"],
                    name=meta[sym]["name"],
                    entry_ts=ts,
                    entry=t.entry_price,
                    stop=t.stop_price,
                    target=t.target_price,
                    exit_ts=df.index[t.exit_idx],
                    exit_px=t.exit_price,
                    reason=t.exit_reason,
                    pnl_pct=t.pnl_pct,
                    risk_pct=(t.entry_price - t.stop_price) / t.entry_price,
                    bounce=t.entry_price / s.wave.trough_low - 1.0,
                    depth=s.wave.depth_pct,
                    bars_below=s.wave.bars_below,
                    hour=int(ts.hour),
                    stacked=bool(ma5[e] > ma10[e] > ma20[e]),
                    mkt_above=above,
                    mkt_ma20_up=up,
                    entry_idx=t.entry_idx,
                    exit_idx=t.exit_idx,
                )
            )
    rows.sort(key=lambda r: r.entry_ts)
    return rows


def apply_filters(rows: Iterable[Row], f: Filters) -> List[Row]:
    out = []
    for r in rows:
        if r.risk_pct > f.max_risk:
            continue
        if f.skip_open_bar and r.hour == 9:
            continue
        if f.market_above_ma20 and r.mkt_above is not True:
            continue
        if r.bounce > f.max_bounce:
            continue
        if f.require_stack and not r.stacked:
            continue
        if r.depth < f.min_depth:
            continue
        out.append(r)
    return out


def in_period(rows: Iterable[Row], name: str) -> List[Row]:
    lo, hi = PERIODS[name]
    return [r for r in rows if lo <= r.entry_ts <= hi and 9 * 60 <= r.entry_ts.hour * 60 + r.entry_ts.minute <= 13 * 60 + 30]


def metrics(rows: Sequence[Row]) -> dict:
    p = [r.pnl_pct for r in rows]
    n = len(p)
    if not n:
        return {"n": 0, "win": 0.0, "avg": 0.0, "med": 0.0, "pf": 0.0, "best": 0.0, "worst": 0.0, "sum": 0.0}
    gains = sum(x for x in p if x > 0)
    losses = -sum(x for x in p if x < 0)
    return {
        "n": n,
        "win": 100.0 * sum(1 for x in p if x > 0) / n,
        "avg": float(np.mean(p)),
        "med": float(statistics.median(p)),
        "pf": gains / losses if losses > 0 else float("inf"),
        "best": max(p),
        "worst": min(p),
        "sum": float(sum(p)),
    }


def fmt(m: dict) -> str:
    return (
        f"n={m['n']:4d} win={m['win']:5.1f}% avg={m['avg']*100:+6.2f}% "
        f"med={m['med']*100:+6.2f}% pf={m['pf']:.2f}"
    )


def bucket(rows: Sequence[Row], key, labels: Sequence[Tuple[str, callable]]) -> None:
    for label, pred in labels:
        sel = [r for r in rows if pred(key(r))]
        print(f"  {label:<18} {fmt(metrics(sel))}")


def cmd_explore(args) -> int:
    cache = load_cache()
    rows = in_period(build_rows(cache, loose_params()), "train")
    print("TRAIN baseline", fmt(metrics(rows)))
    print("risk% (entry-stop)/entry")
    bucket(rows, lambda r: r.risk_pct, [
        ("<3%", lambda x: x < 0.03), ("3-5%", lambda x: 0.03 <= x < 0.05),
        ("5-8%", lambda x: 0.05 <= x < 0.08), (">=8%", lambda x: x >= 0.08)])
    print("entry hour")
    bucket(rows, lambda r: r.hour, [(f"{h:02d}:00", (lambda h: lambda x: x == h)(h)) for h in (9, 10, 11, 12, 13)])
    print("TAIEX 1h close vs MA20")
    bucket(rows, lambda r: r.mkt_above, [("above", lambda x: x is True), ("below", lambda x: x is False)])
    print("TAIEX MA20 rising (vs 5 bars ago)")
    bucket(rows, lambda r: r.mkt_ma20_up, [("up", lambda x: x is True), ("down", lambda x: x is False)])
    print("MA5>MA10>MA20 at entry")
    bucket(rows, lambda r: r.stacked, [("stacked", lambda x: x), ("not", lambda x: not x)])
    print("bounce from trough")
    bucket(rows, lambda r: r.bounce, [
        ("3-5%", lambda x: x < 0.05), ("5-8%", lambda x: 0.05 <= x < 0.08), (">=8%", lambda x: x >= 0.08)])
    print("depth")
    bucket(rows, lambda r: r.depth, [
        ("<6%", lambda x: x < 0.06), ("6-10%", lambda x: 0.06 <= x < 0.10), (">=10%", lambda x: x >= 0.10)])
    print("bars below")
    bucket(rows, lambda r: r.bars_below, [
        ("6-10", lambda x: x <= 10), ("11-20", lambda x: 11 <= x <= 20), ("21-36", lambda x: x >= 21)])
    print("exit reason")
    bucket(rows, lambda r: r.reason, [(k, (lambda k: lambda x: x == k)(k)) for k in ("target", "stop", "time", "open")])
    return 0


PARAM_GRID = {
    "target_r": [1.0, 1.5, 2.0],
    "time_bars": [5, 10, 20],
}
FILTER_GRID = {
    "max_risk": [1.0, 0.08],
    "skip_open_bar": [False, True],
    "market_above_ma20": [False, True],
    "require_stack": [False, True],
    "min_depth": [0.0, 0.06, 0.10],
}


def run_grid(cache: dict, min_train_n: int) -> List[dict]:
    results = []
    for tr, tb in itertools.product(PARAM_GRID["target_r"], PARAM_GRID["time_bars"]):
        params = loose_params(target_r=tr, time_bars=tb)
        rows = build_rows(cache, params)
        for mr, so, mk, st, md in itertools.product(*FILTER_GRID.values()):
            f = Filters(max_risk=mr, skip_open_bar=so, market_above_ma20=mk, require_stack=st, min_depth=md)
            sel = apply_filters(rows, f)
            res = {
                "target_r": tr,
                "time_bars": tb,
                **asdict(f),
                "train": metrics(in_period(sel, "train")),
                "valid": metrics(in_period(sel, "valid")),
                "test": metrics(in_period(sel, "test")),
            }
            results.append(res)
        print(f"done target_r={tr} time_bars={tb}", flush=True)
    return results


def label(r: dict) -> str:
    bits = [f"{r['target_r']:g}R", f"{r['time_bars']}根"]
    if r["max_risk"] < 1:
        bits.append(f"風險≤{r['max_risk']*100:g}%")
    if r["skip_open_bar"]:
        bits.append("不追09:00")
    if r["market_above_ma20"]:
        bits.append("大盤>MA20")
    if r["require_stack"]:
        bits.append("多頭排列")
    if r["min_depth"] > 0:
        bits.append(f"深度≥{r['min_depth']*100:g}%")
    return " · ".join(bits)


def cmd_search(args) -> int:
    cache = load_cache()
    results = run_grid(cache, args.min_train_n)
    ok = [r for r in results if r["train"]["n"] >= args.min_train_n]
    ok.sort(key=lambda r: r["train"]["avg"], reverse=True)
    print(f"\nTOP {args.top} by TRAIN avg (n>={args.min_train_n})")
    for r in ok[: args.top]:
        print(f"{label(r):<40} TRAIN {fmt(r['train'])} | VALID {fmt(r['valid'])}")
    base = next(r for r in results if r["target_r"] == 2.0 and r["time_bars"] == 20 and r["max_risk"] == 1.0
                and not r["skip_open_bar"] and not r["market_above_ma20"] and not r["require_stack"]
                and r["min_depth"] == 0.0)
    print("\nBASELINE", f"TRAIN {fmt(base['train'])} | VALID {fmt(base['valid'])} | TEST {fmt(base['test'])}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("saved", out)
    return 0


CHOSEN = Filters(market_above_ma20=True, min_depth=0.10)
REPORT_DIR = Path(__file__).resolve().parents[1] / "docs" / "tw-1h-reclaim-opt-20261031"
PREV_IMG = Path(__file__).resolve().parents[1] / "docs" / "tw-1h-reclaim-20261017-1030" / "img"


def daily_regime(twii: pd.DataFrame) -> pd.Series:
    """前一個交易日加權收盤 > 日 MA20（當天開盤前就知道，沒有偷看）。"""
    daily = twii["Close"].groupby(twii.index.date).last()
    return (daily > daily.rolling(20).mean()).shift(1)


def _equity_png(path: Path, series: Dict[str, List[Row]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    fig, ax = plt.subplots(figsize=(10.4, 4.6), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    colors = ["#8aa193", "#f0c14b", "#79c0ff", "#f472b6"]
    for (name, rows), col in zip(series.items(), colors):
        if not rows:
            continue
        xs = [r.entry_ts for r in rows]
        ys = np.cumsum([r.pnl_pct * 100 for r in rows])
        ax.plot(xs, ys, color=col, lw=1.4, label=f"{name}（{len(rows)} 筆）")
    for key, shade in (("valid", "#79c0ff22"), ("test", "#ff525233")):
        lo, hi = PERIODS[key]
        ax.axvspan(lo, hi, color=shade, lw=0)
    ax.axhline(0, color="#2a3a33", lw=0.8)
    ax.set_title("累積報酬（各筆 % 相加，未扣費稅）· 藍底=驗證期 · 紅底=測試期", color="#e8f0ea", fontsize=11)
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _market_png(path: Path, twii: pd.DataFrame, rows: List[Row]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tw_1h_reclaim import _setup_cjk

    _setup_cjk()
    m = market_frame(twii).loc["2026-10-01":"2026-10-30"]
    xs = list(range(len(m)))
    counts = pd.Series([r.entry_ts for r in rows]).value_counts()
    n_sig = [int(counts.get(ts, 0)) for ts in m.index]
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(10.4, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, facecolor="#0c1210"
    )
    for a in (ax, axb):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
    ax.plot(xs, m["close"], color="#e8f0ea", lw=1.2, label="加權指數 1h 收盤")
    ax.plot(xs, m["ma20"], color="#f472b6", lw=1.2, label="1h MA20")
    above = m["above"].to_numpy()
    for i, a in enumerate(above):
        if a:
            ax.axvspan(i - 0.5, i + 0.5, color="#3dba7a33", lw=0)
    axb.bar(xs, n_sig, color="#f0c14b", width=0.8)
    axb.set_ylabel("訊號數", color="#8aa193", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#c8d5cc")
    ax.set_title("10 月加權指數 1h · 綠底=站上 1h MA20（優化版濾網放行）· 下方=基準版每根訊號數",
                 color="#e8f0ea", fontsize=10)
    step = max(1, len(m) // 8)
    ticks = list(range(0, len(m), step))
    axb.set_xticks(ticks)
    axb.set_xticklabels([m.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _trade_png(cache: dict, r: Row, out: Path, no: int) -> str:
    stem = f"_{r.code}_{r.entry_ts.strftime('%m%d_%H%M')}.png"
    for p in PREV_IMG.glob(f"t*{stem}"):
        return f"../tw-1h-reclaim-20261017-1030/img/{p.name}"
    from tw_1h_reclaim import draw_trade_png

    df = cache["data"][r.symbol]
    df = df.loc[[ts.date() not in CLOSED_DAYS for ts in df.index]]
    params = loose_params()
    sigs = [s for s in detect_signals(df, params) if df.index[s.entry_idx] == r.entry_ts]
    trade = simulate(df, sigs, params)[0]
    name = f"t{no:02d}{stem}"
    draw_trade_png(df, trade, out / "img" / name, no, title_extra=f"{r.code} {r.name}")
    return f"img/{name}"


def _mrow(title: str, m: dict) -> str:
    def pct(x: float) -> str:
        return f"{x*100:+.2f}%"

    cls = "pnl-win" if m["avg"] > 0 else ("pnl-loss" if m["avg"] < 0 else "")
    if not m["n"]:
        return f"<tr><td>{title}</td><td>0</td><td colspan='5' class='muted'>沒有訊號</td></tr>"
    return (
        f"<tr><td>{title}</td><td>{m['n']}</td><td>{m['win']:.1f}%</td>"
        f"<td class='{cls}'>{pct(m['avg'])}</td><td>{pct(m['med'])}</td>"
        f"<td>{pct(m['best'])} / {pct(m['worst'])}</td><td>{m['pf']:.2f}</td></tr>"
    )


def cmd_report(args) -> int:
    from html import escape

    cache = load_cache()
    out = REPORT_DIR
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = build_rows(cache, loose_params())
    reg = daily_regime(cache["twii"])
    variants = {
        "基準（原規則）": rows,
        "優化版：大盤 1h>MA20 · 深度≥10%": apply_filters(rows, CHOSEN),
        "只加大盤 1h>MA20": apply_filters(rows, Filters(market_above_ma20=True)),
        "候選（事後提出）：前日加權日線>日MA20": [
            r for r in rows if not pd.isna(reg.get(r.entry_ts.date())) and bool(reg.get(r.entry_ts.date()))
        ],
    }
    table = []
    for per, title in (("train", "訓練 2025-11～2026-09"), ("valid", "驗證 10/01～10/16"), ("test", "測試 10/17～10/30")):
        table.append(f"<h3>{title}</h3><table><tr><th>版本</th><th>筆數</th><th>勝率</th><th>平均</th>"
                     f"<th>中位數</th><th>最好 / 最差</th><th>獲利因子</th></tr>")
        for name, sel in variants.items():
            table.append(_mrow(escape(name), metrics(in_period(sel, per))))
        table.append("</table>")
    _equity_png(out / "img" / "equity.png", {k: in_period(v, "train") + in_period(v, "valid") + in_period(v, "test")
                                             for k, v in variants.items()})
    _market_png(out / "img" / "taiex_oct.png", cache["twii"], in_period(rows, "test"))

    chosen_test = in_period(variants["優化版：大盤 1h>MA20 · 深度≥10%"], "test")
    cards = []
    for i, r in enumerate(chosen_test, 1):
        src = _trade_png(cache, r, out, i)
        cls = "pnl-win" if r.pnl_pct > 0 else "pnl-loss"
        reason = {"stop": "停損", "target": "停利 2R", "time": "時間出場", "open": "未平"}.get(r.reason, r.reason)
        cards.append(
            "<article class='trade-card'><header class='card-header'>"
            f"<span class='trade-no'>#{i} · {escape(r.code)} {escape(r.name)}</span>"
            f"<span class='{cls}'>{r.pnl_pct*100:+.2f}%</span></header>"
            f"<p class='muted'>{r.entry_ts.strftime('%m-%d %H:%M')} 進 {r.entry:.2f} → {r.exit_ts.strftime('%m-%d %H:%M')} "
            f"{reason} {r.exit_px:.2f} · 深度 {r.depth*100:.1f}% · 停損距離 {r.risk_pct*100:.1f}%</p>"
            f"<img src='{src}' alt='{escape(r.code)}' style='width:100%;border-radius:10px'/></article>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 破底翻 · 優化檢驗</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:760px;margin:0 auto;padding:14px 12px 40px}}
section{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:19px;margin:0 0 8px}} h2{{font-size:16px;margin:0 0 8px}} h3{{font-size:14px;margin:14px 0 6px;color:#c9d1d9}}
.muted{{color:#8b949e;font-size:13px;line-height:1.6}} li{{font-size:13px;line-height:1.6;margin-bottom:4px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{border-bottom:1px solid #21262d;padding:6px 4px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:#8b949e;font-weight:500}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:12px;margin-bottom:12px}}
.card-header{{display:flex;justify-content:space-between;font-weight:700}}
.verdict{{border-left:4px solid #ff5252;padding-left:10px}}
</style></head><body><div class="page">
<section>
<h1>台股 1h 破底翻 · 優化檢驗</h1>
<p class="muted">只回測、不下單。報酬未扣手續費與稅。宇宙為 2026-10-30 成交額前 200（股價 &lt; 1000），用同一批股票回推一年，有倖存者偏差，訓練期數字偏樂觀。
偵測、進場、停損 2R、20 根時間出場都沿用 <code>examples/tw_1h_reclaim.py</code>，濾網只在訊號產生後套用。週末與 10/23、10/26 休市略過。</p>
<p class="muted">做法：在訓練期用網格挑參數（停利 1/1.5/2R × 時間 5/10/20 根 × 停損距離上限 × 不追 09:00 × 大盤 1h&gt;MA20 × 多頭排列 × 最小深度，共 432 組），驗證期確認，最後只打開一次測試期。</p>
</section>
<section class="verdict">
<h2>結論</h2>
<ul>
<li>訓練期最穩定的改善是「<b>加權 1h 收盤 &gt; 1h MA20</b>」加上「<b>破底深度 ≥ 10%</b>」：訓練 839 筆，平均 +4.14%（基準 +1.98%），驗證 7 筆 +2.27%。改停利或時間出場沒有穩定幫助。</li>
<li><b>這個優化版在 10/17～10/30 失敗</b>：36 筆、勝率 8.3%、平均 −5.89%，比基準 −3.88% 還差。這兩週大盤是下跌趨勢，1h 指數只在 10/21、10/28 反彈時短暫站上 MA20，而訊號正好擠在那幾根，1h 濾網等於把最差的那批放進來。</li>
<li>較慢的「前一日加權日線 &gt; 日 MA20」會讓這兩週完全不進場（0 筆），等於避開全部虧損；但它在訓練期反而<b>降低</b>平均報酬（+1.38%），而且是看過測試期後才提出的，只能當候選，不算驗證過。</li>
<li>建議：Telegram 監看先維持原規則，不切換；若要加大盤開關，用日線趨勢、並等新的資料再驗。</li>
</ul>
</section>
<section><h2>各版本比較</h2>{''.join(table)}</section>
<section><h2>累積報酬</h2><img src="img/equity.png" style="width:100%;border-radius:10px"/></section>
<section><h2>為什麼 1h 大盤濾網在測試期失效</h2><img src="img/taiex_oct.png" style="width:100%;border-radius:10px"/></section>
<section><h2>優化版測試期 {len(chosen_test)} 筆訊號</h2>
<p class="muted">綠三角進場、黃叉出場、黃／藍／粉為 MA5／10／20、紅虛線停損、綠虛線 2R。</p></section>
{''.join(cards)}
</div></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    summary = {name: {per: metrics(in_period(sel, per)) for per in PERIODS} for name, sel in variants.items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from tw_1h_reclaim import _git_branch

    docs = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/docs/"
    view_text = (
        html.replace('src="img/', f'src="{docs}{out.name}/img/')
        .replace("src='img/", f"src='{docs}{out.name}/img/")
        .replace("src='../", f"src='{docs}")
    )
    view = out / "view.html"
    view.write_text(view_text, encoding="utf-8")
    print("html", out / "index.html", "view", view, "test cards", len(chosen_test))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股 1h 破底翻參數搜尋（只回測）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--date", default="20261030")
    f.set_defaults(func=cmd_fetch)
    e = sub.add_parser("explore")
    e.set_defaults(func=cmd_explore)
    s = sub.add_parser("search")
    s.add_argument("--min-train-n", type=int, default=150)
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--out", default="/tmp/tw1h_cache/search.json")
    s.set_defaults(func=cmd_search)
    r = sub.add_parser("report")
    r.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
