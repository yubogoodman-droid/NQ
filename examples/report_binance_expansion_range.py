#!/usr/bin/env python3
"""幣安 15m MA200 附近擴張：指定日期區間回測，產出單一自包含 HTML。

只讀歷史 K，不下單。規則全部從 scan_binance_15m_expansion.py 的實際常數／函式來。
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_binance_15m_expansion as S

OUT = S.REPO / "docs" / "binance" / "expansion-15m-20261017-1030" / "index.html"
START = S.parse_tw("2026-10-17 00:00")
END = S.parse_tw("2026-10-30 23:59")
WARMUP_DAYS = 6


def fetch_range(sym: str, start: datetime, end: datetime) -> dict | None:
    """分段抓 15m，尾段不足 220 根也留（避免最後幾天被丢掉）。"""
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=8), end)
        part = None
        for attempt in range(4):
            try:
                part = S.fetch_klines(
                    sym,
                    limit=1500,
                    start_ms=int(cur.timestamp() * 1000),
                    end_ms=int(nxt.timestamp() * 1000),
                    drop_unclosed=False,
                    interval="15m",
                    min_bars=2,
                )
                break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
        if part is not None:
            chunks.append(part)
        time.sleep(0.18)
        cur = nxt
    if not chunks:
        return None
    t = np.concatenate([p["t"] for p in chunks])
    _, idx = np.unique(t, return_index=True)
    idx.sort()
    out = {k: np.concatenate([p[k] for p in chunks])[idx] for k in ("t", "o", "h", "l", "c", "v")}
    bar_ms = S.BAR_MS["15m"]
    now_ms = int(time.time() * 1000)
    if len(out["t"]) and int(out["t"][-1]) + bar_ms > now_ms:
        out = {k: v[:-1] for k, v in out.items()}
    return out if len(out["c"]) >= S.MIN_BARS else None


def one_symbol(sym: str, start_ms: int, end_ms: int) -> tuple[str, dict | None, list[dict]]:
    fetch_start = datetime.fromtimestamp(start_ms / 1000, S.TZ) - timedelta(days=WARMUP_DAYS)
    fetch_end = datetime.fromtimestamp(end_ms / 1000, S.TZ) + timedelta(minutes=15)
    d0 = fetch_range(sym, fetch_start, fetch_end)
    if d0 is None:
        return sym, None, []
    d = S.indicators(d0)
    trades = []
    for hit in S.select_alerts(d, start_ms, end_ms):
        tr = S.simulate_trade(d, hit)
        if tr is None:
            continue
        tr["symbol"] = sym
        trades.append(tr)
    return sym, d, trades


def run(start: datetime, end: datetime) -> tuple[list[dict], dict[str, dict], list[str]]:
    now = datetime.now(S.TZ)
    if end > now:
        end = now
    print("載入標的…", flush=True)
    symbols = S.universe()
    print(f"{len(symbols)} 個流動永續", flush=True)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    print(
        f"回測 {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}  "
        f"{len(symbols)} 檔 15m（台北時間）",
        flush=True,
    )
    trades: list[dict] = []
    data: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(3) as ex:
        futs = {ex.submit(one_symbol, s, start_ms, end_ms): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            try:
                sym, d, rows = fut.result()
            except Exception as e:
                print("err", futs[fut], e, flush=True)
                continue
            if d is not None:
                data[sym] = d
            trades.extend(rows)
            if done % 25 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)}  訊號 {len(trades)}", flush=True)
    trades.sort(key=lambda t: t["t_signal"])
    before = len(trades)
    trades = S.drop_market_cluster(trades)
    if before != len(trades):
        print(f"大盤集群濾掉 {before - len(trades)} 筆", flush=True)
    return trades, data, symbols


def extra_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"median": 0.0, "best": None, "worst": None}
    pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)
    return {
        "median": float(np.median(pnls)),
        "best": max(trades, key=lambda t: t["pnl_pct"]),
        "worst": min(trades, key=lambda t: t["pnl_pct"]),
    }


def rules_html() -> str:
    keep = "、".join(sorted(S.KEEP))
    hold_h = S.HOLD_BARS * 15 / 60
    return f"""
<h2>觸發規則（照程式，不是口頭版）</h2>
<p class="muted">來源 <code>examples/scan_binance_15m_expansion.py</code> 的常數與 <code>_hit_at</code>／<code>select_alerts</code>／<code>simulate_trade</code>。這份報告是幣安永續，不是台股。</p>
<ol class="rules">
<li><b>週期</b>：幣安 U 本位永續 15 分鐘 K（<code>fapi/v1/klines</code> interval=15m）。時間用台北 UTC+8。</li>
<li><b>標的範圍</b>：<code>exchangeInfo</code> 裡 quote=USDT、status=TRADING、contractType 為 PERPETUAL 或 TRADIFI_PERPETUAL、不是 INDEX；24h quoteVolume ≥ 5,000,000 USDT。另外永遠保留 {escape(keep)}。</li>
<li><b>時段</b>：記號棒開盤小時 ∈ {S.SESSION_HOURS.start}–{S.SESSION_HOURS.stop - 1}（全天）。無台股盤、無視搓。</li>
<li><b>進場條件（同一根 15m 擴張棒，全部成立才算）</b>：
  <ul>
    <li>至少 200 根 K，MA7/14/25/99/120/200、成交量 MA20、前一根 ATR20 都有值。</li>
    <li>陽線：收盤 &gt; 開盤；振幅 &gt; 0。</li>
    <li>收盤位置 (C−L)/(H−L) ≥ {S.MIN_CLOSE_POS:.2f}。</li>
    <li>實體 (C/O−1) ≥ {S.MIN_BODY:.3f}（{S.MIN_BODY * 100:.1f}%）。</li>
    <li>振幅 / 前一根 ATR20 ≥ {S.MIN_RANGE_ATR:.2f}。</li>
    <li>前一根 ATR20 / 前收 ≤ {S.MAX_PRIOR_ATR_PCT:.3f}（前面不能已經在噴）。</li>
    <li>量比 本根量 / 量MA20 ≥ {S.MIN_VOL_RATIO:.2f}，且本根量 ≥ 前根量 × {S.MIN_MARK_VOL_VS_PREV:.1f}。</li>
    <li>收盤創近 {S.LOOKBACK_BREAK} 根最高價新高（比前 {S.LOOKBACK_BREAK} 根的 high 最大值還高）。</li>
    <li>MA7 &gt; MA14 &gt; MA25；MA7/MA14−1 ≥ {S.MIN_STACK_7_14:.3f}；MA7/MA25−1 ≥ {S.MIN_STACK_7_25:.3f}。</li>
    <li>收盤站上整疊均線 max(MA7,14,25,99,120,200)。</li>
    <li>近 {S.SQUEEZE_LOOKBACK} 根裡，六條均線帶寬 min(max/min−1) ≤ {S.MAX_SQUEEZE_RIBBON:.2f}（曾經擠過）。</li>
    <li>收盤 &gt; MA200，且離 MA200 ≤ {S.MAX_MARK_EXT:.0%}。</li>
    <li>近 {S.NEAR_200_LOOKBACK} 根（4 小時）內，曾有收盤 ≤ MA200 × {1 + S.NEAR_200_BAND:.2f}（從 200 附近起來，不是天上掉下來）。</li>
  </ul>
</li>
<li><b>去重</b>：同一檔以記號時間 ÷ {S.ALERT_BUCKET_MS // 3600000} 小時分桶，一桶只留時間序第一根（<code>select_alerts</code>）。</li>
<li><b>大盤叢集</b>：同一根 15m 記號 ≥ {S.MAX_SAME_MARK} 檔就整批不進，之後 {S.CLUSTER_COOLDOWN_MS // 3600000} 小時的跟風也不進。ETH 09-03 22:45 那晚是 14 檔，低於這個門檻，仍會進。</li>
<li><b>進場價</b>：擴張棒收完，<b>下一根 15m 開盤</b>做多。沒有下一根就不做。</li>
<li><b>停損</b>：沒有掛價停損。持有期間若某根 <b>收盤跌破當根 MA200</b>，以該根收盤出場（reason=<code>ma_break</code>）。記號棒的 MA200 只拿來算 R，不是實際停損價。</li>
<li><b>停利</b>：<b>沒有</b>。<code>TARGET_R</code>／<code>MAX_RISK</code> 寫在檔頭但 <code>simulate_trade</code> 沒用到。</li>
<li><b>時間出場</b>：最多抱 {S.HOLD_BARS} 根＝{hold_h:.0f} 小時，以到期那根收盤出（reason=<code>time</code>）。K 線不夠則以最後一根收盤出（reason=<code>eod</code>）。</li>
<li><b>報酬</b>：等權每筆 1 單位，(出場價/進場價−1)。<b>未扣手續費、資金費、稅</b>。</li>
</ol>
"""


def write_html(
    path: Path,
    trades: list[dict],
    data: dict[str, dict],
    *,
    n_symbols: int,
    window_start: str,
    window_end: str,
) -> Path:
    stats = S.summarize_trades(trades)
    extra = extra_stats(trades)
    total_cls = "pnl-win" if stats["pnl"] >= 0 else "pnl-loss"
    avg_cls = "pnl-win" if stats["avg"] >= 0 else "pnl-loss"
    med_cls = "pnl-win" if extra["median"] >= 0 else "pnl-loss"
    best, worst = extra["best"], extra["worst"]
    reason_line = " · ".join(f"{k} {n}" for k, n in stats["by_reason"].items())
    fwd_bits = []
    labels = {1: "15m", 4: "1h", 8: "2h", 16: "4h"}
    for nbar, lab in labels.items():
        fs = stats["fwd"].get(nbar) or {}
        if fs.get("n"):
            fwd_bits.append(f"{lab} 勝率 {fs['win_rate']:.0f}% 均 {fs['avg']:+.2f}%")
    fwd_line = " · ".join(fwd_bits)
    print(f"畫 {len(trades)} 張 15m 圖（內嵌）…", flush=True)
    cards = []
    for i, t in enumerate(trades, 1):
        cls = "pnl-win" if t["pnl_pct"] > 0 else ("pnl-flat" if t["pnl_pct"] == 0 else "pnl-loss")
        reason_cls = {"ma_break": "tag-sl", "time": "tag-time", "eod": "tag-time"}.get(t["reason"], "tag-info")
        img = ""
        d = data.get(t["symbol"])
        b64 = S.draw_trade_b64(t["symbol"], d, t) if d else None
        if b64:
            img = (
                "<div class='mini-chart'><div class='chart-label'>15m　黃菱形＝擴張棒　綠三角＝進場　×＝出場　白線＝MA200</div>"
                f"<img src='data:image/png;base64,{b64}' alt='#{i} {escape(t['symbol'])} 15m' "
                "style='width:100%;display:block'/></div>"
            )
        if i % 20 == 0 or i == len(trades):
            print(f"  圖 {i}/{len(trades)}", flush=True)
        if i == 1 and not img:
            print("畫圖失敗（檢查有沒有裝 matplotlib）", flush=True)
        h1 = (t["fwd"].get(4) or 0) * 100
        h2 = (t["fwd"].get(8) or 0) * 100
        h4 = (t["fwd"].get(16) or 0) * 100
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(t['symbol'])}</span>"
            f"<span class='trade-time'>{escape(S.hm(t['t_entry']))} → {escape(S.hm(t['t_exit']))}</span></div>"
            f"<div class='card-pnl {cls}'>{t['pnl_pct']:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t['reason'])}</span>"
            "<span class='tag tag-info'>MA200 附近擴張</span>"
            f"<span class='tag tag-info'>記號 {escape(S.hm(t.get('t_mark', t['t_signal'])))}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"進場 {t['entry']:g}  （下一根開盤）\n"
            f"出場 {t['exit']:g}  {t['reason']}\n"
            f"離200 {t.get('mark_ext', 0)*100:+.2f}%  實體 {t.get('body', 0)*100:+.2f}%  "
            f"{t.get('range_atr', 0):.1f}×ATR  量比 {t['vol_ratio']:.1f}×\n"
            f"MFE {t['mfe_pct']:+.2f}%  MAE {t['mae_pct']:+.2f}%\n"
            f"固定時間前瞻 1h {h1:+.2f}%  2h {h2:+.2f}%  4h {h4:+.2f}%"
            "</pre>"
            f"{img}"
            "</article>"
        )
    best_s = (
        f"{best['symbol']} {S.hm(best['t_entry'])} {best['pnl_pct']:+.2f}%" if best else "—"
    )
    worst_s = (
        f"{worst['symbol']} {S.hm(worst['t_entry'])} {worst['pnl_pct']:+.2f}%" if worst else "—"
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>幣安 15m MA200 附近擴張 · 2026-10-17～10-30</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:640px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
h2{{font-size:15px;margin:16px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.rules{{margin:0;padding-left:1.2em;font-size:13px;line-height:1.55;color:#c9d1d9}}
.rules li{{margin:0 0 8px}}
.rules ul{{margin:6px 0 0;padding-left:1.1em}}
code{{font-size:12px;background:#0d1117;padding:1px 4px;border-radius:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px 8px;border-radius:10px;overflow:hidden;background:#0c1210}}
.chart-label{{font-size:11px;font-weight:600;color:#8b949e;padding:8px 12px 0}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>幣安 15m MA200 附近擴張 · 兩週回測</h1>
<p class="muted">區間 {escape(window_start)} → {escape(window_end)}（台北時間）· 掃 {n_symbols} 檔永續。只回測、不下單。圖片內嵌在這一個 HTML 裡。</p>
<div class="cards">
<div class="card">訊號數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">平均報酬<b class="{avg_cls}">{stats['avg']:+.2f}%</b></div>
<div class="card">中位數<b class="{med_cls}">{extra['median']:+.2f}%</b></div>
<div class="card">等權合計<b class="{total_cls}">{stats['pnl']:+.1f}%</b></div>
<div class="card">最好一筆<b class="pnl-win">{escape(best_s)}</b></div>
<div class="card">最差一筆<b class="pnl-loss">{escape(worst_s)}</b></div>
</div>
<div class="equity">{S._equity_svg([t['pnl_pct'] for t in trades])}</div>
<p class="muted">MFE 均 {stats['mfe']:+.2f}% · MAE 均 {stats['mae']:+.2f}%</p>
<p class="muted">出場 {escape(reason_line) if reason_line else '—'}</p>
<p class="muted">固定時間前瞻（不管 MA200 出場）{escape(fwd_line) if fwd_line else '—'}</p>
<p class="muted">未扣手續費、資金費、稅。下面每筆一張 15m K 線：進場（綠三角）、均線、出場（×）。</p>
{rules_html()}
</section>
{''.join(cards) if cards else "<div class='empty'>這段沒有訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    start, end = START, END
    trades, data, symbols = run(start, end)
    stats = S.summarize_trades(trades)
    extra = extra_stats(trades)
    print(
        f"trades={stats['count']} WR={stats['win_rate']:.1f}% "
        f"avg={stats['avg']:+.2f}% median={extra['median']:+.2f}% pnl={stats['pnl']:+.1f}%"
    )
    if extra["best"]:
        b, w = extra["best"], extra["worst"]
        print(f"best {b['symbol']} {S.hm(b['t_entry'])} {b['pnl_pct']:+.2f}%")
        print(f"worst {w['symbol']} {S.hm(w['t_entry'])} {w['pnl_pct']:+.2f}%")
    for nbar, lab in ((1, "15m"), (4, "1h"), (8, "2h"), (16, "4h")):
        fs = stats["fwd"].get(nbar) or {}
        if fs.get("n"):
            print(f"  fwd {lab}: n={fs['n']} WR={fs['win_rate']:.1f}% avg={fs['avg']:+.2f}%")
    now = datetime.now(S.TZ)
    window_end = min(end, now)
    out = write_html(
        OUT,
        trades,
        data,
        n_symbols=len(symbols),
        window_start=start.strftime("%Y-%m-%d %H:%M"),
        window_end=window_end.strftime("%Y-%m-%d %H:%M"),
    )
    print(f"html={out}  bytes={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
