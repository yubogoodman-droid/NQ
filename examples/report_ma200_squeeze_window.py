#!/usr/bin/env python3
"""幣安 15m 黏帶：固定日期回測，只讀 K 線，不下單。

訊號規則用 nq/ma200_squeeze.py 與 scan_binance_15m_align.py 的現行程式。
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from statistics import median

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nq.ma200_squeeze import (  # noqa: E402
    HOLD_BARS,
    LOOKBACK,
    MAX_ABOVE_BARS,
    MAX_BOX,
    MAX_ENTRY_EXT,
    MAX_LONG_DETACH,
    MAX_MA200_ABOVE_BOX,
    MAX_MA200_ABOVE_OTHERS,
    MAX_MA200_SLOPE,
    MAX_MA7_VS_200,
    MAX_RANGE_EXPAND,
    MAX_RIBBON,
    MAX_RISK,
    MAX_VOL_RATIO,
    MIN_BARS_AT_OR_BELOW,
    MIN_BODY_FRAC,
    MIN_GAP_BARS,
    MIN_NEAR_FRAC,
    MIN_RANGE_EXPAND,
    MIN_RISK,
    MIN_VOL_RATIO,
    NEAR_200,
    STOP_BUFFER,
    TARGET_R,
    detect_signals,
    simulate_trades,
)

_spec = importlib.util.spec_from_file_location(
    "align_scan", REPO / "examples" / "scan_binance_15m_align.py"
)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

TZ = scan.TZ
OUT = REPO / "docs" / "binance" / "ma200-squeeze-15m-20261017-1030" / "index.html"
START = datetime(2026, 10, 17, tzinfo=TZ)
END = datetime(2026, 10, 31, tzinfo=TZ)  # 含 10/30 全日，不含 10/31


def _trim_unclosed(raw: dict) -> dict:
    now_ms = int(datetime.now(TZ).timestamp() * 1000)
    keep = raw["t"] + scan.BAR_MS <= now_ms
    if keep.all():
        return raw
    return {k: v[keep] for k, v in raw.items()}


def _one(sym: str, warm: datetime, fetch_end: datetime, attempts: int = 4) -> tuple[str, list[dict], str | None]:
    import time

    raw = None
    err: str | None = None
    for attempt in range(attempts):
        try:
            raw = scan.fetch_range(sym, warm, fetch_end)
            err = None
            break
        except Exception as e:
            err = str(e)
            time.sleep(1.5 * (attempt + 1))
    if err:
        return sym, [], err
    if raw is None:
        return sym, [], None
    raw = _trim_unclosed(raw)
    if len(raw["c"]) < scan.MIN_BARS:
        return sym, [], None
    d = scan.prepare(raw)
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    sigs = [
        s
        for s in detect_signals(d)
        if start_ms <= int(d["t"][s.idx]) < end_ms and scan.is_stacked(d, s.idx)
    ]
    rows = [{"symbol": sym, "d": d, "sig": s} for s in sigs]
    return sym, rows, None


def _fmt_px(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6g}"


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "win_rate": 0.0, "avg": 0.0, "median": 0.0, "best": 0.0, "worst": 0.0}
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(pnls),
        "win_rate": 100.0 * wins / len(pnls),
        "avg": float(sum(pnls) / len(pnls)),
        "median": float(median(pnls)),
        "best": float(max(pnls)),
        "worst": float(min(pnls)),
    }


def _rules_html() -> str:
    return f"""
<h2>觸發規則（照現行程式）</h2>
<ol>
<li><b>週期</b>：幣安 U 本位 15 分 K。時間以訊號 K 的開盤時間、台北時間（UTC+8）計。</li>
<li><b>標的</b>：<code>universe(0)</code>，目前 TRADING 的 USDT <code>PERPETUAL</code> 與 <code>TRADIFI_PERPETUAL</code>，排除 <code>underlyingType == INDEX</code>。這不是台股現股，所以 10/23 休市不另外跳過；幣安這天有 K 線就照算。</li>
<li><b>區間</b>：訊號 K 開盤時間落在 2026-10-17 00:00（含）到 2026-10-31 00:00（不含），也就是含 10/30 全日。尚未收盤的那一根不計。暖機從 10/07 起抓，只為了讓 MA200 有夠長的歷史，暖機段不算出訊號。</li>
<li><b>均線</b>：SMA 7 / 14 / 25 / 99 / 120 / 200，量能用 20 根均量。另外要當根 <b>MA5 &gt; MA20 &gt; MA99</b>（<code>is_stacked</code>）。</li>
<li><b>黏帶</b>（看訊號前一根）：六條均線最高/最低 − 1 ≤ {MAX_RIBBON*100:.1f}%。MA7 離 MA200 的絕對差距 ≤ {MAX_MA7_VS_200*100:.1f}%。MA200 不能高於其餘五條最高值的 {MAX_MA200_ABOVE_OTHERS*100:.2f}% 以上。MA99 與 MA120 的較低者，不能讓短均與 MA200 的較低者高出它 {MAX_LONG_DETACH*100:.2f}% 以上。</li>
<li><b>箱體</b>：回看 {LOOKBACK} 根（不含當根）。高低差 / 中價 ≤ {MAX_BOX*100:.1f}%。前一根 MA200 要在箱底×0.995 之上，且不高於箱頂×{1+MAX_MA200_ABOVE_BOX:.3f}。</li>
<li><b>MA200 斜率</b>：當根 MA200 / 20 根前 MA200 − 1 的絕對值 ≤ {MAX_MA200_SLOPE*100:.1f}%。</li>
<li><b>靠近 200</b>：這 {LOOKBACK} 根裡，收盤離 MA200 ≤ {NEAR_200*100:.1f}% 的比例 ≥ {MIN_NEAR_FRAC*100:.0f}%。收盤 ≤ MA200×1.003 的根數 ≥ {MIN_BARS_AT_OR_BELOW}。收盤嚴格站上 MA200 的根數 ≤ {MAX_ABOVE_BARS}。</li>
<li><b>突破 K</b>：陽線；收盤 &gt; MA200；收盤 &gt; 回看區間的收盤高；收盤 &gt; 前一根六條均線的最高值。收盤離當根 MA200 要在 0 到 {MAX_ENTRY_EXT*100:.2f}% 之間。量 / 20 均量在 {MIN_VOL_RATIO:.1f}× 到 {MAX_VOL_RATIO:.1f}×。這根振幅相對回看區間振幅中位數在 {MIN_RANGE_EXPAND:.0f}× 到 {MAX_RANGE_EXPAND:.0f}×。實體 / 振幅 ≥ {MIN_BODY_FRAC:.2f}。</li>
<li><b>同一段只留第一根</b>：<code>detect_signals</code> 兩筆訊號至少隔 {MIN_GAP_BARS} 根。程式另有「前一根已收過收盤箱頂且站上 200 與黏帶上緣就略過」；收盤箱頂的區間含前一根，所以前一根收盤不會大於這個最高值，這條在現行程式裡不會擋掉任何 K。</li>
<li><b>停損、停利的價位</b>（用訊號 K 的收盤當參考進場）：停損 = min(箱底, 當根 MA200) × (1 − {STOP_BUFFER*100:.1f}%)。風險 = (收盤 − 停損) / 收盤，要落在 {MIN_RISK*100:.1f}% 到 {MAX_RISK*100:.1f}%。停利 = 收盤 + {TARGET_R:.0f} × (收盤 − 停損)。</li>
<li><b>實際進場</b>：下一根開盤價。開盤 ≤ 停損，或下一根不存在，或還在上一筆持倉裡（<code>entry_idx &lt;= busy_until</code>），這筆不成交。</li>
<li><b>出場</b>：進場根是訊號的下一根（<code>entry_idx = sig.idx + 1</code>）。從這根檢查到 <code>entry_idx + {HOLD_BARS}</code>（含進場那根，一共 {HOLD_BARS + 1} 根）；資料不夠就停在最後一根。同一根先看最低價觸停損（停損價出場），再看最高價觸停利（停利價出場）；都沒有就在該窗口最後一根的收盤出場，記為時間出場。報酬 = 出場價 / 進場開盤 − 1。<b>未扣手續費與稅</b>。</li>
</ol>
<p class="muted">只讀幣安公開 K 線，沒有下單。</p>
"""


def render_card(n: int, row: dict, img_dir: Path) -> str:
    sig, d, sym = row["sig"], row["d"], row["symbol"]
    trade = row.get("trade")
    img_name = f"{n:03d}_{sym}_{scan.hm(int(d['t'][sig.idx])).replace(' ', '_').replace(':', '')}.png"
    png = scan.draw_chart(sym, d, sig, img_dir / img_name, trade=trade)
    src = scan._b64_img(png) if png and png.exists() else ""
    if trade is not None:
        reason = {"stop": "停損", "target": "停利", "time": "時間出場"}.get(trade.exit_reason, trade.exit_reason)
        held = int(trade.exit_idx) - int(trade.entry_idx) + 1
        short = ""
        if trade.exit_reason == "time" and int(trade.exit_idx) < int(trade.entry_idx) + HOLD_BARS:
            short = f"\n時間出場只走了 {held} 根（資料到抓取當下，窗口還沒滿）"
        pnl_s = f"{trade.pnl_pct:+.2f}%"
        pnl_cls = "pnl-win" if trade.pnl_pct >= 0 else "pnl-loss"
        detail = (
            f"訊號 {scan.hm(int(d['t'][sig.idx]))} 收 {_fmt_px(sig.close)}\n"
            f"進場 {scan.hm(int(d['t'][trade.entry_idx]))} 開 {_fmt_px(trade.entry_price)}\n"
            f"出場 {scan.hm(int(d['t'][trade.exit_idx]))} {reason} {_fmt_px(trade.exit_price)}  {trade.pnl_pct:+.2f}%\n"
            f"停損 {_fmt_px(sig.stop)}　停利 {_fmt_px(sig.target)}\n"
            f"離 200 {sig.ext*100:+.2f}%　黏帶 {sig.ribbon*100:.2f}%　量 {sig.vol_ratio:.1f}x　振幅 {sig.expand:.1f}x"
            f"{short}"
        )
    else:
        pnl_s = "未成交"
        pnl_cls = "pnl-flat"
        detail = (
            f"訊號 {scan.hm(int(d['t'][sig.idx]))} 收 {_fmt_px(sig.close)}\n"
            f"下一根開盤沒成交（開盤已穿停損、沒有下一根，或上一筆還在持倉）\n"
            f"參考停損 {_fmt_px(sig.stop)}　參考停利 {_fmt_px(sig.target)}\n"
            f"離 200 {sig.ext*100:+.2f}%　黏帶 {sig.ribbon*100:.2f}%　量 {sig.vol_ratio:.1f}x　振幅 {sig.expand:.1f}x"
        )
    img_html = f"<img src='{src}' alt='{escape(sym)}'/>" if src else ""
    return (
        "<article class='trade-card'>"
        f"<header class='card-header'><div><div class='trade-no'>#{n} · {escape(sym)}</div>"
        f"<div class='trade-time'>{scan.hm(int(d['t'][sig.idx]))}</div></div>"
        f"<div class='card-pnl {pnl_cls}'>{pnl_s}</div></header>"
        f"<pre class='trade-detail'>{escape(detail)}</pre>"
        f"<div class='mini-chart'>{img_html}</div></article>"
    )


def write_report(rows: list[dict], *, universe_n: int, errors: list[str]) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img_dir = OUT.parent / "img"
    rows = sorted(rows, key=lambda r: int(r["d"]["t"][r["sig"].idx]))
    cards = [render_card(n, row, img_dir) for n, row in enumerate(rows, 1)]
    filled = [r for r in rows if r.get("trade") is not None]
    pnls = [r["trade"].pnl_pct for r in filled]
    st = _stats(pnls)
    best = max(filled, key=lambda r: r["trade"].pnl_pct) if filled else None
    worst = min(filled, key=lambda r: r["trade"].pnl_pct) if filled else None

    def side(row) -> str:
        if row is None:
            return "—"
        t = row["trade"]
        return (
            f"{escape(row['symbol'])} {scan.hm(int(row['d']['t'][row['sig'].idx]))} "
            f"{t.pnl_pct:+.2f}%（{ {'stop':'停損','target':'停利','time':'時間出場'}.get(t.exit_reason, t.exit_reason) }）"
        )

    return _write_html(
        cards,
        universe_n=universe_n,
        errors=errors,
        stats=st,
        best_s=side(best),
        worst_s=side(worst),
        note="",
    )


def _write_html(cards, *, universe_n, errors, stats, best_s, worst_s, note: str) -> Path:
    st = stats
    err_html = ""
    if errors:
        err_html = (
            "<p class='muted'>讀取失敗 "
            + str(len(errors))
            + " 檔："
            + escape(", ".join(errors))
            + "</p>"
        )
    note_html = f"<p class='muted'>{escape(note)}</p>" if note else ""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>15m 黏帶 · 2026-10-17 到 10-30</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:720px;margin:0 auto;padding:16px 12px 40px}}
h1{{font-size:20px;margin:0 0 8px}}
h2{{font-size:16px;margin:18px 0 8px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.55}}
ol{{padding-left:1.2em;line-height:1.55;font-size:14px}}
li{{margin:0 0 8px}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#161b22;padding:10px 12px;border-radius:10px;min-width:108px;border:1px solid #30363d}}
.card span{{display:block;color:#8b949e;font-size:12px}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin:14px 0}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}}
.trade-time{{color:#8b949e;font-size:12px;margin-top:2px}}
.card-pnl{{font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}}
.pnl-loss{{color:#f85149}}
.pnl-flat{{color:#8b949e}}
.trade-detail{{margin:10px 0;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;white-space:pre-wrap}}
img{{width:100%;display:block;border-radius:10px}}
</style></head><body>
<div class="page">
<h1>幣安 15m 黏帶 · 2026-10-17 到 10-30</h1>
<p class="muted">全部 U 本位永續（幣＋股，排除指數）。進場是下一根開盤。報酬未扣手續費與稅。</p>
<div class="cards">
<div class="card"><span>標的</span><b>{universe_n}</b></div>
<div class="card"><span>訊號</span><b>{len(cards)}</b></div>
<div class="card"><span>成交</span><b>{st['n']}</b></div>
<div class="card"><span>勝率</span><b>{st['win_rate']:.1f}%</b></div>
<div class="card"><span>平均</span><b>{st['avg']:+.2f}%</b></div>
<div class="card"><span>中位數</span><b>{st['median']:+.2f}%</b></div>
<div class="card"><span>最好</span><b>{st['best']:+.2f}%</b></div>
<div class="card"><span>最差</span><b>{st['worst']:+.2f}%</b></div>
</div>
<p class="muted">最好：{best_s}<br/>最差：{worst_s}</p>
<p class="muted">勝率、平均、中位數、最好、最差都只算有成交的單。未成交不進這些數字。未扣手續費與稅。</p>
{note_html}
{err_html}
{_rules_html()}
<h2>每一筆</h2>
{''.join(cards) if cards else "<p class='muted'>這段沒有訊號。</p>"}
</div></body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> int:
    now = datetime.now(TZ)
    fetch_end = min(END + timedelta(hours=8), now)
    warm = START - timedelta(days=10)
    print(f"區間 {START.isoformat()} → {END.isoformat()}  抓到 {fetch_end.isoformat()}", flush=True)
    uni = scan.universe(0)
    symbols = [s for s, _ in uni]
    print(f"標的 {len(symbols)}", flush=True)
    rows: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(3) as ex:
        futs = {ex.submit(_one, s, warm, fetch_end): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym, got, err = fut.result()
            if err:
                errors.append(sym)
                print("err", sym, err, flush=True)
            rows.extend(got)
            if done % 40 == 0:
                print(f"  {done}/{len(symbols)}  訊號 {len(rows)}", flush=True)
    scan._attach_trades(rows)
    filled = [r["trade"].pnl_pct for r in rows if r.get("trade") is not None]
    st = _stats(filled)
    print(
        f"訊號 {len(rows)}  成交 {st['n']}  勝率 {st['win_rate']:.1f}%  "
        f"平均 {st['avg']:+.2f}%  中位 {st['median']:+.2f}%  "
        f"最好 {st['best']:+.2f}%  最差 {st['worst']:+.2f}%",
        flush=True,
    )
    path = write_report(rows, universe_n=len(symbols), errors=errors)
    print(f"HTML {path}", flush=True)
    return 0


def _article_key(article: str) -> tuple[str, str]:
    import re

    tm = re.search(r"trade-time'>([^<]+)", article)
    sym = re.search(r"trade-no'>#\d+ · ([^<]+)", article)
    return ((tm.group(1) if tm else ""), (sym.group(1) if sym else ""))


def _renumber(article: str, n: int) -> str:
    import re

    return re.sub(r"(<div class='trade-no'>#)\d+", rf"\g<1>{n}", article, count=1)


def _pnl_of(article: str) -> float | None:
    import re

    m = re.search(r"card-pnl [^']*'>([^<]+)", article)
    if not m:
        return None
    text = m.group(1).strip()
    if text == "未成交":
        return None
    return float(text.replace("%", "").replace("+", ""))


def _reason_of(article: str) -> str:
    import re

    m = re.search(r"出場[^\n<]*?(停損|停利|時間出場)", article)
    return m.group(1) if m else "—"


def merge_retry(new_rows: list[dict], errors: list[str], *, universe_n: int, note: str) -> Path:
    import re

    old = OUT.read_text(encoding="utf-8")
    old_articles = re.findall(r"<article class='trade-card'>.*?</article>", old, flags=re.S)
    img_dir = OUT.parent / "img"
    base = 500
    new_articles = [render_card(base + n, row, img_dir) for n, row in enumerate(new_rows, 1)]
    articles = old_articles + new_articles
    articles.sort(key=_article_key)
    articles = [_renumber(a, n) for n, a in enumerate(articles, 1)]
    pnls = [p for p in (_pnl_of(a) for a in articles) if p is not None]
    st = _stats(pnls)
    filled = [(a, p) for a, p in ((_a, _pnl_of(_a)) for _a in articles) if p is not None]
    if filled:
        best_a, best_p = max(filled, key=lambda x: x[1])
        worst_a, worst_p = min(filled, key=lambda x: x[1])
        btm, bsym = _article_key(best_a)
        wtm, wsym = _article_key(worst_a)
        best_s = f"{bsym} {btm} {best_p:+.2f}%（{_reason_of(best_a)}）"
        worst_s = f"{wsym} {wtm} {worst_p:+.2f}%（{_reason_of(worst_a)}）"
    else:
        best_s = worst_s = "—"
    return _write_html(
        articles,
        universe_n=universe_n,
        errors=errors,
        stats=st,
        best_s=best_s,
        worst_s=worst_s,
        note=note,
    )


# 第一次全宇宙掃描被 418/429 擋掉的檔。只補這些，不重抓已成功的標的。
RETRY_SYMBOLS = [
    "TUSDT", "SKDDUSDT", "WUSDT", "ARUSDT", "STXUSDT", "MUBARAKUSDT", "BRUSDT",
    "NMRUSDT", "COMPUSDT", "PIXELUSDT", "ARIAUSDT", "XANUSDT", "TLMUSDT", "ACEUSDT",
    "GIGGLEUSDT", "HK1810USDT", "MOVRUSDT", "TRBUSDT", "PUNDIXUSDT", "VETUSDT",
    "PROVEUSDT", "PEOPLEUSDT", "SUPERUSDT", "IOTAUSDT", "ATHUSDT", "ZBTUSDT",
    "UBUSDT", "OPNUSDT", "ZILUSDT", "STARUSDT", "YGGUSDT", "OPENUSDT", "IDUSDT",
    "UMAUSDT", "YBUSDT", "GRIFFAINUSDT", "IOTXUSDT", "ASMLUSDT", "TRIAUSDT",
    "TSTUSDT", "NAORISUSDT", "BREVUSDT", "PIEVERSEUSDT", "ENSOUSDT", "GPSUSDT",
    "MYXUSDT", "VELVETUSDT", "INXUSDT", "CLOUSDT", "ARXUSDT", "TTWOUSDT",
    "SWARMSUSDT", "KERNELUSDT", "GMXUSDT", "AIXBTUSDT", "BSBUSDT", "APPUSDT",
    "MMTUSDT", "GOATUSDT", "SKLUSDT", "BLUAIUSDT", "MEUSDT", "COAIUSDT", "XNYUSDT",
    "RECALLUSDT", "VTHOUSDT", "FLUIDUSDT", "AIOTUSDT", "DIAUSDT", "CYSUSDT",
    "CLANKERUSDT", "BASUSDT", "QNTXUSDT", "DYMUSDT", "USDEXUSDT", "DJTUSDT", "SONYUSDT",
]


def retry_failed() -> int:
    import time

    def get_json(path: str, params=None, retries: int = 8):
        last: Exception | None = None
        for i in range(retries):
            try:
                r = scan.SESSION.get(scan.BASE + path, params=params, timeout=20)
                if r.status_code in (418, 429):
                    last = RuntimeError(f"{r.status_code} {path}")
                    time.sleep(2.5 * (i + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                time.sleep(1.2 * (i + 1))
        raise last if last is not None else RuntimeError(f"failed {path}")

    scan.get_json = get_json
    now = datetime.now(TZ)
    fetch_end = min(END + timedelta(hours=8), now)
    warm = START - timedelta(days=10)
    print(f"補掃 {len(RETRY_SYMBOLS)}  抓到 {fetch_end.isoformat()}", flush=True)
    rows: list[dict] = []
    errors: list[str] = []
    for i, sym in enumerate(RETRY_SYMBOLS, 1):
        sym, got, err = _one(sym, warm, fetch_end, attempts=2)
        if err:
            errors.append(sym)
            print(f"err {i}/{len(RETRY_SYMBOLS)} {sym} {err}", flush=True)
        else:
            rows.extend(got)
            print(f"ok {i}/{len(RETRY_SYMBOLS)} {sym} +{len(got)}", flush=True)
        time.sleep(0.15)
    scan._attach_trades(rows)
    import pickle

    pickle.dump(rows, open("/tmp/squeeze_retry_rows.pkl", "wb"))
    note = (
        f"補掃時間 {now.strftime('%Y-%m-%d %H:%M')} 台北。"
        f"{len(RETRY_SYMBOLS) - len(errors)} 檔先前被限流的標的已補掃，新增訊號 {len(rows)}。"
        "其餘標的沿用同日稍早成功讀到的 K 線。只讀公開 K 線，沒有下單。"
    )
    path = merge_retry(rows, errors, universe_n=754, note=note)
    filled = [r for r in rows if r.get("trade") is not None]
    print(f"補掃訊號 {len(rows)} 成交 {len(filled)} 仍失敗 {len(errors)}", flush=True)
    for r in sorted(rows, key=lambda x: int(x["d"]["t"][x["sig"].idx])):
        t = r.get("trade")
        extra = f"{t.pnl_pct:+.2f}% {t.exit_reason}" if t else "未成交"
        print(f"  {r['symbol']} {scan.hm(int(r['d']['t'][r['sig'].idx]))} {extra}", flush=True)
    print(f"HTML {path}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--retry":
        raise SystemExit(retry_failed())
    raise SystemExit(main())
