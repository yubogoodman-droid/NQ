#!/usr/bin/env python3
"""台股成交額前 100：同一套 1m 破底翻 MA Reclaim，回測一週。

點數門檻用股價 / 20000 等比縮放（對齊 NQ）。09–10 用台北時間。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    detect_signals,
    draw_trade_png,
    simulate,
    summarize_trades,
)

TPE = ZoneInfo("Asia/Taipei")
REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "tw-ma-reclaim" / "index.html"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
NQ_REF = 20000.0


@dataclass
class TwHit:
    row: dict
    trade: Any
    df: pd.DataFrame


def _get_json(url: str, retries: int = 4) -> dict | list:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.4 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _num(value: object) -> int:
    return int(str(value).replace(",", "").replace('"', "").strip() or 0)


def _price(value: object) -> float | None:
    text = str(value).replace(",", "").replace('"', "").strip()
    if not text or text in {"--", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_stock_code(code: str) -> bool:
    """上市櫃普通股：四位數字。排除 ETF / 債 / 權證代號。"""
    return code.isdigit() and len(code) == 4 and not code.startswith("00")


def yahoo_symbol(code: str, market: str) -> str:
    return f"{code}.TW" if market == "tse" else f"{code}.TWO"


def tw_pt_scale(price: float, ref: float = NQ_REF) -> float:
    return max(float(price) / ref, 1e-4)


def session_mask(index: pd.DatetimeIndex) -> pd.Series:
    minutes = index.hour * 60 + index.minute
    return (minutes >= 9 * 60) & (minutes <= 13 * 60 + 30)


def last_tw_session_yyyymmdd(now: datetime | None = None) -> str:
    cur = (now or datetime.now(TPE)).date()
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur.strftime("%Y%m%d")


def fetch_top_turnover(date: str, limit: int) -> list[dict]:
    twse = _get_json(
        f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type=ALLBUT0999&response=json"
    )
    if twse.get("stat") != "OK":
        raise RuntimeError(f"TWSE stat={twse.get('stat')}")
    tables = twse.get("tables") or []
    items: list[tuple[int, str, str, str, float | None]] = []
    for table in tables:
        fields = [str(x) for x in (table.get("fields") or [])]
        rows = table.get("data") or []
        if not rows or len(rows[0]) < 9:
            continue
        if not fields or "證券代號" not in fields[0]:
            continue
        for rec in rows:
            code, name = str(rec[0]).strip(), str(rec[1]).strip()
            if not _is_stock_code(code):
                continue
            amt = _num(rec[4])
            if amt > 0:
                items.append((amt, code, name, "tse", _price(rec[8])))
        break

    roc = f"{int(date[:4]) - 1911}{date[4:]}"
    try:
        tpex = _get_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes")
        for rec in tpex:
            if rec.get("Date") not in {roc, date}:
                continue
            code = str(rec["SecuritiesCompanyCode"]).strip()
            name = str(rec["CompanyName"]).strip()
            if not _is_stock_code(code):
                continue
            amt = _num(rec.get("TransactionAmount") or 0)
            if amt > 0:
                items.append((amt, code, name, "otc", _price(rec.get("Close"))))
    except Exception as exc:  # noqa: BLE001
        print(f"[tpex] skip: {exc}", file=sys.stderr)

    best: dict[str, tuple[int, str, str, str, float | None]] = {}
    for amt, code, name, mkt, close in items:
        prev = best.get(code)
        if prev is None or amt > prev[0]:
            best[code] = (amt, code, name, mkt, close)

    ranked = sorted(best.values(), reverse=True)[:limit]
    return [
        {
            "rank": i,
            "code": code,
            "name": name,
            "market": mkt,
            "amount": amt,
            "close": close,
            "symbol": yahoo_symbol(code, mkt),
        }
        for i, (amt, code, name, mkt, close) in enumerate(ranked, 1)
    ]


def filter_by_max_price(rows: list[dict], max_price: float | None, limit: int) -> tuple[list[dict], list[dict]]:
    if max_price is None:
        kept = rows[:limit]
        return kept, []
    kept: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        px = row.get("close")
        if px is None or float(px) > max_price:
            dropped.append(row)
            continue
        kept.append(row)
        if len(kept) >= limit:
            break
    for i, row in enumerate(kept, 1):
        row["rank"] = i
    return kept, dropped


def _chart_payload_to_df(payload: dict) -> pd.DataFrame:
    result = (payload.get("chart") or {}).get("result")
    if not result:
        return pd.DataFrame()
    ts = result[0].get("timestamp") or []
    quote = result[0]["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(TPE),
    )
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return df
    return df.loc[session_mask(df.index)].copy()


def fetch_yahoo_1m(symbol: str, range_: str = "7d", days: int | None = None) -> pd.DataFrame:
    if days is not None and days > 8:
        return fetch_yahoo_1m_days(symbol, days)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1m&range={range_}&includePrePost=false"
    )
    return _chart_payload_to_df(_get_json(url))


def fetch_yahoo_1m_days(symbol: str, days: int = 30, chunk_days: int = 7) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    chunks: list[pd.DataFrame] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1m&period1={int(cur.timestamp())}&period2={int(nxt.timestamp())}"
            f"&includePrePost=false"
        )
        try:
            part = _chart_payload_to_df(_get_json(url))
            if len(part):
                # drop stray bars outside the requested window
                lo, hi = cur.astimezone(TPE), nxt.astimezone(TPE)
                part = part.loc[(part.index >= lo) & (part.index < hi)]
                if len(part):
                    chunks.append(part)
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {symbol} {cur.date()}→{nxt.date()} {exc}", file=sys.stderr)
        cur = nxt
        time.sleep(0.12)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep="last")]


def simulate_scaled(df: pd.DataFrame, sigs, scale: float):
    return simulate(
        df,
        sigs,
        preopen_flat=False,
        ma200_tp_lo=40.0 * scale,
        ma200_tp_hi=55.0 * scale,
        ma200_tp_d60=18.0 * scale,
        ma200_near_over_lo=0.0,
        ma200_near_over_hi=15.0 * scale,
        ma200_near_pts=5.0 * scale,
        ma60_up_near=20.0 * scale,
        ma60_up_min_gap=25.0 * scale,
        ma60_up_buffer=5.0 * scale,
    )


def scan_symbol(row: dict, range_: str, days: int | None = None) -> tuple[list[TwHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0, "n_trade": 0}
    try:
        df = fetch_yahoo_1m(row["symbol"], range_, days=days)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 220:
        meta["error"] = "too_few_bars"
        return [], meta
    px = float(row["close"] or df["Close"].iloc[-1])
    scale = tw_pt_scale(px)
    sigs = detect_signals(df, pt_scale=scale, skip_hour_start=9, skip_hour_end=10)
    trades = simulate_scaled(df, sigs, scale)
    meta["n_sig"] = len(sigs)
    meta["n_trade"] = len(trades)
    return [TwHit(row, t, df) for t in trades], meta


def write_tw_html(path: Path, hits: list[TwHit], universe: list[dict], period: str, date: str) -> Path:
    stats = summarize_trades([h.trade for h in hits])
    cards = []
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        img_name = (
            f"t{i:02d}_{hit.row['code']}_{et.strftime('%m%d_%H%M')}_q{t.quality.lower()}.png"
        )
        label = f"{hit.row['code']} {hit.row['name']}"
        draw_trade_png(df, t, path.parent / "img" / img_name, i, title_extra=label)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.2f}</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            f"<span class='tag'>{escape(t.exit_reason)}</span></div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}  stop {t.stop_price:.2f} (−{risk:.2f})\n"
            f"target {t.target_price:.2f}  exit {t.exit_price:.2f} {t.exit_reason}\n"
            f"破底 {t.signal.break_low:.2f} / 2h低 {t.signal.two_hr_low:.2f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    prices = [r.get("close") for r in universe if r.get("close") is not None]
    px_note = f" · 股價 {min(prices):.0f}–{max(prices):.0f}" if prices else ""
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股破底翻 · 成交額前{len(universe)}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 1m 破底翻 · 成交額前 {len(universe)}</h1>
<p class="muted">{escape(period)} · 基準日 {escape(date)} · {len(universe)} 檔 · 成交額末名約 {cutoff:.1f} 億{escape(px_note)}
<br/>規則同 NQ：破 2h 低、15 根收復 MA20/30、5/10/20 多頭、09–10 不進。點數 × 股價/20000。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{'pnl-win' if stats['total_points']>=0 else 'pnl-loss'}">{stats['total_points']:+.2f}</b></div>
<div class="card">標的<b>{len({h.row['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有通過嚴格過濾的破底翻訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = (
        "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
        f"cursor/nq-1m-ma-reclaim-2484/{rel}/"
    )
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="台股成交額前100 破底翻一週回測")
    p.add_argument("--date", default="", help="YYYYMMDD，預設上一個交易日")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--pool", type=int, default=200, help="先取成交額前 N 再套股價過濾")
    p.add_argument("--max-price", type=float, default=None, help="收盤價超過此值剔除，例如 600")
    p.add_argument("--days", type=int, default=None, help="1m 回看天數；>8 用 7 日切片")
    p.add_argument("--range", dest="range_", default="7d")
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    args = p.parse_args(argv)

    date = args.date or last_tw_session_yyyymmdd()
    pool = max(args.limit, args.pool if args.max_price else args.limit)
    print(
        f"universe date={date} limit={args.limit} pool={pool} "
        f"max_price={args.max_price} days={args.days or args.range_}"
    )
    raw = fetch_top_turnover(date, pool)
    universe, dropped = filter_by_max_price(raw, args.max_price, args.limit)
    if dropped:
        print(
            f"drop price>{args.max_price}: "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:12])
            + (" …" if len(dropped) > 12 else "")
        )
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
        f"{universe[0]['amount']/1e8:.1f}億 / {universe[0]['close']} · "
        f"末 {universe[-1]['code']} {universe[-1]['amount']/1e8:.1f}億 / {universe[-1]['close']}"
    )

    hits: list[TwHit] = []
    errors = 0
    scanned = 0
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol(row, args.range_, days=args.days)
        scanned += 1
        if meta["error"]:
            errors += 1
        hits.extend(stock_hits)
        flag = f" trades={meta['n_trade']}" if meta["n_trade"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row['name']} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    stats = summarize_trades([h.trade for h in hits])
    print(
        f"done scanned={scanned} errors={errors} trades={stats['count']} "
        f"WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.2f}"
    )
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        ts = hit.df.index[t.entry_idx]
        print(
            f"  [{i}] {hit.row['code']} {hit.row['name']} Q{t.quality} "
            f"{ts.strftime('%m-%d %H:%M')} {t.exit_reason} {t.pnl_points:+.2f}"
        )

    html_path = Path(args.html) if args.html else None
    if html_path is None and args.pages:
        if args.days and args.days > 8:
            html_path = REPO / "docs" / "tw-ma-reclaim-30d" / "index.html"
        else:
            html_path = PAGES
    if html_path:
        period_label = f"{args.days}d" if args.days else args.range_
        if args.max_price is not None:
            period_label += f" · 股價≤{args.max_price:g}"
        out = write_tw_html(html_path, hits, universe, period_label, date)
        write_view_html(out)
        print(f"html={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
