#!/usr/bin/env python3
"""回測台股指定日成交額前 N 檔的假跌破後上拉訊號（1 分 K）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.backtest import run_backtest
from nq.spring import FakeBreakdownPattern
from nq.strategy import FakeBreakdownStrategy

TW = timezone(timedelta(hours=8))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _get_json(url: str, retries: int = 4) -> dict | list:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
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


def tw_tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def yahoo_symbol(code: str, market: str) -> str:
    return f"{code}.TW" if market == "tse" else f"{code}.TWO"


def list_trading_days(end: str, n: int) -> list[str]:
    """從 end（YYYYMMDD）往回找 n 個上市交易日。"""
    day = datetime.strptime(end, "%Y%m%d")
    found: list[str] = []
    for _ in range(21):
        if day.weekday() < 5:
            ymd = day.strftime("%Y%m%d")
            payload = _get_json(
                f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={ymd}&type=IND&response=json"
            )
            if payload.get("stat") == "OK":
                found.append(ymd)
                if len(found) >= n:
                    break
        day -= timedelta(days=1)
    return list(reversed(found))


def _tpex_day_quotes(date: str) -> list[tuple[int, str, str, str, float | None]]:
    """date: YYYYMMDD。上櫃每日行情（含歷史日）。"""
    ymd = f"{date[:4]}/{date[4:6]}/{date[6:]}"
    payload = _get_json(
        f"https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?date={ymd}&id=&response=json"
    )
    tables = payload.get("tables") or []
    if not tables:
        return []
    rows = tables[0].get("data") or []
    items: list[tuple[int, str, str, str, float | None]] = []
    for rec in rows:
        code, name = rec[0].strip(), rec[1].strip()
        amt = _num(rec[9])
        if amt > 0:
            items.append((amt, code, name, "otc", _price(rec[2])))
    return items


def fetch_top_turnover(date: str, limit: int) -> list[dict]:
    """date: YYYYMMDD。上市 + 上櫃，依成交金額排序。"""
    twse = _get_json(
        f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type=ALLBUT0999&response=json"
    )
    if twse.get("stat") != "OK":
        raise RuntimeError(f"TWSE stat={twse.get('stat')}")
    items: list[tuple[int, str, str, str, float | None]] = []
    for rec in twse["tables"][8]["data"]:
        code, name = rec[0].strip(), rec[1].strip()
        amt = _num(rec[4])
        if amt > 0:
            items.append((amt, code, name, "tse", _price(rec[8])))
    items.extend(_tpex_day_quotes(date))

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


def fetch_yahoo_1m(symbol: str, range: str = "5d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1m&range={range}&includePrePost=false"
    )
    payload = _get_json(url)
    result = (payload.get("chart") or {}).get("result")
    if not result:
        return pd.DataFrame()
    ts = result[0].get("timestamp") or []
    quote = result[0]["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Taipei"),
    )
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _local_ts(ts) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is None:
        return ts
    return ts.tz_convert("Asia/Taipei")


def _signal_record(sig, trade, local) -> dict:
    assert isinstance(sig.pattern, FakeBreakdownPattern)
    risk = sig.entry - sig.stop_loss
    pnl = None if trade is None else round(trade.pnl_points, 2)
    r_mult = None if pnl is None or risk <= 0 else round(pnl / risk, 2)
    return {
        "date": local.strftime("%Y-%m-%d"),
        "time": local.strftime("%H:%M"),
        "entry": sig.entry,
        "stop": sig.stop_loss,
        "target": sig.target,
        "support": round(sig.pattern.support, 4),
        "resistance": round(sig.pattern.resistance, 4),
        "spring_low": round(sig.pattern.spring_low, 4),
        "break_pct": round(sig.pattern.break_pct * 100, 2),
        "vol_ratio": round(sig.pattern.volume_ratio, 2),
        "exit": None if trade is None else trade.exit_reason,
        "pnl": pnl,
        "r": r_mult,
    }


def scan_stock(
    row: dict,
    day: str,
    strategy_kwargs: dict,
    *,
    df: pd.DataFrame | None = None,
    days: list[str] | None = None,
    yahoo_range: str = "5d",
) -> dict:
    if df is None:
        df = fetch_yahoo_1m(row["symbol"], range=yahoo_range)
    keep_days = set(days or [day])
    out = {**row, "bars": int(len(df)), "signals": []}
    if df.empty:
        out["error"] = "no_1m_data"
        return out
    last_close = float(df["close"].iloc[-1])
    strategy = FakeBreakdownStrategy(tick_size=tw_tick_size(last_close), **strategy_kwargs)
    signals = strategy.generate_signals(df)
    results = run_backtest(df, strategy, max_bars_hold=60)
    by_bar = {r.signal.bar_idx: r for r in results}
    day_signals = []
    for sig in signals:
        local = _local_ts(sig.timestamp)
        if local.strftime("%Y-%m-%d") not in keep_days:
            continue
        trade = by_bar.get(sig.bar_idx)
        day_signals.append(_signal_record(sig, trade, local))
    out["signals"] = day_signals
    out["friday_bars"] = int((df.index.strftime("%Y-%m-%d") == day).sum())
    return out


def _filter_max_price(universe: list[dict], max_price: float | None) -> tuple[list[dict], list[dict]]:
    if max_price is None:
        return universe, []
    kept, skipped = [], []
    for row in universe:
        close = row.get("close")
        if close is not None and close > max_price:
            skipped.append(row)
        else:
            kept.append(row)
    return kept, skipped


def _trade_stats(trades: list[dict]) -> dict:
    wins = [t for t in trades if t.get("exit") == "take_profit"]
    losses = [t for t in trades if t.get("exit") == "stop_loss"]
    times = [t for t in trades if t.get("exit") == "time_stop"]
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    rs = [t["r"] for t in trades if t.get("r") is not None]
    return {
        "signals": len(trades),
        "tp": len(wins),
        "sl": len(losses),
        "time": len(times),
        "pnl_sum": round(sum(pnls), 2) if pnls else 0,
        "pnl_avg": round(sum(pnls) / len(pnls), 2) if pnls else 0,
        "r_sum": round(sum(rs), 2) if rs else 0,
        "r_avg": round(sum(rs) / len(rs), 2) if rs else 0,
    }


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"{label}：{stats['signals']} 筆  TP {stats['tp']} / SL {stats['sl']} / TIME {stats['time']}"
        + (
            f"，合計 {stats['pnl_sum']:+.2f} 點（平均 {stats['pnl_avg']:+.2f}）"
            f"，{stats['r_sum']:+.2f}R（平均 {stats['r_avg']:+.2f}R）"
            if stats["signals"]
            else ""
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="台股成交額前 N 檔假跌破回測")
    parser.add_argument("--date", default="20260814", help="YYYYMMDD，預設上週五；搭配 --days 為結束日")
    parser.add_argument("--days", type=int, default=1, help="往回幾個交易日，例如 5 回測一週")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-price", type=float, default=None, help="收盤價超過此值則剔除，例如 700")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--yahoo-range", default="", help="Yahoo 1 分 K 區間，預設 5d；多日自動 8d")
    parser.add_argument(
        "--after",
        default="09:00",
        help="只計此時之後的進場（開盤雜訊已由 skip_open_minutes 處理）",
    )
    parser.add_argument("--json-out", default="", help="把掃描結果寫成 JSON")
    args = parser.parse_args()
    ymds = list_trading_days(args.date, max(1, args.days)) if args.days > 1 else [args.date]
    iso_days = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in ymds]
    yahoo_range = args.yahoo_range or ("8d" if len(ymds) > 1 else "5d")
    print_empty = len(ymds) == 1

    universes: dict[str, list[dict]] = {}
    skipped_by_day: dict[str, list[dict]] = {}
    allowed: dict[str, set[str]] = {}
    unique: dict[str, dict] = {}
    for ymd, iso in zip(ymds, iso_days):
        raw = fetch_top_turnover(ymd, args.limit)
        kept, skipped = _filter_max_price(raw, args.max_price)
        universes[iso] = kept
        skipped_by_day[iso] = skipped
        allowed[iso] = {r["code"] for r in kept}
        for row in kept:
            unique[row["code"]] = row
        print(
            f"{ymd} 成交額前 {args.limit}"
            + (f"，剔除收盤 > {args.max_price:g} 共 {len(skipped)} 檔" if args.max_price is not None else "")
            + f"，掃描 {len(kept)} 檔"
        )

    print(f"期間 {iso_days[0]}～{iso_days[-1]}，不重複 {len(unique)} 檔，Yahoo {yahoo_range}")
    hits: list[dict] = []
    missing = 0
    scanned = 0
    for row in unique.values():
        try:
            result = scan_stock(
                row,
                iso_days[-1],
                {},
                days=iso_days,
                yahoo_range=yahoo_range,
            )
        except Exception as exc:  # noqa: BLE001
            missing += 1
            print(f"{row['code']:7} {row['name']} | 抓 1 分 K 失敗：{exc}")
            continue
        time.sleep(args.sleep)
        tag = f"{row['code']:7} {row['name']}"
        if result.get("error"):
            missing += 1
            print(f"{tag} | 無 1 分 K")
            continue
        scanned += 1
        kept = [
            s
            for s in result["signals"]
            if s["time"] >= args.after and s["date"] in allowed and row["code"] in allowed[s["date"]]
        ]
        result["signals"] = kept
        if not kept:
            if print_empty:
                amt = row["amount"] / 1e8
                close = row.get("close")
                px = f"收 {close:g}" if close is not None else "收 --"
                print(f"{tag} | 成交 {amt:7.2f} 億 | {px} | 無訊號 | 1分K {result['friday_bars']}")
            continue
        hits.append(result)
        for sig in kept:
            rtxt = "" if sig.get("r") is None else f" {sig['r']:+.2f}R"
            print(
                f"{tag} | {sig['date']} {sig['time']} 做多 {sig['entry']:.2f} "
                f"停 {sig['stop']:.2f} 目標 {sig['target']:.2f} | "
                f"跌破 {sig['break_pct']}% 量 {sig['vol_ratio']}x | "
                f"{sig['exit']} {sig['pnl']:+.2f}{rtxt}"
            )

    trades = [s for h in hits for s in h["signals"]]
    overall = _trade_stats(trades)
    print("\n=== 摘要 ===")
    print(
        f"交易日 {len(ymds)} 天，不重複掃描 {scanned} 檔，缺資料 {missing}，"
        f"有訊號 {len(hits)} 檔、{overall['signals']} 筆"
    )
    _print_stats("合計", overall)
    by_day = {}
    for iso in iso_days:
        day_trades = [t for t in trades if t["date"] == iso]
        stats = _trade_stats(day_trades)
        by_day[iso] = stats
        _print_stats(iso, stats)
    if hits:
        print("有訊號：", "、".join(f"{h['code']}{h['name']}" for h in hits))
    if args.json_out:
        out = {
            "dates": iso_days,
            "limit": args.limit,
            "max_price": args.max_price,
            "skipped_by_day": {k: [{"code": r["code"], "name": r["name"], "close": r.get("close")} for r in v] for k, v in skipped_by_day.items()},
            "missing": missing,
            "hits": hits,
            "summary": {"scanned": scanned, **overall},
            "by_day": by_day,
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {path.resolve()}")


if __name__ == "__main__":
    main()
