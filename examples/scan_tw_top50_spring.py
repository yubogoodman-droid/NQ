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


def fetch_top_turnover(date: str, limit: int) -> list[dict]:
    """date: YYYYMMDD。上市 + 上櫃，依成交金額排序。"""
    twse = _get_json(
        f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type=ALLBUT0999&response=json"
    )
    if twse.get("stat") != "OK":
        raise RuntimeError(f"TWSE stat={twse.get('stat')}")
    items: list[tuple[int, str, str, str]] = []
    for rec in twse["tables"][8]["data"]:
        code, name = rec[0].strip(), rec[1].strip()
        amt = _num(rec[4])
        if amt > 0:
            items.append((amt, code, name, "tse"))

    roc = f"{int(date[:4]) - 1911}{date[4:]}"
    tpex = _get_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes")
    for rec in tpex:
        if rec.get("Date") not in {roc, date}:
            continue
        code = str(rec["SecuritiesCompanyCode"]).strip()
        name = str(rec["CompanyName"]).strip()
        amt = _num(rec.get("TransactionAmount") or 0)
        if amt > 0:
            items.append((amt, code, name, "otc"))

    best: dict[str, tuple[int, str, str, str]] = {}
    for amt, code, name, mkt in items:
        prev = best.get(code)
        if prev is None or amt > prev[0]:
            best[code] = (amt, code, name, mkt)

    ranked = sorted(best.values(), reverse=True)[:limit]
    return [
        {"rank": i, "code": code, "name": name, "market": mkt, "amount": amt, "symbol": yahoo_symbol(code, mkt)}
        for i, (amt, code, name, mkt) in enumerate(ranked, 1)
    ]


def fetch_yahoo_1m(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=5d&includePrePost=false"
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


def scan_stock(row: dict, day: str, strategy_kwargs: dict) -> dict:
    df = fetch_yahoo_1m(row["symbol"])
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
        ts = sig.timestamp
        if getattr(ts, "tzinfo", None) is None:
            local = ts
        else:
            local = ts.tz_convert("Asia/Taipei")
        if local.strftime("%Y-%m-%d") != day:
            continue
        assert isinstance(sig.pattern, FakeBreakdownPattern)
        trade = by_bar.get(sig.bar_idx)
        day_signals.append(
            {
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
                "pnl": None if trade is None else round(trade.pnl_points, 2),
            }
        )
    out["signals"] = day_signals
    out["friday_bars"] = int((df.index.strftime("%Y-%m-%d") == day).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="台股成交額前 N 檔假跌破回測")
    parser.add_argument("--date", default="20260814", help="YYYYMMDD，預設上週五")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument(
        "--after",
        default="09:00",
        help="只計此時之後的進場（開盤雜訊已由 skip_open_minutes 處理）",
    )
    args = parser.parse_args()
    day = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"

    universe = fetch_top_turnover(args.date, args.limit)
    print(f"{args.date} 成交額前 {len(universe)} 檔（上市+上櫃）")
    hits: list[dict] = []
    missing = 0
    for row in universe:
        result = scan_stock(row, day, {})
        time.sleep(args.sleep)
        tag = f"{row['rank']:2} {row['code']:7} {row['name']}"
        if result.get("error"):
            missing += 1
            print(f"{tag} | 無 1 分 K")
            continue
        kept = [s for s in result["signals"] if s["time"] >= args.after]
        result["signals"] = kept
        n = len(kept)
        amt = row["amount"] / 1e8
        if n == 0:
            print(f"{tag} | 成交 {amt:7.2f} 億 | 無訊號 | 1分K {result['friday_bars']}")
            continue
        hits.append(result)
        for sig in kept:
            print(
                f"{tag} | 成交 {amt:7.2f} 億 | {sig['time']} 做多 {sig['entry']:.2f} "
                f"停 {sig['stop']:.2f} 目標 {sig['target']:.2f} | "
                f"跌破 {sig['break_pct']}% 量 {sig['vol_ratio']}x | "
                f"{sig['exit']} {sig['pnl']:+.2f}"
            )

    print("\n=== 摘要 ===")
    print(f"掃描 {len(universe)} 檔，缺資料 {missing}，有訊號 {len(hits)} 檔、{sum(len(h['signals']) for h in hits)} 筆")
    if hits:
        print("有訊號：", "、".join(f"{h['code']}{h['name']}" for h in hits))


if __name__ == "__main__":
    main()
