#!/usr/bin/env python3
"""掃描台股成交額前 N 檔：日線 MA5/10/20 多頭排列且收盤站上 MA200 時跳通知。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.scan_tw_top50_spring import fetch_top_turnover, tw_tick_size
from nq.backtest import run_backtest, summarize
from nq.ma_align import add_daily_mas, is_aligned
from nq.strategy import MaAlignStrategy

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
TW = timezone(timedelta(hours=8))


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


def fetch_yahoo_daily(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2y&includePrePost=false"
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


def _local_date(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("Asia/Taipei") if getattr(ts, "tzinfo", None) else ts
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def scan_stock(row: dict, day: str) -> dict:
    df = fetch_yahoo_daily(row["symbol"])
    out = {**row, "bars": int(len(df)), "alert": None, "aligned": False, "history": []}
    if df.empty or len(df) < 200:
        out["error"] = "no_daily_data"
        return out
    last_close = float(df["close"].iloc[-1])
    strategy = MaAlignStrategy(tick_size=tw_tick_size(last_close))
    trades = run_backtest(df, strategy, max_bars_hold=40)
    by_bar = {t.signal.bar_idx: t for t in trades}
    work = add_daily_mas(df)
    last = work.iloc[-1]
    out["aligned"] = bool(is_aligned(last))
    out["close"] = float(last["close"])
    out["ma5"] = None if pd.isna(last["ma5"]) else float(last["ma5"])
    out["ma10"] = None if pd.isna(last["ma10"]) else float(last["ma10"])
    out["ma20"] = None if pd.isna(last["ma20"]) else float(last["ma20"])
    out["ma200"] = None if pd.isna(last["ma200"]) else float(last["ma200"])

    hist = []
    for sig in strategy.generate_signals(df):
        trade = by_bar.get(sig.bar_idx)
        rec = {
            "date": _local_date(sig.timestamp),
            "entry": sig.entry,
            "stop": sig.stop_loss,
            "target": sig.target,
            "ma5": round(sig.pattern.ma5, 2),
            "ma10": round(sig.pattern.ma10, 2),
            "ma20": round(sig.pattern.ma20, 2),
            "ma200": round(sig.pattern.ma200, 2),
            "exit": None if trade is None else trade.exit_reason,
            "pnl": None if trade is None else round(trade.pnl_points, 2),
        }
        hist.append(rec)
        if rec["date"] == day:
            out["alert"] = rec
    out["history"] = hist
    out["backtest"] = summarize(trades)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="成交額前 N 檔：5/10/20 多頭且站上 200 日")
    parser.add_argument("--date", default="20260814", help="YYYYMMDD，通知日（成交額排名日）")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    day = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"

    universe = fetch_top_turnover(args.date, args.limit)
    print(f"{args.date} 成交額前 {len(universe)} 檔 · 日線 MA5>MA10>MA20 且收盤>MA200")
    alerts: list[dict] = []
    holding: list[dict] = []
    missing = 0
    all_trades: list = []
    for row in universe:
        result = scan_stock(row, day)
        time.sleep(args.sleep)
        tag = f"{row['rank']:2} {row['code']:7} {row['name']}"
        amt = row["amount"] / 1e8
        if result.get("error"):
            missing += 1
            print(f"{tag} | 成交 {amt:7.2f} 億 | 無日線")
            continue
        bt = result["backtest"]
        all_trades.append(bt)
        if result["alert"]:
            alerts.append(result)
            a = result["alert"]
            print(
                f"{tag} | 成交 {amt:7.2f} 億 | 🔔 通知 {a['date']} 收 {a['entry']:.2f} "
                f"MA {a['ma5']:.1f}/{a['ma10']:.1f}/{a['ma20']:.1f}  200日 {a['ma200']:.1f} "
                f"停 {a['stop']:.2f} 目標 {a['target']:.2f} | {a['exit']} {a['pnl']:+.2f}"
            )
        elif result["aligned"]:
            holding.append(result)
            print(
                f"{tag} | 成交 {amt:7.2f} 億 | 已排列站上200  收 {result['close']:.2f} "
                f"MA {result['ma5']:.1f}/{result['ma10']:.1f}/{result['ma20']:.1f}  200日 {result['ma200']:.1f}"
            )
        else:
            print(f"{tag} | 成交 {amt:7.2f} 億 | 未達條件 | 歷史通知 {len(result['history'])} 次")

    hist_n = sum(int(x.get("trades") or 0) for x in all_trades)
    hist_pnl = sum(float(x.get("total_pnl_points") or 0) for x in all_trades)
    print("\n=== 通知 ===")
    if alerts:
        print("今日跳通知：", "、".join(f"{h['code']}{h['name']}" for h in alerts))
    else:
        print(f"{day} 成交額前 {len(universe)} 檔沒有新通知（條件要「今天才成立」）。")
    if holding:
        print("已是多頭排列且站上200日：", "、".join(f"{h['code']}{h['name']}" for h in holding))
    print(
        f"掃描 {len(universe)} 檔，缺資料 {missing}，今日通知 {len(alerts)} 檔，"
        f"已排列 {len(holding)} 檔；近 2 年歷史訊號 {hist_n} 筆、合計 {hist_pnl:+.1f} 點"
    )


if __name__ == "__main__":
    main()
