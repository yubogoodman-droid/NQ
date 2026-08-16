#!/usr/bin/env python3
"""回測台股指定日成交額前 N 檔：1 分 K 收盤剛站上 MA200，且 MA5>MA10>MA20。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.scan_tw_top50_spring import fetch_top_turnover, fetch_yahoo_1m, tw_tick_size
from nq.backtest import run_backtest
from nq.ma_align import MaAlignPattern, detect_ma_align_alerts
from nq.strategy import MaAlignStrategy


def _local(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts
    return ts.tz_convert("Asia/Taipei")


def scan_stock(row: dict, day: str, max_bars_hold: int = 30) -> dict:
    df = fetch_yahoo_1m(row["symbol"])
    out = {**row, "bars": int(len(df)), "signals": [], "alerts": []}
    if df.empty:
        out["error"] = "no_1m_data"
        return out

    last_close = float(df["close"].iloc[-1])
    strategy = MaAlignStrategy(tick_size=tw_tick_size(last_close))
    alerts = detect_ma_align_alerts(df)
    signals = strategy.generate_signals(df)
    results = run_backtest(df, strategy, max_bars_hold=max_bars_hold)
    by_bar = {r.signal.bar_idx: r for r in results}

    day_alerts: list[str] = []
    for pattern in alerts:
        local = _local(df.index[pattern.bar_idx])
        if local.strftime("%Y-%m-%d") != day:
            continue
        day_alerts.append(local.strftime("%H:%M"))
    out["alerts"] = day_alerts

    day_signals = []
    for sig in signals:
        local = _local(sig.timestamp)
        if local.strftime("%Y-%m-%d") != day:
            continue
        assert isinstance(sig.pattern, MaAlignPattern)
        trade = by_bar.get(sig.bar_idx)
        day_signals.append(
            {
                "time": local.strftime("%H:%M"),
                "entry": sig.entry,
                "stop": sig.stop_loss,
                "target": sig.target,
                "ma5": round(sig.pattern.ma5, 4),
                "ma10": round(sig.pattern.ma10, 4),
                "ma20": round(sig.pattern.ma20, 4),
                "ma200": round(sig.pattern.ma200, 4),
                "close": round(sig.pattern.close, 4),
                "exit": None if trade is None else trade.exit_reason,
                "pnl": None if trade is None else round(trade.pnl_points, 2),
            }
        )
    out["signals"] = day_signals
    out["day_bars"] = int(sum(1 for t in df.index if _local(t).strftime("%Y-%m-%d") == day))
    out["n_trades_5d"] = len(results)
    out["pnl_5d"] = round(sum(t.pnl_points for t in results), 2)
    out["wins_5d"] = sum(1 for t in results if t.exit_reason == "take_profit")
    out["stops_5d"] = sum(1 for t in results if t.exit_reason == "stop_loss")
    out["timeouts_5d"] = sum(1 for t in results if t.exit_reason == "time_stop")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="台股成交額前 N 檔 1 分 K 多頭排列通知")
    parser.add_argument("--date", default="20260814", help="YYYYMMDD")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--max-bars-hold", type=int, default=30)
    args = parser.parse_args()
    day = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"

    universe = fetch_top_turnover(args.date, args.limit)
    print(f"{args.date} 成交額前 {len(universe)} 檔（上市+上櫃）  1分K Yahoo range=5d")
    hits: list[dict] = []
    missing = 0
    day_wins = day_stops = day_timeouts = 0
    day_pnl = 0.0
    day_trades = 0
    for row in universe:
        result = scan_stock(row, day, max_bars_hold=args.max_bars_hold)
        time.sleep(args.sleep)
        tag = f"{row['rank']:2} {row['code']:7} {row['name']}"
        if result.get("error"):
            missing += 1
            print(f"{tag} | 無 1 分 K")
            continue
        amt = row["amount"] / 1e8
        n = len(result["alerts"])
        if n == 0:
            print(f"{tag} | 成交 {amt:7.2f} 億 | 無通知 | 1分K {result['day_bars']}")
            continue
        hits.append(result)
        times = ", ".join(result["alerts"])
        print(f"{tag} | 成交 {amt:7.2f} 億 | 通知 {n} 次 {times} | 1分K {result['day_bars']}")
        for sig in result["signals"]:
            if sig["pnl"] is not None:
                day_trades += 1
                day_pnl += sig["pnl"]
                if sig["exit"] == "take_profit":
                    day_wins += 1
                elif sig["exit"] == "stop_loss":
                    day_stops += 1
                elif sig["exit"] == "time_stop":
                    day_timeouts += 1
            exit_s = sig["exit"] or "未進場"
            pnl_s = "-" if sig["pnl"] is None else f"{sig['pnl']:+.2f}"
            print(
                f"     {sig['time']} 做多 {sig['entry']:.2f} 停 {sig['stop']:.2f} "
                f"目標 {sig['target']:.2f} | MA5 {sig['ma5']:.2f} > MA10 {sig['ma10']:.2f} "
                f"> MA20 {sig['ma20']:.2f} | 收 {sig['close']:.2f} > MA200 {sig['ma200']:.2f} | "
                f"{exit_s} {pnl_s}"
            )

    print("\n=== 摘要 ===")
    print(
        f"掃描 {len(universe)} 檔，缺資料 {missing}，當日新通知 {len(hits)} 檔、"
        f"{sum(len(h['alerts']) for h in hits)} 次"
    )
    if hits:
        print("有通知：", "、".join(f"{h['code']}{h['name']}({','.join(h['alerts'])})" for h in hits))
        print(
            f"當日回測（停損 MA20、2R、最多 {args.max_bars_hold} 根）："
            f"{day_trades} 筆 贏 {day_wins} 停 {day_stops} 逾 {day_timeouts}  合計 PnL {day_pnl:.2f}"
        )


if __name__ == "__main__":
    main()
