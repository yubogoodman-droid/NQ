"""Deprecated: stricter 爆量 (2.0×) live scanner."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from shadow_neckline_logic import STRICT, detect_at_index, prepare_indicators

TG_TOKEN = ""
TG_CHAT_ID = ""
MIN_VOLUME_USDT = 50_000_000
SCAN_INTERVAL = 60
PARAMS = STRICT


def send_tg_message(message: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[TG skipped]\n{message}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"❌ TG 發送失敗: {e}")


def main():
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    print(f"📡 原版+強爆量(≥{PARAMS.min_vol_ratio}×) 監控中...")
    last_report_time = {}
    while True:
        try:
            tickers = exchange.fetch_tickers()
            symbols = [
                s
                for s, t in tickers.items()
                if s.endswith(":USDT") and (t.get("quoteVolume") or 0) >= MIN_VOLUME_USDT
            ]
            now = datetime.now()
            for symbol in symbols:
                try:
                    if symbol in last_report_time and now < last_report_time[
                        symbol
                    ] + timedelta(minutes=PARAMS.cooldown_min):
                        continue
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    ok, d = detect_at_index(*prepare_indicators(df), len(df) - 1, PARAMS)
                    if not ok:
                        time.sleep(0.05)
                        continue
                    chg24 = (tickers.get(symbol) or {}).get("percentage")
                    send_tg_message(
                        f"🚨 *【影線頸線｜強爆量】*\n\n"
                        f"💎 `{symbol.split(':')[0]}`\n"
                        f"📈 24h `{None if chg24 is None else round(float(chg24),2)}%`\n"
                        f"💰 `{d['price']}`  爆量 `{d.get('vol_ratio')}×`"
                    )
                    print(f"🎯 {symbol} vol={d.get('vol_ratio')}×")
                    last_report_time[symbol] = now
                    time.sleep(0.05)
                except Exception:
                    continue
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            print(f"❌ {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
