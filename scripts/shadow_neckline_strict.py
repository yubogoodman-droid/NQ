"""
影線頸線監控（Strict）

在 Balanced 共用 SMA99 規則上再加嚴：
- 破位深度 ≥ 0.8%
- 前一根已收在頸線下
- close < SMA25 且 SMA14 < SMA25
- 肩距 12~60、對稱 < 10%
- 冷卻 150m
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from shadow_neckline_logic import STRICT, detect_at_index, prepare_indicators

# ================= 設定區域 =================
TG_TOKEN = ""
TG_CHAT_ID = ""
MIN_VOLUME_USDT = 50_000_000
SCAN_INTERVAL = 60
PARAMS = STRICT
# ===========================================


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
    print(
        f"📡 Strict 監控中... cooldown={PARAMS.cooldown_min}m "
        f"prevConfirm + SMA14<25 + SMA99 rules"
    )
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
            print(f"⏰ {now.strftime('%H:%M:%S')} | 掃描 {len(symbols)} 幣...")

            for symbol in symbols:
                try:
                    if symbol in last_report_time and now < last_report_time[
                        symbol
                    ] + timedelta(minutes=PARAMS.cooldown_min):
                        continue
                    t = tickers.get(symbol) or {}
                    chg24 = t.get("percentage")
                    if chg24 is not None and float(chg24) > PARAMS.max_chg24 * 100:
                        continue

                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    arrays = prepare_indicators(df)
                    ok, d = detect_at_index(*arrays, len(df) - 1, PARAMS)
                    if not ok:
                        time.sleep(0.05)
                        continue

                    msg = (
                        f"🚨 *【影線頸線｜Strict】*\n\n"
                        f"💎 `{symbol.split(':')[0]}` (5M)\n"
                        f"📈 24h: `{None if chg24 is None else round(float(chg24),2)}%`\n"
                        f"💰 `{d['price']}`  破位 `{d['close_break_pct']}%`\n"
                        f"📊 乖離 `{d['bias']}%`  頸線 `{d['line_val']}`\n"
                        f"📏 距SMA99 `{d['dist_ma99_pct']}%`\n"
                        f"⚠️ 前K確認 + SMA14&lt;25 + 遠離SMA99"
                    )
                    send_tg_message(msg)
                    print(f"🎯 {symbol} dist99={d['dist_ma99_pct']}%")
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
