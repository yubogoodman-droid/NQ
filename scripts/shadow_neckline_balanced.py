"""
影線頸線監控（Balanced）— 建議日常使用

邏輯見 shadow_neckline_logic.BALANCED：
- 收盤破頸線 + SMA14、收陰、破位≥0.3%
- SMA25 軟條件
- |ΔSMA99|<2% 不空；仍在 SMA99 上方且距離<8% 不空
- |ΔSMA200|<2% 不空；仍在 SMA200 上方且距離<8% 不空
- 冷卻 60m，24h 漲幅上限 300%
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from shadow_neckline_logic import BALANCED, detect_at_index, prepare_indicators

# ================= 設定區域 =================
TG_TOKEN = ""
TG_CHAT_ID = ""
MIN_VOLUME_USDT = 50_000_000
SCAN_INTERVAL = 60
PARAMS = BALANCED
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
        f"📡 Balanced 監控中... cooldown={PARAMS.cooldown_min}m "
        f"|ΔSMA99|/|ΔSMA200|>={PARAMS.min_abs_dist_ma99*100:.0f}% "
        f"上方≥{PARAMS.max_near_above_ma99*100:.0f}%"
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
                        f"🚨 *【影線頸線｜Balanced】*\n\n"
                        f"💎 `{symbol.split(':')[0]}` (5M)\n"
                        f"📈 24h: `{None if chg24 is None else round(float(chg24),2)}%`\n"
                        f"💰 `{d['price']}`  破位 `{d['close_break_pct']}%`\n"
                        f"📊 乖離 `{d['bias']}%`  頸線 `{d['line_val']}`\n"
                        f"📏 距SMA99 `{d['dist_ma99_pct']}%` · 距SMA200 `{d['dist_ma200_pct']}%`\n"
                        f"⚠️ 收盤破位且遠離 SMA99/SMA200"
                    )
                    send_tg_message(msg)
                    print(
                        f"🎯 {symbol} dist99={d['dist_ma99_pct']}% "
                        f"dist200={d['dist_ma200_pct']}%"
                    )
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
