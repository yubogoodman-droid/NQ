"""
影線頸線監控（原版 + 爆量）— 建議日常使用

原版邏輯不變：
- 三峰肩頭、頭相對 SMA200 乖離 ≥5%、肩距 ≤30、對稱 <15%
- Low 刺破頸線 + SMA14，且高點未破頭

新增：
- 破位 K 量能 ≥ 3 × 近 20 根均量（爆量）
- 拒絕上升頸線（右肩高於左肩）
- 拒絕貼近上彎 SMA200（|距SMA200|<1.5%）
- 拒絕 15分K 戳破 200均線且收在下方
- 拒絕貼近 15m SMA200（|距|<1.5%）
- 拒絕已深跌破 15m SMA200（dist < −3%，如 GWEI）
- 拒絕貼近 SMA99（|距|<1.5%，如 CYS）
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from shadow_neckline_logic import VOLUME, detect_at_index, prepare_indicators

# ================= 設定區域 =================
TG_TOKEN = ""
TG_CHAT_ID = ""
MIN_VOLUME_USDT = 50_000_000
SCAN_INTERVAL = 60
PARAMS = VOLUME
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
        f"📡 原版+爆量 監控中... cooldown={PARAMS.cooldown_min}m "
        f"vol≥{PARAMS.min_vol_ratio:.0f}×{PARAMS.vol_lookback} "
        f"reject_rising_neck={PARAMS.reject_rising_neck} "
        f"reject_near_rising_sma200={PARAMS.reject_near_rising_sma200}"
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

                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    arrays = prepare_indicators(df)
                    ok, d = detect_at_index(*arrays, len(df) - 1, PARAMS)
                    if not ok:
                        time.sleep(0.05)
                        continue

                    t = tickers.get(symbol) or {}
                    chg24 = t.get("percentage")
                    msg = (
                        f"🚨 *【影線頸線｜原版+爆量】*\n\n"
                        f"💎 `{symbol.split(':')[0]}` (5M)\n"
                        f"📈 24h: `{None if chg24 is None else round(float(chg24),2)}%`\n"
                        f"💰 `{d['price']}`  刺破 `{d['close_break_pct']}%`\n"
                        f"📊 乖離 `{d['bias']}%`  爆量 `{d.get('vol_ratio')}×`  "
                        f"頸線 `{d.get('neck_chg_pct')}%`\n"
                        f"⚠️ Low 破頸線+SMA14，爆量，且非上升頸線"
                    )
                    send_tg_message(msg)
                    print(f"🎯 {symbol} vol={d.get('vol_ratio')}× bias={d['bias']}%")
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
