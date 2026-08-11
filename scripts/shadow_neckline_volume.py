"""
影線頸線監控（原版 + 爆量）— 建議日常使用

選幣：Binance USDT-M「24h 漲幅榜前 10」（ticker.percentage）

結構 + 過濾見 shadow_neckline_logic.VOLUME：
- 破位 K 量能 ≥ 2.5 × 近 20 根均量（爆量）
- 拒絕上升頸線（右肩高於左肩）
- 拒絕貼近上彎 SMA200（|距SMA200|<4%）
- 拒絕 15分K 戳破 200均線且收在下方
- 拒絕貼近 15m SMA200（|距|<1.5%）
- 拒絕已深跌破 15m SMA200（dist < −3%）
- 拒絕貼近 SMA99（|距|<1.5%）
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
TOP_N = 10
SCAN_INTERVAL = 60
PARAMS = VOLUME
# ===========================================


def send_tg_message(message: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[TG skipped] {message}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ TG 發送失敗: {e}")


def top_gainers(tickers: dict, n: int = 10) -> list[tuple[str, float]]:
    """USDT 永續，依 24h 漲跌幅% 由高到低取前 n。"""
    rows = []
    for symbol, t in tickers.items():
        if not symbol.endswith(":USDT"):
            continue
        pct = t.get("percentage")
        if pct is None:
            last = t.get("last")
            open_ = t.get("open")
            if last and open_:
                pct = (last / open_ - 1.0) * 100.0
            else:
                continue
        rows.append((symbol, float(pct)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]


def main():
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    print(
        f"📡 原版+爆量 監控中... Top{TOP_N}漲幅榜 "
        f"cooldown={PARAMS.cooldown_min}m "
        f"vol≥{PARAMS.min_vol_ratio:.1f}×{PARAMS.vol_lookback} "
        f"reject_rising_neck={PARAMS.reject_rising_neck} "
        f"reject_near_rising_sma200={PARAMS.reject_near_rising_sma200}"
    )
    last_report_time = {}

    while True:
        try:
            tickers = exchange.fetch_tickers()
            ranked = top_gainers(tickers, TOP_N)
            now = datetime.now()
            preview = ", ".join(f"{s.split(':')[0]} {p:+.1f}%" for s, p in ranked)
            print(f"⏰ {now.strftime('%H:%M:%S')} | Top{TOP_N}: {preview}")

            for symbol, pct in ranked:
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

                    msg = (
                        f"🚨 *【影線頸線｜漲幅榜Top{TOP_N}+爆量】*\n\n"
                        f"💎 `{symbol.split(':')[0]}` (5M)\n"
                        f"📈 24h: `{pct:.2f}%`\n"
                        f"💰 `{d['price']}`  刺破 `{d['close_break_pct']}%`\n"
                        f"📊 乖離 `{d['bias']}%`  爆量 `{d.get('vol_ratio')}×`  "
                        f"頸線 `{d.get('neck_chg_pct')}%`\n"
                        f"⚠️ Low 破頸線+SMA14，Top{TOP_N}漲幅榜，爆量過濾"
                    )
                    send_tg_message(msg)
                    print(f"🎯 {symbol} 24h={pct:+.1f}% vol={d.get('vol_ratio')}× bias={d['bias']}%")
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
