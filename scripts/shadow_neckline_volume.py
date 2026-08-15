"""
影線頸線監控 → 滿足條件即 Telegram 通知

選幣：Binance USDT-M「24h 漲幅榜前 10」（ticker.percentage）

偵測：shadow_neckline_logic.STRUCTURE（結構過濾，**不看爆量**）
- Low 破頸線 + SMA14
- 拒絕上升頸線（右肩高於左肩）
- 拒絕貼近上彎 SMA200（|距SMA200|<4%）
- 拒絕 15分K 戳破 200均線且收在下方
- 拒絕貼近 15m SMA200（|距|<1.5%）
- 拒絕已深跌破 15m SMA200（dist < −3%）
- 拒絕貼近 SMA99（|距|<1.5%）

TG：環境變數 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
（也可設 TG_TOKEN / TG_CHAT_ID）
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from shadow_neckline_logic import STRUCTURE, detect_at_index, prepare_indicators

# ================= 設定區域 =================
TG_TOKEN = (
    os.environ.get("TELEGRAM_BOT_TOKEN")
    or os.environ.get("TG_TOKEN")
    or ""
).strip()
TG_CHAT_ID = (
    os.environ.get("TELEGRAM_CHAT_ID")
    or os.environ.get("TG_CHAT_ID")
    or ""
).strip()
TOP_N = 10
SCAN_INTERVAL = 60
PARAMS = STRUCTURE
# 15m SMA200 需要 ≥200 根 15m ≈ 600 根 5m
OHLCV_LIMIT = 600
# ===========================================


def send_tg_message(message: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[TG skipped — set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID]\n{message}")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ TG HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"❌ TG 發送失敗: {e}")
        return False


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
    tg_ok = bool(TG_TOKEN and TG_CHAT_ID)
    print(
        f"📡 影線頸線監控（結構、不看量） Top{TOP_N}漲幅榜 "
        f"cooldown={PARAMS.cooldown_min}m ohlcv={OHLCV_LIMIT} "
        f"TG={'ON' if tg_ok else 'OFF'}"
    )
    last_report_time: dict[str, datetime] = {}

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

                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=OHLCV_LIMIT)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    arrays = prepare_indicators(df)
                    ok, d = detect_at_index(*arrays, len(df) - 1, PARAMS)
                    if not ok:
                        time.sleep(0.05)
                        continue

                    sym = symbol.split(":")[0]
                    msg = (
                        f"🚨 *【影線頸線｜Top{TOP_N}】*\n\n"
                        f"💎 `{sym}` (5M)\n"
                        f"📈 24h: `{pct:.2f}%`\n"
                        f"💰 `{d['price']}`  刺破 `{d['close_break_pct']}%`\n"
                        f"📊 乖離 `{d['bias']}%`  頸線變化 `{d.get('neck_chg_pct')}%`\n"
                        f"⚠️ Low 破頸線+SMA14（不看爆量）"
                    )
                    sent = send_tg_message(msg)
                    print(
                        f"🎯 {sym} 24h={pct:+.1f}% bias={d['bias']}% "
                        f"break={d['close_break_pct']}% TG={'sent' if sent else 'skip'}"
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
