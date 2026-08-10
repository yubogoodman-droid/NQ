"""
影線頸線監控：只掃「24h 漲幅榜前 10」幣種。

相對原腳本，唯一差異在選幣：
  原：quoteVolume >= MIN_VOLUME_USDT
  新：依 ticker.percentage（24h 漲跌幅%）由高到低取前 TOP_N

偵測邏輯（影線頸線 / SMA200 乖離 / 跌破頸線+SMA14）不變。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import pandas_ta as ta
import requests

# ================= 設定區域 =================
TG_TOKEN = ""  # 請自行填入，勿把 token 寫進公開 repo
TG_CHAT_ID = ""
TOP_N = 10                   # 漲幅榜前 N
SCAN_INTERVAL = 60           # 掃描頻率 (秒)
REPORT_GAP_MINUTES = 30      # 同幣種回報冷卻
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


def detect_shadow_neckline_signal(symbol, df):
    if len(df) < 250:
        return False, None

    close_prices = df["close"].values
    high_prices = df["high"].values
    low_prices = df["low"].values
    df["sma200"] = ta.sma(df["close"], length=200)
    df["sma14"] = ta.sma(df["close"], length=14)

    window = 2
    peaks = []
    for i in range(len(df) - 80, len(df) - window):
        if close_prices[i] == close_prices[i - window : i + window + 1].max():
            peaks.append(i)

    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1_max, h2_max, h3_max = high_prices[p1], high_prices[p2], high_prices[p3]

    start_range = max(0, p2 - 48)
    if h2_max < high_prices[start_range:p2].max():
        return False, None

    curr_sma200 = df.iloc[p2]["sma200"]
    if pd.isna(curr_sma200) or (h2_max - curr_sma200) / curr_sma200 < 0.05:
        return False, None

    if h2_max > h1_max and h2_max > h3_max and (p3 - p1) <= 30:
        if abs(h1_max - h3_max) / max(h1_max, h3_max) < 0.15:
            dx = p3 - p1
            dy = h3_max - h1_max
            slope = dy / dx if dx != 0 else 0

            curr_idx = len(df) - 1
            if curr_idx > p3:
                trigger_line_price = h1_max + slope * (curr_idx - p1)
                curr_sma14 = df.iloc[curr_idx]["sma14"]

                if low_prices[curr_idx] < trigger_line_price and low_prices[curr_idx] < curr_sma14:
                    if high_prices[curr_idx] < h2_max:
                        return True, {
                            "price": close_prices[curr_idx],
                            "bias": round(((h2_max - curr_sma200) / curr_sma200) * 100, 2),
                            "line_val": round(trigger_line_price, 6),
                            "sma14": round(curr_sma14, 6),
                        }

    return False, None


def top_gainers(tickers: dict, n: int = 10) -> list[tuple[str, float]]:
    """回傳 [(symbol, pct), ...]，僅 USDT 永續，依 24h 漲幅由高到低。"""
    rows = []
    for symbol, t in tickers.items():
        if not symbol.endswith(":USDT"):
            continue
        pct = t.get("percentage")
        if pct is None:
            # 後備：用 open / last 推估
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
    print(f"📡 影線頸線監控中（僅漲幅榜前 {TOP_N}）...")

    last_report_time = {}

    while True:
        try:
            tickers = exchange.fetch_tickers()
            ranked = top_gainers(tickers, TOP_N)
            symbols = [s for s, _ in ranked]

            now = datetime.now()
            preview = ", ".join(f"{s.split(':')[0]} {p:+.1f}%" for s, p in ranked)
            print(f"⏰ {now.strftime('%H:%M:%S')} | Top{TOP_N}: {preview}")

            for symbol, pct in ranked:
                try:
                    if symbol in last_report_time:
                        if now < last_report_time[symbol] + timedelta(minutes=REPORT_GAP_MINUTES):
                            continue

                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    has_signal, d = detect_shadow_neckline_signal(symbol, df)
                    if has_signal:
                        msg = (
                            f"🚨 *【影線頸線破位預警｜漲幅榜】*\n\n"
                            f"💎 幣種: `{symbol.split(':')[0]}` (5M)\n"
                            f"📈 24h漲幅: `{pct:.2f}%`\n"
                            f"💰 當前價: `{d['price']}`\n"
                            f"📊 SMA200 乖離: `{d['bias']}%`\n"
                            f"📉 影線頸線位: `{d['line_val']}`\n"
                            f"Ⓜ️ SMA14 位: `{d['sma14']}`\n"
                            f"⚠️ *狀態: 跌破影線連線 + SMA14*"
                        )
                        send_tg_message(msg)
                        print(f"🎯 發現訊號: {symbol} ({pct:+.1f}%)")
                        last_report_time[symbol] = now

                    time.sleep(0.1)
                except Exception:
                    continue

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"❌ 錯誤: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
