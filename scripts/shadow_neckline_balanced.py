"""
影線頸線監控（Balanced 中等降噪版）— 建議日常使用

相對原版：
- 收盤價跌破頸線 + SMA14（不是影線刺破）
- 收盤跌破深度 >= 0.3%
- 當根收陰
- SMA25：收盤跌破 或 破位深度>=1%（軟條件）
- 距 SMA99 太近（|Δ| < 2%）不空；仍在 SMA99 上方且距離 < 8% 也不空（避免砸支撐）
- 肩距 8~50、對稱 < 15%
- 24h 漲幅上限 300%
- 同幣冷卻 60 分鐘

更嚴版本見 shadow_neckline_strict.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# ================= 設定區域 =================
TG_TOKEN = ""  # 請自行填入
TG_CHAT_ID = ""
MIN_VOLUME_USDT = 50_000_000
SCAN_INTERVAL = 60
REPORT_GAP_MINUTES = 60
MAX_CHG24 = 300.0  # percent（放寬以保留 TUT 這類強勢泵幣破位）
MIN_CLOSE_BREAK_PCT = 0.3  # percent
MIN_ABS_DIST_MA99_PCT = 2.0  # |ΔSMA99| 太近不空
MAX_NEAR_ABOVE_MA99_PCT = 8.0  # 仍在 SMA99 上方且距離不足此值 → 不空
# ===========================================


def send_tg_message(message: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[TG skipped]\n{message}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ TG 發送失敗: {e}")


def detect_shadow_neckline_balanced(df: pd.DataFrame, chg24: float | None):
    if len(df) < 250:
        return False, None
    if chg24 is not None and chg24 > MAX_CHG24:
        return False, None

    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    sma200 = ta.sma(df["close"], length=200).to_numpy(float)
    sma14 = ta.sma(df["close"], length=14).to_numpy(float)
    sma25 = ta.sma(df["close"], length=25).to_numpy(float)
    sma99 = ta.sma(df["close"], length=99).to_numpy(float)

    window = 2
    peaks = []
    for i in range(len(df) - 80, len(df) - window):
        if close[i] == close[i - window : i + window + 1].max():
            peaks.append(i)
    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1, h2, h3 = high[p1], high[p2], high[p3]

    start_range = max(0, p2 - 48)
    if h2 < high[start_range:p2].max():
        return False, None

    s200 = sma200[p2]
    if np.isnan(s200):
        return False, None
    bias = (h2 - s200) / s200
    if bias < 0.05:
        return False, None

    span = p3 - p1
    if not (8 <= span <= 50):
        return False, None
    if not (h2 > h1 and h2 > h3):
        return False, None
    if abs(h1 - h3) / max(h1, h3) >= 0.15:
        return False, None

    curr_idx = len(df) - 1
    if curr_idx <= p3:
        return False, None

    slope = (h3 - h1) / (p3 - p1) if p3 != p1 else 0.0
    neck = h1 + slope * (curr_idx - p1)
    s14 = sma14[curr_idx]
    s25 = sma25[curr_idx]
    s99 = sma99[curr_idx]
    if np.isnan(s14) or np.isnan(s25) or np.isnan(s99) or s99 == 0:
        return False, None

    # Close confirmation (main anti-noise vs original wick break)
    if not (close[curr_idx] < neck and close[curr_idx] < s14):
        return False, None
    close_break_pct = (neck - close[curr_idx]) / neck * 100
    if close_break_pct < MIN_CLOSE_BREAK_PCT:
        return False, None
    if not (close[curr_idx] < open_[curr_idx]):
        return False, None
    # Soft SMA25
    if not (close[curr_idx] < s25 or close_break_pct >= 1.0):
        return False, None
    # MA99 proximity:
    # 1) |Δ| < 2% → chop, skip
    # 2) still ABOVE MA99 but within 8% → about to hit support, skip short
    dist_ma99_pct = (close[curr_idx] - s99) / s99 * 100
    if abs(dist_ma99_pct) < MIN_ABS_DIST_MA99_PCT:
        return False, None
    if 0 <= dist_ma99_pct < MAX_NEAR_ABOVE_MA99_PCT:
        return False, None
    if high[curr_idx] >= h2:
        return False, None

    return True, {
        "price": float(close[curr_idx]),
        "bias": round(bias * 100, 2),
        "line_val": round(float(neck), 6),
        "sma14": round(float(s14), 6),
        "sma25": round(float(s25), 6),
        "sma99": round(float(s99), 6),
        "dist_ma99_pct": round(dist_ma99_pct, 2),
        "close_break_pct": round(close_break_pct, 2),
        "span": int(span),
        "chg24": None if chg24 is None else round(chg24, 2),
    }


def main():
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    print(
        f"📡 Balanced 影線頸線監控中... (冷卻 {REPORT_GAP_MINUTES}m / "
        f"收盤確認 / SMA25軟條件 / |ΔSMA99|>={MIN_ABS_DIST_MA99_PCT}% / "
        f"上方距SMA99>={MAX_NEAR_ABOVE_MA99_PCT}% / 24h<{MAX_CHG24}%)"
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
            print(f"⏰ {now.strftime('%H:%M:%S')} | 掃描中 ({len(symbols)} 幣種)...")

            for symbol in symbols:
                try:
                    if symbol in last_report_time:
                        if now < last_report_time[symbol] + timedelta(minutes=REPORT_GAP_MINUTES):
                            continue
                    t = tickers.get(symbol) or {}
                    chg24 = t.get("percentage")
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    has_signal, d = detect_shadow_neckline_balanced(
                        df, None if chg24 is None else float(chg24)
                    )
                    if not has_signal:
                        time.sleep(0.05)
                        continue

                    msg = (
                        f"🚨 *【影線頸線破位｜Balanced】*\n\n"
                        f"💎 幣種: `{symbol.split(':')[0]}` (5M)\n"
                        f"📈 24h: `{d['chg24']}%`\n"
                        f"💰 收盤價: `{d['price']}`\n"
                        f"📊 SMA200 乖離: `{d['bias']}%`\n"
                        f"📉 頸線: `{d['line_val']}`  破位: `{d['close_break_pct']}%`\n"
                        f"Ⓜ️ SMA14/25/99: `{d['sma14']}` / `{d['sma25']}` / `{d['sma99']}`\n"
                        f"📏 距SMA99: `{d['dist_ma99_pct']}%`\n"
                        f"⚠️ *收盤跌破頸線+SMA14；遠離SMA99（上方需≥{MAX_NEAR_ABOVE_MA99_PCT:.0f}%或已跌破）*"
                    )
                    send_tg_message(msg)
                    print(f"🎯 訊號: {symbol} break={d['close_break_pct']}% bias={d['bias']}%")
                    last_report_time[symbol] = now
                    time.sleep(0.05)
                except Exception:
                    continue

            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
