"""
One-day bar-by-bar backtest of the shadow-neckline breakout detector.
Uses the same logic as the live scanner (5m Binance USDM, volume filter).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta

MIN_VOLUME_USDT = 50_000_000
REPORT_GAP_MINUTES = 30
LOOKBACK_BARS = 300  # needed for SMA200 + peak search
SCAN_BARS = 288  # 24h of 5m candles


def detect_at_index(close, high, low, sma200, sma14, curr_idx):
    """
    Evaluate shadow-neckline logic as if `curr_idx` were the latest closed bar.
    Arrays must cover indices 0..curr_idx inclusive.
    """
    if curr_idx + 1 < 250:
        return False, None

    window = 2
    # Peak search window: same as live (last 80 bars before window edge)
    start_i = curr_idx + 1 - 80
    end_i = curr_idx + 1 - window  # exclusive upper for range end in live: len-window
    peaks = []
    for i in range(max(window, start_i), end_i):
        # live: close[i] == close[i-window:i+window+1].max()
        if i + window > curr_idx:
            break
        seg = close[i - window : i + window + 1]
        if close[i] == seg.max():
            peaks.append(i)

    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1_max, h2_max, h3_max = high[p1], high[p2], high[p3]

    start_range = max(0, p2 - 48)
    if h2_max < high[start_range:p2].max():
        return False, None

    curr_sma200 = sma200[p2]
    if np.isnan(curr_sma200) or (h2_max - curr_sma200) / curr_sma200 < 0.05:
        return False, None

    if not (h2_max > h1_max and h2_max > h3_max and (p3 - p1) <= 30):
        return False, None
    if abs(h1_max - h3_max) / max(h1_max, h3_max) >= 0.15:
        return False, None

    dx = p3 - p1
    dy = h3_max - h1_max
    slope = dy / dx if dx != 0 else 0.0

    if curr_idx <= p3:
        return False, None

    trigger_line_price = h1_max + slope * (curr_idx - p1)
    curr_sma14 = sma14[curr_idx]
    if np.isnan(curr_sma14):
        return False, None

    if low[curr_idx] < trigger_line_price and low[curr_idx] < curr_sma14:
        if high[curr_idx] < h2_max:
            return True, {
                "price": float(close[curr_idx]),
                "bias": round(((h2_max - curr_sma200) / curr_sma200) * 100, 2),
                "line_val": round(float(trigger_line_price), 6),
                "sma14": round(float(curr_sma14), 6),
            }

    return False, None


def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms):
    all_rows = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if last_ts >= until_ms:
            break
        time.sleep(max(exchange.rateLimit / 1000.0, 0.05))

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] < until_ms].reset_index(drop=True)
    return df


def main():
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    now = datetime.now(timezone.utc)
    end = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    start = end - timedelta(minutes=5 * SCAN_BARS)
    hist_start = start - timedelta(minutes=5 * LOOKBACK_BARS)

    print(f"📡 Backtest window (UTC): {start.isoformat()} → {end.isoformat()}")
    print(f"   History from: {hist_start.isoformat()}")

    tickers = exchange.fetch_tickers()
    symbols = [
        s
        for s, t in tickers.items()
        if s.endswith(":USDT") and (t.get("quoteVolume") or 0) >= MIN_VOLUME_USDT
    ]
    symbols = sorted(symbols)
    print(f"💰 Symbols over {MIN_VOLUME_USDT:,} USDT 24h volume: {len(symbols)}")

    signals = []
    last_report_time = {}
    errors = 0

    for idx, symbol in enumerate(symbols, 1):
        try:
            df = fetch_ohlcv_range(
                exchange,
                symbol,
                "5m",
                int(hist_start.timestamp() * 1000),
                int(end.timestamp() * 1000),
            )
            if len(df) < 250:
                print(f"[{idx}/{len(symbols)}] {symbol}: skip (only {len(df)} bars)")
                continue

            close = df["close"].to_numpy(dtype=float)
            high = df["high"].to_numpy(dtype=float)
            low = df["low"].to_numpy(dtype=float)
            sma200 = ta.sma(df["close"], length=200).to_numpy(dtype=float)
            sma14 = ta.sma(df["close"], length=14).to_numpy(dtype=float)
            ts = df["timestamp"].to_numpy()

            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            hit_count = 0
            scanned = 0

            for i in range(len(df)):
                if ts[i] < start_ms or ts[i] >= end_ms:
                    continue
                scanned += 1
                has_signal, d = detect_at_index(close, high, low, sma200, sma14, i)
                if not has_signal:
                    continue

                bar_time = datetime.fromtimestamp(ts[i] / 1000.0, tz=timezone.utc)
                prev = last_report_time.get(symbol)
                if prev is not None and bar_time < prev + timedelta(minutes=REPORT_GAP_MINUTES):
                    continue

                last_report_time[symbol] = bar_time
                hit_count += 1
                row = {
                    "symbol": symbol.split(":")[0],
                    "time_utc": bar_time.strftime("%Y-%m-%d %H:%M"),
                    "price": d["price"],
                    "bias": d["bias"],
                    "line_val": d["line_val"],
                    "sma14": d["sma14"],
                }
                signals.append(row)
                print(
                    f"🎯 {row['symbol']} @ {row['time_utc']} | "
                    f"price={row['price']} bias={row['bias']}% "
                    f"neck={row['line_val']} sma14={row['sma14']}"
                )

            print(f"[{idx}/{len(symbols)}] {symbol}: scanned {scanned} bars, hits={hit_count}")
        except Exception as e:
            errors += 1
            print(f"[{idx}/{len(symbols)}] {symbol}: ERROR {e}")

    print("\n========== RESULT ==========")
    print(f"Scan window: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Symbols scanned: {len(symbols)} | Errors: {errors}")
    print(f"Signals (after {REPORT_GAP_MINUTES}m cooldown): {len(signals)}")

    out_path = "/workspace/output/shadow_neckline_backtest_1d.csv"
    if signals:
        out = pd.DataFrame(signals)
        out.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        print(out.to_string(index=False))
    else:
        print("No signals in the last 24 hours under current rules.")
        pd.DataFrame(
            columns=["symbol", "time_utc", "price", "bias", "line_val", "sma14"]
        ).to_csv(out_path, index=False)
        print(f"Saved empty CSV: {out_path}")


if __name__ == "__main__":
    main()
