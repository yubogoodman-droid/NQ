"""
One-day backtest of the shadow-neckline detector using Binance Vision
USDT-M futures 5m klines (fapi is geo-blocked in this environment).

Backtests the last complete UTC day available on data.binance.vision.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

MIN_VOLUME_USDT = 50_000_000
REPORT_GAP_MINUTES = 30
VISION_BASE = "https://data.binance.vision"
S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE_DIR = Path("/tmp/binance_um_klines")
OUT_CSV = Path("/workspace/output/shadow_neckline_backtest_1d.csv")


def detect_at_index(close, high, low, sma200, sma14, curr_idx):
    """Same core logic as the live script, evaluated at curr_idx."""
    if curr_idx + 1 < 250:
        return False, None

    window = 2
    start_i = curr_idx + 1 - 80
    end_i = curr_idx + 1 - window
    peaks = []
    for i in range(max(window, start_i), end_i):
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


def list_um_usdt_symbols() -> list[str]:
    from xml.etree import ElementTree as ET

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    prefix = "data/futures/um/daily/klines/"
    token = None
    symbols: list[str] = []
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(S3_BASE, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for cp in root.findall("s3:CommonPrefixes", ns):
            p = cp.find("s3:Prefix", ns).text
            sym = p[len(prefix) :].strip("/")
            if sym.endswith("USDT"):
                symbols.append(sym)
        nxt = root.find("s3:NextContinuationToken", ns)
        trunc = root.find("s3:IsTruncated", ns)
        if trunc is not None and trunc.text == "true" and nxt is not None:
            token = nxt.text
        else:
            break
    return sorted(symbols)


def zip_url(symbol: str, day: str) -> str:
    return (
        f"{VISION_BASE}/data/futures/um/daily/klines/{symbol}/5m/"
        f"{symbol}-5m-{day}.zip"
    )


def download_day_df(symbol: str, day: str) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}-5m-{day}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return df

    url = zip_url(symbol, day)
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            name = zf.namelist()[0]
            raw = zf.read(name)
    except zipfile.BadZipFile:
        return None

    # Binance Vision CSV may or may not include a header row.
    text = raw.decode("utf-8", errors="replace")
    first = text.split("\n", 1)[0]
    has_header = "open" in first.lower() or "Open" in first
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    if has_header:
        df = pd.read_csv(io.StringIO(text))
        df.columns = cols[: len(df.columns)]
    else:
        df = pd.read_csv(io.StringIO(text), header=None, names=cols)

    keep = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def day_quote_volume(symbol: str, day: str) -> float:
    df = download_day_df(symbol, day)
    if df is None or df.empty:
        return 0.0
    return float(df["quote_volume"].sum())


def latest_complete_day() -> str:
    """Find latest UTC day with BTCUSDT 5m zip available."""
    now = datetime.now(timezone.utc)
    for i in range(1, 8):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        r = requests.head(zip_url("BTCUSDT", day), timeout=20, allow_redirects=True)
        if r.status_code == 200:
            return day
    raise RuntimeError("No recent Vision daily zip found for BTCUSDT")


def main():
    scan_day = latest_complete_day()
    hist_day = (
        datetime.strptime(scan_day, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(f"📡 Data source: Binance Vision USDT-M futures 5m")
    print(f"   Scan day (UTC): {scan_day} 00:00 → 24:00")
    print(f"   History day:    {hist_day}")
    print(f"   Note: live fapi.binance.com is geo-blocked here (HTTP 451).")

    print("📋 Listing UM USDT symbols...")
    symbols = list_um_usdt_symbols()
    print(f"   Found {len(symbols)} symbols")

    print(f"💰 Filtering by {scan_day} quote volume >= {MIN_VOLUME_USDT:,} ...")
    volumes: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        futs = {pool.submit(day_quote_volume, s, scan_day): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            try:
                volumes[s] = fut.result()
            except Exception:
                volumes[s] = 0.0
            if done % 100 == 0 or done == len(symbols):
                print(f"   volume progress {done}/{len(symbols)}")

    liquid = sorted([s for s, v in volumes.items() if v >= MIN_VOLUME_USDT], key=lambda x: -volumes[x])
    print(f"   Liquid symbols: {len(liquid)}")
    for s in liquid[:15]:
        print(f"     {s}: {volumes[s]:,.0f}")

    # Ensure history day cached for liquid symbols
    print("⬇ Downloading history day for liquid symbols...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda s: download_day_df(s, hist_day), liquid))

    signals = []
    last_report_time: dict[str, datetime] = {}
    scan_start = datetime.strptime(scan_day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    scan_end = scan_start + timedelta(days=1)
    start_ms = int(scan_start.timestamp() * 1000)
    end_ms = int(scan_end.timestamp() * 1000)

    for idx, symbol in enumerate(liquid, 1):
        df_h = download_day_df(symbol, hist_day)
        df_s = download_day_df(symbol, scan_day)
        if df_h is None or df_s is None:
            print(f"[{idx}/{len(liquid)}] {symbol}: missing data")
            continue
        df = pd.concat([df_h, df_s], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 250:
            print(f"[{idx}/{len(liquid)}] {symbol}: skip ({len(df)} bars)")
            continue

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        sma200 = ta.sma(df["close"], length=200).to_numpy(dtype=float)
        sma14 = ta.sma(df["close"], length=14).to_numpy(dtype=float)
        ts = df["timestamp"].to_numpy()

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
                "symbol": f"{symbol[:-4]}/USDT",
                "time_utc": bar_time.strftime("%Y-%m-%d %H:%M"),
                "price": d["price"],
                "bias": d["bias"],
                "line_val": d["line_val"],
                "sma14": d["sma14"],
                "day_quote_volume": round(volumes[symbol], 2),
            }
            signals.append(row)
            print(
                f"🎯 {row['symbol']} @ {row['time_utc']} | "
                f"price={row['price']} bias={row['bias']}% "
                f"neck={row['line_val']} sma14={row['sma14']}"
            )

        print(f"[{idx}/{len(liquid)}] {symbol}: scanned {scanned} bars, hits={hit_count}")

    print("\n========== RESULT ==========")
    print(f"Scan day: {scan_day} UTC (Binance USDT-M 5m)")
    print(f"Symbols scanned: {len(liquid)}")
    print(f"Signals (after {REPORT_GAP_MINUTES}m cooldown): {len(signals)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if signals:
        out = pd.DataFrame(signals)
        out.to_csv(OUT_CSV, index=False)
        print(f"Saved: {OUT_CSV}")
        print(out.to_string(index=False))
    else:
        print("No signals on this day under current rules.")
        pd.DataFrame(
            columns=["symbol", "time_utc", "price", "bias", "line_val", "sma14", "day_quote_volume"]
        ).to_csv(OUT_CSV, index=False)
        print(f"Saved empty CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
