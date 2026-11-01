#!/usr/bin/env python3
"""抓 SOXS 永續的起跌放空。5 分定結構，1 分收盤進場。不下單。

    python3 examples/soxs_rollover.py
    python3 examples/soxs_rollover.py --watch
    python3 examples/soxs_rollover.py --demo
    python3 examples/soxs_rollover.py --symbol SOXSUSDT --risk 100 --hours 72
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollover.strategy import (  # noqa: E402
    RolloverSignal,
    RolloverView,
    inspect_rollover,
    qty_for_risk,
)

TPE = "Asia/Taipei"
HOSTS = ("https://fapi.binance.com", "https://www.binance.com")
KIND = {"break": "跌破", "retest": "回測"}


def frame_from_klines(raw: list, interval: str, now_ms: int | None = None) -> pd.DataFrame:
    """幣安 K 線轉成策略要的表。還沒收盤的最後一根丟掉。"""

    bar_ms = 60_000 if interval == "1m" else 300_000
    rows = list(raw)
    if rows and now_ms is not None and int(rows[-1][0]) + bar_ms > now_ms:
        rows = rows[:-1]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        {
            "open": [float(x[1]) for x in rows],
            "high": [float(x[2]) for x in rows],
            "low": [float(x[3]) for x in rows],
            "close": [float(x[4]) for x in rows],
            "volume": [float(x[5]) for x in rows],
        },
        index=pd.to_datetime([int(x[0]) for x in rows], unit="ms", utc=True),
    )
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_klines(symbol: str, interval: str, hours: int) -> pd.DataFrame:
    import requests

    bar_ms = 60_000 if interval == "1m" else 300_000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(hours) * 3_600_000
    last_error: Exception | None = None
    for host in HOSTS:
        try:
            return _fetch_host(requests, host, symbol, interval, start_ms, now_ms, bar_ms)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"抓不到 {symbol} {interval}：{last_error}")


def _fetch_host(requests, host: str, symbol: str, interval: str, start_ms: int, now_ms: int, bar_ms: int) -> pd.DataFrame:
    rows: list = []
    cursor = start_ms
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    while cursor < now_ms:
        response = session.get(
            host + "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1500},
            timeout=20,
        )
        response.raise_for_status()
        chunk = response.json()
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        nxt = int(chunk[-1][0]) + bar_ms
        if nxt <= cursor or len(chunk) < 1500:
            break
        cursor = nxt
        time.sleep(0.05)
    frame = frame_from_klines(rows, interval, now_ms=now_ms)
    if frame.empty:
        raise RuntimeError(f"{host} 沒有回 {symbol} {interval}")
    return frame


def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """離線合成：極點 54.85，較低高點 53.50，頸線 52.35，1 分收盤 52.05 進場。"""

    rows: list[tuple] = [(52.0, 52.10, 51.90, 52.0, 1000.0) for _ in range(40)]
    rows.extend(
        [
            (53.5, 54.85, 53.40, 54.20, 3000),
            (54.0, 54.00, 52.50, 52.70, 2000),
            (52.7, 52.80, 52.35, 52.60, 1500),
            (52.6, 53.20, 52.50, 53.00, 1200),
            (53.0, 53.50, 52.90, 53.30, 1200),
            (53.3, 53.20, 52.80, 52.90, 1100),
            (52.9, 53.00, 52.70, 52.80, 1100),
            (52.8, 52.70, 51.90, 52.05, 4000),
        ]
    )
    index5 = pd.date_range("2026-10-31 00:00", periods=len(rows), freq="5min", tz="UTC")
    df5 = pd.DataFrame(rows, index=index5, columns=["open", "high", "low", "close", "volume"])
    records = []
    stamps = []
    for ts, row in df5.iterrows():
        vol = float(row.volume) / 5.0
        close = float(row.close)
        records.append((float(row.open), float(row.high), float(row.low), close, vol))
        stamps.append(ts)
        for k in range(1, 5):
            records.append((close, close, close, close, vol))
            stamps.append(ts + pd.Timedelta(minutes=k))
    df1 = pd.DataFrame(records, index=pd.DatetimeIndex(stamps), columns=["open", "high", "low", "close", "volume"])
    entry = 47 * 5
    return df5, df1.iloc[: entry + 1]


def load_view(symbol: str, hours: int, demo: bool) -> tuple[str, RolloverView]:
    if demo:
        df5, df1 = demo_frames()
        return "DEMO", inspect_rollover(df5, df1)
    df5 = fetch_klines(symbol, "5m", hours)
    df1 = fetch_klines(symbol, "1m", hours)
    return symbol, inspect_rollover(df5, df1)


def format_report(symbol: str, view: RolloverView, risk: float) -> str:
    lines = [
        f"{symbol}  永續  最後一根已收盤 1 分  {fmt_time(view.timestamp)}",
        f"收盤 {px(view.close)}",
    ]
    if view.extreme is not None and view.lower_high is not None and view.neckline is not None:
        lines.append(
            f"5 分  極點 {px(view.extreme)}  較低高點 {px(view.lower_high)}  頸線 {px(view.neckline)}"
        )
    if view.extension is not None and view.neckline is not None:
        if view.extension > 0:
            lines.append(f"收盤在頸線下 {view.extension * 100:.2f}%")
        else:
            lines.append(f"收盤在頸線上 {abs(view.extension) * 100:.2f}%")
    if view.ma_slow is not None:
        side = "下" if view.close < view.ma_slow else "上"
        lines.append(f"1 分 MA25 {px(view.ma_slow)}  收盤在均線{side}")
    if view.vol_ma is not None:
        enough = "量能夠" if view.volume >= view.vol_ma else "量能不夠"
        lines.append(f"量 {px(view.volume)}  前 10 根均量 {px(view.vol_ma)}  {enough}")

    if view.fresh is not None:
        lines.append("")
        lines.append(f"抓到起跌（{KIND.get(view.fresh.kind, view.fresh.kind)}）")
        lines.extend(_order_lines(view.fresh, risk))
    else:
        lines.append(f"這根不是起跌：{view.note}")

    older = [sig for sig in view.signals if view.fresh is None or sig.timestamp != view.fresh.timestamp]
    if older:
        lines.append("")
        lines.append("這段資料裡較早的進場")
        for sig in older[-5:]:
            session = "  現貨休市" if sig.off_session else ""
            lines.append(
                f"{fmt_time(sig.timestamp)}  {KIND.get(sig.kind, sig.kind)}  "
                f"進場 {px(sig.entry)}  停損 {px(sig.stop)}  目標 {px(sig.target)}  "
                f"極點 {px(sig.extreme)}{session}"
            )
    elif view.fresh is not None and view.fresh.off_session:
        lines.append("現貨休市，永續仍出這根訊號")
    return "\n".join(lines)


def _order_lines(sig: RolloverSignal, risk: float) -> list[str]:
    qty = qty_for_risk(sig.entry, sig.stop, risk)
    lines = [
        f"進場 {px(sig.entry)}",
        f"停損 {px(sig.stop)}",
        f"目標 {px(sig.target)}（1.5 倍停損距離）",
    ]
    if qty > 0:
        lines.append(f"張數 {qty}，這筆最多虧 {px(risk)} USDT，每張風險 {px(sig.risk_per_unit)}")
    else:
        lines.append(f"以 {px(risk)} USDT 的虧損預算，停損距離不夠下一張")
    return lines


def fmt_time(ts: pd.Timestamp) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(TPE).strftime("%Y-%m-%d %H:%M") + " 台北"


def px(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def sleep_until_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1.0, nxt - now))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓 SOXS 永續 5 分結構、1 分起跌。不下單。")
    parser.add_argument("--symbol", default="SOXSUSDT")
    parser.add_argument("--hours", type=int, default=72, help="往回看幾小時，預設 72")
    parser.add_argument("--risk", type=float, default=100.0, help="這筆最多虧的 USDT，用來回推張數")
    parser.add_argument("--watch", action="store_true", help="每根 1 分收盤再看一次")
    parser.add_argument("--demo", action="store_true", help="不打網路，用 54.85 那段合成 K 證明抓得到")
    args = parser.parse_args(argv)

    seen: set[tuple] = set()
    while True:
        try:
            symbol, view = load_view(args.symbol, args.hours, args.demo)
        except Exception as exc:  # noqa: BLE001
            print(f"這次沒抓到資料：{exc}", file=sys.stderr)
            if not args.watch or args.demo:
                return 1
            sleep_until_next_close()
            continue
        key = (view.timestamp, None if view.fresh is None else view.fresh.kind)
        if key not in seen:
            print(format_report(symbol, view, args.risk))
            print()
            seen.add(key)
        if not args.watch or args.demo:
            return 0
        sleep_until_next_close()


if __name__ == "__main__":
    raise SystemExit(main())
