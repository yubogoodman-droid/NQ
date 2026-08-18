#!/usr/bin/env python3
"""
N-day Binance Vision USDT-M backtest for original / 爆量 / 強爆量 tiers.

Universe (aligned with live scanner):
- Candidate pool: prior UTC-day open→close Top CANDIDATE_N (download set)
- Scan set: trailing 24h return Top TOP_N, refreshed each UTC hour
  (same idea as live ticker.percentage Top10 — catches same-day pumps like ACE)

Vision daily zips lag ~1–2d; missing days fall back to BingX USDT-M 5m klines.

Default window: last N complete UTC days on data.binance.vision (or --end).
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from shadow_neckline_logic import (
    RAW,
    STRICT,
    STRUCTURE,
    VOLUME,
    DetectParams,
    detect_at_index,
    params_dict,
    prepare_indicators,
)

VISION_BASE = "https://data.binance.vision"
S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINGX_KLINES = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
CACHE = Path("/tmp/binance_um_klines")
OUT = Path("/workspace/output")
TOP_N = 10  # 滾動 24h 漲幅榜前 N（對齊實盤）
CANDIDATE_N = 30  # 前一日 Top N 作為下載／候選池（含 ACE 類前一日剛進前段）
BARS_24H = 288  # 5m × 288 = 24h
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144}
TIERS = {
    "raw": RAW,
    "structure": STRUCTURE,  # 結構、不看量（實盤）
    "volume": VOLUME,  # ≥2.5× + 結構
    "volume2": STRICT,  # ≥3.5× + 結構
}


def zip_url(symbol: str, day: str) -> str:
    return (
        f"{VISION_BASE}/data/futures/um/daily/klines/{symbol}/5m/"
        f"{symbol}-5m-{day}.zip"
    )


def latest_complete_vision_day() -> str:
    now = datetime.now(timezone.utc)
    for i in range(0, 10):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            r = requests.head(zip_url("BTCUSDT", day), timeout=20, allow_redirects=True)
            if r.status_code == 200:
                return day
        except requests.RequestException:
            continue
    raise RuntimeError("No Vision day found")


def daterange_end_inclusive(end: str, n: int) -> list[str]:
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return [(end_dt - timedelta(days=n - 1 - i)).strftime("%Y-%m-%d") for i in range(n)]


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


def _to_bingx_symbol(symbol: str) -> str:
    s = symbol.replace("/", "").replace("-", "").upper()
    if not s.endswith("USDT"):
        s += "USDT"
    return s[:-4] + "-USDT"


def download_day_df_bingx(symbol: str, day: str) -> pd.DataFrame | None:
    """Fallback when Vision zip is not published yet (or geo-blocked)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{symbol}-5m-{day}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    start_ms = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if start_ms >= now_ms:
        return None
    end_ms = min(start_ms + 86_400_000, now_ms)
    bx = _to_bingx_symbol(symbol)
    rows: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": bx,
            "interval": "5m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        payload = None
        for attempt in range(4):
            try:
                r = requests.get(BINGX_KLINES, params=params, timeout=45)
                r.raise_for_status()
                payload = r.json()
                break
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(0.4 * (attempt + 1))
        if not payload:
            return None
        data = payload.get("data") or []
        if not data:
            break
        data = sorted(data, key=lambda x: int(x["time"]))
        rows.extend(data)
        last = int(data[-1]["time"])
        nxt = last + 300_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={"time": "timestamp"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["quote_volume"] = df["close"] * df["volume"]
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Only keep bars for this UTC day
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < start_ms + 86_400_000)]
    if df.empty:
        return None
    df.to_csv(cache_path, index=False)
    return df


def download_day_df(symbol: str, day: str) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{symbol}-5m-{day}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        r = requests.get(zip_url(symbol, day), timeout=40)
    except requests.RequestException:
        r = None
    if r is not None and r.status_code == 200:
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                raw = zf.read(zf.namelist()[0])
        except zipfile.BadZipFile:
            raw = None
        if raw is not None:
            text = raw.decode("utf-8", errors="replace")
            first = text.split("\n", 1)[0]
            has_header = "open" in first.lower()
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

    return download_day_df_bingx(symbol, day)


def day_return_pct(symbol: str, day: str) -> float | None:
    """UTC 日開→收漲幅 %；資料不足則 None。"""
    df = download_day_df(symbol, day)
    if df is None or df.empty:
        return None
    o = float(df.iloc[0]["open"])
    c = float(df.iloc[-1]["close"])
    if o <= 0:
        return None
    return (c / o - 1.0) * 100.0


def calc_pnl(df: pd.DataFrame, signal_ts: int) -> dict:
    idxs = df.index[df["timestamp"] == signal_ts].tolist()
    if not idxs or idxs[0] + 1 >= len(df):
        return {}
    i = idxs[0]
    entry = float(df.loc[i + 1, "open"])
    out = {"entry": entry}
    for name, n in HORIZONS.items():
        j = i + n
        out[name] = (
            None if j >= len(df) else (entry - float(df.loc[j, "close"])) / entry * 100
        )
    return out


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "symbols": 0, "reclaim_3_rate": None, "by_horizon": {}, "by_day": {}}
    by_h = {}
    for h in HORIZONS:
        x = df[f"pnl_{h}"].dropna()
        by_h[h] = {
            "n": int(len(x)),
            "win_rate": round(float((x > 0).mean() * 100), 1) if len(x) else None,
            "avg": round(float(x.mean()), 3) if len(x) else None,
            "median": round(float(x.median()), 3) if len(x) else None,
        }
    by_day = {}
    for day, g in df.groupby("day"):
        x = g["pnl_1h"].dropna()
        by_day[day] = {
            "n": int(len(g)),
            "symbols": int(g["symbol"].nunique()),
            "win_1h": round(float((x > 0).mean() * 100), 1) if len(x) else None,
            "avg_1h": round(float(x.mean()), 3) if len(x) else None,
        }
    return {
        "n": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "reclaim_3_rate": round(float(df["reclaim_3"].mean() * 100), 1),
        "by_horizon": by_h,
        "by_day": by_day,
        "symbols_list": sorted(df["symbol"].unique().tolist()),
    }


def load_symbol_frame(
    sym: str, days: list[str], warmup: str | list[str]
) -> pd.DataFrame | None:
    warmups = [warmup] if isinstance(warmup, str) else list(warmup)
    parts = []
    for d in [*warmups, *days]:
        df = download_day_df(sym, d)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def hourly_rolling_top(
    frames: dict[str, pd.DataFrame],
    days: list[str],
    candidates_by_day: dict[str, set[str]],
    top_n: int = TOP_N,
    lookback: int = BARS_24H,
) -> dict[str, dict[int, set[str]]]:
    """
    For each UTC day+hour, Top-N by trailing 24h return among that day's candidates.
    Rank uses the last bar at or before the hour start (no intra-hour look-ahead).
    """
    indexed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, df in frames.items():
        indexed[sym] = (
            df["timestamp"].to_numpy(dtype=np.int64),
            df["close"].to_numpy(dtype=float),
        )

    out: dict[str, dict[int, set[str]]] = {}
    for day in days:
        out[day] = {}
        cands = candidates_by_day.get(day, set())
        day_start = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
        for hour in range(24):
            t_ref = day_start + hour * 3_600_000
            rets: list[tuple[float, str]] = []
            for sym in cands:
                pair = indexed.get(sym)
                if pair is None:
                    continue
                ts, close = pair
                i = int(np.searchsorted(ts, t_ref, side="right") - 1)
                if i < lookback:
                    continue
                c0 = float(close[i - lookback])
                c1 = float(close[i])
                if c0 <= 0 or np.isnan(c0) or np.isnan(c1):
                    continue
                rets.append(((c1 / c0 - 1.0) * 100.0, sym))
            rets.sort(key=lambda x: x[0], reverse=True)
            out[day][hour] = {s for _, s in rets[:top_n]}
    return out


def run_tier(
    params: DetectParams,
    frames: dict[str, pd.DataFrame],
    days: list[str],
    candidates_by_day: dict[str, set[str]],
    scan_by_day_hour: dict[str, dict[int, set[str]]] | None = None,
) -> pd.DataFrame:
    if scan_by_day_hour is None:
        scan_by_day_hour = hourly_rolling_top(frames, days, candidates_by_day)
    signals = []
    for day in days:
        start_ms = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
        end_ms = start_ms + 86_400_000
        last_report: dict[str, datetime] = {}
        for sym in sorted(candidates_by_day.get(day, set())):
            df = frames.get(sym)
            if df is None or len(df) < 250:
                continue
            close, high, low, open_, sma200, sma14, sma25, sma99, volume, timestamp, sma200_15 = (
                prepare_indicators(df)
            )
            ts = df["timestamp"].to_numpy()
            for i in range(len(df)):
                tsi = int(ts[i])
                if not (start_ms <= tsi < end_ms):
                    continue
                hour = int((tsi - start_ms) // 3_600_000)
                if sym not in scan_by_day_hour.get(day, {}).get(hour, set()):
                    continue
                ok, d = detect_at_index(
                    close,
                    high,
                    low,
                    open_,
                    sma200,
                    sma14,
                    sma25,
                    sma99,
                    volume,
                    timestamp,
                    sma200_15,
                    i,
                    params,
                )
                if not ok:
                    continue
                tdt = datetime.fromtimestamp(tsi / 1000, tz=timezone.utc)
                prev = last_report.get(sym)
                if prev and tdt < prev + timedelta(minutes=params.cooldown_min):
                    continue
                reclaim = False
                for j in range(1, 4):
                    if i + j < len(df) and high[i + j] > d["line_val"]:
                        reclaim = True
                        break
                pnl = calc_pnl(df, tsi)
                row = {
                    "day": day,
                    "symbol": f"{sym[:-4]}/{sym[-4:]}",
                    "time_utc": tdt.strftime("%Y-%m-%d %H:%M"),
                    **d,
                    "entry": pnl.get("entry"),
                    "reclaim_3": reclaim,
                }
                for h in HORIZONS:
                    row[f"pnl_{h}"] = pnl.get(h)
                signals.append(row)
                last_report[sym] = tdt
    return pd.DataFrame(signals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--end", default=None, help="UTC end day YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    end = args.end or latest_complete_vision_day()
    days = daterange_end_inclusive(end, args.days)
    # Need ~3 prior days so 15m SMA200 (200×15m) is ready at window start
    warmup_n = max(3, 1)
    warmup_start = datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(
        days=warmup_n
    )
    warmups = [(warmup_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(warmup_n)]

    tag = f"{args.days}d"
    print(f"📡 {tag} Vision/BingX backtest")
    print(f"   window: {days[0]} → {days[-1]} ({len(days)} days) warmups={warmups[0]}→{warmups[-1]}")
    print(
        f"   universe: 候選=前一日Top{CANDIDATE_N}；掃描=滾動24h Top{TOP_N}（每UTC整點刷新，對齊實盤）"
    )
    print(f"   tiers: raw / volume(≥2.5×+filters) / volume2(≥3.5×+filters)")

    print("📋 Listing symbols...")
    all_syms = list_um_usdt_symbols()
    print(f"   {len(all_syms)} UM USDT symbols")

    # 候選池排名日 = 訊號日前一日
    rank_day0 = (
        datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    rank_days = [rank_day0, *days[:-1]]  # map: days[i] uses rank_days[i]

    print(f"📈 Candidate pool = prior-day Top{CANDIDATE_N} ...")
    returns_by_day: dict[str, dict[str, float]] = {}
    for rday in sorted(set(rank_days)):
        rets: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=32) as pool:
            futs = {pool.submit(day_return_pct, s, rday): s for s in all_syms}
            done = 0
            for fut in as_completed(futs):
                s = futs[fut]
                done += 1
                try:
                    v = fut.result()
                except Exception:
                    v = None
                if v is not None:
                    rets[s] = v
                if done % 200 == 0:
                    print(f"   {rday} progress {done}/{len(all_syms)}")
        returns_by_day[rday] = rets
        print(f"   {rday}: ranked_from={len(rets)}")

    candidates_by_day: dict[str, set[str]] = {}
    top_detail: dict[str, list[tuple[str, float]]] = {}
    union: set[str] = set()
    for day, rday in zip(days, rank_days):
        ranked = sorted(returns_by_day.get(rday, {}).items(), key=lambda x: x[1], reverse=True)[
            :CANDIDATE_N
        ]
        top_detail[day] = ranked
        uni = {s for s, _ in ranked}
        candidates_by_day[day] = uni
        union |= uni
        preview = ", ".join(f"{s} {p:+.1f}%" for s, p in ranked[:5])
        print(f"   {day} ← {rday} candTop{CANDIDATE_N}: {preview} ...")

    print(f"⬇ Ensuring warmups {warmups[0]}→{warmups[-1]} for {len(union)} symbols...")
    with ThreadPoolExecutor(max_workers=24) as pool:
        for wday in warmups:
            list(pool.map(lambda s, d=wday: download_day_df(s, d), sorted(union)))
        # also ensure scan days (Vision or BingX)
        for d in days:
            list(pool.map(lambda s, day=d: download_day_df(s, day), sorted(union)))

    print("📦 Building frames...")
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(sorted(union), 1):
        df = load_symbol_frame(sym, days, warmups)
        if df is not None and len(df) >= 250:
            frames[sym] = df
        if i % 25 == 0 or i == len(union):
            print(f"   frames {i}/{len(union)} loaded={len(frames)}")

    print(f"🏆 Building hourly rolling-24h Top{TOP_N} scan sets...")
    scan_by_day_hour = hourly_rolling_top(frames, days, candidates_by_day, TOP_N)
    # preview a few hours that matter
    for day in days[-2:]:
        for hour in (0, 13, 23):
            top = scan_by_day_hour.get(day, {}).get(hour, set())
            print(f"   {day} {hour:02d}:00 Top{TOP_N} n={len(top)} sample={sorted(top)[:5]}")

    results = {}
    for name, params in TIERS.items():
        df = run_tier(params, frames, days, candidates_by_day, scan_by_day_hour)
        summary = summarize(df)
        payload = {
            "window": {
                "start": days[0],
                "end": days[-1],
                "days": days,
                "warmups": warmups,
            },
            "source": "binance_vision_um_5m_bingx_fallback",
            "universe": f"prior_day_top{CANDIDATE_N}_candidates__rolling_24h_top{TOP_N}_hourly",
            "top_n": TOP_N,
            "candidate_n": CANDIDATE_N,
            "candidate_by_day": {
                d: [{"symbol": s, "prior_day_pct": round(p, 2)} for s, p in top_detail[d]]
                for d in days
            },
            "tier": name,
            "params": params_dict(params),
            "candidates_by_day": {d: sorted(candidates_by_day[d]) for d in days},
            "summary": summary,
        }
        csv_path = OUT / f"shadow_neckline_{name}_{tag}.csv"
        json_path = OUT / f"shadow_neckline_{name}_{tag}_summary.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        results[name] = summary
        print(f"\n=== {name} n={summary['n']} symbols={summary['symbols']} ===")
        print("horizons:", json.dumps(summary["by_horizon"], ensure_ascii=False))
        print("by_day:")
        for d in days:
            print(f"  {d}: {summary['by_day'].get(d)}")
        if name in ("volume", "structure") and not df.empty:
            ace = df[df["symbol"].str.startswith("ACE/")]
            if not ace.empty:
                print(f"ACE hits ({name}):")
                cols = [c for c in ["time_utc", "price", "vol_ratio", "bias", "pnl_1h", "pnl_4h", "pnl_8h"] if c in df.columns]
                print(ace[cols].to_string(index=False))

    lines = [
        f"# {args.days}-day shadow-neckline ({days[0]} → {days[-1]} UTC, Vision+BingX)",
        "",
        f"Universe: prior-day Top {CANDIDATE_N} candidates → **rolling 24h Top {TOP_N}** (hourly, live-aligned).",
        "",
        "| Tier | n | symbols | 1h win | 1h avg | 4h win | 4h avg | 8h win | 8h avg |",
        "|--|--|--|--|--|--|--|--|--|",
    ]
    for name, s in results.items():
        h1 = s["by_horizon"].get("1h") or {}
        h4 = s["by_horizon"].get("4h") or {}
        h8 = s["by_horizon"].get("8h") or {}
        lines.append(
            f"| {name} | {s['n']} | {s['symbols']} | {h1.get('win_rate')}% | "
            f"{h1.get('avg')}% | {h4.get('win_rate')}% | {h4.get('avg')}% | "
            f"{h8.get('win_rate')}% | {h8.get('avg')}% |"
        )
    lines.append("")
    lines.append(
        "structure = live path: same quality filters, **no volume** (close-break, span≥9, bias≤65%, ext SMA200≤40%, not below MA99)."
    )
    lines.append(
        "volume = structure + 爆量≥2.5× (break-bar OR 4h-peak vs pre-window avg)"
    )
    lines.append(
        "volume2 = structure + 爆量≥3.5× (same peak window)"
    )
    lines.append("")
    lines.append(
        "Near rising SMA200 (|dist|<4%): skips lows into rising MA support (e.g. XAN)."
    )
    lines.append(
        "Deep-below 15m SMA200 (dist < −3%): skips late shorts after a higher-TF dump (e.g. GWEI)."
    )
    lines.append(
        "Rolling 24h Top10 catches same-day pumps that prior-day Top10 misses (e.g. ACE 08-14)."
    )
    report = "\n".join(lines)
    (OUT / f"shadow_neckline_{tag}_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
