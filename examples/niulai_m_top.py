#!/usr/bin/env python3
"""牛来 USDT 永續 · 五分 K M 頭跌破 MA200 做空。

對齊幣安 App 截圖：只做牛來那種對稱深 M（雙峰站在 MA200 上方、頸線回測均線），
收盤同時跌破 MA200 與 MA25 才空。停損在 M 頭高點，目標 2R，或 48 根時間停。

用法:
  python3 examples/niulai_m_top.py
  python3 examples/niulai_m_top.py --days 7 --pages
  python3 examples/niulai_m_top.py --scan --limit 50 --days 3 --pages
  python3 examples/test_niulai_m_top.py
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "niulai-m-top" / "index.html"
SCAN_PAGES = REPO / "docs" / "binance-m-top-3d" / "index.html"
GAINERS_PAGES = REPO / "docs" / "binance-m-top-gainers" / "index.html"
CST = ZoneInfo("Asia/Shanghai")
BINANCE = "https://www.binance.com"
SYMBOL = "牛来USDT"
SYMBOL_TW = "牛來USDT"
INTERVAL = "5m"
LISTING_MS = 1_788_089_400_000  # 2026-08-30 19:30 CST
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"}
)

# 幣安 App 預設均線色
MA_PERIODS = (7, 14, 25, 99, 120, 200)
MA_COLORS = {
    7: "#f0c14a",
    14: "#7ec8e3",
    25: "#f472b6",
    99: "#a78bfa",
    120: "#34d399",
    200: "#c2410c",
}


# ---------------------------------------------------------------------------
# Params / types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MTopParams:
    swing_lookback: int = 3
    high_tolerance_pct: float = 0.02
    second_high_max_overshoot: float = 0.008
    min_bars_between: int = 12
    max_bars_between: int = 72
    min_depth_pct: float = 0.02
    max_bars_to_break: int = 48
    ma_period: int = 200
    target_r: float = 2.0
    time_bars: int = 48
    tick_size: float = 0.00001
    # 牛來同款：雙頂貼在 MA200 上方、頸線回測均線、跌破要帶量
    min_peak_above_ma_pct: float = 0.0
    max_neck_above_ma_pct: float = 1.0
    max_neck_below_ma_pct: float = 1.0
    min_frac_closes_above_ma: float = 0.0
    require_close_below_ma25: bool = False
    min_break_volume_mult: float = 0.0
    local_high_pad: int = 0


def default_params(**overrides: Any) -> MTopParams:
    return MTopParams(**overrides)


def niulai_params(**overrides: Any) -> MTopParams:
    """對齊截圖：對稱 M、深度夠、雙頂明顯站在 MA200 上方、頸線回測均線。

    不要求跌破當根放量：09-07 那筆 22:15 收盤穿 MA200 時量其實偏小，
    瀑布放量在後面幾根。
    """
    data = dict(
        high_tolerance_pct=0.01,
        second_high_max_overshoot=0.003,
        min_bars_between=20,
        max_bars_between=48,
        min_depth_pct=0.04,
        min_peak_above_ma_pct=0.03,
        max_neck_above_ma_pct=0.012,
        max_neck_below_ma_pct=0.02,
        min_frac_closes_above_ma=0.70,
        require_close_below_ma25=True,
        min_break_volume_mult=0.0,
        local_high_pad=16,
    )
    data.update(overrides)
    return MTopParams(**data)


@dataclass(frozen=True)
class MTopPattern:
    first_high_idx: int
    second_high_idx: int
    neckline_idx: int
    first_high: float
    second_high: float
    neckline: float
    breakout_idx: int

    @property
    def peak(self) -> float:
        return max(self.first_high, self.second_high)

    @property
    def depth_pct(self) -> float:
        peak = self.peak
        return (peak - self.neckline) / peak if peak else 0.0

    @property
    def gap(self) -> int:
        return self.second_high_idx - self.first_high_idx


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    entry: float
    stop_loss: float
    target: float
    pattern: MTopPattern
    bar_idx: int
    ma200: float


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_pct: float
    exit_reason: str


@dataclass(frozen=True)
class CoinRow:
    symbol: str
    base: str
    quote_volume: float
    last_price: float
    tick_size: float
    rank: int = 0
    day_pct: float = 0.0


@dataclass
class ScanHit:
    row: CoinRow
    trade: TradeResult
    df: pd.DataFrame


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def get_json(path: str, params: Optional[dict] = None, retries: int = 5) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(BINANCE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def _tick_from_filters(filters: Sequence[dict]) -> float:
    for item in filters:
        if item.get("filterType") == "PRICE_FILTER":
            try:
                tick = float(item.get("tickSize") or 0)
            except (TypeError, ValueError):
                tick = 0.0
            if tick > 0:
                return tick
    return 0.00001


def rank_usdt_perps(
    symbols: Sequence[dict],
    tickers: Sequence[dict],
    limit: int = 50,
) -> list[CoinRow]:
    """USDT 永續（不含股票合約／指數）依 24h 成交額（quoteVolume）排序。"""
    ticker_map = {t.get("symbol"): t for t in tickers}
    ranked: list[CoinRow] = []
    for spec in symbols:
        if spec.get("quoteAsset") != "USDT":
            continue
        if spec.get("status") != "TRADING":
            continue
        if spec.get("contractType") != "PERPETUAL":
            continue
        if spec.get("underlyingType") == "INDEX":
            continue
        symbol = str(spec.get("symbol") or "")
        if not symbol:
            continue
        ticker = ticker_map.get(symbol) or {}
        try:
            quote_volume = float(ticker.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        try:
            last_price = float(ticker.get("lastPrice") or 0)
        except (TypeError, ValueError):
            last_price = 0.0
        ranked.append(
            CoinRow(
                symbol=symbol,
                base=str(spec.get("baseAsset") or symbol.replace("USDT", "")),
                quote_volume=quote_volume,
                last_price=last_price,
                tick_size=_tick_from_filters(spec.get("filters") or []),
            )
        )
    ranked.sort(key=lambda r: r.quote_volume, reverse=True)
    return [
        CoinRow(
            symbol=row.symbol,
            base=row.base,
            quote_volume=row.quote_volume,
            last_price=row.last_price,
            tick_size=row.tick_size,
            rank=i,
        )
        for i, row in enumerate(ranked[: max(0, limit)], 1)
    ]


def fetch_top_universe(limit: int = 50) -> list[CoinRow]:
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = get_json("/fapi/v1/ticker/24hr")
    if not isinstance(info, dict) or not isinstance(tickers, list):
        raise RuntimeError("幣安 exchangeInfo / ticker 格式不符")
    return rank_usdt_perps(info.get("symbols") or [], tickers, limit=limit)


def prev_cst_date(now: Optional[datetime] = None) -> date:
    cur = (now or datetime.now(CST)).astimezone(CST)
    return (cur - timedelta(days=1)).date()


def session_return_pct(df: pd.DataFrame, day: date) -> Optional[float]:
    """該 CST 日第一根開盤到最後一根收盤的漲跌幅。"""
    if df is None or df.empty:
        return None
    start = pd.Timestamp(day, tz=CST)
    end = start + pd.Timedelta(days=1)
    window = df.loc[(df.index >= start) & (df.index < end)]
    if len(window) < 6:
        return None
    open_px = float(window["open"].iloc[0])
    close_px = float(window["close"].iloc[-1])
    if open_px <= 0:
        return None
    return (close_px / open_px) - 1.0


def rank_by_day_pct(
    rows: Sequence[CoinRow],
    pct_by_symbol: Dict[str, float],
    limit: int = 10,
) -> list[CoinRow]:
    scored = [(pct_by_symbol[r.symbol], r) for r in rows if r.symbol in pct_by_symbol]
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[CoinRow] = []
    for i, (pct, row) in enumerate(scored[: max(0, limit)], 1):
        out.append(
            CoinRow(
                symbol=row.symbol,
                base=row.base,
                quote_volume=row.quote_volume,
                last_price=row.last_price,
                tick_size=row.tick_size,
                rank=i,
                day_pct=pct,
            )
        )
    return out


def fetch_gainer_universe(
    limit: int = 10,
    *,
    workers: int = 12,
    min_quote_volume: float = 5_000_000,
    day: Optional[date] = None,
) -> tuple[list[CoinRow], date]:
    """USDT 永續、可交易（24h 成交額 ≥ 5M），依前一個 CST 日漲幅取前 N。"""
    pool = [r for r in fetch_top_universe(limit=10_000) if r.quote_volume >= min_quote_volume]
    target_day = day or prev_cst_date()
    pcts: Dict[str, float] = {}

    def _one(row: CoinRow) -> tuple[str, Optional[float]]:
        try:
            hourly = fetch_klines(row.symbol, interval="1h", lookback_days=3, limit=80)
        except Exception:  # noqa: BLE001
            return row.symbol, None
        return row.symbol, session_return_pct(hourly, target_day)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_ex:
        futs = [pool_ex.submit(_one, row) for row in pool]
        done = 0
        for fut in as_completed(futs):
            done += 1
            symbol, pct = fut.result()
            if pct is not None:
                pcts[symbol] = pct
            if done % 40 == 0 or done == len(pool):
                print(f"  gainers {done}/{len(pool)}  ranked_day={target_day}")
    return rank_by_day_pct(pool, pcts, limit=limit), target_day


def display_name(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def fmt_px(price: float) -> str:
    ax = abs(price)
    if ax >= 1000:
        return f"{price:.2f}"
    if ax >= 1:
        return f"{price:.4f}"
    if ax >= 0.01:
        return f"{price:.5f}"
    if ax >= 0.0001:
        return f"{price:.6f}"
    return f"{price:.8f}"


def fetch_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    start_ms: Optional[int] = LISTING_MS,
    limit: int = 1500,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """拉幣安 U 本位永續 K 線。scan 用 lookback_days；單檔牛來預設從上市日。"""
    if lookback_days is not None:
        start_ms = int((time.time() - lookback_days * 86400) * 1000)
    rows: list[list] = []
    cur = int(start_ms or LISTING_MS)
    interval_ms = {
        "1m": 60_000,
        "5m": 5 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }.get(interval, 5 * 60 * 1000)
    while True:
        raw = get_json(
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit, "startTime": cur},
        )
        if not raw:
            break
        rows.extend(raw)
        nxt = int(raw[-1][0]) + interval_ms
        if nxt <= cur or len(raw) < limit:
            break
        cur = nxt

    seen: set[int] = set()
    uniq: list[list] = []
    for row in rows:
        ts = int(row[0])
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(row)

    if not uniq:
        raise RuntimeError(f"無法取得 {symbol} {interval} K 線")

    now_ms = int(time.time() * 1000)
    if int(uniq[-1][0]) + interval_ms > now_ms:
        uniq = uniq[:-1]
    if not uniq:
        raise RuntimeError(f"{symbol} 尚無已收盤 K 線")

    idx = pd.to_datetime([int(x[0]) for x in uniq], unit="ms", utc=True).tz_convert(CST)
    df = pd.DataFrame(
        {
            "open": [float(x[1]) for x in uniq],
            "high": [float(x[2]) for x in uniq],
            "low": [float(x[3]) for x in uniq],
            "close": [float(x[4]) for x in uniq],
            "volume": [float(x[5]) for x in uniq],
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Pattern / signals
# ---------------------------------------------------------------------------


def _is_swing_high(highs: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(highs) - lookback:
        return False
    window = list(highs[idx - lookback : idx + lookback + 1])
    pivot = highs[idx]
    return pivot == max(window) and window.count(pivot) == 1


def _find_swing_highs(highs: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(highs)) if _is_swing_high(highs, i, lookback)]


def _bump(funnel: Optional[Dict[str, int]], key: str, n: int = 1) -> None:
    if funnel is not None:
        funnel[key] = funnel.get(key, 0) + n


def detect_m_tops(
    df: pd.DataFrame,
    params: Optional[MTopParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> list[MTopPattern]:
    """偵測 M 頭，並在第二峰確認後找收盤跌破 MA200。"""
    params = params or default_params()
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    vols = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(closes))
    ma200 = sma(closes, params.ma_period)
    ma25 = sma(closes, 25) if params.require_close_below_ma25 else None
    look = params.swing_lookback
    swing_highs = _find_swing_highs(highs, look)
    _bump(funnel, "swing_highs", len(swing_highs))

    found: list[MTopPattern] = []
    for a, i1 in enumerate(swing_highs):
        for i2 in swing_highs[a + 1 :]:
            gap = i2 - i1
            if gap < params.min_bars_between:
                continue
            if gap > params.max_bars_between:
                break
            _bump(funnel, "pairs")
            h1, h2 = float(highs[i1]), float(highs[i2])
            avg = (h1 + h2) / 2.0
            if avg <= 0 or abs(h1 - h2) / avg > params.high_tolerance_pct:
                continue
            if h2 > h1 * (1.0 + params.second_high_max_overshoot):
                continue
            left = max(0, i1 - params.local_high_pad)
            right = min(len(highs), i2 + look + 1)
            span_high = float(highs[left:right].max())
            if span_high > max(h1, h2) + 1e-12:
                continue
            inner_lo, inner_hi = i1 + look, i2 - look
            if inner_hi < inner_lo:
                continue
            neck_i = int(np.argmin(lows[inner_lo : inner_hi + 1]) + inner_lo)
            if neck_i <= i1 + 2 or neck_i >= i2 - 2:
                continue
            neck = float(lows[neck_i])
            peak = max(h1, h2)
            depth = (peak - neck) / peak if peak else 0.0
            if depth < params.min_depth_pct:
                continue
            _bump(funnel, "m_shape")
            if np.isnan(ma200[i1]) or np.isnan(ma200[i2]):
                continue
            if not (closes[i1] > ma200[i1] and closes[i2] > ma200[i2]):
                continue
            if params.min_peak_above_ma_pct > 0:
                if h1 < ma200[i1] * (1.0 + params.min_peak_above_ma_pct):
                    continue
                if h2 < ma200[i2] * (1.0 + params.min_peak_above_ma_pct):
                    continue
            if params.max_neck_above_ma_pct < 1.0 and not np.isnan(ma200[neck_i]):
                if neck > ma200[neck_i] * (1.0 + params.max_neck_above_ma_pct):
                    continue
            if params.max_neck_below_ma_pct < 1.0 and not np.isnan(ma200[neck_i]):
                if neck < ma200[neck_i] * (1.0 - params.max_neck_below_ma_pct):
                    continue
            if params.min_frac_closes_above_ma > 0:
                span_c = closes[i1 : i2 + 1]
                span_m = ma200[i1 : i2 + 1]
                ok = ~np.isnan(span_m)
                if not ok.any() or float(np.mean(span_c[ok] > span_m[ok])) < params.min_frac_closes_above_ma:
                    continue
            _bump(funnel, "above_ma200")
            confirm = i2 + look
            if confirm >= len(closes):
                continue
            breakout: int | None = None
            last = min(confirm + params.max_bars_to_break, len(closes))
            for k in range(confirm, last):
                if np.isnan(ma200[k]) or np.isnan(ma200[k - 1]):
                    continue
                if not (closes[k - 1] >= ma200[k - 1] and closes[k] < ma200[k]):
                    continue
                if params.require_close_below_ma25 and ma25 is not None:
                    if np.isnan(ma25[k]) or closes[k] >= ma25[k]:
                        continue
                if params.min_break_volume_mult > 0:
                    prev = vols[max(0, k - 20) : k]
                    med = float(np.median(prev)) if len(prev) else 0.0
                    if med > 0 and vols[k] < params.min_break_volume_mult * med:
                        continue
                breakout = k
                break
            if breakout is None:
                continue
            _bump(funnel, "ma200_break")
            found.append(
                MTopPattern(
                    first_high_idx=i1,
                    second_high_idx=i2,
                    neckline_idx=neck_i,
                    first_high=h1,
                    second_high=h2,
                    neckline=neck,
                    breakout_idx=breakout,
                )
            )

    return _dedupe_patterns(found)


def _dedupe_patterns(patterns: Sequence[MTopPattern]) -> list[MTopPattern]:
    """同一根跌破 MA200 只留最像 M 頭的那個（峰更對稱 + 更深）。"""
    by_break: dict[int, MTopPattern] = {}
    scores: dict[int, float] = {}
    for p in patterns:
        avg = (p.first_high + p.second_high) / 2.0
        equal = 1.0 - abs(p.first_high - p.second_high) / avg if avg else 0.0
        score = p.depth_pct + equal
        cur = by_break.get(p.breakout_idx)
        if cur is None or score > scores[p.breakout_idx]:
            by_break[p.breakout_idx] = p
            scores[p.breakout_idx] = score
    return sorted(by_break.values(), key=lambda p: p.breakout_idx)


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def generate_signals(
    df: pd.DataFrame,
    params: Optional[MTopParams] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> list[Signal]:
    params = params or default_params()
    patterns = detect_m_tops(df, params, funnel=funnel)
    closes = df["close"].to_numpy(float)
    ma200 = sma(closes, params.ma_period)
    signals: list[Signal] = []
    for p in patterns:
        idx = p.breakout_idx
        entry = _round_tick(float(closes[idx]), params.tick_size)
        stop = _round_tick(p.peak, params.tick_size)
        risk = stop - entry
        if risk <= 0:
            continue
        target = _round_tick(entry - params.target_r * risk, params.tick_size)
        signals.append(
            Signal(
                timestamp=df.index[idx],
                entry=entry,
                stop_loss=stop,
                target=target,
                pattern=p,
                bar_idx=idx,
                ma200=float(ma200[idx]),
            )
        )
        _bump(funnel, "signals")
    return signals


def filter_entry_window(df: pd.DataFrame, signals: Sequence[Signal], days: int) -> list[Signal]:
    if not len(df) or days <= 0:
        return list(signals)
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    return [s for s in signals if s.timestamp >= start]


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    params: Optional[MTopParams] = None,
) -> list[TradeResult]:
    """做空：先碰到停損（高點）或目標（2R），否則時間停。同時只持一筆。"""
    params = params or default_params()
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    results: list[TradeResult] = []
    busy_until = -1

    for sig in signals:
        entry_idx = sig.bar_idx
        if entry_idx <= busy_until:
            continue
        entry = sig.entry
        stop = sig.stop_loss
        target = sig.target
        end_idx = min(entry_idx + params.time_bars, len(df) - 1)
        exit_idx = end_idx
        exit_price = float(closes[end_idx])
        reason = "time"
        if end_idx == len(df) - 1 and entry_idx + params.time_bars > end_idx:
            reason = "open"

        for k in range(entry_idx + 1, end_idx + 1):
            if highs[k] >= stop:
                exit_idx, exit_price, reason = k, stop, "stop"
                break
            if lows[k] <= target:
                exit_idx, exit_price, reason = k, target, "target"
                break

        pnl = (entry - exit_price) / entry if entry else 0.0
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=float(exit_price),
                stop_price=stop,
                target_price=target,
                pnl_pct=float(pnl),
                exit_reason=reason,
            )
        )
        busy_until = exit_idx
    return results


def summarize_trades(trades: Sequence[TradeResult]) -> dict:
    n = len(trades)
    if not n:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "closed_win_rate": 0.0,
            "total_pct": 0.0,
            "avg_pct": 0.0,
            "reasons": {},
        }
    closed = [t for t in trades if t.exit_reason != "open"]
    wins = sum(1 for t in closed if t.pnl_pct > 0)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    total = float(sum(t.pnl_pct for t in trades))
    return {
        "count": n,
        "wins": wins,
        "losses": len(closed) - wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "closed_win_rate": 100.0 * wins / len(closed) if closed else 0.0,
        "total_pct": total,
        "avg_pct": total / n,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Charts / HTML
# ---------------------------------------------------------------------------


def _setup_cjk() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break


def _equity_svg(pnls: List[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq]) * 100.0
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(0.5, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys_src[i]):.1f}" for i in range(len(ys_src)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def _trade_window(df: pd.DataFrame, trade: TradeResult, pad_left: int = 28, pad_right: int = 18) -> tuple[int, int]:
    """截圖從 15:50 看到 H1 18:10（約 28 根），跌破後還要留瀑布。"""
    p = trade.signal.pattern
    start = max(0, min(p.first_high_idx, trade.entry_idx) - pad_left)
    end = min(len(df) - 1, max(trade.exit_idx, trade.entry_idx, p.second_high_idx) + pad_right)
    return start, end


def _draw_candles(ax, axv, window: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.7)
        y0 = min(float(o.iloc[k]), float(c.iloc[k]))
        y1 = max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)


def _style_axes(ax, axv) -> None:
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")


def draw_trade_png(
    df: pd.DataFrame,
    trade: TradeResult,
    path: Path,
    trade_no: int,
    title_extra: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk()
    sig = trade.signal
    p = sig.pattern
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    close_full = df["close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(11.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    _style_axes(ax, axv)
    _draw_candles(ax, axv, window)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        lw = 2.5 if n == 200 else (1.25 if n <= 25 else 1.05)
        ax.plot(list(range(len(window))), ma, color=col, lw=lw, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(p.neckline, color="#f0c14a", ls="--", lw=1.0, alpha=0.75)

    h1 = p.first_high_idx - start
    h2 = p.second_high_idx - start
    nk = p.neckline_idx - start
    if 0 <= h1 < len(window) and 0 <= nk < len(window) and 0 <= h2 < len(window):
        ax.plot(
            [h1, nk, h2],
            [p.first_high, p.neckline, p.second_high],
            color="#f0c14a",
            lw=1.35,
            ls="--",
            alpha=0.85,
            zorder=4,
        )
    ex = trade.entry_idx - start
    xx = trade.exit_idx - start
    if 0 <= h1 < len(window):
        ax.scatter([h1], [p.first_high], s=42, color="#f0c14a", zorder=5)
        ax.annotate("H1", (h1, p.first_high), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#f0c14a", fontsize=8)
    if 0 <= h2 < len(window):
        ax.scatter([h2], [p.second_high], s=42, color="#f472b6", zorder=5)
        ax.annotate("H2", (h2, p.second_high), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#f9a8d4", fontsize=8)
    if 0 <= nk < len(window):
        ax.scatter([nk], [p.neckline], s=36, color="#79c0ff", zorder=5)
        ax.annotate("頸線", (nk, p.neckline), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=52, color="#e35d5d", marker="v", zorder=6)
        ax.annotate("S", (ex, trade.entry_price), textcoords="offset points", xytext=(8, -10),
                    ha="left", color="#ff7b72", fontsize=9)
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [xx],
            [trade.exit_price],
            s=40,
            color="#00c805" if trade.pnl_pct > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    extra = f"{title_extra}  " if title_extra else ""
    ax.set_title(
        f"#{trade_no}  {extra}{et.strftime('%m-%d %H:%M')} → {xt.strftime('%m-%d %H:%M')}  "
        f"{trade.exit_reason}  {trade.pnl_pct*100:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_overview_png(df: pd.DataFrame, trade: TradeResult, path: Path, title: str) -> Path:
    """截圖那一段：M 頭 + 跌破 MA200 做空。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk()
    p = trade.signal.pattern
    # 對齊截圖：進場當日下午到夜間
    day = df.index[trade.entry_idx].tz_convert(CST).normalize()
    left = day + pd.Timedelta(hours=15, minutes=50)
    right = day + pd.Timedelta(hours=23, minutes=35)
    mask = (df.index >= left) & (df.index <= right)
    if mask.sum() < 30:
        start, end = _trade_window(df, trade, pad_left=24, pad_right=16)
        window = df.iloc[start : end + 1]
        start_idx = start
    else:
        window = df.loc[mask]
        start_idx = int(df.index.get_loc(window.index[0]))

    close_full = df["close"].astype(float)
    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(11.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1]},
        facecolor="#0c1210",
    )
    _style_axes(ax, axv)
    _draw_candles(ax, axv, window)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start_idx : start_idx + len(window)]
        lw = 2.6 if n == 200 else (1.3 if n <= 25 else 1.1)
        ax.plot(list(range(len(window))), ma, color=col, lw=lw, label=f"MA{n}")

    ax.axhline(p.neckline, color="#f0c14a", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.75)
    h1r, h2r, nkr = p.first_high_idx - start_idx, p.second_high_idx - start_idx, p.neckline_idx - start_idx
    if 0 <= h1r < len(window) and 0 <= nkr < len(window) and 0 <= h2r < len(window):
        ax.plot(
            [h1r, nkr, h2r],
            [p.first_high, p.neckline, p.second_high],
            color="#f0c14a",
            lw=1.4,
            ls="--",
            alpha=0.85,
            zorder=4,
        )

    def _mark(idx: int, y: float, text: str, color: str, dy: int) -> None:
        rel = idx - start_idx
        if 0 <= rel < len(window):
            ax.scatter([rel], [y], s=44, color=color, zorder=5)
            ax.annotate(text, (rel, y), textcoords="offset points", xytext=(0, dy),
                        ha="center", color=color, fontsize=8)

    _mark(p.first_high_idx, p.first_high, "H1", "#f0c14a", 8)
    _mark(p.second_high_idx, p.second_high, "H2", "#f472b6", 8)
    _mark(p.neckline_idx, p.neckline, "頸線", "#79c0ff", -12)
    ex = trade.entry_idx - start_idx
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=64, color="#e35d5d", marker="v", zorder=6)
        ax.annotate("S 跌破MA200", (ex, trade.entry_price), textcoords="offset points",
                    xytext=(10, -12), ha="left", color="#ff7b72", fontsize=9)

    ax.set_title(title, color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            text=True,
        )
        return out.strip() or "main"
    except Exception:  # noqa: BLE001
        return "main"


def write_view_html(src: Path) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def write_html_report(
    path: Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    *,
    days: int,
    funnel: Optional[Dict[str, int]] = None,
) -> Path:
    stats = summarize_trades(trades)
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)

    overview_html = ""
    shot = next((t for t in reversed(trades) if t.signal.pattern.gap >= 20), trades[-1] if trades else None)
    if shot is not None:
        draw_overview_png(
            df,
            shot,
            img_dir / "overview.png",
            f"{SYMBOL_TW} 五分K · M頭跌破 MA200 做空 · {shot.signal.timestamp.strftime('%Y-%m-%d')}",
        )
        overview_html = (
            "<article class='trade-card'>"
            "<header class='card-header'><div class='card-title'>"
            "<span class='trade-no'>截圖這段</span>"
            f"<span class='trade-time'>{escape(shot.signal.timestamp.strftime('%Y-%m-%d %H:%M'))} 跌破 MA200</span>"
            "</div></header>"
            "<p class='muted' style='margin:0 0 10px'>左峰 H1、右峰 H2 在 MA200 上方結成 M 頭，"
            "收盤跌破紅色 MA200 進場做空（S）。停損在雙頂高點，目標 2R。</p>"
            "<div class='mini-chart'><img src='img/overview.png' alt='overview' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "open": "tag-info"}.get(t.exit_reason, "tag-time")
        img_name = f"t{i:02d}_{et.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, t, img_dir / img_name, i)
        p = t.signal.pattern
        risk = t.stop_price - t.entry_price
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · 做空</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>5m</span>"
            f"<span class='tag tag-info'>深度 {p.depth_pct*100:.1f}%</span>"
            f"<span class='tag tag-info'>{p.gap} 根</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.5f}  MA200 {t.signal.ma200:.5f}\n"
            f"stop  {t.stop_price:.5f}  （M頭高 −{risk:.5f}）\n"
            f"target {t.target_price:.5f}  （2R）\n"
            f"exit  {t.exit_price:.5f}  {t.exit_reason}  {t.pnl_pct*100:+.2f}%\n"
            f"H1 {df.index[p.first_high_idx].strftime('%m-%d %H:%M')} {p.first_high:.5f}\n"
            f"H2 {df.index[p.second_high_idx].strftime('%m-%d %H:%M')} {p.second_high:.5f}\n"
            f"頸線 {df.index[p.neckline_idx].strftime('%m-%d %H:%M')} {p.neckline:.5f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='#{i}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    fun = funnel or {}
    reasons = stats.get("reasons") or {}
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_pct"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{SYMBOL_TW} 五分K M頭跌破 MA200</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{SYMBOL_TW} 五分K · M頭跌破 MA200 做空</h1>
<p class="muted">幣安 U 本位永續 · 近 {days} 天 · {escape(start)} → {escape(end)} CST · {len(df)} 根
<br/>只做「牛來那種」M 頭：雙峰幾乎等高（價差 ≤ 1%、右峰不得明顯更高）、間隔約 1.5～4 小時、
中間至少跌 4%，兩個峰都明顯站在 MA200 上方（≥ 3%），頸線回測 MA200（不得遠高或深跌穿），
M 成形期間多數收盤仍在均線上，然後收盤同時跌破 MA200 與 MA25 才空。小振幅、貼均線亂鑽的假 M 不畫。
停損在雙頂高點、目標 2R、或 48 根時間停。加總％是各筆報酬相加，不是複利。</p>
<p class="muted">漏斗：轉折高 {fun.get('swing_highs', 0)} → 配對 {fun.get('pairs', 0)} → M形 {fun.get('m_shape', 0)}
→ 峰在均線上 {fun.get('above_ma200', 0)} → 跌破 MA200 {fun.get('ma200_break', 0)} → 訊號 {fun.get('signals', 0)}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">加總<b class="{total_cls}">{stats['total_pct']*100:+.2f}%</b></div>
<div class="card">平均<b class="{total_cls}">{stats['avg_pct']*100:+.2f}%</b></div>
</div>
<div class="equity">{_equity_svg([t.pnl_pct for t in trades])}</div>
</section>
{overview_html}
{''.join(cards) or "<div class='empty'>這段期間沒有 M 頭跌破 MA200 訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Multi-coin scan
# ---------------------------------------------------------------------------


def scan_symbol(
    row: CoinRow,
    days: int,
    entry_from: Optional[pd.Timestamp] = None,
) -> tuple[list[ScanHit], dict]:
    meta = {
        "symbol": row.symbol,
        "bars": 0,
        "error": "",
        "n_trade": 0,
        "quote_volume": row.quote_volume,
    }
    try:
        df = fetch_klines(row.symbol, lookback_days=max(days + 2, 4))
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:100]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 220:
        meta["error"] = "too_few_bars"
        return [], meta
    params = niulai_params(tick_size=row.tick_size)
    funnel: Dict[str, int] = {}
    sigs = generate_signals(df, params, funnel=funnel)
    sigs = filter_entry_window(df, sigs, days)
    if entry_from is not None:
        sigs = [s for s in sigs if s.timestamp >= entry_from]
    trades = simulate(df, sigs, params)
    meta["n_trade"] = len(trades)
    meta["funnel"] = funnel
    return [ScanHit(row, t, df) for t in trades], meta


def write_scan_html(
    path: Path,
    hits: List[ScanHit],
    universe: List[CoinRow],
    *,
    days: int,
    funnel: Optional[Dict[str, int]] = None,
    heading: Optional[str] = None,
    blurb: Optional[str] = None,
) -> Path:
    stats = summarize_trades([h.trade for h in hits])
    img_dir = path.parent / "img"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)

    cards: List[str] = []
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        df = hit.df
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_pct > 0 else ("pnl-flat" if t.pnl_pct == 0 else "pnl-loss")
        reason_cls = {"target": "tag-tp", "stop": "tag-sl", "open": "tag-info"}.get(t.exit_reason, "tag-time")
        label = display_name(hit.row.symbol)
        safe = "".join(ch if ch.isalnum() else "_" for ch in hit.row.symbol)
        img_name = f"t{i:02d}_{safe}_{et.strftime('%m%d_%H%M')}.png"
        draw_trade_png(df, t, img_dir / img_name, i, title_extra=label)
        p = t.signal.pattern
        risk = t.stop_price - t.entry_price
        qv = hit.row.quote_volume / 1e6
        extra_tags = ""
        if hit.row.day_pct:
            extra_tags += f"<span class='tag tag-info'>昨日 {hit.row.day_pct*100:+.1f}%</span>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)} · 做空</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_pct*100:+.2f}%</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag tag-info'>#{hit.row.rank} {escape(hit.row.symbol)}</span>"
            f"{extra_tags}"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>5m</span>"
            f"<span class='tag tag-info'>深度 {p.depth_pct*100:.1f}%</span>"
            f"<span class='tag tag-info'>{qv:.0f}M</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {fmt_px(t.entry_price)}  MA200 {fmt_px(t.signal.ma200)}\n"
            f"stop  {fmt_px(t.stop_price)}  （M頭高 −{fmt_px(risk)}）\n"
            f"target {fmt_px(t.target_price)}  （2R）\n"
            f"exit  {fmt_px(t.exit_price)}  {t.exit_reason}  {t.pnl_pct*100:+.2f}%\n"
            f"H1 {df.index[p.first_high_idx].strftime('%m-%d %H:%M')} {fmt_px(p.first_high)}\n"
            f"H2 {df.index[p.second_high_idx].strftime('%m-%d %H:%M')} {fmt_px(p.second_high)}\n"
            f"頸線 {df.index[p.neckline_idx].strftime('%m-%d %H:%M')} {fmt_px(p.neckline)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    fun = funnel or {}
    reasons = stats.get("reasons") or {}
    cutoff = universe[-1].quote_volume / 1e6 if universe else 0
    names = {h.row.symbol for h in hits}
    total_cls = "pnl-win" if stats["total_pct"] >= 0 else "pnl-loss"
    title = heading or f"幣安成交額前 {len(universe)} · 牛來型 M 頭做空"
    if blurb is None:
        blurb = (
            f"USDT 永續（不含股票合約）· 24h 成交額前 {len(universe)} · 近 {days} 天"
            f" · 末名約 {cutoff:.0f}M USDT · 只掃牛來那種 M 頭"
        )
    uni_line = ""
    if any(r.day_pct for r in universe):
        uni_line = (
            "<p class='muted'>"
            + " · ".join(
                f"#{r.rank} {escape(display_name(r.symbol))} {r.day_pct*100:+.1f}%"
                for r in universe
            )
            + "</p>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(title)}</h1>
<p class="muted">{blurb}
<br/>只掃「牛來那種」M 頭：雙峰幾乎等高、中間至少跌 4%、雙峰明顯站上 MA200、頸線回測均線（不深跌穿）、成形期間多數收盤在均線上，收盤同時跌破 MA200 與 MA25 才空。貼均線亂鑽的小 M 不畫。停損雙頂高點、2R、或 48 根時間停。加總％是各筆報酬相加，不是複利。</p>
{uni_line}
<p class="muted">漏斗：轉折高 {fun.get('swing_highs', 0)} → 配對 {fun.get('pairs', 0)} → M形 {fun.get('m_shape', 0)}
→ 峰在均線上 {fun.get('above_ma200', 0)} → 跌破 MA200 {fun.get('ma200_break', 0)} → 訊號 {fun.get('signals', 0)}
<br/>出場：2R {reasons.get('target', 0)} · 停損 {reasons.get('stop', 0)} · 時間 {reasons.get('time', 0)} · 未平 {reasons.get('open', 0)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['closed_win_rate']:.1f}%</b></div>
<div class="card">加總<b class="{total_cls}">{stats['total_pct']*100:+.2f}%</b></div>
<div class="card">標的<b>{len(names)}</b></div>
</div>
<div class="equity">{_equity_svg([h.trade.pnl_pct for h in hits])}</div>
</section>
{''.join(cards) or f"<div class='empty'>這段期間前 {len(universe)} 檔沒有 M 頭跌破 MA200 訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def run_scan(
    limit: int = 50,
    days: int = 3,
    workers: int = 8,
    pages: bool = False,
    html_path: Optional[Path] = None,
    *,
    universe: Optional[List[CoinRow]] = None,
    pages_path: Optional[Path] = None,
    heading: Optional[str] = None,
    blurb: Optional[str] = None,
    entry_from: Optional[pd.Timestamp] = None,
) -> int:
    if universe is None:
        universe = fetch_top_universe(limit)
    if not universe:
        print("no universe")
        return 1
    extra = ""
    if universe[0].day_pct:
        extra = f"  {universe[0].day_pct*100:+.1f}%"
    print(
        f"universe {len(universe)}  #{universe[0].rank} {universe[0].symbol} "
        f"{universe[0].quote_volume/1e6:.0f}M{extra} · 末 {universe[-1].symbol} "
        f"{universe[-1].quote_volume/1e6:.0f}M"
    )
    hits: list[ScanHit] = []
    funnel: Dict[str, int] = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(scan_symbol, row, days, entry_from): row for row in universe}
        done = 0
        for fut in as_completed(futs):
            row = futs[fut]
            done += 1
            try:
                coin_hits, meta = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{done:2d}/{len(universe)}] {row.symbol} FAIL {exc}")
                continue
            if meta.get("error"):
                errors += 1
            hits.extend(coin_hits)
            for k, v in (meta.get("funnel") or {}).items():
                funnel[k] = funnel.get(k, 0) + v
            flag = f" trades={meta['n_trade']}" if meta.get("n_trade") else ""
            err = f" {meta.get('error')}" if meta.get("error") else ""
            print(f"[{done:2d}/{len(universe)}] {row.symbol} bars={meta.get('bars', 0)}{flag}{err}")

    hits.sort(key=lambda h: h.df.index[h.trade.entry_idx])
    stats = summarize_trades([h.trade for h in hits])
    print(
        f"done errors={errors} trades={stats['count']} symbols={len({h.row.symbol for h in hits})} "
        f"WR={stats['closed_win_rate']:.1f}% sum={stats['total_pct']*100:+.2f}%"
    )
    for i, hit in enumerate(hits, 1):
        t = hit.trade
        print(
            f"  [{i}] {hit.row.symbol} {hit.df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason:6} {t.pnl_pct*100:+.2f}%"
        )

    out = html_path
    if pages:
        out = pages_path or SCAN_PAGES
    if out:
        write_scan_html(
            Path(out),
            hits,
            universe,
            days=days,
            funnel=funnel,
            heading=heading,
            blurb=blurb,
        )
        view = write_view_html(Path(out))
        print(f"html={Path(out).resolve()}")
        print(f"view={view.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_backtest(days: int = 7, html_path: Optional[Path] = None, pages: bool = False) -> int:
    df = fetch_klines()
    params = niulai_params()
    funnel: Dict[str, int] = {}
    sigs = generate_signals(df, params, funnel=funnel)
    sigs = filter_entry_window(df, sigs, days)
    trades = simulate(df, sigs, params)
    stats = summarize_trades(trades)
    print(f"{SYMBOL} 5m  {df.index[0]} → {df.index[-1]}  bars={len(df)}")
    print(
        f"days={days} trades={stats['count']} WR={stats['closed_win_rate']:.1f}% "
        f"sum={stats['total_pct']*100:+.2f}% avg={stats['avg_pct']*100:+.2f}%"
    )
    if funnel:
        print(
            "funnel "
            f"swing={funnel.get('swing_highs', 0)} pairs={funnel.get('pairs', 0)} "
            f"m={funnel.get('m_shape', 0)} above={funnel.get('above_ma200', 0)} "
            f"break={funnel.get('ma200_break', 0)} sig={funnel.get('signals', 0)}"
        )
    for i, t in enumerate(trades, 1):
        p = t.signal.pattern
        print(
            f"[{i}] {df.index[t.entry_idx].strftime('%m-%d %H:%M')} → "
            f"{df.index[t.exit_idx].strftime('%m-%d %H:%M')} {t.exit_reason:6} "
            f"{t.pnl_pct*100:+.2f}%  entry {t.entry_price:.5f}  "
            f"H1 {df.index[p.first_high_idx].strftime('%H:%M')} / "
            f"H2 {df.index[p.second_high_idx].strftime('%H:%M')}"
        )

    out = html_path
    if pages:
        out = PAGES
    if out:
        write_html_report(Path(out), df, trades, days=days, funnel=funnel)
        view = write_view_html(Path(out))
        print(f"html={Path(out).resolve()}")
        print(f"view={view.resolve()}")
    return 0


def run_gainers_scan(
    limit: int = 10,
    days: int = 2,
    workers: int = 12,
    pages: bool = False,
    html_path: Optional[Path] = None,
) -> int:
    print("ranking previous CST day gainers…")
    universe, day = fetch_gainer_universe(limit=limit, workers=workers)
    if not universe:
        print("no gainers")
        return 1
    for row in universe:
        print(f"  #{row.rank} {row.symbol}  {row.day_pct*100:+.1f}%  qv={row.quote_volume/1e6:.0f}M")
    heading = f"前一天漲幅榜前 {len(universe)} · 牛來型 M 頭做空"
    blurb = (
        f"USDT 永續、24h 成交額 ≥ 5M · {day.isoformat()} CST 全日漲幅前 {len(universe)}"
        f" · 進場從該日 00:00 CST 起。只畫牛來那種對稱深 M，不是任何兩個小高點。"
    )
    return run_scan(
        limit=limit,
        days=days,
        workers=min(workers, 8),
        pages=pages,
        html_path=html_path,
        universe=universe,
        pages_path=GAINERS_PAGES,
        heading=heading,
        blurb=blurb,
        entry_from=pd.Timestamp(day, tz=CST),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="牛来 / 幣安成交額前 N · 五分K M頭跌破 MA200 做空")
    p.add_argument("--days", type=int, default=None, help="回測進場窗（天）；單檔預設 7，scan 預設 3，漲幅榜預設 2")
    p.add_argument("--html", default="", help="輸出 HTML 路徑")
    p.add_argument("--pages", action="store_true", help="寫到 docs/（單檔或 scan 目錄）")
    p.add_argument("--scan", action="store_true", help="掃 USDT 永續成交額前 N 檔")
    p.add_argument("--gainers", action="store_true", help="改掃前一個 CST 日漲幅榜前 N 檔")
    p.add_argument("--limit", type=int, default=None, help="scan 檔數；成交額預設 50，漲幅榜預設 10")
    p.add_argument("--workers", type=int, default=8, help="scan 並行數")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    html_path = Path(args.html) if args.html else None
    if args.gainers:
        return run_gainers_scan(
            limit=10 if args.limit is None else args.limit,
            days=2 if args.days is None else args.days,
            workers=max(args.workers, 12),
            pages=args.pages,
            html_path=html_path,
        )
    if args.scan:
        return run_scan(
            limit=50 if args.limit is None else args.limit,
            days=3 if args.days is None else args.days,
            workers=args.workers,
            pages=args.pages,
            html_path=html_path,
        )
    return run_backtest(
        days=7 if args.days is None else args.days,
        html_path=html_path,
        pages=args.pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
