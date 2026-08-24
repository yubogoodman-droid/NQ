"""NQ 均線糾結後放量突破（起漲點）。

對應手機圖那種走法：先跌、均線收成一束、窄幅盤整，再放量長綠 K
一次站上盤整高點與整束均線。起漲點是突破那根，不是最低點。

訊號只在 1 分 K 上抓。5 分用來對照「1 分條件成立當下」那根 5 分長什麼樣。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

MA_PERIODS: Tuple[int, ...] = (5, 10, 20, 30, 60, 100, 120, 200)


@dataclass
class CoilSignal:
    coil_start_idx: int
    coil_end_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    coil_high: float
    coil_low: float
    coil_range: float
    ribbon_width: float
    vol_ratio: float
    prior_drop: float
    body: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    ma100: float
    ma120: float
    ma200: float
    m5_close: float = float("nan")
    m5_ma20: float = float("nan")
    m5_ma200: float = float("nan")
    m5_dist: float = float("nan")
    d30: float = float("nan")
    d30_dist: float = float("nan")
    quality: str = "C"
    quality_score: int = 0


@dataclass
class CoilTrade:
    signal: CoilSignal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_points: float
    exit_reason: str
    quality: str


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def _col(df: pd.DataFrame, name: str) -> str:
    wanted = name.lower()
    for c in df.columns:
        if str(c).lower() == wanted:
            return c
    raise KeyError(name)


def _ohlcv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = df[_col(df, "open")].to_numpy(float)
    h = df[_col(df, "high")].to_numpy(float)
    l = df[_col(df, "low")].to_numpy(float)
    c = df[_col(df, "close")].to_numpy(float)
    try:
        v = df[_col(df, "volume")].to_numpy(float)
    except KeyError:
        v = np.ones(len(df), dtype=float)
    return o, h, l, c, v


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """把 1 分 OHLCV 聚合成更高週期（右標、右閉，與 Yahoo 5m 對齊）。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_ohlcv requires a DatetimeIndex")
    work = df.copy()
    rename = {}
    for c in work.columns:
        key = str(c).lower()
        if key == "open":
            rename[c] = "Open"
        elif key == "high":
            rename[c] = "High"
        elif key == "low":
            rename[c] = "Low"
        elif key == "close":
            rename[c] = "Close"
        elif key == "volume":
            rename[c] = "Volume"
    work = work.rename(columns=rename)
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    use = {k: v for k, v in agg.items() if k in work.columns}
    out = work[list(use)].resample(rule, label="right", closed="right").agg(use)
    if "Close" in out.columns:
        out = out.dropna(subset=["Close"])
    else:
        out = out.dropna(how="all")
    return out


def m5_asof(df_1m: pd.DataFrame, ts) -> pd.DataFrame:
    """只看到 ts 為止的 5 分 K。最後一根可能還沒收完（當下長相，不偷看後面）。"""
    if df_1m is None or len(df_1m) == 0:
        return resample_ohlcv(df_1m)
    ts = pd.Timestamp(ts)
    if df_1m.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(df_1m.index.tz)
    elif df_1m.index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return resample_ohlcv(df_1m.loc[df_1m.index <= ts], "5min")


def m5_look_at(df_1m: pd.DataFrame, ts) -> Optional[dict]:
    """1 分進場當下，對應那根 5 分的 OHLC / 均線，以及這根後來收完的價。"""
    snap = m5_asof(df_1m, ts)
    if snap.empty:
        return None
    last = snap.iloc[-1]
    full = resample_ohlcv(df_1m, "5min")
    finished = full.loc[last.name] if last.name in full.index else last
    close = snap["Close"].astype(float)
    mas = {}
    for n in (5, 10, 20, 30, 60, 120, 200):
        series = close.rolling(n, min_periods=n).mean()
        mas[n] = float(series.iloc[-1]) if len(series) else float("nan")
    body = float(last["Close"]) - float(last["Open"])
    ma5, ma10, ma20, ma30, ma200 = mas[5], mas[10], mas[20], mas[30], mas[200]
    stack = (not any(np.isnan(x) for x in (ma5, ma10, ma20, ma30))) and (
        ma5 > ma10 > ma20 > ma30
    )
    above_200 = (not np.isnan(ma200)) and float(last["Close"]) > ma200
    above_20 = (not np.isnan(ma20)) and float(last["Close"]) > ma20
    return {
        "bar_time": last.name,
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": float(last["Close"]),
        "volume": float(last["Volume"]) if "Volume" in snap.columns else 0.0,
        "body": body,
        "finished_open": float(finished["Open"]),
        "finished_high": float(finished["High"]),
        "finished_low": float(finished["Low"]),
        "finished_close": float(finished["Close"]),
        "forming": abs(float(last["Close"]) - float(finished["Close"])) > 0.05
        or abs(float(last["High"]) - float(finished["High"])) > 0.05,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma30": ma30,
        "ma60": mas[60],
        "ma120": mas[120],
        "ma200": ma200,
        "stack": stack,
        "above_20": above_20,
        "above_200": above_200,
        "snap": snap,
    }


def quality_of(coil_range: float, ribbon_width: float, vol_ratio: float, body: float) -> Tuple[int, str]:
    score = 0
    if coil_range <= 30.0:
        score += 1
    if ribbon_width <= 25.0:
        score += 1
    if vol_ratio >= 1.8:
        score += 1
    # 起漲點應是剛站上，不是已經噴出去的長實體
    if body <= 25.0:
        score += 1
    if score >= 4:
        return score, "A"
    if score >= 2:
        return score, "B"
    return score, "C"


def m5_asof_mas(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每個 1 分時刻：當時 5 分當根收盤、5分 MA20、5分 MA200、收盤相對 MA200 距離（不偷看後面）。"""
    n = 0 if df is None else len(df)
    close5 = np.full(n, np.nan, dtype=float)
    ma20 = np.full(n, np.nan, dtype=float)
    ma200 = np.full(n, np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)
    if n == 0:
        return close5, ma20, ma200, dist
    _o, _h, _l, c, _v = _ohlcv(df)
    idx = df.index
    completed: List[float] = []
    cur_bucket = None
    cur_c = float("nan")
    for i in range(n):
        ts = pd.Timestamp(idx[i])
        bucket = ts.ceil("5min")
        if cur_bucket is None or bucket != cur_bucket:
            if cur_bucket is not None and not np.isnan(cur_c):
                completed.append(cur_c)
            cur_bucket = bucket
        cur_c = float(c[i])
        close5[i] = cur_c
        if len(completed) >= 19:
            ma20[i] = (float(sum(completed[-19:])) + cur_c) / 20.0
        if len(completed) >= 199:
            ma = (float(sum(completed[-199:])) + cur_c) / 200.0
            ma200[i] = ma
            dist[i] = cur_c - ma
    return close5, ma20, ma200, dist


def m5_asof_ma200_dist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每個 1 分時刻：當時 5 分當根收盤、5分 MA200、收盤相對 MA200 的距離（不偷看後面）。"""
    close5, _ma20, ma200, dist = m5_asof_mas(df)
    return close5, ma200, dist


def nq_session_key(idx: pd.DatetimeIndex) -> np.ndarray:
    """CME 日盤日期鍵（YYYYMMDD）。18:00 ET 起算下一交易日。"""
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("nq_session_key requires a DatetimeIndex")
    t = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize(ET)
    extra = np.where(t.hour.to_numpy() >= 18, 1, 0)
    sess = t.normalize() + pd.to_timedelta(extra, unit="D")
    return (sess.year * 10000 + sess.month * 100 + sess.day).to_numpy(dtype=np.int32)


def _daily_close_series(daily: pd.DataFrame) -> pd.Series:
    col = None
    for c in daily.columns:
        if str(c).lower() == "close":
            col = c
            break
    if col is None:
        raise KeyError("Close")
    s = daily[col].astype(float)
    idx = daily.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(ET)
    out = pd.Series(s.to_numpy(float), index=pd.DatetimeIndex(idx.date))
    return out[~out.index.duplicated(keep="last")].sort_index()


def d30_asof(
    df: pd.DataFrame,
    daily: Optional[pd.DataFrame] = None,
    period: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """每個 1 分時刻：當時日線 MA30（含當根收盤當今日形成中收盤），以及相對距離。不偷看後面。"""
    n = 0 if df is None else len(df)
    ma = np.full(n, np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)
    if n == 0 or period < 2:
        return ma, dist
    _o, _h, _l, c, _v = _ohlcv(df)
    keys = nq_session_key(df.index)
    completed: List[float] = []
    if daily is not None and len(daily):
        hist = _daily_close_series(daily)
        first = int(keys[0])
        for dt, px in hist.items():
            ymd = int(dt.year) * 10000 + int(dt.month) * 100 + int(dt.day)
            if ymd < first:
                completed.append(float(px))
    cur_sess: Optional[int] = None
    cur_c = float("nan")
    need = int(period) - 1
    for i in range(n):
        sess = int(keys[i])
        if cur_sess is None or sess != cur_sess:
            if cur_sess is not None and not np.isnan(cur_c):
                completed.append(cur_c)
            cur_sess = sess
        cur_c = float(c[i])
        if len(completed) >= need:
            ma[i] = (float(sum(completed[-need:])) + cur_c) / float(period)
            dist[i] = cur_c - ma[i]
    return ma, dist


@dataclass
class _LiveCoil:
    start_idx: int
    end_idx: int
    high: float
    low: float
    range: float
    ribbon_width: float
    prior_drop: float


def detect_coil_breakouts(
    df: pd.DataFrame,
    *,
    coil_bars: int = 15,
    min_coil_bars: int = 10,
    max_coil_bars: int = 18,
    max_coil_range: float = 36.0,
    min_coil_range: float = 10.0,
    max_ribbon_width: float = 42.0,
    min_body: float = 0.0,
    min_vol_ratio: float = 2.0,
    vol_lookback: int = 60,
    min_prior_drop: float = 30.0,
    prior_lookback: int = 120,
    hug_buffer: float = 18.0,
    min_hug_frac: float = 0.55,
    min_break_over: float = 1.0,
    max_ma200_drop: float = 25.0,
    ma200_slope_bars: int = 20,
    max_coil_vs_prior: float = 0.70,
    break_window: int = 8,
    stop_buffer: float = 5.0,
    target_r: float = 2.0,
    min_entry_gap: int = 45,
    max_body: float = 40.0,
    max_m5_below_200: float = -1.0,
    require_m5_above_ma20: bool = True,
    require_above_d30: bool = True,
    daily: Optional[pd.DataFrame] = None,
    d30_period: int = 30,
    ma_periods: Sequence[int] = MA_PERIODS,
    funnel: Optional[Dict[str, int]] = None,
) -> List[CoilSignal]:
    """
    抓起漲點：先鎖住均線糾結的盤整箱，再允許之後幾根放量站上箱頂。

    進場是「第一次」收盤站上 MA200，且 MA5>MA10>MA20>MA30、
    MA60 與 MA120 仍在 MA200 下方。不要等後面那根放量長綠。

    另外不接已經噴超過 max_body 的長實體。進場當下，當時 5 分收盤要在
    5 分 MA20 上方，也要站上日線 30 日均線（require_above_d30；日線不夠時略過）。
    5 分 MA200 深度過濾預設關掉（max_m5_below_200<0）。
    """
    if df is None or len(df) == 0:
        return []
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    mas = [sma(c, int(p)) for p in ma_periods]
    _m5_c, m5_ma20, m5_ma200, m5_dist = m5_asof_mas(df)
    d30_ma, d30_dist = d30_asof(df, daily=daily, period=d30_period)
    warmup = max(int(p) for p in ma_periods)
    lo = min(min_coil_bars, coil_bars)
    hi = max(max_coil_bars, coil_bars)
    signals: List[CoilSignal] = []
    last_entry = -(10**9)
    last_coil: Optional[_LiveCoil] = None
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    def find_coil(i: int) -> Optional[_LiveCoil]:
        best: Optional[_LiveCoil] = None
        for length in range(lo, hi + 1):
            start = i - length
            if start < 0:
                continue
            coil_high = float(np.max(h[start:i]))
            coil_low = float(np.min(l[start:i]))
            coil_range = coil_high - coil_low
            if coil_range > max_coil_range or coil_range < min_coil_range:
                continue
            ribbon_prev = np.array([float(ma[i - 1]) for ma in mas], dtype=float)
            ribbon_width = float(ribbon_prev.max() - ribbon_prev.min())
            if ribbon_width > max_ribbon_width:
                continue
            look_from = max(0, start - prior_lookback)
            prior_drop = float(np.max(h[look_from:i]) - np.min(l[look_from:i]))
            if prior_drop < min_prior_drop:
                continue
            pre_start = max(0, start - prior_lookback)
            if start > pre_start + 5:
                pre_range = float(np.max(h[pre_start:start]) - np.min(l[pre_start:start]))
                if pre_range > 0 and coil_range > max_coil_vs_prior * pre_range:
                    continue
            ma200 = mas[-1]
            if (
                i - 1 >= ma200_slope_bars
                and not np.isnan(ma200[i - 1])
                and not np.isnan(ma200[i - 1 - ma200_slope_bars])
            ):
                if float(ma200[i - 1] - ma200[i - 1 - ma200_slope_bars]) < -max_ma200_drop:
                    continue
            inside = 0
            for j in range(start, i):
                rhi = max(float(ma[j]) for ma in mas)
                rlo = min(float(ma[j]) for ma in mas)
                if np.isnan(rhi) or np.isnan(rlo):
                    continue
                if (rlo - hug_buffer) <= c[j] <= (rhi + hug_buffer):
                    inside += 1
            if inside < length * min_hug_frac:
                continue
            cand = _LiveCoil(
                start_idx=start,
                end_idx=i - 1,
                high=coil_high,
                low=coil_low,
                range=coil_range,
                ribbon_width=ribbon_width,
                prior_drop=prior_drop,
            )
            if best is None or cand.range < best.range or (
                cand.range == best.range and length > (best.end_idx - best.start_idx + 1)
            ):
                best = cand
        return best

    i = warmup
    while i < n:
        if any(np.isnan(ma[i]) or np.isnan(ma[i - 1]) for ma in mas):
            i += 1
            continue
        bump("checked")

        if last_coil is not None and (i - last_coil.end_idx) > break_window:
            last_coil = None

        if last_coil is not None and i - last_entry >= min_entry_gap:
            bump("sticky")
            body = float(c[i] - o[i])
            vol_win = v[max(0, i - vol_lookback) : i]
            vol_ref = float(np.median(vol_win)) if len(vol_win) else 0.0
            vol_ratio = float(v[i] / vol_ref) if vol_ref > 1e-9 else 1.0
            ma5 = float(mas[0][i])
            ma10 = float(mas[1][i])
            ma20 = float(mas[2][i])
            ma30 = float(mas[3][i])
            ma60 = float(mas[4][i])
            ma120 = float(mas[6][i])
            ma200 = float(mas[7][i])
            ok_body = body >= min_body
            ok_max_body = max_body <= 0 or body <= max_body
            ok_break = c[i] >= last_coil.high + min_break_over
            ok_stack = ma5 > ma10 > ma20 > ma30
            ok_above_200 = c[i] > ma200
            ok_long_below = ma60 < ma200 and ma120 < ma200
            ok_vol = vol_ref <= 1e-9 or vol_ratio >= min_vol_ratio
            m5d = float(m5_dist[i])
            ok_m5 = (
                max_m5_below_200 < 0
                or np.isnan(m5d)
                or m5d > -max_m5_below_200
            )
            m5_20 = float(m5_ma20[i])
            ok_m5_20 = (not require_m5_above_ma20) or np.isnan(m5_20) or float(_m5_c[i]) > m5_20
            d30v = float(d30_ma[i])
            ok_d30 = (not require_above_d30) or np.isnan(d30v) or float(_m5_c[i]) > d30v
            if ok_body:
                bump("body")
            if ok_max_body:
                bump("not_chase")
            if ok_break:
                bump("above_coil")
            if ok_stack:
                bump("stack")
            if ok_above_200:
                bump("above_200")
            if ok_long_below:
                bump("long_below")
            if ok_vol:
                bump("volume")
            if ok_m5:
                bump("m5_ok")
            if ok_m5_20:
                bump("m5_ma20")
            if ok_d30:
                bump("d30")
            if (
                ok_body
                and ok_max_body
                and ok_break
                and ok_stack
                and ok_above_200
                and ok_long_below
                and ok_vol
                and ok_m5
                and ok_m5_20
                and ok_d30
            ):
                entry = float(c[i])
                stop = last_coil.low - stop_buffer
                risk = entry - stop
                if risk > 0:
                    bump("taken")
                    q_score, q_grade = quality_of(
                        last_coil.range, last_coil.ribbon_width, vol_ratio, body
                    )
                    signals.append(
                        CoilSignal(
                            coil_start_idx=last_coil.start_idx,
                            coil_end_idx=last_coil.end_idx,
                            entry_idx=i,
                            entry_price=entry,
                            stop_price=float(stop),
                            target_price=float(entry + risk * target_r),
                            coil_high=last_coil.high,
                            coil_low=last_coil.low,
                            coil_range=last_coil.range,
                            ribbon_width=last_coil.ribbon_width,
                            vol_ratio=vol_ratio,
                            prior_drop=last_coil.prior_drop,
                            body=body,
                            ma5=ma5,
                            ma10=ma10,
                            ma20=ma20,
                            ma30=ma30,
                            ma60=ma60,
                            ma100=float(mas[5][i]),
                            ma120=ma120,
                            ma200=ma200,
                            m5_close=float(_m5_c[i]),
                            m5_ma20=m5_20,
                            m5_ma200=float(m5_ma200[i]),
                            m5_dist=m5d,
                            d30=d30v,
                            d30_dist=float(d30_dist[i]),
                            quality=q_grade,
                            quality_score=q_score,
                        )
                    )
                    last_entry = i
                    last_coil = None
                    i += min_entry_gap
                    continue

        coil = find_coil(i)
        if coil is not None:
            bump("coil")
            # 價格已離開舊箱時不要用起漲 K 重算盤整，否則箱子會被撐破
            if last_coil is None or c[i] <= last_coil.high:
                last_coil = coil
        i += 1
    return signals


def simulate(
    df: pd.DataFrame,
    signals: Sequence[CoilSignal],
    *,
    max_hold: int = 90,
    be_after_r: float = 0.70,
    fail_closes: int = 2,
    trail_after_r: float = 1.0,
    trail_lock_r: float = 0.3,
) -> List[CoilTrade]:
    """停損＝盤整低點；0.7R 後保本；走到 1R 後停損移到 +0.3R（移動停利）；目標 2R。

    若連續 fail_closes 根收回到盤整高點下方，視為突破失敗，用收盤出場，
    不再等到盤整低點。
    """
    if not signals:
        return []
    _o, h, l, c, _v = _ohlcv(df)
    n = len(c)
    out: List[CoilTrade] = []
    for sig in signals:
        entry_idx = sig.entry_idx
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = entry - stop
        if risk <= 0:
            continue
        cur_stop = stop
        mfe = 0.0
        fail_n = 0
        limit = min(entry_idx + max_hold, n - 1)
        exit_idx = limit
        exit_price = float(c[exit_idx])
        exit_reason = "timeout"
        for k in range(entry_idx + 1, limit + 1):
            mfe = max(mfe, float(h[k] - entry))
            rr = mfe / risk
            if be_after_r > 0 and rr >= be_after_r:
                cur_stop = max(cur_stop, entry)
            if trail_after_r > 0 and trail_lock_r > 0 and rr >= trail_after_r:
                cur_stop = max(cur_stop, entry + risk * trail_lock_r)
            if l[k] <= cur_stop:
                if cur_stop > entry + 1e-9:
                    reason = "trail"
                else:
                    reason = "stop"
                exit_idx, exit_price, exit_reason = k, float(cur_stop), reason
                break
            if h[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break
            if fail_closes > 0 and float(c[k]) < sig.coil_high:
                fail_n += 1
                if fail_n >= fail_closes:
                    exit_idx, exit_price, exit_reason = k, float(c[k]), "fail"
                    break
            else:
                fail_n = 0
        out.append(
            CoilTrade(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(exit_price - entry),
                exit_reason=exit_reason,
                quality=sig.quality,
            )
        )
    return out


def summarize_trades(trades: Sequence[CoilTrade]) -> dict:
    pnls = [float(t.pnl_points) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_q: Dict[str, List[float]] = {}
    for t in trades:
        by_q.setdefault(t.quality, []).append(float(t.pnl_points))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "by_quality": {
            q: {"n": len(v), "wins": sum(1 for p in v if p > 0), "pnl": float(sum(v))}
            for q, v in sorted(by_q.items())
        },
    }


def make_coil_demo_bars(n: int = 420) -> pd.DataFrame:
    """模擬那張 1 分鐘圖：長盤後下殺、均線糾結、07:35 放量突破。"""
    close = np.zeros(n, dtype=float)
    high = np.zeros(n, dtype=float)
    low = np.zeros(n, dtype=float)
    open_ = np.zeros(n, dtype=float)
    vol = np.full(n, 90.0, dtype=float)

    # 02:00 起橫盤，讓 8 條均線黏在一起
    close[0] = 29200.0
    for i in range(1, 250):
        close[i] = 29200.0 + (3.0 if i % 2 == 0 else -3.0)
    # 緩跌到 07:10 附近
    for i in range(250, 310):
        t = (i - 250) / 60.0
        close[i] = 29200.0 - t * 54.0 + (2.0 if i % 2 == 0 else -2.0)
    # 07:10 觸底 29145.75 後拉回
    close[310] = 29145.75
    close[311] = 29162.0
    close[312] = 29178.0
    close[313] = 29188.0
    close[314] = 29192.0
    # 07:15–07:30 盤整 29180–29210
    coil = [
        29188.0,
        29195.0,
        29186.0,
        29202.0,
        29191.0,
        29206.0,
        29184.0,
        29199.0,
        29190.0,
        29204.0,
        29187.0,
        29198.0,
        29193.0,
        29201.0,
        29196.0,
    ]
    for k, px in enumerate(coil):
        close[315 + k] = px
    # 07:35 連續長綠（起漲）
    close[330] = 29218.0
    close[331] = 29238.0
    close[332] = 29255.0
    close[333] = 29272.0
    close[334] = 29283.0
    for i in range(335, n):
        close[i] = 29283.0 - (i - 334) * 1.6

    for i in range(n):
        if i == 0:
            open_[i] = close[i] - 1.0
        else:
            open_[i] = close[i - 1]
        if 315 <= i < 330:
            high[i] = min(29210.0, max(close[i], open_[i]) + 2.5)
            low[i] = max(29180.0, min(close[i], open_[i]) - 2.5)
            vol[i] = 70.0
        elif i == 310:
            high[i] = max(close[i], open_[i]) + 2.0
            low[i] = 29145.75
            vol[i] = 160.0
        elif 330 <= i <= 334:
            high[i] = max(close[i], open_[i]) + 3.0
            low[i] = min(close[i], open_[i]) - 1.5
            vol[i] = 280.0 if i == 330 else 220.0
        else:
            high[i] = max(close[i], open_[i]) + 3.0
            low[i] = min(close[i], open_[i]) - 3.0
        if high[i] < close[i]:
            high[i] = close[i]
        if low[i] > close[i]:
            low[i] = close[i]
        if high[i] < open_[i]:
            high[i] = open_[i]
        if low[i] > open_[i]:
            low[i] = open_[i]

    idx = pd.date_range("2026-08-24 02:00", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )
