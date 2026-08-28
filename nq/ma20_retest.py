"""NQ 破底翻：反彈收復 MA20 後，右肩回踩 MA20 做多（5m / 1m）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Signal:
    break_idx: int
    trough_idx: int
    reclaim_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    support: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    ma60_5m: float = 0.0
    ma60_5m_slope: float = 0.0
    ma20_5m: float = 0.0
    ma20_5m_slope: float = 0.0
    ma30_5m: float = 0.0
    ma30_5m_slope: float = 0.0
    quality: str = "C"
    quality_score: int = 0


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_points: float
    exit_reason: str
    quality: str


def sma(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def rolling_min_prev(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.shift(1).rolling(n, min_periods=n).min().to_numpy(float)


def align_5m_ma(
    df: pd.DataFrame,
    *,
    ma_len: int = 60,
    slope_bars: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """5 分均線與斜率，對齊 df.index，不用未來 1m。

    一分圖用已收盤的 5 分 K（與 TradingView lookahead_off 相同）：
    10:51–10:54 仍用 10:50 收完的那根 5 分均線。
    """
    close = df["Close"].astype(float)
    idx = df.index
    n = len(df)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bar_min = 5.0
    if n >= 2:
        bar_min = max((idx[1] - idx[0]).total_seconds() / 60.0, 0.5)

    if bar_min >= 4.0:
        ma = close.rolling(ma_len, min_periods=ma_len).mean()
        slope = ma - ma.shift(slope_bars)
        return ma.to_numpy(float), slope.to_numpy(float)

    m5 = close.resample("5min", label="right", closed="right").last().dropna()
    ma5 = m5.rolling(ma_len, min_periods=ma_len).mean()
    slope5 = ma5 - ma5.shift(slope_bars)
    ma_out = np.full(n, np.nan, dtype=float)
    slope_out = np.full(n, np.nan, dtype=float)
    m5_idx = ma5.index
    ma_vals = ma5.to_numpy(float)
    sl_vals = slope5.to_numpy(float)
    j = 0
    for i, ts in enumerate(idx):
        while j + 1 < len(m5_idx) and m5_idx[j + 1] <= ts:
            j += 1
        if j < len(m5_idx) and m5_idx[j] <= ts:
            ma_out[i] = ma_vals[j]
            slope_out[i] = sl_vals[j]
    return ma_out, slope_out


def align_5m_ma60(
    df: pd.DataFrame,
    *,
    ma_len: int = 60,
    slope_bars: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    return align_5m_ma(df, ma_len=ma_len, slope_bars=slope_bars)


def near_falling_5m_ma60(
    entry: float,
    ma60_5m: float,
    slope: float,
    near: float,
) -> bool:
    """進場價貼著下彎的 5m MA60（綠線）→ 濾掉。"""
    if near <= 0 or np.isnan(ma60_5m) or np.isnan(slope):
        return False
    return float(slope) < 0.0 and abs(float(entry) - float(ma60_5m)) <= float(near)


def near_falling_5m_ma20_ma30(
    entry: float,
    ma20_5m: float,
    slope20: float,
    ma30_5m: float,
    slope30: float,
    near: float,
    ma60_5m: float = float("nan"),
    slope60: float = float("nan"),
) -> bool:
    """進場夾在下彎 5m MA20 / MA30 蓋頭底下（空頭排列）→ 濾掉。

    要粉紅 < 藍 < 綠（MA20 < MA30 < MA60）且三條都下彎。
    08-18 10:34 是蓋頭；08-24 07:27 均線纏在一起、綠線還在藍線下面，會留。
    """
    if near <= 0:
        return False
    vals = (ma20_5m, slope20, ma30_5m, slope30, ma60_5m, slope60)
    if any(np.isnan(x) for x in vals):
        return False
    if float(slope20) >= 0.0 or float(slope30) >= 0.0 or float(slope60) >= 0.0:
        return False
    if not (float(entry) < float(ma20_5m) and float(entry) < float(ma30_5m)):
        return False
    if not (float(ma20_5m) < float(ma30_5m) < float(ma60_5m)):
        return False
    return (float(ma20_5m) - float(entry)) <= float(near) and (
        float(ma30_5m) - float(entry)
    ) <= float(near)


def quality_at_entry(ma5: float, ma10: float, ma20: float, ma20_slope: float) -> Tuple[int, str]:
    """A：MA20 上彎且短均多頭；B：收在 MA20 之上且 MA5>MA20；C：其餘。"""
    score = 0
    if ma20_slope > 0:
        score += 1
    if ma5 > ma10:
        score += 1
    if ma5 > ma20:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def summarize_trades(trades: Sequence) -> dict:
    pnls = [float(getattr(t, "pnl_points", 0.0)) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_q: Dict[str, List[float]] = {}
    for t in trades:
        by_q.setdefault(getattr(t, "quality", "?"), []).append(float(getattr(t, "pnl_points", 0.0)))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "pnl": float(sum(pnls)),
        "n": n,
        "by_quality": {
            q: {
                "n": len(v),
                "wins": sum(1 for p in v if p > 0),
                "pnl": float(sum(v)),
            }
            for q, v in sorted(by_q.items())
        },
    }


def _in_session(ts, session: str) -> bool:
    if session == "all":
        return True
    h, m = ts.hour, ts.minute
    minutes = h * 60 + m
    if session == "rth":
        return (9 * 60 + 30) <= minutes <= (15 * 60 + 45)
    if session == "day":
        return (8 * 60) <= minutes <= (16 * 60)
    return True


# 同一套破翻回踩；只把「時間」換成根數。點數門檻（深度／刺穿／風險）不變。
# 5m MA20 ≈ 100 分鐘；1m MA20 ≈ 20 分鐘，均線更貼、訊號會比較密。
INTERVAL_DETECT = {
    "5m": dict(
        lookback=24,
        min_break_depth=25.0,
        reclaim_window=24,
        retest_window=18,
        leave_bars=3,
        leave_buffer=10.0,
        touch_above=8.0,
        max_pierce=12.0,
        fail_below=8.0,
        stop_buffer=10.0,
        target_r=1.5,
        max_risk=180.0,
        min_risk=20.0,
        min_entry_gap=12,
        ma20_slope_bars=4,
        ma60_5m_near=40.0,
        ma60_5m_slope_bars=6,
        ma20_5m_near=0.0,  # 五分圖進場本就貼 MA20；蓋頭濾只給一分用
        stop_at_shoulder=False,
    ),
    "1m": dict(
        lookback=120,
        min_break_depth=10.0,
        reclaim_window=120,
        retest_window=90,
        leave_bars=8,
        leave_buffer=6.0,
        touch_above=8.0,
        max_pierce=20.0,
        fail_below=40.0,
        stop_buffer=10.0,
        target_r=1.5,
        max_risk=100.0,
        min_risk=20.0,
        min_entry_gap=60,
        ma20_slope_bars=20,
        ma60_5m_near=40.0,
        ma60_5m_slope_bars=6,
        ma20_5m_near=45.0,
        min_pullback=25.0,
        max_dump_body=30.0,
        max_prev_above=45.0,
        min_retest_bars=30,
        stop_at_shoulder=True,
    ),
}

INTERVAL_SIMULATE = {
    "5m": dict(max_hold=36, ma_exit_after=12),
    "1m": dict(max_hold=180, ma_exit_after=60),
}


def detect_kwargs(interval: str, **overrides) -> dict:
    if interval not in INTERVAL_DETECT:
        raise ValueError(f"unsupported interval {interval!r}, expected {tuple(INTERVAL_DETECT)}")
    kw = dict(INTERVAL_DETECT[interval])
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw


def simulate_kwargs(interval: str, **overrides) -> dict:
    if interval not in INTERVAL_SIMULATE:
        raise ValueError(f"unsupported interval {interval!r}, expected {tuple(INTERVAL_SIMULATE)}")
    kw = dict(INTERVAL_SIMULATE[interval])
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw


def detect_signals(
    df: pd.DataFrame,
    *,
    lookback: int = 24,
    min_break_depth: float = 25.0,
    reclaim_window: int = 24,
    retest_window: int = 18,
    leave_bars: int = 3,
    leave_buffer: float = 10.0,
    touch_above: float = 8.0,
    max_pierce: float = 12.0,
    fail_below: float = 8.0,
    stop_buffer: float = 10.0,
    target_r: float = 1.5,
    max_risk: float = 180.0,
    min_risk: float = 20.0,
    min_entry_gap: int = 12,
    session: str = "rth",
    ma20_len: int = 20,
    ma20_slope_bars: int = 4,
    ma60_5m_near: float = 40.0,
    ma60_5m_slope_bars: int = 6,
    ma20_5m_near: float = 45.0,
    min_pullback: float = 0.0,
    max_dump_body: float = 0.0,
    max_prev_above: float = 0.0,
    min_retest_bars: int = 0,
    stop_at_shoulder: bool = False,
    funnel: Optional[Dict[str, int]] = None,
    last_entry_idx: int = -(10**9),
) -> List[Signal]:
    """
    破底 → 反彈收復 MA20 → 先離開均線 → 回踩 MA20 進場。

    對齊 2026-08-24 藍圈：09:50 低點 28946.75，10:35 收復，11:00–11:20 回踩。
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    if "Open" in df.columns:
        open_ = df["Open"].to_numpy(float)
    else:
        open_ = np.r_[close[0], close[:-1]]

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, ma20_len)
    ma60 = sma(close, 60)
    ma20_5m, ma20_5m_slope = align_5m_ma(df, ma_len=20, slope_bars=ma60_5m_slope_bars)
    ma30_5m, ma30_5m_slope = align_5m_ma(df, ma_len=30, slope_bars=ma60_5m_slope_bars)
    ma60_5m, ma60_5m_slope = align_5m_ma(df, ma_len=60, slope_bars=ma60_5m_slope_bars)
    floor = rolling_min_prev(low, lookback)

    n = len(close)
    warmup = max(lookback, 60, ma20_len)
    signals: List[Signal] = []
    last_entry = int(last_entry_idx)
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        if np.isnan(floor[i]) or np.isnan(ma20[i]):
            i += 1
            continue
        if low[i] >= float(floor[i]):
            i += 1
            continue
        # 日盤策略：夜盤破底不當 破底翻（08-12 08:30 / 08-20 08:36 那種，
        # 09:30 一開盤就當成右肩，其實只是隔夜彈完）。
        if not _in_session(df.index[i], session):
            bump("skip_session")
            i += 1
            continue

        support = float(floor[i])
        break_low = float(low[i])
        depth = support - break_low
        bump("break")
        if depth < min_break_depth:
            bump("shallow")
            i += 1
            continue
        bump("deep_break")

        break_idx = i
        trough_idx = i
        end_scan = min(n - 1, break_idx + reclaim_window)
        reclaim_idx: Optional[int] = None
        k = break_idx + 1
        while k <= end_scan:
            if low[k] < break_low:
                break_low = float(low[k])
                trough_idx = k
                end_scan = min(n - 1, k + reclaim_window)
            if (
                not np.isnan(ma20[k])
                and close[k] > float(ma20[k])
                and break_low < float(ma20[k])
            ):
                reclaim_idx = k
                bump("reclaim")
                break
            k += 1

        if reclaim_idx is None:
            bump("no_reclaim")
            i = trough_idx + 1
            continue

        retest_end = min(n - 1, reclaim_idx + retest_window)
        left_run = 0
        left_ok = False
        peak = float(high[reclaim_idx])
        peak_idx = int(reclaim_idx)
        entry_idx: Optional[int] = None
        dead = False
        for t in range(reclaim_idx + 1, retest_end + 1):
            if np.isnan(ma20[t]):
                continue
            m20 = float(ma20[t])
            if low[t] < break_low:
                bump("new_low")
                dead = True
                break
            if close[t] < m20 - fail_below:
                bump("fail_hold")
                dead = True
                break

            if float(high[t]) >= peak:
                peak = float(high[t])
                peak_idx = t

            if float(low[t]) > m20 + leave_buffer:
                left_run += 1
                if left_run >= leave_bars:
                    left_ok = True
            else:
                left_run = 0

            if not left_ok:
                continue
            # 右肩：現價相對這波反彈高點至少拉回 min_pullback（08-28 圖 11:01 高點
            # 29811 → 11:22 踩 MA20；不能用稍早小回檔把 pulled 黏死，否則 11:06 高點就進）。
            if min_pullback > 0 and peak - float(close[t]) < min_pullback:
                continue

            pierce = m20 - float(low[t])
            if pierce < -touch_above or pierce > max_pierce:
                continue
            if close[t] < m20:
                continue

            # 右肩是坐上 MA20，不是大陰線從天上砸下來碰到均線
            # （07-31 09:38：開 28660 → 收 28604，H-MA +63；08-28 11:23 是小陽線回踩）。
            dump_body = float(open_[t]) - float(close[t])
            prev_above = float(close[t - 1]) - m20 if t > 0 else 0.0
            # 大陰線砸上 MA20 = 這肩失敗（07-31 09:38、08-28 09:49）。
            # 不能只 skip 再買下一根；但 V 後第一個小回踩（08-28 10:27 實體 29 點）
            # 不能整波作廢，否則 11:23 那肩也沒了。
            if max_dump_body > 0 and dump_body >= max_dump_body:
                bump("skip_dump")
                dead = True
                break
            if max_prev_above > 0 and prev_above >= max_prev_above:
                bump("skip_dump")
                continue
            if min_retest_bars > 0 and t - reclaim_idx < min_retest_bars:
                continue

            # 右肩候選：過濾不通過就繼續掃同一波，不要整段放棄
            # （08-28 10:13 破底 29505，09:49 那筆的 60 根間隔擋掉 10:27 首踩，
            #  11:23 才是圖上那肩；舊邏輯 skip_gap 後從 10:28 重找破底，右肩就沒了。）
            ts = df.index[t]
            if not _in_session(ts, session):
                bump("skip_session")
                continue
            if t - last_entry < min_entry_gap:
                bump("skip_gap")
                continue

            entry = float(close[t])
            if stop_at_shoulder:
                shoulder_low = float(np.min(low[peak_idx : t + 1]))
                stop = shoulder_low - stop_buffer
            else:
                stop = break_low - stop_buffer
            risk = entry - stop
            if risk < min_risk:
                bump("skip_tiny_risk")
                continue
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            m60_5 = float(ma60_5m[t]) if not np.isnan(ma60_5m[t]) else float("nan")
            m60_5_s = (
                float(ma60_5m_slope[t]) if not np.isnan(ma60_5m_slope[t]) else float("nan")
            )
            if near_falling_5m_ma60(entry, m60_5, m60_5_s, ma60_5m_near):
                bump("skip_ma60")
                continue

            m20_5 = float(ma20_5m[t]) if not np.isnan(ma20_5m[t]) else float("nan")
            m20_5_s = (
                float(ma20_5m_slope[t]) if not np.isnan(ma20_5m_slope[t]) else float("nan")
            )
            m30_5 = float(ma30_5m[t]) if not np.isnan(ma30_5m[t]) else float("nan")
            m30_5_s = (
                float(ma30_5m_slope[t]) if not np.isnan(ma30_5m_slope[t]) else float("nan")
            )
            if near_falling_5m_ma20_ma30(
                entry, m20_5, m20_5_s, m30_5, m30_5_s, ma20_5m_near, m60_5, m60_5_s
            ):
                bump("skip_ma20_30")
                continue

            entry_idx = t
            bump("retest")
            break

        if dead or entry_idx is None:
            if entry_idx is None and not dead:
                bump("no_retest")
            i = (reclaim_idx if reclaim_idx is not None else trough_idx) + 1
            continue

        entry = float(close[entry_idx])
        if stop_at_shoulder:
            shoulder_low = float(np.min(low[peak_idx : entry_idx + 1]))
            stop = shoulder_low - stop_buffer
        else:
            stop = break_low - stop_buffer
        risk = entry - stop
        m60_5 = float(ma60_5m[entry_idx]) if not np.isnan(ma60_5m[entry_idx]) else float("nan")
        m60_5_s = (
            float(ma60_5m_slope[entry_idx])
            if not np.isnan(ma60_5m_slope[entry_idx])
            else float("nan")
        )
        m20_5 = float(ma20_5m[entry_idx]) if not np.isnan(ma20_5m[entry_idx]) else float("nan")
        m20_5_s = (
            float(ma20_5m_slope[entry_idx])
            if not np.isnan(ma20_5m_slope[entry_idx])
            else float("nan")
        )
        m30_5 = float(ma30_5m[entry_idx]) if not np.isnan(ma30_5m[entry_idx]) else float("nan")
        m30_5_s = (
            float(ma30_5m_slope[entry_idx])
            if not np.isnan(ma30_5m_slope[entry_idx])
            else float("nan")
        )
        slope = 0.0
        if entry_idx >= ma20_slope_bars and not np.isnan(ma20[entry_idx - ma20_slope_bars]):
            slope = float(ma20[entry_idx] - ma20[entry_idx - ma20_slope_bars])
        q_score, q_grade = quality_at_entry(
            float(ma5[entry_idx]),
            float(ma10[entry_idx]),
            float(ma20[entry_idx]),
            slope,
        )
        bump("taken")
        signals.append(
            Signal(
                break_idx=break_idx,
                trough_idx=trough_idx,
                reclaim_idx=reclaim_idx,
                entry_idx=entry_idx,
                entry_price=entry,
                stop_price=stop,
                target_price=entry + risk * target_r,
                break_low=break_low,
                support=support,
                ma5=float(ma5[entry_idx]),
                ma10=float(ma10[entry_idx]),
                ma20=float(ma20[entry_idx]),
                ma60=float(ma60[entry_idx]) if not np.isnan(ma60[entry_idx]) else 0.0,
                ma60_5m=m60_5 if not np.isnan(m60_5) else 0.0,
                ma60_5m_slope=m60_5_s if not np.isnan(m60_5_s) else 0.0,
                ma20_5m=m20_5 if not np.isnan(m20_5) else 0.0,
                ma20_5m_slope=m20_5_s if not np.isnan(m20_5_s) else 0.0,
                ma30_5m=m30_5 if not np.isnan(m30_5) else 0.0,
                ma30_5m_slope=m30_5_s if not np.isnan(m30_5_s) else 0.0,
                quality=q_grade,
                quality_score=q_score,
            )
        )
        last_entry = entry_idx
        i = entry_idx + 1

    return signals


def simulate(
    df: pd.DataFrame,
    signals: List[Signal],
    *,
    max_hold: int = 36,
    ma_exit_after: int = 12,
    be_after_r: float = 0.8,
    trail_after_r: float = 1.2,
    trail_lock_r: float = 0.4,
    ma_exit_period: int = 20,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma_exit = sma(close, ma_exit_period)
    results: List[TradeResult] = []
    busy_until = -1

    for sig in signals:
        entry_idx = sig.entry_idx
        if entry_idx <= busy_until:
            continue
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = entry - stop
        if risk <= 0:
            continue
        cur_stop = stop
        mfe = 0.0
        limit = min(entry_idx + max_hold, len(df) - 1)
        exit_idx = limit
        exit_price = float(close[exit_idx])
        exit_reason = "timeout"

        for k in range(entry_idx + 1, limit + 1):
            mfe = max(mfe, float(high[k] - entry))
            if be_after_r > 0 and mfe / risk >= be_after_r:
                cur_stop = max(cur_stop, entry)
            if trail_after_r > 0 and mfe / risk >= trail_after_r:
                cur_stop = max(cur_stop, entry + trail_lock_r * risk)

            if low[k] <= cur_stop:
                reason = "be_stop" if cur_stop > stop + 1e-9 else "stop"
                exit_idx, exit_price, exit_reason = k, float(cur_stop), reason
                break
            if high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break
            held = k - entry_idx
            if (
                held >= ma_exit_after
                and not np.isnan(ma_exit[k])
                and float(close[k]) < float(ma_exit[k])
            ):
                exit_idx, exit_price, exit_reason = k, float(close[k]), "ma20"
                break

        busy_until = exit_idx
        pnl = exit_price - entry
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(pnl),
                exit_reason=exit_reason,
                quality=sig.quality,
            )
        )
    return results


def drop_open_end_trades(
    df: pd.DataFrame,
    trades: Sequence[TradeResult],
    max_hold: int,
) -> Tuple[List[TradeResult], List[TradeResult]]:
    """樣本最後一根若還在持倉，不算進回測成績。"""
    if not trades or len(df) == 0:
        return list(trades), []
    last = len(df) - 1
    kept: List[TradeResult] = []
    open_trades: List[TradeResult] = []
    for t in trades:
        held = t.exit_idx - t.entry_idx
        if t.exit_idx >= last and t.exit_reason == "timeout" and held < max_hold:
            open_trades.append(t)
        else:
            kept.append(t)
    return kept, open_trades
