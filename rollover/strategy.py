"""SOXS 式 15 分 K 起跌放空。

進場不是極點那一根，而是這三件事同時出現的那根 15 分收盤：

1. 極點之後出現較低高點（高點不再創新高）。
2. 極點與較低高點之間的回檔低點（頸線）被收盤跌破。
3. 同一根收盤在 MA25 之下，而且量能回到前一段均量以上。

停損放在較低高點，不放在極點。影線刺破頸線不算。
量能不夠的那根不進場，只允許之後一次「拉回 MA7–MA25 再收不回去」。
收盤已經離開頸線太遠，視為跌勢中段，不再追。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
BAR_MINUTES = 15


@dataclass(frozen=True)
class RolloverParams:
    """15 分圖上的起跌參數。數字對得上幣安圖上的 MA7、MA25、量 MA10。"""

    swing_lookback: int = 2
    ma_fast: int = 7
    ma_slow: int = 25
    vol_ma_len: int = 10
    min_pullback_pct: float = 0.008
    min_lower_high_gap_pct: float = 0.001
    min_bounce_pct: float = 0.003
    max_bars_to_lower_high: int = 24
    max_bars_after_lower_high: int = 10
    max_entry_extension_pct: float = 0.008
    retest_bars: int = 6
    stop_buffer_pct: float = 0.0
    reward_r: float = 1.5
    max_hold_bars: int = 16


@dataclass(frozen=True)
class RolloverSignal:
    """一根 15 分收盤上的放空訊號。"""

    timestamp: pd.Timestamp
    bar_idx: int
    kind: str  # "break" 或 "retest"
    entry: float
    stop: float
    target: float
    extreme: float
    extreme_idx: int
    neckline: float
    lower_high: float
    lower_high_idx: int
    off_session: bool

    @property
    def risk_per_unit(self) -> float:
        return self.stop - self.entry


@dataclass(frozen=True)
class ShortTrade:
    signal: RolloverSignal
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_per_unit: float


def is_us_cash_session(ts: pd.Timestamp) -> bool:
    """SOXS 現貨時段：美東週一到週五 09:30–16:00。無時區的時間戳視為 UTC。"""

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    local = stamp.tz_convert(NY)
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def qty_for_risk(entry: float, stop: float, risk_usdt: float) -> int:
    """張數 = 這筆可虧的 USDT / (停損價 - 進場價)。

    假設 1 張的損益就是 1 個報價單位。合約面額不是 1 時，再除掉面額。
    停損不在進場上方、或預算不夠一張，回 0。
    """

    distance = stop - entry
    if distance <= 0 or risk_usdt <= 0:
        return 0
    return int(math.floor(risk_usdt / distance + 1e-9))


def detect_rollover_shorts(
    df: pd.DataFrame,
    params: RolloverParams | None = None,
) -> list[RolloverSignal]:
    """在 15 分 OHLCV 上找出起跌放空。同一套極點只出一次訊號。"""

    params = params or RolloverParams()
    frame = _normalize(df)
    _require_15m(frame.index)

    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    volume = frame["volume"].to_numpy(float)
    ma_fast = _sma(close, params.ma_fast)
    ma_slow = _sma(close, params.ma_slow)
    vol_ma = _sma_prev(volume, params.vol_ma_len)

    signals: list[RolloverSignal] = []
    extreme_i: int | None = None
    extreme_px: float | None = None
    lower_i: int | None = None
    lower_px: float | None = None
    neckline: float | None = None
    phase = "seek"  # seek | watch | retest | cool
    retest_left = 0
    bounce_high = 0.0
    cool_until = -1

    n = len(frame)
    left = params.swing_lookback
    for t in range(n):
        pivot_i = t - left
        if pivot_i >= left and _is_pivot_high(high, pivot_i, left):
            px = high[pivot_i]
            # 上一筆結束後，訊號之後的第一個擺動高點當作新極點，
            # 即使它沒有突破舊極點。舊極點沒被突破之前，不另開第二筆。
            fresh_after_trade = phase == "cool" and pivot_i > cool_until
            if fresh_after_trade or extreme_px is None or px >= extreme_px:
                extreme_i = pivot_i
                extreme_px = px
                lower_i = None
                lower_px = None
                neckline = None
                phase = "seek"
            elif (
                phase in ("seek", "watch")
                and extreme_i is not None
                and extreme_i + 1 < pivot_i <= extreme_i + params.max_bars_to_lower_high
                and px < extreme_px * (1 - params.min_lower_high_gap_pct)
                and (lower_px is None or px > lower_px)
            ):
                segment = low[extreme_i + 1 : pivot_i + 1]
                if len(segment) > 0:
                    neck = float(np.min(segment))
                    neck_at = extreme_i + 1 + int(np.argmin(segment))
                    deep = (extreme_px - neck) / extreme_px >= params.min_pullback_pct
                    bounced = px > neck * (1 + params.min_bounce_pct) and neck_at < pivot_i
                    if deep and bounced:
                        lower_i = pivot_i
                        lower_px = px
                        neckline = neck
                        phase = "watch"

        if phase == "watch":
            assert extreme_i is not None and extreme_px is not None
            assert lower_i is not None and lower_px is not None and neckline is not None
            if t - lower_i > params.max_bars_after_lower_high or high[t] >= extreme_px or close[t] > extreme_px:
                phase = "seek"
                extreme_i = None
                extreme_px = None
                lower_i = None
                lower_px = None
                neckline = None
                continue
            if close[t] < neckline:
                extension = (neckline - close[t]) / neckline
                if extension > params.max_entry_extension_pct:
                    phase = "seek"
                    extreme_i = None
                    extreme_px = None
                    lower_i = None
                    lower_px = None
                    neckline = None
                    continue
                ma_ok = not math.isnan(ma_slow[t]) and close[t] < ma_slow[t]
                vol_ok = not math.isnan(vol_ma[t]) and volume[t] >= vol_ma[t]
                if ma_ok and vol_ok:
                    signals.append(
                        _signal(
                            frame,
                            params,
                            t,
                            "break",
                            close[t],
                            lower_px,
                            extreme_px,
                            extreme_i,
                            neckline,
                            lower_i,
                        )
                    )
                    phase = "cool"
                    cool_until = t
                    continue
                if ma_ok and not vol_ok:
                    phase = "retest"
                    retest_left = params.retest_bars
                    bounce_high = high[t]
                    continue

        if phase == "retest":
            assert extreme_i is not None and extreme_px is not None
            assert lower_i is not None and lower_px is not None and neckline is not None
            bounce_high = max(bounce_high, high[t])
            retest_left -= 1
            failed = (
                retest_left < 0
                or high[t] >= extreme_px
                or close[t] > extreme_px
                or bounce_high >= lower_px
            )
            if failed:
                phase = "seek"
                extreme_i = None
                extreme_px = None
                lower_i = None
                lower_px = None
                neckline = None
                continue
            fast = ma_fast[t]
            slow = ma_slow[t]
            if math.isnan(fast) or math.isnan(slow):
                continue
            band_lo = min(fast, slow)
            band_hi = max(fast, slow)
            tagged = high[t] >= band_lo and low[t] <= band_hi
            back_under = close[t] < band_lo and close[t] < neckline
            extend = (band_lo - close[t]) / band_lo if band_lo else 1.0
            if tagged and back_under and 0 <= extend <= params.max_entry_extension_pct:
                signals.append(
                    _signal(
                        frame,
                        params,
                        t,
                        "retest",
                        close[t],
                        bounce_high,
                        extreme_px,
                        extreme_i,
                        neckline,
                        lower_i,
                    )
                )
                phase = "cool"
                cool_until = t

    return signals


def simulate_shorts(
    df: pd.DataFrame,
    signals: list[RolloverSignal],
    params: RolloverParams | None = None,
) -> list[ShortTrade]:
    """放空回測。同一根同時碰到停損和停利，算停損。持倉中不再進新單。"""

    params = params or RolloverParams()
    frame = _normalize(df)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    trades: list[ShortTrade] = []
    busy_until = -1

    for sig in signals:
        if sig.bar_idx <= busy_until:
            continue
        last = min(sig.bar_idx + params.max_hold_bars, len(frame) - 1)
        exit_idx = last
        exit_price = float(close[last])
        reason = "time"
        for i in range(sig.bar_idx + 1, last + 1):
            if high[i] >= sig.stop:
                exit_idx = i
                exit_price = sig.stop
                reason = "stop"
                break
            if low[i] <= sig.target:
                exit_idx = i
                exit_price = sig.target
                reason = "target"
                break
        busy_until = exit_idx
        trades.append(
            ShortTrade(
                signal=sig,
                exit_idx=exit_idx,
                exit_time=frame.index[exit_idx],
                exit_price=exit_price,
                exit_reason=reason,
                pnl_per_unit=sig.entry - exit_price,
            )
        )
    return trades


def _signal(
    frame: pd.DataFrame,
    params: RolloverParams,
    bar_idx: int,
    kind: str,
    entry: float,
    stop_anchor: float,
    extreme: float,
    extreme_idx: int,
    neckline: float,
    lower_high_idx: int,
) -> RolloverSignal:
    stop = stop_anchor * (1 + params.stop_buffer_pct)
    risk = stop - entry
    target = entry - params.reward_r * risk
    ts = frame.index[bar_idx]
    return RolloverSignal(
        timestamp=ts,
        bar_idx=bar_idx,
        kind=kind,
        entry=entry,
        stop=stop,
        target=target,
        extreme=extreme,
        extreme_idx=extreme_idx,
        neckline=neckline,
        lower_high=float(frame["high"].iloc[lower_high_idx]),
        lower_high_idx=lower_high_idx,
        off_session=not is_us_cash_session(ts),
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).lower() for col in df.columns}
    frame = df.rename(columns=renamed)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {sorted(missing)}")
    out = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(frame.index)
    return out


def _require_15m(index: pd.DatetimeIndex) -> None:
    if len(index) < 3:
        return
    delta = pd.Series(index).diff().dropna().median()
    if abs(delta - pd.Timedelta(minutes=BAR_MINUTES)) > pd.Timedelta(seconds=30):
        raise ValueError(
            f"起跌策略只吃 15 分 K，這份資料的間隔是 {delta}。"
            "影線在非交易時段太多，改用 15 分收盤確認。"
        )


def _sma(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).rolling(length, min_periods=length).mean().to_numpy(float)


def _sma_prev(values: np.ndarray, length: int) -> np.ndarray:
    """前 length 根的均值，不含當根，避免放量進場把均量自己抬高。"""

    return pd.Series(values).rolling(length, min_periods=length).mean().shift(1).to_numpy(float)


def _is_pivot_high(highs: np.ndarray, idx: int, lookback: int) -> bool:
    if idx < lookback or idx + lookback >= len(highs):
        return False
    pivot = highs[idx]
    for j in range(idx - lookback, idx + lookback + 1):
        if j != idx and highs[j] >= pivot:
            return False
    return True
