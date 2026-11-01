"""SOXS 起跌放空：5 分定結構，1 分收盤進場。

5 分圖負責極點、較低高點、頸線、停損。
1 分圖負責進場：較低高點確認之後，第一根同時跌破頸線、收在 1 分 MA25 下、
量能回到前 10 根均量，而且收盤還在四小時高點附近。

停損放在 5 分較低高點，不放在極點。影線刺破頸線不算。
量能不夠的那根不進場，只允許之後一次 1 分拉回 MA7–MA25 再收不回去。
收盤已經離開頸線太遠，視為跌勢中段，不再追。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class RolloverParams:
    """5 分結構、1 分進場。MA7、MA25、量 MA10 跟幣安圖上的一樣，算在 1 分進場那根。"""

    swing_lookback: int = 2
    ma_fast: int = 7
    ma_slow: int = 25
    vol_ma_len: int = 10
    min_pullback_pct: float = 0.008
    min_lower_high_gap_pct: float = 0.001
    min_bounce_pct: float = 0.003
    max_bars_to_lower_high: int = 72  # 5 分根數，約 6 小時
    max_bars_after_lower_high: int = 30  # 5 分根數，約 2.5 小時
    max_entry_extension_pct: float = 0.008
    four_hour_bars: int = 240  # 1 分根數，4 小時
    max_below_4h_high_pct: float = 0.02  # 觸發收盤最多比四小時高點低 2%
    retest_bars: int = 15  # 1 分根數
    stop_buffer_pct: float = 0.0
    reward_r: float = 1.5
    max_hold_bars: int = 120  # 1 分根數，約 2 小時


@dataclass(frozen=True)
class RolloverSignal:
    """一根 1 分收盤上的放空訊號。停損價來自 5 分較低高點。"""

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
    four_hour_high: float = 0.0

    @property
    def risk_per_unit(self) -> float:
        return self.stop - self.entry


@dataclass(frozen=True)
class RolloverView:
    """最後一根已收盤 1 分的判斷，以及這段資料裡的全部進場。"""

    signals: tuple[RolloverSignal, ...]
    timestamp: pd.Timestamp
    close: float
    phase: str
    note: str
    extreme: float | None
    lower_high: float | None
    neckline: float | None
    ma_slow: float | None
    volume: float
    vol_ma: float | None
    extension: float | None
    four_hour_high: float | None
    fresh: RolloverSignal | None


@dataclass(frozen=True)
class ShortTrade:
    signal: RolloverSignal
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_per_unit: float


@dataclass(frozen=True)
class _Armed:
    extreme: float
    extreme_idx: int
    neckline: float
    lower_high: float
    lower_high_idx: int
    deadline: pd.Timestamp


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
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    params: RolloverParams | None = None,
) -> list[RolloverSignal]:
    """5 分找出較低高點和頸線，1 分收盤進場。同一套較低高點只出一次訊號。"""

    return list(inspect_rollover(df_5m, df_1m, params).signals)


def inspect_rollover(
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    params: RolloverParams | None = None,
) -> RolloverView:
    """跟 detect 同一套判斷，多帶最後一根 1 分為什麼進、或為什麼還不進。"""

    params = params or RolloverParams()
    frame5 = _normalize(df_5m)
    frame1 = _normalize(df_1m)
    _require_step(frame5.index, 5, "5 分 K")
    _require_step(frame1.index, 1, "1 分 K")
    if len(frame1) == 0:
        raise ValueError("1 分 K 是空的")

    steps = _arm_5m(frame5, params)
    high = frame1["high"].to_numpy(float)
    low = frame1["low"].to_numpy(float)
    close = frame1["close"].to_numpy(float)
    volume = frame1["volume"].to_numpy(float)
    ma_fast = _sma(close, params.ma_fast)
    ma_slow = _sma(close, params.ma_slow)
    vol_ma = _sma_prev(volume, params.vol_ma_len)
    high_4h = pd.Series(high).rolling(params.four_hour_bars, min_periods=params.four_hour_bars).max().to_numpy(float)

    signals: list[RolloverSignal] = []
    step_i = -1
    phase = "idle"
    note = "5 分還沒走出較低高點"
    setup: _Armed | None = None
    filled_lower = -1
    killed: set[int] = set()
    killed_reason: dict[int, str] = {}
    retest_left = 0
    bounce_high = 0.0
    last_levels: _Armed | None = None

    def kill(idx: int, reason: str) -> None:
        killed.add(idx)
        killed_reason[idx] = reason

    for j, ts in enumerate(frame1.index):
        while step_i + 1 < len(steps) and steps[step_i + 1][0] <= ts:
            step_i += 1
        armed = steps[step_i][1] if step_i >= 0 else None
        last_levels = armed

        if phase == "cool":
            if armed is None or armed.lower_high_idx != filled_lower:
                phase = "idle"
                setup = None
                note = "5 分還沒走出較低高點" if armed is None else "前一組較低高點已經進過場"
            else:
                note = "這組較低高點已經進過場"
                continue

        if armed is None or armed.lower_high_idx in killed or ts >= armed.deadline:
            if armed is not None and ts >= armed.deadline:
                killed.add(armed.lower_high_idx)
                killed_reason.setdefault(armed.lower_high_idx, "較低高點之後等太久，這組作廢")
            if armed is None:
                note = "5 分還沒走出較低高點"
            else:
                note = killed_reason.get(armed.lower_high_idx, "這組較低高點已經作廢")
            phase = "idle"
            setup = None
            continue

        if setup is None or setup.lower_high_idx != armed.lower_high_idx:
            phase = "watch"
            setup = armed
            note = "5 分已有較低高點，等 1 分收盤跌破頸線"

        if high[j] >= setup.extreme or close[j] > setup.extreme:
            kill(setup.lower_high_idx, "1 分碰到極點，這組結構作廢")
            note = "1 分碰到極點，這組結構作廢"
            phase = "idle"
            setup = None
            continue

        if phase == "watch" and close[j] < setup.neckline:
            extension = (setup.neckline - close[j]) / setup.neckline
            if extension > params.max_entry_extension_pct:
                kill(setup.lower_high_idx, "收盤離開頸線超過 0.8%，不追")
                note = "收盤離開頸線超過 0.8%，不追"
                phase = "idle"
                setup = None
                continue
            ma_ok = not math.isnan(ma_slow[j]) and close[j] < ma_slow[j]
            vol_ok = not math.isnan(vol_ma[j]) and volume[j] >= vol_ma[j]
            near_high = _near_4h_high(high_4h[j], close[j], params.max_below_4h_high_pct)
            if ma_ok and vol_ok and near_high:
                signals.append(_signal(frame1, params, j, "break", close[j], setup, four_hour_high=float(high_4h[j])))
                filled_lower = setup.lower_high_idx
                phase = "cool"
                note = "這根 1 分收盤是起跌"
                continue
            if ma_ok and vol_ok:
                note = "觸發價離開四小時高點，這根不做"
                continue
            if ma_ok and not vol_ok:
                phase = "retest"
                retest_left = params.retest_bars
                bounce_high = high[j]
                note = "跌破的量能不夠，等拉回 1 分 MA7–MA25"
                continue
            note = "收盤跌破頸線，但還在 1 分 MA25 上面"
        elif phase == "watch":
            note = "這根 1 分還沒收在頸線下"

        if phase == "retest" and setup is not None:
            bounce_high = max(bounce_high, high[j])
            retest_left -= 1
            failed = retest_left < 0 or bounce_high >= setup.lower_high
            if failed:
                if bounce_high >= setup.lower_high:
                    reason = "回測高點回到較低高點，這組不做"
                else:
                    reason = "回測等超過 15 根 1 分，這組不做"
                kill(setup.lower_high_idx, reason)
                note = reason
                phase = "idle"
                setup = None
                continue
            fast = ma_fast[j]
            slow = ma_slow[j]
            if math.isnan(fast) or math.isnan(slow):
                note = "等價格拉回 1 分 MA7–MA25，再收在均線和頸線下面"
                continue
            band_lo = min(fast, slow)
            band_hi = max(fast, slow)
            tagged = high[j] >= band_lo and low[j] <= band_hi
            back_under = close[j] < band_lo and close[j] < setup.neckline
            extend = (band_lo - close[j]) / band_lo if band_lo else 1.0
            near_high = _near_4h_high(high_4h[j], close[j], params.max_below_4h_high_pct)
            if tagged and back_under and 0 <= extend <= params.max_entry_extension_pct and near_high:
                signals.append(
                    _signal(
                        frame1, params, j, "retest", close[j], setup,
                        stop_anchor=bounce_high, four_hour_high=float(high_4h[j]),
                    )
                )
                filled_lower = setup.lower_high_idx
                phase = "cool"
                note = "這根 1 分收盤是起跌"
            elif tagged and back_under and 0 <= extend <= params.max_entry_extension_pct:
                note = "觸發價離開四小時高點，這根不做"
            else:
                note = "等價格拉回 1 分 MA7–MA25，再收在均線和頸線下面"

    last = len(frame1) - 1
    fresh = signals[-1] if signals and signals[-1].bar_idx == last else None
    neck = last_levels.neckline if last_levels is not None else None
    extension = ((neck - float(close[last])) / neck) if neck else None
    slow_last = float(ma_slow[last])
    vol_last = float(vol_ma[last])
    high_last = float(high_4h[last])
    return RolloverView(
        signals=tuple(signals),
        timestamp=frame1.index[last],
        close=float(close[last]),
        phase=phase,
        note=note,
        extreme=None if last_levels is None else last_levels.extreme,
        lower_high=None if last_levels is None else last_levels.lower_high,
        neckline=neck,
        ma_slow=None if math.isnan(slow_last) else slow_last,
        volume=float(volume[last]),
        vol_ma=None if math.isnan(vol_last) else vol_last,
        extension=extension,
        four_hour_high=None if math.isnan(high_last) else high_last,
        fresh=fresh,
    )


def simulate_shorts(
    df: pd.DataFrame,
    signals: list[RolloverSignal],
    params: RolloverParams | None = None,
) -> list[ShortTrade]:
    """放空回測，K 線用進場的那一份（1 分）。同一根同時碰到停損和停利，算停損。"""

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


def _arm_5m(frame: pd.DataFrame, params: RolloverParams) -> list[tuple[pd.Timestamp, _Armed | None]]:
    """每根 5 分收盤之後，當時看得到的結構。1 分只能用已經收完的那根。"""

    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    steps: list[tuple[pd.Timestamp, _Armed | None]] = []
    extreme_i: int | None = None
    extreme_px: float | None = None
    lower_i: int | None = None
    lower_px: float | None = None
    neckline: float | None = None
    phase = "seek"
    left = params.swing_lookback

    for t in range(len(frame)):
        pivot_i = t - left
        if pivot_i >= left and _is_pivot_high(high, pivot_i, left):
            px = float(high[pivot_i])
            if extreme_px is None or px >= extreme_px:
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

        armed: _Armed | None = None
        if (
            phase == "watch"
            and extreme_i is not None
            and extreme_px is not None
            and lower_i is not None
            and lower_px is not None
            and neckline is not None
            and t - lower_i <= params.max_bars_after_lower_high
            and high[t] < extreme_px
            and close[t] <= extreme_px
        ):
            armed = _Armed(
                extreme=extreme_px,
                extreme_idx=extreme_i,
                neckline=neckline,
                lower_high=lower_px,
                lower_high_idx=lower_i,
                deadline=frame.index[lower_i]
                + pd.Timedelta(minutes=5 * (params.max_bars_after_lower_high + 1)),
            )
        elif phase == "watch":
            phase = "seek"
            extreme_i = None
            extreme_px = None
            lower_i = None
            lower_px = None
            neckline = None

        steps.append((frame.index[t] + pd.Timedelta(minutes=5), armed))

    return steps


def _signal(
    frame: pd.DataFrame,
    params: RolloverParams,
    bar_idx: int,
    kind: str,
    entry: float,
    setup: _Armed,
    stop_anchor: float | None = None,
    four_hour_high: float = 0.0,
) -> RolloverSignal:
    anchor = setup.lower_high if stop_anchor is None else stop_anchor
    stop = anchor * (1 + params.stop_buffer_pct)
    risk = stop - entry
    ts = frame.index[bar_idx]
    return RolloverSignal(
        timestamp=ts,
        bar_idx=bar_idx,
        kind=kind,
        entry=entry,
        stop=stop,
        target=entry - params.reward_r * risk,
        extreme=setup.extreme,
        extreme_idx=setup.extreme_idx,
        neckline=setup.neckline,
        lower_high=setup.lower_high,
        lower_high_idx=setup.lower_high_idx,
        off_session=not is_us_cash_session(ts),
        four_hour_high=four_hour_high,
    )


def _near_4h_high(high_4h: float, price: float, max_below: float) -> bool:
    """收盤還在四小時最高價的 max_below 以內。資料不足四小時就不是。"""

    if math.isnan(high_4h) or high_4h <= 0:
        return False
    return (high_4h - price) / high_4h <= max_below


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


def _require_step(index: pd.DatetimeIndex, minutes: int, label: str) -> None:
    if len(index) < 3:
        return
    delta = pd.Series(index).diff().dropna().median()
    if abs(delta - pd.Timedelta(minutes=minutes)) > pd.Timedelta(seconds=30):
        raise ValueError(f"這裡要 {label}，這份資料的間隔是 {delta}。")


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
