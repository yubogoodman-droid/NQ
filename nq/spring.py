"""假跌破（Spring）型態偵測：盤整 → 跌破支撐 → 迅速站回 → 放量上拉。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class FakeBreakdownPattern:
    """已確認的假跌破後上拉型態。"""

    range_start_idx: int
    range_end_idx: int
    spring_idx: int
    reclaim_idx: int
    breakout_idx: int | None
    support: float
    resistance: float
    spring_low: float
    range_pct: float
    break_pct: float
    volume_ratio: float

    @property
    def stop_loss(self) -> float:
        """停損設於假跌破最低點。"""
        return self.spring_low

    @property
    def measured_target(self) -> float:
        """量度目標：箱體高 + (箱體高 − 假跌破低)。"""
        return self.resistance + (self.resistance - self.spring_low)


def _rolling_mean(values: Sequence[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    window = 0.0
    for i, val in enumerate(values):
        window += val
        if i >= period:
            window -= values[i - period]
        if i >= period - 1:
            out[i] = window / period
    return out


def _is_tight_box(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    start: int,
    end: int,
    range_max_pct: float,
) -> tuple[float, float, float] | None:
    """回傳 (support, resistance, range_pct)；不夠橫盤則為 None。"""
    if end <= start:
        return None
    support = min(lows[start : end + 1])
    resistance = max(highs[start : end + 1])
    mid = (support + resistance) / 2
    if mid <= 0:
        return None
    range_pct = (resistance - support) / mid
    if range_pct > range_max_pct or range_pct <= 0:
        return None

    # 避免單邊趨勢穿過箱體：首尾收盤距離不得大於箱高的 60%
    box_h = resistance - support
    if abs(closes[end] - closes[start]) > 0.6 * box_h:
        return None

    # 多數 K 收在箱體內，且上下影線不要把箱體撐得名不副實
    inside = 0
    total = end - start + 1
    for i in range(start, end + 1):
        if support <= closes[i] <= resistance:
            inside += 1
    if inside / total < 0.85:
        return None

    return support, resistance, range_pct


def _ma_cluster_ok(
    ma_values: list[list[float | None]],
    idx: int,
    price: float,
    cluster_pct: float,
) -> bool:
    vals: list[float] = []
    for series in ma_values:
        v = series[idx]
        if v is None:
            return False
        vals.append(v)
    if price <= 0:
        return False
    spread = (max(vals) - min(vals)) / price
    if spread > cluster_pct:
        return False
    ma_mid = sum(vals) / len(vals)
    return abs(price - ma_mid) / price <= cluster_pct * 1.5


def detect_fake_breakdowns(
    df: pd.DataFrame,
    *,
    range_bars: int = 18,
    range_max_pct: float = 0.02,
    ma_periods: tuple[int, ...] = (5, 10, 20),
    ma_cluster_pct: float = 0.012,
    min_break_pct: float = 0.005,
    max_break_pct: float = 0.03,
    max_spring_bars: int = 15,
    max_reclaim_bars: int = 10,
    max_breakout_bars: int = 12,
    vol_ma: int = 20,
    breakout_vol_mult: float = 1.4,
    spring_vol_max_mult: float = 1.35,
    require_volume: bool = True,
    same_session: bool = True,
    skip_open_minutes: int = 5,
) -> list[FakeBreakdownPattern]:
    """
    在 OHLCV DataFrame 上偵測假跌破後上拉。

    四段條件（對應 1 分 K 上「先假跌破再往上拉」）：
    1. 盤整：近期箱體夠窄，均線糾結，價格貼著均線走
    2. 假跌破：短暫跌破箱體低，深度在 min/max_break_pct 之間，量能不宜爆量恐慌
    3. 站回：很快收盤站回原支撐
    4. 上拉：收盤突破箱體高，且量能放大
    5. 開盤濾波：箱體與突破同一交易日，並略過開盤後 skip_open_minutes 分鐘

    Parameters
    ----------
    df : DataFrame
        需含 open, high, low, close；volume 可選（require_volume=True 時必填）。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")
    if require_volume and "volume" not in df.columns:
        raise ValueError("需要 volume 欄位才能做放量確認")

    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    volumes = df["volume"].tolist() if "volume" in df.columns else [0.0] * len(df)
    n = len(df)
    if n < max(ma_periods) + range_bars + max_spring_bars + 2:
        return []

    ma_series = [_rolling_mean(closes, p) for p in ma_periods]
    vol_sma = _rolling_mean(volumes, vol_ma)

    patterns: list[FakeBreakdownPattern] = []
    min_end = max(max(ma_periods), range_bars) - 1

    for range_end in range(min_end, n - 2):
        range_start = range_end - range_bars + 1
        box = _is_tight_box(highs, lows, closes, range_start, range_end, range_max_pct)
        if box is None:
            continue
        support, resistance, range_pct = box
        if not _ma_cluster_ok(ma_series, range_end, closes[range_end], ma_cluster_pct):
            continue

        spring_limit = min(range_end + max_spring_bars, n - 1)
        spring_idx: int | None = None
        spring_low = float("inf")
        broken = False
        for j in range(range_end + 1, spring_limit + 1):
            if lows[j] < support * (1.0 - min_break_pct):
                broken = True
            if broken and lows[j] < spring_low:
                spring_low = lows[j]
                spring_idx = j
            if broken and spring_idx is not None and closes[j] > support:
                break
        if not broken or spring_idx is None:
            continue

        break_pct = (support - spring_low) / support
        if break_pct < min_break_pct or break_pct > max_break_pct:
            continue

        if require_volume:
            spring_start = range_end + 1
            spring_vols = volumes[spring_start : spring_idx + 1]
            base = vol_sma[range_end]
            if spring_vols and base and base > 0:
                if (sum(spring_vols) / len(spring_vols)) > base * spring_vol_max_mult:
                    continue

        reclaim_limit = min(spring_idx + max_reclaim_bars, n - 1)
        reclaim_idx: int | None = None
        failed = False
        for k in range(spring_idx, reclaim_limit + 1):
            if lows[k] < spring_low * 0.999:
                failed = True
                break
            if closes[k] > support:
                reclaim_idx = k
                break
        if failed or reclaim_idx is None:
            continue

        breakout_limit = min(reclaim_idx + max_breakout_bars, n - 1)
        breakout_idx: int | None = None
        volume_ratio = 0.0
        for k in range(reclaim_idx, breakout_limit + 1):
            if lows[k] < spring_low:
                break
            vol_ok = True
            ratio = 0.0
            if require_volume:
                base = vol_sma[k]
                if not base or base <= 0:
                    vol_ok = False
                else:
                    ratio = volumes[k] / base
                    vol_ok = ratio >= breakout_vol_mult
            if closes[k] > resistance and vol_ok:
                breakout_idx = k
                volume_ratio = ratio
                break
        if breakout_idx is None:
            continue
        if not _session_ok(
            df,
            range_start,
            breakout_idx,
            same_session=same_session,
            skip_open_minutes=skip_open_minutes,
        ):
            continue

        patterns.append(
            FakeBreakdownPattern(
                range_start_idx=range_start,
                range_end_idx=range_end,
                spring_idx=spring_idx,
                reclaim_idx=reclaim_idx,
                breakout_idx=breakout_idx,
                support=support,
                resistance=resistance,
                spring_low=spring_low,
                range_pct=range_pct,
                break_pct=break_pct,
                volume_ratio=volume_ratio,
            )
        )

    return _dedupe_patterns(patterns)


def _as_local(ts: pd.Timestamp) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("Asia/Taipei")
    return ts


def _session_ok(
    df: pd.DataFrame,
    range_start: int,
    breakout_idx: int,
    *,
    same_session: bool,
    skip_open_minutes: int,
) -> bool:
    """擋掉跨夜箱體，以及開盤前幾分鐘的跳空雜訊。"""
    start_ts = _as_local(df.index[range_start])
    break_ts = _as_local(df.index[breakout_idx])
    if same_session and start_ts.date() != break_ts.date():
        return False
    if skip_open_minutes <= 0:
        return True
    day = break_ts.date()
    open_ts = None
    for ts in df.index:
        local = _as_local(ts)
        if local.date() == day:
            open_ts = local
            break
    if open_ts is None:
        return True
    return break_ts >= open_ts + pd.Timedelta(minutes=skip_open_minutes)


def _score(p: FakeBreakdownPattern) -> tuple[float, float]:
    """優先緊箱體、假跌破深度適中。"""
    depth_score = min(p.break_pct, 0.02) - abs(p.break_pct - 0.012)
    return (-p.range_pct, depth_score)


def _dedupe_patterns(patterns: Iterable[FakeBreakdownPattern]) -> list[FakeBreakdownPattern]:
    """同一波上拉只留一個訊號：先依突破 K 去重，再合併鄰近 5 根。"""
    by_breakout: dict[int, FakeBreakdownPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        existing = by_breakout.get(p.breakout_idx)
        if existing is None or _score(p) > _score(existing):
            by_breakout[p.breakout_idx] = p

    ordered = sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)
    kept: list[FakeBreakdownPattern] = []
    for p in ordered:
        if kept and p.breakout_idx is not None and kept[-1].breakout_idx is not None:
            if p.breakout_idx - kept[-1].breakout_idx <= 5:
                if _score(p) > _score(kept[-1]):
                    kept[-1] = p
                continue
        kept.append(p)
    return kept
