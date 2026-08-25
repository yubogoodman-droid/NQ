"""W 底（雙底）型態偵測。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WBottomPattern:
    """已確認的 W 底型態。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    breakout_idx: int | None = None

    @property
    def stop_loss(self) -> float:
        """停損設於第二個低點下方。"""
        return self.second_low

    @property
    def target(self) -> float:
        """量度目標：頸線 + (頸線 - 最低點)。"""
        depth = self.neckline - min(self.first_low, self.second_low)
        return self.neckline + depth


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int, allow_tie: bool = False) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    if pivot != min(window):
        return False
    if not allow_tie:
        return window.count(pivot) == 1
    # 平底只取最後一根，避免 515/515/515 整段都不是轉折
    last = lookback
    for j in range(lookback + 1, len(window)):
        if window[j] == pivot:
            last = j
    return last == lookback


def _is_swing_high(highs: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(highs) - lookback:
        return False
    pivot = highs[idx]
    window = highs[idx - lookback : idx + lookback + 1]
    return pivot == max(window) and window.count(pivot) == 1


def _find_swing_lows(lows: Sequence[float], lookback: int, allow_tie: bool = False) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback, allow_tie)]


def detect_w_bottoms(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 3,
    low_tolerance_pct: float = 0.001,
    min_bars_between_lows: int = 5,
    max_bars_between_lows: int = 60,
    require_neckline_break: bool = True,
) -> list[WBottomPattern]:
    """
    在 OHLCV DataFrame 上偵測 W 底型態。

    條件：
    1. 兩個相近的波段低點（價差在 low_tolerance_pct 內）
    2. 兩低點之間有明確的頸線高點
    3. 第二低點確認後，收盤突破頸線（可選）

    Parameters
    ----------
    df : DataFrame
        需含 open, high, low, close 欄位，index 為時間。
    swing_lookback : int
        左右各幾根 K 確認轉折。
    low_tolerance_pct : float
        兩低點允許的最大價差比例（0.001 = 0.1%）。
    min_bars_between_lows : int
        兩低點最少間隔 K 數。
    max_bars_between_lows : int
        兩低點最多間隔 K 數。
    require_neckline_break : bool
        是否要求收盤突破頸線才視為有效。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    lows = df["low"].tolist()
    highs = df["high"].tolist()
    closes = df["close"].tolist()

    swing_lows = _find_swing_lows(lows, swing_lookback)
    patterns: list[WBottomPattern] = []

    for i, first_idx in enumerate(swing_lows):
        for second_idx in swing_lows[i + 1 :]:
            gap = second_idx - first_idx
            if gap < min_bars_between_lows:
                continue
            if gap > max_bars_between_lows:
                break

            first_low = lows[first_idx]
            second_low = lows[second_idx]
            avg_low = (first_low + second_low) / 2
            if avg_low == 0:
                continue
            if abs(first_low - second_low) / avg_low > low_tolerance_pct:
                continue

            # 頸線：兩低點之間的最高波段高點
            neckline_idx: int | None = None
            neckline_price = float("-inf")
            for j in range(first_idx + swing_lookback, second_idx - swing_lookback + 1):
                if _is_swing_high(highs, j, swing_lookback) and highs[j] > neckline_price:
                    neckline_idx = j
                    neckline_price = highs[j]

            if neckline_idx is None:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                for k in range(second_idx + swing_lookback, len(closes)):
                    if closes[k] > neckline_price:
                        breakout_idx = k
                        break
                if breakout_idx is None:
                    continue

            patterns.append(
                WBottomPattern(
                    first_low_idx=first_idx,
                    second_low_idx=second_idx,
                    neckline_idx=neckline_idx,
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    breakout_idx=breakout_idx,
                )
            )

    return _dedupe_patterns(patterns)


def _dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    """同一突破 K 只保留風險報酬最佳的型態。"""
    by_breakout: dict[int, WBottomPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        depth = p.neckline - min(p.first_low, p.second_low)
        risk = p.neckline - p.second_low
        rr = depth / risk if risk > 0 else 0
        existing = by_breakout.get(p.breakout_idx)
        if existing is None:
            by_breakout[p.breakout_idx] = p
            continue
        ex_depth = existing.neckline - min(existing.first_low, existing.second_low)
        ex_risk = existing.neckline - existing.second_low
        ex_rr = ex_depth / ex_risk if ex_risk > 0 else 0
        if rr > ex_rr:
            by_breakout[p.breakout_idx] = p
    return sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)


@dataclass(frozen=True)
class WMa20Signal:
    """五分 K W 底之後，收盤上穿 MA20。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    cross_idx: int
    cross_price: float
    ma20: float

    @property
    def stop_loss(self) -> float:
        return min(self.first_low, self.second_low)

    @property
    def target(self) -> float:
        depth = self.neckline - min(self.first_low, self.second_low)
        return self.neckline + depth

    @property
    def w_depth_pct(self) -> float:
        base = min(self.first_low, self.second_low)
        if base <= 0:
            return 0.0
        return (self.neckline - base) / base


def _ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """接受 open/high/low/close 或 Open/High/Low/Close。"""
    lower = {str(c).lower(): c for c in df.columns}
    missing = {"open", "high", "low", "close"} - set(lower)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")
    out = pd.DataFrame(
        {
            "open": df[lower["open"]],
            "high": df[lower["high"]],
            "low": df[lower["low"]],
            "close": df[lower["close"]],
        },
        index=df.index,
    )
    if "volume" in lower:
        out["volume"] = df[lower["volume"]]
    return out


def detect_w_ma20_crosses(
    df: pd.DataFrame,
    *,
    ma_period: int = 20,
    swing_lookback: int = 2,
    min_bars_between_lows: int = 6,
    max_bars_between_lows: int = 48,
    low_below_pct: float = 0.015,
    low_above_pct: float = 0.025,
    min_neck_pct: float = 0.008,
    min_prior_drop_pct: float = 0.025,
    prior_lookback: int = 36,
    max_bars_to_cross: int = 36,
    invalidate_pct: float = 0.003,
) -> list[WMa20Signal]:
    """
    視覺 W 底（雙底）形成後，收盤由下往上穿過 MA20。

    對齊券商五分 K 圖那種走法：先大跌、做出兩個相近低點，
    反彈過程收盤站上五分 MA20 才通知。不要求先突破頸線。
    """
    ohlc = _ohlc_frame(df)
    if len(ohlc) < ma_period + max_bars_between_lows:
        return []

    highs = ohlc["high"].tolist()
    lows = ohlc["low"].tolist()
    closes = ohlc["close"].tolist()
    ma = ohlc["close"].rolling(ma_period, min_periods=ma_period).mean().tolist()

    swing_lows = _find_swing_lows(lows, swing_lookback, allow_tie=True)
    signals: list[WMa20Signal] = []

    for i, first_idx in enumerate(swing_lows):
        first_low = lows[first_idx]
        if first_low <= 0:
            continue
        look_from = max(0, first_idx - prior_lookback)
        prior_high = max(highs[look_from : first_idx + 1])
        if (prior_high - first_low) / first_low < min_prior_drop_pct:
            continue
        ma1 = ma[first_idx]
        if ma1 != ma1 or first_low >= ma1:
            continue

        for second_idx in swing_lows[i + 1 :]:
            gap = second_idx - first_idx
            if gap < min_bars_between_lows:
                continue
            if gap > max_bars_between_lows:
                break

            second_low = lows[second_idx]
            if second_low <= 0:
                continue
            rel = (second_low - first_low) / first_low
            if rel < -low_below_pct or rel > low_above_pct:
                continue

            ma2 = ma[second_idx]
            if ma2 != ma2 or second_low >= ma2:
                continue

            mid_slice = slice(first_idx + 1, second_idx)
            if second_idx - first_idx < 2:
                continue
            neckline_price = max(highs[mid_slice])
            neckline_idx = first_idx + 1 + highs[mid_slice].index(neckline_price)
            avg_low = (first_low + second_low) / 2
            if (neckline_price - avg_low) / avg_low < min_neck_pct:
                continue

            confirm = second_idx + swing_lookback
            if confirm >= len(closes):
                continue

            floor = min(first_low, second_low) * (1.0 - invalidate_pct)
            first_cross: int | None = None
            start_k = max(second_idx, ma_period)
            end_k = min(len(closes), second_idx + max_bars_to_cross + 1)
            for k in range(start_k, end_k):
                if min(lows[second_idx : k + 1]) < floor:
                    break
                prev_ma, cur_ma = ma[k - 1], ma[k]
                if prev_ma != prev_ma or cur_ma != cur_ma:
                    continue
                # 要價往上穿均線，不要均線掉下來吻到走平的收盤
                if closes[k] <= closes[k - 1]:
                    continue
                if closes[k - 1] <= prev_ma and closes[k] > cur_ma:
                    first_cross = k
                    break
            if first_cross is None:
                continue

            if first_cross >= confirm:
                cross_idx = first_cross
            elif closes[confirm] > ma[confirm]:
                cross_idx = confirm
            else:
                continue
            if min(lows[second_idx : cross_idx + 1]) < floor:
                continue

            signals.append(
                WMa20Signal(
                    first_low_idx=first_idx,
                    second_low_idx=second_idx,
                    neckline_idx=neckline_idx,
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    cross_idx=cross_idx,
                    cross_price=closes[cross_idx],
                    ma20=float(ma[cross_idx]),
                )
            )

    return _dedupe_w_ma20(signals)


def _w_ma20_rank(sig: WMa20Signal) -> tuple[int, float, float]:
    """同一根上穿：優先剛完成的 W，再看兩低點對稱、深度。"""
    recency = -(sig.cross_idx - sig.second_low_idx)
    avg = (sig.first_low + sig.second_low) / 2
    equal = -abs(sig.first_low - sig.second_low) / avg if avg else 0.0
    return (recency, equal, sig.w_depth_pct)


def _dedupe_w_ma20(signals: Iterable[WMa20Signal], min_gap: int = 6) -> list[WMa20Signal]:
    """同一根上穿只留一組 W；30 分鐘內的連續上穿合併成一筆。"""
    by_cross: dict[int, WMa20Signal] = {}
    for sig in signals:
        existing = by_cross.get(sig.cross_idx)
        if existing is None or _w_ma20_rank(sig) > _w_ma20_rank(existing):
            by_cross[sig.cross_idx] = sig
    kept: list[WMa20Signal] = []
    for sig in sorted(by_cross.values(), key=lambda s: s.cross_idx):
        if kept and sig.cross_idx - kept[-1].cross_idx < min_gap:
            continue
        kept.append(sig)
    return kept
