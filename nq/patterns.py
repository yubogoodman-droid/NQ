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
    """五分 K W 底之後，MA5 > MA10 > MA20 多頭排列進場。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    cross_idx: int
    cross_price: float
    ma5: float
    ma10: float
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

    @property
    def stacked(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20 and self.cross_price > self.ma5


def _bar_day(ts: object) -> object:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Taipei")
    return t.date()


def _has_earlier_shelf_low(
    lows: Sequence[float],
    swing_lows: Sequence[int],
    first_idx: int,
    first_low: float,
    index: pd.Index,
    *,
    shelf_pct: float = 0.004,
) -> bool:
    """同一天、更早已經測過同一層低點 → 後面是箱型支撐，不是 W 的左腳。"""
    if first_low <= 0:
        return False
    ceiling = first_low * (1.0 + shelf_pct)
    first_day = _bar_day(index[first_idx])
    for j in swing_lows:
        if j >= first_idx:
            break
        if _bar_day(index[j]) != first_day:
            continue
        if lows[j] <= ceiling:
            return True
    return False


def _is_visual_w_pair(
    first_idx: int,
    second_idx: int,
    neckline_idx: int,
    swing_lows: Sequence[int],
    *,
    min_neck_offset: int = 2,
    min_right_bars: int = 3,
    max_mid_swing_lows: int = 1,
) -> bool:
    """視覺 W：中間有反彈高峰，不是殺完立刻橫盤、中間一堆底。"""
    if neckline_idx - first_idx < min_neck_offset:
        return False
    if second_idx - neckline_idx < min_right_bars:
        return False
    extra = sum(1 for j in swing_lows if first_idx < j < second_idx)
    return extra <= max_mid_swing_lows


def _neck_retrace_ok(
    neckline: float,
    floor: float,
    session_high: float,
    *,
    max_retrace: float = 0.80,
) -> bool:
    """頸線若幾乎漲回殺勢起點，那是 V 型反彈再回測，不是打在低檔的 W。"""
    dump = session_high - floor
    if dump <= 0 or neckline < floor:
        return False
    return (neckline - floor) / dump <= max_retrace


def _w_height_vs_bar_ok(
    highs: Sequence[float],
    lows: Sequence[float],
    first_idx: int,
    second_idx: int,
    neckline: float,
    floor: float,
    *,
    min_ratio: float = 2.3,
) -> bool:
    """W 高度要明顯大於區間內平均振幅，否則只是殺完貼著的淺箱。"""
    if second_idx < first_idx or floor <= 0:
        return False
    ranges = [highs[i] - lows[i] for i in range(first_idx, second_idx + 1)]
    avg = sum(ranges) / len(ranges) if ranges else 0.0
    if avg <= 0:
        return False
    return (neckline - floor) / avg >= min_ratio


def _w_not_floor_box(
    lows: Sequence[float],
    first_idx: int,
    second_idx: int,
    floor: float,
    *,
    max_hug: float = 0.85,
    hug_pct: float = 0.006,
) -> bool:
    """兩低點之間大多數 K 都貼著地板 → 箱型，不是兩個谷。"""
    n = second_idx - first_idx + 1
    if n <= 0 or floor <= 0:
        return False
    ceiling = floor * (1.0 + hug_pct)
    hug = sum(1 for i in range(first_idx, second_idx + 1) if lows[i] <= ceiling)
    return hug / n <= max_hug


def _right_leg_ok(neckline: float, second_low: float, *, min_pct: float = 0.010) -> bool:
    """右腳要從頸線明顯回落，否則只是殺完貼箱、沒有第二個谷。"""
    if second_low <= 0 or neckline <= second_low:
        return False
    return (neckline - second_low) / second_low >= min_pct


def _w_vs_dump_atr_ok(
    highs: Sequence[float],
    lows: Sequence[float],
    sess: int,
    first_idx: int,
    neckline: float,
    floor: float,
    *,
    min_ratio: float = 1.10,
) -> bool:
    """W 高度要蓋過殺勢那一段的平均振幅，否則瀑布圖上只剩一個小台階。"""
    if first_idx < sess or floor <= 0:
        return False
    ranges = [highs[i] - lows[i] for i in range(sess, first_idx + 1)]
    avg = sum(ranges) / len(ranges) if ranges else 0.0
    if avg <= 0:
        return False
    return (neckline - floor) / avg >= min_ratio


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
    low_below_pct: float = 0.002,
    low_above_pct: float = 0.025,
    min_neck_pct: float = 0.015,
    min_prior_drop_pct: float = 0.025,
    min_session_drop_pct: float = 0.035,
    prior_lookback: int = 36,
    max_bars_to_cross: int = 14,
    invalidate_pct: float = 0.003,
    min_neck_offset: int = 2,
    min_right_bars: int = 3,
    max_mid_swing_lows: int = 1,
    earlier_shelf_pct: float = 0.004,
    min_dump_bars: int = 4,
    max_neck_retrace: float = 0.80,
    min_height_vs_bar: float = 2.3,
    max_floor_hug: float = 0.85,
    min_right_leg_pct: float = 0.010,
    min_height_vs_dump_atr: float = 1.10,
) -> list[WMa20Signal]:
    """
    視覺 W 底形成後，五分 K 出現 MA5 > MA10 > MA20 多頭排列才進場。

    對齊券商五分圖：先大跌做出雙底，反彈等到 5/10/20 多排
    （收盤也站上 MA5）才通知。不要求先突破頸線。

    排除缺口後貼著箱型打底（例如旺宏、華邦電）：
    當天五分圖要先有夠深的殺勢（不能只靠隔夜缺口），
    頸線不能貼在第一低旁邊、兩低點之間不能一再測底，
    而且第一低不能已是當天同一層的第二次。

    也排除看起來不像 W 的三種圖：
    開盤當根就當 L1、頸線幾乎漲回殺勢起點（台虹那種 V），
    以及殺完貼著地板的淺箱（台勝科、台玻那種）。
    第二腳明顯更低（下階）、右腳沒從頸線掉下來、
    或第二低之後橫很久才多排，也不算。
    """
    ohlc = _ohlc_frame(df)
    if len(ohlc) < ma_period + max_bars_between_lows:
        return []

    highs = ohlc["high"].tolist()
    lows = ohlc["low"].tolist()
    closes = ohlc["close"].tolist()
    ma5 = ohlc["close"].rolling(5, min_periods=5).mean().tolist()
    ma10 = ohlc["close"].rolling(10, min_periods=10).mean().tolist()
    ma20 = ohlc["close"].rolling(ma_period, min_periods=ma_period).mean().tolist()

    def _stacked(k: int) -> bool:
        if k < 0:
            return False
        a, b, c = ma5[k], ma10[k], ma20[k]
        if a != a or b != b or c != c:
            return False
        return a > b > c and closes[k] > a

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
        sess = first_idx
        first_day = _bar_day(ohlc.index[first_idx])
        while sess > 0 and _bar_day(ohlc.index[sess - 1]) == first_day:
            sess -= 1
        sess_high = max(highs[sess : first_idx + 1])
        if (sess_high - first_low) / first_low < min_session_drop_pct:
            continue
        if first_idx - sess < min_dump_bars:
            continue
        ma1 = ma20[first_idx]
        if ma1 != ma1 or first_low >= ma1:
            continue
        if _has_earlier_shelf_low(
            lows,
            swing_lows,
            first_idx,
            first_low,
            ohlc.index,
            shelf_pct=earlier_shelf_pct,
        ):
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

            ma2 = ma20[second_idx]
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
            if not _is_visual_w_pair(
                first_idx,
                second_idx,
                neckline_idx,
                swing_lows,
                min_neck_offset=min_neck_offset,
                min_right_bars=min_right_bars,
                max_mid_swing_lows=max_mid_swing_lows,
            ):
                continue
            w_floor = min(first_low, second_low)
            if not _neck_retrace_ok(
                neckline_price,
                w_floor,
                sess_high,
                max_retrace=max_neck_retrace,
            ):
                continue
            if not _w_height_vs_bar_ok(
                highs,
                lows,
                first_idx,
                second_idx,
                neckline_price,
                w_floor,
                min_ratio=min_height_vs_bar,
            ):
                continue
            if not _w_not_floor_box(
                lows,
                first_idx,
                second_idx,
                w_floor,
                max_hug=max_floor_hug,
            ):
                continue
            if not _right_leg_ok(
                neckline_price,
                second_low,
                min_pct=min_right_leg_pct,
            ):
                continue
            if not _w_vs_dump_atr_ok(
                highs,
                lows,
                sess,
                first_idx,
                neckline_price,
                w_floor,
                min_ratio=min_height_vs_dump_atr,
            ):
                continue

            confirm = second_idx + swing_lookback
            if confirm >= len(closes):
                continue

            floor = min(first_low, second_low) * (1.0 - invalidate_pct)
            entry: int | None = None
            start_k = max(confirm, ma_period)
            end_k = min(len(closes), second_idx + max_bars_to_cross + 1)
            for k in range(start_k, end_k):
                if min(lows[second_idx : k + 1]) < floor:
                    break
                if not _stacked(k):
                    continue
                # 第一次翻成多排才進，已經多排走的不算
                if _stacked(k - 1):
                    continue
                entry = k
                break
            if entry is None:
                continue
            if min(lows[second_idx : entry + 1]) < floor:
                continue

            signals.append(
                WMa20Signal(
                    first_low_idx=first_idx,
                    second_low_idx=second_idx,
                    neckline_idx=neckline_idx,
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    cross_idx=entry,
                    cross_price=closes[entry],
                    ma5=float(ma5[entry]),
                    ma10=float(ma10[entry]),
                    ma20=float(ma20[entry]),
                )
            )

    return _dedupe_w_ma20(signals)


def _w_ma20_rank(sig: WMa20Signal) -> tuple[int, float, float]:
    """同一根進場：優先剛完成的 W，再看兩低點對稱、深度。"""
    recency = -(sig.cross_idx - sig.second_low_idx)
    avg = (sig.first_low + sig.second_low) / 2
    equal = -abs(sig.first_low - sig.second_low) / avg if avg else 0.0
    return (recency, equal, sig.w_depth_pct)


def _dedupe_w_ma20(signals: Iterable[WMa20Signal], min_gap: int = 6) -> list[WMa20Signal]:
    """同一根多排只留一組 W；30 分鐘內的連續訊號合併成一筆。"""
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
