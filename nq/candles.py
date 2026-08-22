"""一分 K 棒型態偵測。

涵蓋教科書單根／雙根／三根 K，以及南亞科那種「盤整後放量長紅突破」。
偵測在收盤當根完成，進場應等到下一根開盤，避免偷看當根高低。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandlePattern:
    """在 end_idx 收盤確認的型態。"""

    name: str
    name_zh: str
    side: str
    start_idx: int
    end_idx: int
    low: float
    high: float

    @property
    def bars(self) -> int:
        return self.end_idx - self.start_idx + 1


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _bar_range(h: float, l: float) -> float:
    return max(h - l, 0.0)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _is_bull(o: float, c: float) -> bool:
    return c > o


def _is_bear(o: float, c: float) -> bool:
    return c < o


def add_candle_features(df: pd.DataFrame, *, atr_period: int = 14, vol_period: int = 20) -> pd.DataFrame:
    """補 ATR、均量、當日開盤，給過濾與停損用。"""
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
    out["vol_sma"] = out["volume"].rolling(vol_period, min_periods=vol_period).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["session_date"] = [pd.Timestamp(ts).date() for ts in out.index]
    out["day_open"] = out.groupby("session_date")["open"].transform("first")
    return out


def _day_move_pct(close: float, day_open: float) -> float:
    if day_open <= 0 or np.isnan(day_open):
        return 0.0
    return close / day_open - 1.0


def detect_candle_patterns(
    df: pd.DataFrame,
    *,
    max_chase_pct: float = 0.06,
    min_atr_mult: float = 0.45,
    include_range_breakout: bool = True,
) -> list[CandlePattern]:
    """掃描一分 K，回傳已確認的型態（同一根可有多個，之後策略再排序）。"""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    work = add_candle_features(df)
    o = work["open"].to_numpy(dtype=float)
    h = work["high"].to_numpy(dtype=float)
    l = work["low"].to_numpy(dtype=float)
    c = work["close"].to_numpy(dtype=float)
    v = work["volume"].to_numpy(dtype=float)
    atr = work["atr"].to_numpy(dtype=float)
    vol_sma = work["vol_sma"].to_numpy(dtype=float)
    day_open = work["day_open"].to_numpy(dtype=float)

    patterns: list[CandlePattern] = []
    start = 20
    for i in range(start, len(work)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if _bar_range(h[i], l[i]) < min_atr_mult * atr[i] and i >= 2:
            # 當根過小通常不是有效反轉；突破型另判。
            tiny = True
        else:
            tiny = False

        found = _detect_at(
            i,
            o,
            h,
            l,
            c,
            v,
            atr,
            vol_sma,
            tiny=tiny,
        )
        if include_range_breakout:
            brk = _range_breakout_at(i, o, h, l, c, v, atr, vol_sma)
            if brk is not None:
                found.append(brk)

        if not found:
            continue

        move = _day_move_pct(c[i], day_open[i])
        for p in found:
            if p.side == "long" and move > max_chase_pct:
                continue
            if p.side == "short" and move < -max_chase_pct:
                continue
            patterns.append(p)

    return patterns


def _detect_at(
    i: int,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    atr: np.ndarray,
    vol_sma: np.ndarray,
    *,
    tiny: bool,
) -> list[CandlePattern]:
    found: list[CandlePattern] = []
    if not tiny:
        p = _hammer(i, o, h, l, c)
        if p:
            found.append(p)
        p = _shooting_star(i, o, h, l, c)
        if p:
            found.append(p)
        p = _marubozu(i, o, h, l, c, v, atr, vol_sma)
        if p:
            found.append(p)

    if i >= 1:
        p = _engulfing(i, o, h, l, c, v)
        if p:
            found.append(p)
        p = _piercing_or_cloud(i, o, h, l, c)
        if p:
            found.append(p)
        p = _tweezers(i, o, h, l, c, atr)
        if p:
            found.append(p)

    if i >= 2:
        p = _morning_or_evening_star(i, o, h, l, c)
        if p:
            found.append(p)
        p = _three_soldiers_or_crows(i, o, h, l, c, atr)
        if p:
            found.append(p)
    return found


def _hammer(i: int, o, h, l, c) -> CandlePattern | None:
    rng = _bar_range(h[i], l[i])
    body = _body(o[i], c[i])
    if rng <= 0 or body <= 0:
        return None
    lower = _lower_wick(o[i], l[i], c[i])
    upper = _upper_wick(o[i], h[i], c[i])
    if lower < 2.0 * body:
        return None
    if upper > 0.35 * body and upper > 0.15 * rng:
        return None
    if min(o[i], c[i]) > l[i] + 0.45 * rng:
        # 實體要在上半部
        return CandlePattern("hammer", "錘子線", "long", i, i, l[i], h[i])
    return None


def _shooting_star(i: int, o, h, l, c) -> CandlePattern | None:
    rng = _bar_range(h[i], l[i])
    body = _body(o[i], c[i])
    if rng <= 0 or body <= 0:
        return None
    lower = _lower_wick(o[i], l[i], c[i])
    upper = _upper_wick(o[i], h[i], c[i])
    if upper < 2.0 * body:
        return None
    if lower > 0.35 * body and lower > 0.15 * rng:
        return None
    if max(o[i], c[i]) < l[i] + 0.55 * rng:
        return CandlePattern("shooting_star", "射擊之星", "short", i, i, l[i], h[i])
    return None


def _marubozu(i: int, o, h, l, c, v, atr, vol_sma) -> CandlePattern | None:
    rng = _bar_range(h[i], l[i])
    body = _body(o[i], c[i])
    if rng <= 0 or body / rng < 0.82:
        return None
    if body < 1.15 * atr[i]:
        return None
    if not np.isnan(vol_sma[i]) and vol_sma[i] > 0 and v[i] < 1.4 * vol_sma[i]:
        return None
    if _is_bull(o[i], c[i]):
        return CandlePattern("bull_marubozu", "大陽無影", "long", i, i, l[i], h[i])
    if _is_bear(o[i], c[i]):
        return CandlePattern("bear_marubozu", "大陰無影", "short", i, i, l[i], h[i])
    return None


def _engulfing(i: int, o, h, l, c, v) -> CandlePattern | None:
    prev_body = _body(o[i - 1], c[i - 1])
    body = _body(o[i], c[i])
    if prev_body <= 0 or body <= prev_body:
        return None
    if _is_bear(o[i - 1], c[i - 1]) and _is_bull(o[i], c[i]):
        if o[i] <= c[i - 1] and c[i] >= o[i - 1]:
            if v[i] >= v[i - 1]:
                return CandlePattern("bull_engulfing", "多頭吞噬", "long", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    if _is_bull(o[i - 1], c[i - 1]) and _is_bear(o[i], c[i]):
        if o[i] >= c[i - 1] and c[i] <= o[i - 1]:
            if v[i] >= v[i - 1]:
                return CandlePattern("bear_engulfing", "空頭吞噬", "short", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    return None


def _piercing_or_cloud(i: int, o, h, l, c) -> CandlePattern | None:
    prev_body = _body(o[i - 1], c[i - 1])
    body = _body(o[i], c[i])
    if prev_body <= 0 or body <= 0:
        return None
    mid = (o[i - 1] + c[i - 1]) / 2
    if _is_bear(o[i - 1], c[i - 1]) and _is_bull(o[i], c[i]):
        if o[i] < l[i - 1] and c[i] > mid and c[i] < o[i - 1]:
            return CandlePattern("piercing", "刺透線", "long", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    if _is_bull(o[i - 1], c[i - 1]) and _is_bear(o[i], c[i]):
        if o[i] > h[i - 1] and c[i] < mid and c[i] > o[i - 1]:
            return CandlePattern("dark_cloud", "烏雲蓋頂", "short", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    return None


def _tweezers(i: int, o, h, l, c, atr) -> CandlePattern | None:
    # 1 分 K 雜訊多，價差必須接近 ATR 的一小段，不能只用價格百分比。
    atr_i = atr[i] if not np.isnan(atr[i]) else 0.0
    tol = max(abs(l[i]) * 0.00012, 1e-9)
    if atr_i > 0:
        tol = min(tol, 0.10 * atr_i)
    if _body(o[i - 1], c[i - 1]) < 0.25 * atr_i or _body(o[i], c[i]) < 0.25 * atr_i:
        return None
    if abs(l[i] - l[i - 1]) <= tol and _is_bear(o[i - 1], c[i - 1]) and _is_bull(o[i], c[i]):
        return CandlePattern("tweezer_bottom", "鑷底", "long", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    if abs(h[i] - h[i - 1]) <= tol and _is_bull(o[i - 1], c[i - 1]) and _is_bear(o[i], c[i]):
        return CandlePattern("tweezer_top", "鑷頂", "short", i - 1, i, min(l[i - 1], l[i]), max(h[i - 1], h[i]))
    return None


def _morning_or_evening_star(i: int, o, h, l, c) -> CandlePattern | None:
    b0 = _body(o[i - 2], c[i - 2])
    b1 = _body(o[i - 1], c[i - 1])
    b2 = _body(o[i], c[i])
    r0 = _bar_range(h[i - 2], l[i - 2])
    if b0 <= 0 or b2 <= 0 or r0 <= 0:
        return None
    mid0 = (o[i - 2] + c[i - 2]) / 2
    # 中間星線：實體明顯小於第一根
    if b1 > 0.45 * b0:
        return None
    if _is_bear(o[i - 2], c[i - 2]) and _is_bull(o[i], c[i]) and c[i] > mid0:
        if max(o[i - 1], c[i - 1]) < min(o[i - 2], c[i - 2]):
            return CandlePattern(
                "morning_star",
                "晨星",
                "long",
                i - 2,
                i,
                min(l[i - 2], l[i - 1], l[i]),
                max(h[i - 2], h[i - 1], h[i]),
            )
    if _is_bull(o[i - 2], c[i - 2]) and _is_bear(o[i], c[i]) and c[i] < mid0:
        if min(o[i - 1], c[i - 1]) > max(o[i - 2], c[i - 2]):
            return CandlePattern(
                "evening_star",
                "夜星",
                "short",
                i - 2,
                i,
                min(l[i - 2], l[i - 1], l[i]),
                max(h[i - 2], h[i - 1], h[i]),
            )
    return None


def _three_soldiers_or_crows(i: int, o, h, l, c, atr) -> CandlePattern | None:
    if not (_is_bull(o[i - 2], c[i - 2]) and _is_bull(o[i - 1], c[i - 1]) and _is_bull(o[i], c[i])):
        soldiers = False
    else:
        soldiers = c[i] > c[i - 1] > c[i - 2]
        soldiers = soldiers and all(_body(o[j], c[j]) >= 0.55 * _bar_range(h[j], l[j]) for j in (i - 2, i - 1, i))
        soldiers = soldiers and all(_body(o[j], c[j]) >= 0.5 * atr[i] for j in (i - 2, i - 1, i))
    if soldiers:
        return CandlePattern(
            "three_white_soldiers",
            "三白兵",
            "long",
            i - 2,
            i,
            min(l[i - 2], l[i - 1], l[i]),
            max(h[i - 2], h[i - 1], h[i]),
        )

    if not (_is_bear(o[i - 2], c[i - 2]) and _is_bear(o[i - 1], c[i - 1]) and _is_bear(o[i], c[i])):
        return None
    crows = c[i] < c[i - 1] < c[i - 2]
    crows = crows and all(_body(o[j], c[j]) >= 0.55 * _bar_range(h[j], l[j]) for j in (i - 2, i - 1, i))
    crows = crows and all(_body(o[j], c[j]) >= 0.5 * atr[i] for j in (i - 2, i - 1, i))
    if crows:
        return CandlePattern(
            "three_black_crows",
            "三烏鴉",
            "short",
            i - 2,
            i,
            min(l[i - 2], l[i - 1], l[i]),
            max(h[i - 2], h[i - 1], h[i]),
        )
    return None


def _range_breakout_at(
    i: int,
    o,
    h,
    l,
    c,
    v,
    atr,
    vol_sma,
    *,
    lookback: int = 60,
    min_lookback: int = 30,
    max_range_pct: float = 0.04,
    vol_mult: float = 1.8,
) -> CandlePattern | None:
    """盤整區間後，放量長紅收盤站上區間高。對應南亞科 390–405 後垂直拉升。"""
    if i < min_lookback + 5:
        return None
    left = max(0, i - lookback)
    right = i  # 不含當根
    if right - left < min_lookback:
        return None
    if not _is_bull(o[i], c[i]):
        return None

    rng_high = float(np.max(h[left:right]))
    rng_low = float(np.min(l[left:right]))
    mid = (rng_high + rng_low) / 2
    if mid <= 0:
        return None
    width = (rng_high - rng_low) / mid
    if width <= 0.006 or width > max_range_pct:
        return None

    insides = np.sum((h[left:right] <= rng_high) & (l[left:right] >= rng_low))
    if insides < 0.9 * (right - left):
        return None

    # 區間內不要已經接近上限磨很久（要有被壓住再突破的感覺）
    closes = c[left:right]
    if np.mean(closes > rng_low + 0.85 * (rng_high - rng_low)) > 0.45:
        return None

    if c[i] <= rng_high:
        return None
    body = _body(o[i], c[i])
    bar_rng = _bar_range(h[i], l[i])
    if bar_rng <= 0 or body / bar_rng < 0.55:
        return None
    if body < 0.8 * atr[i]:
        return None
    if np.isnan(vol_sma[i]) or vol_sma[i] <= 0 or v[i] < vol_mult * vol_sma[i]:
        return None
    if c[i] - rng_high < 0.12 * atr[i]:
        return None

    return CandlePattern(
        "range_breakout",
        "盤整放量突破",
        "long",
        left,
        i,
        rng_low,
        rng_high,
    )


PRIORITY = {
    "range_breakout": 0,
    "three_white_soldiers": 1,
    "three_black_crows": 1,
    "morning_star": 2,
    "evening_star": 2,
    "bull_engulfing": 3,
    "bear_engulfing": 3,
    "piercing": 4,
    "dark_cloud": 4,
    "hammer": 5,
    "shooting_star": 5,
    "tweezer_bottom": 6,
    "tweezer_top": 6,
    "bull_marubozu": 7,
    "bear_marubozu": 7,
}


def pick_best_pattern(patterns: list[CandlePattern]) -> CandlePattern | None:
    """同一根收盤只留優先序最高的一個型態。"""
    if not patterns:
        return None
    return sorted(patterns, key=lambda p: (PRIORITY.get(p.name, 99), -p.bars))[0]
