#!/usr/bin/env python3
"""五分 K W 底後 5/10/20 多排（無網路）。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.patterns import WMa20Signal, detect_w_bottoms, detect_w_ma20_crosses  # noqa: E402
from watch_tw_w_ma20 import (  # noqa: E402
    TPE,
    TwHit,
    bounce_pct,
    drop_forming_bar,
    fill_chinese_names,
    filter_hits_days,
    filter_price_below,
    has_cjk,
    hit_key,
    is_strict_hit,
    parse_symbols,
    recent_hits,
    stand_pct,
)


def _ohlc(close: np.ndarray, low: np.ndarray | None = None, high: np.ndarray | None = None) -> pd.DataFrame:
    close = np.asarray(close, float)
    n = len(close)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    if high is None:
        high = np.maximum(open_, close) + 0.6
    if low is None:
        low = np.minimum(open_, close) - 0.6
    idx = pd.date_range("2026-08-25 09:00", periods=n, freq="5min", tz=TPE)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 1000.0)},
        index=idx,
    )


def make_yageo_like() -> pd.DataFrame:
    """國巨那種：先從 560 殺到 515，做出 W，再站上 MA20。"""
    n = 96
    close = np.full(n, 556.0)
    for i in range(1, 28):
        close[i] = 556.0 + (i % 4) * 0.3
    close[28:43] = np.linspace(554.0, 516.0, 15)
    close[43:51] = np.linspace(517.0, 528.0, 8)
    close[51:59] = np.linspace(527.0, 518.5, 8)
    close[59] = 519.0
    close[60] = 525.0  # 明確站上 MA20，對齊 521 vs 518.55
    close[61:] = np.linspace(526.0, 538.0, n - 61)
    low = np.minimum(close, np.roll(close, 1)) - 0.4
    low[0] = close[0] - 0.4
    low[42] = 515.0
    low[41] = 516.8
    low[43] = 516.6
    low[58] = 518.0
    low[57] = 519.4
    low[59] = 519.2
    high = np.maximum(close, np.roll(close, 1)) + 0.5
    high[0] = close[0] + 0.5
    high[50] = 529.5
    return _ohlc(close, low=low, high=high)


def make_yageo_plateau() -> pd.DataFrame:
    """國巨 8/25：515 尖底 + 515 平底，11:25 收盤站上 MA20。"""
    n = 70
    close = np.full(n, 556.0)
    close[:20] = 556.0
    # 09:00-09:50 殺到 515
    drop = [535, 533, 532, 532, 524, 523, 521, 525, 521, 518, 519]
    close[20:31] = drop
    # 反彈 10:10-10:35
    close[31:37] = [523, 522, 521, 521, 522, 520]
    # 第二腳平底 515
    close[37:44] = [518, 516, 516, 515, 515, 518, 516]
    # 11:20 起翻上
    close[44:] = [517, 523, 524, 522, 525, 524, 525, 525, 527, 527, 527, 529, 530, 530] + [530] * (n - 58)
    close = close[:n]
    low = np.minimum(close, np.roll(close, 1)) - 0.2
    low[0] = close[0] - 0.2
    low[20:31] = [534, 527, 528, 530, 523, 521, 521, 521, 521, 517, 515]
    low[31:37] = [517, 520, 521, 520, 520, 519]
    low[37:44] = [518, 515, 515, 515, 515, 515, 515]
    low[44:47] = [517, 517, 519]
    high = np.maximum(close, np.roll(close, 1)) + 0.4
    high[0] = close[0] + 0.4
    high[20] = 548.0
    high[31] = 523.0
    return _ohlc(close, low=low, high=high)


def make_nanya_like() -> pd.DataFrame:
    """南亞科那種：殺到 480，第二腳稍高，再站上 MA20。"""
    n = 96
    close = np.full(n, 508.0)
    for i in range(1, 26):
        close[i] = 509.0 + (i % 3) * 0.4
    close[26:40] = np.linspace(507.0, 481.0, 14)
    close[40:48] = np.linspace(482.0, 492.0, 8)
    close[48:56] = np.linspace(491.0, 483.5, 8)
    close[56] = 483.8
    close[57] = 489.5  # 明確站上 MA20，對齊 487 vs 484.43
    close[58:] = np.linspace(490.0, 504.0, n - 58)
    low = np.minimum(close, np.roll(close, 1)) - 0.35
    low[0] = close[0] - 0.35
    low[39] = 480.0
    low[38] = 481.6
    low[40] = 481.4
    low[55] = 482.8
    low[54] = 483.8
    low[56] = 483.6
    high = np.maximum(close, np.roll(close, 1)) + 0.45
    high[0] = close[0] + 0.45
    high[47] = 493.2
    return _ohlc(close, low=low, high=high)


def make_range_after_dump() -> pd.DataFrame:
    """旺宏 8/25：殺到 117.5 後立刻碰到箱頂，接著多次測同一層，不是 W。"""
    n = 90
    close = np.full(n, 126.0)
    close[:12] = 126.0
    close[12:18] = [124.0, 122.0, 120.5, 119.5, 118.5, 118.2]
    # 第一低之後馬上碰到箱頂 119.5，接著橫盤
    rest = []
    box = [119.0, 118.5, 118.0, 118.6, 119.2, 118.4, 118.0, 119.0, 118.2, 118.8]
    while len(rest) < n - 18:
        rest.extend(box)
    close[18:] = rest[: n - 18]
    close[-6:] = [119.5, 120.0, 120.8, 121.5, 122.0, 122.5]
    low = np.minimum(close, np.roll(close, 1)) - 0.15
    low[0] = close[0] - 0.15
    low[17] = 117.5
    close[17] = 118.5
    # 同一層被反覆測
    for i in (25, 33, 41, 49, 57):
        low[i] = 117.5
        close[i] = 117.8
    high = np.maximum(close, np.roll(close, 1)) + 0.2
    high[0] = close[0] + 0.2
    high[12] = 128.0
    high[18] = 119.5
    high[19] = 119.5
    return _ohlc(close, low=low, high=high)


def make_no_w_ma20_cross() -> pd.DataFrame:
    """只有單腳回升上穿 MA20，沒有 W。"""
    n = 80
    close = np.full(n, 100.0)
    close[:30] = 100.0
    close[30:45] = np.linspace(99.5, 90.0, 15)
    close[45:] = np.linspace(90.5, 102.0, n - 45)
    return _ohlc(close)


def make_w_never_cross() -> pd.DataFrame:
    """有 W 但反彈壓在均線下，沒站上 MA20。"""
    n = 90
    close = np.full(n, 200.0)
    close[:28] = 200.0
    close[28:42] = np.linspace(199.0, 180.0, 14)
    close[42:50] = np.linspace(181.0, 188.0, 8)
    close[50:58] = np.linspace(187.0, 181.0, 8)
    close[58:66] = np.linspace(181.5, 184.0, 8)
    close[66:] = np.linspace(183.5, 176.0, n - 66)
    low = np.minimum(close, np.roll(close, 1)) - 0.2
    low[0] = close[0] - 0.2
    low[41] = 179.5
    low[57] = 180.2
    return _ohlc(close, low=low)


def test_yageo_like_alerts() -> None:
    df = make_yageo_like()
    sigs = detect_w_ma20_crosses(df)
    assert sigs, "國巨型 W 底後 5/10/20 多排應有訊號"
    sig = sigs[-1]
    assert df["Low"].iloc[sig.first_low_idx] <= 516.5
    assert sig.second_low >= sig.first_low - 2
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.cross_price > sig.ma5


def test_yageo_plateau_second_bottom() -> None:
    df = make_yageo_plateau()
    sigs = detect_w_ma20_crosses(df)
    assert sigs, "515 平底第二腳也要能出訊號"
    sig = sigs[-1]
    assert abs(sig.first_low - 515.0) < 1.0
    assert abs(sig.second_low - 515.0) < 1.0
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.cross_price > sig.ma5


def test_nanya_like_higher_second_low() -> None:
    df = make_nanya_like()
    sigs = detect_w_ma20_crosses(df)
    assert sigs, "南亞科型 W 底後 5/10/20 多排應有訊號"
    sig = sigs[-1]
    assert sig.second_low >= sig.first_low
    assert abs(sig.first_low - 480.0) < 1.5
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.cross_price > sig.ma5


def test_no_w_no_alert() -> None:
    assert detect_w_ma20_crosses(make_no_w_ma20_cross()) == []


def test_range_after_dump_is_not_a_w() -> None:
    assert detect_w_ma20_crosses(make_range_after_dump()) == [], "橫盤箱型不該算 W 底"


def test_w_without_ma20_no_alert() -> None:
    assert detect_w_ma20_crosses(make_w_never_cross()) == []


def test_falling_ma_onto_flat_close_is_not_a_cross() -> None:
    """均線掉下來吻到走平收盤，不算往上穿。"""
    n = 80
    close = np.full(n, 120.0)
    close[:25] = 120.0
    close[25:38] = np.linspace(119.0, 100.0, 13)
    close[38:46] = np.linspace(101.0, 108.0, 8)
    close[46:54] = np.linspace(107.0, 101.0, 8)
    close[54:] = 102.0
    low = np.minimum(close, np.roll(close, 1)) - 0.15
    low[0] = close[0] - 0.15
    low[37] = 99.6
    low[53] = 100.4
    df = _ohlc(close, low=low)
    assert detect_w_ma20_crosses(df) == []


def test_accepts_lowercase_columns() -> None:
    df = make_yageo_like().rename(columns=str.lower)
    assert detect_w_ma20_crosses(df)


def test_existing_w_bottom_still_works() -> None:
    from run_backtest import make_sample_w_bottom_bars

    df = make_sample_w_bottom_bars()
    detect_w_bottoms(df)


def test_parse_symbols() -> None:
    rows = parse_symbols("2327,2408.TW,6488.TWO")
    assert [r["symbol"] for r in rows] == ["2327.TW", "2408.TW", "6488.TWO"]
    assert rows[2]["market"] == "otc"


def test_drop_forming_bar() -> None:
    df = make_yageo_like()
    last = df.index[-1]
    during = last + pd.Timedelta(minutes=2)
    trimmed = drop_forming_bar(df, now=during.to_pydatetime())
    assert len(trimmed) == len(df) - 1
    closed = drop_forming_bar(df, now=(last + pd.Timedelta(minutes=5, seconds=1)).to_pydatetime())
    assert len(closed) == len(df)
    messy = df.copy()
    messy.index = messy.index[:-1].append(pd.DatetimeIndex([last + pd.Timedelta(minutes=2, seconds=9)]))
    dropped = drop_forming_bar(messy, now=(last + pd.Timedelta(minutes=10)).to_pydatetime())
    assert len(dropped) == len(df) - 1


def test_hit_key_and_recent() -> None:
    df = make_yageo_like()
    sigs = detect_w_ma20_crosses(df)
    hit = type("H", (), {"row": {"symbol": "2327.TW"}, "signal": sigs[-1], "df": df})()
    key = hit_key(hit)
    assert key.startswith("2327.TW|")
    old = recent_hits([hit], lookback_hours=0.01)
    assert old == [] or df.index[sigs[-1].cross_idx] >= datetime.now(TPE) - pd.Timedelta(hours=0.01)


def test_filter_price_below_700() -> None:
    rows = [
        {"code": "2327", "close": 537.0},
        {"code": "2408", "close": 503.0},
        {"code": "2330", "close": 1400.0},
        {"code": "2454", "close": 700.0},
        {"code": "2303", "close": 55.0},
        {"code": "3008", "close": 2500.0},
    ]
    kept, dropped = filter_price_below(rows, 700.0, 100)
    assert [r["code"] for r in kept] == ["2327", "2408", "2303"]
    assert {r["code"] for r in dropped} == {"2330", "2454", "3008"}
    assert kept[0]["rank"] == 1


def test_has_cjk_and_chinese_names() -> None:
    import watch_tw_w_ma20 as mod

    assert has_cjk("國巨")
    assert has_cjk("南亞科")
    assert not has_cjk("YAGEO CORP")
    assert not has_cjk("2327")
    mod._NAME_MAP.clear()
    mod._NAME_MAP.update({"2327": "國巨", "2408": "南亞科"})
    mod._NAME_MAP_LOADED = True
    rows = [
        {"code": "2327", "name": "YAGEO CORP"},
        {"code": "2408", "name": "2408"},
        {"code": "2303", "name": "聯電"},
    ]
    out = fill_chinese_names(rows)
    assert [r["name"] for r in out] == ["國巨", "南亞科", "聯電"]


def _fake_hit(code: str, name: str, when: str, *, bounce: float = 2.0, stand: float = 0.4) -> TwHit:
    ts = pd.Timestamp(when, tz=TPE)
    start = ts.normalize() + pd.Timedelta(hours=9)
    idx = pd.date_range(start, periods=54, freq="5min", tz=TPE)
    close = np.linspace(100, 110, len(idx))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    loc = int(df.index.get_indexer([ts], method="nearest")[0])
    df.iloc[max(0, loc - 20), df.columns.get_loc("High")] = 108.0
    base = 100.0
    cross = base * (1 + bounce / 100.0)
    ma20 = cross / (1 + stand / 100.0)
    ma5 = cross * 0.999
    if ma5 <= ma20:
        ma5 = (cross + ma20) / 2.0
    ma10 = (ma5 + ma20) / 2.0
    sig = WMa20Signal(
        first_low_idx=max(0, loc - 10),
        second_low_idx=max(0, loc - 4),
        neckline_idx=max(0, loc - 7),
        first_low=base,
        second_low=base,
        neckline=base * 1.02,
        cross_idx=loc,
        cross_price=cross,
        ma5=ma5,
        ma10=ma10,
        ma20=ma20,
    )
    row = {"code": code, "name": name, "symbol": f"{code}.TW"}
    return TwHit(row, sig, df)


def test_filter_hits_days_one_week() -> None:
    now = datetime(2026, 8, 25, 15, 0, tzinfo=TPE)
    hits = [
        _fake_hit("2327", "國巨", "2026-08-18 11:25"),
        _fake_hit("2408", "南亞科", "2026-08-19 10:00"),
        _fake_hit("1303", "南亞", "2026-08-25 11:30"),
    ]
    week = filter_hits_days(hits, 7, now=now)
    assert [h.row["code"] for h in week] == ["2408", "1303"]
    today = filter_hits_days(hits, 1, now=now)
    assert [h.row["code"] for h in today] == ["1303"]


def test_strict_keeps_broker_examples() -> None:
    from watch_tw_w_ma20 import depth_pct, prior_drop_pct

    cases = (
        (make_yageo_plateau(), "2327", "國巨"),
        (make_nanya_like(), "2408", "南亞科"),
        (make_yageo_like(), "2327", "國巨"),
    )
    for df, code, name in cases:
        sigs = detect_w_ma20_crosses(df)
        assert sigs, name
        hit = TwHit({"code": code, "name": name, "symbol": f"{code}.TW"}, sigs[-1], df)
        assert is_strict_hit(hit), (
            f"{name} bounce={bounce_pct(hit.signal):.2f} stand={stand_pct(hit.signal):.2f} "
            f"depth={depth_pct(hit.signal):.2f} drop={prior_drop_pct(hit):.2f} "
            f"ts={df.index[hit.signal.cross_idx]}"
        )


def test_strict_rejects_open_gap_and_weak_kiss() -> None:
    open_gap = _fake_hit("2408", "南亞科", "2026-08-25 09:00", bounce=2.0, stand=1.0)
    weak = _fake_hit("2327", "國巨", "2026-08-25 11:25", bounce=0.4, stand=0.1)
    huge = _fake_hit("8039", "台虹", "2026-08-25 10:00", bounce=4.0, stand=1.0)
    assert not is_strict_hit(open_gap)
    assert not is_strict_hit(weak)
    assert not is_strict_hit(huge)


def test_select_chart_hits_picks_strict() -> None:
    from watch_tw_w_ma20 import select_chart_hits

    hits = [
        _fake_hit("2327", "國巨", "2026-08-25 11:25", bounce=1.17, stand=0.47),
        _fake_hit("2408", "南亞科", "2026-08-25 11:30", bounce=1.46, stand=0.53),
        _fake_hit("2303", "聯電", "2026-08-25 11:00", bounce=0.3, stand=0.05),
    ]
    picked = select_chart_hits(hits, per_day=7)
    codes = [h.row["code"] for h in picked]
    assert codes == ["2327", "2408"]


def main() -> int:
    test_yageo_like_alerts()
    test_yageo_plateau_second_bottom()
    test_nanya_like_higher_second_low()
    test_no_w_no_alert()
    test_range_after_dump_is_not_a_w()
    test_w_without_ma20_no_alert()
    test_falling_ma_onto_flat_close_is_not_a_cross()
    test_accepts_lowercase_columns()
    test_existing_w_bottom_still_works()
    test_parse_symbols()
    test_drop_forming_bar()
    test_hit_key_and_recent()
    test_filter_price_below_700()
    test_has_cjk_and_chinese_names()
    test_filter_hits_days_one_week()
    test_strict_keeps_broker_examples()
    test_strict_rejects_open_gap_and_weak_kiss()
    test_select_chart_hits_picks_strict()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
