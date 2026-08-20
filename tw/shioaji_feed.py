"""永豐 Shioaji：一分 K 歷史查詢 + 盤中 tick 合成 K。金鑰只從環境變數讀。"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

TAIPEI = ZoneInfo("Asia/Taipei")
KBARS_GAP_SEC = 0.25  # 低於官方 50 次／10 秒
SUBSCRIBE_LIMIT = 200

_api = None
_api_lock = threading.Lock()
_frames: dict[str, pd.DataFrame] = {}
_frames_lock = threading.Lock()
_frame_ranges: dict[str, tuple[date, date]] = {}
_open_bars: dict[str, dict] = {}
_subscribed: set[str] = set()
_callback_bound = False


def configured() -> bool:
    return bool(
        os.environ.get("SHIOAJI_API_KEY", "").strip()
        and os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    )


def yahoo_symbol_to_code(symbol: str) -> str:
    return str(symbol).split(".", 1)[0].strip()


def kbars_to_frame(kbars: object) -> pd.DataFrame:
    """把 Shioaji Kbars 轉成與 Yahoo 相同的 OHLCV DataFrame。"""
    if kbars is None:
        return _empty()
    if isinstance(kbars, pd.DataFrame):
        raw = kbars
    elif hasattr(kbars, "dict"):
        raw = pd.DataFrame(kbars.dict())
    else:
        try:
            raw = pd.DataFrame({**kbars})
        except TypeError:
            return _empty()
    if raw is None or raw.empty:
        return _empty()
    cols = {str(c).lower(): c for c in raw.columns}
    ts_col = cols.get("ts") or cols.get("datetime")
    if ts_col is None:
        return _empty()
    rename = {}
    for name in ("open", "high", "low", "close", "volume"):
        if name in cols:
            rename[cols[name]] = name
    work = raw.rename(columns=rename)
    if any(col not in work.columns for col in ("open", "high", "low", "close")):
        return _empty()
    if "volume" not in work.columns:
        work["volume"] = 0.0
    index = pd.DatetimeIndex(pd.to_datetime(work[ts_col]))
    if index.tz is None:
        index = index.tz_localize(TAIPEI)
    else:
        index = index.tz_convert(TAIPEI)
    out = work[["open", "high", "low", "close", "volume"]].copy()
    out.index = index
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def concat_daily_frames(parts: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [p for p in parts if p is not None and not p.empty]
    if not nonempty:
        return _empty()
    frame = pd.concat(nonempty).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    out = (
        df.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
    )
    return out.astype(float)


def minute_of_tick(ts: pd.Timestamp) -> pd.Timestamp:
    """Shioaji 一分K 標在該分鐘結束（09:00:08 → 09:01）。"""
    mark = pd.Timestamp(ts)
    if mark.tzinfo is None:
        mark = mark.tz_localize(TAIPEI)
    else:
        mark = mark.tz_convert(TAIPEI)
    if mark.second == 0 and mark.microsecond == 0 and mark.nanosecond == 0:
        return mark
    return mark.ceil("min")


def apply_tick(
    open_bars: dict[str, dict],
    frames: dict[str, pd.DataFrame],
    *,
    code: str,
    price: float,
    volume: float,
    ts: pd.Timestamp,
) -> pd.Timestamp | None:
    """把一筆成交寫進當根 K；換分鐘時把上一根收進 frames。回傳剛收完的 K 時間。"""
    bar_ts = minute_of_tick(ts)
    closed_ts = None
    current = open_bars.get(code)
    if current is not None and current["ts"] != bar_ts:
        closed_ts = current["ts"]
        _append_bar(frames, code, current)
    if current is None or current["ts"] != bar_ts:
        open_bars[code] = {
            "ts": bar_ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
        }
    else:
        current["high"] = max(current["high"], price)
        current["low"] = min(current["low"], price)
        current["close"] = price
        current["volume"] += volume
    return closed_ts


def drop_incomplete_last(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    if df is None or len(df) < 2:
        return df if df is not None else _empty()
    last = pd.Timestamp(df.index[-1])
    now = pd.Timestamp(datetime.now(TAIPEI))
    freq = "5min" if interval == "5m" else "min"
    if last.floor(freq) >= now.floor(freq):
        return df.iloc[:-1]
    return df


def fetch_bars_many(
    symbols: list[str],
    interval: str = "1m",
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
) -> dict[str, pd.DataFrame]:
    api = login()
    start_d, end_d = _window(range_, start, end)
    unique = list(dict.fromkeys(s for s in symbols if s))
    if interval == "1m":
        print(f"下載一分K：{len(unique)} 檔（第一次較久，不是當機）…", flush=True)
    out: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(unique):
        cached = _peek_1m(symbol, start_d, end_d)
        if cached is None and interval == "1m":
            print(f"  {i + 1}/{len(unique)} {symbol}", flush=True)
        frame_1m = cached if cached is not None else _one_minute(api, symbol, start_d, end_d)
        if frame_1m.empty:
            continue
        frame = resample_ohlcv(frame_1m, "5min") if interval == "5m" else frame_1m
        if closed_only:
            frame = drop_incomplete_last(frame, interval=interval)
        if not frame.empty:
            out[symbol] = frame
        if cached is None and i + 1 < len(unique):
            time.sleep(KBARS_GAP_SEC)
    return out


def login():
    global _api, _callback_bound
    if not configured():
        raise RuntimeError("未設定 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY")
    with _api_lock:
        if _api is not None:
            return _api
        import shioaji as sj

        print("永豐登入中（第一次下載商品合約會等 1～2 分鐘）…", flush=True)
        api = sj.Shioaji()
        _sj_login(api)
        print("永豐登入完成。", flush=True)
        if not _callback_bound:
            _bind_tick_callback(api)
            _callback_bound = True
        _api = api
        return api


def _sj_login(api) -> None:
    key = os.environ["SHIOAJI_API_KEY"].strip()
    secret = os.environ["SHIOAJI_SECRET_KEY"].strip()
    try:
        api.login(api_key=key, secret_key=secret, fetch_contract=True)
        return
    except TypeError:
        pass
    api.login(api_key=key, secret_key=secret)
    for name in ("fetch_contracts", "fetch_contract"):
        fn = getattr(api, name, None)
        if not callable(fn):
            continue
        try:
            fn()
        except TypeError:
            try:
                fn(contract_download=True)
            except Exception:  # noqa: BLE001
                pass
        break


def subscribe_symbols(symbols: list[str]) -> list[str]:
    """盤中訂閱成交；回傳成功的 Yahoo 代號。超過 200 檔會截斷。"""
    api = login()
    ok: list[str] = []
    for symbol in symbols[:SUBSCRIBE_LIMIT]:
        code = yahoo_symbol_to_code(symbol)
        if symbol in _subscribed:
            ok.append(symbol)
            continue
        contract = _contract(api, code)
        if contract is None:
            continue
        try:
            api.quote.subscribe(contract, quote_type="tick", version="v1")
        except TypeError:
            api.quote.subscribe(contract, quote_type="tick")
        _subscribed.add(symbol)
        ok.append(symbol)
    return ok


def logout() -> None:
    global _api, _callback_bound
    with _api_lock:
        if _api is None:
            return
        try:
            _api.logout()
        except Exception:  # noqa: BLE001
            pass
        _api = None
        _callback_bound = False
    with _frames_lock:
        _frames.clear()
        _frame_ranges.clear()
        _open_bars.clear()
        _subscribed.clear()


def _peek_1m(symbol: str, start_d: date, end_d: date) -> pd.DataFrame | None:
    with _frames_lock:
        cached = _frames.get(symbol)
        rng = _frame_ranges.get(symbol)
        if cached is None or cached.empty or rng is None:
            return None
        if rng[0] <= start_d and rng[1] >= end_d:
            return cached.copy()
    return None


def _one_minute(api, symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    """1K 每次最多約 270 根（一個台股交易日），按日抓再串起來。"""
    cached = _peek_1m(symbol, start_d, end_d)
    if cached is not None:
        return cached
    code = yahoo_symbol_to_code(symbol)
    contract = _contract(api, code)
    if contract is None:
        return _empty()
    today = datetime.now(TAIPEI).date()
    parts: list[pd.DataFrame] = []
    day = start_d
    first_call = True
    while day < end_d:
        if day.weekday() < 5 and day <= today:
            if not first_call:
                time.sleep(KBARS_GAP_SEC)
            first_call = False
            try:
                kbars = api.kbars(
                    contract=contract,
                    start=day.isoformat(),
                    end=day.isoformat(),
                )
                part = kbars_to_frame(kbars)
                if not part.empty:
                    parts.append(part)
            except Exception:  # noqa: BLE001
                pass
        day += timedelta(days=1)
    frame = concat_daily_frames(parts)
    with _frames_lock:
        _frames[symbol] = frame.copy()
        _frame_ranges[symbol] = (start_d, end_d)
    return frame


def _contract(api, code: str):
    getter = getattr(api, "contracts", None)
    if getter is not None and hasattr(getter, "get"):
        found = getter.get(code)
        if found is not None:
            return found
    stocks = getattr(getattr(api, "Contracts", None), "Stocks", None)
    if stocks is None:
        return None
    try:
        return stocks[code]
    except Exception:  # noqa: BLE001
        return None


def _window(
    range_: str,
    start: date | str | None,
    end: date | str | None,
) -> tuple[date, date]:
    start_d = _as_date(start)
    end_d = _as_date(end)
    today = datetime.now(TAIPEI).date()
    if start_d is None:
        days = 5
        if isinstance(range_, str) and range_.endswith("d"):
            try:
                days = max(1, int(range_[:-1]))
            except ValueError:
                days = 5
        start_d = today - timedelta(days=days)
    if end_d is None:
        end_d = today + timedelta(days=1)
    return start_d, end_d


def _as_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _bind_tick_callback(api) -> None:
    @api.on_tick_stk_v1()
    def _on_tick(exchange, tick) -> None:  # noqa: ARG001
        code = str(getattr(tick, "code", "") or "")
        price = float(getattr(tick, "close", 0) or 0)
        volume = float(getattr(tick, "volume", 0) or 0)
        raw_ts = getattr(tick, "datetime", None) or getattr(tick, "ts", None)
        if not code or price <= 0 or raw_ts is None:
            return
        ts = pd.Timestamp(raw_ts)
        symbol = _symbol_for_code(code)
        with _frames_lock:
            apply_tick(
                _open_bars,
                _frames,
                code=symbol,
                price=price,
                volume=volume,
                ts=ts,
            )


def _symbol_for_code(code: str) -> str:
    for symbol in list(_subscribed) + list(_frames):
        if yahoo_symbol_to_code(symbol) == code:
            return symbol
    return f"{code}.TW"


def _append_bar(frames: dict[str, pd.DataFrame], symbol: str, bar: dict) -> None:
    row = pd.DataFrame(
        {
            "open": [bar["open"]],
            "high": [bar["high"]],
            "low": [bar["low"]],
            "close": [bar["close"]],
            "volume": [bar["volume"]],
        },
        index=[bar["ts"]],
    )
    prev = frames.get(symbol)
    if prev is None or prev.empty:
        frames[symbol] = row
        return
    merged = pd.concat([prev, row]).sort_index()
    frames[symbol] = merged[~merged.index.duplicated(keep="last")]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
