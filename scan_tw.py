#!/usr/bin/env python3
"""台股一分K掃描（單檔）。PyCharm 只要這一個檔，檔名不要叫 tw.py。

最上面四個引號填序號，按 Run 就是盤中隨時監控。只掃一次加 --once。
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ════ 在下面引號裡填序號（只放你自己電腦，不要貼聊天室）════
SHIOAJI_API_KEY = ""        # 永豐 API Key
SHIOAJI_SECRET_KEY = ""     # 永豐 Secret
TELEGRAM_BOT_TOKEN = ""     # Telegram BotFather 給的 token
TELEGRAM_CHAT_ID = ""       # 你的 Telegram chat id
# ═══════════════════════════════════════════════════════════════

SCRIPT_VERSION = "2026-08-24-a"


def _apply_secrets() -> None:
    for name, value in (
        ("SHIOAJI_API_KEY", SHIOAJI_API_KEY),
        ("SHIOAJI_SECRET_KEY", SHIOAJI_SECRET_KEY),
        ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
    ):
        text = str(value).strip()
        if text:
            os.environ[name] = text


_apply_secrets()

TAIPEI = ZoneInfo("Asia/Taipei")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*api\.Contracts is deprecated.*",
)


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 240
# 09:00–09:04 開盤跳空不算「當下那根站上 240」。
ENTRY_AFTER_HOUR = 9
ENTRY_AFTER_MINUTE = 5
# 站穩那根要過開盤前 10 分鐘（09:06 那種開盤第一根不算）。
CONFIRM_AFTER_MINUTE = 10
MAX_BAR_GAP = pd.Timedelta(minutes=2)
HOLD_BARS = 2
# 金叉前連續收在 MA240 下，才算從下面穿上，不是貼著磨。
BARS_BELOW_BEFORE_CROSS = 3


@dataclass(frozen=True)
class AlertSnapshot:
    timestamp: pd.Timestamp
    close: float
    prev_close: float
    ma5: float
    ma10: float
    ma20: float
    ma240: float
    prev_ma240: float

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def crossed_above_ma240(self) -> bool:
        return self.close > self.ma240 and self.prev_close <= self.prev_ma240

    @property
    def ma_span_pct(self) -> float:
        """MA5 與 MA20 的距離／收盤。太小代表均線糾結。"""
        if self.close <= 0:
            return 0.0
        return (self.ma5 - self.ma20) / self.close

    @property
    def ma20_ma240_gap_pct(self) -> float:
        """MA20 與 MA240 的距離／收盤。太大代表兩條線差太遠（像華新科）。"""
        if self.close <= 0:
            return 0.0
        return abs(self.ma240 - self.ma20) / self.close


def is_intraday_entry_bar(prev_ts: pd.Timestamp, ts: pd.Timestamp) -> bool:
    """同一根進場：前一根必須是同一交易日、連續的一分K，且已過開盤跳空時段。"""
    prev = pd.Timestamp(prev_ts)
    cur = pd.Timestamp(ts)
    if prev.tzinfo is not None and cur.tzinfo is None:
        cur = cur.tz_localize(prev.tzinfo)
    elif cur.tzinfo is not None and prev.tzinfo is None:
        prev = prev.tz_localize(cur.tzinfo)
    if cur.date() != prev.date():
        return False
    if cur.hour < ENTRY_AFTER_HOUR or (
        cur.hour == ENTRY_AFTER_HOUR and cur.minute < ENTRY_AFTER_MINUTE
    ):
        return False
    if cur - prev > MAX_BAR_GAP:
        return False
    return True


def mas_are_open(snapshot: AlertSnapshot, min_span: float = 0.004) -> bool:
    """多頭排列且 MA5–MA20 拉開到 min_span 以上（預設 0.4%，短均要扇開）。"""
    return snapshot.bullish_aligned and snapshot.ma_span_pct >= min_span


def ma20_near_ma240(
    snapshot: AlertSnapshot,
    max_gap: float = 0.010,
    min_gap: float = 0.004,
) -> bool:
    """MA20 與 MA240 距離要剛好（預設 0.4%～1.0%）。太近是貼著磨，太遠不像踩線。"""
    gap = snapshot.ma20_ma240_gap_pct
    return min_gap <= gap <= max_gap


def is_confirm_time(ts: pd.Timestamp) -> bool:
    """站穩通知那一根：09:10 以後。"""
    cur = pd.Timestamp(ts)
    if cur.hour < ENTRY_AFTER_HOUR:
        return False
    if cur.hour == ENTRY_AFTER_HOUR and cur.minute < CONFIRM_AFTER_MINUTE:
        return False
    return True


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma240"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def is_ma240_breakout_bullish(df: pd.DataFrame) -> AlertSnapshot | None:
    """最新一根：前一根剛站上 MA240，這一根仍收在 MA240 上（站穩兩根）。"""
    return latest_ma240_breakout_bullish(df, latest_only=True)


def latest_ma240_breakout_bullish(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    latest_only: bool = False,
    min_ma_span: float = 0.0,
    min_ma20_ma240_gap: float = 0.0,
    max_ma20_ma240_gap: float | None = None,
) -> AlertSnapshot | None:
    """
    進場／通知那一根：從 MA240 下面穿上，連續兩根收盤站穩。
    站穩要 09:10 以後；金叉前至少三根收在 240 下。
    latest_only 只看最後一根（watch）。隔夜跳空、開盤前 5 分鐘不算。
    """
    if df is None or len(df) < MA_LONG + HOLD_BARS + BARS_BELOW_BEFORE_CROSS:
        return None
    work = add_moving_averages(df)
    if latest_only:
        loc = len(work) - 1
        snap = _hold_confirm_at(
            work,
            loc,
            min_ma_span=min_ma_span,
            min_ma20_ma240_gap=min_ma20_ma240_gap,
            max_ma20_ma240_gap=max_ma20_ma240_gap,
        )
        if snap is None:
            return None
        if since is not None and snap.timestamp < since:
            return None
        if until is not None and snap.timestamp > until:
            return None
        return snap

    start = MA_LONG + HOLD_BARS + BARS_BELOW_BEFORE_CROSS - 1
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(start, i)
                matched = True
                break
        if not matched:
            return None

    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _hold_confirm_at(
            work,
            i,
            min_ma_span=min_ma_span,
            min_ma20_ma240_gap=min_ma20_ma240_gap,
            max_ma20_ma240_gap=max_ma20_ma240_gap,
        )
        if snap is not None:
            return snap
    return None


def _hold_confirm_at(
    work: pd.DataFrame,
    idx: int,
    *,
    min_ma_span: float = 0.0,
    min_ma20_ma240_gap: float = 0.0,
    max_ma20_ma240_gap: float | None = None,
) -> AlertSnapshot | None:
    """idx 為站穩的第二根；前一根必須是剛站上 240 的進場K。"""
    if idx < 2:
        return None
    if not is_confirm_time(work.index[idx]):
        return None
    if not is_intraday_entry_bar(work.index[idx - 1], work.index[idx]):
        return None
    if not is_intraday_entry_bar(work.index[idx - 2], work.index[idx - 1]):
        return None
    if not _run_below_ma240(work, idx - 1):
        return None
    confirm = _snapshot_at(work, idx)
    cross = _snapshot_at(work, idx - 1)
    if confirm is None or cross is None:
        return None
    if not cross.crossed_above_ma240:
        return None
    if not (confirm.close > confirm.ma240):
        return None
    if not (confirm.bullish_aligned and cross.bullish_aligned):
        return None
    if min_ma_span and not mas_are_open(confirm, min_ma_span):
        return None
    if max_ma20_ma240_gap is not None or min_ma20_ma240_gap > 0:
        max_gap = 1.0 if max_ma20_ma240_gap is None else max_ma20_ma240_gap
        if not ma20_near_ma240(
            confirm, max_gap=max_gap, min_gap=min_ma20_ma240_gap
        ):
            return None
    return confirm


def _run_below_ma240(
    work: pd.DataFrame,
    cross_idx: int,
    bars: int = BARS_BELOW_BEFORE_CROSS,
) -> bool:
    """金叉前連續 bars 根收盤都在 MA240 下（含等於），同一交易日。"""
    if cross_idx < bars:
        return False
    cross_day = pd.Timestamp(work.index[cross_idx]).date()
    for i in range(cross_idx - bars, cross_idx):
        ts = pd.Timestamp(work.index[i])
        if ts.date() != cross_day:
            return False
        row = work.iloc[i]
        if pd.isna(row.get("ma240")):
            return False
        if float(row["close"]) > float(row["ma240"]):
            return False
    return True


def ma240_at(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> tuple[float, float] | None:
    """回傳指定時間（或最新一根）的收盤與 MA240；資料不足則 None。"""
    if df is None or df.empty:
        return None
    work = add_moving_averages(df)
    if ts is None:
        loc = len(work) - 1
    else:
        mark = pd.Timestamp(ts)
        if floor:
            mark = mark.floor(floor)
        loc = int(work.index.get_indexer([mark], method="nearest")[0])
        if loc < 0:
            return None
    row = work.iloc[loc]
    if pd.isna(row.get("ma240")):
        return None
    return float(row["close"]), float(row["ma240"])


def ma240_gap_pct(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> float | None:
    """(收盤 − MA240) / MA240。資料不足或 MA240 ≤ 0 則 None。"""
    pair = ma240_at(df, ts, floor=floor)
    if pair is None or pair[1] <= 0:
        return None
    return (pair[0] - pair[1]) / pair[1]


def close_above_ma240(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
    min_gap: float = 0.0,
) -> bool:
    gap = ma240_gap_pct(df, ts, floor=floor)
    return gap is not None and gap >= min_gap and gap > 0


def _snapshot_at(work: pd.DataFrame, idx: int) -> AlertSnapshot | None:
    if idx < 1:
        return None
    last = work.iloc[idx]
    prev = work.iloc[idx - 1]
    needed = ("ma5", "ma10", "ma20", "ma240")
    if any(pd.isna(last[col]) or pd.isna(prev[col]) for col in needed):
        return None
    return AlertSnapshot(
        timestamp=work.index[idx],
        close=float(last["close"]),
        prev_close=float(prev["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma240=float(last["ma240"]),
        prev_ma240=float(prev["ma240"]),
    )


TWSE_MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_QUOTES_URL = (
    "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php"
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


@dataclass(frozen=True)
class RankedStock:
    rank: int
    symbol: str
    name: str
    price: float
    change: float | None
    change_percent: float | None
    volume_lots: int | None
    turnover: float
    exchange: str

    @property
    def code(self) -> str:
        return self.symbol.split(".", 1)[0]


def fetch_turnover_ranking(
    top: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
    as_of: date | None = None,
) -> tuple[list[RankedStock], str | None]:
    """抓最近一個已公布交易日的成交金額前 N 名（證交所／櫃買，不走永豐快照）。"""
    sess = session or requests.Session()
    start = as_of or date.today()
    last_error: Exception | None = None
    for session_day in iter_recent_sessions(start, limit=10):
        try:
            stocks, label = fetch_daily_turnover_ranking(
                session_day, top=top, session=sess, timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if len(stocks) < 50:
            last_error = ValueError(f"{session_day.isoformat()} 成交額名單只有 {len(stocks)} 檔")
            continue
        if session_day != start:
            label = f"{session_day.isoformat()} 證交所/櫃買成交額（{start.isoformat()} 尚未公布）"
        print(f"成交額名單：{label} {len(stocks)} 檔", flush=True)
        return stocks, label
    detail = f"（{last_error}）" if last_error else ""
    raise RuntimeError(f"成交額排行抓不到（證交所/櫃買）{detail}")


def iter_recent_sessions(start: date, limit: int = 10):
    current = start
    found = 0
    while found < limit:
        if current.weekday() < 5:
            yield current
            found += 1
        current -= timedelta(days=1)


def previous_friday(today: date | None = None) -> date:
    """回傳「上週五」（若今天是週五則回上一個週五）。"""
    current = today or date.today()
    days = current.weekday() - 4
    if days <= 0:
        days += 7
    return current - timedelta(days=days)


def previous_weekdays(today: date | None = None, weeks: int = 1) -> list[date]:
    """過去 N 個完整週的週一到週五（不含本週）。weeks=1 為上週，weeks=2 為上上週＋上週。"""
    if weeks < 1:
        raise ValueError("weeks must be >= 1")
    current = today or date.today()
    this_monday = current - timedelta(days=current.weekday())
    days: list[date] = []
    for week in range(weeks, 0, -1):
        monday = this_monday - timedelta(days=7 * week)
        days.extend(monday + timedelta(days=i) for i in range(5))
    return days


def fetch_daily_turnover_ranking(
    on_date: date,
    top: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> tuple[list[RankedStock], str | None]:
    """上市＋上櫃當日成交金額排行（盤後）。一邊失敗仍用另一邊。"""
    sess = session or requests.Session()
    stocks: list[RankedStock] = []
    errors: list[str] = []
    try:
        stocks.extend(_fetch_twse_daily(on_date, sess, timeout))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"上市 {exc}")
    try:
        stocks.extend(_fetch_tpex_daily(on_date, sess, timeout))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"上櫃 {exc}")
    if len(stocks) < 30:
        detail = "；".join(errors) or "資料不足"
        raise ValueError(f"{on_date.isoformat()} 盤後成交額不可用：{detail}")
    stocks.sort(key=lambda s: s.turnover, reverse=True)
    ranked = [
        RankedStock(
            rank=i,
            symbol=s.symbol,
            name=s.name,
            price=s.price,
            change=s.change,
            change_percent=s.change_percent,
            volume_lots=s.volume_lots,
            turnover=s.turnover,
            exchange=s.exchange,
        )
        for i, s in enumerate(stocks, 1)
    ]
    return ranked[:top], f"{on_date.isoformat()} 盤後成交額"


def parse_twse_mi_index(payload: dict, exchange: str = "TAI") -> list[RankedStock]:
    table = _table_with_fields(payload, need=("證券代號", "成交金額", "收盤價"))
    fields = [str(f).strip() for f in table.get("fields") or []]
    idx = {name: i for i, name in enumerate(fields)}
    stocks: list[RankedStock] = []
    for row in table.get("data") or []:
        stock = _row_to_stock(
            row,
            code_i=idx["證券代號"],
            name_i=idx["證券名稱"],
            close_i=idx["收盤價"],
            turnover_i=idx["成交金額"],
            volume_i=idx.get("成交股數"),
            change_i=idx.get("漲跌價差"),
            sign_i=idx.get("漲跌(+/-)"),
            suffix=".TW",
            exchange=exchange,
        )
        if stock is not None:
            stocks.append(stock)
    return stocks


def parse_tpex_quotes(payload: dict) -> list[RankedStock]:
    table = _table_with_fields(payload, need=("代號", "成交金額"))
    fields = [re.sub(r"<[^>]+>", "", str(f)).strip() for f in table.get("fields") or []]
    idx = {name: i for i, name in enumerate(fields)}
    close_key = next((k for k in idx if k.startswith("收盤")), None)
    turn_key = next((k for k in idx if "成交金額" in k), None)
    vol_key = next((k for k in idx if "成交股數" in k), None)
    if close_key is None or turn_key is None:
        raise ValueError("上櫃收盤行情欄位異常")
    stocks: list[RankedStock] = []
    for row in table.get("data") or []:
        stock = _row_to_stock(
            row,
            code_i=idx["代號"],
            name_i=idx["名稱"],
            close_i=idx[close_key],
            turnover_i=idx[turn_key],
            volume_i=idx.get(vol_key) if vol_key else None,
            change_i=idx.get("漲跌"),
            sign_i=None,
            suffix=".TWO",
            exchange="TWO",
        )
        if stock is not None:
            stocks.append(stock)
    return stocks


def _fetch_twse_daily(on_date: date, sess: requests.Session, timeout: int) -> list[RankedStock]:
    resp = sess.get(
        TWSE_MI_INDEX_URL,
        params={"date": on_date.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"},
        headers=DEFAULT_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("stat", "")).upper() != "OK":
        raise ValueError(f"上市盤後資料不可用：{payload.get('stat')}")
    return parse_twse_mi_index(payload)


def _fetch_tpex_daily(on_date: date, sess: requests.Session, timeout: int) -> list[RankedStock]:
    roc = f"{on_date.year - 1911}/{on_date.strftime('%m/%d')}"
    resp = sess.get(
        TPEX_QUOTES_URL,
        params={"l": "zh-tw", "d": roc, "se": "EW"},
        headers=DEFAULT_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("stat", "")).lower() not in {"ok", "success"}:
        raise ValueError(f"上櫃盤後資料不可用：{payload.get('stat')}")
    return parse_tpex_quotes(payload)


def _table_with_fields(payload: dict, need: tuple[str, ...]) -> dict:
    for table in payload.get("tables") or []:
        fields = [re.sub(r"<[^>]+>", "", str(f)).strip() for f in (table.get("fields") or [])]
        if all(any(req in field for field in fields) for req in need):
            return table
    raise ValueError(f"找不到欄位 {need} 的行情表")


def _row_to_stock(
    row: list,
    *,
    code_i: int,
    name_i: int,
    close_i: int,
    turnover_i: int,
    volume_i: int | None,
    change_i: int | None,
    sign_i: int | None,
    suffix: str,
    exchange: str,
) -> RankedStock | None:
    if not isinstance(row, (list, tuple)) or len(row) <= max(code_i, name_i, close_i, turnover_i):
        return None
    code = str(row[code_i]).strip()
    if not code:
        return None
    price = _to_float(row[close_i])
    turnover = _to_float(row[turnover_i])
    if price is None or turnover is None:
        return None
    change = _to_float(row[change_i]) if change_i is not None and change_i < len(row) else None
    if change is not None and sign_i is not None and sign_i < len(row):
        sign_html = str(row[sign_i])
        if "color:green" in sign_html or re.search(r">\s*-", sign_html):
            change = -abs(change)
        elif "color:red" in sign_html or "+" in sign_html:
            change = abs(change)
    change_percent = None
    if change is not None:
        prev = price - change
        if prev:
            change_percent = change / prev * 100.0
    volume_shares = _to_float(row[volume_i]) if volume_i is not None and volume_i < len(row) else None
    volume_lots = int(volume_shares / 1000) if volume_shares is not None else None
    return RankedStock(
        rank=0,
        symbol=f"{code}{suffix}" if "." not in code else code,
        name=str(row[name_i]).strip() or code,
        price=price,
        change=change,
        change_percent=change_percent,
        volume_lots=volume_lots,
        turnover=turnover,
        exchange=exchange,
    )


def filter_by_price(stocks: list[RankedStock], max_price: float) -> list[RankedStock]:
    """濾掉股價達 max_price 以上（含）的標的。"""
    return [s for s in stocks if s.price < max_price]


def is_etf(stock: RankedStock) -> bool:
    """台股 ETF / ETN：代號 00、02 開頭、含英文字，或名稱帶 ETF/槓桿反向。"""
    code = stock.code.upper()
    if code.startswith(("00", "02")):
        return True
    if any(ch.isalpha() for ch in code):
        return True
    name = stock.name.upper()
    markers = ("ETF", "ETN", "正2", "反1", "主動")
    return any(mark in stock.name or mark in name for mark in markers)


def filter_etfs(stocks: list[RankedStock]) -> list[RankedStock]:
    return [s for s in stocks if not is_etf(s)]


_FINANCIAL_CODE_PREFIXES = ("28", "58")
_FINANCIAL_NAME_MARKERS = ("銀行", "金控", "保險", "產險", "壽險", "證券", "票券", "期貨")
_FINANCIAL_NAME_SUFFIXES = ("金", "銀", "證", "保", "壽", "票", "期")


def is_financial(stock: RankedStock) -> bool:
    """金控、銀行、保險、證券、票券、期貨、租賃（中租）。不把金居、金寶當金融股。"""
    code = stock.code
    if len(code) >= 2 and code[:2] in _FINANCIAL_CODE_PREFIXES:
        return True
    name = re.sub(r"[-*].*$", "", stock.name).strip()
    if any(mark in name for mark in _FINANCIAL_NAME_MARKERS):
        return True
    return any(name.endswith(suf) for suf in _FINANCIAL_NAME_SUFFIXES)


def filter_financials(stocks: list[RankedStock]) -> list[RankedStock]:
    return [s for s in stocks if not is_financial(s)]


def _exchange_of(symbol: str) -> str:
    if symbol.endswith(".TWO"):
        return "TWO"
    return "TAI"


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").replace("+", "").strip()
    if not text or text in {"-", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _parse_percent(value: object) -> float | None:
    return _to_float(value)


KBARS_GAP_SEC = 0.25  # 低於官方 50 次／10 秒
SUBSCRIBE_LIMIT = 200

EMPTY_RETRY_SEC = 600.0

_api = None
_api_lock = threading.Lock()
_rest_lock = threading.Lock()
_empty_at: dict[str, float] = {}
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


def stock_code(symbol: str) -> str:
    return str(symbol).split(".", 1)[0].strip()


def kbars_to_frame(kbars: object) -> pd.DataFrame:
    """把永豐 Kbars 轉成 OHLCV DataFrame。"""
    if kbars is None:
        return _sj_empty()
    if isinstance(kbars, pd.DataFrame):
        raw = kbars
    elif hasattr(kbars, "dict"):
        raw = pd.DataFrame(kbars.dict())
    else:
        try:
            raw = pd.DataFrame({**kbars})
        except TypeError:
            return _sj_empty()
    if raw is None or raw.empty:
        return _sj_empty()
    cols = {str(c).lower(): c for c in raw.columns}
    ts_col = cols.get("ts") or cols.get("datetime")
    if ts_col is None:
        return _sj_empty()
    rename = {}
    for name in ("open", "high", "low", "close", "volume"):
        if name in cols:
            rename[cols[name]] = name
    work = raw.rename(columns=rename)
    if any(col not in work.columns for col in ("open", "high", "low", "close")):
        return _sj_empty()
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
        return _sj_empty()
    frame = pd.concat(nonempty).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    if df is None or df.empty:
        return _sj_empty()
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
        return df if df is not None else _sj_empty()
    last = pd.Timestamp(df.index[-1])
    now = pd.Timestamp(datetime.now(TAIPEI))
    freq = "5min" if interval == "5m" else "min"
    if last.floor(freq) >= now.floor(freq):
        return df.iloc[:-1]
    return df


def sj_fetch_bars_many(
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


def _sj_login(api) -> None:
    key = os.environ["SHIOAJI_API_KEY"].strip()
    secret = os.environ["SHIOAJI_SECRET_KEY"].strip()
    kwargs = {"api_key": key, "secret_key": secret}
    try:
        if "fetch_contract" in inspect.signature(api.login).parameters:
            kwargs["fetch_contract"] = True
    except (TypeError, ValueError):
        pass
    api.login(**kwargs)
    _wait_stock_contracts(api)


def _sj_busy(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return "exclusive" in msg or "timeout" in name or "timeout" in msg


def _v2_contracts(api):
    """Shioaji 1.7 起用 api.contracts；不要碰已廢棄的 api.Contracts。"""
    contracts = getattr(api, "contracts", None)
    if contracts is not None and callable(getattr(contracts, "get", None)):
        return contracts
    return None


def _legacy_stock_bucket(api):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(getattr(api, "Contracts", None), "Stocks", None)


def _wait_stock_contracts(api, timeout_sec: float = 180.0) -> int:
    """login 自己會抓合約，不要再呼叫 fetch_contracts。

    新版用 api.contracts.get('2330') 探測；舊版才數 api.Contracts 檔數。
    """
    print("永豐載入股票合約中（第一次較久，不要再按一次 Run）…", flush=True)
    deadline = time.time() + timeout_sec
    last_print = 0.0
    last_n = -1
    stable = 0
    v2 = _v2_contracts(api) is not None
    while time.time() < deadline:
        if v2:
            if _contract(api, "2330") is not None:
                print("永豐合約就緒。", flush=True)
                return 1
            now = time.time()
            if now - last_print >= 10:
                print("永豐商品合約下載中…", flush=True)
                last_print = now
        else:
            n = _contract_count(api)
            if n != last_n:
                stable = 0
                if n > 0:
                    print(f"永豐商品合約下載中… 目前 {n} 檔", flush=True)
            elif n > 200:
                stable += 1
                if stable >= 3:
                    print(f"永豐合約就緒：{n} 檔", flush=True)
                    return n
            last_n = n
        time.sleep(2)
    n = _contract_count(api)
    if _contract(api, "2330") is not None:
        print("永豐合約就緒。", flush=True)
        return n or 1
    if n > 100:
        print(f"永豐合約未完全穩定（目前 {n} 檔），先繼續…", flush=True)
    else:
        print("商品合約還沒齊，先繼續…", flush=True)
    return n


def _contract_count(api) -> int:
    v2 = _v2_contracts(api)
    if v2 is not None and callable(getattr(v2, "list", None)):
        return 0
    stocks = _legacy_stock_bucket(api)
    if stocks is None:
        return 0
    total = 0
    for exch in ("TSE", "OTC"):
        try:
            bucket = stocks[exch]
        except Exception:  # noqa: BLE001
            bucket = getattr(stocks, exch, None)
        if bucket is None:
            continue
        try:
            if hasattr(bucket, "values"):
                total += len(list(bucket.values()))
            else:
                total += len(list(bucket))
        except Exception:  # noqa: BLE001
            continue
    return total


def login():
    global _api, _callback_bound
    if not configured():
        raise RuntimeError("未設定 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY")
    with _api_lock:
        if _api is not None:
            return _api
        import shioaji as sj

        print("永豐登入中（第一次下載商品合約會等 1～2 分鐘，不要再按一次 Run）…", flush=True)
        api = sj.Shioaji()
        _sj_login(api)
        print("永豐登入完成。", flush=True)
        if not _callback_bound:
            _bind_tick_callback(api)
            _callback_bound = True
        _api = api
        return api


SNAPSHOT_BATCH = 80


def fetch_snapshot_ranking(top: int = 100) -> tuple[list, str | None]:
    """用永豐快照排上市＋上櫃成交金額。"""
    api = login()
    contracts = _stock_contracts(api)
    print(f"永豐快照排行：{len(contracts)} 檔…", flush=True)
    rows: list = []
    for i in range(0, len(contracts), SNAPSHOT_BATCH):
        batch = contracts[i : i + SNAPSHOT_BATCH]
        try:
            snaps = api.snapshots(batch)
        except Exception:  # noqa: BLE001
            continue
        if not snaps:
            continue
        for snap in snaps:
            stock = _ranked_from_snap(snap, RankedStock)
            if stock is not None:
                rows.append(stock)
        time.sleep(0.1)
    rows.sort(key=lambda s: s.turnover, reverse=True)
    ranked = [
        RankedStock(
            rank=i,
            symbol=s.symbol,
            name=s.name,
            price=s.price,
            change=s.change,
            change_percent=s.change_percent,
            volume_lots=s.volume_lots,
            turnover=s.turnover,
            exchange=s.exchange,
        )
        for i, s in enumerate(rows[:top], 1)
    ]
    stamp = datetime.now(TAIPEI).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    return ranked, f"{stamp} 永豐快照"


def _stock_contracts(api) -> list:
    stocks = _legacy_stock_bucket(api)
    if stocks is None:
        return []
    out = []
    seen: set[str] = set()
    for exch in ("TSE", "OTC"):
        bucket = None
        try:
            bucket = stocks[exch]
        except Exception:  # noqa: BLE001
            bucket = getattr(stocks, exch, None)
        if bucket is None:
            continue
        items = []
        if hasattr(bucket, "values"):
            try:
                items = list(bucket.values())
            except Exception:  # noqa: BLE001
                items = []
        if not items:
            try:
                items = list(bucket)
            except Exception:  # noqa: BLE001
                items = []
        for contract in items:
            code = str(getattr(contract, "code", "") or "")
            if len(code) != 4 or not code.isdigit() or code in seen:
                continue
            seen.add(code)
            out.append(contract)
    return out


def _ranked_from_snap(snap, ranked_cls):
    code = str(getattr(snap, "code", "") or "")
    if len(code) != 4 or not code.isdigit():
        return None
    close = _snap_float(getattr(snap, "close", None))
    if not close:
        return None
    turnover = _snap_float(getattr(snap, "total_amount", None)) or 0.0
    if turnover <= 0:
        vol = _snap_float(getattr(snap, "total_volume", None)) or 0.0
        turnover = close * vol * 1000.0
    if turnover <= 0:
        return None
    exch_raw = str(getattr(snap, "exchange", "") or "").upper()
    if "OTC" in exch_raw or exch_raw in {"TWO", "OTC"}:
        suffix, exchange = ".TWO", "TWO"
    else:
        suffix, exchange = ".TW", "TAI"
    name = str(getattr(snap, "name", "") or code)
    change = _snap_float(getattr(snap, "change_price", None))
    chg_pct = _snap_float(getattr(snap, "change_rate", None))
    vol = _snap_float(getattr(snap, "total_volume", None))
    return ranked_cls(
        rank=0,
        symbol=f"{code}{suffix}",
        name=name,
        price=close,
        change=change,
        change_percent=chg_pct,
        volume_lots=int(vol) if vol is not None else None,
        turnover=turnover,
        exchange=exchange,
    )


def _snap_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def subscribe_symbols(symbols: list[str]) -> list[str]:
    """盤中訂閱成交；回傳成功的代號。超過 200 檔會截斷。"""
    api = login()
    ok: list[str] = []
    for symbol in symbols[:SUBSCRIBE_LIMIT]:
        code = stock_code(symbol)
        if symbol in _subscribed:
            ok.append(symbol)
            continue
        contract = _contract(api, code)
        if contract is None:
            continue
        try:
            with _rest_lock:
                try:
                    api.quote.subscribe(contract, quote_type="tick", version="v1")
                except TypeError:
                    api.quote.subscribe(contract, quote_type="tick")
        except Exception as exc:  # noqa: BLE001
            if _sj_busy(exc):
                print(f"訂閱 {symbol} 時永豐忙碌，略過", flush=True)
            continue
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
        _empty_at.clear()
        _open_bars.clear()
        _subscribed.clear()


def _peek_1m(symbol: str, start_d: date, end_d: date) -> pd.DataFrame | None:
    with _frames_lock:
        cached = _frames.get(symbol)
        rng = _frame_ranges.get(symbol)
        if cached is None or rng is None:
            return None
        if rng[0] > start_d or rng[1] < end_d:
            return None
        if cached.empty:
            # 沒資料的標的別每分鐘重打一次 kbars，隔一段時間才重試
            recorded = _empty_at.get(symbol)
            if recorded is None or time.time() - recorded > EMPTY_RETRY_SEC:
                return None
        return cached.copy()


def _kbars_day(api, contract, day: date) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(4):
        try:
            with _rest_lock:
                kbars = api.kbars(
                    contract=contract,
                    start=day.isoformat(),
                    end=day.isoformat(),
                )
            return kbars_to_frame(kbars)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _sj_busy(exc) and attempt < 3:
                wait = 2 * (attempt + 1)
                print(f"永豐忙碌（{day.isoformat()}），{wait} 秒後再試…", flush=True)
                time.sleep(wait)
                continue
            raise
    assert last is not None
    raise last


def _one_minute(api, symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    """1K 每次最多約 270 根（一個台股交易日），按日抓再串起來。"""
    cached = _peek_1m(symbol, start_d, end_d)
    if cached is not None:
        return cached
    code = stock_code(symbol)
    contract = _contract(api, code)
    if contract is None:
        time.sleep(2)
        contract = _contract(api, code)
    if contract is None:
        return _sj_empty()
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
                part = _kbars_day(api, contract, day)
                if not part.empty:
                    parts.append(part)
            except Exception as exc:  # noqa: BLE001
                if _sj_busy(exc):
                    print(f"  {symbol} {day.isoformat()} 失敗：{exc}", flush=True)
        day += timedelta(days=1)
    frame = concat_daily_frames(parts)
    with _frames_lock:
        _frames[symbol] = frame.copy()
        _frame_ranges[symbol] = (start_d, end_d)
        if frame.empty:
            _empty_at[symbol] = time.time()
        else:
            _empty_at.pop(symbol, None)
    return frame


def _contract(api, code: str):
    v2 = _v2_contracts(api)
    if v2 is not None:
        try:
            return v2.get(code)
        except Exception:  # noqa: BLE001
            return None
    stocks = _legacy_stock_bucket(api)
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
    start_d = _sj_as_date(start)
    end_d = _sj_as_date(end)
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


def _sj_as_date(value: date | str | None) -> date | None:
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
        if stock_code(symbol) == code:
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


def _sj_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


_source = "shioaji"


def set_kline_source(source: str) -> None:
    global _source
    _source = "shioaji"


def kline_source() -> str:
    return _source


def using_shioaji() -> bool:
    if not configured():
        raise RuntimeError("未設定 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY")
    return True


def fetch_1m_bars(
    symbol: str,
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
    **_: object,
) -> pd.DataFrame:
    """下載單一標的一分 K。"""
    frames = fetch_bars_many(
        [symbol],
        interval="1m",
        range_=range_,
        closed_only=closed_only,
        start=start,
        end=end,
    )
    return frames.get(symbol, _empty())


def fetch_1m_bars_many(
    symbols: list[str],
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
) -> dict[str, pd.DataFrame]:
    return fetch_bars_many(
        symbols,
        interval="1m",
        range_=range_,
        closed_only=closed_only,
        start=start,
        end=end,
    )


def kline_window_for_date(on_date: date, lookback_days: int = 7) -> tuple[date, date]:
    """回測某日時，往前 lookback_days 抓 K 線（含前一交易日，MA240 才算得出來）。end 不含當天之後。"""
    return on_date - timedelta(days=lookback_days), on_date + timedelta(days=1)


def fetch_bars_many(
    symbols: list[str],
    interval: str = "1m",
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
) -> dict[str, pd.DataFrame]:
    """一次抓多檔 K 線，回傳 symbol -> OHLCV。"""
    unique = list(dict.fromkeys(s for s in symbols if s))
    if not unique:
        return {}
    return sj_fetch_bars_many(
        unique,
        interval=interval,
        range_=range_,
        closed_only=closed_only,
        start=start,
        end=end,
    )


def _as_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@dataclass(frozen=True)
class ScanConfig:
    top: int = 100
    max_price: float = 650.0
    kline_range: str = "5d"
    closed_only: bool = False
    workers: int = 8
    timeout: int = 20
    latest_only: bool = False
    exclude_etf: bool = True
    exclude_financial: bool = True
    on_date: date | None = None
    min_ma_span: float = 0.004
    min_ma20_ma240_gap: float = 0.004
    max_ma20_ma240_gap: float = 0.010
    kline_start: date | None = None
    kline_end: date | None = None
    reuse_universe: list[RankedStock] | None = None
    reuse_rank_time: str | None = None


@dataclass
class ScanHit:
    stock: RankedStock
    snapshot: AlertSnapshot
    bars: int
    frame: pd.DataFrame | None = None
    frame_5m: pd.DataFrame | None = None


@dataclass
class ScanResult:
    scanned_at: datetime
    rank_time: str | None
    universe: list[RankedStock]
    candidates: list[RankedStock]
    hits: list[ScanHit]
    skipped: list[tuple[RankedStock, str]] = field(default_factory=list)
    errors: list[tuple[RankedStock, str]] = field(default_factory=list)
    price_dropped: int = 0
    etf_dropped: int = 0
    financial_dropped: int = 0
    below_5m_dropped: int = 0
    tangled_dropped: int = 0
    far_ma_dropped: int = 0
    as_of: date | None = None


def run_scan(
    config: ScanConfig | None = None,
    session: requests.Session | None = None,
) -> ScanResult:
    cfg = config or ScanConfig()
    sess = session or requests.Session()
    if cfg.on_date is not None:
        universe, rank_time = fetch_daily_turnover_ranking(
            cfg.on_date, top=cfg.top, session=sess, timeout=cfg.timeout
        )
    elif cfg.reuse_universe:
        universe, rank_time = list(cfg.reuse_universe), cfg.reuse_rank_time
    else:
        universe, rank_time = fetch_turnover_ranking(
            top=cfg.top, session=sess, timeout=cfg.timeout
        )
    if len(universe) < 20:
        raise RuntimeError(f"成交額名單只有 {len(universe)} 檔，抓不到證交所/櫃買排行。")
    priced = filter_by_price(universe, cfg.max_price)
    price_dropped = len(universe) - len(priced)
    if cfg.exclude_etf:
        after_etf = filter_etfs(priced)
    else:
        after_etf = priced
    etf_dropped = len(priced) - len(after_etf)
    if cfg.exclude_financial:
        candidates = filter_financials(after_etf)
    else:
        candidates = after_etf
    financial_dropped = len(after_etf) - len(candidates)

    hits: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    errors: list[tuple[RankedStock, str]] = []
    tangled_dropped = 0
    far_ma_dropped = 0

    kline_start = cfg.kline_start
    kline_end = cfg.kline_end
    if cfg.on_date is not None and kline_start is None:
        kline_start, kline_end = kline_window_for_date(cfg.on_date)

    frames: dict[str, pd.DataFrame] = {}
    try:
        frames = fetch_1m_bars_many(
            [s.symbol for s in candidates],
            range_=cfg.kline_range,
            closed_only=cfg.closed_only,
            start=kline_start,
            end=kline_end,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append((candidates[0] if candidates else RankedStock(0, "—", "—", 0.0, None, None, None, 0.0, ""), str(exc)))
        return ScanResult(
            scanned_at=datetime.now(TAIPEI),
            rank_time=rank_time,
            universe=universe,
            candidates=candidates,
            hits=[],
            skipped=[],
            errors=errors,
            price_dropped=price_dropped,
            etf_dropped=etf_dropped,
            financial_dropped=financial_dropped,
            as_of=cfg.on_date,
        )

    for stock in candidates:
        df = frames.get(stock.symbol)
        if df is None or df.empty:
            skipped.append((stock, "無一分 K 資料"))
            continue
        if len(df) < 241:
            skipped.append((stock, f"一分 K 不足 241 根（{len(df)}）"))
            continue
        since = None
        until = None
        if cfg.on_date is not None:
            since = pd.Timestamp(cfg.on_date, tz=TAIPEI)
            until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        elif not cfg.latest_only:
            now = datetime.now(TAIPEI)
            since = pd.Timestamp(now.date(), tz=TAIPEI)
        snapshot = latest_ma240_breakout_bullish(
            df,
            since=since,
            until=until,
            latest_only=cfg.latest_only,
            min_ma_span=cfg.min_ma_span,
            min_ma20_ma240_gap=cfg.min_ma20_ma240_gap,
            max_ma20_ma240_gap=cfg.max_ma20_ma240_gap,
        )
        if snapshot is None:
            continue
        if not mas_are_open(snapshot, cfg.min_ma_span):
            skipped.append((stock, "均線糾結"))
            tangled_dropped += 1
            continue
        if not ma20_near_ma240(
            snapshot,
            max_gap=cfg.max_ma20_ma240_gap,
            min_gap=cfg.min_ma20_ma240_gap,
        ):
            skipped.append((stock, "MA20/MA240 距離不對"))
            far_ma_dropped += 1
            continue
        hits.append(ScanHit(stock=stock, snapshot=snapshot, bars=len(df), frame=df))

    below_5m_dropped = 0
    if hits:
        try:
            frames_5m = fetch_bars_many(
                [h.stock.symbol for h in hits],
                interval="5m",
                range_=cfg.kline_range,
                closed_only=cfg.closed_only,
                start=kline_start,
                end=kline_end,
            )
            for hit in hits:
                hit.frame_5m = frames_5m.get(hit.stock.symbol)
        except Exception as exc:  # noqa: BLE001
            errors.append((hits[0].stock, f"五分K下載失敗：{exc}"))
            for hit in hits:
                skipped.append((hit.stock, "五分K下載失敗"))
            below_5m_dropped = len(hits)
            hits = []
        else:
            hits, dropped_5m, skip_5m = apply_5m_ma240_filter(hits)
            skipped.extend(skip_5m)
            below_5m_dropped = dropped_5m

    hits.sort(key=lambda h: h.stock.rank)
    skipped.sort(key=lambda item: item[0].rank)
    return ScanResult(
        scanned_at=datetime.now(TAIPEI),
        rank_time=rank_time,
        universe=universe,
        candidates=candidates,
        hits=hits,
        skipped=skipped,
        errors=errors,
        price_dropped=price_dropped,
        etf_dropped=etf_dropped,
        financial_dropped=financial_dropped,
        below_5m_dropped=below_5m_dropped,
        tangled_dropped=tangled_dropped,
        far_ma_dropped=far_ma_dropped,
        as_of=cfg.on_date,
    )


def apply_5m_ma240_filter(
    hits: list[ScanHit],
) -> tuple[list[ScanHit], int, list[tuple[RankedStock, str]]]:
    """一分站穩當下的五分K收盤必須高於五分 MA240。"""
    kept: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    for hit in hits:
        frame = hit.frame_5m
        if frame is None or frame.empty:
            skipped.append((hit.stock, "無五分 K 資料"))
            continue
        if not close_above_ma240(frame, hit.snapshot.timestamp, floor="5min"):
            skipped.append((hit.stock, "五分K收盤在 MA240 底下"))
            continue
        kept.append(hit)
    return kept, len(skipped), skipped


def hit_key(hit: ScanHit) -> tuple[str, pd.Timestamp]:
    return hit.stock.symbol, hit.snapshot.timestamp


def send_notifications(title: str, body: str) -> list[str]:
    """送出所有已設定的通知通道，回傳成功通道名稱。"""
    sent: list[str] = []
    if _notify_send(title, body):
        sent.append("desktop")
    if _telegram(title, body):
        sent.append("telegram")
    if _discord(title, body):
        sent.append("discord")
    return sent


def format_hit_message(hits: list[ScanHit]) -> tuple[str, str]:
    if not hits:
        return "台股一分K掃描", "目前沒有符合條件的標的"
    title = f"台股一分K站穩MA240 × {len(hits)}"
    lines = []
    for hit in hits:
        stock = hit.stock
        snap = hit.snapshot
        chg = ""
        if stock.change_percent is not None:
            chg = f" {stock.change_percent:+.2f}%"
        lines.append(
            f"{stock.rank}. {stock.name} {stock.symbol} "
            f"{stock.price:.2f}{chg}\n"
            f"   收 {snap.close:.2f} > MA240 {snap.ma240:.2f} "
            f"（前收 {snap.prev_close:.2f} / 前MA240 {snap.prev_ma240:.2f}）\n"
            f"   MA5 {snap.ma5:.2f} > MA10 {snap.ma10:.2f} > MA20 {snap.ma20:.2f}\n"
            f"   {snap.timestamp.strftime('%H:%M')}  成交額 {stock.turnover/1e8:.2f} 億"
        )
    return title, "\n".join(lines)


def _notify_send(title: str, body: str) -> bool:
    binary = shutil.which("notify-send")
    if not binary:
        return False
    try:
        subprocess.run(
            [binary, "--app-name=tw-1m-screener", title, body[:1000]],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _telegram(title: str, body: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": f"{title}\n{body}"[:4000],
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        return resp.ok
    except requests.RequestException:
        return False


def _discord(title: str, body: str) -> bool:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    try:
        resp = requests.post(
            webhook,
            json={"content": f"**{title}**\n```\n{body[:1800]}\n```"},
            timeout=15,
        )
        return resp.ok
    except requests.RequestException:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股一分K多頭排列＋站穩MA240掃描（單檔）")
    p.add_argument("--top", type=int, default=100, help="成交額前 N 名（預設 100）")
    p.add_argument("--max-price", type=float, default=650.0, help="濾掉此價格以上（預設 650）")
    p.add_argument("--watch", action="store_true", help="盤中持續監控（PyCharm 直接 Run 預設就是這個）")
    p.add_argument("--once", action="store_true", help="只掃一次，不持續監控")
    p.add_argument("--interval", type=int, default=60, help="watch 間隔秒數")
    p.add_argument("--latest-only", action="store_true", help="只看最新一根（watch 模式自動開啟）")
    p.add_argument("--closed-only", action="store_true", help="只用已收盤的一分 K")
    p.add_argument("--include-etf", action="store_true", help="不過濾 ETF")
    p.add_argument("--include-financial", action="store_true", help="不過濾金融股")
    p.add_argument("--date", help="回測指定日 YYYY-MM-DD")
    p.add_argument("--last-friday", action="store_true", help="回測上週五")
    p.add_argument("--last-week", action="store_true", help="回測上週一到五")
    p.add_argument("--quiet-empty", action="store_true", help="沒命中時不印詳細清單")
    return p.parse_args()


def _on_date(args: argparse.Namespace) -> date | None:
    if args.last_friday:
        return previous_friday()
    if args.date:
        return date.fromisoformat(args.date)
    return None


def print_result(result, *, quiet_empty: bool) -> None:
    print(
        f"[{result.scanned_at.strftime('%H:%M:%S')}] "
        f"成交額前 {len(result.universe)}／股價濾掉 {result.price_dropped}／"
        f"ETF濾掉 {result.etf_dropped}／金融濾掉 {result.financial_dropped}／掃描 {len(result.candidates)}／"
        f"均線糾結濾掉 {result.tangled_dropped}／"
        f"MA20/MA240不符 {result.far_ma_dropped}／"
        f"五分MA240濾掉 {result.below_5m_dropped}／"
        f"命中 {len(result.hits)}／略過 {len(result.skipped)}／錯誤 {len(result.errors)}"
    )
    if result.as_of:
        print(f"  回測日期：{result.as_of.isoformat()}")
    print("  K線：永豐 Shioaji")
    if result.rank_time:
        print(f"  排行資料時間：{result.rank_time}")
    no_k = sum(1 for _, reason in result.skipped if "無一分" in reason or "不足" in reason)
    if no_k >= max(5, len(result.candidates) // 2) and not result.hits:
        print("  多數沒有一分K：合約可能還沒下完，或永豐還在忙碌。不要連按 Run。")
    if not result.hits:
        if not quiet_empty:
            print("  目前沒有符合條件的標的。")
        return
    for hit in result.hits:
        s, snap = hit.stock, hit.snapshot
        chg = f"{s.change_percent:+.2f}%" if s.change_percent is not None else ""
        print(
            f"  #{s.rank:3d} {s.name:8s} {s.symbol:10s} "
            f"{s.price:8.2f} {chg:>8s}  "
            f"收 {snap.close:.2f} > MA240 {snap.ma240:.2f}  "
            f"前收 {snap.prev_close:.2f}  "
            f"MA {snap.ma5:.2f}/{snap.ma10:.2f}/{snap.ma20:.2f}  "
            f"{snap.timestamp.strftime('%H:%M')}"
        )


def scan_once(
    args: argparse.Namespace,
    seen: set,
    *,
    ranking: tuple | None = None,
    first: bool = False,
) -> tuple[int, object]:
    on_date = _on_date(args)
    latest_only = (args.latest_only or args.watch) and on_date is None
    if first and args.watch and on_date is None:
        latest_only = False
    result = run_scan(
        ScanConfig(
            top=args.top,
            max_price=args.max_price,
            closed_only=args.closed_only or on_date is not None or (args.watch and using_shioaji()),
            latest_only=latest_only,
            exclude_etf=not args.include_etf,
            exclude_financial=not args.include_financial,
            on_date=on_date,
            kline_range="7d" if on_date is not None else "5d",
            reuse_universe=None if ranking is None else list(ranking[0]),
            reuse_rank_time=None if ranking is None else ranking[1],
        )
    )
    print_result(result, quiet_empty=args.quiet_empty)
    new_hits = [h for h in result.hits if hit_key(h) not in seen]
    for h in result.hits:
        seen.add(hit_key(h))
    if new_hits and on_date is None:
        title, body = format_hit_message(new_hits)
        print()
        print(title)
        print(body)
        channels = send_notifications(title, body)
        if channels:
            print(f"  已通知：{', '.join(channels)}")
        else:
            print("  未設定 Telegram（仍已印在終端機）。")
            print("  在本檔最上面填 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID")
    if result.errors and not args.quiet_empty:
        print("  錯誤：")
        for stock, err in result.errors[:8]:
            print(f"    {stock.symbol} {err}")
        if len(result.errors) > 8:
            print(f"    …另有 {len(result.errors) - 8} 筆")
    return len(new_hits), result


def scan_weekdays(args: argparse.Namespace, days: list, label: str) -> list:
    results = []
    print(f"{label} {days[0].isoformat()}～{days[-1].isoformat()}，每天分開")
    for day in days:
        result = run_scan(
            ScanConfig(
                top=args.top,
                max_price=args.max_price,
                closed_only=True,
                latest_only=False,
                exclude_etf=not args.include_etf,
                exclude_financial=not args.include_financial,
                on_date=day,
            )
        )
        print_result(result, quiet_empty=args.quiet_empty)
        results.append(result)
    print()
    print(f"{label}分日命中：")
    for result in results:
        assert result.as_of is not None
        names = "、".join(h.stock.name for h in result.hits) or "—"
        print(f"  {result.as_of.isoformat()}  {len(result.hits)} 檔  {names}")
    return results


def _sleep_to_next_minute(pad_sec: int = 3) -> None:
    now = datetime.now(TAIPEI)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=pad_sec)
    time.sleep(max(1.0, (nxt - now).total_seconds()))


def _tw_session_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(TAIPEI)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 <= minutes <= 13 * 60 + 35


def _seconds_until_open(now: datetime | None = None) -> float | None:
    """平日開盤前回傳還要等幾秒；已開盤、收盤後或週末回傳 None。"""
    now = now or datetime.now(TAIPEI)
    if now.weekday() >= 5:
        return None
    open_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now >= open_at:
        return None
    return (open_at - now).total_seconds()


def _ensure_shioaji() -> None:
    """沒裝 shioaji 就用目前這個 Python 自動 pip install。"""
    try:
        import shioaji  # noqa: F401
        return
    except ImportError:
        pass
    print("沒有 shioaji，正在裝到目前這個 Python：")
    print(f"  {sys.executable} -m pip install shioaji")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shioaji"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "安裝失敗。請在 PyCharm Terminal 執行：\n"
            f"  {sys.executable} -m pip install shioaji"
        ) from exc
    try:
        import shioaji  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "裝完仍 import 不到。PyCharm 右下角 interpreter 要選這個：\n"
            f"  {sys.executable}"
        ) from exc


def main() -> int:
    print(f"scan_tw {SCRIPT_VERSION}（不該再出現 fetch_contracts / fn()）", flush=True)
    args = parse_args()
    _apply_secrets()
    if not args.once and _on_date(args) is None and not args.last_week:
        args.watch = True
    if not (
        os.environ.get("SHIOAJI_API_KEY", "").strip()
        and os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    ):
        print("請在本檔最上面填 SHIOAJI_API_KEY 與 SHIOAJI_SECRET_KEY")
        return 1
    set_kline_source("shioaji")
    _ensure_shioaji()
    if args.last_week:
        scan_weekdays(args, previous_weekdays(), "回測上週")
        return 0
    seen: set = set()
    _, result = scan_once(args, seen, first=True)
    ranking = (result.universe, result.rank_time)
    if not args.watch or _on_date(args) is not None:
        return 0

    forced = "--watch" in sys.argv
    wait_sec = _seconds_until_open()
    if wait_sec is not None and not forced:
        print(f"\n離 09:00 開盤還有 {wait_sec / 60:.0f} 分鐘。K 線已先抓好，等開盤自動開始監控。")
        print("這段時間不要關視窗，也不要重按 Run。")
        time.sleep(wait_sec + 5)
    elif not _tw_session_open() and not forced:
        print("現在不是台股開盤（平日 09:00–13:30）。收盤後沒有即時成交，掃完就結束。")
        print("明天盤中再按 Run，才會持續監控。")
        logout()
        return 0

    subscribed = subscribe_symbols([s.symbol for s in result.candidates])
    print(f"\nwatch 永豐即時：已訂閱 {len(subscribed)} 檔，每分鐘收完 K 再判斷（Ctrl+C 結束）")
    alerted = 0
    try:
        while True:
            _sleep_to_next_minute()
            if not _tw_session_open() and not forced:
                print("\n13:30 收盤，停止監控。")
                break
            new_hits, result = scan_once(args, seen, ranking=ranking)
            alerted += new_hits
            subscribe_symbols([s.symbol for s in result.candidates])
    except KeyboardInterrupt:
        print("\n已停止。")
    print(f"今天共通知 {alerted} 檔。")
    logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
