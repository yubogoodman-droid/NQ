#!/usr/bin/env python3
"""台股永豐監控（單一檔，可直接放到 PyCharm 執行）。

成交額前 100，濾 ETF／金融／電信／股價 500 以上。
五分或十五分剛站上／剛跌破 MA240 就推 Telegram（預設多方＋空方）。

把下面四行金鑰填好，然後 Run。不要把檔名取成 tw.py。

    python watch_tw.py --test
    python watch_tw.py
    python watch_tw.py --side short
"""

from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# —— 填這裡（不要 commit 真的金鑰）——
SHIOAJI_API_KEY = ""
SHIOAJI_SECRET_KEY = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

HERE = Path(__file__).resolve().parent
SEEN_PATH = HERE / "tw_shioaji_seen.json"
LOCAL_SECRETS = HERE / "local_secrets.py"
TG_SESSION = requests.Session()
TAIPEI = ZoneInfo("Asia/Taipei")



# ===== K線合成 =====

def resample_ohlcv(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """把較短週期 OHLCV 合成較長週期（預設五分 → 十五分）。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.DatetimeIndex(work.index)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {name: how for name, how in agg.items() if name in work.columns}
    out = work.resample(rule, label="left", closed="left").agg(cols)
    return out.dropna(subset=["close"]) if "close" in out.columns else out


# ===== 成交額名單 =====

TWSE_MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_QUOTES_URL = (
    "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php"
)

YAHOO_TURNOVER_URL = "https://tw.stock.yahoo.com/rank/turnover"
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
    """抓最近一個「已公布」交易日的成交金額前 N 名。

    盤中官方當日排行尚未出爐，會改用前一個交易日，避免永豐 snapshots 掃全市場。
    """
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
    """由 start 往回列出最近的平日（含 start，若本身是平日）。"""
    current = start
    found = 0
    while found < limit:
        if current.weekday() < 5:
            yield current
            found += 1
        current -= timedelta(days=1)


def parse_yahoo_ranking_html(html: str) -> tuple[list[RankedStock], str | None]:
    payload = _extract_app_main(html)
    table = (
        payload.get("context", {})
        .get("dispatcher", {})
        .get("stores", {})
        .get("TableStore", {})
        .get("main-0-StockRanking")
    )
    if not isinstance(table, dict) or not table.get("list"):
        raise ValueError("Yahoo 成交額排行資料格式異常")

    rank_time = None
    meta = table.get("listMeta") or {}
    if isinstance(meta, dict):
        rank_time = meta.get("rankTime")

    stocks: list[RankedStock] = []
    for row in table["list"]:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        price = _to_float(row.get("price"))
        if price is None:
            continue
        stocks.append(
            RankedStock(
                rank=int(row.get("rank") or len(stocks) + 1),
                symbol=symbol,
                name=str(row.get("name") or row.get("symbolName") or symbol),
                price=price,
                change=_to_float(row.get("change")),
                change_percent=_parse_percent(row.get("changePercent")),
                volume_lots=_to_int(row.get("volK")),
                turnover=_turnover_from_row(row),
                exchange=_exchange_of(symbol),
            )
        )
    stocks.sort(key=lambda s: s.rank)
    return stocks, rank_time


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


def last_n_weekdays(n: int = 5, today: date | None = None) -> list[date]:
    """含 today（若為平日）往回的最近 n 個平日，由舊到新。"""
    if n < 1:
        raise ValueError("n must be >= 1")
    current = today or date.today()
    days: list[date] = []
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    days.reverse()
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


_TELECOM_CODES = {"2412", "3045", "4904", "3682"}
_TELECOM_NAME_MARKERS = ("電信", "中華電", "台灣大", "遠傳", "亞太電", "台灣之星")


def is_telecom(stock: RankedStock) -> bool:
    """電信營運商（中華電、台灣大、遠傳、亞太電）。智邦、啟碁等通信設備不算。"""
    if stock.code in _TELECOM_CODES:
        return True
    name = re.sub(r"[-*].*$", "", stock.name).strip()
    return any(mark in name for mark in _TELECOM_NAME_MARKERS)


def filter_telecoms(stocks: list[RankedStock]) -> list[RankedStock]:
    return [s for s in stocks if not is_telecom(s)]


def _extract_app_main(html: str) -> dict:
    match = re.search(r"root\.App\.main = (\{.*?\});\s*(?:</script>|\n)", html, re.S)
    if not match:
        raise ValueError("找不到 Yahoo App.main 排行資料")
    raw = match.group(1)
    raw = re.sub(r"(?<![A-Za-z\"])undefined(?![A-Za-z])", "null", raw)
    raw = re.sub(r"(?<![A-Za-z\"])NaN(?![A-Za-z])", "null", raw)
    return json.loads(raw)


def _turnover_from_row(row: dict) -> float:
    # turnoverK 單位為千元
    turnover_k = _to_float(row.get("turnoverK"))
    if turnover_k is not None:
        return turnover_k * 1000.0
    return 0.0


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


# ===== 進出場訊號 =====

MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_MED = 60
MA_LONG = 240
H1_MA = 20
# MA5 相對 MA20 至少拉開這麼多（％），否則算糾結。
MIN_RIBBON_FAN_PCT = 0.50
# MA5–MA10、MA10–MA20 各自相對收盤至少這麼多（％），避免其中兩條黏在一起。
MIN_RIBBON_GAP_PCT = 0.10
# 十五分收盤要在 MA240 上，而且連續至少這麼久。
M15_ABOVE_MA240_MINUTES = 30
M15_BAR_MINUTES = 15


@dataclass(frozen=True)
class AlertSnapshot:
    timestamp: pd.Timestamp
    close: float
    prev_close: float
    ma5: float
    ma10: float
    ma20: float
    ma240: float
    prev_ma5: float
    prev_ma10: float
    prev_ma20: float
    prev_ma240: float
    h1_close: float | None = None
    h1_ma5: float | None = None
    h1_ma10: float | None = None
    h1_ma20: float | None = None
    m15_close: float | None = None
    m15_ma5: float | None = None
    m15_ma10: float | None = None
    m15_ma20: float | None = None
    m15_ma240: float | None = None
    m15_above_ma240_minutes: int | None = None
    side: str = "long"

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def bearish_aligned(self) -> bool:
        return self.ma5 < self.ma10 < self.ma20

    @property
    def mas_rising(self) -> bool:
        return self.ma5 > self.prev_ma5 and self.ma10 > self.prev_ma10 and self.ma20 > self.prev_ma20

    @property
    def mas_falling(self) -> bool:
        return self.ma5 < self.prev_ma5 and self.ma10 < self.prev_ma10 and self.ma20 < self.prev_ma20

    @property
    def ribbon_fan_pct(self) -> float:
        if self.ma20 == 0:
            return 0.0
        return (self.ma5 / self.ma20 - 1.0) * 100.0

    @property
    def gap_5_10_pct(self) -> float:
        if not self.close:
            return 0.0
        return (self.ma5 - self.ma10) / self.close * 100.0

    @property
    def gap_10_20_pct(self) -> float:
        if not self.close:
            return 0.0
        return (self.ma10 - self.ma20) / self.close * 100.0

    @property
    def ribbon_fanned(self) -> bool:
        """多頭排列：MA5 > MA10 > MA20，且三條都比前一根高。"""
        return self.bullish_aligned and self.mas_rising

    @property
    def ribbon_down(self) -> bool:
        """空頭排列：MA5 < MA10 < MA20，且三條都比前一根低。"""
        return self.bearish_aligned and self.mas_falling

    @property
    def crossed_above_ma240(self) -> bool:
        return self.close > self.ma240 and self.prev_close <= self.prev_ma240

    @property
    def crossed_below_ma240(self) -> bool:
        return self.close < self.ma240 and self.prev_close >= self.prev_ma240

    @property
    def close_above_all_mas(self) -> bool:
        return (
            self.close > self.ma5
            and self.close > self.ma10
            and self.close > self.ma20
            and self.close > self.ma240
        )

    @property
    def close_below_all_mas(self) -> bool:
        return (
            self.close < self.ma5
            and self.close < self.ma10
            and self.close < self.ma20
            and self.close < self.ma240
        )

    @property
    def hourly_close_above_ma20(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma20 is not None
            and self.h1_close > self.h1_ma20
        )

    @property
    def hourly_close_above_short_mas(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma5 is not None
            and self.h1_ma10 is not None
            and self.h1_ma20 is not None
            and self.h1_close > self.h1_ma5
            and self.h1_close > self.h1_ma10
            and self.h1_close > self.h1_ma20
        )

    @property
    def close_above_15m_mas(self) -> bool:
        return (
            self.m15_close is not None
            and self.m15_ma5 is not None
            and self.m15_ma10 is not None
            and self.m15_ma20 is not None
            and self.m15_close > self.m15_ma5
            and self.m15_close > self.m15_ma10
            and self.m15_close > self.m15_ma20
        )

    @property
    def fifteen_above_ma240_half_hour(self) -> bool:
        return (
            self.m15_close is not None
            and self.m15_ma240 is not None
            and self.m15_above_ma240_minutes is not None
            and self.m15_close > self.m15_ma240
            and self.m15_above_ma240_minutes >= M15_ABOVE_MA240_MINUTES
        )


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma60"] = close.rolling(MA_MED, min_periods=MA_MED).mean()
    out["ma240"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def hourly_close_and_mas(five_min: pd.DataFrame) -> tuple[float, float, float, float] | None:
    """用截至目前的五分K合成小時K，回傳（收盤, MA5, MA10, MA20）。"""
    hourly = resample_ohlcv(five_min, "1h")
    if len(hourly) < H1_MA or "close" not in hourly.columns:
        return None
    close = hourly["close"]
    ma5 = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    ma10 = close.rolling(MA_MID, min_periods=MA_MID).mean()
    ma20 = close.rolling(H1_MA, min_periods=H1_MA).mean()
    last = (close.iloc[-1], ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1])
    if any(pd.isna(v) for v in last):
        return None
    return float(last[0]), float(last[1]), float(last[2]), float(last[3])


def hourly_close_and_ma20(five_min: pd.DataFrame) -> tuple[float, float] | None:
    """用截至目前的五分K合成小時K，回傳（小時收盤, 小時MA20）。"""
    hourly = hourly_close_and_mas(five_min)
    if hourly is None:
        return None
    return hourly[0], hourly[3]


@dataclass(frozen=True)
class FifteenSnapshot:
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma240: float
    above_ma240_minutes: int


def fifteen_close_and_mas(five_min: pd.DataFrame) -> FifteenSnapshot | None:
    """用截至目前的五分K合成十五分K，含短均與 MA240，以及收盤在 MA240 上多久。"""
    m15 = resample_ohlcv(five_min, "15min")
    if len(m15) < MA_LONG or "close" not in m15.columns:
        return None
    work = add_moving_averages(m15)
    last = work.iloc[-1]
    needed = ("close", "ma5", "ma10", "ma20", "ma240")
    if any(pd.isna(last[col]) for col in needed):
        return None
    minutes = _minutes_above_ma240(work, pd.Timestamp(five_min.index[-1]))
    return FifteenSnapshot(
        close=float(last["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma240=float(last["ma240"]),
        above_ma240_minutes=minutes,
    )


def _minutes_above_ma240(m15: pd.DataFrame, signal_ts: pd.Timestamp) -> int:
    """當根必須收在 MA240 上；已走完的十五分K可用收盤站上（含剛好碰到）。"""
    last = m15.iloc[-1]
    if last["close"] <= last["ma240"] or pd.isna(last["ma240"]):
        return 0
    completed = 0
    for i in range(len(m15) - 2, -1, -1):
        row = m15.iloc[i]
        if pd.isna(row["ma240"]) or row["close"] < row["ma240"]:
            break
        completed += 1
    last_ts = pd.Timestamp(m15.index[-1])
    mark = pd.Timestamp(signal_ts)
    if last_ts.tzinfo is not None:
        mark = mark.tz_convert(last_ts.tzinfo) if mark.tzinfo else mark.tz_localize(last_ts.tzinfo)
    elif mark.tzinfo is not None:
        mark = mark.tz_localize(None)
    elapsed = int((mark - last_ts).total_seconds() // 60) + 5
    elapsed = min(M15_BAR_MINUTES, max(5, elapsed))
    return completed * M15_BAR_MINUTES + elapsed


def _wanted_sides(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("long", "short")
    if side in ("long", "short"):
        return (side,)
    raise ValueError(f"unknown side: {side}")


def _passes(snap: AlertSnapshot, side: str) -> bool:
    if side == "short":
        return snap.ribbon_down and snap.crossed_below_ma240 and snap.close_below_all_mas
    return snap.ribbon_fanned and snap.crossed_above_ma240 and snap.close_above_all_mas


def iter_5m_ma240_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    side: str = "long",
) -> list[AlertSnapshot]:
    """同一交易日連續五分 K。多方：MA5>MA10>MA20 且往上，剛站上 MA240；空方鏡像跌破。含開盤第一根。"""
    if df is None or len(df) < MA_LONG + 1:
        return []
    work = add_moving_averages(df)
    hits: list[AlertSnapshot] = []
    start = MA_LONG
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(start, i)
                matched = True
                break
        if not matched:
            return []
    wanted = _wanted_sides(side)
    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _snapshot_at(work, i)
        if snap is None:
            continue
        for want in wanted:
            if _passes(snap, want):
                hits.append(replace(snap, side=want))
                break
    return hits


def iter_15m_ma240_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    side: str = "long",
) -> list[AlertSnapshot]:
    """同一交易日連續十五分 K。多方剛站上／空方剛跌破十五分 MA240（含開盤第一根）。"""
    if df is None or df.empty or "close" not in df.columns:
        return []
    m15 = resample_ohlcv(df, "15min")
    if len(m15) < MA_LONG + 1:
        return []
    work = add_moving_averages(m15)
    hits: list[AlertSnapshot] = []
    start = MA_LONG
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(start, i)
                matched = True
                break
        if not matched:
            return []
    wanted = _wanted_sides(side)
    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _snapshot_at(work, i)
        if snap is None:
            continue
        matched_side = next((want for want in wanted if _passes(snap, want)), None)
        if matched_side is None:
            continue
        bar_end = pd.Timestamp(ts) + pd.Timedelta(minutes=15)
        five_window = df[df.index < bar_end]
        if five_window.empty:
            continue
        signal_ts = pd.Timestamp(five_window.index[-1])
        hits.append(replace(snap, timestamp=signal_ts, side=matched_side))
    return hits


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
        prev_ma5=float(prev["ma5"]),
        prev_ma10=float(prev["ma10"]),
        prev_ma20=float(prev["ma20"]),
        prev_ma240=float(prev["ma240"]),
    )


# ===== 盤中 tick / Telegram =====

SESSION_OPEN = dt_time(9, 0)
SESSION_CLOSE = dt_time(13, 30)


@dataclass
class OhlcvBar:
    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class BarAggregator:
    minutes: int = 5
    forming: OhlcvBar | None = None
    closed: list[OhlcvBar] = field(default_factory=list)

    def on_tick(self, ts: pd.Timestamp, price: float, volume: float = 0.0) -> OhlcvBar | None:
        """吃一筆成交。若跨進下一根，回傳剛收完的那根。"""
        if price <= 0:
            return None
        mark = _as_taipei(ts)
        bucket = floor_bar(mark, self.minutes)
        if self.forming is None:
            self.forming = OhlcvBar(bucket, price, price, price, price, volume)
            return None
        if bucket > self.forming.start:
            done = self.forming
            self.forming = OhlcvBar(bucket, price, price, price, price, volume)
            self.closed.append(done)
            return done
        self.forming.high = max(self.forming.high, price)
        self.forming.low = min(self.forming.low, price)
        self.forming.close = price
        self.forming.volume += volume
        return None

    def flush_if_due(self, now: pd.Timestamp) -> OhlcvBar | None:
        if self.forming is None:
            return None
        mark = _as_taipei(now)
        if mark >= self.forming.start + pd.Timedelta(minutes=self.minutes):
            done = self.forming
            self.forming = None
            self.closed.append(done)
            return done
        return None


def floor_bar(ts: pd.Timestamp, minutes: int = 5) -> pd.Timestamp:
    mark = _as_taipei(ts)
    minute = (mark.minute // minutes) * minutes
    return mark.replace(minute=minute, second=0, microsecond=0)


def _as_taipei(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    mark = pd.Timestamp(ts)
    if mark.tzinfo is None:
        return mark.tz_localize(TAIPEI)
    return mark.tz_convert(TAIPEI)


def in_session(now: datetime | pd.Timestamp | None = None) -> bool:
    mark = _as_taipei(now or datetime.now(TAIPEI))
    if mark.weekday() >= 5:
        return False
    clock = mark.time()
    return SESSION_OPEN <= clock <= SESSION_CLOSE


def kbars_to_ohlcv(kbars) -> pd.DataFrame:
    """永豐 api.kbars 轉成訊號用的 OHLCV（1 分K）。"""
    if kbars is None:
        return _empty_ohlcv()
    try:
        frame = pd.DataFrame({**kbars})
    except Exception:
        return _empty_ohlcv()
    if frame.empty:
        return _empty_ohlcv()
    cols = {str(c).lower(): c for c in frame.columns}
    ts_col = cols.get("ts") or cols.get("time")
    if ts_col is None:
        return _empty_ohlcv()
    rename = {}
    for dest, aliases in (
        ("open", ("open",)),
        ("high", ("high",)),
        ("low", ("low",)),
        ("close", ("close",)),
        ("volume", ("volume",)),
    ):
        src = next((cols[a] for a in aliases if a in cols), None)
        if src is not None:
            rename[src] = dest
    out = frame.rename(columns=rename)
    out.index = pd.to_datetime(out[ts_col])
    if out.index.tz is None:
        out.index = out.index.tz_localize(TAIPEI)
    else:
        out.index = out.index.tz_convert(TAIPEI)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    return out[keep].astype(float).sort_index()


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def upsert_bar(frame: pd.DataFrame, bar: OhlcvBar) -> pd.DataFrame:
    row = pd.DataFrame(
        {
            "open": [bar.open],
            "high": [bar.high],
            "low": [bar.low],
            "close": [bar.close],
            "volume": [bar.volume],
        },
        index=[_as_taipei(bar.start)],
    )
    if frame is None or frame.empty:
        return row
    work = frame.copy()
    work.loc[row.index[0], row.columns] = row.iloc[0]
    return work.sort_index()


def alerts_on_closed_bar(
    frame: pd.DataFrame,
    bar: OhlcvBar,
    *,
    tf: str,
    side: str = "both",
) -> list[AlertSnapshot]:
    """只收「剛收完這根」對應的通知，避免把歷史交叉重發。"""
    if frame is None or frame.empty:
        return []
    mark = _as_taipei(bar.start)
    day = mark.normalize()
    until = mark + pd.Timedelta(minutes=4, seconds=59)
    if tf == "15m":
        hits = iter_15m_ma240_alerts(
            frame, since=day, until=until + pd.Timedelta(minutes=10), side=side
        )
        return [h for h in hits if _same_15m(h.timestamp, mark)]
    hits = iter_5m_ma240_alerts(frame, since=day, until=until, side=side)
    return [h for h in hits if _as_taipei(h.timestamp) == mark]


def _same_15m(signal_ts: pd.Timestamp, five_start: pd.Timestamp) -> bool:
    sig = _as_taipei(signal_ts)
    left = floor_bar(five_start, 15)
    # 15 分K 在 09:10／09:25… 這根五分收完；訊號時間記最後一根五分。
    return floor_bar(sig, 15) == left and five_start.minute % 15 == 10


def format_telegram(name: str, symbol: str, snap: AlertSnapshot, tf: str) -> str:
    short = getattr(snap, "side", "long") == "short"
    if tf == "15m":
        title = "十五分K 剛跌破 MA240" if short else "十五分K 剛站上 MA240"
    else:
        title = "五分K 剛跌破 MA240" if short else "五分K 剛站上 MA240"
    cmp = "&lt;" if short else "&gt;"
    ts = pd.Timestamp(snap.timestamp).tz_convert(TAIPEI).strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{symbol}"
    lines = [
        f"<b>{title}</b>",
        f"{html.escape(name)} {html.escape(symbol)}",
        f"時間 {ts}",
        f"收盤 {snap.close:.2f} {cmp} MA240 {snap.ma240:.2f}",
        f"短均 {snap.ma5:.2f} / {snap.ma10:.2f} / {snap.ma20:.2f}",
    ]
    lines.append(url)
    return "\n".join(lines)


def should_run_15m(bar: OhlcvBar) -> bool:
    return _as_taipei(bar.start).minute % 15 == 10


# ===== 主程式 =====

def apply_keys() -> None:
    if LOCAL_SECRETS.exists():
        ns: dict = {}
        exec(LOCAL_SECRETS.read_text(encoding="utf-8"), ns)
        mapping = {
            "SHIOAJI_API_KEY": ("SHIOAJI_API_KEY", "API_KEY"),
            "SHIOAJI_SECRET_KEY": ("SHIOAJI_SECRET_KEY", "SECRET_KEY"),
            "TELEGRAM_BOT_TOKEN": ("TELEGRAM_BOT_TOKEN", "TG_TOKEN"),
            "TELEGRAM_CHAT_ID": ("TELEGRAM_CHAT_ID", "TG_CHAT_ID"),
        }
        for env_name, aliases in mapping.items():
            for alias in aliases:
                val = str(ns.get(alias, "")).strip()
                if val:
                    os.environ.setdefault(env_name, val)
                    break
    if SHIOAJI_API_KEY.strip():
        os.environ.setdefault("SHIOAJI_API_KEY", SHIOAJI_API_KEY.strip())
    if SHIOAJI_SECRET_KEY.strip():
        os.environ.setdefault("SHIOAJI_SECRET_KEY", SHIOAJI_SECRET_KEY.strip())
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def telegram_send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = TG_SESSION.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if not r.ok:
            try:
                desc = r.json().get("description", r.text[:200])
            except ValueError:
                desc = r.text[:200]
            print(f"  Telegram HTTP {r.status_code}：{desc}", flush=True)
        return bool(r.ok)
    except requests.RequestException as exc:
        print(f"  Telegram 連線失敗：{exc}", flush=True)
        return False


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    save = sorted(seen)
    if len(save) > 4000:
        save = save[-2000:]
    SEEN_PATH.write_text(json.dumps(save), encoding="utf-8")


def load_universe(top: int, max_price: float) -> tuple[list[RankedStock], str]:
    stocks, label = fetch_turnover_ranking(top=top)
    priced = filter_by_price(stocks, max_price)
    candidates = filter_telecoms(filter_financials(filter_etfs(priced)))
    return candidates, label or "成交額名單"


def login_shioaji(*, simulation: bool):
    try:
        import shioaji as sj
    except ImportError as exc:
        raise SystemExit("請先安裝永豐 API：pip install shioaji") from exc
    key = os.environ.get("SHIOAJI_API_KEY", "").strip()
    secret = os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit("請填 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY（腳本上方或環境變數）")
    api = sj.Shioaji(simulation=simulation)
    api.login(api_key=key, secret_key=secret)
    return api, sj


def resolve_contract(api, stock: RankedStock):
    code = stock.code
    try:
        found = api.Contracts.Stocks[code]
        if found is not None:
            return found
    except Exception:
        pass
    getter = getattr(api, "contracts", None)
    if getter is not None and hasattr(getter, "get"):
        return getter.get(code)
    return None


def fetch_history(api, contract, days: int = 30) -> pd.DataFrame:
    end = datetime.now(TAIPEI).date()
    start = end - timedelta(days=days)
    kbars = api.kbars(
        contract=contract,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    one = kbars_to_ohlcv(kbars)
    if one.empty:
        return one
    return resample_ohlcv(one, "5min")


def push_snap(stock: RankedStock, snap, tf: str, seen: set[str], *, dry: bool) -> bool:
    day = datetime.now(TAIPEI).date().isoformat()
    side = getattr(snap, "side", "long")
    key = f"{day}:{stock.symbol}:{pd.Timestamp(snap.timestamp)}:{tf}:{side}"
    if key in seen:
        return False
    seen.add(key)
    text = format_telegram(stock.name, stock.symbol, snap, tf)
    print(text.replace("&gt;", ">").replace("&lt;", "<"), flush=True)
    if dry:
        return True
    if telegram_send(text):
        print("  → Telegram 已送", flush=True)
    else:
        print("  → Telegram 失敗（檢查 token / chat id）", flush=True)
    return True


def emit_alerts(
    stock: RankedStock,
    frame: pd.DataFrame,
    bar: OhlcvBar,
    tfs: list[str],
    seen: set[str],
    *,
    dry: bool,
    side: str = "both",
) -> None:
    jobs = []
    if "5m" in tfs:
        jobs.append("5m")
    if "15m" in tfs and should_run_15m(bar):
        jobs.append("15m")
    for tf in jobs:
        for snap in alerts_on_closed_bar(frame, bar, tf=tf, side=side):
            push_snap(stock, snap, tf, seen, dry=dry)
    save_seen(seen)


def scan_once(
    api,
    candidates: list[RankedStock],
    tfs: list[str],
    seen: set[str],
    *,
    dry: bool,
    side: str = "both",
) -> int:
    n = 0
    for i, stock in enumerate(candidates, 1):
        contract = resolve_contract(api, stock)
        if contract is None:
            print(f"{stock.symbol} 找不到契約", flush=True)
            continue
        print(f"歷史K {i}/{len(candidates)} {stock.name} {stock.code}", flush=True)
        frame = fetch_history(api, contract)
        if frame.empty or len(frame) < 241:
            continue
        since = pd.Timestamp(datetime.now(TAIPEI).date(), tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        before = len(seen)
        if "5m" in tfs:
            for snap in iter_5m_ma240_alerts(frame, since=since, until=until, side=side):
                push_snap(stock, snap, "5m", seen, dry=dry)
        if "15m" in tfs:
            for snap in iter_15m_ma240_alerts(frame, since=since, until=until, side=side):
                push_snap(stock, snap, "15m", seen, dry=dry)
        save_seen(seen)
        n += len(seen) - before
        time.sleep(0.2)
    return n


def run_watch(args: argparse.Namespace) -> int:
    apply_keys()
    tfs = ["5m", "15m"] if args.tf == "both" else [args.tf]
    side = args.side
    side_zh = {"long": "多方", "short": "空方", "both": "多方＋空方"}[side]
    seen = load_seen()
    candidates, label = load_universe(args.top, args.max_price)
    print(f"{label} → 掃描 {len(candidates)} 檔  tf={'+'.join(tfs)}  {side_zh}", flush=True)

    api, sj = login_shioaji(simulation=args.sim)
    if args.once:
        found = scan_once(api, candidates, tfs, seen, dry=args.dry, side=side)
        print(f"掃完，新通知 {found} 則", flush=True)
        try:
            api.logout()
        except Exception:
            pass
        return 0

    frames: dict[str, pd.DataFrame] = {}
    builders: dict[str, BarAggregator] = {}
    stocks: dict[str, RankedStock] = {}
    pending: queue.Queue[tuple[str, OhlcvBar]] = queue.Queue()

    print("啟動時抓一次歷史K（之後盤中只用 tick）…", flush=True)
    for i, stock in enumerate(candidates, 1):
        contract = resolve_contract(api, stock)
        if contract is None:
            print(f"略過 {stock.symbol}（無契約）", flush=True)
            continue
        print(f"歷史K {i}/{len(candidates)} {stock.name} {stock.code}", flush=True)
        try:
            frames[stock.code] = fetch_history(api, contract)
        except Exception as exc:  # noqa: BLE001
            print(f"  失敗：{exc}", flush=True)
            frames[stock.code] = pd.DataFrame()
        builders[stock.code] = BarAggregator(5)
        stocks[stock.code] = stock
        time.sleep(0.2)

    from shioaji import TickSTKv1, Exchange  # noqa: WPS433

    @api.on_tick_stk_v1()
    def _on_tick(exchange: Exchange, tick: TickSTKv1) -> None:
        code = str(getattr(tick, "code", "") or "")
        builder = builders.get(code)
        if builder is None:
            return
        try:
            price = float(tick.close)
            vol = float(getattr(tick, "volume", 0) or 0)
            ts = pd.Timestamp(tick.datetime)
        except Exception:
            return
        closed = builder.on_tick(ts, price, vol)
        if closed is not None:
            pending.put((code, closed))

    subscribed = 0
    for stock in candidates:
        if stock.code not in builders:
            continue
        contract = resolve_contract(api, stock)
        if contract is None:
            continue
        try:
            api.subscribe(contract, quote_type=sj.constant.QuoteType.Tick)
            subscribed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"訂閱失敗 {stock.code}：{exc}", flush=True)
    print(f"已訂閱 {subscribed} 檔 tick。Ctrl+C 結束。", flush=True)
    if not args.dry:
        telegram_send(
            f"台股監控已啟動\n{label}\n掃描 {len(builders)} 檔　{'＋'.join(tfs)}　{side_zh}"
        )

    try:
        while True:
            now = pd.Timestamp(datetime.now(TAIPEI))
            if not in_session(now) and now.time().replace(tzinfo=None) > datetime.strptime("13:35", "%H:%M").time():
                for code, builder in builders.items():
                    flushed = builder.flush_if_due(now)
                    if flushed is not None:
                        pending.put((code, flushed))
            try:
                code, bar = pending.get(timeout=2)
            except queue.Empty:
                for code, builder in list(builders.items()):
                    flushed = builder.flush_if_due(pd.Timestamp(datetime.now(TAIPEI)))
                    if flushed is not None:
                        pending.put((code, flushed))
                continue
            stock = stocks.get(code)
            if stock is None:
                continue
            frames[code] = upsert_bar(frames.get(code), bar)
            print(
                f"{bar.start.strftime('%H:%M')} 收 {stock.name} {bar.close:.2f}",
                flush=True,
            )
            emit_alerts(stock, frames[code], bar, tfs, seen, dry=args.dry, side=side)
    except KeyboardInterrupt:
        print("\n結束監控", flush=True)
    finally:
        try:
            api.logout()
        except Exception:
            pass
    return 0


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("台股永豐監控測試\n看到這則代表 Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="永豐 Shioaji 台股五分／十五分 MA240 Telegram 監控")
    p.add_argument("--tf", choices=("5m", "15m", "both"), default="both")
    p.add_argument(
        "--side",
        choices=("long", "short", "both"),
        default="both",
        help="多方站上／空方跌破／兩邊都盯（預設 both）",
    )
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--max-price", type=float, default=500.0)
    p.add_argument("--test", action="store_true", help="只測 Telegram")
    p.add_argument("--once", action="store_true", help="抓歷史K掃一次就結束（可盤後）")
    p.add_argument("--dry", action="store_true", help="符合條件只印、不推 Telegram")
    p.add_argument("--sim", action="store_true", help="永豐模擬環境登入")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        return test_telegram()
    return run_watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
