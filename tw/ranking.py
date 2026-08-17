"""成交額排行：Yahoo 股市即時排行（上市＋上櫃合併）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

import requests

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
) -> tuple[list[RankedStock], str | None]:
    """抓取成交金額前 N 名。回傳 (清單, 排行資料時間)。"""
    sess = session or requests.Session()
    resp = sess.get(YAHOO_TURNOVER_URL, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    stocks, rank_time = parse_yahoo_ranking_html(resp.text)
    return stocks[:top], rank_time


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


def fetch_daily_turnover_ranking(
    on_date: date,
    top: int = 100,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> tuple[list[RankedStock], str | None]:
    """上市＋上櫃當日成交金額排行（盤後）。"""
    sess = session or requests.Session()
    twse = _fetch_twse_daily(on_date, sess, timeout)
    tpex = _fetch_tpex_daily(on_date, sess, timeout)
    stocks = twse + tpex
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
