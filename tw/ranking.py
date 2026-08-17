"""成交額排行：Yahoo 股市即時排行（上市＋上櫃合併）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

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
