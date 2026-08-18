"""台股標的池：上市+上櫃普通股，排除 ETF，並依週成交額排名。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

TWSE_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_COMPANY_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tw-ma-short-backtest/1.0)",
    "Accept": "application/json",
}

ETF_NAME_MARKERS = ("ETF", "指數股票", "指數投資", "槓桿", "反向", "期信")


@dataclass(frozen=True)
class TwStock:
    code: str
    name: str
    market: str  # TW or TWO

    @property
    def ticker(self) -> str:
        suffix = "TW" if self.market == "TW" else "TWO"
        return f"{self.code}.{suffix}"


def _is_common_stock_code(code: str) -> bool:
    code = str(code).strip()
    return code.isdigit() and 4 <= len(code) <= 6 and not code.startswith("00")


def _looks_like_etf(name: str) -> bool:
    return any(m in name for m in ETF_NAME_MARKERS)


def _get_json(url: str, timeout: int = 60) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected JSON from {url}")
    return data


def fetch_stock_universe() -> list[TwStock]:
    """上市 + 上櫃公司清單（不含 ETF / 權證 / 00 開頭標的）。"""
    listed = _get_json(TWSE_COMPANY_URL)
    otc = _get_json(TPEX_COMPANY_URL)
    stocks: list[TwStock] = []
    seen: set[str] = set()

    for row in listed:
        code = str(row.get("公司代號", "")).strip()
        name = str(row.get("公司簡稱") or row.get("公司名稱") or "").strip()
        if not _is_common_stock_code(code) or _looks_like_etf(name) or code in seen:
            continue
        seen.add(code)
        stocks.append(TwStock(code=code, name=name, market="TW"))

    for row in otc:
        code = str(row.get("SecuritiesCompanyCode", "")).strip()
        name = str(row.get("CompanyAbbreviation") or row.get("CompanyName") or "").strip()
        if not _is_common_stock_code(code) or _looks_like_etf(name) or code in seen:
            continue
        seen.add(code)
        stocks.append(TwStock(code=code, name=name, market="TWO"))

    stocks.sort(key=lambda s: (s.market, s.code))
    return stocks


def previous_week_end(ts: pd.Timestamp) -> pd.Timestamp:
    """回傳「前一個已結束週」的週五（當日若為週五，仍用上週五，避免用到當週未完資料）。"""
    ts = pd.Timestamp(ts).normalize()
    days_since_friday = (ts.weekday() - 4) % 7
    last_friday = ts - pd.Timedelta(days=days_since_friday)
    if days_since_friday == 0:
        last_friday = last_friday - pd.Timedelta(days=7)
    return last_friday


def weekly_top_n_mask(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    top_n: int = 100,
    max_price: float = 600.0,
) -> pd.DataFrame:
    """
    每個交易日是否屬於「上一週成交額前 N、且該週收盤價 <= max_price」。

    成交額 = 收盤價 × 成交量。用上一完整週（週五截止）避免前瞻偏差。
    """
    turnover = close * volume
    weekly_turnover = turnover.resample("W-FRI").sum(min_count=1)
    weekly_close = close.resample("W-FRI").last()
    weekly_turnover = weekly_turnover.where(weekly_close <= max_price)

    rank = weekly_turnover.rank(axis=1, ascending=False, method="first")
    weekly_eligible = rank <= top_n

    prev_ends = pd.DatetimeIndex([previous_week_end(d) for d in close.index])
    aligned = weekly_eligible.reindex(prev_ends).fillna(False)
    aligned.index = close.index
    return aligned.astype(bool)


def latest_weekly_top(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    stocks: list[TwStock],
    *,
    top_n: int = 100,
    max_price: float = 600.0,
) -> pd.DataFrame:
    """最近一個完整週的成交額前 N（已濾 ETF 池與股價上限）。"""
    names = {s.ticker: s.name for s in stocks}
    turnover = (close * volume).resample("W-FRI").sum(min_count=1)
    weekly_close = close.resample("W-FRI").last()
    last_week = turnover.dropna(how="all").index.max()
    if pd.isna(last_week):
        return pd.DataFrame()
    row = turnover.loc[last_week]
    px = weekly_close.loc[last_week]
    row = row.where(px <= max_price).dropna()
    top = row.sort_values(ascending=False).head(top_n)
    out = pd.DataFrame(
        {
            "ticker": top.index,
            "name": [names.get(t, t) for t in top.index],
            "turnover": top.values,
            "close": px.reindex(top.index).values,
            "rank": range(1, len(top) + 1),
        }
    )
    out.attrs["week_end"] = last_week
    return out


def tickers_ever_eligible(mask: pd.DataFrame) -> list[str]:
    """回傳在 mask 期間至少有一天符合週成交額條件的 ticker。"""
    if mask.empty:
        return []
    flags = mask.fillna(False).astype(bool)
    return [str(c) for c in flags.columns if bool(flags[c].any())]
