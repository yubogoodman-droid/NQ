"""盤中掃描：五分 K 空頭排列且跌破 MA200 就通知。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_bars_many
from tw.ranking import (
    DEFAULT_TURNOVER_TOP,
    RankedStock,
    fetch_turnover_ranking,
    filter_by_price,
    filter_etfs,
    filter_financials,
    filter_telecoms,
)
from tw.signals import AlertSnapshot, iter_5m_ma200_short_alerts

TAIPEI = ZoneInfo("Asia/Taipei")
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(13, 30)


@dataclass(frozen=True)
class WatchConfig:
    top: int = DEFAULT_TURNOVER_TOP
    max_price: float = 650.0
    exclude_etf: bool = True
    exclude_financial: bool = True
    exclude_telecom: bool = True
    kline_range: str = "5d"
    timeout: int = 20
    latest_only: bool = True


@dataclass
class ScanHit:
    stock: RankedStock
    snapshot: AlertSnapshot
    frame: pd.DataFrame


@dataclass
class ScanResult:
    scanned_at: datetime
    rank_time: str | None
    universe: list[RankedStock]
    candidates: list[RankedStock]
    hits: list[ScanHit]
    skipped: list[tuple[RankedStock, str]] = field(default_factory=list)
    price_dropped: int = 0
    etf_dropped: int = 0
    financial_dropped: int = 0
    telecom_dropped: int = 0


def market_open(now: datetime | None = None) -> bool:
    current = now or datetime.now(TAIPEI)
    if current.weekday() >= 5:
        return False
    return MARKET_OPEN <= current.time() <= MARKET_CLOSE


def hit_key(hit: ScanHit) -> str:
    return f"{hit.stock.symbol}|{hit.snapshot.timestamp.isoformat()}"


def run_scan(
    config: WatchConfig | None = None,
    session: requests.Session | None = None,
) -> ScanResult:
    cfg = config or WatchConfig()
    sess = session or requests.Session()
    universe, rank_time = fetch_turnover_ranking(
        top=cfg.top, session=sess, timeout=cfg.timeout
    )
    priced = filter_by_price(universe, cfg.max_price)
    after_etf = filter_etfs(priced) if cfg.exclude_etf else priced
    after_fin = filter_financials(after_etf) if cfg.exclude_financial else after_etf
    candidates = filter_telecoms(after_fin) if cfg.exclude_telecom else after_fin

    frames = fetch_bars_many(
        [s.symbol for s in candidates],
        interval="5m",
        range_=cfg.kline_range,
        closed_only=True,
    )
    now = datetime.now(TAIPEI)
    since = pd.Timestamp(now.date(), tz=TAIPEI)
    until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    hits: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    for stock in candidates:
        df = frames.get(stock.symbol)
        if df is None or df.empty:
            skipped.append((stock, "無五分 K 資料"))
            continue
        if len(df) < 201:
            skipped.append((stock, f"五分 K 不足 201 根（{len(df)}）"))
            continue
        alerts = iter_5m_ma200_short_alerts(
            df, since=since, until=until, latest_only=cfg.latest_only
        )
        for snap in alerts:
            hits.append(ScanHit(stock=stock, snapshot=snap, frame=df))

    hits.sort(key=lambda h: (h.snapshot.timestamp, h.stock.rank))
    return ScanResult(
        scanned_at=now,
        rank_time=rank_time,
        universe=universe,
        candidates=candidates,
        hits=hits,
        skipped=skipped,
        price_dropped=len(universe) - len(priced),
        etf_dropped=len(priced) - len(after_etf),
        financial_dropped=len(after_etf) - len(after_fin),
        telecom_dropped=len(after_fin) - len(candidates),
    )
