"""掃描成交額前 N 名，找出一分 K 多頭排列且剛站上 MA200 的標的。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_1m_bars_many, fetch_bars_many
from tw.ranking import RankedStock, fetch_turnover_ranking, filter_by_price, filter_etfs
from tw.signals import AlertSnapshot, latest_ma200_breakout_bullish

TAIPEI = ZoneInfo("Asia/Taipei")


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


def run_scan(
    config: ScanConfig | None = None,
    session: requests.Session | None = None,
) -> ScanResult:
    cfg = config or ScanConfig()
    sess = session or requests.Session()
    universe, rank_time = fetch_turnover_ranking(
        top=cfg.top, session=sess, timeout=cfg.timeout
    )
    priced = filter_by_price(universe, cfg.max_price)
    price_dropped = len(universe) - len(priced)
    if cfg.exclude_etf:
        candidates = filter_etfs(priced)
    else:
        candidates = priced
    etf_dropped = len(priced) - len(candidates)

    hits: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    errors: list[tuple[RankedStock, str]] = []

    frames: dict[str, pd.DataFrame] = {}
    try:
        frames = fetch_1m_bars_many(
            [s.symbol for s in candidates],
            range_=cfg.kline_range,
            closed_only=cfg.closed_only,
        )
    except Exception as exc:  # noqa: BLE001
        errors.extend((stock, str(exc)) for stock in candidates)
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
        )

    for stock in candidates:
        df = frames.get(stock.symbol)
        if df is None or df.empty:
            skipped.append((stock, "無一分 K 資料"))
            continue
        if len(df) < 201:
            skipped.append((stock, f"一分 K 不足 201 根（{len(df)}）"))
            continue
        since = None
        if not cfg.latest_only:
            now = datetime.now(TAIPEI)
            since = pd.Timestamp(now.date(), tz=TAIPEI)
        snapshot = latest_ma200_breakout_bullish(
            df, since=since, latest_only=cfg.latest_only
        )
        if snapshot is None:
            continue
        hits.append(ScanHit(stock=stock, snapshot=snapshot, bars=len(df), frame=df))

    if hits:
        try:
            frames_5m = fetch_bars_many(
                [h.stock.symbol for h in hits],
                interval="5m",
                range_=cfg.kline_range,
                closed_only=cfg.closed_only,
            )
            for hit in hits:
                hit.frame_5m = frames_5m.get(hit.stock.symbol)
        except Exception as exc:  # noqa: BLE001
            errors.append((hits[0].stock, f"五分K下載失敗：{exc}"))

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
    )


def hit_key(hit: ScanHit) -> tuple[str, pd.Timestamp]:
    return hit.stock.symbol, hit.snapshot.timestamp
