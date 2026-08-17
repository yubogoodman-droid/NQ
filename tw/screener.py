"""掃描成交額前 N 名，找出一分 K 多頭排列且剛站上 MA200 的標的。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_1m_bars
from tw.ranking import RankedStock, fetch_turnover_ranking, filter_by_price
from tw.signals import AlertSnapshot, is_ma200_breakout_bullish

TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class ScanConfig:
    top: int = 100
    max_price: float = 650.0
    kline_range: str = "5d"
    closed_only: bool = False
    workers: int = 8
    timeout: int = 20


@dataclass
class ScanHit:
    stock: RankedStock
    snapshot: AlertSnapshot
    bars: int


@dataclass
class ScanResult:
    scanned_at: datetime
    rank_time: str | None
    universe: list[RankedStock]
    candidates: list[RankedStock]
    hits: list[ScanHit]
    skipped: list[tuple[RankedStock, str]] = field(default_factory=list)
    errors: list[tuple[RankedStock, str]] = field(default_factory=list)


def run_scan(
    config: ScanConfig | None = None,
    session: requests.Session | None = None,
) -> ScanResult:
    cfg = config or ScanConfig()
    sess = session or requests.Session()
    universe, rank_time = fetch_turnover_ranking(
        top=cfg.top, session=sess, timeout=cfg.timeout
    )
    candidates = filter_by_price(universe, cfg.max_price)

    hits: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    errors: list[tuple[RankedStock, str]] = []

    def _one(stock: RankedStock) -> tuple[str, RankedStock, object]:
        local = requests.Session()
        try:
            df = fetch_1m_bars(
                stock.symbol,
                range_=cfg.kline_range,
                session=local,
                timeout=cfg.timeout,
                closed_only=cfg.closed_only,
            )
            if df is None or df.empty:
                return "skip", stock, "無一分 K 資料"
            if len(df) < 201:
                return "skip", stock, f"一分 K 不足 201 根（{len(df)}）"
            snapshot = is_ma200_breakout_bullish(df)
            if snapshot is None:
                return "miss", stock, len(df)
            return "hit", stock, (snapshot, len(df))
        except Exception as exc:  # noqa: BLE001 — 單檔失敗不中斷整批
            return "error", stock, str(exc)

    workers = max(1, min(cfg.workers, len(candidates) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, stock) for stock in candidates]
        for fut in as_completed(futures):
            kind, stock, payload = fut.result()
            if kind == "hit":
                snapshot, bars = payload  # type: ignore[misc]
                hits.append(ScanHit(stock=stock, snapshot=snapshot, bars=bars))
            elif kind == "skip":
                skipped.append((stock, str(payload)))
            elif kind == "error":
                errors.append((stock, str(payload)))

    hits.sort(key=lambda h: h.stock.rank)
    skipped.sort(key=lambda item: item[0].rank)
    errors.sort(key=lambda item: item[0].rank)
    return ScanResult(
        scanned_at=datetime.now(TAIPEI),
        rank_time=rank_time,
        universe=universe,
        candidates=candidates,
        hits=hits,
        skipped=skipped,
        errors=errors,
    )


def hit_key(hit: ScanHit) -> tuple[str, pd.Timestamp]:
    return hit.stock.symbol, hit.snapshot.timestamp
