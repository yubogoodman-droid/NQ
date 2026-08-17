"""掃描成交額前 N 名，找出一分 K 多頭排列且剛站上 MA200 的標的。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_1m_bars_many, fetch_bars_many
from tw.ranking import (
    RankedStock,
    fetch_daily_turnover_ranking,
    fetch_turnover_ranking,
    filter_by_price,
    filter_etfs,
)
from tw.signals import AlertSnapshot, latest_ma200_breakout_bullish, ma200_at, mas_are_open

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
    on_date: date | None = None
    min_ma_span: float = 0.005
    # 金居 / 玉晶光這類：五分圖已明顯站上 MA200，不是剛貼著均線。
    min_5m_ma200_gap: float = 0.06


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
    below_5m_dropped: int = 0
    tangled_dropped: int = 0
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
    else:
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
    tangled_dropped = 0

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
            as_of=cfg.on_date,
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
        until = None
        if cfg.on_date is not None:
            since = pd.Timestamp(cfg.on_date, tz=TAIPEI)
            until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        elif not cfg.latest_only:
            now = datetime.now(TAIPEI)
            since = pd.Timestamp(now.date(), tz=TAIPEI)
        snapshot = latest_ma200_breakout_bullish(
            df, since=since, until=until, latest_only=cfg.latest_only
        )
        if snapshot is None:
            continue
        if not mas_are_open(snapshot, cfg.min_ma_span):
            skipped.append((stock, "均線糾結"))
            tangled_dropped += 1
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
            hits, dropped_5m, skip_5m = apply_5m_ma200_filter(
                hits, min_gap=cfg.min_5m_ma200_gap
            )
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
        below_5m_dropped=below_5m_dropped,
        tangled_dropped=tangled_dropped,
        as_of=cfg.on_date,
    )


def apply_5m_ma200_filter(
    hits: list[ScanHit],
    min_gap: float = 0.06,
) -> tuple[list[ScanHit], int, list[tuple[RankedStock, str]]]:
    """一分金叉當下的五分K收盤必須明顯高於五分 MA200（預設至少 6%）。"""
    kept: list[ScanHit] = []
    skipped: list[tuple[RankedStock, str]] = []
    for hit in hits:
        frame = hit.frame_5m
        if frame is None or frame.empty:
            skipped.append((hit.stock, "無五分 K 資料"))
            continue
        pair = ma200_at(frame, hit.snapshot.timestamp, floor="5min")
        if pair is None or pair[1] <= 0:
            skipped.append((hit.stock, "五分 MA200 不足"))
            continue
        close, ma200 = pair
        gap = (close - ma200) / ma200
        if close <= ma200:
            skipped.append((hit.stock, "五分K收盤在 MA200 底下"))
            continue
        if gap < min_gap:
            skipped.append((hit.stock, f"五分K離 MA200 太近（僅 {gap:.1%}）"))
            continue
        kept.append(hit)
    return kept, len(skipped), skipped


def hit_key(hit: ScanHit) -> tuple[str, pd.Timestamp]:
    return hit.stock.symbol, hit.snapshot.timestamp
