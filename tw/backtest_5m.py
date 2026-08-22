"""台股五分 K 回測：成交額前 N、多頭發散、當根收盤站上 MA200 與十五分短均，且小時K在 MA20 之上。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_bars_many
from tw.ranking import (
    RankedStock,
    fetch_daily_turnover_ranking,
    filter_by_price,
    filter_etfs,
    filter_financials,
    filter_telecoms,
)
from tw.signals import AlertSnapshot, iter_5m_ma200_alerts

TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class BacktestConfig:
    days: int = 5
    top: int = 250
    max_price: float | None = 600.0
    exclude_etf: bool = True
    exclude_financial: bool = True
    exclude_telecom: bool = True
    kline_range: str = "1mo"
    today: date | None = None
    timeout: int = 20


@dataclass
class DayUniverse:
    day: date
    rank_time: str | None
    universe: list[RankedStock]
    candidates: list[RankedStock]
    price_dropped: int
    etf_dropped: int
    financial_dropped: int
    telecom_dropped: int


@dataclass
class BacktestHit:
    day: date
    stock: RankedStock
    snapshot: AlertSnapshot
    frame: pd.DataFrame


@dataclass
class BacktestResult:
    scanned_at: datetime
    days: list[date]
    universes: dict[date, DayUniverse]
    hits: list[BacktestHit]
    skipped: list[tuple[date, RankedStock, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def hits_on(self, day: date) -> list[BacktestHit]:
        return [h for h in self.hits if h.day == day]


def run_5m_backtest(
    config: BacktestConfig | None = None,
    session: requests.Session | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    sess = session or requests.Session()
    as_of = cfg.today or datetime.now(TAIPEI).date()
    universes = _load_session_universes(cfg, as_of, sess)
    days = [item.day for item in universes]
    by_day = {item.day: item for item in universes}

    symbols = list(
        dict.fromkeys(stock.symbol for item in universes for stock in item.candidates)
    )
    print(f"五分K下載 {len(symbols)} 檔（Yahoo {cfg.kline_range}）", flush=True)
    frames = fetch_bars_many(symbols, interval="5m", range_=cfg.kline_range, closed_only=True)

    hits: list[BacktestHit] = []
    skipped: list[tuple[date, RankedStock, str]] = []
    for item in universes:
        since = pd.Timestamp(item.day, tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        for stock in item.candidates:
            df = frames.get(stock.symbol)
            if df is None or df.empty:
                skipped.append((item.day, stock, "無五分 K 資料"))
                continue
            if len(df) < 201:
                skipped.append((item.day, stock, f"五分 K 不足 201 根（{len(df)}）"))
                continue
            alerts = iter_5m_ma200_alerts(df, since=since, until=until)
            if not alerts:
                continue
            for snap in alerts:
                hits.append(
                    BacktestHit(day=item.day, stock=stock, snapshot=snap, frame=df)
                )

    hits.sort(key=lambda h: (h.day, h.snapshot.timestamp, h.stock.rank))
    return BacktestResult(
        scanned_at=datetime.now(TAIPEI),
        days=days,
        universes=by_day,
        hits=hits,
        skipped=skipped,
    )


def _load_session_universes(
    cfg: BacktestConfig,
    as_of: date,
    sess: requests.Session,
) -> list[DayUniverse]:
    current = as_of
    found: list[DayUniverse] = []
    scanned = 0
    last_error: Exception | None = None
    while len(found) < cfg.days and scanned < 18:
        if current.weekday() < 5:
            scanned += 1
            try:
                universe, rank_time = fetch_daily_turnover_ranking(
                    current, top=cfg.top, session=sess, timeout=cfg.timeout
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"{current.isoformat()} 排行略過：{exc}", flush=True)
                current -= timedelta(days=1)
                continue
            if len(universe) < 50:
                last_error = ValueError(f"只有 {len(universe)} 檔")
                current -= timedelta(days=1)
                continue
            priced = (
                filter_by_price(universe, cfg.max_price)
                if cfg.max_price is not None
                else list(universe)
            )
            after_etf = filter_etfs(priced) if cfg.exclude_etf else priced
            after_fin = (
                filter_financials(after_etf) if cfg.exclude_financial else after_etf
            )
            candidates = (
                filter_telecoms(after_fin) if cfg.exclude_telecom else after_fin
            )
            found.append(
                DayUniverse(
                    day=current,
                    rank_time=rank_time,
                    universe=universe,
                    candidates=candidates,
                    price_dropped=len(universe) - len(priced),
                    etf_dropped=len(priced) - len(after_etf),
                    financial_dropped=len(after_etf) - len(after_fin),
                    telecom_dropped=len(after_fin) - len(candidates),
                )
            )
            print(
                f"{current.isoformat()} 成交額前 {len(universe)} → "
                f"價{found[-1].price_dropped}/ETF{found[-1].etf_dropped}/"
                f"金融{found[-1].financial_dropped}/電信{found[-1].telecom_dropped} "
                f"→ 掃描 {len(candidates)}",
                flush=True,
            )
        current -= timedelta(days=1)
    found.reverse()
    if len(found) < cfg.days:
        detail = f"（{last_error}）" if last_error else ""
        raise RuntimeError(f"湊不滿 {cfg.days} 個已公布交易日{detail}")
    return found
