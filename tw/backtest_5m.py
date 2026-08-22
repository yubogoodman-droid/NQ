"""台股五分 K 空頭回測：成交額前 300、5/10/20 空頭排列、跌破 MA200，且 15 分／小時 K 都在 MA20 之下。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.kline import fetch_bars_many
from tw.ranking import (
    DEFAULT_TURNOVER_TOP,
    RankedStock,
    fetch_daily_turnover_ranking,
    filter_by_price,
    filter_etfs,
    filter_financials,
    filter_telecoms,
    turnover_pool_label,
)
from tw.signals import AlertSnapshot, iter_5m_ma200_short_alerts

TAIPEI = ZoneInfo("Asia/Taipei")
FORWARD_BARS = (3, 6, 12)


@dataclass(frozen=True)
class BacktestConfig:
    days: int = 5
    top: int = DEFAULT_TURNOVER_TOP
    max_price: float = 650.0
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
class ForwardMove:
    bars: int
    later_close: float
    pnl_pct: float


@dataclass
class BacktestHit:
    day: date
    stock: RankedStock
    snapshot: AlertSnapshot
    frame: pd.DataFrame
    forwards: dict[int, ForwardMove] = field(default_factory=dict)
    eod: ForwardMove | None = None


@dataclass
class BacktestResult:
    scanned_at: datetime
    days: list[date]
    universes: dict[date, DayUniverse]
    hits: list[BacktestHit]
    skipped: list[tuple[date, RankedStock, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    top: int = DEFAULT_TURNOVER_TOP

    def hits_on(self, day: date) -> list[BacktestHit]:
        return [h for h in self.hits if h.day == day]


def run_5m_short_backtest(
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
            alerts = iter_5m_ma200_short_alerts(df, since=since, until=until)
            if not alerts:
                continue
            for snap in alerts:
                loc = df.index.get_indexer([snap.timestamp], method="nearest")[0]
                hits.append(
                    BacktestHit(
                        day=item.day,
                        stock=stock,
                        snapshot=snap,
                        frame=df,
                        forwards=_forward_moves(df, loc, snap.close),
                        eod=_eod_move(df, loc, snap.close),
                    )
                )

    hits.sort(key=lambda h: (h.day, h.snapshot.timestamp, h.stock.rank))
    return BacktestResult(
        scanned_at=datetime.now(TAIPEI),
        days=days,
        universes=by_day,
        hits=hits,
        skipped=skipped,
        top=cfg.top,
    )


def summarize_forwards(hits: list[BacktestHit]) -> dict[str, dict]:
    """空頭：價格續跌為正報酬。"""
    out: dict[str, dict] = {}
    for bars in FORWARD_BARS:
        pnls = [h.forwards[bars].pnl_pct for h in hits if bars in h.forwards]
        out[f"h{bars}"] = _stats(pnls)
    eod = [h.eod.pnl_pct for h in hits if h.eod is not None]
    out["eod"] = _stats(eod)
    return out


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "wr": 0.0, "avg": 0.0}
    wins = sum(1 for x in pnls if x > 0)
    return {
        "n": len(pnls),
        "wr": wins / len(pnls) * 100.0,
        "avg": sum(pnls) / len(pnls) * 100.0,
    }


def _forward_moves(df: pd.DataFrame, loc: int, entry: float) -> dict[int, ForwardMove]:
    out: dict[int, ForwardMove] = {}
    if loc < 0 or not entry:
        return out
    for bars in FORWARD_BARS:
        j = loc + bars
        if j >= len(df):
            continue
        later = float(df["close"].iloc[j])
        out[bars] = ForwardMove(bars=bars, later_close=later, pnl_pct=(entry - later) / entry)
    return out


def _eod_move(df: pd.DataFrame, loc: int, entry: float) -> ForwardMove | None:
    if loc < 0 or loc >= len(df) or not entry:
        return None
    day = pd.Timestamp(df.index[loc]).date()
    end = loc
    for i in range(loc + 1, len(df)):
        if pd.Timestamp(df.index[i]).date() != day:
            break
        end = i
    if end == loc:
        return None
    later = float(df["close"].iloc[end])
    return ForwardMove(bars=end - loc, later_close=later, pnl_pct=(entry - later) / entry)


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
            priced = filter_by_price(universe, cfg.max_price)
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
                f"{current.isoformat()} 上市＋上櫃 {len(universe)}"
                f"（{turnover_pool_label(cfg.top)}） → "
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
