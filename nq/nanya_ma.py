"""南亞科一分圖均線回測。

對應截圖上的 MA5/10/20/60/120/200：
盤整時短均糾結；啟動時價站上 MA20、短均剛排成 5>10>20 並開始扇開；
長均尚未被甩開。436 那種 5>10>20>60>120>200 大幅扇開是末端，不當進場。
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MA_PERIODS = (5, 10, 20, 60, 120, 200)


@dataclass(frozen=True)
class NanyaMaSignal:
    timestamp: pd.Timestamp
    bar_idx: int
    entry: float
    stop_loss: float
    target: float
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    ma120: float
    ma200: float
    short_span_pct: float
    ext_200_pct: float
    vol_ratio: float
    range_high: float
    range_low: float

    @property
    def risk(self) -> float:
        return self.entry - self.stop_loss


@dataclass
class NanyaMaTrade:
    symbol: str
    signal: NanyaMaSignal
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str
    pnl_points: float
    pnl_pct: float
    pnl_pct_net: float


@dataclass
class NanyaMaStrategy:
    """截圖邏輯：剛離開黏均就進，不等排成末端多頭。"""

    lookback: int = 60
    min_lookback: int = 30
    max_range_pct: float = 0.04
    max_coil_pct: float = 0.006
    min_short_span: float = 0.0015
    max_short_span: float = 0.008
    min_ext_200: float = 0.002
    max_ext_200: float = 0.035
    max_full_stack_pct: float = 0.04
    vol_mult: float = 1.5
    max_chase_pct: float = 0.06
    reward_r: float = 1.5
    atr_stop_buffer: float = 0.20
    max_bars_hold: int = 30
    tick_size: float = 0.5
    entry_after_minute: int = 10
    session_open_hour: int | None = 9

    def generate_signals(self, df: pd.DataFrame) -> list[NanyaMaSignal]:
        work = add_nanya_features(df)
        signals: list[NanyaMaSignal] = []
        start = max(200, self.lookback + 5)
        for i in range(start, len(work) - 1):
            snap = _snapshot_at(work, i, lookback=self.lookback, min_lookback=self.min_lookback)
            if snap is None:
                continue
            if not self._is_setup(work, i, snap):
                continue
            entry_idx = i + 1
            entry = _round_tick(float(work["open"].iloc[entry_idx]), self.tick_size)
            atr = float(work["atr"].iloc[i])
            if np.isnan(atr) or atr <= 0:
                atr = max(entry * 0.002, self.tick_size)
            stop = _round_tick(min(snap["ma20"], snap["range_low"]) - self.atr_stop_buffer * atr, self.tick_size)
            if entry <= stop:
                continue
            target = _round_tick(entry + self.reward_r * (entry - stop), self.tick_size)
            signals.append(
                NanyaMaSignal(
                    timestamp=work.index[entry_idx],
                    bar_idx=entry_idx,
                    entry=entry,
                    stop_loss=stop,
                    target=target,
                    close=snap["close"],
                    ma5=snap["ma5"],
                    ma10=snap["ma10"],
                    ma20=snap["ma20"],
                    ma60=snap["ma60"],
                    ma120=snap["ma120"],
                    ma200=snap["ma200"],
                    short_span_pct=snap["short_span_pct"],
                    ext_200_pct=snap["ext_200_pct"],
                    vol_ratio=snap["vol_ratio"],
                    range_high=snap["range_high"],
                    range_low=snap["range_low"],
                )
            )
        return signals

    def _is_setup(self, work: pd.DataFrame, i: int, snap: dict) -> bool:
        ts = work.index[i]
        if self.session_open_hour is not None:
            if ts.hour < self.session_open_hour or (
                ts.hour == self.session_open_hour and ts.minute < self.entry_after_minute
            ):
                return False
        if snap["day_move_pct"] > self.max_chase_pct:
            return False
        if not snap["aligned_short"]:
            return False
        if snap["close"] <= snap["ma20"]:
            return False
        if snap["close"] <= snap["range_high"]:
            return False
        if not (self.min_short_span <= snap["short_span_pct"] <= self.max_short_span):
            return False
        if not (self.min_ext_200 <= snap["ext_200_pct"] <= self.max_ext_200):
            return False
        if snap["aligned_full"] and snap["stack_span_pct"] > self.max_full_stack_pct:
            return False
        if snap["vol_ratio"] < self.vol_mult:
            return False
        if snap["coil_pct"] > self.max_coil_pct:
            return False
        if snap["range_pct"] > self.max_range_pct or snap["range_pct"] < 0.006:
            return False
        # MA20 開始往上，對應截圖突破後短中均翹起來
        if snap["ma20"] < float(work["ma20"].iloc[i - 3]):
            return False
        return True


def add_nanya_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for period in MA_PERIODS:
        out[f"ma{period}"] = close.rolling(period, min_periods=period).mean()
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=14).mean()
    out["vol_sma"] = out["volume"].rolling(20, min_periods=20).mean()
    out["session_date"] = [pd.Timestamp(ts).date() for ts in out.index]
    out["day_open"] = out.groupby("session_date")["open"].transform("first")
    return out


def _snapshot_at(work: pd.DataFrame, i: int, *, lookback: int, min_lookback: int) -> dict | None:
    needed = ("ma5", "ma10", "ma20", "ma60", "ma120", "ma200", "atr", "vol_sma", "day_open")
    if any(pd.isna(work[c].iloc[i]) for c in needed):
        return None
    left = max(0, i - lookback)
    if i - left < min_lookback:
        return None
    window = work.iloc[left:i]
    rng_high = float(window["high"].max())
    rng_low = float(window["low"].min())
    mid = (rng_high + rng_low) / 2
    if mid <= 0:
        return None
    close = float(work["close"].iloc[i])
    ma5 = float(work["ma5"].iloc[i])
    ma10 = float(work["ma10"].iloc[i])
    ma20 = float(work["ma20"].iloc[i])
    ma60 = float(work["ma60"].iloc[i])
    ma120 = float(work["ma120"].iloc[i])
    ma200 = float(work["ma200"].iloc[i])
    vol = float(work["volume"].iloc[i])
    vol_sma = float(work["vol_sma"].iloc[i])
    day_open = float(work["day_open"].iloc[i])
    if close <= 0 or ma200 <= 0 or vol_sma <= 0:
        return None
    # 盤整段短均要黏過：用突破前最後 20 根的平均短均寬度
    coil_end = window.iloc[-20:] if len(window) >= 20 else window
    coil_span = (coil_end[["ma5", "ma10", "ma20"]].max(axis=1) - coil_end[["ma5", "ma10", "ma20"]].min(axis=1)) / coil_end["close"]
    coil_pct = float(coil_span.median()) if coil_span.notna().any() else 1.0
    return {
        "close": close,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma200": ma200,
        "range_high": rng_high,
        "range_low": rng_low,
        "range_pct": (rng_high - rng_low) / mid,
        "coil_pct": coil_pct,
        "short_span_pct": (ma5 - ma20) / close,
        "stack_span_pct": (ma5 - ma200) / close,
        "ext_200_pct": close / ma200 - 1.0,
        "vol_ratio": vol / vol_sma,
        "day_move_pct": close / day_open - 1.0 if day_open > 0 else 0.0,
        "aligned_short": ma5 > ma10 > ma20,
        "aligned_full": ma5 > ma10 > ma20 > ma60 > ma120 > ma200,
    }


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def run_nanya_ma_backtest(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    strategy: NanyaMaStrategy | None = None,
    cost_bps: float = 8.0,
    flatten_minutes: tuple[int, int] | None = (13, 20),
) -> list[NanyaMaTrade]:
    strategy = strategy or NanyaMaStrategy()
    work = add_nanya_features(df)
    signals = strategy.generate_signals(work)
    results: list[NanyaMaTrade] = []
    busy_until = -1

    for sig in signals:
        if sig.bar_idx <= busy_until or sig.risk <= 0:
            continue
        end_idx = min(sig.bar_idx + strategy.max_bars_hold, len(work) - 1)
        exit_price = float(work["close"].iloc[end_idx])
        exit_time = work.index[end_idx]
        exit_reason = "time_stop"
        exit_idx = end_idx

        for i in range(sig.bar_idx, end_idx + 1):
            ts = work.index[i]
            if flatten_minutes is not None and (ts.hour, ts.minute) >= flatten_minutes and i > sig.bar_idx:
                exit_price = float(work["close"].iloc[i])
                exit_time = ts
                exit_reason = "session_flat"
                exit_idx = i
                break

            low = float(work["low"].iloc[i])
            high = float(work["high"].iloc[i])
            close = float(work["close"].iloc[i])
            ma20 = float(work["ma20"].iloc[i])

            if low <= sig.stop_loss:
                exit_price = sig.stop_loss
                exit_time = ts
                exit_reason = "stop_loss"
                exit_idx = i
                break
            if high >= sig.target:
                exit_price = sig.target
                exit_time = ts
                exit_reason = "take_profit"
                exit_idx = i
                break
            # 收盤跌破 MA20：截圖上第一道回檔線沒了
            if i > sig.bar_idx and close < ma20:
                exit_price = close
                exit_time = ts
                exit_reason = "lost_ma20"
                exit_idx = i
                break

        busy_until = exit_idx
        pnl_points = exit_price - sig.entry
        pnl_pct = pnl_points / sig.entry if sig.entry else 0.0
        results.append(
            NanyaMaTrade(
                symbol=symbol,
                signal=sig,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl_points=pnl_points,
                pnl_pct=pnl_pct,
                pnl_pct_net=pnl_pct - 2.0 * (cost_bps / 10_000.0),
            )
        )
    return results


def summarize_ma_trades(trades: list[NanyaMaTrade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "total_pnl_pct_net": 0.0,
            "avg_pnl_pct_net": 0.0,
            "expectancy_net": 0.0,
        }
    wins = sum(1 for t in trades if t.pnl_pct_net > 0)
    total_net = sum(t.pnl_pct_net for t in trades)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": wins / len(trades),
        "total_pnl_pct": sum(t.pnl_pct for t in trades),
        "total_pnl_pct_net": total_net,
        "avg_pnl_pct_net": total_net / len(trades),
        "expectancy_net": total_net / len(trades),
    }


def _fmt_ts(ts: pd.Timestamp) -> str:
    t = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
    return t.strftime("%m-%d %H:%M")


def build_nanya_ma_report_html(
    *,
    title: str,
    trades: list[NanyaMaTrade],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> str:
    overall = summarize_ma_trades(trades)

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    def wr(x: float) -> str:
        return f"{x * 100:.0f}%"

    sym_rows = "".join(
        f"<tr><td>{html.escape(sym)}</td><td>{bars}</td><td>{s['trades']}</td>"
        f"<td>{wr(s['win_rate'])}</td>"
        f"<td class=\"{'pos' if s['total_pnl_pct_net']>=0 else 'neg'}\">{pct(s['total_pnl_pct_net'])}</td></tr>"
        for sym, s, bars in symbol_stats
    ) or "<tr><td colspan='5'>沒有資料</td></tr>"

    trade_rows = "".join(
        f"<tr><td>{html.escape(t.symbol)}</td><td>{_fmt_ts(t.signal.timestamp)}</td>"
        f"<td>{t.signal.entry:.2f}</td>"
        f"<td>5 {t.signal.ma5:.2f} / 10 {t.signal.ma10:.2f} / 20 {t.signal.ma20:.2f}</td>"
        f"<td>60 {t.signal.ma60:.2f} / 200 {t.signal.ma200:.2f}</td>"
        f"<td>{t.signal.short_span_pct*100:.2f}%</td>"
        f"<td>{t.signal.ext_200_pct*100:.2f}%</td>"
        f"<td>{_fmt_ts(t.exit_time)}</td><td>{html.escape(t.exit_reason)}</td>"
        f"<td class=\"{'pos' if t.pnl_pct_net>=0 else 'neg'}\">{pct(t.pnl_pct_net)}</td></tr>"
        for t in trades[:80]
    ) or "<tr><td colspan='10'>沒有成交</td></tr>"

    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin:0; background:#0b0e11; color:#e6edf3;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif; }}
    .page {{ max-width:1080px; margin:0 auto; padding:20px 16px 40px; }}
    h1 {{ font-size:22px; margin:0 0 8px; }}
    h2 {{ font-size:16px; margin:22px 0 10px; }}
    p,li {{ color:#8b949e; line-height:1.6; font-size:14px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:12px; }}
    .k {{ color:#8b949e; font-size:12px; }}
    .v {{ font-size:20px; font-weight:700; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #21262d; padding:8px 6px; text-align:left; }}
    th {{ color:#8b949e; font-weight:600; }}
    .pos {{ color:#3ddc68; }} .neg {{ color:#ff7b72; }}
    @media (max-width:720px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p>對應南亞科一分圖 MA5/10/20/60/120/200：盤整短均要黏，進場要短均剛扇開、價剛離開 MA200。已排成末端多頭（436 那種）不追。</p>
    <div class="cards">
      <div class="card"><div class="k">成交</div><div class="v">{overall['trades']}</div></div>
      <div class="card"><div class="k">勝率（淨）</div><div class="v">{wr(overall['win_rate'])}</div></div>
      <div class="card"><div class="k">累計淨損益</div><div class="v {'pos' if overall['total_pnl_pct_net']>=0 else 'neg'}">{pct(overall['total_pnl_pct_net'])}</div></div>
      <div class="card"><div class="k">單筆期望值</div><div class="v {'pos' if overall['expectancy_net']>=0 else 'neg'}">{pct(overall['expectancy_net'])}</div></div>
    </div>
    <h2>標的</h2>
    <table>
      <thead><tr><th>代號</th><th>K 數</th><th>筆數</th><th>勝率</th><th>淨損益</th></tr></thead>
      <tbody>{sym_rows}</tbody>
    </table>
    <h2>前 80 筆</h2>
    <table>
      <thead><tr><th>標的</th><th>進場</th><th>價</th><th>短均</th><th>長均</th><th>5–20</th><th>離200</th><th>出場</th><th>原因</th><th>淨%</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
    <h2>規則</h2>
    <ul>{note_html}</ul>
  </div>
</body>
</html>
"""


def save_nanya_ma_report(
    output: str | Path,
    *,
    title: str,
    trades: list[NanyaMaTrade],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_nanya_ma_report_html(title=title, trades=trades, notes=notes, symbol_stats=symbol_stats),
        encoding="utf-8",
    )
    return out
