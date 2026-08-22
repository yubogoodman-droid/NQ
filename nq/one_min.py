"""一分 K 棒型態策略與回測。"""

from __future__ import annotations

import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nq.candles import (
    CORE_PATTERNS,
    CandlePattern,
    add_candle_features,
    detect_candle_patterns,
    pick_best_pattern,
)


@dataclass(frozen=True)
class CandleSignal:
    timestamp: pd.Timestamp
    side: str
    entry: float
    stop_loss: float
    target: float
    pattern: CandlePattern
    bar_idx: int

    @property
    def risk(self) -> float:
        if self.side == "long":
            return self.entry - self.stop_loss
        return self.stop_loss - self.entry


@dataclass
class CandleTrade:
    symbol: str
    signal: CandleSignal
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str
    pnl_points: float
    pnl_pct: float
    pnl_pct_net: float


@dataclass
class OneMinCandleStrategy:
    """
    型態收盤確認，下一根開盤進場。
    停損：型態極值再加一點 ATR 緩衝。
    停利：reward_r 倍風險。
    時間停：max_bars_hold 根後收盤出場。
    """

    reward_r: float = 1.5
    atr_stop_buffer: float = 0.15
    max_bars_hold: int = 20
    max_chase_pct: float = 0.06
    tick_size: float = 0.01

    def generate_signals(self, df: pd.DataFrame) -> list[CandleSignal]:
        work = add_candle_features(df)
        raw = detect_candle_patterns(work, max_chase_pct=self.max_chase_pct)
        by_bar: dict[int, list[CandlePattern]] = defaultdict(list)
        for p in raw:
            by_bar[p.end_idx].append(p)

        signals: list[CandleSignal] = []
        for end_idx in sorted(by_bar):
            entry_idx = end_idx + 1
            if entry_idx >= len(work):
                continue
            pattern = pick_best_pattern(by_bar[end_idx])
            if pattern is None:
                continue
            atr = float(work["atr"].iloc[end_idx])
            if pd.isna(atr) or atr <= 0:
                atr = max(pattern.high - pattern.low, self.tick_size)

            entry = self._round(float(work["open"].iloc[entry_idx]))
            if pattern.side == "long":
                stop = self._round(pattern.low - self.atr_stop_buffer * atr)
                if entry <= stop:
                    continue
                target = self._round(entry + self.reward_r * (entry - stop))
            else:
                stop = self._round(pattern.high + self.atr_stop_buffer * atr)
                if stop <= entry:
                    continue
                target = self._round(entry - self.reward_r * (stop - entry))

            signals.append(
                CandleSignal(
                    timestamp=work.index[entry_idx],
                    side=pattern.side,
                    entry=entry,
                    stop_loss=stop,
                    target=target,
                    pattern=pattern,
                    bar_idx=entry_idx,
                )
            )
        return signals

    def _round(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size


def run_one_min_backtest(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    strategy: OneMinCandleStrategy | None = None,
    cost_bps: float = 8.0,
    flatten_minutes: tuple[int, int] | None = None,
) -> list[CandleTrade]:
    """
    單倉回測。先檢查停損再停利（同根偏保守）。
    cost_bps：單邊成本（bps），買賣各扣一次。
    flatten_minutes：(hour, minute) 之後不再持倉，用當根收盤平。
    """
    strategy = strategy or OneMinCandleStrategy()
    signals = sorted(strategy.generate_signals(df), key=lambda s: s.bar_idx)
    results: list[CandleTrade] = []
    busy_until = -1

    for sig in signals:
        if sig.bar_idx <= busy_until:
            continue
        if sig.risk <= 0:
            continue

        end_idx = min(sig.bar_idx + strategy.max_bars_hold, len(df) - 1)
        exit_price = float(df["close"].iloc[end_idx])
        exit_time = df.index[end_idx]
        exit_reason = "time_stop"
        exit_idx = end_idx

        for i in range(sig.bar_idx, end_idx + 1):
            ts = df.index[i]
            if flatten_minutes is not None:
                hm = (ts.hour, ts.minute)
                if hm >= flatten_minutes and i > sig.bar_idx:
                    exit_price = float(df["close"].iloc[i])
                    exit_time = ts
                    exit_reason = "session_flat"
                    exit_idx = i
                    break

            low = float(df["low"].iloc[i])
            high = float(df["high"].iloc[i])
            # 進場當根：開盤已成交，仍可能當根打到停損／停利
            if sig.side == "long":
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
            else:
                if high >= sig.stop_loss:
                    exit_price = sig.stop_loss
                    exit_time = ts
                    exit_reason = "stop_loss"
                    exit_idx = i
                    break
                if low <= sig.target:
                    exit_price = sig.target
                    exit_time = ts
                    exit_reason = "take_profit"
                    exit_idx = i
                    break

        busy_until = exit_idx
        if sig.side == "long":
            pnl_points = exit_price - sig.entry
        else:
            pnl_points = sig.entry - exit_price
        pnl_pct = pnl_points / sig.entry if sig.entry else 0.0
        pnl_pct_net = pnl_pct - 2.0 * (cost_bps / 10_000.0)

        results.append(
            CandleTrade(
                symbol=symbol,
                signal=sig,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl_points=pnl_points,
                pnl_pct=pnl_pct,
                pnl_pct_net=pnl_pct_net,
            )
        )
    return results


def summarize_trades(trades: list[CandleTrade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "total_pnl_pct_net": 0.0,
            "avg_pnl_pct": 0.0,
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
        "avg_pnl_pct": sum(t.pnl_pct for t in trades) / len(trades),
        "avg_pnl_pct_net": total_net / len(trades),
        "expectancy_net": total_net / len(trades),
    }


def summarize_by_pattern(trades: list[CandleTrade]) -> list[dict]:
    groups: dict[str, list[CandleTrade]] = defaultdict(list)
    for t in trades:
        groups[t.signal.pattern.name].append(t)
    rows = []
    for name, items in groups.items():
        stats = summarize_trades(items)
        stats["name"] = name
        stats["name_zh"] = items[0].signal.pattern.name_zh
        stats["side"] = items[0].signal.side
        rows.append(stats)
    rows.sort(key=lambda r: r["expectancy_net"], reverse=True)
    return rows


def _fmt_ts(ts: pd.Timestamp) -> str:
    t = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
    return t.strftime("%m-%d %H:%M")


def build_one_min_report_html(
    *,
    title: str,
    trades: list[CandleTrade],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> str:
    overall = summarize_trades(trades)
    by_pat = summarize_by_pattern(trades)
    core_trades = [t for t in trades if t.signal.pattern.name in CORE_PATTERNS]
    core = summarize_trades(core_trades)

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    def wr(x: float) -> str:
        return f"{x * 100:.0f}%"

    pat_rows = "".join(
        f"<tr><td>{html.escape(r['name_zh'])}</td><td><code>{html.escape(r['name'])}</code></td>"
        f"<td>{r['trades']}</td><td>{wr(r['win_rate'])}</td>"
        f"<td class=\"{'pos' if r['total_pnl_pct_net']>=0 else 'neg'}\">{pct(r['total_pnl_pct_net'])}</td>"
        f"<td class=\"{'pos' if r['expectancy_net']>=0 else 'neg'}\">{pct(r['expectancy_net'])}</td></tr>"
        for r in by_pat
    ) or "<tr><td colspan='6'>沒有成交</td></tr>"

    sym_rows = "".join(
        f"<tr><td>{html.escape(sym)}</td><td>{bars}</td><td>{s['trades']}</td>"
        f"<td>{wr(s['win_rate'])}</td>"
        f"<td class=\"{'pos' if s['total_pnl_pct_net']>=0 else 'neg'}\">{pct(s['total_pnl_pct_net'])}</td></tr>"
        for sym, s, bars in symbol_stats
    )

    trade_rows = "".join(
        f"<tr><td>{html.escape(t.symbol)}</td><td>{html.escape(t.signal.pattern.name_zh)}</td>"
        f"<td>{'多' if t.signal.side=='long' else '空'}</td>"
        f"<td>{_fmt_ts(t.signal.timestamp)}</td><td>{t.signal.entry:.2f}</td>"
        f"<td>{_fmt_ts(t.exit_time)}</td><td>{t.exit_price:.2f}</td>"
        f"<td>{html.escape(t.exit_reason)}</td>"
        f"<td class=\"{'pos' if t.pnl_pct_net>=0 else 'neg'}\">{pct(t.pnl_pct_net)}</td></tr>"
        for t in trades[:80]
    ) or "<tr><td colspan='9'>沒有成交</td></tr>"

    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0; background: #0b0e11; color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }}
    .page {{ max-width: 960px; margin: 0 auto; padding: 20px 16px 40px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 16px; margin: 22px 0 10px; }}
    p, li {{ color: #8b949e; line-height: 1.6; font-size: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }}
    .cards .card:nth-child(n+5) {{ border-color: #3d4a5c; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 12px; }}
    .k {{ color: #8b949e; font-size: 12px; }}
    .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #21262d; padding: 8px 6px; text-align: left; }}
    th {{ color: #8b949e; font-weight: 600; }}
    .pos {{ color: #3ddc68; }}
    .neg {{ color: #ff7b72; }}
    code {{ color: #79c0ff; }}
    @media (max-width: 720px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p>型態收盤確認、下一根開盤進場；停損在型態極值，停利 1.5R，最多抱 20 根一分 K。已扣單邊成本。</p>
    <div class="cards">
      <div class="card"><div class="k">全部成交</div><div class="v">{overall['trades']}</div></div>
      <div class="card"><div class="k">全部勝率</div><div class="v">{wr(overall['win_rate'])}</div></div>
      <div class="card"><div class="k">全部淨損益</div><div class="v {'pos' if overall['total_pnl_pct_net']>=0 else 'neg'}">{pct(overall['total_pnl_pct_net'])}</div></div>
      <div class="card"><div class="k">全部期望值</div><div class="v {'pos' if overall['expectancy_net']>=0 else 'neg'}">{pct(overall['expectancy_net'])}</div></div>
      <div class="card"><div class="k">核心型態成交</div><div class="v">{core['trades']}</div></div>
      <div class="card"><div class="k">核心勝率</div><div class="v">{wr(core['win_rate'])}</div></div>
      <div class="card"><div class="k">核心淨損益</div><div class="v {'pos' if core['total_pnl_pct_net']>=0 else 'neg'}">{pct(core['total_pnl_pct_net'])}</div></div>
      <div class="card"><div class="k">核心期望值</div><div class="v {'pos' if core['expectancy_net']>=0 else 'neg'}">{pct(core['expectancy_net'])}</div></div>
    </div>
    <p>核心型態：盤整放量突破、三白兵／三烏鴉、晨星／夜星、吞噬、刺透／烏雲。鑷子、錘子、光頭光腳單獨列，一分 K 雜訊通常較大。</p>
    <h2>各型態（依淨期望值排序）</h2>
    <table>
      <thead><tr><th>型態</th><th>代碼</th><th>筆數</th><th>勝率</th><th>淨損益</th><th>期望值</th></tr></thead>
      <tbody>{pat_rows}</tbody>
    </table>
    <h2>標的</h2>
    <table>
      <thead><tr><th>代號</th><th>K 數</th><th>筆數</th><th>勝率</th><th>淨損益</th></tr></thead>
      <tbody>{sym_rows}</tbody>
    </table>
    <h2>前 80 筆明細</h2>
    <table>
      <thead><tr><th>標的</th><th>型態</th><th>向</th><th>進場</th><th>價</th><th>出場</th><th>價</th><th>原因</th><th>淨%</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
    <h2>規則與限制</h2>
    <ul>{note_html}</ul>
  </div>
</body>
</html>
"""


def save_one_min_report(
    output: str | Path,
    *,
    title: str,
    trades: list[CandleTrade],
    notes: list[str],
    symbol_stats: list[tuple[str, dict, int]],
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_one_min_report_html(title=title, trades=trades, notes=notes, symbol_stats=symbol_stats),
        encoding="utf-8",
    )
    return out
