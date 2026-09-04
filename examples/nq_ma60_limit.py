#!/usr/bin/env python3
"""NQ 一分 K：破 2h 低後，半小時內 5/20 多頭排列突破 MA60，掛單 MA60 五分鐘做多。

規則:
  1. 跌破近兩小時低點
  2. 之後 30 根 1m 內：MA5 > MA20，且收盤突破 1m MA60
  3. 在突破當下的 MA60 掛限價買單，只掛 5 分鐘
  4. 5 分鐘內回踩成交才進場；逾時取消
  5. 停損在破底低點；目標 2R

用法:
  python3 examples/nq_ma60_limit.py backtest --period 8d --html output/nq_ma60_limit.html
  python3 examples/nq_ma60_limit.py backtest --period 30d --pages
  python3 examples/nq_ma60_limit.py alert --dry-run --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    CONFIG_ENV,
    ET,
    Signal,
    TradeResult,
    _build_m5_ma60_slope5,
    env,
    load_bars,
    load_dotenv,
    load_yfinance,
    quality_from_slopes,
    rolling_min_prev,
    simulate as _simulate_reclaim,
    sma,
    summarize_trades,
    tg_send,
    to_et,
    write_html_report,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma60-limit" / "index.html"
VIEW_BRANCH = "cursor/nq-ma60-limit-63a8"
STATE_PATH = ROOT / "tg_ma60_limit_state.json"
RULES = "破 2h 低後 30 分鐘內：MA5>MA20 且 1m 收盤突破 MA60，再把限價單掛在 MA60，只留 5 分鐘；逾時不進。停損在破底低點，不用 MA60 上移停損。"


def simulate(df, signals, **kwargs):
    """Same exits as MA Reclaim, but never trail the stop up to MA60."""
    kwargs["use_ma60_up_stop"] = False
    return _simulate_reclaim(df, signals, **kwargs)


@dataclass
class PendingOrder:
    break_idx: int
    setup_idx: int
    expire_idx: int
    limit_price: float
    break_low: float
    two_hr_low: float
    ma5: float
    ma20: float
    ma60: float


def detect_signals(
    df: pd.DataFrame,
    setup_window: int = 30,
    limit_bars: int = 5,
    two_hour_bars: int = 120,
    target_r: float = 2.0,
    min_break_depth: float = 10.0,
    min_entry_gap: int = 15,
    ma60_slope_bars: int = 5,
    pending: Optional[List[PendingOrder]] = None,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """破 2h 低 → 半小時內 5/20 多頭 + 收盤破 MA60 → 掛單 MA60 五分鐘。"""
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    low = df["Low"].to_numpy(float)

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    ma60 = sma(close, 60)
    ma200 = sma(close, 200)
    m5_ma60_slope5 = _build_m5_ma60_slope5(df, ma60_slope_bars)
    two_hr_low = rolling_min_prev(low, two_hour_bars)

    signals: List[Signal] = []
    last_entry = -(10**9)
    n = len(close)
    warmup = max(200, two_hour_bars)
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n:
        if np.isnan(two_hr_low[i]) or np.isnan(ma60[i]):
            i += 1
            continue

        support = float(two_hr_low[i])
        if low[i] >= support:
            i += 1
            continue

        bump("break")
        break_idx = i
        break_low = float(low[i])
        if support - break_low < min_break_depth:
            bump("shallow")
            i += 1
            continue
        bump("deep_break")

        entered = False
        j = break_idx + 1
        last_j = min(break_idx + setup_window, n - 1)
        while j <= last_j:
            if (
                j < 1
                or np.isnan(ma5[j])
                or np.isnan(ma20[j])
                or np.isnan(ma60[j])
                or np.isnan(ma60[j - 1])
            ):
                j += 1
                continue

            bull_stack = float(ma5[j]) > float(ma20[j])
            crossed = float(close[j]) > float(ma60[j]) and float(close[j - 1]) <= float(ma60[j - 1])
            if not (bull_stack and crossed):
                j += 1
                continue

            bump("setup")
            limit = float(ma60[j])
            window_end = j + limit_bars
            scan_end = min(window_end, n - 1)
            filled = False

            for k in range(j + 1, scan_end + 1):
                if float(low[k]) > limit:
                    continue
                if k - last_entry < min_entry_gap:
                    bump("skip_entry_gap")
                    continue
                fill = float(open_[k]) if float(open_[k]) <= limit else limit
                stop = break_low
                risk = fill - stop
                if risk <= 0:
                    bump("skip_bad_risk")
                    continue

                slope5 = 0.0
                if j >= ma60_slope_bars and not np.isnan(ma60[j - ma60_slope_bars]):
                    slope5 = float(ma60[j]) - float(ma60[j - ma60_slope_bars])
                m1_ma5_s5 = 0.0
                if j >= ma60_slope_bars and not np.isnan(ma5[j - ma60_slope_bars]):
                    m1_ma5_s5 = float(ma5[j]) - float(ma5[j - ma60_slope_bars])
                m5_s5 = float(m5_ma60_slope5[j]) if not np.isnan(m5_ma60_slope5[j]) else float("nan")
                q_score, q_grade = quality_from_slopes(m1_ma5_s5, slope5, m5_s5)

                bump("taken")
                signals.append(
                    Signal(
                        break_idx,
                        k,
                        fill,
                        stop,
                        fill + risk * target_r,
                        break_low,
                        support,
                        0.0,
                        break_low,
                        "ma60_limit",
                        float(ma5[j]),
                        float(ma10[j]) if not np.isnan(ma10[j]) else 0.0,
                        float(ma20[j]),
                        float(ma30[j]) if not np.isnan(ma30[j]) else 0.0,
                        float(ma60[j]),
                        float(ma200[j]) if not np.isnan(ma200[j]) else 0.0,
                        quality=q_grade,
                        quality_score=q_score,
                        m1_ma5_slope5=m1_ma5_s5,
                        m1_ma60_slope5=float(slope5),
                        m5_ma60_slope5=m5_s5 if not np.isnan(m5_s5) else 0.0,
                        setup_idx=j,
                        limit_price=limit,
                    )
                )
                last_entry = k
                entered = True
                filled = True
                i = k + 1
                break

            if filled:
                break
            if window_end > n - 1:
                bump("pending")
                if pending is not None:
                    pending.append(
                        PendingOrder(
                            break_idx=break_idx,
                            setup_idx=j,
                            expire_idx=j + limit_bars,
                            limit_price=limit,
                            break_low=break_low,
                            two_hr_low=support,
                            ma5=float(ma5[j]),
                            ma20=float(ma20[j]),
                            ma60=limit,
                        )
                    )
                break
            bump("expired")
            j += 1

        if not entered:
            i = break_idx + 1

    return signals


def write_view_html(src: Path, branch: str = VIEW_BRANCH) -> Path:
    rel = src.parent.relative_to(REPO_ROOT).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8")
    if "圖是靜態" not in text:
        text = text.replace(
            "</h1>\n<p class=\"muted\">",
            "</h1>\n<p class=\"muted\">圖是靜態 1m K 線。手機請往下捲。</p>\n<p class=\"muted\">",
            1,
        )
    text = text.replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _ts_et(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize("UTC").tz_convert(ET)
    return ts.tz_convert(ET)


def _load_limit_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": [], "alerted_exits": [], "alerted_setups": [], "alerted_expires": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": [], "alerted_setups": [], "alerted_expires": []}


def _save_limit_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_key(df, pending: PendingOrder) -> str:
    ts = _ts_et(df.index[pending.setup_idx])
    return f"setup|{ts.isoformat()}|{pending.limit_price:.2f}"


def expire_key(df, pending: PendingOrder) -> str:
    ts = _ts_et(df.index[pending.setup_idx])
    return f"expire|{ts.isoformat()}|{pending.limit_price:.2f}"


def entry_key(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    return f"{ts.isoformat()}|{sig.entry_price:.2f}"


def exit_key(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    return f"{et.isoformat()}->{xt.isoformat()}|{tr.exit_reason}|{tr.pnl_points:.2f}"


def fmt_setup(df, pending: PendingOrder) -> str:
    ts = _ts_et(df.index[pending.setup_idx])
    br = _ts_et(df.index[pending.break_idx])
    last = float(df["Close"].iloc[-1])
    left = max(0, pending.expire_idx - (len(df) - 1))
    return (
        f"📌 <b>掛單 MA60 做多</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"限價: <code>{pending.limit_price:.2f}</code>\n"
        f"有效: <b>{left} 分鐘</b>（逾時取消）\n"
        f"破底: <code>{br.strftime('%H:%M')}</code> low={pending.break_low:.2f}\n"
        f"MA5 {pending.ma5:.1f} &gt; MA20 {pending.ma20:.1f}\n"
        f"現價: <code>{last:.2f}</code>\n"
        f"#掛單 #MA60 #NQ"
    )


def fmt_entry(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    br = _ts_et(df.index[sig.break_idx])
    su = _ts_et(df.index[sig.setup_idx]) if sig.setup_idx else ts
    risk = sig.entry_price - sig.stop_price
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>MA60 掛單成交</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"品質: <b>Q{sig.quality}</b> ({sig.quality_score}/3)\n"
        f"進場: <code>{sig.entry_price:.2f}</code>（掛單 {sig.limit_price:.2f}）\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> ({r_mult:.1f}R)\n"
        f"破底: <code>{br.strftime('%H:%M')}</code> → 破MA60 <code>{su.strftime('%H:%M')}</code>\n"
        f"現價: <code>{last:.2f}</code>\n"
        f"#破底翻 #MA60 #NQ #Q{sig.quality}"
    )


def fmt_expire(df, pending: PendingOrder) -> str:
    ts = _ts_et(df.index[pending.setup_idx])
    return (
        f"⚪ <b>掛單逾時取消</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"限價: <code>{pending.limit_price:.2f}</code>\n"
        f"超過 5 分鐘未回踩，不進場。\n"
        f"#取消 #MA60"
    )


def fmt_exit(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_points > 0 else ("⚪" if tr.pnl_points == 0 else "🔴")
    return (
        f"{emoji} <b>MA60 掛單出場</b>\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.2f}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.2f}\n"
        f"原因: <b>{tr.exit_reason}</b>\n"
        f"盈虧: <b>{tr.pnl_points:+.1f} pts</b> · Q{tr.quality}\n"
        f"#MA60 #出場"
    )


def _collect_expired(df, setup_window: int = 30, limit_bars: int = 5) -> List[PendingOrder]:
    """Replay setups that already expired (for Telegram 取消通知)."""
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    two_hr_low = rolling_min_prev(low, 120)
    n = len(close)
    out: List[PendingOrder] = []
    i = max(200, 120)
    while i < n:
        if np.isnan(two_hr_low[i]) or low[i] >= float(two_hr_low[i]):
            i += 1
            continue
        if float(two_hr_low[i]) - float(low[i]) < 10.0:
            i += 1
            continue
        last_j = min(i + setup_window, n - 1)
        for j in range(i + 1, last_j + 1):
            if j < 1 or np.isnan(ma60[j]) or np.isnan(ma60[j - 1]):
                continue
            if not (float(ma5[j]) > float(ma20[j])):
                continue
            if not (float(close[j]) > float(ma60[j]) and float(close[j - 1]) <= float(ma60[j - 1])):
                continue
            window_end = j + limit_bars
            if window_end > n - 1:
                break
            limit = float(ma60[j])
            touched = any(float(low[k]) <= limit for k in range(j + 1, window_end + 1))
            if not touched:
                out.append(
                    PendingOrder(
                        break_idx=i,
                        setup_idx=j,
                        expire_idx=window_end,
                        limit_price=limit,
                        break_low=float(low[i]),
                        two_hr_low=float(two_hr_low[i]),
                        ma5=float(ma5[j]),
                        ma20=float(ma20[j]),
                        ma60=limit,
                    )
                )
        i += 1
    return out


def scan_once(
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    alert_exits: bool,
    seed_alert: bool,
    lookback_hours: float,
    period: str = "5d",
) -> None:
    df = to_et(load_yfinance("NQ=F", "1m", period))
    pending: List[PendingOrder] = []
    sigs = detect_signals(df, pending=pending)
    trades = simulate(df, sigs)
    expired = _collect_expired(df)
    state = _load_limit_state()
    alerted_e: Set[str] = set(state.get("alerted_entries") or [])
    alerted_x: Set[str] = set(state.get("alerted_exits") or [])
    alerted_s: Set[str] = set(state.get("alerted_setups") or [])
    alerted_z: Set[str] = set(state.get("alerted_expires") or [])
    now = datetime.now(ET)
    cutoff = now.timestamp() - lookback_hours * 3600
    first_run = not STATE_PATH.exists() or (not alerted_e and not state.get("initialized"))

    new_setups = []
    for p in pending:
        k = setup_key(df, p)
        ts = _ts_et(df.index[p.setup_idx])
        if ts.timestamp() < cutoff or k in alerted_s:
            continue
        new_setups.append((k, p, ts))

    new_entries = []
    for sig in sigs:
        k = entry_key(df, sig)
        ts = _ts_et(df.index[sig.entry_idx])
        if ts.timestamp() < cutoff:
            alerted_e.add(k)
            continue
        if k in alerted_e:
            continue
        new_entries.append((k, sig, ts))

    new_expires = []
    for p in expired:
        k = expire_key(df, p)
        ts = _ts_et(df.index[p.setup_idx])
        if ts.timestamp() < cutoff or k in alerted_z:
            continue
        new_expires.append((k, p, ts))

    if first_run and not seed_alert:
        for k, _, _ in new_setups:
            alerted_s.add(k)
        for k, _, _ in new_entries:
            alerted_e.add(k)
        for k, _, _ in new_expires:
            alerted_z.add(k)
        for tr in trades:
            alerted_x.add(exit_key(df, tr))
        state.update(
            {
                "alerted_entries": sorted(alerted_e)[-200:],
                "alerted_exits": sorted(alerted_x)[-200:],
                "alerted_setups": sorted(alerted_s)[-200:],
                "alerted_expires": sorted(alerted_z)[-200:],
                "initialized": True,
                "last_scan": now.isoformat(),
            }
        )
        _save_limit_state(state)
        print(
            f"[{now.strftime('%H:%M:%S')} ET] init: marked {len(new_setups)} setups / "
            f"{len(new_entries)} fills, bars={len(df)} last={df['Close'].iloc[-1]:.2f}"
        )
        return

    sent = 0
    for k, p, ts in new_setups:
        ok = tg_send(token, chat_id, fmt_setup(df, p), dry_run=dry_run)
        if ok:
            alerted_s.add(k)
            sent += 1
            print(f"[setup] {ts} limit={p.limit_price:.2f}")

    for k, sig, ts in new_entries:
        ok = tg_send(token, chat_id, fmt_entry(df, sig), dry_run=dry_run)
        if ok:
            alerted_e.add(k)
            sent += 1
            print(f"[fill] {ts} Q{sig.quality} @ {sig.entry_price:.2f}")

    for k, p, ts in new_expires:
        ok = tg_send(token, chat_id, fmt_expire(df, p), dry_run=dry_run)
        if ok:
            alerted_z.add(k)
            sent += 1
            print(f"[expire] {ts} limit={p.limit_price:.2f}")

    if alert_exits:
        for tr in trades:
            k = exit_key(df, tr)
            xt = _ts_et(df.index[tr.exit_idx])
            if xt.timestamp() < cutoff:
                alerted_x.add(k)
                continue
            if k in alerted_x:
                continue
            ok = tg_send(token, chat_id, fmt_exit(df, tr), dry_run=dry_run)
            if ok:
                alerted_x.add(k)
                sent += 1
                print(f"[exit] {xt} {tr.exit_reason} {tr.pnl_points:+.1f}")

    state.update(
        {
            "alerted_entries": sorted(alerted_e)[-200:],
            "alerted_exits": sorted(alerted_x)[-200:],
            "alerted_setups": sorted(alerted_s)[-200:],
            "alerted_expires": sorted(alerted_z)[-200:],
            "initialized": True,
            "last_scan": now.isoformat(),
        }
    )
    _save_limit_state(state)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] scan ok bars={len(df)} "
        f"sigs={len(sigs)} pending={len(pending)} new_sent={sent} last={df['Close'].iloc[-1]:.2f}"
    )


def cmd_backtest(args) -> int:
    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    funnel: Dict[str, int] = {}
    pending: List[PendingOrder] = []
    sigs = detect_signals(df, funnel=funnel, pending=pending)
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if funnel:
        print(
            "funnel "
            f"break={funnel.get('break', 0)} deep={funnel.get('deep_break', 0)} "
            f"setup={funnel.get('setup', 0)} taken={funnel.get('taken', 0)} "
            f"expired={funnel.get('expired', 0)} pending={funnel.get('pending', 0)}"
        )
    for q, info in stats.get("by_quality", {}).items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        setup = ""
        if t.signal.setup_idx:
            setup = f" setup={df.index[t.signal.setup_idx].strftime('%H:%M')}"
        print(
            f"[{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}{setup} limit={t.signal.limit_price:.2f}"
        )
    if pending:
        print(f"live pending={len(pending)}")
        for p in pending:
            print(
                f"  掛單 {df.index[p.setup_idx].strftime('%m-%d %H:%M')} "
                f"MA60={p.limit_price:.2f} 剩 {max(0, p.expire_idx - (len(df) - 1))}m"
            )

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(
            html_path,
            df,
            trades,
            args.symbol,
            args.period,
            funnel=funnel,
            title="破底後掛單 MA60",
            rules=RULES,
        )
        print(f"html={out}")
        if getattr(args, "pages", False):
            view = write_view_html(out)
            print(f"view={view}")
    return 0


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2

    if args.test:
        ok = tg_send(
            token or "",
            chat_id or "",
            f"✅ MA60 掛單 bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    print(
        f"MA60 limit TG | interval={args.interval}s | exits={not args.no_exits} | "
        f"dry_run={args.dry_run} | lookback={args.lookback_hours}h"
    )
    while True:
        try:
            scan_once(
                token or "",
                chat_id or "",
                dry_run=args.dry_run,
                alert_exits=not args.no_exits,
                seed_alert=args.seed_alert,
                lookback_hours=args.lookback_hours,
                period=args.period,
            )
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(max(15, args.interval))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 破 2h 低後掛單 MA60 五分鐘")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-limit/index.html")
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("alert", help="Telegram 輪詢（掛單 / 成交 / 逾時）")
    a.add_argument("--interval", type=int, default=None)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--no-exits", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=None)
    a.add_argument("--period", default="5d")
    a.set_defaults(func=cmd_alert)

    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma60-limit/index.html")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "alert":
        if args.interval is None:
            args.interval = int(env("POLL_SECONDS", "60") or 60)
        if args.lookback_hours is None:
            args.lookback_hours = float(env("LOOKBACK_HOURS", "36") or 36)
        return cmd_alert(args)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
