#!/usr/bin/env python3
"""台股 1h 破底翻 — 只推 Telegram 訊號，不下單。

對齊 examples/tw_1h_reclaim.py 的元晶型規則。盤中 09:00–13:30 Asia/Taipei
才評估／發送；忽略 08:30–09:00 盤前試撮。只用已收完的 1h K，進行中的
那根不會觸發。每個訊號寫入 output/tw_1h_reclaim_alert_state.json，重啟不重發。

Telegram 憑證放 repo 根目錄 tg_config.env（勿提交）:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...

目標機器人 @Tw6688bot、chat id 1297264584（寫在 env，不要寫進程式）。

    python3 examples/watch_tw_1h_reclaim.py --dry-run --once
    python3 examples/watch_tw_1h_reclaim.py --test --dry-run
    python3 examples/watch_tw_1h_reclaim.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import session_mask  # noqa: E402
from tw_1h_reclaim import (  # noqa: E402
    REPO,
    TPE,
    ReclaimParams,
    Signal,
    detect_signals,
    fetch_top_turnover,
    fetch_yahoo_1h,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    loose_params,
    resolve_twse_date,
    strict_params,
)

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"

DEFAULT_STATE_PATH = REPO / "output" / "tw_1h_reclaim_alert_state.json"
DEFAULT_LIMIT = 200
DEFAULT_MAX_PRICE = 1000.0
DEFAULT_RANGE = "2mo"
DEFAULT_GRACE_SECONDS = 20
SEEN_CAP = 4000
MIN_FETCH_BARS = 40
MIN_COMPLETED_BARS = 24
SESSION_OPEN = (9, 0)
SESSION_CLOSE = (13, 30)
# Yahoo 60m 在本策略裡是 K 開盤時間：09/10/11/12/13。13:00 那根 13:30 收完。
BAR_CLOSE_BY_OPEN = {
    (9, 0): (10, 0),
    (10, 0): (11, 0),
    (11, 0): (12, 0),
    (12, 0): (13, 0),
    (13, 0): (13, 30),
    (13, 30): (13, 30),
}
EVAL_CLOCKS = ((10, 0), (11, 0), (12, 0), (13, 0), (13, 30))


# ---------------------------------------------------------------------------
# Session / completed 1h bars
# ---------------------------------------------------------------------------


def to_tpe(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(TPE)
    return t.tz_convert(TPE)


def session_minutes(now: datetime) -> int:
    now_t = to_tpe(now)
    return int(now_t.hour) * 60 + int(now_t.minute)


def in_regular_session(now: datetime) -> bool:
    """Regular TWSE session 09:00–13:30 Asia/Taipei, weekdays only.

    Pre-open / test matching (before 09:00) is out. The 13:30 minute is in,
    so the 13:00–13:30 bar can be evaluated at the close.
    """
    now_t = to_tpe(now)
    if int(now_t.weekday()) >= 5:
        return False
    mins = session_minutes(now_t)
    open_m = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_m = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return open_m <= mins <= close_m


def bar_close_time(bar_open) -> pd.Timestamp:
    ts = to_tpe(bar_open).replace(second=0, microsecond=0)
    hm = (int(ts.hour), int(ts.minute))
    close_hm = BAR_CLOSE_BY_OPEN.get(hm)
    if close_hm is None:
        return ts + pd.Timedelta(hours=1)
    ch, cm = close_hm
    return ts.replace(hour=ch, minute=cm)


def is_completed_1h_bar(bar_open, now: datetime) -> bool:
    return to_tpe(now) >= bar_close_time(bar_open)


def drop_incomplete_1h_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop the in-progress candle and anything that has not closed yet."""
    if df is None or df.empty:
        return df
    keep = [is_completed_1h_bar(ts, now) for ts in df.index]
    return df.loc[keep].copy()


def regular_session_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[session_mask(df.index)].copy()


def should_scan(now: datetime) -> bool:
    """Scan only in the regular session, after the first 1h bar can be complete."""
    if not in_regular_session(now):
        return False
    return session_minutes(now) >= 10 * 60


def next_eval_at(now: datetime, grace_seconds: int = DEFAULT_GRACE_SECONDS) -> datetime:
    now_t = to_tpe(now).to_pydatetime()
    grace = timedelta(seconds=max(0, grace_seconds))
    d = now_t.date()
    for i in range(0, 12):
        day = d + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for hour, minute in EVAL_CLOCKS:
            t = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TPE) + grace
            if t > now_t:
                return t
    raise RuntimeError("could not find next 1h eval time")


# ---------------------------------------------------------------------------
# Dedup state
# ---------------------------------------------------------------------------


def signal_key(symbol: str, entry_ts, entry_price: float) -> str:
    ts = to_tpe(entry_ts)
    return f"{symbol}|{ts.isoformat(timespec='seconds')}|{float(entry_price):.4f}"


def load_seen(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"alerted": [], "initialized": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"alerted": [], "initialized": False}
    if not isinstance(data, dict):
        return {"alerted": [], "initialized": False}
    alerted = data.get("alerted") or data.get("alerted_entries") or []
    if not isinstance(alerted, list):
        alerted = []
    return {
        "alerted": [str(x) for x in alerted],
        "initialized": bool(data.get("initialized")),
        "last_scan": data.get("last_scan"),
    }


def save_seen(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alerted = list(state.get("alerted") or [])
    if len(alerted) > SEEN_CAP:
        alerted = alerted[-SEEN_CAP:]
    payload = {
        "alerted": alerted,
        "initialized": True,
        "last_scan": state.get("last_scan"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def consume_key(order: List[str], keys: Set[str], key: str) -> None:
    if key in keys:
        return
    keys.add(key)
    order.append(key)


@dataclass(frozen=True)
class SelectedAlert:
    key: str
    signal: Signal
    send: bool


def select_alerts(
    symbol: str,
    df: pd.DataFrame,
    signals: Sequence[Signal],
    now: datetime,
    seen: Set[str],
) -> List[SelectedAlert]:
    """Pick new alerts. Other-day signals are marked seen but not sent.

    An in-progress last bar must already have been dropped from ``df``.
    Today's completed-bar signals that are not in ``seen`` are sent.
    """
    today = to_tpe(now).date()
    out: List[SelectedAlert] = []
    for sig in signals:
        ts = to_tpe(df.index[sig.entry_idx])
        key = signal_key(symbol, ts, sig.entry_price)
        if key in seen:
            continue
        if not is_completed_1h_bar(ts, now):
            continue
        send = ts.date() == today
        out.append(SelectedAlert(key=key, signal=sig, send=send))
    return out


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def load_dotenv(path: Path = CONFIG_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def tg_send(token: str, chat_id: str, text: str, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text, flush=True)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    data = r.json()
    if not data.get("ok"):
        print(f"[tg] API error: {data}", file=sys.stderr)
        return False
    return True


def format_alert(row: dict, df: pd.DataFrame, sig: Signal) -> str:
    ts = to_tpe(df.index[sig.entry_idx])
    w = sig.wave
    entry = float(sig.entry_price)
    stop = float(w.trough_low)
    risk = entry - stop
    if risk <= 0:
        risk = entry * 0.01
        stop = entry - risk
    target = entry + 2.0 * risk
    bounce = (entry / w.trough_low - 1.0) if w.trough_low else 0.0
    trough_ts = to_tpe(df.index[w.trough_idx])
    code = escape(str(row.get("code") or ""))
    name = escape(str(row.get("name") or ""))
    symbol = escape(str(row.get("symbol") or ""))
    return (
        f"🟢 <b>台股 1h 破底翻</b>（只通知、不下單）\n"
        f"<b>{code} {name}</b> <code>{symbol}</code>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} TPE</code>\n"
        f"進場: <code>{entry:.2f}</code>\n"
        f"停損: <code>{stop:.2f}</code>  目標: <code>{target:.2f}</code> (2R)\n"
        f"破底: <code>{w.trough_low:.2f}</code> @ {trough_ts.strftime('%m-%d %H:%M')}\n"
        f"深度 {w.depth_pct * 100:.1f}% · 下面 {w.bars_below} 根 · 彈 {bounce * 100:.1f}%\n"
        f"MA5 {sig.ma5:.2f} / MA10 {sig.ma10:.2f} / MA20 {sig.ma20:.2f}\n"
        f"#破底翻 #台股 #1h"
    )


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


@dataclass
class SymbolScan:
    row: dict
    df: pd.DataFrame
    alerts: List[SelectedAlert]
    error: str = ""


def prepare_live_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    df = regular_session_bars(df)
    return drop_incomplete_1h_bars(df, now)


def scan_prepared(
    row: dict,
    df: pd.DataFrame,
    now: datetime,
    seen: Set[str],
    params: ReclaimParams,
) -> SymbolScan:
    session = regular_session_bars(df)
    empty = pd.DataFrame()
    if session is None or len(session) < MIN_FETCH_BARS:
        return SymbolScan(row=row, df=session if session is not None else empty, alerts=[], error="too_few_bars")
    live = drop_incomplete_1h_bars(session, now)
    if live is None or len(live) < MIN_COMPLETED_BARS:
        return SymbolScan(row=row, df=live if live is not None else empty, alerts=[], error="too_few_bars")
    sigs = detect_signals(live, params)
    alerts = select_alerts(str(row.get("symbol") or ""), live, sigs, now, seen)
    return SymbolScan(row=row, df=live, alerts=alerts, error="")


def load_universe(limit: int, max_price: Optional[float], date: str = "") -> List[dict]:
    ymd = resolve_twse_date(date or last_tw_session_yyyymmdd())
    pool = 0 if limit <= 0 else max(limit, 400 if max_price else limit)
    raw = fetch_top_turnover(ymd, pool)
    price_cap = None if max_price is None or max_price <= 0 else float(max_price)
    helper_cap = None if price_cap is None else price_cap - 1e-9
    universe, _dropped = filter_by_max_price(raw, helper_cap, limit)
    return universe


def scan_once(
    *,
    token: str,
    chat_id: str,
    now: datetime,
    universe: Sequence[dict],
    seen_order: List[str],
    seen_keys: Set[str],
    params: ReclaimParams,
    range_: str,
    dry_run: bool,
    workers: int,
    sleep_s: float,
    fetch_symbol=None,
) -> Tuple[int, int, int]:
    """Scan universe. Returns (scanned, errors, sent). Mutates seen_* on success."""
    fetch_symbol = fetch_symbol or (lambda row: fetch_yahoo_1h(row["symbol"], range_))
    scanned = 0
    errors = 0
    sent = 0
    results: List[SymbolScan] = []

    def work(row: dict) -> SymbolScan:
        try:
            df = fetch_symbol(row)
        except Exception as exc:  # noqa: BLE001
            return SymbolScan(row=row, df=pd.DataFrame(), alerts=[], error=str(exc)[:80])
        return scan_prepared(row, df, now, seen_keys, params)

    if workers <= 1:
        for i, row in enumerate(universe, 1):
            results.append(work(row))
            if sleep_s > 0 and i < len(universe):
                time.sleep(sleep_s)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, row): row for row in universe}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        SymbolScan(row=futs[fut], df=pd.DataFrame(), alerts=[], error=str(exc)[:80])
                    )

    for item in results:
        scanned += 1
        if item.error:
            errors += 1
            continue
        for alert in item.alerts:
            consume_key(seen_order, seen_keys, alert.key)
            if not alert.send:
                continue
            text = format_alert(item.row, item.df, alert.signal)
            ts = to_tpe(item.df.index[alert.signal.entry_idx])
            ok = tg_send(token, chat_id, text, dry_run=dry_run)
            if ok:
                sent += 1
                print(
                    f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] "
                    f"{item.row.get('code')} {item.row.get('name')} 破底翻 @ {alert.signal.entry_price:.2f} "
                    f"{ts.strftime('%Y-%m-%d %H:%M')}",
                    flush=True,
                )
            else:
                # keep key consumed only after a successful send so a restart retries
                seen_keys.discard(alert.key)
                if alert.key in seen_order:
                    seen_order.remove(alert.key)
                print(
                    f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] Telegram send failed "
                    f"{item.row.get('code')} {alert.key}",
                    file=sys.stderr,
                    flush=True,
                )
    return scanned, errors, sent


def sleep_until(target: datetime) -> None:
    while True:
        now = datetime.now(TPE)
        remaining = (to_tpe(target) - to_tpe(now)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30.0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_test(token: str, chat_id: str, dry_run: bool) -> int:
    ok = tg_send(
        token,
        chat_id,
        f"✅ 台股 1h 破底翻監看測試（只通知、不下單）\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} TPE",
        dry_run=dry_run,
    )
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）", flush=True)
    return 0 if ok else 1


def run_loop(args: argparse.Namespace) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN") or ""
    chat_id = env("TELEGRAM_CHAT_ID") or ""
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2

    if args.test:
        return cmd_test(token, chat_id, args.dry_run)

    state_path = Path(args.state).expanduser() if args.state else DEFAULT_STATE_PATH
    state = load_seen(state_path)
    seen_order = list(state.get("alerted") or [])
    seen_keys = set(seen_order)
    params = strict_params() if args.strict else loose_params()
    price_cap = None if args.max_price is None or args.max_price <= 0 else float(args.max_price)

    print(
        "TW 1h 破底翻 TG | alerts-only, no orders | "
        f"session=09:00-13:30 TPE | completed-1h-bars | "
        f"dry_run={args.dry_run} limit={args.limit} max_price={price_cap} "
        f"strict={args.strict} state={state_path}",
        flush=True,
    )

    universe: List[dict] = []
    universe_day = ""

    def refresh_universe(now: datetime) -> List[dict]:
        nonlocal universe, universe_day
        day = to_tpe(now).strftime("%Y-%m-%d")
        if universe and universe_day == day:
            return universe
        print(f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] loading universe…", flush=True)
        try:
            universe = load_universe(args.limit, price_cap, args.date)
            universe_day = day
        except Exception as exc:  # noqa: BLE001
            print(f"[error] universe: {exc}", file=sys.stderr)
            traceback.print_exc()
            return universe
        if universe:
            print(
                f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] universe {len(universe)} "
                f"{universe[0]['code']} … {universe[-1]['code']}",
                flush=True,
            )
        else:
            print(f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] universe empty", flush=True)
        return universe

    def persist(now: datetime) -> None:
        save_seen(
            state_path,
            {
                "alerted": seen_order,
                "initialized": True,
                "last_scan": to_tpe(now).isoformat(timespec="seconds"),
            },
        )

    def round_once(now: datetime) -> None:
        if not should_scan(now):
            print(
                f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] skip: outside regular session "
                f"or first 1h bar still open (09:00–13:30 TPE, completed bars only)",
                flush=True,
            )
            return
        rows = refresh_universe(now)
        if not rows:
            return
        t0 = time.time()
        scanned, errors, sent = scan_once(
            token=token,
            chat_id=chat_id,
            now=now,
            universe=rows,
            seen_order=seen_order,
            seen_keys=seen_keys,
            params=params,
            range_=args.range_,
            dry_run=args.dry_run,
            workers=args.workers,
            sleep_s=args.sleep,
        )
        persist(now)
        print(
            f"[{to_tpe(now).strftime('%H:%M:%S')} TPE] scan ok scanned={scanned} "
            f"errors={errors} sent={sent} {time.time() - t0:.1f}s",
            flush=True,
        )

    now = datetime.now(TPE)
    try:
        round_once(now)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        traceback.print_exc()
        if args.once:
            return 1
    if args.once:
        return 0

    print("watch 中，每個 1h 收盤後掃描（Ctrl+C 停）。不下單。", flush=True)
    try:
        while True:
            target = next_eval_at(datetime.now(TPE), args.grace)
            print(f"[{datetime.now(TPE).strftime('%H:%M:%S')} TPE] next eval {target.isoformat()}", flush=True)
            sleep_until(target)
            try:
                round_once(datetime.now(TPE))
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {exc}", file=sys.stderr)
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
        persist(datetime.now(TPE))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股 1h 破底翻 Telegram 監看（只通知、不下單）")
    p.add_argument("--once", action="store_true", help="只掃一輪然後結束")
    p.add_argument("--dry-run", action="store_true", help="印出訊號、不呼叫 Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--strict", action="store_true", help="進場還要 MA5>MA10>MA20（同回測 --strict）")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="成交額前 N；0 = 不限")
    p.add_argument("--max-price", type=float, default=DEFAULT_MAX_PRICE, help="股價達此值以上剔除；0 不過濾")
    p.add_argument("--range", dest="range_", default=DEFAULT_RANGE, metavar="RANGE", help="Yahoo 1h 下載區間")
    p.add_argument("--date", default="", help="宇宙成交額基準日 YYYYMMDD，預設上一個交易日")
    p.add_argument("--sleep", type=float, default=0.0, help="workers=1 時，檔與檔之間的間隔秒")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grace", type=int, default=DEFAULT_GRACE_SECONDS, help="收盤後等多久再抓 Yahoo（秒）")
    p.add_argument("--state", default="", help="去重狀態檔，預設 output/tw_1h_reclaim_alert_state.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
