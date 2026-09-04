#!/usr/bin/env python3
"""幣安 1h MA25 下破底、等 7/14/25 多頭排列 → 推 Telegram。

憑證放專案根目錄 tg_config.env（勿提交），或填下面兩個變數：

    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

用法:
  python3 examples/watch_binance_1h_ma25.py --test
  python3 examples/watch_binance_1h_ma25.py --dry-run --once
  python3 examples/watch_binance_1h_ma25.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_1h_ma25_reclaim import (  # noqa: E402
    KEEP,
    STRICT_DETECT,
    Signal,
    TPE,
    TradeResult,
    detect_signals,
    draw_trade_png,
    fetch_klines,
    simulate,
    ticker_quote_volume,
    universe,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO / "tg_config.env"
STATE_PATH = REPO / "output" / "binance_1h_ma25_tg.json"
PHOTO_DIR = Path("/tmp")

# —— 也可以直接填這裡 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""


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


def apply_keys() -> None:
    load_dotenv()
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def tg_creds() -> Tuple[str, str]:
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
    )


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": [], "alerted_exits": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _ts_tpe(ts) -> datetime:
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=TPE)
    return ts.tz_convert(TPE)


def entry_key(symbol: str, sig: Signal, df) -> str:
    ts = _ts_tpe(df.index[sig.entry_idx])
    return f"{symbol}|{ts.isoformat()}|{sig.entry_price:.8g}"


def exit_key(symbol: str, tr: TradeResult, df) -> str:
    et = _ts_tpe(df.index[tr.entry_idx])
    xt = _ts_tpe(df.index[tr.exit_idx])
    return f"{symbol}|{et.isoformat()}->{xt.isoformat()}|{tr.exit_reason}|{tr.pnl_pct:.4f}"


def open_trade(sig: Signal) -> TradeResult:
    return TradeResult(
        signal=sig,
        entry_idx=sig.entry_idx,
        exit_idx=sig.entry_idx,
        entry_price=sig.entry_price,
        exit_price=sig.entry_price,
        stop_price=sig.stop_price,
        target_price=sig.target_price,
        pnl_pct=0.0,
        exit_reason="open",
        quality=sig.quality,
    )


def fmt_entry(symbol: str, df, sig: Signal) -> str:
    ts = _ts_tpe(df.index[sig.entry_idx])
    br = _ts_tpe(df.index[sig.break_idx])
    last = float(df["Close"].iloc[-1])
    risk = sig.entry_price - sig.stop_price
    risk_pct = (risk / sig.entry_price) * 100.0 if sig.entry_price else 0.0
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0.0
    return (
        f"🟢 <b>1h MA25 破底後多頭排列</b>\n"
        f"<b>{escape(symbol)}</b>  Q{escape(sig.quality)}  {escape(sig.shape)}\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')}</code> 台北\n"
        f"進場: <code>{sig.entry_price:.6g}</code>\n"
        f"停損: <code>{sig.stop_price:.6g}</code>  (−{risk_pct:.2f}%)  收盤跌破破底K\n"
        f"目標: <code>{sig.target_price:.6g}</code>  ({r_mult:.1f}R)\n"
        f"破底: <code>{br.strftime('%m-%d %H:%M')}</code>  {sig.bottom:.6g}\n"
        f"深度 {sig.depth_pct * 100:.2f}%  在下 {sig.bars_below}h\n"
        f"MA7 {sig.ma7:.6g} &gt; MA14 {sig.ma14:.6g} &gt; MA25 {sig.ma25:.6g}\n"
        f"現價: <code>{last:.6g}</code>\n"
        f"#MA25 #排列 #{escape(symbol)}"
    )


def fmt_exit(symbol: str, df, tr: TradeResult) -> str:
    et = _ts_tpe(df.index[tr.entry_idx])
    xt = _ts_tpe(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_pct > 0 else ("⚪" if tr.pnl_pct == 0 else "🔴")
    return (
        f"{emoji} <b>1h MA25 出場</b>  {escape(symbol)}\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.6g}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.6g}\n"
        f"原因: <b>{escape(tr.exit_reason)}</b>\n"
        f"盈虧: <b>{tr.pnl_pct:+.2f}%</b> · Q{escape(tr.quality)}\n"
        f"#MA25 #出場 #{escape(symbol)}"
    )


def telegram_send(text: str, *, photo: Optional[Path] = None, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("&gt;", ">"))
        return True
    token, chat_id = tg_creds()
    if not token or not chat_id:
        return False
    try:
        if photo is not None and photo.exists():
            with photo.open("rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": text[:1024],
                        "parse_mode": "HTML",
                    },
                    files={"photo": f},
                    timeout=25,
                )
            if r.ok:
                return True
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if not r.ok:
            print(f"[tg] HTTP {r.status_code}: {r.text[:240]}", file=sys.stderr)
            return False
        data = r.json()
        if not data.get("ok"):
            print(f"[tg] API error: {data}", file=sys.stderr)
            return False
        return True
    except requests.RequestException as exc:
        print(f"[tg] {exc}", file=sys.stderr)
        return False


def draw_entry_png(symbol: str, df, sig: Signal) -> Optional[Path]:
    ts = _ts_tpe(df.index[sig.entry_idx])
    path = PHOTO_DIR / f"ma25_{symbol}_{ts.strftime('%m%d_%H%M')}.png"
    try:
        draw_trade_png(df, open_trade(sig), path, 1, title_extra=symbol)
        return path if path.exists() else None
    except Exception as exc:  # noqa: BLE001
        print(f"[chart] {symbol} {exc}", file=sys.stderr)
        return None


def pick_symbols(args) -> List[str]:
    extras = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if extras and not args.universe:
        return extras
    symbols = universe(args.min_quote, extra=extras)
    if args.limit and len(symbols) > args.limit:
        qv = ticker_quote_volume()
        keep = [s for s in symbols if s in KEEP or s in extras]
        rest = [s for s in symbols if s not in keep]
        rest.sort(key=lambda s: qv.get(s, 0.0), reverse=True)
        symbols = keep + rest[: max(0, args.limit - len(keep))]
        seen: Set[str] = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]
    return symbols


def detect_kw(args) -> dict:
    if args.strict:
        return dict(STRICT_DETECT)
    return dict(
        min_bars_below=args.min_bars,
        max_bars_below=args.max_bars,
        min_depth_pct=args.min_depth / 100.0,
    )


def scan_symbol(symbol: str, days: int, kw: dict) -> Tuple[str, Optional[Any], List[Signal], List[TradeResult], str]:
    try:
        df = fetch_klines(symbol, days=days)
    except Exception as exc:  # noqa: BLE001
        return symbol, None, [], [], str(exc)[:80]
    if df is None or len(df) < 40:
        return symbol, df, [], [], "too_few_bars"
    sigs = detect_signals(df, **kw)
    trades = simulate(df, sigs)
    return symbol, df, sigs, trades, ""


def wait_next_hour_close(pad_sec: int = 20) -> None:
    now = time.time()
    nxt = (int(now) // 3600 + 1) * 3600 + pad_sec
    time.sleep(max(1, nxt - now))


def scan_once(args, symbols: Sequence[str], *, seed_if_first: bool) -> None:
    kw = detect_kw(args)
    state = load_state()
    alerted_e: Set[str] = set(state.get("alerted_entries") or [])
    alerted_x: Set[str] = set(state.get("alerted_exits") or [])
    now = datetime.now(TPE)
    cutoff = now - timedelta(hours=args.lookback_hours)
    first_run = seed_if_first and (not state.get("initialized"))

    new_entries: List[Tuple[str, Any, Signal, str]] = []
    new_exits: List[Tuple[str, Any, TradeResult, str]] = []
    errors = 0

    def work(sym: str):
        return scan_symbol(sym, args.days, kw)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(work, s): s for s in symbols}
        for fut in as_completed(futs):
            symbol, df, sigs, trades, err = fut.result()
            if err or df is None:
                errors += 1
                if err:
                    print(f"  {symbol} {err}", file=sys.stderr)
                continue
            for sig in sigs:
                k = entry_key(symbol, sig, df)
                ts = _ts_tpe(df.index[sig.entry_idx])
                if ts < cutoff:
                    alerted_e.add(k)
                    continue
                if k in alerted_e:
                    continue
                new_entries.append((symbol, df, sig, k))
            if args.no_exits:
                continue
            for tr in trades:
                k = exit_key(symbol, tr, df)
                xt = _ts_tpe(df.index[tr.exit_idx])
                if xt < cutoff or tr.exit_reason == "open":
                    alerted_x.add(k)
                    continue
                if k in alerted_x:
                    continue
                new_exits.append((symbol, df, tr, k))

    new_entries.sort(key=lambda x: x[1].index[x[2].entry_idx])
    new_exits.sort(key=lambda x: x[1].index[x[2].exit_idx])

    if first_run and not args.seed_alert:
        for *_, k in new_entries:
            alerted_e.add(k)
        for *_, k in new_exits:
            alerted_x.add(k)
        state["alerted_entries"] = sorted(alerted_e)[-400:]
        state["alerted_exits"] = sorted(alerted_x)[-400:]
        state["initialized"] = True
        state["last_scan"] = now.isoformat()
        save_state(state)
        print(
            f"[{now.strftime('%m-%d %H:%M')}] 初次啟動，標記近期 "
            f"{len(new_entries)} 筆進場 / {len(new_exits)} 筆出場，之後有新的才推。"
        )
        return

    sent = 0
    for symbol, df, sig, k in new_entries:
        text = fmt_entry(symbol, df, sig)
        photo = None if args.no_photo else draw_entry_png(symbol, df, sig)
        ok = telegram_send(text, photo=photo, dry_run=args.dry_run)
        ts = _ts_tpe(df.index[sig.entry_idx])
        if ok:
            alerted_e.add(k)
            sent += 1
            print(f"[進場] {symbol} {ts.strftime('%m-%d %H:%M')} Q{sig.quality} @ {sig.entry_price:.6g}")
        else:
            token, _ = tg_creds()
            if not token and not args.dry_run:
                print(f"[進場·未送] {symbol} 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
            else:
                print(f"[進場·失敗] {symbol}", file=sys.stderr)

    for symbol, df, tr, k in new_exits:
        ok = telegram_send(fmt_exit(symbol, df, tr), dry_run=args.dry_run)
        if ok:
            alerted_x.add(k)
            sent += 1
            print(f"[出場] {symbol} {tr.exit_reason} {tr.pnl_pct:+.2f}%")

    state["alerted_entries"] = sorted(alerted_e)[-400:]
    state["alerted_exits"] = sorted(alerted_x)[-400:]
    state["initialized"] = True
    state["last_scan"] = now.isoformat()
    save_state(state)
    print(
        f"[{now.strftime('%m-%d %H:%M')}] 掃 {len(symbols)} 檔  err={errors}  "
        f"新進場 {len(new_entries)} 新出場 {len(new_exits)} 已送 {sent}"
    )


def cmd_test(dry_run: bool) -> int:
    ok = telegram_send(
        f"✅ 1h MA25 監看測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
        dry_run=dry_run,
    )
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="幣安 1h MA25 破底後排列 → Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--once", action="store_true", help="掃一次就結束")
    p.add_argument("--dry-run", action="store_true", help="只印不送")
    p.add_argument("--seed-alert", action="store_true", help="第一次啟動也把近期訊號推出去")
    p.add_argument("--no-exits", action="store_true", help="不出場通知")
    p.add_argument("--no-photo", action="store_true", help="進場不帶圖")
    p.add_argument("--strict", action="store_true", help="筆畫那種 W")
    p.add_argument("--symbols", default="", help="只看這些，逗號分隔")
    p.add_argument("--universe", action="store_true", help="指定 --symbols 仍掃流動永續")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--min-quote", type=float, default=8_000_000)
    p.add_argument("--days", type=int, default=14, help="每檔拉幾天 1h（含均線暖身）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-bars", type=int, default=4)
    p.add_argument("--max-bars", type=int, default=36)
    p.add_argument("--min-depth", type=float, default=1.8)
    p.add_argument("--lookback-hours", type=float, default=8, help="只推這幾小時內的新訊號")
    p.add_argument("--interval", type=int, default=0, help="秒；0=等下一根 1h 收盤再掃")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    apply_keys()
    args = build_parser().parse_args(argv)
    if args.test:
        return cmd_test(args.dry_run)

    token, chat_id = tg_creds()
    if not args.dry_run and (not token or not chat_id):
        print("請在 tg_config.env 或腳本上方填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID", file=sys.stderr)
        return 2

    print("載入標的…", flush=True)
    symbols = pick_symbols(args)
    mode = "嚴格 W" if args.strict else "寬鬆"
    print(
        f"監看 {len(symbols)} 檔 1h · {mode} · 破底後 7/14/25 排列才推"
        f"{' · dry-run' if args.dry_run else ''}",
        flush=True,
    )
    uni_ts = time.time()

    def round_once(*, seed_if_first: bool) -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 6 * 3600:
            symbols = pick_symbols(args)
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        scan_once(args, symbols, seed_if_first=seed_if_first)

    round_once(seed_if_first=True)
    if args.once:
        return 0
    print("watch 中，每根 1h 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            if args.interval > 0:
                time.sleep(max(15, args.interval))
            else:
                wait_next_hour_close()
            round_once(seed_if_first=False)
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
