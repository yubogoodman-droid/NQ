#!/usr/bin/env python3
"""永豐 Shioaji 盤中監控：成交額前 100 掃描池，五分／十五分剛站上 MA200 就推 Telegram。

盤中用 tick 訂閱合成 K（不要重覆打歷史 kbars）。
啟動時抓一次歷史 1 分K 當均線底。

在下面填金鑰，或改用環境變數：

    SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID

    python3 examples/watch_tw_shioaji.py --test          # 只測 Telegram
    python3 examples/watch_tw_shioaji.py --once          # 用歷史K掃一次就結束
    python3 examples/watch_tw_shioaji.py                 # 盤中一直盯
    python3 examples/watch_tw_shioaji.py --tf 5m
    python3 examples/watch_tw_shioaji.py --tf 15m
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


def find_nq_root() -> Path:
    """找到含 tw/kline.py 的專案根目錄；若被 tw.py 擋住就直接說明。"""
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        shadow = cand / "tw.py"
        pkg = cand / "tw" / "kline.py"
        if shadow.is_file() and not pkg.is_file():
            raise SystemExit(
                f"找到 {shadow}\n"
                "這個 tw.py 會讓 Python 以為 tw 不是套件，所以出現：\n"
                "  ModuleNotFoundError: No module named 'tw.kline'; 'tw' is not a package\n"
                "請把 tw.py 刪掉或改名，並把整個 tw 資料夾放到專案裡。"
            )
        if pkg.is_file():
            if shadow.is_file():
                raise SystemExit(
                    f"找到 {shadow}，它會蓋掉 tw 資料夾。請刪掉或改名 tw.py。"
                )
            return cand
    raise SystemExit(
        "找不到 tw\\kline.py。\n"
        "請用 PyCharm 打開整個 NQ 專案（裡面要有 tw 資料夾），\n"
        "不要只把一支腳本複製到 PythonProject2。\n"
        "套件：https://github.com/yubogoodman-droid/NQ/tree/cursor/tw-5m-ma200-5d-327f/tw"
    )


ROOT = find_nq_root()
sys.path.insert(0, str(ROOT))

from tw.kline import resample_ohlcv
from tw.live import (
    TAIPEI,
    BarAggregator,
    OhlcvBar,
    alerts_on_closed_bar,
    format_telegram,
    in_session,
    kbars_to_ohlcv,
    should_run_15m,
    upsert_bar,
)
from tw.ranking import (
    RankedStock,
    fetch_turnover_ranking,
    filter_by_price,
    filter_etfs,
    filter_financials,
    filter_telecoms,
)
from tw.signals import iter_15m_ma200_alerts, iter_5m_ma200_alerts

# —— 本機金鑰放 examples/local_secrets.py（已 gitignore，不要填進這個檔再 commit）——
SHIOAJI_API_KEY = ""
SHIOAJI_SECRET_KEY = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

SEEN_PATH = ROOT / "output" / "tw_shioaji_seen.json"
LOCAL_SECRETS = Path(__file__).resolve().parent / "local_secrets.py"
if not LOCAL_SECRETS.exists():
    LOCAL_SECRETS = ROOT / "examples" / "local_secrets.py"
if not LOCAL_SECRETS.exists():
    LOCAL_SECRETS = ROOT / "local_secrets.py"
TG_SESSION = requests.Session()


def apply_keys() -> None:
    if LOCAL_SECRETS.exists():
        ns: dict = {}
        exec(LOCAL_SECRETS.read_text(encoding="utf-8"), ns)
        mapping = {
            "SHIOAJI_API_KEY": ("SHIOAJI_API_KEY", "API_KEY"),
            "SHIOAJI_SECRET_KEY": ("SHIOAJI_SECRET_KEY", "SECRET_KEY"),
            "TELEGRAM_BOT_TOKEN": ("TELEGRAM_BOT_TOKEN", "TG_TOKEN"),
            "TELEGRAM_CHAT_ID": ("TELEGRAM_CHAT_ID", "TG_CHAT_ID"),
        }
        for env_name, aliases in mapping.items():
            for alias in aliases:
                val = str(ns.get(alias, "")).strip()
                if val:
                    os.environ.setdefault(env_name, val)
                    break
    if SHIOAJI_API_KEY.strip():
        os.environ.setdefault("SHIOAJI_API_KEY", SHIOAJI_API_KEY.strip())
    if SHIOAJI_SECRET_KEY.strip():
        os.environ.setdefault("SHIOAJI_SECRET_KEY", SHIOAJI_SECRET_KEY.strip())
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def telegram_send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = TG_SESSION.post(
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
            try:
                desc = r.json().get("description", r.text[:200])
            except ValueError:
                desc = r.text[:200]
            print(f"  Telegram HTTP {r.status_code}：{desc}", flush=True)
        return bool(r.ok)
    except requests.RequestException as exc:
        print(f"  Telegram 連線失敗：{exc}", flush=True)
        return False


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    save = sorted(seen)
    if len(save) > 4000:
        save = save[-2000:]
    SEEN_PATH.write_text(json.dumps(save), encoding="utf-8")


def load_universe(top: int, max_price: float) -> tuple[list[RankedStock], str]:
    stocks, label = fetch_turnover_ranking(top=top)
    priced = filter_by_price(stocks, max_price)
    candidates = filter_telecoms(filter_financials(filter_etfs(priced)))
    return candidates, label or "成交額名單"


def login_shioaji(*, simulation: bool):
    try:
        import shioaji as sj
    except ImportError as exc:
        raise SystemExit("請先安裝永豐 API：pip install shioaji") from exc
    key = os.environ.get("SHIOAJI_API_KEY", "").strip()
    secret = os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit("請填 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY（腳本上方或環境變數）")
    api = sj.Shioaji(simulation=simulation)
    api.login(api_key=key, secret_key=secret)
    return api, sj


def resolve_contract(api, stock: RankedStock):
    code = stock.code
    try:
        found = api.Contracts.Stocks[code]
        if found is not None:
            return found
    except Exception:
        pass
    getter = getattr(api, "contracts", None)
    if getter is not None and hasattr(getter, "get"):
        return getter.get(code)
    return None


def fetch_history(api, contract, days: int = 20) -> pd.DataFrame:
    end = datetime.now(TAIPEI).date()
    start = end - timedelta(days=days)
    kbars = api.kbars(
        contract=contract,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    one = kbars_to_ohlcv(kbars)
    if one.empty:
        return one
    return resample_ohlcv(one, "5min")


def push_snap(stock: RankedStock, snap, tf: str, seen: set[str], *, dry: bool) -> bool:
    day = datetime.now(TAIPEI).date().isoformat()
    key = f"{day}:{stock.symbol}:{pd.Timestamp(snap.timestamp)}:{tf}"
    if key in seen:
        return False
    seen.add(key)
    text = format_telegram(stock.name, stock.symbol, snap, tf)
    print(text.replace("&gt;", ">"), flush=True)
    if dry:
        return True
    if telegram_send(text):
        print("  → Telegram 已送", flush=True)
    else:
        print("  → Telegram 失敗（檢查 token / chat id）", flush=True)
    return True


def emit_alerts(
    stock: RankedStock,
    frame: pd.DataFrame,
    bar: OhlcvBar,
    tfs: list[str],
    seen: set[str],
    *,
    dry: bool,
) -> None:
    jobs = []
    if "5m" in tfs:
        jobs.append("5m")
    if "15m" in tfs and should_run_15m(bar):
        jobs.append("15m")
    for tf in jobs:
        for snap in alerts_on_closed_bar(frame, bar, tf=tf):
            push_snap(stock, snap, tf, seen, dry=dry)
    save_seen(seen)


def scan_once(api, candidates: list[RankedStock], tfs: list[str], seen: set[str], *, dry: bool) -> int:
    n = 0
    for i, stock in enumerate(candidates, 1):
        contract = resolve_contract(api, stock)
        if contract is None:
            print(f"{stock.symbol} 找不到契約", flush=True)
            continue
        print(f"歷史K {i}/{len(candidates)} {stock.name} {stock.code}", flush=True)
        frame = fetch_history(api, contract)
        if frame.empty or len(frame) < 201:
            continue
        since = pd.Timestamp(datetime.now(TAIPEI).date(), tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        before = len(seen)
        if "5m" in tfs:
            for snap in iter_5m_ma200_alerts(frame, since=since, until=until):
                push_snap(stock, snap, "5m", seen, dry=dry)
        if "15m" in tfs:
            for snap in iter_15m_ma200_alerts(frame, since=since, until=until):
                push_snap(stock, snap, "15m", seen, dry=dry)
        save_seen(seen)
        n += len(seen) - before
        time.sleep(0.2)
    return n


def run_watch(args: argparse.Namespace) -> int:
    apply_keys()
    tfs = ["5m", "15m"] if args.tf == "both" else [args.tf]
    seen = load_seen()
    candidates, label = load_universe(args.top, args.max_price)
    print(f"{label} → 掃描 {len(candidates)} 檔  tf={'+'.join(tfs)}", flush=True)

    api, sj = login_shioaji(simulation=args.sim)
    if args.once:
        found = scan_once(api, candidates, tfs, seen, dry=args.dry)
        print(f"掃完，新通知 {found} 則", flush=True)
        try:
            api.logout()
        except Exception:
            pass
        return 0

    frames: dict[str, pd.DataFrame] = {}
    builders: dict[str, BarAggregator] = {}
    stocks: dict[str, RankedStock] = {}
    pending: queue.Queue[tuple[str, OhlcvBar]] = queue.Queue()

    print("啟動時抓一次歷史K（之後盤中只用 tick）…", flush=True)
    for i, stock in enumerate(candidates, 1):
        contract = resolve_contract(api, stock)
        if contract is None:
            print(f"略過 {stock.symbol}（無契約）", flush=True)
            continue
        print(f"歷史K {i}/{len(candidates)} {stock.name} {stock.code}", flush=True)
        try:
            frames[stock.code] = fetch_history(api, contract)
        except Exception as exc:  # noqa: BLE001
            print(f"  失敗：{exc}", flush=True)
            frames[stock.code] = pd.DataFrame()
        builders[stock.code] = BarAggregator(5)
        stocks[stock.code] = stock
        time.sleep(0.2)

    from shioaji import TickSTKv1, Exchange  # noqa: WPS433

    @api.on_tick_stk_v1()
    def _on_tick(exchange: Exchange, tick: TickSTKv1) -> None:
        code = str(getattr(tick, "code", "") or "")
        builder = builders.get(code)
        if builder is None:
            return
        try:
            price = float(tick.close)
            vol = float(getattr(tick, "volume", 0) or 0)
            ts = pd.Timestamp(tick.datetime)
        except Exception:
            return
        closed = builder.on_tick(ts, price, vol)
        if closed is not None:
            pending.put((code, closed))

    subscribed = 0
    for stock in candidates:
        if stock.code not in builders:
            continue
        contract = resolve_contract(api, stock)
        if contract is None:
            continue
        try:
            api.subscribe(contract, quote_type=sj.constant.QuoteType.Tick)
            subscribed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"訂閱失敗 {stock.code}：{exc}", flush=True)
    print(f"已訂閱 {subscribed} 檔 tick。Ctrl+C 結束。", flush=True)
    if not args.dry:
        telegram_send(
            f"台股監控已啟動\n{label}\n掃描 {len(builders)} 檔　{'＋'.join(tfs)}"
        )

    try:
        while True:
            now = pd.Timestamp(datetime.now(TAIPEI))
            if not in_session(now) and now.time().replace(tzinfo=None) > datetime.strptime("13:35", "%H:%M").time():
                for code, builder in builders.items():
                    flushed = builder.flush_if_due(now)
                    if flushed is not None:
                        pending.put((code, flushed))
            try:
                code, bar = pending.get(timeout=2)
            except queue.Empty:
                for code, builder in list(builders.items()):
                    flushed = builder.flush_if_due(pd.Timestamp(datetime.now(TAIPEI)))
                    if flushed is not None:
                        pending.put((code, flushed))
                continue
            stock = stocks.get(code)
            if stock is None:
                continue
            frames[code] = upsert_bar(frames.get(code), bar)
            print(
                f"{bar.start.strftime('%H:%M')} 收 {stock.name} {bar.close:.2f}",
                flush=True,
            )
            emit_alerts(stock, frames[code], bar, tfs, seen, dry=args.dry)
    except KeyboardInterrupt:
        print("\n結束監控", flush=True)
    finally:
        try:
            api.logout()
        except Exception:
            pass
    return 0


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("台股永豐監控測試\n看到這則代表 Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="永豐 Shioaji 台股五分／十五分 MA200 Telegram 監控")
    p.add_argument("--tf", choices=("5m", "15m", "both"), default="both")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--max-price", type=float, default=500.0)
    p.add_argument("--test", action="store_true", help="只測 Telegram")
    p.add_argument("--once", action="store_true", help="抓歷史K掃一次就結束（可盤後）")
    p.add_argument("--dry", action="store_true", help="符合條件只印、不推 Telegram")
    p.add_argument("--sim", action="store_true", help="永豐模擬環境登入")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        return test_telegram()
    return run_watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
