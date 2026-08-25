#!/usr/bin/env python3
"""台股一分 K：MA5>MA10>MA20 多頭排列，收盤剛上 MA240 就通知。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ════ 在下面引號裡填序號（只放你自己電腦，不要貼聊天室、不要 commit）════
SHIOAJI_API_KEY = ""        # 永豐 API Key
SHIOAJI_SECRET_KEY = ""     # 永豐 Secret
TELEGRAM_BOT_TOKEN = ""     # Telegram BotFather 給的 token
TELEGRAM_CHAT_ID = ""       # 你的 Telegram chat id
# ═══════════════════════════════════════════════════════════════


def _apply_secrets() -> None:
    for name, value in (
        ("SHIOAJI_API_KEY", SHIOAJI_API_KEY),
        ("SHIOAJI_SECRET_KEY", SHIOAJI_SECRET_KEY),
        ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
    ):
        text = str(value).strip()
        if text:
            os.environ[name] = text


_apply_secrets()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.stem == "tw":
        raise SystemExit(
            "這個檔不能叫 tw.py。\n"
            "請用整個專案來跑，不要只貼這一個檔：\n"
            "  git clone https://github.com/yubogoodman-droid/NQ.git\n"
            "PyCharm 打開 NQ 資料夾，執行 examples/scan_tw_1m.py"
        )
    for folder in (here.parent, here.parent.parent):
        if (folder / "tw" / "kline.py").is_file():
            return folder
    raise SystemExit(
        "找不到 tw 套件（tw/kline.py）。\n"
        "掃描不是單檔，需要整個專案。請 clone：\n"
        "  git clone https://github.com/yubogoodman-droid/NQ.git\n"
        "然後在 NQ 資料夾裡跑 examples/scan_tw_1m.py"
    )


sys.path.insert(0, str(_repo_root()))

from tw.kline import set_kline_source, using_shioaji
from tw.notify import format_hit_message, send_notifications
from tw.ranking import previous_friday, previous_weekdays
from tw.report import save_scan_html, save_week_index
from tw.screener import ScanConfig, hit_key, run_scan

TAIPEI = ZoneInfo("Asia/Taipei")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股一分K多頭排列＋收盤上MA240掃描")
    p.add_argument("--top", type=int, default=200, help="成交額前 N 名（預設 200）")
    p.add_argument("--max-price", type=float, default=650.0, help="濾掉此價格以上（預設 650）")
    p.add_argument("--watch", action="store_true", help="盤中每分鐘重掃，同一根 K 不重複通知")
    p.add_argument(
        "--source",
        choices=("auto", "shioaji", "yahoo"),
        default="shioaji",
        help="K線來源（預設永豐；要 Yahoo 才用 --source yahoo）",
    )
    p.add_argument("--interval", type=int, default=60, help="watch 間隔秒數")
    p.add_argument("--latest-only", action="store_true", help="只看最新一根（watch 模式自動開啟）")
    p.add_argument("--closed-only", action="store_true", help="只用已收盤的一分 K（不含當根未收）")
    p.add_argument("--include-etf", action="store_true", help="不過濾 ETF")
    p.add_argument("--include-financial", action="store_true", help="不過濾金融股")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-o", "--output", default="docs/tw/index.html")
    p.add_argument("--date", help="回測指定日（YYYY-MM-DD），只找當天站上 MA240")
    p.add_argument("--last-friday", action="store_true", help="回測上週五")
    p.add_argument("--last-week", action="store_true", help="回測上週一到五，每天分開一頁")
    p.add_argument("--quiet-empty", action="store_true", help="沒命中時不印詳細清單")
    return p.parse_args()


def _on_date(args: argparse.Namespace) -> date | None:
    if args.last_friday:
        return previous_friday()
    if args.date:
        return date.fromisoformat(args.date)
    return None


def print_result(result, *, quiet_empty: bool) -> None:
    print(
        f"[{result.scanned_at.strftime('%H:%M:%S')}] "
        f"成交額前 {len(result.universe)}／股價濾掉 {result.price_dropped}／"
        f"ETF濾掉 {result.etf_dropped}／金融濾掉 {result.financial_dropped}／掃描 {len(result.candidates)}／"
        f"命中 {len(result.hits)}／略過 {len(result.skipped)}／錯誤 {len(result.errors)}"
    )
    if result.as_of:
        print(f"  回測日期：{result.as_of.isoformat()}")
    if using_shioaji():
        print("  K線：永豐 Shioaji")
    else:
        print("  K線：Yahoo（延遲）")
    if result.rank_time:
        print(f"  排行資料時間：{result.rank_time}")
    no_k = sum(1 for _, reason in result.skipped if "無一分" in reason or "不足" in reason)
    if no_k >= max(5, len(result.candidates) // 2) and not result.hits:
        print("  多數沒有一分K：合約可能還沒下完，或永豐還在忙碌。不要連按 Run。")
    if not result.hits:
        if not quiet_empty:
            print("  目前沒有符合條件的標的。")
        return
    for hit in result.hits:
        s, snap = hit.stock, hit.snapshot
        chg = f"{s.change_percent:+.2f}%" if s.change_percent is not None else ""
        print(
            f"  #{s.rank:3d} {s.name:8s} {s.symbol:10s} "
            f"{s.price:8.2f} {chg:>8s}  "
            f"收 {snap.close:.2f} > MA240 {snap.ma240:.2f}  "
            f"前收 {snap.prev_close:.2f}  "
            f"MA {snap.ma5:.2f}/{snap.ma10:.2f}/{snap.ma20:.2f}  "
            f"{snap.timestamp.strftime('%H:%M')}"
        )


def scan_once(
    args: argparse.Namespace,
    seen: set,
    *,
    ranking: tuple | None = None,
    first: bool = False,
) -> tuple[int, object]:
    on_date = _on_date(args)
    output = args.output
    if on_date is not None and output == "docs/tw/index.html":
        output = f"docs/tw/backtest-{on_date.isoformat()}.html"
    latest_only = (args.latest_only or args.watch) and on_date is None
    if first and args.watch and on_date is None:
        latest_only = False
    result = run_scan(
        ScanConfig(
            top=args.top,
            max_price=args.max_price,
            closed_only=args.closed_only or on_date is not None or (args.watch and using_shioaji()),
            workers=args.workers,
            latest_only=latest_only,
            exclude_etf=not args.include_etf,
            exclude_financial=not args.include_financial,
            on_date=on_date,
            kline_range="7d" if on_date is not None else "5d",
            reuse_universe=None if ranking is None else list(ranking[0]),
            reuse_rank_time=None if ranking is None else ranking[1],
        )
    )
    print_result(result, quiet_empty=args.quiet_empty)
    path = save_scan_html(result, output)
    print(f"  報告：{path}")

    new_hits = [h for h in result.hits if hit_key(h) not in seen]
    for h in result.hits:
        seen.add(hit_key(h))
    if new_hits and on_date is None:
        title, body = format_hit_message(new_hits)
        print()
        print(title)
        print(body)
        channels = send_notifications(title, body)
        if channels:
            print(f"  已通知：{', '.join(channels)}")
        else:
            print("  未設定 Telegram（仍已印在終端機）。")
            print("  在本檔最上面填 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID")
    if result.errors and not args.quiet_empty:
        print("  錯誤：")
        for stock, err in result.errors[:8]:
            print(f"    {stock.symbol} {err}")
        if len(result.errors) > 8:
            print(f"    …另有 {len(result.errors) - 8} 筆")
    return len(new_hits), result


def scan_weekdays(args: argparse.Namespace, days: list, index_path: str, label: str) -> list:
    results = []
    print(f"{label} {days[0].isoformat()}～{days[-1].isoformat()}，每天分開")
    for day in days:
        output = f"docs/tw/backtest-{day.isoformat()}.html"
        result = run_scan(
            ScanConfig(
                top=args.top,
                max_price=args.max_price,
                closed_only=True,
                workers=args.workers,
                latest_only=False,
                exclude_etf=not args.include_etf,
                exclude_financial=not args.include_financial,
                on_date=day,
            )
        )
        print_result(result, quiet_empty=args.quiet_empty)
        path = save_scan_html(result, output)
        print(f"  報告：{path}")
        results.append(result)
    index = save_week_index(results, index_path)
    print(f"  目錄：{index}")
    print()
    print(f"{label}分日命中：")
    for result in results:
        assert result.as_of is not None
        names = "、".join(h.stock.name for h in result.hits) or "—"
        print(f"  {result.as_of.isoformat()}  {len(result.hits)} 檔  {names}")
    return results


def _sleep_to_next_minute(pad_sec: int = 3) -> None:
    now = datetime.now(TAIPEI)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=pad_sec)
    time.sleep(max(1.0, (nxt - now).total_seconds()))


def _tw_session_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(TAIPEI)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 <= minutes <= 13 * 60 + 35


def _seconds_until_open(now: datetime | None = None) -> float | None:
    """平日開盤前回傳還要等幾秒；已開盤、收盤後或週末回傳 None。"""
    now = now or datetime.now(TAIPEI)
    if now.weekday() >= 5:
        return None
    open_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now >= open_at:
        return None
    return (open_at - now).total_seconds()


def scan_last_week(args: argparse.Namespace) -> int:
    return len(scan_weekdays(args, previous_weekdays(), "docs/tw/week-last.md", "回測上週"))


def _ensure_shioaji() -> None:
    try:
        import shioaji  # noqa: F401
        return
    except ImportError:
        pass
    print("沒有 shioaji，正在裝到目前這個 Python：")
    print(f"  {sys.executable} -m pip install shioaji")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shioaji"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "安裝失敗。請在 PyCharm Terminal 執行：\n"
            f"  {sys.executable} -m pip install shioaji"
        ) from exc
    try:
        import shioaji  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "裝完仍 import 不到。PyCharm 右下角 interpreter 要選這個：\n"
            f"  {sys.executable}"
        ) from exc


def main() -> int:
    args = parse_args()
    _apply_secrets()
    if args.source == "shioaji" and not (
        os.environ.get("SHIOAJI_API_KEY", "").strip()
        and os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    ):
        print("請在 examples/scan_tw_1m.py 最上面填 SHIOAJI_API_KEY 與 SHIOAJI_SECRET_KEY")
        return 1
    set_kline_source(args.source)
    if using_shioaji():
        _ensure_shioaji()
    if args.last_week:
        scan_last_week(args)
        return 0
    seen: set = set()
    _, result = scan_once(args, seen, first=True)
    ranking = (result.universe, result.rank_time)
    if not args.watch or _on_date(args) is not None:
        return 0
    live = using_shioaji()
    wait_sec = _seconds_until_open()
    if live and wait_sec is not None:
        print(f"\n離 09:00 開盤還有 {wait_sec / 60:.0f} 分鐘。K 線已先抓好，等開盤自動開始監控。")
        time.sleep(wait_sec + 5)
    if live:
        from tw.shioaji_feed import subscribe_symbols

        subscribed = subscribe_symbols([s.symbol for s in result.candidates])
        print(f"\nwatch 永豐即時：已訂閱 {len(subscribed)} 檔，每分鐘收完 K 再判斷（Ctrl+C 結束）")
    else:
        print(f"\nwatch 模式（Yahoo 延遲），每 {args.interval} 秒重掃（Ctrl+C 結束）")
    alerted = 0
    try:
        while True:
            if live:
                _sleep_to_next_minute()
                if not _tw_session_open():
                    print("\n13:30 收盤，停止監控。")
                    break
            else:
                time.sleep(max(15, args.interval))
            new_hits, result = scan_once(args, seen, ranking=ranking)
            alerted += new_hits
            if live:
                from tw.shioaji_feed import subscribe_symbols

                subscribe_symbols([s.symbol for s in result.candidates])
    except KeyboardInterrupt:
        print("\n已停止。")
    print(f"今天共通知 {alerted} 檔。")
    if live:
        from tw.shioaji_feed import logout

        logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
