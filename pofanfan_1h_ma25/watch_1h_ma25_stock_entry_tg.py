#!/usr/bin/env python3
"""幣安股票 TradFi 永續 · 1H MA25 破底翻回測進場 → Telegram。

PyCharm：打開本檔 → Run（可加參數 --test / --once）。
Universe：TRADIFI_PERPETUAL + EQUITY / HK_EQUITY / KR_EQUITY / CN_EQUITY（無 meme/加密幣）。
訊號：與 binance_1h_ma25_reclaim.detect_signals / 訓練破底翻 相同的回測進場。
Pretty 預設開：4h≥MA25（容差2.0%）、破底前24h暴衝≤15%、站回後2~9根再回測。
v2/v3 濾網：破底低<MA200、破底前跌≥3.5%、一字線≤2根、4h MA120 24根跌>6%丟、排除反向ETF。
v7：進場收盤離 MA25 ≤2 ATR；MA60/99/120 全在價上且帶寬≤1.5 ATR 則丟。
僅進場推播，無出場／停損／5m M／36h 邏輯。
第一次只記現況不洗歷史；之後新進場才推。
憑證：專案根 tg_config.env（Binance @NQ6688688bot，非台灣 bot）。
API：https://www.binance.com + /fapi/v1/...（由 reclaim 模組處理）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

CONFIG_ENV = REPO / "tg_config.env"
SEEN_PATH = REPO / "output" / "ma25_stock_entry_tg_seen.json"

# 也可直接填；優先讀 tg_config.env
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TPE = ZoneInfo("Asia/Taipei")
POLL_SECONDS = 3600
LIMIT = 80
DAYS = 47  # 對齊回測 days+40；4h MA120 跌幅需 ≥144 根 4h
LOOKBACK_HOURS = 6
WORKERS = 3
# v3：排除反向 / 放空槓桿 ETF（SQQQ/SOXS/SKDD/TBT/TZA，見 reclaim.INVERSE_SYMBOLS）
EXCLUDE_INVERSE = True

# pretty ON + 與 1H破底翻TG / 訓練破底翻 對齊
DETECT_KW = dict(
    min_bars_below=4,
    max_bars_below=36,
    min_depth_pct=0.018,
    min_impulse_pct=0.0,
    min_undercut_pct=0.014,
    min_flush_atr=0.0,
    target_r=2.0,
    max_retest_wait=18,
    retest_tag_pct=0.003,
    retest_tag_atr=0.4,
    max_ma_drift_pct=1.5,
    min_ribbon_atr=0.50,
    ma200_near_pct=0.0,  # v2：破底低須 < MA200（%，原 1.0）
    min_ma_cluster_atr=1.0,
    min_break_vol=0.8,
    min_mid_ma_atr=0.50,
    require_h4_ma25=True,
    h4_ma25_tol_pct=2.0,  # %（原 1.5）
    max_prior_spike_pct=0.15,
    prior_spike_bars=24,
    min_reclaim_entry_bars=2,  # DELL 2026-10-02 站回→回測 2 根；3 會整段丟掉
    # v2 shape filters
    min_decline_pct=0.035,  # 分數：破底前跌≥3.5%
    max_flat_bars=2,  # 進場前24根一字線最多2根
    max_reclaim_entry_bars=9,  # 站回→進場最多9根
    # v3：4h MA120 近24根4h 跌幅 > 6% 則丟（單位 %，6 = 6%）
    h4_max_ma_drop_pct=6.0,
    h4_drop_ma=120,
    h4_drop_bars=24,
    # v7：HIMS 2026-10-02 收盤高出 MA25 2.91 ATR；ANET 2026-09-29 頭上 MA60/99/120 帶寬 0.61 ATR
    max_entry_ext_atr=2.0,
    max_overhead_knot_atr=1.5,
)


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
    # 備用：shared/tg_notify 的 env（若專案 env 缺欄位）
    shared_env = Path("/workspace/shared/tg_config.env")
    if shared_env.exists() and shared_env != CONFIG_ENV:
        load_dotenv(shared_env)
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def load_reclaim():
    try:
        import binance_1h_ma25_reclaim as m  # noqa: WPS433
        # BingX port: Binance is HTTP 451 here. Route REST through the BingX shim, no vision fallback.
        sys.path.insert(0, "/workspace/shared")
        import bingx_fapi  # noqa: WPS433
        m.get_json = bingx_fapi.get_json
        m._REST_BANNED = False
        m._fetch_klines_vision = lambda *a, **k: m._klines_to_df([])
        m.BASE = bingx_fapi.BASE
        return m
    except ImportError:
        print("找不到 binance_1h_ma25_reclaim.py，請跟本檔放同一層。", file=sys.stderr)
        raise


def tg_send(text: str) -> bool:
    token = (TELEGRAM_BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        # 嘗試共用 helper（仍讀同一組憑證）
        try:
            sys.path.insert(0, "/workspace/shared")
            from tg_notify import send as shared_send  # noqa: WPS433

            return bool(shared_send(text, parse_mode=None))
        except Exception as exc:  # noqa: BLE001
            print(f"[tg] 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（看 tg_config.env）: {exc}")
            print(text)
            return False
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return False
    return True


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        return set(data if isinstance(data, list) else data.get("keys") or [])
    except Exception:
        return set()


def save_seen(keys: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(keys), ensure_ascii=False, indent=2), encoding="utf-8")


def sig_key(symbol: str, df, sig) -> str:
    ts = df.index[sig.entry_idx]
    return f"{symbol}|{ts.strftime('%Y-%m-%dT%H:%M')}|{sig.entry_price:.8g}"


def fmt_sig(symbol: str, df, sig) -> str:
    ts = df.index[sig.entry_idx]
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC").tz_convert(TPE)
    else:
        ts = ts.tz_convert(TPE)
    br = df.index[sig.break_idx]
    if getattr(br, "tzinfo", None) is not None:
        br = br.tz_convert(TPE)
    risk = sig.entry_price - sig.stop_price
    risk_pct = (risk / sig.entry_price * 100) if sig.entry_price else 0
    return (
        f"股票永續 1H 破底翻 · 回測進場\n"
        f"{symbol}\n"
        f"時間 {ts.strftime('%m-%d %H:%M')} 台北\n"
        f"進場 {sig.entry_price:.6g}\n"
        f"參考停損 {sig.stop_price:.6g}  （破底 {br.strftime('%m-%d %H:%M')}）\n"
        f"風險 {risk_pct:.2f}%（僅參考，本監控不出場）\n"
        f"MA25 {sig.ma25:.6g}\n"
        f"品質 Q{sig.quality} {sig.shape}  破底量 {sig.break_vol_ratio:.2f}x"
    )


def pick_symbols(m, limit: int) -> list[str]:
    """股票 TradFi 永續（universe 已過濾）；成交額排序，最多 limit 檔。"""
    print("載入股票 TradFi 永續…", flush=True)
    symbols = m.universe(8_000_000)
    qv = m.ticker_quote_volume()
    # KEEP 裡若在股票 universe 才優先（AVGO/ONDS 等）
    keep = [s for s in symbols if s in getattr(m, "KEEP", set())]
    rest = [s for s in symbols if s not in keep]
    rest.sort(key=lambda s: qv.get(s, 0.0), reverse=True)
    out = keep + rest[: max(0, limit - len(keep))]
    seen: set[str] = set()
    out = [s for s in out if not (s in seen or seen.add(s))]
    if EXCLUDE_INVERSE:
        before = list(out)
        out = m.filter_symbols(out, exclude_inverse=True)
        dropped = [s for s in before if s not in out]
        if dropped:
            print(f"排除反向商品 {len(dropped)} 檔: {','.join(dropped)}", flush=True)
    return out


def scan_symbol(m, symbol: str):
    try:
        df = m.fetch_klines(symbol, interval="1h", days=DAYS, drop_forming=True)
    except Exception as exc:
        return symbol, None, str(exc)[:80]
    if df is None or len(df) < 40:
        return symbol, [], ""
    try:
        sigs = m.detect_signals(df, **DETECT_KW)
    except Exception as exc:
        return symbol, None, str(exc)[:80]
    cutoff = datetime.now(TPE) - timedelta(hours=LOOKBACK_HOURS)
    hits = []
    for sig in sigs:
        ts = df.index[sig.entry_idx]
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TPE)
        if ts >= cutoff:
            hits.append((df, sig))
    return symbol, hits, ""


def scan_once(m, symbols: list[str], *, seed: bool) -> int:
    seen = load_seen()
    new_n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(scan_symbol, m, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            symbol, hits, err = fut.result()
            if err:
                print(f"[{done}/{len(symbols)}] {symbol} {err}", flush=True)
                continue
            if hits:
                print(f"[{done}/{len(symbols)}] {symbol} hits={len(hits)}", flush=True)
            elif done % 10 == 0 or done == len(symbols):
                print(f"[{done}/{len(symbols)}] …", flush=True)
            for df, sig in hits or []:
                key = sig_key(symbol, df, sig)
                if key in seen:
                    continue
                seen.add(key)
                if seed:
                    continue
                text = fmt_sig(symbol, df, sig)
                print(text, flush=True)
                tg_send(text)
                new_n += 1
    save_seen(seen)
    return new_n


def main() -> int:
    p = argparse.ArgumentParser(description="幣安股票 TradFi 1H 破底翻進場 → Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram，不掃盤")
    p.add_argument("--once", action="store_true", help="seed + 掃一輪後結束")
    p.add_argument("--limit", type=int, default=LIMIT, help="最多幾檔股票永續")
    args = p.parse_args()

    apply_keys()

    if args.test:
        ok = tg_send(
            f"股票永續 1H 破底翻 entry test "
            f"{datetime.now(TPE).strftime('%Y-%m-%d %H:%M')} 台北"
        )
        print("Telegram 測試", "成功" if ok else "失敗", flush=True)
        return 0 if ok else 1

    m = load_reclaim()
    symbols = pick_symbols(m, args.limit)
    print(
        f"監控 {len(symbols)} 檔股票 TradFi  每 {POLL_SECONDS}s 掃一次  Ctrl+C 停",
        flush=True,
    )
    print("pretty：4h MA25±2.0% / 暴衝≤15% / 站回後3~9根 / <MA200 / 跌≥3.5% / 一字≤2 / 4h MA120跌≤6% / 無反向 · 僅進場", flush=True)
    print(f"seen → {SEEN_PATH}", flush=True)
    print("第一次先記現況，不推歷史單。", flush=True)
    scan_once(m, symbols, seed=True)
    tg_send(
        f"股票永續 1H 破底翻進場監控已開  {len(symbols)} 檔  "
        f"{datetime.now(TPE).strftime('%H:%M')} 台北"
    )
    while True:
        try:
            n = scan_once(m, symbols, seed=False)
            if n:
                print(f"新訊 {n} 筆", flush=True)
        except KeyboardInterrupt:
            print("stopped")
            return 0
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
