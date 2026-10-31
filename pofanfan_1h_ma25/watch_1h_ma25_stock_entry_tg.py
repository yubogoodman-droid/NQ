#!/usr/bin/env python3
"""幣安 + BingX 股票 TradFi 永續 · 1H MA25 破底翻 v11 進出場全自動監控 → Telegram（含 K 線圖推播 + 出場追蹤）。

功能強化（v11）：
1. 進場 + 出場全流程推播：
   - 進場訊號觸發時，自動繪製當下 1H + 4H 對照 K 線圖（`sendPhoto`）連同進場價、4 個明確出場點與「以損定量建議槓桿」推播至 Telegram。
   - 自動將未平倉訊號寫入 `output/ma25_stock_open_positions.json`，每小時輪詢追蹤後續 K 線（含 5m M 頭）：
     一旦觸發 ① 破底收盤停損 (`stop`)、② 連 5 根 1H 收盤掉回 MA25 (`lost_ma25`)、③ 1.2R 武裝後五分 M 頭破頸停利 (`m5_neck`)、或 ④ 滿 24h 到期 (`timeout`)，
     立即推播「出場通知 + 結算 K 線圖」！
2. 支援雙交易所股票池（`--exchange binance | bingx | both`，預設 `both`）：
   - 先掃幣安 216 檔 `TRADIFI_PERPETUAL`，再掃 BingX 獨有的 167 檔美股中小型飆股（共 383+ 檔聯集池）。
3. 內建 v11 跨平台防停損條件：
   - `min_bars_below=8`（在 MA25 下方至少 8h，排除 4~6h 急刺假底）
   - `ma200_near_pct=-2.5`（破底低於 1H MA200 至少 2.5%）
   - `h4_ma60_max_rise_pct=0.0`（4H MA60 近 12 根斜率 ≤ 0%，拒高檔初跌段接刀）
   - `min_entry_rng_pct=0.45`（進場回測棒高低振幅 ≥ 0.45%，拒極窄十字星）
   - `min_overhead_ma_atr=0.25`（距離上方最近長均線 MA60/99/120/200 ≥ 0.25 ATR，不買在長均線天花板正下方）
   - `min_entry_vol=0.15`、`require_ma7_above_ma14=True`、`lose_ma_closes=5`、`max_hold=24`、`m5_arm_r=1.2`
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

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "shared"))

CONFIG_ENV = REPO / "tg_config.env"
SEEN_PATH = REPO / "output" / "ma25_stock_entry_tg_seen.json"
OPEN_POS_PATH = REPO / "output" / "ma25_stock_open_positions.json"
CHART_DIR = REPO / "output" / "tg_charts"

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TPE = ZoneInfo("Asia/Taipei")
POLL_SECONDS = 3600
LIMIT = 400
DAYS = 47
LOOKBACK_HOURS = 6
WORKERS = 4
EXCLUDE_INVERSE = True
MAX_HOLD_HOURS = 24
LOSE_MA_CLOSES = 5
M5_ARM_R = 1.2

# v11 跨平台終極防停損參數
DETECT_KW = dict(
    min_bars_below=8,
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
    ma200_near_pct=-2.5,
    min_ma_cluster_atr=1.0,
    min_break_vol=0.8,
    min_mid_ma_atr=0.50,
    require_h4_ma25=True,
    h4_ma25_tol_pct=2.0,
    max_prior_spike_pct=0.15,
    prior_spike_bars=24,
    min_reclaim_entry_bars=2,
    min_decline_pct=0.035,
    max_flat_bars=2,
    max_reclaim_entry_bars=9,
    h4_max_ma_drop_pct=6.0,
    h4_drop_ma=120,
    h4_drop_bars=24,
    max_entry_ext_atr=2.0,
    max_overhead_knot_atr=1.5,
    min_entry_vol=0.15,
    require_ma7_above_ma14=True,
    h4_ma60_max_rise_pct=0.0,
    min_entry_rng_pct=0.45,
    min_overhead_ma_atr=0.25,
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
    shared_env = Path("/workspace/shared/tg_config.env")
    if shared_env.exists() and shared_env != CONFIG_ENV:
        load_dotenv(shared_env)
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def load_reclaim():
    import binance_1h_ma25_reclaim as m  # noqa: WPS433

    m.DRAW_W_OVERLAY = False
    m.M5_ARM_R = M5_ARM_R
    return m


def tg_send(text: str, photo_path: Path | None = None) -> bool:
    token = (TELEGRAM_BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        print(f"[tg-dry] {text}")
        if photo_path:
            print(f"[tg-dry-photo] {photo_path}")
        return False
    try:
        if photo_path and photo_path.is_file():
            with photo_path.open("rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024]},
                    files={"photo": f},
                    timeout=35,
                )
            if r.ok:
                return True
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=30,
        )
        if not r.ok:
            print(f"[tg] HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[tg] send error: {exc}", file=sys.stderr)
        return False


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


def load_open_positions() -> list[dict]:
    if not OPEN_POS_PATH.exists():
        return []
    try:
        return json.loads(OPEN_POS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_open_positions(positions: list[dict]) -> None:
    OPEN_POS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPEN_POS_PATH.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")


def sig_key(symbol: str, df, sig) -> str:
    ts = df.index[sig.entry_idx]
    return f"{symbol}|{ts.strftime('%Y-%m-%dT%H:%M')}|{sig.entry_price:.8g}"


def suggested_leverage(risk_pct: float, target_acct_risk_pct: float = 12.0) -> float:
    """以損定量：單筆若觸發破底停損，控制在帳戶約 12% 風險，槓桿限制在 1.5x ~ 4.5x。"""
    if risk_pct <= 0:
        return 3.0
    return float(np.clip(target_acct_risk_pct / max(1.0, risk_pct), 1.5, 4.5))


def fmt_entry_sig(symbol: str, source_ex: str, df, sig) -> str:
    ts = df.index[sig.entry_idx]
    ts = ts.tz_convert(TPE) if getattr(ts, "tzinfo", None) is not None else ts.tz_localize("UTC").tz_convert(TPE)
    br = df.index[sig.break_idx]
    br = br.tz_convert(TPE) if getattr(br, "tzinfo", None) is not None else br
    risk = sig.entry_price - sig.stop_price
    risk_pct = (risk / sig.entry_price * 100.0) if sig.entry_price else 0.0
    arm_12r = sig.entry_price + M5_ARM_R * risk
    exit_24h = ts + timedelta(hours=MAX_HOLD_HOURS)
    lev = suggested_leverage(risk_pct)
    return (
        f"🟢 股票永續 1H 破底翻 v11 · 進場訊號 [{source_ex}]\n"
        f"{symbol}\n"
        f"時間 {ts.strftime('%m-%d %H:%M')} 台北\n"
        f"進場 {sig.entry_price:.6g}（MA25 {sig.ma25:.6g}）\n"
        f"建議槓桿 {lev:.1f}x（以損定量：風險 {risk_pct:.2f}% ≈ 帳戶 12% 風險）\n"
        f"出場① 破底停損：{sig.stop_price:.6g}（1H收盤跌破破底 {br.strftime('%m-%d %H:%M')}，-{risk_pct:.2f}%）\n"
        f"出場② 結構止損：進場後若連 {LOSE_MA_CLOSES} 根 1H 收盤掉回 MA25 下方，提前止損出場\n"
        f"出場③ 破頸停利：1H收盤≥{arm_12r:.6g} (+{M5_ARM_R:.1f}R) 後，五分K M頭破頸線≥1%出場（或先平半倉移保本）\n"
        f"出場④ 時間到期：滿 {MAX_HOLD_HOURS}h 到期出場（{exit_24h.strftime('%m-%d %H:%M')} 台北 1H收盤）\n"
        f"品質 Q{sig.quality} {sig.shape} · 在下 {sig.bars_below}h · 破底量 {sig.break_vol_ratio:.2f}x · 進場量 {sig.vol_ratio:.2f}x"
    )


def fmt_exit_sig(pos: dict, trade) -> str:
    reason_zh = {
        "m5_neck": "🎯 五分K M頭破頸停利 (m5_neck)",
        "timeout": f"⏰ 滿 {MAX_HOLD_HOURS}h 時間到期出場 (timeout)",
        "lost_ma25": f"🛡️ 連 {LOSE_MA_CLOSES} 根 1H 掉回 MA25 提前止損 (lost_ma25)",
        "stop": "🛑 1H 收盤跌破破底停損 (stop)",
    }.get(trade.exit_reason, trade.exit_reason)
    lev = float(pos.get("lev") or 3.0)
    acct_ret = trade.pnl_pct * lev
    return (
        f"🔔 股票永續 1H 破底翻 v11 · 出場通知 [{pos.get('exchange', 'Binance')}]\n"
        f"{pos['symbol']}\n"
        f"進場 {pos['entry_tpe']} @ {trade.entry_price:.6g}\n"
        f"出場 {pos['exit_tpe']} @ {trade.exit_price:.6g}\n"
        f"原因 {reason_zh}\n"
        f"單筆報酬 {trade.pnl_pct:+.2f}%（{lev:.1f}x 帳戶損益 {acct_ret:+.2f}%）"
    )


def fetch_symbol_klines(m, symbol: str, exchange: str, interval: str = "1h", days: int = DAYS):
    if exchange == "BingX":
        import bingx_fapi  # noqa: WPS433

        orig_get, orig_base, orig_vis = m.get_json, m.BASE, m._fetch_klines_vision
        try:
            m.get_json = bingx_fapi.get_json
            m.BASE = bingx_fapi.BASE
            m._REST_BANNED = False
            m._fetch_klines_vision = lambda *a, **k: m._klines_to_df([])
            return m.fetch_klines(symbol, interval=interval, days=days, drop_forming=True)
        finally:
            m.get_json, m.BASE, m._fetch_klines_vision = orig_get, orig_base, orig_vis
    return m.fetch_klines(symbol, interval=interval, days=days, drop_forming=True)


def build_symbol_pool(m, mode: str, limit: int) -> list[tuple[str, str]]:
    """回傳 [(symbol, exchange_name), ...]。both 模式優先用 Binance，BingX 獨有標的用 BingX。"""
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    if mode in ("binance", "both"):
        try:
            bn_syms = m.universe(8_000_000)
            bn_syms = m.filter_symbols(bn_syms, exclude_inverse=EXCLUDE_INVERSE)
            for s in bn_syms:
                if s not in seen:
                    seen.add(s)
                    pool.append((s, "Binance"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Binance universe failed: {exc}", flush=True)

    if mode in ("bingx", "both"):
        try:
            import bingx_fapi  # noqa: WPS433

            info = bingx_fapi.get_json("/fapi/v1/exchangeInfo")
            bx_syms = [x["symbol"] for x in (info.get("symbols") or [])]
            bx_syms = m.filter_symbols(bx_syms, exclude_inverse=EXCLUDE_INVERSE)
            for s in bx_syms:
                if s not in seen:
                    seen.add(s)
                    pool.append((s, "BingX"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] BingX universe failed: {exc}", flush=True)

    return pool[:limit] if limit > 0 else pool


def check_open_positions(m) -> int:
    positions = load_open_positions()
    if not positions:
        return 0
    still_open: list[dict] = []
    closed_cnt = 0
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for pos in positions:
        sym = pos["symbol"]
        exch = pos.get("exchange", "Binance")
        entry_tpe = pos["entry_tpe"]
        try:
            df = fetch_symbol_klines(m, sym, exch, interval="1h", days=DAYS)
            if df is None or len(df) < 40:
                still_open.append(pos)
                continue
            sigs = m.detect_signals(df, **DETECT_KW)
            matched = [s for s in sigs if df.index[s.entry_idx].strftime("%Y-%m-%d %H:%M") == entry_tpe]
            if not matched:
                still_open.append(pos)
                continue
            sig = matched[0]
            bars_since_entry = (len(df) - 1) - sig.entry_idx
            if bars_since_entry < 1:
                still_open.append(pos)
                continue
            df5 = fetch_symbol_klines(m, sym, exch, interval="5m", days=4)
            trades = m.simulate(
                df,
                [sig],
                df_5m=df5,
                m5_exit=True,
                max_hold=MAX_HOLD_HOURS,
                lose_ma_closes=LOSE_MA_CLOSES,
                m5_arm_r=M5_ARM_R,
            )
            if not trades:
                still_open.append(pos)
                continue
            tr = trades[0]
            # 若 exit_reason 是 timeout 但還沒滿 24 根，代表還在持有中！
            if tr.exit_reason == "timeout" and (tr.exit_idx - tr.entry_idx) < MAX_HOLD_HOURS:
                still_open.append(pos)
                continue
            pos["exit_tpe"] = df.index[tr.exit_idx].strftime("%Y-%m-%d %H:%M")
            chart_path = CHART_DIR / f"exit_{sym}_{df.index[tr.exit_idx].strftime('%m%d_%H%M')}.png"
            m.draw_trade_png(df, tr, chart_path, 1, title_extra=f"{sym} [{exch}]")
            msg = fmt_exit_sig(pos, tr)
            print(msg, flush=True)
            tg_send(msg, photo_path=chart_path)
            closed_cnt += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] check_open {sym}: {exc}", flush=True)
            still_open.append(pos)
    save_open_positions(still_open)
    return closed_cnt


def scan_symbol(m, symbol: str, exch: str):
    try:
        df = fetch_symbol_klines(m, symbol, exch, interval="1h", days=DAYS)
    except Exception as exc:  # noqa: BLE001
        return symbol, exch, None, str(exc)[:80]
    if df is None or len(df) < 40:
        return symbol, exch, [], ""
    try:
        sigs = m.detect_signals(df, **DETECT_KW)
    except Exception as exc:  # noqa: BLE001
        return symbol, exch, None, str(exc)[:80]
    cutoff = datetime.now(TPE) - timedelta(hours=LOOKBACK_HOURS)
    hits = []
    for sig in sigs:
        ts = df.index[sig.entry_idx]
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TPE)
        if ts >= cutoff:
            hits.append((df, sig))
    return symbol, exch, hits, ""


def scan_once(m, pool: list[tuple[str, str]], *, seed: bool) -> int:
    seen = load_seen()
    open_pos = load_open_positions()
    new_n = 0
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(scan_symbol, m, s, exch): (s, exch) for s, exch in pool}
        done = 0
        for fut in as_completed(futs):
            done += 1
            symbol, exch, hits, err = fut.result()
            if err:
                continue
            if hits:
                print(f"[{done}/{len(pool)}] {symbol} ({exch}) hits={len(hits)}", flush=True)
            elif done % 40 == 0 or done == len(pool):
                print(f"[{done}/{len(pool)}] …", flush=True)
            for df, sig in hits or []:
                key = sig_key(symbol, df, sig)
                if key in seen:
                    continue
                seen.add(key)
                if seed:
                    continue
                # 畫進場當下的 K 線圖（此時 exit_idx 暫設為 entry_idx）
                dummy_tr = m.TradeResult(
                    signal=sig,
                    entry_idx=sig.entry_idx,
                    exit_idx=sig.entry_idx,
                    entry_price=sig.entry_price,
                    exit_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    target_price=sig.target_price,
                    pnl_pct=0.0,
                    exit_reason="entry",
                    quality=sig.quality,
                )
                et_str = df.index[sig.entry_idx].strftime("%Y-%m-%d %H:%M")
                chart_path = CHART_DIR / f"entry_{symbol}_{df.index[sig.entry_idx].strftime('%m%d_%H%M')}.png"
                m.draw_trade_png(df, dummy_tr, chart_path, 1, title_extra=f"{symbol} [{exch}]")
                text = fmt_entry_sig(symbol, exch, df, sig)
                print(text, flush=True)
                tg_send(text, photo_path=chart_path)
                risk_pct = (sig.entry_price - sig.stop_price) / sig.entry_price * 100.0 if sig.entry_price else 3.0
                open_pos.append({
                    "key": key,
                    "symbol": symbol,
                    "exchange": exch,
                    "entry_tpe": et_str,
                    "entry_price": sig.entry_price,
                    "stop_price": sig.stop_price,
                    "lev": round(suggested_leverage(risk_pct), 2),
                })
                new_n += 1
    save_seen(seen)
    save_open_positions(open_pos)
    return new_n


def main() -> int:
    p = argparse.ArgumentParser(description="幣安 + BingX 股票永續 1H 破底翻 v11 進出場監控 → Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram，不掃盤")
    p.add_argument("--once", action="store_true", help="seed + 掃一輪後結束")
    p.add_argument("--exchange", choices=["binance", "bingx", "both"], default="both", help="掃描交易所池（預設 both 聯集 383+ 檔）")
    p.add_argument("--limit", type=int, default=LIMIT, help="最多幾檔股票永續")
    args = p.parse_args()

    apply_keys()

    if args.test:
        ok = tg_send(
            f"股票永續 1H 破底翻 v11 (進出場+圖卡監控) test "
            f"{datetime.now(TPE).strftime('%Y-%m-%d %H:%M')} 台北"
        )
        print("Telegram 測試", "成功" if ok else "失敗", flush=True)
        return 0 if ok else 1

    m = load_reclaim()
    pool = build_symbol_pool(m, args.exchange, args.limit)
    print(
        f"監控 {len(pool)} 檔股票永續 ({args.exchange}) · 每 {POLL_SECONDS}s 掃一次 · 含 K 線圖與出場追蹤",
        flush=True,
    )
    scan_once(m, pool, seed=True)
    tg_send(
        f"股票永續 1H 破底翻 v11 進出場監控已開 · {len(pool)} 檔 ({args.exchange}) · "
        f"{datetime.now(TPE).strftime('%H:%M')} 台北"
    )
    while True:
        try:
            closed_n = check_open_positions(m)
            new_n = scan_once(m, pool, seed=False)
            if new_n or closed_n:
                print(f"新進場 {new_n} 筆 · 新出場 {closed_n} 筆", flush=True)
        except KeyboardInterrupt:
            print("stopped")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
