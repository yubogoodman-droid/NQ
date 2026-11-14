#!/usr/bin/env python3
"""SOXS 1 分 K「大賺型」假突破做空狙擊器（即時監看 + 預掛單推播 + 回測驗證）。

專門鎖定最容易走出 2%～5% 瀑布急殺的 S 級假突破空單（4 大特徵全中才觸發）：
  1. 美股流動性窗：08:35 ET 起允許盤前築頂，只在 09:20–16:00 ET（冬令台北 22:20–05:00）掛單／進場
  2. 先急拉累積多頭停損燃料：假高相對前 45 分鐘低點漲幅 ≥ 3.0%，且假高乖離 MA60 ≥ 2.0%
  3. 創高量縮背離：假高當根成交量 ≤ 前 35 分鐘起漲最大量的 65%（買盤枯竭）
  4. 破 MA25/30 且下方有均線中空帶：收盤跌破 MA25 與 MA30（MA7 < MA14），且掛單價距離下方 MA99 ≥ 0.8%

兩種推播時機（同一波只各推一次，帶 1m K 線圖）：
  • 【預掛空單準備】：剛收盤確認跌破 MA25/30 且符合大賺結構 → 馬上給出下一根 MA14/MA25 建議限價空單、停損、2.5R 目標價
  • 【掛單觸價進場】：1～10 根內反抽碰到 MA14/MA25 掛單帶正式進場 → 推播進場確認卡

用法：
  python3 examples/watch_soxs_big_win_tg.py --backtest --days 30   # 先看近 30 天大賺狙擊回測
  python3 examples/watch_soxs_big_win_tg.py --test                 # 測試 Telegram 通不通
  python3 examples/watch_soxs_big_win_tg.py --once --dry-run       # 單次掃描剛收盤 1m（印在終端機）
  python3 examples/watch_soxs_big_win_tg.py                        # 持續每根 1m 收盤監看並推 Telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from soxs_fake_breakout import (  # noqa: E402
    BASE,
    MA_COLORS,
    MA_WIDTH,
    NY,
    SESSION,
    TPE,
    Params,
    _limit_price,
    _stop_price,
    _use_cjk_font,
    fetch_klines,
    in_session,
    run_backtest,
    score_window,
    sma,
    summarize,
)

# —— 可直接填，或放在專案根目錄 tg_config.env ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

CONFIG_ENV = REPO / "tg_config.env"
SEEN_PATH = REPO / "output" / "soxs_big_win_seen.json"
DEFAULT_SYMBOLS = ("SOXSUSDT", "SQQQUSDT")

SNIPER_PARAMS = Params(
    lookback=120,
    min_sep=15,
    min_overshoot=0.002,
    max_overshoot=0.035,
    break_window=25,
    retest_window=10,
    target_r=2.5,
    max_hold=30,
    min_risk=0.006,
    max_risk=0.025,
    min_gap_ma99=0.008,
    min_pump45_pct=0.030,
    min_peak_ext60_pct=0.020,
    max_peak_vol_ratio=0.65,
    session="ny_open",
    mode="limit",
    days=30,
)


@dataclass(frozen=True)
class SniperSetup:
    symbol: str
    kind: str  # "arm" (跌破預掛準備) | "fill" (回測觸價進場)
    prev_idx: int
    prev_high: float
    peak_idx: int
    peak_high: float
    bd_idx: int
    signal_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    ma99: float
    gap_ma99_pct: float
    pump45_pct: float
    peak_ext60_pct: float
    peak_vol_ratio: float


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


def fetch_recent_1m(symbol: str, limit: int = 500) -> pd.DataFrame:
    """抓最近 limit 根已收盤 1 分 K（輕量單次請求，適合每分鐘輪詢）。"""
    now_ms = int(time.time() * 1000)
    last_err: Optional[Exception] = None
    raw = None
    for attempt in range(4):
        try:
            resp = SESSION.get(
                BASE + "/fapi/v1/klines",
                params={"symbol": symbol, "interval": "1m", "limit": limit},
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(1.2 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    if raw is None:
        raise RuntimeError(f"{symbol} klines failed: {last_err}")
    if raw and int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if not raw:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def detect_live_setups(
    symbol: str,
    df: pd.DataFrame,
    params: Params = SNIPER_PARAMS,
    recent_bars: int = 3,
) -> List[SniperSetup]:
    """掃描最近剛收盤的 recent_bars 根 K 線，抓出「剛跌破預掛（arm）」與「剛回測成交（fill）」訊號。"""
    n = len(df)
    if n < max(params.lookback + 5, 130):
        return []

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    ma7 = sma(c, 7)
    ma14 = sma(c, 14)
    ma25 = sma(c, 25)
    ma30 = sma(c, 30)
    ma60 = sma(c, 60)
    ma99 = sma(c, 99)

    min_signal_idx = max(0, n - max(1, recent_bars))
    out: List[SniperSetup] = []
    seen_peaks: Set[tuple] = set()

    i = max(params.lookback + 1, 105)
    while i < n:
        if not in_session(df.index[i], params.session, for_wave=True):
            i += 1
            continue
        if np.isnan(ma30[i]) or np.isnan(ma99[i]):
            i += 1
            continue

        win = h[i - params.lookback : i - params.min_sep + 1]
        prev_rel = int(np.argmax(win))
        prev_idx = i - params.lookback + prev_rel
        prev_high = float(win[prev_rel])
        if prev_high <= 0 or np.max(h[prev_idx + 1 : i]) > prev_high or h[i] <= prev_high:
            i += 1
            continue
        if not np.any(c[prev_idx + 1 : i] < ma25[prev_idx + 1 : i]):
            i += 1
            continue

        peak_h = float(h[i])
        peak_idx = i
        bd_idx: Optional[int] = None
        for j in range(i, min(n - 1, i + params.break_window) + 1):
            if h[j] > peak_h:
                peak_h = float(h[j])
                peak_idx = j
            ov = peak_h / prev_high - 1.0
            if ov > params.max_overshoot:
                break
            if j <= peak_idx or ov < params.min_overshoot:
                continue
            if c[j] < ma25[j] and c[j] < ma30[j] and c[j] < prev_high and ma7[j] < ma14[j]:
                bd_idx = j
                break
        if bd_idx is None:
            i += 1
            continue

        low45 = float(np.min(l[max(0, peak_idx - 45) : peak_idx + 1]))
        pump45 = (peak_h / low45 - 1.0) if low45 > 0 else 0.0
        pk_ext60 = (peak_h / float(ma60[peak_idx]) - 1.0) if not np.isnan(ma60[peak_idx]) else 0.0
        pump_max_v = float(np.max(v[max(0, peak_idx - 35) : peak_idx])) if peak_idx > 0 else float(v[peak_idx])
        pk_vol_ratio = (float(v[peak_idx]) / pump_max_v) if pump_max_v > 0 else 1.0

        if (
            (params.min_pump45_pct > 0 and pump45 < params.min_pump45_pct)
            or (params.min_peak_ext60_pct > 0 and pk_ext60 < params.min_peak_ext60_pct)
            or (params.max_peak_vol_ratio > 0 and pk_vol_ratio > params.max_peak_vol_ratio)
        ):
            i = bd_idx + 1
            continue

        # 1. 剛收盤在 bd_idx：發「預掛空單準備 (arm)」訊號
        if bd_idx >= min_signal_idx and (peak_idx, "arm") not in seen_peaks:
            lim = _limit_price(float(c[bd_idx]), float(ma14[bd_idx]), float(ma25[bd_idx]))
            m99_now = float(ma99[bd_idx])
            gap99 = (lim - m99_now) / lim if lim > 0 else 0.0
            stop = _stop_price(peak_h, lim, params)
            # 允許 09:15 ET 後提早 5 分鐘推播預掛提醒，或已在 ny_open 進場窗內
            ny_ts = df.index[bd_idx].tz_convert(NY)
            ny_min = ny_ts.hour * 60 + ny_ts.minute
            arm_time_ok = in_session(df.index[bd_idx], params.session) or (
                params.session == "ny_open" and ny_ts.weekday() < 5 and (9 * 60 + 10) <= ny_min < 16 * 60
            )
            if stop is not None and gap99 >= params.min_gap_ma99 and arm_time_ok:
                risk = stop - lim
                out.append(
                    SniperSetup(
                        symbol=symbol,
                        kind="arm",
                        prev_idx=prev_idx,
                        prev_high=prev_high,
                        peak_idx=peak_idx,
                        peak_high=peak_h,
                        bd_idx=bd_idx,
                        signal_idx=bd_idx,
                        entry_price=lim,
                        stop_price=stop,
                        target_price=lim - params.target_r * risk,
                        ma99=m99_now,
                        gap_ma99_pct=gap99,
                        pump45_pct=pump45,
                        peak_ext60_pct=pk_ext60,
                        peak_vol_ratio=pk_vol_ratio,
                    )
                )
                seen_peaks.add((peak_idx, "arm"))

        # 2. 跌破後 1～retest_window 根內剛觸價：發「回測觸價進場 (fill)」訊號
        for k in range(bd_idx + 1, min(n - 1, bd_idx + params.retest_window) + 1):
            prev_c = float(c[k - 1])
            if prev_c >= ma25[k - 1]:
                if prev_c > prev_high:
                    break
                continue
            lim = _limit_price(prev_c, float(ma14[k - 1]), float(ma25[k - 1]))
            prev_m99 = float(ma99[k - 1])
            if (lim - prev_m99) / lim < params.min_gap_ma99:
                continue
            stop = _stop_price(peak_h, lim, params)
            if stop is None or h[k] < lim or not in_session(df.index[k], params.session):
                continue
            fill = float(o[k]) if o[k] > lim else lim
            if fill >= stop:
                break
            if k >= min_signal_idx and (peak_idx, "fill") not in seen_peaks:
                risk = stop - fill
                out.append(
                    SniperSetup(
                        symbol=symbol,
                        kind="fill",
                        prev_idx=prev_idx,
                        prev_high=prev_high,
                        peak_idx=peak_idx,
                        peak_high=peak_h,
                        bd_idx=bd_idx,
                        signal_idx=k,
                        entry_price=fill,
                        stop_price=stop,
                        target_price=fill - params.target_r * risk,
                        ma99=prev_m99,
                        gap_ma99_pct=(fill - prev_m99) / fill,
                        pump45_pct=pump45,
                        peak_ext60_pct=pk_ext60,
                        peak_vol_ratio=pk_vol_ratio,
                    )
                )
                seen_peaks.add((peak_idx, "fill"))
            break

        i = bd_idx + 1

    return out


def format_setup_message(df: pd.DataFrame, s: SniperSetup) -> str:
    ts_tpe = df.index[s.signal_idx].tz_convert(TPE).strftime("%m-%d %H:%M")
    ts_ny = df.index[s.signal_idx].tz_convert(NY).strftime("%H:%M ET")
    pk_tpe = df.index[s.peak_idx].tz_convert(TPE).strftime("%H:%M")
    risk_pct = (s.stop_price - s.entry_price) / s.entry_price * 100.0
    tp_pct = (s.entry_price - s.target_price) / s.entry_price * 100.0
    header = (
        f"🎯 <b>【S級大賺狙擊 · 觸價進場】</b> {s.symbol}"
        if s.kind == "fill"
        else f"⚡ <b>【S級大賺狙擊 · 預掛空單】</b> {s.symbol}"
    )
    action_line = (
        f"進場成交：<b>{s.entry_price:.3f}</b>（回測 MA14/25 觸價）"
        if s.kind == "fill"
        else f"建議預掛空：<b>{s.entry_price:.3f}</b>（MA14/25 反壓帶，等 1–10 根）"
    )
    return (
        f"{header}\n"
        f"時間：{ts_tpe} 台北（{ts_ny}）\n"
        f"{action_line}\n"
        f"停損價：<b>{s.stop_price:.3f}</b>（+{risk_pct:.2f}%，假高 {s.peak_high:.2f} @ {pk_tpe}）\n"
        f"停利價：<b>{s.target_price:.3f}</b>（-{tp_pct:.2f}%，2.5R / 最長拿 30 分鐘）\n"
        f"—— 大賺四特徵檢核 ——\n"
        f"• 均線中空帶：離 MA99（{s.ma99:.2f}）空間 <b>+{s.gap_ma99_pct * 100:.2f}%</b>\n"
        f"• 前 45m 急拉：<b>+{s.pump45_pct * 100:.2f}%</b>（過前高 {s.prev_high:.2f}）\n"
        f"• 假高乖離 MA60：<b>+{s.peak_ext60_pct * 100:.2f}%</b>\n"
        f"• 創高量縮比：起漲最大量的 <b>{s.peak_vol_ratio * 100:.0f}%</b>"
    )


def draw_setup_chart(df: pd.DataFrame, s: SniperSetup, path: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:  # noqa: BLE001
        return None

    _use_cjk_font()
    start = max(0, min(s.prev_idx, s.signal_idx - 80) - 10)
    end = min(len(df) - 1, s.signal_idx + 5)
    window = df.iloc[start : end + 1]
    xs = np.arange(len(window))
    o = window["Open"].to_numpy(float)
    h = window["High"].to_numpy(float)
    l = window["Low"].to_numpy(float)
    c = window["Close"].to_numpy(float)
    v = window["Volume"].to_numpy(float)
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}, facecolor="#0c1210"
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors = []
    for k in range(len(window)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.65)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.33, y0), 0.66, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, v, width=0.8, color=colors, linewidth=0)

    for period, col in MA_COLORS.items():
        ma = close_full.rolling(period, min_periods=period).mean().iloc[start : end + 1]
        ax.plot(xs, ma, color=col, lw=MA_WIDTH[period], label=f"MA{period}")

    ax.axhline(s.prev_high, color="#8aa193", ls="--", lw=0.85, alpha=0.7)
    ax.axhline(s.entry_price, color="#f0b90b", ls="--", lw=0.95, alpha=0.9)
    ax.axhline(s.stop_price, color="#e35d5d", ls=":", lw=1.05, alpha=0.95)
    ax.axhline(s.target_price, color="#3dba7a", ls=":", lw=1.05, alpha=0.9)

    px_x = s.peak_idx - start
    sx_x = s.signal_idx - start
    if 0 <= px_x < len(window):
        ax.scatter([px_x], [s.peak_high], s=40, color="#f0c14b", zorder=6)
        ax.annotate("假高", (px_x, s.peak_high), textcoords="offset points", xytext=(0, 7), ha="center", color="#f0c14b", fontsize=8)
    if 0 <= sx_x < len(window):
        tag = "空" if s.kind == "fill" else "掛空"
        ax.scatter([sx_x], [s.entry_price], s=46, color="#ff5252", marker="v", zorder=7)
        ax.annotate(tag, (sx_x, s.entry_price), textcoords="offset points", xytext=(0, -14), ha="center", color="#ff8a80", fontsize=8)

    ts_str = df.index[s.signal_idx].tz_convert(TPE).strftime("%m-%d %H:%M")
    mode_str = "觸價進場" if s.kind == "fill" else "預掛準備"
    ax.set_title(
        f"{s.symbol} 1m S級大賺狙擊（{mode_str}） {ts_str} TPE  entry={s.entry_price:.2f}  sl={s.stop_price:.2f}  tp={s.target_price:.2f}",
        color="#e8f0ea",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=8)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[k].tz_convert(TPE).strftime("%H:%M") for k in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def setup_key(df: pd.DataFrame, s: SniperSetup) -> str:
    peak_ms = int(df.index[s.peak_idx].timestamp() * 1000)
    return f"{s.symbol}:{peak_ms}:{s.kind}"


def load_seen(path: Path = SEEN_PATH) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return set()


def save_seen(seen: Set[str], path: Path = SEEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen)[-500:], indent=2), encoding="utf-8")


def tg_send(text: str, photo: Optional[Path] = None, dry_run: bool = False) -> bool:
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    print("\n" + plain, flush=True)
    if dry_run:
        if photo:
            print(f"  [dry-run 圖檔] {photo}", flush=True)
        return True

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("  → 未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，僅印在終端機", flush=True)
        return False
    try:
        if photo and photo.exists():
            with open(photo, "rb") as fh:
                r = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                    files={"photo": fh},
                    timeout=25,
                )
            if r.ok:
                return True
        r = SESSION.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException as exc:
        print(f"  → Telegram 傳送失敗: {exc}", file=sys.stderr, flush=True)
        return False


def wait_next_minute_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1.0, nxt - now))


def run_sniper_backtest(symbols: Sequence[str], days: int) -> int:
    for sym in symbols:
        print(f"=== {sym} 大賺狙擊回測（近 {days} 天） ===", flush=True)
        df = fetch_klines(sym, days)
        params = Params(**{**SNIPER_PARAMS.__dict__, "days": days})
        trades, funnel = run_backtest(df, params)
        s30 = summarize(trades)
        w60 = score_window(df, trades, 60, params.fee_pct)
        print(
            f"30m 結算：筆數={s30['count']}  勝率={s30['win_rate']:.1f}% ({s30['wins']}/{s30['count']})  "
            f"淨報酬={s30['net_pct']:+.2f}%  平均={s30['avg_pct']:+.2f}%  合計={s30['sum_r']:+.2f}R  出場={s30['reasons']}"
        )
        print(
            f"60m 延伸：勝率={w60['win_rate']:.1f}% ({w60['wins']}/{w60['count']})  "
            f"淨報酬={w60['net_pct']:+.2f}%  出場={w60['reasons']}"
        )
        print(f"漏斗：{funnel}")
        for idx, t in enumerate(trades, 1):
            et = df.index[t.entry_idx].tz_convert(TPE).strftime("%m-%d %H:%M")
            xt = df.index[t.exit_idx].tz_convert(TPE).strftime("%H:%M")
            print(
                f"  [{idx:02d}] {et} -> {xt}  空={t.entry_price:.2f} 停損={t.stop_price:.2f} 出={t.exit_price:.2f} "
                f"{t.exit_reason:6s} {t.net_pct * 100:+.2f}% ({t.r_multiple:+.2f}R) | "
                f"離MA99={t.gap_ma99_pct * 100:+.2f}% 急拉={t.pump45_pct * 100:.1f}% "
                f"乖離MA60={t.peak_ext60_pct * 100:.1f}% 創高量比={t.peak_vol_ratio * 100:.0f}%"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SOXS 1m S級大賺假突破狙擊器（回測 + 即時監看推播）")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="逗號分隔標的，預設 SOXSUSDT,SQQQUSDT")
    p.add_argument("--backtest", action="store_true", help="跑歷史大賺狙擊回測並印出逐筆明細")
    p.add_argument("--days", type=int, default=30, help="回測天數（搭配 --backtest）")
    p.add_argument("--once", action="store_true", help="只掃描剛收盤的 1m 一次即結束")
    p.add_argument("--recent-bars", type=int, default=3, help="每次掃描往回看幾根剛收盤 1m")
    p.add_argument("--dry-run", action="store_true", help="只印終端機與畫圖，不發 Telegram")
    p.add_argument("--test", action="store_true", help="發一則 Telegram 測試訊息")
    args = p.parse_args(argv)

    apply_keys()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    if args.test:
        now_tpe = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
        ok = tg_send(f"✅ <b>SOXS S級大賺狙擊器</b> 連線測試成功\n台北時間：{now_tpe}", dry_run=args.dry_run)
        return 0 if ok else 1

    if args.backtest:
        return run_sniper_backtest(symbols, args.days)

    seen = load_seen()
    print(f"啟動 S級大賺假突破監看：{', '.join(symbols)}（美東 09:20–16:00 進場窗）", flush=True)

    def scan_round() -> None:
        now_tpe = datetime.now(TPE)
        now_ny = now_tpe.astimezone(NY)
        found = 0
        for sym in symbols:
            try:
                df = fetch_recent_1m(sym, limit=500)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] {sym}: {exc}", file=sys.stderr, flush=True)
                continue
            setups = detect_live_setups(sym, df, SNIPER_PARAMS, recent_bars=args.recent_bars)
            for s in setups:
                key = setup_key(df, s)
                if key in seen:
                    continue
                chart_path = Path("/tmp") / f"sniper_{s.symbol}_{s.kind}_{s.signal_idx}.png"
                photo = draw_setup_chart(df, s, chart_path)
                tg_send(format_setup_message(df, s), photo=photo, dry_run=args.dry_run)
                seen.add(key)
                found += 1
        if found:
            save_seen(seen)
        print(
            f"[{now_tpe.strftime('%H:%M:%S')} TPE / {now_ny.strftime('%H:%M')} ET] "
            f"掃描 {len(symbols)} 檔完成，新訊號={found}",
            flush=True,
        )

    scan_round()
    if args.once:
        return 0
    try:
        while True:
            wait_next_minute_close()
            scan_round()
    except KeyboardInterrupt:
        print("\n已停止監看。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
