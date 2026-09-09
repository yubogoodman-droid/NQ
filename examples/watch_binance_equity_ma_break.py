#!/usr/bin/env python3
"""幣安美股永續：開盤瀑布 Telegram。

找你那種 15m：盤整後長陰放量摔出 MA7/14/25/99/120/200。
進場當下 1h K 須 MA7 < MA14 < MA25 空頭排列（用截至該根 15m 的小時收盤，不偷看未走完的小時 K）。
只在美東 09:00–10:00（開盤前後各半小時）報，週末不掃。

    python3 examples/watch_binance_equity_ma_break.py --test
    python3 examples/watch_binance_equity_ma_break.py --backtest --days 7
    python3 examples/watch_binance_equity_ma_break.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

import numpy as np
import requests

# —— 也可填這裡；有 tg_config.env 會蓋過來 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

ET = ZoneInfo("America/New_York")
TZ8 = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_equity_ma_break_seen.json"
CONFIG_ENV = REPO / "tg_config.env"
INTERVAL = "15m"
INTERVAL_MS = 900_000
HOUR_MS = 3_600_000
MA_PERIODS = (7, 14, 25, 99, 120, 200)
HOUR_MA_PERIODS = (7, 14, 25)
SESSION_START = (9, 0)  # 開盤前 30 分
SESSION_END = (10, 0)  # 開盤後 30 分
CASH_CLOSE = (16, 0)  # 回測「當日收」仍看到美東 16:00
MIN_BODY_PCT = 1.20
MIN_RANGE_PCT = 2.50
MIN_DROP_PCT = 2.50
MIN_VOL_RATIO = 1.80
DROP_LOOKBACK = 16
HORIZONS = ((1, "15m"), (2, "30m"), (4, "1h"), (8, "2h"), (16, "4h"))
PAGES_HTML = REPO / "docs" / "binance" / "ma-break-7d.html"
CHART_DIR = REPO / "docs" / "binance" / "img" / "ma-break"
_MPL_LOCK = threading.Lock()

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"}
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
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))
    raise last


def universe() -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "TRADIFI_PERPETUAL":
            continue
        if s.get("underlyingType") != "EQUITY":
            continue
        out.append(s["symbol"])
    return sorted(out)


def fetch_klines(sym: str, limit: int = 260) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": INTERVAL, "limit": limit})
    if not raw or len(raw) < 210:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + INTERVAL_MS > now_ms:
        raw = raw[:-1]
    if len(raw) < 210:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def indicators(d: dict) -> dict:
    out = dict(d)
    c = d["c"]
    for n in MA_PERIODS:
        out[f"m{n}"] = sma(c, n)
    return out


def below_all(d: dict, i: int) -> bool:
    if i < 0 or i >= len(d["c"]):
        return False
    vals = [d[f"m{n}"][i] for n in MA_PERIODS]
    if any(np.isnan(v) for v in vals):
        return False
    px = d["c"][i]
    return all(px < v for v in vals)


def is_fresh_break(d: dict, i: int) -> bool:
    return below_all(d, i) and not below_all(d, i - 1)


def bar_body_pct(d: dict, i: int) -> float:
    o = float(d["o"][i])
    if o == 0:
        return 0.0
    return (o - float(d["c"][i])) / o * 100.0


def bar_range_pct(d: dict, i: int) -> float:
    c = float(d["c"][i])
    if c == 0:
        return 0.0
    return (float(d["h"][i]) - float(d["l"][i])) / c * 100.0


def vol_ratio(d: dict, i: int) -> float:
    v20 = sma(d["v"], 20)[i]
    if np.isnan(v20) or v20 <= 0:
        return 0.0
    return float(d["v"][i] / v20)


def drop_from_swing_pct(d: dict, i: int) -> float:
    a0 = max(0, i - DROP_LOOKBACK)
    hi = float(np.max(d["h"][a0 : i + 1]))
    if hi <= 0:
        return 0.0
    return (hi - float(d["c"][i])) / hi * 100.0


def break_depth_pct(d: dict, i: int) -> float:
    mas = [float(d[f"m{n}"][i]) for n in MA_PERIODS]
    if any(np.isnan(v) for v in mas):
        return 0.0
    px = float(d["c"][i])
    if px <= 0:
        return 0.0
    return (min(mas) / px - 1) * 100


def prev_near_ribbon(d: dict, i: int) -> bool:
    if i < 1:
        return False
    pc = d["c"][i - 1]
    return any(pc >= d[f"m{n}"][i - 1] for n in (7, 14, 25, 200))


def hourly_ohlcv(d: dict, i_end: int | None = None) -> dict:
    """15m 合成 1h。i_end 有值時只看到該根 15m（含），不含之後。"""
    n = len(d["c"]) if i_end is None else min(max(i_end + 1, 0), len(d["c"]))
    empty = {
        "t": np.array([], np.int64),
        "o": np.array([]),
        "h": np.array([]),
        "l": np.array([]),
        "c": np.array([]),
        "v": np.array([]),
    }
    if n <= 0:
        return empty
    hour = (d["t"][:n] // HOUR_MS) * HOUR_MS
    change = np.ones(n, dtype=bool)
    change[1:] = hour[1:] != hour[:-1]
    starts = np.flatnonzero(change)
    ot, oo, hh, ll, cc, vv = [], [], [], [], [], []
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else n
        ot.append(int(hour[a]))
        oo.append(float(d["o"][a]))
        hh.append(float(np.max(d["h"][a:b])))
        ll.append(float(np.min(d["l"][a:b])))
        cc.append(float(d["c"][b - 1]))
        vv.append(float(np.sum(d["v"][a:b])))
    return {
        "t": np.array(ot, np.int64),
        "o": np.array(oo),
        "h": np.array(hh),
        "l": np.array(ll),
        "c": np.array(cc),
        "v": np.array(vv),
    }


def hourly_closes_asof(d: dict, i: int) -> np.ndarray:
    """1h 收盤：每根 UTC 小時取截至 bar i 的最後一筆 15m 收盤，不含之後的 15m。"""
    if i < 0:
        return np.array([], dtype=float)
    t = d["t"][: i + 1]
    c = d["c"][: i + 1]
    if len(t) == 0:
        return np.array([], dtype=float)
    hour = t // HOUR_MS
    last = np.concatenate((hour[1:] != hour[:-1], np.array([True])))
    return np.asarray(c[last], dtype=float)


def hourly_series_for_chart(d: dict, i_sig: int) -> tuple[dict, int]:
    """1h 圖：訊號當根用 as-of，後面小時用完整 K 方便對比走勢。"""
    full = hourly_ohlcv(d, None)
    asof = hourly_ohlcv(d, i_sig)
    if len(asof["t"]) == 0:
        return full, 0
    sig_t = int(asof["t"][-1])
    if len(full["t"]) == 0:
        return asof, len(asof["t"]) - 1
    idx = int(np.searchsorted(full["t"], sig_t))
    if idx >= len(full["t"]) or int(full["t"][idx]) != sig_t:
        return asof, len(asof["t"]) - 1
    out = {k: np.copy(v) for k, v in full.items()}
    for k in ("o", "h", "l", "c", "v"):
        out[k][idx] = asof[k][-1]
    return out, idx


def hourly_mas_asof(d: dict, i: int) -> tuple[float, float, float] | None:
    hc = hourly_closes_asof(d, i)
    if len(hc) < max(HOUR_MA_PERIODS):
        return None
    m7, m14, m25 = (float(np.mean(hc[-n:])) for n in HOUR_MA_PERIODS)
    return m7, m14, m25


def hourly_bearish(d: dict, i: int) -> bool:
    mas = hourly_mas_asof(d, i)
    if mas is None:
        return False
    m7, m14, m25 = mas
    return m7 < m14 < m25


def is_signal(d: dict, i: int) -> bool:
    if i < 210 or not is_fresh_break(d, i):
        return False
    if d["c"][i] >= d["o"][i]:
        return False
    if not prev_near_ribbon(d, i):
        return False
    if bar_body_pct(d, i) < MIN_BODY_PCT:
        return False
    if bar_range_pct(d, i) < MIN_RANGE_PCT:
        return False
    if drop_from_swing_pct(d, i) < MIN_DROP_PCT:
        return False
    if vol_ratio(d, i) < MIN_VOL_RATIO:
        return False
    return hourly_bearish(d, i)


def bar_close_et(open_ms: int) -> datetime:
    return datetime.fromtimestamp((open_ms + INTERVAL_MS) / 1000, ET)


def bar_in_session(open_ms: int) -> bool:
    close = bar_close_et(open_ms)
    if close.weekday() >= 5:
        return False
    t = close.hour * 60 + close.minute
    start = SESSION_START[0] * 60 + SESSION_START[1]
    end = SESSION_END[0] * 60 + SESSION_END[1]
    return start <= t <= end


def now_should_scan(now: datetime | None = None) -> bool:
    local = (now or datetime.now(timezone.utc)).astimezone(ET)
    if local.weekday() >= 5:
        return False
    start = local.replace(hour=SESSION_START[0], minute=SESSION_START[1], second=0, microsecond=0)
    end = local.replace(hour=SESSION_END[0], minute=SESSION_END[1], second=30, microsecond=0)
    return start <= local <= end


def next_window_start(now: datetime | None = None) -> datetime:
    local = (now or datetime.now(timezone.utc)).astimezone(ET)
    start = local.replace(hour=SESSION_START[0], minute=SESSION_START[1], second=0, microsecond=0)
    if local < start and local.weekday() < 5:
        return start
    day = local + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.replace(hour=SESSION_START[0], minute=SESSION_START[1], second=0, microsecond=0)


def close_hm_et(open_ms: int, interval_ms: int = INTERVAL_MS) -> str:
    return datetime.fromtimestamp((open_ms + interval_ms) / 1000, ET).strftime("%m-%d %H:%M")


def hm_et(open_ms: int) -> str:
    return close_hm_et(open_ms, INTERVAL_MS)


def hm8(open_ms: int) -> str:
    return datetime.fromtimestamp((open_ms + INTERVAL_MS) / 1000, TZ8).strftime("%m-%d %H:%M")


def scan_symbol(sym: str, *, backfill: bool) -> list[dict]:
    raw = fetch_klines(sym)
    if raw is None:
        return []
    d = indicators(raw)
    n = len(d["c"])
    if backfill:
        idxs = range(210, n)
    else:
        idxs = [i for i in (n - 1, n - 2) if i >= 210]
    events = []
    for i in idxs:
        if not is_signal(d, i):
            continue
        if not bar_in_session(int(d["t"][i])):
            continue
        events.append({"symbol": sym, "i": i, "d": d})
    return events


def key_of(ev: dict) -> str:
    return f"{ev['symbol']}:{int(ev['d']['t'][ev['i']])}"


def short_fwd_pct(entry: float, later: float) -> float:
    if entry == 0:
        return 0.0
    return (entry - later) / entry * 100.0


def trade_from_bar(sym: str, d: dict, i: int) -> dict:
    entry = float(d["c"][i])
    n = len(d["c"])
    fwd: dict[str, float | None] = {}
    for bars, name in HORIZONS:
        j = i + bars
        fwd[name] = None if j >= n else short_fwd_pct(entry, float(d["c"][j]))
    close_et = bar_close_et(int(d["t"][i]))
    eod = None
    for j in range(i + 1, n):
        cj = bar_close_et(int(d["t"][j]))
        if cj.date() != close_et.date():
            break
        eod = short_fwd_pct(entry, float(d["c"][j]))
        if (cj.hour, cj.minute) == CASH_CLOSE:
            break
    j1 = min(n, i + 9)
    mae = mfe = None
    if j1 > i + 1:
        hi = float(np.max(d["h"][i + 1 : j1]))
        lo = float(np.min(d["l"][i + 1 : j1]))
        mae = (hi - entry) / entry * 100.0
        mfe = (entry - lo) / entry * 100.0
    hmas = hourly_mas_asof(d, i)
    return {
        "symbol": sym,
        "t": int(d["t"][i]),
        "et": hm_et(int(d["t"][i])),
        "tw": hm8(int(d["t"][i])),
        "entry": entry,
        "depth": break_depth_pct(d, i),
        "body": bar_body_pct(d, i),
        "rng": bar_range_pct(d, i),
        "vr": vol_ratio(d, i),
        "drop": drop_from_swing_pct(d, i),
        "h1m7": None if hmas is None else hmas[0],
        "h1m14": None if hmas is None else hmas[1],
        "h1m25": None if hmas is None else hmas[2],
        "fwd": fwd,
        "eod": eod,
        "mae": mae,
        "mfe": mfe,
        "img": None,
    }


def backtest_symbol(sym: str, cutoff_ms: int) -> list[dict]:
    raw = fetch_klines(sym, limit=1000)
    if raw is None:
        return []
    d = indicators(raw)
    out = []
    for i in range(210, len(d["c"])):
        close_ms = int(d["t"][i]) + INTERVAL_MS
        if close_ms < cutoff_ms:
            continue
        if not is_signal(d, i) or not bar_in_session(int(d["t"][i])):
            continue
        tr = trade_from_bar(sym, d, i)
        fname = f"{sym}_{bar_close_et(int(d['t'][i])).strftime('%m%d_%H%M')}.png"
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        png = CHART_DIR / fname
        if draw_chart(sym, d, i, str(png)):
            tr["img"] = f"./img/ma-break/{fname}"
        out.append(tr)
    return out


def horizon_stats(trades: list[dict], name: str) -> dict | None:
    xs = [t["fwd"][name] for t in trades if t["fwd"].get(name) is not None]
    return _pnl_stats(xs)


def _pnl_stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    wins = sum(1 for x in xs if x > 0)
    return {
        "n": len(xs),
        "wr": wins / len(xs) * 100.0,
        "avg": mean(xs),
        "med": median(xs),
        "sum": sum(xs),
    }


def collect_backtest(days: int) -> tuple[list[str], list[dict], int]:
    symbols = universe()
    start = (datetime.now(ET) - timedelta(days=days)).replace(
        hour=SESSION_START[0], minute=SESSION_START[1], second=0, microsecond=0
    )
    cutoff_ms = int(start.timestamp() * 1000)
    trades: list[dict] = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(backtest_symbol, s, cutoff_ms): s for s in symbols}
        for fut in as_completed(futs):
            try:
                trades.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    trades.sort(key=lambda t: (t["t"], t["symbol"]))
    return symbols, trades, cutoff_ms


def _fmt_pnl(x: float | None) -> str:
    if x is None:
        return "—"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls}">{x:+.2f}%</span>'


def _fmt_plain(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2f}%"


def write_backtest_html(
    path: Path,
    *,
    days: int,
    symbols: list[str],
    trades: list[dict],
    cutoff_ms: int,
) -> Path:
    start = datetime.fromtimestamp(cutoff_ms / 1000, ET).strftime("%Y-%m-%d %H:%M")
    end = datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    cards = []
    for t in reversed(trades):
        name = t["symbol"].replace("USDT", "")
        h1 = t["fwd"].get("1h")
        img_html = ""
        if t.get("img"):
            png = (path.parent / t["img"].replace("./", "")).resolve()
            if png.exists():
                b64 = base64.b64encode(png.read_bytes()).decode("ascii")
                img_html = f'<img src="data:image/png;base64,{b64}" alt="{escape(name)}"/>'
        hmas = ""
        if t.get("h1m7") is not None:
            hmas = f"<br/>1h MA7 {t['h1m7']:g}　MA14 {t['h1m14']:g}　MA25 {t['h1m25']:g}"
        cards.append(
            "<div class='card'>"
            f"<h2>{escape(name)}　{escape(t['et'])}　空1h {_fmt_pnl(h1)}</h2>"
            f"{img_html}"
            f"<p class='note'>上 15m　下 1h　收 {t['entry']:g}　實體 {t['body']:.1f}%　振幅 {t['rng']:.1f}%　回撤 {t['drop']:.1f}%　量 {t['vr']:.1f}×"
            f"{hmas}"
            f"<br/>15m {_fmt_pnl(t['fwd']['15m'])}　1h {_fmt_pnl(t['fwd']['1h'])}　2h {_fmt_pnl(t['fwd']['2h'])}　當日 {_fmt_pnl(t['eod'])}</p>"
            "</div>"
        )
    kpi_bits = []
    for _, name in HORIZONS:
        st = horizon_stats(trades, name)
        if not st:
            continue
        kpi_bits.append(
            f'<div class="kpi"><div class="k">空 {name}</div>'
            f'<div class="v">{st["wr"]:.0f}% / {st["avg"]:+.2f}%</div></div>'
        )
    eod_st = _pnl_stats([t["eod"] for t in trades if t["eod"] is not None])
    if eod_st:
        kpi_bits.append(
            f'<div class="kpi"><div class="k">空到當日收</div>'
            f'<div class="v">{eod_st["wr"]:.0f}% / {eod_st["avg"]:+.2f}%</div></div>'
        )
    by_day: dict[str, int] = {}
    for t in trades:
        by_day[t["et"][:5]] = by_day.get(t["et"][:5], 0) + 1
    day_line = " · ".join(f"{d} {n}筆" for d, n in sorted(by_day.items()))
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>開盤瀑布 · 近 {days} 日</title>
<style>
:root{{--bg:#0c1210;--panel:#14201b;--ink:#e8f0ea;--muted:#8aa193;--line:rgba(232,240,234,.12);--long:#3dba7a;--short:#e35d5d}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0c1210;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:760px;margin:0 auto;padding:16px 12px 40px}}
h1{{font-size:20px;margin:0 0 6px}}
.sub{{color:var(--muted);font-size:13px;line-height:1.55;margin:0 0 14px}}
.kpis{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}}
.kpi{{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:10px 12px}}
.kpi .k{{color:var(--muted);font-size:11px}} .kpi .v{{font-size:16px;margin-top:4px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:14px;overflow-x:auto}}
.card h2{{font-size:15px;margin:0 0 8px}}
img{{width:100%;height:auto;display:block;border-radius:10px;background:#101814;margin-bottom:8px}}
.pos{{color:var(--long)}} .neg{{color:var(--short)}}
.note{{color:var(--muted);font-size:12px;line-height:1.5;margin:8px 0 0}}
</style>
</head>
<body>
<div class="wrap">
  <h1>開盤瀑布 · 近 {days} 日</h1>
  <p class="sub">幣安美股永續 {len(symbols)} 檔 · 美東 {start} → {end} · 只計開盤前後各半小時 · 進場 1h 空頭排列 · 訊號收盤做空</p>
  <div class="kpis">
    <div class="kpi"><div class="k">筆數 / 檔數</div><div class="v">{len(trades)} / {len({t["symbol"] for t in trades})}</div></div>
    {"".join(kpi_bits)}
  </div>
  <div class="card">
    <p class="note">每日：{escape(day_line) if day_line else "無"}</p>
    <p class="note">開盤窗 09:00–10:00。長陰實體≥1.2%、振幅≥2.5%、近 4h 高回撤≥2.5%、量≥1.8×、收盤低於 15m MA7/14/25/99/120/200。進場當下 1h MA7&lt;MA14&lt;MA25 空頭排列。圖上 15m、下 1h，黃虛線＝進場。綠＝空單賺。</p>
  </div>
  {"".join(cards) if cards else "<div class='card'>無訊號</div>"}
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def print_backtest(trades: list[dict]) -> None:
    print(f"訊號 {len(trades)} 筆、{len({t['symbol'] for t in trades})} 檔", flush=True)
    for _, name in HORIZONS:
        st = horizon_stats(trades, name)
        if not st:
            print(f"  {name}: 尚未走完")
            continue
        print(
            f"  空 {name}: n={st['n']} 勝率 {st['wr']:.1f}% 均 {st['avg']:+.2f}% 中位 {st['med']:+.2f}% 合計 {st['sum']:+.2f}%",
            flush=True,
        )
    eod_st = _pnl_stats([t["eod"] for t in trades if t["eod"] is not None])
    if eod_st:
        print(
            f"  空到當日收: n={eod_st['n']} 勝率 {eod_st['wr']:.1f}% 均 {eod_st['avg']:+.2f}% 中位 {eod_st['med']:+.2f}%",
            flush=True,
        )
    for t in trades:
        if t["symbol"] != "VRTUSDT":
            continue
        h1s = ""
        if t.get("h1m7") is not None:
            h1s = f"  1h均 {t['h1m7']:.2f}<{t['h1m14']:.2f}<{t['h1m25']:.2f}"
        print(
            f"  VRT {t['et']} 實體 {t['body']:.2f}% 振幅 {t['rng']:.2f}% 回撤 {t['drop']:.2f}%{h1s}  15m {_fmt_plain(t['fwd']['15m'])}  1h {_fmt_plain(t['fwd']['1h'])}  2h {_fmt_plain(t['fwd']['2h'])}",
            flush=True,
        )


def run_backtest(days: int, html_path: Path) -> int:
    print(f"回測近 {days} 日美股開盤瀑布…", flush=True)
    if CHART_DIR.exists():
        shutil.rmtree(CHART_DIR)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    symbols, trades, cutoff_ms = collect_backtest(days)
    print(f"掃完 {len(symbols)} 檔 {time.time()-t0:.1f}s", flush=True)
    print_backtest(trades)
    out = write_backtest_html(html_path, days=days, symbols=symbols, trades=trades, cutoff_ms=cutoff_ms)
    print("html", out)
    return 0


def format_event(ev: dict) -> str:
    d, i, sym = ev["d"], ev["i"], ev["symbol"]
    px = float(d["c"][i])
    mas = "　".join(f"MA{n} {float(d[f'm{n}'][i]):g}" for n in MA_PERIODS)
    hmas = hourly_mas_asof(d, i)
    hline = ""
    if hmas is not None:
        hline = f"1h MA7 {hmas[0]:g}　MA14 {hmas[1]:g}　MA25 {hmas[2]:g}　空頭排列\n"
    return (
        f"<b>開盤瀑布</b>  {sym}  15m\n"
        f"美東 {hm_et(int(d['t'][i]))}　台北 {hm8(int(d['t'][i]))}\n"
        f"收 {px:g}　實體 {bar_body_pct(d, i):.2f}%　振幅 {bar_range_pct(d, i):.2f}%\n"
        f"量 {vol_ratio(d, i):.1f}×　近4h高回撤 {drop_from_swing_pct(d, i):.2f}%\n"
        f"{mas}\n"
        f"{hline}"
        f"盤整後長陰放量，收盤低於 7/14/25/99/120/200"
    )


def draw_chart(sym: str, d: dict, i: int, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    pal = {7: "#f0c14a", 14: "#26c6da", 25: "#d28cff", 99: "#42a5f5", 120: "#5fd2c2", 200: "#e8f0ea"}

    def panel(ax, axv, src: dict, mark: int, *, lookback: int, lookfwd: int, interval_ms: int, title: str) -> None:
        a0 = max(0, mark - lookback)
        a1 = min(len(src["c"]), mark + lookfwd)
        sl = slice(a0, a1)
        xs = np.arange(a1 - a0)
        o, h, l, c, v = src["o"][sl], src["h"][sl], src["l"][sl], src["c"][sl], src["v"][sl]
        for a in (ax, axv):
            a.set_facecolor("#101814")
            a.tick_params(colors="#8aa193", labelsize=8)
            for sp in a.spines.values():
                sp.set_color("#2a3a33")
        for k in range(len(c)):
            col = "#3dba7a" if c[k] >= o[k] else "#e35d5d"
            ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
            y0, y1 = min(o[k], c[k]), max(o[k], c[k])
            if y1 == y0:
                y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
            ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
            axv.bar(xs[k], v[k], width=0.8, color=col + "99", linewidth=0)
        for n, col in pal.items():
            m = sma(src["c"], n)[sl]
            if len(m) == 0 or np.all(np.isnan(m)):
                continue
            ax.plot(xs, m, color=col, lw=1.08, label=f"MA{n}")
        x = mark - a0
        if 0 <= x < len(c):
            ax.axvline(x, color="#c9a227", ls="--", lw=0.9)
            ax.scatter([x], [c[x]], s=36, color="#e35d5d", zorder=5)
        ax.set_title(title, color="#e8f0ea", fontsize=12)
        ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
        if len(xs):
            ticks = list(range(0, len(xs), max(1, len(xs) // 6)))
            if ticks[-1] != len(xs) - 1:
                ticks.append(len(xs) - 1)
            axv.set_xticks(ticks)
            fmt = "%m-%d %H" if interval_ms >= HOUR_MS else "%H:%M"
            axv.set_xticklabels(
                [
                    datetime.fromtimestamp((int(src["t"][a0 + t]) + interval_ms) / 1000, ET).strftime(fmt)
                    for t in ticks
                ],
                color="#8aa193",
                fontsize=7,
            )

    with _MPL_LOCK:
        h1, hi = hourly_series_for_chart(d, i)
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(9.2, 8.8),
            facecolor="#0c1210",
            gridspec_kw={"height_ratios": [3.2, 0.95, 3.2, 0.95]},
        )
        ax15, ax15v, ax1h, ax1hv = axes
        panel(ax15, ax15v, d, i, lookback=36, lookfwd=18, interval_ms=INTERVAL_MS, title=f"{sym}  15m")
        if len(h1["c"]) >= 8:
            panel(ax1h, ax1hv, h1, hi, lookback=40, lookfwd=12, interval_ms=HOUR_MS, title=f"{sym}  1h")
        else:
            for a in (ax1h, ax1hv):
                a.set_facecolor("#101814")
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color("#2a3a33")
            ax1h.set_title(f"{sym}  1h（K線不足）", color="#8aa193", fontsize=12)
        fig.tight_layout(pad=0.45)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
    return path


def telegram_send(text: str, photo: str | None = None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = SESSION.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                    files={"photo": f},
                    timeout=25,
                )
            if r.ok:
                return True
        r = SESSION.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        return bool(r.ok)
    except requests.RequestException:
        return False


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)))


def scan_all(symbols: list[str], *, backfill: bool) -> list[dict]:
    events = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(scan_symbol, s, backfill=backfill): s for s in symbols}
        for fut in as_completed(futs):
            try:
                events.extend(fut.result())
            except Exception as e:
                print("err", futs[fut], e, flush=True)
    events.sort(key=lambda e: (int(e["d"]["t"][e["i"]]), e["symbol"]))
    return events


def notify(ev: dict, *, dry_run: bool) -> None:
    text = format_event(ev)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("&gt;", ">")
    )
    print("\n" + plain, flush=True)
    if dry_run:
        print("  → dry-run，不送 Telegram", flush=True)
        return
    tmp = Path("/tmp") / f"ma_break_{ev['symbol']}_{ev['i']}.png"
    photo = draw_chart(ev["symbol"], ev["d"], ev["i"], str(tmp))
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送", flush=True)
        return
    if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗", flush=True)


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 900 + 1) * 900 + 3
    time.sleep(max(1, nxt - now))


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("美股 15m 跌破均線監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="幣安美股永續 15m 開盤瀑布（開盤前後各半小時）")
    p.add_argument("--once", action="store_true", help="掃一次就結束")
    p.add_argument("--backfill", action="store_true", help="掃今日美東時段已收盤的 15m，不是只看剛收的兩根")
    p.add_argument("--force", action="store_true", help="不管美東時段，立刻掃")
    p.add_argument("--dry-run", action="store_true", help="只印、不送 Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram")
    p.add_argument("--backtest", action="store_true", help="回測近 N 日（不做 Telegram）")
    p.add_argument("--days", type=int, default=7, help="回測天數，預設 7")
    p.add_argument("--html", default="", help="回測 HTML 路徑")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()
    if args.backtest:
        html_path = Path(args.html) if args.html else PAGES_HTML
        return run_backtest(max(1, args.days), html_path)

    seen = load_seen()
    print("載入美股永續…", flush=True)
    symbols = universe()
    print(f"監看 {len(symbols)} 檔 EQUITY。開盤窗瀑布才推。", flush=True)
    uni_ts = time.time()

    def round_once(*, backfill: bool) -> None:
        nonlocal symbols, uni_ts
        if time.time() - uni_ts > 1800:
            symbols = universe()
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        events = scan_all(symbols, backfill=backfill)
        new = [e for e in events if key_of(e) not in seen]
        print(
            f"[{datetime.now(TZ8).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev, dry_run=args.dry_run)
        if new and not args.dry_run:
            save_seen(seen)

    if not args.force and not now_should_scan():
        nxt = next_window_start()
        print(
            f"現在不是偵測時段（美東 09:00–10:00，開盤前後各半小時）。下次開始 {nxt.strftime('%Y-%m-%d %H:%M %Z')}",
            flush=True,
        )
        if args.once:
            return 0

    if args.once:
        if args.force or now_should_scan():
            round_once(backfill=args.backfill)
        return 0

    print("watch 中，每根 15m 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            if not args.force and not now_should_scan():
                nxt = next_window_start()
                sec = max(5.0, (nxt - datetime.now(ET)).total_seconds())
                print(f"休眠到 {nxt.strftime('%m-%d %H:%M %Z')}（{sec/60:.0f} 分）", flush=True)
                time.sleep(min(sec, 3600))
                continue
            wait_next_close()
            if args.force or now_should_scan():
                round_once(backfill=False)
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
