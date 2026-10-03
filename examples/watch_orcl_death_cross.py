#!/usr/bin/env python3
"""1m 空：4 小時新高後，同一根死亡交叉且破 MA25，形狀對齊 ORCL 22:49。

對 2026-10-02 ORCL：22:44 創四小時高 144.95；22:49（+5 分）同一根收 144.44。
當根還要：MA7 領先夠久、高點後至少幾分鐘、從高點連陰、收在 K 棒下緣、已經離高點一段。

用法：
  python3 examples/watch_orcl_death_cross.py --all --scan --date 2026-10-02 --pages
  python3 examples/watch_orcl_death_cross.py --scan
  python3 examples/watch_orcl_death_cross.py --all
  python3 examples/watch_orcl_death_cross.py --test
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

# —— 可選：直接填這裡 ——
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO_ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO_ROOT / "output" / "orcl_death_cross_seen.json"
PAGES_HTML = REPO_ROOT / "docs" / "binance" / "death-cross-1m" / "index.html"
DEFAULT_SYMBOLS = ("ORCLUSDT",)
KEEP = {"ORCLUSDT"}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"})


@dataclass(frozen=True)
class ShortHit:
    i: int
    close: float
    m7: float
    m14: float
    m25: float
    lead: int
    crossed_ma25: bool
    peak_i: int | None = None
    peak_high: float | None = None
    bars_after_high: int | None = None
    drop_from_high: float | None = None
    close_loc: float | None = None
    reds_from_high: int | None = None


@dataclass(frozen=True)
class ScanRow:
    symbol: str
    ts_ms: int
    hit: ShortHit
    fwd15: float | None
    fwd30: float | None
    low15: float | None
    low30: float | None
    quote_vol: float = 0.0


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


def lead_bars(m7: np.ndarray, m14: np.ndarray, i: int) -> int:
    """死亡交叉前，MA7 連續 ≥ MA14 的根數（不含當根）。"""
    n = 0
    j = i - 1
    while j >= 0 and not np.isnan(m7[j]) and not np.isnan(m14[j]) and m7[j] >= m14[j]:
        n += 1
        j -= 1
    return n


def new_high_mask(high: np.ndarray, lookback: int = 240) -> np.ndarray:
    """high[i] 是否創下過去 lookback 根（不含當根）的新高。4 小時 1m = 240。"""
    n = len(high)
    out = np.zeros(n, dtype=bool)
    if n <= lookback or lookback < 1:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(high, lookback)
    out[lookback:] = high[lookback:] > windows[: n - lookback].max(axis=1)
    return out


def latest_high_in_window(mask: np.ndarray, i: int, within: int) -> int | None:
    """i 往前 within 根（含當根）裡，最近一次新高的 index。"""
    if i < 0 or i >= len(mask) or within < 0:
        return None
    start = max(0, i - within)
    seg = np.flatnonzero(mask[start : i + 1])
    if len(seg) == 0:
        return None
    return int(start + seg[-1])


def detect_shorts(
    close: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    min_lead: int = 15,
    require_cross_ma25: bool = True,
    require_4h_high: bool = False,
    high_lookback: int = 240,
    within_bars: int = 30,
    orcl_shape: bool = True,
    min_after_high: int = 4,
    min_drop_from_high: float = 0.25,
    min_reds_from_high: int = 3,
    max_close_loc: float = 0.25,
) -> list[ShortHit]:
    """收盤根：同一根 K 上 MA7 下穿 MA14，且收盤由上跌破 MA25（對齊 ORCL 22:49）。

    require_4h_high：訊號前 within_bars 根內，必須創下 high_lookback 根新高。
    orcl_shape（預設開）：再壓成 ORCL 那種——領先夠久、高點後幾分鐘才破、
    從高點連陰、收在 K 棒下緣、收盤已離 4h 高至少 min_drop_from_high%。
    沒傳 open/low 時，跳過連陰與收盤位置。
    """
    hi_mask = new_high_mask(high, high_lookback) if (require_4h_high and high is not None) else None
    hits: list[ShortHit] = []
    for i in range(1, len(close)):
        vals = (close[i], close[i - 1], m7[i], m7[i - 1], m14[i], m14[i - 1], m25[i], m25[i - 1])
        if np.isnan(vals).any():
            continue
        death = m7[i - 1] >= m14[i - 1] and m7[i] < m14[i]
        if not death:
            continue
        below = close[i] < m25[i]
        if not below:
            continue
        crossed = close[i - 1] >= m25[i - 1] and close[i] < m25[i]
        if require_cross_ma25 and not crossed:
            continue
        lead = lead_bars(m7, m14, i)
        if lead < min_lead:
            continue
        peak_i = peak_high = bars_after = drop = loc = reds = None
        if require_4h_high:
            if hi_mask is None:
                continue
            peak_i = latest_high_in_window(hi_mask, i, within_bars)
            if peak_i is None:
                continue
            peak_high = float(high[peak_i])  # type: ignore[index]
            bars_after = i - peak_i
            drop = pct_move(peak_high, float(close[i]))
            if orcl_shape:
                if bars_after < min_after_high:
                    continue
                if drop > -min_drop_from_high:
                    continue
                if open_ is not None:
                    reds = int(np.sum(close[peak_i : i + 1] < open_[peak_i : i + 1]))
                    if reds < min_reds_from_high:
                        continue
                    if close[i] >= open_[i]:
                        continue
                if low is not None and high is not None:
                    rng = float(high[i] - low[i])
                    if rng <= 0:
                        continue
                    loc = float((close[i] - low[i]) / rng)
                    if loc > max_close_loc:
                        continue
        hits.append(
            ShortHit(
                i=i,
                close=float(close[i]),
                m7=float(m7[i]),
                m14=float(m14[i]),
                m25=float(m25[i]),
                lead=lead,
                crossed_ma25=bool(crossed),
                peak_i=peak_i,
                peak_high=peak_high,
                bars_after_high=bars_after,
                drop_from_high=drop,
                close_loc=loc,
                reds_from_high=reds,
            )
        )
    return hits


def pct_move(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def forward_moves(
    close: np.ndarray,
    low: np.ndarray,
    i: int,
    *,
    n15: int = 15,
    n30: int = 30,
) -> tuple[float | None, float | None, float | None, float | None]:
    """訊號後 15/30 根收盤漲跌、以及期間最低點相對進場收盤。"""
    n = len(close)
    if i < 0 or i >= n:
        return None, None, None, None
    c0 = float(close[i])

    def at(k: int) -> tuple[float | None, float | None]:
        j = i + k
        if j >= n:
            j = n - 1
        if j <= i:
            return None, None
        sl = low[i : j + 1]
        return pct_move(c0, float(close[j])), pct_move(c0, float(np.min(sl)))

    f15, l15 = at(n15)
    f30, l30 = at(n30)
    return f15, f30, l15, l30


def taipei_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m-%d")


def day_bounds_ms(day: str) -> tuple[int, int]:
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def cutoff_ms(*, day: str | None, hours: int, now_ms: int | None = None) -> tuple[int, int]:
    """回傳 [start, end) 毫秒。有 day 就用台北日；否則用近 hours。"""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if day:
        return day_bounds_ms(day)
    return now - hours * 3600 * 1000, now + 1


def in_window(ms: int, start_ms: int, end_ms: int) -> bool:
    return start_ms <= ms < end_ms


def filter_universe(
    info_symbols: list[dict],
    tickers: dict[str, dict],
    *,
    min_quote_vol: float = 0.0,
    keep: set[str] | None = None,
) -> list[str]:
    keep = keep or set()
    out: list[str] = []
    for s in info_symbols:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") not in ("PERPETUAL", "TRADIFI_PERPETUAL"):
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        if qv < min_quote_vol and sym not in keep:
            continue
        out.append(sym)
    return out


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                last = RuntimeError(f"HTTP 429 {path}")
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))
    raise last if last else RuntimeError(f"GET failed {path}")


def fetch_klines(sym: str, limit: int = 400, *, drop_forming: bool = True) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "1m", "limit": limit})
    if not raw or len(raw) < 30:
        return None
    now_ms = int(time.time() * 1000)
    if drop_forming and int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if len(raw) < 30:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def with_ma(d: dict) -> dict:
    c = d["c"]
    out = dict(d)
    out["m7"], out["m14"], out["m25"] = sma(c, 7), sma(c, 14), sma(c, 25)
    return out


def universe(*, min_quote_vol: float = 0.0) -> tuple[list[str], dict[str, float]]:
    info = get_json("/fapi/v1/exchangeInfo")
    raw_tickers = get_json("/fapi/v1/ticker/24hr")
    tickers = {t["symbol"]: t for t in raw_tickers}
    symbols = filter_universe(info["symbols"], tickers, min_quote_vol=min_quote_vol, keep=KEEP)
    vols = {s: float((tickers.get(s) or {}).get("quoteVolume") or 0) for s in symbols}
    return symbols, vols


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


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


def draw_chart(sym: str, d: dict, hit: ShortHit, path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    a0 = max(0, hit.i - 80)
    a1 = min(len(d["c"]), hit.i + 8)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl]
    fig, ax = plt.subplots(figsize=(10.4, 5.2), facecolor="#0c1210")
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
    pal = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff"}
    for n, col in pal.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.15, label=f"MA{n}")
    x = hit.i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#e35d5d", ls="--", lw=0.95)
        ax.scatter([x], [c[x]], s=38, color="#e35d5d", zorder=5)
    ax.set_title(f"{sym}  1m  空  4h新高後死亡交叉且破 MA25", color="#e8f0ea", fontsize=12)
    if hit.peak_i is not None:
        px = hit.peak_i - a0
        if 0 <= px < len(c):
            ax.axvline(px, color="#c9a227", ls=":", lw=0.9)
            ax.scatter([px], [d["h"][hit.peak_i]], s=28, color="#c9a227", zorder=5, marker="^")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc", ncol=3)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_hit(sym: str, d: dict, hit: ShortHit) -> str:
    ts = hm(int(d["t"][hit.i]))
    x25 = "這根同時跌破 MA25" if hit.crossed_ma25 else "收盤已在 MA25 下方"
    link = binance_href(sym)
    high_line = ""
    if hit.bars_after_high is not None and hit.peak_i is not None:
        dump = ""
        if hit.drop_from_high is not None:
            dump = f"，離高 {hit.drop_from_high:+.2f}%"
        high_line = f"四小時新高 {hm(int(d['t'][hit.peak_i]))} 後 {hit.bars_after_high} 分鐘{dump}\n"
    extra = ""
    if hit.reds_from_high is not None:
        extra += f"從高點連陰 {hit.reds_from_high} 根\n"
    if hit.close_loc is not None:
        extra += f"收在 K 棒下方 {hit.close_loc:.0%}\n"
    return (
        f"🔻 <b>{sym} 空</b>  1m\n"
        f"{high_line}"
        f"MA7 / MA14 <b>死亡交叉</b>，{x25}\n"
        f"{extra}"
        f"時間 {ts}（台北）\n"
        f"收 {hit.close:g}\n"
        f"MA7 {hit.m7:.4f}　MA14 {hit.m14:.4f}　MA25 {hit.m25:.4f}\n"
        f"MA7 領先 {hit.lead} 根後下穿\n"
        f'<a href="{link}">{sym}</a>'
    )


def key_of(sym: str, d: dict, hit: ShortHit) -> str:
    return f"{sym}:{int(d['t'][hit.i])}"


def detect_kw_from_args(args) -> dict:
    shape = not args.loose
    return {
        "min_lead": args.min_lead,
        "require_cross_ma25": args.require_cross_ma25,
        "require_4h_high": not args.no_4h_high,
        "high_lookback": max(1, int(round(args.high_hours * 60))),
        "within_bars": max(0, args.within_minutes),
        "orcl_shape": shape,
        "min_after_high": 0 if not shape else args.min_after,
        "min_drop_from_high": 0.0 if not shape else args.min_drop_high,
        "min_reds_from_high": 0 if not shape else args.min_reds,
        "max_close_loc": 1.0 if not shape else args.max_close_loc,
    }


def scan_symbol(sym: str, *, limit: int, **detect_kw) -> tuple[dict, list[ShortHit]]:
    raw = fetch_klines(sym, limit=limit)
    if raw is None:
        return {}, []
    d = with_ma(raw)
    return d, detect_shorts(
        d["c"],
        d["m7"],
        d["m14"],
        d["m25"],
        high=d["h"],
        low=d["l"],
        open_=d["o"],
        **detect_kw,
    )


def rows_from_hits(sym: str, d: dict, hits: list[ShortHit], *, quote_vol: float = 0.0) -> list[ScanRow]:
    rows: list[ScanRow] = []
    for hit in hits:
        f15, f30, l15, l30 = forward_moves(d["c"], d["l"], hit.i)
        rows.append(
            ScanRow(
                symbol=sym,
                ts_ms=int(d["t"][hit.i]),
                hit=hit,
                fwd15=f15,
                fwd30=f30,
                low15=l15,
                low30=l30,
                quote_vol=quote_vol,
            )
        )
    return rows


def scan_symbol_rows(
    sym: str,
    *,
    limit: int,
    quote_vol: float = 0.0,
    **detect_kw,
) -> tuple[dict, list[ScanRow]]:
    d, hits = scan_symbol(sym, limit=limit, **detect_kw)
    if not d:
        return {}, []
    return d, rows_from_hits(sym, d, hits, quote_vol=quote_vol)


def scan_universe(
    symbols: list[str],
    *,
    limit: int,
    workers: int,
    vols: dict[str, float] | None = None,
    keep_bars: bool = False,
    **detect_kw,
) -> tuple[list[ScanRow], dict[str, dict]]:
    vols = vols or {}
    rows: list[ScanRow] = []
    bars: dict[str, dict] = {}
    err = 0

    def one(sym: str) -> tuple[str, dict, list[ScanRow]]:
        d, rs = scan_symbol_rows(
            sym,
            limit=limit,
            quote_vol=float(vols.get(sym) or 0),
            **detect_kw,
        )
        return sym, d, rs

    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(one, s): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                s, d, rs = fut.result()
            except Exception as e:
                err += 1
                print("err", sym, e, flush=True)
                continue
            rows.extend(rs)
            if keep_bars and d:
                bars[s] = d
            if n % 80 == 0 or n == len(symbols):
                print(f"  … {n}/{len(symbols)}  訊號 {len(rows)}  失敗 {err}", flush=True)
    rows.sort(key=lambda r: (r.ts_ms, r.symbol))
    return rows, bars


def notify(sym: str, d: dict, hit: ShortHit, *, dry_run: bool) -> None:
    text = format_hit(sym, d, hit)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("&gt;", ">")
        .split('<a href=')[0]
        .strip()
    )
    print("\n" + plain)
    if dry_run:
        print("  → dry-run，不送 Telegram")
        return
    tmp = Path("/tmp") / f"orcl_dx_{sym}_{hit.i}.png"
    photo = draw_chart(sym, d, hit, str(tmp))
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送")
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
    else:
        print("  → Telegram 送出失敗，檢查 token 與 chat id")


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1, nxt - now))


def print_scan(sym: str, d: dict, hits: list[ShortHit], *, hours: int, day: str | None = None) -> None:
    if not d:
        print(f"{sym} 沒資料")
        return
    start, end = cutoff_ms(day=day, hours=hours, now_ms=int(d["t"][-1]) + 1)
    recent = [h for h in hits if in_window(int(d["t"][h.i]), start, end)]
    label = day if day else f"近 {hours}h"
    print(f"\n{sym} {label}  4h新高後、同一根死亡交叉且破 MA25（ORCL 形）：{len(recent)} 筆")
    for h in recent:
        x = "同根破25" if h.crossed_ma25 else "已在25下"
        after = f"  高點後{h.bars_after_high}m" if h.bars_after_high is not None else ""
        dump = f"  離高{h.drop_from_high:+.2f}%" if h.drop_from_high is not None else ""
        print(
            f"  {hm(int(d['t'][h.i]))}  收 {h.close:g}  "
            f"MA7 {h.m7:.4f}  MA14 {h.m14:.4f}  MA25 {h.m25:.4f}  "
            f"lead {h.lead}  {x}{after}{dump}"
        )


def orcl_like(rows: list[ScanRow], *, min_dump: float = -1.0) -> list[ScanRow]:
    """主訊號已是 ORCL 形；這裡再留之後 30 根有砸過 min_dump% 的。"""
    out = []
    for r in rows:
        if not r.hit.crossed_ma25 or r.hit.lead < 10:
            continue
        if r.low30 is not None and r.low30 > min_dump:
            continue
        out.append(r)
    out.sort(
        key=lambda r: (
            r.low30 if r.low30 is not None else r.low15 if r.low15 is not None else 0.0,
            -r.hit.lead,
        )
    )
    return out


def binance_href(sym: str) -> str:
    return "https://www.binance.com/zh-TW/futures/" + quote(sym, safe="")


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def print_market_scan(rows: list[ScanRow], *, start_ms: int, end_ms: int, n_symbols: int, top: int = 40) -> None:
    recent = [r for r in rows if in_window(r.ts_ms, start_ms, end_ms)]
    same = sum(1 for r in recent if r.hit.crossed_ma25)
    names = sorted({r.symbol for r in recent})
    print(
        f"\n幣安 USDT 永續 {n_symbols} 檔　4h新高後、同一根死亡交叉且破MA25、ORCL形："
        f"{len(recent)} 筆 / {len(names)} 檔"
    )
    ranked = sorted(
        recent,
        key=lambda r: (
            r.low30 if r.low30 is not None else r.low15 if r.low15 is not None else 0.0,
            -r.hit.lead,
        ),
    )
    print(f"\n急殺最深（訊號後 30 根最低，前 {top}）：")
    for r in ranked[:top]:
        x = "同根破25" if r.hit.crossed_ma25 else "已在25下"
        print(
            f"  {r.symbol:<14} {hm(r.ts_ms)}  收 {r.hit.close:g}  "
            f"30m低 {fmt_pct(r.low30)}  15m {fmt_pct(r.fwd15)}  "
            f"lead {r.hit.lead}  高點後{r.hit.bars_after_high}m  離高{fmt_pct(r.hit.drop_from_high)}  {x}"
        )
    like = orcl_like(recent)
    print(f"\n之後砸得比較深（30m 至少 −1%，前 {top}）：{len(like)} 筆 / {len({r.symbol for r in like})} 檔")
    for r in like[:top]:
        print(
            f"  {r.symbol:<14} {hm(r.ts_ms)}  收 {r.hit.close:g}  "
            f"30m低 {fmt_pct(r.low30)}  15m {fmt_pct(r.fwd15)}  "
            f"lead {r.hit.lead}  高點後{r.hit.bars_after_high}m  離高{fmt_pct(r.hit.drop_from_high)}"
        )


def row_to_json(r: ScanRow) -> dict:
    return {
        "symbol": r.symbol,
        "ts": r.ts_ms,
        "time": hm(r.ts_ms),
        "close": r.hit.close,
        "ma7": r.hit.m7,
        "ma14": r.hit.m14,
        "ma25": r.hit.m25,
        "lead": r.hit.lead,
        "crossed_ma25": r.hit.crossed_ma25,
        "peak_high": r.hit.peak_high,
        "bars_after_high": r.hit.bars_after_high,
        "drop_from_high": r.hit.drop_from_high,
        "close_loc": r.hit.close_loc,
        "reds_from_high": r.hit.reds_from_high,
        "fwd15": r.fwd15,
        "fwd30": r.fwd30,
        "low15": r.low15,
        "low30": r.low30,
        "quote_vol": r.quote_vol,
    }


def write_html_report(
    path: str | Path,
    rows: list[ScanRow],
    *,
    start_ms: int,
    end_ms: int,
    n_symbols: int,
    title: str,
    top: int = 50,
) -> Path:
    recent = [r for r in rows if in_window(r.ts_ms, start_ms, end_ms)]
    ranked = sorted(
        recent,
        key=lambda r: (
            r.low30 if r.low30 is not None else r.low15 if r.low15 is not None else 0.0,
            -r.hit.lead,
        ),
    )
    same = sum(1 for r in recent if r.hit.crossed_ma25)
    names = sorted({r.symbol for r in recent})
    like = orcl_like(recent)
    by_sym: dict[str, list[ScanRow]] = {}
    for r in recent:
        by_sym.setdefault(r.symbol, []).append(r)
    sym_rows = []
    for sym, rs in by_sym.items():
        lows = [x.low30 for x in rs if x.low30 is not None]
        last = rs[-1]
        worst = min(lows) if lows else None
        with_low = [x for x in rs if x.low30 is not None]
        dump = min(with_low, key=lambda z: z.low30) if with_low else last
        sym_rows.append((worst if worst is not None else 0.0, sym, len(rs), dump, last))
    sym_rows.sort()

    def pct_cell(v: float | None) -> str:
        if v is None:
            return '<td class="muted">—</td>'
        cls = "dn" if v < 0 else "up"
        return f'<td class="{cls}">{v:+.2f}%</td>'

    def row_tr(r: ScanRow, extra: str) -> str:
        return (
            "<tr>"
            f"<td><a href='{escape(binance_href(r.symbol))}'>{escape(r.symbol)}</a></td>"
            f"<td>{escape(hm(r.ts_ms))}</td>"
            f"<td>{r.hit.close:g}</td>"
            f"{pct_cell(r.low30)}{pct_cell(r.fwd15)}{pct_cell(r.fwd30)}"
            f"<td>{r.hit.lead}</td>"
            f"<td>{r.hit.bars_after_high if r.hit.bars_after_high is not None else '—'}</td>"
            f"{pct_cell(r.hit.drop_from_high)}"
            f"<td>{extra}</td>"
            "</tr>"
        )

    top_html = [
        row_tr(
            r,
            f"連陰{r.hit.reds_from_high}" if r.hit.reds_from_high is not None else "ORCL形",
        )
        for r in ranked[:top]
    ]
    like_html = [row_tr(r, f"lead {r.hit.lead}") for r in like[:top]]
    sym_html = []
    for _w, sym, n, dump, last in sym_rows[:200]:
        x = "同根破25" if dump.hit.crossed_ma25 else "已在25下"
        sym_html.append(
            "<tr>"
            f"<td><a href='{escape(binance_href(sym))}'>{escape(sym)}</a></td>"
            f"<td>{n}</td>"
            f"<td>{escape(hm(dump.ts_ms))}</td>"
            f"<td>{dump.hit.close:g}</td>"
            f"{pct_cell(dump.low30)}"
            f"<td>{x}</td>"
            f"<td>{escape(hm(last.ts_ms))}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
<style>
body{{margin:0;background:#0c1210;color:#e8f0ea;font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:18px 14px 48px}}
h1{{font-size:1.35rem;margin:0 0 8px}}
h2{{font-size:1.02rem;margin:22px 0 8px}}
.sub{{color:#8aa193;line-height:1.55;margin:0 0 14px;font-size:.92rem}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}}
.chip{{border:1px solid rgba(232,240,234,.12);background:#14201b;border-radius:999px;padding:7px 12px;font-size:.82rem}}
.chip b{{color:#c9a227}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th,td{{padding:7px 6px;border-bottom:1px solid rgba(232,240,234,.1);text-align:left}}
th{{color:#8aa193;font-weight:500}}
a{{color:#c9a227;text-decoration:none}}
.dn{{color:#e35d5d}} .up{{color:#3dba7a}} .muted{{color:#8aa193}}
.note{{margin-top:16px;color:#8aa193;font-size:.8rem;line-height:1.5}}
</style>
</head>
<body>
<div class="wrap">
<h1>{escape(title)}</h1>
<p class="sub">創下過去 4 小時新高後 30 分鐘內，<b>同一根</b> 1 分鐘 K：MA7 下穿 MA14 且收盤跌破 MA25。再壓成 ORCL 22:49 那種：領先 ≥15 根、高點後 ≥4 分、從高點至少 3 根陰、收在 K 棒下緣、離高點 ≥0.25%。急殺深度是訊號後 30 根最低點。不是進出場建議。</p>
<div class="chips">
  <div class="chip">掃 <b>{n_symbols}</b> 檔</div>
  <div class="chip">訊號 <b>{len(recent)}</b> 筆</div>
  <div class="chip">有訊號 <b>{len(names)}</b> 檔</div>
  <div class="chip">之後砸 ≥1% <b>{len(like)}</b></div>
</div>
<h2>之後砸得比較深（30m ≥1%，前 {min(top, len(like))}）</h2>
<table>
<thead><tr><th>標的</th><th>時間</th><th>收</th><th>30m低</th><th>15m</th><th>30m收</th><th>lead</th><th>高點後</th><th>離高</th><th></th></tr></thead>
<tbody>
{"".join(like_html) or "<tr><td colspan='10' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
<h2>全部 ORCL 形（前 {min(top, len(ranked))}）</h2>
<table>
<thead><tr><th>標的</th><th>時間</th><th>收</th><th>30m低</th><th>15m</th><th>30m收</th><th>lead</th><th>高點後</th><th>離高</th><th></th></tr></thead>
<tbody>
{"".join(top_html) or "<tr><td colspan='10' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
<h2>有訊號的標的</h2>
<table>
<thead><tr><th>標的</th><th>筆數</th><th>最深那筆</th><th>收</th><th>30m低</th><th>破25</th><th>最後一筆</th></tr></thead>
<tbody>
{"".join(sym_html) or "<tr><td colspan='7' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
<p class="note">ORCL 10-02：22:44 創四小時高 144.95；22:49（+5 分）同一根死亡交叉且跌破 MA25，lead 22、連陰 4、收在下緣、離高 −0.35%，之後砸到 141。</p>
</div>
</body>
</html>
"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    hits_path = out.parent / "hits.json"
    payload = {
        "title": title,
        "n_symbols": n_symbols,
        "count": len(recent),
        "symbols_hit": len(names),
        "same_bar_ma25": same,
        "orcl_like": len(like),
        "top": [row_to_json(r) for r in ranked[:200]],
        "like": [row_to_json(r) for r in like[:200]],
        "by_symbol": [
            {
                "symbol": sym,
                "n": n,
                "worst_time": hm(dump.ts_ms),
                "worst_close": dump.hit.close,
                "worst_low30": dump.low30,
                "last_time": hm(last.ts_ms),
            }
            for _w, sym, n, dump, last in sym_rows
        ],
    }
    hits_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send("ORCL 1m 死亡交叉監看測試\n如果你看到這則，Telegram 已通。")
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYMBOLS)
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return out or list(DEFAULT_SYMBOLS)


def load_symbol_list(args) -> tuple[list[str], dict[str, float]]:
    if args.all:
        print("載入幣安 USDT 永續…", flush=True)
        symbols, vols = universe(min_quote_vol=args.min_quote_vol)
        print(f"共 {len(symbols)} 檔（min 成交額 {args.min_quote_vol:g}）", flush=True)
        return symbols, vols
    return parse_symbols(args.symbols), {}


def scan_window_label(args) -> tuple[int, int, str]:
    start, end = cutoff_ms(day=args.date, hours=args.hours)
    if args.date:
        return start, end, f"台北 {args.date}"
    return start, end, f"近 {args.hours}h"


def run_scan(args, symbols: list[str], vols: dict[str, float]) -> int:
    start, end, label = scan_window_label(args)
    detect_kw = detect_kw_from_args(args)
    t0 = time.time()
    extra = "4h新高+ORCL形" if detect_kw["require_4h_high"] and detect_kw["orcl_shape"] else (
        "4h新高+30m" if detect_kw["require_4h_high"] else "不過濾4h高"
    )
    extra += f"  min_lead={detect_kw['min_lead']}"
    if args.all or len(symbols) > 1:
        print(
            f"掃 {len(symbols)} 檔 1m　{label}　min_lead={args.min_lead}　{extra}",
            flush=True,
        )
        rows, _ = scan_universe(
            symbols,
            limit=args.limit,
            workers=args.workers,
            vols=vols,
            **detect_kw,
        )
        print(f"掃完 {time.time()-t0:.1f}s", flush=True)
        print_market_scan(rows, start_ms=start, end_ms=end, n_symbols=len(symbols), top=args.top)
        html_path = args.html
        if args.pages:
            html_path = html_path or str(PAGES_HTML)
        if html_path:
            out = write_html_report(
                html_path,
                rows,
                start_ms=start,
                end_ms=end,
                n_symbols=len(symbols),
                title=f"幣安 1m 空 · ORCL 形（同根死亡交叉且破 MA25）· {label}",
                top=args.top,
            )
            print(f"html={out}")
        return 0
    for sym in symbols:
        d, hits = scan_symbol(sym, limit=args.limit, **detect_kw)
        print_scan(sym, d, hits, hours=args.hours, day=args.date)
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="1m 4h新高後 MA7/MA14 死亡交叉且破 MA25 → 掃幣安 / Telegram")
    p.add_argument("--symbols", default="ORCLUSDT", help="逗號分隔，預設 ORCLUSDT")
    p.add_argument("--all", action="store_true", help="掃幣安所有 USDT 永續（含股票型如 ORCL）")
    p.add_argument("--min-quote-vol", type=float, default=0.0, help="24h 成交額下限，--all 時用")
    p.add_argument("--workers", type=int, default=12, help="並行下載 K 線")
    p.add_argument("--min-lead", type=int, default=15, help="死亡交叉前 MA7≥MA14 最少根數，ORCL 是 22")
    p.add_argument(
        "--no-same-bar-ma25",
        dest="require_cross_ma25",
        action="store_false",
        help="允許死亡交叉時價已在 MA25 下（不要求同一根跌破，不像 ORCL）",
    )
    p.add_argument("--high-hours", type=float, default=4.0, help="新高回看幾小時，預設 4")
    p.add_argument("--within-minutes", type=int, default=30, help="新高後幾分鐘內要出現死亡交叉，預設 30")
    p.add_argument("--min-after", type=int, default=4, help="高點後至少過幾分鐘才算，ORCL 是 +5")
    p.add_argument("--min-drop-high", type=float, default=0.25, help="收盤離 4h 高至少百分之幾，ORCL 約 0.35")
    p.add_argument("--min-reds", type=int, default=3, help="從高點到訊號至少幾根陰線，ORCL 是 4")
    p.add_argument("--max-close-loc", type=float, default=0.25, help="收盤在 K 棒位置，0=最低 1=最高，ORCL 收在最低")
    p.add_argument("--loose", action="store_true", help="關掉 ORCL 形，只留 4h 新高+同根死亡交叉")
    p.add_argument("--no-4h-high", action="store_true", help="關掉四小時新高條件")
    p.add_argument("--scan", action="store_true", help="印出歷史訊號後結束")
    p.add_argument("--hours", type=int, default=24, help="沒指定 --date 時，回看小時數")
    p.add_argument("--date", default=None, help="台北日 YYYY-MM-DD，只看這一天")
    p.add_argument("--top", type=int, default=40, help="市場掃描列出急殺最深幾筆")
    p.add_argument("--html", default=None, help="寫入 HTML 報告路徑")
    p.add_argument("--pages", action="store_true", help="寫到 docs/binance/death-cross-1m/")
    p.add_argument("--limit", type=int, default=1500, help="K 線根數")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--dry-run", action="store_true", help="只印不送 Telegram")
    p.set_defaults(require_cross_ma25=True)
    args = p.parse_args()
    if args.loose and "--min-lead" not in sys.argv:
        args.min_lead = 5
    apply_keys()
    if args.test:
        return test_telegram()

    symbols, vols = load_symbol_list(args)
    if args.scan:
        return run_scan(args, symbols, vols)

    seen = load_seen()
    detect_kw = detect_kw_from_args(args)
    look = detect_kw["high_lookback"] + detect_kw["within_bars"] + 25
    watch_limit = max(look + 10, 120 if not args.all else look + 10, args.limit if not args.all else look + 10)
    if args.all:
        watch_limit = max(look + 10, 320)
    print(
        f"監看 {len(symbols)} 檔  1m  空  "
        f"4h新高後{args.within_minutes}分內  同一根死亡交叉且破MA25  "
        f"ORCL形={'開' if detect_kw['orcl_shape'] else '關'}  "
        f"min_lead={args.min_lead}  min_after={detect_kw['min_after_high']}",
        flush=True,
    )
    if args.all:
        print("全市場每分鐘掃一次；報告請用 --scan --pages。", flush=True)
    uni_ts = time.time()

    def round_once() -> None:
        nonlocal symbols, vols, uni_ts
        if args.all and time.time() - uni_ts > 1800:
            symbols, vols = universe(min_quote_vol=args.min_quote_vol)
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        t0 = time.time()
        n_new = 0
        if args.all or len(symbols) > 4:
            with ThreadPoolExecutor(max(1, args.workers)) as ex:
                futs = {
                    ex.submit(
                        scan_symbol,
                        s,
                        limit=watch_limit,
                        **detect_kw,
                    ): s
                    for s in symbols
                }
                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        d, hits = fut.result()
                    except Exception as e:
                        print("err", sym, e, flush=True)
                        continue
                    if not d or not hits:
                        continue
                    last = len(d["c"]) - 1
                    fresh = [h for h in hits if h.i in (last, last - 1)]
                    new = [h for h in fresh if key_of(sym, d, h) not in seen]
                    for hit in new:
                        seen.add(key_of(sym, d, hit))
                        notify(sym, d, hit, dry_run=args.dry_run)
                        n_new += 1
        else:
            for sym in symbols:
                d, hits = scan_symbol(
                    sym,
                    limit=watch_limit,
                    **detect_kw,
                )
                if not hits or not d:
                    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {sym} 無訊號", flush=True)
                    continue
                last = len(d["c"]) - 1
                fresh = [h for h in hits if h.i in (last, last - 1)]
                new = [h for h in fresh if key_of(sym, d, h) not in seen]
                print(
                    f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {sym} "
                    f"歷史 {len(hits)}　剛收盤新訊號 {len(new)}",
                    flush=True,
                )
                for hit in new:
                    seen.add(key_of(sym, d, hit))
                    notify(sym, d, hit, dry_run=args.dry_run)
                    n_new += 1
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} 用 {time.time()-t0:.1f}s　新訊號 {n_new}",
            flush=True,
        )
        if n_new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print("watch 中，每根 1m 收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
