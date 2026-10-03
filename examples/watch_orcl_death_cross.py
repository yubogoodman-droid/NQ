#!/usr/bin/env python3
"""1m 空：4 小時新高後，同一根死亡交叉且破 MA25，形狀對齊 ORCL 22:49。

PyCharm：打開 examples/pycharm_orcl_watch.py，改最上面的設定，按綠三角 Run。
符合訊號會印出來、電腦彈窗，並推 Telegram。

對 2026-10-02 ORCL：22:44 創四小時高 144.95；22:49（+5 分）同一根收 144.44。
當根還要：MA7 領先夠久、高點後至少幾分鐘、從高點連陰、收在 K 棒下緣、已經離高點一段。

用法：
  python3 examples/pycharm_orcl_watch.py
  python3 examples/watch_orcl_death_cross.py --all
  python3 examples/watch_orcl_death_cross.py --all --scan --date 2026-10-02 --pages
  python3 examples/watch_orcl_death_cross.py --test
"""
from __future__ import annotations

import base64
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

# —— PyCharm：沒帶命令列參數時用這裡；有帶 CLI 則 CLI 優先 ——
WATCH_ALL = True
SYMBOLS = "ORCLUSDT"
DRY_RUN = False
TEST_ONLY = False
ONCE = False
DESKTOP_POPUP = True
WORKERS = 12
TELEGRAM_BOT_TOKEN = ""  # BotFather，也可改放專案根目錄 tg_config.env
TELEGRAM_CHAT_ID = ""

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO_ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO_ROOT / "output" / "orcl_death_cross_seen.json"
PAGES_HTML = REPO_ROOT / "docs" / "binance" / "death-cross-1m" / "index.html"
PAGES_WEEK = REPO_ROOT / "docs" / "binance" / "death-cross-1m-7d" / "index.html"
MAX_FETCH_BARS = 16000
MAX_EMBED_CHARTS = 80
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


def cutoff_ms(
    *,
    day: str | None,
    hours: int,
    days: int | None = None,
    now_ms: int | None = None,
) -> tuple[int, int]:
    """回傳 [start, end) 毫秒。有 day 就用台北日；有 days 用近 N 天；否則近 hours。"""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if day:
        return day_bounds_ms(day)
    if days is not None and days > 0:
        return now - int(days) * 86_400_000, now + 1
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


def fetch_klines(
    sym: str,
    limit: int = 400,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    drop_forming: bool = True,
) -> dict | None:
    raw: list = []
    if start_ms is not None:
        cursor = int(start_ms)
        stop = int(end_ms) if end_ms is not None else int(time.time() * 1000)
        while cursor < stop:
            batch = get_json(
                "/fapi/v1/klines",
                params={
                    "symbol": sym,
                    "interval": "1m",
                    "startTime": cursor,
                    "endTime": stop,
                    "limit": 1500,
                },
            )
            if not batch:
                break
            raw.extend(batch)
            nxt = int(batch[-1][0]) + 60_000
            if nxt <= cursor or len(batch) < 1500:
                break
            cursor = nxt
            if len(raw) >= MAX_FETCH_BARS:
                break
    else:
        params = {"symbol": sym, "interval": "1m", "limit": min(int(limit), 1500)}
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        raw = get_json("/fapi/v1/klines", params=params) or []
    if not raw or len(raw) < 30:
        return None
    now_ms = int(time.time() * 1000)
    if drop_forming and int(raw[-1][0]) + 60_000 > now_ms:
        raw = raw[:-1]
    if len(raw) < 30:
        return None
    seen: set[int] = set()
    t, o, h, l, c, v = [], [], [], [], [], []
    for x in raw:
        ts = int(x[0])
        if ts in seen:
            continue
        seen.add(ts)
        t.append(ts)
        o.append(float(x[1]))
        h.append(float(x[2]))
        l.append(float(x[3]))
        c.append(float(x[4]))
        v.append(float(x[5]))
    return {
        "t": np.array(t, np.int64),
        "o": np.array(o),
        "h": np.array(h),
        "l": np.array(l),
        "c": np.array(c),
        "v": np.array(v),
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


def _clean_notify_text(s: str, n: int) -> str:
    out = []
    for ch in (s or "").replace("\r", " "):
        if ch == "\n":
            out.append(" ")
        elif ch.isprintable() or ch == " ":
            out.append(ch)
    return "".join(out).strip()[:n]


def chart_tmp_path(sym: str, i: int) -> Path:
    root = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = Path.cwd()
    safe = "".join(c for c in sym if c.isalnum()) or "SYM"
    return root / f"orcl_dx_{safe}_{i}.png"


def desktop_notify(title: str, body: str) -> None:
    """Windows 氣泡 / macOS 通知 / Linux notify-send；失敗就略過。"""
    title = _clean_notify_text(title, 72) or "ORCL 1m 空"
    body = _clean_notify_text(body, 180) or "符合訊號"
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "Add-Type -AssemblyName System.Drawing; "
                "$n = New-Object System.Windows.Forms.NotifyIcon; "
                "$n.Icon = [System.Drawing.SystemIcons]::Warning; "
                "$n.Visible = $true; "
                f"$n.ShowBalloonTip(10000, {json.dumps(title, ensure_ascii=True)}, "
                f"{json.dumps(body, ensure_ascii=True)}, "
                "[System.Windows.Forms.ToolTipIcon]::Warning); "
                "Start-Sleep 10; $n.Dispose()"
            )
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        if platform.system() == "Darwin":
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f"display notification {json.dumps(body)} with title {json.dumps(title)}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        subprocess.Popen(
            ["notify-send", "--app-name=ORCL 1m 空", title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _cjk_font():
    try:
        from matplotlib import font_manager
    except Exception:
        return None
    for p in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(p).exists():
            return font_manager.FontProperties(fname=p)
    return None


def render_chart_png(sym: str, d: dict, hit: ShortHit, *, before: int = 70, after: int = 28) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    a0 = max(0, hit.i - before)
    a1 = min(len(d["c"]), hit.i + after + 1)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl]
    fig, ax = plt.subplots(figsize=(10.2, 4.6), facecolor="#0c1210")
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
    extra = ""
    if hit.drop_from_high is not None:
        extra += f"  離高 {hit.drop_from_high:+.2f}%"
    if hit.bars_after_high is not None:
        extra += f"  +{hit.bars_after_high}m"
    title = f"{sym}  1m  {hm(int(d['t'][hit.i]))}{extra}"
    fp = _cjk_font()
    if fp is not None:
        ax.set_title(title, color="#e8f0ea", fontsize=11, fontproperties=fp)
    else:
        ax.set_title(title, color="#e8f0ea", fontsize=11)
    if hit.peak_i is not None:
        px = hit.peak_i - a0
        if 0 <= px < len(c):
            ax.axvline(px, color="#c9a227", ls=":", lw=0.9)
            ax.scatter([px], [d["h"][hit.peak_i]], s=28, color="#c9a227", zorder=5, marker="^")
    times = d["t"][sl]
    step = max(1, len(xs) // 6)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(
        [datetime.fromtimestamp(int(t) / 1000, TZ).strftime("%H:%M") for t in times[::step]],
        color="#8aa193",
        fontsize=8,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc", ncol=3)
    fig.tight_layout(pad=0.45)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def draw_chart(sym: str, d: dict, hit: ShortHit, path: str) -> str | None:
    raw = render_chart_png(sym, d, hit, before=80, after=8)
    if not raw:
        return None
    Path(path).write_bytes(raw)
    return path


def chart_data_uri(sym: str, d: dict, hit: ShortHit) -> str | None:
    raw = render_chart_png(sym, d, hit)
    if not raw:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def charts_for_like(
    like: list[ScanRow],
    bars: dict[str, dict],
    *,
    max_charts: int = MAX_EMBED_CHARTS,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in like[: max(0, max_charts)]:
        hit = r.hit
        d = bars.get(r.symbol)
        if d is None or not np.any(d["t"] == r.ts_ms):
            d = fetch_klines(r.symbol, start_ms=r.ts_ms - 90 * 60_000, end_ms=r.ts_ms + 45 * 60_000)
            if not d:
                continue
            idxs = np.where(d["t"] == r.ts_ms)[0]
            if len(idxs) == 0:
                continue
            i = int(idxs[0])
            peak_i = hit.peak_i
            if hit.bars_after_high is not None:
                cand = i - int(hit.bars_after_high)
                peak_i = cand if 0 <= cand < len(d["t"]) else None
            hit = replace(hit, i=i, peak_i=peak_i)
        uri = chart_data_uri(r.symbol, d, hit)
        if uri:
            out[f"{r.symbol}:{r.ts_ms}"] = uri
    return out


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


def scan_symbol(
    sym: str,
    *,
    limit: int,
    start_ms: int | None = None,
    end_ms: int | None = None,
    **detect_kw,
) -> tuple[dict, list[ShortHit]]:
    raw = fetch_klines(sym, limit=limit, start_ms=start_ms, end_ms=end_ms)
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
    start_ms: int | None = None,
    end_ms: int | None = None,
    **detect_kw,
) -> tuple[dict, list[ScanRow]]:
    d, hits = scan_symbol(sym, limit=limit, start_ms=start_ms, end_ms=end_ms, **detect_kw)
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
    start_ms: int | None = None,
    end_ms: int | None = None,
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
            start_ms=start_ms,
            end_ms=end_ms,
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
            if keep_bars and d and rs:
                bars[s] = d
            if n % 80 == 0 or n == len(symbols):
                print(f"  … {n}/{len(symbols)}  訊號 {len(rows)}  失敗 {err}", flush=True)
    rows.sort(key=lambda r: (r.ts_ms, r.symbol))
    return rows, bars


def notify(sym: str, d: dict, hit: ShortHit, *, dry_run: bool, desktop: bool = True) -> None:
    text = format_hit(sym, d, hit)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("&gt;", ">")
        .split('<a href=')[0]
        .strip()
    )
    print("\n" + plain)
    if desktop:
        extra = ""
        if hit.bars_after_high is not None:
            extra = f" 高點後{hit.bars_after_high}m"
        desktop_notify(f"{sym} 空 1m", f"{hm(int(d['t'][hit.i]))}  收 {hit.close:g}{extra}  lead {hit.lead}")
        print("  → 電腦通知已跳")
    if dry_run:
        print("  → dry-run，不送 Telegram")
        return
    tmp = chart_tmp_path(sym, hit.i)
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


def print_scan(
    sym: str, d: dict, hits: list[ShortHit], *, hours: int, day: str | None = None, days: int | None = None
) -> None:
    if not d:
        print(f"{sym} 沒資料")
        return
    start, end = cutoff_ms(day=day, hours=hours, days=days, now_ms=int(d["t"][-1]) + 1)
    recent = [h for h in hits if in_window(int(d["t"][h.i]), start, end)]
    if day:
        label = day
    elif days:
        label = f"近 {days} 天"
    else:
        label = f"近 {hours}h"
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


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _med(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def summarize_rows(rows: list[ScanRow]) -> dict:
    """進場＝訊號收盤做空；fwd30 是價格，空損益 = −fwd30。"""
    lows = [r.low30 for r in rows if r.low30 is not None]
    fwds = [r.fwd30 for r in rows if r.fwd30 is not None]
    pnl = [-x for x in fwds]
    n = len(rows)
    dump05 = sum(1 for x in lows if x <= -0.5)
    dump1 = sum(1 for x in lows if x <= -1.0)
    dump2 = sum(1 for x in lows if x <= -2.0)
    wins = sum(1 for x in pnl if x > 0)
    by_day: list[dict] = []
    days = sorted({taipei_day(r.ts_ms) for r in rows})
    for day in days:
        rs = [r for r in rows if taipei_day(r.ts_ms) == day]
        dlows = [r.low30 for r in rs if r.low30 is not None]
        dfwds = [r.fwd30 for r in rs if r.fwd30 is not None]
        dpnl = [-x for x in dfwds]
        by_day.append(
            {
                "day": day,
                "n": len(rs),
                "dump1": sum(1 for x in dlows if x <= -1.0),
                "low30_avg": _avg(dlows),
                "pnl30_avg": _avg(dpnl),
                "win": sum(1 for x in dpnl if x > 0),
            }
        )
    return {
        "n": n,
        "symbols": len({r.symbol for r in rows}),
        "low30_avg": _avg(lows),
        "low30_med": _med(lows),
        "fwd30_avg": _avg(fwds),
        "pnl30_avg": _avg(pnl),
        "pnl30_med": _med(pnl),
        "win": wins,
        "win_pct": (100.0 * wins / len(pnl)) if pnl else None,
        "dump05": dump05,
        "dump1": dump1,
        "dump2": dump2,
        "dump1_pct": (100.0 * dump1 / len(lows)) if lows else None,
        "by_day": by_day,
    }


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
    st = summarize_rows(recent)
    print("\n回測（進場＝訊號收盤做空，看之後 30 分鐘，不是建議）")
    print(
        f"  30m 最低 平均 {fmt_pct(st['low30_avg'])}  中位 {fmt_pct(st['low30_med'])}  "
        f"砸≥0.5% {st['dump05']}  ≥1% {st['dump1']}  ≥2% {st['dump2']}"
    )
    wp = f"{st['win_pct']:.1f}%" if st["win_pct"] is not None else "—"
    print(
        f"  30m 收盤空損益 平均 {fmt_pct(st['pnl30_avg'])}  中位 {fmt_pct(st['pnl30_med'])}  "
        f"空賺 {st['win']}/{st['n']}（{wp}）"
    )
    if st["by_day"]:
        print("  按日：")
        for d in st["by_day"]:
            print(
                f"    {d['day']}  {d['n']:>4} 筆  砸≥1% {d['dump1']:>3}  "
                f"30m低均 {fmt_pct(d['low30_avg'])}  空損益均 {fmt_pct(d['pnl30_avg'])}"
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
    charts: dict[str, str] | None = None,
) -> Path:
    del top  # 網頁一律寫完整清單，不再截前 N 筆
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
    charts = charts or {}
    st = summarize_rows(recent)
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

    def loc_txt(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:.2f}"

    def row_tr(r: ScanRow, extra: str, *, href: str | None = None) -> str:
        name = escape(r.symbol)
        link = f"<a href='{escape(binance_href(r.symbol))}'>{name}</a>"
        if href:
            link += f" <a class='jump' href='{escape(href)}'>圖</a>"
        return (
            "<tr>"
            f"<td>{link}</td>"
            f"<td>{escape(hm(r.ts_ms))}</td>"
            f"<td>{r.hit.close:g}</td>"
            f"{pct_cell(r.low30)}{pct_cell(r.fwd15)}{pct_cell(r.fwd30)}"
            f"<td>{r.hit.lead}</td>"
            f"<td>{r.hit.bars_after_high if r.hit.bars_after_high is not None else '—'}</td>"
            f"{pct_cell(r.hit.drop_from_high)}"
            f"<td>{r.hit.reds_from_high if r.hit.reds_from_high is not None else '—'}</td>"
            f"<td>{loc_txt(r.hit.close_loc)}</td>"
            f"<td>{extra}</td>"
            "</tr>"
        )

    def like_card(i: int, r: ScanRow) -> str:
        aid = f"hit-{i:02d}"
        uri = charts.get(f"{r.symbol}:{r.ts_ms}", "")
        img = f'<img src="{uri}" alt="{escape(r.symbol)} {escape(hm(r.ts_ms))}"/>' if uri else ""
        loc = loc_txt(r.hit.close_loc)
        reds = r.hit.reds_from_high if r.hit.reds_from_high is not None else "—"
        after = r.hit.bars_after_high if r.hit.bars_after_high is not None else "—"
        return (
            f'<article class="card" id="{aid}">'
            '<div class="card-h">'
            f"<div><a href='{escape(binance_href(r.symbol))}'>{escape(r.symbol)}</a>"
            f" <span class='muted'>{escape(hm(r.ts_ms))}</span></div>"
            f'<div class="dn">{fmt_pct(r.low30)}</div>'
            "</div>"
            '<div class="meta">'
            f"收 {r.hit.close:g}　lead {r.hit.lead}　高點後 {after}m　"
            f"離高 {fmt_pct(r.hit.drop_from_high)}　連陰 {reds}　收位 {loc}　"
            f"15m {fmt_pct(r.fwd15)}　30m收 {fmt_pct(r.fwd30)}"
            "</div>"
            f"{img}"
            "</article>"
        )

    like_html = [
        row_tr(
            r,
            f"#{i:02d}",
            href=f"#hit-{i:02d}" if f"{r.symbol}:{r.ts_ms}" in charts else None,
        )
        for i, r in enumerate(like, 1)
    ]
    all_html = [
        row_tr(
            r,
            "砸≥1%" if r.low30 is not None and r.low30 <= -1.0 else "",
        )
        for r in ranked
    ]
    cards = [like_card(i, r) for i, r in enumerate(like, 1) if f"{r.symbol}:{r.ts_ms}" in charts]
    n_charts = len(cards)
    wp = f"{st['win_pct']:.1f}%" if st["win_pct"] is not None else "—"
    dump1p = f"{st['dump1_pct']:.1f}%" if st["dump1_pct"] is not None else "—"
    day_html = []
    for drow in st["by_day"]:
        day_html.append(
            "<tr>"
            f"<td>{escape(drow['day'])}</td>"
            f"<td>{drow['n']}</td>"
            f"<td>{drow['dump1']}</td>"
            f"{pct_cell(drow['low30_avg'])}"
            f"{pct_cell(drow['pnl30_avg'])}"
            f"<td>{drow['win']}</td>"
            "</tr>"
        )
    sym_html = []
    for _w, sym, n, dump, last in sym_rows:
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
.wrap{{max-width:1080px;margin:0 auto;padding:18px 14px 56px}}
h1{{font-size:1.35rem;margin:0 0 8px}}
h2{{font-size:1.05rem;margin:28px 0 10px}}
.sub{{color:#8aa193;line-height:1.55;margin:0 0 14px;font-size:.92rem}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}}
.chip{{border:1px solid rgba(232,240,234,.12);background:#14201b;border-radius:999px;padding:7px 12px;font-size:.82rem}}
.chip b{{color:#c9a227}}
.toc{{display:flex;flex-wrap:wrap;gap:10px 14px;margin:0 0 18px;font-size:.86rem}}
.toc a{{color:#c9a227}}
.rules{{margin:0 0 8px;padding:0 0 0 1.2em;color:#c8d5cc;line-height:1.7;font-size:.9rem}}
.rules li{{margin:2px 0}}
.scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
th,td{{padding:6px 6px;border-bottom:1px solid rgba(232,240,234,.1);text-align:left;white-space:nowrap}}
th{{color:#8aa193;font-weight:500}}
a{{color:#c9a227;text-decoration:none}}
.jump{{font-size:.75rem;margin-left:4px;color:#8aa193}}
.dn{{color:#e35d5d}} .up{{color:#3dba7a}} .muted{{color:#8aa193}}
.note{{margin-top:16px;color:#8aa193;font-size:.8rem;line-height:1.5}}
.filter{{margin:0 0 10px;padding:8px 10px;width:min(320px,100%);border:1px solid rgba(232,240,234,.14);border-radius:8px;background:#101814;color:#e8f0ea}}
.card{{border:1px solid rgba(232,240,234,.12);background:#101814;border-radius:12px;padding:12px;margin:0 0 14px}}
.card-h{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;font-size:1.02rem;font-weight:600}}
.card-h .dn{{font-variant-numeric:tabular-nums}}
.meta{{color:#8aa193;font-size:.8rem;margin:6px 0 8px;line-height:1.5}}
.card img{{width:100%;border-radius:8px;display:block;background:#0c1210}}
</style>
</head>
<body>
<div class="wrap">
<h1>{escape(title)}</h1>
<p class="sub">回測：進場＝訊號收盤做空，看之後 30 分鐘收盤與最低。砸 ≥1% 最多嵌 {MAX_EMBED_CHARTS} 張圖。不是進出場建議。</p>
<div class="chips">
  <div class="chip">掃 <b>{n_symbols}</b> 檔</div>
  <div class="chip">訊號 <b>{len(recent)}</b> 筆</div>
  <div class="chip">有訊號 <b>{len(names)}</b> 檔</div>
  <div class="chip">砸 ≥1% <b>{len(like)}</b>（{dump1p}）</div>
  <div class="chip">30m低均 <b>{fmt_pct(st['low30_avg'])}</b></div>
  <div class="chip">空30m均 <b>{fmt_pct(st['pnl30_avg'])}</b></div>
  <div class="chip">空勝率 <b>{wp}</b></div>
  <div class="chip">圖 <b>{n_charts}</b></div>
</div>
<nav class="toc">
  <a href="#rules">規則</a>
  <a href="#bt">按日</a>
  <a href="#dump">砸 ≥1%（{len(like)}）</a>
  <a href="#all">全部 {len(ranked)} 筆</a>
  <a href="#syms">{len(sym_rows)} 檔</a>
</nav>
<h2 id="rules">規則（對齊 ORCL 22:49）</h2>
<ol class="rules">
  <li>創下過去 <b>4 小時新高</b></li>
  <li>之後 <b>30 分鐘內</b>，<b>同一根</b> 1 分鐘 K：MA7 下穿 MA14（死亡交叉）<b>且</b>收盤由上跌破 MA25</li>
  <li>交叉前 MA7 領先 ≥ <b>15</b> 根（ORCL 是 22）</li>
  <li>高點後至少 <b>4</b> 分鐘（ORCL 是 +5，不是新高當根）</li>
  <li>從高點到訊號至少 <b>3</b> 根陰線（ORCL 是 4）</li>
  <li>收盤在 K 棒下緣（位置 ≤ 0.25，ORCL 收在最低）</li>
  <li>收盤已離 4h 高 ≥ <b>0.25%</b>（ORCL 約 −0.35%）</li>
</ol>
<p class="note">空損益 = −（訊號後 30 分鐘收盤漲跌）。30m 最低是這段最深不利（對空是有利）。砸 ≥2% {st['dump2']} 筆、≥0.5% {st['dump05']} 筆。金點線是 4 小時高，紅虛線是訊號根。</p>
<h2 id="bt">按日</h2>
<div class="scroll">
<table>
<thead><tr><th>台北日</th><th>筆數</th><th>砸≥1%</th><th>30m低均</th><th>空30m均</th><th>空賺筆數</th></tr></thead>
<tbody>
{"".join(day_html) or "<tr><td colspan='6' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
</div>
<h2 id="dump">之後砸 ≥1%（全部 {len(like)} 筆{f'，圖 {n_charts}' if n_charts else ''}）</h2>
<div class="scroll">
<table>
<thead><tr><th>標的</th><th>時間</th><th>收</th><th>30m低</th><th>15m</th><th>30m收</th><th>lead</th><th>高點後</th><th>離高</th><th>連陰</th><th>收位</th><th></th></tr></thead>
<tbody>
{"".join(like_html) or "<tr><td colspan='12' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
</div>
{"".join(cards)}
<h2 id="all">全部 ORCL 形（{len(ranked)} 筆）</h2>
<input class="filter" id="q" placeholder="篩選標的或時間…" autocomplete="off"/>
<div class="scroll">
<table>
<thead><tr><th>標的</th><th>時間</th><th>收</th><th>30m低</th><th>15m</th><th>30m收</th><th>lead</th><th>高點後</th><th>離高</th><th>連陰</th><th>收位</th><th></th></tr></thead>
<tbody id="all-body">
{"".join(all_html) or "<tr><td colspan='12' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
</div>
<h2 id="syms">有訊號的標的（{len(sym_rows)} 檔）</h2>
<div class="scroll">
<table>
<thead><tr><th>標的</th><th>筆數</th><th>最深那筆</th><th>收</th><th>30m低</th><th>破25</th><th>最後一筆</th></tr></thead>
<tbody>
{"".join(sym_html) or "<tr><td colspan='7' class='muted'>沒有訊號</td></tr>"}
</tbody>
</table>
</div>
<p class="note">ORCL 10-02：22:44 創四小時高 144.95；22:49（+5 分）同一根死亡交叉且跌破 MA25，lead 22、連陰 4、收在下緣、離高 −0.35%，之後砸到 141。</p>
</div>
<script>
(function(){{
  var q = document.getElementById('q');
  var rows = document.querySelectorAll('#all-body tr');
  if (!q) return;
  q.addEventListener('input', function(){{
    var s = (q.value || '').toUpperCase();
    rows.forEach(function(tr){{
      tr.style.display = !s || tr.textContent.toUpperCase().indexOf(s) >= 0 ? '' : 'none';
    }});
  }});
}})();
</script>
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
        "all": [row_to_json(r) for r in ranked],
        "top": [row_to_json(r) for r in ranked],
        "like": [row_to_json(r) for r in like],
        "stats": st,
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


def test_telegram(*, desktop: bool = True) -> int:
    apply_keys()
    if desktop:
        desktop_notify("ORCL 1m 空", "測試通知：電腦彈窗已通。")
        print("電腦彈窗已送一則測試")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（腳本最上面或 tg_config.env），先用電腦彈窗。")
        return 0
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
    start, end = cutoff_ms(day=args.date, hours=args.hours, days=getattr(args, "days", None))
    if args.date:
        return start, end, f"台北 {args.date}"
    if getattr(args, "days", None):
        a = datetime.fromtimestamp(start / 1000, TZ).strftime("%m-%d %H:%M")
        b = datetime.fromtimestamp((end - 1) / 1000, TZ).strftime("%m-%d %H:%M")
        return start, end, f"近 {args.days} 天（{a} → {b} 台北）"
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
        span_h = (end - start) / 3_600_000
        html_path = args.html
        if args.pages:
            html_path = html_path or str(PAGES_WEEK if span_h >= 36 else PAGES_HTML)
        want_charts = bool(html_path) and not args.no_charts
        keep_bars = want_charts and span_h < 36
        rows, bars = scan_universe(
            symbols,
            limit=args.limit,
            workers=args.workers,
            vols=vols,
            keep_bars=keep_bars,
            start_ms=start - (detect_kw["high_lookback"] + 25) * 60_000,
            end_ms=end + 40 * 60_000,
            **detect_kw,
        )
        print(f"掃完 {time.time()-t0:.1f}s", flush=True)
        print_market_scan(rows, start_ms=start, end_ms=end, n_symbols=len(symbols), top=args.top)
        if html_path:
            like = orcl_like([r for r in rows if in_window(r.ts_ms, start, end)])
            charts = charts_for_like(like, bars) if want_charts else {}
            if want_charts:
                print(f"嵌入砸≥1% 圖 {len(charts)}/{len(like)}", flush=True)
            out = write_html_report(
                html_path,
                rows,
                start_ms=start,
                end_ms=end,
                n_symbols=len(symbols),
                title=f"幣安 1m 空 · ORCL 形（同根死亡交叉且破 MA25）· {label}",
                top=args.top,
                charts=charts,
            )
            print(f"html={out}")
        return 0
    for sym in symbols:
        d, hits = scan_symbol(
            sym,
            limit=args.limit,
            start_ms=start - (detect_kw["high_lookback"] + 25) * 60_000,
            end_ms=end + 40 * 60_000,
            **detect_kw,
        )
        print_scan(sym, d, hits, hours=args.hours, day=args.date, days=getattr(args, "days", None))
    return 0


def build_parser():
    import argparse

    p = argparse.ArgumentParser(description="1m 4h新高後 MA7/MA14 死亡交叉且破 MA25 → 掃幣安 / 通知")
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
    p.add_argument("--hours", type=int, default=24, help="沒指定 --date / --days 時，回看小時數")
    p.add_argument("--days", type=int, default=None, help="回測近 N 天（例如 7），台北時間往回算")
    p.add_argument("--date", default=None, help="台北日 YYYY-MM-DD，只看這一天")
    p.add_argument("--top", type=int, default=40, help="市場掃描列出急殺最深幾筆")
    p.add_argument("--html", default=None, help="寫入 HTML 報告路徑")
    p.add_argument("--pages", action="store_true", help="寫到 docs/binance/death-cross-1m/（單檔含規則、砸>=1%%圖、全部訊號）")
    p.add_argument("--no-charts", action="store_true", help="HTML 不嵌圖")
    p.add_argument("--limit", type=int, default=1500, help="K 線根數")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測通知通不通")
    p.add_argument("--dry-run", action="store_true", help="只印不送 Telegram（電腦彈窗仍會跳）")
    p.add_argument("--no-desktop", dest="desktop", action="store_false", help="關掉電腦彈窗")
    p.set_defaults(require_cross_ma25=True, desktop=True)
    return p


def apply_pycharm_defaults(args, argv: list[str]):
    """沒帶命令列參數時，用檔案最上面那組 PyCharm 設定。"""
    if argv:
        return args
    args.all = WATCH_ALL
    args.symbols = SYMBOLS
    args.dry_run = DRY_RUN
    args.once = ONCE
    args.test = TEST_ONLY
    args.desktop = DESKTOP_POPUP
    args.workers = WORKERS
    return args


def watch_args_from_pycharm(
    *,
    watch_all: bool = True,
    symbols: str = "ORCLUSDT",
    dry_run: bool = False,
    test_only: bool = False,
    once: bool = False,
    desktop: bool = True,
    workers: int = 12,
):
    args = build_parser().parse_args([])
    args.all = watch_all
    args.symbols = symbols
    args.dry_run = dry_run
    args.test = test_only
    args.once = once
    args.desktop = desktop
    args.workers = workers
    return args


def run_watch(args, symbols: list[str], vols: dict[str, float]) -> int:
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
        print("全市場每分鐘掃一次；符合就跳通知。報告請用 --scan --pages。", flush=True)
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
                        notify(sym, d, hit, dry_run=args.dry_run, desktop=getattr(args, "desktop", True))
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
                    notify(sym, d, hit, dry_run=args.dry_run, desktop=getattr(args, "desktop", True))
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
    print("watch 中，每根 1m 收盤掃一次（PyCharm 按紅方塊 Stop / Ctrl+C）", flush=True)
    try:
        while True:
            wait_next_close()
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


def run_pycharm(
    *,
    watch_all: bool = True,
    symbols: str = "ORCLUSDT",
    dry_run: bool = False,
    test_only: bool = False,
    once: bool = False,
    desktop: bool = True,
    workers: int = 12,
    telegram_bot_token: str = "",
    telegram_chat_id: str = "",
) -> int:
    if telegram_bot_token.strip():
        os.environ["TELEGRAM_BOT_TOKEN"] = telegram_bot_token.strip()
    if telegram_chat_id.strip():
        os.environ["TELEGRAM_CHAT_ID"] = telegram_chat_id.strip()
    apply_keys()
    args = watch_args_from_pycharm(
        watch_all=watch_all,
        symbols=symbols,
        dry_run=dry_run,
        test_only=test_only,
        once=once,
        desktop=desktop,
        workers=workers,
    )
    if args.test:
        return test_telegram(desktop=args.desktop)
    names, vols = load_symbol_list(args)
    return run_watch(args, names, vols)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    args = p.parse_args(argv)
    apply_pycharm_defaults(args, argv)
    if args.loose and "--min-lead" not in argv:
        args.min_lead = 5
    apply_keys()
    if args.test:
        return test_telegram(desktop=getattr(args, "desktop", True))

    symbols, vols = load_symbol_list(args)
    if args.scan:
        return run_scan(args, symbols, vols)
    return run_watch(args, symbols, vols)


if __name__ == "__main__":
    raise SystemExit(main())
