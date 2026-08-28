#!/usr/bin/env python3
"""台股成交額前 200：五分 K 回測 240MA，Telegram 跳通知。

對齊 XQ 五分圖的 5/10/20/60/120/240MA。**1815 富喬預設必抓**。
價從均線上方拉開後，這根五分 K 的最低價碰到 240MA、收盤仍守住，就推一則（帶圖）。
開盤回測也抓。

用法:
  python3 examples/watch_tw_5m_ma240.py --test
  python3 examples/watch_tw_5m_ma240.py --dry-run --once --limit 30
  python3 examples/watch_tw_5m_ma240.py --scan --dry-run --limit 200 --days 10 --pages
  python3 examples/watch_tw_5m_ma240.py              # 每根五分收盤掃一次

預設濾掉股價 >700。HTML 圖用 data URI 嵌進去，htmlpreview 才看得到。

Telegram 憑證放 tg_config.env（勿提交）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    TPE,
    UA,
    _chart_payload_to_df,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO / "output" / "tw_ma240_seen.json"
PAGES = REPO / "docs" / "tw-5m-ma240" / "index.html"
MA_PERIODS = (5, 10, 20, 60, 120, 240)
MA_COLORS = {
    5: "#4ea3ff",
    10: "#3dba7a",
    20: "#f0c14a",
    60: "#ff8a4c",
    120: "#7fd3f0",
    240: "#c084fc",
}
KEEP_DEFAULT = ("1815",)
KNOWN_MARKET = {"1815": "otc"}
KNOWN_NAME = {"1815": "富喬"}


@dataclass
class RetestHit:
    idx: int
    code: str
    name: str
    symbol: str
    rank: int
    amount: int
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    mas: dict[int, float]
    touch_pct: float
    ext_pct: float
    pierced: bool

    @property
    def ma240(self) -> float:
        return float(self.mas[240])

    @property
    def dist_pct(self) -> float:
        return (self.close / self.ma240 - 1.0) * 100.0

    @property
    def key(self) -> str:
        ts = self.ts
        if getattr(ts, "tzinfo", None) is None:
            stamp = str(ts)
        else:
            stamp = ts.tz_convert(TPE).isoformat()
        return f"{self.symbol}:{stamp}"


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


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= n:
        out[n - 1 :] = np.convolve(arr, np.ones(n) / n, mode="valid")
    return out


def tw_tick(price: float) -> float:
    p = abs(float(price))
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.1
    if p < 500:
        return 0.5
    if p < 1000:
        return 1.0
    return 5.0


def touch_band(ma: float, touch_pct: float, ticks: int = 1) -> float:
    return max(abs(ma) * float(touch_pct), tw_tick(ma) * ticks)


def fetch_yahoo_5m(symbol: str, range_: str = "1mo") -> tuple[pd.DataFrame, str]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=5m&range={range_}&includePrePost=false"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last: Exception | None = None
    payload: dict | None = None
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                payload = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return pd.DataFrame(), ""
            last = exc
            time.sleep(0.8 * (i + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 * (i + 1))
    if payload is None:
        raise RuntimeError(f"GET failed {url}: {last}")
    df = _chart_payload_to_df(payload)
    result = (payload.get("chart") or {}).get("result") or []
    name = ""
    if result:
        name = str((result[0].get("meta") or {}).get("shortName") or "")
    return df, name


def fetch_symbol_5m(row: dict, range_: str = "1mo") -> pd.DataFrame:
    code = str(row.get("code") or "")
    market = str(row.get("market") or KNOWN_MARKET.get(code) or "tse")
    symbol = str(row.get("symbol") or yahoo_symbol(code, market))
    row["symbol"] = symbol
    row["market"] = market
    df, name = fetch_yahoo_5m(symbol, range_)
    if len(df) < 250:
        alt_mkt = "otc" if market == "tse" else "tse"
        alt = yahoo_symbol(code, alt_mkt)
        df2, name2 = fetch_yahoo_5m(alt, range_)
        if len(df2) > len(df):
            row["symbol"] = alt
            row["market"] = alt_mkt
            df, name = df2, name2
    if name and (not row.get("name") or row.get("name") == row.get("code")):
        row["name"] = KNOWN_NAME.get(code) or name
    elif not row.get("name") or row.get("name") == code:
        row["name"] = KNOWN_NAME.get(code, row.get("name") or code)
    return df


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].to_numpy(dtype=float)
    for n in MA_PERIODS:
        out[f"MA{n}"] = sma(close, n)
    return out


def detect_retests(
    df: pd.DataFrame,
    *,
    ma_n: int = 240,
    touch_pct: float = 0.002,
    ticks: int = 1,
    min_away_pct: float = 0.01,
    away_lookback: int = 24,
    min_above: int = 10,
    max_close_below_pct: float = 0.002,
    max_close_above_pct: float = 0.0045,
    min_ma_slope_pct: float = -0.15,
    slope_bars: int = 12,
    skip_open_minutes: int = 0,
    cooldown_bars: int = 6,
    start: int | None = None,
) -> list[int]:
    """從上方回測 240MA：先拉開，這根低點碰到均線，收盤仍守住。

    開盤回測也算（富喬 1815 今天就是 09:05 刺破收回）。
    刺破收回、或低點貼到均線後彈開，都算。
    黏著均線走、或收盤明顯跌破，不算。
    """
    if df is None or len(df) < ma_n + away_lookback + 2:
        return []
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    ma = sma(close, ma_n)
    n = len(close)
    lo = ma_n + max(away_lookback, slope_bars)
    i0 = lo if start is None else max(lo, start)
    hits: list[int] = []
    for i in range(i0, n):
        ts = df.index[i]
        if skip_open_minutes and hasattr(ts, "hour"):
            minutes = int(ts.hour) * 60 + int(ts.minute)
            open_m = 9 * 60
            if open_m <= minutes < open_m + skip_open_minutes:
                continue
        m = ma[i]
        prev = ma[i - 1]
        if np.isnan(m) or np.isnan(prev) or m <= 0:
            continue
        if np.isnan(ma[i - slope_bars]) or ma[i - slope_bars] <= 0:
            continue
        slope_pct = (m / ma[i - slope_bars] - 1.0) * 100.0
        if slope_pct < min_ma_slope_pct:
            continue
        band = touch_band(m, touch_pct, ticks=ticks)
        if low[i] > m + band:
            continue
        if high[i] < m - band:
            continue
        if close[i] < m * (1.0 - max_close_below_pct):
            continue
        pierced = bool(low[i] < m)
        if low[i - 1] <= prev + touch_band(prev, touch_pct, ticks=ticks):
            continue
        a0 = i - away_lookback
        window_c = close[a0:i]
        window_m = ma[a0:i]
        valid = ~np.isnan(window_m) & (window_m > 0)
        if int(valid.sum()) < min_above:
            continue
        rel = window_c[valid] / window_m[valid] - 1.0
        if int((rel > 0).sum()) < min_above:
            continue
        if float(rel.max()) < min_away_pct:
            continue
        if hits and i - hits[-1] < cooldown_bars:
            continue
        hits.append(i)
    return hits


def hit_from_row(df: pd.DataFrame, i: int, row: dict, ext_lookback: int = 24) -> RetestHit:
    mas = {}
    for n in MA_PERIODS:
        col = f"MA{n}"
        if col not in df.columns:
            df = add_mas(df)
        mas[n] = float(df[col].iloc[i])
    ma240 = mas[240]
    a0 = max(0, i - ext_lookback)
    ext = float((df["Close"].iloc[a0:i] / df["MA240"].iloc[a0:i] - 1.0).max() * 100.0)
    low = float(df["Low"].iloc[i])
    return RetestHit(
        idx=i,
        code=str(row.get("code") or ""),
        name=str(row.get("name") or ""),
        symbol=str(row.get("symbol") or ""),
        rank=int(row.get("rank") or 0),
        amount=int(row.get("amount") or 0),
        ts=pd.Timestamp(df.index[i]),
        open=float(df["Open"].iloc[i]),
        high=float(df["High"].iloc[i]),
        low=low,
        close=float(df["Close"].iloc[i]),
        volume=float(df["Volume"].iloc[i] if "Volume" in df.columns else 0),
        mas=mas,
        touch_pct=(low / ma240 - 1.0) * 100.0,
        ext_pct=ext,
        pierced=low < ma240,
    )


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        raw = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return set(raw)
        return set(raw.get("keys") or [])
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = sorted(seen)[-4000:]
    SEEN_PATH.write_text(json.dumps(keep, ensure_ascii=False, indent=0), encoding="utf-8")


def telegram_send(text: str, *, photo: str | None = None, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    token = (env("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (env("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return False
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as fh:
                r = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": text[:1024],
                        "parse_mode": "HTML",
                    },
                    files={"photo": fh},
                    timeout=25,
                )
            if r.ok:
                return True
            print(f"[tg] photo HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
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
            print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return False
        data = r.json()
        return bool(data.get("ok"))
    except Exception as exc:  # noqa: BLE001
        print(f"[tg] {exc}", file=sys.stderr)
        return False


def _fmt_px(x: float) -> str:
    if x >= 500:
        return f"{x:.0f}"
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def format_hit(hit: RetestHit, *, live: bool = False) -> str:
    ts = hit.ts
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(TPE)
    kind = "刺破收回" if hit.pierced else "貼到均線"
    title = "回測中" if live else "回測 5分 240MA"
    rank = f"成交額第 {hit.rank}" if hit.rank else ""
    amt = f"{hit.amount / 1e8:.1f} 億" if hit.amount else ""
    vol = f"{hit.volume:,.0f}" if hit.volume else "--"
    mas = "  ".join(f"{n}MA {_fmt_px(hit.mas[n])}" for n in MA_PERIODS)
    return (
        f"📡 <b>{title}</b>  {kind}\n"
        f"<b>{hit.code} {hit.name}</b>  {hit.symbol}\n"
        f"時間 <code>{ts.strftime('%m-%d %H:%M')}</code>  五分K\n"
        f"現價 <code>{_fmt_px(hit.close)}</code>  "
        f"240MA <code>{_fmt_px(hit.ma240)}</code>  "
        f"收盤 {hit.dist_pct:+.2f}%\n"
        f"最低 <code>{_fmt_px(hit.low)}</code>  距均 {hit.touch_pct:+.2f}%\n"
        f"回測前拉開 {hit.ext_pct:+.2f}%\n"
        f"{mas}\n"
        f"這根量 {vol}  {rank}  {amt}\n"
        f"#回測240MA #五分K #{hit.code}"
    )


def session_dates_back(n: int, now: datetime | None = None) -> set:
    """最近 n 個台股交易日（週一到週五，不含國定假日）。"""
    cur = (now or datetime.now(TPE)).astimezone(TPE).date()
    out = []
    guard = 0
    while len(out) < max(0, n) and guard < 21:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
        guard += 1
    return set(out)


def hit_session_date(hit: RetestHit):
    ts = hit.ts
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(TPE)
    return pd.Timestamp(ts).date()


def filter_hits_days(hits: list[RetestHit], days: int, now: datetime | None = None) -> list[RetestHit]:
    if not days or days <= 0:
        return hits
    keep = session_dates_back(days, now)
    return [h for h in hits if hit_session_date(h) in keep]


def format_hit_line(hit: RetestHit) -> str:
    ts = hit.ts
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(TPE)
    kind = "刺破收回" if hit.pierced else "貼到均線"
    return (
        f"{ts.strftime('%m-%d %H:%M')}  {hit.code} {hit.name}  {kind}  "
        f"{_fmt_px(hit.close)} / 240MA {_fmt_px(hit.ma240)}  低點{hit.touch_pct:+.2f}%"
    )


def _setup_cjk_font(plt) -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    wanted = ("WenQuanYi Micro Hei", "Noto Sans CJK TC", "Droid Sans Fallback")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def draw_chart(
    df: pd.DataFrame,
    hit: RetestHit,
    path: str,
    *,
    figsize: tuple[float, float] = (8.8, 4.8),
    dpi: int = 90,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    _setup_cjk_font(plt)
    work = add_mas(df)
    a0 = max(0, hit.idx - 70)
    a1 = min(len(work), hit.idx + 8)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o = work["Open"].to_numpy()[sl]
    h = work["High"].to_numpy()[sl]
    l = work["Low"].to_numpy()[sl]
    c = work["Close"].to_numpy()[sl]
    v = work["Volume"].to_numpy()[sl] if "Volume" in work.columns else np.zeros(a1 - a0)
    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0b0e11",
    )
    for a in (ax, axv):
        a.set_facecolor("#10141a")
        a.tick_params(colors="#8b949e", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#30363d")
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#ff5c5c"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.75)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-9) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        axv.bar(xs[k], v[k], width=0.8, color=("#3dba7a99" if up else "#ff5c5c99"), linewidth=0)
    for n, col in MA_COLORS.items():
        lw = 2.15 if n == 240 else 1.05
        ax.plot(xs, work[f"MA{n}"].to_numpy()[sl], color=col, lw=lw, label=f"{n}MA")
    mark = hit.idx - a0
    if 0 <= mark < len(c):
        ax.axvline(mark, color="#c084fc", ls="--", lw=0.9, alpha=0.85)
        ax.scatter([mark], [c[mark]], s=42, color="#c084fc", zorder=5)
    title = f"{hit.code} {hit.name}  5分  回測240MA {_fmt_px(hit.ma240)}"
    ax.set_title(title, color="#e6edf3", fontsize=12)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c9d1d9", ncol=6)
    axv.set_ylabel("量", color="#8b949e", fontsize=8)
    fig.tight_layout(pad=0.55)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def want_chart(hit: RetestHit, mode: str) -> bool:
    if mode in {"none", "off"}:
        return False
    if hit.code == "1815":
        return True
    if mode in {"fuqiao", "1815"}:
        return False
    if mode in {"pierce", "auto", ""}:
        return bool(hit.pierced)
    return True


def png_to_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def hit_card_html(hit: RetestHit, n: int, img_src: str | None) -> str:
    ts = hit.ts
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(TPE)
    kind = "刺破收回" if hit.pierced else "貼到均線"
    cls = "pierce" if hit.pierced else "hug"
    pin = " pin" if hit.code == "1815" else ""
    rank = f"成交額第 {hit.rank}" if hit.rank else ""
    amt = f"{hit.amount / 1e8:.1f} 億" if hit.amount else ""
    vol = f"{hit.volume:,.0f}" if hit.volume else "--"
    mas = "  ".join(f"{nma}MA {_fmt_px(hit.mas[nma])}" for nma in MA_PERIODS)
    img = ""
    if img_src:
        img = (
            f"<div class='mini-chart'><img src='{escape(img_src, quote=True)}' "
            f"alt='{escape(hit.code)} {escape(hit.name)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
        )
    return (
        f"<article class='trade-card{pin}' id='h{n}'>"
        "<header class='card-header'>"
        "<div class='card-title'>"
        f"<span class='trade-no'>#{n} · {escape(hit.code)} {escape(hit.name)} · {kind}</span>"
        f"<span class='trade-time'>{escape(ts.strftime('%Y-%m-%d %H:%M'))}  五分K</span>"
        "</div>"
        f"<div class='card-pnl {cls}'>低點 {hit.touch_pct:+.2f}%</div>"
        "</header>"
        "<div class='tags'>"
        f"<span class='tag tag-info'>{escape(hit.symbol)}</span>"
        f"<span class='tag {cls}'>{kind}</span>"
        f"<span class='tag'>收 {hit.dist_pct:+.2f}%</span>"
        "</div>"
        "<pre class='trade-detail'>"
        f"現價 {_fmt_px(hit.close)}  240MA {_fmt_px(hit.ma240)}  最低 {_fmt_px(hit.low)}\n"
        f"回測前拉開 {hit.ext_pct:+.2f}%\n{escape(mas)}\n"
        f"這根量 {vol}  {rank}  {amt}"
        "</pre>"
        f"{img}</article>"
    )


def write_html_report(
    path: Path,
    hits: list[RetestHit],
    universe: list[dict],
    period: str,
    date: str,
    *,
    chart_mode: str = "pierce",
    max_price: float | None = 700.0,
) -> Path:
    path = Path(path)
    img_dir = path.parent / "img"
    if img_dir.is_dir():
        for old in img_dir.glob("*.png"):
            old.unlink()
        try:
            img_dir.rmdir()
        except OSError:
            pass

    pierce = sum(1 for h in hits if h.pierced)
    fuqiao = [h for h in hits if h.code == "1815"]
    by_day: dict[str, list[RetestHit]] = {}
    for h in hits:
        by_day.setdefault(str(hit_session_date(h)), []).append(h)

    cards_fuqiao = []
    cards_days = []
    n = 0
    charted = 0

    def emit(hit: RetestHit) -> str:
        nonlocal n, charted
        n += 1
        img_src = None
        df = getattr(hit, "_df", None)
        if df is not None and want_chart(hit, chart_mode):
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                if draw_chart(df, hit, tmp):
                    img_src = png_to_data_uri(Path(tmp))
                    charted += 1
            finally:
                Path(tmp).unlink(missing_ok=True)
        return hit_card_html(hit, n, img_src)

    if fuqiao:
        cards_fuqiao.append("<h2>富喬 1815</h2>")
        for h in fuqiao:
            cards_fuqiao.append(emit(h))
    for day in sorted(by_day):
        day_hits = by_day[day]
        cards_days.append(
            f"<h2 id='d{day}'>{escape(day)} · {len(day_hits)} 筆</h2>"
        )
        for h in day_hits:
            cards_days.append(emit(h))

    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    px_note = f" · 股價≤{max_price:g}" if max_price else ""
    day_nav = " ".join(
        f"<a href='#d{d}'>{d[5:]} {len(by_day[d])}</a>" for d in sorted(by_day)
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股五分K 回測240MA · {escape(period)}</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} h2{{font-size:15px;margin:18px 0 10px;color:#c084fc}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.day-nav{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px}}
.day-nav a{{font-size:12px;color:#79c0ff;text-decoration:none;border:1px solid #30363d;border-radius:999px;padding:4px 10px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.trade-card.pin{{border-color:#c084fc}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700;white-space:nowrap}} .pierce{{color:#c084fc}} .hug{{color:#7fd3f0}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.tag.pierce{{color:#c084fc;border-color:#6e4a8a}} .tag.hug{{color:#7fd3f0}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 5分 回測 240MA · {escape(period)}</h1>
<p class="muted">基準日 {escape(date)} · 成交額前 {len(universe)} · 末名約 {cutoff:.1f} 億{px_note} · 1815 富喬必抓
<br/>從 240MA 上方拉開後回測碰到、收盤守住。開盤也算。圖嵌在頁面裡（富喬 + 刺破收回）。</p>
<div class="cards">
<div class="card">筆數<b>{len(hits)}</b></div>
<div class="card">標的<b>{len({h.code for h in hits})}</b></div>
<div class="card">刺破收回<b>{pierce}</b></div>
<div class="card">富喬<b>{len(fuqiao)}</b></div>
</div>
<div class="day-nav">{day_nav}</div>
</section>
{''.join(cards_fuqiao)}
{''.join(cards_days) or "<div class='empty'>這段期間沒有回測 240MA 訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"html={path} charts={charted}", flush=True)
    return path


def write_view_html(src: Path) -> Path:
    src = Path(src)
    out = src.with_name("view.html")
    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def market_session(now: datetime | None = None) -> bool:
    cur = (now or datetime.now(TPE)).astimezone(TPE)
    if cur.weekday() >= 5:
        return False
    minutes = cur.hour * 60 + cur.minute
    return (9 * 60) <= minutes <= (13 * 60 + 32)


def seconds_until_next_5m(now: datetime | None = None, extra: float = 8.0) -> float:
    cur = (now or datetime.now(TPE)).astimezone(TPE)
    epoch = cur.timestamp()
    nxt = (int(epoch) // 300 + 1) * 300 + extra
    return max(1.0, nxt - epoch)


def next_session_open(now: datetime | None = None) -> datetime:
    cur = (now or datetime.now(TPE)).astimezone(TPE)
    target = cur.replace(hour=9, minute=0, second=8, microsecond=0)
    if cur >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def seconds_until_next_scan(now: datetime | None = None) -> float:
    cur = (now or datetime.now(TPE)).astimezone(TPE)
    if market_session(cur):
        return seconds_until_next_5m(cur)
    if cur.weekday() < 5:
        open_at = cur.replace(hour=9, minute=0, second=8, microsecond=0)
        if cur < open_at:
            return max(1.0, (open_at - cur).total_seconds())
    nxt = next_session_open(cur)
    return max(1.0, (nxt - cur).total_seconds())


def stub_row(code: str) -> dict:
    market = KNOWN_MARKET.get(code, "tse")
    return {
        "rank": 0,
        "code": code,
        "name": KNOWN_NAME.get(code, code),
        "market": market,
        "amount": 0,
        "close": None,
        "symbol": yahoo_symbol(code, market),
        "pinned": True,
    }


def pin_keep(rows: list[dict], keep: Iterable[str]) -> list[dict]:
    wanted = [c.strip() for c in keep if c.strip()]
    if not wanted:
        return rows
    by_code = {r["code"]: r for r in rows}
    pinned: list[dict] = []
    for code in wanted:
        if code in by_code:
            row = by_code[code]
            row["pinned"] = True
            if not row.get("name") or row.get("name") == code:
                row["name"] = KNOWN_NAME.get(code, row.get("name") or code)
            pinned.append(row)
        else:
            pinned.append(stub_row(code))
    rest = [r for r in rows if r["code"] not in set(wanted)]
    return pinned + rest


def select_universe(
    rows: list[dict],
    limit: int,
    keep: Iterable[str] | None = None,
    max_price: float | None = 700.0,
) -> tuple[list[dict], list[dict]]:
    kept, dropped = filter_by_max_price(rows, max_price, limit)
    keep_codes = [c.strip() for c in (keep or KEEP_DEFAULT) if c.strip()]
    return pin_keep(kept, keep_codes), dropped


def load_universe(
    limit: int,
    date: str = "",
    codes: Iterable[str] | None = None,
    keep: Iterable[str] | None = None,
    max_price: float | None = 700.0,
    pool: int = 400,
) -> tuple[str, list[dict]]:
    ymd = resolve_twse_date(date or last_tw_session_yyyymmdd())
    keep_codes = [c.strip() for c in (keep or KEEP_DEFAULT) if c.strip()]
    fetch_n = max(limit, pool if max_price else limit)
    if codes:
        fetch_n = max(fetch_n, 200)
    rows = fetch_top_turnover(ymd, fetch_n)
    if codes:
        wanted = [c.strip() for c in codes if c.strip()]
        for code in keep_codes:
            if code not in wanted:
                wanted = [code] + wanted
        by_code = {r["code"]: r for r in rows}
        picked: list[dict] = []
        for code in wanted:
            if code in by_code:
                row = by_code[code]
                row["pinned"] = code in keep_codes
                if not row.get("name") or row.get("name") == code:
                    row["name"] = KNOWN_NAME.get(code, row.get("name") or code)
                picked.append(row)
            else:
                picked.append(stub_row(code))
        return ymd, picked
    universe, dropped = select_universe(rows, limit, keep_codes, max_price)
    if dropped:
        preview = ", ".join(
            f"{r['code']} {r.get('close')}" for r in dropped[:12]
        )
        extra = " …" if len(dropped) > 12 else ""
        print(f"drop price>{max_price}: {preview}{extra}", flush=True)
    return ymd, universe


def scan_row(
    row: dict,
    *,
    range_: str,
    recent: int | None,
    **detect_kw: Any,
) -> tuple[list[RetestHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_hit": 0}
    try:
        df = fetch_symbol_5m(row, range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 250:
        meta["error"] = "too_few_bars"
        return [], meta
    df = add_mas(df)
    start = None if recent is None else max(0, len(df) - recent)
    idxs = detect_retests(df, start=start, **detect_kw)
    hits = [hit_from_row(df, i, row) for i in idxs]
    for hit in hits:
        hit._df = df  # type: ignore[attr-defined]
    meta["n_hit"] = len(hits)
    return hits, meta


def scan_universe(
    universe: list[dict],
    *,
    range_: str = "1mo",
    recent: int | None = 3,
    workers: int = 4,
    sleep: float = 0.05,
    **detect_kw: Any,
) -> tuple[list[RetestHit], list[dict]]:
    hits: list[RetestHit] = []
    metas: list[dict] = []

    def job(row: dict) -> tuple[list[RetestHit], dict]:
        time.sleep(sleep)
        return scan_row(row, range_=range_, recent=recent, **detect_kw)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(job, row): row for row in universe}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                stock_hits, meta = fut.result()
            except Exception as exc:  # noqa: BLE001
                stock_hits, meta = [], {**row, "bars": 0, "error": str(exc)[:80], "n_hit": 0}
            hits.extend(stock_hits)
            metas.append(meta)
    hits.sort(key=lambda h: pd.Timestamp(h.ts))
    return hits, metas


def detect_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        ma_n=args.ma,
        touch_pct=args.touch_pct,
        ticks=args.ticks,
        min_away_pct=args.min_away_pct,
        away_lookback=args.away_lookback,
        min_above=args.min_above,
        max_close_below_pct=args.max_close_below_pct,
        max_close_above_pct=args.max_close_above_pct,
        min_ma_slope_pct=args.min_ma_slope_pct,
        skip_open_minutes=args.skip_open_minutes,
        cooldown_bars=args.cooldown_bars,
    )


def notify(hit: RetestHit, *, dry_run: bool, with_chart: bool = True) -> bool:
    text = format_hit(hit)
    plain = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<code>", "")
        .replace("</code>", "")
    )
    print("\n" + plain, flush=True)
    photo = None
    df = getattr(hit, "_df", None)
    if with_chart and df is not None:
        tmp = Path("/tmp") / f"tw_ma240_{hit.code}_{hit.idx}.png"
        photo = draw_chart(df, hit, str(tmp))
    ok = telegram_send(text, photo=photo, dry_run=dry_run)
    if dry_run:
        return ok
    if ok:
        print("  → Telegram 已送", flush=True)
    elif not (env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID")):
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗", flush=True)
    return ok


def round_once(
    universe: list[dict],
    seen: set[str],
    args: argparse.Namespace,
    *,
    first_run: bool,
) -> set[str]:
    t0 = time.time()
    recent = None if args.scan else args.lookback_bars
    hits, metas = scan_universe(
        universe,
        range_=args.range,
        recent=recent,
        workers=args.workers,
        sleep=args.sleep,
        **detect_kwargs(args),
    )
    if args.scan and getattr(args, "days", 0):
        hits = filter_hits_days(hits, args.days)
    errors = sum(1 for m in metas if m.get("error"))
    new = [h for h in hits if h.key not in seen]
    now = datetime.now(TPE)
    print(
        f"[{now.strftime('%H:%M:%S')}] 掃完 {len(universe)} 用 {time.time()-t0:.1f}s "
        f"hits={len(hits)} new={len(new)} err={errors}",
        flush=True,
    )
    if first_run and not args.seed and not args.scan:
        for h in new:
            seen.add(h.key)
        print(f"init: 記住 {len(new)} 筆已出現的回測，不重發（--seed 可改成要推）", flush=True)
        save_seen(seen)
        return seen
    if args.scan:
        if getattr(args, "days", 0):
            hits = filter_hits_days(hits, args.days)
            dates = sorted(session_dates_back(args.days))
            by_day: dict[str, int] = {}
            for h in hits:
                key = str(hit_session_date(h))
                by_day[key] = by_day.get(key, 0) + 1
            pierce = sum(1 for h in hits if h.pierced)
            print(
                f"近 {args.days} 個交易日 {dates[0]}→{dates[-1]}  "
                f"訊號 {len(hits)} 筆、{len({h.code for h in hits})} 檔"
                f"（刺破收回 {pierce}、貼到均線 {len(hits) - pierce}）",
                flush=True,
            )
            if by_day:
                print(
                    "分日 " + "  ".join(f"{d[5:]} {by_day[d]}" for d in sorted(by_day)),
                    flush=True,
                )
        for h in hits:
            if h.code == "1815" or not (getattr(args, "pages", False) or getattr(args, "html", "")):
                print(format_hit_line(h), flush=True)
        html_path = None
        if getattr(args, "pages", False):
            html_path = PAGES
        elif getattr(args, "html", ""):
            html_path = Path(args.html)
        if html_path:
            period = f"近{args.days}個交易日" if args.days else str(args.range)
            write_html_report(
                html_path,
                hits,
                universe,
                period,
                getattr(args, "universe_date", "") or "",
                chart_mode=getattr(args, "chart_mode", "pierce") or "pierce",
                max_price=getattr(args, "max_price", 700) or None,
            )
            write_view_html(html_path)
            print(f"view={html_path.with_name('view.html')}", flush=True)
        return seen
    for h in new:
        if notify(h, dry_run=args.dry_run, with_chart=not args.dry_run):
            seen.add(h.key)
        elif args.dry_run:
            seen.add(h.key)
    if new:
        save_seen(seen)
    return seen


def cmd_test(dry_run: bool) -> int:
    load_dotenv()
    ok = telegram_send(
        f"✅ 台股五分K 回測240MA 測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
        dry_run=dry_run,
    )
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股成交額前200 五分K 回測240MA 跳通知")
    p.add_argument("--limit", type=int, default=200, help="成交額前 N（預設 200）")
    p.add_argument("--pool", type=int, default=400, help="先取成交額前 N 再套股價過濾")
    p.add_argument("--max-price", type=float, default=700, help="收盤價超過此值剔除（預設 700；0=不過濾）")
    p.add_argument("--date", default="", help="YYYYMMDD 成交額基準日，預設上一個交易日")
    p.add_argument("--codes", default="", help="只看這些代號，逗號分隔，例如 1815,2330")
    p.add_argument("--keep", default="1815", help="必抓代號，預設 1815 富喬")
    p.add_argument("--range", default="1mo", help="Yahoo 五分K 區間")
    p.add_argument("--ma", type=int, default=240)
    p.add_argument("--touch-pct", type=float, default=0.002, help="碰到 240MA 的百分比容忍（預設 0.20%%）")
    p.add_argument("--ticks", type=int, default=1, help="碰到 240MA 至少幾檔（與百分比取較寬）")
    p.add_argument("--min-away-pct", type=float, default=0.01, help="回測前至少拉開（預設 1.0%%）")
    p.add_argument("--away-lookback", type=int, default=24, help="回測前看幾根五分K（預設 24=2小時）")
    p.add_argument("--min-above", type=int, default=10)
    p.add_argument("--max-close-below-pct", type=float, default=0.002)
    p.add_argument("--max-close-above-pct", type=float, default=0.0045, help="未刺破時，收盤最多高於均線多少")
    p.add_argument("--min-ma-slope-pct", type=float, default=-0.15)
    p.add_argument("--skip-open-minutes", type=int, default=0, help="開盤前幾分鐘不計（預設 0，開盤回測也抓）")
    p.add_argument("--cooldown-bars", type=int, default=6, help="同一檔兩次回測至少間隔幾根五分K")
    p.add_argument("--lookback-bars", type=int, default=3, help="即時模式只看最近幾根已收K")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.08)
    p.add_argument("--once", action="store_true", help="只掃一輪")
    p.add_argument("--scan", action="store_true", help="掃完整段五分K（近一個月回測清單）")
    p.add_argument("--days", type=int, default=0, help="--scan 只列最近幾個交易日，例如 10=兩週")
    p.add_argument("--html", default="", help="寫 HTML 報告路徑")
    p.add_argument("--pages", action="store_true", help="寫到 docs/tw-5m-ma240/index.html")
    p.add_argument("--chart-mode", default="pierce", help="圖：pierce=富喬+刺破 / all / fuqiao / none")
    p.add_argument("--dry-run", action="store_true", help="只印、不推 Telegram")
    p.add_argument("--test", action="store_true", help="只測 Telegram")
    p.add_argument("--seed", action="store_true", help="第一次就把已出現的回測推出去")
    p.add_argument("--no-wait-session", action="store_true", help="盤外也掃")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test:
        return cmd_test(args.dry_run)
    has_tg = bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))
    if not has_tg and not args.dry_run:
        if args.scan or args.once:
            args.dry_run = True
            print("未設定 Telegram，改為 --dry-run", flush=True)
        else:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID（見 tg_config.env.example）", file=sys.stderr)
            print("或先加 --dry-run 只在終端機看訊號。", file=sys.stderr)
            return 2

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    keep = [c.strip() for c in args.keep.split(",") if c.strip()]
    max_price = args.max_price if args.max_price else None
    print(
        f"universe limit={args.limit} pool={args.pool} max_price={max_price} "
        f"range={args.range} ma={args.ma} "
        f"touch={args.touch_pct:.3%} away={args.min_away_pct:.3%} keep={','.join(keep) or '-'}",
        flush=True,
    )
    date, universe = load_universe(
        args.limit, args.date, codes, keep=keep, max_price=max_price, pool=args.pool
    )
    args.universe_date = date
    args.max_price = max_price
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"date={date} n={len(universe)} "
        f"{universe[0]['code']} {universe[0]['name']} "
        f"{universe[0]['amount']/1e8:.1f}億 → "
        f"{universe[-1]['code']} {universe[-1]['amount']/1e8:.1f}億",
        flush=True,
    )

    seen = load_seen()
    first = not bool(seen)
    if args.scan:
        args.seed = True
        round_once(universe, seen, args, first_run=False)
        return 0

    seen = round_once(universe, seen, args, first_run=first)
    if args.once:
        return 0

    print("watch 中，每根五分K收盤掃一次（Ctrl+C 停）", flush=True)
    uni_ts = time.time()
    try:
        while True:
            wait = seconds_until_next_scan()
            if not args.no_wait_session and not market_session() and wait > 120:
                nxt = datetime.now(TPE) + timedelta(seconds=wait)
                print(f"盤外，等到 {nxt.strftime('%m-%d %H:%M')} 再開盤", flush=True)
            time.sleep(min(wait, 300))
            if not args.no_wait_session and not market_session():
                continue
            if time.time() - uni_ts > 6 * 3600:
                date, universe = load_universe(
                    args.limit, args.date, codes, keep=keep, max_price=max_price, pool=args.pool
                )
                uni_ts = time.time()
                print(f"更新標的 {len(universe)} date={date}", flush=True)
            try:
                seen = round_once(universe, seen, args, first_run=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {exc}", file=sys.stderr)
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
