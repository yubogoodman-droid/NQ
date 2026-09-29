#!/usr/bin/env python3
"""台股 1h：MA5>MA10>MA20 多頭排列且收盤站上 MA60 → Telegram。

條件（小時 K 收盤）：
  • MA5 > MA10 > MA20
  • 收盤 > MA60
  • 上一根還沒同時滿足（剛形成才通知，避免每小時洗版）

成交額前 200、股價 < 1000（跟 1h 破底翻同一池）。
每根小時 K 收盤後掃一次；GitHub Actions 在盤中整點代跑。

    python3 examples/tw_1h_stack_ma60.py --test
    python3 examples/tw_1h_stack_ma60.py --dry-run --once
    python3 examples/tw_1h_stack_ma60.py --once
    python3 examples/tw_1h_stack_ma60.py              # 等到下一根 1h 收盤再掃
    python3 examples/tw_1h_stack_ma60.py --now        # 現在已站上的名單
    python3 examples/tw_1h_stack_ma60.py scan --days 7 --pages
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    REPO,
    TPE,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
)
from tw_1h_reclaim import (  # noqa: E402
    _session_close_indices,
    fetch_yahoo_1h,
    sma,
)

CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO / "output" / "tw_1h_stack_ma60_seen.json"
PAGES = REPO / "docs" / "tw-1h-stack-ma60" / "index.html"
PAGES_7D = REPO / "docs" / "tw-1h-stack-ma60-7d" / "index.html"
MA_COLORS = {5: "#f0c14b", 10: "#79c0ff", 20: "#f472b6", 60: "#e6edf3"}
SCAN_LAG = timedelta(minutes=2)
HOUR_CLOSES = ((10, 0), (11, 0), (12, 0), (13, 0), (13, 30))


@dataclass(frozen=True)
class Signal:
    idx: int
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float


@dataclass
class Hit:
    row: dict
    signal: Signal
    df: pd.DataFrame
    fwd_1d: Optional[float] = None
    fwd_3d: Optional[float] = None
    fwd_5d: Optional[float] = None


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def setup_at(
    i: int,
    close: np.ndarray,
    ma5: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
) -> bool:
    if i < 0 or i >= len(close):
        return False
    vals = (close[i], ma5[i], ma10[i], ma20[i], ma60[i])
    if np.isnan(vals).any():
        return False
    return bool(ma5[i] > ma10[i] > ma20[i] and close[i] > ma60[i])


def detect_signals(df: pd.DataFrame) -> List[Signal]:
    """剛形成：本根多頭排列且站上 MA60，上一根還沒同時滿足。"""
    if df is None or len(df) < 61:
        return []
    close = df["Close"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    out: List[Signal] = []
    for i in range(1, len(close)):
        if setup_at(i, close, ma5, ma10, ma20, ma60) and not setup_at(
            i - 1, close, ma5, ma10, ma20, ma60
        ):
            out.append(
                Signal(
                    idx=i,
                    close=float(close[i]),
                    ma5=float(ma5[i]),
                    ma10=float(ma10[i]),
                    ma20=float(ma20[i]),
                    ma60=float(ma60[i]),
                )
            )
    return out


def current_setup(df: pd.DataFrame) -> Optional[Signal]:
    if df is None or len(df) < 61:
        return None
    close = df["Close"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    i = len(close) - 1
    if not setup_at(i, close, ma5, ma10, ma20, ma60):
        return None
    return Signal(
        idx=i,
        close=float(close[i]),
        ma5=float(ma5[i]),
        ma10=float(ma10[i]),
        ma20=float(ma20[i]),
        ma60=float(ma60[i]),
    )


def bar_close_time(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.hour == 13 and ts.minute >= 30:
        return ts
    if ts.hour == 13:
        return ts + pd.Timedelta(minutes=30)
    return ts + pd.Timedelta(hours=1)


def drop_forming(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    now_ts = pd.Timestamp(now or datetime.now(TPE))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(TPE)
    else:
        now_ts = now_ts.tz_convert(TPE)
    last = df.index[-1]
    if now_ts < bar_close_time(last) + pd.Timedelta(seconds=5):
        return df.iloc[:-1].copy()
    return df


def filter_recent(df: pd.DataFrame, signals: Sequence[Signal], bars: int) -> List[Signal]:
    if bars <= 0 or not len(df):
        return list(signals)
    lo = len(df) - bars
    return [s for s in signals if s.idx >= lo]


def filter_entry_window(df: pd.DataFrame, signals: Sequence[Signal], days: int) -> List[Signal]:
    if not len(df) or days <= 0:
        return list(signals)
    start = df.index[-1] - pd.Timedelta(days=days)
    return [s for s in signals if df.index[s.idx] >= start]


def fill_fwd(hit: Hit) -> Hit:
    """訊號日之後第 N 個交易日收盤報酬，不含當日稍後收盤。"""
    df = hit.df
    close = df["Close"].to_numpy(float)
    entry_idx = hit.signal.idx
    entry_date = df.index[entry_idx].date()
    later_days = [i for i in _session_close_indices(df.index) if df.index[i].date() > entry_date]

    def _at(sessions: int) -> Optional[float]:
        if len(later_days) < sessions or close[entry_idx] == 0:
            return None
        nxt = later_days[sessions - 1]
        return float(close[nxt] / close[entry_idx] - 1.0)

    hit.fwd_1d = _at(1)
    hit.fwd_3d = _at(3)
    hit.fwd_5d = _at(5)
    return hit


def summarize_fwd(hits: Sequence[Hit]) -> dict:
    def _avg(attr: str) -> tuple[Optional[float], int, int]:
        xs = [getattr(h, attr) for h in hits if getattr(h, attr) is not None]
        if not xs:
            return None, 0, 0
        return float(sum(xs) / len(xs)), len(xs), sum(1 for x in xs if x > 0)

    avg1, n1, w1 = _avg("fwd_1d")
    avg3, n3, w3 = _avg("fwd_3d")
    avg5, n5, w5 = _avg("fwd_5d")
    return {
        "count": len(hits),
        "names": len({h.row["code"] for h in hits}),
        "fwd_1d": avg1,
        "fwd_1d_n": n1,
        "fwd_1d_wr": 100.0 * w1 / n1 if n1 else 0.0,
        "fwd_3d": avg3,
        "fwd_3d_n": n3,
        "fwd_3d_wr": 100.0 * w3 / n3 if n3 else 0.0,
        "fwd_5d": avg5,
        "fwd_5d_n": n5,
        "fwd_5d_wr": 100.0 * w5 / n5 if n5 else 0.0,
    }


# ---------------------------------------------------------------------------
# Telegram / state
# ---------------------------------------------------------------------------


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


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def load_seen() -> Set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("keys") or data.get("alerted") or []
        return set(data)
    except Exception:  # noqa: BLE001
        return set()


def save_seen(seen: Iterable[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(seen)[-800:]
    SEEN_PATH.write_text(
        json.dumps({"keys": keys, "updated": datetime.now(TPE).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def hit_key(hit: Hit) -> str:
    ts = hit.df.index[hit.signal.idx]
    return f"{hit.row['symbol']}|{pd.Timestamp(ts).isoformat()}|{hit.signal.close:.2f}"


def tg_send(
    token: str,
    chat_id: str,
    text: str,
    *,
    dry_run: bool = False,
    photo: Optional[str] = None,
) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if not token or not chat_id:
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
        return False
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    try:
        if photo and Path(photo).exists():
            with open(photo, "rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": text[:1024],
                        "parse_mode": "HTML",
                    },
                    files={"photo": f},
                    timeout=30,
                )
            if r.ok and (r.json() or {}).get("ok", True):
                return True
            print(f"[tg] photo HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        if not r.ok:
            print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return False
        data = r.json()
        if not data.get("ok"):
            print(f"[tg] API error: {data}", file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[tg] {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Chart / HTML
# ---------------------------------------------------------------------------


def _setup_cjk() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break


def draw_hit_png(hit: Hit, path: Path, pad_left: int = 50, pad_right: int = 2) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:  # noqa: BLE001
        return None

    _setup_cjk()
    df = hit.df
    sig = hit.signal
    start = max(0, sig.idx - pad_left)
    end = min(len(df) - 1, sig.idx + pad_right)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.8)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35 if n != 60 else 1.6, label=f"MA{n}")

    ex = sig.idx - start
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#00e676", ls="--", lw=0.9)
        ax.scatter([ex], [sig.close], s=46, color="#00e676", marker="^", zorder=6)
        ax.axhline(sig.ma60, color="#e6edf3", ls=":", lw=0.8, alpha=0.7)

    ts = df.index[sig.idx]
    label = f"{hit.row['code']} {hit.row.get('name', '')}".strip()
    ext = (sig.close / sig.ma60 - 1.0) * 100 if sig.ma60 else 0.0
    ax.set_title(
        f"{label}  {ts.strftime('%Y-%m-%d %H:%M')}  站上 MA60 {ext:+.2f}%",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fmt_hit(hit: Hit) -> str:
    sig = hit.signal
    ts = pd.Timestamp(hit.df.index[sig.idx]).tz_convert(TPE)
    ext = (sig.close / sig.ma60 - 1.0) * 100 if sig.ma60 else 0.0
    name = escape(str(hit.row.get("name") or ""))
    code = escape(str(hit.row.get("code") or ""))
    sym = escape(str(hit.row.get("symbol") or ""))
    return (
        f"🟢 <b>1h 多頭排列 · 站上 MA60</b>\n"
        f"<b>{code} {name}</b>  <code>{sym}</code>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} 台北</code>\n"
        f"收盤: <code>{sig.close:.2f}</code>  MA60 <code>{sig.ma60:.2f}</code> ({ext:+.2f}%)\n"
        f"MA5 <code>{sig.ma5:.2f}</code> &gt; "
        f"MA10 <code>{sig.ma10:.2f}</code> &gt; "
        f"MA20 <code>{sig.ma20:.2f}</code>\n"
        f"#台股 #1h #MA60"
    )


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            text=True,
        )
        return out.strip() or "main"
    except Exception:  # noqa: BLE001
        return "main"


def write_view_html(src: Path) -> Path:
    src = src.resolve()
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{_git_branch()}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def _fmt_fwd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


def _pnl_cls(value: Optional[float]) -> str:
    if value is None or value == 0:
        return "pnl-flat"
    return "pnl-win" if value > 0 else "pnl-loss"


def dump_hits_json(path: Path, hits: List[Hit], stats: dict, extra: dict) -> Path:
    rows = []
    for hit in hits:
        sig = hit.signal
        ts = hit.df.index[sig.idx]
        rows.append(
            {
                "code": hit.row["code"],
                "name": hit.row.get("name"),
                "symbol": hit.row.get("symbol"),
                "time": str(ts),
                "close": sig.close,
                "ma5": sig.ma5,
                "ma10": sig.ma10,
                "ma20": sig.ma20,
                "ma60": sig.ma60,
                "above_ma60_pct": (sig.close / sig.ma60 - 1.0) if sig.ma60 else None,
                "fwd_1d": hit.fwd_1d,
                "fwd_3d": hit.fwd_3d,
                "fwd_5d": hit.fwd_5d,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stats": stats, "extra": extra, "hits": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_html(path: Path, hits: List[Hit], universe: List[dict], period: str, date: str) -> Path:
    stats = summarize_fwd(hits)
    cards: List[str] = []
    for i, hit in enumerate(hits, 1):
        sig = hit.signal
        ts = hit.df.index[sig.idx]
        img_name = f"t{i:03d}_{hit.row['code']}_{ts.strftime('%m%d_%H%M')}.png"
        draw_hit_png(hit, path.parent / "img" / img_name)
        ext = (sig.close / sig.ma60 - 1.0) * 100 if sig.ma60 else 0.0
        label = f"{hit.row['code']} {hit.row.get('name', '')}".strip()
        show_fwd = hit.fwd_1d is not None
        headline = hit.fwd_1d if show_fwd else ext / 100.0
        headline_txt = _fmt_fwd(hit.fwd_1d) if show_fwd else f"{ext:+.2f}%"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(ts.strftime('%Y-%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {_pnl_cls(headline)}'>{headline_txt}</div>"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            f"<span class='tag'>+1d {_fmt_fwd(hit.fwd_1d)}</span>"
            f"<span class='tag'>+3d {_fmt_fwd(hit.fwd_3d)}</span>"
            f"<span class='tag'>+5d {_fmt_fwd(hit.fwd_5d)}</span></div>"
            "<pre class='trade-detail'>"
            f"close {sig.close:.2f}  站上 MA60 {ext:+.2f}%\n"
            f"MA5 {sig.ma5:.2f} > MA10 {sig.ma10:.2f} > MA20 {sig.ma20:.2f}  MA60 {sig.ma60:.2f}\n"
            f"fwd +1d {_fmt_fwd(hit.fwd_1d)}  +3d {_fmt_fwd(hit.fwd_3d)}  +5d {_fmt_fwd(hit.fwd_5d)}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    cutoff = universe[-1]["amount"] / 1e8 if universe else 0
    avg1 = _fmt_fwd(stats["fwd_1d"])
    avg3 = _fmt_fwd(stats["fwd_3d"])
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 1h 多頭排列 · 站上 MA60</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.55}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 1h · MA5&gt;MA10&gt;MA20 且站上 MA60</h1>
<p class="muted">{escape(period)} · 基準日 {escape(date)} · {len(universe)} 檔 · 成交額末名約 {cutoff:.1f} 億
<br/>小時 K 收盤同時滿足多頭排列與收盤 &gt; MA60，且上一根還沒同時滿足才算一筆（剛形成）。
卡片右上是訊號後下一個交易日收盤報酬（還沒走完就顯示站上 MA60 幅度）。</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">標的<b>{stats['names']}</b></div>
<div class="card">+1d 勝率<b>{stats['fwd_1d_wr']:.0f}%</b></div>
<div class="card">平均 +1d<b class="{_pnl_cls(stats['fwd_1d'])}">{avg1}</b></div>
<div class="card">平均 +3d<b class="{_pnl_cls(stats['fwd_3d'])}">{avg3}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有訊號</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Universe / scan
# ---------------------------------------------------------------------------


def load_universe(limit: int, pool: int, max_price: Optional[float], date: str = "") -> tuple[List[dict], str]:
    ymd = resolve_twse_date(date or last_tw_session_yyyymmdd())
    take = 0 if limit <= 0 else max(limit, pool if max_price else limit)
    raw = fetch_top_turnover(ymd, take)
    helper_cap = None if max_price is None or max_price <= 0 else float(max_price) - 1e-9
    universe, dropped = filter_by_max_price(raw, helper_cap, limit)
    if dropped:
        print(
            "drop price>="
            + str(max_price)
            + ": "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:8])
            + (" …" if len(dropped) > 8 else "")
        )
    return universe, ymd


def _scan_one(row: dict, range_: str, now: Optional[datetime]) -> tuple[pd.DataFrame, dict]:
    meta = {**row, "bars": 0, "error": ""}
    try:
        df = drop_forming(fetch_yahoo_1h(row["symbol"], range_), now)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return pd.DataFrame(), meta
    meta["bars"] = int(len(df))
    if len(df) < 61:
        meta["error"] = meta["error"] or "too_few_bars"
    return df, meta


def scan_universe(
    universe: Sequence[dict],
    *,
    range_: str = "2mo",
    workers: int = 4,
    now: Optional[datetime] = None,
) -> tuple[List[tuple[dict, pd.DataFrame]], int]:
    frames: List[tuple[dict, pd.DataFrame]] = []
    errors = 0
    total = len(universe)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_scan_one, row, range_, now): row for row in universe}
        for fut in as_completed(futs):
            row = futs[fut]
            done += 1
            try:
                df, meta = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{done:3d}/{total}] {row['symbol']} {row['name']} err {exc}", flush=True)
                continue
            flag = f" {meta['error']}" if meta["error"] else ""
            if meta["error"]:
                errors += 1
            print(f"[{done:3d}/{total}] {row['symbol']} {row['name']} bars={meta['bars']}{flag}", flush=True)
            if meta["error"] or df.empty:
                continue
            frames.append((row, df))
    frames.sort(key=lambda x: x[0].get("rank") or 0)
    return frames, errors


def collect_hits(
    frames: Sequence[tuple[dict, pd.DataFrame]],
    *,
    days: int = 0,
    lookback_bars: int = 0,
    now_only: bool = False,
) -> List[Hit]:
    hits: List[Hit] = []
    for row, df in frames:
        if now_only:
            sig = current_setup(df)
            if sig is None:
                continue
            hits.append(fill_fwd(Hit(row, sig, df)))
            continue
        sigs = detect_signals(df)
        if days:
            sigs = filter_entry_window(df, sigs, days)
        if lookback_bars:
            sigs = filter_recent(df, sigs, lookback_bars)
        for sig in sigs:
            hits.append(fill_fwd(Hit(row, sig, df)))
    hits.sort(key=lambda h: h.df.index[h.signal.idx])
    return hits


def notify_hits(
    hits: Sequence[Hit],
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    seen: Set[str],
    with_chart: bool = True,
) -> int:
    sent = 0
    for hit in hits:
        key = hit_key(hit)
        if key in seen:
            continue
        text = fmt_hit(hit)
        print("\n" + text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">").replace("<code>", "").replace("</code>", ""))
        photo = None
        if with_chart:
            tmp = Path("/tmp") / f"tw1h_{hit.row['code']}_{hit.signal.idx}.png"
            drawn = draw_hit_png(hit, tmp)
            photo = str(drawn) if drawn else None
        ok = tg_send(token, chat_id, text, dry_run=dry_run, photo=photo)
        if ok or dry_run:
            seen.add(key)
            sent += 1
            if ok and not dry_run:
                print("  → Telegram 已送")
        time.sleep(0.35)
    return sent


def session_scan_times(day) -> List[datetime]:
    return [
        datetime(day.year, day.month, day.day, hh, mm, tzinfo=TPE) + SCAN_LAG
        for hh, mm in HOUR_CLOSES
    ]


def next_scan_time(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(TPE)
    d = now.date()
    for _ in range(12):
        if d.weekday() < 5:
            for t in session_scan_times(d):
                if t > now:
                    return t
        d += timedelta(days=1)
    raise RuntimeError("no upcoming scan time")


def wait_next_close() -> None:
    nxt = next_scan_time()
    now = datetime.now(TPE)
    sec = max(1.0, (nxt - now).total_seconds())
    print(f"等到 {nxt.strftime('%Y-%m-%d %H:%M')} 台北（{sec/60:.1f} 分）", flush=True)
    time.sleep(sec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_hits(hits: Sequence[Hit], title: str) -> None:
    stats = summarize_fwd(hits)
    print(
        f"{title}: {stats['count']} 筆 / {stats['names']} 檔  "
        f"+1d {_fmt_fwd(stats['fwd_1d'])} WR={stats['fwd_1d_wr']:.0f}%  "
        f"+3d {_fmt_fwd(stats['fwd_3d'])}"
    )
    for i, hit in enumerate(hits, 1):
        sig = hit.signal
        ts = hit.df.index[sig.idx]
        ext = (sig.close / sig.ma60 - 1.0) * 100 if sig.ma60 else 0.0
        print(
            f"  [{i}] {hit.row['code']} {hit.row.get('name', '')} "
            f"{ts.strftime('%m-%d %H:%M')} close={sig.close:.2f} "
            f"MA60={sig.ma60:.2f} {ext:+.2f}%  "
            f"+1d {_fmt_fwd(hit.fwd_1d)} +3d {_fmt_fwd(hit.fwd_3d)}"
        )


def run_scan_round(
    args,
    *,
    now_only: bool,
    lookback_bars: int,
    notify: bool,
) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN", "") or ""
    chat_id = env("TELEGRAM_CHAT_ID", "") or ""
    dry_run = bool(args.dry_run)
    if notify and not dry_run and (not token or not chat_id):
        if env("GITHUB_ACTIONS"):
            print("GitHub Actions 未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，改印訊號", flush=True)
            dry_run = True
        else:
            print("未填 Telegram 憑證，改 --dry-run（見 tg_config.env.example）", flush=True)
            dry_run = True

    price_cap = None if args.max_price is None or args.max_price <= 0 else float(args.max_price)
    universe, date = load_universe(args.limit, args.pool, price_cap, args.date)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    print(
        f"universe date={date} n={len(universe)} range={args.range_} "
        f"lookback_bars={lookback_bars} now_only={now_only}",
        flush=True,
    )
    t0 = time.time()
    frames, errors = scan_universe(
        universe, range_=args.range_, workers=args.workers, now=datetime.now(TPE)
    )
    hits = collect_hits(
        frames,
        days=getattr(args, "days", 0) or 0,
        lookback_bars=0 if now_only else lookback_bars,
        now_only=now_only,
    )
    print(
        f"scanned={len(frames)} errors={errors} hits={len(hits)} "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    title = "現在站上" if now_only else "新形成"
    print_hits(hits, title)

    if notify:
        seen = load_seen()
        sent = notify_hits(hits, token, chat_id, dry_run=dry_run, seen=seen, with_chart=not args.no_chart)
        save_seen(seen)
        print(f"notified={sent} dry_run={dry_run}", flush=True)

    html_path = Path(args.html).resolve() if getattr(args, "html", "") else None
    days = getattr(args, "days", 0) or 0
    if html_path is None and getattr(args, "pages", False):
        html_path = PAGES_7D if days == 7 else PAGES
    if html_path:
        period = f"{'now' if now_only else (str(days or 'live') + 'd')} · Yahoo {args.range_} 1h"
        if args.max_price:
            period += f" · 股價<{args.max_price:g}"
        write_html(html_path, hits, universe, period, date)
        write_view_html(html_path)
        extra = {
            "date": date,
            "days": days,
            "range": args.range_,
            "limit": args.limit,
            "max_price": args.max_price,
            "generated": datetime.now(TPE).isoformat(timespec="seconds"),
        }
        dump_hits_json(html_path.with_name("hits.json"), hits, summarize_fwd(hits), extra)
        print(f"html={html_path}")
        print(f"view={html_path.with_name('view.html')}")
    return 0


def cmd_alert(args) -> int:
    load_dotenv()
    if args.test:
        token = env("TELEGRAM_BOT_TOKEN", "") or ""
        chat_id = env("TELEGRAM_CHAT_ID", "") or ""
        ok = tg_send(
            token,
            chat_id,
            f"✅ 台股 1h 多頭排列站上 MA60 測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
            dry_run=args.dry_run,
        )
        print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
        return 0 if ok else 1

    lookback = args.lookback_bars
    print(
        f"台股 1h MA5>MA10>MA20 站上 MA60 | once={args.once} dry_run={args.dry_run} "
        f"lookback={lookback} bars",
        flush=True,
    )
    while True:
        try:
            run_scan_round(args, now_only=False, lookback_bars=lookback, notify=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        wait_next_close()
    return 0


def cmd_now(args) -> int:
    return run_scan_round(args, now_only=True, lookback_bars=0, notify=args.notify)


def cmd_scan(args) -> int:
    return run_scan_round(
        args,
        now_only=False,
        lookback_bars=0,
        notify=False,
    )


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--date", default="", help="YYYYMMDD，成交額排名基準日")
    p.add_argument("--limit", type=int, default=200, help="成交額前 N；0 = 不限")
    p.add_argument("--pool", type=int, default=400, help="先取成交額前 N 再套股價過濾")
    p.add_argument("--max-price", type=float, default=1000, help="股價達此值以上剔除；0 不過濾")
    p.add_argument("--range", dest="range_", default="2mo", help="Yahoo 1h 下載區間")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-chart", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股 1h 多頭排列站上 MA60 Telegram")
    add_common(p)
    p.add_argument("--once", action="store_true", help="只掃一次（剛收盤的 1～2 根）")
    p.add_argument("--test", action="store_true", help="只測 Telegram")
    p.add_argument("--now", action="store_true", help="列出目前已站上的標的")
    p.add_argument("--notify", action="store_true", help="--now 時也推播")
    p.add_argument("--lookback-bars", type=int, default=2, help="alert 只看最近 N 根已收盤 1h")
    p.add_argument("--days", type=int, default=0, help="scan 子命令：進場落在最近 N 日")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("scan", help="回看最近 N 日剛形成的訊號（不推播）")
    add_common(s)
    s.add_argument("--days", type=int, default=14)
    s.set_defaults(func=cmd_scan)

    n = sub.add_parser("now", help="目前已多頭排列且站上 MA60 的名單")
    add_common(n)
    n.add_argument("--notify", action="store_true")
    n.set_defaults(func=cmd_now)

    a = sub.add_parser("alert", help="Telegram 監看")
    add_common(a)
    a.add_argument("--once", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--lookback-bars", type=int, default=2)
    a.set_defaults(func=cmd_alert)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None):
        return args.func(args)
    if args.now:
        return cmd_now(args)
    return cmd_alert(args)


if __name__ == "__main__":
    raise SystemExit(main())
