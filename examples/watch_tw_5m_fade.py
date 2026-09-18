#!/usr/bin/env python3
"""台股 5 分 K：5/10/20 空頭排列且收盤剛跌破 MA240 就推 Telegram。

多方「站上 MA240」的鏡像。對齊券商圖 5/10/20/60/120/240：MA5 < MA10 < MA20
且三條下彎，當根收盤從 MA240 上跌到下方，收盤也低於 5/10/20。開盤第一根也算。
陽明 2609 2026-08-26 09:05 那種圖。

用法:
  python3 examples/watch_tw_5m_fade.py scan --symbols 2609 --range 7d --pages
  python3 examples/watch_tw_5m_fade.py scan --limit 80 --range 7d --pages
  python3 examples/watch_tw_5m_fade.py alert --test
  python3 examples/watch_tw_5m_fade.py alert --dry-run --once
  python3 examples/watch_tw_5m_fade.py alert

Telegram 憑證放 tg_config.env（勿提交）:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    TPE,
    UA,
    _chart_payload_to_df,
    _get_json,
    fetch_top_turnover,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "tw-5m-fade" / "index.html"
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
SEEN_PATH = REPO / "output" / "tw_5m_fade_seen.json"
STATE_PATH = Path(__file__).resolve().parent / "tw_5m_fade_state.json"
BRANCH = "cursor/tw-5m-fade-short-9faf"

# 截圖同款均線色：5 藍、10 綠、20 橘、60 青、120 紫、240 粉
MA_PERIODS = (5, 10, 20, 60, 120, 240)
MA_COLORS = {
    5: "#3b82f6",
    10: "#22c55e",
    20: "#f59e0b",
    60: "#14b8a6",
    120: "#a855f7",
    240: "#f472b6",
}


@dataclass
class FadeSignal:
    break_idx: int
    entry_idx: int
    entry_price: float
    break_high: float
    ma240: float
    prev_close: float
    dist_pct: float
    ma5: float
    ma10: float
    ma20: float
    volume_ratio: float


@dataclass
class FadeTrade:
    signal: FadeSignal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_pct: float
    exit_reason: str


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def drop_incomplete_5m(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """丟掉非整點 5 分 K，以及還沒收盤的當根。"""
    if df is None or df.empty:
        return df
    idx = df.index
    aligned = (idx.minute % 5 == 0) & (idx.second == 0)
    out = df.loc[aligned].copy()
    if out.empty:
        return out
    cur = now or datetime.now(TPE)
    last = out.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize(TPE)
    if cur.tzinfo is None:
        cur = cur.replace(tzinfo=TPE)
    if cur < last.tz_convert(cur.tzinfo) + pd.Timedelta(minutes=5):
        out = out.iloc[:-1]
    return out


def fetch_yahoo_5m(symbol: str, range_: str = "5d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=5m&range={range_}&includePrePost=false"
    )
    return drop_incomplete_5m(_chart_payload_to_df(_get_json(url)))


def parse_symbols(text: str) -> list[dict]:
    rows: list[dict] = []
    for i, raw in enumerate(text.split(","), 1):
        token = raw.strip().upper()
        if not token:
            continue
        if token.endswith(".TW") or token.endswith(".TWO"):
            code, market = token.split(".", 1)
            mkt = "tse" if market == "TW" else "otc"
            symbol = token
        else:
            code = token
            mkt = "tse"
            symbol = yahoo_symbol(code, "tse")
        rows.append(
            {
                "rank": i,
                "code": code,
                "name": "",
                "market": mkt,
                "amount": 0,
                "close": None,
                "symbol": symbol,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def sma(arr, n: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).rolling(n, min_periods=n).mean().to_numpy(float)


def _finite(*vals: float) -> bool:
    return all(v is not None and not np.isnan(v) for v in vals)


def ribbon_down(
    ma5: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    i: int,
    *,
    require_falling: bool = True,
) -> bool:
    """5/10/20 空頭排列；預設三條都要比前一根低。"""
    if i < 1 or not _finite(ma5[i], ma10[i], ma20[i], ma5[i - 1], ma10[i - 1], ma20[i - 1]):
        return False
    if not (ma5[i] < ma10[i] < ma20[i]):
        return False
    if require_falling and not (ma5[i] < ma5[i - 1] and ma10[i] < ma10[i - 1] and ma20[i] < ma20[i - 1]):
        return False
    return True


def detect_signals(
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    skip_before: tuple[int, int] | None = None,
    require_pretty: bool = True,
) -> list[FadeSignal]:
    """5/10/20 空排（下彎）且收盤剛跌破 MA240、收也低於 5/10/20。開盤第一根也算。"""
    if df is None or len(df) < 241:
        return []
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.zeros(len(df))
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma240 = sma(close, 240)
    n = len(close)
    signals: list[FadeSignal] = []

    for i in range(240, n):
        ts = df.index[i]
        if skip_before and (ts.hour, ts.minute) < skip_before:
            continue
        if not _finite(ma5[i], ma10[i], ma20[i], ma240[i], ma240[i - 1], close[i], close[i - 1]):
            continue
        stacked = ribbon_down(ma5, ma10, ma20, i, require_falling=require_pretty)
        crossed = float(close[i]) < float(ma240[i]) and float(close[i - 1]) >= float(ma240[i - 1])
        below_all = (
            float(close[i]) < float(ma5[i])
            and float(close[i]) < float(ma10[i])
            and float(close[i]) < float(ma20[i])
            and float(close[i]) < float(ma240[i])
        )
        if not (stacked and crossed and below_all):
            continue
        vol_avg = float(np.mean(volume[max(0, i - vol_lookback) : i]) or 0.0)
        vol_ratio = float(volume[i] / vol_avg) if vol_avg > 0 else 0.0
        m240 = float(ma240[i])
        px = float(close[i])
        signals.append(
            FadeSignal(
                break_idx=i,
                entry_idx=i,
                entry_price=px,
                break_high=float(high[i]),
                ma240=m240,
                prev_close=float(close[i - 1]),
                dist_pct=(m240 - px) / m240 if m240 else 0.0,
                ma5=float(ma5[i]),
                ma10=float(ma10[i]),
                ma20=float(ma20[i]),
                volume_ratio=vol_ratio,
            )
        )
    return signals


def simulate(df: pd.DataFrame, sigs: Sequence[FadeSignal]) -> list[FadeTrade]:
    if df.empty or not sigs:
        return []
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    idx = df.index
    trades: list[FadeTrade] = []
    for sig in sigs:
        entry = sig.entry_price
        stop = max(sig.break_high, sig.ma240) * 1.003
        risk = stop - entry
        if risk <= 0:
            continue
        target = entry - 2.0 * risk
        exit_idx = sig.entry_idx
        exit_px = entry
        reason = "eod"
        for k in range(sig.entry_idx + 1, len(df)):
            if float(high[k]) >= stop:
                exit_idx, exit_px, reason = k, stop, "stop"
                break
            if float(low[k]) <= target:
                exit_idx, exit_px, reason = k, target, "target"
                break
            last_of_day = k == len(df) - 1 or idx[k].date() != idx[k + 1].date()
            if last_of_day:
                exit_idx, exit_px, reason = k, float(close[k]), "eod"
                break
        trades.append(
            FadeTrade(
                signal=sig,
                entry_idx=sig.entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_px,
                stop_price=stop,
                target_price=target,
                pnl_pct=(entry - exit_px) / entry,
                exit_reason=reason,
            )
        )
    return trades


def summarize_trades(trades: Sequence[FadeTrade]) -> dict:
    pnls = [t.pnl_pct for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_pct": float(sum(pnls) * 100.0),
    }


# ---------------------------------------------------------------------------
# Chart / HTML
# ---------------------------------------------------------------------------


def _use_cjk_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break


def draw_signal_png(
    df: pd.DataFrame,
    sig: FadeSignal,
    path: Path,
    title: str,
    trade: FadeTrade | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _use_cjk_font()
    end = trade.exit_idx if trade is not None else sig.entry_idx
    start = max(0, sig.break_idx - 36)
    stop = min(len(df) - 1, end + 16)
    window = df.iloc[start : stop + 1]
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
        facecolor="#0c1014",
    )
    for a in (ax, axv):
        a.set_facecolor("#10141a")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#ef4444" if up else "#22c55e"  # 台股：紅漲綠跌
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#ef444499" if up else "#22c55e99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : stop + 1]
        ax.plot(list(xs), ma, color=col, lw=1.45 if n <= 20 else 1.05, label=f"{n}MA")

    bx, ex = sig.break_idx - start, sig.entry_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.ma240], s=42, color="#f472b6", zorder=6)
        ax.annotate(
            f"破MA240 {sig.ma240:.1f}",
            (bx, sig.ma240),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#f9a8d4",
            fontsize=8,
        )
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#4ade80", ls="--", lw=0.9)
        ax.scatter([ex], [sig.entry_price], s=44, color="#4ade80", marker="v", zorder=6)

    ts = df.index[sig.entry_idx]
    ax.set_title(
        f"{title}  {ts.strftime('%m-%d %H:%M')}  "
        f"破 MA240 {sig.ma240:.1f} → {sig.entry_price:.1f}  5<10<20",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_html_report(
    path: Path,
    hits: list[tuple[dict, FadeSignal, FadeTrade | None, pd.DataFrame]],
    universe: list[dict],
    period: str,
) -> Path:
    stats = summarize_trades([h[2] for h in hits if h[2] is not None])
    cards = []
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        et = df.index[sig.entry_idx]
        bt = df.index[sig.break_idx]
        label = f"{row['code']} {row.get('name') or ''}".strip()
        img_name = f"t{i:02d}_{row['code']}_{et.strftime('%m%d_%H%M')}.png"
        draw_signal_png(df, sig, path.parent / "img" / img_name, label, trade=trade)
        pnl = ""
        if trade is not None:
            cls = "pnl-win" if trade.pnl_pct > 0 else ("pnl-flat" if trade.pnl_pct == 0 else "pnl-loss")
            pnl = f"<div class='card-pnl {cls}'>{trade.pnl_pct*100:+.2f}%</div>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))}</span></div>"
            f"{pnl}"
            "</header>"
            f"<div class='tags'><span class='tag tag-info'>{escape(row['symbol'])}</span>"
            f"<span class='tag'>5分K</span><span class='tag'>5&lt;10&lt;20</span>"
            f"<span class='tag'>破MA240</span></div>"
            "<pre class='trade-detail'>"
            f"進場 {sig.entry_price:.2f}  破 MA240 {sig.ma240:.2f} @ {bt.strftime('%H:%M')}\n"
            f"前收 {sig.prev_close:.2f}  距年線 {sig.dist_pct*100:.2f}%\n"
            f"MA5 {sig.ma5:.2f}  MA10 {sig.ma10:.2f}  MA20 {sig.ma20:.2f}"
            f"  MA240 {sig.ma240:.2f}"
            "</pre>"
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股 5分K 空頭排列跌破 MA240</title>
<style>
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#ef4444}} .pnl-loss{{color:#22c55e}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 5分K 空頭排列跌破 MA240</h1>
<p class="muted">{escape(period)} · {len(universe)} 檔
<br/>5MA &lt; 10MA &lt; 20MA 且三條下彎，當根收盤剛跌破 MA240，收盤也低於 5/10/20。開盤第一根也算。</p>
<div class="cards">
<div class="card">筆數<b>{len(hits)}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總報酬<b class="{'pnl-win' if stats['total_pct']>=0 else 'pnl-loss'}">{stats['total_pct']:+.2f}%</b></div>
<div class="card">標的<b>{len({h[0]['code'] for h in hits})}</b></div>
</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有 5/10/20 空排跌破 MA240</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path, branch: str = BRANCH) -> Path:
    rel = src.parent.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{branch}/{rel}/"
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Telegram
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


def tg_send(token: str, chat_id: str, text: str, photo: Path | None = None, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    if photo is not None and photo.exists():
        with photo.open("rb") as fh:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                files={"photo": fh},
                timeout=30,
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
        headers={"User-Agent": UA},
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    return bool(r.json().get("ok"))


def load_state() -> dict[str, Any]:
    path = STATE_PATH if STATE_PATH.exists() else SEEN_PATH
    if not path.exists():
        return {"alerted": [], "initialized": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted": [], "initialized": False}
    if isinstance(raw, list):
        return {"alerted": raw, "initialized": True}
    return raw


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(state, ensure_ascii=False, indent=2)
    STATE_PATH.write_text(text, encoding="utf-8")
    SEEN_PATH.write_text(text, encoding="utf-8")


def signal_key(row: dict, df: pd.DataFrame, sig: FadeSignal) -> str:
    ts = df.index[sig.entry_idx]
    return f"{row['symbol']}|{ts.isoformat()}|{sig.entry_price:.2f}"


def fmt_alert(row: dict, df: pd.DataFrame, sig: FadeSignal) -> str:
    et = df.index[sig.entry_idx]
    bt = df.index[sig.break_idx]
    last = float(df["Close"].iloc[-1])
    name = row.get("name") or row["code"]
    return (
        f"🟢 <b>5/10/20 空頭排列跌破 MA240</b>\n"
        f"{escape(str(name))} <code>{escape(row['code'])}</code>\n"
        f"時間: <code>{et.strftime('%Y-%m-%d %H:%M')} 台北</code>\n"
        f"現價: <code>{sig.entry_price:.2f}</code>（最新 {last:.2f}）\n"
        f"破線: <code>{bt.strftime('%H:%M')}</code> MA240={sig.ma240:.2f}\n"
        f"前收 {sig.prev_close:.2f} → 收 {sig.entry_price:.2f}（距年線 {sig.dist_pct*100:.2f}%）\n"
        f"MA5 {sig.ma5:.2f} &lt; MA10 {sig.ma10:.2f} &lt; MA20 {sig.ma20:.2f}\n"
        f"#台股 #五分K #空頭排列 #MA240 #{row['code']}"
    )


def in_tw_session(now: datetime | None = None, pad_min: int = 8) -> bool:
    cur = now or datetime.now(TPE)
    if cur.weekday() >= 5:
        return False
    minutes = cur.hour * 60 + cur.minute
    return (9 * 60) <= minutes <= (13 * 60 + 30 + pad_min)


def wait_next_5m_close() -> None:
    now = datetime.now(TPE)
    elapsed = now.minute % 5
    extra = 12 - now.second
    wait = (5 - elapsed) * 60 + extra
    if wait < 5:
        wait += 5 * 60
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Scan / alert loops
# ---------------------------------------------------------------------------


def merge_universe(base: list[dict], extra: list[dict]) -> list[dict]:
    seen = {r["code"] for r in base}
    out = list(base)
    for row in extra:
        if row["code"] in seen:
            continue
        seen.add(row["code"])
        out.append(row)
    return out


def hit_on_day(df: pd.DataFrame, sig: FadeSignal, day) -> bool:
    return df.index[sig.entry_idx].date() == day


def hit_prices(row: dict, sig: FadeSignal, df: pd.DataFrame) -> list[float]:
    out: list[float] = [float(sig.entry_price), float(sig.break_high), float(sig.ma240)]
    if row.get("close") is not None:
        out.append(float(row["close"]))
    if df is None or not len(df):
        return out
    out.append(float(df["Close"].iloc[-1]))
    ts = df.index[sig.entry_idx]
    same_day = df.index.normalize() == ts.normalize()
    if same_day.any():
        out.append(float(df.loc[same_day, "High"].max()))
    return out


def hit_within_max_price(row: dict, sig: FadeSignal, df: pd.DataFrame, max_price: float | None) -> bool:
    if max_price is None:
        return True
    return all(px <= max_price for px in hit_prices(row, sig, df))


def resolve_on_day(args) -> object | None:
    if getattr(args, "today", False):
        return datetime.now(TPE).date()
    text = getattr(args, "on", "") or ""
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def resolve_universe(args) -> list[dict]:
    extra = parse_symbols(getattr(args, "also", "") or "")
    if getattr(args, "symbols", ""):
        return merge_universe(parse_symbols(args.symbols), extra)
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    pool = max(args.limit, args.pool if args.max_price else args.limit)
    print(f"universe date={date} limit={args.limit} pool={pool} max_price={args.max_price}")
    raw = fetch_top_turnover(date, pool)
    universe, dropped = filter_by_max_price(raw, args.max_price, args.limit)
    if dropped:
        print(
            "drop price>"
            + str(args.max_price)
            + ": "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:10])
            + (" …" if len(dropped) > 10 else "")
        )
    if universe:
        print(
            f"keep {len(universe)}  {universe[0]['code']} {universe[0]['name']} "
            f"{universe[0]['amount']/1e8:.1f}億 / {universe[0]['close']}"
        )
    return merge_universe(universe, extra)


def scan_symbol(
    row: dict,
    range_: str,
    *,
    require_pretty: bool = True,
) -> tuple[list[tuple[FadeSignal, pd.DataFrame]], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0}
    try:
        df = fetch_yahoo_5m(row["symbol"], range_)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 241:
        meta["error"] = "too_few_bars"
        return [], meta
    if row.get("close") is None and len(df):
        row["close"] = float(df["Close"].iloc[-1])
    sigs = detect_signals(df, require_pretty=require_pretty)
    meta["n_sig"] = len(sigs)
    return [(s, df) for s in sigs], meta


def cmd_scan(args) -> int:
    universe = resolve_universe(args)
    if not universe:
        print("no universe", file=sys.stderr)
        return 1
    hits: list[tuple[dict, FadeSignal, FadeTrade | None, pd.DataFrame]] = []
    errors = 0
    on_day = resolve_on_day(args)
    if on_day is not None:
        print(f"filter day={on_day}")
    pretty = not getattr(args, "loose", False)
    for i, row in enumerate(universe, 1):
        pairs, meta = scan_symbol(row, args.range_, require_pretty=pretty)
        if meta["error"]:
            errors += 1
        trades_by_entry = {}
        if pairs:
            df0 = pairs[0][1]
            for t in simulate(df0, [s for s, _ in pairs]):
                trades_by_entry[t.entry_idx] = t
        for sig, df in pairs:
            if on_day is not None and not hit_on_day(df, sig, on_day):
                continue
            if not hit_within_max_price(row, sig, df, getattr(args, "max_price", None)):
                continue
            hits.append((row, sig, trades_by_entry.get(sig.entry_idx), df))
        flag = f" sigs={meta['n_sig']}" if meta["n_sig"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row.get('name','')} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, args.sleep))

    hits.sort(key=lambda h: h[3].index[h[1].entry_idx])
    stats = summarize_trades([h[2] for h in hits if h[2] is not None])
    print(
        f"done errors={errors} signals={len(hits)} "
        f"WR={stats['win_rate']:.1f}% pnl={stats['total_pct']:+.2f}%"
    )
    for i, (row, sig, trade, df) in enumerate(hits, 1):
        ts = df.index[sig.entry_idx]
        extra = f" {trade.exit_reason} {trade.pnl_pct*100:+.2f}%" if trade else ""
        print(
            f"  [{i}] {row['code']} {row.get('name','')} {ts.strftime('%m-%d %H:%M')} "
            f"MA240 {sig.ma240:.2f} dist {sig.dist_pct*100:.2f}%{extra}"
        )

    html_path = Path(args.html) if args.html else (PAGES if args.pages else None)
    if html_path:
        period = args.range_
        on_day = resolve_on_day(args)
        if on_day is not None:
            period = f"{on_day.isoformat()} · {args.range_}資料"
        if args.max_price is not None:
            period += f" · 股價≤{args.max_price:g}"
        if pretty:
            period += " · 均線下彎"
        out = write_html_report(html_path, hits, universe, period)
        write_view_html(out)
        print(f"html={out}")
    return 0


def scan_once(
    universe: list[dict],
    token: str,
    chat_id: str,
    *,
    range_: str,
    dry_run: bool,
    seed_alert: bool,
    sleep_s: float,
    require_pretty: bool = True,
) -> None:
    state = load_state()
    alerted = set(state.get("alerted") or [])
    first_run = not state.get("initialized")
    new_items: list[tuple[str, dict, FadeSignal, pd.DataFrame]] = []
    for row in universe:
        pairs, meta = scan_symbol(row, range_, require_pretty=require_pretty)
        if meta["error"]:
            print(f"  skip {row['symbol']} {meta['error']}", file=sys.stderr)
        for sig, df in pairs:
            key = signal_key(row, df, sig)
            if key in alerted:
                continue
            new_items.append((key, row, sig, df))
        time.sleep(max(0.05, sleep_s))

    now = datetime.now(TPE)
    if first_run and not seed_alert:
        for key, _, _, _ in new_items:
            alerted.add(key)
        state["alerted"] = sorted(alerted)[-400:]
        state["initialized"] = True
        state["last_scan"] = now.isoformat()
        save_state(state)
        print(f"[{now.strftime('%H:%M:%S')}] init: marked {len(new_items)} recent signals")
        return

    sent = 0
    for key, row, sig, df in new_items:
        tmp = Path("/tmp") / f"tw5m_fade_{row['code']}_{sig.entry_idx}.png"
        try:
            draw_signal_png(df, sig, tmp, f"{row['code']} {row.get('name') or ''}")
        except Exception as exc:  # noqa: BLE001
            print(f"[chart] {exc}", file=sys.stderr)
            tmp = None
        ok = tg_send(token, chat_id, fmt_alert(row, df, sig), photo=tmp, dry_run=dry_run)
        if ok:
            alerted.add(key)
            sent += 1
            ts = df.index[sig.entry_idx]
            print(f"[alert] {row['code']} {ts} @ {sig.entry_price:.2f}")
    state["alerted"] = sorted(alerted)[-400:]
    state["initialized"] = True
    state["last_scan"] = now.isoformat()
    save_state(state)
    print(f"[{now.strftime('%H:%M:%S')}] scan ok new_sent={sent} pending={len(new_items)}")


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN") or ""
    chat_id = env("TELEGRAM_CHAT_ID") or ""
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2
    if args.test:
        ok = tg_send(
            token,
            chat_id,
            f"✅ 台股 5分K 空排跌破 MA240 bot 測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    universe = resolve_universe(args)
    if not universe:
        return 1
    print(
        f"TW 5m fade TG | n={len(universe)} | dry_run={args.dry_run} | "
        f"range={args.range_} | pretty={not args.loose} | session_only={not args.all_hours}"
    )
    while True:
        try:
            if args.all_hours or in_tw_session():
                scan_once(
                    universe,
                    token,
                    chat_id,
                    range_=args.range_,
                    dry_run=args.dry_run,
                    seed_alert=args.seed_alert,
                    sleep_s=args.sleep,
                    require_pretty=not args.loose,
                )
            else:
                print(f"[{datetime.now(TPE).strftime('%H:%M:%S')}] outside session, skip")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        wait_next_5m_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股 5分K 空頭排列跌破 MA240 通知")
    sub = p.add_subparsers(dest="cmd")

    def add_universe(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--date", default="", help="成交額基準日 YYYYMMDD")
        sp.add_argument("--limit", type=int, default=80)
        sp.add_argument("--pool", type=int, default=160)
        sp.add_argument("--max-price", type=float, default=None)
        sp.add_argument("--symbols", default="", help="逗號分隔代號，例如 2330,2303")
        sp.add_argument("--also", default="", help="額外併入掃描的代號")
        sp.add_argument("--range", dest="range_", default="7d")
        sp.add_argument("--sleep", type=float, default=0.2)
        sp.add_argument(
            "--loose",
            action="store_true",
            help="不要求均線下彎，只要 5<10<20 且跌破 MA240",
        )

    s = sub.add_parser("scan", help="回看近幾日並可出 HTML")
    add_universe(s)
    s.add_argument("--today", action="store_true", help="只留台北今天的訊號")
    s.add_argument("--on", default="", help="只留這一天 YYYY-MM-DD")
    s.add_argument("--pages", action="store_true")
    s.add_argument("--html", default="")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("alert", help="Telegram 輪詢（每根 5 分 K 收盤掃一次）")
    add_universe(a)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--seed-alert", action="store_true", help="第一次也把近期訊號推出去")
    a.add_argument("--all-hours", action="store_true")
    a.set_defaults(func=cmd_alert)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
