#!/usr/bin/env python3
"""幣安 1 分 K：7>14>25 黏在 MA200，剛站上（99/120 可以還在上面）。

    python3 examples/binance_1m_bull.py backtest --top 10 --today --pages
    python3 examples/binance_1m_bull.py alert --test
    python3 examples/binance_1m_bull.py alert --once --dry-run
    python3 examples/binance_1m_bull.py alert
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.binance import SESSION, fetch_klines, universe
from nq.ma1m_bull import (
    FIVE_MIN_MS,
    HORIZONS,
    SignalRow,
    add_mas,
    bar_index_at,
    detect_combo,
    forward_moves,
    resample_ohlcv,
    sma,
    summarize_rows,
)

TZ = timezone(timedelta(hours=8))
REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = REPO / "output" / "binance_1m_bull_seen.json"
PAGES = REPO / "docs" / "binance" / "ma1m-bull.html"
PUBLIC = "https://yubogoodman-droid.github.io/NQ/binance/ma1m-bull.html"
PAGES_IMG = "https://raw.githubusercontent.com/yubogoodman-droid/NQ/gh-pages/binance/"
IMG_VER = "kiss0155"
CIRCLE = "#F6465D"
# 幣安 App 淺色盤：黃/橘/紫/藍/青 + 深灰 MA200
PAL = {7: "#F0B90B", 14: "#FF6D00", 25: "#D500F9", 99: "#2962FF", 120: "#00B8D4", 200: "#474D57"}
VOL_MA = {5: "#F0B90B", 10: "#D500F9"}
UP = "#0ECB81"
DOWN = "#F6465D"
BG = "#FFFFFF"
GRID = "#EAECEF"
TEXT = "#1E2329"
MUTED = "#707A8A"
LABELS = {5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h"}


def apply_keys() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return
    for folder in (REPO, Path(__file__).resolve().parent, Path.cwd()):
        env = folder / "tg_config.env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        local = folder / "telegram_local.py"
        if local.is_file():
            ns: dict = {}
            exec(local.read_text(encoding="utf-8"), ns)
            tok = str(ns.get("TELEGRAM_BOT_TOKEN", "")).strip()
            cid = str(ns.get("TELEGRAM_CHAT_ID", "")).strip()
            if tok:
                os.environ.setdefault("TELEGRAM_BOT_TOKEN", tok)
            if cid:
                os.environ.setdefault("TELEGRAM_CHAT_ID", cid)


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or f"s{abs(hash(symbol)) % 10_000_000_000}"


def img_src(rel: str) -> str:
    """Pages 絕對網址，避免相對路徑在 preview / 舊頁被清掉後圖不見。"""
    return f"{PAGES_IMG}{rel}?v={IMG_VER}"


def default_date(now: datetime | None = None) -> str:
    """台北日。凌晨 2 點前改用前一日（才有完整一天可回測）。"""
    cur = now or datetime.now(TZ)
    day = cur.date()
    if cur.hour < 2:
        day -= timedelta(days=1)
    return day.isoformat()


def day_window_ms(date: str, days: int = 1) -> tuple[int, int]:
    end = datetime.fromisoformat(date).replace(tzinfo=TZ) + timedelta(days=1)
    start = end - timedelta(days=max(1, int(days)))
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def window_label(date: str, days: int = 1) -> str:
    """台北日區間，例如 2026-07-27 → 2026-08-25。"""
    days = max(1, int(days))
    end = datetime.fromisoformat(date).date()
    start = end - timedelta(days=days - 1)
    if days == 1:
        return end.isoformat()
    return f"{start.isoformat()} → {end.isoformat()}"


def kline_fetch_days(window_days: int) -> int:
    """回測窗再多抓 1 天，加上 extra_bars 才夠算 MA200。"""
    return max(int(window_days), 1) + 1


def pool_label_of(pool: str, top: int | None) -> str:
    top_txt = f"成交額前 {top}" if top and top > 0 else "全部"
    if pool == "crypto":
        return f"USDT 加密永續{top_txt}"
    if pool == "both":
        return f"USDT 股票{top_txt} ＋ 加密{top_txt}"
    return f"USDT 股票合約{top_txt}"


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)), encoding="utf-8")


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
    except Exception:
        return False


def _style_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def entry_mark(d: dict, row: SignalRow, tf: str) -> tuple[int, float]:
    """進場＝訊號下一根 1m 開盤。1m 圖圈那根；5m 圖圈含進場時間的那根。"""
    if tf == "1m":
        nxt = row.sig.idx + 1
        if 0 <= nxt < len(d["c"]):
            px = float(row.entry) if row.entry == row.entry else float(d["o"][nxt])
            return nxt, px
        return row.sig.idx, float(row.sig.close)
    ts = int(row.time_ms) + 60_000
    i = bar_index_at(d["t"], ts)
    px = float(row.entry) if row.entry == row.entry else float(d["c"][i])
    return i, px


def draw_chart(
    sym: str,
    d: dict,
    row: SignalRow,
    path: Path,
    title_note: str = "",
    *,
    tf: str = "1m",
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle

        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        return None
    mark_i, mark_px = entry_mark(d, row, tf)
    lookback, lookfwd = (80, 30) if tf == "1m" else (70, 20)
    a0 = max(0, mark_i - lookback)
    a1 = min(len(d["c"]), mark_i + lookfwd + 1)
    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.35, 1]},
        facecolor=BG,
    )
    for a in (ax, axv):
        _style_ax(a)
    colors_v = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = UP if up else DOWN
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.85)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.38, y0), 0.76, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append(col)
    axv.bar(xs, v, width=0.82, color=colors_v, linewidth=0, alpha=0.92)
    vma5 = sma(d["v"], 5)[sl]
    vma10 = sma(d["v"], 10)[sl]
    axv.plot(xs, vma5, color=VOL_MA[5], lw=1.05, label="MA(5)")
    axv.plot(xs, vma10, color=VOL_MA[10], lw=1.05, label="MA(10)")
    for n, col in PAL.items():
        series = sma(d["c"], n)[sl]
        val = series[mark_i - a0] if 0 <= mark_i - a0 < len(series) else np.nan
        lab = f"MA({n}): {val:g}" if not np.isnan(val) else f"MA({n})"
        ax.plot(xs, series, color=col, lw=1.55 if n == 200 else 1.15, label=lab)
    x = mark_i - a0
    if 0 <= x < len(c):
        ax.axvline(x, color="#F0B90B", ls="--", lw=0.85, alpha=0.85)
        ax.axhline(mark_px, color=CIRCLE, ls=":", lw=0.7, alpha=0.75)
        ax.annotate(
            f"{mark_px:g}",
            xy=(xs[-1], mark_px),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#fff",
            bbox={"boxstyle": "round,pad=0.15", "fc": CIRCLE, "ec": "none"},
        )
        ax.relim()
        ax.autoscale_view()
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.07
        ax.set_ylim(ymin - pad, ymax + pad)
        ymin, ymax = ax.get_ylim()
        ax.add_patch(
            Ellipse(
                (x, mark_px),
                width=5.4 if tf == "1m" else 4.6,
                height=(ymax - ymin) * 0.18,
                fill=False,
                edgecolor=CIRCLE,
                lw=2.15,
                zorder=7,
            )
        )
        ax.annotate(
            "進",
            xy=(x, mark_px),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=CIRCLE,
            zorder=8,
        )
    name = f"{sym} 永續"
    ax.set_title(f"{name}   {tf}   {hm(row.time_ms)}{title_note}", color=TEXT, fontsize=12, loc="left", pad=8)
    ax.legend(
        loc="upper left",
        fontsize=7,
        frameon=False,
        labelcolor=TEXT,
        ncol=3,
        borderaxespad=0.2,
    )
    axv.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=TEXT, ncol=2)
    axv.set_ylabel("VOL", color=MUTED, fontsize=8)
    axv.yaxis.set_label_position("right")
    nbar = len(xs)
    step = max(1, nbar // 6)
    ticks = list(range(0, nbar, step))
    labels = [datetime.fromtimestamp(int(d["t"][a0 + k]) / 1000, TZ).strftime("%H:%M") for k in ticks]
    axv.set_xticks(ticks)
    axv.set_xticklabels(labels, color=MUTED, fontsize=8)
    ax.set_xlim(-0.7, nbar - 0.15)
    fig.tight_layout(pad=0.45, h_pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    return path


def scan_symbol(
    item: tuple[int, str, float, str],
    date: str,
    min_gap: int,
    cross_only: bool,
    ribbon_kw: dict | None = None,
    days: int = 1,
) -> tuple[str, dict | None, list[SignalRow], str]:
    rank, sym, qv, kind = item
    lo, hi = day_window_ms(date, days)
    try:
        raw = fetch_klines(sym, interval="1m", days=kline_fetch_days(days), extra_bars=260)
    except Exception as exc:  # noqa: BLE001
        return sym, None, [], str(exc)[:80]
    if raw is None or len(raw["c"]) < 220:
        return sym, None, [], "too_few_bars"
    d = add_mas(raw)
    rows: list[SignalRow] = []
    for sig in detect_combo(d, min_gap_bars=min_gap, cross_only=cross_only, **(ribbon_kw or {})):
        ts = int(d["t"][sig.idx])
        if ts < lo or ts >= hi:
            continue
        entry, moves = forward_moves(d, sig)
        if np.isnan(entry):
            continue
        rows.append(
            SignalRow(
                symbol=sym,
                sig=sig,
                time_ms=ts,
                entry=entry,
                quote_volume=qv,
                rank=rank,
                kind=kind,
                moves=moves,
            )
        )
    return sym, d, rows, ""


def format_alert(row: SignalRow) -> str:
    kind = "剛站上 1m MA200" if row.crossed_200 else "多頭排列剛成立"
    below = f"底下 {row.bars_below} 根" if row.bars_below else "已在 1m MA200 上"
    return (
        f"<b>1m 多頭排列上站 1m MA200</b>\n"
        f"<b>{row.symbol}</b>  一分K  {hm(row.time_ms)}\n"
        f"{kind} · {below}\n"
        f"現價 {row.sig.close:g}　進 {row.entry:g}\n"
        f"1m MA200 {row.sig.ma200:g} &gt; 7 {row.sig.m7:g} &gt; 14 {row.sig.m14:g} "
        f"&gt; 25 {row.sig.m25:g} &gt; 99 {row.sig.m99:g} &gt; 120 {row.sig.m120:g}\n"
        f"黏帶全距 {row.sig.ribbon_pct:.2f}%　短均距 {row.sig.short_pct:.2f}%　"
        f"偏離 1m MA200 {row.ext_pct:+.2f}%　量比 {row.vol_ratio:.2f}x"
    )


def key_of(row: SignalRow) -> str:
    return f"{row.symbol}:{row.time_ms}"


def _card_img(sym: str, d: dict | None, row: SignalRow, img_dir: Path, title_note: str = "") -> str:
    stamp = datetime.fromtimestamp(row.time_ms / 1000, TZ).strftime("%m%d_%H%M")
    base = f"{file_base(sym)}_{stamp}"
    img_1m = f"{base}.png"
    img_5m = f"{base}_5m.png"
    if d is None:
        return ""
    out1 = draw_chart(sym, d, row, img_dir / img_1m, title_note=title_note, tf="1m")
    if out1 is None:
        return ""
    d5 = add_mas(resample_ohlcv(d, FIVE_MIN_MS))
    out5 = draw_chart(sym, d5, row, img_dir / img_5m, title_note=title_note, tf="5m") if len(d5["c"]) >= 8 else None
    blocks = [
        "<div class='tf-block'><div class='tf-lab'>1 分 K · 紅圈＝下一根開盤進場</div>"
        f"<div class='mini-chart'><img src='{escape(img_src('img/ma1m-bull/' + img_1m))}' "
        f"alt='{escape(sym)} 1m'/></div></div>"
    ]
    if out5 is not None:
        blocks.append(
            "<div class='tf-block'><div class='tf-lab'>5 分 K 對照（同一時間）</div>"
            f"<div class='mini-chart'><img src='{escape(img_src('img/ma1m-bull/' + img_5m))}' "
            f"alt='{escape(sym)} 5m'/></div></div>"
        )
    return "".join(blocks)


def write_html(
    path: Path,
    rows: list[SignalRow],
    frames: dict[str, dict],
    *,
    date: str,
    universe_n: int,
    names: list[str],
    max_charts: int,
    pool_label: str = "USDT 股票合約成交額前 10",
    days: int = 1,
    pool: str = "stocks",
) -> Path:
    stats = {h: summarize_rows(rows, h) for h in HORIZONS}
    date_label = window_label(date, days)
    cross_n = sum(1 for r in rows if r.crossed_200)
    stock_n = sum(1 for r in rows if r.kind != "crypto")
    crypto_n = sum(1 for r in rows if r.kind == "crypto")
    if pool == "crypto":
        pool_note = "只掃幣安 <strong>USDT 加密永續</strong>，不含股票、黃金原油等商品、指數。"
    elif pool == "both":
        pool_note = "股票與加密 <strong>各取成交額前 N</strong>（USDT 永續）。不含黃金原油等商品、指數。"
    else:
        pool_note = "只掃幣安 <strong>USDT 股票合約</strong>（美股／韓股／港股／A 股／Pre-IPO），不含加密、黃金原油等商品。"
    cards = []
    limit = len(rows) if max_charts <= 0 else max_charts
    gallery = rows[:limit]
    img_dir = path.parent / "img" / "ma1m-bull"
    if img_dir.exists():
        for old in img_dir.glob("*.png"):
            old.unlink()
    for i, row in enumerate(gallery, 1):
        d = frames.get(row.symbol)
        img_html = _card_img(row.symbol, d, row, img_dir)
        kind = "剛站上 1m MA200" if row.crossed_200 else "排列成立"
        r15 = row.moves.get(15)
        pnl = r15.ret_pct if r15 and r15.ret_pct is not None else None
        cls = "pnl-win" if pnl is not None and pnl > 0 else ("pnl-loss" if pnl is not None and pnl < 0 else "pnl-flat")
        pnl_txt = f"{pnl:+.2f}%" if pnl is not None else "—"
        fwd = "  ".join(
            f"{LABELS[h]} {row.moves[h].ret_pct:+.2f}%"
            if row.moves.get(h) and row.moves[h].ret_pct is not None
            else f"{LABELS[h]} —"
            for h in (5, 15, 30, 60)
        )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>{escape(row.symbol)} 永續</span>"
            f"<span class='trade-time'>#{i} · 1m · {escape(hm(row.time_ms))} · {escape(row.kind_label)} · 成交額第 {row.rank}</span></div>"
            f"<div class='card-pnl {cls}'>{escape(pnl_txt)}</div>"
            "</header>"
            f"<div class='px {cls}'>{row.sig.close:g} <span class='px-sub'>{escape(kind)} · ext {row.ext_pct:+.2f}%</span></div>"
            f"<div class='tags'><span class='tag'>{escape(row.kind_label)}</span>"
            f"<span class='tag'>200&gt;7&gt;14&gt;25&gt;99&gt;120</span>"
            f"<span class='tag'>黏帶 {row.sig.ribbon_pct:.2f}%</span>"
            f"<span class='tag'>量 {row.vol_ratio:.1f}x</span></div>"
            "<pre class='trade-detail'>"
            f"close {row.sig.close:g}  entry {row.entry:g}\n"
            f"MA7 {row.sig.m7:g}  14 {row.sig.m14:g}  25 {row.sig.m25:g}  99 {row.sig.m99:g}  120 {row.sig.m120:g}  200 {row.sig.ma200:g}\n"
            f"均線全距 {row.sig.ribbon_pct:.2f}%  短均距 {row.sig.short_pct:.2f}%\n"
            f"{fwd}"
            "</pre>"
            f"{img_html}"
            "</article>"
        )
    table_rows = []
    for i, row in enumerate(rows, 1):
        kind = "上站" if row.crossed_200 else "排列"
        cells = "".join(
            (
                f"<td class='{'pos' if m.ret_pct > 0 else 'neg' if m.ret_pct < 0 else ''}'>{m.ret_pct:+.2f}%</td>"
                if (m := row.moves.get(h)) and m.ret_pct is not None
                else "<td>—</td>"
            )
            for h in (5, 15, 30, 60, 240)
        )
        table_rows.append(
            "<tr>"
            f"<td>{i}</td><td>{escape(row.symbol)}</td><td>{escape(row.kind_label)}</td>"
            f"<td>{escape(hm(row.time_ms))}</td>"
            f"<td>{escape(kind)}</td><td>{row.ext_pct:+.2f}%</td>{cells}"
            "</tr>"
        )
    kpis = []
    for h in (15, 30, 60, 240):
        s = stats[h]
        kpis.append(
            f"<div class='card'>{escape(LABELS[h])} 勝率"
            f"<b>{s['wr']:.1f}%</b><span class='muted'>{s['n']} 筆 · 均 {s['avg']:+.2f}%</span></div>"
        )
    extra = ""
    if len(rows) > len(gallery):
        extra = f"<p class='muted'>圖表只畫前 {len(gallery)} 筆，表格含全部 {len(rows)} 筆。</p>"
    names_txt = "、".join(names[:12]) + ("…" if len(names) > 12 else "")
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安 1m 7&gt;14&gt;25&gt;99&gt;120 上站 MA200 · {escape(date_label)}</title>
<style>
body{{margin:0;background:#f5f6f7;color:#1e2329;font-family:-apple-system,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
.wide{{max-width:920px}}
.summary{{background:#fff;border:1px solid #eaecef;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#707a8a;font-size:13px;line-height:1.5}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#fafafa;padding:10px 12px;border-radius:10px;min-width:110px;border:1px solid #eaecef}}
.card b{{display:block;font-size:18px;margin-top:4px}}
.trade-card{{background:#fff;border:1px solid #eaecef;border-radius:14px;padding:14px;margin-bottom:14px}}
.card-header{{display:flex;justify-content:space-between;gap:10px}}
.trade-no{{font-weight:700}} .trade-time{{font-size:12px;color:#707a8a}}
.card-pnl{{font-weight:700}} .pnl-win{{color:#0ecb81}} .pnl-loss{{color:#f6465d}} .pnl-flat{{color:#707a8a}}
.px{{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:6px 0 4px}}
.px.pnl-win{{color:#0ecb81}} .px.pnl-loss{{color:#f6465d}}
.px-sub{{font-size:13px;font-weight:500;color:#707a8a}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #eaecef;color:#474d57;background:#fafafa}}
.trade-detail{{background:#fafafa;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap;color:#474d57}}
.tf-block{{margin-top:10px}}
.tf-lab{{font-size:12px;color:#707a8a;margin:0 0 6px}}
.mini-chart img{{width:100%;display:block;border-radius:8px;border:1px solid #eaecef}}
.empty{{text-align:center;color:#707a8a;padding:40px 12px;border:1px solid #eaecef;border-radius:14px;background:#fff}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}}
th,td{{padding:6px 4px;border-bottom:1px solid #eaecef;text-align:right}}
th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){{text-align:left}}
.pos{{color:#0ecb81}} .neg{{color:#f6465d}}
</style></head><body>
<div class="page wide">
<section class="summary">
<h1>幣安一分K · 7&gt;14&gt;25&gt;99&gt;120 黏帶上站 MA200</h1>
<p class="muted">{escape(date_label)} 台北時間 · {escape(pool_label)} · {len(rows)} 筆訊號（剛站上 {cross_n}；股票 {stock_n}／加密 {crypto_n}）
<br/>規則（截圖紅圈）：短均先黏帶，長期在 MA200 下，<strong>放量剛站上</strong>。排列 收盤 &gt; MA200 &gt; 7 &gt; 14 &gt; 25 &gt; 99 &gt; 120。進場用下一根開盤（圖上紅圈）。每筆附 1 分 K ＋ 當下 5 分 K。
<br/>{pool_note}
<br/>標的：{escape(names_txt)}</p>
<div class="cards">
<div class="card">筆數<b>{len(rows)}</b></div>
<div class="card">股票<b>{stock_n}</b></div>
<div class="card">加密<b>{crypto_n}</b></div>
<div class="card">標的<b>{len({r.symbol for r in rows})}</b></div>
{''.join(kpis)}
</div>
{extra}
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有多頭排列上站 1m MA200</div>"}
<section class="summary">
<h1>全部訊號</h1>
<table>
<thead><tr><th>#</th><th>標的</th><th>池</th><th>時間</th><th>種類</th><th>偏離</th>
<th>5m</th><th>15m</th><th>30m</th><th>1h</th><th>4h</th></tr></thead>
<tbody>{''.join(table_rows) or "<tr><td colspan='11'>無</td></tr>"}</tbody>
</table>
</section>
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path) -> Path:
    """Absolute GitHub Pages image URLs so htmlpreview / raw HTML still show the new charts."""
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{PAGES_IMG}img/")
    out = src.with_name("ma1m-bull-view.html")
    out.write_text(text, encoding="utf-8")
    return out


def ribbon_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_ribbon_pct": None if args.max_ribbon <= 0 else args.max_ribbon,
        "max_short_pct": None if args.max_short <= 0 else args.max_short,
        "max_prior_short": None if getattr(args, "max_prior_short", 0.15) <= 0 else args.max_prior_short,
        "min_vol_ratio": float(getattr(args, "min_vol", 1.4)),
        "min_below": int(getattr(args, "min_below", 20)),
    }


def run_backtest(args: argparse.Namespace) -> int:
    date = args.date or default_date()
    days = getattr(args, "days", 1)
    pool = getattr(args, "pool", "stocks")
    cross_only = not args.all_stack
    rkw = ribbon_kwargs(args)
    print(
        f"date={date} days={days} window={window_label(date, days)} pool={pool} "
        f"top={args.top or 'all'} min_gap={args.min_gap} "
        f"cross_only={cross_only} pack≤{rkw['max_ribbon_pct']} short≤{rkw['max_short_pct']} "
        f"prior≤{rkw['max_prior_short']} vol≥{rkw['min_vol_ratio']} below≥{rkw['min_below']}",
        flush=True,
    )
    uni = universe(top_n=args.top, pool=pool)
    if not uni:
        print("no universe", file=sys.stderr)
        return 1
    label = pool_label_of(pool, args.top)
    print(
        f"{label} {len(uni)}  #{1} {uni[0][0]} {uni[0][1]/1e6:.0f}M  "
        f"末 {uni[-1][0]} {uni[-1][1]/1e6:.0f}M",
        flush=True,
    )
    items = [(i, sym, qv, kind) for i, (sym, qv, kind) in enumerate(uni, 1)]
    rows: list[SignalRow] = []
    frames: dict[str, dict] = {}
    errors = 0
    with ThreadPoolExecutor(6) as ex:
        futs = {
            ex.submit(scan_symbol, it, date, args.min_gap, cross_only, rkw, days): it
            for it in items
        }
        for fut in as_completed(futs):
            rank, sym, _qv, kind = futs[fut]
            try:
                _s, d, hits, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{rank:2d}/{len(items)}] {sym} {kind} err {exc}", flush=True)
                continue
            if err:
                errors += 1
            if d is not None:
                frames[sym] = d
            rows.extend(hits)
            flag = f" hits={len(hits)}" if hits else ""
            print(f"[{rank:2d}/{len(items)}] {sym} {kind} bars={0 if d is None else len(d['c'])}{flag} {err}", flush=True)
    rows.sort(key=lambda r: r.time_ms)
    stock_n = sum(1 for r in rows if r.kind != "crypto")
    crypto_n = sum(1 for r in rows if r.kind == "crypto")
    print(
        f"done errors={errors} signals={len(rows)} stock={stock_n} crypto={crypto_n} "
        f"symbols={len({r.symbol for r in rows})}",
        flush=True,
    )
    for h in HORIZONS:
        s = summarize_rows(rows, h)
        print(f"  {LABELS[h]:>4s}  n={s['n']:3d}  WR={s['wr']:.1f}%  avg={s['avg']:+.2f}%  med={s['med']:+.2f}%")
    for kind_name, subset in (("股票", [r for r in rows if r.kind != "crypto"]), ("加密", [r for r in rows if r.kind == "crypto"])):
        if not subset:
            continue
        s15 = summarize_rows(subset, 15)
        print(f"  {kind_name}  n={s15['n']:3d}  15m WR={s15['wr']:.1f}%  avg={s15['avg']:+.2f}%")
    for i, row in enumerate(rows, 1):
        kind = "上站" if row.crossed_200 else "排列"
        r15 = row.moves.get(15)
        rtxt = f"{r15.ret_pct:+.2f}%" if r15 and r15.ret_pct is not None else "—"
        print(
            f"  [{i:3d}] {row.kind_label} {row.symbol:12s} {hm(row.time_ms)} {kind} "
            f"ribbon={row.sig.ribbon_pct:.2f}% ext={row.ext_pct:+.2f}% 15m={rtxt}"
        )

    html_path = Path(args.html) if args.html else (PAGES if args.pages else None)
    if html_path:
        out = write_html(
            html_path,
            rows,
            frames,
            date=date,
            universe_n=len(uni),
            names=[s for s, _qv, _k in uni],
            max_charts=args.charts,
            pool_label=label,
            days=days,
            pool=pool,
        )
        view = write_view_html(out)
        print(f"html={out}")
        print(f"preview={PUBLIC}")
        print(f"view={view}")
    return 0


def wait_next_close() -> None:
    now = time.time()
    nxt = (int(now) // 60 + 1) * 60 + 2
    time.sleep(max(1, nxt - now))


def scan_live(
    sym: str,
    qv: float,
    rank: int,
    *,
    kind: str = "stock",
    cross_only: bool = True,
    ribbon_kw: dict | None = None,
) -> list[SignalRow]:
    raw = fetch_klines(sym, interval="1m", limit=260)
    if raw is None or len(raw["c"]) < 220:
        return []
    d = add_mas(raw)
    n = len(d["c"])
    out = []
    for sig in detect_combo(d, cross_only=cross_only, **(ribbon_kw or {})):
        if sig.idx not in (n - 1, n - 2):
            continue
        entry, moves = forward_moves(d, sig)
        entry = float(d["c"][sig.idx]) if np.isnan(entry) else entry
        row = SignalRow(
            symbol=sym,
            sig=sig,
            time_ms=int(d["t"][sig.idx]),
            entry=entry,
            quote_volume=qv,
            rank=rank,
            kind=kind,
            moves=moves,
        )
        row._frame = d  # type: ignore[attr-defined]
        out.append(row)
    return out


def notify(row: SignalRow, *, dry_run: bool) -> None:
    text = format_alert(row)
    plain = text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">")
    print("\n" + plain, flush=True)
    if dry_run:
        print("  → dry-run，不送 Telegram", flush=True)
        return
    photo = None
    d = getattr(row, "_frame", None)
    if d is not None:
        tmp = Path("/tmp") / f"ma1m_{file_base(row.symbol)}_{row.time_ms}.png"
        photo = str(draw_chart(row.symbol, d, row, tmp) or "")
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送", flush=True)
    elif not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡", flush=True)
    else:
        print("  → Telegram 送出失敗，檢查 token 與 chat id", flush=True)


def run_alert(args: argparse.Namespace) -> int:
    apply_keys()
    if args.test:
        ok = telegram_send("一分K 多頭排列上站 1m MA200 測試\n如果你看到這則，Telegram 已通。")
        print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
        return 0 if ok else 1

    seen = load_seen()
    print("載入標的…", flush=True)
    pool = getattr(args, "pool", "stocks")
    uni = universe(top_n=args.top, pool=pool)
    label = pool_label_of(pool, args.top)
    print(
        f"監看 {label} {len(uni)} 個。黏帶後放量剛站上 1m MA200 才推。",
        flush=True,
    )
    uni_ts = time.time()
    rkw = ribbon_kwargs(args)

    def round_once() -> None:
        nonlocal uni, uni_ts
        if time.time() - uni_ts > 1800:
            uni = universe(top_n=args.top, pool=pool)
            uni_ts = time.time()
            print(f"更新標的 {len(uni)}", flush=True)
        t0 = time.time()
        events: list[SignalRow] = []
        with ThreadPoolExecutor(8) as ex:
            futs = {
                ex.submit(
                    scan_live,
                    sym,
                    qv,
                    i,
                    kind=kind,
                    cross_only=not args.all_stack,
                    ribbon_kw=rkw,
                ): sym
                for i, (sym, qv, kind) in enumerate(uni, 1)
            }
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as e:
                    print("err", futs[fut], e, flush=True)
        new = [e for e in events if key_of(e) not in seen]
        new.sort(key=lambda r: r.time_ms)
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(uni)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev, dry_run=args.dry_run)
        if new:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="幣安一分K：7>14>25>99>120 黏帶上站 1m MA200")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="回測 USDT 永續（預設股票成交額前 10、今天）")
    b.add_argument("--top", type=int, default=10, help="成交額前 N；both 時股票、加密各取 N；0=該池全部")
    b.add_argument("--pool", choices=("stocks", "crypto", "both"), default="stocks", help="stocks / crypto / both")
    b.add_argument("--date", default="", help="YYYY-MM-DD，台北日，預設今天（凌晨 2 點前用昨天）")
    b.add_argument("--today", action="store_true", help="明確指定用今天（同預設）")
    b.add_argument("--days", type=int, default=1, help="往回幾天（含 --date 當天）")
    b.add_argument("--min-gap", type=int, default=0, help="同一標的訊號最少間隔根數")
    b.add_argument("--all-stack", action="store_true", help="含已在 MA200 上才排好均線（會很多）")
    b.add_argument("--max-ribbon", type=float, default=0.65, help="短均+MA200 包距%上限；0=不限")
    b.add_argument("--max-short", type=float, default=0.50, help="MA7/14/25 全距%上限；0=不限")
    b.add_argument("--max-prior-short", type=float, default=0.15, help="站上前 20 根短均最小距%上限；0=不限")
    b.add_argument("--min-vol", type=float, default=1.4, help="量比下限；0=不限")
    b.add_argument("--min-below", type=int, default=20, help="站上前至少連續幾根在 MA200 下")
    b.add_argument("--pages", action="store_true")
    b.add_argument("--html", default="")
    b.add_argument("--charts", type=int, default=0, help="圖表筆數；0=全部")
    b.set_defaults(func=run_backtest)

    a = sub.add_parser("alert", help="掃 USDT 永續，符合就推 Telegram")
    a.add_argument("--top", type=int, default=10, help="成交額前 N；both 時股票、加密各取 N；0=該池全部")
    a.add_argument("--pool", choices=("stocks", "crypto", "both"), default="stocks", help="stocks / crypto / both")
    a.add_argument("--once", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--all-stack", action="store_true", help="含已在 MA200 上才排好均線（會很多）")
    a.add_argument("--max-ribbon", type=float, default=0.65, help="短均+MA200 包距%上限；0=不限")
    a.add_argument("--max-short", type=float, default=0.50, help="MA7/14/25 全距%上限；0=不限")
    a.add_argument("--max-prior-short", type=float, default=0.15, help="站上前 20 根短均最小距%上限；0=不限")
    a.add_argument("--min-vol", type=float, default=1.4, help="量比下限；0=不限")
    a.add_argument("--min-below", type=int, default=20, help="站上前至少連續幾根在 MA200 下")
    a.set_defaults(func=run_alert)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
