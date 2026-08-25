#!/usr/bin/env python3
"""台股五分 K：W 底之後收盤上穿 MA20，推 Telegram。

對齊券商 App 那種圖：大跌做出雙底，反彈站上五分 MA20 就跳通知。

用法:
  python3 examples/watch_tw_w_ma20.py --test
  python3 examples/watch_tw_w_ma20.py --dry-run --once
  python3 examples/watch_tw_w_ma20.py --once --symbols 2327,2408
  python3 examples/watch_tw_w_ma20.py scan --limit 100 --pages
  python3 examples/watch_tw_w_ma20.py scan --range 10d --days 7 --pages  # 回測一週
  python3 examples/watch_tw_w_ma20.py          # 盤中每根 5 分收盤掃一次

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
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.patterns import WMa20Signal, detect_w_ma20_crosses  # noqa: E402
from scan_tw_ma_reclaim import (  # noqa: E402
    REPO,
    TPE,
    _chart_payload_to_df,
    _get_json,
    fetch_top_turnover,
    last_tw_session_yyyymmdd,
    resolve_twse_date,
    yahoo_symbol,
)

try:
    import requests
except ImportError:
    requests = None  # type: ignore

STATE_PATH = REPO / "output" / "tw_w_ma20_seen.json"
CONFIG_ENV = REPO / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = Path(__file__).resolve().parent / "tg_config.env"
PAGES = REPO / "docs" / "tw-w-ma20" / "index.html"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SESSION_OPEN = 9 * 60
SESSION_CLOSE = 13 * 60 + 30


@dataclass
class TwHit:
    row: dict
    signal: WMa20Signal
    df: pd.DataFrame


def load_dotenv(path: Path = CONFIG_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value not in (None, "") else default


def fetch_yahoo_5m(symbol: str, range_: str = "5d") -> tuple[pd.DataFrame, str]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=5m&range={range_}&includePrePost=false"
    )
    payload = _get_json(url)
    result = ((payload.get("chart") or {}).get("result") or [None])[0] or {}
    name = str((result.get("meta") or {}).get("shortName") or "").strip()
    return _chart_payload_to_df(payload), name


def drop_forming_bar(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Yahoo 五分 K 時間戳是開盤；未收盤的最後一根先丟掉。"""
    if df.empty:
        return df
    cur = now or datetime.now(TPE)
    if cur.tzinfo is None:
        cur = cur.replace(tzinfo=TPE)
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize(TPE)
    aligned = last.second == 0 and last.microsecond == 0 and last.minute % 5 == 0
    if (not aligned) or cur < last + timedelta(minutes=5):
        return df.iloc[:-1].copy()
    return df


def parse_symbols(text: str) -> list[dict]:
    rows: list[dict] = []
    for i, raw in enumerate((text or "").split(","), 1):
        token = raw.strip().upper()
        if not token:
            continue
        if token.endswith(".TW") or token.endswith(".TWO"):
            code = token.split(".", 1)[0]
            market = "otc" if token.endswith(".TWO") else "tse"
            symbol = token
        else:
            code = token
            market = "tse"
            symbol = yahoo_symbol(code, market)
        rows.append(
            {
                "rank": i,
                "code": code,
                "name": code,
                "market": market,
                "amount": 0,
                "close": None,
                "symbol": symbol,
            }
        )
    return rows


def has_cjk(text: object) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


_NAME_MAP: dict[str, str] = {}
_NAME_MAP_LOADED = False


def remember_names(rows: list[dict]) -> None:
    for row in rows:
        name = str(row.get("name") or "").strip()
        if row.get("code") and has_cjk(name):
            _NAME_MAP[str(row["code"])] = name.rstrip("*").strip()


def load_tw_name_map() -> dict[str, str]:
    """上市櫃中文股名；同一行程只抓一次。"""
    global _NAME_MAP_LOADED
    if _NAME_MAP_LOADED:
        return _NAME_MAP
    try:
        date = resolve_twse_date(last_tw_session_yyyymmdd())
        remember_names(fetch_top_turnover(date, 10000))
        _NAME_MAP_LOADED = True
    except Exception as exc:  # noqa: BLE001
        print(f"[name] {exc}", file=sys.stderr)
    return _NAME_MAP


def fill_chinese_names(rows: list[dict]) -> list[dict]:
    remember_names(rows)
    missing = [row for row in rows if not has_cjk(row.get("name"))]
    if missing:
        load_tw_name_map()
    out: list[dict] = []
    for row in rows:
        current = str(row.get("name") or "").rstrip("*").strip()
        if has_cjk(current):
            out.append({**row, "name": current})
            continue
        zh = _NAME_MAP.get(str(row.get("code") or ""))
        out.append({**row, "name": zh} if zh else {**row, "name": current or row.get("name")})
    return out


def filter_price_below(rows: list[dict], max_price: float | None, limit: int) -> tuple[list[dict], list[dict]]:
    """股價達到 max_price（含）以上的剔除，例如 700 以上不看。"""
    if max_price is None:
        kept = rows[:limit]
        return kept, []
    kept: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        px = row.get("close")
        if px is None or float(px) >= max_price:
            dropped.append(row)
            continue
        kept.append(row)
        if len(kept) >= limit:
            break
    for i, row in enumerate(kept, 1):
        row["rank"] = i
    return kept, dropped


def hit_under_max_price(hit: TwHit, max_price: float | None) -> bool:
    if max_price is None:
        return True
    last = float(hit.df["Close"].iloc[-1])
    listed = hit.row.get("close")
    px = max(last, float(listed) if listed is not None else last)
    return px < max_price


def hit_ts(hit: TwHit) -> pd.Timestamp:
    ts = hit.df.index[hit.signal.cross_idx]
    if getattr(ts, "tzinfo", None) is None:
        return pd.Timestamp(ts).tz_localize(TPE)
    return pd.Timestamp(ts).tz_convert(TPE)


def filter_hits_days(hits: list[TwHit], days: int, now: datetime | None = None) -> list[TwHit]:
    """只留最近 N 個日曆日（含今天）。days=7 就是一週。"""
    if not days or days <= 0:
        return hits
    cur = now or datetime.now(TPE)
    if cur.tzinfo is None:
        cur = cur.replace(tzinfo=TPE)
    start = cur.date() - timedelta(days=days - 1)
    return [h for h in hits if hit_ts(h).date() >= start]


def bounce_pct(sig: WMa20Signal) -> float:
    base = min(sig.first_low, sig.second_low)
    if base <= 0:
        return 0.0
    return (sig.cross_price / base - 1.0) * 100.0


def stand_pct(sig: WMa20Signal) -> float:
    if sig.ma20 <= 0:
        return 0.0
    return (sig.cross_price / sig.ma20 - 1.0) * 100.0


def is_notable_hit(hit: TwHit, *, min_bounce: float = 1.2, min_stand: float = 0.25) -> bool:
    return bounce_pct(hit.signal) >= min_bounce and stand_pct(hit.signal) >= min_stand


def load_universe(args: argparse.Namespace) -> list[dict]:
    if getattr(args, "symbols", ""):
        return fill_chinese_names(parse_symbols(args.symbols))
    date = resolve_twse_date(args.date or last_tw_session_yyyymmdd())
    max_price = args.max_price
    pool = max(args.limit, args.pool if max_price else args.limit)
    raw = fetch_top_turnover(date, pool)
    universe, dropped = filter_price_below(raw, max_price, args.limit)
    if dropped:
        print(
            f"drop price>={max_price:g}: "
            + ", ".join(f"{r['code']} {r['close']}" for r in dropped[:8])
            + (" …" if len(dropped) > 8 else ""),
            file=sys.stderr,
        )
    print(
        f"universe date={date} n={len(universe)} max_price={max_price} "
        f"{universe[0]['code']} {universe[0]['name']}" if universe else "universe empty",
        file=sys.stderr,
    )
    return fill_chinese_names(universe)


def scan_symbol(row: dict, range_: str, *, live: bool) -> tuple[list[TwHit], dict]:
    meta = {**row, "bars": 0, "error": "", "n_sig": 0}
    try:
        df, _yahoo_name = fetch_yahoo_5m(row["symbol"], range_)
        if live:
            df = drop_forming_bar(df)
        # 股名用上市櫃中文，不用 Yahoo 英文簡稱
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:80]
        return [], meta
    meta["bars"] = int(len(df))
    if len(df) < 60:
        meta["error"] = "too_few_bars"
        return [], meta
    sigs = detect_w_ma20_crosses(df)
    meta["n_sig"] = len(sigs)
    return [TwHit(row, s, df) for s in sigs], meta


def _cjk_font() -> None:
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
            return


def draw_signal_png(df: pd.DataFrame, sig: WMa20Signal, path: Path, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    _cjk_font()
    pad_left, pad_right = 36, 12
    start = max(0, sig.first_low_idx - pad_left)
    end = min(len(df) - 1, sig.cross_idx + pad_right)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)
    ma20 = close_full.rolling(20, min_periods=20).mean().iloc[start : end + 1]
    ma5 = close_full.rolling(5, min_periods=5).mean().iloc[start : end + 1]
    ma10 = close_full.rolling(10, min_periods=10).mean().iloc[start : end + 1]
    ma60 = close_full.rolling(60, min_periods=60).mean().iloc[start : end + 1]

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
        col = "#e35d5d" if up else "#3dba7a"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.7)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#e35d5d99" if up else "#3dba7a99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    ax.plot(list(xs), ma5, color="#42a5f5", lw=1.0, label="MA5")
    ax.plot(list(xs), ma10, color="#66bb6a", lw=1.0, label="MA10")
    ax.plot(list(xs), ma20, color="#ffca28", lw=1.6, label="MA20")
    ax.plot(list(xs), ma60, color="#ab47bc", lw=1.0, label="MA60")

    marks = (
        (sig.first_low_idx, sig.first_low, "L1", "#80deea"),
        (sig.second_low_idx, sig.second_low, "L2", "#80deea"),
        (sig.neckline_idx, sig.neckline, "頸", "#ffb74d"),
        (sig.cross_idx, sig.cross_price, "MA20", "#00e676"),
    )
    for idx, price, label, color in marks:
        x = idx - start
        if 0 <= x < len(window):
            ax.scatter([x], [price], s=36, color=color, zorder=6)
            ax.annotate(
                label,
                (x, price),
                textcoords="offset points",
                xytext=(0, -13 if label != "MA20" else 10),
                ha="center",
                color=color,
                fontsize=8,
            )
    cx = sig.cross_idx - start
    if 0 <= cx < len(window):
        ax.axvline(cx, color="#00e676", ls="--", lw=0.9, alpha=0.8)

    ax.set_title(title, color="#e8f0ea", fontsize=12)
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


def fmt_alert(hit: TwHit) -> str:
    df, sig, row = hit.df, hit.signal, hit.row
    ts = df.index[sig.cross_idx]
    l1t = df.index[sig.first_low_idx]
    l2t = df.index[sig.second_low_idx]
    last = float(df["Close"].iloc[-1])
    name = row.get("name") or row["code"]
    return (
        f"🟢 <b>五分K W底 上 MA20</b>\n"
        f"<b>{escape(str(row['code']))} {escape(str(name))}</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} 台北</code>\n"
        f"收盤: <code>{sig.cross_price:.2f}</code>  MA20: <code>{sig.ma20:.2f}</code>\n"
        f"L1: <code>{l1t.strftime('%H:%M')}</code> {sig.first_low:.2f}　"
        f"L2: <code>{l2t.strftime('%H:%M')}</code> {sig.second_low:.2f}\n"
        f"頸線: <code>{sig.neckline:.2f}</code>　量度: <code>{sig.target:.2f}</code>\n"
        f"停損: <code>{sig.stop_loss:.2f}</code>　現價: <code>{last:.2f}</code>\n"
        f"#W底 #MA20 #五分K #{escape(str(row['code']))}"
    )


def hit_key(hit: TwHit) -> str:
    ts = hit.df.index[hit.signal.cross_idx]
    return f"{hit.row['symbol']}|{ts.isoformat()}|{hit.signal.cross_price:.2f}"


def load_seen() -> set[str]:
    if not STATE_PATH.exists():
        return set()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return set(data.get("keys") or data if isinstance(data, list) else [])
    except Exception:
        return set()


def save_seen(keys: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"keys": sorted(keys)[-400:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def tg_send(token: str, chat_id: str, text: str, photo: Path | None = None, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID", file=sys.stderr)
        return False
    try:
        if photo is not None and photo.exists():
            with photo.open("rb") as fh:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": "HTML"},
                    files={"photo": fh},
                    timeout=30,
                )
            if resp.ok:
                return True
            print(f"[tg] photo HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        if not resp.ok:
            print(f"[tg] HTTP {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
            return False
        return bool(resp.json().get("ok"))
    except Exception as exc:  # noqa: BLE001
        print(f"[tg] {exc}", file=sys.stderr)
        return False


def _day_summary_html(hits: list[TwHit]) -> str:
    buckets: dict[str, list[TwHit]] = {}
    for hit in hits:
        buckets.setdefault(hit_ts(hit).strftime("%m-%d"), []).append(hit)
    if not buckets:
        return ""
    parts = []
    extra = []
    for day, group in buckets.items():
        seen: list[str] = []
        seen_codes: set[str] = set()
        for hit in group:
            code = str(hit.row["code"])
            if code in seen_codes:
                continue
            seen_codes.add(code)
            seen.append(f"{code} {hit.row.get('name') or ''}".strip())
        parts.append(
            "<div class='card'>"
            f"{escape(day)}<b>{len(group)}</b>"
            f"<span class='muted' style='font-size:11px'>{len(seen_codes)} 檔</span>"
            "</div>"
        )
        extra.append(
            f"<p class='muted' style='margin:6px 0 0'><b>{escape(day)}</b>　"
            + "、".join(escape(n) for n in seen)
            + "</p>"
        )
    return "<div class='cards'>" + "".join(parts) + "</div>" + "".join(extra)


def _compact_hit_list_html(hits: list[TwHit]) -> str:
    if not hits:
        return ""
    lines = []
    last_day = ""
    for hit in hits:
        ts = hit_ts(hit)
        day = ts.strftime("%m-%d")
        if day != last_day:
            lines.append(f"—— {day} ——")
            last_day = day
        sig = hit.signal
        lines.append(
            f"{ts.strftime('%H:%M')}  {hit.row['code']} {hit.row.get('name') or ''}  "
            f"{sig.cross_price:.2f}  彈回 {bounce_pct(sig):.2f}%  站上 {stand_pct(sig):.2f}%"
        )
    return (
        "<article class='trade-card'><header class='card-header'>"
        "<div class='card-title'><span class='trade-no'>全部訊號</span></div></header>"
        "<pre class='trade-detail'>" + escape("\n".join(lines)) + "</pre></article>"
    )


def write_html(
    path: Path,
    hits: list[TwHit],
    universe: list[dict],
    period: str,
    chart_hits: list[TwHit] | None = None,
) -> Path:
    cards: list[str] = []
    img_dir = path.parent / "img"
    show_all_cards = chart_hits is None or chart_hits is hits
    detail_hits = hits if show_all_cards else list(chart_hits or [])
    if img_dir.exists() and not show_all_cards:
        for old in img_dir.glob("t*.png"):
            old.unlink()
    chart_i = 0
    for i, hit in enumerate(detail_hits, 1):
        sig = hit.signal
        ts = hit_ts(hit)
        label = f"{hit.row['code']} {hit.row['name']}"
        chart_i += 1
        img_name = f"t{chart_i:02d}_{hit.row['code']}_{ts.strftime('%m%d_%H%M')}.png"
        title = f"{label}  5分K W底上MA20  {ts.strftime('%m-%d %H:%M')}"
        draw_signal_png(hit.df, sig, img_dir / img_name, title)
        img_html = (
            f"<div class='mini-chart'><img src='img/{escape(img_name)}' alt='{escape(label)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
        )
        tags = (
            f"<div class='tags'><span class='tag tag-info'>{escape(hit.row['symbol'])}</span>"
            "<span class='tag'>W底</span><span class='tag'>上MA20</span>"
        )
        if is_notable_hit(hit):
            tags += "<span class='tag'>像樣</span>"
        tags += "</div>"
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · {escape(label)}</span>"
            f"<span class='trade-time'>{escape(ts.strftime('%Y-%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl pnl-win'>{sig.cross_price:.2f}</div>"
            "</header>"
            + tags
            + "<pre class='trade-detail'>"
            f"收盤 {sig.cross_price:.2f}  MA20 {sig.ma20:.2f}\n"
            f"L1 {sig.first_low:.2f} / L2 {sig.second_low:.2f} / 頸線 {sig.neckline:.2f}\n"
            f"停損 {sig.stop_loss:.2f}  量度 {sig.target:.2f}"
            f"　彈回 {bounce_pct(sig):.2f}%  站上 {stand_pct(sig):.2f}%"
            "</pre>"
            + img_html
            + "</article>"
        )
    if not show_all_cards:
        cards.append(_compact_hit_list_html(hits))
    note = ""
    if not show_all_cards:
        note = (
            "<br/>圖卡只畫彈回 ≥1.2% 且站上 MA20 ≥0.25% 的（較像樣）；"
            "全部筆數在頁面最下方清單。"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>台股五分K W底上MA20</title>
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
.card-pnl{{font-weight:700}} .pnl-win{{color:#00c805}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tag{{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid #30363d;color:#79c0ff}}
.trade-detail{{background:#0d1117;padding:10px;border-radius:10px;font-size:12px;white-space:pre-wrap}}
.empty{{text-align:center;color:#8b949e;padding:40px 12px;border:1px solid #30363d;border-radius:14px}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>台股 5分K W底上 MA20</h1>
<p class="muted">{escape(period)} · 掃描 {len(universe)} 檔
<br/>規則：五分 K 做出 W 底（先大跌、兩個相近低點），收盤由下往上穿過五分 MA20 才通知。{note}</p>
<div class="cards">
<div class="card">筆數<b>{len(hits)}</b></div>
<div class="card">標的<b>{len({h.row['code'] for h in hits})}</b></div>
<div class="card">像樣<b>{sum(1 for h in hits if is_notable_hit(h))}</b></div>
</div>
{_day_summary_html(hits)}
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有 W 底上 MA20</div>"}
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_view_html(src: Path) -> Path:
    src = src.resolve()
    rel = src.parent.relative_to(REPO).as_posix()
    base = (
        "https://raw.githubusercontent.com/yubogoodman-droid/NQ/"
        f"cursor/tw-5m-w-ma20-alert-a91a/{rel}/"
    )
    text = src.read_text(encoding="utf-8").replace("src='img/", f"src='{base}img/")
    out = src.with_name("view.html")
    out.write_text(text, encoding="utf-8")
    return out


def collect_hits(universe: list[dict], range_: str, sleep: float, live: bool) -> list[TwHit]:
    universe = fill_chinese_names(universe)
    hits: list[TwHit] = []
    for i, row in enumerate(universe, 1):
        stock_hits, meta = scan_symbol(row, range_, live=live)
        hits.extend(stock_hits)
        flag = f" hits={meta['n_sig']}" if meta["n_sig"] else ""
        err = f" {meta['error']}" if meta["error"] else ""
        print(f"[{i:3d}/{len(universe)}] {row['symbol']} {row.get('name','')} bars={meta['bars']}{flag}{err}")
        time.sleep(max(0.05, sleep))
    hits.sort(key=lambda h: h.df.index[h.signal.cross_idx])
    return hits


def recent_hits(hits: list[TwHit], lookback_hours: float) -> list[TwHit]:
    if lookback_hours <= 0:
        return hits
    now = datetime.now(TPE)
    cutoff = now.timestamp() - lookback_hours * 3600
    out = []
    for hit in hits:
        ts = hit.df.index[hit.signal.cross_idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize(TPE)
        if ts.timestamp() >= cutoff:
            out.append(hit)
    return out


def notify_hits(hits: list[TwHit], *, token: str, chat_id: str, dry_run: bool, seed: bool) -> int:
    seen = load_seen()
    first_run = not STATE_PATH.exists() or not seen
    fresh = [h for h in hits if hit_key(h) not in seen]
    if first_run and not seed:
        for hit in fresh:
            seen.add(hit_key(hit))
        save_seen(seen)
        print(f"init: marked {len(fresh)} recent signals, no spam")
        return 0
    sent = 0
    for hit in fresh:
        ts = hit.df.index[hit.signal.cross_idx]
        label = f"{hit.row['code']} {hit.row.get('name','')}"
        png = REPO / "output" / f"tw_w_ma20_{hit.row['code']}_{ts.strftime('%m%d_%H%M')}.png"
        draw_signal_png(hit.df, hit.signal, png, f"{label}  5分K W底上MA20")
        ok = tg_send(token, chat_id, fmt_alert(hit), photo=png, dry_run=dry_run)
        if ok:
            seen.add(hit_key(hit))
            sent += 1
            print(f"[alert] {label} {ts} @ {hit.signal.cross_price:.2f}")
    save_seen(seen)
    return sent


def in_session(now: datetime | None = None) -> bool:
    cur = now or datetime.now(TPE)
    minutes = cur.hour * 60 + cur.minute
    return cur.weekday() < 5 and SESSION_OPEN - 5 <= minutes <= SESSION_CLOSE + 8


def wait_next_5m() -> None:
    now = datetime.now(TPE)
    if not in_session(now):
        # 休市就等到下一個交易日 09:05
        nxt = now.replace(hour=9, minute=5, second=5, microsecond=0)
        while nxt <= now or nxt.weekday() >= 5:
            nxt += timedelta(days=1)
            nxt = nxt.replace(hour=9, minute=5, second=5, microsecond=0)
        time.sleep(max(5, (nxt - now).total_seconds()))
        return
    # 等到下一根五分收盤 + 4 秒
    elapsed = now.minute % 5
    add = 5 - elapsed if elapsed else 5
    nxt = (now + timedelta(minutes=add)).replace(second=4, microsecond=0)
    time.sleep(max(2, (nxt - now).total_seconds()))


def cmd_scan(args: argparse.Namespace) -> int:
    universe = load_universe(args)
    if not universe:
        return 1
    hits = collect_hits(universe, args.range, args.sleep, live=False)
    hits = [h for h in hits if hit_under_max_price(h, args.max_price)]
    days = int(getattr(args, "days", 0) or 0)
    if getattr(args, "today", False):
        hits = filter_hits_days(hits, 1)
    elif days:
        hits = filter_hits_days(hits, days)
    print(f"done hits={len(hits)}")
    for i, hit in enumerate(hits, 1):
        ts = hit_ts(hit)
        print(
            f"  [{i}] {hit.row['code']} {hit.row.get('name','')} "
            f"{ts.strftime('%m-%d %H:%M')} close={hit.signal.cross_price:.2f} "
            f"ma20={hit.signal.ma20:.2f} L1={hit.signal.first_low:.2f} L2={hit.signal.second_low:.2f}"
            f" bounce={bounce_pct(hit.signal):.2f}% stand={stand_pct(hit.signal):.2f}%"
        )
    html_path = Path(args.html).resolve() if args.html else None
    if html_path is None and args.pages:
        if getattr(args, "today", False):
            html_path = REPO / "docs" / "tw-w-ma20-today" / "index.html"
        elif days >= 7:
            html_path = REPO / "docs" / "tw-w-ma20-week" / "index.html"
        else:
            html_path = PAGES
    if html_path:
        period = args.range
        if args.max_price:
            period += f" · 股價<{args.max_price:g}"
        if getattr(args, "today", False):
            period = datetime.now(TPE).strftime("%Y-%m-%d") + " · " + period
        elif days:
            start = (datetime.now(TPE).date() - timedelta(days=days - 1)).isoformat()
            end = datetime.now(TPE).date().isoformat()
            period = f"{start}～{end} · {period}"
        notable = [h for h in hits if is_notable_hit(h)]
        if len(hits) > 40:
            if len(notable) > 50:
                notable = sorted(
                    notable,
                    key=lambda h: (bounce_pct(h.signal), stand_pct(h.signal)),
                    reverse=True,
                )[:50]
                notable.sort(key=lambda h: hit_ts(h))
            chart_hits = notable
        else:
            chart_hits = hits
        out = write_html(html_path, hits, universe, period, chart_hits=chart_hits)
        write_view_html(out)
        print(f"html={out} charts={len(chart_hits)}")
    return 0


def cmd_alert(args: argparse.Namespace) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN") or ""
    chat_id = env("TELEGRAM_CHAT_ID") or ""
    if args.test:
        ok = tg_send(
            token,
            chat_id,
            f"✅ 台股五分K W底上MA20 測試\n{datetime.now(TPE).strftime('%Y-%m-%d %H:%M:%S')} 台北",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2

    print(
        f"TW 5m W+MA20 | range={args.range} | lookback={args.lookback_hours}h | "
        f"dry_run={args.dry_run} | once={args.once}"
    )
    while True:
        try:
            universe = load_universe(args)
            hits = collect_hits(universe, args.range, args.sleep, live=True)
            hits = [h for h in hits if hit_under_max_price(h, args.max_price)]
            fresh = recent_hits(hits, args.lookback_hours)
            sent = notify_hits(
                fresh,
                token=token,
                chat_id=chat_id,
                dry_run=args.dry_run,
                seed=args.seed_alert,
            )
            print(
                f"[{datetime.now(TPE).strftime('%H:%M:%S')}] "
                f"scan ok symbols={len(universe)} hits={len(hits)} recent={len(fresh)} sent={sent}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        wait_next_5m()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="台股五分K W底上MA20 通知")
    sub = p.add_subparsers(dest="cmd")

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--date", default="", help="成交額排名基準日 YYYYMMDD")
        sp.add_argument("--limit", type=int, default=100)
        sp.add_argument("--pool", type=int, default=200)
        sp.add_argument("--max-price", type=float, default=700, help="股價達此值以上剔除，預設 700")
        sp.add_argument("--symbols", default="", help="指定代號，例如 2327,2408")
        sp.add_argument("--range", dest="range_", default="5d")
        sp.add_argument("--sleep", type=float, default=0.2)

    s = sub.add_parser("scan", help="回看最近幾天的 W 底上 MA20")
    add_common(s)
    s.add_argument("--pages", action="store_true")
    s.add_argument("--html", default="")
    s.add_argument("--today", action="store_true", help="只留今天的訊號")
    s.add_argument("--days", type=int, default=0, help="只留最近 N 個日曆日，例如 7=一週")
    s.set_defaults(func=cmd_scan, range=None)

    a = sub.add_parser("alert", help="Telegram 輪詢")
    add_common(a)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=8.0)
    a.set_defaults(func=cmd_alert, range=None)

    add_common(p)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--seed-alert", action="store_true")
    p.add_argument("--lookback-hours", type=float, default=8.0)
    p.add_argument("--pages", action="store_true")
    p.add_argument("--html", default="")
    p.add_argument("--today", action="store_true", help="只留今天的訊號")
    p.add_argument("--days", type=int, default=0, help="只留最近 N 個日曆日，例如 7=一週")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "range", None) is None:
        args.range = args.range_
    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "alert":
        return cmd_alert(args)
    if args.pages or args.html:
        return cmd_scan(args)
    return cmd_alert(args)


if __name__ == "__main__":
    raise SystemExit(main())
