"""15 分 / 1 小時 MA200 剛站上 → Telegram（與回測同一套規則）。

15m / 1h：收盤 > MA7>25 且本根剛站上該週期 MA200。
15m 還要剛站上後連續 3 根收盤都在 MA200 上才推（訊號是第 3 根）。
15m Telegram 圖由上到下是 15m / 1h / 4h（4h 只對照、不擋單）。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from nq.binance import INTERVAL_MS, SESSION, fetch_klines, universe
from nq.ma15_bull import (
    add_15m_mas,
    above_htf_ma200,
    bar_above_ma200,
    confirm_reclaim_hold,
    detect_combo,
    htf_ma200_at,
    htf_ma25_not_down,
    htf_ma25_now_prev,
    htf_ma7_25_at,
    htf_ma7_25_stack,
    quality_reclaim,
    sma,
)

TZ = timezone(timedelta(hours=8))
_REPO = Path(__file__).resolve().parents[1]
SEEN_PATH = Path(os.environ.get("MA_BULL_SEEN", str(_REPO / "output" / "ma_bull_tg_seen.json")))
PAL = {7: "#f0c14a", 14: "#ff8a4c", 25: "#d28cff", 200: "#ffffff"}
DISPLAY = {
    "HK1810USDT": "小米",
    "CRCLUSDT": "CRCL",
    "HK0700USDT": "騰訊",
    "TENCENTUSDT": "騰訊",
    "MEITUANUSDT": "美團",
    "KUAISHOUUSDT": "快手",
    "POPMARTUSDT": "泡泡瑪特",
}

TF_WATCH = {
    "15m": {
        "signal": "15m",
        "htf": "1h",
        "htf_ms": INTERVAL_MS["1h"],
        "sig_limit": 280,
        "htf_limit": 250,
        "require_htf": False,
        "min_below": None,
        "min_vol": None,
        "max_ext": None,
        "max_rng24": None,
        "max_bars_above": None,
        "require_btc_1h": False,
        "require_h1_ma25_up": True,
        "require_h1_stack": True,
        "lookback": 48,
        "min_hold_bars": 3,
        "title": "15m MA200 上連 3 根",
    },
    "1h": {
        "signal": "1h",
        "htf": "4h",
        "htf_ms": INTERVAL_MS["4h"],
        "sig_limit": 420,
        "htf_limit": 250,
        "require_htf": False,
        "min_below": None,
        "min_vol": None,
        "max_ext": None,
        "max_rng24": None,
        "max_bars_above": None,
        "require_btc_1h": False,
        "require_h1_ma25_up": True,
        "require_h1_stack": False,
        "lookback": 80,
        "title": "1h 剛站上 MA200",
    },
}


def _load_telegram_local() -> None:
    for folder in (Path.cwd(), _REPO, Path(__file__).resolve().parent):
        f = folder / "telegram_local.py"
        if not f.is_file():
            continue
        ns: dict = {}
        exec(f.read_text(encoding="utf-8"), ns)
        tok = str(ns.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat = str(ns.get("TELEGRAM_CHAT_ID", "")).strip()
        if tok:
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", tok)
        if chat:
            os.environ.setdefault("TELEGRAM_CHAT_ID", chat)
        return


def apply_keys() -> None:
    _load_telegram_local()


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def file_base(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or f"s{abs(hash(symbol)) % 10_000_000_000}"


def sym_label(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    name = DISPLAY.get(symbol)
    if name and name != base:
        return f"{name} {base}"
    return base


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
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")


def _paint_ohlcv(ax, axv, d: dict, a0: int, a1: int, mark_i: int | None) -> None:
    from matplotlib.patches import Rectangle

    sl = slice(a0, a1)
    xs = np.arange(a1 - a0)
    o, h, l, c, v = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl]
    colors_v = []
    for k in range(len(c)):
        up = c[k] >= o[k]
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    axv.bar(xs, v, width=0.8, color=colors_v, linewidth=0)
    for n, col in PAL.items():
        ax.plot(xs, sma(d["c"], n)[sl], color=col, lw=1.35 if n == 200 else 1.15, label=f"MA{n}")
    if mark_i is not None and a0 <= mark_i < a1:
        x = mark_i - a0
        ax.axvline(x, color="#c9a227", ls="--", lw=0.95)
        ax.scatter([x], [c[x]], s=36, color="#c9a227", zorder=5)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=4)


def tf_bar_idx(d: dict, time_ms: int, bar_ms: int) -> int | None:
    t = d["t"]
    open_ms = int(time_ms) - (int(time_ms) % bar_ms)
    w = np.where(t == open_ms)[0]
    if len(w):
        return int(w[0])
    w = np.where(t <= time_ms)[0]
    return int(w[-1]) if len(w) else None


def _panel_title(sym: str, tf: str, d: dict, mark_i: int, extra: str = "", note: str = "") -> str:
    h_close = float(d["c"][mark_i])
    h_ma = float(d["m200"][mark_i]) if not np.isnan(d["m200"][mark_i]) else None
    vs = ""
    if h_ma:
        vs = f"  close {h_close:g} vs {tf} MA200 {h_ma:g} ({(h_close / h_ma - 1) * 100:+.2f}%)"
    return f"{sym}  {tf}  {hm(int(d['t'][mark_i]))}{vs}{extra}{note}"


def _save_stacked_chart(path: str, panels: list[tuple[dict, str, int, int, int, str]]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = len(panels)
    if n < 1:
        return None
    ratios = []
    for i in range(n):
        ratios.extend([3.1, 0.9] if i == 0 else [2.4, 0.75])
    fig, axes = plt.subplots(
        n * 2,
        1,
        figsize=(10.6, 5.4 + 4.6 * (n - 1)),
        sharex=False,
        gridspec_kw={"height_ratios": ratios},
        facecolor="#0c1210",
    )
    if n == 1:
        axes = [axes[0], axes[1]]
    for a in axes:
        _style_ax(a)
    for i, (frame, _tf, a0, a1, mark_i, title) in enumerate(panels):
        ax_p, ax_v = axes[2 * i], axes[2 * i + 1]
        _paint_ohlcv(ax_p, ax_v, frame, a0, a1, mark_i)
        ax_p.set_title(title, color="#e8f0ea", fontsize=11 if i == 0 else 12)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def load_4h_frame(sym: str) -> dict | None:
    raw = fetch_klines(sym, interval="4h", days=40, extra_bars=220)
    if raw is None or len(raw["c"]) < 200:
        return None
    return add_15m_mas(raw)


def draw_chart(
    sym: str,
    d: dict,
    sig,
    spec: dict,
    path: str,
    d_htf: dict | None,
    d_4h: dict | None = None,
) -> str | None:
    i = sig.idx
    ts = int(d["t"][i])
    a0 = max(0, i - spec["lookback"])
    a1 = min(len(d["c"]), i + 4)
    title_sym = file_base(sym) if any(ord(ch) >= 128 for ch in sym) else sym
    extra = (
        f"  below={sig.bars_below}  above={sig.bars_above}  "
        f"vol={sig.vol_ratio:.2f}x  ext={sig.ext_pct:+.2f}%"
    )
    panels: list[tuple[dict, str, int, int, int, str]] = [
        (d, spec["signal"], a0, a1, i, f"{title_sym}  {spec['signal']}  {hm(ts)}  reclaim MA200{extra}")
    ]
    hi = tf_bar_idx(d_htf, ts, spec["htf_ms"]) if d_htf is not None and len(d_htf.get("c", [])) else None
    if d_htf is not None and hi is not None:
        b0 = max(0, hi - 48)
        b1 = min(len(d_htf["c"]), hi + 2)
        note = "  compare only" if spec["htf"] == "4h" else ""
        panels.append((d_htf, spec["htf"], b0, b1, hi, _panel_title(title_sym, spec["htf"], d_htf, hi, note=note)))
    if spec["signal"] == "15m" and d_4h is not None and len(d_4h.get("c", [])):
        i4 = tf_bar_idx(d_4h, ts, INTERVAL_MS["4h"])
        if i4 is not None:
            c0 = max(0, i4 - 48)
            c1 = min(len(d_4h["c"]), i4 + 2)
            panels.append(
                (d_4h, "4h", c0, c1, i4, _panel_title(title_sym, "4h", d_4h, i4, note="  compare only"))
            )
    return _save_stacked_chart(path, panels)


def load_btc_1h() -> dict | None:
    raw = fetch_klines("BTCUSDT", interval="1h", limit=250, extra_bars=8)
    if raw is None or len(raw["c"]) < 200:
        return None
    return add_15m_mas(raw)


def passes_notify(sig, spec: dict, d, d_htf, ts: int, btc_1h: dict | None) -> bool:
    if spec["require_htf"] and not above_htf_ma200(d_htf, ts, sig.close, spec["htf_ms"]):
        return False
    if spec.get("require_btc_1h") and not bar_above_ma200(btc_1h, ts, INTERVAL_MS["1h"]):
        return False
    if spec.get("require_h1_ma25_up"):
        src = d if spec["signal"] == "1h" else d_htf
        if not htf_ma25_not_down(src, ts, sig.close, INTERVAL_MS["1h"]):
            return False
    if spec.get("require_h1_stack"):
        src = d if spec["signal"] == "1h" else d_htf
        if not htf_ma7_25_stack(src, ts, sig.close, INTERVAL_MS["1h"]):
            return False
    if spec.get("min_hold_bars"):
        return True
    return quality_reclaim(
        sig,
        min_below=spec["min_below"],
        min_vol=spec["min_vol"],
        max_ext=spec["max_ext"],
        max_rng24=spec.get("max_rng24"),
        max_bars_above=spec.get("max_bars_above"),
    )


def scan_symbol(sym: str, spec: dict, btc_1h: dict | None = None) -> list[dict]:
    raw = fetch_klines(sym, interval=spec["signal"], limit=spec["sig_limit"], extra_bars=8)
    if raw is None or len(raw["c"]) < 220:
        return []
    d = add_15m_mas(raw)
    raw_h = fetch_klines(sym, interval=spec["htf"], limit=spec["htf_limit"], extra_bars=8)
    d_htf = add_15m_mas(raw_h) if raw_h is not None and len(raw_h["c"]) >= 200 else None
    n = len(d["c"])
    combos = detect_combo(d, min_gap_bars=0)
    by_idx = {s.idx: s for s in combos}
    hold = spec.get("min_hold_bars")
    events = []
    for closed in (n - 1, n - 2):
        if closed < 200:
            continue
        if hold:
            start = closed - (hold - 1)
            if start < 200:
                continue
            sig0 = by_idx.get(start)
            if sig0 is None or not sig0.crossed_200:
                continue
            held = confirm_reclaim_hold(d, sig0, hold)
            if held is None or held.idx != closed:
                continue
            ts = int(d["t"][held.idx])
            if not passes_notify(held, spec, d, d_htf, ts, btc_1h):
                continue
            events.append({"symbol": sym, "sig": held, "d": d, "d_htf": d_htf, "spec": spec})
            continue
        sig = by_idx.get(closed)
        if sig is None:
            continue
        ts = int(d["t"][sig.idx])
        if not passes_notify(sig, spec, d, d_htf, ts, btc_1h):
            continue
        events.append({"symbol": sym, "sig": sig, "d": d, "d_htf": d_htf, "spec": spec})
    return events


def format_ev(ev: dict) -> str:
    d, sig, spec = ev["d"], ev["sig"], ev["spec"]
    sym = ev["symbol"]
    ts = hm(int(d["t"][sig.idx]))
    tf, htf = spec["signal"], spec["htf"]
    ma_h = (
        htf_ma200_at(ev.get("d_htf"), int(d["t"][sig.idx]), sig.close, spec["htf_ms"])
        if ev.get("d_htf") is not None
        else None
    )
    htxt = f"{htf} MA200 {ma_h:g}　距 {(sig.close / ma_h - 1) * 100:+.2f}%" if ma_h else f"{htf} MA200 —"
    if spec["require_htf"]:
        hline = f"且現價 &gt; {htxt}"
    else:
        hline = f"{htxt}（參考，不擋單）"
    extra = ""
    if spec.get("min_hold_bars"):
        extra += f"{tf} MA200 上已連 {spec['min_hold_bars']} 根（剛站上後維持）\n"
    elif not sig.crossed_200:
        extra += f"站上後第 {sig.bars_above} 根才收出 7&gt;25\n"
    if spec["min_below"] is not None:
        extra += (
            f"底下已跌 {sig.bars_below} 根　高低差 {sig.rng24:.2f}%　"
            f"量比 {sig.vol_ratio:.2f}×　距 {tf} MA200 {sig.ext_pct:+.2f}%\n"
        )
    else:
        extra += f"距 {tf} MA200 {sig.ext_pct:+.2f}%　量比 {sig.vol_ratio:.2f}×\n"
    if spec.get("require_btc_1h"):
        extra += "BTC 當時在 1h MA200 上\n"
    if spec.get("require_h1_ma25_up"):
        src = d if spec["signal"] == "1h" else ev.get("d_htf")
        now, prev = htf_ma25_now_prev(src, int(d["t"][sig.idx]), sig.close, INTERVAL_MS["1h"])
        if now is not None and prev is not None:
            extra += f"1h MA25 {now:g} ≥ 前一根 {prev:g}（未下彎）\n"
    if spec.get("require_h1_stack"):
        src = d if spec["signal"] == "1h" else ev.get("d_htf")
        m7, m25 = htf_ma7_25_at(src, int(d["t"][sig.idx]), sig.close, INTERVAL_MS["1h"])
        if m7 is not None and m25 is not None:
            extra += f"1h MA7 {m7:g} &gt; MA25 {m25:g}（多頭排列）\n"
    if spec["signal"] == "15m":
        extra += "圖附 4h 對照（不擋單）\n"
    return (
        f"<b>{spec['title']}</b>\n"
        f"<b>{sym_label(sym)}</b>  {sym}\n"
        f"{ts}  收 {sig.close:g}\n"
        f"收盤 &gt; MA7 {sig.m7:g} &gt; MA25 {sig.m25:g}\n"
        f"且 &gt; {tf} MA200 {sig.ma200:g}\n"
        f"{extra}"
        f"{hline}"
    )


def key_of(ev: dict) -> str:
    return f"{ev['spec']['signal']}:{ev['symbol']}:{int(ev['d']['t'][ev['sig'].idx])}"


def notify(ev: dict) -> None:
    text = format_ev(ev)
    print("\n" + text.replace("<b>", "").replace("</b>", "").replace("&gt;", ">"))
    spec = ev["spec"]
    tmp = Path(tempfile.gettempdir()) / f"ma_{spec['signal']}_{ev['symbol']}_{ev['sig'].idx}.png"
    d_4h = load_4h_frame(ev["symbol"]) if spec["signal"] == "15m" else None
    photo = draw_chart(ev["symbol"], ev["d"], ev["sig"], spec, str(tmp), ev.get("d_htf"), d_4h=d_4h)
    ok = telegram_send(text, photo=photo)
    if ok:
        print("  → Telegram 已送")
    else:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("  → 還沒填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，只印在這裡")
        else:
            print("  → Telegram 送出失敗，檢查 token 與 chat id")


def next_close_unix(interval_ms: int) -> float:
    now = time.time()
    step = interval_ms / 1000
    return (int(now) // int(step) + 1) * step + 3


def wait_next(specs: list[dict]) -> None:
    nxt = min(next_close_unix(INTERVAL_MS[s["signal"]]) for s in specs)
    time.sleep(max(1, nxt - time.time()))


def due_to_scan(spec: dict, *, force: bool) -> bool:
    if force or spec["signal"] == "15m":
        return True
    now = time.time()
    hour_close = (int(now) // 3600) * 3600
    return (now - hour_close) < 14 * 60


def test_telegram() -> int:
    apply_keys()
    ok = telegram_send(
        "MA200 監看測試\n"
        "15m：剛站上 15m MA200（收盤 > 7>25），連 3 根都在年線上，且 1h MA25 未下彎、1h 7>25\n"
        "1h：剛站上 1h MA200（收盤 > 7>25），且 1h MA25 未下彎\n"
        "如果你看到這則，Telegram 已通。"
    )
    print("Telegram 測試", "成功" if ok else "失敗（檢查 token / chat id）")
    return 0 if ok else 1


def spec_note(spec: dict) -> str:
    n = spec.get("max_bars_above")
    hold = spec.get("min_hold_bars")
    if n is not None:
        return (
            f"{spec['signal']} 剛站上 MA200，或站上後 {n} 根內收出 7>25"
            f"（{spec['htf']} 只畫圖、不擋單）"
        )
    return (
        f"{spec['signal']} 剛站上 MA200、收盤 > 7>25"
        + (f"，再連 {hold} 根都在 MA200 上" if hold else "")
        + ("，且 1h MA25 未下彎" if spec.get("require_h1_ma25_up") else "")
        + ("，且 1h 7>25" if spec.get("require_h1_stack") else "")
        + f"（{spec['htf']} SMA200 只畫圖、不擋單）"
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="15 分 / 1 小時 MA200 剛站上 → Telegram（與回測同一套規則）")
    p.add_argument("--once", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--stocks", action="store_true", help="只掃幣安 TradFi 股票永續（不含商品）")
    p.add_argument("--tf", choices=("15m", "1h", "both"), default="both", help="預設 15 分與 1 小時都監看")
    p.add_argument("--limit-symbols", type=int, default=0)
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    names = ("15m", "1h") if args.tf == "both" else (args.tf,)
    specs = [TF_WATCH[n] for n in names]

    seen = load_seen()
    print("載入標的…", flush=True)
    symbols = universe(stocks_only=args.stocks)
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    scope = "幣安股票永續" if args.stocks else "成交額前100永續"
    print(f"監看 {len(symbols)} 個{scope}。", flush=True)
    for spec in specs:
        print("  · " + spec_note(spec), flush=True)
    uni_ts = time.time()
    first = True

    def round_once() -> None:
        nonlocal symbols, uni_ts, first
        if time.time() - uni_ts > 1800:
            symbols = universe(stocks_only=args.stocks)
            if args.limit_symbols:
                symbols = symbols[: args.limit_symbols]
            uni_ts = time.time()
            print(f"更新標的 {len(symbols)}", flush=True)
        force = first or args.once
        first = False
        t0 = time.time()
        events = []
        jobs = [(s, spec) for spec in specs if due_to_scan(spec, force=force) for s in symbols]
        if not jobs:
            print(
                f"[{datetime.now(TZ).strftime('%H:%M:%S')}] 這輪沒有要掃的週期",
                flush=True,
            )
            return
        btc_1h = None
        if any(spec.get("require_btc_1h") for _, spec in jobs):
            btc_1h = load_btc_1h()
            if btc_1h is None:
                print("警告：抓不到 BTC 1h，這輪 1h 大盤過濾全不過", flush=True)
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(scan_symbol, sym, spec, btc_1h): (sym, spec) for sym, spec in jobs}
            for fut in as_completed(futs):
                try:
                    events.extend(fut.result())
                except Exception as e:
                    sym, spec = futs[fut]
                    print("err", spec["signal"], sym, e, flush=True)
        new = [e for e in events if key_of(e) not in seen]
        scanned = sorted({spec["signal"] for _, spec in jobs})
        print(
            f"[{datetime.now(TZ).strftime('%H:%M:%S')}] "
            f"掃完 {len(symbols)} × {'+'.join(scanned)} 用 {time.time()-t0:.1f}s　新訊號 {len(new)}",
            flush=True,
        )
        for ev in new:
            seen.add(key_of(ev))
            notify(ev)
        if new:
            save_seen(seen)

    round_once()
    if args.once:
        return 0
    print(f"watch 中（{' + '.join(s['signal'] for s in specs)}），收盤掃一次（Ctrl+C 停）", flush=True)
    try:
        while True:
            wait_next(specs)
            round_once()
    except KeyboardInterrupt:
        print("\n已停止。")
        save_seen(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
