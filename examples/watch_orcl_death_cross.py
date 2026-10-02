#!/usr/bin/env python3
"""ORCL 1m 空：MA7/MA14 死亡交叉且收盤破 MA25 → Telegram。

對 2026-10-02 那張圖：高點 144.95（22:44 台北）後，
22:49 收盤 144.44 同時 MA7 下穿 MA14、收盤跌破 MA25，之後砸到 141。
截圖 23:48（收 141.34、MA7 141.55 / MA14 141.78 / MA25 141.84）是訊號後的結果。

用法：
  python3 examples/watch_orcl_death_cross.py --scan
  python3 examples/watch_orcl_death_cross.py --test
  python3 examples/watch_orcl_death_cross.py --dry-run --once
  python3 examples/watch_orcl_death_cross.py

Telegram 憑證放 tg_config.env（勿提交），或本檔最上面。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
DEFAULT_SYMBOLS = ("ORCLUSDT",)
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


def detect_shorts(
    close: np.ndarray,
    m7: np.ndarray,
    m14: np.ndarray,
    m25: np.ndarray,
    *,
    min_lead: int = 5,
    require_cross_ma25: bool = False,
) -> list[ShortHit]:
    """收盤根：MA7 下穿 MA14，且收盤 < MA25。"""
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
        hits.append(
            ShortHit(
                i=i,
                close=float(close[i]),
                m7=float(m7[i]),
                m14=float(m14[i]),
                m25=float(m25[i]),
                lead=lead,
                crossed_ma25=bool(crossed),
            )
        )
    return hits


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
    ax.set_title(f"{sym}  1m  空  MA7×MA14 死亡交叉且破 MA25", color="#e8f0ea", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#c8d5cc", ncol=3)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_hit(sym: str, d: dict, hit: ShortHit) -> str:
    ts = hm(int(d["t"][hit.i]))
    x25 = "這根同時跌破 MA25" if hit.crossed_ma25 else "收盤已在 MA25 下方"
    link = f"https://www.binance.com/zh-TW/futures/{sym}"
    return (
        f"🔻 <b>{sym} 空</b>  1m\n"
        f"MA7 / MA14 <b>死亡交叉</b>，{x25}\n"
        f"時間 {ts}（台北）\n"
        f"收 {hit.close:g}\n"
        f"MA7 {hit.m7:.4f}　MA14 {hit.m14:.4f}　MA25 {hit.m25:.4f}\n"
        f"MA7 領先 {hit.lead} 根後下穿\n"
        f'<a href="{link}">{sym}</a>'
    )


def key_of(sym: str, d: dict, hit: ShortHit) -> str:
    return f"{sym}:{int(d['t'][hit.i])}"


def scan_symbol(sym: str, *, min_lead: int, require_cross_ma25: bool, limit: int) -> tuple[dict, list[ShortHit]]:
    raw = fetch_klines(sym, limit=limit)
    if raw is None:
        return {}, []
    d = with_ma(raw)
    return d, detect_shorts(
        d["c"], d["m7"], d["m14"], d["m25"], min_lead=min_lead, require_cross_ma25=require_cross_ma25
    )


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


def print_scan(sym: str, d: dict, hits: list[ShortHit], *, hours: int) -> None:
    if not d:
        print(f"{sym} 沒資料")
        return
    cutoff = int(d["t"][-1]) - hours * 3600 * 1000
    recent = [h for h in hits if int(d["t"][h.i]) >= cutoff]
    print(f"\n{sym} 近 {hours}h  死亡交叉且破 MA25：{len(recent)} 筆")
    for h in recent:
        x = "同根破25" if h.crossed_ma25 else "已在25下"
        print(
            f"  {hm(int(d['t'][h.i]))}  收 {h.close:g}  "
            f"MA7 {h.m7:.4f}  MA14 {h.m14:.4f}  MA25 {h.m25:.4f}  "
            f"lead {h.lead}  {x}"
        )


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


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="ORCL 1m MA7/MA14 死亡交叉且破 MA25 → Telegram")
    p.add_argument("--symbols", default="ORCLUSDT", help="逗號分隔，預設 ORCLUSDT")
    p.add_argument("--min-lead", type=int, default=5, help="死亡交叉前 MA7≥MA14 最少根數，濾雜訊")
    p.add_argument("--require-cross-ma25", action="store_true", help="要求這根同時由上跌破 MA25")
    p.add_argument("--scan", action="store_true", help="印出近幾小時歷史訊號後結束")
    p.add_argument("--hours", type=int, default=24, help="--scan 回看小時數")
    p.add_argument("--limit", type=int, default=1500, help="K 線根數")
    p.add_argument("--once", action="store_true", help="只掃剛收盤的那一分，然後結束")
    p.add_argument("--test", action="store_true", help="只測 Telegram 通不通")
    p.add_argument("--dry-run", action="store_true", help="只印不送 Telegram")
    args = p.parse_args()
    apply_keys()
    if args.test:
        return test_telegram()

    symbols = parse_symbols(args.symbols)
    if args.scan:
        for sym in symbols:
            d, hits = scan_symbol(
                sym, min_lead=args.min_lead, require_cross_ma25=args.require_cross_ma25, limit=args.limit
            )
            print_scan(sym, d, hits, hours=args.hours)
        return 0

    seen = load_seen()
    print(
        f"監看 {', '.join(symbols)}  1m  空  "
        f"MA7×MA14 死亡交叉且收盤<MA25  min_lead={args.min_lead}",
        flush=True,
    )

    def round_once() -> None:
        for sym in symbols:
            d, hits = scan_symbol(
                sym, min_lead=args.min_lead, require_cross_ma25=args.require_cross_ma25, limit=max(120, args.limit)
            )
            if not hits:
                print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {sym} 無訊號", flush=True)
                continue
            # 只推剛收盤、以及前一根（怕整點掃晚了）
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


if __name__ == "__main__":
    raise SystemExit(main())
