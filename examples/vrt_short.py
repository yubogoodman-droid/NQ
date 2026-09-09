#!/usr/bin/env python3
"""VRTUSDT 永續空單快照：15m / 日線圖 + 手機 HTML。

    python3 examples/vrt_short.py
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.patches import Rectangle

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
ROOT = Path(__file__).resolve().parents[1]
OUT_HTML = ROOT / "docs" / "binance" / "vrt-short.html"
OUT_IMG = ROOT / "docs" / "binance" / "img"
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"}
)

FONT = None
for p in (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(p).exists():
        fm.fontManager.addfont(p)
        FONT = fm.FontProperties(fname=p)
        plt.rcParams["font.family"] = FONT.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

PAL = {7: "#f0c14a", 14: "#26c6da", 25: "#d28cff", 99: "#42a5f5", 120: "#5fd2c2", 200: "#e8f0ea"}
BG, PANEL, INK, MUTED = "#0c1210", "#101814", "#e8f0ea", "#8aa193"
UP, DN, ACCENT = "#3dba7a", "#e35d5d", "#c9a227"


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def fetch_klines(interval: str, limit: int = 300) -> dict:
    raw = get_json("/fapi/v1/klines", {"symbol": "VRTUSDT", "interval": interval, "limit": limit})
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
        "qv": np.array([float(x[7]) for x in raw]),
    }


def hm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def ymd(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m-%d")


def last_sma(c: np.ndarray, n: int) -> float | None:
    s = sma(c, n)
    v = s[-1]
    return None if np.isnan(v) else float(v)


def fmt(n: float | None, d: int = 2) -> str:
    if n is None:
        return "—"
    return f"{n:.{d}f}"


def draw_candles(ax, xs, o, h, l, c) -> list[str]:
    colors = []
    for k in range(len(c)):
        col = UP if c[k] >= o[k] else DN
        ax.vlines(xs[k], l[k], h[k], color=col, lw=0.7)
        y0, y1 = min(o[k], c[k]), max(o[k], c[k])
        if y1 == y0:
            y1 = y0 + max(h[k] - l[k], 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.3))
        colors.append(col + "99")
    return colors


def style_axes(*axes) -> None:
    for a in axes:
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED, labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")
        a.grid(True, color="#1c2a24", lw=0.5)


def xticks(ax, t: np.ndarray, step: int, fmt_fn) -> None:
    idx = list(range(0, len(t), step))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([fmt_fn(int(t[i])) for i in idx], color=MUTED)


def draw_15m(d: dict, levels: dict, path: Path) -> None:
    # 今日 15m + 前一晚一點上下文
    start = None
    for i, ms in enumerate(d["t"]):
        if ymd(int(ms)) == ymd(int(d["t"][-1])):
            start = max(0, i - 8)
            break
    if start is None:
        start = max(0, len(d["c"]) - 64)
    sl = slice(start, len(d["c"]))
    xs = np.arange(len(d["c"][sl]))
    o, h, l, c, v, t = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl], d["t"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.4, 6.1), sharex=True, gridspec_kw={"height_ratios": [3.15, 1]}, facecolor=BG
    )
    style_axes(ax, axv)
    cols = draw_candles(ax, xs, o, h, l, c)
    axv.bar(xs, v, width=0.8, color=cols, linewidth=0)
    full_c = d["c"]
    for n, col in PAL.items():
        ax.plot(xs, sma(full_c, n)[sl], color=col, lw=1.08, label=f"MA{n}")
    for key, y, color, ls in (
        ("or_high", levels["or_high"], ACCENT, "--"),
        ("or2", levels["or2"], DN, ":"),
        ("shelf", levels["shelf"], "#7eb6ff", "--"),
        ("t1", levels["t1"], UP, ":"),
    ):
        ax.axhline(y, color=color, ls=ls, lw=0.85, alpha=0.9)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    kw = {"fontproperties": FONT} if FONT else {}
    ax.set_title("VRTUSDT  15m · 美股開盤瀑布", color=INK, fontsize=12, **kw)
    axv.set_ylabel("量", color=MUTED, fontsize=8, **kw)
    xticks(axv, t, 8, hm)
    fig.tight_layout(pad=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_daily(d: dict, path: Path) -> None:
    n = min(40, len(d["c"]))
    sl = slice(-n, None)
    xs = np.arange(n)
    o, h, l, c, v, t = d["o"][sl], d["h"][sl], d["l"][sl], d["c"][sl], d["v"][sl], d["t"][sl]
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.15, 1]}, facecolor=BG
    )
    style_axes(ax, axv)
    cols = draw_candles(ax, xs, o, h, l, c)
    axv.bar(xs, v, width=0.8, color=cols, linewidth=0)
    for nma, col in ((7, PAL[7]), (14, PAL[14]), (25, PAL[25])):
        ax.plot(xs, sma(d["c"], nma)[-n:], color=col, lw=1.1, label=f"MA{nma}")
    ax.axhline(259.0, color=ACCENT, ls=":", lw=0.85)
    ax.axhline(269.2, color=UP, ls=":", lw=0.85)
    kw = {"fontproperties": FONT} if FONT else {}
    ax.set_title("VRTUSDT  日線 · 近 40 日", color=INK, fontsize=12, **kw)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=3)
    xticks(axv, t, 5, lambda ms: datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d"))
    fig.tight_layout(pad=0.55)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def collect() -> dict:
    ticker = get_json("/fapi/v1/ticker/24hr", {"symbol": "VRTUSDT"})
    prem = get_json("/fapi/v1/premiumIndex", {"symbol": "VRTUSDT"})
    oi = get_json("/fapi/v1/openInterest", {"symbol": "VRTUSDT"})
    ls = get_json("/futures/data/globalLongShortAccountRatio", {"symbol": "VRTUSDT", "period": "15m", "limit": 3})
    tpos = get_json("/futures/data/topLongShortPositionRatio", {"symbol": "VRTUSDT", "period": "15m", "limit": 3})
    taker = get_json("/futures/data/takerlongshortRatio", {"symbol": "VRTUSDT", "period": "15m", "limit": 4})
    k15 = fetch_klines("15m", 220)
    k1h = fetch_klines("1h", 80)
    k1d = fetch_klines("1d", 60)

    last = float(ticker["lastPrice"])
    hi = float(ticker["highPrice"])
    lo = float(ticker["lowPrice"])
    chg = float(ticker["priceChangePercent"])
    qv = float(ticker["quoteVolume"])
    oi_c = float(oi["openInterest"])

    # 美東 09:30 = 台 21:30 那根 15m
    or_i = None
    for i, ms in enumerate(k15["t"]):
        dt = datetime.fromtimestamp(int(ms) / 1000, TZ)
        if dt.strftime("%Y-%m-%d %H:%M") == ymd(int(k15["t"][-1])) + " 21:30":
            or_i = i
            break
    if or_i is None:
        or_i = int(np.argmax(k15["h"][-20:]) + len(k15["h"]) - 20)
    or_high = float(k15["h"][or_i])
    or_low = float(k15["l"][or_i])
    or_rng = or_high - or_low
    or15 = or_high - 1.5 * or_rng
    or2 = or_high - 2.0 * or_rng
    or25 = or_high - 2.5 * or_rng

    # 破位平台：開盤前 20:00–21:15 的低點區
    shelf = 286.5
    for i, ms in enumerate(k15["t"]):
        dt = datetime.fromtimestamp(int(ms) / 1000, TZ)
        if dt.strftime("%H:%M") == "20:00" and ymd(int(ms)) == ymd(int(k15["t"][-1])):
            shelf = float(np.min(k15["l"][i : i + 5]))  # 20:00–21:00 盤整低
            break

    trs = []
    for i in range(len(k1d["c"])):
        if i == 0:
            trs.append(k1d["h"][i] - k1d["l"][i])
        else:
            prev = k1d["c"][i - 1]
            trs.append(max(k1d["h"][i] - k1d["l"][i], abs(k1d["h"][i] - prev), abs(k1d["l"][i] - prev)))
    atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
    day_rng = hi - lo

    d7 = None
    if len(k1d["c"]) >= 8:
        d7 = (k1d["c"][-1] / k1d["c"][-8] - 1) * 100
    d30 = None
    if len(k1d["c"]) >= 31:
        d30 = (k1d["c"][-1] / k1d["c"][-31] - 1) * 100

    return {
        "asof": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
        "last": last,
        "chg": chg,
        "hi": hi,
        "lo": lo,
        "qv": qv,
        "oi": oi_c,
        "oi_usd": oi_c * last,
        "mark": float(prem["markPrice"]),
        "index": float(prem["indexPrice"]),
        "funding": float(prem.get("lastFundingRate") or 0),
        "ls": float(ls[-1]["longShortRatio"]) if ls else None,
        "tpos": float(tpos[-1]["longShortRatio"]) if tpos else None,
        "taker": float(taker[-1]["buySellRatio"]) if taker else None,
        "taker_prev": [float(x["buySellRatio"]) for x in taker] if taker else [],
        "ma": {n: last_sma(k15["c"], n) for n in (7, 14, 25, 99, 120, 200)},
        "ma_d": {n: last_sma(k1d["c"], n) for n in (7, 14, 25)},
        "or_high": or_high,
        "or_low": or_low,
        "or_rng": or_rng,
        "or15": or15,
        "or2": or2,
        "or25": or25,
        "shelf": shelf,
        "atr": atr,
        "day_rng": day_rng,
        "d7": d7,
        "d30": d30,
        "h1": float(k1h["c"][-1]),
        "k15": k15,
        "k1d": k1d,
        "t1": 269.2,
        "t2": 259.0,
        "t3": 249.7,
    }


def html_page(s: dict, img15: Path, imgd: Path) -> str:
    last, chg = s["last"], s["chg"]
    chase = last <= s["or2"] + 1.5
    verdict = "方向空，不要追低" if chase else "方向空，等反彈再空"
    ma = s["ma"]
    ls = fmt(s["ls"], 2)
    tpos = fmt(s["tpos"], 2)
    d7 = f"{s['d7']:+.2f}%" if s["d7"] is not None else "—"
    d30 = f"{s['d30']:+.2f}%" if s["d30"] is not None else "—"
    bounce = f"{ma[7]:.1f}" if ma[7] else "277"
    mid = f"{ma[14]:.1f}" if ma[14] else "282"
    m25 = f"{ma[25]:.1f}" if ma[25] else "285"

    def kpi(k, v, cls=""):
        return f'<div class="kpi"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'

    chg_cls = "neg" if chg < 0 else "pos"
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>VRTUSDT 空 · {s['asof']}</title>
<style>
:root{{--bg:#0c1210;--panel:#14201b;--ink:#e8f0ea;--muted:#8aa193;--line:rgba(232,240,234,.12);--long:#3dba7a;--short:#e35d5d;--accent:#c9a227}}
*{{box-sizing:border-box}}
body{{margin:0;background:linear-gradient(165deg,#0c1210,#14201b 45%,#0a0f0d);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC",sans-serif}}
.wrap{{max-width:560px;margin:0 auto;padding:16px 14px 40px}}
h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:var(--muted);font-size:13px;line-height:1.55;margin:0 0 14px}}
.banner{{background:rgba(227,93,93,.12);border:1px solid rgba(227,93,93,.35);border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.banner b{{display:block;font-size:18px;color:var(--short);margin-bottom:6px}}
.banner p{{margin:0;color:var(--ink);font-size:14px;line-height:1.55}}
.kpis{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}}
.kpi{{border:1px solid var(--line);background:rgba(20,32,27,.72);border-radius:12px;padding:10px 12px}}
.kpi .k{{color:var(--muted);font-size:11px}}
.kpi .v{{font-size:18px;margin-top:3px;font-variant-numeric:tabular-nums}}
.pos{{color:var(--long)}} .neg{{color:var(--short)}} .mark{{color:var(--accent)}}
.card{{background:rgba(20,32,27,.72);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:14px}}
.card h2{{font-size:15px;margin:0 0 8px}}
.card p,.card li{{font-size:14px;line-height:1.6;color:#d5e0d8}}
.card ol,.card ul{{margin:0;padding-left:1.2em}}
img{{width:100%;height:auto;display:block;border-radius:10px;background:#101814}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 4px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:500}}
.note{{color:var(--muted);font-size:12px;line-height:1.5;margin:10px 0 0}}
</style>
</head>
<body>
<div class="wrap">
  <h1>VRTUSDT 永續 · 空</h1>
  <p class="sub">Vertiv 股票永續 · 快照 {s['asof']}（台北）· 不是進出場建議</p>
  <div class="banner">
    <b>{verdict}</b>
    <p>美東開盤那根 15m 高 {fmt(s['or_high'])} / 低 {fmt(s['or_low'])}，2 倍開盤區間目標在 {fmt(s['or2'])}，現價 {fmt(last)} 已經走到。瀑布發生了，現在市價空是追刀。要空就等反彈。</p>
  </div>
  <div class="kpis">
    {kpi("現價", fmt(last), chg_cls)}
    {kpi("24h", f"{chg:+.2f}%", chg_cls)}
    {kpi("24h 高 / 低", f"{fmt(s['hi'])} / {fmt(s['lo'])}")}
    {kpi("今日振幅 / ATR14", f"{fmt(s['day_rng'])} / {fmt(s['atr'])}")}
    {kpi("7 日 / 30 日", f"{d7} / {d30}")}
    {kpi("成交額 24h", f"{s['qv']/1e6:.2f}M")}
    {kpi("OI", f"{s['oi_usd']/1e3:.0f}k USDT")}
    {kpi("帳戶多空 / 大戶倉", f"{ls} / {tpos}")}
  </div>

  <div class="card">
    <h2>15 分 K</h2>
    <img src="data:image/png;base64,{b64(img15)}" alt="VRT 15m"/>
    <p class="note">黃虛線開盤高 {fmt(s['or_high'])} · 紅點線 2×OR {fmt(s['or2'])} · 藍虛線破位平台 {fmt(s['shelf'])} · 綠點線日線目標 {fmt(s['t1'])}</p>
  </div>

  <div class="card">
    <h2>為什麼現價不追空</h2>
    <ol>
      <li>從 {fmt(s['or_high'])} 殺到 {fmt(last)}，開盤區間已走滿 2 倍，短線目標兌現。</li>
      <li>15m 全在均線下：MA7 {fmt(ma[7])} / MA14 {fmt(ma[14])} / MA25 {fmt(ma[25])} / MA200 {fmt(ma[200])}。空頭排列成立，但進場點在末端。</li>
      <li>今日振幅 {fmt(s['day_rng'])} ≈ {s['day_rng']/s['atr']:.1f} 倍 ATR，是趨勢日，不是剛破位。</li>
      <li>24h 成交只有 {s['qv']/1e6:.2f}M、OI 約 {s['oi_usd']/1e3:.0f}k，TradFi 永續很薄，低點市價空容易滑價。</li>
    </ol>
  </div>

  <div class="card">
    <h2>要空就這樣排</h2>
    <table>
      <tr><th>區</th><th>價</th><th>用途</th></tr>
      <tr><td>反彈空 A</td><td>{bounce}（15m MA7）</td><td>第一個能做的回檔空</td></tr>
      <tr><td>反彈空 B</td><td>{mid}–{m25}</td><td>較佳：MA14/25 + 供給</td></tr>
      <tr><td>反彈空 C</td><td>{fmt(s['shelf'])}–287</td><td>最好：破位平台失敗</td></tr>
      <tr><td>停損</td><td>15m 收過 287，或假高 {fmt(s['or_high'])}</td><td>結構壞掉就走</td></tr>
      <tr><td>目標 1</td><td>{fmt(s['t1'])}</td><td>日線 MA14/25、9/3 收</td></tr>
      <tr><td>目標 2</td><td>{fmt(s['t2'])}</td><td>8/28–31 平台</td></tr>
      <tr><td>目標 3</td><td>{fmt(s['t3'])}</td><td>9/1 低；2.5×OR {fmt(s['or25'])}</td></tr>
    </table>
    <p class="note">現價 {fmt(last)} 空到 269 只有約 {fmt(last-s['t1'])} 點，停損若放 {bounce} 要承擔約 {fmt(float(bounce)-last)} 點，盈虧比不夠。</p>
  </div>

  <div class="card">
    <h2>日線位置</h2>
    <img src="data:image/png;base64,{b64(imgd)}" alt="VRT daily"/>
    <p>昨天衝到 {fmt(s['hi'])} 附近把收購利多一次漲完，今天把那根陽線吐掉。日線 MA7 {fmt(s['ma_d'][7])}，MA14 {fmt(s['ma_d'][14])}、MA25 {fmt(s['ma_d'][25])} 還在 269 附近，那是空單第一目標，不是已經跌完。</p>
  </div>

  <div class="card">
    <h2>倉位與盤勢</h2>
    <ul>
      <li>帳戶多空比 {ls}、大戶持倉比 {tpos}，多頭仍擠，下跌還有燃料。</li>
      <li>主動買賣比最近 {fmt(s['taker'], 2)}（&lt;1 偏賣）。</li>
      <li>標記 {fmt(s['mark'])} / 指數 {fmt(s['index'])}，溢價約 {fmt(s['mark']-s['index'])}。</li>
      <li>NVDA 同期也弱，這是美股開盤後的股票永續，不是幣圈獨立行情。</li>
    </ul>
  </div>

  <p class="note">僅供型態對照。VRTUSDT 是股票永續，流動性差，實盤請自己控倉、滑價與保證金。資料來自幣安 U 本位，快照時間 {s['asof']}。</p>
</div>
</body>
</html>
"""


def main() -> None:
    s = collect()
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    p15 = OUT_IMG / "vrt_15m.png"
    pdaily = OUT_IMG / "vrt_1d.png"
    draw_15m(s["k15"], s, p15)
    draw_daily(s["k1d"], pdaily)
    OUT_HTML.write_text(html_page(s, p15, pdaily), encoding="utf-8")
    snap = {k: v for k, v in s.items() if k not in ("k15", "k1d")}
    print(json.dumps(snap, indent=2, ensure_ascii=False, default=str))
    print("wrote", OUT_HTML)


if __name__ == "__main__":
    main()
