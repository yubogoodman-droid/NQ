#!/usr/bin/env python3
"""Phone-card gallery for structure shorts, matching the MA25 view.html template."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from generate_10d_charts import load_range
from generate_signal_charts import STEM_ALIAS, file_stem, resample_ohlcv_15m

CACHE = Path("/tmp/binance_um_klines")
REPO = Path("/workspace")
BRANCH = "cursor/shadow-neckline-backtest-bdfc"
PAD_HOURS = 14
EXIT_BARS_8H = 96  # 5m × 96
MA_COLORS = {
    7: "#f0c14a",
    14: "#d28cff",  # 頸線確認用，加粗（對齊模板粉線 MA25）
    25: "#26a69a",
    99: "#42a5f5",
    200: "#c9a227",
}


def _setup_cjk() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break


def _style_ax(ax) -> None:
    ax.set_facecolor("#101814")
    ax.tick_params(colors="#8aa193", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3a33")


def _paint_candles(ax, xs, o, h, l, c):
    from matplotlib.patches import Rectangle

    colors = []
    for k in range(len(xs)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0 = min(float(o.iloc[k]), float(c.iloc[k]))
        y1 = max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(
            Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25)
        )
        colors.append("#3dba7a99" if up else "#e35d5d99")
    return colors


def _loc_on_tf(index: pd.Index, ts_ms: int) -> int | None:
    if len(index) == 0:
        return None
    pos = int(index.searchsorted(ts_ms, side="right") - 1)
    if 0 <= pos < len(index):
        return pos
    return None


def _fmt_px(v: float) -> str:
    if v >= 100:
        return f"{v:.2f}"
    if v >= 1:
        return f"{v:.4g}"
    return f"{v:.6g}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):+.2f}%"


def load_symbol_df(symbol: str, start: str, end: str) -> pd.DataFrame:
    stem = file_stem(symbol)
    df = load_range(stem, start, end)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for n in (7, 14, 25, 99, 200):
        df[f"sma{n}"] = df["close"].rolling(n, min_periods=n).mean()
    return df


def draw_trade_png(df: pd.DataFrame, row: pd.Series, path: Path, trade_no: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk()
    sig_ts = int(pd.Timestamp(row["time_utc"], tz="UTC").timestamp() * 1000)
    idxs = df.index[df["timestamp"] == sig_ts].tolist()
    if not idxs:
        nearest = (df["timestamp"] - sig_ts).abs().idxmin()
        idxs = [int(nearest)]
    i = int(idxs[0])
    entry_i = i + 1 if i + 1 < len(df) else i
    exit_i = i + EXIT_BARS_8H if i + EXIT_BARS_8H < len(df) else min(len(df) - 1, i + 48)
    pad = PAD_HOURS * 12  # 5m bars per hour
    start = max(0, i - pad)
    end = min(len(df) - 1, max(exit_i, i) + 18)
    window = df.iloc[start : end + 1].reset_index(drop=True)
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None

    fig, (ax, axv, ax15) = plt.subplots(
        3,
        1,
        figsize=(10.4, 8.0),
        gridspec_kw={"height_ratios": [3.0, 0.75, 2.15]},
        facecolor="#0c1210",
    )
    ax.sharex(axv)
    for a in (ax, axv, ax15):
        _style_ax(a)

    colors_v = _paint_candles(ax, xs, o, h, l, c)
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        key = f"sma{n}"
        if key not in window.columns:
            continue
        lw = 2.15 if n == 14 else (1.25 if n <= 25 else 1.0)
        ax.plot(list(xs), window[key], color=col, lw=lw, label=f"MA{n}")

    if pd.notna(row.get("line_val")):
        ax.axhline(float(row["line_val"]), color="#e35d5d", ls=":", lw=1.0, alpha=0.85, label="頸線")

    bx, ex, xx = i - start, entry_i - start, exit_i - start
    if 0 <= bx < len(window):
        ax.axvline(bx, color="#ff4d4f", ls="--", lw=0.9, alpha=0.8)
        ax.scatter([bx], [float(window["low"].iloc[bx])], s=42, color="#ff4d4f", marker="v", zorder=6)
        ax.annotate(
            "訊號",
            (bx, float(window["low"].iloc[bx])),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            color="#ff7b72",
            fontsize=8,
        )
    entry_px = float(row["entry"]) if pd.notna(row.get("entry")) else float(window["open"].iloc[max(0, ex)])
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#ffd666", ls="--", lw=0.9)
        ax.scatter([ex], [entry_px], s=46, color="#ffd666", marker="o", zorder=6)
        ax.annotate(
            "進場",
            (ex, entry_px),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#ffd666",
            fontsize=8,
        )
    if 0 <= xx < len(window):
        exit_px = float(window["close"].iloc[xx])
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        pnl8 = row.get("pnl_8h")
        win = pd.notna(pnl8) and float(pnl8) > 0
        ax.scatter(
            [xx],
            [exit_px],
            s=40,
            color="#00c805" if win else "#ff5252",
            marker="x",
            zorder=6,
        )
        ax.annotate(
            "8h",
            (xx, exit_px),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#f0c14b",
            fontsize=8,
        )

    pnl8 = row.get("pnl_8h")
    sign = ""
    pnl_txt = "—"
    if pd.notna(pnl8):
        sign = "+" if float(pnl8) >= 0 else ""
        pnl_txt = f"{sign}{float(pnl8):.2f}%"
    t0 = str(row["time_utc"])[5:]
    ax.set_title(
        f"#{trade_no}  {row['symbol']}  5m  {t0}  8h {pnl_txt}",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [
            datetime.fromtimestamp(int(window["timestamp"].iloc[j]) / 1000, tz=timezone.utc).strftime(
                "%m-%d %H:%M"
            )
            for j in ticks
        ],
        color="#8aa193",
    )

    lo_ms, hi_ms = int(window["timestamp"].iloc[0]), int(window["timestamp"].iloc[-1])
    tf15 = resample_ohlcv_15m(df)
    w15 = tf15[(tf15["timestamp"] >= lo_ms) & (tf15["timestamp"] <= hi_ms)].copy()
    if len(w15) >= 2:
        for n in (7, 14, 25, 99, 200):
            tf15[f"sma{n}"] = tf15["close"].rolling(n, min_periods=n).mean()
        w15 = tf15[(tf15["timestamp"] >= lo_ms) & (tf15["timestamp"] <= hi_ms)].copy().reset_index(drop=True)
        xs15 = range(len(w15))
        _paint_candles(ax15, xs15, w15["open"], w15["high"], w15["low"], w15["close"])
        for n, col in MA_COLORS.items():
            key = f"sma{n}"
            if key not in w15.columns:
                continue
            lw = 2.15 if n == 14 else 0.95
            ax15.plot(list(xs15), w15[key], color=col, lw=lw, label=f"MA{n}")
        for ts_ms, col, mark in (
            (sig_ts, "#ff4d4f", "訊號"),
            (int(df.loc[entry_i, "timestamp"]), "#ffd666", "進場"),
        ):
            px = _loc_on_tf(w15["timestamp"], ts_ms)
            if px is not None:
                ax15.axvline(px, color=col, ls="--", lw=0.85, alpha=0.85)
                if mark:
                    ax15.scatter([px], [float(w15["close"].iloc[px])], s=28, color=col, zorder=5)
        ax15.text(
            0.01,
            0.92,
            "15m 對照",
            transform=ax15.transAxes,
            color="#c8d5cc",
            fontsize=9,
            va="top",
        )
        ax15.legend(loc="upper right", fontsize=6, frameon=False, labelcolor="#c8d5cc", ncol=6)
        step15 = max(1, len(w15) // 6)
        ticks15 = list(range(0, len(w15), step15))
        ax15.set_xticks(ticks15)
        ax15.set_xticklabels(
            [
                datetime.fromtimestamp(int(w15["timestamp"].iloc[j]) / 1000, tz=timezone.utc).strftime(
                    "%m-%d %H:%M"
                )
                for j in ticks15
            ],
            color="#8aa193",
        )
    else:
        ax15.set_visible(False)

    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _equity_svg(pnls: list[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(0.2, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys_src[i]):.1f}" for i in range(len(ys_src)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def _page_css() -> str:
    return """*{box-sizing:border-box}
body{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}
.page{max-width:560px;margin:0 auto;padding:14px 12px 32px}
h1{font-size:18px;margin:0 0 6px}
.muted{color:#8b949e;font-size:13px;line-height:1.5}
.summary{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.card{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}
.card b{display:block;font-size:20px;margin-top:4px}
.equity{margin:10px 0 4px}
.trade-card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}
.card-header{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}
.trade-no{font-size:15px;font-weight:700}
.trade-time{font-size:12px;color:#8b949e}
.card-pnl{font-size:16px;font-weight:700;white-space:nowrap}
.pnl-win{color:#00c805} .pnl-loss{color:#ff5252} .pnl-flat{color:#8b949e}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.tag{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}
.tag-tp{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}
.tag-sl{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}
.tag-time{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}
.tag-info{background:rgba(210,140,255,0.12);color:#e2b6ff;border-color:rgba(210,140,255,0.28)}
.tag-fresh{background:rgba(88,166,255,0.16);color:#79c0ff;border-color:rgba(88,166,255,0.35)}
.trade-detail{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}
.mini-chart{margin:0 -6px -4px;border-radius:10px;overflow:hidden}
.back{display:inline-block;margin-bottom:10px;color:#8b949e;font-size:13px;text-decoration:none}
.back:hover{color:#e6edf3}
"""


def wr_avg(s: pd.Series) -> tuple[float | None, float | None]:
    x = s.dropna()
    if len(x) == 0:
        return None, None
    return float((x > 0).mean() * 100), float(x.mean())


def render_gallery(
    csv_path: Path,
    out_html: Path,
    img_dir: Path,
    *,
    kline_start: str,
    kline_end: str,
    title: str,
    subtitle: str,
    img_src_prefix: str,
) -> None:
    sig = pd.read_csv(csv_path).sort_values("time_utc").reset_index(drop=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    df_cache: dict[str, pd.DataFrame] = {}
    cards: list[str] = []
    pnls_8h: list[float] = []

    for i, row in sig.iterrows():
        n = int(i) + 1
        symbol = row["symbol"]
        stem = file_stem(symbol)
        if stem not in df_cache:
            df_cache[stem] = load_symbol_df(symbol, kline_start, kline_end)
        ts = str(row["time_utc"]).replace("-", "").replace(":", "").replace(" ", "_")
        img_name = f"t{n:02d}_{stem}_{ts}.png"
        draw_trade_png(df_cache[stem], row, img_dir / img_name, n)
        print(f"[{n}/{len(sig)}] {img_name}")

        pnl = float(row["pnl_8h"]) if pd.notna(row.get("pnl_8h")) else None
        if pnl is not None:
            pnls_8h.append(pnl)
        cls = "pnl-flat" if pnl is None else ("pnl-win" if pnl > 0 else "pnl-loss")
        pnl_txt = "—" if pnl is None else f"{pnl:+.2f}%"
        reason_cls = "tag-time" if pnl is None else ("tag-tp" if pnl > 0 else "tag-sl")
        reason = "8h—" if pnl is None else ("8h贏" if pnl > 0 else "8h輸")
        t0 = str(row["time_utc"])
        t1 = ""
        if pd.notna(row.get("pnl_8h")):
            end_ts = pd.Timestamp(row["time_utc"]) + pd.Timedelta(hours=8)
            t1 = f" → {end_ts.strftime('%m-%d %H:%M')}"
        bias = float(row["bias"]) if pd.notna(row.get("bias")) else None
        span = int(row["span"]) if pd.notna(row.get("span")) else None
        dist200 = float(row["dist_ma200_pct"]) if pd.notna(row.get("dist_ma200_pct")) else None
        entry = float(row["entry"]) if pd.notna(row.get("entry")) else None
        neck = float(row["line_val"]) if pd.notna(row.get("line_val")) else None
        sma14 = float(row["sma14"]) if pd.notna(row.get("sma14")) else None
        detail_lines = [
            f"entry {_fmt_px(entry) if entry is not None else '—'}",
            (
                f"neck  {_fmt_px(neck) if neck is not None else '—'}"
                f"  sma14 {_fmt_px(sma14) if sma14 is not None else '—'}"
            ),
            (
                f"bias  {bias:.1f}%"
                + (f"  span {span}" if span is not None else "")
                + (f"  距SMA200 {dist200:.1f}%" if dist200 is not None else "")
                if bias is not None
                else "bias  —"
            ),
            (
                f"1h  {_fmt_pct(row.get('pnl_1h'))}"
                f"  4h {_fmt_pct(row.get('pnl_4h'))}"
                f"  8h {_fmt_pct(row.get('pnl_8h'))}"
            ),
        ]
        detail = "\n".join(detail_lines)
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{n} · {escape(symbol)} · 結構</span>"
            f"<span class='trade-time'>{escape(t0)}{escape(t1)}</span></div>"
            f"<div class='card-pnl {cls}'>{pnl_txt}</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{reason}</span>"
            "<span class='tag tag-info'>5m + 15m</span>"
            "<span class='tag tag-info'>結構</span>"
            "</div>"
            f"<pre class='trade-detail'>{escape(detail)}</pre>"
            f"<div class='mini-chart'><img src='{escape(img_src_prefix + img_name)}' alt='#{n} {escape(symbol)}' "
            "style='width:100%;display:block;border-radius:10px'/></div>"
            "</article>"
        )

    wr1, a1 = wr_avg(sig["pnl_1h"])
    wr4, a4 = wr_avg(sig["pnl_4h"])
    wr8, a8 = wr_avg(sig["pnl_8h"])
    med8 = float(sig["pnl_8h"].dropna().median()) if sig["pnl_8h"].notna().any() else 0.0
    total8 = float(sig["pnl_8h"].dropna().sum()) if sig["pnl_8h"].notna().any() else 0.0
    total_cls = "pnl-win" if total8 >= 0 else "pnl-loss"
    n = len(sig)

    def pct(v):
        return "—" if v is None else f"{v:.1f}%"

    def avg(v):
        return "—" if v is None else f"{v:+.2f}%"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(title)}</title>
<style>
{_page_css()}
</style></head><body>
<div class="page">
<a class="back" href="./index.html">← 總覽</a>
<section class="summary">
<h1>{escape(title)}</h1>
<p class="muted">{escape(subtitle)}</p>
<div class="cards">
<div class="card">筆數<b>{n}</b></div>
<div class="card">8h 勝率<b>{pct(wr8)}</b></div>
<div class="card">加總<b class="{total_cls}">{total8:+.2f}%</b></div>
<div class="card">8h 均<b>{avg(a8)}</b></div>
</div>
<p class="muted">1h {pct(wr1)} {avg(a1)} · 4h {pct(wr4)} {avg(a4)} · 8h 中位 {med8:+.2f}%</p>
<p class="muted">粉線 SMA14（破位確認）· 金線 SMA200 · 紅虛線頸線 · 黃點進場 · 叉 = 8h</p>
<div class="equity">{_equity_svg(pnls_8h)}</div>
</section>
{''.join(cards) or "<div class='empty'>這段期間沒有結構訊號</div>"}
</div>
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print("wrote", out_html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/workspace/output/shadow_neckline_structure_7d.csv")
    parser.add_argument("--out", default="/workspace/docs/charts/seven_day.html")
    parser.add_argument("--img-dir", default="/workspace/docs/charts/seven_day/img")
    parser.add_argument("--start", default="2026-08-19")
    parser.add_argument("--end", default="2026-08-29")
    parser.add_argument("--title", default="影線頸線 · 近 7 日結構空")
    parser.add_argument(
        "--subtitle",
        default=(
            "2026-08-22 → 08-28 UTC · 滾動 24h Top10 · 結構、不看量"
            "（收盤確認、跨度≥9、乖離≤65%、距SMA200≤40%、不破 MA99）。"
            "正值 = 空單獲利。每張卡底下附 15m 對照。"
        ),
    )
    parser.add_argument("--branch", default=BRANCH)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out = Path(args.out)
    # relative paths for GitHub Pages
    rel_prefix = "seven_day/img/"
    render_gallery(
        Path(args.csv),
        out,
        img_dir,
        kline_start=args.start,
        kline_end=args.end,
        title=args.title,
        subtitle=args.subtitle,
        img_src_prefix=rel_prefix,
    )
    # htmlpreview-friendly copy with absolute raw.githubusercontent.com URLs
    rel = img_dir.relative_to(REPO).as_posix()
    base = f"https://raw.githubusercontent.com/yubogoodman-droid/NQ/{args.branch}/{rel}/"
    text = out.read_text(encoding="utf-8").replace("src='seven_day/img/", f"src='{base}")
    view = img_dir.parent / "view.html"
    view.write_text(text, encoding="utf-8")
    print("wrote", view)


if __name__ == "__main__":
    main()
