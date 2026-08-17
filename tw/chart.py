"""台股 K 線圖：跟掃描頁同一套畫法（漲紅跌綠 PNG、整數 X、縱軸對準 MA20/MA200）。"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from tw.backtest import TradeResult

MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    200: "#ce93d8",
}
# 一分K 只畫跌破附近：拉太長會把更早的回檔畫成「MA200 在天花板」。
CHART_BARS_1M = (24, 10)
CHART_BARS_5M = (70, 12)
UP = "#ef5350"
DOWN = "#26a69a"
BG = "#161b22"
FG = "#e6edf3"
GRID = "#30363d"

_FONT = None
for _path in (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(_path).exists():
        _FONT = fm.FontProperties(fname=_path)
        break
FONT = _FONT


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for period in (5, 10, 20, 200):
        out[f"ma{period}"] = close.rolling(period, min_periods=period).mean()
    return out


def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work.index = pd.DatetimeIndex(work.index)
    out = work.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in work.columns else ("close", "count"),
    )
    return out.dropna(subset=["close"])


def centered_ylim(mid: float, ref_price: float, span_pct: float = 0.03) -> tuple[float, float]:
    """縱軸以 mid 為中心，高度為價格的 span_pct。"""
    half = abs(ref_price) * span_pct / 2.0 if ref_price else 1.0
    if half <= 0:
        half = 1.0
    pad = half * 0.04
    return mid - half - pad, mid + half + pad


def axis_ylim(
    window: pd.DataFrame,
    ref_price: float,
    min_span_pct: float = 0.03,
) -> tuple[float, float]:
    """縱軸至少覆蓋 min_span_pct，避免一分K把均線差放大成整張圖。"""
    lows: list[float] = [float(window["low"].min())]
    highs: list[float] = [float(window["high"].max())]
    for col in ("ma5", "ma10", "ma20", "ma200"):
        if col not in window.columns:
            continue
        series = window[col].astype(float)
        if series.notna().any():
            lows.append(float(series.min()))
            highs.append(float(series.max()))
    lo, hi = min(lows), max(highs)
    if not (lo < hi):
        pad = abs(ref_price) * 0.01 or 1.0
        return lo - pad, hi + pad
    span = hi - lo
    min_span = abs(ref_price) * min_span_pct if ref_price else 0.0
    if span < min_span:
        mid = (lo + hi) / 2.0
        lo = mid - min_span / 2.0
        hi = mid + min_span / 2.0
        span = min_span
    pad = span * 0.04
    return lo - pad, hi + pad


def window_around(df: pd.DataFrame, ts: pd.Timestamp, before: int, after: int) -> pd.DataFrame:
    if df.empty:
        return df
    loc = df.index.get_indexer([ts], method="nearest")[0]
    if loc < 0:
        return df.iloc[-before - after :]
    start = max(0, int(loc) - before)
    end = min(len(df), int(loc) + after + 1)
    return df.iloc[start:end]


def render_trade_chart_png(
    df: pd.DataFrame,
    trade: TradeResult,
    *,
    timeframe: str = "1m",
) -> bytes | None:
    frame = resample_5m(df) if timeframe == "5m" else df
    if frame is None or frame.empty:
        return None
    work = add_moving_averages(frame)
    mark_ts = pd.Timestamp(trade.signal.timestamp)
    exit_ts = pd.Timestamp(trade.exit_time)
    if timeframe == "5m":
        mark_ts = mark_ts.floor("5min")
        exit_ts = exit_ts.floor("5min")
    before, after = CHART_BARS_5M if timeframe == "5m" else CHART_BARS_1M
    if timeframe == "1m":
        after = max(after, min(int(trade.hold_bars) + 2, 16))
    window = window_around(work, mark_ts, before=before, after=after)
    if window.empty:
        return None

    n = len(window)
    xs = list(range(n))
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    body_w = 0.7
    for i, (_, row) in enumerate(window.iterrows()):
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = UP if c >= o else DOWN
        ax.vlines(i, l, h, color=color, linewidth=0.9, zorder=2)
        body = abs(c - o)
        if body < 1e-9:
            body = max((h - l) * 0.04, 1e-6)
        ax.add_patch(
            Rectangle(
                (i - body_w / 2, min(o, c)),
                body_w,
                body,
                facecolor=color,
                edgecolor=color,
                linewidth=0.4,
                zorder=3,
            )
        )

    for period, color in MA_COLORS.items():
        col = f"ma{period}"
        if col not in window.columns:
            continue
        vals = window[col].astype(float)
        if vals.notna().sum() == 0:
            continue
        ax.plot(
            xs,
            vals,
            color=color,
            linewidth=2.0 if period == 200 else 1.35,
            label=f"MA{period}",
            zorder=4,
            solid_capstyle="round",
        )

    loc = int(window.index.get_indexer([mark_ts], method="nearest")[0])
    tf_name = "五分K" if timeframe == "5m" else "一分K"
    if 0 <= loc < n:
        ax.scatter(
            [loc],
            [float(window["close"].iloc[loc])],
            marker="v",
            s=70,
            color="#ff7b72",
            zorder=5,
            label="跌破",
        )
    exit_loc = int(window.index.get_indexer([exit_ts], method="nearest")[0])
    if 0 <= exit_loc < n and exit_loc != loc:
        ax.scatter(
            [exit_loc],
            [float(trade.exit_price)],
            marker="x",
            s=55,
            color="#7ee787" if trade.pnl_pct > 0 else "#ff7b72",
            zorder=5,
            label="回補",
        )

    sig = trade.signal
    ref = float(sig.entry) if sig.entry else float(window["close"].iloc[-1])
    if timeframe == "1m":
        mid = (float(sig.ma20) + float(sig.ma200)) / 2.0
        ax.set_ylim(*centered_ylim(mid, ref, span_pct=0.03))
    else:
        ax.set_ylim(*axis_ylim(window, ref, min_span_pct=0.03))
    ax.set_xlim(-1, n)
    bar_time = (
        pd.Timestamp(window.index[loc]).strftime("%H:%M") if 0 <= loc < n else pd.Timestamp(sig.timestamp).strftime("%H:%M")
    )
    gap = (sig.ma20 - sig.ma200) / sig.ma200 if sig.ma200 else 0.0
    if timeframe == "1m":
        title = f"{sig.name} {sig.ticker}  {tf_name}  {bar_time}  MA20/MA200 差 {gap:.1%}"
    else:
        title = f"{sig.name} {sig.ticker}  {tf_name}  {bar_time}"
    ax.set_title(title, color=FG, fontsize=11, pad=8, fontproperties=_FONT)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=FG,
        ncol=6,
        prop=_FONT,
    )

    step = max(1, n // 6)
    ticks = list(range(0, n, step))
    if n - 1 not in ticks:
        ticks.append(n - 1)
    labels = [pd.Timestamp(window.index[i]).strftime("%H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def save_trade_charts(
    df: pd.DataFrame,
    trade: TradeResult,
    chart_dir: Path,
    stem: str,
) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for tf in ("1m", "5m"):
        png = render_trade_chart_png(df, trade, timeframe=tf)
        if not png:
            continue
        path = chart_dir / f"{stem}-{tf}.png"
        path.write_bytes(png)
        saved[tf] = path
    return saved
