#!/usr/bin/env python3
"""台股假跌破：2026-10-17～10-30 兩週回測，產出單一內嵌圖片的 HTML。"""

from __future__ import annotations

import base64
import html
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.chart_spring_top50 import save_png
from examples.scan_tw_top50_spring import (
    _filter_max_price,
    fetch_top_turnover,
    fetch_yahoo_1m,
    list_trading_days,
    scan_stock,
    tw_tick_size,
)
from nq.backtest import run_backtest
from nq.spring_report import add_mas
from nq.strategy import FakeBreakdownStrategy

START = "20261017"
END = "20261030"
LIMIT = 100
MAX_PRICE = 700.0
YAHOO_RANGE = "8d"
AFTER = "09:00"
MAX_BARS_HOLD = 60
HTML_PATH = Path("docs/spring/two_week_20261017_1030/index.html")
PNG_DIR = Path("output/spring_charts/two_week_20261017_1030")


def _regular_session(df: pd.DataFrame, trading_iso: set[str]) -> pd.DataFrame:
    """只留交易日 09:00–13:30（含）的 1 分 K，丟掉盤前／休市日。"""
    if df.empty:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        local = idx.tz_localize("Asia/Taipei")
    else:
        local = idx.tz_convert("Asia/Taipei")
    minutes = local.hour * 60 + local.minute
    days = local.strftime("%Y-%m-%d")
    mask = days.isin(trading_iso) & (minutes >= 9 * 60) & (minutes <= 13 * 60 + 30)
    out = df.loc[mask]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _local(ts) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is None:
        return ts
    return ts.tz_convert("Asia/Taipei")


def _fmt_px(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _img_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _exit_label(exit_reason: str, pnl: float) -> tuple[str, str]:
    if exit_reason == "take_profit":
        return "tp", f"TP {pnl:+.2f}"
    if exit_reason == "stop_loss":
        return "sl", f"SL {pnl:+.2f}"
    return "time", f"TIME {pnl:+.2f}"


def build_html(
    *,
    trading_iso: list[str],
    skipped_holidays: list[str],
    yahoo_days: list[str],
    scanned: int,
    missing: int,
    trades: list[dict],
    gallery: list[dict],
) -> str:
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    rs = [t["r"] for t in trades]
    rets = [t["pnl"] / t["entry"] for t in trades if t["entry"]]
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = (len(wins) / n) if n else 0.0
    best = max(trades, key=lambda t: t["r"]) if trades else None
    worst = min(trades, key=lambda t: t["r"]) if trades else None

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def med(xs: list[float]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    by_day_rows = []
    for iso in trading_iso:
        day_t = [t for t in trades if t["date"] == iso]
        if not day_t:
            by_day_rows.append(
                f"<tr><td>{iso}</td><td>0</td><td>0 / 0 / 0</td><td>0.00</td><td>0.00</td></tr>"
            )
            continue
        tp = sum(1 for t in day_t if t["exit"] == "take_profit")
        sl = sum(1 for t in day_t if t["exit"] == "stop_loss")
        tm = sum(1 for t in day_t if t["exit"] == "time_stop")
        by_day_rows.append(
            "<tr>"
            f"<td>{iso}</td><td>{len(day_t)}</td><td>{tp} / {sl} / {tm}</td>"
            f"<td>{sum(t['pnl'] for t in day_t):+.2f}</td>"
            f"<td>{sum(t['r'] for t in day_t):+.2f}</td>"
            "</tr>"
        )

    trade_rows = []
    for t in trades:
        trade_rows.append(
            "<tr>"
            f"<td>{t['date'][5:]}</td><td>{t['time']}</td><td>{t['code']}</td><td>{html.escape(t['name'])}</td>"
            f"<td>{_fmt_px(t['entry'])}</td><td>{_fmt_px(t['stop'])}</td><td>{_fmt_px(t['target'])}</td>"
            f"<td>{t['break_pct']:.2f}%</td><td>{t['vol_ratio']:.2f}x</td>"
            f"<td>{t['exit']}</td><td>{t['pnl']:+.2f}</td><td>{t['r']:+.2f}</td>"
            "</tr>"
        )

    cards = []
    for item in gallery:
        cards.append(
            f"""
    <article class="card{' feature' if item['feature'] else ''}">
      <h2>{html.escape(item['label'])} · {html.escape(item['time'])} <span class="tag {item['cls']}">{html.escape(item['tag'])}</span></h2>
      <p>{html.escape(item['detail'])}</p>
      <img src="{item['src']}" alt="{html.escape(item['label'])}" />
    </article>"""
        )

    best_txt = "—"
    worst_txt = "—"
    if best:
        best_txt = (
            f"{best['date']} {best['time']} {best['code']} {best['name']} "
            f"{best['pnl']:+.2f} 點 / {best['r']:+.2f}R（{_exit_label(best['exit'], best['pnl'])[1]}）"
        )
    if worst:
        worst_txt = (
            f"{worst['date']} {worst['time']} {worst['code']} {worst['name']} "
            f"{worst['pnl']:+.2f} 點 / {worst['r']:+.2f}R（{_exit_label(worst['exit'], worst['pnl'])[1]}）"
        )

    yahoo_note = "、".join(yahoo_days) if yahoo_days else "無"
    holiday_note = "、".join(skipped_holidays) if skipped_holidays else "無"

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>假跌破 · 台股兩週回測 2026-10-17～10-30</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0b0e11; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif; }}
    .page {{ max-width: 860px; margin: 0 auto; padding: 20px 14px 40px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 17px; margin: 22px 0 8px; }}
    p, li {{ color: #c9d1d9; font-size: 14px; line-height: 1.65; }}
    .sub {{ color: #8b949e; font-size: 13px; margin: 0 0 16px; }}
    code {{ background: #161b22; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0 16px; }}
    th, td {{ border: 1px solid #30363d; padding: 6px 8px; text-align: left; }}
    th {{ background: #161b22; color: #8b949e; }}
    .stat {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 12px 0 18px; }}
    .stat div {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px 12px; }}
    .stat b {{ display: block; font-size: 18px; margin-top: 4px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 12px; margin-bottom: 14px; }}
    .card.feature {{ border-color: #ef5350; }}
    .card h2 {{ font-size: 15px; margin: 0 0 8px; }}
    .card p {{ font-size: 12px; color: #8b949e; margin: 0 0 8px; }}
    .card img {{ width: 100%; height: auto; border-radius: 10px; display: block; background: #0d1117; }}
    .tag {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; }}
    .time {{ background: rgba(255,193,7,.12); color: #f0c14b; }}
    .tp {{ background: rgba(0,200,5,.15); color: #3ddc68; }}
    .sl {{ background: rgba(255,82,82,.15); color: #ff7b72; }}
    .mas {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; font-size: 12px; font-weight: 700; }}
    .mas span {{ padding: 4px 8px; border-radius: 8px; background: #161b22; border: 1px solid #30363d; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>假跌破 1 分 K · 台股兩週回測</h1>
    <p class="sub">期間 2026-10-17～2026-10-30。報酬為點數／R，<strong>未扣手續費與稅</strong>。僅供回測學習，不構成投資建議。</p>
    <p>幣安：此策略標的是台股 1 分 K（上市＋上櫃成交額前 100），repo 沒有幣安掃描，因此沒有跑幣安。</p>

    <h2>觸發規則（照 repo 實際程式）</h2>
    <ul>
      <li><strong>週期</strong>：1 分 K（<code>FakeBreakdownStrategy</code> + <code>detect_fake_breakdowns</code>）。</li>
      <li><strong>標的範圍</strong>：每個交易日 TWSE＋TPEX 成交額前 {LIMIT}，收盤價 &gt; {MAX_PRICE:g} 剔除；該日不在當日名單者不計。資料來源 Yahoo 1 分 K，<code>includePrePost=false</code>。</li>
      <li><strong>盤別</strong>：只留 09:00–13:30（含）K 線，忽略試搓／盤前。休市日 K 線刪除，不參與箱體。</li>
      <li><strong>盤整箱體</strong>：約 {18} 根、箱幅 ≤ 2%；首尾收盤距離不得大於箱高 60%；至少 85% 收盤在箱內；MA5／10／20 糾結（均線間距／價格 ≤ 1.2%，價格距均線中點 ≤ 1.8%）。</li>
      <li><strong>假跌破</strong>：低點跌破支撐 0.5%～3%；假跌破期間均量不得超過成交量 MA20 的 1.35 倍（避免恐慌量）。最多等 15 根出現跌破。</li>
      <li><strong>站回</strong>：假跌破後最多 10 根收盤站回支撐；期間低點不得再破假跌破低的 0.999 倍。</li>
      <li><strong>進場條件</strong>：站回後最多 24 根，收盤站上箱頂，且當根量 ≥ 成交量 MA20 × 1.4。允許跨夜箱體（<code>same_session=False</code>）。略過開盤後 5 分鐘（當日第一根 K + 5 分鐘之前不進）。掃描另要求進場時間 ≥ 09:00。</li>
      <li><strong>進場價</strong>：突破根收盤價，依台股 tick 四捨五入（<code>tw_tick_size</code>）。</li>
      <li><strong>停損</strong>：假跌破最低點，同樣 round 到 tick。</li>
      <li><strong>停利</strong>：進場價 + 2R（風險 = 進場 − 停損），round 到 tick。</li>
      <li><strong>時間出場</strong>：進場後最多抱 60 根 1 分 K，未觸及停損／停利則以該根收盤出場（<code>run_backtest(..., max_bars_hold=60)</code>）。</li>
      <li><strong>重疊</strong>：持倉中忽略新訊號，平倉後下一根才可再進；未實際開倉的訊號不列入。</li>
    </ul>

    <h2>資料範圍</h2>
    <ul>
      <li>日曆窗：{START[:4]}-{START[4:6]}-{START[6:]}～{END[:4]}-{END[4:6]}-{END[6:]}。</li>
      <li>TWSE 有行情的交易日：{"、".join(trading_iso) if trading_iso else "無"}。</li>
      <li>休市／無日線（略過）：{holiday_note}。</li>
      <li>Yahoo 1 分 K 實際有到的日期：{yahoo_note}。10/19 若在交易日清單但未出現在此列，代表已超出 Yahoo 1 分 K 約 8 日視窗，未納入偵測。</li>
      <li>不重複掃描 {scanned} 檔，缺 1 分 K {missing} 檔。</li>
    </ul>

    <h2>統計結果</h2>
    <div class="stat">
      <div>訊號數<b>{n}</b></div>
      <div>勝率（點數&gt;0）<b>{win_rate:.1%}</b></div>
      <div>平均點數<b>{avg(pnls):+.2f}</b></div>
      <div>中位數點數<b>{med(pnls):+.2f}</b></div>
      <div>平均 R<b>{avg(rs):+.2f}</b></div>
      <div>中位數 R<b>{med(rs):+.2f}</b></div>
      <div>平均報酬率<b>{avg(rets):+.2%}</b></div>
      <div>中位數報酬率<b>{med(rets):+.2%}</b></div>
    </div>
    <p>最好一筆（依 R）：{html.escape(best_txt)}<br/>最差一筆（依 R）：{html.escape(worst_txt)}</p>
    <p>合計 {sum(pnls):+.2f} 點、{sum(rs):+.2f}R。TP {sum(1 for t in trades if t['exit']=='take_profit')}／SL {sum(1 for t in trades if t['exit']=='stop_loss')}／TIME {sum(1 for t in trades if t['exit']=='time_stop')}。</p>

    <h2>逐日</h2>
    <table>
      <thead><tr><th>日期</th><th>筆數</th><th>TP / SL / TIME</th><th>點數</th><th>R</th></tr></thead>
      <tbody>
        {''.join(by_day_rows)}
      </tbody>
    </table>

    <h2>訊號明細</h2>
    <table>
      <thead><tr><th>日期</th><th>時間</th><th>代碼</th><th>名稱</th><th>進場</th><th>停損</th><th>目標</th><th>跌破</th><th>量</th><th>出場</th><th>點數</th><th>R</th></tr></thead>
      <tbody>
        {''.join(trade_rows) if trade_rows else '<tr><td colspan="12">無訊號</td></tr>'}
      </tbody>
    </table>

    <h2>K 線圖（進場 ▲、假跌破 ●、出場 ×、均線）</h2>
    <div class="mas">
      <span style="color:#ffa726">MA5</span>
      <span style="color:#ffeb3b">MA10</span>
      <span style="color:#66bb6a">MA20</span>
      <span style="color:#42a5f5">MA60</span>
      <span style="color:#26c6da">MA120</span>
      <span style="color:#ab47bc">MA200</span>
    </div>
    {''.join(cards) if cards else '<p>無圖。</p>'}
  </div>
</body>
</html>
"""


def main() -> None:
    ymds = [d for d in list_trading_days(END, 15) if START <= d <= END]
    iso_days = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in ymds]
    trading_iso = set(iso_days)

    holidays: list[str] = []
    cursor = datetime.strptime(START, "%Y%m%d")
    end_dt = datetime.strptime(END, "%Y%m%d")
    while cursor <= end_dt:
        ymd = cursor.strftime("%Y%m%d")
        if cursor.weekday() < 5 and ymd not in ymds:
            holidays.append(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
        cursor += timedelta(days=1)

    print("交易日", iso_days)
    print("休市", holidays)

    universes: dict[str, list[dict]] = {}
    allowed: dict[str, set[str]] = {}
    unique: dict[str, dict] = {}
    for ymd, iso in zip(ymds, iso_days):
        raw = fetch_top_turnover(ymd, LIMIT)
        kept, skipped = _filter_max_price(raw, MAX_PRICE)
        universes[iso] = kept
        allowed[iso] = {r["code"] for r in kept}
        for row in kept:
            unique[row["code"]] = row
        print(f"{ymd} 掃描 {len(kept)} 檔，剔除 > {MAX_PRICE:g} 共 {len(skipped)}")

    print(f"不重複 {len(unique)} 檔")
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

    hits: list[dict] = []
    missing = 0
    scanned = 0
    yahoo_days: set[str] = set()
    cache: dict[str, pd.DataFrame] = {}

    for row in unique.values():
        try:
            raw_df = fetch_yahoo_1m(row["symbol"], range=YAHOO_RANGE)
        except Exception as exc:  # noqa: BLE001
            missing += 1
            print(f"{row['code']} {row['name']} 抓 1 分 K 失敗：{exc}")
            time.sleep(0.25)
            continue
        time.sleep(0.25)
        df = _regular_session(raw_df, trading_iso)
        if not df.empty:
            yahoo_days.update(df.index.tz_convert("Asia/Taipei").strftime("%Y-%m-%d").unique().tolist())
        cache[row["code"]] = df
        result = scan_stock(row, iso_days[-1], {}, df=df, days=iso_days, yahoo_range=YAHOO_RANGE)
        if result.get("error") or df.empty:
            missing += 1
            print(f"{row['code']} {row['name']} | 無 1 分 K")
            continue
        scanned += 1
        kept = [
            s
            for s in result["signals"]
            if s["time"] >= AFTER
            and s["date"] in allowed
            and row["code"] in allowed[s["date"]]
            and s.get("exit") is not None
            and "09:00" <= s["time"] <= "13:30"
        ]
        result["signals"] = kept
        if not kept:
            continue
        hits.append(result)
        for sig in kept:
            print(
                f"{row['code']} {row['name']} | {sig['date']} {sig['time']} "
                f"{sig['exit']} {sig['pnl']:+.2f} {sig['r']:+.2f}R"
            )

    trades: list[dict] = []
    for h in hits:
        for s in h["signals"]:
            trades.append({**s, "code": h["code"], "name": h["name"], "symbol": h["symbol"]})
    trades.sort(key=lambda t: (t["date"], t["time"], t["code"]))

    gallery: list[dict] = []
    wanted = {(t["code"], t["date"], t["time"]) for t in trades}
    for h in hits:
        df1 = cache.get(h["code"])
        if df1 is None or df1.empty:
            continue
        df = add_mas(df1)
        strategy = FakeBreakdownStrategy(tick_size=tw_tick_size(float(df["close"].iloc[-1])))
        by_bar = {t.signal.bar_idx: t for t in run_backtest(df, strategy, max_bars_hold=MAX_BARS_HOLD)}
        for sig in strategy.generate_signals(df):
            local = _local(sig.timestamp)
            key = (h["code"], local.strftime("%Y-%m-%d"), local.strftime("%H:%M"))
            if key not in wanted:
                continue
            trade = by_bar.get(sig.bar_idx)
            stamp = f"{h['code']}_{local.strftime('%Y%m%d_%H%M')}"
            png = PNG_DIR / f"{stamp}.png"
            save_png(df, sig, trade, png, f"{h['code']} {h['name']}  {local.strftime('%Y-%m-%d')} 1m")
            cls, tag = _exit_label(trade.exit_reason, trade.pnl_points) if trade else ("time", "訊號")
            p = sig.pattern
            gallery.append(
                {
                    "label": f"{h['code']} {h['name']}",
                    "time": f"{local.strftime('%m-%d')} {local.strftime('%H:%M')}",
                    "sort": local.isoformat(),
                    "cls": cls,
                    "tag": tag,
                    "detail": (
                        f"進 {_fmt_px(sig.entry)} / 停 {_fmt_px(sig.stop_loss)} / 目標 {_fmt_px(sig.target)}"
                        f" · 跌破 {p.break_pct * 100:.2f}% · 量 {p.volume_ratio:.2f}x"
                    ),
                    "src": _img_data(png),
                    "feature": trade is not None and trade.exit_reason == "take_profit",
                }
            )
    gallery.sort(key=lambda item: (not item["feature"], item["sort"]))

    page = build_html(
        trading_iso=iso_days,
        skipped_holidays=holidays,
        yahoo_days=sorted(yahoo_days),
        scanned=scanned,
        missing=missing,
        trades=trades,
        gallery=gallery,
    )
    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"HTML {HTML_PATH.resolve()}  {len(trades)} 筆  {len(gallery)} 圖")
    if trades:
        pnls = [t["pnl"] for t in trades]
        rs = [t["r"] for t in trades]
        print(
            f"勝率 {sum(p>0 for p in pnls)/len(pnls):.1%}  "
            f"平均 {sum(pnls)/len(pnls):+.2f} 點 / {sum(rs)/len(rs):+.2f}R  "
            f"中位數 {statistics.median(pnls):+.2f} / {statistics.median(rs):+.2f}R"
        )


if __name__ == "__main__":
    main()
