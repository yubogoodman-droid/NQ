"""掃描結果 HTML 報告。"""

from __future__ import annotations

import html
from pathlib import Path

from tw.screener import ScanResult


def save_scan_html(result: ScanResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(result), encoding="utf-8")
    return out


def _render(result: ScanResult) -> str:
    scanned = result.scanned_at.strftime("%Y-%m-%d %H:%M:%S")
    rank_time = html.escape(result.rank_time or "—")
    hit_rows = "\n".join(_hit_card(h) for h in result.hits) or (
        '<p class="empty">目前沒有符合「MA5&gt;MA10&gt;MA20 且剛站上 MA200」的標的。</p>'
    )
    filtered = len(result.universe) - len(result.candidates)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>台股一分K MA200 站上掃描</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b98a8;
      --up: #ff5c7a;
      --accent: #7dd3a7;
    }}
    body {{
      margin: 0; background: var(--bg); color: var(--text);
      font-family: "PingFang TC", "Noto Sans TC", sans-serif;
      padding: 16px;
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 8px; }}
    .meta {{ color: var(--muted); font-size: .9rem; line-height: 1.6; }}
    .card {{
      background: var(--card); border-radius: 14px; padding: 14px 16px;
      margin: 12px 0;
    }}
    .name {{ font-weight: 700; font-size: 1.05rem; }}
    .sym {{ color: var(--muted); margin-left: 6px; }}
    .price {{ color: var(--up); font-weight: 700; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: .92rem; }}
    .empty {{ color: var(--muted); }}
    .ok {{ color: var(--accent); }}
  </style>
</head>
<body>
  <h1>台股一分K 多頭排列 × 站上 MA200</h1>
  <div class="meta">
    掃描時間 {html.escape(scanned)}（台北）<br/>
    成交額排行時間 {rank_time}<br/>
    排行 {len(result.universe)} 檔／濾掉 ≥650 共 {filtered} 檔／實際掃描 {len(result.candidates)} 檔<br/>
    命中 <span class="ok">{len(result.hits)}</span>
    ／資料不足 {len(result.skipped)} ／錯誤 {len(result.errors)}
  </div>
  {hit_rows}
</body>
</html>
"""


def _hit_card(hit) -> str:
    s = hit.stock
    snap = hit.snapshot
    chg = ""
    if s.change_percent is not None:
        chg = f" ({s.change_percent:+.2f}%)"
    ts = snap.timestamp.strftime("%H:%M")
    return f"""
  <article class="card">
    <div class="name">{html.escape(s.name)}<span class="sym">{html.escape(s.symbol)}</span></div>
    <div class="row">
      <span>成交額第 {s.rank} 名</span>
      <span class="price">{s.price:.2f}{html.escape(chg)}</span>
    </div>
    <div class="row"><span>這根 {ts} 收盤</span><span>{snap.close:.2f}</span></div>
    <div class="row"><span>一分 MA200</span><span>{snap.ma200:.2f}</span></div>
    <div class="row"><span>前一根收／MA200</span><span>{snap.prev_close:.2f} / {snap.prev_ma200:.2f}</span></div>
    <div class="row"><span>MA5 / 10 / 20</span>
      <span>{snap.ma5:.2f} &gt; {snap.ma10:.2f} &gt; {snap.ma20:.2f}</span></div>
    <div class="row"><span>成交額</span><span>{s.turnover/1e8:.2f} 億</span></div>
  </article>
"""
