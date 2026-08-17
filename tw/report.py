"""掃描結果 HTML 報告（可放到 GitHub Pages）。"""

from __future__ import annotations

import html
from pathlib import Path

from tw.screener import ScanHit, ScanResult


def save_scan_html(result: ScanResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(result), encoding="utf-8")
    return out


def _render(result: ScanResult) -> str:
    scanned = result.scanned_at.strftime("%Y-%m-%d %H:%M:%S")
    rank_time = html.escape(result.rank_time or "—")
    hit_rows = "\n".join(_hit_card(i, h) for i, h in enumerate(result.hits, 1)) or (
        '<p class="empty">目前沒有符合條件的個股。</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <meta http-equiv="refresh" content="180"/>
  <title>台股一分K · 多頭排列站上 MA200</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0e11;
      --card: #161b22;
      --line: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --up: #ff5c7a;
      --ok: #7ee787;
      --chip: #21262d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{ max-width: 560px; margin: 0 auto; padding: 16px 14px 40px; }}
    h1 {{ font-size: 1.2rem; margin: 0 0 6px; }}
    .lead {{ color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0 0 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
    .chip {{
      background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
      padding: 4px 10px; font-size: 12px; color: var(--muted);
    }}
    .summary {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; margin-bottom: 14px; font-size: .9rem; line-height: 1.65;
      color: var(--muted);
    }}
    .summary .ok {{ color: var(--ok); font-weight: 700; font-size: 1.05rem; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 14px; margin: 0 0 12px; display: block; color: inherit; text-decoration: none;
    }}
    .card:hover {{ border-color: #6e7681; }}
    .top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }}
    .name {{ font-weight: 700; font-size: 1.05rem; }}
    .sym {{ color: var(--muted); font-weight: 500; margin-left: 6px; font-size: .9rem; }}
    .price {{ color: var(--up); font-weight: 700; white-space: nowrap; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: .9rem; color: var(--muted); }}
    .row b {{ color: var(--text); font-weight: 600; }}
    .empty {{ color: var(--muted); }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>台股一分K · 剛站上 MA200</h1>
    <p class="lead">成交額前 100、濾掉 ETF 與股價 650 以上。一分K MA5&gt;MA10&gt;MA20，且這根收盤剛站上 MA200（前一根還沒）。</p>
    <div class="chips">
      <span class="chip">不含 ETF</span>
      <span class="chip">股價 &lt; 650</span>
      <span class="chip">MA5 &gt; 10 &gt; 20</span>
      <span class="chip">金叉 MA200</span>
    </div>
    <div class="summary">
      命中 <span class="ok">{len(result.hits)}</span> 檔<br/>
      掃描時間 {html.escape(scanned)}（台北）<br/>
      排行時間 {rank_time}<br/>
      前 100 名 → 濾掉股價 {result.price_dropped}、ETF {result.etf_dropped} → 掃描 {len(result.candidates)} 檔
    </div>
    {hit_rows}
    <footer>僅供研究，不構成投資建議。點卡片可開 Yahoo 報價。頁面約 3 分鐘自動重整（資料隨掃描更新）。</footer>
  </div>
</body>
</html>
"""


def _hit_card(index: int, hit: ScanHit) -> str:
    s = hit.stock
    snap = hit.snapshot
    chg = ""
    if s.change_percent is not None:
        chg = f" {s.change_percent:+.2f}%"
    ts = snap.timestamp.strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{html.escape(s.symbol)}"
    return f"""
    <a class="card" href="{url}" target="_blank" rel="noopener">
      <div class="top">
        <div class="name">{index}. {html.escape(s.name)}<span class="sym">{html.escape(s.symbol)}</span></div>
        <div class="price">{s.price:.2f}{html.escape(chg)}</div>
      </div>
      <div class="row"><span>金叉時間</span><b>{ts}</b></div>
      <div class="row"><span>收盤 / MA200</span><b>{snap.close:.2f} &gt; {snap.ma200:.2f}</b></div>
      <div class="row"><span>前收 / 前MA200</span><b>{snap.prev_close:.2f} / {snap.prev_ma200:.2f}</b></div>
      <div class="row"><span>MA5 / 10 / 20</span><b>{snap.ma5:.2f} &gt; {snap.ma10:.2f} &gt; {snap.ma20:.2f}</b></div>
      <div class="row"><span>成交額排名</span><b>#{s.rank} · {s.turnover/1e8:.2f} 億</b></div>
    </a>
"""
