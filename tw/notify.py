"""桌面／Telegram／Discord 通知。"""

from __future__ import annotations

import os
import shutil
import subprocess

import requests

from tw.screener import ScanHit


def send_notifications(title: str, body: str) -> list[str]:
    """送出所有已設定的通知通道，回傳成功通道名稱。"""
    sent: list[str] = []
    if _notify_send(title, body):
        sent.append("desktop")
    if _telegram(title, body):
        sent.append("telegram")
    if _discord(title, body):
        sent.append("discord")
    return sent


def format_hit_message(hits: list[ScanHit]) -> tuple[str, str]:
    if not hits:
        return "台股一分K掃描", "目前沒有符合條件的標的"
    title = f"台股一分K站穩MA200 × {len(hits)}"
    lines = []
    for hit in hits:
        stock = hit.stock
        snap = hit.snapshot
        chg = ""
        if stock.change_percent is not None:
            chg = f" {stock.change_percent:+.2f}%"
        lines.append(
            f"{stock.rank}. {stock.name} {stock.symbol} "
            f"{stock.price:.2f}{chg}\n"
            f"   收 {snap.close:.2f} > MA200 {snap.ma200:.2f} "
            f"（前收 {snap.prev_close:.2f} / 前MA200 {snap.prev_ma200:.2f}）\n"
            f"   MA5 {snap.ma5:.2f} > MA10 {snap.ma10:.2f} > MA20 {snap.ma20:.2f}\n"
            f"   {snap.timestamp.strftime('%H:%M')}  成交額 {stock.turnover/1e8:.2f} 億"
        )
    return title, "\n".join(lines)


def _notify_send(title: str, body: str) -> bool:
    binary = shutil.which("notify-send")
    if not binary:
        return False
    try:
        subprocess.run(
            [binary, "--app-name=tw-1m-screener", title, body[:1000]],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _telegram(title: str, body: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": f"{title}\n{body}"[:4000],
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        return resp.ok
    except requests.RequestException:
        return False


def _discord(title: str, body: str) -> bool:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    try:
        resp = requests.post(
            webhook,
            json={"content": f"**{title}**\n```\n{body[:1800]}\n```"},
            timeout=15,
        )
        return resp.ok
    except requests.RequestException:
        return False
