#!/usr/bin/env python3
"""Stamp two-line P&L + Chinese-exit-reason badges onto 30-day MA25 cards."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC_HTML = Path("/workspace/NQ/docs/binance-1h-ma25-30d/index.html")
SRC_IMG = Path("/workspace/NQ/docs/binance-1h-ma25-30d/img")
OUT_DIR = Path("/workspace/ma25-retest/30d-pnl")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
CJK_INDEX = 3  # Noto Serif CJK TC

# High-contrast fills (opaque)
WIN_FILL = (22, 163, 74, 245)    # green #16a34a
LOSS_FILL = (220, 38, 38, 245)   # red   #dc2626
FLAT_FILL = (107, 114, 128, 245)
TEXT_FILL = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 90)

REASON_MAP = {
    "timeout": "抱36h",
    "stop": "破底停損",
    "m5_neck": "五分M破頸",
    "m5_m": "五分M破頸",
    "lost_ma25": "掉MA25",
    "target": "2R",
}

# First tag after card-pnl that is tag-time / tag-sl / tag-tp is the exit reason.
ARTICLE_RE = re.compile(
    r"<article class='trade-card'>.*?"
    r"class='card-pnl (pnl-\w+)'>([^<]+)</div>.*?"
    r"<span class='tag tag-(?:time|sl|tp)'>([^<]+)</span>.*?"
    r"<img src='img/([^']+)'",
    re.DOTALL,
)


def parse_trades(html: str) -> list[dict]:
    trades = []
    for cls, raw_pct, raw_reason, fname in ARTICLE_RE.findall(html):
        pct = raw_pct.strip()
        key = raw_reason.strip()
        zh = REASON_MAP.get(key, key)
        # Unicode minus for display (user asked for −2.69%)
        if pct.startswith("-"):
            display = "\u2212" + pct[1:]
        elif pct.startswith("\u2212"):
            display = pct
        else:
            display = pct
        trades.append(
            {
                "cls": cls,
                "raw": pct,
                "display": display,
                "reason_key": key,
                "reason": zh,
                "fname": fname,
            }
        )
    return trades


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    w, h = size
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    return mask


def _bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1], b


def make_badge(pct: str, reason: str, fill: tuple, img_w: int) -> Image.Image:
    """Two-line rounded pill: P&L on top, Chinese exit reason below."""
    pct_size = max(18, int(img_w * 0.040))
    reason_size = max(12, int(pct_size * 0.56))

    tmp = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dr = ImageDraw.Draw(tmp)

    def fonts(ps: int, rs: int):
        pf = ImageFont.truetype(FONT_PATH, ps)
        rf = ImageFont.truetype(CJK_FONT, rs, index=CJK_INDEX)
        return pf, rf

    pct_font, reason_font = fonts(pct_size, reason_size)
    pw, ph, _ = _bbox(dr, pct, pct_font)
    rw, rh, _ = _bbox(dr, reason, reason_font)

    gap = max(1, int(pct_size * 0.08))
    pad_x = max(12, int(pct_size * 0.50))
    pad_y = max(7, int(pct_size * 0.28))
    badge_w = max(pw, rw) + pad_x * 2
    badge_h = ph + gap + rh + pad_y * 2

    # Keep the chip in a readable band (~16–24% of image width).
    lo, hi = 0.155, 0.245
    for _ in range(14):
        ratio = badge_w / img_w
        if lo <= ratio <= hi:
            break
        scale = 1.07 if ratio < lo else 0.93
        pct_size = max(14, int(pct_size * scale))
        reason_size = max(11, int(pct_size * 0.56))
        pct_font, reason_font = fonts(pct_size, reason_size)
        pw, ph, _ = _bbox(dr, pct, pct_font)
        rw, rh, _ = _bbox(dr, reason, reason_font)
        gap = max(1, int(pct_size * 0.08))
        pad_x = max(12, int(pct_size * 0.50))
        pad_y = max(7, int(pct_size * 0.28))
        badge_w = max(pw, rw) + pad_x * 2
        badge_h = ph + gap + rh + pad_y * 2

    radius = badge_h // 2  # pill

    shadow_pad = max(3, img_w // 400)
    canvas_w = badge_w + shadow_pad * 4
    canvas_h = badge_h + shadow_pad * 4
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    shadow_img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    sm = rounded_mask((badge_w, badge_h), radius)
    shadow_fill = Image.new("RGBA", (badge_w, badge_h), SHADOW)
    shadow_img.paste(shadow_fill, (0, 0), sm)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_pad + 1))
    canvas.paste(shadow_img, (shadow_pad * 2 + 1, shadow_pad * 2 + 1), shadow_img)

    body = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    bm = rounded_mask((badge_w, badge_h), radius)
    body_fill = Image.new("RGBA", (badge_w, badge_h), fill)
    body.paste(body_fill, (0, 0), bm)

    rim = ImageDraw.Draw(body)
    rim.rounded_rectangle(
        (1, 1, badge_w - 2, badge_h - 2),
        radius=max(1, radius - 1),
        outline=(0, 0, 0, 60),
        width=max(1, img_w // 700),
    )

    draw = ImageDraw.Draw(body)
    pw, ph, pb = _bbox(draw, pct, pct_font)
    rw, rh, rb = _bbox(draw, reason, reason_font)
    gap = max(1, int(pct_size * 0.08))
    total_h = ph + gap + rh
    top = (badge_h - total_h) // 2

    px = (badge_w - pw) // 2 - pb[0]
    py = top - pb[1]
    rx = (badge_w - rw) // 2 - rb[0]
    ry = top + ph + gap - rb[1]
    draw.text((px, py), pct, font=pct_font, fill=TEXT_FILL)
    draw.text((rx, ry), reason, font=reason_font, fill=TEXT_FILL)

    ox = shadow_pad
    oy = shadow_pad
    canvas.paste(body, (ox, oy), body)
    return canvas


def stamp(src: Path, dst: Path, display: str, reason: str, cls: str) -> None:
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    if cls == "pnl-win":
        fill = WIN_FILL
    elif cls == "pnl-loss":
        fill = LOSS_FILL
    else:
        fill = FLAT_FILL
    badge = make_badge(display, reason, fill, w)
    bw, bh = badge.size
    # Top-right of the whole PNG, small margin. Title bar is at the top,
    # so the pill covers the tiny title P&L rather than candlesticks.
    margin = max(8, int(w * 0.012))
    x = w - bw - margin + 4  # +4 accounts for shadow padding on the right
    y = max(4, margin - 2)
    out = im.copy()
    out.alpha_composite(badge, (x, y))
    rgb = Image.new("RGB", out.size, (255, 255, 255))
    rgb.paste(out, mask=out.split()[-1])
    rgb.save(dst, "PNG", optimize=True)


def main() -> None:
    html = SRC_HTML.read_text(encoding="utf-8")
    trades = parse_trades(html)
    if not trades:
        raise SystemExit("parsed 0 trades from HTML")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    written = set()
    for i, t in enumerate(trades, 1):
        src = SRC_IMG / t["fname"]
        if not src.is_file():
            raise SystemExit(f"missing source image: {src}")
        dst = OUT_DIR / t["fname"]
        stamp(src, dst, t["display"], t["reason"], t["cls"])
        written.add(t["fname"])
        rows.append((i, str(dst), t["display"], t["reason"]))

    # Drop leftover cards from an older scan so the folder matches HTML.
    for leftover in OUT_DIR.glob("*.png"):
        if leftover.name not in written:
            leftover.unlink()

    print(f"stamped={len(rows)}")
    for n, path, pnl, reason in rows:
        print(f"{n}\t{path}\t{pnl}\t{reason}")


if __name__ == "__main__":
    main()
