#!/usr/bin/env python3
"""Render the demo storyboard to a silent 720x960 H.264 picture track."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "story-to-handdrawn-video"
STORY = json.loads((DOC / "story.json").read_text(encoding="utf-8"))
OUT = DOC / "assets" / "demo-preview.mp4"
W, H = 720, 960
FPS = 30
PAGE_SEC = 4.8
FONT_CANDIDATES = [
    DOC / "assets" / "fonts" / "MaShanZheng-Regular.ttf",
    DOC / "assets" / "fonts" / "LiuJianMaoCao-Regular.ttf",
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def clamp(v: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, v))


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for ch in text:
        trial = line + ch
        if draw.textlength(trial, font=font) > max_w and line:
            lines.append(line)
            line = ch
        else:
            line = trial
    if line:
        lines.append(line)
    return lines[:3]


def layered(local: float) -> tuple[float, float, float]:
    return (
        clamp(local / 1.05),
        clamp((local - 0.95) / 1.2),
        clamp((local - 2.1) / 1.2),
    )


def wipe(base: Image.Image, overlay: Image.Image, progress: float) -> Image.Image:
    if progress <= 0:
        return base
    if progress >= 1:
        return overlay
    w = max(1, int(W * progress))
    out = base.copy()
    out.paste(overlay.crop((0, 0, w, H)), (0, 0))
    return out


def draw_caption(img: Image.Image, caption: str, sub: str, index_label: str, cover: bool, text_p: float) -> None:
    draw = ImageDraw.Draw(img)
    font = load_font(54 if cover else 40)
    small = load_font(20)
    lines = wrap(draw, caption, font, W - 96)
    clip_w = int(W * clamp(text_p))
    if clip_w <= 0:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(layer)
    y0 = 70 if cover else 58
    fill = (42, 36, 28, 255)
    for i, ln in enumerate(lines):
        d2.text((W / 2, y0 + i * (56 if cover else 44)), ln, font=font, fill=fill, anchor="ma")
    if cover and sub:
        d2.text((W / 2, y0 + len(lines) * 56 + 6), sub, font=small, fill=(138, 123, 104, 255), anchor="ma")
    d2.text((W - 36, H - 28), index_label, font=small, fill=(42, 36, 28, 90), anchor="rm")
    cropped = layer.crop((0, 0, clip_w, H))
    img.paste(cropped, (0, 0), cropped)


def render_page(color: Image.Image, bw: Image.Image, caption: str, sub: str, label: str, cover: bool, local: float) -> Image.Image:
    text_p, bw_p, color_p = layered(local)
    frame = Image.new("RGB", (W, H), (255, 254, 251))
    frame = wipe(frame, bw, bw_p)
    frame = wipe(frame, color, color_p)
    rgba = frame.convert("RGBA")
    draw_caption(rgba, caption, sub, label, cover, text_p)
    return rgba.convert("RGB")


def main() -> None:
    pages = []
    for i, p in enumerate(STORY["pages"]):
        color = Image.open(DOC / p["color"]).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
        bw = Image.open(DOC / p["bw"]).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
        pages.append((color, bw, p["caption"], p.get("sub") or "", f"{i + 1:02d}", p["kind"] == "cover"))

    n_frames = int(round(len(pages) * PAGE_SEC * FPS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast", "-an",
        "-movflags", "+faststart",
        str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for f in range(n_frames):
        t = f / FPS
        idx = min(len(pages) - 1, int(t / PAGE_SEC))
        local = t - idx * PAGE_SEC
        color, bw, caption, sub, label, cover = pages[idx]
        frame = render_page(color, bw, caption, sub, label, cover, local)
        proc.stdin.write(frame.tobytes())
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise SystemExit(stderr.decode("utf-8", "ignore")[-2000:])
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {n_frames} frames)")


if __name__ == "__main__":
    main()
