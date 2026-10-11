#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess

DOC = Path(__file__).resolve().parents[1] / "docs" / "story-to-handdrawn-video"
story = json.loads((DOC / "story.json").read_text(encoding="utf-8"))
assert story["title"] == "窗边的约定"
assert len(story["pages"]) == 6
for page in story["pages"]:
    color = DOC / page["color"]
    bw = DOC / page["bw"]
    assert color.is_file(), color
    assert bw.is_file(), bw

mp4 = DOC / "assets" / "demo-preview.mp4"
assert mp4.is_file() and mp4.stat().st_size > 100_000, "demo mp4 missing"
probe = subprocess.check_output(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height,nb_frames,duration",
     "-of", "csv=p=0", str(mp4)],
    text=True,
).strip()
assert probe.startswith("720,960"), probe
print("assets ok", probe)
