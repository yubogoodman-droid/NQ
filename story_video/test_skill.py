#!/usr/bin/env python3
"""Smoke-check that the vendored story-to-handdrawn-video skill is installed."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".cursor" / "skills" / "story-to-handdrawn-video" / "SKILL.md"
WRAP = ROOT / ".cursor" / "skills" / "story-to-handdrawn-video" / "scripts" / "run_story_video.py"
VENDOR = ROOT / "vendor" / "story-to-handdrawn-video"

assert SKILL.is_file(), SKILL
assert WRAP.is_file(), WRAP
assert (VENDOR / "package.json").is_file(), VENDOR
assert "story-to-handdrawn-video" in SKILL.read_text(encoding="utf-8")

help_text = subprocess.check_output(
    [sys.executable, str(WRAP), "--help"],
    cwd=ROOT,
    text=True,
)
assert "--mode" in help_text
assert "--list-styles" in help_text
print("skill ok", VENDOR)
