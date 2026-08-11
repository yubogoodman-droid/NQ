"""Deprecated alias: use shadow_neckline_volume.py (原版+爆量)."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("shadow_neckline_volume.py")), run_name="__main__")
