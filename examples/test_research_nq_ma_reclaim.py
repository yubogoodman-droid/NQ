#!/usr/bin/env python3
"""NQ research helpers (no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_nq_ma_reclaim import extra_vs_base, session_bucket  # noqa: E402


def test_session_bucket() -> None:
    assert session_bucket(20) == "亞盤 18–03"
    assert session_bucket(1) == "亞盤 18–03"
    assert session_bucket(4) == "倫敦 03–09"
    assert session_bucket(9) == "美開 09–10"
    assert session_bucket(11) == "美股 10–16"
    assert session_bucket(16) == "尾盤 16–18"


def test_extra_vs_base() -> None:
    base = [SimpleNamespace(entry_idx=10), SimpleNamespace(entry_idx=20)]
    other = [
        SimpleNamespace(entry_idx=10),
        SimpleNamespace(entry_idx=15),
        SimpleNamespace(entry_idx=20),
    ]
    extra = extra_vs_base(base, other)
    assert [t.entry_idx for t in extra] == [15]


def main() -> int:
    test_session_bucket()
    test_extra_vs_base()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
