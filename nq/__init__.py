"""NQ 策略模組。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "WBottomPattern",
    "detect_w_bottoms",
    "NQWBottomStrategy",
    "Signal",
    "save_report_html",
]


def __getattr__(name: str) -> Any:
    if name in {"WBottomPattern", "detect_w_bottoms"}:
        from nq.patterns import WBottomPattern, detect_w_bottoms

        return {"WBottomPattern": WBottomPattern, "detect_w_bottoms": detect_w_bottoms}[name]
    if name in {"NQWBottomStrategy", "Signal"}:
        from nq.strategy import NQWBottomStrategy, Signal

        return {"NQWBottomStrategy": NQWBottomStrategy, "Signal": Signal}[name]
    if name == "save_report_html":
        from nq.report import save_report_html

        return save_report_html
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
