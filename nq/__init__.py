"""NQ / 短線型態進場策略模組。"""

from nq.ma_align import MaAlignPattern, detect_ma_align_alerts
from nq.patterns import WBottomPattern, detect_w_bottoms
from nq.report import save_report_html
from nq.spring import FakeBreakdownPattern, detect_fake_breakdowns
from nq.spring_chart import save_spring_html_chart
from nq.strategy import FakeBreakdownStrategy, MaAlignStrategy, NQWBottomStrategy, Signal

__all__ = [
    "WBottomPattern",
    "detect_w_bottoms",
    "FakeBreakdownPattern",
    "detect_fake_breakdowns",
    "MaAlignPattern",
    "detect_ma_align_alerts",
    "FakeBreakdownStrategy",
    "MaAlignStrategy",
    "NQWBottomStrategy",
    "Signal",
    "save_report_html",
    "save_spring_html_chart",
]
