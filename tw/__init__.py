"""台股五分 K 空頭掃描：5/10/20 空頭排列，收盤跌破 MA200，且小時K在 MA20 之下。"""

from tw.backtest_5m import BacktestHit, BacktestResult, run_5m_short_backtest
from tw.signals import AlertSnapshot, iter_5m_ma200_short_alerts

__all__ = [
    "AlertSnapshot",
    "BacktestHit",
    "BacktestResult",
    "iter_5m_ma200_short_alerts",
    "run_5m_short_backtest",
]
