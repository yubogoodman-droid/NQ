# AGENTS.md

## Cursor Cloud specific instructions

This is a small pure-Python project (Python 3.12) that backtests an NQ futures
"W-bottom" (雙底) long-entry strategy on 5-minute candles. There is no database,
web server, or long-running service — everything runs as one-shot CLI scripts.

### Environment

- Dependencies (`pandas`, `numpy`, `yfinance`, `plotly`) are installed into the
  user site-packages by the startup update script (`pip3 install -r requirements.txt`).
  The base image has no `python3-venv`, so a virtualenv is intentionally NOT used;
  run everything with the system `python3` (packages resolve from `~/.local`).

### Running the app

- Offline demo backtest (no network, uses synthetic W-bottom data — best smoke test):
  `python3 examples/run_backtest.py --demo`
- Backtest your own data: `python3 examples/run_backtest.py --csv your_nq_5m.csv`
  (CSV needs columns `datetime,open,high,low,close`).
- Generate HTML report/chart from live data via `yfinance` (needs network egress):
  `python3 examples/chart_today.py --report --pages` — see `README.md` for all
  chart variants. NOTE: `--pages` overwrites the tracked file `docs/index.html`;
  use `-o output/<name>.html` for scratch output (`output/*.html` is gitignored)
  to avoid committing regenerated report artifacts.

### Lint / test / build

- There is no linter config, no automated test suite, and no build step in this
  repo. "Testing" a change means running the CLI scripts above and checking the
  printed signals / backtest summary (and the generated HTML for chart changes).
