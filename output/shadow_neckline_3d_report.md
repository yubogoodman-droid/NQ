# 3-day shadow-neckline (2026-08-14 → 2026-08-16 UTC, Vision+BingX)

Universe: prior-day Top 30 candidates → **rolling 24h Top 10** (hourly, live-aligned).

| Tier | n | symbols | 1h win | 1h avg | 4h win | 4h avg | 8h win | 8h avg |
|--|--|--|--|--|--|--|--|--|
| raw | 71 | 22 | 36.6% | -0.127% | 44.1% | -0.571% | 52.2% | -0.462% |
| structure | 5 | 5 | 60.0% | 7.023% | 60.0% | 8.406% | 60.0% | 16.333% |
| volume | 5 | 5 | 60.0% | 7.023% | 60.0% | 8.406% | 60.0% | 16.333% |
| volume2 | 4 | 4 | 75.0% | 8.812% | 50.0% | 10.171% | 50.0% | 19.972% |

structure = live path: same quality filters, **no volume** (close-break, span≥9, bias≤65%, ext SMA200≤40%, not below MA99).
volume = structure + 爆量≥2.5× (break-bar OR 4h-peak vs pre-window avg)
volume2 = structure + 爆量≥3.5× (same peak window)

Near rising SMA200 (|dist|<4%): skips lows into rising MA support (e.g. XAN).
Deep-below 15m SMA200 (dist < −3%): skips late shorts after a higher-TF dump (e.g. GWEI).
Rolling 24h Top10 catches same-day pumps that prior-day Top10 misses (e.g. ACE 08-14).