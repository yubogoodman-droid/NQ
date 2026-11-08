# 7-day shadow-neckline (2026-08-22 → 2026-08-28 UTC, Vision+BingX)

Universe: prior-day Top 30 candidates → **rolling 24h Top 10** (hourly, live-aligned).

| Tier | n | symbols | 1h win | 1h avg | 4h win | 4h avg | 8h win | 8h avg |
|--|--|--|--|--|--|--|--|--|
| raw | 241 | 74 | 47.7% | -0.492% | 54.0% | -0.899% | 63.5% | -0.051% |
| structure | 22 | 13 | 36.4% | -0.875% | 54.5% | -1.686% | 72.7% | 0.327% |
| volume | 20 | 13 | 40.0% | -0.827% | 55.0% | -1.862% | 70.0% | -0.093% |
| volume2 | 18 | 11 | 44.4% | -0.834% | 61.1% | -1.18% | 72.2% | 0.858% |

structure = live path: same quality filters, **no volume** (close-break, span≥9, bias≤65%, ext SMA200≤40%, not below MA99).
volume = structure + 爆量≥2.5× (break-bar OR 4h-peak vs pre-window avg)
volume2 = structure + 爆量≥3.5× (same peak window)

Near rising SMA200 (|dist|<4%): skips lows into rising MA support (e.g. XAN).
Deep-below 15m SMA200 (dist < −3%): skips late shorts after a higher-TF dump (e.g. GWEI).
Rolling 24h Top10 catches same-day pumps that prior-day Top10 misses (e.g. ACE 08-14).