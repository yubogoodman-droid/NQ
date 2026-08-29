# 10-day shadow-neckline (2026-08-05 → 2026-08-14 UTC, Vision+BingX)

Universe: prior-day Top 30 candidates → **rolling 24h Top 10** (hourly, live-aligned).

| Tier | n | symbols | 1h win | 1h avg | 4h win | 4h avg | 8h win | 8h avg |
|--|--|--|--|--|--|--|--|--|
| raw | 361 | 90 | 41.5% | -0.385% | 46.8% | -0.312% | 50.1% | -0.229% |
| structure | 45 | 28 | 45.5% | -0.338% | 54.5% | 1.959% | 50.0% | 1.05% |
| volume | 37 | 23 | 50.0% | 0.188% | 63.9% | 2.689% | 52.8% | -0.068% |
| volume2 | 32 | 21 | 45.2% | -0.483% | 58.1% | 2.029% | 51.6% | -1.657% |

structure = live path: same quality filters, **no volume** (close-break, span≥9, bias≤65%, ext SMA200≤40%, not below MA99).
volume = structure + 爆量≥2.5× (break-bar OR 4h-peak vs pre-window avg)
volume2 = structure + 爆量≥3.5× (same peak window)

Near rising SMA200 (|dist|<4%): skips lows into rising MA support (e.g. XAN).
Deep-below 15m SMA200 (dist < −3%): skips late shorts after a higher-TF dump (e.g. GWEI).
Rolling 24h Top10 catches same-day pumps that prior-day Top10 misses (e.g. ACE 08-14).