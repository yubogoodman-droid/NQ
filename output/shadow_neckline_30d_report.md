# 30-day shadow-neckline (2026-07-18 → 2026-08-16 UTC, Vision+BingX)

Universe: prior-day Top 30 candidates → **rolling 24h Top 10** (hourly, live-aligned).

| Tier | n | symbols | 1h win | 1h avg | 4h win | 4h avg | 8h win | 8h avg |
|--|--|--|--|--|--|--|--|--|
| raw | 1035 | 175 | 45.2% | 0.033% | 51.7% | 0.073% | 51.1% | -0.087% |
| structure | 142 | 73 | 45.1% | 0.033% | 48.6% | -0.206% | 49.3% | -0.349% |
| volume | 112 | 61 | 46.4% | 0.312% | 50.9% | 0.39% | 51.8% | -0.023% |
| volume2 | 92 | 57 | 44.6% | 0.163% | 46.7% | 0.07% | 50.0% | -0.76% |

structure = live path: same quality filters, **no volume** (close-break, span≥9, bias≤65%, ext SMA200≤40%, not below MA99).
volume = structure + 爆量≥2.5× (break-bar OR 4h-peak vs pre-window avg)
volume2 = structure + 爆量≥3.5× (same peak window)

Near rising SMA200 (|dist|<4%): skips lows into rising MA support (e.g. XAN).
Deep-below 15m SMA200 (dist < −3%): skips late shorts after a higher-TF dump (e.g. GWEI).
Rolling 24h Top10 catches same-day pumps that prior-day Top10 misses (e.g. ACE 08-14).