"""Shared bar-by-bar engine for live and replay. Signal-only."""
import data, rules
MA1 = (7, 14, 25, 30, 200); MA5 = (25, 99, 120, 200)

def build(df1raw, now_ms):
    d1 = data.add_ma(data.closed_1m(df1raw, now_ms), MA1)
    d5 = data.add_ma(data.resample_5m(d1, now_ms), MA5)
    return d1, d5

def step(d1, d5, st, out):
    """Process every closed bar newer than st['last1'/'last5'] in order (catch-up safe)."""
    for i in d1.index[d1.t > st["last1"]]:
        if i >= 1: rules.eval_1m(d1, i, st, out)
        st["last1"] = int(d1.t[i])
    for j in d5.index[d5.t > st["last5"]]:
        if j >= 1: rules.eval_5m(d5, j, st, out)
        st["last5"] = int(d5.t[j])
