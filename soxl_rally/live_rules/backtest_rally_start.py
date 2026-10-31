"""SOXL 1m 'start of rally' (起漲點) rule research on BingX NCSKSOXL2USD-USDT.
Signal research only - no orders. Uses cached bingx_1m.csv (run fetch_bingx_1m.py to refresh)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/workspace/soxl_ma_watch/"
DIP = 0.015      # low must be >= 1.5% below MA200
LOOK = 30        # bars to look back for the dip
COOL = 60        # min bars between signals of same rule
HOR = [30, 60, 120]
EVENT_LOW_T = pd.Timestamp("2026-10-08 21:55", tz="Asia/Taipei")

df = pd.read_csv(D + "bingx_1m.csv")
df["dt"] = pd.to_datetime(df.time, unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
for n in [7, 14, 25, 99, 120, 200]:
    df[f"ma{n}"] = df.close.rolling(n).mean()
df["vr"] = df.volume / df.volume.rolling(20).mean().shift(1)
df["dipbar"] = df.low <= df.ma200 * (1 - DIP)
df["dip_recent"] = df.dipbar.rolling(LOOK).max().astype(bool)
df["dip_recent60"] = df.dipbar.rolling(60).max().astype(bool)
df["ma7up"] = df.ma7 > df.ma7.shift(1)
df["cluster"] = df[["ma99", "ma120", "ma200"]].max(axis=1)
df["swinglow"] = df.low.rolling(LOOK).min()
c = df.close

def cross_up(a, b): return (a > b) & (a.shift(1) <= b.shift(1))

rules = {}
# A: dip >=1.5% below MA200 in last 30 bars, first close back above MA25 with MA7 turning up
rules["A_dip_reclaimMA25"] = df.dip_recent & cross_up(c, df.ma25) & df.ma7up
# B: MA7 crosses above MA25 while MA25 still below MA200, after dip, with a >=2x volume bar in last 10
rules["B_ma7x25_vol"] = df.dip_recent & cross_up(df.ma7, df.ma25) & (df.ma25 < df.ma200) & (df.vr.rolling(10).max() >= 2)
# C: first close above MA99/120/200 cluster after a dip below it (dip within last 60 bars)
rules["C_cluster_break"] = df.dip_recent60 & cross_up(c, df.cluster)
# E: capitulation volume (>=4x) at the dip, then first close above MA14 with MA7 rising
climax = df.dipbar & (df.vr >= 4)
rules["E_climax_reclaimMA14"] = climax.rolling(15).max().astype(bool) & cross_up(c, df.ma14) & df.ma7up

# D: higher-low + break of prior swing high (bar-by-bar, no lookahead)
lo, hi, cl = df.low.values, df.high.values, c.values
import rally_rule   # shared with live watch.py so live == backtest
sigD = rally_rule.d_raw(lo, hi, cl, df.dipbar.values, LOOK)
rules["D_HL_break_swinghigh"] = pd.Series(sigD, index=df.index)

# ---- context filter (found in research): US regular session + sharp drop ----
DROP = 0.025   # dip low >= 2.5% below the highest high of the prior 90 bars
ny = df.dt.dt.tz_convert("America/New_York")
mins = ny.dt.hour * 60 + ny.dt.minute
df["rth"] = (ny.dt.dayofweek < 5) & (mins >= 570) & (mins < 960)
df["drop90"] = df.low.rolling(30).min() / df.high.rolling(90).max() - 1
FILT = (df.rth & (df.drop90 <= -DROP)).values
for name in list(rules):
    rules[name + "+F"] = pd.Series(rules[name].fillna(False).values & FILT, index=df.index)

def simulate(i, tp, sl_mode, maxbars=120):
    e = cl[i]
    sl = df.swinglow.iat[i] * 0.999 if sl_mode == "swing" else e * 0.99
    risk = e - sl
    tpp = e * (1 + tp)
    for k in range(i + 1, min(i + 1 + maxbars, len(df))):
        if lo[k] <= sl: return (sl / e - 1), (sl - e) / risk, "SL", k - i
        if hi[k] >= tpp: return tp, (tpp - e) / risk, "TP", k - i
    k = min(i + maxbars, len(df) - 1)
    return cl[k] / e - 1, (cl[k] - e) / risk, "TO", k - i

EXITS = [(0.02, "swing"), (0.03, "swing"), (0.02, "pct1"), (0.03, "pct1")]
all_rows, summ = [], []
event_idx = int(df.index[df.dt == EVENT_LOW_T][0])
for name, s in rules.items():
    s = s.fillna(False).values
    last = -10**9; idxs = []
    for i in np.where(s)[0]:
        if i - last >= COOL and i + 1 < len(df): idxs.append(i); last = i
    for i in idxs:
        r = dict(rule=name, time=df.dt.iat[i].strftime("%Y-%m-%d %H:%M"), entry=cl[i],
                 swinglow=df.swinglow.iat[i], dist_ma200_pct=round((cl[i]/df.ma200.iat[i]-1)*100, 2),
                 risk_swing_pct=round((1 - df.swinglow.iat[i]*0.999/cl[i])*100, 2),
                 bars_after_low=i - (i - LOOK + int(np.argmin(lo[i-LOOK:i+1]))))
        for h in HOR:
            seg = slice(i + 1, min(i + 1 + h, len(df)))
            r[f"maxgain{h}"] = round((hi[seg].max()/cl[i]-1)*100, 2)
            r[f"maxdd{h}"] = round((lo[seg].min()/cl[i]-1)*100, 2)
        for tp, sm in EXITS:
            ret, R, why, n = simulate(i, tp, sm)
            k = f"tp{int(tp*100)}_{sm}"
            r[k+"_ret"] = round(ret*100, 2); r[k+"_R"] = round(R, 2); r[k+"_exit"] = why
        all_rows.append(r)
    sub = pd.DataFrame([x for x in all_rows if x["rule"] == name])
    ev = sub[(pd.to_datetime(sub.time).dt.tz_localize("Asia/Taipei") >= EVENT_LOW_T) &
             (pd.to_datetime(sub.time).dt.tz_localize("Asia/Taipei") <= EVENT_LOW_T + pd.Timedelta(minutes=45))] if len(sub) else sub
    row = dict(rule=name, n=len(sub))
    if len(sub):
        for h in HOR:
            row[f"avg_maxgain{h}"] = sub[f"maxgain{h}"].mean().round(2)
            row[f"avg_maxdd{h}"] = sub[f"maxdd{h}"].mean().round(2)
        row["hit_+2%_in60"] = f"{(sub.maxgain60 >= 2).mean()*100:.0f}%"
        for tp, sm in EXITS:
            k = f"tp{int(tp*100)}_{sm}"
            row[k+"_win"] = f"{(sub[k+'_ret'] > 0).mean()*100:.0f}%"
            row[k+"_avgR"] = round(sub[k+"_R"].mean(), 2)
            row[k+"_avgret%"] = round(sub[k+"_ret"].mean(), 2)
        row["avg_risk_swing%"] = round(sub.risk_swing_pct.mean(), 2)
    if len(ev):
        e0 = ev.iloc[0]
        mins = int((pd.Timestamp(e0.time, tz="Asia/Taipei") - EVENT_LOW_T).total_seconds() // 60)
        row["caught_1008"] = f"{e0.time} @ {e0.entry} (+{mins}min after low, +{(e0.entry/146.59-1)*100:.1f}% off low)"
    else:
        row["caught_1008"] = "NO"
    summ.append(row)

trades = pd.DataFrame(all_rows); summary = pd.DataFrame(summ)
# baseline: every RTH bar as entry (to judge edge)
ii = np.where(df.rth.values[:-121])[0]
base = dict(rule="BASELINE_all_RTH_bars", n=len(ii))
for h in HOR:
    base[f"avg_maxgain{h}"] = round(np.mean([hi[i+1:i+1+h].max()/cl[i]-1 for i in ii])*100, 2)
    base[f"avg_maxdd{h}"] = round(np.mean([lo[i+1:i+1+h].min()/cl[i]-1 for i in ii])*100, 2)
summary = pd.concat([summary, pd.DataFrame([base])], ignore_index=True)
trades["half"] = np.where(trades.time < "2026-09-10", "H1", "H2")
split = trades.groupby(["rule", "half"]).tp3_swing_R.mean().unstack().round(2)
trades.to_csv(D + "rally_start_signals.csv", index=False)
summary.to_csv(D + "rally_start_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(summary.T.to_string()); print(split)
split.to_csv(D + "rally_start_split.csv")

# chart of the 10/8 event
w = df[(df.dt >= EVENT_LOW_T - pd.Timedelta(minutes=60)) & (df.dt <= EVENT_LOW_T + pd.Timedelta(minutes=100))]
import sys
best = sys.argv[1:3] if len(sys.argv) >= 3 else ["D_HL_break_swinghigh+F", "E_climax_reclaimMA14+F"]
fig, ax = plt.subplots(figsize=(15, 7))
x = w.dt.dt.tz_localize(None)
for _, b in w.iterrows():
    t = b["dt"].tz_localize(None); col = "tab:green" if b.close >= b.open else "tab:red"
    ax.vlines(t, b.low, b.high, color=col, lw=0.8)
    ax.vlines(t, min(b.open, b.close), max(b.open, b.close), color=col, lw=3.5)
for n, cc in zip([7, 14, 25, 99, 120, 200], ["gold", "magenta", "deepskyblue", "orange", "brown", "purple"]):
    ax.plot(x, w[f"ma{n}"], color=cc, lw=1.2, label=f"MA{n}")
mk = ["^", "*"]; mc = ["blue", "black"]
for j, rn in enumerate(best):
    for _, t in trades[trades.rule == rn].iterrows():
        tt = pd.Timestamp(t.time)
        if x.min() <= tt <= x.max():
            ax.scatter(tt, t.entry*0.997 - j*0.35, marker=mk[j], s=180, color=mc[j], zorder=5,
                       label=f"{rn} {t.time[11:]} @{t.entry}")
lowr = w.loc[w.low.idxmin()]
ax.annotate(f"low {lowr.low} {lowr['dt'].strftime('%H:%M')}", (lowr["dt"].tz_localize(None), lowr.low),
            xytext=(0, -25), textcoords="offset points", ha="center")
ax.set_title("SOXL BingX 1m (NCSKSOXL2USD-USDT) 2026-10-08 rally start, UTC+8")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(D + "rally_start_1008.png", dpi=110)
