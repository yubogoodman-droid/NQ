"""Signal rules. Signal-only: this module never places orders."""
import math
import h4high_ma30_rule, h4low_ma_rule
from datetime import datetime, timezone, timedelta
TPE = timezone(timedelta(hours=8))
def tpe(ms): return datetime.fromtimestamp(ms / 1000, TPE)
def ok(*v): return all(x is not None and not (isinstance(x, float) and math.isnan(x)) for x in v)

def default_state():
    return {"last1": 0, "last5": 0, "cd": {}, "r1_armed": True, "asia": {}}

def _cool(st, key, ms, mins):
    last = st["cd"].get(key, 0)
    if ms - last < mins * 60000: return False
    st["cd"][key] = ms; return True

def us_session(ms):
    d = tpe(ms); hm = d.hour * 60 + d.minute
    if hm >= 21 * 60 + 30: return d.weekday() < 5
    if hm < 4 * 60: return (d - timedelta(days=1)).weekday() < 5
    return False

def eval_1m(df, i, st, out):
    r = df.iloc[i]; p = df.iloc[i - 1]; t = int(r.t); close_ms = t + 60000
    # 1 空頭排列跌破 MA200 (armed)
    if ok(r.ma200, p.ma200, r.ma7, r.ma25):
        if r.close > r.ma200: st["r1_armed"] = True
        if st["r1_armed"] and r.ma7 < r.ma25 and r.close < r.ma200 and p.close >= p.ma200:
            st["r1_armed"] = False
            if _cool(st, "r1", close_ms, 5):
                out("r1", t, "🔴 SOXL 一分K 空頭排列跌破 MA200",
                    f"收 {r.close:.2f} < MA200 {r.ma200:.2f}；MA7 {r.ma7:.2f} < MA25 {r.ma25:.2f}")
    # 2 多頭排列由下站上 MA200
    if ok(r.ma200, p.ma200, r.ma7, r.ma14, r.ma30):
        if r.ma7 > r.ma14 > r.ma30 and p.close < p.ma200 and r.close > r.ma200 and _cool(st, "r2", close_ms, 15):
            out("r2", t, "🟢 SOXL 一分K 多頭排列站上 MA200",
                f"收 {r.close:.2f} > MA200 {r.ma200:.2f}；MA7 {r.ma7:.2f} > MA14 {r.ma14:.2f} > MA30 {r.ma30:.2f}")
    # 3 起漲點
    if i >= 90 and us_session(t) and ok(r.ma200):
        w = df.iloc[i - 89:i + 1]
        h = w.high.idxmax(); H = df.high[h]
        after = df.iloc[h + 1:i]
        if len(after) >= 4:
            l1 = after.low.idxmin(); L1 = df.low[l1]
            if L1 <= H * 0.975 and l1 + 2 <= i - 1:
                seg = df.iloc[l1 + 2:i]
                l2 = seg.low.idxmin(); L2 = df.low[l2]
                if l2 - l1 >= 2 and L2 > L1 and ok(df.ma200[l2]) and L2 >= df.ma200[l2] * 0.985:
                    B = df.high[l1 + 1:l2].max()
                    if r.close > B and p.close <= B and abs(r.close / r.ma200 - 1) <= 0.01 and _cool(st, "r3", close_ms, 30):
                        out("r3", t, "🚀 SOXL 一分K 起漲點（急跌後更高低點突破）",
                            f"90K高 {H:.2f} → 低1 {L1:.2f}（{(L1/H-1)*100:.1f}%）→ 低2 {L2:.2f}；收 {r.close:.2f} 突破反彈高 {B:.2f}；MA200 {r.ma200:.2f}")
    # 7 亞洲盤區間 06:00-07:00
    d = tpe(t); day = d.strftime("%Y-%m-%d"); a = st["asia"]
    if a.get("day") != day:
        a.clear(); a.update(day=day, hi=None, lo=None, sent=False, up=False, dn=False)
    if d.hour == 6:
        a["hi"] = max(a["hi"] or r.high, r.high); a["lo"] = min(a["lo"] or r.low, r.low)
        if d.minute == 59 and not a["sent"]:
            a["sent"] = True
            out("r7", t, "🌏 SOXL 一分K 亞洲盤區間（06:00–07:00）", f"高 {a['hi']:.2f} / 低 {a['lo']:.2f}")
    elif d.hour >= 7 and a["hi"] is not None:
        if not a["up"] and r.close > a["hi"]:
            a["up"] = True; out("r7", t, "🟢 SOXL 一分K ASIA up 突破亞洲盤高", f"收 {r.close:.2f} > 區間高 {a['hi']:.2f}")
        if not a["dn"] and r.close < a["lo"]:
            a["dn"] = True; out("r7", t, "🔴 SOXL 一分K ASIA dn 跌破亞洲盤低", f"收 {r.close:.2f} < 區間低 {a['lo']:.2f}")

    # 8 四小時新高後跌破 MA30
    h4high_ma30_rule.eval_1m(df, i, st, out, tpe, _cool)
    h4low_ma_rule.D1 = df  # r9 needs 1m lows

def eval_5m(df, j, st, out):
    r = df.iloc[j]; p = df.iloc[j - 1]; t = int(r.t); close_ms = t + 300000
    # 4 收盤站上/跌破 MA200
    if j >= 200 and ok(r.ma200, p.ma200):
        if p.close <= p.ma200 and r.close > r.ma200 and _cool(st, "r4up", close_ms, 15):
            out("r4", t, "🟢 SOXL 5分K 站上 MA200（五分K收盤）", f"收 {r.close:.2f} > MA200 {r.ma200:.2f}")
        if p.close >= p.ma200 and r.close < r.ma200 and _cool(st, "r4dn", close_ms, 15):
            out("r4", t, "🔴 SOXL 5分K 跌破 MA200（五分K收盤）", f"收 {r.close:.2f} < MA200 {r.ma200:.2f}")
    # 5 反彈碰下彎 MA25
    if j >= 31 and ok(r.ma25, p.ma25, df.ma25[j - 6]):
        m = r.ma25
        if m < df.ma25[j - 6] and p.close < p.ma25 and r.close < m and abs(r.high / m - 1) <= 0.001 and _cool(st, "r5", close_ms, 30):
            out("r5", t, "🟠 SOXL 五分K 反彈碰下彎 MA25",
                f"高 {r.high:.2f} 觸 MA25 {m:.2f}（6根前 {df.ma25[j-6]:.2f}）；收 {r.close:.2f}")
    # 6 由下假突破做空
    if j >= 230 and ok(r.ma200, df.ma200[j - 24], r.ma99, r.ma120, df.ma99[j - 12], df.ma120[j - 12]):
        if r.close < r.ma200 and r.ma200 < df.ma200[j - 24] and r.ma99 < df.ma99[j - 12] and r.ma120 < df.ma120[j - 12]:
            for k in range(j - 1, j - 13, -1):
                if df.close[k] > df.ma200[k]:
                    if df.close[k - 1] > df.ma200[k - 1]:
                        continue  # still above; search for first breakout bar
                    if all(df.close[x] < df.ma200[x] for x in range(k - 6, k)) and _cool(st, "r6", close_ms, 30):
                        out("r6", t, "🔻 SOXL 五分K 由下假突破做空",
                            f"收 {r.close:.2f} 回落 MA200 {r.ma200:.2f} 下（{tpe(int(df.t[k])).strftime('%H:%M')} 曾站上）；參考停損 {r.close*1.01:.2f}（+1%）目標 {r.close*0.98:.2f}（−2%）")
                    break
                else:
                    break  # previous bar below MA200 → not the first close back below
    # 9 四小時新低後站上 MA25/MA30 三根守住
    h4low_ma_rule.eval_5m(df, j, st, out, tpe, _cool)
