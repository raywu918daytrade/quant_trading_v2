"""MA 斜率 × 當沖.
A. D1 均線斜率 (MA5/10/20/60 的 3/5 日變化率, 加速度, 排列) 十分位 → 開→收漂移 (未扣成本) 與 開盤空 / 開盤多 的現實結算
B. D1 斜率條件 × ORB5 跌破 + 區間高停損 (家族進場) / × M1 死叉 + 進場前高停損
C. M1 均線: 09:30 / 10:00 MA20 斜率快照 × 方向; 死叉/黃金交叉觸發 (全 universe 與各 D1 條件)
D. 組合
-> stats/ma_slope_scan.md"""
import glob, numpy as np, pandas as pd
from config import PROC, STATS, COST, EXPLORE_END
FORCE = 0.08; SLIP = 0.002; LOCK = 0.07; STOP_SLIP = 0.001
d1 = pd.read_parquet(PROC / "d1_feat.parquet")
d1 = d1.sort_values(["stock_id", "date"]).reset_index(drop=True); g = d1.groupby("stock_id", sort=False)
for n in [5, 10, 20, 60]:
    ma = g.close.transform(lambda s: s.shift(1).rolling(n).mean())
    d1[f"ma{n}"] = ma
    for k in [3, 5]: d1[f"s{n}_{k}"] = (ma / ma.groupby(d1.stock_id).shift(k) - 1) * 100          # 昨日 MA 相對 k 日前 MA 的變化 (%)
    d1[f"acc{n}"] = d1[f"s{n}_5"] - d1[f"s{n}_5"].groupby(d1.stock_id).shift(5)                  # 加速度: 5 日斜率 vs 5 日前的 5 日斜率
d1["ma_bull"] = (d1.pc > d1.ma5) & (d1.ma5 > d1.ma20) & (d1.ma20 > d1.ma60); d1["ma_bear"] = (d1.pc < d1.ma5) & (d1.ma5 < d1.ma20) & (d1.ma20 < d1.ma60)
d1 = d1[d1.univ & (d1.date >= "2020-01-01")]
orb = pd.concat([pd.read_parquet(f, columns=["stock_id", "date", "auc_c", "o5_h", "o5_l", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open", "dn5_cumdv", "entry_open"]) for f in sorted(glob.glob(str(PROC / "orb_dn" / "*.parquet")))], ignore_index=True)
ma = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "m1_ma" / "*.parquet")))], ignore_index=True).drop(columns=["m1_open", "auc_c", "m1_high", "m1_low", "entry_open"])
df = d1.merge(orb, on=["stock_id", "date"], how="inner").merge(ma, on=["stock_id", "date"], how="left")
df = df[df.auc_c.notna()].reset_index(drop=True)
df["period"] = np.where(df.date <= EXPLORE_END, "explore", "validate"); df["year"] = df.date.dt.year
df["lu"] = df.pc * 1.1; df["ld"] = df.pc * 0.9; df["fup"] = df.pc * (1 + FORCE); df["fdn"] = df.pc * (1 - FORCE)
ALLDAYS = np.sort(df.date.unique())
print("rows", len(df))

def settle(s, side, ent, hi, lo, ex, stop_px=None):
    """現實結算. stop_px: 已含成交價 (NaN = 無停損) ; 停損後不再有強制風險."""
    if stop_px is not None:
        ex = stop_px.where(stop_px.notna(), ex)
        if side == "short": hi = hi.where(stop_px.isna(), np.minimum(hi, stop_px))
        else: lo = lo.where(stop_px.isna(), np.maximum(lo, stop_px))
    if side == "short":
        forced = hi >= s.fup; exf = np.where(forced, s.fup * (1 + SLIP), ex); locked = (~forced) & (exf >= s.lu * 0.999); r = (-(exf / ent - 1) - COST - np.where(locked, LOCK, 0)) * 100
    else:
        forced = lo <= s.fdn; exf = np.where(forced, s.fdn * (1 - SLIP), ex); locked = (~forced) & (exf <= s.ld * 1.001); r = ((exf / ent - 1) - COST - np.where(locked, LOCK, 0)) * 100
    out = s[["date", "stock_id", "period", "year", "dow", "mkt_gap", "dv20", "pc", "atr20", "gap"]].copy(); out["r"] = r; out["forced"] = np.asarray(forced); return out

def tr_open(mask, side):
    s = df[mask & df.entry_open.notna()]
    s = s[(s.o5_h < s.lu * 0.999)] if side == "short" else s[(s.o5_l > s.ld * 1.001)]
    return settle(s, side, s.entry_open, s.m1_high_x if "m1_high_x" in s.columns else s.high, s.low, s.auc_c)
def tr_orb(mask, stop=True):
    s = df[mask & df.dn5_t.notna() & (df.dn5_t <= 180) & (df.o5_h < df.lu * 0.999) & (df.o5_h < df.fup) & (df.dn5_px < df.fup)]
    st = (np.maximum(s.dn5_stop_hi_px, s.dn5_stop_hi_open) * (1 + STOP_SLIP)) if stop else pd.Series(np.nan, index=s.index)
    return settle(s, "short", s.dn5_px, s.dn5_hi_after, s.low, s.auc_c, pd.Series(np.asarray(st), index=s.index))
def tr_cross(mask, side, stop=True):
    nm = "dc" if side == "short" else "gc"
    s = df[mask & df[f"{nm}_t"].notna()]
    if side == "short": s = s[(s[f"{nm}_stop_lvl"] < s.lu * 0.999) & (s[f"{nm}_stop_lvl"] < s.fup)]
    else: s = s[(s[f"{nm}_stop_lvl"] > s.ld * 1.001) & (s[f"{nm}_stop_lvl"] > s.fdn)]
    if stop:
        st = np.where(s[f"{nm}_stop_t"].notna(), (np.maximum(s[f"{nm}_stop_lvl"], s[f"{nm}_stop_open"]) * (1 + STOP_SLIP)) if side == "short" else (np.minimum(s[f"{nm}_stop_lvl"], s[f"{nm}_stop_open"]) * (1 - STOP_SLIP)), np.nan)
    else: st = np.full(len(s), np.nan)
    o = settle(s, side, s[f"{nm}_px"], s[f"{nm}_hi_after"], s[f"{nm}_lo_after"], s.auc_c, pd.Series(st, index=s.index)); o["t"] = s[f"{nm}_t"].values; return o

def summ(x, label):
    if len(x) < 50: return dict(label=label, n=len(x))
    day = x.groupby("date").r.mean()
    o = dict(label=label, n=len(x), per_day=len(x) / x.date.nunique(), mean=x.r.mean(), med=x.r.median(), win=(x.r > 0).mean() * 100, forced=x.forced.mean() * 100, t_all=day.mean() / day.std() * np.sqrt(len(day)))
    for lab, p_ in [("ex", "explore"), ("va", "validate")]:
        q = x[x.period == p_]; dq = q.groupby("date").r.mean(); o[f"{lab}_mean"] = q.r.mean(); o[f"{lab}_t"] = dq.mean() / dq.std() * np.sqrt(len(dq)) if len(dq) > 2 else np.nan
    for y in range(2020, 2027): o[f"y{y}"] = x[x.year == y].r.mean()
    return o
def port(x, N, label):
    x = x.sort_values(["date"]); top = x.groupby("date").head(N); day = top.groupby("date").r.mean(); s = pd.Series(0.0, index=ALLDAYS); s.loc[day.index] = day.values
    eq = (1 + s / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100; mon = s.groupby(s.index.to_period("M")).sum(); yr = s.groupby(s.index.year).mean()
    return dict(label=label, N=N, trade_days=int((s != 0).sum()), avg_n=top.groupby("date").size().mean(), per_trade=top.r.mean(), win=(top.r > 0).mean() * 100, forced=top.forced.mean() * 100, day_mean=s.mean(), day_std=s.std(),
                sharpe=s.mean() / s.std() * np.sqrt(245), mdd=mdd, worst_day=s.min(), worst_month=mon.min(), ex_day=s[s.index <= EXPLORE_END].mean(), va_day=s[s.index > EXPLORE_END].mean(), **{f"y{y}": yr.get(y, np.nan) for y in range(2020, 2027)})
def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    rr = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in rr.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
C = [("條件", "label"), ("筆數", "n"), ("每日筆", "per_day"), ("每筆淨%", "mean"), ("中位", "med"), ("勝率", "win"), ("強制%", "forced"), ("t(全)", "t_all"), ("探索%", "ex_mean"), ("探索t", "ex_t"), ("驗證%", "va_mean"), ("驗證t", "va_t")] + [(str(y), f"y{y}") for y in range(2020, 2027)]
CP = [("組合", "label"), ("N", "N"), ("交易日", "trade_days"), ("日均檔", "avg_n"), ("每筆%", "per_trade"), ("勝率", "win"), ("強制%", "forced"), ("日淨%", "day_mean"), ("日std", "day_std"), ("Sharpe", "sharpe"), ("MDD%", "mdd"), ("最差日", "worst_day"), ("最差月", "worst_month"), ("探索日淨", "ex_day"), ("驗證日淨", "va_day")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

md = ["# MA 斜率 × 當沖", "", f"universe 2020-01~{df.date.max().date()}, {len(df):,} 股票日。D1 斜率 = 昨日 MA 相對 k 日前 MA 的變化 % (只用昨日以前資料)。現實結算同前 (+8%/−8% 強制、鎖死 −7%、成本 0.23%)。", ""]

# ===== A. D1 斜率十分位 → 開→收漂移 =====
rows = []
for col, lab in [("s5_3", "MA5 3日斜率"), ("s10_5", "MA10 5日斜率"), ("s20_5", "MA20 5日斜率"), ("s60_5", "MA60 5日斜率"), ("acc10", "MA10 加速度"), ("acc20", "MA20 加速度")]:
    x = df[df[col].notna()].copy(); x["dec"] = pd.qcut(x[col], 10, labels=False, duplicates="drop")
    grp = x.groupby("dec").agg(lo=(col, "min"), hi=(col, "max"), n=("intra", "size"), intra=("intra", "mean"), gap=("gap", "mean"), win=("intra", lambda s: (s > 0).mean() * 100), forced=("touch_up", "mean"))
    for d_, r in grp.iterrows():
        rows.append(dict(label=f"{lab} 十分位 {int(d_)+1} ({r.lo:.1f}~{r.hi:.1f}%)", n=int(r.n), intra=r.intra * 100, gap=r.gap * 100, win=r.win, touch=r.forced * 100))
md += ["## A. D1 斜率十分位: 當日 開→收 平均 (未扣成本; 負 = 偏空)", "", table(rows, [("切分", "label"), ("n", "n"), ("開→收%", "intra"), ("跳空%", "gap"), ("開→收>0 %", "win"), ("盤中摸漲停%", "touch")]), ""]

# ===== B. D1 斜率條件 × 進場方式 =====
COND = {"全部": pd.Series(True, index=df.index),
        "MA5 3日斜率 ≥ +5%": df.s5_3 >= 5, "MA5 3日斜率 ≥ +8%": df.s5_3 >= 8, "MA5 3日斜率 ≤ −5%": df.s5_3 <= -5,
        "MA10 5日斜率 ≥ +5%": df.s10_5 >= 5, "MA10 5日斜率 ≥ +10%": df.s10_5 >= 10, "MA10 5日斜率 ≤ −5%": df.s10_5 <= -5,
        "MA20 5日斜率 ≥ +5%": df.s20_5 >= 5, "MA20 5日斜率 ≥ +10%": df.s20_5 >= 10, "MA20 5日斜率 ≤ −5%": df.s20_5 <= -5,
        "MA60 5日斜率 ≥ +3%": df.s60_5 >= 3, "MA60 5日斜率 ≤ −3%": df.s60_5 <= -3,
        "MA10 向上且加速 (斜率>0, 加速度>0)": (df.s10_5 > 0) & (df.acc10 > 0), "MA10 向上但減速": (df.s10_5 > 0) & (df.acc10 <= 0),
        "MA10 向下且加速": (df.s10_5 < 0) & (df.acc10 < 0), "MA10 向下但減速": (df.s10_5 < 0) & (df.acc10 >= 0),
        "多頭排列 (價>MA5>MA20>MA60)": df.ma_bull, "空頭排列": df.ma_bear,
        "MA5 斜率 ≥5% ∩ 跳空 ≥2%": (df.s5_3 >= 5) & (df.gap >= 0.02), "MA20 斜率 ≥10% ∩ 跳空 ≥2%": (df.s20_5 >= 10) & (df.gap >= 0.02),
        "MA5 斜率 ≥5% ∩ ATR ≥3%": (df.s5_3 >= 5) & (df.atr20 >= 0.03), "MA20 斜率 ≥10% ∩ ATR ≥3%": (df.s20_5 >= 10) & (df.atr20 >= 0.03)}
for sec, fn, nm in [("B1. 09:00 開盤空 (收盤出)", lambda m: tr_open(m, "short"), "開盤空"), ("B2. 09:00 開盤多 (收盤出)", lambda m: tr_open(m, "long"), "開盤多"),
                    ("B3. 跌破開盤 5 分區間低 → 空, 區間高停損", lambda m: tr_orb(m, True), "ORB5 空+停損"),
                    ("B4. M1 死叉 (MA10 下穿 MA20, MA20 斜率<0) → 空, 進場前高停損", lambda m: tr_cross(m, "short", True), "死叉空+停損"),
                    ("B5. M1 黃金交叉 → 多, 進場前低停損", lambda m: tr_cross(m, "long", True), "金叉多+停損")]:
    rows = [summ(fn(m), k) for k, m in COND.items()]
    md += [f"## {sec}", "", table(rows, C), ""]

# ===== C. M1 MA20 斜率快照 =====
rows = []
for T in ["s0930", "s1000"]:
    x = df[df[f"{T}_ma20s"].notna()].copy(); x["dec"] = pd.qcut(x[f"{T}_ma20s"], 5, labels=False, duplicates="drop")
    px = x[f"{T}_px"]
    for d_, grp in x.groupby("dec"):
        lo_, hi_ = grp[f"{T}_ma20s"].min(), grp[f"{T}_ma20s"].max()
        gs = grp[grp[f"{T}_runhi"] < grp.fup]; sh = settle(gs, "short", gs[f"{T}_px"], gs.high, gs.low, gs.auc_c)
        gl = grp[grp[f"{T}_runlo"] > grp.fdn]; lg = settle(gl, "long", gl[f"{T}_px"], gl.high, gl.low, gl.auc_c)
        rows.append(dict(label=f"{'09:30' if T=='s0930' else '10:00'} MA20 5分斜率 五分位 {int(d_)+1} ({lo_:.2f}~{hi_:.2f}%)", n=len(grp), rest=((grp.auc_c / grp[f"{T}_px"] - 1).mean() * 100),
                         short=sh.r.mean(), long_=lg.r.mean(), pma=(grp[f"{T}_pma20"].mean() * 100)))
md += ["## C. M1 MA20 斜率快照 (09:30 / 10:00) → 之後到收盤", "", table(rows, [("切分", "label"), ("n", "n"), ("之後到收盤 %", "rest"), ("空 每筆淨%", "short"), ("多 每筆淨%", "long_"), ("價/MA20−1 %", "pma")]), "",
       "注意: 快照時價格通常已偏離, 斜率最負的那組進場價也最低。", ""]
rows = []
for col, lab in [("dc", "死叉 空+停損"), ("gc", "金叉 多+停損")]:
    side = "short" if col == "dc" else "long"
    x = tr_cross(COND["全部"], side, True)
    rows.append(summ(x, f"{lab} 全部")); rows.append(summ(x[x.t <= 60], f"{lab} 10:00 前")); rows.append(summ(x[(x.t > 60) & (x.t <= 120)], f"{lab} 10:00~11:00")); rows.append(summ(x[x.t > 120], f"{lab} 11:00 後"))
    rows.append(summ(tr_cross(COND["全部"], side, False), f"{lab.split(' ')[0]} 無停損 (收盤出)"))
    rows.append(summ(tr_cross(COND["全部"] & (df.atr20 >= 0.03), side, True), f"{lab} ATR≥3%"))
    rows.append(summ(tr_cross(COND["全部"] & (df.atr20 >= 0.03) & (df.gap >= 0.02 if side == "short" else df.gap <= -0.02), side, True), f"{lab} ATR≥3% ∩ 跳空 {'≥+2%' if side=='short' else '≤−2%'}"))
md += ["## C2. M1 交叉觸發細分", "", table(rows, C), ""]

# ===== D. 組合 =====
rows = []
def filt(x): return x[(x.dow != 0) & (x.mkt_gap >= -0.01) & (x.dv20 >= 3e8) & (x.pc <= 300)]
for k in ["MA5 3日斜率 ≥ +5%", "MA10 5日斜率 ≥ +10%", "MA20 5日斜率 ≥ +10%", "MA10 向上且加速 (斜率>0, 加速度>0)", "MA5 斜率 ≥5% ∩ ATR ≥3%", "MA20 斜率 ≥10% ∩ ATR ≥3%"]:
    rows.append(port(filt(tr_open(COND[k], "short")), 10, f"{k} / 開盤空"))
    rows.append(port(filt(tr_orb(COND[k], True)), 10, f"{k} / ORB5 空+區間高停損"))
    rows.append(port(filt(tr_cross(COND[k], "short", True)), 10, f"{k} / 死叉空+停損"))
rows.append(port(filt(tr_orb((df.gap >= 0.02) & (df.atr20 >= 0.03), True)), 10, "對照: 跳空≥2% ∩ ATR≥3% / ORB5 空+區間高停損 (前一節定案)"))
rows.append(port(filt(tr_orb((df.gap >= 0.02) & (df.atr20 >= 0.03) & (df.s5_3 >= 3), True)), 10, "跳空≥2% ∩ ATR≥3% ∩ MA5 斜率≥3% / ORB5 空+停損"))
rows.append(port(filt(tr_orb((df.gap >= 0.02) & (df.atr20 >= 0.03) & (df.s20_5 >= 5), True)), 10, "跳空≥2% ∩ ATR≥3% ∩ MA20 斜率≥5% / ORB5 空+停損"))
rows.append(port(filt(tr_orb((df.gap >= 0.02) & (df.atr20 >= 0.03) & (df.s20_5 < 5), True)), 10, "跳空≥2% ∩ ATR≥3% ∩ MA20 斜率<5% / ORB5 空+停損"))
md += ["## D. 每日 10 檔組合 (濾網: 非週一、大盤跳空≥−1%、成交額≥3億、股價≤300)", "", table(rows, CP), ""]
(STATS / "ma_slope_scan.md").write_text("\n".join(md), encoding="utf-8"); print("written")
