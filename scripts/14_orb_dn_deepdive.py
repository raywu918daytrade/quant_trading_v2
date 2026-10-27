"""D1 過熱 × M1 開盤區間向下突破 → 空 的深入掃描.
A. 過熱變數切檔 (W=15 基準): 跳空 / 昨日漲幅 / 5 日漲幅 / 20 日漲幅 / 昨日量比 / 昨日上影 / 昨日漲停 / ATR / 20 日位置
B. 區間寬度 W=5/10/15/30 × 進場版本 (突破價 / 收盤確認 / 突破後 5 分收) × 截止時間
C. 出場: 收盤 / 11:00 / 12:00 / 13:00 / 三種停損
D. 濾網 + 每日 N 檔組合 (日報酬序列、Sharpe、MDD、逐年、容量)
現實結算: +8% 強制回補 (+0.2% 滑價), 鎖死 −7%, 成本 0.23%; 進場前 (區間內) 已到 +8% 或摸漲停的不進.
-> stats/orb_dn_deepdive.md, stats/orb_dn_daily.csv, data/processed/orb_dn_trades.parquet"""
import glob, numpy as np, pandas as pd
from config import PROC, STATS, COST, EXPLORE_END
FORCE = 0.08; SLIP = 0.002; LOCK = 0.07; STOP_SLIP = 0.001
d1 = pd.read_parquet(PROC / "d1_feat.parquet"); d1 = d1[d1.univ & (d1.date >= "2020-01-01")]
d1 = d1[["stock_id", "date", "name", "pc", "open", "high", "low", "close", "volume", "gap", "p_ret", "p_intra", "p_rng", "p_uplim", "p_vr", "vr5_20", "ret5", "ret20", "pos20", "atr20", "sd20", "dv20", "mkt_gap", "mkt_pret", "mkt_breadth", "dow", "range10", "touch_up"]]
# 昨日上影線 (需從 d1_feat 重算: 用 S_shooting 近似不夠, 直接重建)
raw = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open", "high", "low", "close", "pc"]).sort_values(["stock_id", "date"])
raw["uw"] = (raw.high - raw[["open", "close"]].max(axis=1)) / raw.pc; raw["body"] = (raw.close - raw.open).abs() / raw.pc
raw["p_uw"] = raw.groupby("stock_id", sort=False).uw.shift(1); raw["p_body"] = raw.groupby("stock_id", sort=False).body.shift(1)
d1 = d1.merge(raw[["stock_id", "date", "p_uw", "p_body"]], on=["stock_id", "date"], how="left")
m1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "orb_dn" / "*.parquet")))], ignore_index=True)
df = d1.merge(m1, on=["stock_id", "date"], how="inner"); df = df[df.auc_c.notna()].reset_index(drop=True)
df["period"] = np.where(df.date <= EXPLORE_END, "explore", "validate"); df["year"] = df.date.dt.year
df["lu"] = df.pc * 1.1; df["fpx"] = df.pc * (1 + FORCE)
ALLDAYS = np.sort(df.date.unique())

def trades(mask, W=15, entry="brk", exit_="close", deadline=180):
    """回傳每筆: date, stock_id, r (%), forced, t, entry, cumdv (突破時累積成交額, 元), 以及 d1 欄位."""
    p = f"dn{W}_"
    s = df[mask & df[p + "t"].notna() & (df[p + "t"] <= deadline)].copy()
    s = s[(s[f"o{W}_h"] < s.lu * 0.999) & (s[f"o{W}_h"] < s.fpx)]                      # 區間內沒摸漲停、沒到 +8%
    if entry == "brk": ent = s[p + "px"]; t = s[p + "t"]; hi = s[p + "hi_after"]
    elif entry == "cc": s = s[s[p + "cc_t"].notna() & (s[p + "cc_t"] <= deadline)]; ent = s[p + "cc_px"]; t = s[p + "cc_t"]; hi = s[p + "cc_hi_after"]
    elif entry == "c5": s = s[s[p + "c5"].notna()]; ent = s[p + "c5"]; t = s[p + "t"] + 5; hi = s[p + "hi_after"]   # 近似: 突破後 5 分收盤進, hi_after 含這 5 分 (偏保守)
    s = s[ent < s.fpx]
    ent = ent.loc[s.index]; hi = hi.loc[s.index]; t = t.loc[s.index]
    if exit_ == "close": ex = s.auc_c.copy()
    elif exit_ in ("c1100", "c1200", "c1300"): ex = s[exit_].where(s[exit_].notna(), s.auc_c)
    elif exit_.startswith("stop_"):
        st = s[p + exit_ + "_px"]
        if (p + exit_ + "_open") in s.columns: st = np.maximum(st, s[p + exit_ + "_open"]) * (1 + STOP_SLIP)   # 跳空穿越停損價 → 以該 bar 開盤成交, 另加滑價
        st = pd.Series(st, index=s.index); ex = st.where(st.notna(), s.auc_c)
        hi = hi.where(st.isna(), np.minimum(hi, st))                                             # 停損出場後不再有強制回補風險
    forced = hi >= s.fpx
    exf = np.where(forced, s.fpx * (1 + SLIP), ex)
    locked = (~forced) & (exf >= s.lu * 0.999)
    r = (-(exf / ent - 1) - COST - np.where(locked, LOCK, 0)) * 100
    out = s[["date", "stock_id", "period", "year", "gap", "p_ret", "ret5", "ret20", "p_vr", "p_uw", "atr20", "dv20", "pc", "mkt_gap", "dow", "p_uplim", "pos20"]].copy()
    out["r"] = r; out["forced"] = forced.values; out["stopped"] = (s[p + exit_ + "_px"].notna().values if exit_.startswith("stop_") else np.zeros(len(s), bool)); out["t_stop"] = (s[p + exit_ + "_t"].values if exit_.startswith("stop_") else np.nan); out["t"] = t.values; out["entry"] = ent.values; out["cumdv"] = s[p + "cumdv"].values; out["naive"] = (-(s.auc_c / ent - 1) - COST).values * 100
    return out

def summ(x, label):
    if len(x) < 50: return dict(label=label, n=len(x))
    day = x.groupby("date").r.mean()
    o = dict(label=label, n=len(x), per_day=len(x) / x.date.nunique(), mean=x.r.mean(), med=x.r.median(), win=(x.r > 0).mean() * 100, forced=x.forced.mean() * 100, t_entry=x.t.mean(),
             t_all=day.mean() / day.std() * np.sqrt(len(day)))
    for lab, p_ in [("ex", "explore"), ("va", "validate")]:
        q = x[x.period == p_]; dq = q.groupby("date").r.mean(); o[f"{lab}_mean"] = q.r.mean(); o[f"{lab}_t"] = dq.mean() / dq.std() * np.sqrt(len(dq)) if len(dq) > 2 else np.nan
    for y in range(2020, 2027): o[f"y{y}"] = x[x.year == y].r.mean()
    return o
def port(x, N, label, rank_col="t", asc=True):
    x = x.sort_values(["date", rank_col], ascending=[True, asc]); top = x.groupby("date").head(N)
    day = top.groupby("date").r.mean(); s = pd.Series(0.0, index=ALLDAYS); s.loc[day.index] = day.values
    eq = (1 + s / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100; mon = s.groupby(s.index.to_period("M")).sum()
    yr = s.groupby(s.index.year).mean()
    o = dict(label=label, N=N, trade_days=int((s != 0).sum()), avg_n=top.groupby("date").size().mean(), per_trade=top.r.mean(), win=(top.r > 0).mean() * 100, forced=top.forced.mean() * 100,
             day_mean=s.mean(), day_std=s.std(), sharpe=s.mean() / s.std() * np.sqrt(245), mdd=mdd, worst_day=s.min(), worst_month=mon.min(), neg_months=(mon < 0).mean() * 100,
             ex_day=s[s.index <= EXPLORE_END].mean(), va_day=s[s.index > EXPLORE_END].mean(), **{f"y{y}": yr.get(y, np.nan) for y in range(2020, 2027)})
    return s, o
def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    rr = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in rr.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
C = [("條件", "label"), ("筆數", "n"), ("每日筆", "per_day"), ("每筆淨%", "mean"), ("中位", "med"), ("勝率", "win"), ("強制%", "forced"), ("平均進場分", "t_entry"), ("t(全)", "t_all"),
     ("探索%", "ex_mean"), ("探索t", "ex_t"), ("驗證%", "va_mean"), ("驗證t", "va_t")] + [(str(y), f"y{y}") for y in range(2020, 2027)]
CP = [("組合", "label"), ("N", "N"), ("交易日", "trade_days"), ("日均檔", "avg_n"), ("每筆%", "per_trade"), ("勝率", "win"), ("強制%", "forced"), ("日淨%", "day_mean"), ("日std", "day_std"),
      ("Sharpe", "sharpe"), ("MDD%", "mdd"), ("最差日", "worst_day"), ("最差月", "worst_month"), ("負月%", "neg_months"), ("探索日淨", "ex_day"), ("驗證日淨", "va_day")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

md = ["# D1 過熱 × M1 開盤區間向下突破 → 空: 深入掃描", "",
      f"universe 2020-01~{df.date.max().date()}, {len(df):,} 股票日。進場 = 第一根 low 跌破開盤 W 分鐘區間低點的 bar, 進場價 = min(區間低, bar 開)；12:00 前。出場 13:30 收盤競價。",
      f"現實結算: 進場後到昨收 +{FORCE*100:.0f}% 立即回補 (+{SLIP*100:.1f}%)、收盤鎖漲停 −{LOCK*100:.0f}%、成本 {COST*100:.2f}%；區間內已摸漲停或到 +8% 的不進。t 以每日平均報酬計。", ""]

# ===== A. 過熱變數切檔 (W=15, 全 universe 無過熱條件為基準) =====
ALL = pd.Series(True, index=df.index)
rows = [summ(trades(ALL), "全 universe (無過熱條件)")]
for lab, cond in [("跳空 <0", df.gap < 0), ("跳空 0~2%", (df.gap >= 0) & (df.gap < 0.02)), ("跳空 2~4%", (df.gap >= 0.02) & (df.gap < 0.04)), ("跳空 4~6%", (df.gap >= 0.04) & (df.gap < 0.06)), ("跳空 ≥6%", df.gap >= 0.06),
                  ("昨日漲幅 <0", df.p_ret < 0), ("昨日漲幅 0~3%", (df.p_ret >= 0) & (df.p_ret < 0.03)), ("昨日漲幅 3~6%", (df.p_ret >= 0.03) & (df.p_ret < 0.06)), ("昨日漲幅 6~9.5%", (df.p_ret >= 0.06) & (df.p_ret < 0.095)), ("昨日漲停", df.p_uplim.astype(bool)),
                  ("5 日漲幅 <0", df.ret5 < 0), ("5 日漲幅 0~5%", (df.ret5 >= 0) & (df.ret5 < 0.05)), ("5 日漲幅 5~10%", (df.ret5 >= 0.05) & (df.ret5 < 0.10)), ("5 日漲幅 10~20%", (df.ret5 >= 0.10) & (df.ret5 < 0.20)), ("5 日漲幅 ≥20%", df.ret5 >= 0.20),
                  ("20 日漲幅 <0", df.ret20 < 0), ("20 日漲幅 0~15%", (df.ret20 >= 0) & (df.ret20 < 0.15)), ("20 日漲幅 15~30%", (df.ret20 >= 0.15) & (df.ret20 < 0.30)), ("20 日漲幅 30~50%", (df.ret20 >= 0.30) & (df.ret20 < 0.50)), ("20 日漲幅 ≥50%", df.ret20 >= 0.50),
                  ("昨日量比 <1", df.p_vr < 1), ("昨日量比 1~2", (df.p_vr >= 1) & (df.p_vr < 2)), ("昨日量比 2~3", (df.p_vr >= 2) & (df.p_vr < 3)), ("昨日量比 ≥3", df.p_vr >= 3),
                  ("昨日上影 <1%", df.p_uw < 0.01), ("昨日上影 1~2%", (df.p_uw >= 0.01) & (df.p_uw < 0.02)), ("昨日上影 2~4%", (df.p_uw >= 0.02) & (df.p_uw < 0.04)), ("昨日上影 ≥4%", df.p_uw >= 0.04),
                  ("ATR20 <2%", df.atr20 < 0.02), ("ATR20 2~3%", (df.atr20 >= 0.02) & (df.atr20 < 0.03)), ("ATR20 3~5%", (df.atr20 >= 0.03) & (df.atr20 < 0.05)), ("ATR20 5~7%", (df.atr20 >= 0.05) & (df.atr20 < 0.07)), ("ATR20 ≥7%", df.atr20 >= 0.07),
                  ("20 日位置 <0.5", df.pos20 < 0.5), ("20 日位置 0.5~0.9", (df.pos20 >= 0.5) & (df.pos20 < 0.9)), ("20 日位置 ≥0.9 (在高點)", df.pos20 >= 0.9),
                  ("昨日摸漲停沒鎖", df.touch_up.shift(0) & False)]:
    if cond.sum() == 0: continue
    rows.append(summ(trades(cond), lab))
md += ["## A. 過熱變數切檔 (W=15 分區間, 突破價進, 收盤出)", "", table(rows, C), ""]

# ===== 過熱定義候選 (組合) =====
HOT = {
    "H1 跳空≥2%": df.gap >= 0.02,
    "H2 昨日漲幅≥3%": df.p_ret >= 0.03,
    "H3 5日漲幅≥10%": df.ret5 >= 0.10,
    "H4 20日漲幅≥30%": df.ret20 >= 0.30,
    "H5 昨日上影≥2%": df.p_uw >= 0.02,
    "H6 昨日量比≥2": df.p_vr >= 2,
    "H7 ATR≥5%": df.atr20 >= 0.05,
    "H8 跳空≥2% ∩ ATR≥3%": (df.gap >= 0.02) & (df.atr20 >= 0.03),
    "H9 跳空≥2% ∩ 5日漲≥5%": (df.gap >= 0.02) & (df.ret5 >= 0.05),
    "H10 (跳空≥2% ∪ 昨漲≥3% ∪ 上影≥2%) ∩ ATR≥3%": ((df.gap >= 0.02) | (df.p_ret >= 0.03) | (df.p_uw >= 0.02)) & (df.atr20 >= 0.03),
    "H11 跳空≥1% ∩ 20日漲≥15% ∩ ATR≥3%": (df.gap >= 0.01) & (df.ret20 >= 0.15) & (df.atr20 >= 0.03),
    "H12 過熱分數≥2 (跳空≥2,昨漲≥3,5日≥10,20日≥30,上影≥2,量比≥2 各 1 分)": ((df.gap >= 0.02).astype(int) + (df.p_ret >= 0.03).astype(int) + (df.ret5 >= 0.10).astype(int) + (df.ret20 >= 0.30).astype(int) + (df.p_uw >= 0.02).astype(int) + (df.p_vr >= 2).astype(int)) >= 2,
    "H13 過熱分數≥3": ((df.gap >= 0.02).astype(int) + (df.p_ret >= 0.03).astype(int) + (df.ret5 >= 0.10).astype(int) + (df.ret20 >= 0.30).astype(int) + (df.p_uw >= 0.02).astype(int) + (df.p_vr >= 2).astype(int)) >= 3,
}
rows = [summ(trades(m), k) for k, m in HOT.items()]
md += ["## A2. 過熱定義候選 (W=15)", "", table(rows, C), ""]
best_hot = max(HOT, key=lambda k: min(summ(trades(HOT[k]), k).get("ex_t", -9) or -9, summ(trades(HOT[k]), k).get("va_t", -9) or -9))
# 改用固定主定義 (避免挑最好的過度擬合): 跳空≥2% ∩ ATR≥3% 若存在
MAIN = "H8 跳空≥2% ∩ ATR≥3%"; hm = HOT[MAIN]
md += [f"以下用主定義 **{MAIN}** (min(探索t,驗證t) 最高的是 {best_hot})。", ""]

# ===== B. 區間寬度 × 進場版本 × 截止 =====
rows = []
for W in [5, 10, 15, 30]:
    for entry, el in [("brk", "突破價進"), ("cc", "收盤確認進 (第一根收在區間低之下)"), ("c5", "突破後 5 分收盤進")]:
        rows.append(summ(trades(hm, W, entry), f"W={W} / {el}"))
for W in [5, 15]:
    for dl, dll in [(30, "09:30 前"), (60, "10:00 前"), (120, "11:00 前"), (180, "12:00 前")]:
        rows.append(summ(trades(hm, W, "brk", deadline=dl), f"W={W} / 突破價進 / 截止 {dll}"))
md += ["## B. 區間寬度 × 進場版本 × 突破截止時間 (主定義)", "", table(rows, C), ""]

# ===== C. 出場 =====
rows = []
for W in [5, 15]:
    for ex, exl in [("close", "13:30 收盤"), ("c1300", "13:00"), ("c1200", "12:00"), ("c1100", "11:00"), ("stop_in", "停損: 收回區間內 (否則收盤)"), ("stop_vwap", "停損: 站回 VWAP"), ("stop_hi", "停損: 突破區間高")]:
        rows.append(summ(trades(hm, W, "brk", ex), f"W={W} / {exl}"))
rows2 = []
for W in [5, 15]:
    for ex, exl in [("stop_k05", "停損 = 區間低 + 0.5×寬 (區間中點)"), ("stop_hi", "停損 = 區間高 (k=1)"), ("stop_k15", "停損 = 區間低 + 1.5×寬"), ("stop_k20", "停損 = 區間低 + 2×寬")]:
        tt = trades(hm, W, "brk", ex); o = summ(tt, f"W={W} / {exl}"); o["stopped"] = tt.stopped.mean() * 100; o["t_stop"] = tt.t_stop.mean()
        o["r_stopped"] = tt[tt.stopped].r.mean(); o["r_unstopped"] = tt[~tt.stopped].r.mean(); o["naive_stopped"] = tt[tt.stopped].naive.mean()
        rows2.append(o)
C2 = C[:9] + [("停損率%", "stopped"), ("平均停損分", "t_stop"), ("被停損那些 每筆%", "r_stopped"), ("同批若抱到收盤 %", "naive_stopped"), ("未停損 每筆%", "r_unstopped")] + C[9:13]
md += ["## C. 出場方式 (主定義)", "", table(rows, C), "", "### C2. 停損高度格點 (停損成交價 = max(停損價, 該分鐘開盤) + 0.1% 滑價)", "", table(rows2, C2), ""]

# ===== D. 濾網 + 組合 =====
base = trades(hm, 15, "brk")
base.to_parquet(PROC / "orb_dn_trades.parquet", index=False)
rows = [summ(base, "主定義 W=15 基底")]
for lab, m in [("非週一", base.dow != 0), ("大盤跳空 ≥ -1%", base.mkt_gap >= -0.01), ("大盤跳空 ≥ 0", base.mkt_gap >= 0), ("大盤跳空 < -1%", base.mkt_gap < -0.01), ("個股跳空 ≤ 6%", base.gap <= 0.06), ("個股跳空 > 6%", base.gap > 0.06),
               ("昨日非漲停", ~base.p_uplim.astype(bool)), ("昨日漲停", base.p_uplim.astype(bool)), ("股價 ≤ 300", base.pc <= 300), ("20 日均成交額 ≥ 3 億", base.dv20 >= 3e8), ("20 日均成交額 ≥ 5 億", base.dv20 >= 5e8),
               ("突破在 09:30 前", base.t <= 30), ("突破在 09:30~10:30", (base.t > 30) & (base.t <= 90)), ("突破在 10:30 後", base.t > 90), ("突破時累積成交額 ≥ 2000 萬", base.cumdv >= 2e4), ("突破時累積成交額 500~2000 萬", (base.cumdv >= 5e3) & (base.cumdv < 2e4)), ("突破時累積成交額 < 500 萬", base.cumdv < 5e3)]:
    rows.append(summ(base[m], lab))
md += ["## D1. 濾網 (主定義 W=15)", "", table(rows, C), ""]
rows = []; series = {}
for N in [5, 10, 20, 999]:
    s, o = port(base, N, "基底 (最早突破優先)"); rows.append(o); series[f"base_N{N}"] = s
for rk, asc, rl in [("gap", False, "跳空大優先"), ("atr20", False, "ATR 大優先"), ("ret5", False, "5 日漲幅大優先"), ("cumdv", False, "突破時成交額大優先")]:
    s, o = port(base, 10, f"基底, {rl}", rk, asc); rows.append(o)
f1 = base[(base.dow != 0) & (base.mkt_gap >= -0.01)]
for N in [5, 10, 20]:
    s, o = port(f1, N, "非週一 ∩ 大盤跳空≥-1%"); rows.append(o); series[f"f1_N{N}"] = s
f2 = f1[(f1.dv20 >= 3e8) & (f1.pc <= 300)]
for N in [5, 10]:
    s, o = port(f2, N, "+ 成交額≥3億 ∩ 股價≤300 (可執行)"); rows.append(o); series[f"f2_N{N}"] = s
f3 = f2[f2.t <= 60]
s, o = port(f3, 10, "+ 只做 10:00 前突破"); rows.append(o); series["f3_N10"] = s
# 寬鬆過熱 (H10) 容量版
wide = trades(HOT["H10 (跳空≥2% ∪ 昨漲≥3% ∪ 上影≥2%) ∩ ATR≥3%"], 15, "brk"); wide = wide[(wide.dow != 0) & (wide.mkt_gap >= -0.01) & (wide.dv20 >= 3e8) & (wide.pc <= 300)]
for N in [10, 20]:
    s, o = port(wide, N, "寬鬆過熱 H10 + 可執行濾網"); rows.append(o); series[f"wide_N{N}"] = s
# 停損出場版本
sh = trades(hm, 15, "brk", "stop_hi"); sh = sh[(sh.dow != 0) & (sh.mkt_gap >= -0.01) & (sh.dv20 >= 3e8) & (sh.pc <= 300)]
for N in [10, 20]:
    s, o = port(sh, N, "主定義 + 可執行濾網 + 停損突破區間高"); rows.append(o); series[f"stophi_N{N}"] = s
sv = trades(hm, 15, "brk", "stop_vwap"); sv = sv[(sv.dow != 0) & (sv.mkt_gap >= -0.01) & (sv.dv20 >= 3e8) & (sv.pc <= 300)]
s, o = port(sv, 10, "主定義 + 可執行濾網 + 停損站回 VWAP"); rows.append(o)
for hk in ["H5 昨日上影≥2%", "H7 ATR≥5%", "H11 跳空≥1% ∩ 20日漲≥15% ∩ ATR≥3%", "H9 跳空≥2% ∩ 5日漲≥5%"]:
    for ex in ["close", "stop_hi"]:
        tt = trades(HOT[hk], 15, "brk", ex); tt = tt[(tt.dow != 0) & (tt.mkt_gap >= -0.01) & (tt.dv20 >= 3e8) & (tt.pc <= 300)]
        s, o = port(tt, 10, f"{hk} + 可執行濾網 + {'收盤出' if ex=='close' else '停損區間高'}"); rows.append(o); series[f"{hk[:3].strip()}_{ex}_N10"] = s
# 對照: 只有 ATR≥3% 不加過熱 (看 D1 過熱到底加了多少) ; W=5 版本
for lab, m in [("對照: 只 ATR≥3% (無過熱) ", df.atr20 >= 0.03), ("對照: 全 universe", pd.Series(True, index=df.index))]:
    for ex in ["close", "stop_hi"]:
        tt = trades(m, 15, "brk", ex); tt = tt[(tt.dow != 0) & (tt.mkt_gap >= -0.01) & (tt.dv20 >= 3e8) & (tt.pc <= 300)]
        s, o = port(tt, 10, f"{lab}+ 可執行濾網 + {'收盤出' if ex=='close' else '停損區間高'}"); rows.append(o); series[f"{lab[:6]}_{ex}"] = s
t5 = trades(hm, 5, "brk", "stop_hi"); t5 = t5[(t5.dow != 0) & (t5.mkt_gap >= -0.01) & (t5.dv20 >= 3e8) & (t5.pc <= 300)]
s, o = port(t5, 10, "主定義 W=5 + 可執行濾網 + 停損區間高"); rows.append(o); series["stophi_W5_N10"] = s
pd.DataFrame(series).to_csv(STATS / "orb_dn_daily.csv")
# 日報酬與大盤當日日內的相關
mkt_intra = df.groupby("date").apply(lambda x: (x.close / x.open - 1).mean()) * 100
cor = {k: np.corrcoef(v.reindex(mkt_intra.index).fillna(0), mkt_intra)[0, 1] for k, v in series.items() if k in ("f2_N10", "stophi_N10", "wide_N20")}
md_extra = [f"日報酬與「大盤當日平均開→收」的相關: " + ", ".join(f"{k} {v:.2f}" for k, v in cor.items()) + " (負 = 大盤日內跌時賺, 大盤日內漲時賠)。"]
md += ["## D2. 每日 N 檔等權組合 (去重: 每股票日一筆; 無交易日計 0; Sharpe 年化 245 日)", "", table(rows, CP), ""] + md_extra + ["",
       "容量: 每檔 60 萬 vs 突破時累積成交額 → 見 D1 的「突破時累積成交額」兩列；進場在突破那一分鐘, 實際滑價未計。", ""]
(STATS / "orb_dn_deepdive.md").write_text("\n".join(md), encoding="utf-8")
print("written")
