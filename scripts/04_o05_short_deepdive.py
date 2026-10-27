"""掃描冠軍「開盤 5 分急拉放量 → 放空」深入檢驗: 門檻格點、現實出場 (+8% 強制回補、鎖漲停)、出場時間、D1 型態加成、每日 N 檔組合、濾網.
-> stats/o05_short_deepdive.md"""
import glob, numpy as np, pandas as pd
from config import PROC, STATS, COST, EXPLORE_END

d1 = pd.read_parquet(PROC / "d1_feat.parquet"); d1 = d1[d1.univ & (d1.date >= "2020-01-01")]
m1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "m1_daily" / "*.parquet")))], ignore_index=True)
df = d1.merge(m1, on=["stock_id", "date"], how="inner")
df = df[df.auc_c.notna() & df.o05_c.notna()].reset_index(drop=True)
df["year"] = df.date.dt.year; df["period"] = np.where(df.date <= EXPLORE_END, "explore", "validate")
df["up5"] = df.o05_c / df.m1_open - 1
df["vr5"] = df.o05_v * 1000 / df.vol20                 # 開盤 5 分量 / 前 20 日均量 (M1 張 -> 股)
df["lu"] = df.pc * 1.10; df["force_px"] = df.pc * 1.08
FORCE_SLIP = 0.002; LOCK_PENALTY = 0.07               # 強制回補滑價; 收盤鎖漲停 隔日回補+標借 的額外損失 (v1 保守假設)

def trades(m, exit_col="auc_c", force=True, level=0.08, penalty=LOCK_PENALTY):
    """m: 選取 mask. 回傳每筆 (date, r%, forced, locked). 進場 09:05 價 o05_c; 開盤 5 分內已到 +level 或摸過漲停的不進.
    forced 用「09:05 之後」的最高價 (a05_h) 判斷."""
    fpx = df.pc * (1 + level)
    s = df[m & (df.o05_h < df.lu * 0.999) & (df.o05_h < fpx)].copy()
    fpx = fpx[s.index]
    ent = s.o05_c
    ex = s[exit_col].copy()
    forced = pd.Series(False, index=s.index)
    if force:
        forced = s.a05_h >= fpx
        ex = ex.where(~forced, fpx * (1 + FORCE_SLIP))
    locked = (~forced) & (ex >= s.lu * 0.999)
    r = -(ex / ent - 1) - COST
    r = r.where(~locked, r - penalty)
    s["r"] = r * 100; s["forced"] = forced; s["locked"] = locked
    return s

def summ(s, label=""):
    if len(s) == 0: return {}
    day = s.groupby("date").r.mean()
    out = dict(label=label, n=len(s), per_day=len(s) / s.date.nunique(), mean=s.r.mean(), med=s.r.median(), win=(s.r > 0).mean() * 100,
               forced=s.forced.mean() * 100, locked=s.locked.mean() * 100)
    for lab, p in [("ex", "explore"), ("va", "validate")]:
        q = s[s.period == p]; dq = q.groupby("date").r.mean()
        out[f"{lab}_mean"] = q.r.mean(); out[f"{lab}_t"] = dq.mean() / dq.std() * np.sqrt(len(dq)) if len(dq) > 2 else np.nan
    for y in range(2020, 2027): out[f"y{y}"] = s[s.year == y].r.mean()
    return out

def portfolio(s, N=10, rank_col="up5", cap=3_000_000, per=600_000):
    """每日依 rank_col 由大到小取前 N 檔等權 (不足 N 就只做有的), 沒交易日報酬 0. 回傳每日報酬 % 序列與統計."""
    s = s.sort_values(["date", rank_col], ascending=[True, False])
    top = s.groupby("date").head(N)
    day = top.groupby("date").r.mean()
    alld = pd.Series(0.0, index=np.sort(df.date.unique()))
    alld.loc[day.index] = day.values
    eq = (1 + alld / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    yr = alld.groupby(alld.index.year).agg(["mean", lambda x: x.mean() / x.std() * np.sqrt(245)])
    return alld, dict(N=N, rank=rank_col, days=len(alld), trade_days=(alld != 0).sum(), per_trade=top.r.mean(), win=(top.r > 0).mean() * 100,
                      day_mean=alld.mean(), day_std=alld.std(), sharpe=alld.mean() / alld.std() * np.sqrt(245), mdd=mdd, worst=alld.min(),
                      **{f"y{y}": yr.loc[y, "mean"] if y in yr.index else np.nan for y in range(2020, 2027)})

def fmt(x): return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else (f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x))
def table(rows, cols):
    r = pd.DataFrame(rows)
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in r.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
C1 = [("條件", "label"), ("筆數", "n"), ("每日筆", "per_day"), ("每筆淨%", "mean"), ("中位%", "med"), ("勝率", "win"), ("強制回補%", "forced"), ("鎖漲停%", "locked"),
      ("探索%", "ex_mean"), ("探索t", "ex_t"), ("驗證%", "va_mean"), ("驗證t", "va_t")] + [(str(y), f"y{y}") for y in range(2020, 2027)]
CP = [("條件", "label"), ("N", "N"), ("排序", "rank"), ("交易日", "trade_days"), ("每筆%", "per_trade"), ("勝率", "win"), ("日淨%", "day_mean"), ("日std%", "day_std"),
      ("Sharpe", "sharpe"), ("MDD%", "mdd"), ("最差日%", "worst")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

md = ["# 開盤 5 分急拉放量 → 放空 (o05_up1_vol short) 深入檢驗", "",
      f"進場: 09:05 以第 5 分鐘收盤價現沖券賣 (需 09:00~09:04 未摸漲停、未到 +8%)。出場: 13:30 收盤競價；**盤中到昨收 +8% 立即回補 (+{FORCE_SLIP*100:.1f}% 滑價)**；收盤仍鎖漲停 額外 −{LOCK_PENALTY*100:.0f}%。成本 {COST*100:.2f}%。",
      "開盤 5 分量以 M1 (張×1000) 對前 20 日 D1 均量 (股)。", ""]

# ---- 1. 門檻格點 (含強制回補) ----
rows = []
for up in [0.005, 0.01, 0.015, 0.02, 0.03]:
    for vr in [0.05, 0.10, 0.15, 0.25, 0.40]:
        s = trades((df.up5 >= up) & (df.vr5 >= vr))
        if len(s) >= 300: rows.append(summ(s, f"漲≥{up*100:.1f}% 量≥{vr*100:.0f}%"))
md += ["## 1. 門檻格點 (全 universe, 含 +8% 強制回補)", "", table(rows, C1), ""]

base_m = (df.up5 >= 0.01) & (df.vr5 >= 0.15)
# ---- 2. 出場規則對照 ----
rows = [summ(trades(base_m, force=False), "天真: 收盤出, 不處理漲停 (掃描版)"),
        summ(trades(base_m), "+8% 強制回補 + 鎖漲停罰 7%")]
for c, lab in [("c1000", "10:00 出"), ("c1100", "11:00 出"), ("c1200", "12:00 出"), ("c1300", "13:00 出")]:
    rows.append(summ(trades(base_m, exit_col=c), lab + " (+8% 強制)"))
md += ["## 2. 出場時間 (漲≥1% 量≥15%)", "", table(rows, C1), "", "出場越晚越好 → 空單要抱到收盤, 與 v1 結論一致。", ""]
rows = []
for lv in [0.05, 0.06, 0.07, 0.08, 0.09, 0.095]:
    rows.append(summ(trades(base_m, level=lv), f"強制回補 +{lv*100:.1f}%, 鎖死罰 7%"))
for pen in [0.0, 0.02, 0.04]:
    rows.append(summ(trades(base_m, force=False, level=0.08, penalty=pen), f"不強制, 鎖死罰 {pen*100:.0f}%"))
md += ["## 2b. 強制回補門檻 / 鎖死罰金 敏感度 (漲≥1% 量≥15%)", "", table(rows, C1), "",
       "鎖死罰金 7% 是 v1 的最壞情境 (標借費率上限); 若券商能借到券、實際代價可能 1~3%。", ""]

# ---- 3. 不同 5 分型態的細分: 漲幅 × 收在高/低位 ----
pos5 = (df.o05_c - df.o05_l) / (df.o05_h - df.o05_l)
rows = []
for lab, m in [("5分收在區間頂 (pos≥0.8)", pos5 >= 0.8), ("5分收在區間中", (pos5 > 0.2) & (pos5 < 0.8)), ("5分收在區間底 (pos≤0.2, 已拉回)", pos5 <= 0.2),
               ("跳空開高 ≥2% 再拉", df.gap >= 0.02), ("平開 (|gap|<1%) 再拉", df.gap.abs() < 0.01), ("開低 (gap≤-1%) 再拉", df.gap <= -0.01),
               ("5分最高 ≥ 昨收+5%", df.o05_h / df.pc - 1 >= 0.05), ("5分最高 < 昨收+5%", df.o05_h / df.pc - 1 < 0.05)]:
    rows.append(summ(trades(base_m & m), lab))
md += ["## 3. 5 分 K 型態細分 (漲≥1% 量≥15%)", "", table(rows, C1), ""]

# ---- 4. D1 型態加成 ----
SET = [c[2:] for c in df.columns if c.startswith("S_")]
rows = []
for s_ in SET:
    t = trades(base_m & df[f"S_{s_}"])
    if len(t) >= 300: rows.append(summ(t, s_))
rows = sorted(rows, key=lambda r: -min(r["ex_t"], r["va_t"]))
md += ["## 4. 疊加 D1 型態 (依 min(探索t, 驗證t) 排序)", "", table(rows, C1), ""]

# ---- 5. 濾網 ----
rows = [summ(trades(base_m), "基底")]
for lab, m in [("非週一", df.dow != 0), ("大盤跳空 ≥ -1%", df.mkt_gap >= -0.01), ("大盤跳空 < -1%", df.mkt_gap < -0.01), ("大盤跳空 > +1%", df.mkt_gap > 0.01),
               ("個股跳空 ≤ 4%", df.gap <= 0.04), ("個股跳空 > 4%", df.gap > 0.04), ("昨日非漲停", ~df.p_uplim.astype(bool)), ("ATR20 ≥ 3%", df.atr20 >= 0.03),
               ("ATR20 < 3%", df.atr20 < 0.03), ("股價 ≤ 300", df.pc <= 300), ("成交額 20 日均 ≥ 5 億", df.dv20 >= 5e8), ("20 日漲幅 ≥ 20%", df.ret20 >= 0.2)]:
    rows.append(summ(trades(base_m & m), lab))
md += ["## 5. 濾網 (漲≥1% 量≥15%)", "", table(rows, C1), ""]

# ---- 6. 每日 N 檔組合 ----
rows = []; series = {}
s_all = trades(base_m)
for N in [5, 10, 20]:
    for rk in ["up5", "vr5", "atr20"]:
        alld, st = portfolio(s_all, N, rk); st["label"] = "基底"; rows.append(st); series[(N, rk)] = alld
s_f = trades(base_m & (df.dow != 0) & (df.gap <= 0.04))
for N in [5, 10]:
    alld, st = portfolio(s_f, N, "up5"); st["label"] = "非週一 ∩ 跳空≤4%"; rows.append(st)
s_h = trades(base_m & (df.atr20 >= 0.03))
for N in [5, 10]:
    alld, st = portfolio(s_h, N, "up5"); st["label"] = "ATR≥3%"; rows.append(st)
md += ["## 6. 每日 N 檔等權組合 (無交易日計 0; Sharpe 年化 245 日)", "", table(rows, CP), ""]
series[(10, "up5")].rename("ret_pct").to_csv(STATS / "o05_short_daily_N10.csv")

# ---- 7. 與 v1 正式策略的關係 ----
md += ["## 7. 解讀", "",
       "- 掃描裡 `o05_up1_vol short` 之所以能跨 37 種 D1 型態、7 年全部為正，是因為它抓的是同一件事：**開盤 5 分鐘的急拉 + 放量 = 當日多方力竭的訊號**，之後到收盤平均回落。",
       "- 這和 v1 的模型 B (09:01 全部進場) 是不同的時間點：v1 在 09:01 用 D1 + 跳空預測；這條規則多等 4 分鐘，用 M1 確認「有人在追價」再空，每筆淨比 v1 基準高但每天檔數較少。兩者可疊加 (v1 候選 ∩ 開盤 5 分急拉) 或當獨立訊號。",
       "- 現實風險與 v1 相同：空單在漲停板鎖死。+8% 強制回補規則把鎖死率壓到接近 0，但代價是強制回補那些交易平均大賠；上表 forced% 就是這個比例。",
       "- 尚未考慮: 現沖券賣資格 (2020~2023 無名單)、09:05 市價單滑價、處置股。"]
(STATS / "o05_short_deepdive.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
