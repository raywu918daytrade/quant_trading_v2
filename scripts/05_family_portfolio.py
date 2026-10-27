"""把掃描存活的組合組成每日可交易的組合 (去重股票日、每日最多 N 檔等權), 算真實的日報酬序列.
A. 「D1 過熱 × M1 向下突破」家族  B. 所有兩段 t≥2 的空單組合聯集 (排除高強制回補率的開盤急拉)  C. 基準: 高 ATR 開盤空 (≈ v1 的 ATR 基準)
-> stats/family_portfolio.md, stats/family_daily.csv"""
import numpy as np, pandas as pd
from config import PROC, STATS
tr = pd.read_parquet(PROC / "scan_trades.parquet")
res = pd.read_csv(STATS / "scan_all.csv")
alldays = np.sort(pd.read_parquet(PROC / "d1_feat.parquet", columns=["date"]).date.unique()); alldays = alldays[alldays >= np.datetime64("2020-01-01")]
tr["year"] = tr.date.dt.year

def dedupe(t, rank_col, asc):
    """同一股票日多個組合觸發 -> 留最早進場 (t 最小) 那筆"""
    return t.sort_values(["date", "stock_id", "t"]).drop_duplicates(["stock_id", "date"]).sort_values(["date", rank_col], ascending=[True, asc])

def port(t, N, label, rank_col="t", asc=True, capital=3e6, per=6e5):
    t = dedupe(t, rank_col, asc)
    top = t.groupby("date").head(N)
    day = top.groupby("date").r.mean()
    s = pd.Series(0.0, index=alldays); s.loc[day.index] = day.values
    eq = (1 + s / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    cnt = top.groupby("date").size()
    mon = s.groupby(s.index.to_period("M")).sum()
    yr = s.groupby(s.index.year)
    out = dict(label=label, N=N, trade_days=int((s != 0).sum()), avg_n=cnt.mean(), per_trade=top.r.mean(), win=(top.r > 0).mean() * 100, forced=top.forced.mean() * 100,
               day_mean=s.mean(), day_std=s.std(), sharpe=s.mean() / s.std() * np.sqrt(245), mdd=mdd, worst_day=s.min(), worst_month=mon.min(), neg_months=(mon < 0).mean() * 100,
               ex_day=s[s.index <= "2023-12-31"].mean(), va_day=s[s.index > "2023-12-31"].mean(),
               **{f"y{y}": yr.mean().get(y, np.nan) for y in range(2020, 2027)})
    return s, out

def fmt(x): return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else (f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x))
def table(rows, cols):
    r = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in r.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
CP = [("組合", "label"), ("N", "N"), ("交易日", "trade_days"), ("日均檔", "avg_n"), ("每筆%", "per_trade"), ("勝率", "win"), ("強制%", "forced"), ("日淨%", "day_mean"), ("日std", "day_std"),
      ("Sharpe", "sharpe"), ("MDD%", "mdd"), ("最差日", "worst_day"), ("最差月", "worst_month"), ("負月%", "neg_months"), ("探索日淨", "ex_day"), ("驗證日淨", "va_day")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

sh = tr[tr.side == "short"]
# ---- A. 家族 ----
HOT = ["gap_up3_vol", "gap_up3", "shooting", "uplim_first", "uplim", "bigred_vol", "vol_climax", "ret20_hot", "brk20_vol", "gap_up_afterLU", "uplim_2d", "brk60_vol", "pos20_top"]
DNBRK = ["orb5_dn_early", "orb15_dn_early", "orb30_dn_early", "orb30_dn", "orb15_dn", "orb5_dn"]
famA = sh[sh.setup.isin(HOT) & sh.trigger.isin(DNBRK)]
combosA = famA.groupby(["setup", "trigger"]).size().reset_index(name="n")
rows = []; series = {}
for N in [5, 10, 20, 999]:
    s, o = port(famA, N, "A 過熱×向下突破 (最早觸發優先)"); rows.append(o); series[f"A_N{N}"] = s
s, o = port(famA, 10, "A, 依 gap 大到小", "gap", False); rows.append(o)
s, o = port(famA, 10, "A, 依 ATR 大到小", "atr20", False); rows.append(o)
s, o = port(famA[famA.trigger.str.endswith("early")], 10, "A 只用 10:00 前突破"); rows.append(o)
s, o = port(famA[famA.gap <= 0.04], 10, "A ∩ 個股跳空 ≤4%"); rows.append(o)
s, o = port(famA[famA.dow != 0], 10, "A ∩ 非週一"); rows.append(o)
s, o = port(famA[famA.mkt_gap >= -0.01], 10, "A ∩ 大盤跳空 ≥ -1%"); rows.append(o)
s, o = port(famA[(famA.dow != 0) & (famA.gap <= 0.04) & (famA.mkt_gap >= -0.01)], 10, "A ∩ 三個濾網"); rows.append(o)
# ---- B. 所有穩健空單聯集 ----
ok = res[(res.side == "short") & (res.ex_n >= 300) & (res.va_n >= 200) & (res.ex_t >= 2) & (res.va_t >= 2)]
okB = ok[~ok.trigger.isin(["o05_up1_vol", "o30_up2"])]                                   # 排除高強制回補的追空型
famB = sh.merge(okB[["setup", "trigger"]], on=["setup", "trigger"])
for N in [10, 20]:
    s, o = port(famB, N, f"B 兩段 t≥2 空單聯集 ({len(okB)} 組)"); rows.append(o); series[f"B_N{N}"] = s
famB2 = sh.merge(ok[["setup", "trigger"]], on=["setup", "trigger"])
s, o = port(famB2, 10, f"B 含追空型 ({len(ok)} 組)"); rows.append(o)
# ---- C. 基準 ----
base = sh[(sh.setup == "hi_atr") & (sh.trigger == "open")]
for N in [10, 20]:
    s, o = port(base, N, "C 基準: 高 ATR 09:00 空, 依 ATR 排", "atr20", False); rows.append(o); series[f"C_N{N}"] = s
# all-universe open short ranked by ATR (closest to v1 ATR baseline)
allopen = sh[(sh.setup == "all") & (sh.trigger == "open")] if ((sh.setup == "all") & (sh.trigger == "open")).any() else base
s, o = port(allopen, 10, "C 基準: 全 universe 09:00 空, 依 ATR 排", "atr20", False); rows.append(o)

pd.DataFrame(series).to_csv(STATS / "family_daily.csv")
# A 家族各組合貢獻
contrib = famA.groupby(["setup", "trigger"]).agg(n=("r", "size"), mean=("r", "mean"), win=("r", lambda x: (x > 0).mean() * 100), forced=("forced", "mean")).reset_index()
contrib["forced"] *= 100
# A 家族去重後每年筆數與每筆
dA = dedupe(famA, "t", True)
yrA = dA.astype({"year": str}).groupby("year").agg(n=("r", "size"), per_day=("date", lambda x: len(x) / x.nunique()), mean=("r", "mean"), win=("r", lambda x: (x > 0).mean() * 100), forced=("forced", "mean"), t_entry=("t", "mean")).reset_index()
yrA["forced"] *= 100
# 進場時間分布
dA["t_bin"] = pd.cut(dA.t, [0, 15, 30, 60, 90, 120, 180], labels=["≤09:15", "09:15-30", "09:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-12:00"])
tb = dA.groupby("t_bin", observed=True).agg(n=("r", "size"), mean=("r", "mean"), win=("r", lambda x: (x > 0).mean() * 100)).reset_index()

md = ["# 組合層級回測: 掃描存活組合 → 每日 N 檔等權", "",
      "每筆以現實結算 (+8% 強制回補、鎖死 −7%、成本 0.23%)。同一股票日被多個組合觸發只算一次 (取最早進場)。無交易日報酬 0 併入日均。Sharpe 年化 245 日。", "",
      "**警告**: 家族 A 的成員是看完探索+驗證兩段後挑的，驗證期數字對 A 不是乾淨樣本外；B 的篩選 (兩段 t≥2) 同樣用到了驗證期。真正的樣本外要靠之後的新資料。", "",
      "## 1. 組合績效", "", table(rows, CP), "",
      "## 2. 家族 A 成員 (D1 過熱 × M1 開盤區間向下突破 → 空) 的各自貢獻", "",
      table(contrib.sort_values("mean", ascending=False).to_dict("records"), [("D1", "setup"), ("M1", "trigger"), ("筆數", "n"), ("每筆%", "mean"), ("勝率", "win"), ("強制%", "forced")]), "",
      "## 3. 家族 A (去重後) 逐年", "",
      table(yrA.to_dict("records"), [("年", "year"), ("筆數", "n"), ("每日筆", "per_day"), ("每筆%", "mean"), ("勝率", "win"), ("強制%", "forced"), ("平均進場分 (09:00 起)", "t_entry")]), "",
      "## 4. 家族 A 依進場時間", "", table(tb.to_dict("records"), [("進場", "t_bin"), ("筆數", "n"), ("每筆%", "mean"), ("勝率", "win")]), ""]
(STATS / "family_portfolio.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
