"""0050 盤中爆量 → 之後會不會大漲? 空單要不要提早回補?
爆量 = 0050 某分鐘成交量 / 前 20 個交易日同一分鐘成交量中位數 ≥ K (3、5、8); 分爆量收紅 (close>open) / 收黑。只看 09:05~13:24。
A. 0050 爆量後到收盤 (與 15/30/60 分後) 的報酬與大漲機率, 對照同時段一般分鐘。
B. 策略: 當天第一次爆量時仍持有的空單, 「爆量那分鐘收盤價立即回補 (+0.1% 滑價)」vs「照規則」(停損 / 收盤)。
-> stats/etf0050_volume_spike.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
m = pd.read_parquet(PROC / "etf0050_m1.parquet").sort_values(["date", "t"]).reset_index(drop=True)
m.loc[m.date < "2025-06-18", "volume"] *= 4                       # 1 拆 4 前的張數換算成拆後
# 每分鐘基準量: 前 20 個交易日同一分鐘的中位數
piv = m.pivot_table(index="date", columns="t", values="volume")
base = piv.rolling(20, min_periods=15).median().shift(1)
m = m.merge(base.stack().rename("vbase").reset_index(), on=["date", "t"], how="left")
m["vr"] = m.volume / m.vbase
gg = m.groupby("date", sort=False)
m["v5"] = gg.volume.transform(lambda x: x.rolling(5).sum()); m["cumv"] = gg.volume.cumsum()
for col in ["v5", "cumv"]:
    pv = m.pivot_table(index="date", columns="t", values=col)
    m = m.merge(pv.rolling(20, min_periods=15).median().shift(1).stack().rename(col + "_base").reset_index(), on=["date", "t"], how="left")
m["vr5"] = m.v5 / m.v5_base; m["cvr"] = m.cumv / m.cumv_base
m = m.sort_values(["date", "t"]).reset_index(drop=True)
m["r5m"] = m.close / m.groupby("date").close.shift(5) - 1
m["r_open"] = m.close / m.groupby("date").open.transform("first") - 1
g = m.groupby("date", sort=False)
m["day_close"] = g.close.transform("last")
for k in [15, 30, 60]:
    m[f"c{k}"] = g.close.shift(-k)
m["r_close"] = (m.day_close / m.close - 1) * 100
for k in [15, 30, 60]: m[f"r{k}"] = (m[f"c{k}"] / m.close - 1) * 100
m["up_bar"] = m.close > m.open; m["dn_bar"] = m.close < m.open
m["bar_ret"] = (m.close / m.open - 1) * 100
W = m[(m.t >= 5) & (m.t <= 264) & m.vr.notna()].copy()
W["hb"] = pd.cut(W.t, [4, 30, 60, 120, 180, 264], labels=["09:05~09:30", "09:30~10:00", "10:00~11:00", "11:00~12:00", "12:00~13:24"])

def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
def ev(x, label):
    return dict(label=label, n=len(x), days=x.date.nunique(), r15=x.r15.mean(), r30=x.r30.mean(), r60=x.r60.mean(), rc=x.r_close.mean(), med=x.r_close.median(),
                p_up05=(x.r_close >= 0.5).mean() * 100, p_up1=(x.r_close >= 1).mean() * 100, p_dn05=(x.r_close <= -0.5).mean() * 100, p_dn1=(x.r_close <= -1).mean() * 100)
CE = [("條件", "label"), ("次數", "n", 0), ("天數", "days", 0), ("15分後%", "r15", 3), ("30分後%", "r30", 3), ("60分後%", "r60", 3), ("到收盤%", "rc", 3), ("到收盤中位", "med", 3),
      ("收盤前漲≥0.5% 機率", "p_up05", 1), ("漲≥1%", "p_up1", 1), ("跌≥0.5%", "p_dn05", 1), ("跌≥1%", "p_dn1", 1)]

md = ["# 0050 盤中爆量後會不會大漲? 空單要不要提早回補?", "",
      "注意: 單一分鐘 ≥3 倍幾乎每天都有 (平均一天幾十次, 多為造市/套利大單), 不是交易員說的爆量; 主要結論看「嚴格的爆量定義」與 B 節。", "",
      "爆量 = 0050 該分鐘成交量 / 前 20 個交易日同一分鐘量的中位數 (開盤、收盤本來量就大, 要跟同時段比)。拆股前的張數已 ×4。只看 09:05~13:24。報酬自爆量那分鐘收盤價起算。", "",
      "## A. 0050 本身", ""]
rows = [ev(W, "對照: 所有分鐘")]
first = {}
for K in [3, 5, 8]:
    s = W[W.vr >= K]
    rows += [ev(s, f"爆量 ≥{K} 倍 (全部)"), ev(s[s.up_bar], f"爆量 ≥{K} 倍 收紅"), ev(s[s.dn_bar], f"爆量 ≥{K} 倍 收黑")]
    f1 = s.groupby("date").head(1); first[K] = f1
    rows += [ev(f1, f"爆量 ≥{K} 倍, 當天第一次"), ev(f1[f1.up_bar], f"爆量 ≥{K} 倍, 當天第一次且收紅")]
md += [table(rows, CE), ""]
s5 = W[(W.vr >= 5) & W.up_bar]
rows = []
for hb in W.hb.cat.categories:
    a = W[W.hb == hb]; b = s5[s5.hb == hb]
    rows += [ev(a, f"{hb} 所有分鐘"), ev(b, f"{hb} 爆量≥5倍收紅")]
md += ["### 依時段 (爆量 ≥5 倍收紅 vs 同時段所有分鐘)", "", table(rows, CE), ""]
base_t = W.groupby("t").agg(b_rc=("r_close", "mean"), b_up05=("r_close", lambda x: (x >= 0.5).mean() * 100), b_dn05=("r_close", lambda x: (x <= -0.5).mean() * 100))
W10_ = W[W.t >= 10]
EV_A = {"5分鐘量 ≥3 倍且 5 分鐘漲 ≥0.3%": (W10_.vr5 >= 3) & (W10_.r5m >= 0.003), "5分鐘量 ≥5 倍且 5 分鐘漲 ≥0.5%": (W10_.vr5 >= 5) & (W10_.r5m >= 0.005),
        "累計量 ≥2 倍且相對開盤漲 ≥0.5%": (W10_.cvr >= 2) & (W10_.r_open >= 0.005), "累計量 ≥2 倍且相對開盤漲 ≥1%": (W10_.cvr >= 2) & (W10_.r_open >= 0.01),
        "(對照) 5分鐘量 ≥3 倍且 5 分鐘跌 ≥0.3%": (W10_.vr5 >= 3) & (W10_.r5m <= -0.003)}
rows = []
for name, mk in EV_A.items():
    e = W10_[mk].groupby("date").head(1).join(base_t, on="t")
    r = ev(e, name + " (當天第一次)"); r["ex_rc"] = (e.r_close - e.b_rc).mean(); r["ex_up"] = r["p_up05"] - e.b_up05.mean(); r["ex_dn"] = r["p_dn05"] - e.b_dn05.mean(); r["tmed"] = e.t.median()
    rows.append(r)
md += ["### 嚴格的爆量定義 (每年幾十天, 當天第一次; 「超額」= 減去同一分鐘所有日子的平均)", "",
       table(rows, CE[:3] + [("中位時間(分)", "tmed", 0)] + CE[3:8] + [("到收盤超額%", "ex_rc", 3), ("漲≥0.5% 機率", "p_up05", 1), ("超額(百分點)", "ex_up", 1), ("跌≥0.5% 機率", "p_dn05", 1), ("超額(百分點)", "ex_dn", 1)]), ""]
big = W[(W.vr >= 5) & W.up_bar & (W.bar_ret >= 0.2)]
md += [f"爆量 ≥5 倍且那一分鐘漲 ≥0.2% 的強攻: {len(big)} 次 / {big.date.nunique()} 天, 到收盤平均 {big.r_close.mean():+.3f}%, 漲≥0.5% 機率 {(big.r_close>=0.5).mean()*100:.1f}%, 跌≥0.5% 機率 {(big.r_close<=-0.5).mean()*100:.1f}%。", ""]
pd.DataFrame({"freq_days_per_year_K3": [first[3].date.nunique() / 6.8], "K5": [first[5].date.nunique() / 6.8], "K8": [first[8].date.nunique() / 6.8]})

# ---------------- B. 策略: 爆量時立即回補 vs 照規則 ----------------
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
T["period"] = np.where(T.date <= EXPLORE_END, "探索", "驗證")
D = pd.read_parquet(PROC / "orb_history" / "days.parquet"); ALL = pd.DatetimeIndex(D.date)
# 每筆交易需要「爆量那分鐘」的股價: 從 M1 讀
W10 = W[W.t >= 10]
EVENTS = {"5分鐘量 ≥3 倍且 5 分鐘漲 ≥0.3%": (W10.vr5 >= 3) & (W10.r5m >= 0.003),
          "5分鐘量 ≥5 倍且 5 分鐘漲 ≥0.5%": (W10.vr5 >= 5) & (W10.r5m >= 0.005),
          "累計量 ≥2 倍且相對開盤漲 ≥0.5%": (W10.cvr >= 2) & (W10.r_open >= 0.005),
          "累計量 ≥2 倍且相對開盤漲 ≥1%": (W10.cvr >= 2) & (W10.r_open >= 0.01),
          "(對照) 5分鐘量 ≥3 倍且 5 分鐘跌 ≥0.3%": (W10.vr5 >= 3) & (W10.r5m <= -0.003)}
need = []
for name, mk in EVENTS.items():
    e = W10[mk].groupby("date").head(1)[["date", "t"]].rename(columns={"t": "ts"}); e["rule"] = name; need.append(e)
need = pd.concat(need)
cand = T.merge(need, on="date")
cand = cand[(cand.entry_t < cand.ts) & (cand.exit_t > cand.ts)]            # 爆量時仍持有
px = []
for ym, grp in cand.groupby(cand.date.dt.strftime("%Y_%m")):
    mm = L.load_m1_month(ym).merge(grp[["stock_id", "date"]].drop_duplicates(), on=["stock_id", "date"])[["stock_id", "date", "t", "close"]]
    x = grp[["stock_id", "date", "ts", "rule"]].rename(columns={"ts": "t"}).astype({"t": float}).sort_values("t")
    px.append(pd.merge_asof(x, mm.astype({"t": float}).sort_values("t"), on="t", by=["stock_id", "date"], direction="backward"))
px = pd.concat(px).rename(columns={"t": "ts", "close": "px_ts"})
cand = cand.astype({"ts": float}).merge(px, on=["stock_id", "date", "ts", "rule"])
cand["r_cover"] = (-(cand.px_ts * 1.001 / cand.entry_px - 1) - COST) * 100
cand["pnl_cover"] = cand.r_cover / 100 * cand.entry_px * cand.lots * 1000
rows, caps = [], []
def capstats(pnl_by_trade, label, n_aff):
    pn = pnl_by_trade.groupby(T.date).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, aff=n_aff, total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(), worst=pn.min(),
                ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
caps.append(capstats(T.pnl, "現行: 不管 0050, 照規則", 0))
for rule, c in cand.groupby("rule", sort=False):
    rows.append(dict(label=rule, n=len(c), days=c.date.nunique(), keep=c.r_pct.mean(), cover=c.r_cover.mean(), diff=(c.r_cover - c.r_pct).mean(),
                     better=(c.r_cover > c.r_pct).mean() * 100, ex=(c[c.period == "探索"].r_cover - c[c.period == "探索"].r_pct).mean(), va=(c[c.period == "驗證"].r_cover - c[c.period == "驗證"].r_pct).mean()))
    alt = T.pnl.copy(); idx = T.reset_index().merge(c[["stock_id", "date", "pnl_cover"]], on=["stock_id", "date"]).set_index("index")
    alt.loc[idx.index] = idx.pnl_cover.values
    caps.append(capstats(alt, f"{rule} 就把仍持有的空單全部回補", len(c)))
md += ["## B. 策略: 0050 當天第一次爆量時, 仍持有的空單立即回補 vs 照規則", "",
       table(rows, [("觸發", "label"), ("受影響筆數", "n", 0), ("天數", "days", 0), ("照規則 每筆%", "keep"), ("爆量時回補 每筆%", "cover"), ("差 (回補−照規則)", "diff"), ("回補較好的比例%", "better", 1), ("探索期差", "ex"), ("驗證期差", "va")]), "",
       "### 資金層級 (固定 300 萬整數張)", "",
       table(caps, [("規則", "label"), ("受影響筆數", "aff", 0), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]), ""]
# 頻率
md += [f"頻率 (2020-01 ~ 2026-10, 約 6.8 年): 當天出現 ≥3 倍爆量 {first[3].date.nunique()} 天、≥5 倍 {first[5].date.nunique()} 天、≥8 倍 {first[8].date.nunique()} 天 (總交易日 {W.date.nunique()})。"]
(STATS / "etf0050_volume_spike.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
