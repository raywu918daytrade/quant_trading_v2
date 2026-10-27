"""0050 能不能幫定案做空規則判斷多空?
A. 診斷: 策略日損益 vs 0050 當日開→收。
B. 濾網: 進場前就知道的 0050 狀態 (盤前 D1、09:00 跳空、09:05 前 5 分、進場那一刻) 分組 → 每筆報酬; 固定 300 萬整數張的資金層級比較。
C. 反向做多: 同一批候選 (跳空≥2% ∩ ATR≥3% ...) 突破開盤 5 分區間高 → 多, 跌破區間低停損, 否則收盤; 依 0050 狀態分組。
-> data/processed/etf0050_m1.parquet, data/processed/long_orb_trades.parquet, stats/etf0050_filter.md"""
import glob, numpy as np, pandas as pd
import orb_live as L
from config import RAW, PROC, STATS, COST, EXPLORE_END

# ---------------- 0050 D1 (還原分割) ----------------
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RAW / "d1" / "*.parquet")))]).query("stock_id=='0050'").sort_values("date").reset_index(drop=True)
d["date"] = pd.to_datetime(d.date)
for i in d.index[1:]:
    ratio = d.close[i - 1] / d.open[i]
    if ratio > 2.5:                                            # 分割 (2025-06-18 1 拆 4)
        f = round(ratio); d.loc[: i - 1, ["open", "high", "low", "close"]] /= f; d.loc[: i - 1, "volume"] *= f; print("0050 分割還原", d.date[i].date(), f)
d["ret"] = d.close / d.close.shift(1) - 1; d["intra"] = d.close / d.open - 1
E = pd.DataFrame({"date": d.date})
E["e_gap"] = d.open / d.close.shift(1) - 1
E["e_pret"] = d.ret.shift(1); E["e_pintra"] = d.intra.shift(1)
E["e_ret5"] = d.close.shift(1) / d.close.shift(6) - 1; E["e_ret20"] = d.close.shift(1) / d.close.shift(21) - 1
E["e_ma20_dev"] = d.close.shift(1) / d.close.shift(1).rolling(20).mean() - 1; E["e_ma60_dev"] = d.close.shift(1) / d.close.shift(1).rolling(60).mean() - 1
up = (d.ret > 0).astype(int); grp = (up != up.shift()).cumsum(); streak = up.groupby(grp).cumsum().where(up == 1, -(1 - up).groupby(grp).cumsum())
E["e_streak"] = streak.shift(1)
E["e_intra"] = d.intra                                         # 當日開→收 (只用於診斷, 盤中不知道)

# ---------------- 0050 M1 ----------------
mp = PROC / "etf0050_m1.parquet"
if not mp.exists():
    parts = []
    for f in sorted(glob.glob(str(RAW / "m1" / "*.parquet"))):
        m = pd.read_parquet(f, filters=[("stock_id", "=", "0050")]); parts.append(m)
    m = pd.concat(parts); m["ts"] = pd.to_datetime(m.date); m["date"] = m.ts.dt.normalize(); m["t"] = m.ts.dt.hour * 60 + m.ts.dt.minute - 540
    m = m[(m.t >= 0) & (m.t <= 270)].sort_values(["date", "t"])[["date", "t", "open", "high", "low", "close", "volume"]]
    m.to_parquet(mp, index=False)
m = pd.read_parquet(mp)
g = m.groupby("date", sort=False)
m["day_open"] = g.open.transform("first")
m["vwap"] = (m.close * m.volume).groupby(m.date).cumsum() / g.volume.cumsum()
o5 = m[m.t < 5].groupby("date").agg(e5_h=("high", "max"), e5_l=("low", "min"), e5_c=("close", "last"))
m = m.join(o5, on="date")
after5 = m.t >= 5
m["hi_since5"] = m.high.where(after5).groupby(m.date).cummax(); m["lo_since5"] = m.low.where(after5).groupby(m.date).cummin()
day = o5.join(g.open.first().rename("e_open_m1"))
day["e_o5"] = day.e5_c / day.e_open_m1 - 1; day["e_o5_pos"] = (day.e5_c - day.e5_l) / (day.e5_h - day.e5_l)
E = E.merge(day[["e_o5", "e_o5_pos"]].reset_index(), on="date", how="left")

# ---------------- 交易 + 進場當下 0050 狀態 ----------------
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet"); D = pd.read_parquet(PROC / "orb_history" / "days.parquet")
T = T.merge(E, on="date", how="left")
snap = m[["date", "t", "close", "vwap", "day_open", "e5_h", "e5_l", "hi_since5", "lo_since5"]].rename(columns={"t": "st"}).sort_values("st")
T["st"] = T.entry_t.astype(float)
T = pd.merge_asof(T.sort_values("st"), snap.astype({"st": float}), on="st", by="date", allow_exact_matches=False, direction="backward")
T["e_ret_entry"] = T.close / T.day_open - 1                                  # 進場前一分鐘 0050 相對開盤
T["e_vwap_entry"] = T.close / T.vwap - 1
T["e_brk_low"] = (T.lo_since5 < T.e5_l).fillna(False); T["e_brk_high"] = (T.hi_since5 > T.e5_h).fillna(False)
T = T.sort_values(["date", "entry_t", "stock_id"]).reset_index(drop=True)
T["period"] = np.where(T.date <= EXPLORE_END, "探索", "驗證")
ALL = pd.DatetimeIndex(D.date)

def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
def grp_stats(x, label):
    return dict(label=label, n=len(x), share=len(x) / len(T) * 100, mean=x.r_pct.mean(), win=(x.r_pct > 0).mean() * 100, stop=x.reason.str.startswith("停損").mean() * 100,
                ex=x[x.period == "探索"].r_pct.mean(), va=x[x.period == "驗證"].r_pct.mean(), pnl=x.pnl.sum() / 1e4)
CG = [("條件", "label"), ("筆數", "n", 0), ("占比%", "share", 1), ("每筆淨%", "mean"), ("勝率", "win", 1), ("停損%", "stop", 1), ("探索%", "ex"), ("驗證%", "va"), ("損益合計(萬)", "pnl", 0)]
def cap_stats(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(),
                worst_day=pn.min(), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
CC = [("規則", "label"), ("筆數", "n", 0), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst_day", 1), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]

md = ["# 0050 能不能幫做空規則判斷多空", "",
      f"交易 = 定案規則歷史 (orb_history, 2020-01 ~ {T.date.max().date()}, {len(T):,} 筆, 固定 300 萬整數張)。0050 日 K 已還原 2025-06-18 的 1 拆 4。",
      "進場當下的 0050 狀態用「進場那一分鐘之前」的最後一根 1 分 K (不偷看)。探索期 ≤2023, 驗證期 2024~。", ""]

# ---- A. 診斷 ----
dd_ = D.set_index("date").join(E.set_index("date")[["e_intra"]])
corr = dd_[dd_.n_trades > 0][["pnl", "e_intra"]].corr().iloc[0, 1]
T["q_intra"] = pd.qcut(T.e_intra, 5, labels=["最弱 20%", "弱", "中", "強", "最強 20%"])
rows = [grp_stats(T[T.q_intra == q], f"0050 當日開→收 {q} ({T[T.q_intra == q].e_intra.min()*100:+.2f}~{T[T.q_intra == q].e_intra.max()*100:+.2f}%)") for q in T.q_intra.cat.categories]
md += ["## A. 診斷: 策略有多依賴大盤日內方向 (事後資訊, 不能當訊號)", "", f"有交易的日子, 當日損益與 0050 開→收相關 {corr:.2f}。", "", table(rows, CG), ""]

# ---- B. 濾網: 分組 ----
def qbuckets(col, label, q=5, x=None):
    x = T if x is None else x
    b = pd.qcut(x[col], q, duplicates="drop")
    return [grp_stats(x[b == k], f"{label} {k.left*100:+.2f}~{k.right*100:+.2f}%") for k in b.cat.categories]
md += ["## B. 進場前可知的 0050 狀態 → 每筆報酬", ""]
for col, lab in [("e_gap", "0050 今日跳空"), ("e_o5", "0050 前 5 分鐘漲跌"), ("e_ret_entry", "進場時 0050 相對開盤"), ("e_vwap_entry", "進場時 0050 相對均價線"),
                 ("e_pret", "0050 昨日漲跌"), ("e_pintra", "0050 昨日開→收"), ("e_ret5", "0050 近 5 日"), ("e_ret20", "0050 近 20 日"), ("e_ma20_dev", "0050 昨收相對 MA20"), ("e_ma60_dev", "0050 昨收相對 MA60")]:
    md += [f"### {lab}", "", table(qbuckets(col, lab), CG), ""]
rows = [grp_stats(T[T.e_brk_low], "進場前 0050 已跌破自己的開盤 5 分低"), grp_stats(T[~T.e_brk_low], "未跌破"),
        grp_stats(T[T.e_brk_high], "進場前 0050 已突破自己的開盤 5 分高"), grp_stats(T[~T.e_brk_high], "未突破")]
for k in [-3, -2, -1, 1, 2, 3]:
    s = T[(T.e_streak <= k) if k < 0 else (T.e_streak >= k)]; rows.append(grp_stats(s, f"0050 連{'跌' if k < 0 else '漲'} ≥{abs(k)} 日"))
md += ["### 結構狀態", "", table(rows, CG), ""]

# ---- B2. 濾網規則 (事前定好門檻, 不從表裡挑最佳) ----
RULES = [("基底 (現行規則)", pd.Series(True, index=T.index)),
         ("進場時 0050 在均價線之下才做", T.e_vwap_entry < 0),
         ("進場時 0050 未漲超過 +0.3% 才做", T.e_ret_entry <= 0.003),
         ("0050 前 5 分鐘 ≤ 0 才做", T.e_o5 <= 0),
         ("0050 跳空 ≤ +1% 才做", T.e_gap <= 0.01),
         ("0050 未連漲 3 日才做", T.e_streak < 3),
         ("0050 昨收在 MA20 之下才做", T.e_ma20_dev < 0),
         ("進場前 0050 未突破自己 5 分高才做", ~T.e_brk_high)]
rows = [cap_stats(T[m_], lab) for lab, m_ in RULES]
md += ["## B2. 濾網對資金的影響 (固定 300 萬整數張, 門檻事先訂, 非最佳化)", "", table(rows, CC), "",
       "濾網只是不做某些交易, 不會讓其他交易多做。年均以全部交易日 ×245 計。", ""]
T.to_parquet(PROC / "orb_trades_with_0050.parquet", index=False)
(STATS / "etf0050_filter.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
