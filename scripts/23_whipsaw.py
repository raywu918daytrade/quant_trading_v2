"""個股盤中來回洗盤的特徵.
單筆分類: 被停損且收盤又跌回進場價之下 = 「洗掉」(停損砍錯); 被停損且收盤仍在區間高之上 = 「真反彈」; 未停損 = 「抱到收盤」.
單日: 進場 ≥5 檔且停損 ≥70% = 洗盤日.
比較進場前可知的特徵, 並測事先訂好的濾網. -> stats/whipsaw.md"""
import glob, numpy as np, pandas as pd
from config import PROC, STATS, EXPLORE_END
T = pd.read_parquet(PROC / "orb_trades_with_0050.parquet")
orb = pd.concat([pd.read_parquet(f, columns=["stock_id", "date", "auc_c", "o5_v", "o5_dv", "m1_open", "dn5_lo_after"]) for f in sorted(glob.glob(str(PROC / "orb_dn" / "*.parquet")))])
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "pc", "open", "p_ret", "ret5", "ret20", "p_vr", "vol20", "atr20", "dv20", "pos20", "mkt_breadth", "mkt_gap"])
T = T.drop(columns=[c for c in ["pc", "open", "atr20", "dv20", "mkt_gap"] if c in T.columns]).merge(orb, on=["stock_id", "date"], how="left").merge(f, on=["stock_id", "date"], how="left")
T["stopped"] = T.reason.str.startswith("停損")
T["cls"] = np.select([T.stopped & (T.auc_c < T.entry_px), T.stopped & (T.auc_c >= T.rh), T.stopped], ["洗掉", "真反彈", "停損後收在區間內"], "抱到收盤")
T["width"] = (T.rh / T.rl - 1) * 100; T["stop_dist"] = (T.rh / T.entry_px - 1) * 100
T["o5_vr"] = T.o5_v * 1000 / T.vol20                         # 開盤 5 分鐘量 / 20 日均量
T["open_drop"] = (T.m1_open / T.rl - 1) * 100                 # 開盤到區間低的距離 (開高走低多少)
T["entry_vs_pc"] = (T.entry_px / T.pc - 1) * 100              # 進場價相對昨收
T["width_atr"] = T.width / (T.atr20 * 100)                    # 開盤區間寬 / ATR
T["at0905"] = T.entry_t <= 5
# ---- 日層級 ----
D = pd.read_parquet(PROC / "orb_history" / "days.parquet").set_index("date")
day = T.groupby("date").agg(n=("r_pct", "size"), stop_share=("stopped", "mean"), wash_share=("cls", lambda x: (x == "洗掉").mean()), n0905=("at0905", "sum"),
                             e_gap=("e_gap", "first"), e_o5=("e_o5", "first"), e_pret=("e_pret", "first"), e_ret5=("e_ret5", "first"), e_ret20=("e_ret20", "first"),
                             e_intra=("e_intra", "first"), breadth=("mkt_breadth", "first"), mkt_gap=("mkt_gap", "first"))
day = day.join(D[["pnl", "n_cand", "n_watch"]]); day["pnl_w"] = day.pnl / 1e4
m = pd.read_parquet(PROC / "etf0050_m1.parquet"); e = m.groupby("date").agg(o=("open", "first"), c=("close", "last"), h=("high", "max"), l=("low", "min"))
e.loc[e.index < "2025-06-18", ["o", "c", "h", "l"]] /= 4
e["rng"] = (e.h - e.l) / e.c.shift(1) * 100; e["e_rng5"] = e.rng.shift(1).rolling(5).mean(); e["e_rng20"] = e.rng.shift(1).rolling(20).mean()
day = day.join(e[["e_rng5", "e_rng20"]])
# 策略自身近期: 前一個「有交易日」的停損率與損益, 前 5 個有交易日停損率
day = day.sort_index()
day["prev_stop"] = day.stop_share.shift(1); day["prev_pnl"] = day.pnl_w.shift(1); day["prev5_stop"] = day.stop_share.shift(1).rolling(5).mean()
day["period"] = np.where(day.index <= EXPLORE_END, "探索", "驗證")
day["wash_day"] = (day.n >= 5) & (day.stop_share >= 0.7)

def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
md = ["# 個股盤中來回洗盤的特徵", "",
      f"資料: 定案規則 2020-01 ~ {T.date.max().date()}, {len(T):,} 筆、{len(day)} 個有交易日。固定 300 萬、每檔 30 萬整數張。", ""]
# ---- 1. 單筆分類 ----
c = T.groupby("cls").agg(n=("r_pct", "size"), r=("r_pct", "mean"), pnl=("pnl", "sum")).reindex(["抱到收盤", "真反彈", "停損後收在區間內", "洗掉"])
c["share"] = c.n / len(T) * 100; c["pnl"] /= 1e4
T["r_hold"] = (-(T.auc_c / T.entry_px - 1) - 0.0023) * 100
rows = [dict(label=k, n=int(r.n), share=r.share, r=r.r, hold=T[T.cls == k].r_hold.mean(), pnl=r.pnl) for k, r in c.iterrows()]
md += ["## 1. 單筆分類", "", table(rows, [("類別", "label"), ("筆數", "n", 0), ("占比%", "share", 1), ("實際每筆%", "r"), ("若抱到收盤每筆%", "hold"), ("實際損益合計(萬)", "pnl", 0)]), "",
       "「洗掉」= 被停損, 但收盤價又低於進場價 (抱著就會賺)。「真反彈」= 被停損且收盤仍高於區間高 (停損做對)。", ""]
# ---- 2. 單筆特徵比較 ----
FEAT = [("gap", "個股跳空%", 100), ("atr20", "ATR20%", 100), ("width", "開盤 5 分區間寬%", 1), ("width_atr", "區間寬 / ATR", 1), ("stop_dist", "停損距離%", 1),
        ("open_drop", "開盤到區間低%", 1), ("entry_vs_pc", "進場價相對昨收%", 1), ("o5_vr", "開盤 5 分量 / 日均量", 1), ("entry_t", "進場分鐘 (09:00 起)", 1),
        ("p_ret", "昨日漲幅%", 100), ("ret5", "近 5 日漲幅%", 100), ("ret20", "近 20 日漲幅%", 100), ("p_vr", "昨日量比", 1), ("pos20", "20 日區間位置", 1),
        ("e_o5", "0050 前 5 分%", 100), ("e_ret_entry", "進場時 0050 相對開盤%", 100)]
rows = []
for col, lab, k in FEAT:
    r = dict(label=lab)
    for cl in ["抱到收盤", "真反彈", "洗掉"]: r[cl] = T[T.cls == cl][col].median() * k
    rows.append(r)
md += ["## 2. 單筆特徵 (中位數)", "", table(rows, [("特徵", "label"), ("抱到收盤", "抱到收盤"), ("真反彈", "真反彈"), ("洗掉", "洗掉")]), ""]
# 各特徵五分位 → 停損率與洗掉率與每筆
rows = []
for col, lab, k in FEAT[:9]:
    b = pd.qcut(T[col], 5, duplicates="drop")
    for kk in b.cat.categories:
        x = T[b == kk]
        rows.append(dict(label=f"{lab} {kk.left*k:.2f}~{kk.right*k:.2f}", n=len(x), stop=x.stopped.mean() * 100, wash=(x.cls == "洗掉").mean() * 100, r=x.r_pct.mean(),
                         ex=x[x.date <= EXPLORE_END].r_pct.mean(), va=x[x.date > EXPLORE_END].r_pct.mean()))
md += ["### 個股特徵五分位", "", table(rows, [("條件", "label"), ("筆數", "n", 0), ("停損率%", "stop", 1), ("洗掉率%", "wash", 1), ("每筆%", "r"), ("探索%", "ex"), ("驗證%", "va")]), ""]
# ---- 3. 洗盤日 ----
w = day[day.wash_day]; o = day[~day.wash_day & (day.n >= 5)]
DF = [("n", "進場檔數", 1), ("n0905", "09:05 同時進場檔數", 1), ("n_cand", "跳空≥2% 候選數", 1), ("mkt_gap", "大盤平均跳空%", 100), ("breadth", "開盤上漲家數比%", 100),
      ("e_gap", "0050 跳空%", 100), ("e_o5", "0050 前 5 分%", 100), ("e_pret", "0050 昨日%", 100), ("e_ret5", "0050 近 5 日%", 100), ("e_ret20", "0050 近 20 日%", 100),
      ("e_rng5", "0050 近 5 日平均振幅%", 1), ("e_rng20", "0050 近 20 日平均振幅%", 1), ("prev_stop", "策略前一交易日停損率%", 100), ("prev5_stop", "策略前 5 交易日停損率%", 100),
      ("e_intra", "(事後) 0050 當日開→收%", 100)]
rows = [dict(label=lab, w=w[col].mean() * k, o=o[col].mean() * k, wm=w[col].median() * k, om=o[col].median() * k) for col, lab, k in DF]
md += ["## 3. 洗盤日 vs 其他日 (進場 ≥5 檔的日子)", "",
       f"洗盤日 (≥5 檔且停損 ≥70%): {len(w)} 天 / {(day.n >= 5).sum()} 天, 平均單日 {w.pnl_w.mean():.1f} 萬、合計 {w.pnl_w.sum():.0f} 萬；其他日平均 {o.pnl_w.mean():.1f} 萬。洗盤日中「洗掉」單占 {w.wash_share.mean()*100:.0f}%。", "",
       f"各年洗盤日數: " + "、".join(f"{y} {n}" for y, n in w.groupby(w.index.year).size().items()), "",
       table(rows, [("特徵", "label"), ("洗盤日 平均", "w"), ("其他日 平均", "o"), ("洗盤日 中位", "wm"), ("其他日 中位", "om")]), ""]
# 洗盤是否連續 (前一日停損率 → 今日)
rows = []
for lab, mk in [("前一交易日停損率 <30%", day.prev_stop < 0.3), ("30~50%", (day.prev_stop >= 0.3) & (day.prev_stop < 0.5)), ("50~70%", (day.prev_stop >= 0.5) & (day.prev_stop < 0.7)), ("≥70%", day.prev_stop >= 0.7)]:
    x = day[mk]; rows.append(dict(label=lab, n=len(x), stop=x.stop_share.mean() * 100, wash=x.wash_day.mean() * 100, pnl=x.pnl_w.mean(), ex=x[x.period == "探索"].pnl_w.mean(), va=x[x.period == "驗證"].pnl_w.mean()))
for lab, col in [("0050 近 5 日振幅", "e_rng5"), ("09:05 同時進場檔數", "n0905"), ("大盤上漲家數比", "breadth")]:
    b = pd.qcut(day[col], 4, duplicates="drop")
    for kk in b.cat.categories:
        x = day[b == kk]; rows.append(dict(label=f"{lab} {kk.left:.2f}~{kk.right:.2f}", n=len(x), stop=x.stop_share.mean() * 100, wash=x.wash_day.mean() * 100, pnl=x.pnl_w.mean(),
                                            ex=x[x.period == "探索"].pnl_w.mean(), va=x[x.period == "驗證"].pnl_w.mean()))
md += ["## 4. 事前可知的日層級條件 → 當日停損率與損益", "", table(rows, [("條件", "label"), ("天數", "n", 0), ("當日停損率%", "stop", 1), ("洗盤日機率%", "wash", 1), ("平均單日(萬)", "pnl", 2), ("探索期", "ex", 2), ("驗證期", "va", 2)]), ""]
# ---- 5. 濾網 (事先訂) ----
ALL = pd.DatetimeIndex(D.index)
def cap(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(), worst=pn.min(),
                ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
T = T.merge(day[["prev_stop", "prev5_stop", "e_rng5"]], left_on="date", right_index=True, how="left")
rows = [cap(T, "現行")]
for lab, mk in [("前一交易日停損率 ≥70% 就休息一天", T.prev_stop >= 0.7), ("前 5 交易日平均停損率 ≥55% 就休息", T.prev5_stop >= 0.55),
                ("停損距離 <1% 的不做 (區間太窄)", T.stop_dist < 1.0), ("區間寬 / ATR < 0.2 的不做", T.width_atr < 0.2),
                ("開盤 5 分量 < 日均量 5% 的不做", T.o5_vr < 0.05)]:
    rows.append(cap(T[~mk.fillna(False)], lab))
md += ["## 5. 濾網 (門檻事先訂, 非最佳化)", "", table(rows, [("規則", "label"), ("筆數", "n", 0), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)])]
(STATS / "whipsaw.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
