"""「先上再下」的做空會不會更好?
A. 開盤 N 分 (N=5/10/15, 即 09:05/09:10/09:15 之前) 區間內最高點比最低點先出現 = 先上再下; 跌破 N 分區間低放空、N 分區間高停損。
B. (N=5) 09:05 之後先突破區間高、再跌破區間低才放空 (假突破後再空), 依突破時間 ≤09:10 / ≤09:15 / ≤09:20。
候選與濾網同現行 (跳空≥2%、ATR≥3%、成交額≥3億、股價≤300、非週一、大盤跳空≥−1%、開盤 5 分量≥日均量 5%), 每日最多 10 檔 (先跌破先做, 名額遞補)。
-> stats/up_then_down.md"""
import glob, numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"])
f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_v"] + [f"{p}{N}_{c}" for N in [5, 10, 15] for p, cs in [("o", ["h", "l"]), ("dn", ["t", "px", "hi_after", "stop_hi_px", "stop_hi_open"])] for c in cs]
cols = [c.replace("o5_v", "o5_v") for c in cols]
orb = pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms], ignore_index=True)
s = f.merge(orb, on=["stock_id", "date"]); s = s[s.auc_c.notna()]
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
s["lu"] = s.pc * 1.1; s["fpx"] = s.pc * 1.08
# 開盤 N 分內 最高 / 最低 出現的分鐘 (取第一次)
sp = PROC / "open_shape.parquet"
if not sp.exists():
    out = []
    for ym, grp in s.groupby(s.date.dt.strftime("%Y_%m")):
        m = L.load_m1_month(ym).merge(grp[["stock_id", "date"]], on=["stock_id", "date"]); m = m[m.t < 15].sort_values(["stock_id", "date", "t"])
        rows = {}
        for N in [5, 10, 15]:
            x = m[m.t < N]; g = x.groupby(["stock_id", "date"])
            hi_t = x.loc[g.high.idxmax(), ["stock_id", "date", "t"]].set_index(["stock_id", "date"]).t
            lo_t = x.loc[g.low.idxmin(), ["stock_id", "date", "t"]].set_index(["stock_id", "date"]).t
            rows[f"hi_t{N}"] = hi_t; rows[f"lo_t{N}"] = lo_t
        out.append(pd.DataFrame(rows).reset_index())
    pd.concat(out).to_parquet(sp, index=False)
shp = pd.read_parquet(sp); s = s.merge(shp, on=["stock_id", "date"], how="left")
for N in [5, 10, 15]:
    s[f"shape{N}"] = np.select([s[f"hi_t{N}"] < s[f"lo_t{N}"], s[f"hi_t{N}"] > s[f"lo_t{N}"]], ["先上再下", "先下再上"], "同一分鐘")
# B: 09:05 後第一次突破區間高的時間 (m1_daily 的 orb5_up_t)
up = pd.concat([pd.read_parquet(p, columns=["stock_id", "date", "orb5_up_t"]) for p in sorted(glob.glob(str(PROC / "m1_daily" / "*.parquet")))])
s = s.merge(up, on=["stock_id", "date"], how="left")

def trades(x, N):
    p = f"dn{N}_"
    x = x[x[p + "t"].notna() & (x[p + "t"] <= 180) & (x[f"o{N}_h"] < x.lu * 0.999) & (x[f"o{N}_h"] < x.fpx) & (x[p + "px"] < x.fpx)].copy()
    st = np.maximum(x[p + "stop_hi_px"], x[p + "stop_hi_open"]) * (1 + P["STOP_SLIP"]); ex = st.where(st.notna(), x.auc_c)
    hi = x[p + "hi_after"].where(st.isna(), np.minimum(x[p + "hi_after"], st)); forced = hi >= x.fpx
    exf = np.where(forced, x.fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= x.lu * 0.999)
    x["r_pct"] = (-(exf / x[p + "px"] - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
    x["lots"] = np.maximum((3e5 // (x[f"o{N}_l"] * 1000)).astype(int), 1); x["pnl"] = x.r_pct / 100 * x[p + "px"] * x.lots * 1000
    x["stopped"] = st.notna().values; x["bt"] = x[p + "t"]
    return x
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
def pick(x): return x.sort_values(["date", "bt", "stock_id"]).groupby("date").head(10)
def ts(x, label): return dict(label=label, n=len(x), per=x.r_pct.mean(), win=(x.r_pct > 0).mean() * 100, stop=x.stopped.mean() * 100, ex=x[x.date <= EXPLORE_END].r_pct.mean(), va=x[x.date > EXPLORE_END].r_pct.mean())
def cap(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), per=x.r_pct.mean(), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(),
                worst=pn.min(), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
CT = [("條件", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("勝率", "win", 1), ("停損率%", "stop", 0), ("探索期每筆", "ex"), ("驗證期每筆", "va")]
CC = [("規則", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("累計(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]
md = ["# 「先上再下」的做空會不會更好", "", "候選與濾網同現行, 固定 300 萬、每檔 30 萬整數張, 每日最多 10 檔 (先跌破先做, 名額遞補)。", "",
      "## A. 開盤區間內「先上再下」 (最高點比最低點先出現)", ""]
rows_t, rows_c = [], []
for N, lab in [(5, "09:05"), (10, "09:10"), (15, "09:15")]:
    x = pick(trades(s, N))
    rows_t.append(ts(x, f"區間到 {lab}: 全部 (每日前 10)"))
    for sh in ["先上再下", "先下再上", "同一分鐘"]:
        rows_t.append(ts(x[x[f"shape{N}"] == sh], f"區間到 {lab}: 其中 {sh}"))
    rows_c.append(cap(x, f"區間到 {lab}, 全部做" + (" (現行)" if N == 5 else "")))
    rows_c.append(cap(pick(trades(s[s[f"shape{N}"] == "先上再下"], N)), f"區間到 {lab}, 只做先上再下"))
    rows_c.append(cap(pick(trades(s[s[f"shape{N}"] != "先上再下"], N)), f"區間到 {lab}, 只做非先上再下"))
md += ["### 每筆 (同一批每日前 10 檔內, 依區間形狀分)", "", table(rows_t, CT), "", "### 資金層級 (只做某一種形狀時, 名額由同形狀的其他候選遞補)", "", table(rows_c, CC), ""]
# B
x5 = trades(s, 5)
x5["up_first"] = x5.orb5_up_t.notna() & (x5.orb5_up_t < x5.dn5_t)
rows_t = [ts(pick(x5), "現行全部 (每日前 10)")]
sel = pick(x5)
rows_t += [ts(sel[sel.up_first], "其中: 跌破前曾先突破區間高 (假突破後再空)"), ts(sel[~sel.up_first], "其中: 跌破前沒有突破區間高")]
rows_c = [cap(pick(x5), "現行")]
for lim, lab in [(10, "09:10"), (15, "09:15"), (20, "09:20")]:
    mk = x5.up_first & (x5.orb5_up_t <= lim) & (x5.dn5_t <= lim)
    rows_t.append(ts(x5[mk], f"先突破區間高、再跌破區間低, 兩者都在 {lab} 前"))
    rows_c.append(cap(pick(x5[mk]), f"只做「{lab} 前先上再下」"))
    rows_c.append(cap(pick(x5[~mk]), f"排除「{lab} 前先上再下」"))
md += ["## B. 09:05 之後先突破區間高、再跌破區間低", "", table(rows_t, CT), "", "### 資金層級", "", table(rows_c, CC)]
(STATS / "up_then_down.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
