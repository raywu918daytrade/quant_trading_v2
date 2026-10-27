"""反向做多: 同一批候選 (跳空≥2% ∩ ATR≥3% ∩ 成交額≥3億 ∩ 股價≤300) 突破開盤 5 分區間高 → 現股當沖買進, 跌破區間低停損, 否則收盤賣出。
依進場當下 0050 狀態分組, 看 0050 強的時候做多是否成立; 並比較「0050 強做多 / 其他做空」的組合。 -> stats/etf0050_long.md"""
import glob, numpy as np, pandas as pd
import orb_live as L
from config import RAW, PROC, STATS, COST, EXPLORE_END
STOP_SLIP = 0.001
lp = PROC / "long_orb_trades.parquet"
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "mkt_gap", "dow"])
cand = f[f.univ & (f.date >= "2020-01-01") & (f.gap >= 0.02) & (f.atr20 >= 0.03) & (f.dv20 >= 3e8) & (f.pc <= 300)]
if not lp.exists():
    out = []
    for ym in sorted({d.strftime("%Y_%m") for d in cand.date}):
        m = L.load_m1_month(ym).merge(cand[["stock_id", "date", "pc"]], on=["stock_id", "date"])
        m = m.sort_values(["stock_id", "date", "t"])
        key = ["stock_id", "date"]
        r5 = m[m.t < 5].groupby(key).agg(rh=("high", "max"), rl=("low", "min"))
        m = m.join(r5, on=key); m["lu"] = m.pc.map(L.limit_up)
        brk = m[(m.t >= 5) & (m.t <= 180) & (m.high > m.rh)].groupby(key).head(1).set_index(key)
        tr = pd.DataFrame({"entry_t": brk.t, "entry_px": np.maximum(brk.rh, brk.open), "rh": brk.rh, "rl": brk.rl, "pc": brk.pc, "lu": brk.lu})
        tr = tr[(tr.entry_px < tr.lu * 0.999) & (tr.rh < tr.lu * 0.999)]
        y = m.join(tr[["entry_t"]], on=key, how="inner"); y = y[y.t > y.entry_t]
        st = y[y.low < y.rl].groupby(key).head(1).set_index(key)
        tr["stop_px"] = (np.minimum(st.rl, st.open) * (1 - STOP_SLIP)).reindex(tr.index); tr["stop_t"] = st.t.reindex(tr.index)
        tr["close_px"] = m.groupby(key).close.last().reindex(tr.index)
        out.append(tr.reset_index()); print(ym, len(tr), flush=True)
    LT = pd.concat(out, ignore_index=True); LT.to_parquet(lp, index=False)
LT = pd.read_parquet(lp).merge(cand[["stock_id", "date", "gap", "atr20", "mkt_gap", "dow"]], on=["stock_id", "date"])
LT["exit_px"] = LT.stop_px.where(LT.stop_px.notna(), LT.close_px); LT["stopped"] = LT.stop_px.notna()
LT["r_pct"] = (LT.exit_px / LT.entry_px - 1 - COST) * 100
LT["lots"] = np.maximum((3e5 // (LT.rl * 1000)).astype(int), 1); LT["pnl"] = LT.r_pct / 100 * LT.entry_px * LT.lots * 1000
LT["period"] = np.where(LT.date <= EXPLORE_END, "探索", "驗證")

# 0050 狀態 (與 19 相同的定義)
m = pd.read_parquet(PROC / "etf0050_m1.parquet"); g = m.groupby("date", sort=False)
m["day_open"] = g.open.transform("first"); m["vwap"] = (m.close * m.volume).groupby(m.date).cumsum() / g.volume.cumsum()
o5 = m[m.t < 5].groupby("date").agg(e5_h=("high", "max"), e5_l=("low", "min"), e5_c=("close", "last")); m = m.join(o5, on="date")
m["hi_since5"] = m.high.where(m.t >= 5).groupby(m.date).cummax()
snap = m[["date", "t", "close", "vwap", "day_open", "e5_h", "hi_since5", "e5_c"]].rename(columns={"t": "st"}).astype({"st": float}).sort_values("st")
LT["st"] = LT.entry_t.astype(float)
LT = pd.merge_asof(LT.sort_values("st"), snap, on="st", by="date", allow_exact_matches=False, direction="backward")
LT["e_ret_entry"] = LT.close / LT.day_open - 1; LT["e_vwap_entry"] = LT.close / LT.vwap - 1; LT["e_brk_high"] = (LT.hi_since5 > LT.e5_h).fillna(False)
LT["e_o5"] = LT.e5_c / LT.day_open - 1
LT = LT.sort_values(["date", "entry_t", "stock_id"]).reset_index(drop=True)

S = pd.read_parquet(PROC / "orb_trades_with_0050.parquet")          # 做空 (定案規則) 帶 0050 狀態
D = pd.read_parquet(PROC / "orb_history" / "days.parquet"); ALL = pd.DatetimeIndex(D.date)
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
def gs(x, label): return dict(label=label, n=len(x), mean=x.r_pct.mean(), win=(x.r_pct > 0).mean() * 100, stop=x.stopped.mean() * 100 if "stopped" in x else np.nan,
                              ex=x[x.period == "探索"].r_pct.mean(), va=x[x.period == "驗證"].r_pct.mean())
CG = [("條件", "label"), ("筆數", "n", 0), ("每筆淨%", "mean"), ("勝率", "win", 1), ("停損%", "stop", 1), ("探索%", "ex"), ("驗證%", "va")]
def top10(x): return x.sort_values(["date", "entry_t", "stock_id"]).groupby("date").head(10)
def cap(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(),
                ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
CC = [("組合", "label"), ("筆數", "n", 0), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]

md = ["# 0050 強的時候改做多?", "",
      "做多 = 同一批候選 (跳空≥2% ∩ ATR≥3% ∩ 成交額≥3億 ∩ 股價≤300) 在 09:05~12:00 第一次突破開盤 5 分區間高 → 以 max(區間高, 該分鐘開盤) 買進; 跌破區間低停損 (min(區間低, 該分鐘開盤) −0.1%); 否則收盤賣出; 成本 0.23%。",
      "0050 狀態一律用進場前一分鐘。", "", "## 1. 做多: 全部與依 0050 狀態", ""]
rows = [gs(LT, "全部突破做多")]
for lab, mk in [("進場時 0050 漲 > +0.3%", LT.e_ret_entry > 0.003), ("進場時 0050 漲 > +0.5%", LT.e_ret_entry > 0.005), ("進場時 0050 在均價線之上", LT.e_vwap_entry > 0),
                ("進場前 0050 已突破自己 5 分高", LT.e_brk_high), ("0050 前 5 分 > +0.2%", LT.e_o5 > 0.002),
                ("三者皆是: 0050 漲>0.3% ∩ 在均價線上 ∩ 已突破 5 分高", (LT.e_ret_entry > 0.003) & (LT.e_vwap_entry > 0) & LT.e_brk_high),
                ("進場時 0050 跌 < −0.3%", LT.e_ret_entry < -0.003)]:
    rows.append(gs(LT[mk], lab))
b = pd.qcut(LT.e_ret_entry, 5)
for k in b.cat.categories: rows.append(gs(LT[b == k], f"進場時 0050 相對開盤 {k.left*100:+.2f}~{k.right*100:+.2f}%"))
md += [table(rows, CG), ""]
# 同樣分組下的做空 (定案規則)
rows = []
for lab, mk in [("全部", pd.Series(True, index=S.index)), ("進場時 0050 漲 > +0.3%", S.e_ret_entry > 0.003), ("進場時 0050 漲 > +0.5%", S.e_ret_entry > 0.005),
                ("進場時 0050 在均價線之上", S.e_vwap_entry > 0), ("進場前 0050 已突破自己 5 分高", S.e_brk_high),
                ("三者皆是", (S.e_ret_entry > 0.003) & (S.e_vwap_entry > 0) & S.e_brk_high)]:
    x = S[mk]; rows.append(dict(label=lab, n=len(x), mean=x.r_pct.mean(), win=(x.r_pct > 0).mean() * 100, stop=x.reason.str.startswith("停損").mean() * 100,
                                ex=x[x.period == "探索"].r_pct.mean(), va=x[x.period == "驗證"].r_pct.mean()))
md += ["## 2. 對照: 同樣 0050 狀態下, 定案做空的表現", "", table(rows, CG), ""]
# 組合
strong_L = (LT.e_ret_entry > 0.003) & (LT.e_vwap_entry > 0) & LT.e_brk_high
strong_S = (S.e_ret_entry > 0.003) & (S.e_vwap_entry > 0) & S.e_brk_high
L_nf = LT[(LT.dow != 0) & (LT.mkt_gap >= -0.01)]
rows = [cap(S, "現行: 只做空 (每日前 10 檔)"),
        cap(top10(L_nf), "只做多 (同濾網, 每日前 10 檔)"),
        cap(top10(L_nf[(L_nf.e_ret_entry > 0.003) & (L_nf.e_vwap_entry > 0) & L_nf.e_brk_high]), "只在 0050 強時做多"),
        cap(S[~strong_S], "做空, 但 0050 強時跳過"),
        cap(pd.concat([S[~strong_S][["date", "entry_t", "stock_id", "pnl"]], L_nf[(L_nf.e_ret_entry > 0.003) & (L_nf.e_vwap_entry > 0) & L_nf.e_brk_high][["date", "entry_t", "stock_id", "pnl"]]]).pipe(top10),
            "0050 強時做多、其他時候做空 (合計每日前 10 檔)")]
md += ["## 3. 資金層級 (固定 300 萬、每檔 30 萬整數張)", "", table(rows, CC), "",
       "「0050 強」= 進場前一分鐘 0050 相對開盤 > +0.3%、在均價線之上、且已突破自己的開盤 5 分高; 門檻事先訂。"]
(STATS / "etf0050_long.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
