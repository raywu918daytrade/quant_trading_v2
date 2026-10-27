"""進場時 VWAP 距離: 是否獨立於開盤區間寬度? 以及 09:05 後跌破時才判斷、名額遞補的濾網效果。 -> 追加到 stats/vwap.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
P = L.P
T = pd.read_parquet(PROC / "orb_vwap_sim.parquet")
T["width"] = (T.rh / T.rl - 1) * 100
md = ["", "## 3. VWAP 距離是不是開盤區間寬度的另一種說法?", "", f"相關係數 (VWAP 距離 vs 區間寬): {T[['vw_gap', 'width']].corr().iloc[0, 1]:.2f}", "",
      "同一個區間寬度五分位內, 再依 VWAP 距離分高低兩半的每筆報酬 (%):", "", "| 區間寬 | VWAP 距離低半 | VWAP 距離高半 | 差 |", "|---|---|---|---|"]
wb = pd.qcut(T.width, 5)
for k in wb.cat.categories:
    x = T[wb == k]; med = x.vw_gap.median(); lo_, hi_ = x[x.vw_gap <= med].r_base.mean(), x[x.vw_gap > med].r_base.mean()
    md.append(f"| {k.left:.2f}~{k.right:.2f}% | {lo_:.2f} | {hi_:.2f} | {hi_ - lo_:+.2f} |")
# 遞補濾網 (向量化, 與 orb_live.expected_trades 同一套)
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"]); f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "dn5_t", "dn5_px", "dn5_vwap_at", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
s = f.merge(pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms]), on=["stock_id", "date"]); s = s[s.auc_c.notna()]
lu = s.pc * 1.1; fpx = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < lu * 0.999) & (s.o5_h < fpx) & (s.dn5_px < fpx) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
lu = s.pc * 1.1; fpx = s.pc * 1.08
st = np.maximum(s.dn5_stop_hi_px, s.dn5_stop_hi_open) * (1 + P["STOP_SLIP"]); ex = st.where(st.notna(), s.auc_c)
hi = s.dn5_hi_after.where(st.isna(), np.minimum(s.dn5_hi_after, st)); forced = hi >= fpx
exf = np.where(forced, fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= lu * 0.999)
s["r_pct"] = (-(exf / s.dn5_px - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
s["lots"] = np.maximum((3e5 // (s.o5_l * 1000)).astype(int), 1); s["pnl"] = s.r_pct / 100 * s.dn5_px * s.lots * 1000
s["vw_gap"] = (s.dn5_vwap_at / s.dn5_px - 1) * 100
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
def pick(x): return x.sort_values(["date", "dn5_t", "stock_id"]).groupby("date").head(10)
def cap(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), per=x.r_pct.mean(), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(),
                worst=pn.min(), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245)
rows = [cap(pick(s), "現行 (含開盤量 ≥5%)")]
for th in [0.3, 0.6, 0.9]:
    rows.append(cap(pick(s[s.vw_gap >= th]), f"跌破時價格須低於 VWAP ≥{th}% (不足的跳過, 名額遞補)"))
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
cc = [("規則", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("累計(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]
md += ["", "## 4. 當進場濾網 (名額由其他候選遞補; 門檻 0.6% 是從上表五分位看來的, 有挑選偏誤)", "", "| " + " | ".join(c[0] for c in cc) + " |", "|" + "---|" * len(cc)]
for r in rows: md.append("| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cc) + " |")
with open(STATS / "vwap.md", "a", encoding="utf-8") as fh: fh.write("\n".join(md) + "\n")
print("\n".join(md))
