"""09:05 就能知道的洗盤特徵 (開盤 5 分區間寬 / ATR、開盤 5 分量 / 日均量) 先排除, 讓其他候選遞補每日 10 個名額, 重算資金層級。 -> 追加到 stats/whipsaw.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"])
f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
orb = pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms], ignore_index=True)
s = f.merge(orb, on=["stock_id", "date"]); s = s[s.auc_c.notna()]
lu = s.pc * 1.1; fpx = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < lu * 0.999) & (s.o5_h < fpx) & (s.dn5_px < fpx)].copy()
lu = s.pc * 1.1; fpx = s.pc * 1.08
st = np.maximum(s.dn5_stop_hi_px, s.dn5_stop_hi_open) * (1 + P["STOP_SLIP"]); ex = st.where(st.notna(), s.auc_c)
hi = s.dn5_hi_after.where(st.isna(), np.minimum(s.dn5_hi_after, st)); forced = hi >= fpx
exf = np.where(forced, fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= lu * 0.999)
s["r_pct"] = (-(exf / s.dn5_px - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
s["lots"] = np.maximum((3e5 // (s.o5_l * 1000)).astype(int), 1); s["pnl"] = s.r_pct / 100 * s.dn5_px * s.lots * 1000
s["width_atr"] = (s.o5_h / s.o5_l - 1) / s.atr20; s["o5_vr"] = s.o5_v * 1000 / s.vol20; s["stop_dist"] = s.o5_h / s.dn5_px - 1
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
def pick(x): return x.sort_values(["date", "dn5_t", "stock_id"]).groupby("date").head(10)
def cap(x, label):
    pn = x.groupby("date").pnl.sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), per=x.r_pct.mean(), total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245), mdd=(eq - eq.cummax()).min(),
                worst=pn.min(), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245,
                yrs="  ".join(f"{y}:{v:.0f}" for y, v in pn.groupby(pn.index.year).sum().items()))
rows = [cap(pick(s), "現行")]
for lab, keep in [("排除 區間寬/ATR < 0.2", s.width_atr >= 0.2), ("排除 開盤 5 分量 < 日均量 5%", s.o5_vr >= 0.05), ("排除 開盤 5 分量 < 日均量 8%", s.o5_vr >= 0.08),
                  ("排除 停損距離 < 1%", s.stop_dist >= 0.01), ("排除 區間寬/ATR<0.2 或 開盤量<5%", (s.width_atr >= 0.2) & (s.o5_vr >= 0.05))]:
    rows.append(cap(pick(s[keep]), lab + " (名額由其他候選遞補)"))
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
cols = [("規則", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1),
        ("探索期年均", "ex", 0), ("驗證期年均", "va", 0), ("各年(萬)", "yrs")]
md = ["", "## 6. 09:05 就排除, 名額由其他候選遞補 (正確的濾網算法)", "", "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
for r in rows: md.append("| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |")
print("\n".join(md))
with open(STATS / "whipsaw.md", "a", encoding="utf-8") as fh: fh.write("\n".join(md) + "\n")
