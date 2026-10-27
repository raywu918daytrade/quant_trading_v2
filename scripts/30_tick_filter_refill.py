"""一檔跳動單位占股價 (以區間低計, 09:05 已知) ≤ 門檻才監看, 名額由其他候選遞補; 在滑價 0/1/2 檔下比較。
滑 k 檔: 進場 = min(區間低, 跌破那分鐘開盤) − k 檔; 停損 = max(區間高, 觸發那分鐘開盤) + k 檔 (k=0 時沿用 +0.1%)。 -> stats/tick_filter.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"]); f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
s = f.merge(pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms]), on=["stock_id", "date"]); s = s[s.auc_c.notna()]
s["lu"] = s.pc * 1.1; s["fpx"] = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < s.lu * 0.999) & (s.o5_h < s.fpx) & (s.dn5_px < s.fpx) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
s["tick_l"] = s.o5_l.map(L.tick_size); s["tick_h"] = s.o5_h.map(L.tick_size); s["tick_pct"] = s.tick_l / s.o5_l * 100
s["lots"] = np.maximum((3e5 // (s.o5_l * 1000)).astype(int), 1)
def settle(x, k):
    ent = x.dn5_px - k * x.tick_l
    raw = np.maximum(x.dn5_stop_hi_px, x.dn5_stop_hi_open)
    st = raw * (1 + P["STOP_SLIP"]) if k == 0 else raw + k * x.tick_h
    ex = st.where(x.dn5_stop_hi_px.notna(), x.auc_c)
    hi = x.dn5_hi_after.where(x.dn5_stop_hi_px.isna(), np.minimum(x.dn5_hi_after, st)); forced = hi >= x.fpx
    exf = np.where(forced, x.fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= x.lu * 0.999)
    r = (-(exf / ent - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
    return r, r / 100 * ent * x.lots * 1000
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
def pick(x): return x.sort_values(["date", "dn5_t", "stock_id"]).groupby("date").head(10)
rows = []
for th, lab in [(99, "現行 (不限)"), (0.2, "一檔 ≤0.2%"), (0.15, "一檔 ≤0.15%")]:
    x = pick(s[s.tick_pct <= th])
    for k in [0, 1, 2]:
        r, pnl = settle(x, k)
        pn = pnl.groupby(x.date).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); dd = eq - eq.cummax()
        uw = (dd < 0).astype(int); mon = pn.groupby(pn.index.to_period("M")).sum(); yr = pn.groupby(pn.index.year).sum()
        rows.append(dict(label=lab, k=k, n=len(x), per_day=len(x) / (pn != 0).sum(), per=r.mean(), win=(r > 0).mean() * 100, py=pn.sum() / len(pn) * 245, sh=pn.mean() / pn.std() * np.sqrt(245),
                         mdd=dd.min(), ratio=(-dd.min() / (pn.sum() / len(pn) * 245)) if pn.sum() > 0 else np.nan, worst=pn.min(), wm=mon.min(), negm=(mon < 0).mean() * 100,
                         longest=int(uw.groupby((uw == 0).cumsum()).sum().max()), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245,
                         va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245, yrs="  ".join(f"{y}:{v:.0f}" for y, v in yr.items()), amt=(x.lots * x.dn5_px * 1000).mean() / 1e4))
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
cols = [("監看條件", "label"), ("滑價(檔)", "k", 0), ("筆數", "n", 0), ("有交易日平均檔數", "per_day", 1), ("平均投入(萬)", "amt", 1), ("每筆%", "per"), ("勝率", "win", 1), ("年均(萬)", "py", 0), ("Sharpe", "sh"),
        ("最大回撤(萬)", "mdd", 0), ("回撤÷年均", "ratio"), ("最差日(萬)", "worst", 1), ("最差月(萬)", "wm", 1), ("負報酬月%", "negm", 0), ("最長停滯(交易日)", "longest", 0),
        ("探索期年均", "ex", 0), ("驗證期年均", "va", 0), ("各年(萬)", "yrs")]
md = ["# 一檔占比門檻 + 名額遞補, 在不同滑價下", "", "固定 300 萬、每檔 30 萬整數張; 候選與濾網同現行; 一檔占比以開盤 5 分區間低計 (09:05 已知)。", "",
      "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
for r in rows: md.append("| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |")
dist = s.tick_pct
md += ["", f"候選中一檔占比分布: ≤0.15% 占 {(dist <= 0.15).mean()*100:.0f}%、≤0.2% 占 {(dist <= 0.2).mean()*100:.0f}%。占比小的價位帶: 10~50 元段的 33~50 元、50~100 元段的 67~100 元、100~500 元段的 333~500 元 (本規則股價 ≤300, 所以 100 元以上幾乎都 >0.15%)。"]
(STATS / "tick_filter.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
