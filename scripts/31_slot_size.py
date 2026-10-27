"""部位配置: 每檔 30 萬最多 10 檔 vs 每檔 60 萬最多 5 檔 (固定總資金 300 萬), 現行 / 一檔 ≤0.15%, 滑價 0/1/2 檔。
另算流動性: 部位 / 開盤 5 分鐘成交額、部位 / 20 日均成交額。 -> stats/slot_size.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"]); f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "o5_dv", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
s = f.merge(pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms]), on=["stock_id", "date"]); s = s[s.auc_c.notna()]
s["lu"] = s.pc * 1.1; s["fpx"] = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < s.lu * 0.999) & (s.o5_h < s.fpx) & (s.dn5_px < s.fpx) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
s["tick_l"] = s.o5_l.map(L.tick_size); s["tick_h"] = s.o5_h.map(L.tick_size); s["tick_pct"] = s.tick_l / s.o5_l * 100
def settle(x, k, slot):
    lots = np.maximum((slot // (x.o5_l * 1000)).astype(int), 1)
    ent = x.dn5_px - k * x.tick_l
    raw = np.maximum(x.dn5_stop_hi_px, x.dn5_stop_hi_open)
    st = raw * (1 + P["STOP_SLIP"]) if k == 0 else raw + k * x.tick_h
    ex = st.where(x.dn5_stop_hi_px.notna(), x.auc_c)
    hi = x.dn5_hi_after.where(x.dn5_stop_hi_px.isna(), np.minimum(x.dn5_hi_after, st)); forced = hi >= x.fpx
    exf = np.where(forced, x.fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= x.lu * 0.999)
    r = (-(exf / ent - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
    return r, r / 100 * ent * lots * 1000, lots * ent * 1000
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
rows = []
for th, flab in [(99, "現行"), (0.15, "一檔 ≤0.15%")]:
    for slot, maxn in ([(3e5, 10), (6e5, 5)] if th > 1 else [(3e5, 10), (6e5, 5), (1e6, 3)]):
        x = s[s.tick_pct <= th].sort_values(["date", "dn5_t", "stock_id"]).groupby("date").head(maxn)
        for k in [0, 1, 2]:
            r, pnl, amt = settle(x, k, slot)
            pn = pnl.groupby(x.date).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); dd = eq - eq.cummax()
            uw = (dd < 0).astype(int); mon = pn.groupby(pn.index.to_period("M")).sum(); used = amt.groupby(x.date).sum()
            py = pn.sum() / len(pn) * 245
            rows.append(dict(label=f"{flab}, 每檔 {slot/1e4:.0f} 萬 最多 {maxn} 檔", k=k, n=len(x), per_day=len(x) / x.date.nunique(), used=used.mean() / 1e4, per=r.mean(), py=py,
                             sh=pn.mean() / pn.std() * np.sqrt(245), mdd=dd.min(), ratio=(-dd.min() / py) if py > 0 else np.nan, worst=pn.min(), wm=mon.min(), negm=(mon < 0).mean() * 100,
                             longest=int(uw.groupby((uw == 0).cumsum()).sum().max()), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245,
                             va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245,
                             liq5=(amt / (x.o5_dv * 1000) * 100).median(), liq5_90=(amt / (x.o5_dv * 1000) * 100).quantile(0.9), liqd=(amt / x.dv20 * 100).median()))
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
cols = [("配置", "label"), ("滑價(檔)", "k", 0), ("筆數", "n", 0), ("有交易日平均檔數", "per_day", 1), ("有交易日平均投入(萬)", "used", 0), ("每筆%", "per"), ("年均(萬)", "py", 0), ("Sharpe", "sh"),
        ("最大回撤(萬)", "mdd", 0), ("回撤÷年均", "ratio"), ("最差日(萬)", "worst", 1), ("最差月(萬)", "wm", 1), ("負報酬月%", "negm", 0), ("最長停滯(交易日)", "longest", 0),
        ("探索期年均", "ex", 0), ("驗證期年均", "va", 0), ("部位/開盤5分成交額 中位%", "liq5", 1), ("90分位%", "liq5_90", 1), ("部位/20日均成交額 中位%", "liqd", 2)]
md = ["# 部位配置: 每檔 30 萬 × 10 / 60 萬 × 5 / 100 萬 × 3", "", "總資金固定 300 萬; 候選與濾網同現行; 先跌破先做, 名額遞補; 整數張、最少 1 張。", "",
      "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
for r in rows: md.append("| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |")
(STATS / "slot_size.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
