"""定案規則的停損細節: 停損距離、停損率、停損時間、被停損/未停損的報酬, 以及停損高度格點 (區間低 + k×寬)。 -> stats/orb_stop_detail.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "mkt_gap", "dow"])
f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "dn5_t", "dn5_px", "dn5_hi_after"] + [f"dn5_stop_{k}_{c}" for k in ["hi", "k05", "k15", "k20"] for c in ["t", "px", "open"]]
orb = pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms], ignore_index=True)
s = f.merge(orb, on=["stock_id", "date"]); s = s[s.auc_c.notna()]
lu = s.pc * 1.1; fpx = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < lu * 0.999) & (s.o5_h < fpx) & (s.dn5_px < fpx)].sort_values(["date", "dn5_t", "stock_id"])
s = s.groupby("date").head(10).copy()                      # 與定案規則同一批 7,046 筆 (進場與停損無關)
s["lu"] = s.pc * 1.1; s["fpx"] = s.pc * 1.08
s["width"] = (s.o5_h / s.o5_l - 1) * 100                   # 區間寬 %
s["stop_dist"] = (s.o5_h / s.dn5_px - 1) * 100             # 停損距離 (相對進場價) %

def settle(stop):
    """stop: None = 不停損; 否則 'hi'/'k05'/'k15'/'k20'."""
    if stop is None:
        st = pd.Series(np.nan, index=s.index)
    else:
        st = np.maximum(s[f"dn5_stop_{stop}_px"], s[f"dn5_stop_{stop}_open"]) * (1 + P["STOP_SLIP"])
    ex = st.where(st.notna(), s.auc_c)
    hi = s.dn5_hi_after.where(st.isna(), np.minimum(s.dn5_hi_after, st))
    forced = hi >= s.fpx; exf = np.where(forced, s.fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= s.lu * 0.999)
    r = (-(exf / s.dn5_px - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
    return pd.Series(r, index=s.index), st.notna(), pd.Series(forced, index=s.index), pd.Series(locked, index=s.index)

days = np.sort(f.date.unique())
def port_stats(r):
    cap = r.groupby(s.date).sum() / 10; x = pd.Series(0.0, index=days); x.loc[cap.index] = cap.values
    eq = (1 + x / 100).cumprod(); return x.mean(), x.mean() / x.std() * np.sqrt(245), (eq / eq.cummax() - 1).min() * 100, x.min()

rows = []
for stop, lab in [(None, "不停損 (抱到收盤, 只有 +8% 強制回補)"), ("k05", "區間中點 (區間低 + 0.5×寬)"), ("hi", "區間高 (定案)"), ("k15", "區間低 + 1.5×寬"), ("k20", "區間低 + 2×寬")]:
    r, stp, fo, lk = settle(stop)
    dm, sh, mdd, wd = port_stats(r)
    rows.append(dict(停損位置=lab, 每筆淨=r.mean(), 勝率=(r > 0).mean() * 100, 停損率=stp.mean() * 100, 強制回補=fo.mean() * 100, 鎖漲停=lk.mean() * 100,
                     被停損那些平均=r[stp].mean() if stp.any() else np.nan, 未停損那些平均=r[~stp].mean(), 最差一筆=r.min(), 資金日報酬=dm, Sharpe=sh, MDD=mdd, 最差日=wd,
                     探索=r[s.date <= "2023-12-31"].mean(), 驗證=r[s.date > "2023-12-31"].mean()))
tab = pd.DataFrame(rows)

r, stp, _, _ = settle("hi"); s["r"] = r; s["stopped"] = stp
s["stop_after_min"] = s.dn5_stop_hi_t - s.dn5_t
st = s[s.stopped]
naive = (-(st.auc_c / st.dn5_px - 1) - COST) * 100        # 被停損的那些若抱到收盤 (不含 +8% 強制)
bins = pd.cut(st.stop_after_min, [0, 5, 15, 30, 60, 120, 270], labels=["≤5 分", "5~15", "15~30", "30~60", "60~120", ">120"])
by_t = st.groupby(bins, observed=True).agg(筆數=("r", "size"), 平均=("r", "mean")).reset_index().rename(columns={"stop_after_min": "進場後多久被停損"})
wb = pd.qcut(s.stop_dist, 5)
by_w = s.groupby(wb, observed=True).agg(筆數=("r", "size"), 停損率=("stopped", lambda x: x.mean() * 100), 每筆淨=("r", "mean"), 被停損平均=("r", lambda x: x[s.loc[x.index, "stopped"]].mean())).reset_index()
by_w["停損距離"] = by_w.stop_dist.astype(str); by_w = by_w.drop(columns="stop_dist")

md = ["# 定案規則的停損細節 (2020-01 ~ 2026-10, 7,046 筆)", "",
      f"停損距離 (區間高 / 進場價 − 1): 中位 {s.stop_dist.median():.2f}%, 25~75% {s.stop_dist.quantile(.25):.2f}~{s.stop_dist.quantile(.75):.2f}%, 90% {s.stop_dist.quantile(.9):.2f}%, 最大 {s.stop_dist.max():.2f}%。區間寬中位 {s.width.median():.2f}%。",
      f"被停損的 {len(st):,} 筆: 平均 {st.r.mean():.2f}%, 中位 {st.r.median():.2f}%；這批若抱到收盤 (不含 +8% 強制) 平均 {naive.mean():.2f}%, 中位 {naive.median():.2f}%, 其中收盤反而賺錢的 {(naive > 0).mean()*100:.0f}%。",
      f"進場到停損的時間 中位 {st.stop_after_min.median():.0f} 分鐘。", "",
      "## 停損位置比較", "", tab.round(2).to_markdown(index=False), "",
      "## 進場後多久被停損", "", by_t.round(2).to_markdown(index=False), "",
      "## 依停損距離五分位", "", by_w.round(2).to_markdown(index=False)]
(STATS / "orb_stop_detail.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
