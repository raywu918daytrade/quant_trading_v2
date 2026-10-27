"""定案規則 (orb_live) 的漲停無法平倉風險: 逐筆用 M1 檢查進場後是否碰漲停、停損那一分鐘是否已在漲停、是否整天鎖死到收盤.
-> stats/orb_limit_risk.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, RAW, COST

e = L.expected_trades(pd.Timestamp("2020-01-01"), pd.Timestamp("2026-10-22"))
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "pc"])
e = e.merge(f, on=["stock_id", "date"])
orb = pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=["stock_id", "date", "o5_h", "dn5_stop_hi_t", "auc_c"]) for ym in sorted({d.strftime("%Y_%m") for d in e.date})])
e = e.merge(orb, on=["stock_id", "date"])
e["lu"] = e.pc.map(L.limit_up)
# 隔日開盤 (D1)
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open"]).sort_values(["stock_id", "date"])
d1["next_open"] = d1.groupby("stock_id").open.shift(-1)
e = e.merge(d1[["stock_id", "date", "next_open"]], on=["stock_id", "date"], how="left")

rows = []
for ym, grp in e.groupby(e.date.dt.strftime("%Y_%m")):
    m = L.load_m1_month(ym)
    m = m.merge(grp[["stock_id", "date", "dn5_t", "dn5_stop_hi_t", "lu", "o5_h"]], on=["stock_id", "date"])
    m = m[m.t > m.dn5_t]                                               # 進場之後的分 K
    at_lu = m.high >= m.lu - 1e-6
    full_locked = m.low >= m.lu - 1e-6                                 # 整分鐘都在漲停價 (買不到低於漲停的價)
    g = m.assign(at_lu=at_lu, full_locked=full_locked).groupby(["stock_id", "date"])
    a = g.agg(hit_lu=("at_lu", "any"), first_lu_t=("t", lambda s: np.nan))
    a["first_lu_t"] = m[at_lu].groupby(["stock_id", "date"]).t.min()
    a["n_locked_min"] = g.full_locked.sum()
    sb = m[m.t == m.dn5_stop_hi_t].set_index(["stock_id", "date"])     # 停損那一分鐘
    a["stop_open"] = sb.open; a["stop_high"] = sb.high; a["stop_low"] = sb.low
    # 停損那一分鐘之後, 第一根有成交在漲停價以下的分鐘 (可以買回的最早時間)
    after_stop = m[m.t >= m.dn5_stop_hi_t]
    unl = after_stop[after_stop.low < after_stop.lu - 1e-6].groupby(["stock_id", "date"]).agg(unlock_t=("t", "min"))
    a = a.join(unl)
    rows.append(a.reset_index())
x = e.merge(pd.concat(rows), on=["stock_id", "date"], how="left")
x["stopped"] = x.dn5_stop_hi_t.notna()
x["stop_in_lu_bar"] = x.stopped & (x.stop_high >= x.lu - 1e-6)             # 停損那分鐘就碰到漲停
x["stop_bar_all_lu"] = x.stopped & (x.stop_low >= x.lu - 1e-6)             # 停損那分鐘整分鐘都在漲停 (完全買不到)
x["stuck_to_close"] = x.stopped & x.unlock_t.isna()                         # 停損後到收盤都沒有低於漲停的成交
x["close_lu"] = x.auc_c >= x.lu - 1e-6

n = len(x); s = x[x.stopped]
L1 = [f"# 定案規則的漲停無法平倉風險 (2020-01 ~ 2026-10, {n:,} 筆)", "",
      "規則上停損價 (開盤 5 分區間高) 一定低於漲停, 價格要到漲停必先穿過停損價; 風險只在「從停損價下方瞬間拉到漲停並鎖死」。", "",
      "| 項目 | 筆數 | 占全部 |", "|---|---|---|",
      f"| 全部交易 | {n:,} | 100% |",
      f"| 進場後盤中曾碰漲停 | {int(x.hit_lu.sum())} | {x.hit_lu.mean()*100:.2f}% |",
      f"| 被停損 | {len(s):,} | {len(s)/n*100:.1f}% |",
      f"| 停損那一分鐘就碰到漲停 | {int(x.stop_in_lu_bar.sum())} | {x.stop_in_lu_bar.mean()*100:.2f}% |",
      f"| 停損那一分鐘整分鐘都在漲停 (當下完全買不到) | {int(x.stop_bar_all_lu.sum())} | {x.stop_bar_all_lu.mean()*100:.3f}% |",
      f"| 停損後一直到收盤都沒有低於漲停的成交 (真正卡住) | {int(x.stuck_to_close.sum())} | {x.stuck_to_close.mean()*100:.3f}% |",
      f"| 沒停損且收盤在漲停 (理論上不可能) | {int((~x.stopped & x.close_lu).sum())} | |", ""]
# 停損那分鐘碰漲停的交易: 回測假設的成交價 vs 漲停價 vs 解鎖後最早可買
y = x[x.stop_in_lu_bar].copy()
if len(y):
    y["bt_fill"] = np.maximum(y.o5_h, y.stop_open) * 1.001
    y["r_bt"] = y.r_pct
    y["r_at_lu"] = (-(y.lu / y.dn5_px - 1) - COST) * 100
    L1 += ["## 停損那一分鐘就碰到漲停的交易", "",
           y.assign(date=y.date.dt.date)[["date", "stock_id", "pc", "dn5_px", "o5_h", "lu", "stop_open", "stop_low", "r_bt", "r_at_lu", "unlock_t", "n_locked_min"]].round(2).to_markdown(index=False), "",
           "r_bt = 回測報酬 (停損成交 = max(區間高, 該分鐘開盤)+0.1%)；r_at_lu = 若以漲停價買回的報酬；unlock_t = 停損後第一根有低於漲停成交的分鐘 (09:00=0)；n_locked_min = 進場後整分鐘鎖在漲停的分鐘數。", ""]
stuck = x[x.stuck_to_close]
if len(stuck):
    stuck = stuck.assign(r_next_open=(-(stuck.next_open / stuck.dn5_px - 1) - COST - 0.07) * 100)
    L1 += ["## 真正卡到收盤的交易 (隔日開盤回補 + 7% 標借罰金)", "", stuck.assign(date=stuck.date.dt.date)[["date", "stock_id", "dn5_px", "lu", "auc_c", "next_open", "r_pct", "r_next_open"]].round(2).to_markdown(index=False), ""]
# 漲停前的距離: 停損價離漲停多遠
x["gap_to_lu"] = (x.lu / x.o5_h - 1) * 100
L1 += ["## 停損價 (區間高) 離漲停的距離", "",
       f"- 中位數 {x.gap_to_lu.median():.1f}%，10% 分位 {x.gap_to_lu.quantile(.1):.1f}%，最小 {x.gap_to_lu.min():.2f}%。",
       f"- 距離 <2% 的交易 {int((x.gap_to_lu < 2).sum())} 筆 ({(x.gap_to_lu < 2).mean()*100:.1f}%)，其中停損那分鐘碰漲停 {int(x[x.gap_to_lu < 2].stop_in_lu_bar.sum())} 筆。"]
(STATS / "orb_limit_risk.md").write_text("\n".join(L1), encoding="utf-8")
print("\n".join(L1))

# ===== 停損延遲 k 分鐘才送單: 那時是否已鎖漲停、要等多久才買得到 =====
res = []
for ym, grp in x[x.stopped].groupby(x[x.stopped].date.dt.strftime("%Y_%m")):
    m = L.load_m1_month(ym).merge(grp[["stock_id", "date", "dn5_stop_hi_t", "lu", "dn5_px"]], on=["stock_id", "date"])
    m = m[m.t >= m.dn5_stop_hi_t].sort_values(["stock_id", "date", "t"])
    for (sid, d), b in m.groupby(["stock_id", "date"]):
        t0 = b.dn5_stop_hi_t.iloc[0]; lu = b.lu.iloc[0]
        for k in [1, 2, 5]:
            bb = b[b.t >= t0 + k]
            if bb.empty: continue
            first = bb.iloc[0]
            locked_now = first.low >= lu - 1e-6
            ok = bb[bb.low < lu - 1e-6]
            res.append(dict(stock_id=sid, date=d, k=k, locked_now=locked_now, wait=(ok.t.iloc[0] - first.t) if len(ok) else np.nan, stuck=ok.empty,
                            px=(first.open if not locked_now else (ok.iloc[0].open if len(ok) else np.nan)), entry=b.dn5_px.iloc[0], lu=lu))
r = pd.DataFrame(res)
L2 = ["", "## 停損單晚 k 分鐘才送出的情況 (被停損的 2,697 筆)", "", "| 延遲 | 送單那分鐘已整分鐘鎖漲停 | 其中要等的分鐘數 (中位) | 一直鎖到收盤 |", "|---|---|---|---|"]
for k, g in r.groupby("k"):
    lk = g[g.locked_now]
    L2.append(f"| {k} 分鐘 | {len(lk)} 筆 ({len(lk)/len(g)*100:.2f}%) | {lk.wait.median():.0f} | {int(g.stuck.sum())} 筆 |")
(STATS / "orb_limit_risk.md").write_text("\n".join(L1 + L2), encoding="utf-8")
print("\n".join(L2))
