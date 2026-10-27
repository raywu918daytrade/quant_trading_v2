"""個股 VWAP (自 09:00 起的成交量加權均價) 能不能改善定案規則? 逐分鐘模擬 6,526 筆 (orb_history).
出場變體 (都保留 +8% 強制回補、成本 0.23%; 均價線訊號在該分鐘收盤判斷, 下一分鐘開盤 +0.1% 成交):
  base   現行: 突破區間高停損 (max(區間高, 該分鐘開)+0.1%), 否則收盤
  V1     停損改為「收盤站回 VWAP」(不用區間高)
  V2     停損收緊為 min(區間高, 進場時 VWAP)
  V3     現行 + 11:00 後收盤站回 VWAP 就回補
  V4     現行 + 12:00 後收盤站回 VWAP 就回補
  V5     現行 + 先賺 2% (最低價 ≤ 進場×0.98) 之後, 收盤站回 VWAP 就回補
  V6     現行 + 先賺 1% 之後, 收盤站回 VWAP 就回補
進場: 依「跌破時 VWAP 在進場價之上多少」分組。 -> stats/vwap.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
SL, FS, LOCK = 0.001, 0.002, 0.07
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "pc"]); T = T.merge(f, on=["stock_id", "date"])
T["lu"] = T.pc * 1.1; T["fpx"] = T.pc * 1.08
VARS = ["base", "V1", "V2", "V3", "V4", "V5", "V6"]
res = {v: np.full(len(T), np.nan) for v in VARS}; vw_entry = np.full(len(T), np.nan); exit_t = {v: np.full(len(T), np.nan) for v in VARS}
T = T.reset_index(drop=True)
for ym, grp in T.groupby(T.date.dt.strftime("%Y_%m")):
    m = L.load_m1_month(ym).merge(grp[["stock_id", "date"]], on=["stock_id", "date"]).sort_values(["stock_id", "date", "t"])
    bars = {k: g for k, g in m.groupby(["stock_id", "date"], sort=False)}
    for i, r in grp.iterrows():
        b = bars.get((r.stock_id, r.date))
        if b is None: continue
        t = b.t.values; o = b.open.values; h = b.high.values; lo = b.low.values; c = b.close.values; v = b.volume.values.astype(float)
        vw = np.cumsum(c * v) / np.maximum(np.cumsum(v), 1)
        ie = np.searchsorted(t, r.entry_t)
        if ie >= len(t) or t[ie] != r.entry_t: continue
        ent = r.entry_px; vw_entry[i] = vw[ie]
        for var in VARS:
            stop_lvl = r.rh if var != "V2" else min(r.rh, vw[ie])
            use_rh = var != "V1"
            px = np.nan; pending = False; minlow = ent; et = 270
            for j in range(ie + 1, len(t)):
                if pending:                                   # 前一分鐘收盤站回 VWAP → 這分鐘開盤回補
                    px = o[j] * (1 + SL); et = t[j]; break
                if use_rh and h[j] > stop_lvl:
                    px = max(stop_lvl, o[j]) * (1 + SL); et = t[j]; break
                minlow = min(minlow, lo[j])
                vw_sig = c[j] > vw[j]
                if var == "V1" and vw_sig: pending = True
                elif var == "V3" and t[j] >= 120 and vw_sig: pending = True
                elif var == "V4" and t[j] >= 180 and vw_sig: pending = True
                elif var == "V5" and minlow <= ent * 0.98 and vw_sig: pending = True
                elif var == "V6" and minlow <= ent * 0.99 and vw_sig: pending = True
                if var == "V1" and h[j] >= r.fpx:              # V1 沒有區間高停損 → 需要 +8% 強制回補
                    px = r.fpx * (1 + FS); et = t[j]; break
            if np.isnan(px): px = c[-1]; et = 270
            if px >= r.fpx and var != "V1": px = r.fpx * (1 + FS)
            locked = (et == 270) and (px >= r.lu * 0.999)
            res[var][i] = (-(px / ent - 1) - COST - (LOCK if locked else 0)) * 100; exit_t[var][i] = et
    print(ym, flush=True) if ym.endswith("12") else None
for v in VARS: T[f"r_{v}"] = res[v]; T[f"t_{v}"] = exit_t[v]
T["vw_gap"] = (vw_entry / T.entry_px - 1) * 100                     # 進場時 VWAP 高於進場價多少 %
T.to_parquet(PROC / "orb_vwap_sim.parquet", index=False)
print("base 與歷史 r_pct 差異 (應≈0):", float((T.r_base - T.r_pct).abs().max()))

ALL = pd.DatetimeIndex(pd.read_parquet(PROC / "orb_history" / "days.parquet").date)
def cap(rcol, label, x=None):
    x = T if x is None else x
    pnl = x[rcol] / 100 * x.entry_px * x.lots * 1000
    pn = pnl.groupby(x.date).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); r = pn / 300 * 100
    return dict(label=label, n=len(x), per=x[rcol].mean(), win=(x[rcol] > 0).mean() * 100, total=pn.sum(), per_year=pn.sum() / len(pn) * 245, sharpe=r.mean() / r.std() * np.sqrt(245),
                mdd=(eq - eq.cummax()).min(), worst=pn.min(), ex=x[x.date <= EXPLORE_END][rcol].mean(), va=x[x.date > EXPLORE_END][rcol].mean(),
                early=(x[rcol.replace("r_", "t_")] < 270).mean() * 100 if rcol.replace("r_", "t_") in x else np.nan)
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |\n"
    return out
LAB = {"base": "現行: 區間高停損, 否則收盤", "V1": "V1 停損改為收盤站回 VWAP", "V2": "V2 停損收緊為 min(區間高, 進場時 VWAP)", "V3": "V3 現行 + 11:00 後站回 VWAP 回補",
       "V4": "V4 現行 + 12:00 後站回 VWAP 回補", "V5": "V5 現行 + 先賺 2% 後站回 VWAP 回補", "V6": "V6 現行 + 先賺 1% 後站回 VWAP 回補"}
rows = [cap(f"r_{v}", LAB[v]) for v in VARS]
md = ["# 個股 VWAP 能不能改善定案規則", "",
      f"定案規則 2020-01 ~ {T.date.max().date()} 的 {len(T):,} 筆, 用 1 分 K 逐分鐘重新模擬出場。VWAP = 09:00 起 Σ(收盤×量)/Σ量。均價線訊號在該分鐘收盤判斷, 下一分鐘開盤 +0.1% 回補。固定 300 萬、每檔 30 萬整數張。", "",
      "## 1. 出場方式", "", table(rows, [("出場", "label"), ("每筆%", "per"), ("勝率", "win", 1), ("提前出場%", "early", 0), ("累計損益(萬)", "total", 0), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"),
                                         ("最大回撤(萬)", "mdd", 0), ("最差日(萬)", "worst", 1), ("探索期每筆", "ex"), ("驗證期每筆", "va")]), ""]
# 2. 進場: VWAP 距離
b = pd.qcut(T.vw_gap, 5)
rows = []
for k in b.cat.categories:
    x = T[b == k]; rows.append(dict(label=f"VWAP 高於進場價 {k.left:.2f}~{k.right:.2f}%", n=len(x), per=x.r_base.mean(), win=(x.r_base > 0).mean() * 100, stop=(x.t_base < 270).mean() * 100,
                                    ex=x[x.date <= EXPLORE_END].r_base.mean(), va=x[x.date > EXPLORE_END].r_base.mean()))
md += ["## 2. 進場時離 VWAP 多遠 (現行出場)", "", "第一次跌破區間低時, 價格一定在 VWAP 之下 (VWAP 是開盤以來成交均價, 都 ≥ 區間低)。", "",
       table(rows, [("條件", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("勝率", "win", 1), ("停損率%", "stop", 0), ("探索%", "ex"), ("驗證%", "va")]), ""]
(STATS / "vwap.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
