"""三角收斂 (擺盪點版): 5 根 K 一組找高低點 — 擺盪高點 = 當日高是前後各 2 日 (共 5 日) 的最高; 擺盪低點對稱 (需 2 日後才確認).
從今日往回取最近 2 個擺盪高點 (H1 早, H2 晚) 與最近 2 個擺盪低點 (L1 早, L2 晚): 收斂 = H2 < H1 且 L2 > L1; 兩高點之間所有日高 < H2, 兩低點之間所有日低 > L2; 4 點時間交錯 (HLHL 或 LHLH).
上緣 = H1→H2 連線, 下緣 = L1→L2 連線, 外推到今日 = 突破價 res / 支撐 sup. 條件: 昨收在線內, 最近點之後的日高/低未破線, 4 點跨度 MIN_SPAN~MAX_SPAN 日.
-> data/processed/triangle_blocks.parquet"""
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from config import PROC
W = 2; TOL = 0.003; MIN_SPAN, MAX_SPAN = 8, 60

def one(stock):
    s = stock.sort_values("date").reset_index(drop=True); n = len(s)
    if n < 15: return None
    hi = s.high.values.astype(float); lo = s.low.values.astype(float); cl = s.close.values.astype(float); pc = s.pc.values.astype(float)
    ph = np.zeros(n, bool); pl = np.zeros(n, bool)
    for i in range(W, n - W):
        if np.isfinite(hi[i]) and hi[i] >= np.nanmax(hi[i-W:i+W+1]) and np.sum(hi[i-W:i+W+1] == hi[i]) == 1: ph[i] = True
        if np.isfinite(lo[i]) and lo[i] <= np.nanmin(lo[i-W:i+W+1]) and np.sum(lo[i-W:i+W+1] == lo[i]) == 1: pl[i] = True
    PH = np.where(ph)[0]; PL = np.where(pl)[0]
    rows = []
    for t in range(10, n):
        if not np.isfinite(pc[t]): continue
        conf = t - 1 - W                                   # 到昨日為止能確認的擺盪點 index 上限
        hs = PH[PH <= conf]; ls = PL[PL <= conf]
        if len(hs) < 2 or len(ls) < 2: continue
        h1, h2 = hs[-2], hs[-1]; l1, l2 = ls[-2], ls[-1]
        if not (hi[h2] < hi[h1] and lo[l2] > lo[l1]): continue
        # 兩個高點之間不能有其他高點 (中間所有日高 < 第 2 高點); 兩個低點之間不能有其他低點
        if h2 - h1 > 1 and np.nanmax(hi[h1+1:h2]) >= hi[h2]: continue
        if l2 - l1 > 1 and np.nanmin(lo[l1+1:l2]) <= lo[l2]: continue
        # 4 點在時間上必須交錯 (高-低-高-低 或 低-高-低-高)
        order = "".join(k for _, k in sorted([(h1, "H"), (h2, "H"), (l1, "L"), (l2, "L")]))
        if order not in ("HLHL", "LHLH"): continue
        first = min(h1, l1); span = t - first
        if span < MIN_SPAN or span > MAX_SPAN: continue
        sh = (hi[h2] - hi[h1]) / (h2 - h1); sl = (lo[l2] - lo[l1]) / (l2 - l1)
        res = hi[h1] + sh * (t - h1); sup = lo[l1] + sl * (t - l1)
        if not (res > sup): continue
        y = t - 1
        if not (cl[y] < hi[h1] + sh * (y - h1) and cl[y] > lo[l1] + sl * (y - l1)): continue
        # 最近擺盪點之後到昨日, 日高未破上緣 / 日低未破下緣
        xs = np.arange(h2 + 1, t); 
        if len(xs) and np.any(hi[xs] > hi[h1] + sh * (xs - h1) + TOL * pc[t]): continue
        xs = np.arange(l2 + 1, t)
        if len(xs) and np.any(lo[xs] < lo[l1] + sl * (xs - l1) - TOL * pc[t]): continue
        # 兩線之間 (h1,l1 之前) 區間內所有日高/低不穿線 (切線性質)
        xs = np.arange(h1, t)
        if np.any(hi[xs] > hi[h1] + sh * (xs - h1) + TOL * pc[t]): continue
        xs = np.arange(l1, t)
        if np.any(lo[xs] < lo[l1] + sl * (xs - l1) - TOL * pc[t]): continue
        # 更早是否還有連續下降高點 / 上升低點 (第 3 個點)
        nh = 2 + sum(1 for k in range(len(hs) - 3, -1, -1) if hi[hs[k]] > hi[hs[k+1]] and all(hi[hs[j]] > hi[hs[j+1]] for j in range(k, len(hs) - 1)))
        nl = 2 + sum(1 for k in range(len(ls) - 3, -1, -1) if lo[ls[k]] < lo[ls[k+1]] and all(lo[ls[j]] < lo[ls[j+1]] for j in range(k, len(ls) - 1)))
        early_rng = hi[h1] - lo[l1]; late_rng = hi[h2] - lo[l2]
        rows.append(dict(stock_id=s.stock_id.iloc[0], date=s.date.iloc[t], res=res, sup=sup, slope_h=sh / pc[t], slope_l=sl / pc[t], width=(res - sup) / pc[t],
                         contr=late_rng / early_rng if early_rng > 0 else np.nan, span=span, iH1=h1, iH2=h2, iL1=l1, iL2=l2, H1=hi[h1], H2=hi[h2], L1=lo[l1], L2=lo[l2], n_desc_h=nh, n_asc_l=nl))
    return pd.DataFrame(rows) if rows else None

if __name__ == "__main__":
    d = pd.read_parquet(PROC / "d1_feat.parquet")
    parts = [g for _, g in d.groupby("stock_id", sort=False)]
    with ProcessPoolExecutor(8) as ex: outs = [o for o in ex.map(one, parts, chunksize=20) if o is not None]
    tri = pd.concat(outs, ignore_index=True)
    tri = tri.merge(d[["stock_id", "date", "name", "univ", "pc", "open", "high", "low", "close", "gap", "vol20", "dv20", "atr20", "p_vr", "vr5_20", "mkt_gap", "dow"]], on=["stock_id", "date"])
    tri = tri[tri.univ].reset_index(drop=True)
    tri.to_parquet(PROC / "triangle_blocks.parquet", index=False)
    u = int(d.univ.sum())
    print(f"universe 股票日 {u:,}; 三角成立 {len(tri):,} ({len(tri)/u*100:.2f}%), 每日平均 {len(tri)/tri.date.nunique():.1f} 檔")
    print("跨度 (日) 分布:", tri.span.describe()[["25%", "50%", "75%"]].astype(int).to_dict(), "| 收斂比 (後段區間/前段):", tri.contr.describe()[["25%", "50%", "75%"]].round(2).to_dict())
    print("高點連續下降個數:", tri.n_desc_h.value_counts().sort_index().to_dict(), "| 低點連續上升個數:", tri.n_asc_l.value_counts().sort_index().to_dict())
