"""三角突破確認後的第一個 MACD 柱狀體背離 (D1, 12/26/9).
頂背離 (向上突破後): 柱狀體正值高峰 (hist[i]>hist[i-1] 且 hist[i]>hist[i+1], hist[i]>0) 低於前一個正值高峰, 但該高峰附近 (前後 2 日) 的價格最高 > 前一高峰附近的價格最高.
底背離 (向下跌破後): 對稱 (負值谷, 谷變淺, 價格創新低). 高峰/谷需隔一日確認 → 訊號日 D = 峰/谷日 + 1 (收盤後可知), 之後 T+1.. 自 D 起算.
搜尋視窗: 突破日後 LOOK 日內. -> stats/macd_divergence.md"""
import numpy as np, pandas as pd
from config import PROC, STATS, COST
LOOK = 30; N = 5
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open", "high", "low", "close", "pc", "gap", "intra", "ret", "univ", "mkt_gap"]).sort_values(["stock_id", "date"]).reset_index(drop=True)
g = d1.groupby("stock_id", sort=False)
ema12 = g.close.transform(lambda s: s.ewm(span=12, adjust=False).mean()); ema26 = g.close.transform(lambda s: s.ewm(span=26, adjust=False).mean())
d1["dif"] = ema12 - ema26; d1["dea"] = d1.groupby("stock_id", sort=False).dif.transform(lambda s: s.ewm(span=9, adjust=False).mean()); d1["hist"] = d1.dif - d1.dea
d1["i"] = g.cumcount()
idx = d1.set_index(["stock_id", "date"]).i
def build_events(contr_max):
    tri = pd.read_parquet(PROC / "triangle_blocks.parquet"); tri = tri[tri.contr <= contr_max][["stock_id", "date", "res", "sup", "contr"]]
    e = tri.merge(d1[["stock_id", "date", "high", "low", "close"]], on=["stock_id", "date"])
    e["up_conf"] = (e.high > e.res) & (e.close > e.res); e["dn_conf"] = (e.low < e.sup) & (e.close < e.sup)
    e = e[e.up_conf | e.dn_conf].sort_values(["stock_id", "date"])
    keep = []; last = {}
    for i, r in e.iterrows():
        if r.stock_id in last and (r.date - last[r.stock_id]).days <= 14: continue
        keep.append(i); last[r.stock_id] = r.date
    e = e.loc[keep].copy()
    e["kind"] = np.where(e.up_conf, "向上突破 確認 (收在上緣之上)", "向下跌破 確認 (收在下緣之下)")
    e["i0"] = idx.reindex(pd.MultiIndex.from_frame(e[["stock_id", "date"]])).values
    return e[["stock_id", "date", "kind", "res", "sup", "contr", "i0"]]
arr = {c: d1[c].values for c in ["open", "high", "low", "close", "pc", "gap", "intra", "ret", "hist", "mkt_gap"]}
stock_end = d1.groupby("stock_id", sort=False).i.max(); stock_start = d1.groupby("stock_id", sort=False).apply(lambda x: x.index[0])

def run(ev):
  rows = []
  for _, e in ev.iterrows():
      base = stock_start[e.stock_id]; n_end = stock_end[e.stock_id]
      up = e.kind.startswith("向上"); sgn = 1 if up else -1
      t0 = base + int(e.i0)
      h = arr["hist"]; peaks = []   # (day index, hist value, price extreme near it)
      found = None
      for i in range(t0 - 1, min(t0 + LOOK, base + n_end - 1)):
          if i + 1 > base + n_end: break
          if sgn > 0 and h[i] > 0 and h[i] > h[i - 1] and h[i] > h[i + 1]:
              px = arr["high"][max(i - 2, base): i + 3].max(); peaks.append((i, h[i], px))
          if sgn < 0 and h[i] < 0 and h[i] < h[i - 1] and h[i] < h[i + 1]:
              px = arr["low"][max(i - 2, base): i + 3].min(); peaks.append((i, h[i], px))
          if len(peaks) >= 2 and i >= t0:
              (i1, h1, p1), (i2, h2, p2) = peaks[-2], peaks[-1]
              if i2 == i and ((sgn > 0 and h2 < h1 and p2 > p1) or (sgn < 0 and h2 > h1 and p2 < p1)):
                  found = i; break
      if found is None:
          rows.append(dict(stock_id=e.stock_id, date=e.date, kind=e.kind, contr=e.contr, has_div=False)); continue
      D = found + 1                                           # 訊號日 (確認峰/谷)
      r = dict(stock_id=e.stock_id, date=e.date, kind=e.kind, contr=e.contr, has_div=True, days_to_div=D - t0, hist_peak1=peaks[-2][1], hist_peak2=peaks[-1][1],
               px_move=(peaks[-1][2] / peaks[-2][2] - 1) * 100, close_D=arr["close"][D], D_vs_brk=(arr["close"][D] / arr["close"][t0] - 1) * 100,
               still_out=float(arr["close"][D] > e.res) if up else float(arr["close"][D] < e.sup), sig_date=d1.date.values[D])
      for k in range(1, N + 1):
          j = D + k
          if j > base + n_end: break
          for c in ["open", "high", "low", "close", "pc", "gap", "intra", "ret"]: r[f"{c}_{k}"] = arr[c][j]
      rows.append(r)
  return pd.DataFrame(rows)


def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1])) for c in cols) + " |\n"
    return out
def dt(x, k, side):
    o = x[f"open_{k}"] * (1 + 0.0003 if side == "long" else 1 - 0.0003); c = x[f"close_{k}"]; pc = x[f"pc_{k}"]; hi = x[f"high_{k}"]; lo = x[f"low_{k}"]
    if side == "long":
        forced = lo <= pc * 0.92; ex = np.where(forced, pc * 0.92 * 0.998, c); locked = (~forced) & (ex <= pc * 0.9 * 1.001); ok = o > pc * 0.92; r = (ex / o - 1 - COST - np.where(locked, 0.07, 0)) * 100
    else:
        forced = hi >= pc * 1.08; ex = np.where(forced, pc * 1.08 * 1.002, c); locked = (~forced) & (ex >= pc * 1.1 * 0.999); ok = o < pc * 1.08; r = (-(ex / o - 1) - COST - np.where(locked, 0.07, 0)) * 100
    return pd.Series(np.where(ok, r, np.nan), index=x.index)

md = ["# 三角突破後第一個 MACD 柱狀體背離 (D1 12/26/9)", "",
      f"事件: 三角突破確認日 (2018~2026) 後 {LOOK} 日內的第一個背離。向上突破 → 頂背離 (價創高、柱峰變矮)；向下跌破 → 底背離 (價創低、柱谷變淺)。訊號日 D = 峰/谷隔天 (收盤後確認)，T+k 自 D 起算。", ""]
for contr_max, cl in [(0.8, "收斂比 ≤0.8"), (9, "不限收斂比 (樣本較多)")]:
  res = run(build_events(contr_max)); res.to_parquet(PROC / f"macd_div_events_{contr_max}.parquet", index=False)
  md += [f"# ===== {cl} =====", ""]
  for kind in ["向上突破 確認 (收在上緣之上)", "向下跌破 確認 (收在下緣之下)"]:
      x = res[res.kind == kind]; dv = x[x.has_div].dropna(subset=["close_3"]); sgn = 1 if "向上" in kind else -1
      md += [f"## {kind}: 突破 {len(x)} 筆, {LOOK} 日內出現背離 {int(x.has_div.sum())} 筆 ({x.has_div.mean()*100:.0f}%)", "",
             f"- 背離出現在突破後第 {dv.days_to_div.median():.0f} 日 (中位, 25~75%: {dv.days_to_div.quantile(.25):.0f}~{dv.days_to_div.quantile(.75):.0f})；訊號日收盤相對突破日收盤 {dv.D_vs_brk.mean():+.1f}% (中位 {dv.D_vs_brk.median():+.1f}%)；訊號日仍在線外 {dv.still_out.mean()*100:.0f}%。",
             f"- 兩峰之間價格 {'創高' if sgn>0 else '創低'}幅度 中位 {dv.px_move.median():+.1f}%。", ""]
      rows = []
      for k in range(1, N + 1):
          cum = (dv[f"close_{k}"] / dv.close_D - 1)
          rows.append(dict(day=f"T+{k}", n=int(dv[f"close_{k}"].notna().sum()), gap=dv[f"gap_{k}"].mean() * 100, intra=dv[f"intra_{k}"].mean() * 100, intra_win=(dv[f"intra_{k}"] > 0).mean() * 100,
                           ret=dv[f"ret_{k}"].mean() * 100, cum=cum.mean() * 100, cum_med=cum.median() * 100, p_up3=(cum >= 0.03).mean() * 100, p_dn3=(cum <= -0.03).mean() * 100))
      md += ["訊號日之後每天 (平均 %, 未扣成本):", "", table(rows, [("日", "day"), ("n", "n"), ("跳空", "gap"), ("開→收", "intra"), ("開→收>0 %", "intra_win"), ("收→收", "ret"), ("累積自 D 收盤", "cum"), ("累積中位", "cum_med"), ("累積≥+3% %", "p_up3"), ("累積≤−3% %", "p_dn3")]), ""]
      rows = []
      for side in ["long", "short"]:
          for lab, ks in [("T+1", [1]), ("T+1~3", [1, 2, 3]), ("T+1~5", [1, 2, 3, 4, 5])]:
              rs = pd.concat([dt(dv, k, side) for k in ks]).dropna()
              rows.append(dict(side="多" if side == "long" else "空", days=lab, n=len(rs), mean=rs.mean(), win=(rs > 0).mean() * 100, t=rs.mean() / rs.std() * np.sqrt(len(rs)) if len(rs) > 2 else np.nan))
      md += ["背離訊號後當沖 (D1 近似: 開盤進收盤出, 成本 0.23%, ±8% 強制, 鎖死 −7%):", "", table(rows, [("方向", "side"), ("持有日", "days"), ("筆數", "n"), ("每筆淨%", "mean"), ("勝率", "win"), ("t", "t")]), ""]
      # 沒有背離的那些: 之後表現 (對照)
      nd = x[~x.has_div]
      md += [f"- 對照: {LOOK} 日內沒出現背離的 {len(nd)} 筆 (多半是突破後很快回落或 MACD 柱沒形成第二個峰)。", ""]
(STATS / "macd_divergence.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
