"""三角突破後的延續: 以 D1 統計突破事件後 T+1..T+10 每天的 跳空 / 開→收 (當沖可抓) / 收→收, 並測「突破後順向當沖 N 天」.
事件 (全部用 D1, 2018 起): 三角日 (07 定義, 收斂比 <= CONTR) 當天 高 > res (向上突破) 或 低 < sup (向下跌破);
  確認 = 收盤仍在線外, 假突破 = 收回線內. 同一檔連續三角日只取第一次突破 (之後 10 日內不重複).
-> stats/triangle_post_breakout.md"""
import numpy as np, pandas as pd
from config import PROC, STATS, COST
CONTR = 0.8; K = 10
tri = pd.read_parquet(PROC / "triangle_blocks.parquet"); tri = tri[tri.contr <= CONTR]
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open", "high", "low", "close", "pc", "gap", "intra", "ret", "univ"]).sort_values(["stock_id", "date"]).reset_index(drop=True)
g = d1.groupby("stock_id", sort=False)
for k in range(1, K + 1):
    for c in ["open", "high", "low", "close", "pc", "gap", "intra", "ret"]:
        d1[f"{c}_{k}"] = g[c].shift(-k)
ev = tri.merge(d1.drop(columns=["open", "high", "low", "close", "pc", "gap", "univ"]), on=["stock_id", "date"])
ev["up_brk"] = ev.high > ev.res; ev["dn_brk"] = ev.low < ev.sup
ev["up_conf"] = ev.up_brk & (ev.close > ev.res); ev["dn_conf"] = ev.dn_brk & (ev.close < ev.sup)
ev = ev[ev.up_brk | ev.dn_brk].sort_values(["stock_id", "date"])
# 去重: 同檔 10 日內只留第一次
keep = []; last = {}
for i, r in ev.iterrows():
    if r.stock_id in last and (r.date - last[r.stock_id]).days <= 14: continue
    keep.append(i); last[r.stock_id] = r.date
ev = ev.loc[keep]
ev["kind"] = np.select([ev.up_conf, ev.up_brk & ~ev.up_conf, ev.dn_conf, ev.dn_brk & ~ev.dn_conf], ["向上突破 確認 (收在上緣之上)", "向上突破 假 (收回線內)", "向下跌破 確認 (收在下緣之下)", "向下跌破 假 (收回線內)"], "其他")
ev = ev[ev.kind != "其他"]
ev.to_parquet(PROC / "triangle_events.parquet", index=False)
base_intra = d1[d1.univ].intra.mean() * 100; base_gap = d1[d1.univ].gap.mean() * 100
print(ev.kind.value_counts().to_string())

def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    rr = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in rr.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
md = [f"# 三角突破後的延續 (D1, 2018~2026, 收斂比 ≤{CONTR})", "",
      f"事件數: " + ", ".join(f"{k} {v}" for k, v in ev.kind.value_counts().items()) + f"。對照: universe 平均每日 開→收 {base_intra:.2f}%、跳空 {base_gap:.2f}%。",
      "T+0 = 突破當天。表中數字是平均 % (未扣成本)；「開→收」就是當沖多單能抓的部分，負值代表當沖應做空。", ""]
for kind, grp in ev.groupby("kind"):
    rows = []
    rows.append(dict(day="T+0 (突破日)", gap=grp.gap.mean() * 100, intra=grp.intra.mean() * 100, intra_win=(grp.intra > 0).mean() * 100, ret=grp.ret.mean() * 100, cum=grp.ret.mean() * 100, above=(grp.close > grp.res).mean() * 100 if "向上" in kind else (grp.close < grp.sup).mean() * 100))
    cum = np.zeros(len(grp))
    for k in range(1, K + 1):
        c = grp[f"close_{k}"] / grp.close - 1
        rows.append(dict(day=f"T+{k}", gap=grp[f"gap_{k}"].mean() * 100, intra=grp[f"intra_{k}"].mean() * 100, intra_win=(grp[f"intra_{k}"] > 0).mean() * 100, intra_med=grp[f"intra_{k}"].median() * 100,
                         ret=grp[f"ret_{k}"].mean() * 100, cum=c.mean() * 100, cum_med=c.median() * 100, above=(grp[f"close_{k}"] > grp.res).mean() * 100 if "向上" in kind else (grp[f"close_{k}"] < grp.sup).mean() * 100,
                         n=int(grp[f"close_{k}"].notna().sum())))
    md += [f"## {kind}  (n={len(grp)})", "", table(rows, [("日", "day"), ("n", "n"), ("跳空%", "gap"), ("開→收%", "intra"), ("開→收 勝率", "intra_win"), ("開→收 中位", "intra_med"), ("收→收%", "ret"), ("累積 (自突破日收盤)%", "cum"), ("累積 中位", "cum_med"), ("仍在線外%", "above")]), ""]

# ---- 突破後順向當沖 N 天 (D1 近似: 開盤進 +0.03%, 收盤出, 成本 0.23%; 空單 高 ≥ 昨收+8% 強制 (+0.2%), 收盤鎖漲停 -7%; 多單對稱) ----
def dt_ret(grp, k, side):
    o = grp[f"open_{k}"] * (1 + 0.0003 if side == "long" else 1 - 0.0003); c = grp[f"close_{k}"]; pc = grp[f"pc_{k}"]; hi = grp[f"high_{k}"]; lo = grp[f"low_{k}"]
    if side == "long":
        forced = lo <= pc * 0.92; ex = np.where(forced, pc * 0.92 * 0.998, c); locked = (~forced) & (ex <= pc * 0.9 * 1.001); ok = o > pc * 0.92
        r = (ex / o - 1 - COST - np.where(locked, 0.07, 0)) * 100
    else:
        forced = hi >= pc * 1.08; ex = np.where(forced, pc * 1.08 * 1.002, c); locked = (~forced) & (ex >= pc * 1.1 * 0.999); ok = o < pc * 1.08
        r = (-(ex / o - 1) - COST - np.where(locked, 0.07, 0)) * 100
    return pd.Series(np.where(ok, r, np.nan), index=grp.index)
rows = []
for kind, side in [("向上突破 確認 (收在上緣之上)", "long"), ("向上突破 確認 (收在上緣之上)", "short"), ("向上突破 假 (收回線內)", "short"), ("向上突破 假 (收回線內)", "long"),
                   ("向下跌破 確認 (收在下緣之下)", "short"), ("向下跌破 確認 (收在下緣之下)", "long"), ("向下跌破 假 (收回線內)", "long"), ("向下跌破 假 (收回線內)", "short")]:
    grp = ev[ev.kind == kind]
    for lab, ks in [("T+1", [1]), ("T+1~3", [1, 2, 3]), ("T+1~5", list(range(1, 6))), ("T+2~5", list(range(2, 6))), ("T+1~10", list(range(1, 11)))]:
        df_ = pd.concat([pd.DataFrame({"r": dt_ret(grp, k, side).values, "year": grp.date.dt.year.values}) for k in ks]).dropna()
        rs = df_.r; yrs = df_.year.values
        row = dict(kind=kind, side="多" if side == "long" else "空", days=lab, n=len(rs), mean=rs.mean(), win=(rs > 0).mean() * 100, t=rs.mean() / rs.std() * np.sqrt(len(rs)))
        for y in range(2018, 2027): row[f"y{y}"] = rs[yrs == y].mean() if (yrs == y).any() else np.nan
        rows.append(row)
md += ["## 突破後順向/逆向當沖 (D1 近似結算: 開盤進, 收盤出, 成本 0.23%, ±8% 強制, 鎖死 −7%)", "",
       table(rows, [("事件", "kind"), ("方向", "side"), ("持有日", "days"), ("筆數", "n"), ("每筆淨%", "mean"), ("勝率", "win"), ("t", "t")] + [(str(y), f"y{y}") for y in range(2018, 2027)]), "",
       "t 以逐筆計 (同一事件的多天彼此相關, 實際 t 會偏小)。"]
(STATS / "triangle_post_breakout.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
