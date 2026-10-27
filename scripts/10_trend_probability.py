"""三角突破後 3 日內出現「明顯趨勢」的機率, 多種定義, 並附 universe 隨機日基準. -> stats/trend_probability.md"""
import numpy as np, pandas as pd
from config import PROC, STATS
N = 3
ev = pd.read_parquet(PROC / "triangle_events.parquet")
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open", "high", "low", "close", "intra", "ret", "univ", "atr20"]).sort_values(["stock_id", "date"]).reset_index(drop=True)
g = d1.groupby("stock_id", sort=False)
for k in range(1, N + 1):
    for c in ["open", "high", "low", "close", "intra", "ret"]: d1[f"{c}_{k}"] = g[c].shift(-k)
base = d1[d1.univ & (d1.date >= "2018-01-01")].dropna(subset=[f"close_{N}"])
hi_atr = base[base.atr20 >= 0.03]     # 三角股多半波動較高, 另附高波動基準

def probs(x, sign):
    """x: 有 close, close_k, high_k, low_k, intra_k, ret_k 的表. sign=+1 看向上趨勢, -1 看向下. 回傳各定義的機率 %"""
    c0 = x.close
    cum = sign * (x[f"close_{N}"] / c0 - 1)
    cums = [sign * (x[f"close_{k}"] / c0 - 1) for k in range(1, N + 1)]
    hi3 = pd.concat([x[f"high_{k}"] for k in range(1, N + 1)], axis=1).max(axis=1); lo3 = pd.concat([x[f"low_{k}"] for k in range(1, N + 1)], axis=1).min(axis=1)
    ext = (hi3 / c0 - 1) if sign > 0 else (1 - lo3 / c0)          # 順向最大觸及幅度 (正)
    adv = (lo3 / c0 - 1) if sign > 0 else (1 - hi3 / c0)          # 逆向最大觸及幅度 (負)
    intra_sum = sign * sum(x[f"intra_{k}"] for k in range(1, N + 1))
    rets = [sign * x[f"ret_{k}"] for k in range(1, N + 1)]
    o = dict(n=len(x))
    for th in [0.02, 0.03, 0.05]: o[f"cum≥{th*100:.0f}%"] = (cum >= th).mean() * 100
    for th in [0.03, 0.05]: o[f"曾觸及{th*100:.0f}%"] = (ext >= th).mean() * 100
    o["3日收盤全在突破收盤同側"] = (pd.concat(cums, axis=1) > 0).all(axis=1).mean() * 100
    o["3日連續同向收"] = (pd.concat(rets, axis=1) > 0).all(axis=1).mean() * 100
    for th in [0.02, 0.03]: o[f"3日日內合計≥{th*100:.0f}%"] = (intra_sum >= th).mean() * 100
    o["反向≥3% (3日累積)"] = (cum <= -0.03).mean() * 100
    o["逆勢曾觸及−3%"] = (adv <= -0.03).mean() * 100
    o["cum 中位%"] = cum.median() * 100; o["cum 平均%"] = cum.mean() * 100
    return o
rows = []
for kind in ["向上突破 確認 (收在上緣之上)", "向上突破 假 (收回線內)", "向下跌破 確認 (收在下緣之下)", "向下跌破 假 (收回線內)"]:
    x = ev[ev.kind == kind].dropna(subset=[f"close_{N}"]); sign = 1 if "向上" in kind else -1
    rows.append(dict(label=f"{kind} → 順向", **probs(x, sign)))
    rows.append(dict(label=f"{kind} → 逆向", **probs(x, -sign)))
rows.append(dict(label="基準: universe 任意日 → 向上", **probs(base, 1)))
rows.append(dict(label="基準: universe 任意日 → 向下", **probs(base, -1)))
rows.append(dict(label="基準: ATR20≥3% 任意日 → 向上", **probs(hi_atr, 1)))
rows.append(dict(label="基準: ATR20≥3% 任意日 → 向下", **probs(hi_atr, -1)))
df = pd.DataFrame(rows)
cols = [c for c in df.columns if c != "label"]
md = [f"# 三角突破後 {N} 日內出現明顯趨勢的機率", "",
      "事件 = 突破日 (T+0, 收斂比 ≤0.8, 2018~2026); 「cum」= T+3 收盤相對 T+0 收盤; 「曾觸及」= T+1~T+3 盤中最高 (或最低) 相對 T+0 收盤; 「日內合計」= T+1~T+3 三天 開→收 相加 (當沖能抓的)。",
      "「順向」= 與突破方向相同 (向上突破看漲、向下跌破看跌); 「逆向」= 相反。基準列是 universe 所有股票日的無條件機率, 供比較。單位 %。", "",
      "| 事件 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
for _, r in df.iterrows(): md.append("| " + r.label + " | " + " | ".join(f"{r[c]:.0f}" if c == "n" else f"{r[c]:.1f}" for c in cols) + " |")
(STATS / "trend_probability.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
