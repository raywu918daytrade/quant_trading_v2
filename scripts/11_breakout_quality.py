"""突破日品質切分: 量比 (突破日量/前20日均量) × 收盤離線距離, 看 3 日內順向/逆向趨勢機率與幅度. -> stats/breakout_quality.md"""
import numpy as np, pandas as pd
from config import PROC, STATS
N = 3
ev = pd.read_parquet(PROC / "triangle_events.parquet")
vol = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "volume", "mkt_gap"]).rename(columns={"mkt_gap": "mkt_gap_d1"})
ev = ev.merge(vol, on=["stock_id", "date"], how="left")
ev["vr"] = ev.volume / ev.vol20
ev["dist"] = np.where(ev.kind.str.startswith("向上"), ev.close / ev.res - 1, ev.sup / ev.close - 1)   # 收盤超出線外的幅度 (正)
ev["dist"] = np.where(ev.kind.str.startswith("向上"), ev.close / ev.res - 1, 1 - ev.close / ev.sup)
ev = ev[ev.kind.str.contains("確認")].dropna(subset=[f"close_{N}"]).copy()

def stats(x, sign):
    c0 = x.close
    cum = sign * (x[f"close_{N}"] / c0 - 1)
    hi3 = pd.concat([x[f"high_{k}"] for k in range(1, N + 1)], axis=1).max(axis=1); lo3 = pd.concat([x[f"low_{k}"] for k in range(1, N + 1)], axis=1).min(axis=1)
    ext = (hi3 / c0 - 1) if sign > 0 else (1 - lo3 / c0); adv = (lo3 / c0 - 1) if sign > 0 else (1 - hi3 / c0)
    intra = sign * sum(x[f"intra_{k}"] for k in range(1, N + 1))
    intra1 = sign * x["intra_1"]
    return dict(n=len(x), p_fwd3=(cum >= 0.03).mean() * 100, p_rev3=(cum <= -0.03).mean() * 100, p_touch3=(ext >= 0.03).mean() * 100, p_adv3=(adv <= -0.03).mean() * 100,
                cum_mean=cum.mean() * 100, cum_med=cum.median() * 100, intra3=intra.mean() * 100, intra3_win=(intra > 0).mean() * 100, intra1=intra1.mean() * 100, intra1_win=(intra1 > 0).mean() * 100)
C = [("切分", "label"), ("n", "n"), ("順向 3日≥3%", "p_fwd3"), ("逆向 3日≥3%", "p_rev3"), ("順向曾觸及3%", "p_touch3"), ("逆向曾觸及3%", "p_adv3"), ("3日累積平均%", "cum_mean"), ("中位", "cum_med"),
     ("T+1~3 日內合計%", "intra3"), ("合計>0 %", "intra3_win"), ("T+1 日內%", "intra1"), ("T+1 日內>0 %", "intra1_win")]
def fmt(v): return f"{v:.0f}" if isinstance(v, (int, np.integer)) else (f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows):
    out = "| " + " | ".join(c[0] for c in C) + " |\n|" + "---|" * len(C) + "\n"
    for r in rows: out += "| " + " | ".join(fmt(r.get(c[1], "")) for c in C) + " |\n"
    return out
md = ["# 突破日品質 × 3 日內趨勢", "", "事件 = 三角突破確認日 (收在線外), 收斂比 ≤0.8, 2018~2026。量比 = 突破日成交量 / 前 20 日均量；離線距離 = 收盤超出上緣 (或下緣) 的 %。",
      "「順向」= 突破方向；機率欄位 = 3 日累積收→收 ≥3% (或 ≤−3%) 的比例；「日內合計」= T+1~T+3 開→收相加 (順向為正), 是當沖能抓到的部分 (未扣成本)。", ""]
for kind, sign in [("向上突破 確認 (收在上緣之上)", 1), ("向下跌破 確認 (收在下緣之下)", -1)]:
    x = ev[ev.kind == kind]
    rows = [dict(label="全部", **stats(x, sign))]
    for lab, m in [("量比 <1", x.vr < 1), ("量比 1~2", (x.vr >= 1) & (x.vr < 2)), ("量比 2~3", (x.vr >= 2) & (x.vr < 3)), ("量比 ≥3", x.vr >= 3), ("量比 ≥2", x.vr >= 2)]:
        rows.append(dict(label=lab, **stats(x[m], sign)))
    for lab, m in [("離線 <1%", x.dist < 0.01), ("離線 1~3%", (x.dist >= 0.01) & (x.dist < 0.03)), ("離線 ≥3%", x.dist >= 0.03), ("離線 ≥2%", x.dist >= 0.02)]:
        rows.append(dict(label=lab, **stats(x[m], sign)))
    for lab, m in [("量比 ≥2 ∩ 離線 ≥2%", (x.vr >= 2) & (x.dist >= 0.02)), ("量比 ≥2 ∩ 離線 <2%", (x.vr >= 2) & (x.dist < 0.02)), ("量比 <2 ∩ 離線 ≥2%", (x.vr < 2) & (x.dist >= 0.02)), ("量比 <2 ∩ 離線 <2%", (x.vr < 2) & (x.dist < 0.02))]:
        rows.append(dict(label=lab, **stats(x[m], sign)))
    for lab, m in [("突破日跳空同向 (|gap|≥1%)", sign * x.gap >= 0.01), ("突破日非跳空", sign * x.gap < 0.01), ("大盤當日跳空同向", sign * x.mkt_gap_d1 > 0), ("大盤當日跳空反向", sign * x.mkt_gap_d1 <= 0),
                   ("突破日漲跌 ≥5% (大 K)", sign * x.ret >= 0.05), ("突破日漲跌 <5%", sign * x.ret < 0.05)]:
        rows.append(dict(label=lab, **stats(x[m], sign)))
    md += [f"## {kind} (n={len(x)})", "", f"量比分布: 中位 {x.vr.median():.1f}, 75% {x.vr.quantile(.75):.1f}；離線距離 中位 {x.dist.median()*100:.1f}%, 75% {x.dist.quantile(.75)*100:.1f}%。", "", table(rows), ""]
md += ["對照 (trend_probability.md): 高波動股 (ATR20≥3%) 任意日 3 日累積 ≥3% 的機率 向上 26% / 向下 27%, 曾觸及 3% 56% / 49%。"]
(STATS / "breakout_quality.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
