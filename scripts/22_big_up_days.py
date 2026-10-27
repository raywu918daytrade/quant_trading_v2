"""大盤大漲日 (0050 當日開→收) 策略的單日損益分布與最大虧損 (固定 300 萬、每檔 30 萬整數張). -> stats/big_up_days.md"""
import numpy as np, pandas as pd
from config import PROC, STATS
m = pd.read_parquet(PROC / "etf0050_m1.parquet")
e = m.groupby("date").agg(o=("open", "first"), c=("close", "last"), h=("high", "max"), l=("low", "min"))
e["e_intra"] = (e.c / e.o - 1) * 100; e["e_hi"] = (e.h / e.o - 1) * 100
D = pd.read_parquet(PROC / "orb_history" / "days.parquet").set_index("date").join(e[["e_intra", "e_hi"]])
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
D["pnl_w"] = D.pnl / 1e4
worst_trade = T.groupby("date").pnl.min() / 1e4
D["worst_trade_w"] = worst_trade
tr_days = D[D.n_trades > 0]

def fmt(v, nd=1): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
def table(df, cols):
    out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows(): out += "| " + " | ".join(fmt(r[c[1]], c[2] if len(c) > 2 else 1) for c in cols) + " |\n"
    return out
rows = []
for lab, mk in [("全部有交易的日子", tr_days.e_intra > -99), ("0050 開→收 ≥ +0.5%", tr_days.e_intra >= 0.5), ("≥ +1%", tr_days.e_intra >= 1), ("≥ +1.5%", tr_days.e_intra >= 1.5), ("≥ +2%", tr_days.e_intra >= 2),
                ("0050 開→收 ≤ −1% (對照: 大跌日)", tr_days.e_intra <= -1)]:
    x = tr_days[mk]
    rows.append(dict(label=lab, days=len(x), avg=x.pnl_w.mean(), lose=(x.pnl_w < 0).mean() * 100, p10=x.pnl_w.quantile(0.1), p5=x.pnl_w.quantile(0.05), worst=x.pnl_w.min(),
                     worst_date=str(x.pnl_w.idxmin().date()) if len(x) else "", total=x.pnl_w.sum(), wt=x.worst_trade_w.min(), stops=x.n_stop.sum() / x.n_trades.sum() * 100))
R = pd.DataFrame(rows)
md = ["# 大盤大漲日的單日虧損 (固定 300 萬、每檔 30 萬整數張)", "",
      f"大漲日以 0050 當日開盤→收盤定義 (策略只抱日內)。資料 2020-01 ~ {D.index.max().date()}, 有交易的日子 {len(tr_days)} 天。", "",
      table(R, [("條件", "label"), ("天數", "days", 0), ("平均單日損益(萬)", "avg"), ("虧損日比例%", "lose", 0), ("最差 10% 分位(萬)", "p10"), ("最差 5% 分位(萬)", "p5"),
                ("最大單日虧損(萬)", "worst"), ("發生日", "worst_date"), ("這些日子合計(萬)", "total", 0), ("單筆最大虧損(萬)", "wt"), ("停損率%", "stops", 0)]), ""]
w = tr_days.sort_values("pnl_w").head(15).reset_index()
w["date"] = w.date.dt.date.astype(str); w["ret_int"] = w.pnl_w / 300 * 100
md += ["## 最差 15 個交易日", "", table(w, [("日期", "date"), ("當日損益(萬)", "pnl_w"), ("占 300 萬%", "ret_int", 2), ("0050 開→收%", "e_intra", 2), ("0050 盤中最高%", "e_hi", 2), ("進場檔數", "n_trades", 0), ("停損檔數", "n_stop", 0), ("最差單筆(萬)", "worst_trade_w")]), ""]
n_up = (w.e_intra >= 1).sum()
md += [f"最差 15 天中, 0050 當日開→收 ≥ +1% 的有 {n_up} 天, ≥ +0.5% 的有 {(w.e_intra >= 0.5).sum()} 天。", ""]
# 依 0050 開→收 分段的平均損益
b = pd.cut(tr_days.e_intra, [-99, -1, -0.5, 0, 0.5, 1, 1.5, 2, 99], labels=["≤−1%", "−1~−0.5%", "−0.5~0%", "0~+0.5%", "+0.5~+1%", "+1~+1.5%", "+1.5~+2%", "≥+2%"])
g = tr_days.groupby(b, observed=True).agg(days=("pnl_w", "size"), avg=("pnl_w", "mean"), worst=("pnl_w", "min"), lose=("pnl_w", lambda x: (x < 0).mean() * 100)).reset_index()
g = g.rename(columns={"e_intra": "seg"}); g["seg"] = g.seg.astype(str)
md += ["## 依 0050 當日開→收 分段", "", table(g, [("0050 開→收", "seg"), ("天數", "days", 0), ("平均(萬)", "avg"), ("最差(萬)", "worst"), ("虧損日比例%", "lose", 0)])]
(STATS / "big_up_days.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))
