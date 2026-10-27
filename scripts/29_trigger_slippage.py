"""觸價單 (預掛) 的實際滑價: 進場比區間低再差 k 檔、停損比區間高再差 k 檔 (取代原本 +0.1%)。收盤競價出場無滑價。 -> 追加到 stats/delay_drawdown.md"""
import numpy as np, pandas as pd
import orb_live as L
from config import PROC, STATS, EXPLORE_END
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
D = pd.read_parquet(PROC / "orb_history" / "days.parquet"); ALL = pd.DatetimeIndex(D.date)
stopped = T.reason.str.startswith("停損")
rows = []
for k in [0, 1, 2, 3]:
    ent_slip = k * T.rl.map(L.tick_size) / T.entry_px * 100
    stop_slip = np.where(stopped, k * T.rh.map(L.tick_size) / T.entry_px * 100 - 0.1 * T.exit_px / T.entry_px, 0)   # 換掉原本的 0.1%
    r = T.r_pct - ent_slip - stop_slip
    pn = (r / 100 * T.entry_px * T.lots * 1000).groupby(T.date).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); dd = eq - eq.cummax()
    uw = (dd < 0).astype(int); mon = pn.groupby(pn.index.to_period("M")).sum()
    rows.append(f"| 觸價單, 進場與停損各滑 {k} 檔 | {r.mean():.2f} | {pn.sum()/len(pn)*245:,.0f} | {pn.mean()/pn.std()*np.sqrt(245):.2f} | {dd.min():,.0f} | {-dd.min()/(pn.sum()/len(pn)*245):.2f} | {pn.min():.1f} | {mon.min():.1f} | {(mon<0).mean()*100:.0f} | {int(uw.groupby((uw==0).cumsum()).sum().max())} |")
print(f"平均一檔占價格: 區間低 {(T.rl.map(L.tick_size)/T.rl*100).mean():.3f}%")
md = ["", "## 觸價單預掛 + 滑價 (最實際的執行方式)", "", f"一檔 (跳動單位) 平均約占股價 {(T.rl.map(L.tick_size)/T.rl*100).mean():.2f}%。", "",
      "| 情境 | 每筆% | 年均(萬) | Sharpe | 最大回撤(萬) | 回撤÷年均 | 最差日(萬) | 最差月(萬) | 負報酬月% | 最長停滯(交易日) |", "|---|---|---|---|---|---|---|---|---|---|"] + rows
with open(STATS / "delay_drawdown.md", "a", encoding="utf-8") as fh: fh.write("\n".join(md) + "\n")
print("\n".join(md))
