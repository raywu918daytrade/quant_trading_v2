"""畫最近 N 檔 (擺盪點版三角) 供驗證: 標出 4 個擺盪點、上下緣、突破價. 用法: python3 plot_triangles2.py [N] [min_desc_h] [min_asc_l]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from config import PROC, STATS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MINH = int(sys.argv[2]) if len(sys.argv) > 2 else 2; MINL = int(sys.argv[3]) if len(sys.argv) > 3 else 2; MAXC = float(sys.argv[4]) if len(sys.argv) > 4 else 9
for f in ["PingFang TC", "Heiti TC", "Arial Unicode MS"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
tri = pd.read_parquet(PROC / "triangle_blocks.parquet")
tri = tri[(tri.n_desc_h >= MINH) & (tri.n_asc_l >= MINL) & (tri.contr <= MAXC)].sort_values("date", ascending=False).drop_duplicates("stock_id").head(N)
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "open", "high", "low", "close"])
rows = []
fig, axes = plt.subplots((N + 1) // 2, 2, figsize=(16, 3.8 * ((N + 1) // 2))); axes = axes.ravel()
for ax, (_, r) in zip(axes, tri.iterrows()):
    full = d1[d1.stock_id == r.stock_id].sort_values("date").reset_index(drop=True)
    t = int(full.index[full.date == r.date][0])
    a = max(0, int(min(r.iH1, r.iL1)) - 8); s = full.iloc[a:t + 1].reset_index(drop=True); off = a
    x = np.arange(len(s)); up = s.close >= s.open
    ax.bar(x, (s.close - s.open).abs(), bottom=np.minimum(s.open, s.close), width=0.6, color=np.where(up, "#d33", "#2a7"), zorder=3)
    ax.vlines(x, s.low, s.high, color=np.where(up, "#d33", "#2a7"), lw=1, zorder=2)
    xh = np.array([r.iH1, r.iH2, t]) - off; xl = np.array([r.iL1, r.iL2, t]) - off
    ax.plot(xh, [r.H1, r.H2, r.res], "b-", lw=1.3); ax.plot(xl, [r.L1, r.L2, r.sup], "b-", lw=1.3)
    ax.plot(xh[:2], [r.H1, r.H2], "bv", ms=8, zorder=5); ax.plot(xl[:2], [r.L1, r.L2], "b^", ms=8, zorder=5)
    ax.axvline(t - off, color="k", lw=0.8, ls=":")
    ax.annotate(f"突破 {r.res:.1f}", (t - off, r.res), xytext=(5, 0), textcoords="offset points", fontsize=8, color="b")
    ax.annotate(f"{r.sup:.1f}", (t - off, r.sup), xytext=(5, -8), textcoords="offset points", fontsize=8, color="b")
    today = s.iloc[-1]
    ax.set_title(f"{r.stock_id} {r['name']}  判斷日 {r.date.date()} | 跨度 {int(r.span)} 日 | 下降高點 {int(r.n_desc_h)} 個, 上升低點 {int(r.n_asc_l)} 個 | 收斂比 {r.contr:.2f} 寬 {r.width*100:.1f}% | 今日 {today.open:.1f}→{today.close:.1f} (高 {today.high:.1f})", fontsize=8.5)
    ax.set_xticks(x[::5]); ax.set_xticklabels([q.strftime("%m/%d") for q in s.date.iloc[::5]], fontsize=7); ax.grid(alpha=.3)
    rows.append(dict(stock_id=r.stock_id, name=r["name"], date=r.date.date(), span=int(r.span), n_desc_h=int(r.n_desc_h), n_asc_l=int(r.n_asc_l), H1=r.H1, H2=r.H2, L1=r.L1, L2=r.L2,
                     res=round(r.res, 2), sup=round(r.sup, 2), contr=round(r.contr, 2), width_pct=round(r.width * 100, 2), prev_close=r.pc, today_open=today.open, today_high=today.high, today_low=today.low, today_close=today.close))
for ax in axes[len(tri):]: ax.axis("off")
plt.tight_layout(); out = STATS / f"triangles2_recent_h{MINH}_l{MINL}_c{MAXC}.png"; plt.savefig(out, dpi=110); print(out)
pd.DataFrame(rows).to_csv(STATS / f"triangles2_recent_h{MINH}_l{MINL}_c{MAXC}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
