"""畫出最近 N 檔被判為對稱三角的 D1 K 線 (含回歸上下緣) 供人工驗證. 用法: python3 plot_triangles.py [N] [tri_sym|tri_asc|tri_desc]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from config import PROC, STATS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
KIND = sys.argv[2] if len(sys.argv) > 2 else "tri_sym"
L = 20
for f in ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

fl = pd.read_parquet(PROC / "triangle_flags.parquet")
fl = fl[fl[KIND]].sort_values("date", ascending=False).drop_duplicates("stock_id").head(N)
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "name", "open", "high", "low", "close", "volume"])
rows = []
fig, axes = plt.subplots((N + 1) // 2, 2, figsize=(16, 3.6 * ((N + 1) // 2))); axes = axes.ravel()
for ax, (_, r) in zip(axes, fl.iterrows()):
    s = d1[(d1.stock_id == r.stock_id) & (d1.date <= r.date)].sort_values("date").tail(L + 11).reset_index(drop=True)
    i0 = len(s) - 1                      # 今日 (判斷日 = 突破觀察日)
    win = s.iloc[i0 - L:i0]              # 三角視窗 = 昨日往前 20 日
    x = np.arange(len(s))
    up = s.close >= s.open
    ax.bar(x, (s.close - s.open).abs(), bottom=np.minimum(s.open, s.close), width=0.6, color=np.where(up, "#d33", "#2a7"), zorder=3)
    ax.vlines(x, s.low, s.high, color=np.where(up, "#d33", "#2a7"), lw=1, zorder=2)
    xw = x[i0 - L:i0]
    line_h = r.res - r.slope_h * r.pc * (L - np.arange(L))      # 外推到今日的值往回推
    line_l = r.sup - r.slope_l * r.pc * (L - np.arange(L))
    ax.plot(np.append(xw, i0), np.append(line_h, r.res), "b--", lw=1.2); ax.plot(np.append(xw, i0), np.append(line_l, r.sup), "b--", lw=1.2)
    ax.axvspan(i0 - L - 0.5, i0 - 0.5, color="#eef", zorder=0)
    ax.axvline(i0, color="k", lw=0.8, ls=":")
    ax.annotate(f"突破價 {r.res:.1f}", (i0, r.res), xytext=(5, 0), textcoords="offset points", fontsize=8, color="b")
    ax.annotate(f"{r.sup:.1f}", (i0, r.sup), xytext=(5, -8), textcoords="offset points", fontsize=8, color="b")
    today = s.iloc[i0]
    ax.set_title(f"{r.stock_id} {s.name.iloc[-1]}  判斷日 {r.date.date()} | 高點斜率 {r.slope_h*100:.2f}%/日 低點 {r.slope_l*100:+.2f}%/日 | 收斂 {r.contr:.2f} 寬 {r.width*100:.1f}%  (今日 {today.open:.1f}→{today.close:.1f})", fontsize=9)
    ax.set_xticks(x[::5]); ax.set_xticklabels([t.strftime("%m/%d") for t in s.date.iloc[::5]], fontsize=7); ax.grid(alpha=.3)
    rows.append(dict(stock_id=r.stock_id, name=s.name.iloc[-1], date=r.date.date(), slope_h_pct=round(r.slope_h * 100, 3), slope_l_pct=round(r.slope_l * 100, 3), contr=round(r.contr, 2),
                     width_pct=round(r.width * 100, 2), res=round(r.res, 2), sup=round(r.sup, 2), prev_close=r.pc, today_open=today.open, today_high=today.high, today_low=today.low, today_close=today.close))
for ax in axes[len(fl):]: ax.axis("off")
plt.tight_layout(); out = STATS / f"triangles_recent_{KIND}.png"; plt.savefig(out, dpi=110); print(out)
t = pd.DataFrame(rows); t.to_csv(STATS / f"triangles_recent_{KIND}.csv", index=False); print(t.to_string(index=False))
