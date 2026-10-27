"""定案規則的資金曲線與回撤圖.
預設: 整數張、固定資金不複利 (每檔額度 = 資金/10, 張數 = floor(額度 / (區間低×1000)), 最少 1 張; 損益直接累加)。
--compound: 舊版口徑 (小數張、每檔 = 當時資金/10、複利, 對數座標)。
資料: data/processed/orb_history/ (build_history.py)。用法: python3 plot_equity.py [--capital 3000000] [--compound]"""
import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager
from config import PROC, STATS

ap = argparse.ArgumentParser(); ap.add_argument("--capital", type=float, default=3_000_000); ap.add_argument("--compound", action="store_true"); a = ap.parse_args()
for f in ["PingFang TC", "Heiti TC", "Arial Unicode MS"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID, BLUE, RED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#e34948"

D = pd.read_parquet(PROC / "orb_history" / "days.parquet").set_index("date"); T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
cap0 = a.capital / 1e4                                                      # 萬
if a.compound:
    eq = cap0 * (1 + D.ret_cap / 100).cumprod()
    mode = "小數張、每檔 = 當時資金 / 10、複利"
else:
    T["lots_c"] = np.maximum((a.capital / 10 // (T.rl * 1000)).astype(int), 1)
    T["pnl_c"] = T.r_pct / 100 * T.entry_px * T.lots_c * 1000
    pnl = T.groupby("date").pnl_c.sum().reindex(D.index, fill_value=0.0) / 1e4
    eq = cap0 + pnl.cumsum()
    mode = f"整數張、固定 {cap0:,.0f} 萬不複利 (每檔額度 {cap0/10:,.0f} 萬, 張數無條件捨去, 最少 1 張)"
dd_amt = eq - eq.cummax()
dd = (eq / eq.cummax() - 1) * 100 if a.compound else dd_amt / cap0 * 100      # 不複利: 回撤占起始資金 (實際操作資金) 的比例

eps, start, in_dd = [], None, False
for i, (d, v) in enumerate(dd.items()):
    if v < 0 and not in_dd: in_dd, start = True, dd.index[i - 1]
    if in_dd and v == 0: seg = dd[start:d]; eps.append((start, seg.idxmin(), d, seg.min())); in_dd = False
if in_dd: seg = dd[start:]; eps.append((start, seg.idxmin(), dd.index[-1], seg.min()))
top = sorted(eps, key=lambda x: x[3])[:5]

fig, (ax, ad) = plt.subplots(2, 1, figsize=(14, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.08), facecolor=SURF)
for x in (ax, ad):
    x.set_facecolor(SURF); x.grid(color=GRID, lw=1)
    for sp in ["top", "right"]: x.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: x.spines[sp].set_color("#c9c8c2")
    x.tick_params(colors=INK2, labelsize=9)
for (a0, b0, c0, v) in top:
    for x in (ax, ad): x.axvspan(a0, c0, color="#f2d6d6", alpha=0.45, lw=0, zorder=0)
ax.plot(eq.index, eq.values, color=BLUE, lw=2, zorder=3)
if a.compound:
    ax.set_yscale("log"); tk = [v for v in [300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000] if v <= eq.max() * 1.3]
    ax.set_yticks(tk); ax.set_yticklabels([f"{v:,}" for v in tk]); ax.minorticks_off(); ax.set_ylim(eq.min() * 0.92, eq.max() * 1.18)
else:
    ax.set_ylim(cap0 * 0.9, eq.max() * 1.06); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.axhline(cap0, color="#b8b7b1", lw=1, zorder=1)
ax.text(eq.index[0] + pd.Timedelta(days=60), cap0 * 0.985, f"起始 {cap0:,.0f} 萬", color=INK2, fontsize=9, va="top")
ax.annotate(f"{eq.iloc[-1]:,.0f} 萬", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=10, va="center")
split = pd.Timestamp("2024-01-01")
for x in (ax, ad): x.axvline(split, color="#8a8984", lw=1, zorder=2)
yt = ax.get_ylim()[1] * (0.97 if a.compound else 1) - (0 if a.compound else (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02)
ax.text(split, yt, " 驗證期 →", color=INK2, fontsize=9, va="top"); ax.text(split, yt, "← 探索期 ", color=INK2, fontsize=9, va="top", ha="right")
lng = max(eps, key=lambda x: len(dd[x[0]:x[2]]))
ax.annotate(f"停滯 {len(dd[lng[0]:lng[2]]) - 1} 個交易日\n({lng[0].date()} ~ {lng[2].date()}, 最深 {lng[3]:.1f}%)", (lng[1], eq[lng[1]]), xytext=(0, -46), textcoords="offset points",
            ha="center", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color="#8a8984", lw=1))
ax.set_ylabel("資金 (萬元" + (", 對數座標)" if a.compound else ")"), color=INK2, fontsize=10)
mdd_i = dd.idxmin()
yv = dd if a.compound else dd_amt
ad.fill_between(yv.index, yv.values, 0, color=RED, alpha=0.35, lw=0, zorder=2); ad.plot(yv.index, yv.values, color=RED, lw=1, zorder=3)
txt = f"最大回撤 {dd.min():.1f}%\n{mdd_i.date()}" if a.compound else f"最大回撤 {dd_amt[mdd_i]:,.0f} 萬 (起始資金的 {dd.min():.1f}%)\n{mdd_i.date()}"
ad.annotate(txt, (mdd_i, yv[mdd_i]), xytext=(10, 4), textcoords="offset points", fontsize=9, color=INK)
ad.set_ylabel("回撤 (%)" if a.compound else "回撤 (萬元)", color=INK2, fontsize=10); ad.set_ylim(yv.min() * 1.35, abs(yv.min()) * 0.05)
# 年度: 不複利顯示 年獲利 (萬) 與占起始資金 %
yp = eq.groupby(eq.index.year).last() - eq.groupby(eq.index.year).first().shift(-0)
first_by_year = eq.groupby(eq.index.year).first(); prev_end = eq.groupby(eq.index.year).last().shift(1).fillna(cap0)
gain = eq.groupby(eq.index.year).last() - prev_end
lab = {y: (f"{y}\n{(eq.groupby(eq.index.year).last()[y] / prev_end[y] - 1) * 100:+.0f}%" if a.compound else f"{y}\n{g:+,.0f} 萬 ({g / cap0 * 100:+.0f}%)") for y, g in gain.items()}
ad.xaxis.set_major_locator(mdates.YearLocator()); ad.xaxis.set_major_formatter(plt.NullFormatter())
mids = [pd.Timestamp(f"{y}-07-01") if y < eq.index[-1].year else eq.index[eq.index.year == y][len(eq.index[eq.index.year == y]) // 2] for y in gain.index]
ad.set_xticks(mids, minor=True); ad.set_xticklabels([lab[y] for y in gain.index], minor=True, fontsize=9, color=INK2)
ad.tick_params(axis="x", which="minor", length=0, pad=6); ad.tick_params(axis="x", which="major", length=4)
for x in (ax, ad): x.set_xlim(eq.index[0] - pd.Timedelta(days=20), eq.index[-1] + pd.Timedelta(days=110))
day_ret = eq.diff().fillna(eq.iloc[0] - cap0) / cap0 * 100 if not a.compound else D.ret_cap
sh = day_ret.mean() / day_ret.std() * np.sqrt(245)
fig.subplots_adjust(top=0.86)
fig.text(0.125, 0.955, f"定案規則回測資金曲線 ({eq.index[0].date()} ~ {eq.index[-1].date()})", fontsize=14, color=INK, va="bottom")
fig.text(0.125, 0.94, f"跳空≥2% ∩ ATR≥3% ∩ 開盤 5 分量≥日均量 5%，跌破開盤 5 分區間低放空，區間高停損，否則收盤回補。{mode}。含成本 0.23% 與 +8% 強制回補 / 鎖漲停罰金。\n"
         f"{len(T):,} 筆　累計損益 {eq.iloc[-1] - cap0:,.0f} 萬　Sharpe {sh:.2f}　最大回撤 {dd_amt.min():,.0f} 萬 ({dd.min():.1f}%{'' if a.compound else ' ，占起始資金'})　淺紅底 = 最深 5 次回撤期間。理想成交價，未計進場與停損的延遲。",
         fontsize=9, color=INK2, va="top", linespacing=1.6)
out = STATS / ("equity_curve_compound.png" if a.compound else "equity_curve.png"); fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURF); print(out)
print(f"期末 {eq.iloc[-1]:,.0f} 萬  累計損益 {eq.iloc[-1] - cap0:,.0f} 萬  年均 {(eq.iloc[-1] - cap0) / len(eq) * 245:,.0f} 萬  Sharpe {sh:.2f}  MDD {dd.min():.2f}% ({dd_amt.min():,.0f} 萬)")
print("各年損益:", "  ".join(f"{y}: {g:+,.0f} 萬" for y, g in gain.items()))
