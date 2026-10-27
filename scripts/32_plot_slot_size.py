"""一檔 ≤0.15% 三種配置 (30 萬×10 / 60 萬×5 / 100 萬×3) 的資金曲線與回撤, 滑價 0/1/2 檔小圖組。 -> stats/slot_size_equity.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager
import orb_live as L
from config import PROC, STATS, COST
P = L.P
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"]); f = f[f.univ & (f.date >= "2020-01-01")]
yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
s = f.merge(pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms]), on=["stock_id", "date"]); s = s[s.auc_c.notna()]
s["lu"] = s.pc * 1.1; s["fpx"] = s.pc * 1.08
s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
      & (s.o5_h < s.lu * 0.999) & (s.o5_h < s.fpx) & (s.dn5_px < s.fpx) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
s["tick_l"] = s.o5_l.map(L.tick_size); s["tick_h"] = s.o5_h.map(L.tick_size); s = s[s.tick_l / s.o5_l * 100 <= 0.15]
ALL = pd.DatetimeIndex(np.sort(f.date.unique()))
def curve(slot, maxn, k):
    x = s.sort_values(["date", "dn5_t", "stock_id"]).groupby("date").head(maxn)
    lots = np.maximum((slot // (x.o5_l * 1000)).astype(int), 1); ent = x.dn5_px - k * x.tick_l
    raw = np.maximum(x.dn5_stop_hi_px, x.dn5_stop_hi_open); st = raw * (1 + P["STOP_SLIP"]) if k == 0 else raw + k * x.tick_h
    ex = st.where(x.dn5_stop_hi_px.notna(), x.auc_c); hi = x.dn5_hi_after.where(x.dn5_stop_hi_px.isna(), np.minimum(x.dn5_hi_after, st)); forced = hi >= x.fpx
    exf = np.where(forced, x.fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= x.lu * 0.999)
    r = -(exf / ent - 1) - COST - np.where(locked, P["LOCK"], 0)
    pn = (r * ent * lots * 1000).groupby(x.date).sum().reindex(ALL, fill_value=0) / 1e4
    return 300 + pn.cumsum()
CONF = [((3e5, 10), "每檔 30 萬 × 最多 10 檔", "#2a78d6"), ((6e5, 5), "每檔 60 萬 × 最多 5 檔", "#eb6834"), ((1e6, 3), "每檔 100 萬 × 最多 3 檔", "#1baf7a")]
E = {(k, lab): curve(*cfg, k) for k in [0, 1, 2] for cfg, lab, _ in CONF}

for fn in ["PingFang TC", "Heiti TC", "Arial Unicode MS"]:
    if any(fn == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = fn; break
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
fig, axs = plt.subplots(2, 3, figsize=(17, 8.6), sharex=True, sharey="row", gridspec_kw=dict(height_ratios=[2.6, 1.3], hspace=0.08, wspace=0.06), facecolor=SURF)
ymax = max(e.max() for e in E.values()); ymin = min(e.min() for e in E.values())
dmin = min((e - e.cummax()).min() for e in E.values())
for j, k in enumerate([0, 1, 2]):
    ax, ad = axs[0, j], axs[1, j]
    for x in (ax, ad):
        x.set_facecolor(SURF); x.grid(color=GRID, lw=1)
        for sp in ["top", "right"]: x.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: x.spines[sp].set_color("#c9c8c2")
        x.tick_params(colors=INK2, labelsize=8.5)
    ends = []
    for cfg, lab, col in CONF:
        eq = E[(k, lab)]; dd = eq - eq.cummax()
        ax.plot(eq.index, eq.values, color=col, lw=2, label=lab); ad.plot(dd.index, dd.values, color=col, lw=1.1)
        ends.append((eq.iloc[-1], lab, col, dd.min()))
    last = -1e9
    for v, lab, col, mdd in sorted(ends):
        y = max(v, last + (ymax - ymin) * 0.07); last = y
        ax.annotate(f"{v - 300:+,.0f} 萬", (eq.index[-1], v), xytext=(eq.index[-1] + pd.Timedelta(days=25), y), textcoords="data", fontsize=9, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.axhline(300, color="#b8b7b1", lw=1)
    ax.set_title(f"成交比區間差 {k} 檔", fontsize=11.5, color=INK, loc="left")
    mdds = "  ".join(f"{m:,.0f}" for _, _, _, m in sorted(ends, key=lambda z: [c[1] for c in CONF].index(z[1])))
    ad.text(0.02, 0.06, f"最大回撤 (萬): {mdds}", transform=ad.transAxes, fontsize=8.5, color=INK2)
    ad.xaxis.set_major_locator(mdates.YearLocator(2)); ad.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for x in (ax, ad): x.set_xlim(ALL[0] - pd.Timedelta(days=20), ALL[-1] + pd.Timedelta(days=330))
axs[0, 0].set_ylabel("資金 (萬元)", color=INK2); axs[1, 0].set_ylabel("回撤 (萬元)", color=INK2)
axs[0, 0].set_ylim(min(ymin, 290) - 10, ymax * 1.05); axs[1, 0].set_ylim(dmin * 1.15, 3)
axs[0, 0].legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
fig.subplots_adjust(top=0.86)
fig.text(0.125, 0.955, "一檔 ≤0.15% 規則: 三種部位配置的資金曲線與回撤 (總資金 300 萬, 2020-01 ~ 2026-10)", fontsize=14, color=INK, va="bottom")
fig.text(0.125, 0.94, "先跌破先做, 名額遞補, 整數張。曲線右側為累計損益。回撤列的三個數字依序為 30 萬×10 / 60 萬×5 / 100 萬×3。", fontsize=9.5, color=INK2, va="top")
out = STATS / "slot_size_equity.png"; fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURF); print(out)
for k in [0, 1, 2]: print(k, {lab: round(E[(k, lab)].iloc[-1] - 300) for _, lab, _ in CONF})
