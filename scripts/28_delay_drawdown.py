"""實盤延遲對報酬與回撤的影響 (定案規則 6,526 筆, 1 分 K 逐分鐘重新模擬; 固定 300 萬、每檔 30 萬整數張)。
進場: 理想 = min(區間低, 跌破那分鐘開盤); "c" = 跌破那分鐘收盤; d 分 = 跌破後第 d 分鐘開盤 (若這段期間已觸及停損價或成交價 ≥ 區間高 → 放棄這筆)。
停損: 理想 = max(區間高, 觸發那分鐘開盤)+0.1%; "c" = 觸發那分鐘收盤+0.1%; d 分 = 觸發後第 d 分鐘開盤+0.1%。
延遲回補遇到整分鐘鎖漲停 → 等到第一根有低於漲停成交的分鐘以漲停價成交; 鎖到收盤 → 漲停價回補並再扣 7% (標借)。
-> stats/delay_drawdown.md, stats/delay_equity.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager
import orb_live as L
from config import PROC, STATS, COST, EXPLORE_END
SL, LOCK = 0.001, 0.07
T = pd.read_parquet(PROC / "orb_history" / "trades.parquet")
f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "pc"]); T = T.merge(f, on=["stock_id", "date"]).reset_index(drop=True)
T["lu"] = T.pc.map(L.limit_up); T["lu10"] = T.pc * 1.1; T["fpx"] = T.pc * 1.08
SC = [("理想 (目前回測)", 0, 0), ("同一分鐘收盤成交 (約 10~60 秒延遲)", "c", "c"), ("延遲 1 分鐘", 1, 1), ("延遲 2 分鐘", 2, 2), ("延遲 5 分鐘", 5, 5),
      ("只有進場延遲 1 分", 1, 0), ("只有停損延遲 1 分", 0, 1), ("只有停損延遲 2 分", 0, 2), ("只有停損延遲 5 分", 0, 5)]
R = {s[0]: np.full(len(T), np.nan) for s in SC}; LK = {s[0]: np.zeros(len(T), bool) for s in SC}

def first_idx(t, start_i, tmin):
    k = start_i
    while k < len(t) and t[k] < tmin: k += 1
    return k

for ym, grp in T.groupby(T.date.dt.strftime("%Y_%m")):
    m = L.load_m1_month(ym).merge(grp[["stock_id", "date"]], on=["stock_id", "date"]).sort_values(["stock_id", "date", "t"])
    bars = {k: (g.t.values, g.open.values, g.high.values, g.low.values, g.close.values) for k, g in m.groupby(["stock_id", "date"], sort=False)}
    for i, r in grp.iterrows():
        bb = bars.get((r.stock_id, r.date))
        if bb is None: continue
        t, o, h, lo, c = bb
        ie = np.searchsorted(t, r.entry_t)
        if ie >= len(t) or t[ie] != r.entry_t: continue
        for name, de, ds in SC:
            # ---- 進場 ----
            if de == 0: ent, js = min(r.rl, o[ie]), ie + 1
            elif de == "c": ent, js = c[ie], ie + 1
            else:
                k = first_idx(t, ie, t[ie] + de)
                if k >= len(t) or h[ie:k].max() > r.rh or o[k] >= r.rh: continue          # 等待期間已碰停損 → 放棄
                ent, js = o[k], k                                                       # 以該分鐘開盤進場, 該分鐘起算停損
            if ent >= r.fpx: continue
            # ---- 停損 / 收盤 ----
            px, locked = np.nan, False
            for j in range(js, len(t)):
                if h[j] > r.rh:
                    if ds == 0: px = max(r.rh, o[j]) * (1 + SL)
                    elif ds == "c": px = c[j] * (1 + SL)
                    else:
                        k2 = first_idx(t, j, t[j] + ds)
                        px = o[k2] * (1 + SL) if k2 < len(t) else c[-1] * (1 + SL)
                        kk = k2
                        if kk < len(t) and lo[kk] >= r.lu - 1e-6:                     # 送單時整分鐘鎖漲停
                            while kk < len(t) and lo[kk] >= r.lu - 1e-6: kk += 1
                            if kk < len(t): px = r.lu
                            else: px, locked = r.lu, True
                    break
            if np.isnan(px):
                px = c[-1]; locked = px >= r.lu10 * 0.999
            R[name][i] = (-(px / ent - 1) - COST - (LOCK if locked else 0)) * 100; LK[name][i] = locked
    if ym.endswith("12"): print(ym, flush=True)

D = pd.read_parquet(PROC / "orb_history" / "days.parquet"); ALL = pd.DatetimeIndex(D.date)
print("理想 vs 歷史 r_pct 最大差:", float(np.nanmax(np.abs(R[SC[0][0]] - T.r_pct.values))))
rows, curves = [], {}
for name, de, ds in SC:
    r = pd.Series(R[name]); ok = r.notna()
    pnl = (r / 100 * T.entry_px * T.lots * 1000)[ok]                                    # 金額用原本的張數
    pn = pnl.groupby(T.date[ok]).sum().reindex(ALL, fill_value=0) / 1e4; eq = 300 + pn.cumsum(); dd = eq - eq.cummax()
    uw = (dd < 0).astype(int); longest = int(uw.groupby((uw == 0).cumsum()).sum().max())
    mon = pn.groupby(pn.index.to_period("M")).sum()
    rows.append(dict(label=name, n=int(ok.sum()), per=r[ok].mean(), win=(r[ok] > 0).mean() * 100, per_year=pn.sum() / len(pn) * 245, sharpe=pn.mean() / pn.std() * np.sqrt(245),
                     mdd=dd.min(), ratio=-dd.min() / (pn.sum() / len(pn) * 245), worst=pn.min(), worst_m=mon.min(), neg_m=(mon < 0).mean() * 100, longest=longest,
                     locked=int(LK[name].sum()), ex=pn[pn.index <= EXPLORE_END].sum() / (pn.index <= EXPLORE_END).sum() * 245, va=pn[pn.index > EXPLORE_END].sum() / (pn.index > EXPLORE_END).sum() * 245))
    curves[name] = eq
def fmt(v, nd=2): return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
cols = [("情境", "label"), ("筆數", "n", 0), ("每筆%", "per"), ("勝率", "win", 1), ("年均(萬)", "per_year", 0), ("Sharpe", "sharpe"), ("最大回撤(萬)", "mdd", 0), ("回撤÷年均", "ratio"),
        ("最差日(萬)", "worst", 1), ("最差月(萬)", "worst_m", 1), ("負報酬月%", "neg_m", 0), ("最長停滯(交易日)", "longest", 0), ("鎖漲停回補不了", "locked", 0), ("探索期年均", "ex", 0), ("驗證期年均", "va", 0)]
md = ["# 實盤延遲對報酬與回撤的影響", "", f"定案規則 2020-01 ~ {T.date.max().date()} 的 {len(T):,} 筆, 1 分 K 逐分鐘重新模擬。固定 300 萬、每檔 30 萬 (張數不變)。", "",
      "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
for r in rows: md.append("| " + " | ".join(fmt(r.get(c[1]), c[2] if len(c) > 2 else 2) for c in cols) + " |")
md += ["", "延遲進場時, 若等待期間已碰到停損價就放棄該筆 (所以筆數變少)。延遲停損遇到整分鐘鎖漲停, 等到打開才以漲停價回補; 鎖到收盤則以漲停價回補並扣 7%。"]
(STATS / "delay_drawdown.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))

# ---- 圖 ----
for fn in ["PingFang TC", "Heiti TC", "Arial Unicode MS"]:
    if any(fn == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = fn; break
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
show = [("理想 (目前回測)", "#2a78d6"), ("同一分鐘收盤成交 (約 10~60 秒延遲)", "#eb6834"), ("延遲 1 分鐘", "#1baf7a"), ("延遲 2 分鐘", "#eda100")]
fig, (ax, ad) = plt.subplots(2, 1, figsize=(14, 8.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.6], hspace=0.08), facecolor=SURF)
for x in (ax, ad):
    x.set_facecolor(SURF); x.grid(color=GRID, lw=1)
    for sp in ["top", "right"]: x.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: x.spines[sp].set_color("#c9c8c2")
    x.tick_params(colors=INK2, labelsize=9)
ends = sorted([(curves[n].iloc[-1], n) for n, _ in show])
lab_y, last = {}, -1e9
for v, n in ends:                                   # 期末標籤往上錯開, 避免重疊
    y = max(v, last + 22); lab_y[n] = y; last = y
for name, col in show:
    eq = curves[name]; ax.plot(eq.index, eq.values, color=col, lw=2, label=name)
    ax.annotate(f"{eq.iloc[-1]:,.0f} 萬", (eq.index[-1], eq.iloc[-1]), xytext=(eq.index[-1] + pd.Timedelta(days=12), lab_y[name]), textcoords="data", fontsize=9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8) if abs(lab_y[name] - eq.iloc[-1]) > 1 else None)
    dd = eq - eq.cummax(); ad.plot(dd.index, dd.values, color=col, lw=1.2)
ax.axhline(300, color="#b8b7b1", lw=1); ax.set_ylabel("資金 (萬元)", color=INK2); ad.set_ylabel("回撤 (萬元)", color=INK2)
ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
ad.xaxis.set_major_locator(mdates.YearLocator()); ad.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for x in (ax, ad): x.set_xlim(ALL[0] - pd.Timedelta(days=20), ALL[-1] + pd.Timedelta(days=130))
fig.subplots_adjust(top=0.9)
fig.text(0.125, 0.955, "實盤延遲對資金曲線與回撤的影響 (固定 300 萬、每檔 30 萬整數張)", fontsize=14, color=INK, va="bottom")
fig.text(0.125, 0.94, "理想 = 跌破 / 突破那一刻成交；同一分鐘收盤 ≈ 程式輪詢 10~60 秒延遲；延遲 1 / 2 分 = 之後那一分鐘開盤成交。進場與停損同時延遲。", fontsize=9, color=INK2, va="top")
fig.savefig(STATS / "delay_equity.png", dpi=120, bbox_inches="tight", facecolor=SURF); print(STATS / "delay_equity.png")
