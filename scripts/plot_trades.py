"""畫定案規則最近 N 筆交易的當日 1 分 K (回放 orb_live 邏輯). 用法: python3 plot_trades.py [N] [--end YYYY-MM-DD]"""
import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
import orb_live as L
from config import STATS

ap = argparse.ArgumentParser(); ap.add_argument("n", nargs="?", type=int, default=10); ap.add_argument("--end"); a = ap.parse_args()
for f in ["PingFang TC", "Heiti TC", "Arial Unicode MS"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
UP, DN = "#d33", "#2a7"

d1 = L.load_d1(since="2025_06")
dates = sorted(d1.date.unique(), reverse=True)
if a.end: dates = [d for d in dates if d <= pd.Timestamp(a.end)]
trades, m1c = [], {}
for d in dates:
    ym = pd.Timestamp(d).strftime("%Y_%m")
    if ym not in m1c:
        try: m1c[ym] = L.load_m1_month(ym)
        except FileNotFoundError: continue
    if not (m1c[ym].date == d).any(): continue
    tr, info, watch, _ = L.replay_day(d, d1, m1c[ym])
    if len(tr):
        tr["date"] = pd.Timestamp(d); tr = tr.merge(watch[["stock_id", "pc", "open", "atr20"]], on="stock_id"); trades.append(tr)
    if sum(len(t) for t in trades) >= a.n: break
T = pd.concat(trades).sort_values(["date", "entry_t"], ascending=[False, False]).head(a.n).reset_index(drop=True)

def hhmm(t): return f"{9 + int(t) // 60:02d}:{int(t) % 60:02d}"
n = len(T); rows = (n + 1) // 2
fig = plt.figure(figsize=(18, 5.2 * rows))
outer = fig.add_gridspec(rows, 2, hspace=0.36, wspace=0.14, top=1 - 0.95 / (5.2 * rows), bottom=0.02)
for i, r in T.iterrows():
    m = m1c[r.date.strftime("%Y_%m")]; b = m[(m.date == r.date) & (m.stock_id == r.stock_id)].sort_values("t")
    inner = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax = fig.add_subplot(inner[0]); av = fig.add_subplot(inner[1], sharex=ax)
    up = b.close >= b.open; col = np.where(up, UP, DN)
    ax.vlines(b.t, b.low, b.high, color=col, lw=0.7, zorder=2)
    ax.bar(b.t, (b.close - b.open).abs().clip(lower=(b.close * 0.0004)), bottom=np.minimum(b.open, b.close), width=0.75, color=col, zorder=3)
    av.bar(b.t, b.volume, width=0.8, color=col)
    ax.axvspan(-0.5, 4.5, color="#ffe9a8", alpha=0.6, zorder=0)
    ax.axhline(r.rh, color="#e67e00", ls="--", lw=1.2, zorder=1); ax.axhline(r.rl, color="#1f6fd1", ls="--", lw=1.2, zorder=1)
    ax.axhline(r.pc, color="#888", ls=":", lw=1, zorder=1)
    lo_, hi_ = b.low.min(), b.high.max(); pad = (hi_ - lo_) * 0.08
    for lvl, lab, c in [(r.pc * 1.08, "+8%", "#9b59b6"), (L.limit_up(r.pc), "漲停", UP)]:
        if lvl <= hi_ + pad * 2: ax.axhline(lvl, color=c, ls=":", lw=1); ax.text(273, lvl, lab, color=c, fontsize=8, va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(273, r.rh, f"停損 {r.rh:g}", color="#e67e00", fontsize=8, va="center", bbox=dict(fc="white", ec="none", pad=0.5)); ax.text(273, r.rl, f"進場 {r.rl:g}", color="#1f6fd1", fontsize=8, va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(273, r.pc, f"昨收 {r.pc:g}", color="#666", fontsize=8, va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    ax.scatter([r.entry_t], [r.entry_px], marker="v", s=110, color="#1f6fd1", edgecolor="k", zorder=5)
    ax.annotate(f"空 {r.entry_px:g}\n{hhmm(r.entry_t)}", (r.entry_t, r.entry_px), xytext=(8, -26), textcoords="offset points", fontsize=8, color="#1f6fd1")
    xc = "#e67e00" if r.reason.startswith("停損") else "#333"
    ax.scatter([r.exit_t], [r.exit_px], marker="^", s=110, color=xc, edgecolor="k", zorder=5)
    ax.annotate(f"補 {r.exit_px:.2f}\n{hhmm(r.exit_t)}", (r.exit_t, r.exit_px), xytext=(-48 if r.exit_t > 200 else 8, 10), textcoords="offset points", fontsize=8, color=xc)
    if r.reason.startswith("停損"): ax.axvspan(r.entry_t, r.exit_t, color="#f5c6c6", alpha=0.35, zorder=0)
    else: ax.axvspan(r.entry_t, r.exit_t, color="#cdeccd", alpha=0.35, zorder=0)
    res = f"{r.r_pct:+.2f}%"
    ax.set_title(f"{r.stock_id} {r['name']}  {r.date.date()}  跳空 {r.gap*100:+.1f}%  ATR {r.atr20*100:.1f}%  |  {r.reason}  {res}", fontsize=10.5,
                 color=(UP if r.r_pct > 0 else DN), loc="left")
    ax.set_ylim(min(lo_, r.rl) - pad, max(hi_, r.rh) + pad * 2.5); ax.set_xlim(-3, 300); ax.grid(alpha=0.25)
    plt.setp(ax.get_xticklabels(), visible=False)
    av.set_xticks(range(0, 271, 30)); av.set_xticklabels([hhmm(t) for t in range(0, 271, 30)], fontsize=8)
    av.tick_params(axis="y", labelsize=7); av.grid(alpha=0.25); av.set_ylabel("張", fontsize=8)
fig.suptitle(f"定案規則最近 {n} 筆交易 (黃底 = 開盤 5 分區間；藍虛線 = 區間低/進場；橘虛線 = 區間高/停損；綠底 = 持有到收盤；紅底 = 被停損)", fontsize=12, y=1 - 0.15 / (5.2 * rows))
out = STATS / f"trades_recent{n}.png"; fig.savefig(out, dpi=105, bbox_inches="tight"); print(out)
cols = ["date", "stock_id", "name", "gap", "rl", "rh", "entry_t", "entry_px", "exit_t", "exit_px", "reason", "r_pct"]
T2 = T[cols].assign(date=T.date.dt.date, gap=(T.gap * 100).round(2), entry_t=T.entry_t.map(hhmm), exit_t=T.exit_t.map(hhmm), r_pct=T.r_pct.round(2), exit_px=T.exit_px.round(2))
T2.to_csv(STATS / f"trades_recent{n}.csv", index=False); print(T2.to_string(index=False))
