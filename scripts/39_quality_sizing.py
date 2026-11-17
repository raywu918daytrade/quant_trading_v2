#!/usr/bin/env python3
"""研究: 訊號品質 (09:05 已知) 要拿來「砍掉差的」還是「好的多押」? 各品質訊號是不是同一件事? 滑價 0 / 1 / 2 檔下誰站得住。
只讀資料、印報告, 不改線上規則。
輸入: data/processed/d1_sr_orb.parquet (先跑 scripts/33_d1_sr_orb.py)。框架同 34_rank_breakers.py: 現行規則、不設上限的全部跌破單,
組合時先到先做 (同分鐘依代號)、每日最多 10 檔、每檔 = 資金/10, 被砍的名額由之後跌破的單遞補。

品質訊號 (門檻一律用探索期 ≤2023 的分位數):
  區間寬   = 開盤 5 分 (區間高 / 區間低 − 1)            ← 上一輪「區間寬前 40% 開 2 倍」
  5 分成交額 = 開盤 5 分量 × 區間低                      ← 34 §5 的候選 (探索中位 ≈1.5 億)
  5 分量比 = 開盤 5 分量 / 20 日均量                      ← 34 §5 的候選
  碰壓力   = 區間高碰到日K壓力 (lvl_in)                  ← 33 的候選 (1.5 / 0.75 格)
  一檔占比 = 跳動單位 / 區間低                            ← 30 的濾網
滑價: 進場與停損各不利 k 檔 (收盤競價不滑)。
加權有兩種上限: 「10 檔」(加權後曝險可超過 100%) 與「10 格資金」(權重累計 ≤10 格, 放不下就不做) — 後者才是同資金的公平比較。
用法: python3 scripts/39_quality_sizing.py [--inp data/processed/d1_sr_orb.parquet] [--out stats/quality_sizing.md]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--out", default="stats/quality_sizing.md")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)

Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok].copy()
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy()
EXP = pd.Timestamp(EXPLORE_END); A["exp"] = A.date <= EXP
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
A["tick_pct"] = tick(A.rl.values) / A.rl * 100
nleg = 1 + (A.s_why == "停損")
for k in (1, 2): A[f"s_r{k}"] = A.s_r - k * A.tick_pct * nleg
RC = {0: "s_r", 1: "s_r1", 2: "s_r2"}

A["f_width"] = (A.rh / A.rl - 1) * 100
A["f_amt"] = A.o5_v * 1000 * A.rl / 1e8          # 億
A["f_vr"] = A.o5_vr * 100
A["f_lvl"] = A.lvl_in.fillna(0).astype(float)
A["f_tick"] = A.tick_pct
A["f_gap"] = A.gap * 100; A["f_atr"] = A.atr20 * 100
FEAT = {"f_width": "區間寬%", "f_amt": "5 分成交額(億)", "f_vr": "5 分量比%", "f_lvl": "區間碰壓力", "f_tick": "一檔占比%", "f_gap": "跳空%", "f_atr": "ATR20%"}
q = lambda c, p: float(A.loc[A.exp, c].quantile(p))
TH = dict(w20=q("f_width", .2), w60=q("f_width", .6), amt50=q("f_amt", .5), vr50=q("f_vr", .5))

P("# 訊號品質: 砍掉差的 vs 好的多押")
P("")
P(f"期間 {A.date.min().date()} ~ {A.date.max().date()}；探索 ≤{EXPLORE_END}、驗證之後。現行規則不設上限 {len(A):,} 筆, 每筆 {A.s_r.mean():.3f}% "
  f"(滑 1 檔 {A.s_r1.mean():.3f}%、2 檔 {A.s_r2.mean():.3f}%)。每筆含成本 0.23%、+8% 強制回補、收盤鎖漲停 −7%。")
P(f"門檻 (探索期分位): 區間寬 20 分位 {TH['w20']:.2f}%、60 分位 {TH['w60']:.2f}%；5 分成交額中位 {TH['amt50']:.2f} 億；5 分量比中位 {TH['vr50']:.1f}%。")

# ---------- 1. 相關 ----------
P(""); P("## 1. 品質訊號彼此的相關 (Spearman, 全部筆數)")
cs = list(FEAT)
C = A[cs].rank().corr()
P(""); P("| | " + " | ".join(FEAT[c] for c in cs) + " |"); P("|---|" + "---|" * len(cs))
for r in cs: P(f"| {FEAT[r]} | " + " | ".join(f"{C.loc[r, c]:+.2f}" for c in cs) + " |")

# ---------- 2. 單一訊號五分位 ----------
P(""); P("## 2. 單一訊號五分位 (每筆%, 小→大)")
P(""); P("| 訊號 | 滑 0 全期 | 滑 0 探索 | 滑 0 驗證 | 滑 1 全期 | 滑 1 驗證 | 滑 2 驗證 | 停損率% |"); P("|---|---|---|---|---|---|---|---|")
for c in ["f_width", "f_amt", "f_vr", "f_tick", "f_gap", "f_atr"]:
    qq = pd.qcut(A[c].rank(method="first"), 5, labels=False)
    f = lambda m, rc: " / ".join(f"{A.loc[(qq == i) & m, rc].mean():.2f}" for i in range(5))
    al = pd.Series(True, index=A.index)
    P(f"| {FEAT[c]} | {f(al, 's_r')} | {f(A.exp, 's_r')} | {f(~A.exp, 's_r')} | {f(al, 's_r1')} | {f(~A.exp, 's_r1')} | {f(~A.exp, 's_r2')} | "
      + " / ".join(f"{(A.loc[qq == i, 's_why'] == '停損').mean()*100:.0f}" for i in range(5)) + " |")
for v in (0, 1):
    m = A.f_lvl == v
    P(f"| 區間碰壓力 = {v} ({m.mean()*100:.0f}%) | {A.loc[m, 's_r'].mean():.2f} | {A.loc[m & A.exp, 's_r'].mean():.2f} | {A.loc[m & ~A.exp, 's_r'].mean():.2f} | "
      f"{A.loc[m, 's_r1'].mean():.2f} | {A.loc[m & ~A.exp, 's_r1'].mean():.2f} | {A.loc[m & ~A.exp, 's_r2'].mean():.2f} | {(A.loc[m, 's_why'] == '停損').mean()*100:.0f} |")

# ---------- 3. 交叉: 是不是同一件事 ----------
P(""); P("## 3. 交叉表: 區間寬 × 5 分成交額 (每筆%: 滑 0 / 滑 1 [筆數])")
P("如果兩者是同一件事, 控制其中一個後另一個就不該再有差。")
wq = pd.cut(A.f_width, [-1, TH["w20"], TH["w60"], 99], labels=["寬 <20 分位", "寬 20~60", "寬 ≥60 分位"])
aq = np.where(A.f_amt >= TH["amt50"], "成交額 ≥ 中位", "成交額 < 中位")
for nm, m in [("全期", A.exp | ~A.exp), ("驗證期", ~A.exp)]:
    P(""); P(f"**{nm}**"); P(""); P("| | 成交額 < 中位 | 成交額 ≥ 中位 |"); P("|---|---|---|")
    for w in wq.cat.categories:
        cells = []
        for am in ["成交額 < 中位", "成交額 ≥ 中位"]:
            g = A[m.values & (wq == w).values & (aq == am)]
            cells.append(f"{g.s_r.mean():.2f} / {g.s_r1.mean():.2f} [{len(g):,}]")
        P(f"| {w} | " + " | ".join(cells) + " |")
P(""); P("## 3b. 交叉表: 區間碰壓力 × 5 分成交額 (每筆%: 滑 0 / 滑 1 [筆數], 全期)")
P(""); P("| | 成交額 < 中位 | 成交額 ≥ 中位 |"); P("|---|---|---|")
for v in (0, 1):
    cells = []
    for am in ["成交額 < 中位", "成交額 ≥ 中位"]:
        g = A[(A.f_lvl == v).values & (aq == am)]; cells.append(f"{g.s_r.mean():.2f} / {g.s_r1.mean():.2f} [{len(g):,}]")
    P(f"| 碰壓力 = {v} | " + " | ".join(cells) + " |")

# ---------- 4. 組合 ----------
def run(keep=None, w=None, rc="s_r", cap="n"):
    """先到先做; keep=False 的不做 (名額遞補); w = 權重 (格); cap='n': 最多 10 檔; cap='cash': 權重累計 ≤10 格。"""
    z = A.assign(w=1.0 if w is None else w)
    if keep is not None: z = z[keep]
    z = z.sort_values(["date", "s_et", "stock_id"])
    if cap == "n": z = z.groupby("date").head(10)
    else: z = z[z.groupby("date").w.cumsum() <= 10 + 1e-9]
    dly = (z.w * z[rc]).groupby(z.date).sum().reindex(ALLDAYS, fill_value=0.0) / 10; dly.index = ALLDAYS
    expo = (z.w.groupby(z.date).sum() / 10).reindex(ALLDAYS, fill_value=0.0)
    return dly, z, expo
def line(name, dly, z, expo, rc):
    eq = dly.cumsum(); mdd = (eq - eq.cummax()).min(); e = dly.index <= EXP
    sh = dly.mean() / dly.std() * np.sqrt(250)
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    tr = expo[expo > 0]
    return (f"| {name} | {len(z):,} | {(z.w * z[rc]).sum() / z.w.sum():.3f} | {dly.mean()*250*3:.0f} | {sh:.2f} | {mdd*3:.0f} | {dly.min()*3:.1f} | "
            f"{tr.mean()*100:.0f}% / {tr.max()*100:.0f}% | {dly[e].mean()*250*3:.0f} / {dly[~e].mean()*250*3:.0f} | {yr} |")
HD = ["| 方法 | 筆數 | 加權每筆% | 年均 (萬, 300萬) | Sharpe | MDD (萬) | 最差日 (萬) | 有交易日曝險 平均/最大 | 探索/驗證 年均 (萬) | 2020~2026 日淨% |",
      "|---|---|---|---|---|---|---|---|---|---|"]
tk_ok = A.f_tick <= 0.2; amt_ok = A.f_amt >= TH["amt50"]; vr_ok = A.f_vr >= TH["vr50"]; w_ok = A.f_width >= TH["w20"]
wide = (A.f_width >= TH["w60"]).astype(float); lvl = A.f_lvl
V = [("現行", None, None),
     ("砍 5 分成交額 < 中位", amt_ok, None),
     ("砍 5 分量比 < 中位", vr_ok, None),
     ("砍 區間寬 < 20 分位", w_ok, None),
     ("砍 一檔 > 0.2%", A.f_tick <= 0.2, None),
     ("加權: 區間寬 ≥60 分位 2 格", None, 1 + wide),
     ("加權: 碰壓力 1.5 格 / 其他 0.75 格", None, 0.75 + 0.75 * lvl),
     ("加權: 5 分成交額 ≥ 中位 2 格", None, 1 + amt_ok.astype(float)),
     ("砍成交額 + 砍區間寬", amt_ok & w_ok, None),
     ("砍成交額 + 區間寬加權", amt_ok, 1 + wide),
     ("砍成交額 + 碰壓力加權", amt_ok, 0.75 + 0.75 * lvl),
     ("砍成交額 + 區間寬加權 + 碰壓力加權", amt_ok, (1 + wide) * (0.75 + 0.75 * lvl)),
     ("砍成交額 + 砍一檔 > 0.2%", amt_ok & tk_ok, None),
     ("砍成交額 + 砍一檔 > 0.2% + 碰壓力加權", amt_ok & tk_ok, 0.75 + 0.75 * lvl),
     ("砍成交額 + 砍一檔 > 0.2% + 區間寬加權 + 碰壓力加權", amt_ok & tk_ok, (1 + wide) * (0.75 + 0.75 * lvl)),
     ]
for slip in (0, 1, 2):
    rc = RC[slip]
    for cap, capn in [("n", "最多 10 檔 (加權後曝險可超過 100%)"), ("cash", "權重累計 ≤10 格 (同資金上限)")]:
        P(""); P(f"## 4{'abc'[slip]}{'12'[cap == 'cash']}. 組合 — 滑 {slip} 檔, {capn}")
        P(""); [P(h) for h in HD]
        for nm, kp, w in V:
            if cap == "cash" and w is None and nm != "現行": continue      # 不加權的兩種上限一樣
            P(line(nm, *run(kp, w, rc, cap), rc))

# ---------- 5. 同曝險對照: 加權 vs 全部等比放大 ----------
P(""); P("## 5. 加權 vs 「全部等比放大到同樣平均曝險」(最多 10 檔)")
P("加權的價值 = 同樣平均曝險下 Sharpe / 回撤有沒有比較好; 只是曝險變大而多賺的不算。")
P(""); P("| 方法 | 滑價 | 年均 | Sharpe | MDD | 對照: 現行 × 同曝險倍數 年均 / MDD | 驗證期年均 (對照) |"); P("|---|---|---|---|---|---|---|")
d0, z0, e0 = run(None, None, "s_r")
for nm, kp, w in V[5:]:
    for slip in (0, 1, 2):
        rc = RC[slip]
        d, z, e = run(kp, w, rc); db, zb, eb = run(None, None, rc)
        k = e.mean() / eb.mean()
        ds = db * k; eqb = ds.cumsum(); eqx = d.cumsum(); ex = d.index <= EXP
        P(f"| {nm} | {slip} | {d.mean()*750:.0f} | {d.mean()/d.std()*np.sqrt(250):.2f} | {(eqx - eqx.cummax()).min()*3:.0f} | "
          f"×{k:.2f}: {ds.mean()*750:.0f} / {(eqb - eqb.cummax()).min()*3:.0f} | {d[~ex].mean()*750:.0f} ({ds[~ex].mean()*750:.0f}) |")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out}")
