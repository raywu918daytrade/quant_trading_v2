#!/usr/bin/env python3
"""研究 41: 降低「每個月的最大回撤」— 出場法 + 月內風控, 用 11-18 起上線的新規則 (品質篩選 + 權重) 重算。只讀資料、印報告, 不改線上規則。

輸入: data/processed/d1_sr_orb.parquet (先跑 33) 與 data/processed/exit_paths.npz/.parquet (先跑 40 --build-only)。
交易 = 線上新規則: 39 的「現行」條件 ∩ 開盤 5 分成交額 ≥1.5 億 ∩ 一檔/區間低 ≤0.2%;
  權重 = (區間寬 ≥2.1% → 2, 否則 1) × (區間高碰日K壓力 → 1.5, 否則 0.75); 先跌破先做, 當日權重累計 ≤10 格 (每格 30 萬)。
出場: 同 40 (停損保留、盤中市價出場扣 0.1%、同分鐘先算停損)。
月內回撤 = 每個月從月初 0 起算的累計損益, 相對當月最高點的最大跌幅 (萬, 300 萬本金)。
月內風控 (只用到前一天為止的資訊): 當月累計虧損 / 月內回撤到門檻後, 當月剩下的日子停做或減半。
RL: 同 40 的批次 Q-learning, 但報酬改成「虧損加重」的效用 u(x) = x (x≥0), (1+κ)x (x<0) — 單筆 RL 看不到整個月的組合路徑, 用這個近似「討厭虧損」。
走動式: 2022~2026 每年只用之前年份挑參數 / 訓練; 標準 = 訓練期 年均 / |平均月內回撤| 最大, 且年均 ≥ 現行 70%。
用法: python3 scripts/41_monthly_dd.py [--out stats/monthly_dd.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COST, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="stats/monthly_dd.md")
ap.add_argument("--cache", default=str(PROC / "exit_paths.npz"))
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
NT, STOP_SLIP, SLIP, FORCE, FORCE_SLIP = 271, 0.001, 0.001, 0.08, 0.002
T0 = time.time()

# ---------------- 交易 (新規則) ----------------
Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy()
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
A["tick_pct"] = tick(A.rl.values) / A.rl * 100
A = A[(A.o5_v * 1000 * A.rl >= 1.5e8) & (A.tick_pct <= 0.2)].copy()
A["w"] = np.where((A.rh / A.rl - 1) * 100 >= 2.1, 2.0, 1.0) * np.where(A.lvl_in.fillna(0) > 0, 1.5, 0.75)
A = A.sort_values(["date", "s_et", "stock_id"])
A = A[A.groupby("date").w.cumsum() <= 10 + 1e-9]
Z = np.load(a.cache); M = pd.read_parquet(Path(a.cache).with_suffix(".parquet")); M["stock_id"] = M.stock_id.astype(str); M["date"] = pd.to_datetime(M.date)
M["j"] = np.arange(len(M))
meta = A[["stock_id", "date", "w", "s_r", "s_et", "tick_pct"]].merge(M, on=["stock_id", "date"], how="inner").reset_index(drop=True)
P(f"(新規則交易 {len(A):,} 筆, 對到 1 分K 路徑 {len(meta):,} 筆; 進場分鐘一致 {(meta.s_et == meta.et).mean()*100:.1f}%)")
meta = meta[meta.s_et == meta.et].reset_index(drop=True)
ALLDAYS = pd.DatetimeIndex(Z["days"])
J = meta.j.values
O, H, L, C, V = (Z[k][J].astype("float64") for k in "ohlcv")
N = len(meta); WT = meta.w.values; TKP = meta.tick_pct.values / 100
E = meta.E.values[:, None]; RH = meta.rh.values[:, None]; RL = meta.rl.values[:, None]; PC = meta.pc.values[:, None]; ET = meta.et.values[:, None]
TT = np.arange(NT)[None, :]; AFT = TT > ET
CF = pd.DataFrame(C).ffill(axis=1).values
typ = np.nan_to_num((H + L + C) / 3 * V); vv = np.nan_to_num(V)
cv = np.cumsum(vv, 1); VWAP = np.where(cv > 0, np.cumsum(typ, 1) / np.maximum(cv, 1e-9), CF)
LA = np.where(AFT, L, np.nan)
RUNMIN = np.minimum(np.fmin.accumulate(np.where(np.isnan(LA), np.inf, LA), axis=1), E)
RUNMIN_PREV = np.where(AFT, np.concatenate([E, RUNMIN[:, :-1]], 1), E)
YEAR = meta.date.dt.year.values
ret = lambda px: (-(px / (E if np.ndim(px) == 2 else E[:, 0]) - 1) - COST) * 100
def first(mask):
    hit = mask.any(1); return np.where(hit, mask.argmax(1), 10_000)
STOP_HIT = AFT & (H > RH)
def capf(px): return np.where(px >= PC * (1 + FORCE) if np.ndim(px) == 2 else px >= PC[:, 0] * (1 + FORCE), (PC if np.ndim(px) == 2 else PC[:, 0]) * (1 + FORCE) * (1 + FORCE_SLIP), px)
def simulate(sig=None, limit=None, stop_lvl=None, t_end=270):
    lvl = RH if stop_lvl is None else np.minimum(RH, stop_lvl)
    ar = np.arange(N)
    i_s = first(AFT & (H > lvl) & (TT <= t_end))
    lv_at = lvl[ar, np.minimum(i_s, NT - 1)] if lvl.shape[1] > 1 else lvl[:, 0]
    o = O[ar, np.minimum(i_s, NT - 1)]
    px_s = capf(np.maximum(lv_at, np.where(np.isnan(o), lv_at, o)) * (1 + STOP_SLIP))
    i_l = np.full(N, 10_000); px_l = np.full(N, np.nan)
    if limit is not None:
        i_l = first(AFT & (L < limit) & (TT <= t_end))
        oo = O[ar, np.minimum(i_l, NT - 1)]; lm = limit[ar, np.minimum(i_l, NT - 1)]
        px_l = np.minimum(lm, np.where(np.isnan(oo), lm, oo))
    i_g = np.full(N, 10_000) if sig is None else first(sig & AFT & (TT < t_end))
    px_g = CF[ar, np.minimum(i_g, NT - 1)] * (1 + SLIP)
    px_e = CF[:, t_end] * (1 + (SLIP if t_end < 270 else 0))
    idx = np.stack([i_s, i_l, i_g, np.full(N, t_end)]); k = np.argmin(idx, 0)
    px = np.choose(k, [px_s, px_l, px_g, px_e])
    intraday = ~((k == 3) & (t_end == 270))                       # 盤中出場 (滑 1 檔敏感度用)
    return ret(px), intraday

def daily(r):
    s = pd.Series(WT * r / 10, index=meta.date).groupby(level=0).sum()
    return s.reindex(ALLDAYS, fill_value=0.0)
def mdd_month(dl):
    out = {}
    for m, x in dl.groupby(dl.index.to_period("M")):
        eq = np.concatenate([[0.0], x.cumsum().values]); out[m] = (eq - np.maximum.accumulate(eq)).min() * 3
    return pd.Series(out)
def st(dl):
    eq = dl.cumsum(); mdd = (eq - eq.cummax()).min() * 3
    mm = mdd_month(dl); mret = dl.groupby(dl.index.to_period("M")).sum() * 3
    sd = dl.std()
    return dict(ann=dl.mean() * 750, sh=dl.mean() / sd * np.sqrt(250) if sd > 0 else 0, mdd=mdd, mm_mean=mm.mean(), mm_med=mm.median(),
                mm_p90=mm.quantile(0.1), mm_worst=mm.min(), n10=int((mm <= -10).sum()), nmon=len(mm), negm=float((mret < 0).mean() * 100),
                worstm=mret.min(), score=(dl.mean() * 750) / max(-mm.mean(), 1e-9))
def sub(dl, years): return dl[dl.index.year.isin(years)]
HD = ["| 方法 | 年均 (萬) | Sharpe | 全期最大回撤 | 月內回撤 平均 | 中位 | 最差 10% | 最差月 | 月內回撤 ≤−10 萬的月數 | 虧損月% | 最差月損益 |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
def line(nm, s): return (f"| {nm} | {s['ann']:.0f} | {s['sh']:.2f} | {s['mdd']:.0f} | {s['mm_mean']:.1f} | {s['mm_med']:.1f} | {s['mm_p90']:.1f} | {s['mm_worst']:.1f} | "
                         f"{s['n10']} / {s['nmon']} | {s['negm']:.0f} | {s['worstm']:.1f} |")

base_r, base_in = simulate()
P(f"(自我檢查: 本程式現行出場 vs 33 回測每筆差 最大 {np.abs(base_r - meta.s_r.values).max():.3f}%, 平均 {np.abs(base_r - meta.s_r.values).mean():.4f}%)")

# ---------------- 出場規則 ----------------
def R_target(p): return simulate(limit=np.broadcast_to(E * (1 - p / 100), (N, NT)))
def R_trail(x, act):
    on = RUNMIN_PREV <= E * (1 - act / 100); return simulate(stop_lvl=np.where(on, RUNMIN_PREV * (1 + x / 100), np.inf))
def R_be(act, b):
    on = RUNMIN_PREV <= E * (1 - act / 100); return simulate(stop_lvl=np.where(on, E * (1 - b / 100), np.inf))
def R_tighter(f):        # 停損收緊到 區間低 + f × 區間寬 (f=1 = 現行)
    return simulate(stop_lvl=np.broadcast_to(RL + f * (RH - RL), (N, NT)))
EXITS = {"現行": (base_r, base_in)}
for p in (3, 4, 5): EXITS[f"賺 {p}% 回補"] = R_target(p)
for act in (1, 2, 3):
    for x in (2, 3): EXITS[f"賺 {act}% 後從最低反彈 {x}% 出"] = R_trail(x, act)
for act in (1, 1.5, 2, 3): EXITS[f"賺 {act}% 後停損移到進場價"] = R_be(act, 0)
for f in (0.75, 0.9): EXITS[f"停損收緊到區間 {int(f*100)}% 位置"] = R_tighter(f)
print(f"出場規則完成 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 月內風控 (作用在任一出場法的日損益上) ----------------
def overlay(dl, kind, thr, cut=0.0):
    """kind='loss': 當月累計 ≤ −thr 萬後, 當月剩下日子部位 × cut; kind='dd': 當月回撤 ≤ −thr 萬後 × cut (月內創新高不恢復)。只用到前一天。"""
    out = dl.to_numpy(copy=True); idx = dl.index; per = idx.to_period("M")
    cum = 0.0; peak = 0.0; scale = 1.0; cur = None
    for i in range(len(out)):
        if per[i] != cur: cur = per[i]; cum = 0.0; peak = 0.0; scale = 1.0
        x = out[i] * scale; out[i] = x
        cum += x * 3; peak = max(peak, cum)
        if (kind == "loss" and cum <= -thr) or (kind == "dd" and cum - peak <= -thr): scale = cut
    return pd.Series(out, index=idx)
def overlay_streak(dl, k, cut=0.5):
    """連續 k 個「有交易且虧損」的日子後, 隔天起部位 × cut, 直到出現一個賺錢日。"""
    out = dl.to_numpy(copy=True); n = 0; scale = 1.0
    for i in range(len(out)):
        x = out[i] * scale; out[i] = x
        if dl.values[i] < 0: n += 1
        elif dl.values[i] > 0: n = 0; scale = 1.0
        if n >= k: scale = cut
    return pd.Series(out, index=dl.index)
OV = {"無": lambda d: d}
for thr in (5, 8, 10, 15):
    OV[f"月虧 {thr} 萬停做"] = (lambda d, t=thr: overlay(d, "loss", t, 0.0))
    OV[f"月虧 {thr} 萬減半"] = (lambda d, t=thr: overlay(d, "loss", t, 0.5))
    OV[f"月回撤 {thr} 萬停做"] = (lambda d, t=thr: overlay(d, "dd", t, 0.0))
for k in (2, 3): OV[f"連虧 {k} 天減半"] = (lambda d, k=k: overlay_streak(d, k))

# ---------------- RL (虧損加重效用) ----------------
TB = np.digitize(TT + 0 * ET, [30, 60, 90, 120, 180, 240]); PNL = (E - CF) / E * 100
STATE = ((TB * 7 + np.digitize(PNL, [-1, 0, 1, 2, 3, 5])) * 4 + np.digitize((CF - RUNMIN) / E * 100, [0.5, 1, 2])) * 2 + (CF > VWAP).astype(int)
NS = 392; T_LAST = 264
EXITV = ret(CF * (1 + SLIP))
o_stop = np.where(np.isnan(O), RH, O)
STOPV = (-(capf(np.maximum(RH, o_stop) * (1 + STOP_SLIP)) / E - 1) - COST) * 100
CLOSEV = ret(CF[:, 270])
def values(pol):
    D = np.zeros((N, NT + 1)); D[:, NT] = CLOSEV; D[:, 270] = np.where(STOP_HIT[:, 270], STOPV[:, 270], CLOSEV)
    for u in range(269, -1, -1):
        ex = pol[STATE[:, u]] & (u <= T_LAST)
        D[:, u] = np.where(STOP_HIT[:, u], STOPV[:, u], np.where(ex, EXITV[:, u], D[:, u + 1]))
    return D
def apply(pol):
    D = values(pol); return D[np.arange(N), np.minimum(ET[:, 0] + 1, NT)]
def train(mask, kappa, iters=4, min_trades=60, margin=0.02):
    u = lambda x: np.where(x >= 0, x, (1 + kappa) * x)
    pol = np.zeros(NS, bool); rows = np.where(mask)[0]
    dec = AFT[rows] & (TT <= T_LAST) & ~STOP_HIT[rows]; rr, cc = np.nonzero(dec)
    s_ = STATE[rows][rr, cc]; ev = u(EXITV[rows][rr, cc]); w_ = WT[rows][rr]
    for _ in range(iters):
        hv = u(values(pol)[rows][rr, cc + 1])
        df = pd.DataFrame(dict(s=s_, d=(ev - hv) * w_, w=w_, n=rr))
        g = df.groupby("s").agg(d=("d", "sum"), w=("w", "sum"), nt=("n", "nunique")); g["d"] = g.d / g.w
        new = np.zeros(NS, bool); new[g[(g.nt >= min_trades) & (g.d > margin)].index.values] = True
        if (new == pol).all(): break
        pol = new
    return pol
P(f"(自我檢查: RL 環境「永不提前平倉」vs 現行 每筆差 {np.abs(apply(np.zeros(NS, bool)) - base_r).max():.2e}%)")
TEST_YEARS = [y for y in range(2022, 2027) if (YEAR == y).any()]
for kappa in (0.0, 0.5, 1.0, 2.0):
    r_o = base_r.copy()
    for y in TEST_YEARS:
        r_o = np.where(YEAR == y, apply(train(YEAR < y, kappa)), r_o)
    EXITS[f"RL 虧損加重 κ={kappa} (走動式)"] = (r_o, np.ones(N, bool))
print(f"RL 完成 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 報表 ----------------
P("# 降低每個月的最大回撤 (新規則: 品質篩選 + 權重)")
P("")
P(f"期間 {meta.date.min().date()} ~ {meta.date.max().date()}, {N:,} 筆, 300 萬本金, 每格 30 萬、當日 ≤10 格。成本 0.23%, 盤中出場另扣 0.1%; 停損保留。")
P("月內回撤 = 每月從 0 起算、相對當月最高點的最大跌幅 (萬)。風控只用到前一天的資訊。")
P(""); P("## 1. 各出場法 (無風控, 全期; 探索 ≤2023 挑參數會有偏誤, 走動式見 §3)"); P(""); [P(h) for h in HD]
DL = {nm: daily(r) for nm, (r, _) in EXITS.items()}
for nm in EXITS:
    if not nm.startswith("RL"): P(line(nm, st(DL[nm])))

P(""); P("## 2. 月內風控 (現行出場, 全期)"); P(""); [P(h) for h in HD]
for on, f in OV.items(): P(line(on, st(f(DL["現行"]))))

# 走動式: 出場法 × 風控 一起挑
P(""); P(f"## 3. 走動式樣本外 ({TEST_YEARS[0]}~{TEST_YEARS[-1]}): 每年用之前年份挑「出場法 × 風控」組合")
COMBO = {}
for en in EXITS:
    if en.startswith("RL"): continue
    for on, f in OV.items(): COMBO[(en, on)] = f(DL[en])
def pick(train_years, pool):
    b = st(sub(COMBO[("現行", "無")], train_years))
    best, bs = ("現行", "無"), b["score"]
    for key, dl in pool.items():
        s = st(sub(dl, train_years))
        if s["ann"] < 0.7 * b["ann"]: continue
        if s["score"] > bs: best, bs = key, s["score"]
    return best
def wf(pool):
    parts, ch = [], []
    for y in TEST_YEARS:
        k = pick([yy for yy in range(2020, y)], pool); ch.append(f"{y}: {k[0]} + {k[1]}")
        parts.append(sub(COMBO[k], [y]))
    return pd.concat(parts), ch
oos_base = sub(COMBO[("現行", "無")], TEST_YEARS)
res = {"只換出場法": wf({k: v for k, v in COMBO.items() if k[1] == "無"}),
       "只加風控": wf({k: v for k, v in COMBO.items() if k[0] == "現行"}),
       "出場法 × 風控 一起挑": wf(COMBO)}
P(""); [P(h) for h in HD]
P(line("**現行 (無風控)**", st(oos_base)))
for nm, (dl, ch) in res.items(): P(line(nm, st(dl)))
for nm in EXITS:
    if nm.startswith("RL"): P(line(nm, st(sub(DL[nm], TEST_YEARS))))
P(""); P("各年選到:")
for nm, (dl, ch) in res.items(): P(f"- {nm}: " + "; ".join(ch))

# 固定候選: 逐年月內回撤
CAND = [("現行", "無"), ("賺 2% 後停損移到進場價", "無"), ("賺 2% 後從最低反彈 3% 出", "無"), ("現行", "月虧 10 萬停做"), ("現行", "月回撤 10 萬停做"),
        ("賺 2% 後停損移到進場價", "月虧 10 萬停做"), ("賺 2% 後停損移到進場價", "月回撤 10 萬停做")]
YS = sorted(set(YEAR))
P(""); P("## 4. 固定候選逐年: 年均 (萬) / 平均月內回撤 / 最差月內回撤")
P(""); P("| 組合 | " + " | ".join(str(y) for y in YS) + " |"); P("|---|" + "---|" * len(YS))
for key in CAND:
    dl = COMBO[key]
    P(f"| {key[0]} + {key[1]} | " + " | ".join(f"{st(sub(dl, [y]))['ann']:.0f} / {st(sub(dl, [y]))['mm_mean']:.1f} / {st(sub(dl, [y]))['mm_worst']:.1f}" for y in YS) + " |")

# 每月最大回撤分布
P(""); P("## 5. 每月最大回撤分布 (全期月數)")
bins = [-1e9, -20, -15, -10, -5, -2, 0.01]; lab = ["≤−20", "−20~−15", "−15~−10", "−10~−5", "−5~−2", "−2~0"]
P(""); P("| 組合 | " + " | ".join(lab) + " |"); P("|---|" + "---|" * len(lab))
for key in CAND:
    mm = mdd_month(COMBO[key]); c = pd.cut(mm, bins, labels=lab).value_counts().reindex(lab, fill_value=0)
    P(f"| {key[0]} + {key[1]} | " + " | ".join(str(int(v)) for v in c.values) + " |")

# 滑 1 檔敏感度: 進場 +1 檔, 盤中出場 +1 檔
P(""); P("## 6. 滑 1 檔 (進場與盤中出場各不利 1 檔; 收盤競價不滑)")
P(""); [P(h) for h in HD]
for nm in ["現行", "賺 2% 後停損移到進場價", "賺 2% 後從最低反彈 3% 出", "賺 3% 後停損移到進場價"]:
    r, intr = EXITS[nm]; r1 = r - TKP * 100 * (1 + intr)
    P(line(nm, st(daily(r1))))
    P(line(nm + " + 月虧 10 萬停做", st(overlay(daily(r1), "loss", 10, 0.0))))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
