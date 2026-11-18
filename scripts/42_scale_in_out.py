#!/usr/bin/env python3
"""研究 42: 進場後盤中動態加碼 / 減碼 / 補回 (選股與第一筆進場不變)。只讀資料、印報告, 不改線上規則。

輸入: data/processed/d1_sr_orb.parquet (先跑 33) 與 data/processed/exit_paths.npz/.parquet (先跑 40 --build-only)。
交易 = 11-18 起線上新規則 (同 41): 品質篩選 + 權重 0.75/1.5/3 格, 當日 ≤10 格, 每格 30 萬, 本金 300 萬。
每筆交易拆成好幾「腳」(leg), 每腳各自進出、各自扣成本 0.23%; 盤中市價 / 觸價單另扣 0.1% 滑價, 限價單 (反彈到某價賣) 不扣。
  停損一律保留 (反彈破區間高, 全部腳一起出); 同一分鐘碰到停損與其他動作一律算先停損; 13:30 收盤全部出。
規則族群:
  A 獲利加碼: 賺到 X% (跌破 進場×(1−X)) 再空 s 份; 停損維持區間高, 或加碼後全部停損移到第一筆進場價。
  B 二次跌破加碼 (拉上來又殺下去): 從最低點反彈 ≥b% 之後, 再跌破前低時加空 s 份; 加碼那份停損 = 區間高 或 反彈高點。
  C 反彈加碼: 跌深 ≥x% 後反彈碰到 區間低 / VWAP / 區間中間 時再空 (限價碰到就成交, 或碰到但收不上去才用收盤價空)。
  D 減碼再補回: 已賺 ≥act% 且從最低反彈 ≥r% 時先回補 f (一半或全部), 之後再跌破前低 (或從回補價回落 1%) 時空回去。
  E 依強弱調整初始部位: 跳空大 / 開盤 5 分成交額大 的那幾檔進場就 ×1.5。
  F RL 加碼: 狀態同 40/41 (時段 × 浮動損益 × 從最低反彈 × VWAP 上/下), 學「在哪些狀態加空半份」, 走動式。
基準: 現行, 以及「直接等量加大部位」(現行 × k)。部位越大損益與回撤等比放大, 所以公平比較 = 同樣的月內回撤 (或同樣的最大回撤) 下年均多少。
走動式: 2022~2026 每年只用之前年份挑參數; 標準 = 訓練期 年均 / |平均月內回撤| 最大, 且要贏現行, 否則維持現行。
用法: python3 scripts/42_scale_in_out.py [--out stats/scale_in_out.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COST, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="stats/scale_in_out.md")
ap.add_argument("--cache", default=str(PROC / "exit_paths.npz"))
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
NT, STOP_SLIP, SLIP, FORCE, FORCE_SLIP = 271, 0.001, 0.001, 0.08, 0.002
ADD_DL = 210                     # 加碼 / 補回 最晚 12:30
NO = 10_000
T0 = time.time()

# ---------------- 交易 (新規則, 同 41) ----------------
Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy()
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
A["tick_pct"] = tick(A.rl.values) / A.rl * 100
A["o5_amt"] = A.o5_v * 1000 * A.rl
A = A[(A.o5_amt >= 1.5e8) & (A.tick_pct <= 0.2)].copy()
A["w"] = np.where((A.rh / A.rl - 1) * 100 >= 2.1, 2.0, 1.0) * np.where(A.lvl_in.fillna(0) > 0, 1.5, 0.75)
A = A.sort_values(["date", "s_et", "stock_id"])
A = A[A.groupby("date").w.cumsum() <= 10 + 1e-9]
Z = np.load(a.cache); M = pd.read_parquet(Path(a.cache).with_suffix(".parquet")); M["stock_id"] = M.stock_id.astype(str); M["date"] = pd.to_datetime(M.date)
M["j"] = np.arange(len(M))
meta = A[["stock_id", "date", "w", "s_r", "s_et", "tick_pct", "o5_amt"]].merge(M, on=["stock_id", "date"], how="inner").reset_index(drop=True)
P(f"(新規則交易 {len(A):,} 筆, 對到 1 分K 路徑 {len(meta):,} 筆; 進場分鐘一致 {(meta.s_et == meta.et).mean()*100:.1f}%)")
meta = meta[meta.s_et == meta.et].reset_index(drop=True)
ALLDAYS = pd.DatetimeIndex(Z["days"])
J = meta.j.values
O, H, L, C, V = (Z[k][J].astype("float64") for k in "ohlcv")
N = len(meta); WT = meta.w.values; TKP = meta.tick_pct.values / 100; AR = np.arange(N)
E = meta.E.values[:, None]; RH = meta.rh.values[:, None]; RL = meta.rl.values[:, None]; PC = meta.pc.values[:, None]; ET = meta.et.values[:, None]
TT = np.arange(NT)[None, :]; AFT = TT > ET
CF = pd.DataFrame(C).ffill(axis=1).values
OO = np.where(np.isnan(O), CF, O)
HH = np.where(np.isnan(H), CF, H); LL = np.where(np.isnan(L), CF, L)
typ = np.nan_to_num((H + L + C) / 3 * V); vv = np.nan_to_num(V)
cv = np.cumsum(vv, 1); VWAP = np.where(cv > 0, np.cumsum(typ, 1) / np.maximum(cv, 1e-9), CF)
VWAP_PREV = np.concatenate([VWAP[:, :1], VWAP[:, :-1]], 1)
LA = np.where(AFT, L, np.nan)
RUNMIN = np.minimum(np.fmin.accumulate(np.where(np.isnan(LA), np.inf, LA), axis=1), E)
RUNMIN_PREV = np.where(AFT, np.concatenate([E, RUNMIN[:, :-1]], 1), E)
HA = np.where(AFT & ~np.isnan(H), H, -np.inf)
YEAR = meta.date.dt.year.values
def first(mask):
    hit = mask.any(1); return np.where(hit, mask.argmax(1), NO)
def at(X, i): return X[AR, np.minimum(i, NT - 1)]
def capf(px): return np.where(px >= PC[:, 0] * (1 + FORCE), PC[:, 0] * (1 + FORCE) * (1 + FORCE_SLIP), px)

def leg(i_in, px_in, stop=None, sig=None, same_min=False, t_end=270):
    """一腳空單: i_in 進場分鐘 (NO = 沒進場), px_in 進場價。之後碰停損 (預設區間高) / sig 市價出 / 收盤出。
    回傳 (報酬% (沒進場=0), 出場分鐘, 出場是否盤中)。"""
    has = i_in < NO; ii = i_in[:, None]
    act = (TT > ii) | ((TT == ii) & same_min)
    lvl = np.broadcast_to(RH, (N, NT)) if stop is None else np.minimum(np.broadcast_to(RH, (N, NT)), stop)
    i_s = first(act & (H > lvl) & (TT <= t_end))
    lv = at(lvl, i_s); o = at(O, i_s)
    px_s = capf(np.maximum(lv, np.where(np.isnan(o), lv, o)) * (1 + STOP_SLIP))
    i_g = np.full(N, NO) if sig is None else first(sig & (TT > ii) & (TT < t_end))
    px_g = at(CF, i_g) * (1 + SLIP)
    px_e = CF[:, t_end]
    idx = np.stack([i_s, i_g, np.full(N, t_end)]); k = np.argmin(idx, 0)
    px = np.choose(k, [px_s, px_g, px_e])
    r = np.where(has, (-(px / np.where(has, px_in, 1) - 1) - COST) * 100, 0.0)
    return r, np.where(has, idx.min(0), -1), has & (k != 2)

class Plan:
    """一筆交易的所有腳: 加總報酬 (以第一筆 1 份為單位)、每分鐘部位 (算同時曝險)、市價成交次數 (滑 1 檔敏感度)。"""
    def __init__(s): s.r = np.zeros(N); s.Q = np.zeros((N, NT)); s.nmk = np.zeros(N); s.size = np.zeros(N)
    def add(s, size, i_in, out, mkt_in):
        r, i_out, intr = out; has = i_in < NO; size = np.broadcast_to(np.asarray(size, float), (N,))
        s.r += size * r
        s.Q += size[:, None] * ((TT >= i_in[:, None]) & (TT <= i_out[:, None]) & has[:, None])
        s.nmk += size * has * (mkt_in.astype(float) + intr); s.size += size * has
        return s

base_out = leg(ET[:, 0], E[:, 0])
P(f"(自我檢查: 本程式現行 vs 33 回測每筆差 最大 {np.abs(base_out[0] - meta.s_r.values).max():.3f}%)")
def base_plan(size=1.0, stop=None):
    return Plan().add(size, ET[:, 0], leg(ET[:, 0], E[:, 0], stop=stop), np.ones(N, bool))

# ---------------- 統計 ----------------
DCODE = pd.DatetimeIndex(meta.date).normalize(); DAYS_T, dcode = np.unique(DCODE.values, return_inverse=True)
def daily(r, wt=None):
    wt = WT if wt is None else wt
    s = pd.Series(wt * r / 10, index=meta.date).groupby(level=0).sum()
    return s.reindex(ALLDAYS, fill_value=0.0)
def mdd_month(dl):
    out = {}
    for m, x in dl.groupby(dl.index.to_period("M")):
        eq = np.concatenate([[0.0], x.cumsum().values]); out[m] = (eq - np.maximum.accumulate(eq)).min() * 3
    return pd.Series(out)
def st(dl):
    eq = dl.cumsum(); mdd = (eq - eq.cummax()).min() * 3
    mm = mdd_month(dl); mret = dl.groupby(dl.index.to_period("M")).sum() * 3; sd = dl.std()
    return dict(ann=dl.mean() * 750, sh=dl.mean() / sd * np.sqrt(250) if sd > 0 else 0, mdd=mdd, mm_mean=mm.mean(), mm_worst=mm.min(),
                worstm=mret.min(), score=(dl.mean() * 750) / max(-mm.mean(), 1e-9))
def peak_grids(plan, wt=None):
    wt = WT if wt is None else wt
    G = np.zeros((len(DAYS_T), NT)); np.add.at(G, dcode, wt[:, None] * plan.Q)
    pk = G.max(1); return pk.mean(), np.percentile(pk, 95), pk.max()
def sub(dl, years): return dl[dl.index.year.isin(years)]
BASE = base_plan(); B_ST = st(daily(BASE.r))
HD = ["| 方法 | 每筆勝率% | 年均 (萬) | Sharpe | 最大回撤 | 平均月回撤 | 最差月回撤 | 最差月損益 | 同月回撤下年均 | 同最大回撤下年均 | 平均加碼份數 | 每日最高曝險 平均 / 最大 (格) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
def line(nm, plan, wt=None, mask=None, dl=None):
    r = plan.r; dl = daily(r, wt) if dl is None else dl; s = st(dl); m = np.ones(N, bool) if mask is None else mask
    win = (r[m] > 0).mean() * 100
    eq_mm = s["ann"] * B_REF["mm_mean"] / s["mm_mean"] if s["mm_mean"] < 0 else np.nan
    eq_dd = s["ann"] * B_REF["mdd"] / s["mdd"] if s["mdd"] < 0 else np.nan
    pk = peak_grids(plan, wt)
    extra = (plan.size * (1 if wt is None else wt / WT))[m].mean() - 1
    return (f"| {nm} | {win:.0f} | {s['ann']:.0f} | {s['sh']:.2f} | {s['mdd']:.0f} | {s['mm_mean']:.1f} | {s['mm_worst']:.1f} | {s['worstm']:.1f} | "
            f"{eq_mm:.0f} | {eq_dd:.0f} | {extra:+.2f} | {pk[0]:.1f} / {pk[2]:.1f} |")
B_REF = B_ST

# ---------------- 規則 ----------------
PLANS = {}   # 名稱 → (族群, Plan, 權重或 None)
def reg(fam, nm, plan, wt=None): PLANS[nm] = (fam, plan, wt)
reg("基準", "現行", BASE)
for k in (1.25, 1.5): reg("基準", f"直接加大部位 ×{k}", base_plan(k))

# A 獲利加碼
for X in (1, 2, 3):
    lvl = E * (1 - X / 100)
    i_a = first(AFT & (L < lvl) & (TT <= ADD_DL)); px_a = np.minimum(lvl[:, 0], at(OO, i_a)) * (1 - SLIP)
    on = TT > i_a[:, None]
    for s_ in (0.5, 1.0):
        reg("A", f"A 賺 {X}% 加空 {s_} 份 (停損區間高)", Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_a, leg(i_a, px_a), np.ones(N, bool)))
        be = np.where(on, E, np.inf)
        reg("A", f"A 賺 {X}% 加空 {s_} 份, 全部停損移到進場價", Plan().add(1, ET[:, 0], leg(ET[:, 0], E[:, 0], stop=be), np.ones(N, bool))
            .add(s_, i_a, leg(i_a, px_a, stop=be), np.ones(N, bool)))
    if X == 2:
        reg("A", "(對照) 只做 賺 2% 停損移到進場價, 不加碼", Plan().add(1, ET[:, 0], leg(ET[:, 0], E[:, 0], stop=np.where(on, E, np.inf)), np.ones(N, bool)))

# B 二次跌破加碼
BNC = np.where(AFT, (HH - RUNMIN_PREV) / E * 100, 0)
BNC_MAX_PREV = np.concatenate([np.zeros((N, 1)), np.maximum.accumulate(BNC, 1)[:, :-1]], 1)
HMAX = np.maximum.accumulate(HA, 1)
for b in (1, 2, 3):
    i_b = first(AFT & (BNC_MAX_PREV >= b) & (L < RUNMIN_PREV) & (TT <= ADD_DL))
    px_b = np.minimum(at(RUNMIN_PREV, i_b), at(OO, i_b)) * (1 - SLIP)
    hi = at(HMAX, np.maximum(i_b - 1, 0))[:, None] * np.ones((1, NT))
    for s_ in (0.5, 1.0):
        reg("B", f"B 反彈 ≥{b}% 後再破前低 加空 {s_} 份 (停損區間高)", Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_b, leg(i_b, px_b), np.ones(N, bool)))
        reg("B", f"B 反彈 ≥{b}% 後再破前低 加空 {s_} 份 (加碼停損=反彈高點)", Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_b, leg(i_b, px_b, stop=hi), np.ones(N, bool)))

# C 反彈加碼 (反彈到某價沒站回再空)
LEVELS = {"區間低": np.broadcast_to(RL, (N, NT)), "VWAP": VWAP_PREV, "區間中間": np.broadcast_to((RL + RH) / 2, (N, NT))}
for ln, LV in LEVELS.items():
    for x in (0, 2):
        armed = RUNMIN_PREV <= E * (1 - x / 100) - 1e-9 if x > 0 else RUNMIN_PREV < E - 1e-9
        ok = AFT & armed & (LV < RH) & (LV > RUNMIN_PREV) & (TT <= ADD_DL)
        i_t = first(ok & (HH >= LV)); px_t = np.maximum(at(LV, i_t), at(OO, i_t))
        i_r = first(ok & (HH >= LV) & (CF < LV)); px_r = at(CF, i_r) * (1 - SLIP)
        for s_ in (0.5, 1.0):
            reg("C", f"C 跌 ≥{x}% 後反彈碰{ln} 限價加空 {s_} 份", Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_t, leg(i_t, px_t, same_min=True), np.zeros(N, bool)))
            reg("C", f"C 跌 ≥{x}% 後反彈碰{ln}收不上去 加空 {s_} 份", Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_r, leg(i_r, px_r), np.ones(N, bool)))

# D 減碼再補回
for act in (0, 2):
    for r_ in (1.5, 2, 3):
        cov = AFT & (RUNMIN <= E * (1 - act / 100)) & (CF >= RUNMIN * (1 + r_ / 100)) & (TT <= 264)
        i_c = first(cov)
        rm_c = at(RUNMIN, i_c)[:, None]; px_c = at(CF, i_c) * (1 + SLIP)
        aftc = (TT > i_c[:, None]) & (TT <= ADD_DL)
        REF = {"不補回": None, "再破前低補回": ((L < rm_c), rm_c), "從回補價回落 1% 補回": ((L < px_c[:, None] * 0.99), px_c[:, None] * 0.99)}
        for f in (0.5, 1.0):
            keep = Plan().add(1 - f, ET[:, 0], base_out, np.ones(N, bool)) if f < 1 else Plan()
            cut = leg(ET[:, 0], E[:, 0], sig=cov)
            for rn, spec in REF.items():
                p = Plan(); p.r = keep.r.copy(); p.Q = keep.Q.copy(); p.nmk = keep.nmk.copy(); p.size = keep.size.copy()
                p.add(f, ET[:, 0], cut, np.ones(N, bool))
                if spec is not None:
                    trig, lv = spec
                    i_f = first(aftc & trig)
                    # 只有「是因為減碼訊號出場」才補回 (同分鐘碰停損算停損; 停損 / 收盤出場的不補)
                    by_sig = (i_c < base_out[1]) & (i_c < 270)
                    i_f = np.where(by_sig, i_f, NO)
                    px_f = np.minimum(at(lv if lv.shape[1] > 1 else np.broadcast_to(lv, (N, NT)), i_f), at(OO, i_f)) * (1 - SLIP)
                    p.add(f, i_f, leg(i_f, px_f), np.ones(N, bool))
                    p.size -= f * (i_f < NO)          # 補回不算加碼份數
                reg("D", f"D 賺 ≥{act}% 後反彈 {r_}% 回補 {'一半' if f < 1 else '全部'}, {rn}", p)

# E 依強弱調整初始部位
for g in (3, 4, 5):
    wt = WT * np.where(meta.gap.values >= g / 100, 1.5, 1.0); reg("E", f"E 跳空 ≥{g}% 的 ×1.5", BASE, wt)
for amt in (3, 5):
    wt = WT * np.where(meta.o5_amt.values >= amt * 1e8, 1.5, 1.0); reg("E", f"E 開盤 5 分成交額 ≥{amt} 億的 ×1.5", BASE, wt)
print(f"規則完成 {len(PLANS)} 組 ({time.time()-T0:.0f}s)", flush=True)

# F RL 加碼: 在狀態集合裡第一次出現時市價加空半份 (停損區間高 / 收盤出)
TB = np.digitize(TT + 0 * ET, [30, 60, 90, 120, 180, 240]); PNL = (E - CF) / E * 100
STATE = ((TB * 7 + np.digitize(PNL, [-1, 0, 1, 2, 3, 5])) * 4 + np.digitize((CF - RUNMIN) / E * 100, [0.5, 1, 2])) * 2 + (CF > VWAP).astype(int)
NS = 392
r0, i_x0, _ = base_out
# 第 u 分鐘收盤空 → 停損區間高與第一筆相同, 所以出場價 = 第一筆的出場價 (u 要早於第一筆出場)
px_x0 = E[:, 0] * (1 - (r0 / 100 + COST))   # 由 r0 反推第一筆出場價
ADDV = (-(px_x0[:, None] / (CF * (1 - SLIP)) - 1) - COST) * 100
DEC = AFT & (TT < i_x0[:, None]) & (TT <= ADD_DL)
def train_add(mask, margin, min_trades=60):
    rows = np.where(mask)[0]; rr, cc = np.nonzero(DEC[rows])
    df = pd.DataFrame(dict(n=rows[rr], s=STATE[rows][rr, cc], v=ADDV[rows][rr, cc], w=WT[rows][rr]))
    df = df.drop_duplicates(["n", "s"])                               # 每筆每狀態只看第一次
    df["vw"] = df.v * df.w
    g = df.groupby("s").agg(vw=("vw", "sum"), w=("w", "sum"), nt=("n", "nunique")); g["m"] = g.vw / g.w
    pol = np.zeros(NS, bool); pol[g[(g.nt >= min_trades) & (g.m > margin)].index.values] = True; return pol
def apply_add(pol, s_=0.5):
    i_f = first(DEC & pol[STATE]); px = at(CF, i_f) * (1 - SLIP)
    return Plan().add(1, ET[:, 0], base_out, np.ones(N, bool)).add(s_, i_f, leg(i_f, px), np.ones(N, bool))
TEST_YEARS = [y for y in range(2022, 2027) if (YEAR == y).any()]
RLP = {}
for margin in (0.3, 0.6, 1.0):
    p = Plan(); p.r = BASE.r.copy(); p.Q = BASE.Q.copy(); p.nmk = BASE.nmk.copy(); p.size = BASE.size.copy()
    nst = []
    for y in TEST_YEARS:
        pol = train_add(YEAR < y, margin); q = apply_add(pol); m = YEAR == y; nst.append(int(pol.sum()))
        p.r[m] = q.r[m]; p.Q[m] = q.Q[m]; p.nmk[m] = q.nmk[m]; p.size[m] = q.size[m]
    RLP[f"F RL 加碼半份 (門檻 {margin}%, 走動式)"] = (p, nst)
pol_all = train_add(np.ones(N, bool), 0.6)
print(f"RL 完成 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 報表 ----------------
P("# 盤中加碼 / 減碼 / 補回 (新規則: 品質篩選 + 權重)")
P("")
P(f"期間 {meta.date.min().date()} ~ {meta.date.max().date()}, {N:,} 筆, 300 萬本金, 每格 30 萬。每腳成本 0.23%, 市價 / 觸價另扣 0.1%; 停損保留 (全部一起出)。")
P("「同月回撤下年均」= 把部位縮放到和現行一樣的平均月回撤時的年均 (部位越大損益、回撤等比放大); 要 > 現行才是真的比「直接加大部位」好。「同最大回撤下年均」同理。")
DLS = {nm: daily(pl.r, wt) for nm, (fam, pl, wt) in PLANS.items()}
for fam, title in [("基準", "基準"), ("A", "A 獲利加碼"), ("B", "B 二次跌破加碼 (拉上來又殺下去)"), ("C", "C 反彈加碼"), ("D", "D 減碼再補回"), ("E", "E 依強弱調整初始部位")]:
    P(""); P(f"## 全期: {title}"); P(""); [P(h) for h in HD]
    for nm, (f_, pl, wt) in PLANS.items():
        if f_ == fam: P(line(nm, pl, wt, dl=DLS[nm]))

# 走動式
def pick(train_years, pool):
    best, bs = "現行", st(sub(DLS["現行"], train_years))["score"]
    for nm in pool:
        s = st(sub(DLS[nm], train_years))
        if s["score"] > bs: best, bs = nm, s["score"]
    return best
def wf(pool):
    parts, ch = [], []
    r = BASE.r.copy(); wt = WT.copy(); Q = BASE.Q.copy(); size = BASE.size.copy()
    for y in TEST_YEARS:
        k = pick(list(range(2020, y)), pool); ch.append(f"{y}: {k}"); m = YEAR == y
        fam, pl, w_ = PLANS[k]; r[m] = pl.r[m]; Q[m] = pl.Q[m]; size[m] = pl.size[m]
        if w_ is not None: wt[m] = w_[m]
    p = Plan(); p.r = r; p.Q = Q; p.size = size
    return p, wt, ch
B_REF = st(sub(DLS["現行"], TEST_YEARS))
OOS = YEAR >= TEST_YEARS[0]
def oline(nm, pl, wt=None):
    dl = sub(daily(pl.r, wt), TEST_YEARS); return line(nm, pl, wt, mask=OOS, dl=dl)
P(""); P(f"## 走動式樣本外 ({TEST_YEARS[0]}~{TEST_YEARS[-1]}): 每年只用之前年份挑該族群最好的參數 (標準 年均/|平均月回撤|, 要贏現行)")
P(""); [P(h) for h in HD]
P(oline("**現行**", BASE))
for k in (1.25, 1.5): P(oline(f"直接加大部位 ×{k}", PLANS[f"直接加大部位 ×{k}"][1]))
CH = []
for fam, title in [("A", "A 獲利加碼"), ("B", "B 二次跌破加碼"), ("C", "C 反彈加碼"), ("D", "D 減碼再補回"), ("E", "E 依強弱初始部位")]:
    pool = [nm for nm, v in PLANS.items() if v[0] == fam]
    pl, wt, ch = wf(pool); P(oline(title, pl, wt)); CH.append(f"- {title}: " + "; ".join(ch))
pl, wt, ch = wf([nm for nm, v in PLANS.items() if v[0] in "ABCDE"]); P(oline("A~E 一起挑", pl, wt)); CH.append("- A~E 一起挑: " + "; ".join(ch))
for nm, (pl, nst) in RLP.items(): P(oline(nm, pl)); CH.append(f"- {nm}: 每年學到加碼的狀態數 {nst}")
P(""); P("各年選到:"); [P(c) for c in CH]
P(f"- (RL 全期訓練 門檻 0.6%: 學到 {int(pol_all.sum())} 個加碼狀態)")

# 樣本外: 每族群全部參數 (看是不是整族都好, 不是只有挑出來那個)
B_REF = st(sub(DLS["現行"], TEST_YEARS))
P(""); P("## 樣本外 2022~2026 每組參數的「同月回撤下年均」分布 (現行 = {:.0f} 萬)".format(B_REF["ann"]))
P(""); P("| 族群 | 組數 | 贏現行的組數 | 最好 | 中位 | 最差 |"); P("|---|---|---|---|---|---|")
for fam in "ABCDE":
    v = []
    for nm, (f_, pl, wt) in PLANS.items():
        if f_ != fam: continue
        s = st(sub(DLS[nm], TEST_YEARS)); v.append(s["ann"] * B_REF["mm_mean"] / s["mm_mean"])
    v = np.array(v); P(f"| {fam} | {len(v)} | {(v > B_REF['ann']).sum()} | {v.max():.0f} | {np.median(v):.0f} | {v.min():.0f} |")

# 前 8 名 (全期 同月回撤下年均) 的逐年與滑 1 檔
B_REF = B_ST
rank = sorted([nm for nm, v in PLANS.items() if v[0] in "ABCDE"], key=lambda nm: -st(DLS[nm])["score"])[:8]
YS = sorted(set(YEAR))
P(""); P("## 全期 年均/平均月回撤 前 8 名: 逐年 年均 (萬) / 平均月回撤")
P(""); P("| 方法 | " + " | ".join(str(y) for y in YS) + " |"); P("|---|" + "---|" * len(YS))
for nm in ["現行"] + rank:
    dl = DLS[nm]; P(f"| {nm} | " + " | ".join(f"{st(sub(dl, [y]))['ann']:.0f} / {st(sub(dl, [y]))['mm_mean']:.1f}" for y in YS) + " |")
P(""); P("## 滑 1 檔 (每次市價 / 觸價成交各不利 1 檔; 限價與收盤不滑)"); P(""); [P(h) for h in HD]
for nm in ["現行"] + rank:
    fam, pl, wt = PLANS[nm]; q = Plan(); q.r = pl.r - TKP * 100 * pl.nmk; q.Q = pl.Q; q.size = pl.size
    P(line(nm, q, wt))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
