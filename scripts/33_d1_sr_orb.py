#!/usr/bin/env python3
"""研究: 先用 D1 找壓力 / 支撐, 對 M1 開盤 5 分區間突破 (ORB5) 當沖有沒有幫助。只讀資料、印報告, 不改線上規則。
資料: HF db/d1 (日K), db/m1 (分K), db/info/info.parquet; 對照 db/orb/history/trades.parquet (定案規則回放)。

D1 壓力 / 支撐 (全部只用今天以前的日K):
  - 擺盪高 / 低點 (pivot): 該日高 (低) 是前後各 K=5 根內最高 (最低), 第 i+5 日收盤後才確認; 只取近 250 日。
  - 另加: 昨高、昨低、昨收、前 20 / 60 / 250 日最高 / 最低。
  壓力 = 參考價之上最近的一條; 支撐 = 參考價之下最近的一條。距離以 % 表示。

交易 (每檔每日, 不設每日檔數上限, 看每筆效果; 最後再做每日最多 10 檔的組合):
  空 (現行策略的骨架): 09:05~12:00 第一根跌破開盤 5 分低 → 進 min(區間低, 該分開); 反彈破區間高停損 max(區間高, 該分開)×1.001;
      +8% 強制回補 (+0.2%)、收盤鎖漲停 −7%; 否則 13:30 收盤。
  多 (對稱): 第一根突破開盤 5 分高 → 進 max(區間高, 該分開); 跌破區間低停損 min(區間低, 該分開)×0.999; −8% 強制賣、收盤鎖跌停 −7%。
  成本 0.23%。排除: 開盤 5 分無量 / 處置股一字線 / 開盤 5 分量 < 日均 5% / 5 分內摸漲停或 +8% (空) / 摸跌停或 −8% (多) / 進場價已在漲跌停。
用法: python3 scripts/33_d1_sr_orb.py [--start 2020-01] [--out stats/d1_sr_orb.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, HF_PREFIX, COST, EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2020-01")
ap.add_argument("--out", default="stats/d1_sr_orb.md")
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
hf = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset")
FORCE, LOCK, STOP_SLIP, FORCE_SLIP, W, DEADLINE = 0.08, 0.07, 0.001, 0.002, 5, 180
T0 = time.time()

files = set(HfApi().list_repo_files(HF_REPO, repo_type="dataset"))
d1m = sorted(f for f in files if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:13] >= f"{int(a.start[:4]) - 2}_01")
m1m = sorted(f for f in files if f.startswith("db/m1/") and f.endswith(".parquet") and f[6:13] >= a.start.replace("-", "_"))
print(f"d1 {len(d1m)} 月 ({d1m[0]}~{d1m[-1]}), m1 {len(m1m)} 月 ({m1m[0] if m1m else '-'}~{m1m[-1] if m1m else '-'})")

# ---------------- D1 ----------------
info = pd.read_parquet(hf("db/info/info.parquet"))
d = pd.concat([pd.read_parquet(hf(f), columns=["stock_id", "date", "open", "high", "low", "close", "volume"]) for f in d1m], ignore_index=True)
d["stock_id"] = d.stock_id.astype(str); d["date"] = pd.to_datetime(d.date).dt.normalize()
d = d.drop_duplicates(["stock_id", "date"], keep="last").merge(info[["stock_id", "market_type"]].astype({"stock_id": str}), on="stock_id", how="left")
d = d[d.market_type.isin(["twse", "tpex"]) & d.stock_id.str.fullmatch(r"\d{4}") & ~d.stock_id.str.startswith("00")]
d = d[(d.close > 0) & (d.high > 0)].sort_values(["stock_id", "date"]).reset_index(drop=True)
for c in ["open", "high", "low", "close", "volume"]: d[c] = d[c].astype("float64")
g = d.groupby("stock_id", sort=False)
d["pc"] = g.close.shift(1); d["ph"] = g.high.shift(1); d["pl"] = g.low.shift(1)
d["rng"] = (d.high - d.low) / d.pc
d["dvv"] = d.close * d.volume
sh = lambda s, n, f: g[s].transform(lambda x: getattr(x.shift(1).rolling(n, min_periods=n), f)())
d["atr20"] = sh("rng", 20, "mean"); d["dv20"] = sh("dvv", 20, "mean"); d["vol20"] = sh("volume", 20, "mean")
for n in [20, 60, 250]:
    d[f"h{n}"] = sh("high", n, "max"); d[f"l{n}"] = sh("low", n, "min")
d["gap"] = d.open / d.pc - 1
d["di"] = g.cumcount()
liq = d[(d.dv20 >= 1e8) & d.pc.notna() & d.gap.notna()]
mkt = liq.groupby("date").gap.mean().rename("mkt_gap")
d = d.merge(mkt, on="date", how="left")
print(f"D1 完成 {len(d):,} 列 ({time.time()-T0:.0f}s)", flush=True)

# 擺盪點: 每檔的 (確認日 index, 價)
K = a.k
piv = {}
for sid, x in d.groupby("stock_id", sort=False):
    hi, lo = x.high.values, x.low.values; n = len(x)
    if n < 2 * K + 1: piv[sid] = (np.array([], int), np.array([]), np.array([], int), np.array([])); continue
    from numpy.lib.stride_tricks import sliding_window_view as swv
    wh = swv(hi, 2 * K + 1).max(1); wl = swv(lo, 2 * K + 1).min(1)
    ih = np.where(hi[K:n - K] >= wh)[0] + K; il = np.where(lo[K:n - K] <= wl)[0] + K
    piv[sid] = (ih, hi[ih], il, lo[il])
print(f"pivot 完成 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- M1: 每檔每日 ORB5 多 / 空 結果 ----------------
U = d[(d.dv20 >= 1e8) & (d.pc >= 10) & d.atr20.notna()][["stock_id", "date", "pc", "open", "gap", "atr20", "dv20", "vol20", "mkt_gap", "di",
                                                          "ph", "pl", "h20", "l20", "h60", "l60", "h250", "l250", "close"]].rename(columns={"open": "o", "close": "c_d1"})
U["lu"] = U.pc * 1.10; U["ld"] = U.pc * 0.90
R = []
for f in m1m:
    ym = f[6:13]; t1 = time.time()
    m = pd.read_parquet(hf(f), columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    m["stock_id"] = m.stock_id.astype(str); ts = pd.to_datetime(m.date)
    m["t"] = (ts.dt.hour * 60 + ts.dt.minute - 540).astype(int); m["date"] = ts.dt.normalize()
    m = m[(m.t >= 0) & (m.t <= 270)]
    u = U[U.date.dt.strftime("%Y_%m") == ym]
    m = m.merge(u[["stock_id", "date"]], on=["stock_id", "date"])
    if not len(m): print(ym, "無資料"); continue
    m = m.sort_values(["stock_id", "date", "t"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close"]: m[c] = m[c].astype("float64")
    key = ["stock_id", "date"]
    b5 = m[m.t < W].assign(flat=lambda x: (x.open == x.high) & (x.high == x.low) & (x.low == x.close))
    rg = b5.groupby(key).agg(rh=("high", "max"), rl=("low", "min"), o5_v=("volume", "sum"), o5_n=("t", "size"), o5_flat=("flat", "all")).reset_index()
    cl = m.groupby(key).agg(close=("close", "last"), dlo=("low", "min"), dhi=("high", "max")).reset_index()
    x = u.merge(rg, on=key).merge(cl, on=key)
    m = m.merge(x[key + ["rh", "rl", "pc"]], on=key)
    aft = m[m.t >= W]
    out = x.copy()
    for side in ["s", "l"]:
        cond = (aft.low < aft.rl) if side == "s" else (aft.high > aft.rh)
        e = aft[cond & (aft.t <= DEADLINE)].groupby(key).first()[["t", "open"]].rename(columns={"t": f"{side}_et", "open": f"{side}_eo"}).reset_index()
        mm = aft.merge(e, on=key)
        mm = mm[mm.t > mm[f"{side}_et"]]
        if side == "s":
            stp = mm[mm.high > mm.rh].groupby(key).first()[["t", "open"]]
            frc = mm[mm.high >= mm.pc * (1 + FORCE)].groupby(key).first()[["t", "open"]]
        else:
            stp = mm[mm.low < mm.rl].groupby(key).first()[["t", "open"]]
            frc = mm[mm.low <= mm.pc * (1 - FORCE)].groupby(key).first()[["t", "open"]]
        e = e.merge(stp.rename(columns={"t": f"{side}_st", "open": f"{side}_so"}).reset_index(), on=key, how="left")
        e = e.merge(frc.rename(columns={"t": f"{side}_ft", "open": f"{side}_fo"}).reset_index(), on=key, how="left")
        out = out.merge(e, on=key, how="left")
    R.append(out)
    print(f"{ym}: {len(x):,} 股票日 ({time.time()-t1:.0f}s, 累計 {time.time()-T0:.0f}s)", flush=True)
    try: Path(hf(f)).resolve().unlink()
    except Exception: pass
X = pd.concat(R, ignore_index=True)
del R

# 結算
X["o5_vr"] = X.o5_v * 1000 / X.vol20
X["bad"] = X.rh.isna() | ((X.o5_n <= 3) & X.o5_flat) | (X.o5_vr < 0.05)
# 空
s = X.s_et.notna()
X["s_px"] = np.minimum(X.rl, X.s_eo)
stop_px = np.maximum(X.rh, X.s_so) * (1 + STOP_SLIP)
force_first = X.s_ft.notna() & (X.s_st.isna() | (X.s_ft < X.s_st))   # 先到 +8% (未先停損; 停損價 ≥ +8% 時也算強制)
ex = np.where(X.s_st.notna(), stop_px, X.close)
ex = np.where(X.s_st.notna() & (stop_px >= X.pc * (1 + FORCE)), X.pc * (1 + FORCE) * (1 + FORCE_SLIP), ex)
ex = np.where(force_first, np.maximum(X.pc * (1 + FORCE), X.s_fo) * (1 + FORCE_SLIP), ex)
locked = X.s_st.isna() & ~force_first & (X.close >= X.lu * 0.999)
X["s_r"] = np.where(s, (-(ex / X.s_px - 1) - COST - np.where(locked, LOCK, 0)) * 100, np.nan)
X["s_why"] = np.select([~s, force_first, X.s_st.notna(), locked], ["", "強制", "停損", "鎖漲停"], "收盤")
X["s_ok"] = s & ~X.bad & (X.rh < X.lu * 0.999) & (X.rh < X.pc * (1 + FORCE)) & (X.s_px > X.ld * 1.001)
# 多
l = X.l_et.notna()
X["l_px"] = np.maximum(X.rh, X.l_eo)
stop_px = np.minimum(X.rl, X.l_so) * (1 - STOP_SLIP)
force_first = X.l_ft.notna() & (X.l_st.isna() | (X.l_ft < X.l_st))
ex = np.where(X.l_st.notna(), stop_px, X.close)
ex = np.where(X.l_st.notna() & (stop_px <= X.pc * (1 - FORCE)), X.pc * (1 - FORCE) * (1 - FORCE_SLIP), ex)
ex = np.where(force_first, np.minimum(X.pc * (1 - FORCE), X.l_fo) * (1 - FORCE_SLIP), ex)
locked = X.l_st.isna() & ~force_first & (X.close <= X.ld * 1.001)
X["l_r"] = np.where(l, ((ex / X.l_px - 1) - COST - np.where(locked, LOCK, 0)) * 100, np.nan)
X["l_why"] = np.select([~l, force_first, X.l_st.notna(), locked], ["", "強制", "停損", "鎖跌停"], "收盤")
X["l_ok"] = l & ~X.bad & (X.rl > X.ld * 1.001) & (X.rl > X.pc * (1 - FORCE)) & (X.l_px < X.lu * 0.999)
print(f"結算完成 {len(X):,} 股票日; 空 {X.s_ok.sum():,} 筆, 多 {X.l_ok.sum():,} 筆 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 最近的壓力 / 支撐 ----------------
def levels(sid, di, row):
    ih, vh, il, vl = piv.get(sid, (np.array([], int),) * 4)
    mh = (ih + K <= di - 1) & (ih >= di - 250); ml = (il + K <= di - 1) & (il >= di - 250)
    extra = [row.ph, row.pl, row.pc, row.h20, row.l20, row.h60, row.l60, row.h250, row.l250]
    return np.concatenate([vh[mh], vl[ml], np.array([v for v in extra if np.isfinite(v)])]), vh[mh]
def above(L, x):
    v = L[L > x * 1.001]; return v.min() if len(v) else np.nan
def below(L, x):
    v = L[L < x * 0.999]; return v.max() if len(v) else np.nan
Y = X[X.s_ok | X.l_ok].copy()
cols = {k: np.full(len(Y), np.nan) for k in ["res_o", "sup_s", "res_l", "sup_l", "res_s", "lvl_in"]}
for j, r in enumerate(Y.itertuples(index=False)):
    L, PH = levels(r.stock_id, r.di, r)
    cols["res_o"][j] = above(L, r.o)                                # 開盤之上最近壓力
    cols["lvl_in"][j] = float(((PH > r.rl) & (PH <= r.rh * 1.003)).any() or any(np.isfinite(v) and r.rl < v <= r.rh * 1.003 for v in [r.ph, r.h20, r.h60, r.h250]))
    if r.s_ok:
        cols["sup_s"][j] = below(L, r.s_px); cols["res_s"][j] = above(L, r.s_px)
    if r.l_ok:
        cols["res_l"][j] = above(L, r.l_px); cols["sup_l"][j] = below(L, r.l_px)
for k, v in cols.items(): Y[k] = v
Y["d_res_o"] = (Y.res_o / Y.o - 1) * 100
Y["d_sup_s"] = (1 - Y.sup_s / Y.s_px) * 100          # 空單進場價到下方支撐的空間
Y["d_res_l"] = (Y.res_l / Y.l_px - 1) * 100          # 多單進場價到上方壓力的空間
Y["d_res_s"] = (Y.res_s / Y.s_px - 1) * 100
Y["o_vs_h60"] = Y.o / Y.h60 - 1; Y["o_vs_h250"] = Y.o / Y.h250 - 1
Y["year"] = Y.date.dt.year; Y["exp"] = Y.date <= EXPLORE_END
Y["dow"] = Y.date.dt.dayofweek
Y.drop(columns=[c for c in Y.columns if c.endswith(("_eo", "_so", "_fo"))]).to_parquet(PROC / "d1_sr_orb.parquet", index=False)
print(f"壓力支撐完成 ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 報表 ----------------
def tstat(v):
    v = v.dropna(); return v.mean() / v.std() * np.sqrt(len(v)) if len(v) > 2 and v.std() > 0 else np.nan
def row(name, z, rc, why):
    r = z[rc]
    if len(r) < 30: return f"| {name} | {len(r)} | | | | | | | |"
    e, v = z.exp, ~z.exp
    return (f"| {name} | {len(r):,} | {r.mean():.3f} | {(r > 0).mean()*100:.0f} | {tstat(r):.1f} | {r[e].mean():.3f} / {r[v].mean():.3f} | "
            f"{(z[why] == '停損').mean()*100:.0f} | {(z[why].isin(['強制', '鎖漲停', '鎖跌停'])).mean()*100:.1f} | "
            + " ".join(f"{z[z.year == y][rc].mean():.2f}" if (z.year == y).sum() >= 20 else "–" for y in range(2020, 2027)) + " |")
HDR = ["| 分組 | 筆數 | 每筆% | 勝率% | t | 探索/驗證 每筆% | 停損% | 強制+鎖% | 2020~2026 每筆% |", "|---|---|---|---|---|---|---|---|---|"]
def table(title, z, rc, why, groups):
    P(""); P(f"### {title}"); P(""); [P(h) for h in HDR]
    P(row("全部", z, rc, why))
    for name, msk in groups: P(row(name, z[msk.reindex(z.index, fill_value=False)], rc, why))
def cuts(z, col, edges, fmt="{:g}"):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append((f"{col} {fmt.format(lo)} ~ {fmt.format(hi)}", (z[col] >= lo) & (z[col] < hi)))
    out.append((f"{col} 無 (NaN)", z[col].isna()))
    return out

hist = None
try:
    hist = pd.read_parquet(hf(f"{HF_PREFIX}/history/trades.parquet")); hist["date"] = pd.to_datetime(hist.date)
except Exception as e:
    print("history 讀不到:", e)

P("# D1 壓力 / 支撐 × M1 開盤 5 分區間突破 (ORB5)")
P("")
P(f"期間 {Y.date.min().date()} ~ {Y.date.max().date()}；探索 ≤{EXPLORE_END}、驗證之後。壓力 / 支撐 = 近 250 日擺盪高低點 (前後 {K} 根) + 昨高 / 昨低 / 昨收 + 20/60/250 日高低, 只用今天以前的日K。")
P("每筆含成本 0.23%、+8% 強制回補、收盤鎖漲跌停 −7%；停損 = 開盤 5 分區間另一端。表內每檔每日都算 (不設 10 檔上限)。")

# ===== A. 現行空單 =====
S = Y[Y.s_ok].copy()
pool = (S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300)
live = pool & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)
A = S[live]
P(""); P("## A. 現行策略 (跳空≥2% ∩ ATR≥3% ∩ 均額≥3億 ∩ 股價≤300 ∩ 非週一 ∩ 大盤跳空≥−1%, 跌破 5 分低放空)")
if hist is not None:
    P(f"對照: 線上回放 history/trades.parquet {len(hist):,} 筆 (每日 ≤10 檔), 每筆 {hist.r_pct.mean():.3f}%；本表同規則不設上限 {len(A):,} 筆。")
table("A1. 開盤位置 vs 前高 (開盤 / 前 N 日最高 − 1)", A, "s_r", "s_why", [
    ("開盤 > 250 日高 (創一年新高開)", A.o > A.h250), ("開盤 > 60 日高, ≤ 250 日高", (A.o > A.h60) & (A.o <= A.h250)),
    ("開盤 > 20 日高, ≤ 60 日高", (A.o > A.h20) & (A.o <= A.h60)), ("開盤 ≤ 20 日高 (開在區間內)", A.o <= A.h20),
    ("開盤 > 昨高 (真跳空)", A.o > A.ph), ("開盤 ≤ 昨高 (開在昨日 K 棒內)", A.o <= A.ph)])
table("A2. 開盤離上方最近壓力 (%)", A, "s_r", "s_why", cuts(A, "d_res_o", [0, 1, 2, 4, 8, 99]))
table("A3. 開盤 5 分區間內有日K壓力 (區間高碰到壓力後跌破)", A, "s_r", "s_why", [("區間內有壓力線", A.lvl_in == 1), ("區間內無壓力線", A.lvl_in == 0)])
table("A4. 進場價到下方最近支撐的空間 (%)", A, "s_r", "s_why", cuts(A, "d_sup_s", [0, 1, 2, 3, 5, 99]))
z = A.assign(d_pc=(A.s_px / A.pc - 1) * 100)
table("A5. 進場價離昨收 (跳空缺口) 的距離 (%)", z, "s_r", "s_why", cuts(z, "d_pc", [-99, 0, 1, 2, 3, 5, 99]))

# ===== B. 擴大: 不限跳空的空單 =====
B_ = S[pool & (S.dow != 0) & (S.mkt_gap >= -0.01)]
P(""); P("## B. 空單擴大 (同池但不限跳空), 看「跌破日K支撐」有沒有額外效果")
table("B1. 依跳空與是否跌破昨低 / 20 日低", B_, "s_r", "s_why", [
    ("跳空 <0", B_.gap < 0), ("跳空 0~2%", (B_.gap >= 0) & (B_.gap < 0.02)), ("跳空 ≥2% (現行)", B_.gap >= 0.02),
    ("跳空 <2% 且 進場 < 昨低", (B_.gap < 0.02) & (B_.s_px < B_.pl)), ("跳空 <2% 且 進場 < 20 日低", (B_.gap < 0.02) & (B_.s_px < B_.l20)),
    ("跳空 <2% 且 進場 ≥ 昨低", (B_.gap < 0.02) & (B_.s_px >= B_.pl))])
table("B2. 跳空 <2%: 進場到下方支撐空間 (%)", B_[B_.gap < 0.02], "s_r", "s_why", cuts(B_[B_.gap < 0.02], "d_sup_s", [0, 1, 2, 3, 5, 99]))

# ===== C. 多單 =====
Lg = Y[Y.l_ok].copy()
Lp = (Lg.dv20 >= 3e8) & (Lg.pc <= 300)
C = Lg[Lp]
P(""); P("## C. 做多: 突破開盤 5 分高買進, 跌破 5 分低停損 (均額≥3億, 股價≤300)")
table("C1. 依跳空 / ATR", C, "l_r", "l_why", [
    ("跳空 <−2%", C.gap < -0.02), ("跳空 −2~0%", (C.gap >= -0.02) & (C.gap < 0)), ("跳空 0~2%", (C.gap >= 0) & (C.gap < 0.02)), ("跳空 ≥2%", C.gap >= 0.02),
    ("ATR ≥3%", C.atr20 >= 0.03), ("ATR <3%", C.atr20 < 0.03)])
table("C2. 進場價相對日K壓力: 是否突破前高", C, "l_r", "l_why", [
    ("進場 > 250 日高 (突破一年新高)", C.l_px > C.h250), ("進場 > 60 日高", C.l_px > C.h60), ("進場 > 20 日高", C.l_px > C.h20),
    ("進場 > 昨高", C.l_px > C.ph), ("進場 ≤ 昨高", C.l_px <= C.ph), ("進場 ≤ 20 日高 (還在壓力下)", C.l_px <= C.h20)])
table("C3. 進場價到上方最近壓力的空間 (%) (無 = 上方沒有壓力, 創新高)", C, "l_r", "l_why", cuts(C, "d_res_l", [0, 1, 2, 3, 5, 99]))
table("C4. 進場到下方支撐 (停損附近有沒有支撐) (%)", C.assign(d_sup_l=(1 - C.sup_l / C.l_px) * 100), "l_r", "l_why",
      cuts(C.assign(d_sup_l=(1 - C.sup_l / C.l_px) * 100), "d_sup_l", [0, 1, 2, 3, 5, 99]))
z = C[(C.l_px > C.h60) & (C.gap < 0.02)]
table("C5. 突破 60 日高 ∩ 跳空 <2% (盤中真正突破日K壓力): 依 ATR / 大盤跳空", z, "l_r", "l_why", [
    ("ATR ≥3%", z.atr20 >= 0.03), ("ATR <3%", z.atr20 < 0.03), ("大盤跳空 ≥0", z.mkt_gap >= 0), ("大盤跳空 <0", z.mkt_gap < 0),
    ("進場時間 ≤09:15", z.l_et <= 15), ("進場時間 >09:15", z.l_et > 15)])

# ===== D. 組合 (每日最多 10 檔, 依進場時間) =====
def port(z, rc, et, name, cap=10):
    z = z.sort_values(["date", et]); z = z.groupby("date").head(cap)
    days = pd.Series(sorted(Y.date.unique()))
    if "dow" in z: pass
    dly = (z.groupby("date")[rc].sum() / cap).reindex(days, fill_value=0.0)
    dly.index = days
    eq = (1 + dly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    sh = dly.mean() / dly.std() * np.sqrt(250) if dly.std() > 0 else np.nan
    e = dly.index <= EXPLORE_END
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return f"| {name} | {len(z)/max((dly != 0).sum(),1):.1f} | {len(z):,} | {z[rc].mean():.3f} | {dly.mean():.3f} | {sh:.2f} | {mdd:.0f}% | {dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |"
P(""); P("## D. 組合: 每日最多 10 檔 (依進場時間先到先做), 每檔 1/10 資金, 無交易日計 0")
P(""); P("| 組合 | 有交易日均檔 | 筆數 | 每筆% | 日淨% | Sharpe | MDD | 探索/驗證 日淨% | 2020~2026 日淨% |"); P("|---|---|---|---|---|---|---|---|---|")
P(port(A, "s_r", "s_et", "現行空單 (A 全部)"))
for name, msk in [("A ∩ 開盤 ≤ 60 日高", A.o <= A.h60), ("A ∩ 開盤 > 60 日高", A.o > A.h60), ("A ∩ 區間內有壓力", A.lvl_in == 1),
                  ("A ∩ 開盤離壓力 <2%", A.d_res_o < 2), ("A ∩ 下方支撐空間 ≥2% 或無", ~(A.d_sup_s < 2)), ("A ∩ 下方支撐空間 <2%", A.d_sup_s < 2)]:
    P(port(A[msk], "s_r", "s_et", name))
P(port(C, "l_r", "l_et", "多: 全部 ORB5 向上突破"))
for name, msk in [("多: 進場 > 60 日高 ∩ 跳空 <2%", (C.l_px > C.h60) & (C.gap < 0.02)), ("多: 進場 > 250 日高", C.l_px > C.h250),
                  ("多: 上方壓力空間 ≥3% 或無", ~(C.d_res_l < 3)), ("多: 進場 ≤ 20 日高", C.l_px <= C.h20)]:
    P(port(C[msk], "l_r", "l_et", name))

# ===== E. 現行空單: 壓力訊號拿來排優先 / 調部位 (不是硬過濾) =====
def port_rule(z, flag, reserve=0, w_on=1.0, w_off=1.0, cap=10, name=""):
    """依進場時間先到先做; 非壓力單只在已進場檔數 < cap−reserve 時才做 (保留 reserve 格給壓力單)。部位: 壓力單 w_on 格、其他 w_off 格。"""
    z = z.sort_values(["date", "s_et"]); rows = []
    for dt, gdf in z.groupby("date", sort=False):
        n = 0
        for r, f in zip(gdf.s_r.values, gdf[flag].values):
            if n >= cap: break
            if not f and n >= cap - reserve: continue
            rows.append((dt, r, w_on if f else w_off)); n += 1
    t = pd.DataFrame(rows, columns=["date", "r", "w"])
    days = pd.Series(sorted(Y.date.unique()))
    dly = (t.assign(x=t.r * t.w).groupby("date").x.sum() / cap).reindex(days, fill_value=0.0); dly.index = days
    eq = (1 + dly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    e = dly.index <= EXPLORE_END
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return f"| {name} | {len(t):,} | {t.r.mean():.3f} | {dly.mean():.3f} | {dly.mean()/dly.std()*np.sqrt(250):.2f} | {mdd:.0f}% | {dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |"
A = A.assign(f_lvl=A.lvl_in == 1, f_gap=A.o > A.ph, f_any=(A.lvl_in == 1) | (A.o > A.h20))
nday = A.groupby("date").size()
P(""); P("## E. 壓力訊號當「優先序 / 部位權重」而不是硬過濾 (現行空單)")
P(f"同規則不設上限時, 有訊號的日子平均 {nday.mean():.1f} 檔跌破, {(nday > 10).mean()*100:.0f}% 的日子超過 10 檔 (第 11 檔起被丟掉)。")
P(""); P("| 規則 | 筆數 | 每筆% | 日淨% | Sharpe | MDD | 探索/驗證 日淨% | 2020~2026 日淨% |"); P("|---|---|---|---|---|---|---|---|")
P(port_rule(A, "f_lvl", 0, name="現行 (先到先做, 10 格)"))
for fl, lab in [("f_lvl", "區間內有壓力"), ("f_any", "區間內有壓力 或 開盤>20日高")]:
    for rv in [2, 4, 6]:
        P(port_rule(A, fl, rv, name=f"保留 {rv} 格給「{lab}」"))
    P(port_rule(A, fl, 0, 1.5, 0.75, name=f"「{lab}」1.5 格, 其他 0.75 格"))
    P(port_rule(A, fl, 0, 1.25, 0.75, name=f"「{lab}」1.25 格, 其他 0.75 格"))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
