#!/usr/bin/env python3
"""研究 40: 進場後什麼時候出場比較好 (選股 / 進場完全沿用現行規則, 只換出場)。只讀資料、印報告, 不改線上規則。

交易: 同 33/39 的「現行」: 跳空 ≥2%、ATR20 ≥3%、20 日均額 ≥3 億、昨收 10~300、非週一、大盘平均跳空 ≥−1%,
  09:05~12:00 第一根跌破開盤 5 分低 → 空 min(區間低, 該分開); 每日先到先做最多 10 檔、等權 (每檔 = 資金/10)。
  (線上另有區間寬 / 碰壓力加權, 這裡不加權; 比的是同一批交易換出場法的差別。)
出場 (進場那一分鐘不檢查, 同回測):
  - 一律保留現行停損: 反彈破區間高 → max(區間高, 該分開) ×1.001。
  - 規則出場 (盤中市價) 一律多扣 0.1% 滑價 (約一檔); 獲利目標是限價, 要「穿過」目標價才算成交, 不扣滑價。
  - 同一分鐘同時碰停損和其他出場, 一律當作先停損 (保守)。
  - 現行 = 停損或 13:30 收盤; 成本 0.23%。
走動式驗證 (walk-forward): 每個測試年只用之前年份挑參數 / 訓練, 2022~2026 串起來就是樣本外。
挑參數的標準 (Ray 2026-11-18: 以回撤小為準): 訓練期 年均 / |最大回撤| (Calmar) 最大, 且年均不低於現行的 70%。
RL: 批次 Q-learning (fitted Q / 政策迭代) — 狀態 = 時段 × 浮動損益 × 從最低點反彈 × 在 VWAP 上/下, 動作 = 續抱 / 平倉;
  價格路徑不受動作影響, 所以用歷史 1 分 K 重播即可精確算出每個狀態「現在平」vs「續抱並照目前政策走」的平均報酬。
用法: python3 scripts/40_exit_research.py [--start 2020-01] [--rebuild] [--out stats/exit_research.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, COST, EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2020-01")
ap.add_argument("--out", default="stats/exit_research.md")
ap.add_argument("--cache", default=str(PROC / "exit_paths.npz"))
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--build-only", action="store_true", help="只建快取 (給 41 用)")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
W, DEADLINE, NT = 5, 180, 271          # t = 分鐘 (09:00 = 0, 13:30 = 270)
STOP_SLIP, SLIP, FORCE, FORCE_SLIP = 0.001, 0.001, 0.08, 0.002
T0 = time.time()

# =============================== 1. 建立每筆交易進場後的 1 分 K 路徑 ===============================
def build():
    from huggingface_hub import hf_hub_download, HfApi
    hf = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset")
    files = set(HfApi().list_repo_files(HF_REPO, repo_type="dataset"))
    d1m = sorted(f for f in files if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:13] >= f"{int(a.start[:4]) - 1}_01")
    m1m = sorted(f for f in files if f.startswith("db/m1/") and f.endswith(".parquet") and f[6:13] >= a.start.replace("-", "_"))
    print(f"d1 {len(d1m)} 月, m1 {len(m1m)} 月 ({m1m[0]}~{m1m[-1]})", flush=True)
    info = pd.read_parquet(hf("db/info/info.parquet"))
    d = pd.concat([pd.read_parquet(hf(f), columns=["stock_id", "date", "open", "high", "low", "close", "volume"]) for f in d1m], ignore_index=True)
    d["stock_id"] = d.stock_id.astype(str); d["date"] = pd.to_datetime(d.date).dt.normalize()
    d = d.drop_duplicates(["stock_id", "date"], keep="last").merge(info[["stock_id", "market_type"]].astype({"stock_id": str}), on="stock_id", how="left")
    d = d[d.market_type.isin(["twse", "tpex"]) & d.stock_id.str.fullmatch(r"\d{4}") & ~d.stock_id.str.startswith("00")]
    d = d[(d.close > 0) & (d.high > 0)].sort_values(["stock_id", "date"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]: d[c] = d[c].astype("float64")
    g = d.groupby("stock_id", sort=False)
    d["pc"] = g.close.shift(1)
    d["rng"] = (d.high - d.low) / d.pc; d["dvv"] = d.close * d.volume
    sh = lambda s, n: g[s].transform(lambda x: x.shift(1).rolling(n, min_periods=n).mean())
    d["atr20"] = sh("rng", 20); d["dv20"] = sh("dvv", 20); d["vol20"] = sh("volume", 20)
    d["gap"] = d.open / d.pc - 1
    d["pchg"] = d.pc / g.close.shift(2) - 1                                   # 昨日漲跌
    up = (g.close.diff() > 0).astype(int)
    d["streak"] = up.groupby(d.stock_id).transform(lambda x: x.groupby((x == 0).cumsum()).cumsum().shift(1)).fillna(0)   # 到昨天連漲幾天
    d["h20"] = g.high.transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())
    d["d_h20"] = d.open / d.h20 - 1                                           # 開盤 vs 前 20 日最高
    liq = d[(d.dv20 >= 1e8) & d.pc.notna() & d.gap.notna()]
    d = d.merge(liq.groupby("date").gap.mean().rename("mkt_gap"), on="date", how="left")
    U = d[(d.dv20 >= 3e8) & (d.pc >= 10) & (d.pc <= 300) & (d.atr20 >= 0.03) & (d.gap >= 0.02) & (d.mkt_gap >= -0.01)
          & (d.date.dt.dayofweek != 0)][["stock_id", "date", "pc", "vol20", "gap", "atr20", "pchg", "streak", "d_h20", "mkt_gap"]].copy()
    print(f"D1 完成, 候選股票日 {len(U):,} ({time.time()-T0:.0f}s)", flush=True)
    META, ARR, DAYS = [], {k: [] for k in "ohlcv"}, []
    for f in m1m:
        ym = f[6:13]; t1 = time.time()
        m = pd.read_parquet(hf(f), columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
        m["stock_id"] = m.stock_id.astype(str); ts = pd.to_datetime(m.date)
        m["t"] = (ts.dt.hour * 60 + ts.dt.minute - 540).astype(int); m["date"] = ts.dt.normalize()
        m = m[(m.t >= 0) & (m.t <= 270)]
        DAYS += sorted(m.date.unique())
        u = U[U.date.dt.strftime("%Y_%m") == ym]
        key = ["stock_id", "date"]
        m = m.merge(u[key], on=key)
        if not len(m): print(ym, "無候選"); continue
        m = m.drop_duplicates(key + ["t"], keep="last").sort_values(key + ["t"]).reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]: m[c] = m[c].astype("float64")
        b5 = m[m.t < W].assign(flat=lambda x: (x.open == x.high) & (x.high == x.low) & (x.low == x.close))
        rg = b5.groupby(key).agg(rh=("high", "max"), rl=("low", "min"), o5_v=("volume", "sum"), o5_n=("t", "size"), o5_flat=("flat", "all")).reset_index()
        x = u.merge(rg, on=key)
        mm = m.merge(x[key + ["rl"]], on=key)
        aft = mm[(mm.t >= W) & (mm.t <= DEADLINE) & (mm.low < mm.rl)]
        e = aft.groupby(key).first()[["t", "open"]].rename(columns={"t": "et", "open": "eo"}).reset_index()
        x = x.merge(e, on=key)
        x["E"] = np.minimum(x.rl, x.eo)
        bad = ((x.o5_n <= 3) & x.o5_flat) | (x.o5_v * 1000 / x.vol20 < 0.05)
        ok = ~bad & (x.rh < x.pc * 1.1 * 0.999) & (x.rh < x.pc * (1 + FORCE)) & (x.E > x.pc * 0.9 * 1.001)
        x = x[ok].reset_index(drop=True)
        if not len(x): continue
        x["i"] = np.arange(len(x))
        z = m.merge(x[key + ["i"]], on=key)
        for k, c in zip("ohlcv", ["open", "high", "low", "close", "volume"]):
            A = np.full((len(x), NT), np.nan, dtype="float32"); A[z.i.values, z.t.values] = z[c].values; ARR[k].append(A)
        META.append(x[key + ["pc", "rh", "rl", "et", "E", "gap", "atr20", "pchg", "streak", "d_h20", "mkt_gap"]])
        print(f"{ym}: {len(x)} 筆 ({time.time()-t1:.0f}s, 累計 {time.time()-T0:.0f}s)", flush=True)
        try: Path(hf(f)).resolve().unlink()
        except Exception: pass
    meta = pd.concat(META, ignore_index=True)
    np.savez_compressed(a.cache, **{k: np.concatenate(v) for k, v in ARR.items()}, days=np.array(sorted(set(DAYS)), dtype="datetime64[ns]"))
    meta.to_parquet(Path(a.cache).with_suffix(".parquet"), index=False)

if a.rebuild or not Path(a.cache).exists(): build()
if a.build_only: sys.exit(0)
Z = np.load(a.cache); meta = pd.read_parquet(Path(a.cache).with_suffix(".parquet"))
ALLDAYS = pd.DatetimeIndex(Z["days"])
# 只保留每日先到先做的前 10 檔 (出場法不影響誰進場: 線上名額看「已進場」不看「同時持有」)
meta["j"] = np.arange(len(meta))
meta = meta.sort_values(["date", "et", "stock_id"]); meta = meta.groupby("date").head(10).sort_values("j")
J = meta.j.values
O, H, L, C, V = (Z[k][J].astype("float64") for k in "ohlcv")
meta = meta.reset_index(drop=True)
N = len(meta)
E = meta.E.values[:, None]; RH = meta.rh.values[:, None]; RL = meta.rl.values[:, None]; PC = meta.pc.values[:, None]; ET = meta.et.values[:, None]
TT = np.arange(NT)[None, :]
AFT = TT > ET                                         # 進場後的分鐘
CF = pd.DataFrame(C).ffill(axis=1).values             # 收盤價前值填補
last_close = CF[:, -1]
typ = np.nan_to_num((H + L + C) / 3 * V); vv = np.nan_to_num(V)
VWAP = np.cumsum(typ, 1) / np.maximum(np.cumsum(vv, 1), 1e-9)
VWAP = np.where(np.cumsum(vv, 1) > 0, VWAP, CF)
LA = np.where(AFT, L, np.nan)
RUNMIN = np.fmin.accumulate(np.where(np.isnan(LA), np.inf, LA), axis=1); RUNMIN = np.minimum(RUNMIN, E)      # 含本分鐘
RUNMIN_PREV = np.concatenate([E, RUNMIN[:, :-1]], 1); RUNMIN_PREV = np.where(AFT, RUNMIN_PREV, E)          # 到上一分鐘
YEAR = meta.date.dt.year.values
ret = lambda px: (-(px / (E if np.ndim(px) == 2 else E[:, 0]) - 1) - COST) * 100
print(f"交易 {N:,} 筆 ({meta.date.min().date()}~{meta.date.max().date()}), 交易日 {len(ALLDAYS):,} ({time.time()-T0:.0f}s)", flush=True)

def first(mask):
    hit = mask.any(1); return np.where(hit, mask.argmax(1), 10_000)

# 現行停損
STOP_HIT = AFT & (H > RH)
def stop_px_of(lvl, idx):
    o = O[np.arange(N), np.minimum(idx, NT - 1)]
    px = np.maximum(lvl, np.where(np.isnan(o), lvl, o)) * (1 + STOP_SLIP)
    return np.where(px >= PC[:, 0] * (1 + FORCE), PC[:, 0] * (1 + FORCE) * (1 + FORCE_SLIP), px)

def simulate(sig=None, limit=None, stop_lvl=None, t_end=270):
    """sig: N×NT 布林, 該分鐘收盤時平倉 (市價 + SLIP); limit: N×NT 限價回補價 (要穿過才成交); stop_lvl: N×NT 動態停損價 (取 min(區間高, 它));
    t_end: 該分鐘收盤強制平倉 (270 = 13:30 收盤競價, 不扣滑價)。回傳 (每筆報酬%, 出場分鐘, 出場方式)。"""
    lvl = RH if stop_lvl is None else np.minimum(RH, stop_lvl)
    hit = AFT & (H > lvl) & (TT <= t_end)
    i_s = first(hit)
    lv_at = lvl[np.arange(N), np.minimum(i_s, NT - 1)] if lvl.shape[1] > 1 else lvl[:, 0]
    px_s = stop_px_of(lv_at, i_s)
    i_l = np.full(N, 10_000); px_l = np.full(N, np.nan)
    if limit is not None:
        lh = AFT & (L < limit) & (TT <= t_end); i_l = first(lh)
        oo = O[np.arange(N), np.minimum(i_l, NT - 1)]; lm = limit[np.arange(N), np.minimum(i_l, NT - 1)]
        px_l = np.minimum(lm, np.where(np.isnan(oo), lm, oo))
    i_g = np.full(N, 10_000)
    if sig is not None: i_g = first(sig & AFT & (TT < t_end))
    px_g = CF[np.arange(N), np.minimum(i_g, NT - 1)] * (1 + SLIP)
    px_e = CF[:, t_end] * (1 + (SLIP if t_end < 270 else 0))
    idx = np.stack([i_s, i_l, i_g, np.full(N, t_end)])
    k = np.argmin(idx, 0)                                   # 同分鐘: 停損 > 限價 > 訊號 > 收盤 (保守)
    px = np.choose(k, [px_s, px_l, px_g, px_e])
    return ret(px), idx[k, np.arange(N)], k

def daily(r):
    s = pd.Series(r / 10, index=meta.date).groupby(level=0).sum()
    return s.reindex(ALLDAYS, fill_value=0.0)
def stats(r, mask=None):
    m = np.ones(N, bool) if mask is None else mask
    dl = daily(np.where(m, r, 0.0))
    if mask is not None:
        yrs = sorted(set(YEAR[m])); dl = dl[dl.index.year.isin(yrs)]
    eq = dl.cumsum(); mdd = (eq - eq.cummax()).min()
    sd = dl.std()
    return dict(n=int(m.sum()), per=float(r[m].mean()), win=float((r[m] > 0).mean() * 100), ann=dl.mean() * 750,
                sh=dl.mean() / sd * np.sqrt(250) if sd > 0 else 0, mdd=mdd * 3, worst=dl.min() * 3,
                calmar=(dl.mean() * 750) / max(-mdd * 3, 1e-9))
HD = ["| 出場法 | 筆數 | 每筆% | 勝率% | 年均 (萬, 300萬) | Sharpe | 最大回撤 (萬) | 最差日 (萬) | 年均/回撤 |", "|---|---|---|---|---|---|---|---|---|"]
def line(name, s): return (f"| {name} | {s['n']:,} | {s['per']:.3f} | {s['win']:.0f} | {s['ann']:.0f} | {s['sh']:.2f} | {s['mdd']:.0f} | {s['worst']:.1f} | {s['calmar']:.2f} |")

# =============================== 2. 規則族群 ===============================
def R_time(T): return simulate(t_end=T)
def R_target(p): return simulate(limit=np.broadcast_to(E * (1 - p / 100), (N, NT)))
def R_target_rng(k): return simulate(limit=np.broadcast_to(E - k * (RH - RL), (N, NT)))
def R_trail(x, act):
    on = RUNMIN_PREV <= E * (1 - act / 100)
    return simulate(stop_lvl=np.where(on, RUNMIN_PREV * (1 + x / 100), np.inf))
def R_be(act, b):
    on = RUNMIN_PREV <= E * (1 - act / 100)
    return simulate(stop_lvl=np.where(on, E * (1 - b / 100), np.inf))
def R_timestop(k, mpct):
    sig = (TT == ET + k) & (CF > E * (1 - mpct / 100)); return simulate(sig=sig)
def R_vwap(T):
    return simulate(sig=(TT >= T) & (CF > VWAP))
def R_trail_late(x, T):
    on = TT >= T
    return simulate(stop_lvl=np.where(on, RUNMIN_PREV * (1 + x / 100), np.inf))
FAM = {
    "固定時間出場": [(f"{9 + (T + 0) // 60}:{T % 60:02d} 平", lambda T=T: R_time(T)) for T in (60, 90, 120, 180, 240, 255, 265, 270)],
    "獲利目標 (固定%)": [(f"賺 {p}% 回補", lambda p=p: R_target(p)) for p in (1, 1.5, 2, 3, 4, 5, 6)],
    "獲利目標 (區間寬倍數)": [(f"賺 {k} 倍區間寬", lambda k=k: R_target_rng(k)) for k in (0.5, 1, 1.5, 2, 3)],
    "移動停利": [(f"賺 {act}% 後, 從最低反彈 {x}% 出", lambda x=x, act=act: R_trail(x, act)) for act in (0, 1, 2, 3) for x in (1, 1.5, 2, 3)],
    "保本停損": [(f"賺 {act}% 後停損移到 進場價−{b}%", lambda act=act, b=b: R_be(act, b)) for act in (1, 2, 3) for b in (0, 0.5, 1)],
    "時間停損": [(f"進場 {k} 分後仍沒賺 {mp}% 就出", lambda k=k, mp=mp: R_timestop(k, mp)) for k in (15, 30, 60, 90) for mp in (0, 0.5, 1)],
    "站回 VWAP 出": [(f"{9 + T // 60}:{T % 60:02d} 後站回 VWAP 出", lambda T=T: R_vwap(T)) for T in (5, 120, 180, 240)],
    "盤後段移動停利": [(f"{9 + T // 60}:{T % 60:02d} 後從最低反彈 {x}% 出", lambda x=x, T=T: R_trail_late(x, T)) for T in (180, 240) for x in (0.5, 1, 2)],
}
base_r, base_i, base_k = simulate()
EXPM = meta.date.values <= np.datetime64(EXPLORE_END)
TEST_YEARS = [y for y in range(2022, 2027) if (YEAR == y).any()]
OOS = np.isin(YEAR, TEST_YEARS)

P("# 進場後什麼時候出場比較好 (選股 / 進場不變)")
P("")
P(f"期間 {meta.date.min().date()} ~ {meta.date.max().date()}, 現行規則每日先到先做最多 10 檔 (等權), 共 {N:,} 筆。每筆含成本 0.23%; "
  f"規則出場 (市價) 另扣 0.1% 滑價; 停損一律保留 (反彈破區間高)。")
P(f"樣本外 = {TEST_YEARS[0]}~{TEST_YEARS[-1]} 逐年走動式: 每年只用之前年份挑參數。挑參數標準 = 訓練期 年均/最大回撤 最大 (且年均 ≥ 現行 70%)。")
P(f"現行出場分布: 停損 {(base_k == 0).mean()*100:.0f}%、收盤 {(base_k == 3).mean()*100:.0f}%。")

RES = {}
for fam, items in FAM.items():
    for nm, fn in items:
        r, i, k = fn(); RES[(fam, nm)] = r
print(f"規則回測完成 ({time.time()-T0:.0f}s)", flush=True)

P(""); P("## 1. 全部參數 (全期 / 探索 ≤2023 / 驗證 2024~) — 只供參考, 有事後挑選偏誤")
P(""); P("| 出場法 | 全期 每筆% | 全期 年均 | 全期 回撤 | 探索 年均 / 回撤 | 驗證 年均 / 回撤 | 驗證 Sharpe | 全期 停損外出場% |"); P("|---|---|---|---|---|---|---|---|")
def short(nm, r, k=None):
    s, se, sv = stats(r), stats(r, EXPM), stats(r, ~EXPM)
    return f"| {nm} | {s['per']:.3f} | {s['ann']:.0f} | {s['mdd']:.0f} | {se['ann']:.0f} / {se['mdd']:.0f} | {sv['ann']:.0f} / {sv['mdd']:.0f} | {sv['sh']:.2f} |"
P(short("**現行 (停損 / 13:30 收盤)**", base_r) + " – |")
for fam, items in FAM.items():
    P(f"| *{fam}* | | | | | | | |")
    for nm, _ in items:
        r = RES[(fam, nm)]; P(short(nm, r) + f" {(np.abs(r - base_r) > 1e-9).mean()*100:.0f} |")

# =============================== 3. 走動式挑參數 ===============================
def pick(cands, train_mask):
    b = stats(base_r, train_mask)
    best, bs = None, b["calmar"]                  # 現行也是候選: 都沒贏過現行就維持現行
    for nm, r in cands:
        s = stats(r, train_mask)
        if s["ann"] < 0.7 * b["ann"]: continue
        if s["calmar"] > bs: best, bs = nm, s["calmar"]
    return best
WF = {}
for fam, items in FAM.items():
    cands = [(nm, RES[(fam, nm)]) for nm, _ in items]
    r_oos = base_r.copy(); chosen = []
    for y in TEST_YEARS:
        nm = pick(cands, YEAR < y)
        chosen.append(f"{y}:{nm or '現行'}")
        if nm: r_oos = np.where(YEAR == y, RES[(fam, nm)], r_oos)
    WF[fam] = (r_oos, chosen)
allc = [(f"{fam}/{nm}", RES[(fam, nm)]) for fam, items in FAM.items() for nm, _ in items]
r_oos = base_r.copy(); chosen = []
for y in TEST_YEARS:
    nm = pick(allc, YEAR < y); chosen.append(f"{y}:{nm or '現行'}")
    if nm: r_oos = np.where(YEAR == y, dict(allc)[nm], r_oos)
WF["所有規則一起挑"] = (r_oos, chosen)

# =============================== 4. RL: 批次 Q-learning (續抱 / 平倉) ===============================
TB = np.digitize(TT + 0 * ET, [30, 60, 90, 120, 180, 240])                 # 時段: <09:30, <10:00, <10:30, <11:00, <12:00, <13:00, 13:00~
PNL = (E - CF) / E * 100                                                   # 浮動損益% (空單)
PB = np.digitize(PNL, [-1, 0, 1, 2, 3, 5])
GB = np.digitize((CF - RUNMIN) / E * 100, [0.5, 1, 2])                     # 從進場後最低點反彈%
VB = (CF > VWAP).astype(int)
STATE = ((TB * 7 + PB) * 4 + GB) * 2 + VB
NS = 7 * 7 * 4 * 2
EXITV = ret(CF * (1 + SLIP))                                              # 該分鐘收盤平倉的報酬
o_stop = np.where(np.isnan(O), RH, O)
_sp = np.maximum(RH, o_stop) * (1 + STOP_SLIP)
STOPV = (-(np.where(_sp >= PC * (1 + FORCE), PC * (1 + FORCE) * (1 + FORCE_SLIP), _sp) / E - 1) - COST) * 100   # 同 stop_px_of
CLOSEV = ret(CF[:, 270])
T_LAST = 264                                                               # 13:24 收盤前最後一次決策
def values(pol, S=None):
    """pol: NS 布林 (True = 平倉)。回傳 D (N×NT): 「持有進入第 u 分鐘」的報酬, 照政策走。"""
    D = np.zeros((N, NT + 1)); D[:, NT] = CLOSEV; D[:, 270] = np.where(STOP_HIT[:, 270], STOPV[:, 270], CLOSEV)
    for u in range(269, -1, -1):
        ex = pol[(STATE if S is None else S)[:, u]] & (u <= T_LAST)
        D[:, u] = np.where(STOP_HIT[:, u], STOPV[:, u], np.where(ex, EXITV[:, u], D[:, u + 1]))
    return D
def train(mask, iters=4, min_trades=60, margin=0.02, S=None, ns=None, lam=0.0):
    """lam > 0: 風險趨避, 比較 平均 − lam×變異數 (續抱的報酬變異大, 越大越傾向提早平倉 → 降回撤)。"""
    S = STATE if S is None else S; ns = NS if ns is None else ns
    pol = np.zeros(ns, bool)
    rows = np.where(mask)[0]
    dec = AFT[rows] & (TT <= T_LAST) & ~STOP_HIT[rows]
    rr, cc = np.nonzero(dec)
    st = S[rows][rr, cc]; ev = EXITV[rows][rr, cc]
    for it in range(iters):
        D = values(pol, S)[rows]
        hv = D[rr, cc + 1]
        df = pd.DataFrame(dict(s=st, d=ev - hv, n=rr, h=hv, e=ev))
        g = df.groupby("s").agg(d=("d", "mean"), nt=("n", "nunique"), vh=("h", "var"), ve=("e", "var"))
        g["d"] = g.d - lam * (g.ve.fillna(0) - g.vh.fillna(0))
        new = np.zeros(ns, bool); ok = g[(g.nt >= min_trades) & (g.d > margin)].index.values; new[ok] = True
        if (new == pol).all(): break
        pol = new
    return pol
def apply(pol, S=None):
    D = values(pol, S)
    # 實際報酬 = 進場那分鐘之後第一分鐘起持有
    return D[np.arange(N), np.minimum(ET[:, 0] + 1, NT)]
def rl_desc(pol):
    names_t = ["<09:30", "09:30~10", "10~10:30", "10:30~11", "11~12", "12~13", "13~"]
    names_p = ["賠>1%", "賠0~1%", "賺0~1%", "賺1~2%", "賺2~3%", "賺3~5%", "賺>5%"]
    names_g = ["反彈<0.5%", "反彈0.5~1%", "反彈1~2%", "反彈>2%"]; names_v = ["VWAP下", "VWAP上"]
    out = []
    for s in np.where(pol)[0]:
        v = s % 2; g_ = (s // 2) % 4; p_ = (s // 8) % 7; t_ = s // 56
        out.append(f"{names_t[t_]}·{names_p[p_]}·{names_g[g_]}·{names_v[v]}")
    return out
chk = apply(np.zeros(NS, bool))
P(""); P(f"(自我檢查: RL 環境在「永不提前平倉」時與現行回測每筆差 {np.abs(chk - base_r).max():.2e}%)")
rl_oos = base_r.copy(); rl_info = []
for y in TEST_YEARS:
    pol = train(YEAR < y); r = apply(pol)
    rl_oos = np.where(YEAR == y, r, rl_oos); rl_info.append((y, int(pol.sum())))
pol_all = train(np.ones(N, bool))
# RL + D1: 狀態再乘上一個日K特徵 (訓練期中位數切兩半; 連漲以 ≥2 天切), 一次加一個, 看樣本外有沒有變好
D1F = {"跳空幅度": "gap", "ATR20": "atr20", "昨日漲跌": "pchg", "連漲天數": "streak", "開盤 vs 前20日高": "d_h20", "大盤平均跳空": "mkt_gap"}
d1_rows = []
for lab, col in D1F.items():
    v = meta[col].values.astype(float)
    r_o = base_r.copy(); ns_on = []
    for y in TEST_YEARS:
        tr = YEAR < y
        cut = 2 if col == "streak" else np.nanmedian(v[tr])
        b = (np.nan_to_num(v, nan=cut) >= cut).astype(int)[:, None]
        S2 = STATE * 2 + b; pol = train(tr, S=S2, ns=NS * 2)
        r_o = np.where(YEAR == y, apply(pol, S2), r_o); ns_on.append(int(pol.sum()))
    WF[f"RL + 日K: {lab}"] = (r_o, [f"{y}:{k} 個狀態平倉" for y, k in zip(TEST_YEARS, ns_on)])
WF["RL (Q-learning 續抱/平倉)"] = (rl_oos, [f"{y}:{k} 個狀態平倉" for y, k in rl_info])
for lam in (0.02, 0.05, 0.1):
    r_o = base_r.copy(); ns_on = []
    for y in TEST_YEARS:
        pol = train(YEAR < y, lam=lam); r_o = np.where(YEAR == y, apply(pol), r_o); ns_on.append(int(pol.sum()))
    WF[f"RL 風險趨避 λ={lam}"] = (r_o, [f"{y}:{k} 個狀態平倉" for y, k in zip(TEST_YEARS, ns_on)])
# 13:25 前平倉的版本 (收盤競價改成 13:25 市價)
r1325 = R_time(265)[0]
print(f"RL 完成 ({time.time()-T0:.0f}s)", flush=True)

# =============================== 5. 報表 ===============================
P(""); P(f"## 2. 樣本外 (走動式, {TEST_YEARS[0]}~{TEST_YEARS[-1]}) — 這張才算數")
P(""); [P(h) for h in HD]
P(line("**現行**", stats(base_r, OOS)))
for nm, (r, ch) in WF.items(): P(line(nm, stats(r, OOS)))
P(""); P("各年選到的參數:")
for nm, (r, ch) in WF.items(): P(f"- {nm}: " + "; ".join(ch))

P(""); P("## 3. 樣本外逐年年均 (萬) / 最大回撤 (萬)")
P(""); P("| 出場法 | " + " | ".join(str(y) for y in TEST_YEARS) + " |"); P("|---|" + "---|" * len(TEST_YEARS))
for nm, r in [("現行", base_r)] + [(k, v[0]) for k, v in WF.items()]:
    cells = []
    for y in TEST_YEARS:
        s = stats(r, YEAR == y); cells.append(f"{s['ann']:.0f} / {s['mdd']:.0f}")
    P(f"| {nm} | " + " | ".join(cells) + " |")

P(""); P("## 3b. 固定參數候選 逐年 (全期 2020~, 年均 萬 / 最大回撤 萬) — 看改善是不是只集中在某一年")
CAND = [("現行", base_r), ("移動停利 賺2%後反彈3%", RES[("移動停利", "賺 2% 後, 從最低反彈 3% 出")]),
        ("保本 賺2%後停損移到進場價", RES[("保本停損", "賺 2% 後停損移到 進場價−0%")]),
        ("保本 賺3%後停損移到進場價", RES[("保本停損", "賺 3% 後停損移到 進場價−0%")]),
        ("13:00 後反彈 2% 出", RES[("盤後段移動停利", "13:00 後從最低反彈 2% 出")])]
YS = sorted(set(YEAR))
P(""); P("| 出場法 | " + " | ".join(str(y) for y in YS) + " | 全期 |"); P("|---|" + "---|" * (len(YS) + 1))
for nm, r in CAND:
    P(f"| {nm} | " + " | ".join(f"{stats(r, YEAR == y)['ann']:.0f} / {stats(r, YEAR == y)['mdd']:.0f}" for y in YS) + f" | {stats(r)['ann']:.0f} / {stats(r)['mdd']:.0f} |")
P(""); P("回撤最深的 5 段 (現行): 起訖日、現行 vs 候選 的回撤 (萬)")
dl0 = daily(base_r); eq0 = dl0.cumsum(); dd0 = eq0 - eq0.cummax()
peaks = []
x = dd0.copy()
for _ in range(5):
    tmin = x.idxmin()
    if x[tmin] >= 0: break
    st = eq0[:tmin].idxmax(); en = tmin
    row = f"- {st.date()} ~ {en.date()}: 現行 {(eq0[en] - eq0[st]) * 3:.0f}"
    for nm, r in CAND[1:]:
        e = daily(r).cumsum(); row += f"、{nm} {(e[en] - e[st]) * 3:.0f}"
    P(row); x[st:en + pd.Timedelta(days=1)] = 0
    nxt = eq0[en:]; rec = nxt[nxt >= eq0[st]]
    if len(rec): x[en:rec.index[0]] = 0

P(""); P("## 4. 對照")
P(""); [P(h) for h in HD]
P(line("現行 (全期)", stats(base_r)))
P(line("13:25 市價平倉取代收盤競價 (全期)", stats(r1325)))
P(line("RL 用全期訓練再回測全期 (樣本內, 只看上限)", stats(apply(pol_all))))
P(""); P(f"RL 全期訓練學到「該平倉」的狀態 ({int(pol_all.sum())} 個 / {NS}): " + ("、".join(rl_desc(pol_all)) if pol_all.any() else "無 (一律續抱)"))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
