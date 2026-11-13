#!/usr/bin/env python3
"""研究: 現行空單「跌破開盤 5 分低」的那一下, 若同時跌穿日K關鍵價位 (前 20~60 日), 勝率 / 報酬會不會更高。只讀資料、印報告。
輸入: data/processed/d1_sr_orb.parquet (先跑 scripts/33_d1_sr_orb.py) + HF db/d1 (重算關鍵價位)。口徑同 33: 成本 0.23%、+8% 強制、鎖漲停 −7%、停損區間高。

關鍵價位 (全部只用今天以前的日K) 與分類 (以開盤 5 分低 rl = 跌破點為準, 容忍 TOL):
  跌穿   : 關鍵位在 rl ± TOL 內 → 跌破區間低 = 同時跌穿關鍵位
  區間內 : 關鍵位在 (rl+TOL, rh] → 5 分內就已經跌回關鍵位之下 (或是上方壓力)
  下方近 : 關鍵位在 [rl−2%, rl−TOL) → 進場後不遠就碰到支撐
  沒有   : 以上都不是
定義: 前 N 日低 (N=20/40/60)、前 N 日高 (前高轉支撐)、擺盪高 / 低點 (前後 5 根, 近 60 日)、
      密集成交區 (近 60 日, 典型價 1% 分箱、成交量 ≥ 1.5 倍平均的箱)、缺口邊緣 (昨高 = 本次跳空缺口下緣; 昨收)。
用法: python3 scripts/35_key_level_break.py [--tol 0.005] [--out stats/key_level_break.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--tol", type=float, default=0.005)
ap.add_argument("--out", default="stats/key_level_break.md")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
hf = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset")
T0 = time.time(); K = 5

Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy().reset_index(drop=True)
A["exp"] = A.date <= EXPLORE_END
print(f"現行空單 (不設上限) {len(A):,} 筆")

# ---- D1 ----
files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
d1m = sorted(f for f in files if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:13] >= "2019_01")
d = pd.concat([pd.read_parquet(hf(f), columns=["stock_id", "date", "high", "low", "close", "volume"]) for f in d1m], ignore_index=True)
d["stock_id"] = d.stock_id.astype(str); d["date"] = pd.to_datetime(d.date).dt.normalize()
d = d[d.stock_id.isin(set(A.stock_id)) & (d.close > 0)].drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"])
for c in ["high", "low", "close", "volume"]: d[c] = d[c].astype("float64")
print(f"D1 {len(d):,} 列 ({time.time()-T0:.0f}s)", flush=True)

TYPES = ["前20日低", "前40日低", "前60日低", "前20日高", "前40日高", "前60日高", "擺盪低點", "擺盪高點", "密集成交區", "缺口下緣(昨高)", "昨收"]
LV = {t: [None] * len(A) for t in TYPES}          # 每筆: 該定義的價位 array
from numpy.lib.stride_tricks import sliding_window_view as swv
for sid, idx in A.groupby("stock_id").groups.items():
    x = d[d.stock_id == sid]
    if len(x) < 70: continue
    dt = x.date.values; hi, lo, cl, vo = x.high.values, x.low.values, x.close.values, x.volume.values
    n = len(x)
    wh = swv(hi, 2 * K + 1).max(1); wl = swv(lo, 2 * K + 1).min(1)
    ph_i = np.where(hi[K:n - K] >= wh)[0] + K; pl_i = np.where(lo[K:n - K] <= wl)[0] + K
    tp = (hi + lo + cl) / 3
    for j in idx:
        r = A.loc[j]
        i = np.searchsorted(dt, np.datetime64(r.date))          # 今天的位置 (今天以前 = [:i])
        if i < 61: continue
        for N in (20, 40, 60):
            LV[f"前{N}日低"][j] = np.array([lo[i - N:i].min()]); LV[f"前{N}日高"][j] = np.array([hi[i - N:i].max()])
        okh = (ph_i + K <= i - 1) & (ph_i >= i - 60); okl = (pl_i + K <= i - 1) & (pl_i >= i - 60)
        LV["擺盪高點"][j] = hi[ph_i[okh]]; LV["擺盪低點"][j] = lo[pl_i[okl]]
        t60, v60 = tp[i - 60:i], vo[i - 60:i]
        b = np.floor(np.log(t60) / np.log(1.01)); ub, inv = np.unique(b, return_inverse=True)
        vb = np.bincount(inv, weights=v60)
        LV["密集成交區"][j] = 1.01 ** (ub[vb >= 1.5 * vb.mean()] + 0.5)
        LV["缺口下緣(昨高)"][j] = np.array([hi[i - 1]]); LV["昨收"][j] = np.array([cl[i - 1]])
print(f"關鍵價位完成 ({time.time()-T0:.0f}s)", flush=True)

def classify(lv, rl, rh, tol):
    if lv is None or not len(lv): return "沒有"
    if (np.abs(lv / rl - 1) <= tol).any(): return "跌穿"
    if ((lv > rl * (1 + tol)) & (lv <= rh)).any(): return "區間內"
    if ((lv < rl * (1 - tol)) & (lv >= rl * 0.98)).any(): return "下方近"
    return "沒有"
for t in TYPES:
    A[f"c_{t}"] = [classify(LV[t][j], A.rl[j], A.rh[j], a.tol) for j in range(len(A))]
KEYS = ["前20日低", "前40日低", "前60日低", "擺盪低點", "密集成交區", "前20日高", "前40日高", "前60日高", "擺盪高點", "缺口下緣(昨高)"]
A["c_任一(不含昨收)"] = np.where((A[[f"c_{t}" for t in KEYS]] == "跌穿").any(axis=1), "跌穿",
                         np.where((A[[f"c_{t}" for t in KEYS]] == "區間內").any(axis=1), "區間內",
                                  np.where((A[[f"c_{t}" for t in KEYS]] == "下方近").any(axis=1), "下方近", "沒有")))
A["n_brk"] = (A[[f"c_{t}" for t in KEYS]] == "跌穿").sum(axis=1)

# ---- 報表 ----
def stats(z):
    r = z.s_r
    if len(r) < 30: return None
    e = z.exp
    return (len(r), r.mean(), (r > 0).mean() * 100, (z.s_why == "停損").mean() * 100, r[e].mean(), r[~e].mean(), (r[e] > 0).mean() * 100, (r[~e] > 0).mean() * 100)
def line(name, z):
    s = stats(z)
    if s is None: return f"| {name} | {len(z)} | | | | | |"
    return f"| {name} | {s[0]:,} ({s[0]/len(A)*100:.0f}%) | {s[1]:.3f} | {s[2]:.0f} | {s[3]:.0f} | {s[4]:.3f} / {s[5]:.3f} | {s[6]:.0f} / {s[7]:.0f} |"
HDR = ["| 分類 | 筆數 (占比) | 每筆% | 勝率% | 停損% | 探索/驗證 每筆% | 探索/驗證 勝率% |", "|---|---|---|---|---|---|---|"]

P("# 第一根跌破開盤 5 分低時, 同時跌穿日K關鍵價位 → 勝率會不會更高")
P("")
P(f"對象: 現行空單規則 (跳空≥2% ∩ ATR≥3% ∩ 均額≥3億 ∩ 股價≤300 ∩ 非週一 ∩ 大盤跳空≥−1%), 不設每日 10 檔上限, {len(A):,} 筆, {A.date.min().date()} ~ {A.date.max().date()}。探索 ≤{EXPLORE_END}, 驗證之後。")
P(f"分類以開盤 5 分低 (跌破點) 為準, 容忍 ±{a.tol*100:.1f}%: **跌穿** = 關鍵位就在區間低附近 (跌破區間低 = 跌穿關鍵位)；**區間內** = 關鍵位在區間低之上、區間高之下；**下方近** = 區間低之下 2% 內有關鍵位 (進場後不遠就碰支撐)；**沒有** = 以上皆非。")
P("")
P("全部: " + line("全部", A).split("|", 2)[2])
for t in KEYS + ["昨收", "任一(不含昨收)"]:
    P(""); P(f"### {t}"); P(""); [P(h) for h in HDR]
    for c in ["跌穿", "區間內", "下方近", "沒有"]:
        P(line(c, A[A[f"c_{t}"] == c]))
P(""); P("### 同時跌穿幾條關鍵位 (不含昨收)"); P(""); [P(h) for h in HDR]
for k, m in [("0 條", A.n_brk == 0), ("1 條", A.n_brk == 1), ("2 條", A.n_brk == 2), ("≥3 條", A.n_brk >= 3)]:
    P(line(k, A[m]))

# 容忍度敏感度 (任一)
P(""); P("### 容忍度敏感度: 任一關鍵位「跌穿」 vs 其他"); P("")
P("| 容忍 | 跌穿 筆數 | 跌穿 每筆% / 勝率% | 其他 每筆% / 勝率% | 跌穿 探索/驗證 每筆% | 其他 探索/驗證 每筆% |"); P("|---|---|---|---|---|---|")
for tol in [0.002, 0.005, 0.01, 0.02]:
    brk = np.zeros(len(A), bool)
    for t in KEYS:
        brk |= np.array([LV[t][j] is not None and len(LV[t][j]) > 0 and bool((np.abs(LV[t][j] / A.rl[j] - 1) <= tol).any()) for j in range(len(A))])
    z1, z0 = A[brk], A[~brk]
    P(f"| ±{tol*100:.1f}% | {len(z1):,} | {z1.s_r.mean():.3f} / {(z1.s_r>0).mean()*100:.0f} | {z0.s_r.mean():.3f} / {(z0.s_r>0).mean()*100:.0f} | "
      f"{z1[z1.exp].s_r.mean():.3f} / {z1[~z1.exp].s_r.mean():.3f} | {z0[z0.exp].s_r.mean():.3f} / {z0[~z0.exp].s_r.mean():.3f} |")

# ---- 組合: 硬過濾 / 部位加權 ----
def port(z, flag=None, keep_only=False, w_on=1.0, w_off=1.0, cap=10, name=""):
    z = z.sort_values(["date", "s_et", "stock_id"]); z = z.groupby("date").head(cap) if not keep_only else z[z[flag]].groupby("date").head(cap)
    w = np.where(z[flag], w_on, w_off) if flag else 1.0
    dly = (z.s_r * w).groupby(z.date).sum().div(cap).reindex(ALLDAYS, fill_value=0.0); dly.index = ALLDAYS
    eq = (1 + dly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    e = dly.index <= EXPLORE_END
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return f"| {name} | {len(z):,} | {z.s_r.mean():.3f} | {(z.s_r>0).mean()*100:.0f} | {dly.mean():.3f} | {dly.mean()/dly.std()*np.sqrt(250):.2f} | {mdd:.0f}% | {dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |"
A["f_brk"] = A["c_任一(不含昨收)"] == "跌穿"
A["f_low"] = A[[f"c_{t}" for t in ["前20日低", "前40日低", "前60日低", "擺盪低點", "密集成交區"]]].eq("跌穿").any(axis=1)
A["f_hi"] = A[[f"c_{t}" for t in ["前20日高", "前40日高", "前60日高", "擺盪高點", "缺口下緣(昨高)"]]].eq("跌穿").any(axis=1)
A["f_lvlin"] = A.lvl_in == 1
A["f_both"] = A.f_brk | A.f_lvlin
P(""); P("## 組合: 每日最多 10 檔 (依跌破時間先到先做), 每檔 1/10, 無交易日計 0")
P(""); P("| 規則 | 筆數 | 每筆% | 勝率% | 日淨% | Sharpe | MDD | 探索/驗證 日淨% | 2020~2026 日淨% |"); P("|---|---|---|---|---|---|---|---|---|")
P(port(A, name="現行 (先到先做)"))
for f, lab in [("f_brk", "跌穿任一關鍵位"), ("f_low", "跌穿前低/擺盪低/密集區"), ("f_hi", "跌穿前高/擺盪高/缺口下緣"), ("f_both", "跌穿任一 或 區間內碰壓力 (33 的訊號)")]:
    P(port(A, f, keep_only=True, name=f"只做「{lab}」"))
    P(port(A, f, w_on=1.5, w_off=0.75, name=f"「{lab}」1.5 格, 其他 0.75 格"))
P("")
P("注意: 1.5 / 0.75 格版本總曝險可能 >100%, 日淨增加有一部分是曝險; 比 Sharpe 較公平。")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
