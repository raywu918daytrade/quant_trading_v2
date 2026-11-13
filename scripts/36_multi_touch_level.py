#!/usr/bin/env python3
"""研究: 過去 20/40/60 日有多個高點 (或低點) 重複落在同一價位帶 (彼此差 1% 或 2% 內) = 被測試多次的關鍵位。
現行空單「跌破開盤 5 分低」時若穿越這種多次觸碰的價位, 勝率會不會更高? 觸碰次數越多越好嗎? 只讀資料、印報告。
輸入: data/processed/d1_sr_orb.parquet (先跑 scripts/33_d1_sr_orb.py) + HF db/d1。口徑同 33/35。

觸碰點: 局部高 (低) 點 = 該日高 (低) 是前後各 2 根內最高 (最低), 第 i+2 日後才確認; 只用今天以前、近 N 日。
價位帶: 以每個觸碰點為中心, 帶內 (±W) 的觸碰點數 = 觸碰次數, 帶的價位 = 帶內觸碰點平均。
位置 (以開盤 5 分低 rl 為準, 容忍 ±0.5%): 跌穿 / 區間內 (rl, rh] / 下方近 [rl−2%, rl) / 沒有; 同位置多條帶時取觸碰次數最多的。
用法: python3 scripts/36_multi_touch_level.py [--out stats/multi_touch_level.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--tol", type=float, default=0.005)
ap.add_argument("--out", default="stats/multi_touch_level.md")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
hf = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset")
T0 = time.time(); K = 2; TOL = a.tol

Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy().reset_index(drop=True)
A["exp"] = A.date <= EXPLORE_END

files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
d1m = sorted(f for f in files if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:13] >= "2019_01")
d = pd.concat([pd.read_parquet(hf(f), columns=["stock_id", "date", "high", "low", "close"]) for f in d1m], ignore_index=True)
d["stock_id"] = d.stock_id.astype(str); d["date"] = pd.to_datetime(d.date).dt.normalize()
d = d[d.stock_id.isin(set(A.stock_id)) & (d.close > 0)].drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"])
print(f"A {len(A):,} 筆, D1 {len(d):,} 列 ({time.time()-T0:.0f}s)", flush=True)

NS, WS, SIDES = (20, 40, 60), (0.01, 0.02), ("高點", "低點", "高低合併")
LOCS = ["跌穿", "區間內", "下方近"]
res = {}   # (side, N, W, loc) -> array 觸碰次數 (0 = 該位置沒有帶)
for key in [(s, n, w, l) for s in SIDES for n in NS for w in WS for l in LOCS]:
    res[key] = np.zeros(len(A), int)

def bands(p, w):
    """每個觸碰點為中心的帶: (價位, 次數)"""
    if not len(p): return np.array([]), np.array([], int)
    m = np.abs(p[:, None] / p[None, :] - 1) <= w
    cnt = m.sum(1); lvl = (m * p[None, :]).sum(1) / cnt
    return lvl, cnt

from numpy.lib.stride_tricks import sliding_window_view as swv
for sid, idx in A.groupby("stock_id").groups.items():
    x = d[d.stock_id == sid]
    if len(x) < 70: continue
    dt = x.date.values; hi, lo = x.high.values, x.low.values; n = len(x)
    ph_i = np.where(hi[K:n - K] >= swv(hi, 2 * K + 1).max(1))[0] + K
    pl_i = np.where(lo[K:n - K] <= swv(lo, 2 * K + 1).min(1))[0] + K
    for j in idx:
        rl, rh = A.rl[j], A.rh[j]
        i = np.searchsorted(dt, np.datetime64(A.date[j]))
        if i < 61: continue
        for N in NS:
            H = hi[ph_i[(ph_i + K <= i - 1) & (ph_i >= i - N)]]; L = lo[pl_i[(pl_i + K <= i - 1) & (pl_i >= i - N)]]
            for side, p in (("高點", H), ("低點", L), ("高低合併", np.concatenate([H, L]))):
                for W in WS:
                    lvl, cnt = bands(p, W)
                    if not len(lvl): continue
                    r = lvl / rl - 1
                    for loc, m in (("跌穿", np.abs(r) <= TOL), ("區間內", (lvl > rl * (1 + TOL)) & (lvl <= rh)), ("下方近", (r < -TOL) & (r >= -0.02))):
                        if m.any(): res[(side, N, W, loc)][j] = cnt[m].max()
print(f"價位帶完成 ({time.time()-T0:.0f}s)", flush=True)

def cell(z):
    if len(z) < 30: return f"{len(z)} 筆 | | "
    e = z.exp
    return f"{len(z):,} | {z.s_r.mean():.3f} / {(z.s_r>0).mean()*100:.0f} | {z[e].s_r.mean():.3f} / {z[~e].s_r.mean():.3f}"

P("# 多次觸碰的價位帶 (過去 20/40/60 日) 被第一根跌破穿越 → 勝率會不會更高")
P("")
P(f"對象: 現行空單規則, 不設每日 10 檔上限, {len(A):,} 筆 (全部每筆 {A.s_r.mean():.3f}%、勝率 {(A.s_r>0).mean()*100:.0f}%、停損 {(A.s_why=='停損').mean()*100:.0f}%), {A.date.min().date()} ~ {A.date.max().date()}。探索 ≤{EXPLORE_END}, 驗證之後。")
P(f"觸碰點 = 前後 2 根內的局部高 / 低點；價位帶寬 ±1% 或 ±2%；觸碰次數 = 帶內觸碰點數。位置以開盤 5 分低為準 (容忍 ±{TOL*100:.1f}%)。")
P("每格: 筆數 | 每筆% / 勝率% | 探索/驗證 每筆%")
for loc in LOCS:
    P(""); P(f"## 價位帶在「{loc}」, 依觸碰次數"); P("")
    P("| 帶 | 期間 | 寬 | 1 次 (單點) | | | 2 次 | | | 3 次 | | | ≥4 次 | | | 沒有這種帶 | | |")
    P("|---|---|---|" + "---|" * 15)
    for side in SIDES:
        for N in NS:
            for W in WS:
                c = res[(side, N, W, loc)]
                cells = [cell(A[c == 1]), cell(A[c == 2]), cell(A[c == 3]), cell(A[c >= 4]), cell(A[c == 0])]
                P(f"| {side} | {N} 日 | ±{W*100:.0f}% | " + " | ".join(cells) + " |")

# 停損率對照 (最常用的設定)
P(""); P("## 停損率 (高低合併, 60 日, ±1%)"); P("")
P("| 位置 | 1 次 | 2 次 | 3 次 | ≥4 次 | 沒有 |"); P("|---|---|---|---|---|---|")
for loc in LOCS:
    c = res[("高低合併", 60, 0.01, loc)]
    P(f"| {loc} | " + " | ".join(f"{(A[m].s_why=='停損').mean()*100:.0f}% ({m.sum():,})" for m in [c == 1, c == 2, c == 3, c >= 4, c == 0]) + " |")

# ---- 組合 ----
def port(flag=None, keep_only=False, w_on=1.0, w_off=1.0, cap=10, name=""):
    z = A.assign(f=flag if flag is not None else False).sort_values(["date", "s_et", "stock_id"])
    if keep_only: z = z[z.f]
    z = z.groupby("date").head(cap)
    w = np.where(z.f, w_on, w_off)
    dly = (z.s_r * w).groupby(z.date).sum().div(cap).reindex(ALLDAYS, fill_value=0.0); dly.index = ALLDAYS
    eq = (1 + dly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    e = dly.index <= EXPLORE_END
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return f"| {name} | {len(z):,} | {z.s_r.mean():.3f} | {(z.s_r>0).mean()*100:.0f} | {dly.mean():.3f} | {dly.mean()/dly.std()*np.sqrt(250):.2f} | {mdd:.0f}% | {dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |"
P(""); P("## 組合: 每日最多 10 檔 (依跌破時間先到先做), 每檔 1/10, 無交易日計 0"); P("")
P("| 規則 | 筆數 | 每筆% | 勝率% | 日淨% | Sharpe | MDD | 探索/驗證 日淨% | 2020~2026 日淨% |"); P("|---|---|---|---|---|---|---|---|---|")
P(port(name="現行 (先到先做)"))
for side in SIDES:
    for N in (20, 60):
        for W in WS:
            f = res[(side, N, W, "跌穿")] >= 2
            P(port(f, keep_only=True, name=f"只做 跌穿 {side}{N}日 ±{W*100:.0f}% 觸碰≥2"))
            P(port(f, w_on=1.5, w_off=0.75, name=f"跌穿 {side}{N}日 ±{W*100:.0f}% 觸碰≥2: 1.5 格, 其他 0.75 格"))
            P(port(f, w_on=2.0, w_off=1.0, name=f"跌穿 {side}{N}日 ±{W*100:.0f}% 觸碰≥2: 加碼 2 格, 其他 1 格"))
for N in (20, 60):
    f = res[("高低合併", N, 0.01, "跌穿")] >= 3
    P(port(f, w_on=2.0, w_off=1.0, name=f"跌穿 高低合併{N}日 ±1% 觸碰≥3: 加碼 2 格, 其他 1 格"))
P("")
P("注意: 1.5 / 0.75 格版本總曝險可能 >100%, 比 Sharpe 較公平。設定很多 (3 種帶 × 3 期間 × 2 寬), 單一格的好結果要看兩段是否都成立。")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out} ({time.time()-T0:.0f}s)")
