#!/usr/bin/env python3
"""研究: 同一天很多檔跌破開盤 5 分低時, 要怎麼評估 / 排序 / 篩選。只讀資料、印報告, 不改線上規則。
輸入: data/processed/d1_sr_orb.parquet (先跑 scripts/33_d1_sr_orb.py 產生; 每檔每日 ORB5 空單結果 + 日K壓力特徵)。

現行: 每日最多 10 檔, 依跌破時間先到先做; 同一分鐘同時跌破時 history 回放依代號排序 (等於隨機)。
本腳本:
  1. 每天有幾檔跌破、多少集中在 09:05 那一分鐘、被 10 格上限丟掉的單 (第 11 檔起) 每筆多少。
  2. 單一特徵: 全部筆數的五分位每筆報酬, 以及「同一天內」的排名效果 (每筆減當日平均, 只看當天 ≥6 檔跌破的日子)。
  3. 組合: 先到先做 / 同分鐘依分數 / 只做分數過門檻 / 事後依分數取前 N (上限參考, 實盤做不到) / 不同檔數上限。
     分數方向與門檻只用探索期 (≤2023) 決定, 驗證期 (2024~) 是樣本外。
  4. 滑價: 0 檔與 1 檔 (進場、停損各滑 1 檔; 收盤競價不滑)。
用法: python3 scripts/34_rank_breakers.py [--inp data/processed/d1_sr_orb.parquet] [--out stats/rank_breakers.md]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXPLORE_END, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--out", default="stats/rank_breakers.md")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)

Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok].copy()
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy()
EXP = pd.Timestamp(EXPLORE_END)
A["exp"] = A.date <= EXP

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
A["tick_pct"] = tick(A.rl.values) / A.rl * 100
# 1 檔滑價: 進場少賣 1 檔, 停損多買 1 檔 (收盤競價不滑)
A["s_r1"] = A.s_r - A.tick_pct * (1 + (A.s_why == "停損"))

# ---- 特徵 (全部 09:05 前或跌破當下已知) ----
A["f_gap"] = A.gap * 100                                   # 跳空 %
A["f_atr"] = A.atr20 * 100                                 # ATR20 %
A["f_width"] = (A.rh / A.rl - 1) * 100                     # 開盤 5 分區間寬 % (≈ 停損距離)
A["f_stopd"] = (A.rh / A.s_px - 1) * 100                   # 停損距離 (區間高 / 進場價) %
A["f_vr"] = A.o5_vr * 100                                  # 開盤 5 分量 / 20 日均量 %
A["f_o5amt"] = np.log10(A.o5_v * 1000 * A.rl)              # 開盤 5 分成交額 (log10 元)
A["f_dv20"] = np.log10(A.dv20)                             # 20 日均成交額 (log10)
A["f_et"] = A.s_et                                         # 跌破時間 (分, 9:00 起)
A["f_lvl"] = A.lvl_in.astype(float)                        # 區間高碰到日K壓力
A["f_hi20"] = (A.o > A.h20).astype(float)                  # 開盤 > 20 日高
A["f_rhpc"] = (A.rh / A.pc - 1) * 100                      # 區間高離昨收 % (離漲停多近)
A["f_entgap"] = (A.s_px / A.pc - 1) * 100                  # 進場價離昨收 % (缺口還剩多少)
A["f_opos"] = ((A.o - A.rl) / (A.rh - A.rl)).clip(0, 1)    # 開盤在 5 分區間的位置 (1 = 開在最高)
A["f_tick"] = A.tick_pct                                   # 一檔占股價 %
A["f_price"] = np.log10(A.pc)                              # 股價 (log10)
FEAT = {"f_gap": "跳空幅度", "f_atr": "ATR20", "f_width": "5 分區間寬", "f_stopd": "停損距離", "f_vr": "5 分量比", "f_o5amt": "5 分成交額",
        "f_dv20": "20 日均額", "f_et": "跌破時間", "f_lvl": "區間碰日K壓力", "f_hi20": "開盤>20日高", "f_rhpc": "區間高離昨收",
        "f_entgap": "進場價離昨收", "f_opos": "開盤在區間位置", "f_tick": "一檔占比", "f_price": "股價"}

# ---- 類股: 同產業 (universe 內, 均額≥1億) 在本檔跌破那一分鐘前 (含) 已跌破開盤 5 分低的比例 ----
def load_industry():
    import os
    if os.environ.get("SMOKE_IND"):   # 本機測試用
        ids = pd.read_parquet(PROC / "orb5_peers.parquet").stock_id.astype(str).unique()
        return pd.DataFrame({"stock_id": ids, "ind": [str(hash(i) % 5) for i in ids]}), "smoke"
    from huggingface_hub import hf_hub_download
    from config import HF_REPO
    info = pd.read_parquet(hf_hub_download(HF_REPO, "db/info/info.parquet", repo_type="dataset"))
    print("info 欄位:", list(info.columns), flush=True)
    col = next((c for c in ["industry_category", "industry", "產業別", "sector", "category"] if c in info.columns), None)
    if col is not None:
        ind = info[["stock_id", col]].rename(columns={col: "ind"}).astype(str)
    else:   # 退路: 證交所 / 櫃買 OpenAPI 公司基本資料的產業別代碼
        import requests
        rows = []
        for url, k_id, k_ind in [("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號", "產業別"),
                                 ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "SecuritiesIndustryCode")]:
            try:
                js = requests.get(url, timeout=30).json(); rows += [(str(r[k_id]).strip(), str(r[k_ind]).strip()) for r in js]
            except Exception as e: print("產業別下載失敗", url, e)
        ind = pd.DataFrame(rows, columns=["stock_id", "ind"])
        col = "OpenAPI 產業別"
    ind = ind[ind.ind.notna() & ~ind.ind.isin(["", "nan", "None"])].drop_duplicates("stock_id")
    return ind, col
IND, IND_SRC = load_industry()
PEER = pd.read_parquet(PROC / "orb5_peers.parquet"); PEER["date"] = pd.to_datetime(PEER.date); PEER["stock_id"] = PEER.stock_id.astype(str)
PEER = PEER[~PEER.bad].merge(IND, on="stock_id")
A = A.merge(IND, on="stock_id", how="left")
_pk = PEER.groupby(["date", "ind"])
_et = {k: np.sort(g.s_et.fillna(9999).values) for k, g in _pk}
_gp = _pk.gap.mean(); _nc = _pk.gap.apply(lambda v: (v >= 0.02).sum())
pb, pn, pgap, pcand = [], [], [], []
for r in A[["date", "ind", "s_et", "gap"]].itertuples(index=False):
    v = _et.get((r.date, r.ind))
    if v is None or len(v) < 4: pb.append(np.nan); pn.append(len(v) if v is not None else 0); pgap.append(np.nan); pcand.append(np.nan); continue
    n = len(v) - 1                                                    # 扣掉自己
    pb.append((np.searchsorted(v, r.s_et, side="right") - 1) / n)     # 自己也在 ≤ s_et 裡, 扣 1
    pn.append(n); pgap.append((_gp[(r.date, r.ind)] * (n + 1) - r.gap) / n * 100); pcand.append(_nc[(r.date, r.ind)] - 1)
A["f_peer_brk"] = pb; A["peer_n"] = pn; A["f_peer_gap"] = pgap; A["f_peer_cand"] = pcand
FEAT.update({"f_peer_brk": "同類股已跌破比例", "f_peer_gap": "同類股平均跳空", "f_peer_cand": "同類股跳空≥2%檔數"})

n_day = A.groupby("date").size()
A = A.merge(n_day.rename("nday"), left_on="date", right_index=True)
A["dm"] = A.s_r - A.groupby("date").s_r.transform("mean")          # 當日去均值
A["dm1"] = A.s_r1 - A.groupby("date").s_r1.transform("mean")

P("# 多檔同時跌破時的評估與排序")
P("")
P(f"期間 {A.date.min().date()} ~ {A.date.max().date()}；探索 ≤{EXPLORE_END}、驗證之後。規則同現行 (跳空≥2% ∩ ATR≥3% ∩ 均額≥3億 ∩ 股價≤300 ∩ 非週一 ∩ 大盤跳空≥−1%), 不設上限 {len(A):,} 筆, 每筆 {A.s_r.mean():.3f}% (滑 1 檔 {A.s_r1.mean():.3f}%)。")
P("每筆含成本 0.23%、+8% 強制回補、收盤鎖漲停 −7%；停損 = 開盤 5 分區間高。")

# ---------- 1. 跌破檔數分布 ----------
P(""); P("## 1. 每天跌破幾檔、集中在哪一分鐘")
P(""); P("| 當日跌破檔數 | 天數 | 占有訊號日% | 每筆% | 有 09:05 那分鐘同時跌破 >10 檔的天數 |"); P("|---|---|---|---|---|")
first_min = A[A.s_et == A.groupby("date").s_et.transform("min")].groupby("date").size()
for lo, hi in [(1, 3), (3, 6), (6, 11), (11, 16), (16, 999)]:
    dd = n_day[(n_day >= lo) & (n_day < hi)].index
    z = A[A.date.isin(dd)]
    P(f"| {lo}~{hi-1 if hi < 999 else '+'} | {len(dd)} | {len(dd)/len(n_day)*100:.0f} | {z.s_r.mean():.3f} | {(first_min.reindex(dd) > 10).sum()} |")
P(f"\n有訊號日 {len(n_day)} 天, 平均 {n_day.mean():.1f} 檔、中位 {n_day.median():.0f} 檔；>10 檔的日子 {(n_day > 10).mean()*100:.0f}%。")
P(f"跌破時間: 09:05 那一分鐘 {(A.s_et == 5).mean()*100:.0f}%、09:10 前 {(A.s_et < 10).mean()*100:.0f}%、09:30 前 {(A.s_et < 30).mean()*100:.0f}%。")
Z = A.sort_values(["date", "s_et", "stock_id"]).assign(k=lambda x: x.groupby("date").cumcount() + 1)
for lab, m in [("先到先做第 1~5 檔", Z.k <= 5), ("第 6~10 檔", (Z.k > 5) & (Z.k <= 10)), ("第 11~15 檔 (現在被丟掉)", (Z.k > 10) & (Z.k <= 15)), ("第 16 檔以後", Z.k > 15)]:
    z = Z[m]; P(f"- {lab}: {len(z):,} 筆, 每筆 {z.s_r.mean():.3f}% (探索 {z[z.exp].s_r.mean():.3f} / 驗證 {z[~z.exp].s_r.mean():.3f}), 滑 1 檔 {z.s_r1.mean():.3f}%, 勝率 {(z.s_r > 0).mean()*100:.0f}%")

# ---------- 1b. 類股 ----------
P(""); P(f"## 1b. 類股: 同產業大多已跌破時再放空, 勝率會不會比較高 (產業來源: {IND_SRC})")
P("同類股 = 同產業、均額≥1億的上市櫃股票 (不限跳空, 扣掉自己, 至少 3 檔)；比例 = 本檔跌破那一分鐘 (含) 為止已跌破自己開盤 5 分低的檔數 / 同類股檔數。")
P(""); P("| 同類股已跌破比例 | 筆數 | 每筆% | 勝率% | 停損% | 探索 / 驗證 每筆% | 滑 1 檔 每筆% |"); P("|---|---|---|---|---|---|---|")
def srow(nm, z):
    if len(z) < 30: P(f"| {nm} | {len(z)} | | | | | |"); return
    P(f"| {nm} | {len(z):,} | {z.s_r.mean():.3f} | {(z.s_r > 0).mean()*100:.0f} | {(z.s_why == '停損').mean()*100:.0f} | {z[z.exp].s_r.mean():.3f} / {z[~z.exp].s_r.mean():.3f} | {z.s_r1.mean():.3f} |")
srow("全部", A); srow("無產業 / 同類股 <3 檔", A[A.f_peer_brk.isna()])
for lo, hi in [(0, .2), (.2, .4), (.4, .5), (.5, .6), (.6, .8), (.8, 1.01)]:
    srow(f"{lo:.0%} ~ {min(hi,1):.0%}", A[(A.f_peer_brk >= lo) & (A.f_peer_brk < hi)])
srow("≥50% (同類股大多已跌破)", A[A.f_peer_brk >= .5]); srow("<50%", A[A.f_peer_brk < .5])
for lab, m in [("09:05 那分鐘跌破", A.s_et == 5), ("09:06 以後跌破", A.s_et > 5)]:
    z = A[m]; srow(f"{lab} ∩ ≥50%", z[z.f_peer_brk >= .5]); srow(f"{lab} ∩ <50%", z[z.f_peer_brk < .5])
srow("同類股跳空≥2% 另有 ≥2 檔 (類股齊漲)", A[A.f_peer_cand >= 2]); srow("同類股跳空≥2% 另有 0~1 檔", A[A.f_peer_cand < 2])
srow("類股齊漲 ∩ 已跌破 ≥50%", A[(A.f_peer_cand >= 2) & (A.f_peer_brk >= .5)])
P(f"\n同類股檔數中位 {A.peer_n.median():.0f}；比例中位 {A.f_peer_brk.median():.0%}。")

# ---------- 2. 單一特徵 ----------
def ic_by_day(z, col, rc="s_r"):
    """每天 (≥6 檔) 特徵與報酬的 Spearman, 回傳平均與 t"""
    v = []
    for _, g in z[z.nday >= 6].groupby("date"):
        if g[col].nunique() < 3: continue
        v.append(g[col].rank().corr(g[rc].rank()))
    v = pd.Series(v).dropna()
    return (v.mean(), v.mean() / v.std() * np.sqrt(len(v))) if len(v) > 5 else (np.nan, np.nan)
P(""); P("## 2. 單一特徵")
P("全體五分位 = 所有筆數依特徵切五等分的每筆%；當日 IC = 只看當天 ≥6 檔跌破的日子, 當天內特徵排名與報酬排名的相關 (正 = 特徵越大越好)。")
P("「當日高半 − 低半」= 當天內特徵較大那一半的去均值報酬減較小那一半 (%/筆), 這才是「多檔時挑哪些」的效果。")
P(""); P("| 特徵 | 全體五分位每筆% (小→大) | 當日 IC 探索 (t) | 當日 IC 驗證 (t) | 當日高半−低半 探索 / 驗證 | 滑 1 檔 驗證 |"); P("|---|---|---|---|---|---|")
ICX = {}
M6 = A[A.nday >= 6].copy()
for col, nm in FEAT.items():
    if A[col].nunique() > 5:
        q = pd.qcut(A[col].rank(method="first"), 5, labels=False)
        qs = " / ".join(f"{A.s_r[q == i].mean():.2f}" for i in range(5))
    else:
        qs = " / ".join(f"{v:g}: {A.s_r[A[col] == v].mean():.2f}" for v in sorted(A[col].dropna().unique()))
    ie, te = ic_by_day(A[A.exp], col); iv, tv = ic_by_day(A[~A.exp], col)
    ICX[col] = (ie, te)
    hi = M6[col] > M6.groupby("date")[col].transform("median"); lo = M6[col] < M6.groupby("date")[col].transform("median")
    sp = lambda m, c="dm": M6[m & hi][c].mean() - M6[m & lo][c].mean()
    P(f"| {nm} | {qs} | {ie:+.3f} ({te:+.1f}) | {iv:+.3f} ({tv:+.1f}) | {sp(M6.exp):+.2f} / {sp(~M6.exp):+.2f} | {sp(~M6.exp, 'dm1'):+.2f} |")

# ---------- 3. 組合分數 (方向與挑選只用探索期) ----------
pick = [c for c, (ie, te) in ICX.items() if np.isfinite(te) and abs(te) >= 2.0 and c not in ("f_et",)]
sign = {c: np.sign(ICX[c][0]) for c in pick}
P(""); P("## 3. 組合分數")
P("探索期當日 IC |t| ≥ 2 的特徵 (不含跌破時間), 以當日百分位排名 × 探索期方向 平均成「綜合分數」: " + ", ".join(f"{FEAT[c]}({'+' if sign[c] > 0 else '−'})" for c in pick))
def pct_rank(z, c):  # 當日百分位 (單檔日 = 0.5)
    return z.groupby("date")[c].rank(pct=True).where(z.nday > 1, 0.5).fillna(0.5)
A["sc_combo"] = sum(sign[c] * (pct_rank(A, c) - 0.5) for c in pick) / max(len(pick), 1) if pick else 0.0
# 絕對分數 (門檻用, 不依當天其他檔): 各特徵在探索期的全體百分位
for c in pick:
    A[c] = A[c].astype(float)
    ref = np.sort(A.loc[A.exp, c].dropna().values)
    A[f"ab_{c}"] = np.searchsorted(ref, A[c].values) / len(ref)
A["ab_combo"] = sum(sign[c] * (A[f"ab_{c}"] - 0.5) for c in pick) / max(len(pick), 1) if pick else 0.0
ie, te = ic_by_day(A[A.exp], "sc_combo"); iv, tv = ic_by_day(A[~A.exp], "sc_combo")
P(f"綜合分數 當日 IC: 探索 {ie:+.3f} (t {te:+.1f}) / 驗證 {iv:+.3f} (t {tv:+.1f})")
q = pd.qcut(A.ab_combo.rank(method="first"), 5, labels=False)
P("絕對綜合分數五分位 (低→高) 每筆%: 探索 " + " / ".join(f"{A[(q == i) & A.exp].s_r.mean():.2f}" for i in range(5))
  + "；驗證 " + " / ".join(f"{A[(q == i) & ~A.exp].s_r.mean():.2f}" for i in range(5))
  + "；驗證滑 1 檔 " + " / ".join(f"{A[(q == i) & ~A.exp].s_r1.mean():.2f}" for i in range(5)))

# ---------- 4. 組合回測 ----------
def run(z, order, cap=10, keep=None, size=10, rc="s_r"):
    """order: 排序欄位 list (asc); keep: 布林欄位 (False 的不做); 每檔 = 資金/size; 有 cap 上限。回傳每日報酬 Series 與筆數表"""
    z = z if keep is None else z[z[keep]]
    z = z.sort_values(["date"] + order).groupby("date").head(cap)
    dly = (z.groupby("date")[rc].sum() / size).reindex(ALLDAYS, fill_value=0.0); dly.index = ALLDAYS
    return dly, z
def line(name, dly, z, rc="s_r"):
    eq = dly.cumsum(); mdd = (eq - eq.cummax()).min()          # 單利 (固定資金), 單位 % of 資金
    e = dly.index <= EXP
    sh = dly.mean() / dly.std() * np.sqrt(250)
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return (f"| {name} | {len(z):,} | {z[rc].mean():.3f} | {(z[rc] > 0).mean()*100:.0f} | {dly.mean():.3f} | {dly.mean()*250*3:.0f} | {sh:.2f} | {mdd:.0f}% | {dly.min():.1f}% | "
            f"{dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |")
HD = ["| 方法 | 筆數 | 每筆% | 勝率% | 日淨% | 年均 (萬, 300萬) | Sharpe | MDD (單利) | 最差日 | 探索/驗證 日淨% | 2020~2026 日淨% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
A["neg_combo"] = -A.sc_combo; A["neg_ab"] = -A.ab_combo
for c in FEAT: A[f"neg_{c}"] = -A[c]
for c in pick:
    A[f"rk_{c}"] = -sign[c] * A[c]                     # 越小越優先
thr = {}
for qq in [0.2, 0.33, 0.5]:
    t = A.loc[A.exp, "ab_combo"].quantile(qq); thr[qq] = t; A[f"keep{int(qq*100)}"] = A.ab_combo >= t

for slip, rc in [(0, "s_r"), (1, "s_r1")]:
    P(""); P(f"## 4{'ab'[slip]}. 組合回測 (每檔 = 資金/10, 最多 10 檔, 無交易日計 0) — 滑 {slip} 檔")
    P(""); [P(h) for h in HD]
    P(line("現行: 先到先做, 同分鐘依代號", *run(A, ["s_et", "stock_id"], rc=rc), rc))
    for c in pick:
        P(line(f"先到先做, 同分鐘依「{FEAT[c]}」", *run(A, ["s_et", f"rk_{c}"], rc=rc), rc))
    P(line("先到先做, 同分鐘依綜合分數", *run(A, ["s_et", "neg_combo"], rc=rc), rc))
    for qq in [0.2, 0.33, 0.5]:
        P(line(f"先到先做, 只做絕對綜合分數 ≥ 探索期 {int(qq*100)} 分位", *run(A, ["s_et", "neg_combo"], keep=f"keep{int(qq*100)}", rc=rc), rc))
    for th in [0.4, 0.5, 0.6]:
        P(line(f"先到先做, 只做同類股已跌破 ≥{th:.0%} (無產業的照做)", *run(A.assign(kp=~(A.f_peer_brk < th)), ["s_et", "stock_id"], keep="kp", rc=rc), rc))
    P(line("先到先做, 同分鐘依同類股已跌破比例 (高先)", *run(A.assign(kq=-A.f_peer_brk.fillna(0.5)), ["s_et", "kq"], rc=rc), rc))
    P(line("事後: 全天跌破者依綜合分數取前 10 (上限參考)", *run(A, ["neg_combo"], rc=rc), rc))
    P(line("事後: 全天跌破者隨機取 10 (對照)", *run(A.assign(rnd=np.random.default_rng(0).random(len(A))), ["rnd"], rc=rc), rc))
    for c in ["f_gap", "f_stopd", "f_vr", "f_tick"]:
        P(line(f"事後: 依「{FEAT[c]}」大→小取前 10", *run(A, [f"neg_{c}"], rc=rc), rc))
        P(line(f"事後: 依「{FEAT[c]}」小→大取前 10", *run(A, [c], rc=rc), rc))
    P(line("檔數上限 5 (每檔 1/10, 先到先做)", *run(A, ["s_et", "stock_id"], cap=5, rc=rc), rc))
    P(line("檔數上限 5 (每檔 1/5, 先到先做)", *run(A, ["s_et", "stock_id"], cap=5, size=5, rc=rc), rc))
    P(line("檔數上限 5 (每檔 1/5, 同分鐘依綜合分數)", *run(A, ["s_et", "neg_combo"], cap=5, size=5, rc=rc), rc))
    P(line("檔數上限 15 (每檔 1/10, 曝險可到 150%)", *run(A, ["s_et", "stock_id"], cap=15, rc=rc), rc))
    P(line("全做 (每檔 1/10, 不設上限)", *run(A, ["s_et", "stock_id"], cap=999, rc=rc), rc))
    P(line("全做, 每檔 = 資金/當日檔數 (事後等權攤, 上限參考)", run(A.assign(**{rc: A[rc] / A.nday * 10}), ["s_et"], cap=999, rc=rc)[0], A, rc))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out}")
