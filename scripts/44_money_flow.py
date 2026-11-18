#!/usr/bin/env python3
"""研究 44: 資金流向 (三大法人 / 融資融券 / 八大行庫 / 分點) 能不能提高現行空單的期望值。只讀資料、印報告, 不改線上規則。
輸入: data/processed/d1_sr_orb.parquet (先跑 33) + data/flows/ (43) + data/chips/ (37) + HF db/d1 (算大盤金額用的收盤價)。
對象: 現行規則 (2026-11-18 起: 跳空≥2%、ATR≥3%、均額≥3億、股價≤300、非週一、大盤跳空≥−1%、開盤5分成交額≥1.5億、一檔≤0.2%;
      權重 = (區間寬≥2.1% → 2, 否則 1) × (區間高碰日K壓力 → 1.5, 否則 0.75); 先到先做, 當日權重累計 ≤10 格)。成本 0.23%。
指標一律用進場日「前一個交易日」收盤後公布的數字 (盤前已知, 沒有偷看)。
門檻一律用探索期 (≤2023) 的分位數決定, 驗證期 (2024~) 只看結果 = 樣本外。
用法: python3 scripts/44_money_flow.py [--out stats/money_flow.md]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXPLORE_END, PROC, ROOT, HF_REPO

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--out", default="stats/money_flow.md")
ap.add_argument("--d1", default="", help="測試用: 本地日K parquet (不從 HF 下載)")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
FL, CH = ROOT / "data" / "flows", ROOT / "data" / "chips"
EXP = pd.Timestamp(EXPLORE_END)

# ---------------- 交易 (現行規則) ----------------
Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy()
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
A["tick_pct"] = tick(A.rl.values) / A.rl * 100
A["amt"] = A.o5_v * 1000 * A.rl
A = A[(A.amt >= 1.5e8) & (A.tick_pct <= 0.2)].copy()
A["w0"] = np.where((A.rh / A.rl - 1) * 100 >= 2.1, 2.0, 1.0) * np.where(A.lvl_in.fillna(False).astype(bool), 1.5, 0.75)
A["s_r1"] = A.s_r - A.tick_pct * (1 + (A.s_why == "停損"))
A["exp"] = A.date <= EXP
A = A.sort_values("date").reset_index(drop=True)

# ---------------- HF 日K (收盤價 / 成交量) ----------------
def wide(df, c):
    return df.pivot_table(index="date", columns="stock_id", values=c, aggfunc="last")
if a.d1: D = pd.read_parquet(a.d1)
else:
    from huggingface_hub import hf_hub_download, HfApi
    files = sorted(f for f in HfApi().list_repo_files(HF_REPO, repo_type="dataset") if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:10] >= "2019")
    D = pd.concat([pd.read_parquet(hf_hub_download(HF_REPO, f, repo_type="dataset"), columns=["stock_id", "date", "close", "volume"]) for f in files], ignore_index=True)
D["stock_id"] = D.stock_id.astype(str); D["date"] = pd.to_datetime(D.date).dt.normalize()
D = D[D.stock_id.str.fullmatch(r"\d{4}") & ~D.stock_id.str.startswith("00")].drop_duplicates(["stock_id", "date"], keep="last")
CL = wide(D, "close"); TV = (CL * wide(D, "volume")).sum(axis=1)                 # 全市場 (4 碼) 成交值

# ---------------- 三大法人 (個股) ----------------
I = pd.concat([pd.read_parquet(p).assign(mkt=p.name.split("_")[1]) for p in sorted(FL.glob("inst_*.parquet"))], ignore_index=True)
I["date"] = pd.to_datetime(I.date); I["stock_id"] = I.stock_id.astype(str).str.strip()
I = I[I.stock_id.str.fullmatch(r"\d{4}") & ~I.stock_id.str.startswith("00")].drop_duplicates(["date", "stock_id"], keep="last")
P(f"<!-- 三大法人 {len(I):,} 列, {I.date.min().date()} ~ {I.date.max().date()}, {I.stock_id.nunique()} 檔, 缺值 {I[['fi','it','dl','tot']].isna().mean().round(3).to_dict()} -->")
days_i = sorted(I.date.unique())
def holes(df, Wd):
    """上市 / 上櫃各自哪幾天沒抓到 → 該市場的股票那天設 NaN (不要當成 0)。"""
    mk_of = df.sort_values("date").groupby("stock_id").mkt.last()
    for m, g in df.groupby("mkt"):
        miss = ~Wd.index.isin(g.date.unique())
        cols = [c for c in Wd.columns if mk_of.get(c) == m]
        if miss.any(): Wd.loc[miss, cols] = np.nan
        print(f"  {m}: 缺 {int(miss.sum())} 個交易日 (最後 {g.date.max().date()})", flush=True)
    return Wd
feat = {}           # name -> wide (date × stock), 值 = 該日收盤後已知
for c in ["fi", "it", "dl", "tot"]:
    W = holes(I, wide(I, c).reindex(days_i).fillna(0.0))   # 沒列出 = 當天法人沒買賣; 整個市場缺的日子 = NaN
    feat[f"{c}1"] = W; feat[f"{c}5"] = W.rolling(5, min_periods=5).sum()
    pos = (W > 0).astype(int).where(W.notna()); neg = (W < 0).astype(int).where(W.notna())
    # 連續買超 (+) / 賣超 (−) 天數
    def streak(b):
        grp = (b == 0).cumsum(); return b.groupby(grp).cumsum()
    feat[f"{c}_st"] = (pos.fillna(0).apply(streak) - neg.fillna(0).apply(streak)).where(W.notna())

# ---------------- 融資融券 (個股) ----------------
M = pd.concat([pd.read_parquet(p).assign(mkt=p.name.split("_")[1]) for p in sorted(CH.glob("margin_*.parquet"))], ignore_index=True)
M["date"] = pd.to_datetime(M.date); M["stock_id"] = M.stock_id.astype(str)
M = M[M.stock_id.str.fullmatch(r"\d{4}")].drop_duplicates(["stock_id", "date"], keep="last")
P(f"<!-- 融資融券 {len(M):,} 列, {M.date.min().date()} ~ {M.date.max().date()} -->")
days_m = sorted(M.date.unique())
MB = wide(M, "mb").reindex(days_m); SB = wide(M, "sb").reindex(days_m)
feat["mb1"] = (MB - MB.shift(1)) * 1000; feat["mb5"] = (MB - MB.shift(5)) * 1000      # 張 → 股
feat["sb1"] = (SB - SB.shift(1)) * 1000; feat["sb5"] = (SB - SB.shift(5)) * 1000

# ---------------- 八大行庫 / 分點 (有資料才做) ----------------
for nm in ["gb", "bk"]:          # gb = 八大行庫合計買賣超 (股); bk = 隔日沖分點合計買賣超 (股) — 目前沒有免費歷史資料
    fs = sorted(FL.glob(f"{nm}_*.parquet"))
    if not fs: continue
    G = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    G["date"] = pd.to_datetime(G.date); G["stock_id"] = G.stock_id.astype(str)
    P(f"<!-- {nm} {len(G):,} 列, {G.date.min().date()} ~ {G.date.max().date()} -->")
    Wg = wide(G, "net").reindex(sorted(G.date.unique())).fillna(0.0)
    feat[f"{nm}1"] = Wg; feat[f"{nm}5"] = Wg.rolling(5, min_periods=5).sum()

# ---------------- 當沖 (隔日沖替代指標) ----------------
fs = sorted(FL.glob("dt_*.parquet"))
if fs:
    T = pd.concat([pd.read_parquet(p).assign(mkt=p.name.split("_")[1][2:]) for p in fs], ignore_index=True)
    T["date"] = pd.to_datetime(T.date); T["stock_id"] = T.stock_id.astype(str).str.strip()
    T = T[T.stock_id.str.fullmatch(r"\d{4}")].drop_duplicates(["date", "stock_id"], keep="last")
    P(f"<!-- 當沖 {len(T):,} 列, {T.date.min().date()} ~ {T.date.max().date()} -->")
    Wt = holes(T, wide(T, "dtv").reindex(sorted(T.date.unique())).fillna(0.0))
    VO = wide(D, "volume").reindex(index=Wt.index, columns=Wt.columns)
    feat["dtr1"] = (Wt / VO * 100).where(VO > 0)                      # 前一日當沖率 % (當沖股數 / 成交股數)
    feat["dtr5"] = (Wt.rolling(5).sum() / VO.rolling(5).sum() * 100).where(VO.rolling(5).sum() > 0)
    MDT = (Wt * CL.reindex(index=Wt.index, columns=Wt.columns)).sum(axis=1) / TV.reindex(Wt.index) * 100   # 大盤當沖占成交值 %
    MDT[Wt.isna().any(axis=1)] = np.nan                                  # 有市場缺資料的日子不算大盤

def attach(X, name, W, norm=True):
    """X 每列 (date, stock_id) 取 W 在 date 之前最後一個交易日的值; norm: ÷ 20 日均量 (股) × 100。"""
    L = W.stack().rename(name).reset_index().rename(columns={"date": "fdate"}).sort_values("fdate")
    L["stock_id"] = L.stock_id.astype(str)
    m = pd.merge_asof(X[["date", "stock_id"]].reset_index().sort_values("date"), L, left_on="date", right_on="fdate",
                      by="stock_id", allow_exact_matches=False, direction="backward").set_index("index").sort_index()
    v = m[name].where((m.date - m.fdate).dt.days <= 7)
    return v / X.vol20 * 100 if norm else v
NORM = {k: not (k.endswith("_st") or k.startswith("dtr")) for k in feat}
for k, W in feat.items(): A[k] = attach(A, k, W, NORM[k])

# ---------------- 大盤層面 (日) ----------------
mk = pd.DataFrame(index=pd.DatetimeIndex(days_i))
for c in ["fi", "it", "dl", "tot"]:
    W = feat[f"{c}1"]; cl = CL.reindex(index=W.index, columns=W.columns)
    amt = (W * cl).sum(axis=1)                                                   # 元
    amt[W.isna().any(axis=1)] = np.nan                                           # 有市場缺資料的日子不算大盤
    tv20 = TV.reindex(W.index).rolling(20, min_periods=10).mean()
    mk[f"M_{c}1"] = amt / tv20 * 100; mk[f"M_{c}5"] = amt.rolling(5).sum() / tv20 * 100   # 占 20 日均成交值 %
MBt = M.groupby(["date", "mkt"]).mb.sum().unstack().dropna().sum(axis=1)
mk = mk.join((MBt / MBt.shift(5) - 1).mul(100).rename("M_mb5"), how="outer").join((MBt / MBt.shift(1) - 1).mul(100).rename("M_mb1"), how="outer")
for nm in ["gb", "bk"]:
    if f"{nm}1" in feat:
        W = feat[f"{nm}1"]; cl = CL.reindex(index=W.index, columns=W.columns)
        amt = (W * cl).sum(axis=1); tv20 = TV.reindex(W.index).rolling(20, min_periods=10).mean()
        mk[f"M_{nm}1"] = amt / tv20 * 100; mk[f"M_{nm}5"] = amt.rolling(5).sum() / tv20 * 100
if fs: mk = mk.join(MDT.rename("M_dtr1"), how="outer")
mk = mk.sort_index(); mk.index.name = "kdate"
A = pd.merge_asof(A.sort_values("date"), mk.reset_index(), left_on="date", right_on="kdate", allow_exact_matches=False).reset_index(drop=True)
A.loc[(A.date - A.kdate).dt.days > 7, [c for c in mk.columns]] = np.nan

NAMES = {"fi1": "外資 前1日買賣超 (%均量)", "fi5": "外資 前5日累計 (%均量)", "fi_st": "外資 連買(+)/連賣(−)天數",
         "it1": "投信 前1日 (%均量)", "it5": "投信 前5日累計", "it_st": "投信 連買/連賣天數",
         "dl1": "自營商 前1日 (%均量)", "dl5": "自營商 前5日累計", "dl_st": "自營商 連買/連賣天數",
         "tot1": "三大法人 前1日 (%均量)", "tot5": "三大法人 前5日累計", "tot_st": "三大法人 連買/連賣天數",
         "mb1": "融資 前1日增減 (%均量)", "mb5": "融資 前5日增減 (%均量)", "sb1": "融券 前1日增減 (%均量)", "sb5": "融券 前5日增減 (%均量)",
         "gb1": "八大行庫 前1日 (%均量)", "gb5": "八大行庫 前5日累計", "bk1": "隔日沖分點 前1日 (%均量)", "bk5": "隔日沖分點 前5日累計", "dtr1": "前1日當沖率 %", "dtr5": "前5日當沖率 %",
         "M_fi1": "大盤 外資 前1日 (%均成交值)", "M_fi5": "大盤 外資 前5日", "M_it1": "大盤 投信 前1日", "M_it5": "大盤 投信 前5日",
         "M_dl1": "大盤 自營商 前1日", "M_dl5": "大盤 自營商 前5日", "M_tot1": "大盤 三大法人 前1日", "M_tot5": "大盤 三大法人 前5日",
         "M_mb1": "大盤 融資餘額 1日增減%", "M_mb5": "大盤 融資餘額 5日增減%", "M_gb1": "大盤 八大行庫 前1日", "M_gb5": "大盤 八大行庫 前5日",
         "M_bk1": "大盤 隔日沖分點 前1日", "M_bk5": "大盤 隔日沖分點 前5日", "M_dtr1": "大盤 前1日當沖占成交值 %"}
FEATS = [k for k in NAMES if k in A.columns]

# ---------------- 組合 ----------------
def run(keep=None, mult=None, rc="s_r"):
    z = A.assign(w=A.w0 * (1.0 if mult is None else mult))
    if keep is not None: z = z[keep]
    z = z[z.w > 0].sort_values(["date", "s_et", "stock_id"])
    z = z[z.groupby("date").w.cumsum() <= 10 + 1e-9]
    dly = (z.w * z[rc]).groupby(z.date).sum().reindex(ALLDAYS, fill_value=0.0) / 10; dly.index = ALLDAYS
    return dly, z
def stats(d):
    e = d.index <= EXP; eq = d.cumsum()
    sh = lambda x: x.mean() / x.std() * np.sqrt(250) if x.std() > 0 else np.nan
    return dict(ann=d.mean() * 750, ann_e=d[e].mean() * 750, ann_v=d[~e].mean() * 750, sh=sh(d), sh_e=sh(d[e]), sh_v=sh(d[~e]),
                mdd=(eq - eq.cummax()).min() * 3)
BASE = {rc: stats(run(rc=rc)[0]) for rc in ["s_r", "s_r1"]}
b0 = BASE["s_r"]

P("# 資金流向 (三大法人 / 融資融券 / 八大行庫 / 分點) → 現行空單期望值")
P("")
P(f"對象: 現行規則 (含 11-18 起的成交額 / 一檔篩選與權重), 不設上限 {len(A):,} 筆, {A.date.min().date()} ~ {A.date.max().date()}；探索 ≤{EXPLORE_END} ({A.exp.sum():,} 筆)、驗證 2024~ ({(~A.exp).sum():,} 筆)。")
P("指標都是進場日**前一交易日**收盤後的數字。門檻用探索期分位數定, 驗證期 = 樣本外。")
P(f"現行組合 (300 萬, 累計 ≤10 格): 年均 {b0['ann']:.0f} 萬 (探索 {b0['ann_e']:.0f} / 驗證 {b0['ann_v']:.0f})、Sharpe {b0['sh']:.2f} ({b0['sh_e']:.2f} / {b0['sh_v']:.2f})、MDD {b0['mdd']:.0f} 萬；"
  f"滑 1 檔 年均 {BASE['s_r1']['ann']:.0f} 萬、Sharpe {BASE['s_r1']['sh']:.2f}。")
P(f"資料涵蓋 (占筆數): " + "、".join(f"{NAMES[k].split(' ')[0] if not k.startswith('M_') else NAMES[k][:6]} {A[k].notna().mean()*100:.0f}%" for k in FEATS if k.endswith("1")))

# ---------------- 1. 五分位 ----------------
P(""); P("## 1. 各指標五分位 (門檻 = 探索期分位; 每筆% 未加權、含成本)")
P("Q1 = 最小 (最多賣超 / 減少), Q5 = 最大 (最多買超 / 增加)。")
P(""); P("| 指標 | 門檻 (Q1/Q2/Q3/Q4 上緣) | 探索 每筆% Q1→Q5 | 驗證 每筆% Q1→Q5 | 驗證 勝率% Q1→Q5 | 單調? |"); P("|---|---|---|---|---|---|")
QE = {}
for k in FEATS:
    z = A[A[k].notna()]
    e = np.unique(np.nanquantile(z.loc[z.exp, k], [0.2, 0.4, 0.6, 0.8]))
    QE[k] = e
    q = np.searchsorted(e, z[k].values, side="left")
    nq = len(e) + 1
    me = [z.s_r[(q == i) & z.exp.values].mean() for i in range(nq)]
    mv = [z.s_r[(q == i) & ~z.exp.values].mean() for i in range(nq)]
    wv = [(z.s_r[(q == i) & ~z.exp.values] > 0).mean() * 100 for i in range(nq)]
    rk = lambda v: pd.Series(v).rank().corr(pd.Series(range(nq), dtype=float))      # Spearman (不用 scipy)
    sp_e, sp_v = rk(me), rk(mv)
    mono = "同向" if np.sign(sp_e) == np.sign(sp_v) and abs(sp_e) >= 0.7 and abs(sp_v) >= 0.7 else ""
    P(f"| {NAMES[k]} | {' / '.join(f'{x:.2g}' for x in e)} | {' / '.join(f'{x:.2f}' for x in me)} | {' / '.join(f'{x:.2f}' for x in mv)} | "
      f"{' / '.join(f'{x:.0f}' for x in wv)} | {mono} |")

# ---------------- 2. 規則掃描 ----------------
P(""); P("## 2. 規則掃描: 兩端 (≤探索 20 分位 / ≥80 分位) 不做、0.5 格、1.5 格")
P(f"Δ = 相對現行 (年均 萬, Sharpe)。探索 / 驗證 分開算。✔ = 探索有改善且驗證也改善 (年均不降、Sharpe +0.1 以上)。")
P(""); P("| 指標 | 端 | 占比 | 探索 每筆% 此端/其他 | 驗證 每筆% 此端/其他 | 不做 Δ年均 探/驗 | 不做 ΔSharpe 探/驗 | 0.5格 ΔSharpe 探/驗 | 1.5格 ΔSharpe 探/驗 | |")
P("|---|---|---|---|---|---|---|---|---|---|")
CAND = []
for k in FEATS:
    lo, hi = np.nanquantile(A.loc[A.exp, k], [0.2, 0.8])
    for side, m in [("低", A[k] <= lo), ("高", A[k] >= hi)]:
        m = m.fillna(False).values
        if m.sum() < 50: continue
        e = A.exp.values
        st = {}
        for lab, keep, mult in [("skip", ~m, None), ("half", None, np.where(m, 0.5, 1.0)), ("boost", None, np.where(m, 1.5, 1.0))]:
            st[lab] = stats(run(keep, mult)[0])
        d = lambda lab, f: (st[lab][f + "_e"] - b0[f + "_e"], st[lab][f + "_v"] - b0[f + "_v"])
        ok = []
        for lab in ["skip", "half", "boost"]:
            ae, av = d(lab, "ann"); se, sv = d(lab, "sh")
            good = se >= 0.1 and sv >= 0.1 and av >= -3 and ae >= -3
            if good: ok.append(lab); CAND.append((k, side, lab, m, sv, av))
        P(f"| {NAMES[k]} | {side} | {m.mean()*100:.0f}% | {A.s_r[m & e].mean():.2f} / {A.s_r[~m & e].mean():.2f} | {A.s_r[m & ~e].mean():.2f} / {A.s_r[~m & ~e].mean():.2f} | "
          f"{d('skip','ann')[0]:+.0f} / {d('skip','ann')[1]:+.0f} | {d('skip','sh')[0]:+.2f} / {d('skip','sh')[1]:+.2f} | "
          f"{d('half','sh')[0]:+.2f} / {d('half','sh')[1]:+.2f} | {d('boost','sh')[0]:+.2f} / {d('boost','sh')[1]:+.2f} | {' '.join('✔' + x for x in ok)} |")

# ---------------- 3. 候選細看 ----------------
P(""); P("## 3. 掃描過關的候選: 逐年、滑 1 檔、回撤")
P(f"掃描共 {sum(1 for _ in FEATS) * 2 * 3} 組規則, 隨機也會有幾組過關 → 要逐年多數成立、滑 1 檔仍有效才算。")
P(""); P("| 規則 | 年均 (萬) | Sharpe | MDD (萬) | 滑1檔 年均 / Sharpe | 驗證 年均 / Sharpe | 逐年 Δ年均 (2020~2026) | 贏的年數 |"); P("|---|---|---|---|---|---|---|---|")
def yearly(d): return np.array([d[d.index.year == y].mean() * 750 for y in range(2020, 2027)])
d_base = run()[0]; yb = yearly(d_base)
P(f"| 現行 | {b0['ann']:.0f} | {b0['sh']:.2f} | {b0['mdd']:.0f} | {BASE['s_r1']['ann']:.0f} / {BASE['s_r1']['sh']:.2f} | {b0['ann_v']:.0f} / {b0['sh_v']:.2f} | " + " ".join(f"{x:.0f}" for x in yb) + " | |")
for k, side, lab, m, sv, av in sorted(CAND, key=lambda x: -x[4])[:15]:
    keep, mult = (~m, None) if lab == "skip" else (None, np.where(m, 0.5 if lab == "half" else 1.5, 1.0))
    d0 = run(keep, mult)[0]; s0 = stats(d0); s1 = stats(run(keep, mult, "s_r1")[0])
    dy = yearly(d0) - yb
    nm = {"skip": "不做", "half": "0.5 格", "boost": "1.5 格"}[lab]
    P(f"| {NAMES[k]} {side}端 {nm} | {s0['ann']:.0f} | {s0['sh']:.2f} | {s0['mdd']:.0f} | {s1['ann']:.0f} / {s1['sh']:.2f} | {s0['ann_v']:.0f} / {s0['sh_v']:.2f} | "
      + " ".join(f"{x:+.0f}" for x in dy) + f" | {(dy > 0).sum()}/7 |")
if not CAND: P("| (沒有規則過關) | | | | | | | |")

# ---------------- 4. 舊候選: 大盤融資 5 日減 ≥3% 隔天不做 (新規則重算) ----------------
P(""); P("## 4. 舊候選重算: 大盤融資餘額 5 日減 ≥ X% 隔天不做 / 0.5 格 (研究 38 用舊規則 Sharpe 2.86→3.30)")
P(""); P("| 規則 | 停做天數 | 年均 (萬) | Sharpe | MDD (萬) | 探索 / 驗證 年均 | 探索 / 驗證 Sharpe | 滑1檔 年均 / Sharpe | 逐年 Δ年均 |"); P("|---|---|---|---|---|---|---|---|---|")
P(f"| 現行 | 0 | {b0['ann']:.0f} | {b0['sh']:.2f} | {b0['mdd']:.0f} | {b0['ann_e']:.0f} / {b0['ann_v']:.0f} | {b0['sh_e']:.2f} / {b0['sh_v']:.2f} | {BASE['s_r1']['ann']:.0f} / {BASE['s_r1']['sh']:.2f} | |")
for th in [1.0, 2.0, 3.0, 4.0]:
    m = (A.M_mb5 <= -th).fillna(False).values
    nd = mk.index[(mk.M_mb5.shift(1) <= -th)].isin(ALLDAYS).sum()
    for lab, keep, mult in [("不做", ~m, None), ("0.5 格", None, np.where(m, 0.5, 1.0))]:
        d0 = run(keep, mult)[0]; s0 = stats(d0); s1 = stats(run(keep, mult, "s_r1")[0])
        P(f"| 5 日減 ≥{th:.0f}% {lab} | {nd} | {s0['ann']:.0f} | {s0['sh']:.2f} | {s0['mdd']:.0f} | {s0['ann_e']:.0f} / {s0['ann_v']:.0f} | {s0['sh_e']:.2f} / {s0['sh_v']:.2f} | "
          f"{s1['ann']:.0f} / {s1['sh']:.2f} | " + " ".join(f"{x:+.0f}" for x in yearly(d0) - yb) + " |")

P("")
P("說明: 「不做」= 該筆跳過, 名額讓給之後跌破的單; 0.5 / 1.5 格 = 該筆權重 ×0.5 / ×1.5 (累計 ≤10 格)。Sharpe 不受部位大小影響, 用來比加權是否真的較好。")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out}")
