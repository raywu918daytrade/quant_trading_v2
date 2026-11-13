#!/usr/bin/env python3
"""研究: 籌碼 (融資融券、小台散戶多空比) 在進場前一日的狀態, 和現行空單每筆的勝率 / 報酬 / 停損率有沒有穩定關係 (怕被軋空)。只讀資料、印報告。
輸入: data/processed/d1_sr_orb.parquet (先跑 33) + data/chips/ (先跑 37)。對象 / 口徑同 35: 現行空單規則、不設每日上限、成本 0.23%。
指標一律用「進場日前一個交易日」收盤後公布的數字 (盤前就知道, 沒有偷看):
  券資比 = 融券餘額 / 融資餘額；回補天數 = 融券餘額 / 前 20 日均量；融券 1 日 / 5 日增減；融資 5 日增減；
  停券 / 強制回補: 註記出現停止融券, 或未來 7 個交易日內將停止融券 (股東會 / 除權息停券, 事前公告; 最後回補日前常見軋空)；
  小台散戶多空比 = −三大法人小台未平倉淨額 / 小台全市場未平倉；大盤融資餘額 5 日增減 (上市+上櫃加總)。
用法: python3 scripts/38_chips_squeeze.py [--out stats/chips_squeeze.md]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXPLORE_END, PROC, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(PROC / "d1_sr_orb.parquet"))
ap.add_argument("--out", default="stats/chips_squeeze.md")
a = ap.parse_args()
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
CH = ROOT / "data" / "chips"

Y = pd.read_parquet(a.inp); Y["date"] = pd.to_datetime(Y.date); Y["stock_id"] = Y.stock_id.astype(str)
ALLDAYS = pd.Series(sorted(Y.date.unique()))
S = Y[Y.s_ok]
A = S[(S.dv20 >= 3e8) & (S.atr20 >= 0.03) & (S.pc <= 300) & (S.gap >= 0.02) & (S.dow != 0) & (S.mkt_gap >= -0.01)].copy().reset_index(drop=True)
A["exp"] = A.date <= EXPLORE_END

# ---- 融資融券 ----
M = pd.concat([pd.read_parquet(p).assign(mkt=p.name.split("_")[1]) for p in sorted(CH.glob("margin_*.parquet"))], ignore_index=True)
M["date"] = pd.to_datetime(M.date); M["stock_id"] = M.stock_id.astype(str)
M = M.drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"]).reset_index(drop=True)
P(f"<!-- 融資融券 {len(M):,} 列, {M.date.min().date()} ~ {M.date.max().date()}, {M.stock_id.nunique()} 檔 -->")
notes = M.note.fillna("").astype(str)
print("註記出現的字元:", pd.Series(list("".join(notes))).value_counts().head(20).to_dict(), flush=True)
M["stop_short"] = notes.str.contains("X")
g = M.groupby("stock_id")
M["sb5"] = g.sb.shift(5); M["mb5"] = g.mb.shift(5); M["sb1"] = g.sb.shift(1)
# 未來 1~7 個交易日內出現停止融券 (且今天還沒停) → 回補壓力窗口
fut = pd.concat([g.stop_short.shift(-k) for k in range(1, 8)], axis=1).fillna(False).astype(bool).any(axis=1)
M["pre_stop"] = fut & ~M.stop_short
# 大盤融資: 上市、上櫃都有資料的日子才算 (缺一邊會造成假的大減)
mkp = M.groupby(["date", "mkt"]).mb.sum().unstack()
print("每日檔數 (上市/上櫃) 中位數:", M.groupby(["date", "mkt"]).size().unstack().median().to_dict(),
      "缺一邊的日子:", int(mkp.isna().any(axis=1).sum()), flush=True)
mkp = mkp.dropna()
for c in mkp.columns:   # 單邊單日跳動 >15% 視為資料異常
    bad = (mkp[c] / mkp[c].shift(1) - 1).abs() > 0.15
    if bad.any(): print(f"{c} 單日融資總額跳動 >15%:", list(mkp.index[bad].astype(str)), flush=True)
mk = mkp.sum(axis=1); mk = (mk / mk.shift(5) - 1) * 100
print("大盤融資 5 日增減分位:", mk.quantile([0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]).round(2).to_dict(), flush=True)
print("5 日減 ≥3% 的日子:", list(mk.index[mk <= -3].strftime("%Y-%m-%d")), flush=True)
M = M.rename(columns={"date": "mdate"})

A = A.sort_values("date")
X = pd.merge_asof(A, M[["mdate", "stock_id", "mb", "sb", "sb1", "sb5", "mb5", "ml", "stop_short", "pre_stop"]].sort_values("mdate"),
                  left_on="date", right_on="mdate", by="stock_id", allow_exact_matches=False, direction="backward")
X.loc[(X.date - X.mdate).dt.days > 7, ["mb", "sb", "sb1", "sb5", "mb5", "ml"]] = np.nan
X["has"] = X.mb.notna()
X["sr"] = np.where(X.mb > 0, X.sb / X.mb * 100, np.nan)
X["dtc"] = X.sb * 1000 / X.vol20
X["dsb1"] = (X.sb - X.sb1) * 1000 / X.vol20 * 100      # 融券 1 日增減 (占 20 日均量 %)
X["dsb5"] = np.where(X.sb5 > 0, (X.sb / X.sb5 - 1) * 100, np.nan)
X["dmb5"] = np.where(X.mb5 > 0, (X.mb / X.mb5 - 1) * 100, np.nan)
X["mbuse"] = np.where(X.ml > 0, X.mb / (X.mb + X.ml) * 100, np.nan)
X["sbv"] = X.sb * 1000 / X.vol20 * 100
# 大盤融資 5 日增減 (前一交易日)
mkd = mk.rename("mkt_dmb5").reset_index().rename(columns={"date": "kdate"}).dropna()
X = pd.merge_asof(X.sort_values("date"), mkd.sort_values("kdate"), left_on="date", right_on="kdate", allow_exact_matches=False)
# ---- 小台散戶多空比 ----
T = pd.read_parquet(CH / "mtx.parquet"); T["date"] = pd.to_datetime(T.date)
T = T.dropna(subset=["inst_net", "oi_all"]).sort_values("date")
T["rr"] = -T.inst_net / T.oi_all * 100
T["drr5"] = T.rr - T.rr.shift(5)
P(f"<!-- 小台 {len(T)} 日, {T.date.min().date()} ~ {T.date.max().date()}, 散戶多空比 中位數 {T.rr.median():.1f}% -->")
X = pd.merge_asof(X.sort_values("date"), T[["date", "rr", "drr5"]].rename(columns={"date": "tdate"}), left_on="date", right_on="tdate", allow_exact_matches=False)
X = X.reset_index(drop=True)

# ---- 報表 ----
def line(name, z):
    r = z.s_r; e = z.exp
    if len(r) < 30: return f"| {name} | {len(r)} | | | | | | |"
    return (f"| {name} | {len(r):,} ({len(r)/len(X)*100:.0f}%) | {r.mean():.3f} | {(r>0).mean()*100:.0f} | {(z.s_why=='停損').mean()*100:.0f} | "
            f"{r[e].mean():.3f} / {r[~e].mean():.3f} | {(r[e]>0).mean()*100:.0f} / {(r[~e]>0).mean()*100:.0f} |")
HDR = ["| 分組 | 筆數 (占比) | 每筆% | 勝率% | 停損% | 探索/驗證 每筆% | 探索/驗證 勝率% |", "|---|---|---|---|---|---|---|"]
def bucket(title, col, edges=None, q=5, note=""):
    P(""); P(f"### {title}");
    if note: P(note)
    P(""); [P(h) for h in HDR]
    z = X[X[col].notna()]
    if edges is None:
        edges = np.unique(np.nanquantile(z[col], np.linspace(0, 1, q + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z[col] > lo) & (z[col] <= hi) if lo != -np.inf else (z[col] <= hi)
        lab = f"≤ {hi:.1f}" if lo == -np.inf else (f"> {lo:.1f}" if hi == np.inf else f"{lo:.1f} ~ {hi:.1f}")
        P(line(lab, z[m]))
    if X[col].isna().any(): P(line("無資料", X[X[col].isna()]))

P("# 融資融券 / 小台散戶多空比 → 放空會不會比較容易被軋")
P("")
P(f"對象: 現行空單規則 (跳空≥2% ∩ ATR≥3% ∩ 均額≥3億 ∩ 股價≤300 ∩ 非週一 ∩ 大盤跳空≥−1%), 不設每日上限, {len(X):,} 筆, {X.date.min().date()} ~ {X.date.max().date()}。探索 ≤{EXPLORE_END}, 驗證之後。")
P(f"指標都是進場日**前一交易日**的數字。有融資融券資料的 {X.has.mean()*100:.0f}% 筆。")
P(""); [P(h) for h in HDR]; P(line("全部", X))

bucket("券資比 % (融券餘額 / 融資餘額)", "sr", np.array([-np.inf, 2, 5, 10, 20, 30, np.inf]))
bucket("券資比 % (五分位)", "sr")
bucket("融券餘額 / 20 日均量 % (回補天數 ×100)", "sbv", np.array([-np.inf, 1, 3, 5, 10, 20, np.inf]))
bucket("融券 1 日增減 (占 20 日均量 %)", "dsb1", np.array([-np.inf, -1, -0.2, 0.2, 1, 3, np.inf]), note="正 = 前一天融券增加 (有人放空), 負 = 前一天融券回補")
bucket("融券 5 日增減 %", "dsb5", np.array([-np.inf, -30, -10, 10, 30, 100, np.inf]))
bucket("融資 5 日增減 %", "dmb5", np.array([-np.inf, -5, 0, 5, 10, 20, np.inf]))
bucket("融資使用率 % (融資餘額 / (餘額+限額))", "mbuse")

P(""); P("### 停止融券 / 強制回補窗口 (股東會、除權息停券)"); P(""); [P(h) for h in HDR]
P(line("前一日已在停券期", X[X.stop_short.fillna(False).astype(bool)]))
P(line("未來 7 個交易日內將停券 (回補窗口)", X[X.pre_stop.fillna(False).astype(bool)]))
P(line("其他", X[~X.stop_short.fillna(False).astype(bool) & ~X.pre_stop.fillna(False).astype(bool)]))

P(""); P("### 組合: 券資比高 × 前一日融券增加 (軋空候選)"); P(""); [P(h) for h in HDR]
hi_sr = X.sr >= 20; up = X.dsb1 > 0.2
for lab, m in [("券資比≥20 且 融券增加", hi_sr & up), ("券資比≥20 且 融券沒增加", hi_sr & ~up), ("券資比<20 且 融券增加", ~hi_sr & up), ("券資比<20 且 融券沒增加", ~hi_sr & ~up & X.has)]:
    P(line(lab, X[m]))

bucket("小台散戶多空比 % (前一日, 五分位)", "rr", note="正 = 散戶偏多 (法人偏空)。常見說法: 散戶越偏多越不利多方 (反指標)")
bucket("小台散戶多空比 5 日變化 (百分點, 五分位)", "drr5")
bucket("大盤融資餘額 5 日增減 % (五分位)", "mkt_dmb5")

# ---- 組合 ----
def port(z, flag=None, keep_only=False, w_on=1.0, w_off=1.0, cap=10, name=""):
    z = z.sort_values(["date", "s_et", "stock_id"])
    z = z[z[flag]].groupby("date").head(cap) if keep_only else z.groupby("date").head(cap)
    w = np.where(z[flag], w_on, w_off) if flag else 1.0
    dly = (z.s_r * w).groupby(z.date).sum().div(cap).reindex(ALLDAYS, fill_value=0.0); dly.index = ALLDAYS
    eq = (1 + dly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    e = dly.index <= EXPLORE_END
    yr = " ".join(f"{dly[dly.index.year == y].mean():.2f}" for y in range(2020, 2027))
    return f"| {name} | {len(z):,} | {z.s_r.mean():.3f} | {(z.s_r>0).mean()*100:.0f} | {dly.mean():.3f} | {dly.mean()/dly.std()*np.sqrt(250):.2f} | {mdd:.0f}% | {dly[e].mean():.3f} / {dly[~e].mean():.3f} | {yr} |"
X["ok_sr30"] = ~(X.sr >= 30); X["ok_sr20"] = ~(X.sr >= 20); X["ok_sbv"] = ~(X.sbv >= 10)
X["ok_stop"] = ~(X.stop_short.fillna(False).astype(bool) | X.pre_stop.fillna(False).astype(bool))
X["ok_sq"] = ~(hi_sr & up)
P(""); P("## 組合: 每日最多 10 檔 (依跌破時間先到先做), 每檔 1/10, 無交易日計 0")
P(""); P("| 規則 | 筆數 | 每筆% | 勝率% | 日淨% | Sharpe | MDD | 探索/驗證 日淨% | 2020~2026 日淨% |"); P("|---|---|---|---|---|---|---|---|---|")
P(port(X, name="現行 (先到先做)"))
for f, lab in [("ok_sr20", "券資比≥20%"), ("ok_sr30", "券資比≥30%"), ("ok_sbv", "融券餘額≥10% 均量"), ("ok_stop", "停券 / 回補窗口"), ("ok_sq", "券資比≥20 且融券增加")]:
    P(port(X, f, keep_only=True, name=f"跳過「{lab}」(名額遞補)"))
    P(port(X, f, w_on=1.0, w_off=0.5, name=f"「{lab}」改 0.5 格"))
# 散戶多空比: 日層級
lo_, hi_ = np.nanquantile(T.rr, [0.2, 0.8])
for lab, ok in [(f"散戶多空比 ≥ {hi_:.1f}% (前 20% 偏多) 當天不做", ~(X.rr >= hi_)), (f"散戶多空比 ≤ {lo_:.1f}% (前 20% 偏空) 當天不做", ~(X.rr <= lo_))]:
    X["ok_rr"] = ok
    P(port(X, "ok_rr", keep_only=True, name=lab))
for th in [-1.0, -1.7, -3.0]:
    X["ok_mk"] = ~(X.mkt_dmb5 <= th)
    P(port(X, "ok_mk", keep_only=True, name=f"大盤融資 5 日減 ≥{-th:.1f}% 當天不做"))
    P(port(X, "ok_mk", w_on=1.0, w_off=0.5, name=f"大盤融資 5 日減 ≥{-th:.1f}% 當天 0.5 格"))
X["ok_has"] = X.has
P(port(X, "ok_has", keep_only=True, name="跳過「無融資融券資料」"))
P("")
P("### 大盤融資 5 日減 ≥1.7% 的日子, 逐年 (每筆% / 筆數)")
P("")
P("| 年 | 減 ≥1.7% 每筆% (筆) | 其他 每筆% (筆) |"); P("|---|---|---|")
for y in range(2020, 2027):
    z = X[X.date.dt.year == y]; m = z.mkt_dmb5 <= -1.7
    P(f"| {y} | {z[m].s_r.mean():.3f} ({m.sum()}) | {z[~m].s_r.mean():.3f} ({(~m).sum()}) |")
P("")
P("注意: 「跳過」= 該筆不做, 當天名額讓給後面跌破的股票。0.5 格版本總曝險較低, 比 Sharpe 較公平。")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
print(f"寫入 {a.out}")
