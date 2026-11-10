#!/usr/bin/env python3
"""診斷: 收盤回補 (13:30 集合競價) vs 13:20~13:24 自己市價平倉, 哪個好。只讀資料、印報告, 不改規則。
對象: 定案規則歷史 (HF db/orb/history/trades.parquet) 中 13:20 時仍持有的空單 (收盤回補的, 以及 13:20 後才停損的)。
提早平倉價 = 該分鐘分 K 收盤 × (1 + 滑價), 滑價預設 0.1% (市價單吃一檔左右); 若該分鐘已在漲停 (≥漲停×0.999) 視為買不到, 沿用原結果。
用法: python3 scripts/diag_exit_time.py [--slip 0.001] [--exclude 2026-11-10] [--out stats/exit_time.md]"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, HF_PREFIX, COST

ap = argparse.ArgumentParser()
ap.add_argument("--slip", type=float, default=0.001)
ap.add_argument("--exclude", default="2026-11-10")
ap.add_argument("--out", default="stats/exit_time.md")
a = ap.parse_args()
EXCL = {pd.Timestamp(x) for x in a.exclude.split(",") if x}
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)
hf = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset")

tr = pd.read_parquet(hf(f"{HF_PREFIX}/history/trades.parquet"))
tr["date"] = pd.to_datetime(tr.date); tr = tr[~tr.date.isin(EXCL)].copy()
tr["stock_id"] = tr.stock_id.astype(str)
print("trades", len(tr), tr.columns.tolist()); print(tr.head(3).to_string())
def to_t(v):
    if isinstance(v, str) and ":" in v:
        h, mm = v.split(":")[:2]; return int(h) * 60 + int(mm) - 540
    try: return int(float(v))
    except Exception: return np.nan
tr["xt"] = tr.exit_t.map(to_t)
hold = tr[tr.xt >= 260].copy()          # 13:20 時仍持有
print("13:20 仍持有", len(hold), "其中收盤回補", hold.reason.str.startswith("收盤").sum())

files = set(HfApi().list_repo_files(HF_REPO, repo_type="dataset"))
B = []
for ym in sorted(hold.date.dt.strftime("%Y_%m").unique()):
    f = f"db/m1/{ym}.parquet"
    if f not in files: print("沒有", f); continue
    m = pd.read_parquet(hf(f), columns=["stock_id", "date", "close", "high"])
    m["stock_id"] = m.stock_id.astype(str); ts = pd.to_datetime(m.date)
    m["t"] = ts.dt.hour * 60 + ts.dt.minute - 540; m["d"] = ts.dt.normalize()
    h = hold[hold.date.dt.strftime("%Y_%m") == ym][["date", "stock_id"]]
    m = m[(m.t >= 255) & (m.t <= 270)].merge(h, left_on=["d", "stock_id"], right_on=["date", "stock_id"])
    B.append(m[["date", "stock_id", "t", "close", "high"]])
    print(ym, len(m), flush=True)
    try: Path(hf(f)).resolve().unlink()
    except Exception: pass
bars = pd.concat(B, ignore_index=True)
px = bars.pivot_table(index=["date", "stock_id"], columns="t", values="close", aggfunc="last")
# 該分鐘沒成交 → 用前一分鐘收盤
px = px.reindex(columns=range(255, 271)).ffill(axis=1)
hold = hold.merge(px, left_on=["date", "stock_id"], right_index=True, how="left")

lu = hold.pc * 1.10 if "pc" in hold else hold.entry_px * np.nan
if "lu" in hold: lu = hold.lu
P("# 收盤回補 vs 13:20~13:24 提早平倉")
P("")
P(f"對象: 定案規則回測 {hold.date.min().date()} ~ {hold.date.max().date()} 中 13:20 時仍持有的空單 {len(hold)} 筆 (排除 {', '.join(str(d.date()) for d in sorted(EXCL))})。")
P(f"提早平倉價 = 該分鐘 1 分 K 收盤 × (1+{a.slip*100:.1f}%) 市價滑價；成本 {COST*100:.2f}% 不變；該分鐘在漲停買不到者沿用原結果。原結果 = 規則實際 (13:30 收盤競價, 含鎖漲停 −7% 罰金, 或 13:20 後停損)。")
chk = hold[hold.reason.str.startswith("收盤")]
if 270 in chk:
    dd = (chk[270] / chk.exit_px - 1).abs()
    P(f"核對: 收盤回補的 13:30 分 K 收盤 = 回測回補價 的比例 {(dd < 1e-4).mean()*100:.0f}% (分 K 13:30 那根就是收盤競價)。")
P("")
P("| 平倉時間 | 筆數 | 原每筆% | 提早每筆% | 差 (提早−原) %/筆 | 提早較好的比例 | 2020~2023 差 | 2024~ 差 | 合計差 (萬, 每筆 25 萬) |")
P("|---|---|---|---|---|---|---|---|---|")
res = {}
for t in range(260, 265):
    p = hold[t]; ok = p.notna() & ~(p >= (hold.pc * 1.1) * 0.999 if "pc" in hold else False)
    r_alt = np.where(ok, (-(p * (1 + a.slip) / hold.entry_px - 1) - COST) * 100, hold.r_pct)
    d = pd.Series(r_alt - hold.r_pct, index=hold.index); res[t] = d
    e = hold.date < "2024-01-01"
    P(f"| 13:{t-240:02d} | {len(hold)} | {hold.r_pct.mean():.3f} | {np.nanmean(r_alt):.3f} | {d.mean():+.3f} | {(d > 0).mean()*100:.0f}% | {d[e].mean():+.3f} | {d[~e].mean():+.3f} | {d.sum()*0.25/100*100:+.1f} |")
P("")
d = res[262]
P("## 13:22 平倉 vs 收盤: 依年份")
P("")
P("| 年 | 筆數 | 差 %/筆 | 提早較好比例 |")
P("|---|---|---|---|")
for y, z in d.groupby(hold.date.dt.year):
    P(f"| {y} | {len(z)} | {z.mean():+.3f} | {(z > 0).mean()*100:.0f}% |")
P("")
lk = hold.get("locked", pd.Series(False, index=hold.index)).fillna(False).astype(bool)
P(f"鎖漲停收盤的筆數 {int(lk.sum())}: 這些提早平倉平均差 {res[262][lk].mean():+.2f}%/筆；其餘 {res[262][~lk].mean():+.3f}%/筆。")
mv = (hold[270] / hold[262] - 1) * 100
P(f"13:22 → 13:30 個股平均變動 {mv.mean():+.3f}% (正 = 尾盤上漲, 對空單不利), 中位 {mv.median():+.3f}%, 上漲比例 {(mv > 0).mean()*100:.0f}%。")
P("")
P("滑價敏感度 (13:22): " + "；".join(f"滑價 {s*100:.2f}% → 差 {((np.where(hold[262].notna(), (-(hold[262]*(1+s)/hold.entry_px-1)-COST)*100, hold.r_pct)) - hold.r_pct).mean():+.3f}%/筆" for s in [0, 0.0005, 0.001, 0.002]))

Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text("\n".join(OUT) + "\n")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f: f.write("\n".join(OUT) + "\n")
