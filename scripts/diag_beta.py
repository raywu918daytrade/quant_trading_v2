#!/usr/bin/env python3
"""診斷: 策略日損益跟大盤 (加權指數 / 0050 / 流動股等權) 的連動有多大, 以及對沖 / 大盤濾網能不能降低。只讀資料、印報告, 不改規則。
資料: HF db/orb/history/{days,trades}.parquet (回測 = 定案規則歷史)、db/orb/web/live_index.json (即時訊號 vs 回測)、db/orb/paper/summary.csv (模擬單)、
      db/d1/*.parquet (0050 與流動股等權)、證交所 MI_5MINS_HIST (加權指數日 OHLC)。
用法: python3 scripts/diag_beta.py [--exclude 2026-11-10] [--out stats/market_beta.md]   (雲端: .github/workflows/diag_beta.yml)"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, HF_PREFIX

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", default="2026-11-10", help="排除的日期 (逗號分隔), 例如系統異常日")
ap.add_argument("--out", default="stats/market_beta.md")
ap.add_argument("--start", default="2020-01-01")
a = ap.parse_args()
EXCL = {pd.Timestamp(x) for x in a.exclude.split(",") if x}
OUT = []
def P(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); OUT.append(t)

def hf(path):
    return hf_hub_download(HF_REPO, path, repo_type="dataset", force_download=True)

# ---------------- 策略 ----------------
days = pd.read_parquet(hf(f"{HF_PREFIX}/history/days.parquet"))
trades = pd.read_parquet(hf(f"{HF_PREFIX}/history/trades.parquet"))
days["date"] = pd.to_datetime(days.date); trades["date"] = pd.to_datetime(trades.date)
days = days[(days.date >= a.start) & ~days.date.isin(EXCL)]
trades = trades[(trades.date >= a.start) & ~trades.date.isin(EXCL)]
print("history", days.date.min().date(), "~", days.date.max().date(), len(days), "days", len(trades), "trades")

# ---------------- 大盤 ----------------
files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
months = sorted(f for f in files if f.startswith("db/d1/") and f.endswith(".parquet") and f[6:10] >= a.start[:4])
D = []
for f in months:
    x = pd.read_parquet(hf(f)); x["stock_id"] = x.stock_id.astype(str); D.append(x)
d1 = pd.concat(D, ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"])
d1["pc"] = d1.groupby("stock_id").close.shift()
e = d1[d1.stock_id == "0050"].set_index("date").copy()
# 0050 2025-06-18 1 拆 4: 用比例算報酬, 拆股當天的昨收除以 4
e.loc[e.index == "2025-06-18", "pc"] = e.loc[e.index == "2025-06-18", "pc"] / 4
m = pd.DataFrame(index=e.index)
m["e_oc"] = e.close / e.open - 1; m["e_cc"] = e.close / e.pc - 1; m["e_gap"] = e.open / e.pc - 1
# 流動股等權 (4 碼, 昨收 ≥10, 前 20 日均成交額 ≥1 億)
s = d1[d1.stock_id.str.fullmatch(r"[1-9]\d{3}")].copy()
s["dv"] = s.close * s.volume; s["dv20"] = s.groupby("stock_id").dv.transform(lambda v: v.shift().rolling(20, min_periods=10).mean())
s = s[(s.pc >= 10) & (s.dv20 >= 1e8) & (s.open > 0)]
if s.dv20.median() < 1e8 and s.volume.median() < 1e5: print("⚠ volume 單位可能是張, 成交額門檻需 ×1000")
g = s.assign(oc=s.close / s.open - 1, cc=s.close / s.pc - 1).groupby("date")[["oc", "cc"]].mean()
m = m.join(g.rename(columns={"oc": "ew_oc", "cc": "ew_cc"}), how="outer")

# 加權指數 (證交所, 每月一次)
T = []
for ym in pd.period_range(a.start[:7], pd.Timestamp.now().strftime("%Y-%m"), freq="M"):
    u = f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date={ym.year}{ym.month:02d}01&response=json"
    for k in range(3):
        try:
            j = requests.get(u, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).json()
            for r in j.get("data", []):
                y, mo, dd = r[0].split("/"); f2 = lambda v: float(str(v).replace(",", ""))
                T.append(dict(date=pd.Timestamp(int(y) + 1911, int(mo), int(dd)), open=f2(r[1]), high=f2(r[2]), low=f2(r[3]), close=f2(r[4])))
            break
        except Exception as ex:
            time.sleep(5 * (k + 1))
    time.sleep(2.5)
if T:
    t = pd.DataFrame(T).drop_duplicates("date").set_index("date").sort_index()
    t["pc"] = t.close.shift()
    m = m.join(pd.DataFrame({"tx_oc": t.close / t.open - 1, "tx_cc": t.close / t.pc - 1, "tx_gap": t.open / t.pc - 1, "tx_range": (t.high - t.low) / t.pc}), how="left")
    print("TAIEX", t.index.min().date(), "~", t.index.max().date(), len(t))
else:
    print("⚠ 證交所加權指數抓不到, 只用 0050 / 等權")
m = m * 100

# ---------------- 合併 ----------------
x = days.set_index("date")[["trade_ok", "n_trades", "ret_cap", "pnl", "mkt_gap", "breadth", "n_stop"]].join(m, how="left")
x["mkt_gap"] *= 100
x["traded"] = x.n_trades > 0
xt = x[x.traded].copy()
MK = [c for c in ["tx_oc", "tx_cc", "e_oc", "e_cc", "ew_oc", "ew_cc"] if c in x and x[c].notna().any()]
NAME = dict(tx_oc="加權 開→收", tx_cc="加權 昨收→收", e_oc="0050 開→收", e_cc="0050 昨收→收", ew_oc="流動股等權 開→收", ew_cc="流動股等權 昨收→收")
MAIN = "tx_oc" if "tx_oc" in MK else "e_oc"

def stats(y, xm):
    k = y.notna() & xm.notna(); y, xm = y[k], xm[k]
    if len(y) < 5: return dict(n=len(y), corr=np.nan, beta=np.nan, r2=np.nan)
    c = np.corrcoef(y, xm)[0, 1]; b = np.cov(y, xm)[0, 1] / xm.var()
    return dict(n=len(y), corr=c, beta=b, r2=c * c, alpha=y.mean() - b * xm.mean())

def risk(r):
    r = r.fillna(0); eq = r.cumsum()
    return dict(mean=r.mean(), sd=r.std(), sharpe=r.mean() / r.std() * np.sqrt(245) if r.std() > 0 else np.nan,
                mdd=(eq - eq.cummax()).min(), worst=r.min(), win=(r[r != 0] > 0).mean() * 100)

P("# 策略日損益 vs 大盤 連動診斷")
P("")
P(f"資料: 定案規則回測歷史 (HF db/orb/history) {x.index.min().date()} ~ {x.index.max().date()}, 排除 {', '.join(str(d.date()) for d in sorted(EXCL))}。"
  f" 日報酬 = 當日各筆淨報酬合計 / 10 (每檔 10% 資金, 300 萬), 單位 %；只看有交易的日子 ({len(xt)} 天)。")
P("")
P("## 1. 相關與 beta (有交易的日子)")
P("")
P("| 大盤指標 | 天數 | 相關 | beta | R² | alpha (%/日) |")
P("|---|---|---|---|---|---|")
for c in MK:
    q = stats(xt.ret_cap, xt[c]); P(f"| {NAME[c]} | {q['n']} | {q['corr']:.2f} | {q['beta']:.2f} | {q['r2']:.2f} | {q.get('alpha', np.nan):.3f} |")
P("")
P("beta 是「大盤當日 +1% → 策略 (占總資金) 變動幾 %」。策略約 60~90% 資金放空, 若空單跟大盤 1:1 連動, 理論 beta 約 −0.6~−0.9。")
P("")

# 年度與近期
P(f"## 2. 各期間 (對 {NAME[MAIN]})")
P("")
P("| 期間 | 天數 | 策略日均% | 大盤日均% | 相關 | beta | R² |")
P("|---|---|---|---|---|---|---|")
per = [(str(y), xt[xt.index.year == y]) for y in sorted(set(xt.index.year))]
last = xt.index.max()
for n in [60, 20]:
    per.append((f"最近 {n} 個交易日 (有交易)", xt.tail(n)))
per.append(("2026-10-23 起 (雲端上線後)", xt[xt.index >= "2026-10-23"]))
for lab, z in per:
    q = stats(z.ret_cap, z[MAIN]); P(f"| {lab} | {q['n']} | {z.ret_cap.mean():.2f} | {z[MAIN].mean():.2f} | {q['corr']:.2f} | {q['beta']:.2f} | {q['r2']:.2f} |")
P("")
roll = xt.ret_cap.rolling(60).corr(xt[MAIN])
P(f"滾動 60 個交易日相關: 中位 {roll.median():.2f}, 10% 分位 {roll.quantile(.1):.2f}, 90% 分位 {roll.quantile(.9):.2f}, 目前 {roll.iloc[-1]:.2f}。")
P("")

# 分組
P(f"## 3. 依當日大盤 ({NAME[MAIN]}) 分組")
P("")
P("| 大盤 | 天數 | 策略日均% | 賺錢天% | 停損比% |")
P("|---|---|---|---|---|")
bins = [-99, -1, -0.5, 0, 0.5, 1, 99]
xt["mb"] = pd.cut(xt[MAIN], bins)
for b, z in xt.groupby("mb", observed=True):
    P(f"| {b} | {len(z)} | {z.ret_cap.mean():.2f} | {(z.ret_cap > 0).mean()*100:.0f} | {z.n_stop.sum()/max(z.n_trades.sum(),1)*100:.0f} |")
P("")

# ---------------- 對沖 ----------------
P("## 4. 對沖模擬: 空股票 + 同時做多大盤 (台指期 / 0050)")
P("")
P(f"做法: 有交易的日子, 開盤買進 h × (進場檔數 / 10) × 300 萬 的大盤 (以 {NAME[MAIN]} 計, 13:30 平倉), 每天扣 0.02% 名目的期貨成本+滑價。")
P("近似: 對沖部位從開盤持有到收盤, 但空單 09:05 後才進、被停損的會提早出場 → 實際對沖會有誤差。h 先用探索期 (≤2023) 的 beta 決定, 驗證期 (2024~) 才是樣本外。")
P("")
ex = xt[xt.index < "2024-01-01"]; va = xt[xt.index >= "2024-01-01"]
expo = xt.n_trades.clip(upper=10) / 10
b_ex = stats(ex.ret_cap, ex[MAIN] * expo[ex.index])["beta"]
P(f"探索期 策略報酬對「大盤 × 曝險比例」的 beta = {b_ex:.2f} → 最小變異對沖比 h* = {-b_ex:.2f}")
P("")
P("| 對沖比 h | 期間 | 日均% | 日波動% | Sharpe | MDD (%資金) | 最差日% | 與大盤相關 |")
P("|---|---|---|---|---|---|---|---|")
full = x.copy(); full["expo"] = (full.n_trades.clip(upper=10) / 10).where(full.traded, 0)
for h in [0, 0.25, 0.5, round(-b_ex, 2), 1.0]:
    hed = full.ret_cap + h * full.expo * full[MAIN].fillna(0) - (0.02 * h * full.expo)
    for lab, msk in [("全期", full.index >= "1900"), ("驗證 2024~", full.index >= "2024-01-01")]:
        r = risk(hed[msk]); c = stats(hed[msk & full.traded], full.loc[msk & full.traded, MAIN])["corr"]
        P(f"| {h} | {lab} | {r['mean']:.3f} | {r['sd']:.2f} | {r['sharpe']:.2f} | {r['mdd']:.1f} | {r['worst']:.2f} | {c:.2f} |")
P("")

# 逐筆殘差
P("## 5. 逐筆: 個股自己的部分 vs 大盤帶動的部分")
P("")
tt = trades.merge(m[[MAIN]], left_on="date", right_index=True, how="left").dropna(subset=[MAIN])
q = stats(tt.r_pct, tt[MAIN])
P(f"每筆淨報酬 (放空, %) 對 {NAME[MAIN]}: 筆數 {q['n']}, 相關 {q['corr']:.2f}, beta {q['beta']:.2f}, R² {q['r2']:.2f}。")
P(f"→ 單筆報酬只有 {q['r2']*100:.0f}% 的變異來自大盤, 其餘是個股自己的回落/反彈; 但同一天 5~10 檔的個股雜訊會互相抵消, 大盤的部分不會, 所以日損益層級的 R² ({stats(xt.ret_cap, xt[MAIN])['r2']*100:.0f}%) 比單筆高。")
res = tt.r_pct - q["beta"] * tt[MAIN]
P(f"去掉大盤後每筆平均 {res.mean():.2f}% (原 {tt.r_pct.mean():.2f}%), 去掉大盤後勝率 {(res > 0).mean()*100:.0f}% (原 {(tt.r_pct > 0).mean()*100:.0f}%)。")
P("")

# 事前可知的大盤條件
P("## 6. 開盤前可知的大盤條件 → 當日策略 (濾網候選)")
P("")
P("| 條件 | 天數 | 策略日均% | 賺錢天% | 與大盤相關 |")
P("|---|---|---|---|---|")
conds = [("大盤平均跳空 < 0%", xt.mkt_gap < 0), ("大盤平均跳空 0~1%", (xt.mkt_gap >= 0) & (xt.mkt_gap < 1)), ("大盤平均跳空 ≥ 1%", xt.mkt_gap >= 1),
         ("上漲家數比 < 50%", xt.breadth < .5), ("上漲家數比 ≥ 70%", xt.breadth >= .7)]
if "tx_cc" in xt:
    pv = m.tx_cc.shift().reindex(xt.index)
    conds += [("加權昨日 ≤ −1%", pv <= -1), ("加權昨日 −1~+1%", (pv > -1) & (pv < 1)), ("加權昨日 ≥ +1%", pv >= 1)]
for lab, k in conds:
    z = xt[k.fillna(False)]
    if len(z): P(f"| {lab} | {len(z)} | {z.ret_cap.mean():.2f} | {(z.ret_cap > 0).mean()*100:.0f} | {stats(z.ret_cap, z[MAIN])['corr']:.2f} |")
P("")

# ---------------- 近期: 回測 vs 即時 vs 模擬 ----------------
P("## 7. 最近 25 個交易日: 大盤 vs 策略 (回測 / 即時訊號 / 永豐模擬)")
P("")
live = {}
try:
    li = json.loads(Path(hf(f"{HF_PREFIX}/web/live_index.json")).read_text())
    rows = li if isinstance(li, list) else li.get("days", li.get("rows", []))
    for r in rows:
        if isinstance(r, dict) and "date" in r: live[r["date"]] = r
    if rows: print("live_index keys:", list(rows[0].keys()) if isinstance(rows[0], dict) else type(rows[0]))
except Exception as ex:
    print("live_index 讀取失敗", ex)
paper = pd.DataFrame()
try:
    paper = pd.read_csv(hf(f"{HF_PREFIX}/paper/summary.csv")); print("paper summary cols:", list(paper.columns))
    paper["date"] = paper.date.astype(str)
except Exception as ex:
    print("paper summary 讀取失敗", ex)
def lv(r, *ks):
    for k in ks:
        v = r
        for kk in k.split("."):
            v = v.get(kk) if isinstance(v, dict) else None
        if isinstance(v, (int, float)): return v
    return None
cols = [c for c in ["tx_oc", "tx_cc", "e_oc"] if c in x]
P("| 日期 | " + " | ".join(NAME[c] for c in cols) + " | 回測 檔數 | 回測 日報酬% | 回測 損益(萬) | 即時訊號 損益(萬) | 模擬單 損益(萬) |")
P("|" + "---|" * (len(cols) + 6))
recent = x[x.index >= x.index[-25]] if len(x) >= 25 else x
alld = sorted(set(recent.index.strftime("%Y-%m-%d")) | {d for d in live if d >= recent.index.min().strftime("%Y-%m-%d")} |
              set(paper.date[paper.date >= recent.index.min().strftime("%Y-%m-%d")]) if len(paper) else set(recent.index.strftime("%Y-%m-%d")))
for d in alld:
    ts = pd.Timestamp(d)
    if ts in EXCL: continue
    r = x.loc[ts] if ts in x.index else None
    mk = [f"{m.loc[ts, c]:+.2f}" if ts in m.index and pd.notna(m.loc[ts, c]) else "" for c in cols]
    L = live.get(d, {}); pl = lv(L, "live_pnl", "live.pnl", "pnl_live")
    pp = paper[paper.date == d]; ppn = pp.sim_pnl.iloc[-1] if len(pp) and "sim_pnl" in pp and pd.notna(pp.sim_pnl.iloc[-1]) else None
    rt = f"{int(r.n_trades)} | {r.ret_cap:+.2f} | {r.pnl/1e4:+.1f}" if r is not None else " | | "
    P(f"| {d} | " + " | ".join(mk) + f" | {rt} | {'' if pl is None else f'{pl/1e4:+.1f}'} | {'' if ppn is None else f'{ppn/1e4:+.1f}'} |")
P("")
if live: P("即時訊號 raw (最近 5 天): " + json.dumps({d: live[d] for d in sorted(live)[-5:]}, ensure_ascii=False, default=str)[:3000])
if len(paper): P("模擬單 summary 最近:\n" + paper.tail(12).to_string(index=False)[:4000])

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(OUT) + "\n")
import os
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f: f.write("\n".join(OUT) + "\n")
print("寫入", a.out)
