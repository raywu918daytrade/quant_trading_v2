#!/usr/bin/env python3
"""收盤價對帳: 訊號 (db/orb/live/<date>/<date>_trades.csv) 與模擬單 (db/orb/paper/<date>_trades.csv) 的收盤回補價
一律以 D1 (HF db/d1, 收盤後由 DAY_TRADE 寫入的官方收盤價) 為準; 對不上就改正並重算報酬 / 損益, 回傳改了什麼。

為什麼要有這道檢查 (2026-11-09):
  - 11-05 訊號端 13:31 抓 MIS 時收盤競價結果還沒出來, 用了 13:25 前最後成交 (驊宏資 42.15 / 萬海 112.5, 實際 43.4 / 113.0), 多算約 8,500 元
  - 11-04 永豐模擬環境的收盤回報價錯誤 (順達 283.5 vs 271.5、光寶科 235 vs 225、合晶 77.4 vs 83.5)
  兩個 bug 都已修 (MIS.closes() 等 13:30 後的成交價、模擬單收盤腳用真實收盤價), 但存下來的紀錄沒有重算, 也沒有任何東西會自動發現。
  cloud_nightly.py 每晚呼叫 reconcile(): 改正 + 印出 + (有 LINE 憑證時) 推播, 再重算網頁的實盤紀錄。

只動「收盤回補」那一腳 (停損 / 進場是盤中成交價, 沒有官方對照); 鎖漲停罰金 (locked) 維持原判斷。
用法:
  python3 scripts/orb_live.py refresh --months 3 --m1-months 0     # 先有 D1
  python3 scripts/fix_close.py                 # 只檢查, 印出差異
  HF_TOKEN=... python3 scripts/fix_close.py --upload [--line]
"""
import argparse, csv, io, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_live as OL
from config import COST, HF_REPO, HF_PREFIX

TOL = 1e-6


def d1_closes(since_months=6):
    since = (pd.Timestamp.now() - pd.DateOffset(months=since_months)).strftime("%Y_%m")
    d = OL.load_d1(since=since)
    return {(str(r.date.date()), r.stock_id): round(float(r.close), 2) for r in d[["date", "stock_id", "close"]].itertuples() if np.isfinite(r.close)}


class Repo:
    def __init__(self, tok):
        from huggingface_hub import HfApi
        self.tok = tok; self.api = HfApi(token=tok)
        self.files = set(self.api.list_repo_files(HF_REPO, repo_type="dataset"))
        self.ops = []

    def read(self, path):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(HF_REPO, path, repo_type="dataset", token=self.tok, force_download=True)).read_bytes()

    def put(self, path, data: bytes):
        from huggingface_hub import CommitOperationAdd
        self.ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data))

    def commit(self, msg):
        if self.ops: self.api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=self.ops, commit_message=msg)
        n = len(self.ops); self.ops = []; return n


def _csv_bytes(df): return df.to_csv(index=False, float_format="%.10g").encode("utf-8")


# ============================== 訊號 (orb_live run) ==============================
def fix_live(repo, C, log):
    pre = f"{HF_PREFIX}/live/"; fixed = set()
    for f in sorted(x for x in repo.files if x.startswith(pre) and x.endswith("_trades.csv")):
        ds = f.split("/")[-1][:10]
        tr = pd.read_csv(io.BytesIO(repo.read(f)), dtype={"stock_id": str})
        if not len(tr) or "reason" not in tr: continue
        new_px = {}
        for i, r in tr.iterrows():
            c = C.get((ds, r.stock_id))
            if r.reason != "收盤回補" or c is None or not np.isfinite(r.exit_px) or abs(r.exit_px - c) < TOL: continue
            rr = -(c / r.entry_px - 1) - COST - (OL.P["LOCK"] if bool(r.get("locked", False)) else 0)
            log.append(f"訊號 {ds} {r.stock_id} {r['name']}: 收盤 {r.exit_px:g} → {c:g}, 損益 {r.pnl:+,.0f} → {rr * r.entry_px * r.lots * 1000:+,.0f}")
            tr.loc[i, ["exit_px", "r_pct", "pnl"]] = [c, rr * 100, rr * r.entry_px * r.lots * 1000]; new_px[r.stock_id] = c
        if not new_px: continue
        repo.put(f, _csv_bytes(tr)); fixed.add(ds)
        ev = f.replace("_trades.csv", "_events.csv")                       # 沒有表頭, 逐行 append 的事件檔: CLOSE 列的 price (第 5 欄)
        if ev in repo.files:
            rows = list(csv.reader(io.StringIO(repo.read(ev).decode("utf-8"))))
            for r in rows:
                if len(r) >= 6 and r[3] == "CLOSE" and r[1] in new_px: r[4] = f"{new_px[r[1]]:g}"
            buf = io.StringIO(); csv.writer(buf, lineterminator="\n").writerows(rows); repo.put(ev, buf.getvalue().encode("utf-8"))
    return fixed


# ============================== 模擬單 (paper_orb run) ==============================
def fix_paper(repo, C, log):
    import paper_orb as PO
    pre = f"{HF_PREFIX}/paper/"; fixed = {}
    for f in sorted(x for x in repo.files if x.startswith(pre) and x.endswith("_trades.csv")):
        ds = f.split("/")[-1][:10]
        tr = pd.read_csv(io.BytesIO(repo.read(f)), dtype={"stock_id": str})
        if not len(tr) or "exit_leg" not in tr: continue
        if "broker_buy_px" not in tr: tr["broker_buy_px"] = np.nan
        changed = set()
        for i, r in tr.iterrows():
            c = C.get((ds, r.stock_id))
            if r.exit_leg != "close" or c is None or not np.isfinite(r.buy_px): continue
            if abs(r.buy_px - c) < TOL and abs(r.close_px - c) < TOL: continue
            q = min(r.sell_lots, r.buy_lots)
            sim = 1 - c / r.sell_px - COST; ideal = 1 - c / r.rl - COST
            new_pnl = (r.sell_px - c) * 1000 * q - COST * r.sell_px * 1000 * q
            log.append(f"模擬 {ds} {r.stock_id} {r['name']}: 收盤 {r.buy_px:g} → {c:g}, 損益 {r.sim_pnl:+,.0f} → {new_pnl:+,.0f}")
            tr.loc[i, ["broker_buy_px", "buy_px", "close_px", "exit_slip_ticks", "sim_net_pct", "sim_pnl", "ideal_net_pct", "ideal_pnl"]] = [
                r.broker_buy_px if np.isfinite(r.broker_buy_px) else r.buy_px, c, c, 0.0, sim * 100, new_pnl, ideal * 100, ideal * r.rl * 1000 * r.lots]
            changed.add(r.stock_id)
        if not changed: continue
        repo.put(f, _csv_bytes(tr))
        legs = None; sf = f.replace("_trades.csv", "_slippage.csv")
        if sf in repo.files:
            legs = pd.read_csv(io.BytesIO(repo.read(sf)), dtype={"stock_id": str})
            m = (legs.leg == "close") & legs.stock_id.isin(changed)
            if m.any():
                c = legs.loc[m].stock_id.map(lambda s: C[(ds, s)])
                if "broker_fill_px" not in legs: legs["broker_fill_px"] = np.nan
                legs.loc[m, "broker_fill_px"] = legs.loc[m, "broker_fill_px"].fillna(legs.loc[m, "fill_px"])
                legs.loc[m, "signal_px"] = c; legs.loc[m, "fill_px"] = c
                legs.loc[m, ["slip_yuan", "slip_ticks", "slip_pct"]] = 0.0
                repo.put(sf, _csv_bytes(legs))
        fixed[ds] = (tr, legs)
    if fixed: _paper_summaries(repo, fixed)
    return set(fixed)


def _paper_summaries(repo, fixed):
    """改過的日子: 重算 summary 的損益欄位 (其餘欄位照舊), 更新 <date>_summary.json / summary.csv / slippage.csv / latest.json."""
    pre = f"{HF_PREFIX}/paper/"
    def mean(s): s = pd.to_numeric(s, errors="coerce").dropna(); return round(float(s.mean()), 3) if len(s) else None
    upd = {}
    for ds, (t, g) in fixed.items():
        done = t[t.sell_px.notna() & t.buy_px.notna()]
        u = dict(sim_avg_net_pct=mean(done.sim_net_pct), ideal_avg_net_pct=mean(t.ideal_net_pct),
                 sim_pnl=round(float(done.sim_pnl.sum())) if len(done) else None, ideal_pnl=round(float(pd.to_numeric(t.ideal_pnl, errors="coerce").sum())))
        if g is not None and len(g): u["close_slip_ticks"] = mean(g[(g.filled.astype(str) == "True") & (g.leg == "close")].slip_ticks)
        upd[ds] = u
        jf = f"{pre}{ds}_summary.json"
        if jf in repo.files:
            j = {**json.loads(repo.read(jf)), **u}; b = json.dumps(j, ensure_ascii=False, indent=1).encode("utf-8")
            repo.put(jf, b)
            lf = f"{pre}latest.json"
            if lf in repo.files and json.loads(repo.read(lf)).get("date") == ds: repo.put(lf, b)
    sp = f"{pre}summary.csv"
    if sp in repo.files:
        s = pd.read_csv(io.BytesIO(repo.read(sp)), dtype=str)
        for ds, u in upd.items():
            for k, v in u.items():
                if k not in s: s[k] = ""
                s.loc[s.date == ds, k] = "" if v is None else str(v)
        repo.put(sp, s.to_csv(index=False).encode("utf-8"))
    cp = f"{pre}slippage.csv"
    if cp in repo.files:
        s = pd.read_csv(io.BytesIO(repo.read(cp)), dtype={"stock_id": str})
        new = [g for ds, (_, g) in fixed.items() if g is not None]
        if new:
            s = pd.concat([s[~s.date.astype(str).isin([ds for ds, (_, g) in fixed.items() if g is not None])], *new], ignore_index=True).sort_values(["date"], kind="stable")
            repo.put(cp, _csv_bytes(s))


def reconcile(tok, upload=True, line=False):
    """回傳 (改過的訊號日期, 改過的模擬單日期, 說明列)."""
    C = d1_closes(); repo = Repo(tok); log = []
    live, paper = fix_live(repo, C, log), fix_paper(repo, C, log)
    print("收盤價對帳:", "全部一致" if not log else f"{len(log)} 筆不一致"); [print("  " + x) for x in log]
    if log and upload:
        n = repo.commit(f"orb fix close prices vs D1 ({len(log)} legs)"); print(f"已改正並上傳 {n} 個檔")
    if log and line:
        try: OL.Line(True).push(OL.LC.card("⚠ 收盤價已自動改正", [OL.LC.note("\n".join(log[:15]) + (f"\n…共 {len(log)} 筆" if len(log) > 15 else "")),
                                       OL.LC.small("以日K收盤價為準, 網頁與紀錄已一併更新")], hint=f"{len(log)} 筆", kind="warn",
                                       alt=f"⚠ 收盤價對帳已自動改正 {len(log)} 筆: " + "；".join(log[:5])))
        except Exception as e: print("LINE 推播失敗:", e)
    return live, paper, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true"); ap.add_argument("--line", action="store_true")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    if a.upload and not tok: sys.exit("缺環境變數 HF_TOKEN")
    reconcile(tok, upload=a.upload, line=a.line)
