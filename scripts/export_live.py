#!/usr/bin/env python3
"""實盤紀錄: 把每天的實盤檔 (orb_live.py run 寫的 <date>_trades/_events/_watch/_excluded/_cand.csv) 與同日回測 (orb_history) 逐檔對齊.
輸出 (網頁「實盤紀錄」分頁讀):
  live_index.json       每日: 實盤 / 回測 筆數、平均報酬、損益 (元)、兩者差、平均進場差幾檔
  live/<date>.json      逐檔: 實盤放空價/時間 vs 回測進場價/時間、差幾檔 (orb_live.tick_size)、出場方式與報酬; 只有一邊有的股票標示原因
差幾檔的正負: 正 = 實盤比回測不利 (進場: 實盤放空價較低; 出場: 實盤回補價較高)。
實盤來源:
  HF   db/orb/live/<date>/<date>_*.csv  (雲端 live.yml 收盤後上傳)
  本機 signals/<date>_*.csv             (本機 orb_live.py run 的輸出, 或任何同樣命名的資料夾)
用法:
  python3 scripts/export_live.py                      # 讀 HF, 寫 data/web_export/live_index.json + live/<date>.json
  python3 scripts/export_live.py --src signals        # 讀本機資料夾
  HF_TOKEN=... python3 scripts/export_live.py --upload   # 匯出後上傳到 HF db/orb/web/ (平常由 cloud_nightly.py 做)
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_live as L
from config import ROOT, PROC, HF_REPO, HF_PREFIX

OUT = ROOT / "data" / "web_export"
KINDS = ["trades", "events", "watch", "excluded", "cand"]
EV_COLS = ["time", "stock_id", "name", "event", "price", "ref", "note"]
BT_STATE = {"WAIT": "回測未跌破區間低", "EXPIRED": "回測未跌破區間低", "FULL": "回測跌破但已滿 10 檔"}


# ============================== 實盤來源 ==============================
class LocalSrc:
    """資料夾內 <date>_<kind>.csv (平放, 與 orb_live.py run 的 signals/ 相同)."""
    def __init__(self, d):
        self.d = Path(d)
    def dates(self):
        return sorted({p.name[:10] for k in KINDS for p in self.d.glob(f"????-??-??_{k}.csv")})
    def path(self, date, kind):
        p = self.d / f"{date}_{kind}.csv"
        return p if p.exists() else None


class HFSrc:
    """HF db/orb/live/<date>/<date>_<kind>.csv; 列表一次, 需要時才下載 (public dataset, 讀不需 token)."""
    def __init__(self, token=None):
        from huggingface_hub import HfApi
        self.token = token
        try:
            fs = HfApi(token=token).list_repo_files(HF_REPO, repo_type="dataset")
        except Exception as e:
            print(f"⚠ HF 列表失敗 ({e}), 當作沒有實盤資料"); fs = []
        pre = f"{HF_PREFIX}/live/"
        self.files = {f[len(pre):].split("/")[-1]: f for f in fs if f.startswith(pre) and f.endswith(".csv")}
    def dates(self):
        return sorted({n[:10] for n in self.files if n[10:11] == "_" and n[11:-4] in KINDS})
    def path(self, date, kind):
        f = self.files.get(f"{date}_{kind}.csv")
        if not f: return None
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(HF_REPO, f, repo_type="dataset", token=self.token))


def _csv(p):
    if p is None: return None
    try: d = pd.read_csv(p, dtype={"stock_id": str})
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=["stock_id"])
    return d if "stock_id" in d.columns else pd.DataFrame(columns=["stock_id"])


def _events(p):
    """events.csv 沒有表頭 (orb_live.py 逐行 append); note 若含逗號就併回 note."""
    if p is None: return pd.DataFrame(columns=EV_COLS)
    rows = [r[:6] + [",".join(r[6:])] for r in csv.reader(open(p, encoding="utf-8")) if len(r) >= 6]
    d = pd.DataFrame(rows, columns=EV_COLS) if rows else pd.DataFrame(columns=EV_COLS)
    d["price"] = pd.to_numeric(d.price, errors="coerce"); d["ref"] = pd.to_numeric(d.ref, errors="coerce")
    return d


def read_live(src, date):
    x = {k: _csv(src.path(date, k)) for k in ["trades", "watch", "excluded", "cand"]}
    x["events"] = _events(src.path(date, "events"))
    if x["trades"] is not None: x["status"] = "已結算"
    elif x["watch"] is not None: x["status"] = "未結算"            # 盤中中斷, 沒有 trades.csv → 用事件重建
    elif x["cand"] is not None: x["status"] = "不交易"
    else: x["status"] = "無資料"
    if x["trades"] is None and len(x["events"]):
        x["trades"] = trades_from_events(x["events"], x["watch"])
    return x


def trades_from_events(ev, watch):
    """沒有 trades.csv 時, 用 SHORT / STOP / CLOSE 事件拼出進出場 (沒有出場的報酬為空)."""
    lots = dict(zip(watch.stock_id, watch.lots)) if watch is not None and "lots" in watch.columns else {}
    rows = []
    for sid, g in ev.groupby("stock_id", sort=False):
        s = g[g.event == "SHORT"]
        if s.empty: continue
        e = g[g.event.isin(["STOP", "CLOSE"])]
        r = dict(stock_id=sid, name=s.name.iloc[0], entry_t=_hms_to_t(s.time.iloc[0]), entry_px=s.price.iloc[0], lots=lots.get(sid, np.nan),
                 exit_t=np.nan, exit_px=np.nan, reason="", r_pct=np.nan, pnl=np.nan, locked=False)
        if len(e):
            r.update(exit_t=_hms_to_t(e.time.iloc[0]), exit_px=e.price.iloc[0], reason="停損 (突破區間高)" if e.event.iloc[0] == "STOP" else "收盤回補")
            ret = -(r["exit_px"] / r["entry_px"] - 1) - L.COST
            r.update(r_pct=ret * 100, pnl=ret * r["entry_px"] * r["lots"] * 1000)
        rows.append(r)
    return pd.DataFrame(rows, columns=["stock_id", "name", "entry_t", "entry_px", "lots", "exit_t", "exit_px", "reason", "r_pct", "pnl", "locked"])


# ============================== 工具 ==============================
def num(v, d=4):
    try: v = float(v)
    except (TypeError, ValueError): return None
    return round(v, d) if np.isfinite(v) else None


def txt(v):
    return "" if v is None or v != v else str(v)


def _hms_to_t(s):
    try: h, m, *sec = [int(x) for x in str(s).split(":")]
    except ValueError: return np.nan
    return h * 60 + m + (sec[0] / 60 if sec else 0) - 540


def t_str(t, sec=True):
    """09:00 起算的分鐘 → HH:MM(:SS)."""
    if t is None or not np.isfinite(t): return None
    s = int(round(t * 60)) + 9 * 3600
    return f"{s // 3600:02d}:{s // 60 % 60:02d}" + (f":{s % 60:02d}" if sec else "")


def ticks(a, b):
    """a → b 差幾檔 (b > a 為正), 跨價位級距時逐檔累加; 不在檔位上的價格 (回測停損含滑價) 以最後一檔的比例計."""
    a, b = num(a, 3), num(b, 3)
    if a is None or b is None: return None
    sgn = 1
    if b < a: a, b, sgn = b, a, -1
    n, p = 0.0, a
    while p < b - 1e-6 and n < 5000:
        t = L.tick_size(p + 1e-9); p = round(p + t, 3); n += 1
    if p > b + 1e-6: n -= (p - b) / t
    return round(sgn * n, 1)


def side(r):
    """一筆交易 → JSON (實盤時間帶秒, 回測是分 K 只到分)."""
    if r is None: return None
    live = r.get("_live", False)
    return dict(entry_t=num(r.get("entry_t"), 2), entry_time=t_str(num(r.get("entry_t"), 4), sec=live), entry_px=num(r.get("entry_px"), 2),
                exit_t=num(r.get("exit_t"), 2), exit_time=t_str(num(r.get("exit_t"), 4), sec=live), exit_px=num(r.get("exit_px"), 2),
                reason=txt(r.get("reason")), locked=bool(r.get("locked")) if r.get("locked") == r.get("locked") else False,
                lots=num(r.get("lots"), 0), r_pct=num(r.get("r_pct")), pnl=num(r.get("pnl"), 0))


# ============================== 對齊 ==============================
def compare_day(date, lv, TR, WT, DAYS):
    d = pd.Timestamp(date)
    drow = DAYS[DAYS.date == d]
    bt_ready = len(drow) > 0
    bt_ok = bool(drow.trade_ok.iloc[0]) if bt_ready else None
    bt_reason = txt(drow.reasons.iloc[0]) if bt_ready else ""
    bt = TR[TR.date == d].copy(); bw = WT[WT.date == d]
    lt = lv["trades"] if lv["trades"] is not None else pd.DataFrame(columns=["stock_id"])
    lw = lv["watch"] if lv["watch"] is not None else pd.DataFrame(columns=["stock_id"])
    lx = lv["excluded"] if lv["excluded"] is not None else pd.DataFrame(columns=["stock_id"])
    ev = lv["events"]
    full = set(ev[ev.event == "FULL"].stock_id)
    lrec = {r["stock_id"]: {**r, "_live": True} for r in lt.to_dict("records")}
    brec = {r["stock_id"]: r for r in bt.to_dict("records")}
    winfo = {r["stock_id"]: r for r in pd.concat([bw, lw, lx], ignore_index=True).drop_duplicates("stock_id").to_dict("records")}

    def why_no_bt(sid):
        if not bt_ready: return "回測尚未更新 (分 K 還沒到)"
        if not bt_ok: return f"回測判定不交易 ({bt_reason})"
        w = bw[bw.stock_id == sid]
        if w.empty: return "不在回測監看名單 (開盤價 / 跳空 / 池條件不同)"
        s, x = w.state.iloc[0], w.excluded.iloc[0]
        return f"回測剔除 ({x})" if s == "EXCLUDED" else BT_STATE.get(s, f"回測狀態 {s}")

    def why_no_live(sid):
        if lv["status"] == "不交易": return "實盤判定不交易"
        if lv["status"] == "無資料": return "沒有實盤檔"
        if sid in full: return "實盤跌破但已滿 10 檔"
        x = lx[lx.stock_id == sid]
        if len(x): return f"實盤剔除 ({txt(x.excluded.iloc[0])})" if "excluded" in x.columns else "實盤剔除"
        if sid in set(lw.stock_id): return "實盤監看中但 12:00 前未偵測到跌破"
        return "不在實盤監看名單 (開盤價 / 跳空 / 當沖資格不同)"

    rows = []
    for sid in list(dict.fromkeys(list(lrec) + list(brec))):
        a, b = lrec.get(sid), brec.get(sid)
        w = winfo.get(sid, {})
        ref = b or a
        row = dict(stock_id=sid, name=(a or b or w).get("name", ""), gap=num(ref.get("gap", w.get("gap"))), rl=num(ref.get("rl", w.get("rl")), 2), rh=num(ref.get("rh", w.get("rh")), 2),
                   side="both" if a and b else "live" if a else "bt", live=side(a), bt=side(b), entry_ticks=None, exit_ticks=None, r_diff=None, pnl_diff=None, note="")
        if a and b:
            row["entry_ticks"] = ticks(a["entry_px"], b["entry_px"])          # 回測進場 - 實盤放空 (正 = 實盤賣得較低)
            row["exit_ticks"] = ticks(b["exit_px"], a["exit_px"])             # 實盤回補 - 回測出場 (正 = 實盤買得較高)
            if row["live"]["r_pct"] is not None and row["bt"]["r_pct"] is not None:
                row["r_diff"] = round(row["live"]["r_pct"] - row["bt"]["r_pct"], 4); row["pnl_diff"] = round((row["live"]["pnl"] or 0) - (row["bt"]["pnl"] or 0))
            if txt(a.get("reason"))[:2] != txt(b.get("reason"))[:2]: row["note"] = "出場方式不同"
        elif a: row["note"] = why_no_bt(sid)
        else: row["note"] = why_no_live(sid)
        rows.append(row)
    order = {"both": 0, "live": 1, "bt": 2}
    rows.sort(key=lambda r: (order[r["side"]], (r["live"] or r["bt"])["entry_t"] or 999, r["stock_id"]))

    def agg(xs, key):
        v = [x[key]["r_pct"] for x in xs if x[key] and x[key]["r_pct"] is not None]
        p = [x[key]["pnl"] for x in xs if x[key] and x[key]["pnl"] is not None]
        return (round(float(np.mean(v)), 4) if v else None), (round(float(np.sum(p))) if p else 0)
    live_rows = [r for r in rows if r["live"]]; bt_rows = [r for r in rows if r["bt"]]; both = [r for r in rows if r["side"] == "both"]
    la, lp = agg(live_rows, "live"); ba, bp = agg(bt_rows, "bt")
    tk = [r["entry_ticks"] for r in both if r["entry_ticks"] is not None]
    note = []
    if lv["status"] != "已結算": note.append(f"實盤{lv['status']}")
    if not bt_ready: note.append("回測尚未更新")
    elif not bt_ok: note.append(f"回測不交易 ({bt_reason})")
    note = "；".join(note)
    summ = dict(date=date, live_status=lv["status"], bt_ready=bt_ready, bt_trade_ok=bt_ok, n_live=len(live_rows), n_bt=len(bt_rows), n_both=len(both),
                n_live_only=sum(r["side"] == "live" for r in rows), n_bt_only=sum(r["side"] == "bt" for r in rows),
                live_avg=la, bt_avg=ba, live_pnl=lp, bt_pnl=bp if bt_ready else None, diff_pnl=lp - bp if bt_ready else None, diff_avg=round(la - ba, 4) if la is not None and ba is not None else None,
                both_live_avg=agg(both, "live")[0], both_bt_avg=agg(both, "bt")[0], entry_ticks_avg=round(float(np.mean(tk)), 2) if tk else None, note=note)
    events = [{k: (num(v, 2) if k in ("price", "ref") else ("" if v != v else v)) for k, v in e.items()} for e in ev.to_dict("records")]
    excluded = [dict(stock_id=r["stock_id"], name=txt(r.get("name")), reason=txt(r.get("excluded"))) for r in lx.to_dict("records")]
    return summ, dict(date=date, summary=summ, rows=rows, events=events, excluded=excluded)


def build_index(days):
    """累計曲線與總計. 回測還沒更新的日子: 回測與差額為空, 回測累計不前進 (總差額只算兩邊都有的日子)."""
    days = sorted(days, key=lambda x: x["date"])
    cl = cb = 0
    for x in days:
        cl += x["live_pnl"]; x["cum_live"] = cl
        if x["bt_ready"]: cb += x["bt_pnl"]
        x["cum_bt"] = cb if x["bt_ready"] else None
    ready = [x for x in days if x["bt_ready"]]
    tk = [x["entry_ticks_avg"] for x in days if x["entry_ticks_avg"] is not None]

    def wavg(k, n):
        xs = [x for x in days if x[k] is not None and x[n]]
        return round(sum(x[k] * x[n] for x in xs) / sum(x[n] for x in xs), 4) if xs else None
    stats = dict(n_days=len(days), first=days[0]["date"] if days else None, last=days[-1]["date"] if days else None,
                 n_live_days=sum(1 for x in days if x["n_live"]), n_ready=len(ready), n_live=sum(x["n_live"] for x in days), n_bt=sum(x["n_bt"] for x in ready),
                 n_both=sum(x["n_both"] for x in days), live_pnl=cl, live_pnl_ready=sum(x["live_pnl"] for x in ready), bt_pnl=cb,
                 diff_pnl=sum(x["diff_pnl"] for x in ready), live_avg=wavg("live_avg", "n_live"), bt_avg=wavg("bt_avg", "n_bt"),
                 entry_ticks_avg=round(float(np.mean(tk)), 2) if tk else None)
    return dict(stats=stats, days=days)


def load_history():
    H = PROC / "orb_history"
    return pd.read_parquet(H / "trades.parquet"), pd.read_parquet(H / "watch.parquet"), pd.read_parquet(H / "days.parquet")


def build(src, TR, WT, DAYS, prev=None, rebuild=False):
    """回傳 (index, {date: detail}). prev = 上次的 live_index.json: 回測已就緒的日子沿用, 不重算 (只重算新日子與回測未就緒的)."""
    old = {x["date"]: x for x in (prev or {}).get("days", [])} if not rebuild else {}
    days, details = [], {}
    for ds in src.dates():
        if ds in old and old[ds].get("bt_ready"):
            days.append({k: v for k, v in old[ds].items() if k not in ("cum_live", "cum_bt")}); continue
        summ, det = compare_day(ds, read_live(src, ds), TR, WT, DAYS)
        days.append(summ); details[ds] = det
    return build_index(days), details


def write(index, details, out=OUT):
    (out / "live").mkdir(parents=True, exist_ok=True)
    js = lambda o: json.dumps(L._jclean(o), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    paths = [out / "live_index.json"]; paths[0].write_text(js(index), encoding="utf-8")
    for ds, det in details.items():
        p = out / "live" / f"{ds}.json"; p.write_text(js(det), encoding="utf-8"); paths.append(p)
    return paths


def commit_ops(paths, out=OUT):
    from huggingface_hub import CommitOperationAdd
    return [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/web/{p.relative_to(out).as_posix()}", path_or_fileobj=str(p)) for p in paths]


def fetch_prev_index(token=None):
    from huggingface_hub import hf_hub_download
    try:
        return json.loads(Path(hf_hub_download(HF_REPO, f"{HF_PREFIX}/web/live_index.json", repo_type="dataset", force_download=True, token=token)).read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="本機實盤資料夾 (例 signals); 不給 = 讀 HF")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    import os
    tok = os.environ.get("HF_TOKEN")
    src = LocalSrc(a.src) if a.src else HFSrc(tok)
    idx, det = build(src, *load_history())
    files = write(idx, det, Path(a.out))
    s = idx["stats"]
    print(f"實盤 {s['n_days']} 天 ({s['first']} ~ {s['last']}), 實盤 {s['n_live']} 筆 / 回測 {s['n_bt']} 筆; 回測已更新的 {s['n_ready']} 天: 實盤 {s['live_pnl_ready']:+,} 元 vs 回測 {s['bt_pnl']:+,} 元 → {a.out}")
    if a.upload:
        from huggingface_hub import HfApi
        if not tok: sys.exit("缺環境變數 HF_TOKEN (需 HF 寫入權限)")
        HfApi(token=tok).create_commit(repo_id=HF_REPO, repo_type="dataset", operations=commit_ops(files, Path(a.out)), commit_message="orb live record")
        print(f"已上傳 {len(files)} 個檔案")
