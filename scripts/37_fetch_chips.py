#!/usr/bin/env python3
"""研究用: 抓籌碼歷史資料 (只寫 data/chips/, 不碰線上程式)。在 GHA 跑 (雲端 session 連不到證交所 / 櫃買 / 期交所)。
  1. 個股融資融券餘額 (每日): 證交所 MI_MARGN (上市) + 櫃買 margin balance (上櫃) → data/chips/margin_YYYY.parquet
     欄位: date, stock_id, mb (融資餘額, 張), ml (融資限額), sb (融券餘額, 張), ss (融券賣出), sc (融券買進=回補), sl (融券限額), off (資券互抵), note (註記)
  2. 小台指 (MTX) 三大法人未平倉 + 全市場未平倉 (每日) → data/chips/mtx.parquet
     散戶多空比 = −(三大法人多空未平倉淨額) / 全市場未平倉 (散戶 = 全市場 − 法人)
已抓過的日期會跳過 (data/chips 由 actions/cache 保留), 中斷可續跑。
用法: python3 scripts/37_fetch_chips.py [--start 2019-12-01] [--test]"""
import argparse, io, sys, time, threading, traceback
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "chips"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2019-12-01")
ap.add_argument("--end", default=str(date.today()))
ap.add_argument("--test", action="store_true", help="只抓幾天, 印欄位")
a = ap.parse_args()
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
LOCK = threading.Lock()

def num(x):
    if x is None: return np.nan
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except ValueError: return np.nan

def get(url, method="GET", tries=5, **kw):
    for k in range(tries):
        try:
            r = requests.request(method, url, headers=UA, timeout=30, **kw)
            if r.status_code == 200: return r
            print(f"  {r.status_code} {url}", flush=True)
        except Exception as e:
            print(f"  err {type(e).__name__} {url}", flush=True)
        time.sleep(10 * (k + 1))
    return None

def pick(fields, rows, want):
    """依欄位名找 index; 融資/融券同名欄位 (買進/賣出/今日餘額/限額) 用出現順序區分。want: {name: (keyword, occurrence)}"""
    out = {}
    for k, (kw, occ) in want.items():
        hits = [i for i, f in enumerate(fields) if kw in str(f)]
        out[k] = hits[occ] if len(hits) > occ else None
    return out

def df_from(rows, idx, d):
    recs = []
    for r in rows:
        sid = str(r[idx["sid"]]).strip()
        if not sid or not sid[0].isdigit(): continue
        rec = {"date": d, "stock_id": sid}
        for k in ["mb", "ml", "sb", "ss", "sc", "sl", "off"]:
            rec[k] = num(r[idx[k]]) if idx.get(k) is not None else np.nan
        rec["note"] = str(r[idx["note"]]).strip() if idx.get("note") is not None and idx["note"] < len(r) else ""
        recs.append(rec)
    return pd.DataFrame(recs)

def twse(d):
    r = get(f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=ALL&response=json")
    if r is None: return None
    j = r.json()
    if j.get("stat") not in ("OK", None) and "tables" not in j: return pd.DataFrame()     # 非交易日
    for t in j.get("tables", []):
        f = t.get("fields", []); rows = t.get("data", [])
        if not rows or not any("代號" in str(x) for x in f) or len(f) < 14: continue
        if a.test: print("TWSE fields", f, rows[0], flush=True)
        # 代號,名稱,[融資]買進,賣出,現金償還,前日餘額,今日餘額,次一營業日限額,[融券]買進,賣出,現券償還,前日餘額,今日餘額,次一營業日限額,資券互抵,註記
        idx = pick(f, rows, {"sid": ("代號", 0), "mb": ("今日餘額", 0), "ml": ("限額", 0), "sc": ("買進", 1), "ss": ("賣出", 1),
                             "sb": ("今日餘額", 1), "sl": ("限額", 1), "off": ("互抵", 0), "note": ("註記", 0)})
        return df_from(rows, idx, d)
    return pd.DataFrame()

def tpex(d):
    roc = f"{d.year - 1911}/{d:%m/%d}"
    r = get("https://www.tpex.org.tw/www/zh-tw/margin/balance", "POST", data={"date": f"{d:%Y/%m/%d}", "id": "", "response": "json"})
    rows, f = [], []
    if r is not None:
        try:
            j = r.json()
            for t in j.get("tables", []):
                if t.get("data"): rows, f = t["data"], t.get("fields", []); break
        except Exception: pass
    if not rows:   # 舊網址
        r = get(f"https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=json&d={roc}")
        if r is None: return None
        try: j = r.json()
        except Exception: return None
        rows = j.get("aaData") or []
        f = []
    if not rows: return pd.DataFrame()
    if a.test: print("TPEx fields", f, rows[0], flush=True)
    if f and any("代號" in str(x) for x in f):
        # 代號,名稱,前資餘額,資買,資賣,現償,資餘額,資屬證金,資使用率,資限額,前券餘額,券賣,券買,券償,券餘額,券屬證金,券使用率,券限額,資券相抵,備註
        idx = pick(f, rows, {"sid": ("代號", 0), "mb": ("資餘額", 1), "ml": ("資限額", 0), "ss": ("券賣", 0), "sc": ("券買", 0),
                             "sb": ("券餘額", 1), "sl": ("券限額", 0), "off": ("相抵", 0), "note": ("備註", 0)})
        if idx["mb"] is None: idx["mb"] = pick(f, rows, {"x": ("資餘額", 0)})["x"]
        if idx["sb"] is None: idx["sb"] = pick(f, rows, {"x": ("券餘額", 0)})["x"]
    else:
        idx = {"sid": 0, "mb": 6, "ml": 9, "ss": 11, "sc": 12, "sb": 14, "sl": 17, "off": 18, "note": 19}
    return df_from(rows, idx, d)

def run_margin(name, fn, days, sleep):
    have = set()
    for p in OUT.glob(f"margin_{name}_*.parquet"):
        have |= set(pd.read_parquet(p, columns=["date"]).date.astype(str))
    done = set((OUT / f"empty_{name}.txt").read_text().split()) if (OUT / f"empty_{name}.txt").exists() else set()
    todo = [d for d in days if str(d) not in have and str(d) not in done]
    print(f"[{name}] 已有 {len(have)} 日, 待抓 {len(todo)} 日", flush=True)
    buf, n = {}, 0
    def flush():
        for y, parts in buf.items():
            p = OUT / f"margin_{name}_{y}.parquet"
            old = [pd.read_parquet(p)] if p.exists() else []
            pd.concat(old + parts, ignore_index=True).drop_duplicates(["date", "stock_id"], keep="last").to_parquet(p, index=False)
        buf.clear()
        (OUT / f"empty_{name}.txt").write_text("\n".join(sorted(done)))
    for d in todo:
        try: x = fn(d)
        except Exception:
            traceback.print_exc(); x = None
        if x is None: print(f"[{name}] {d} 失敗", flush=True)
        elif len(x) == 0: done.add(str(d))
        else:
            x["date"] = str(d); buf.setdefault(d.year, []).append(x); n += 1
        if n and n % 50 == 0 and buf:
            flush(); print(f"[{name}] {d} 累計 {n} 日", flush=True)
        time.sleep(sleep)
    flush(); print(f"[{name}] 完成, 新增 {n} 日", flush=True)

def taifex_month(s, e):
    q = {"queryStartDate": f"{s:%Y/%m/%d}", "queryEndDate": f"{e:%Y/%m/%d}"}
    r1 = get("https://www.taifex.com.tw/cht/3/futContractsDateDown", "POST", data={**q, "commodityId": "MXF"})
    r2 = get("https://www.taifex.com.tw/cht/3/futDataDown", "POST", data={**q, "down_type": "1", "commodity_id": "MTX", "commodity_id2": ""})
    if r1 is None or r2 is None: return None
    try:
        A = pd.read_csv(io.StringIO(r1.content.decode("big5", "ignore")), index_col=False)
        B = pd.read_csv(io.StringIO(r2.content.decode("big5", "ignore")), index_col=False)
    except Exception:
        traceback.print_exc(); return None
    if a.test: print("TAIFEX inst", list(A.columns), A.head(3).values.tolist(), "\nTAIFEX all", list(B.columns), B.head(3).values.tolist(), flush=True)
    A.columns = [c.strip() for c in A.columns]; B.columns = [c.strip() for c in B.columns]
    if "日期" not in A.columns or "交易日期" not in B.columns: return pd.DataFrame()
    nc = [c for c in A.columns if "未平倉" in c and "淨額" in c and "口數" in c][0]
    lc = [c for c in A.columns if c.startswith("多方未平倉口數")][0]; sc_ = [c for c in A.columns if c.startswith("空方未平倉口數")][0]
    for c in [nc, lc, sc_]: A[c] = A[c].map(num)
    inst = A.groupby("日期").agg(inst_net=(nc, "sum"), inst_long=(lc, "sum"), inst_short=(sc_, "sum")).reset_index().rename(columns={"日期": "date"})
    B["oi"] = B["未沖銷契約數"].map(num)
    sess = [c for c in B.columns if "交易時段" in c]
    if sess: B = B[B[sess[0]].astype(str).str.contains("一般")]
    B = B[~B["到期月份(週別)"].astype(str).str.contains("/")]
    tot = B.groupby("交易日期").oi.sum().reset_index().rename(columns={"交易日期": "date", "oi": "oi_all"})
    m = inst.merge(tot, on="date", how="outer")
    m["date"] = pd.to_datetime(m.date.astype(str).str.strip()).dt.strftime("%Y-%m-%d")
    return m

def run_mtx(start, end):
    p = OUT / "mtx.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date"])
    have = set(old.date.astype(str).str[:7])
    parts, s = [old], date(start.year, start.month, 1)
    while s <= end:
        e = min((pd.Timestamp(s) + pd.offsets.MonthEnd(0)).date(), end)
        if f"{s:%Y-%m}" not in have or s.year == end.year and s.month == end.month:
            x = taifex_month(s, e)
            print(f"[MTX] {s:%Y-%m} {'失敗' if x is None else len(x)}", flush=True)
            if x is not None and len(x): parts.append(x)
            time.sleep(3)
            if a.test: break
        s = (pd.Timestamp(s) + pd.offsets.MonthBegin(1)).date()
    m = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    m.to_parquet(p, index=False); print(f"[MTX] 共 {len(m)} 日", flush=True)

start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
days = [start + timedelta(i) for i in range((end - start).days + 1)]
days = [d for d in days if d.weekday() < 5]
if a.test: days = [date(2020, 3, 5), date(2023, 6, 6), date(2025, 9, 10)]
th = [threading.Thread(target=run_margin, args=("twse", twse, days, 3.2)),
      threading.Thread(target=run_margin, args=("tpex", tpex, days, 2.0)),
      threading.Thread(target=run_mtx, args=(start, end))]
[t.start() for t in th]; [t.join() for t in th]
for p in sorted(OUT.glob("*.parquet")):
    x = pd.read_parquet(p); print(p.name, len(x), x.date.min() if len(x) else "", x.date.max() if len(x) else "")
if a.test:
    for p in sorted(OUT.glob("*.parquet")): print(pd.read_parquet(p).head(5).to_string())
