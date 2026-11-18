#!/usr/bin/env python3
"""研究用: 抓資金流向歷史資料 (只寫 data/flows/, 不碰線上程式)。在 GHA 跑 (雲端 session 連不到證交所 / 櫃買 / FinMind)。
  1. 個股三大法人買賣超 (每日, 股數): 證交所 T86 (上市) + 櫃買 insti/dailyTrade (上櫃) → data/flows/inst_{twse,tpex}_YYYY.parquet
     欄位: date, stock_id, fi (外資, 不含外資自營商), it (投信), dl (自營商合計), tot (三大法人合計)
  2. 個股當沖成交股數 (每日): 證交所 TWTB4U + 櫃買 → data/flows/dt_{dttwse,dttpex}_YYYY.parquet (date, stock_id, dtv)
     隔日沖分點的替代指標: 前一天當沖比重高 = 短線資金多。
  3. 八大行庫 / 分點: 免費公開來源沒有歷史資料 (證交所買賣日報表要驗證碼且只有當天) → --probe 只試 FinMind 免 token 能不能拿到。
已抓過的日期會跳過 (data/flows 由 actions/cache 保留), 中斷可續跑。
用法: python3 scripts/43_fetch_flows.py [--start 2019-12-01] [--probe]"""
import argparse, json, time, threading, traceback
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "flows"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2019-12-01")
ap.add_argument("--end", default=str(date.today()))
ap.add_argument("--max-min", type=float, default=50, help="每個來源最多抓幾分鐘, 時間到就停 (下次接著抓)")
ap.add_argument("--probe", action="store_true", help="只試抓幾天 / 試 FinMind 資料集, 印欄位")
a = ap.parse_args()
UA = {"User-Agent": "Mozilla/5.0 (research script)"}

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

def col(f, *kws, excl=()):
    """第一個同時含全部關鍵字、且不含 excl 任一字的欄位 index。"""
    for i, x in enumerate(f):
        s = str(x)
        if all(k in s for k in kws) and not any(e in s for e in excl): return i
    return None

def to_df(rows, idx, d):
    recs = []
    for r in rows:
        sid = str(r[idx["sid"]]).strip().replace("=", "").replace('"', "")
        if not sid or not sid[0].isdigit(): continue
        rec = {"date": str(d), "stock_id": sid}
        for k in ["fi", "it", "dl", "tot"]:
            rec[k] = num(r[idx[k]]) if idx.get(k) is not None else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)

def twse(d):
    r = get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    if r is None: return None
    j = r.json()
    f, rows = j.get("fields") or [], j.get("data") or []
    if not rows:
        for t in j.get("tables", []):
            if t.get("data"): f, rows = t.get("fields", []), t["data"]; break
    if not rows: return pd.DataFrame()       # 非交易日
    if a.probe: print("TWSE T86 fields", f, rows[0], flush=True)
    ex = lambda name: next((i for i, x in enumerate(f) if str(x).strip() == name), None)
    idx = {"sid": col(f, "代號"),
           "fi": ex("外陸資買賣超股數(不含外資自營商)") if ex("外陸資買賣超股數(不含外資自營商)") is not None else ex("外資買賣超股數"),
           "it": ex("投信買賣超股數"), "dl": ex("自營商買賣超股數"), "tot": ex("三大法人買賣超股數")}
    if a.probe: print("TWSE idx", idx, flush=True)
    return to_df(rows, idx, d)

def tpex(d):
    r = get("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade", "POST",
            data={"type": "Daily", "sect": "AL", "date": f"{d:%Y/%m/%d}", "id": "", "response": "json"})
    f, rows = [], []
    if r is not None:
        try:
            j = r.json()
            for t in j.get("tables", []):
                if t.get("data"): f, rows = t.get("fields", []), t["data"]; break
        except Exception: pass
    if not rows:
        roc = f"{d.year - 1911}/{d:%m/%d}"
        r = get(f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=AL&t=D&d={roc}")
        if r is None: return None
        try: rows = r.json().get("aaData") or []
        except Exception: return None
        f = []
    if not rows: return pd.DataFrame()
    if a.probe: print("TPEx fields", f, rows[0], flush=True)
    # 欄位名不分組 (都叫 買進/賣出/買賣超股數), 依序: 外資及陸資(不含外資自營商) 2-4, 外資自營商 5-7, 外資及陸資合計 8-10,
    # 投信 11-13, 自營商(自行) 14-16, 自營商(避險) 17-19, 自營商合計 20-22, 三大法人合計 23 (舊網址 aaData 同順序)
    if len(rows[0]) < 24: print(f"[tpex] {d} 欄數 {len(rows[0])} 不符", flush=True); return None
    idx = {"sid": 0, "fi": 4, "it": 13, "dl": 22, "tot": 23}
    if a.probe: print("TPEx idx", idx, flush=True)
    return to_df(rows, idx, d)

def dt_twse(d):
    """當日沖銷交易標的 (上市): 當沖成交股數 / 買進金額 / 賣出金額 → dtv (當沖股數)。"""
    r = get(f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d:%Y%m%d}&selectType=All&response=json")
    if r is None: return None
    j = r.json(); f, rows = [], []
    for t in j.get("tables", []):
        if t.get("data") and any("代號" in str(x) for x in t.get("fields", [])): f, rows = t["fields"], t["data"]
    if not rows: return pd.DataFrame()
    if a.probe: print("TWSE 當沖 fields", f, rows[0], flush=True)
    i_s, i_v = col(f, "代號"), col(f, "成交股數")
    return pd.DataFrame({"date": str(d), "stock_id": [str(r[i_s]).strip() for r in rows], "dtv": [num(r[i_v]) for r in rows]})

DT_TPEX_FAIL = [0]
def dt_tpex(d):
    """當日沖銷交易統計 (上櫃): 試幾個網址, 取含「成交股數」欄位的表; 連續 5 個交易日都找不到就放棄 (不記成已抓)。"""
    if DT_TPEX_FAIL[0] >= 5: raise StopIteration
    roc = f"{d.year - 1911}/{d:%m/%d}"
    tries = [("GET", f"https://www.tpex.org.tw/web/stock/trading/intraday_stat/intraday_trading_stat_result.php?l=zh-tw&d={roc}&o=json", None)]   # 實測只有這個有個股表
    for m, u, data in tries:
        r = get(u, m, tries=1, data=data) if data else get(u, tries=1)
        if r is None: continue
        try: j = r.json()
        except Exception: continue
        tabs = [(t.get("fields", []), t.get("data") or []) for t in j.get("tables", [])]
        if j.get("aaData"): tabs.append((j.get("fields") or [], j["aaData"]))
        for f, rows in tabs:
            if a.probe or DT_TPEX_FAIL[0] == 0 and not rows: pass
            i_s, i_v = col(f, "代號"), col(f, "成交股數")
            if rows and i_s is not None and i_v is not None:
                if a.probe: print("TPEx 當沖 OK", u, f, rows[0], flush=True)
                DT_TPEX_FAIL[0] = -999
                return pd.DataFrame({"date": str(d), "stock_id": [str(r[i_s]).strip() for r in rows], "dtv": [num(r[i_v]) for r in rows]})
        print(f"  TPEx 當沖 {u.split('tpex.org.tw')[1][:60]}: 表 {[ (len(f), len(rw)) for f, rw in tabs]} 欄位 {[f for f, _ in tabs][:2]}", flush=True)
    if DT_TPEX_FAIL[0] >= 0: DT_TPEX_FAIL[0] += 1
    return None

def run(name, fn, days, sleep):
    have = set()
    pre = "dt" if name.startswith("dt") else "inst"
    for p in OUT.glob(f"{pre}_{name}_*.parquet"):
        have |= set(pd.read_parquet(p, columns=["date"]).date.astype(str))
    ef = OUT / f"empty_{name}.txt"
    done = set(ef.read_text().split()) if ef.exists() else set()
    todo = [d for d in days if str(d) not in have and str(d) not in done]
    print(f"[{name}] 已有 {len(have)} 日, 待抓 {len(todo)} 日", flush=True)
    buf, n = {}, 0
    def flush():
        for y, parts in buf.items():
            p = OUT / f"{pre}_{name}_{y}.parquet"
            old = [pd.read_parquet(p)] if p.exists() else []
            pd.concat(old + parts, ignore_index=True).drop_duplicates(["date", "stock_id"], keep="last").to_parquet(p, index=False)
        buf.clear(); ef.write_text("\n".join(sorted(done)))
    t0 = time.time()
    for d in todo:
        if time.time() - t0 > a.max_min * 60:
            print(f"[{name}] 時間到 ({a.max_min:g} 分), 停在 {d}", flush=True); break
        try: x = fn(d)
        except StopIteration:
            print(f"[{name}] 找不到可用來源, 放棄", flush=True); break
        except Exception:
            traceback.print_exc(); x = None
        if x is None: print(f"[{name}] {d} 失敗", flush=True)
        elif len(x) == 0: done.add(str(d))
        else: buf.setdefault(d.year, []).append(x); n += 1
        if n and n % 50 == 0 and buf: flush(); print(f"[{name}] {d} 累計 {n} 日", flush=True)
        time.sleep(sleep)
    flush(); print(f"[{name}] 完成, 新增 {n} 日", flush=True)

def probe_finmind():
    """八大行庫 / 分點: 試 FinMind 免 token。"""
    base = "https://api.finmindtrade.com/api/v4/data"
    tests = [("TaiwanStockGovernmentBankBuySell", {"start_date": "2024-01-02", "end_date": "2024-01-02"}),
             ("TaiwanStockGovernmentBankBuySell", {"data_id": "2330", "start_date": "2024-01-02", "end_date": "2024-01-10"}),
             ("TaiwanStockTradingDailyReport", {"data_id": "2330", "date": "2024-01-02"}),
             ("TaiwanStockTradingDailyReport", {"data_id": "2330", "start_date": "2024-01-02", "end_date": "2024-01-02"}),
             ("TaiwanStockTradingDailyReportSecIdAgg", {"data_id": "2330", "start_date": "2024-01-02", "end_date": "2024-01-05"}),
             ("TaiwanStockInstitutionalInvestorsBuySell", {"data_id": "2330", "start_date": "2024-01-02", "end_date": "2024-01-05"}),
             ("TaiwanStockTotalInstitutionalInvestors", {"start_date": "2024-01-02", "end_date": "2024-01-05"})]
    for ds, q in tests:
        r = get(base, params={"dataset": ds, **q}, tries=1)
        if r is None: print(f"[FinMind] {ds} {q}: 無回應", flush=True); continue
        try: j = r.json()
        except Exception: print(f"[FinMind] {ds}: 非 JSON {r.text[:200]}", flush=True); continue
        dat = j.get("data") or []
        print(f"[FinMind] {ds} {q}: status {j.get('status')} msg {j.get('msg')} 筆數 {len(dat)} 例 {json.dumps(dat[:2], ensure_ascii=False)[:400]}", flush=True)
        time.sleep(3)
    for u in ["https://bsr.twse.com.tw/bshtm/", "https://www.tpex.org.tw/web/stock/aftertrading/broker_trading/brokerBS.php?l=zh-tw"]:
        r = get(u, tries=1); print(f"[分點] {u}: {'無回應' if r is None else r.status_code} {'' if r is None else ('驗證碼' if 'CaptchaControl' in r.text or 'captcha' in r.text.lower() else '')}", flush=True)

start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
days = [start + timedelta(i) for i in range((end - start).days + 1)]
days = [d for d in days if d.weekday() < 5]
if a.probe:
    probe_finmind()
    for d in [date(2020, 3, 2), date(2024, 1, 2), date(2026, 11, 17)]:
        for nm, fn in [("twse", twse), ("tpex", tpex), ("dttwse", dt_twse), ("dttpex", dt_tpex)]:
            x = fn(d); print(nm, d, None if x is None else len(x), flush=True)
            if x is not None and len(x): print(x.head(3).to_string(), "\n", x[x.stock_id.isin(["2330", "2618", "6488"])].to_string(), flush=True)
    raise SystemExit
th = [threading.Thread(target=run, args=("twse", twse, days, 5.0)),
      threading.Thread(target=run, args=("tpex", tpex, days, 3.0)),
      threading.Thread(target=run, args=("dttwse", dt_twse, days, 5.0)),
      threading.Thread(target=run, args=("dttpex", dt_tpex, days, 3.0))]
[t.start() for t in th]; [t.join() for t in th]
for p in sorted(OUT.glob("*.parquet")):
    x = pd.read_parquet(p); print(p.name, len(x), x.date.min(), x.date.max(), x.drop(columns=["date", "stock_id"]).isna().mean().round(3).to_dict())
