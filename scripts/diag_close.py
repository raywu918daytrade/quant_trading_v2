"""診斷收盤價: 證交所 / 櫃買官方日成交資訊的收盤價 vs HF D1 close vs HF 分 K 最後一根 close。
用法: python3 scripts/diag_close.py 2026-11-04:3211,2301,6182,6290,3016 2026-10-29:6226,4956 ..."""
import sys, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import orb_live as OL

H = {"User-Agent": "Mozilla/5.0"}
_cache = {}
def official(sid, mt, d):
    key = (sid, mt, d.strftime("%Y%m"))
    if key not in _cache:
        rows = []
        try:
            if mt == "twse":
                j = requests.get("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY", params=dict(date=d.strftime("%Y%m01"), stockNo=sid, response="json"), headers=H, timeout=20).json()
                rows = j.get("data") or []
            else:
                j = requests.get("https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock", params=dict(code=sid, date=d.strftime("%Y/%m/01"), response="json"), headers=H, timeout=20).json()
                rows = (j.get("tables") or [{}])[0].get("data") or []
        except Exception as e: print("官方資料失敗", sid, e)
        _cache[key] = rows; time.sleep(1.5)
    roc = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
    for r in _cache[key]:
        if str(r[0]).strip() == roc:
            try: return float(str(r[6]).replace(",", ""))
            except ValueError: return None
    return None

d1 = OL.load_d1(since="2026_10")
out = []
for arg in sys.argv[1:]:
    ds, ids = arg.split(":"); d = pd.Timestamp(ds)
    try: m1 = OL.load_m1_month(d.strftime("%Y_%m")); m1 = m1[m1.date == d]
    except Exception as e: print("分 K 讀取失敗", e); m1 = pd.DataFrame(columns=["stock_id", "t", "close"])
    for sid in ids.split(","):
        r = d1[(d1.date == d) & (d1.stock_id == sid)]
        mt = r.market_type.iloc[0] if len(r) else "twse"
        b = m1[m1.stock_id == sid].sort_values("t")
        out.append(dict(date=ds, stock_id=sid, name=r.name.iloc[0] if len(r) else "", market=mt, official=official(sid, mt, d),
                        d1_close=round(float(r.close.iloc[0]), 2) if len(r) else None,
                        m1_last_t=int(b.t.iloc[-1]) if len(b) else None, m1_last=round(float(b.close.iloc[-1]), 2) if len(b) else None))
x = pd.DataFrame(out)
x["d1_ok"] = (x.d1_close - x.official).abs() < 1e-6
print(x.to_string(index=False))
print(f"D1 與官方一致 {int(x.d1_ok.sum())} / {x.official.notna().sum()} 筆")
