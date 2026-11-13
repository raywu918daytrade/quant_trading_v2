"""盤中比對永豐快照 (模擬環境登入) 與 MIS: 開 / 高 / 低 / 現價 / 昨收 / 漲停 / 量 / 時間。
用法: SINOPAC_API_KEY=... SINOPAC_SECRET_KEY=... python3 scripts/diag_quotes.py [代號 ...]"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_live as OL, paper_orb as PO
import shioaji as sj
ids = sys.argv[1:] or ["2330", "2317", "1569", "2221", "3231", "2377", "3029", "6148"]
info = OL.load_info() if hasattr(OL, "load_info") else None
api = PO.connect_simulation_only(sj)
mk = []
for s in ids:
    c = api.Contracts.Stocks[s]; mk.append("twse" if c is not None and str(c.exchange).endswith("TSE") else "tpex")
q = OL.Quotes(OL.MIS(), api)
a = q._sj(ids).set_index("stock_id"); b = OL.MIS().snapshot(ids, mk).set_index("stock_id")
pd.set_option("display.width", 250)
print("永豐快照:\n", a.to_string()); print("MIS:\n", b.to_string())
print("closes 測試 (盤中應該都抓不到, 只驗證流程):")
try: print(q.closes(ids[:2], mk[:2], deadline=(0, 0, 0)))
except Exception as e: print("closes error", e)
api.logout()
