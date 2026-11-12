"""盤中對照: HF db/orb/live.json (即時訊號, MIS 輪詢) vs db/orb/paper/live.json (永豐模擬單)。逐檔列出進場價 / 時間 / 出場 / 損益與差距。
用法: python3 scripts/diag_today.py"""
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_REPO, HF_PREFIX, COST
import paper_orb as PO

def get(p): return json.loads(Path(hf_hub_download(HF_REPO, p, repo_type="dataset", force_download=True)).read_text(encoding="utf-8"))
L, P = get(f"{HF_PREFIX}/live.json"), get(f"{HF_PREFIX}/paper/live.json")
print("訊號 live.json", L.get("date"), L.get("phase"), L.get("updated_at"), "| 模擬 live.json", P.get("date"), P.get("phase"), P.get("updated_at"))
ev = {}
for e in P.get("events", []):
    if e.get("event") == "SHORT": ev.setdefault(e["stock_id"], e)
lev = {}
for e in L.get("events", []):
    if e.get("event") == "SHORT": lev.setdefault(e["stock_id"], e)
lw = {r["stock_id"]: r for r in L.get("watch", [])}; pw = {r["stock_id"]: r for r in P.get("positions", [])}
ONLY = set(sys.argv[1:])            # 額外列出的股票 (即使兩邊都沒進場)
rows = []
for sid in sorted(set(lw) | set(pw)):
    a, b = lw.get(sid, {}), pw.get(sid, {})
    le, pe = a.get("entry_px"), b.get("sell_px")
    if not (le == le and le) and not pe and sid not in ONLY: continue
    lots = b.get("lots") or a.get("lots")
    lp = a.get("pnl_pct"); pp = b.get("pnl_pct")
    rows.append(dict(sid=sid, name=a.get("name") or b.get("name"), rl=a.get("rl") or b.get("rl"), lots=lots,
                     sig_state=a.get("state"), sig_entry=le, sig_t=lev.get(sid, {}).get("time"), sig_exit=a.get("exit_px"),
                     sim_state=b.get("state"), sim_rl=b.get("rl"), sim_rh=b.get("rh"), sig_rh=a.get("rh"), skipped=(b.get("skipped") or "")[:40], sim_sell=pe, sim_lots=b.get("sell_lots"), sim_trig_t=str(ev.get(sid, {}).get("tick_time", ""))[:12],
                     sim_recv_t=str(ev.get(sid, {}).get("recv_time", ""))[:12], sim_buy=b.get("buy_px"), last=b.get("last"),
                     entry_ticks=PO.ticks_between(le, pe) if (le == le and le and pe) else None,
                     sig_pnl=round(lp / 100 * le * a.get("lots", 0) * 1000) if lp is not None and le == le and le else None,
                     sim_pnl=round(pp / 100 * pe * (b.get("sell_lots") or 0) * 1000) if pp is not None and pe else None))
x = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(x.to_string(index=False))
if len(x):
    both = x[x.sig_pnl.notna() & x.sim_pnl.notna()]
    print(f"訊號合計 {x.sig_pnl.sum():+,.0f}  模擬合計 {x.sim_pnl.sum():+,.0f}  差 {x.sim_pnl.sum() - x.sig_pnl.sum():+,.0f}")
    print(f"同股 {len(both)} 檔: 訊號 {both.sig_pnl.sum():+,.0f} 模擬 {both.sim_pnl.sum():+,.0f} 差 {both.sim_pnl.sum() - both.sig_pnl.sum():+,.0f}  (進場差 正=模擬賣得較低)")
print("模擬 events:"); [print(" ", {k: e.get(k) for k in ["stock_id", "name", "event", "price", "tick_px", "tick_time", "recv_time", "note"]}) for e in P.get("events", [])[:60] if e.get("event") != "FULL" or e.get("stock_id") in ONLY]
print("訊號 events:"); [print(" ", e) for e in L.get("events", [])[:40]]
for sid in ONLY: print("模擬原始", sid, pw.get(sid)); print("訊號原始", sid, lw.get(sid))
# 訊號端現價: watch.last 與 bars (MIS z) 是否有值
print("訊號 last:", {r["stock_id"]: r.get("last") for r in L.get("watch", []) if r.get("entry_px")})
bars = L.get("bars") or {}
print("訊號 bars 根數:", {s: len(v) for s, v in bars.items()}, "例:", {s: v[-2:] for s, v in list(bars.items())[:2]})
if ONLY:                                     # 直接打一次 MIS, 看 z 欄位
    import orb_live as OL
    m = OL.MIS(); mk = {r["stock_id"]: r.get("market_type") for r in L.get("watch", [])}
    import requests
    codes = "|".join(f"{'tse' if mk.get(s) == 'twse' else 'otc'}_{s}.tw" for s in ONLY) + "|" + "|".join(f"tse_{s}.tw" for s in ONLY)
    js = m.s.get(f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={codes}&json=1&delay=0", timeout=10).json()
    for x in js.get("msgArray", []): print("MIS 原始", {k: x.get(k) for k in ["c", "n", "t", "z", "pz", "o", "h", "l", "y", "v", "tv", "a", "b", "ex"]})
