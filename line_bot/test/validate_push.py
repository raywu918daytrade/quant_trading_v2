"""用假資料組出 orb_live / fix_close 會推的每種卡片, 丟給 LINE 的 validate API 檢查格式 (不會真的推出去、不算額度)。
沒有 LINE_CHANNEL_ACCESS_TOKEN 時只檢查能不能組出來。CI: LINE test workflow 在真的推之前先跑這個。"""
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import numpy as np, pandas as pd
import orb_live as OL

ds = "2026-11-10"
s = {"2330": dict(name="台積電", rl=1010.0, rh=1025.0, lots=2, entry_px=1009.0, state="SHORT"),
     "3017": dict(name="奇鋐", rl=820.0, rh=835.0, lots=3, entry_px=819.0, state="STOPPED")}
book = SimpleNamespace(s=s)
evs = [dict(time="09:21:30", stock_id="2330", name="台積電", event="SHORT", price=1009.0), dict(time="09:21:30", stock_id="3017", name="奇鋐", event="STOP", price=836.0)]
mem = {"2330": {"z": 1001.0}}
tr = pd.DataFrame([dict(stock_id="2330", name="台積電", entry_t=21.5, entry_px=1009.0, exit_px=1001.0, reason="收盤回補", lots=2, r_pct=0.56, pnl=11300.0),
                   dict(stock_id="3017", name="奇鋐", entry_t=12.0, entry_px=819.0, exit_px=836.0, reason="停損 (突破區間高)", lots=3, r_pct=-2.31, pnl=-56700.0)])
info = dict(mkt_gap=0.0042, breadth=0.61, n_watch=7)
cards = {
    "signal": OL.card_events(evs[:1], book, ds), "stop": OL.card_events(evs[1:], book, ds), "mixed": OL.card_events(evs, book, ds),
    "reminder": OL.card_reminder(["2330"], book, mem, ds), "summary": OL.card_summary(tr, ds, info), "summary_empty": OL.card_summary(tr.iloc[:0], ds, info),
    "notrade": OL.card_notrade(ds, "⛔ 今日不交易", ["大盤平均跳空 -1.20% < -1%"], dict(mkt_gap=-0.012, breadth=0.2), 3),
    "monday": OL.card_notrade(ds, "😴 今日不交易", ["週一, 規則不交易"]), "holiday": OL.card_notrade(ds, "😴 休市 / 非交易日", ["非交易日或休市"], extra="MIS 回傳日期 20261107 ≠ 今天"),
    "start": OL.card_start(ds, info, tr), "error": OL.card_error("orb_live", "KeyError: 'z'"), "text": OL.LC.from_text("❌ 2026-11-10 09:00 live.yml 失敗 / 中斷\n請看 GitHub Actions log", "error"),
    "src_mis": OL.card_src(False), "src_tick": OL.card_src(True),
    "paper_err": OL.card_paper_err("模擬單補單出錯", KeyError("x")), "paper_login": OL.card_paper_err("永豐模擬環境未登入", title="❌ 今天沒有模擬單"),
    "long_text": OL.LC.from_text("⚠ 13:27:02 訊號資料來源 → MIS 輪詢 (永豐 tick 未啟用或已斷線)", "warn"),
    "fix_close": OL.LC.card("⚠ 收盤價已自動改正", [OL.LC.note("2026-11-10 2330 收盤 1001→1000")], hint="1 筆", kind="warn"),
}
json.dumps(cards, ensure_ascii=False, allow_nan=False)        # NaN 混進去會被 LINE 拒收
tok = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
if not tok: print(f"組出 {len(cards)} 種卡片 OK (沒有 token, 略過 LINE validate)"); sys.exit(0)
import requests
bad = 0
for k, m in cards.items():
    r = requests.post("https://api.line.me/v2/bot/message/validate/push", headers={"Authorization": f"Bearer {tok}"}, json={"messages": [m]}, timeout=10)
    print(f"{k:14s} {r.status_code} {'' if r.ok else r.text[:300]}"); bad += not r.ok
sys.exit(1 if bad else 0)
