"""LINE 推播額度檢查 (只讀, 不推播、不佔額度): 本月上限 / 已用, 以及最近幾天每天實際送出的 push 則數。
用法: python3 scripts/diag_line.py   (需要 LINE_CHANNEL_ACCESS_TOKEN)"""
import os, sys
from datetime import datetime, timedelta, timezone
import requests

tok = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
if not tok: sys.exit("沒有 LINE_CHANNEL_ACCESS_TOKEN")
H = {"Authorization": f"Bearer {tok}"}
def get(path, **params):
    r = requests.get("https://api.line.me/v2/bot/" + path, headers=H, params=params, timeout=10)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200])
q, c = get("message/quota"), get("message/quota/consumption")
print("本月額度:", q, "\n本月已用:", c)
tw = datetime.now(timezone(timedelta(hours=8))); tot = 0
for d in range(1, tw.day):                    # 本月 1 日到昨天 (LINE 的每日統計隔天才有)
    ds = tw.replace(day=d).strftime("%Y%m%d"); st, j = get("message/delivery/push", date=ds)
    k = j.get("success", 0) if isinstance(j, dict) else 0; tot += k or 0
    print(ds, "push 送出:", st, j)
if c[0] == 200: print(f"本月到昨天 push {tot} 則 → 今天已送出約 {c[1].get('totalUsage', 0) - tot} 則 (已用量也含 multicast / broadcast)")
