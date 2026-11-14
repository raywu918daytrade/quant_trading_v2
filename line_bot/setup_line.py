#!/usr/bin/env python3
"""LINE 官方帳號設定 (由 .github/workflows/line_bot.yml 執行, 也可本機跑):
  richmenu           建立 / 更新聊天室下方的選單 (持倉 / 損益 / 績效 / 狀態 / 網頁), 設成預設; 舊的同名選單會刪掉
  webhook <url>      把 Messaging API 的 webhook 網址設成 Cloudflare Worker, 並測試連線
token 只從環境變數 LINE_CHANNEL_ACCESS_TOKEN 讀, 不印出。"""
import json, os, sys
from pathlib import Path
import requests

API = "https://api.line.me/v2/bot"
NAME = "orb-menu"
SITE = "https://fade.just1stock.com/web/"
HDR = {"Authorization": f"Bearer {os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '')}"}

def call(method, url, **kw):
    r = requests.request(method, url, headers={**HDR, **kw.pop("headers", {})}, timeout=20, **kw)
    if not r.ok: sys.exit(f"{method} {url.split('/v2/bot')[-1]} 失敗 {r.status_code}: {r.text[:300]}")
    return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else {}

def richmenu():
    for m in call("GET", f"{API}/richmenu/list").get("richmenus", []):
        if m.get("name") == NAME: call("DELETE", f"{API}/richmenu/{m['richMenuId']}"); print("刪除舊選單", m["richMenuId"])
    acts = [{"type": "message", "text": "持倉"}, {"type": "message", "text": "損益"}, {"type": "message", "text": "績效"}, {"type": "message", "text": "狀態"},
            {"type": "uri", "uri": SITE + "today.html"}]
    w = 2500 // len(acts)
    body = {"size": {"width": 2500, "height": 843}, "selected": True, "name": NAME, "chatBarText": "查持倉 / 損益",
            "areas": [{"bounds": {"x": i * w, "y": 0, "width": w, "height": 843}, "action": a} for i, a in enumerate(acts)]}
    mid = call("POST", f"{API}/richmenu", json=body)["richMenuId"]
    img = (Path(__file__).parent / "richmenu.png").read_bytes()
    call("POST", f"https://api-data.line.me/v2/bot/richmenu/{mid}/content", data=img, headers={"Content-Type": "image/png"})
    call("POST", f"{API}/user/all/richmenu/{mid}")
    print("✓ 選單已建立並設為預設:", mid)

def webhook(url):
    call("PUT", f"{API}/channel/webhook/endpoint", json={"endpoint": url})
    info = call("GET", f"{API}/channel/webhook/endpoint")
    print("✓ webhook 網址:", info.get("endpoint"), "| Use webhook:", "開" if info.get("active") else "關 (請到 LINE Developers → Messaging API → 打開 Use webhook)")
    t = call("POST", f"{API}/channel/webhook/test", json={"endpoint": url})
    print("webhook 測試:", "成功" if t.get("success") else f"失敗 {t.get('statusCode')} {t.get('reason')} {t.get('detail')}")

if __name__ == "__main__":
    if not HDR["Authorization"].split()[-1]: sys.exit("缺少 LINE_CHANNEL_ACCESS_TOKEN")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "richmenu": richmenu()
    elif cmd == "webhook" and len(sys.argv) > 2: webhook(sys.argv[2])
    else: sys.exit(__doc__)
