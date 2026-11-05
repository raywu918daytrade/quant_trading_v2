#!/usr/bin/env python3
"""LINE 推播 (LINE Messaging API push message)。LINE Notify 已於 2025-03-31 停止服務, 改用官方帳號 (Messaging API channel) 推給自己。

設定 (只從環境變數或 --env-file 讀, 不寫死、不印出 token):
  LINE_CHANNEL_ACCESS_TOKEN   Messaging API channel 的 long-lived channel access token
  LINE_TO                     收訊者的 userId (U 開頭 33 字元) 或群組 groupId (C 開頭)
  LINE_NOTIFY                 選填: 0 / off 關閉; 預設有 token 就開
  LINE_BATCH_SEC              選填: 合併幾秒內的事件成一則訊息 (預設 3)。LINE 依「則數 × 收訊人數」計費 (免費方案每月有則數上限),
                              同一則訊息塞多個事件只算一則, 所以短時間內的多個事件會合併送出。

送出在背景執行緒, 不會卡住行情 / 下單; 失敗只印警告。額度用完 (HTTP 429) 時當天停止推播。
測試: python3 scripts/line_notify.py "測試訊息" [--env-file .env]
"""
import json, os, queue, sys, threading, time, urllib.error, urllib.request
from datetime import datetime, timedelta, timezone

API = "https://api.line.me/v2/bot/message/push"
MAX_CHARS = 4900          # 單則 text 上限 5000 字
TZ = timezone(timedelta(hours=8))


class LineNotifier:
    def __init__(self, prefix="", token=None, to=None, batch_sec=None, enabled=None):
        self.prefix = prefix
        self.token = token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
        self.to = to or os.environ.get("LINE_TO", "").strip()
        self.batch_sec = float(batch_sec if batch_sec is not None else os.environ.get("LINE_BATCH_SEC", 3))
        off = os.environ.get("LINE_NOTIFY", "").strip().lower() in ("0", "off", "false", "no")
        self.enabled = bool(self.token and self.to and not off) if enabled is None else (enabled and bool(self.token and self.to))
        self.q = queue.Queue(); self.dead = False; self._t = None; self.n_ok = 0
        if self.enabled:
            self._t = threading.Thread(target=self._loop, name="line-notify", daemon=True); self._t.start()
        elif (self.token or self.to) and not off:
            print("⚠ LINE 推播未啟用: LINE_CHANNEL_ACCESS_TOKEN 與 LINE_TO 都要設定")

    # ---- 對外 ----
    def send(self, text):
        """排入佇列 (立即返回)。batch_sec 內的訊息合併成一則。"""
        if self.enabled and not self.dead and text: self.q.put(str(text))

    def flush(self, timeout=15):
        """程式結束前呼叫, 等佇列送完。"""
        if not self.enabled or self._t is None: return
        self.q.put(None); self._t.join(timeout)

    # ---- 內部 ----
    def _loop(self):
        while True:
            item = self.q.get()
            if item is None: return
            buf = [item]; end = time.time() + self.batch_sec; stop = False
            while (left := end - time.time()) > 0:
                try: nxt = self.q.get(timeout=left)
                except queue.Empty: break
                if nxt is None: stop = True; break
                buf.append(nxt)
            self._post(buf)
            if stop: return

    def _post(self, texts):
        if self.dead: return
        body = "\n\n".join(texts)
        if self.prefix: body = f"{self.prefix}\n{body}"
        chunks = [body[i:i + MAX_CHARS] for i in range(0, len(body), MAX_CHARS)][:5]     # 一次 push 最多 5 個 message 物件, 仍算一則
        data = json.dumps({"to": self.to, "messages": [{"type": "text", "text": c} for c in chunks]}).encode()
        req = urllib.request.Request(API, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        for k in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as r: r.read(); self.n_ok += 1; return
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:200]
                if e.code == 429:
                    print(f"⚠ LINE 推播額度已用完或太頻繁 (429): {msg}; 本次執行停止推播"); self.dead = True; return
                if e.code in (400, 401, 403):
                    print(f"⚠ LINE 推播失敗 ({e.code}): {msg}; 請檢查 LINE_CHANNEL_ACCESS_TOKEN / LINE_TO, 本次執行停止推播"); self.dead = True; return
                print(f"⚠ LINE 推播失敗 ({e.code}, {k + 1}/3): {msg}")
            except Exception as e:
                print(f"⚠ LINE 推播失敗 ({k + 1}/3): {e}")
            time.sleep(2 * (k + 1))


def px(v):
    """價格格式: 去掉多餘的 0; NaN / None 顯示 —。"""
    try:
        v = float(v)
        if v != v: return "—"
        return f"{v:.2f}".rstrip("0").rstrip(".")
    except Exception: return "—"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="送一則 LINE 測試訊息")
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--env-file", action="append", default=[])
    a = ap.parse_args()
    for f in a.env_file:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    n = LineNotifier("[ORB 測試]", batch_sec=0)
    if not n.enabled: sys.exit("缺少 LINE_CHANNEL_ACCESS_TOKEN 或 LINE_TO")
    n.send(a.text or f"LINE 推播測試 {datetime.now(TZ):%Y-%m-%d %H:%M:%S}"); n.flush()
    print("已送出, 請看 LINE" if n.n_ok else "送出失敗 (見上面的錯誤訊息)")


if __name__ == "__main__":
    main()
