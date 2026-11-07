"""即時行情來源切換: 證交所 MIS (預設) 或 叢集內報價系統的 Redis Stream。
orb_live / paper_orb / diag_gap 只呼叫 make_quotes(), 回傳物件都有 MIS 的兩個方法:
  snapshot(sids, markets) → DataFrame[stock_id, o, h, l, z, y, u, w, v, mis_t, mis_d]   (今日開 / 高 / 低 / 最新 / 昨收 / 漲停 / 跌停 / 累計量(張) / 時間 / 日期 YYYYMMDD)
  closes(sids, markets)   → {stock_id: 收盤價}  (13:30 收盤競價之後的價)

環境變數:
  QUOTE_SOURCE     mis (預設) | redis
  REDIS_URL        例 redis://redis.market-data.svc.cluster.local:6379/0   (K3s 內, 無密碼, 不可對外)
  REDIS_M1_STREAM  預設 stream:m1 (market-data-collector 的 1 分 K stream; 欄位 stock_id, date, ts, open, high, low, close, volume)

Redis 版的限制 (2026-11-07, 尚未在真實盤中驗證, 所以預設仍是 MIS):
  - y (昨收) / u (漲停) / w (跌停) 沒有 → NaN; orb_live.open_stage 會改用 D1 昨收, range_stage 用 昨收×1.1 算漲停 (與回測一致)。
  - 只有 1 分 K: 盤中「最新價」是最後一根 K 的收盤, 跌破判定粒度是 1 分鐘 (與回測相同), 比 MIS 每 10 秒輪詢慢但口徑更接近回測。
  - 報價系統只訂 tick_universe 的前 N 檔; 池內股票若不在訂閱範圍就沒有行情 → snapshot 少列, 09:02 大盤平均跳空會以有行情的檔計算。
    上線前要比對同一天 MIS 與 Redis 的 09:02 跳空、09:05 區間、進場與回補價 (見 README「K3s 部署」)。
"""
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

TZ = ZoneInfo("Asia/Taipei")
COLS = ["stock_id", "o", "h", "l", "z", "y", "u", "w", "v", "mis_t", "mis_d"]


def make_quotes():
    src = os.environ.get("QUOTE_SOURCE", "mis").lower()
    if src == "redis":
        return RedisQuotes()
    if src != "mis": print(f"⚠ QUOTE_SOURCE={src} 不認得, 用 MIS")
    from orb_live import MIS
    return MIS()


def _bar_time(f):
    """stream 欄位 → 台灣時間 datetime (優先 ts epoch 秒/毫秒, 其次 date 字串)."""
    ts = f.get("ts")
    if ts not in (None, ""):
        try:
            x = float(ts)
            if x > 1e12: x /= 1000
            return datetime.fromtimestamp(x, TZ)
        except (TypeError, ValueError): pass
    d = f.get("date")
    if d:
        try: return pd.Timestamp(d).tz_localize(TZ) if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d).tz_convert(TZ)
        except (TypeError, ValueError): pass
    return None


class RedisQuotes:
    """從 Redis Stream 的 1 分 K 累積出 MIS.snapshot 同格式的今日 開/高/低/最新/量。
    每次 snapshot() 先把 stream 讀到最新 (非阻塞), 以 (stock_id, 分鐘) 為 key 存最後一次收到的 K (同一分鐘重送就覆蓋)。"""
    def __init__(self, url=None, stream=None, day=None):
        import redis
        self.url = url or os.environ.get("REDIS_URL")
        if not self.url: raise RuntimeError("QUOTE_SOURCE=redis 但沒有 REDIS_URL")
        self.r = redis.Redis.from_url(self.url, decode_responses=True, socket_timeout=10)
        self.stream = stream or os.environ.get("REDIS_M1_STREAM", "stream:m1")
        self.day = day or datetime.now(TZ).date()
        start = datetime.combine(self.day, dtime(8, 0), TZ)                 # stream id 是毫秒時間戳: 從今天 08:00 開始讀
        self.last_id = f"{int(start.timestamp() * 1000)}-0"
        self.bars = {}                                                        # sid -> {HH:MM: dict(o,h,l,c,v,t)}
        self.r.ping(); print(f"Redis 行情: {self.stream} @ {self.url.split('@')[-1]}")

    def _pull(self, batch=20000):
        while True:
            res = self.r.xread({self.stream: self.last_id}, count=batch)
            if not res: return
            n = 0
            for _, entries in res:
                for eid, f in entries:
                    self.last_id = eid; n += 1
                    t = _bar_time(f); sid = f.get("stock_id")
                    if not sid or t is None or t.date() != self.day: continue
                    try: o, h, l, c, v = (float(f["open"]), float(f["high"]), float(f["low"]), float(f["close"]), float(f.get("volume", 0) or 0))
                    except (KeyError, TypeError, ValueError): continue
                    self.bars.setdefault(sid, {})[t.strftime("%H:%M")] = dict(o=o, h=h, l=l, c=c, v=v, t=t)
            if n < batch: return

    def _row(self, sid, before=None):
        b = self.bars.get(sid)
        if not b: return None
        ks = sorted(k for k in b if before is None or k < before)
        if not ks: return None
        first, last = b[ks[0]], b[ks[-1]]
        return dict(stock_id=sid, o=first["o"], h=max(b[k]["h"] for k in ks), l=min(b[k]["l"] for k in ks), z=last["c"], y=np.nan, u=np.nan, w=np.nan,
                    v=sum(b[k]["v"] for k in ks), mis_t=last["t"].strftime("%H:%M:%S"), mis_d=self.day.strftime("%Y%m%d"))

    def snapshot(self, sids, markets=None, before=None, **_):
        """markets 不用 (Redis 不分上市櫃). before='09:05' 可只取該分鐘之前的 K (與回測 range_from_bars 的 t < 5 一致)."""
        self._pull()
        rows = [r for r in (self._row(s, before) for s in sids) if r]
        return pd.DataFrame(rows, columns=COLS)

    def closes(self, sids, markets=None, deadline=(13, 36, 0), retry=20):
        """收盤價 = 13:30 那根 K 的收盤 (收盤集合競價); 沒到就每 retry 秒重讀到 deadline, 仍沒有的用最後一根 K."""
        import time
        end = datetime.now(TZ).replace(hour=deadline[0], minute=deadline[1], second=deadline[2], microsecond=0)
        while True:
            self._pull()
            got = {s: self.bars[s]["13:30"]["c"] for s in sids if s in self.bars and "13:30" in self.bars[s]}
            todo = [s for s in sids if s not in got]
            if not todo or datetime.now(TZ) >= end: break
            print(f"收盤 K 尚未到 ({len(todo)} 檔: {' '.join(todo)}), {retry} 秒後重讀"); time.sleep(retry)
        if todo: print(f"⚠ {deadline[0]}:{deadline[1]:02d} 仍無 13:30 K: {' '.join(todo)}, 改用最後一根 K 的收盤")
        last = {s: self._row(s)["z"] for s in todo if self._row(s)}
        return {**last, **got}
