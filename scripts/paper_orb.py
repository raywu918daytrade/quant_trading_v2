#!/usr/bin/env python3
"""永豐 Shioaji 模擬環境自動執行 orb_live 的訊號, 量測實際滑價 (成交價 vs 區間低 / 區間高 / 收盤價差幾檔)。
選股與進出場規則完全沿用 scripts/orb_live.py (build_pool / open_stage / range_stage / Book), 只把「訊號」換成模擬委託。
永遠是模擬環境 (Shioaji(simulation=True) 寫死 + 不啟用 CA + account_balance 查核), 不可能真的下單。

流程 (台灣時間, 與 orb_live run 相同的時間結構):
  08:50   盤前池 (只用前一日以前 D1)
  09:02   證交所 MIS 抓開盤價 → 大盤平均跳空 / 候選; 登入永豐模擬環境, 訂閱候選的逐筆 tick
  09:05   MIS 區間高 / 低 → 監看名單 (同 orb_live 的 range_stage, 含當沖資格與開盤 5 分量濾網)
  盤中    每筆 tick 丟進 orb_live.Book (每日最多 10 檔、每檔 30 萬整數張、先跌破先做、12:00 後失效):
            tick < 區間低 → 立即市價「現股當沖先賣」(daytrade_short)
            tick > 區間高 → 立即市價買回 (停損); 只買回券賣實際成交的張數, 未成交的券賣單先取消
          永豐沒有股票觸價單 API, 所以是「tick 偵測 → 市價單」, 並記錄偵測延遲 (收到 tick 的本機時間 − 交易所 tick 時間)
  13:25   其餘部位以漲停價限價 ROD 買回 (13:25 起收盤集合競價不收市價單, op_code 88), 以 13:30 收盤價成交
  13:35   結算: 每檔每一腳 (進場 / 停損 / 收盤) 的 訊號價、觸發 tick、委託 / 成交時間、成交價、滑價 (元 / 檔 / %)、損益
          寫 paper/<date>_trades.csv、_slippage.csv、_events.csv、_summary.json; --publish 時上傳 HF db/orb/paper/,
          並累計 db/orb/paper/summary.csv (每日) 與 db/orb/paper/slippage.csv (每一腳)

滑價符號: 正 = 對我們不利。券賣: 訊號價 − 成交價; 買回: 成交價 − 訊號價。檔數用台股升降單位逐檔數 (跨 50 / 100 元也正確)。
  拆成兩段: 觸發滑價 (訊號價 → 第一筆跌破/突破的 tick, 至少 1 檔) + 執行滑價 (觸發 tick → 模擬成交價)。
  注意: 模擬環境的成交價是永豐依當時行情撮合的模擬值, 不含真實的委託簿衝擊; 主要量到的是偵測延遲與觸發後的價格移動。

憑證: 環境變數 SINOPAC_API_KEY / SINOPAC_SECRET_KEY (或 --env-file), 絕不印出; HF_TOKEN 只有 --publish 才用。
用法:
  python3 scripts/paper_orb.py run [--publish] [--force] [--env-file F]   # 08:50 後啟動, 自動跑到 13:35
  python3 scripts/paper_orb.py run --dry --now                             # 不等時間、不下單: 算名單 + (有憑證時) 登入查合約/快照
  python3 scripts/paper_orb.py replay --date 2026-10-22                    # 離線: 用歷史分 K 合成 tick + 假券商, 走完整個下單/結算流程
"""
import argparse, csv, io, json, os, queue, sys, threading, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_live as OL
from config import ROOT, COST, HF_REPO, HF_PREFIX

TZ = OL.TZ
PAPER = ROOT / "paper"
HF_PAPER = f"{HF_PREFIX}/paper"
TAG = "orbv2"                 # custom_field (≤6 字): 同一個模擬帳戶也跑 v1 paper_trade, 用它分辨回報
SIMULATION = True             # 安全常數: 改成 False 程式會拒絕執行
AUCTION_T = (13, 25, 10)      # 收盤集合競價開始後, 剩餘部位以漲停價限價買回
END_T = (13, 35, 0)
LIVE_EVERY = 180              # 盤中至少每 3 分鐘上傳一次 live.json (有事件立即上傳)


def now(): return datetime.now(TZ)
def at(h, m, s=0): return now().replace(hour=h, minute=m, second=s, microsecond=0)
def log(*a): print(now().strftime("%H:%M:%S"), *a, flush=True)
def hms(dt): return dt.strftime("%H:%M:%S.%f")[:12] if isinstance(dt, datetime) else ""
def ms(a, b): return round((a - b).total_seconds() * 1000) if isinstance(a, datetime) and isinstance(b, datetime) else None

def sleep_until(t, step=None):
    while (d := (t - now()).total_seconds()) > 0:
        if step: step(min(d, 0.5))
        else: time.sleep(min(d, 5))


# ============================== 升降單位 ==============================
SEG = [(0, 10, 0.01), (10, 50, 0.05), (50, 100, 0.1), (100, 500, 0.5), (500, 1000, 1.0), (1000, 1e9, 5.0)]   # 與 orb_live.tick_size 相同的級距

def tick_index(p):
    """價格在台股升降單位格子上的位置 (0 元起算的檔數)."""
    n = 0.0
    for lo, hi, t in SEG:
        if p >= hi: n += (hi - lo) / t
        else: return n + (p - lo) / t
    return n

def ticks_between(a, b):
    """a 比 b 高幾檔 (可為負 / 小數: 均價不在格子上時)."""
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)): return None
    return round(tick_index(a) - tick_index(b), 2)


# ============================== 憑證 / 登入 ==============================
def load_env_file(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        os.environ.setdefault(k, v)

def has_keys(): return bool(os.environ.get("SINOPAC_API_KEY") and os.environ.get("SINOPAC_SECRET_KEY"))

def connect_simulation_only(sj):
    """登入永豐『模擬環境』並做三道查核 (同 v1 paper_trade.py), 任何一道不過就退出, 絕不碰正式帳號:
    1. 常數 SIMULATION 必須為 True, 且 Shioaji(simulation=True) 寫死。
    2. 從不呼叫 activate_ca(): 正式環境的股票委託必須憑證簽章。
    3. account_balance(): 模擬環境回傳 date='' 與餘額 0; 看到真實日期或餘額就停止。"""
    if SIMULATION is not True: sys.exit("安全停止: SIMULATION 不是 True")
    api = sj.Shioaji(simulation=True)
    api.login(api_key=os.environ["SINOPAC_API_KEY"], secret_key=os.environ["SINOPAC_SECRET_KEY"])
    try:
        bal = api.account_balance()
        date_ = str(getattr(bal, "date", "") or ""); amt = float(getattr(bal, "acc_balance", 0) or 0)
    except Exception as e:
        try: api.logout()
        except Exception: pass
        sys.exit(f"安全停止: 無法查核 account_balance ({e})")
    if date_ != "" or amt != 0:
        try: api.logout()
        except Exception: pass
        sys.exit("安全停止: account_balance 回傳真實資料, 看起來不是模擬環境, 拒絕執行")
    log(f"✓ 模擬環境確認 (shioaji {getattr(sj, '__version__', '?')}): simulation=True、未啟用 CA、account_balance 為模擬預設值")
    return api

def to_dt(ts):
    """Shioaji 成交 ts (epoch 秒 / 奈秒 / datetime) → 台灣時間 datetime."""
    if isinstance(ts, datetime): return ts.replace(tzinfo=TZ) if ts.tzinfo is None else ts.astimezone(TZ)
    try: x = float(ts)
    except (TypeError, ValueError): return None
    if x > 1e12: x /= 1e9
    return datetime.fromtimestamp(x, TZ) if x > 0 else None


# ============================== 券商 ==============================
class SinopacBroker:
    """永豐模擬環境下單。只有 MKT/ROD 與 (13:25 後) 漲停價 LMT/ROD 兩種委託。"""
    def __init__(self, api, sj):
        self.api, self.sj = api, sj; self.contracts = {}

    def contract(self, sid):
        if sid not in self.contracts:
            c = self.api.Contracts.Stocks[sid]          # .get() 回 BaseContract, 沒有 day_trade / limit_up
            if c is None: raise KeyError(sid)
            self.contracts[sid] = c
        return self.contracts[sid]

    def can_short(self, sid):
        try: c = self.contract(sid)
        except Exception as e: return f"合約查詢失敗: {e}"
        return "" if str(c.day_trade) == "DayTrade.Yes" else f"永豐合約不可現沖券賣 ({c.day_trade})"

    def limit_up(self, sid): return float(self.contract(sid).limit_up)

    def _order(self, sid, action, lots, daytrade_short=False, limit_px=None):
        sj = self.sj
        kw = dict(price=limit_px or 0, quantity=int(lots), action=action, price_type=sj.StockPriceType.LMT if limit_px else sj.StockPriceType.MKT,
                  order_type=sj.OrderType.ROD, order_lot=sj.StockOrderLot.Common, account=self.api.stock_account, custom_field=TAG)
        if daytrade_short: kw["daytrade_short"] = True
        else: kw["order_cond"] = sj.StockOrderCond.Cash
        tr = self.api.place_order(self.contract(sid), self.api.Order(**kw))
        return tr, str(getattr(tr.status, "status", ""))

    def sell_short(self, sid, lots): return self._order(sid, self.sj.Action.Sell, lots, daytrade_short=True)
    def buy(self, sid, lots, limit_px=None): return self._order(sid, self.sj.Action.Buy, lots, limit_px=limit_px)
    def status(self, tr): return str(getattr(tr.status, "status", ""))
    def cancel(self, tr): self.api.cancel_order(tr)

    def deals(self, tr):
        """update_status 後的成交明細 [(px, qty, dt)] (callback 可能漏)."""
        try: return [(float(d.price), int(d.quantity), to_dt(getattr(d, "ts", None))) for d in (tr.status.deals or [])]
        except Exception: return []

    def refresh(self):
        try: self.api.update_status(self.api.stock_account)
        except Exception as e: log("update_status 失敗:", e)


class SimBroker:
    """離線回放用的假券商: 市價單以下一筆 tick 成交 (模擬 1 筆延遲), 漲停價限價單以收盤價成交。"""
    class T:
        def __init__(self, sid, act, lots, lmt): self.sid, self.act, self.lots, self.lmt, self.st, self.fills = sid, act, lots, lmt, "PendingSubmit", []

    def __init__(self, limit_ups, on_deal):
        self.lu, self.on_deal, self.pending = limit_ups, on_deal, []
    def can_short(self, sid): return ""
    def limit_up(self, sid): return self.lu[sid]
    def sell_short(self, sid, lots): return self._new(sid, "Sell", lots, None)
    def buy(self, sid, lots, limit_px=None): return self._new(sid, "Buy", lots, limit_px)
    def _new(self, sid, act, lots, lmt):
        t = self.T(sid, act, lots, lmt); self.pending.append(t); return t, t.st
    def status(self, t): return t.st
    def cancel(self, t):
        if t.st != "Filled": t.st = "Cancelled"; self.pending = [x for x in self.pending if x is not t]
    def deals(self, t): return list(t.fills)
    def refresh(self): pass
    def _fill(self, t, px, dt):
        t.fills.append((px, t.lots, dt)); t.st = "Filled"; self.on_deal(t.sid, t.act, px, t.lots, dt)
    def feed(self, sid, px, dt):
        for t in [x for x in self.pending if x.sid == sid and x.lmt is None]:
            self.pending.remove(t); self._fill(t, px, dt + timedelta(milliseconds=30))
    def final(self, closes, dt):
        for t in list(self.pending):
            self.pending.remove(t); self._fill(t, closes[t.sid], dt)


# ============================== 執行 ==============================
class PaperOrb:
    """把 tick 餵給 orb_live.Book, Book 的 SHORT / STOP 事件轉成委託, 記錄每一腳的時間與成交。
    tick callback (Shioaji 執行緒) 只更新 Book 並排隊; 下單在主執行緒 process() 做, 避免卡住行情。"""

    def __init__(self, book, watch, broker, ds):
        self.book, self.broker, self.ds = book, broker, ds
        self.w = {r.stock_id: r._asdict() for r in watch.itertuples()}
        self.lock = threading.RLock(); self.q = queue.Queue(); self.ready = False; self.closing = False
        self.last = {}; self.last_t = {}; self.n_ticks = defaultdict(int)
        self.leg = defaultdict(dict)            # sid -> {"entry": {...}, "exit": {...}}
        self.orders = defaultdict(list)         # (sid, "Sell"/"Buy") -> [trade]
        self.deals = defaultdict(list)          # (sid, "Sell"/"Buy") -> [(px, qty, dt)]
        self.skipped = {}; self.events = []; self.dirty = False
        self._cur = None; self.clock = now     # replay 換成合成 tick 的時間
        self.book.log = self._on_book_event

    # ---- 行情 ----
    def on_tick(self, *args):
        tick = args[-1]                          # 新版 1 參數 (tick), 舊版 2 參數 (exchange, tick)
        try:
            if getattr(tick, "simtrade", False) or getattr(tick, "suspend", False): return
            recv = now(); dt = getattr(tick, "datetime", None)
            dt = (dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)) if isinstance(dt, datetime) else recv
            self.on_price(str(tick.code), float(tick.close), dt, recv)
        except Exception as e:
            log("tick callback error:", e)

    def on_price(self, sid, px, dt, recv):
        with self.lock:
            self.last[sid] = px; self.last_t[sid] = dt; self.n_ticks[sid] += 1
            if not self.ready or self.closing or sid not in self.book.s or self.book.s[sid]["state"] not in ("WAIT", "SHORT"): return   # 13:25 收盤回補後不再觸發停損
            self._cur = (px, dt, recv)
            self.book.step(OL.tmin(dt), sid, px, px, px, px)

    def _on_book_event(self, e):
        px, dt, recv = self._cur
        e = {**e, "tick_px": px, "tick_time": dt, "recv_time": recv}
        self.events.append(e); self.q.put(e); self.dirty = True

    # ---- 成交回報 ----
    def on_order(self, stat, msg):
        try:
            s = str(stat)
            if "Deal" in s:
                if msg.get("custom_field") not in (None, "", TAG): return          # v1 paper_trade 的回報
                act = "Sell" if "Sell" in str(msg.get("action")) else "Buy"
                self.on_deal(str(msg.get("code")), act, float(msg["price"]), int(msg["quantity"]), to_dt(msg.get("ts")) or now())
            else:
                o = msg.get("order", {}) or {}; op = msg.get("operation", {}) or {}
                if o.get("custom_field") not in (None, "", TAG): return
                if op.get("op_code") not in ("00", None):
                    sid = str((msg.get("contract") or {}).get("code")); txt = f"op_code={op.get('op_code')} {op.get('op_msg')}"
                    log(f"委託異常 {sid} {o.get('action')} {txt}")
                    L = self.leg.get(sid, {}).get("entry" if "Sell" in str(o.get("action")) else "exit")
                    if L is not None: L["status"] = f"{L.get('status', '')} | {txt}".strip(" |")
        except Exception as e:
            log("order callback error:", e)

    def on_deal(self, sid, act, px, qty, dt):
        with self.lock:
            self.deals[(sid, act)].append((px, qty, dt)); self.dirty = True
        log(f"成交 {sid} {act} {qty} 張 @ {px}")

    def sync_deals(self):
        """callback 可能漏: 用 update_status 的 trade.status.deals 補齊 (以張數多者為準)."""
        self.broker.refresh()
        with self.lock:
            for key, trs in self.orders.items():
                ds = [d for tr in trs for d in self.broker.deals(tr)]
                if sum(q for _, q, _ in ds) > sum(q for _, q, _ in self.deals[key]): self.deals[key] = ds

    def filled(self, key):
        ds = self.deals[key]; q = sum(q for _, q, _ in ds)
        return ((sum(p * q for p, q, _ in ds) / q) if q else None, q, min((d for _, _, d in ds if d), default=None))

    # ---- 下單 ----
    def process(self, timeout=0.5):
        try: e = self.q.get(timeout=timeout)
        except queue.Empty: return
        while True:
            self._handle(e)
            try: e = self.q.get_nowait()
            except queue.Empty: break

    def _handle(self, e):
        sid = e["stock_id"]; p = self.w[sid]
        if e["event"] == "SHORT":
            L = self.leg[sid]["entry"] = dict(signal_px=p["rl"], trigger_px=e["tick_px"], tick_time=e["tick_time"], recv_time=e["recv_time"])
            why = self.skipped.get(sid) or self.broker.can_short(sid)
            if why:
                self.skipped[sid] = why; L["status"] = "skipped"; log(f"✗ {sid} 跌破但不下單: {why}"); return
            try:
                tr, st = self.broker.sell_short(sid, p["lots"]); L["order_time"] = self.clock(); L["status"] = st
                self.orders[(sid, "Sell")].append(tr)
                log(f"券賣 {sid} {p['name']} {p['lots']} 張 市價 (觸發 {e['tick_px']} < 區間低 {p['rl']:g}, tick {hms(e['tick_time'])}, 收到 {hms(e['recv_time'])}) → {st}")
            except Exception as ex:
                L["status"] = f"委託失敗: {ex}"; self.skipped[sid] = L["status"]; log("✗", sid, L["status"])
        elif e["event"] == "STOP":
            self.leg[sid]["exit"] = dict(leg="stop", signal_px=p["rh"], trigger_px=e["tick_px"], tick_time=e["tick_time"], recv_time=e["recv_time"])
            self.cover(sid)
        elif e["event"] == "FULL":
            log(f"{sid} {p['name']} 跌破但當日已滿 {self.book.max_n} 檔")

    def cover(self, sid, limit_px=None):
        """買回 = 券賣『實際成交』張數; 還沒成交的券賣單先取消 (避免回補後才成交變裸空)."""
        L = self.leg[sid]["exit"]
        for tr in self.orders.get((sid, "Sell"), []):
            st = self.broker.status(tr)
            if "Filled" not in st or "PartFilled" in st:
                try: self.broker.cancel(tr); log(f"取消未成交券賣單 {sid} ({st})")
                except Exception as ex: log(f"取消券賣單 {sid} 失敗 (可能已成交): {ex}")
        _, q, _ = self.filled((sid, "Sell"))
        if q <= 0:
            self.sync_deals(); _, q, _ = self.filled((sid, "Sell"))
        if q <= 0:
            L["status"] = "no_fill"; L["ordered"] = 0; log(f"{sid} 券賣未成交, 不買回 (之後若才回報成交, reconcile 會補買回)"); return
        self._buy(sid, q, limit_px)

    def _buy(self, sid, q, limit_px):
        L = self.leg[sid]["exit"]
        try:
            tr, st = self.broker.buy(sid, q, limit_px=limit_px); L.setdefault("order_time", self.clock()); L["status"] = st
            L["ordered"] = L.get("ordered", 0) + q
            self.orders[(sid, "Buy")].append(tr)
            log(f"買回 {sid} {q} 張 {f'限價 {limit_px:g} (收盤競價)' if limit_px else '市價'} ({L['leg']}) → {st}")
        except Exception as ex:
            L["status"] = f"委託失敗: {ex}"; log("✗ 買回委託失敗", sid, ex)

    def reconcile(self):
        """券賣成交張數 > 已送出的買回張數 (取消券賣與成交賽跑時會發生) → 補送買回差額."""
        for sid, legs in list(self.leg.items()):
            L = legs.get("exit")
            if not L or "ordered" not in L: continue
            _, q, _ = self.filled((sid, "Sell"))
            if q > L["ordered"]:
                log(f"⚠ {sid} 券賣成交 {q} 張 > 已買回 {L['ordered']} 張, 補買回差額")
                self._buy(sid, q - L["ordered"], self.broker.limit_up(sid) if self.clock() >= self.clock().replace(hour=AUCTION_T[0], minute=AUCTION_T[1], second=AUCTION_T[2], microsecond=0) else None)

    def close_all(self, limit=True):
        """13:25 後: Book 仍在 SHORT 的部位, 以漲停價限價買回 (= 以收盤價成交)."""
        with self.lock:
            self.closing = True; holding = [s for s, p in self.book.s.items() if p["state"] == "SHORT"]
        for sid in holding:
            if "exit" in self.leg[sid] or self.leg[sid].get("entry", {}).get("status") == "skipped": continue
            self.leg[sid]["exit"] = dict(leg="close", signal_px=np.nan, trigger_px=np.nan, tick_time=None, recv_time=None)
            lim = None
            if limit:
                try: lim = self.broker.limit_up(sid)
                except Exception: lim = self.w[sid]["lu"]
            self.cover(sid, lim)
        return holding

    def expire(self, t):
        with self.lock:
            for p in self.book.s.values():
                if p["state"] == "WAIT" and t >= OL.P["DEADLINE"] + 1: p["state"] = "EXPIRED"

    # ---- 結算 ----
    def settle(self):
        closes = {sid: self.last.get(sid, np.nan) for sid in self.book.s}
        self.book.close_all(closes)                                    # Book 的理論結果: 進場 = 觸發 tick, 收盤 = 最後一筆 tick
        bk = self.book.trades(); bk = bk.set_index("stock_id") if len(bk) else bk
        trades, legs = [], []
        for sid, p in self.book.s.items():
            if sid not in self.leg and p["state"] != "FULL": continue
            w = self.w[sid]; E = self.leg.get(sid, {}).get("entry", {}); X = self.leg.get(sid, {}).get("exit", {})
            sell_px, sell_q, sell_t = self.filled((sid, "Sell")); buy_px, buy_q, buy_t = self.filled((sid, "Buy"))
            close_px = closes.get(sid, np.nan)
            if X.get("leg") == "close": X["signal_px"] = close_px
            # 回測口徑 (理想): 進場 = 區間低, 停損 = 區間高 ×(1+0.1%), 收盤 = 收盤價
            ideal_exit = w["rh"] * (1 + OL.P["STOP_SLIP"]) if X.get("leg") == "stop" else close_px
            r_ideal = 1 - ideal_exit / w["rl"] - COST if (E and np.isfinite(ideal_exit)) else np.nan
            q = min(sell_q, buy_q)
            r_sim = (1 - buy_px / sell_px - COST) if (sell_px and buy_px) else np.nan
            pnl = ((sell_px - buy_px) * 1000 * q - COST * sell_px * 1000 * q) if (sell_px and buy_px) else np.nan
            if sid in self.skipped and not sell_px: status = f"略過: {self.skipped[sid]}"
            elif p["state"] == "FULL": status = "跌破但額滿"
            elif not sell_px: status = "券賣未成交"
            elif not buy_px: status = "未回補"
            else: status = {"stop": "停損", "close": "收盤回補"}.get(X.get("leg"), X.get("leg", ""))
            trades.append(dict(date=self.ds, stock_id=sid, name=w["name"], pc=w["pc"], gap=round(w["gap"] * 100, 2), rl=w["rl"], rh=w["rh"], lots=w["lots"],
                               status=status, exit_leg=X.get("leg", ""),
                               sell_px=sell_px, sell_lots=sell_q, sell_time=hms(sell_t), buy_px=buy_px, buy_lots=buy_q, buy_time=hms(buy_t), close_px=close_px,
                               entry_slip_ticks=ticks_between(w["rl"], sell_px) if sell_px else None,
                               exit_slip_ticks=ticks_between(buy_px, X.get("signal_px")) if (buy_px and X) else None,
                               sim_net_pct=r_sim * 100, sim_pnl=pnl,
                               book_net_pct=bk.loc[sid, "r_pct"] if len(bk) and sid in bk.index else np.nan,
                               ideal_net_pct=r_ideal * 100, ideal_pnl=r_ideal * w["rl"] * 1000 * w["lots"]))
            for name, L, act, fpx, fq, ft in [("entry", E, "Sell", sell_px, sell_q, sell_t), ("exit", X, "Buy", buy_px, buy_q, buy_t)]:
                if not L: continue
                sig, trig = L.get("signal_px"), L.get("trigger_px")
                sgn = 1 if act == "Sell" else -1                        # 正 = 不利: 賣低了 / 買高了
                def slip(a, b): return round(sgn * (a - b), 4) + 0.0 if (a is not None and b is not None and np.isfinite(a) and np.isfinite(b)) else None
                def tks(a, b): x = ticks_between(a, b); return None if x is None else sgn * x + 0.0
                legs.append(dict(date=self.ds, stock_id=sid, name=w["name"], leg=L.get("leg", "entry"), side=act, lots=fq or 0,
                                 signal_px=sig, trigger_px=trig, fill_px=fpx,
                                 slip_yuan=slip(sig, fpx), slip_ticks=tks(sig, fpx), slip_pct=(slip(sig, fpx) / sig * 100) if (slip(sig, fpx) is not None and sig) else None,
                                 trigger_slip_ticks=tks(sig, trig), exec_slip_ticks=tks(trig, fpx),
                                 tick_size=OL.tick_size(sig) if sig is not None and np.isfinite(sig) else None,
                                 tick_time=hms(L.get("tick_time")), recv_time=hms(L.get("recv_time")), order_time=hms(L.get("order_time")), fill_time=hms(ft),
                                 detect_ms=ms(L.get("recv_time"), L.get("tick_time")), order_ms=ms(L.get("order_time"), L.get("recv_time")),
                                 fill_ms=ms(ft, L.get("order_time")), tick_to_fill_ms=ms(ft, L.get("tick_time")),
                                 order_status=L.get("status", ""), filled=bool(fq)))
        return pd.DataFrame(trades), pd.DataFrame(legs)

    def live_rows(self):
        rows = []
        with self.lock:
            for sid, p in self.book.s.items():
                w = self.w[sid]; sp, sq, _ = self.filled((sid, "Sell")); bp, bq, _ = self.filled((sid, "Buy")); last = self.last.get(sid)
                ref = bp or last
                rows.append(dict(stock_id=sid, name=w["name"], rl=w["rl"], rh=w["rh"], lots=w["lots"], state=p["state"], state_txt=OL.STATE_TXT.get(p["state"], p["state"]),
                                 sell_px=sp, sell_lots=sq, buy_px=bp, last=last, skipped=self.skipped.get(sid, ""),
                                 entry_slip_ticks=ticks_between(w["rl"], sp) if sp else None,
                                 pnl_pct=(1 - ref / sp - COST) * 100 if (sp and ref) else None))
        return rows


# ============================== 輸出 / 發布 ==============================
def summarize(ds, info, trades, legs, n_watch, mode):
    def mean(s): s = pd.to_numeric(s, errors="coerce").dropna(); return round(float(s.mean()), 3) if len(s) else None
    def med(s): s = pd.to_numeric(s, errors="coerce").dropna(); return round(float(s.median()), 1) if len(s) else None
    t = trades if len(trades) else pd.DataFrame(columns=["sell_px", "buy_px", "status", "exit_leg", "sim_net_pct", "sim_pnl", "ideal_net_pct", "ideal_pnl", "book_net_pct"])
    g = legs if len(legs) else pd.DataFrame(columns=["leg", "filled", "slip_ticks", "trigger_slip_ticks", "exec_slip_ticks", "detect_ms", "tick_to_fill_ms"])
    gf = g[g.filled == True]
    done = t[t.sell_px.notna() & t.buy_px.notna()]
    return dict(date=ds, mode=mode, trade_ok=bool(info.get("trade_ok", False)), reasons="；".join(info.get("reasons", [])),
                mkt_gap_pct=None if info.get("mkt_gap") is None else round(info["mkt_gap"] * 100, 2),
                n_cand=info.get("n_cand", 0), n_watch=n_watch, n_signal=len(t), n_filled=int(t.sell_px.notna().sum()),
                n_stop=int((done.exit_leg == "stop").sum()), n_close=int((done.exit_leg == "close").sum()), n_unclosed=int((t.status == "未回補").sum()),
                n_skipped=int(t.status.astype(str).str.startswith("略過").sum()), n_full=int((t.status == "跌破但額滿").sum()),
                entry_slip_ticks=mean(gf[gf.leg == "entry"].slip_ticks), entry_trigger_ticks=mean(gf[gf.leg == "entry"].trigger_slip_ticks),
                entry_exec_ticks=mean(gf[gf.leg == "entry"].exec_slip_ticks),
                stop_slip_ticks=mean(gf[gf.leg == "stop"].slip_ticks), close_slip_ticks=mean(gf[gf.leg == "close"].slip_ticks),
                detect_ms_median=med(g.detect_ms), tick_to_fill_ms_median=med(gf[gf.leg != "close"].tick_to_fill_ms),
                sim_avg_net_pct=mean(done.sim_net_pct), ideal_avg_net_pct=mean(t.ideal_net_pct), book_avg_net_pct=mean(t.book_net_pct),
                sim_pnl=None if not len(done) else round(float(done.sim_pnl.sum())), ideal_pnl=round(float(pd.to_numeric(t.ideal_pnl, errors="coerce").sum())) if len(t) else 0,
                finished_at=now().strftime("%Y-%m-%d %H:%M:%S"))

def write_day(ds, trades, legs, events, summ, out=PAPER):
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / f"{ds}_trades.csv", index=False, encoding="utf-8-sig", float_format="%.4g")
    legs.to_csv(out / f"{ds}_slippage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: (hms(v) if isinstance(v, datetime) else v) for k, v in e.items()} for e in events]).to_csv(out / f"{ds}_events.csv", index=False, encoding="utf-8-sig")
    (out / f"{ds}_summary.json").write_text(json.dumps(OL._jclean(summ), ensure_ascii=False, indent=1), encoding="utf-8")

class Pub:
    """paper 狀態上傳 HF db/orb/paper/ (同 orb_live.Publisher 的做法: token 只從環境變數讀, 不印出)."""
    def __init__(self, enabled, min_gap=LIVE_EVERY):
        self.enabled, self.min_gap, self.last, self.api = enabled, min_gap, 0.0, None
        if enabled:
            tok = os.environ.get("HF_TOKEN")
            if not tok: log("⚠ --publish 但沒有 HF_TOKEN, 只寫本機 paper/"); self.enabled = False
            else:
                from huggingface_hub import HfApi
                self.api = HfApi(token=tok)

    def live(self, state, force=False):
        PAPER.mkdir(exist_ok=True); p = PAPER / "live.json"
        state = {**state, "updated_at": now().strftime("%Y-%m-%d %H:%M:%S")}
        p.write_text(json.dumps(OL._jclean(state), ensure_ascii=False, default=OL._jdefault, allow_nan=False), encoding="utf-8")
        if not self.enabled or (not force and time.time() - self.last < self.min_gap): return
        self.last = time.time()
        for k in range(3):
            try:
                self.api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{HF_PAPER}/live.json", repo_id=HF_REPO, repo_type="dataset",
                                     commit_message=f"orb paper live {state.get('date')} {state.get('phase')}"); return
            except Exception as e: log(f"HF 上傳失敗 ({k + 1}/3): {e}"); time.sleep(3)

    def _cumulative(self, name, today_df, ds):
        """下載 HF 上的累計檔, 換掉同一天的列後回傳 CSV bytes; 讀取失敗 (非「檔案不存在」) 回 None, 避免覆蓋掉歷史."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        hist = pd.DataFrame()
        for k in range(3):
            try:
                hist = pd.read_csv(hf_hub_download(HF_REPO, f"{HF_PAPER}/{name}", repo_type="dataset", force_download=True), dtype={"stock_id": str}); break
            except EntryNotFoundError: break
            except Exception as e:
                if k == 2: log(f"⚠ 讀 HF {name} 失敗 ({e}), 本次不更新累計檔"); return None
                time.sleep(3)
        if len(hist) and "date" in hist: hist = hist[hist.date.astype(str) != ds]
        out = pd.concat([hist, today_df], ignore_index=True)
        return out.to_csv(index=False).encode("utf-8")

    def day(self, ds, summ, legs):
        if not self.enabled: return
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(f"{HF_PAPER}/{f.name}", str(f)) for f in sorted(PAPER.glob(f"{ds}_*")) if f.is_file()]
        if (PAPER / f"{ds}_summary.json").exists(): ops.append(CommitOperationAdd(f"{HF_PAPER}/latest.json", str(PAPER / f"{ds}_summary.json")))
        for name, df in [("summary.csv", pd.DataFrame([OL._jclean(summ)])), ("slippage.csv", legs)]:
            b = self._cumulative(name, df, ds)
            if b is not None: ops.append(CommitOperationAdd(f"{HF_PAPER}/{name}", b))
        for k in range(3):
            try:
                self.api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops,
                                       commit_message=f"orb paper {ds}: {summ.get('n_filled', 0)} filled, entry slip {summ.get('entry_slip_ticks')} ticks")
                log(f"已上傳 {HF_PAPER}/ ({ds}, {len(ops)} 檔)"); return
            except Exception as e: log(f"HF commit 失敗 ({k + 1}/3): {e}"); time.sleep(5)

def no_trade_day(ds, info, pub, note):
    summ = summarize(ds, info, pd.DataFrame(), pd.DataFrame(), 0, f"simulation · {note}")
    write_day(ds, pd.DataFrame(columns=["date", "stock_id"]), pd.DataFrame(columns=["date", "stock_id", "leg"]), [], summ)
    pub.live(dict(date=ds, phase="不交易", note=note), force=True); pub.day(ds, summ, pd.DataFrame())
    log("今日不交易:", note)


# ============================== run (盤中) ==============================
def cmd_run(a):
    for f in a.env_file: load_env_file(f)
    today = pd.Timestamp(now().date()); ds = str(today.date())
    if today.dayofweek >= 5 and not a.dry: sys.exit("週末不跑")
    if not has_keys() and not a.dry: sys.exit("缺少 SINOPAC_API_KEY / SINOPAC_SECRET_KEY")
    PAPER.mkdir(exist_ok=True); pub = Pub(a.publish and not a.dry)
    pool, last = OL.cmd_pool(argparse.Namespace(date=None))
    if today.dayofweek in OL.P["NO_DOW"] and not a.force:
        no_trade_day(ds, dict(trade_ok=False, reasons=["週一不交易"]), pub, "週一不交易"); return
    elig = None if a.no_eligibility else OL.fetch_eligible(ds)
    mis = OL.MIS()
    # ---- 09:02 開盤 ----
    if not a.now: OL.wait_until(9, 2, 0)
    snap = mis.snapshot(pool.stock_id.tolist(), pool.market_type.tolist())
    mis_day = snap.mis_d.dropna().mode()
    if len(mis_day) == 0 or mis_day.iloc[0] != today.strftime("%Y%m%d"):
        msg = f"MIS 回傳日期 {mis_day.iloc[0] if len(mis_day) else '無'} ≠ 今天, 可能非交易日或尚未開盤"
        log(msg + ", 結束。"); pub.live(dict(date=ds, phase="非交易日", note=msg), force=True); sys.exit(0 if a.publish else 2)
    cand, info = OL.open_stage(pool, snap.rename(columns={"o": "open"}), today); info["n_cand"] = len(cand)
    log(f"09:02 開盤: 大盤平均跳空 {info['mkt_gap']*100:+.2f}%；候選 {len(cand)} 檔")
    if not info["trade_ok"] and not a.force:
        no_trade_day(ds, info, pub, "；".join(info["reasons"])); return
    # ---- 登入模擬環境, 訂閱候選 tick ----
    api = broker = None
    if has_keys():
        import shioaji as sj
        api = connect_simulation_only(sj); broker = SinopacBroker(api, sj)
    elif a.dry: log("dry: 沒有永豐憑證, 跳過登入 (只驗證選股流程)")
    holder = {}
    if api is not None:
        api.set_order_callback(lambda stat, msg: holder["po"].on_order(stat, msg) if "po" in holder else None)
        api.set_on_tick_stk_v1_callback(lambda *x: holder["po"].on_tick(*x) if "po" in holder else None)
        for sid in cand.stock_id:
            try: api.subscribe(broker.contract(sid), quote_type=sj.QuoteType.Tick)
            except Exception as e: log(f"訂閱 {sid} 失敗: {e}")
        log(f"已訂閱 {len(cand)} 檔候選 tick")
    pub.live(dict(date=ds, phase="等待 09:05", n_cand=len(cand)), force=True)
    # ---- 09:05 區間 ----
    if not a.now: OL.wait_until(9, 5, 0)
    s5 = mis.snapshot(cand.stock_id.tolist(), cand.market_type.tolist()).rename(columns={"h": "rh", "l": "rl", "u": "lu_live", "v": "o5_v"})
    watch, excl = OL.range_stage(cand, s5[["stock_id", "rh", "rl", "lu_live", "o5_v"]], eligible=elig); info["n_watch"] = len(watch)
    book = OL.Book(watch, "live", capital=a.capital)
    watch["lots"] = watch.stock_id.map(lambda s: book.s[s]["lots"])
    watch.to_csv(PAPER / f"{ds}_watch.csv", index=False, encoding="utf-8-sig")
    log(f"09:05 監看名單 {len(watch)} 檔 (每檔 {book.slot:,.0f} 元):\n" + (OL.fmt_watch(watch) if len(watch) else "(無)"))
    if len(excl): log("剔除: " + ", ".join(f"{r.stock_id}{r.name}({r.excluded})" for r in excl.itertuples()))
    po = PaperOrb(book, watch, broker, ds)
    if broker is not None:
        for sid in watch.stock_id:
            why = broker.can_short(sid)
            if why: po.skipped[sid] = why; log(f"注意 {sid}: {why}, 跌破時不下單 (Book 仍計入名額, 與訊號一致)")
        for sid in set(cand.stock_id) - set(watch.stock_id):
            try: api.unsubscribe(broker.contract(sid), quote_type=sj.QuoteType.Tick)
            except Exception: pass
    if a.dry:
        if api is not None and len(watch):
            for s in api.snapshots([broker.contract(x) for x in watch.stock_id]):
                log(f"快照 {s.code}: open={s.open} high={s.high} low={s.low} close={s.close} vol={s.total_volume}")
        log("dry run 結束 (未下單)")
        if api is not None: api.logout()
        return
    if api is None: sys.exit("缺少永豐憑證")
    holder["po"] = po; po.ready = True
    state = lambda phase, note="": dict(date=ds, phase=phase, note=note, mkt_gap=info.get("mkt_gap"), n_cand=info["n_cand"], n_watch=len(watch),
                                        slot=book.slot, positions=po.live_rows(), events=[{k: (hms(v) if isinstance(v, datetime) else v) for k, v in e.items()} for e in po.events])
    pub.live(state("監看中"), force=True)
    # ---- 盤中: tick 驅動 Book, 主執行緒下單 ----
    t_auc = at(*AUCTION_T); last_sync = last_push = time.time()
    while now() < t_auc:
        po.process(0.5); po.reconcile(); po.expire(OL.tmin(now()))
        if time.time() - last_sync >= 30: po.sync_deals(); last_sync = time.time()
        if po.dirty or time.time() - last_push >= LIVE_EVERY:
            pub.live(state("監看中"), force=po.dirty); po.dirty = False; last_push = time.time()
    # ---- 13:25 收盤競價回補 ----
    po.process(0.1); po.sync_deals()
    holding = po.close_all(limit=True)
    log(f"13:25 收盤競價回補 {len(holding)} 檔 (漲停價限價): {holding}")
    pub.live(state("等待收盤", f"收盤競價回補 {len(holding)} 檔"), force=True)
    sleep_until(at(*END_T), step=lambda d: (po.process(d), po.reconcile()))
    po.sync_deals(); po.reconcile()
    trades, legs = po.settle()
    summ = summarize(ds, info, trades, legs, len(watch), "simulation")
    write_day(ds, trades, legs, po.events, summ)
    log("結算:", json.dumps(OL._jclean(summ), ensure_ascii=False))
    if len(trades): print(trades[["stock_id", "name", "status", "rl", "sell_px", "entry_slip_ticks", "rh", "buy_px", "exit_leg", "exit_slip_ticks", "sim_net_pct", "ideal_net_pct"]].to_string(index=False))
    pub.live(state("已結算"), force=True); pub.day(ds, summ, legs)
    try: api.logout()
    except Exception: pass


# ============================== replay (離線驗證) ==============================
def cmd_replay(a):
    """用歷史分 K 合成 tick (每分鐘 開→低→高→收 或 開→高→低→收), 走完整個 下單 / 停損 / 收盤 / 結算 流程。
    市價單以下一筆合成 tick 成交, 所以會有約 1 筆的執行滑價; 用來驗證程式, 不是滑價估計。"""
    date = pd.Timestamp(a.date); ds = str(date.date())
    d1 = OL.load_d1(since=(date - pd.DateOffset(months=4)).strftime("%Y_%m"))
    m1 = OL.load_m1_month(date.strftime("%Y_%m"))
    pool, _ = OL.build_pool(d1, date)
    cand, info = OL.open_stage(pool, d1[d1.date == date][["stock_id", "open"]], date); info["n_cand"] = len(cand)
    log(f"回放 {ds}: 大盤跳空 {info['mkt_gap']*100:+.2f}%  候選 {len(cand)}  trade_ok={info['trade_ok']} {info['reasons']}")
    out = PAPER / "replay"
    if not info["trade_ok"] and not a.force:
        summ = summarize(ds, info, pd.DataFrame(), pd.DataFrame(), 0, "replay · 不交易"); write_day(ds, pd.DataFrame(), pd.DataFrame(), [], summ, out); log("今日不交易"); return
    bars = m1[(m1.date == date) & m1.stock_id.isin(cand.stock_id)].sort_values(["t", "stock_id"]).copy()
    bars[["open", "high", "low", "close"]] = bars[["open", "high", "low", "close"]].astype(float).round(2)    # 分 K 是 float32
    rng = bars[bars.t < OL.P["W"]].groupby("stock_id").agg(rh=("high", "max"), rl=("low", "min"), o5_v=("volume", "sum")).reset_index()
    watch, excl = OL.range_stage(cand, rng, lu_mode="pc"); info["n_watch"] = len(watch)
    book = OL.Book(watch, "live", capital=a.capital)
    watch["lots"] = watch.stock_id.map(lambda s: book.s[s]["lots"])
    log(f"監看 {len(watch)} 檔:\n" + (OL.fmt_watch(watch) if len(watch) else "(無)"))
    holder = {}
    broker = SimBroker({r.stock_id: OL.limit_up(r.pc) for r in watch.itertuples()}, lambda *x: holder["po"].on_deal(*x))
    po = PaperOrb(book, watch, broker, ds); holder["po"] = po; po.ready = True
    base = datetime.combine(date.date(), datetime.min.time(), TZ).replace(hour=9)
    clk = {"dt": base}; po.clock = lambda: clk["dt"]
    b = bars[(bars.t >= OL.P["W"]) & bars.stock_id.isin(watch.stock_id)]
    auction_done = False
    for r in b.itertuples():
        if r.t >= 265 and not auction_done:                     # 13:25
            clk["dt"] = base + timedelta(minutes=265, seconds=10); po.close_all(limit=True); auction_done = True
        seq = [r.open, r.low, r.high, r.close] if r.close >= r.open else [r.open, r.high, r.low, r.close]
        for k, px in enumerate(seq):
            dt = base + timedelta(minutes=int(r.t), seconds=15 * k); clk["dt"] = dt + timedelta(milliseconds=45)
            broker.feed(r.stock_id, float(px), dt)
            po.on_price(r.stock_id, float(px), dt, dt + timedelta(milliseconds=40))
            po.process(0)
        po.expire(r.t)
    if not auction_done: po.close_all(limit=True)
    po.reconcile()
    broker.final({sid: po.last.get(sid, np.nan) for sid in book.s}, base + timedelta(minutes=270))
    trades, legs = po.settle()
    summ = summarize(ds, info, trades, legs, len(watch), "replay (合成 tick, 假券商)")
    write_day(ds, trades, legs, po.events, summ, out)
    log("結算:", json.dumps(OL._jclean(summ), ensure_ascii=False))
    if len(trades): print(trades[["stock_id", "name", "status", "rl", "sell_px", "entry_slip_ticks", "rh", "buy_px", "exit_leg", "exit_slip_ticks", "sim_net_pct", "ideal_net_pct"]].to_string(index=False))
    tr_bt, _, _, _ = OL.replay_day(date, d1, m1)               # 對照: orb_live 回放 (分 K) 進場的股票應完全相同
    got = set(trades[trades.sell_px.notna()].stock_id) if len(trades) else set(); exp = set(tr_bt.stock_id) if len(tr_bt) else set()
    log(f"對照 orb_live replay: 進場 {len(exp)} 檔 vs paper {len(got)} 檔, " + ("股票完全相同 ✓" if got == exp else f"不同 ✗ 只在 orb_live {sorted(exp - got)} 只在 paper {sorted(got - exp)}"))
    log(f"輸出 {out}/{ds}_*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--capital", type=float, default=3_000_000, help="總資金, 每檔額度 = 資金/10 (同 orb_live)")
    p.add_argument("--publish", action="store_true", help="上傳 HF db/orb/paper/ (需 HF_TOKEN)")
    p.add_argument("--force", action="store_true", help="週一或大盤跳空 < -1%% 仍跑")
    p.add_argument("--no-eligibility", action="store_true", help="不查證交所/櫃買現沖券賣資格")
    p.add_argument("--dry", action="store_true", help="不下單: 算名單 + (有憑證時) 登入查合約/快照")
    p.add_argument("--now", action="store_true", help="不等 09:02 / 09:05 (測試用)")
    p.add_argument("--env-file", action="append", default=[])
    p = sp.add_parser("replay"); p.add_argument("--date", required=True); p.add_argument("--capital", type=float, default=3_000_000); p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "replay": cmd_replay}[a.cmd](a)

if __name__ == "__main__":
    main()
