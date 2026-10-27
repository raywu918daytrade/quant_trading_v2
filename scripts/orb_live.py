#!/usr/bin/env python3
"""每日選股 + 盤中訊號: D1 過熱 (跳空 ≥2% ∩ ATR20 ≥3%) × 跌破開盤 5 分區間 → 現沖券賣.
規則來源: stats/orb_dn_deepdive.md (W=5 版本, 保守回測每筆 0.42%、Sharpe 2.7)。只產生訊號, 不下單。

用法 (台灣時間):
  python3 scripts/orb_live.py refresh                 # 盤前 (前一日 16:00 後或當日 08:40 前): 從 HF 更新最近 3 個月 D1
  python3 scripts/orb_live.py run                     # 08:50 後啟動, 自動走完 盤前池 → 09:02 開盤 → 09:05 區間 → 盤中監看 → 13:31 結算
  python3 scripts/orb_live.py pool [--date D]         # 只算盤前池
  python3 scripts/orb_live.py replay --date D         # 用歷史 M1 回放某一天 (與 run 同一套邏輯)
  python3 scripts/orb_live.py verify --start A --end B  # 回放多日, 與回測 (14_orb_dn_deepdive) 逐筆對帳

規則:
  池 (只用前一日以前資料): 上市櫃 4 碼非 ETF、昨收 10~300、20 日均成交額 ≥3 億、ATR20 ≥3%。
  開盤: 今日跳空 ≥2%；非週一；大盤平均跳空 (20 日均成交額 ≥1 億的全部股票) ≥ −1%。
  09:05 記錄開盤 5 分鐘區間高 / 低與成交量；區間內已摸漲停或已到昨收 +8% 的剔除；
        開盤 5 分鐘量 < 前 20 日平均日量的 5% 的剔除 (2026-10-26 加入, 容易被來回洗, 見 stats/whipsaw.md)。
  09:05 ~ 12:00 第一次跌破區間低 → 現沖券賣 (每日最多 10 檔, 先跌破先做)。
  反彈突破區間高 → 市價回補 (停損)；否則 13:25 前掛市價參加 13:30 收盤競價回補。
"""
import argparse, glob, json, math, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, RAW, PROC, COST, HF_REPO, HF_PREFIX

TZ = ZoneInfo("Asia/Taipei")
LIVE = ROOT / "data" / "live"
SIG = ROOT / "signals"
REPO = HF_REPO
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

P = dict(LIQ_DV20=1e8, POOL_DV20=3e8, ATR_MIN=0.03, PRICE_MIN=10.0, PRICE_MAX=300.0,
         GAP_MIN=0.02, MKT_GAP_MIN=-0.01, NO_DOW={0}, W=5, DEADLINE=180, MAX_N=10, O5_VR_MIN=0.05,
         FORCE=0.08, STOP_SLIP=0.001, FORCE_SLIP=0.002, LOCK=0.07)


# ============================== 價格工具 ==============================
def tick_size(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

def limit_up(pc):
    t = tick_size(pc * 1.10)
    return math.floor(pc * 1.10 / t + 1e-9) * t

def now_tw():
    return datetime.now(TZ)

def tmin(ts: datetime):
    """台灣時間 → 09:00 起算的分鐘 (浮點)."""
    return (ts.hour * 60 + ts.minute + ts.second / 60) - 540


# ============================== 資料 ==============================
def _info():
    for p in [LIVE / "meta" / "info.parquet", RAW / "meta" / "info.parquet"]:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError("找不到 info.parquet")

def load_d1(since: str = None):
    """RAW (v1 原始月檔) + LIVE (refresh 下載的最新月檔, 同股同日以 LIVE 為準)."""
    parts = []
    for src in [RAW / "d1", LIVE / "d1"]:
        for f in sorted(glob.glob(str(src / "*.parquet"))):
            if since is None or Path(f).stem >= since:
                parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d.date)
    d = d.drop_duplicates(["stock_id", "date"], keep="last")
    info = _info()
    d = d.merge(info[["stock_id", "name", "market_type"]], on="stock_id", how="left")
    d = d[d.market_type.isin(["twse", "tpex"]) & d.stock_id.str.fullmatch(r"\d{4}") & ~d.stock_id.str.startswith("00")]
    return d.sort_values(["stock_id", "date"]).reset_index(drop=True)

def m1_path(ym):
    """分 K 月檔: LIVE (refresh 下載, 較新) 優先, 否則 RAW."""
    lp, rp = LIVE / "m1" / f"{ym}.parquet", RAW / "m1" / f"{ym}.parquet"
    return lp if lp.exists() else rp

def refresh(months=3, m1_months=1):
    from huggingface_hub import hf_hub_download
    (LIVE / "d1").mkdir(parents=True, exist_ok=True); (LIVE / "meta").mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(now_tw().date())
    for i in range(months):
        ym = (today - pd.DateOffset(months=i)).strftime("%Y_%m")
        try:
            p = hf_hub_download(REPO, f"db/d1/{ym}.parquet", repo_type="dataset", force_download=True)
            (LIVE / "d1" / f"{ym}.parquet").write_bytes(Path(p).read_bytes())
            print(f"更新 D1 {ym}: 最後日期 {pd.read_parquet(LIVE / 'd1' / f'{ym}.parquet').date.max()}")
        except Exception as e:
            print(f"D1 {ym} 下載失敗: {e}")
    p = hf_hub_download(REPO, "db/info/info.parquet", repo_type="dataset", force_download=True)
    (LIVE / "meta" / "info.parquet").write_bytes(Path(p).read_bytes())
    print("更新 info.parquet")
    (LIVE / "m1").mkdir(parents=True, exist_ok=True)
    for i in range(m1_months):
        ym = (today - pd.DateOffset(months=i)).strftime("%Y_%m")
        try:
            p = hf_hub_download(REPO, f"db/m1/{ym}.parquet", repo_type="dataset", force_download=True)
            (LIVE / "m1" / f"{ym}.parquet").write_bytes(Path(p).read_bytes())
            print(f"更新 M1 {ym}: 最後時間 {pd.read_parquet(LIVE / 'm1' / f'{ym}.parquet', columns=['date']).date.max()}")
        except Exception as e:
            print(f"M1 {ym} 下載失敗: {e}")


# ============================== 盤前池 ==============================
def build_pool(d1, date):
    """date 當天的池 (只用 date 之前的 D1). 回傳所有流動股 (供大盤跳空) 並標記 in_pool."""
    date = pd.Timestamp(date).normalize()
    h = d1[d1.date < date].copy()
    g = h.groupby("stock_id", sort=False)
    h["pcl"] = g.close.shift(1)
    h["rng"] = (h.high - h.low) / h.pcl
    h["dvv"] = h.close * h.volume
    t = h.groupby("stock_id", sort=False).tail(20)
    a = t.groupby("stock_id", sort=False).agg(n=("dvv", "count"), rng_n=("rng", "count"), atr20=("rng", "mean"), dv20=("dvv", "mean"), vol20=("volume", "mean"),
                                               pc=("close", "last"), last_date=("date", "last"), name=("name", "last"), market_type=("market_type", "last")).reset_index()
    a = a[(a.n == 20) & (a.rng_n == 20) & a.pc.notna()]
    a = a[a.dv20 >= P["LIQ_DV20"]].copy()
    a["in_pool"] = (a.dv20 >= P["POOL_DV20"]) & (a.atr20 >= P["ATR_MIN"]) & (a.pc >= P["PRICE_MIN"]) & (a.pc <= P["PRICE_MAX"])
    a["limit_up"] = a.pc.map(limit_up)
    a["force_px"] = a.pc * (1 + P["FORCE"])
    return a.reset_index(drop=True), h.date.max()


# ============================== 開盤 / 區間 ==============================
def open_stage(pool, opens, date):
    x = pool.merge(opens[["stock_id", "open"]], on="stock_id", how="left")
    x["gap"] = x.open / x.pc - 1
    mkt_gap = x.gap.mean()
    breadth = (x.gap > 0).mean()
    cand = x[x.in_pool & (x.gap >= P["GAP_MIN"])].copy()
    dow = pd.Timestamp(date).dayofweek
    reasons = []
    if dow in P["NO_DOW"]: reasons.append("週一不交易")
    if not (mkt_gap >= P["MKT_GAP_MIN"]): reasons.append(f"大盤平均跳空 {mkt_gap*100:.2f}% < {P['MKT_GAP_MIN']*100:.0f}%")
    return cand.sort_values("gap", ascending=False).reset_index(drop=True), dict(mkt_gap=mkt_gap, breadth=breadth, n_open=int(x.open.notna().sum()), n_liq=len(x), trade_ok=not reasons, reasons=reasons)

def range_stage(cand, rng, lu_mode="tick", eligible=None):
    """rng: stock_id, rh, rl (+ lu_live). 回傳 (watch, excluded)."""
    c = cand.merge(rng, on="stock_id", how="left")
    if lu_mode == "pc":                       # 回放對帳: 與回測一致用 昨收×1.1
        c["lu"] = c.pc * 1.10
    else:
        c["lu"] = c["lu_live"].fillna(c.limit_up) if "lu_live" in c.columns else c.limit_up
    why = pd.Series("", index=c.index)
    why[c.rh.isna() | c.rl.isna()] = "開盤 5 分無成交"
    why[(why == "") & (c.rh >= c.lu * 0.999)] = "開盤 5 分內摸漲停"
    why[(why == "") & (c.rh >= c.force_px)] = "開盤 5 分內已到昨收 +8%"
    if "o5_v" in c.columns:                                                   # 張 → 股, 對前 20 日均量 (股)
        c["o5_vr"] = c.o5_v * 1000 / c.vol20
        why[(why == "") & (c.o5_vr < P["O5_VR_MIN"])] = f"開盤 5 分量 < 日均量 {P['O5_VR_MIN']*100:.0f}%"
    if eligible is not None:
        why[(why == "") & ~c.stock_id.map(lambda s: eligible.get(s) is True)] = "不可現沖券賣"
    c["excluded"] = why
    return c[why == ""].reset_index(drop=True), c[why != ""].reset_index(drop=True)


# ============================== 盤中狀態機 ==============================
class Book:
    """每檔: WAIT → SHORT → (STOP 停損 | CLOSE 收盤回補). WAIT 在 12:00 後或額滿時失效.
    step() 的輸入是一段期間 (回放 = 1 分 K; 即時 = 兩次輪詢之間) 的 最低 lo / 最高 hi / 第一價 first / 最後價 last."""

    def __init__(self, watch, mode, capital=3_000_000, max_n=None, log=None):
        self.mode = mode; self.max_n = max_n or P["MAX_N"]; self.n_entered = 0; self.log = log
        self.slot = capital / self.max_n
        self.s = {}
        for r in watch.itertuples():
            self.s[r.stock_id] = dict(sid=r.stock_id, name=r.name, pc=r.pc, gap=r.gap, rh=r.rh, rl=r.rl, fpx=r.force_px, lu=r.lu, state="WAIT",
                                      entry_t=np.nan, entry_px=np.nan, exit_t=np.nan, exit_px=np.nan, reason="", forced=False, locked=False,
                                      lots=max(int(self.slot // (r.rl * 1000)), 1))   # 股價 >300 時 30 萬不足 1 張, 至少做 1 張
        self.events = []

    def _ev(self, t, sid, ev, px, ref, note=""):
        p = self.s[sid]
        hhmm = f"{9 + int((t + 540 - 540) // 60):02d}:{int(t) % 60:02d}" if self.mode == "bar" else now_tw().strftime("%H:%M:%S")
        e = dict(time=hhmm, t=round(t, 2), stock_id=sid, name=p["name"], event=ev, price=px, ref=ref, note=note)
        self.events.append(e)
        if self.log: self.log(e)

    def active(self):
        return [k for k, v in self.s.items() if v["state"] in ("WAIT", "SHORT")]

    def step(self, t, sid, lo, hi, first, last):
        p = self.s[sid]
        if p["state"] == "WAIT":
            if t >= P["DEADLINE"] + 1:
                p["state"] = "EXPIRED"; return
            if lo < p["rl"]:
                if self.n_entered >= self.max_n:
                    p["state"] = "FULL"; self._ev(t, sid, "FULL", last, p["rl"], "跌破但當日已滿 10 檔"); return
                p["state"] = "SHORT"; self.n_entered += 1
                p["entry_t"] = t
                p["entry_px"] = min(p["rl"], first) if self.mode == "bar" else (last if np.isfinite(last) else p["rl"])
                self._ev(t, sid, "SHORT", p["entry_px"], p["rl"], f"跌破區間低 {p['rl']:g} → 現沖券賣 {p['lots']} 張; 停損 {p['rh']:g}")
            return                                                  # 進場那一段不檢查停損 (與回測一致)
        if p["state"] == "SHORT" and hi > p["rh"]:
            px = max(p["rh"], first) * (1 + P["STOP_SLIP"]) if self.mode == "bar" else (last if np.isfinite(last) else p["rh"])
            if self.mode == "bar" and px >= p["fpx"]:
                px = p["fpx"] * (1 + P["FORCE_SLIP"]); p["forced"] = True
            p["state"] = "STOPPED"; p["exit_t"] = t; p["exit_px"] = px; p["reason"] = "停損 (突破區間高)"
            self._ev(t, sid, "STOP", px, p["rh"], f"突破區間高 {p['rh']:g} → 市價回補")

    def close_all(self, closes: dict, t=270):
        for sid, p in self.s.items():
            if p["state"] == "SHORT":
                c = closes.get(sid, np.nan)
                p["state"] = "CLOSED"; p["exit_t"] = t; p["exit_px"] = c; p["reason"] = "收盤回補"
                p["locked"] = bool(np.isfinite(c) and c >= p["lu"] * 0.999)
                self._ev(t, sid, "CLOSE", c, np.nan, "收盤競價回補" + (" (鎖漲停!)" if p["locked"] else ""))

    def trades(self):
        rows = []
        for p in self.s.values():
            if not np.isfinite(p["entry_px"]): continue
            r = -(p["exit_px"] / p["entry_px"] - 1) - COST - (P["LOCK"] if p["locked"] else 0)
            rows.append(dict(stock_id=p["sid"], name=p["name"], gap=p["gap"], rl=p["rl"], rh=p["rh"], entry_t=p["entry_t"], entry_px=p["entry_px"],
                             exit_t=p["exit_t"], exit_px=p["exit_px"], reason=p["reason"], forced=p["forced"], locked=p["locked"], lots=p["lots"],
                             r_pct=r * 100, pnl=r * p["entry_px"] * p["lots"] * 1000))      # 元; 含成本與鎖漲停罰金
        return pd.DataFrame(rows)


# ============================== 即時行情 (證交所 MIS) ==============================
def _f(v):
    try:
        x = float(str(v).split("_")[0])
        return x if x > 0 else np.nan
    except Exception:
        return np.nan

class MIS:
    def __init__(self):
        import requests
        self.s = requests.Session(); self.s.headers.update(UA)
        try: self.s.get("https://mis.twse.com.tw/stock/index.jsp", timeout=10)
        except Exception as e: print("MIS 首頁連線失敗:", e)

    def snapshot(self, sids, markets, batch=40, sleep=0.3):
        codes = [f"{'tse' if m == 'twse' else 'otc'}_{s}.tw" for s, m in zip(sids, markets)]
        out = []
        for i in range(0, len(codes), batch):
            url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=" + "|".join(codes[i:i + batch]) + f"&json=1&delay=0&_={int(time.time() * 1000)}"
            js = {"msgArray": []}
            for k in range(3):
                try: js = self.s.get(url, timeout=10).json(); break
                except Exception as e:
                    if k == 2: print("MIS 失敗:", e)
                    time.sleep(1)
            for m in js.get("msgArray", []):
                out.append(dict(stock_id=m.get("c"), o=_f(m.get("o")), h=_f(m.get("h")), l=_f(m.get("l")), z=_f(m.get("z")), y=_f(m.get("y")),
                                u=_f(m.get("u")), w=_f(m.get("w")), v=_f(m.get("v")), mis_t=m.get("t"), mis_d=m.get("d")))
            if i + batch < len(codes): time.sleep(sleep)
        df = pd.DataFrame(out, columns=["stock_id", "o", "h", "l", "z", "y", "u", "w", "v", "mis_t", "mis_d"])
        return df[df.stock_id.notna() & (df.stock_id != "")].reset_index(drop=True)


# ============================== 當沖資格 ==============================
def fetch_eligible(date):
    """{stock_id: True=可現沖券賣, False=暫停先賣}; 不在名單 = 不可當沖. 失敗回 None."""
    import requests
    ymd = str(date)[:10].replace("-", ""); (LIVE / "meta").mkdir(parents=True, exist_ok=True)
    cache = LIVE / "meta" / f"daytrade_{ymd}.json"
    if cache.exists(): return json.load(open(cache))
    try:
        out = {}
        url = "https://www.twse.com.tw/exchangeReport/TWTB4U?response=json" + ("" if ymd == now_tw().strftime("%Y%m%d") else f"&date={ymd}")
        d = requests.get(url, headers=UA, timeout=20).json()
        for t in d.get("tables", []):
            if (t.get("fields") or [""])[0] == "證券代號":
                for row in t.get("data", []): out[str(row[0]).strip()] = str(row[2]).strip() == ""
        d = requests.get(f"https://www.tpex.org.tw/www/zh-tw/intraday/list?date={ymd[:4]}/{ymd[4:6]}/{ymd[6:]}&response=json", headers=UA, timeout=20).json()
        for t in d.get("tables", []):
            for row in t.get("data", []): out[str(row[0]).strip()] = (str(row[2]).strip() if len(row) > 2 else "") == ""
        if len(out) < 500: raise RuntimeError(f"名單只有 {len(out)} 檔")
        json.dump(out, open(cache, "w"), ensure_ascii=False); return out
    except Exception as e:
        print(f"⚠ 當沖資格名單取得失敗 ({e}), 本日不過濾資格"); return None


# ============================== 發布到 HF (雲端網頁讀) ==============================
class Publisher:
    """把盤中狀態寫到 signals/live.json, 並 (有 HF_TOKEN 時) 上傳到 HF db/orb/live.json。
    有事件立即上傳; 否則至少間隔 min_gap 秒。token 只從環境變數讀, 不印出。"""
    def __init__(self, enabled, min_gap=180):
        import os
        self.enabled = enabled; self.min_gap = min_gap; self.last = 0.0; self.api = None
        if enabled:
            tok = os.environ.get("HF_TOKEN")
            if not tok: print("⚠ --publish 但沒有 HF_TOKEN, 只寫本機 signals/live.json"); self.enabled = False
            else:
                from huggingface_hub import HfApi
                self.api = HfApi(token=tok)
    def push(self, state, force=False):
        state = {**state, "updated_at": now_tw().strftime("%Y-%m-%d %H:%M:%S")}
        SIG.mkdir(exist_ok=True); p = SIG / "live.json"
        p.write_text(json.dumps(_jclean(state), ensure_ascii=False, default=_jdefault, allow_nan=False), encoding="utf-8")
        if not self.enabled or (not force and time.time() - self.last < self.min_gap): return
        for k in range(3):
            try:
                self.api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{HF_PREFIX}/live.json", repo_id=HF_REPO, repo_type="dataset",
                                     commit_message=f"orb live {state.get('date')} {state.get('phase')}")
                self.last = time.time(); return
            except Exception as e:
                print(f"HF 上傳失敗 ({k + 1}/3): {e}"); time.sleep(5)
    def upload_day(self, ds):
        if not self.enabled: return
        files = [f for f in SIG.glob(f"{ds}_*") if f.is_file()]
        for f in files:
            try: self.api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{HF_PREFIX}/live/{ds}/{f.name}", repo_id=HF_REPO, repo_type="dataset", commit_message=f"orb day {ds} {f.name}")
            except Exception as e: print("HF 上傳失敗:", f.name, e)

def _jclean(o):
    """遞迴把 NaN / inf 轉成 None、numpy 型別轉成 Python 型別 (標準 JSON 不接受 NaN)."""
    if isinstance(o, dict): return {k: _jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jclean(v) for v in o]
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, pd.Timestamp): return str(o)
    return o

def _jdefault(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return str(o)
    return str(o)

STATE_TXT = {"WAIT": "等待跌破", "SHORT": "已放空", "STOPPED": "已停損", "CLOSED": "收盤回補", "EXPIRED": "未跌破 (12:00 失效)", "FULL": "跌破但額滿"}
def live_state(ds, phase, info=None, book=None, mem=None, excl=None, cand=None, note=""):
    st = {"date": ds, "phase": phase, "note": note, "rule": "跳空≥2% ∩ ATR20≥3% ∩ 開盤5分量≥日均量5%; 跌破開盤5分區間低放空, 區間高停損, 否則收盤回補"}
    if info: st.update({k: info.get(k) for k in ["mkt_gap", "breadth", "n_cand", "n_watch", "trade_ok", "reasons", "n_open", "n_liq"]})
    if book is not None:
        rows = []
        for sid, p in book.s.items():
            last = (mem or {}).get(sid, {}).get("z", np.nan)
            ref = p["exit_px"] if np.isfinite(p["exit_px"]) else last
            pnl = (-(ref / p["entry_px"] - 1) - COST) * 100 if np.isfinite(p["entry_px"]) and np.isfinite(ref) else None
            rows.append(dict(stock_id=sid, name=p["name"], pc=p["pc"], gap=p["gap"], rl=p["rl"], rh=p["rh"], force_px=p["fpx"], lots=p["lots"], state=p["state"], state_txt=STATE_TXT.get(p["state"], p["state"]),
                             entry_px=p["entry_px"], entry_t=p["entry_t"], exit_px=p["exit_px"], exit_t=p["exit_t"], last=last, pnl_pct=pnl))
        st["watch"] = rows; st["events"] = book.events; st["slot"] = book.slot
        tr = book.trades()
        if len(tr): st["realized"] = dict(n=len(tr), avg_pct=float(tr.r_pct.mean()), pnl=float(tr.pnl.sum()))
    if excl is not None and len(excl): st["excluded"] = [dict(stock_id=r.stock_id, name=r.name, gap=r.gap, reason=r.excluded) for r in excl.itertuples()]
    if cand is not None and len(cand) and book is None: st["candidates"] = [dict(stock_id=r.stock_id, name=r.name, gap=r.gap, pc=r.pc) for r in cand.itertuples()]
    return st


# ============================== 輸出 ==============================
def fmt_watch(w):
    cols = ["stock_id", "name", "pc", "open", "gap", "atr20", "rl", "rh", "force_px", "lots"]
    x = w.copy(); x["gap"] = (x.gap * 100).round(2); x["atr20"] = (x.atr20 * 100).round(1)
    if "lots" not in x.columns: x["lots"] = ""
    return x[[c for c in cols if c in x.columns]].rename(columns={"stock_id": "代號", "name": "名稱", "pc": "昨收", "open": "開盤", "gap": "跳空%", "atr20": "ATR%",
                                                                  "rl": "區間低(進場)", "rh": "區間高(停損)", "force_px": "+8%", "lots": "張"}).to_string(index=False)

def summarize(tr, date, out_dir, info):
    lines = [f"===== {date} 結算 =====", f"大盤平均跳空 {info.get('mkt_gap', np.nan)*100:.2f}%  候選 {info.get('n_cand', 0)}  監看 {info.get('n_watch', 0)}  進場 {len(tr)}"]
    if len(tr):
        lines.append(tr.assign(r_pct=tr.r_pct.round(2), pnl=tr.pnl.round(0))[["stock_id", "name", "entry_px", "exit_px", "reason", "lots", "r_pct", "pnl"]].to_string(index=False))
        lines.append(f"平均每筆 {tr.r_pct.mean():+.2f}%  勝率 {(tr.r_pct > 0).mean()*100:.0f}%  合計損益 {tr.pnl.sum():+,.0f} 元")
    txt = "\n".join(lines); print(txt)
    (out_dir / f"{date}_summary.txt").write_text(txt, encoding="utf-8")
    tr.to_csv(out_dir / f"{date}_trades.csv", index=False)


# ============================== 回放 ==============================
def load_m1_month(ym):
    m = pd.read_parquet(m1_path(ym))
    m["ts"] = pd.to_datetime(m.date); m["date"] = m.ts.dt.normalize()
    m["t"] = (m.ts.dt.hour * 60 + m.ts.dt.minute - 540).astype(int)
    return m[(m.t >= 0) & (m.t <= 270)].drop(columns=["ts"])

def replay_day(date, d1, m1, eligible=None, verbose=False):
    date = pd.Timestamp(date)
    pool, last = build_pool(d1, date)
    opens = d1[d1.date == date][["stock_id", "open"]]
    cand, info = open_stage(pool, opens, date)
    info["n_cand"] = len(cand)
    if not info["trade_ok"]:
        return pd.DataFrame(), info, pd.DataFrame(), pd.DataFrame()
    bars = m1[(m1.date == date) & m1.stock_id.isin(cand.stock_id)].sort_values(["t", "stock_id"])
    rng = bars[bars.t < P["W"]].groupby("stock_id").agg(rh=("high", "max"), rl=("low", "min"), o5_v=("volume", "sum")).reset_index()
    watch, excl = range_stage(cand, rng, lu_mode="pc", eligible=eligible)
    info["n_watch"] = len(watch)
    book = Book(watch, "bar", log=(lambda e: print(f"  {e['time']} {e['stock_id']} {e['name']} {e['event']} {e['price']:.2f}  {e['note']}")) if verbose else None)
    b = bars[(bars.t >= P["W"]) & bars.stock_id.isin(watch.stock_id)]
    for r in b.itertuples():
        if book.s[r.stock_id]["state"] in ("WAIT", "SHORT"):
            book.step(r.t, r.stock_id, r.low, r.high, r.open, r.close)
    book.close_all(bars.groupby("stock_id").close.last().to_dict())
    info["states"] = pd.DataFrame(list(book.s.values()))           # 每檔最終狀態 (CLOSED/STOPPED/WAIT/EXPIRED/FULL), 供歷史網頁
    return book.trades(), info, watch, excl


def expected_trades(start, end):
    """回測 (14_orb_dn_deepdive: W=5, 跳空≥2% ∩ ATR≥3%, 停損區間高, 可執行濾網, 開盤 5 分量 ≥ 日均量 5%, 每日前 10 檔) 的逐筆結果, 供對帳."""
    f = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ", "pc", "gap", "atr20", "dv20", "vol20", "mkt_gap", "dow"])
    f = f[f.univ & (f.date >= start) & (f.date <= end)]
    yms = sorted({d.strftime("%Y_%m") for d in pd.to_datetime(f.date.unique())})
    cols = ["stock_id", "date", "auc_c", "o5_h", "o5_l", "o5_v", "dn5_t", "dn5_px", "dn5_hi_after", "dn5_stop_hi_px", "dn5_stop_hi_open"]
    orb = pd.concat([pd.read_parquet(PROC / "orb_dn" / f"{ym}.parquet", columns=cols) for ym in yms], ignore_index=True)
    s = f.merge(orb, on=["stock_id", "date"]); s = s[s.auc_c.notna()]
    lu = s.pc * 1.1; fpx = s.pc * 1.08
    s = s[(s.gap >= 0.02) & (s.atr20 >= 0.03) & (s.dow != 0) & (s.mkt_gap >= -0.01) & (s.dv20 >= 3e8) & (s.pc <= 300) & s.dn5_t.notna() & (s.dn5_t <= 180)
          & (s.o5_h < lu * 0.999) & (s.o5_h < fpx) & (s.dn5_px < fpx) & (s.o5_v * 1000 / s.vol20 >= P["O5_VR_MIN"])].copy()
    lu = s.pc * 1.1; fpx = s.pc * 1.08
    st = np.maximum(s.dn5_stop_hi_px, s.dn5_stop_hi_open) * (1 + P["STOP_SLIP"])
    ex = st.where(st.notna(), s.auc_c)
    hi = s.dn5_hi_after.where(st.isna(), np.minimum(s.dn5_hi_after, st))
    forced = hi >= fpx; exf = np.where(forced, fpx * (1 + P["FORCE_SLIP"]), ex); locked = (~forced) & (exf >= lu * 0.999)
    s["r_pct"] = (-(exf / s.dn5_px - 1) - COST - np.where(locked, P["LOCK"], 0)) * 100
    s = s.sort_values(["date", "dn5_t", "stock_id"])
    return s.groupby("date").head(P["MAX_N"])[["date", "stock_id", "dn5_t", "dn5_px", "r_pct"]]


# ============================== 指令 ==============================
def cmd_pool(a):
    date = pd.Timestamp(a.date or now_tw().date())
    d1 = load_d1(since=(date - pd.DateOffset(months=4)).strftime("%Y_%m"))
    pool, last = build_pool(d1, date)
    SIG.mkdir(exist_ok=True); pool.to_csv(SIG / f"{date.date()}_pool.csv", index=False)
    p = pool[pool.in_pool]
    print(f"{date.date()} 盤前池: D1 最後日期 {last.date()}；流動股 {len(pool)} 檔 (算大盤跳空用)；池內 {len(p)} 檔 (ATR≥3%、成交額≥3億、股價 10~300)")
    if (date - last).days > 4: print(f"⚠ D1 最後日期 {last.date()} 距今 {(date - last).days} 天, 請先執行 refresh")
    return pool, last

def wait_until(hh, mm, ss=0):
    target = now_tw().replace(hour=hh, minute=mm, second=ss, microsecond=0)
    while (dt := (target - now_tw()).total_seconds()) > 0:
        print(f"\r等待到 {hh:02d}:{mm:02d}:{ss:02d} (還有 {int(dt)} 秒)   ", end="", flush=True); time.sleep(min(dt, 15))
    print()

def notify(title, msg):
    try: subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}" sound name "Glass"'], timeout=5)
    except Exception: pass

def cmd_run(a):
    today = pd.Timestamp(now_tw().date()); ds = str(today.date())
    SIG.mkdir(exist_ok=True)
    pub = Publisher(a.publish)
    pool, last = cmd_pool(argparse.Namespace(date=None))
    pub.push(live_state(ds, "盤前", note=f"盤前池 {int(pool.in_pool.sum())} 檔 (D1 最後 {last.date()})"), force=True)
    if today.dayofweek in P["NO_DOW"] and not a.force:
        print("今天週一, 規則不交易 (加 --force 仍可跑完, 僅供觀察)。")
        pub.push(live_state(ds, "不交易", note="週一不交易"), force=True); return
    elig = None if a.no_eligibility else fetch_eligible(ds)
    mis = MIS()
    # ---- 09:02 開盤價 / 大盤跳空 ----
    if not a.now: wait_until(9, 2, 0)
    snap = mis.snapshot(pool.stock_id.tolist(), pool.market_type.tolist())
    mis_day = snap.mis_d.dropna().mode()
    if len(mis_day) == 0 or mis_day.iloc[0] != today.strftime("%Y%m%d"):
        msg = f"MIS 回傳日期 {mis_day.iloc[0] if len(mis_day) else '無'} ≠ 今天, 可能非交易日或尚未開盤"
        print(msg + ", 結束。"); pub.push(live_state(ds, "非交易日", note=msg), force=True); sys.exit(0 if a.publish else 2)
    cand, info = open_stage(pool, snap.rename(columns={"o": "open"}), today)
    info["n_cand"] = len(cand)
    print(f"09:02 開盤: 有開盤價 {info['n_open']}/{info['n_liq']}；大盤平均跳空 {info['mkt_gap']*100:+.2f}%  上漲家數比 {info['breadth']*100:.0f}%；跳空≥2% 的池內候選 {len(cand)} 檔")
    if not info["trade_ok"]:
        print("今日不交易: " + "；".join(info["reasons"]))
        if not a.force:
            cand.to_csv(SIG / f"{ds}_cand.csv", index=False)
            pub.push(live_state(ds, "不交易", info, cand=cand, note="；".join(info["reasons"])), force=True); pub.upload_day(ds); return
    pub.push(live_state(ds, "等待 09:05", info, cand=cand), force=True)
    # ---- 09:05 區間 ----
    if not a.now: wait_until(9, 5, 0)
    s5 = mis.snapshot(cand.stock_id.tolist(), cand.market_type.tolist()).rename(columns={"h": "rh", "l": "rl", "u": "lu_live", "v": "o5_v"})
    watch, excl = range_stage(cand, s5[["stock_id", "rh", "rl", "lu_live", "o5_v"]], eligible=elig)
    info["n_watch"] = len(watch)
    book = Book(watch, "live", capital=a.capital, log=None)
    watch["lots"] = watch.stock_id.map(lambda s: book.s[s]["lots"])
    watch.to_csv(SIG / f"{ds}_watch.csv", index=False); excl.to_csv(SIG / f"{ds}_excluded.csv", index=False)
    print(f"\n09:05 監看名單 {len(watch)} 檔 (每檔額度 {book.slot:,.0f} 元):\n" + (fmt_watch(watch) if len(watch) else "(無)"))
    if len(excl): print("剔除: " + ", ".join(f"{r.stock_id}{r.name}({r.excluded})" for r in excl.itertuples()))
    mem = {r.stock_id: dict(h=r.rh, l=r.rl, z=np.nan) for r in watch.itertuples()}
    pub.push(live_state(ds, "監看中", info, book, mem, excl), force=True)
    evf = open(SIG / f"{ds}_events.csv", "a", encoding="utf-8")
    new_ev = []
    def log(e):
        print(f"🔔 {e['time']} {e['stock_id']} {e['name']} {e['event']} @ {e['price']:.2f}  {e['note']}")
        evf.write(",".join(str(e[k]) for k in ["time", "stock_id", "name", "event", "price", "ref", "note"]) + "\n"); evf.flush()
        new_ev.append(e)
        if a.notify and e["event"] in ("SHORT", "STOP"): notify(f"{e['event']} {e['stock_id']} {e['name']}", f"{e['price']:.2f} {e['note']}")
    book.log = log
    mk = dict(zip(watch.stock_id, watch.market_type))
    reminded = False
    # ---- 盤中監看 ----
    while book.active() and now_tw().hour * 60 + now_tw().minute < 13 * 60 + 30:
        loop_start = time.time(); t = tmin(now_tw())
        ids = book.active()
        snap = mis.snapshot(ids, [mk[s] for s in ids])
        for r in snap.itertuples():
            m = mem[r.stock_id]
            hi = np.nanmax([r.z, r.h if r.h > m["h"] else np.nan]); lo = np.nanmin([r.z, r.l if r.l < m["l"] else np.nan])
            last = r.z if np.isfinite(r.z) else m["z"]
            if np.isfinite(hi) or np.isfinite(lo):
                book.step(t, r.stock_id, lo if np.isfinite(lo) else np.inf, hi if np.isfinite(hi) else -np.inf, last, last)
            m["h"] = np.nanmax([m["h"], r.h]); m["l"] = np.nanmin([m["l"], r.l]); m["z"] = last
        for sid in ids:                                     # 12:00 後 WAIT 失效
            if book.s[sid]["state"] == "WAIT" and t >= P["DEADLINE"] + 1: book.s[sid]["state"] = "EXPIRED"
        holding = [s for s, p in book.s.items() if p["state"] == "SHORT"]
        if not reminded and t >= 264 and holding:           # 13:24
            print(f"⏰ 13:24 提醒: 持有 {len(holding)} 檔 {holding}, 13:25 前掛市價買回參加收盤競價"); reminded = True
            if a.notify: notify("收盤回補提醒", f"{len(holding)} 檔: {' '.join(holding)}")
            pub.push(live_state(ds, "收盤回補提醒", info, book, mem, excl, note=f"13:25 前掛市價買回 {len(holding)} 檔"), force=True)
        pub.push(live_state(ds, "監看中", info, book, mem, excl), force=bool(new_ev)); new_ev.clear()
        if not book.active() or (not holding and t >= P["DEADLINE"] + 1): break
        time.sleep(max(a.poll - (time.time() - loop_start), 1))
    # ---- 收盤結算 ----
    holding = [s for s, p in book.s.items() if p["state"] == "SHORT"]
    if holding:
        pub.push(live_state(ds, "等待收盤", info, book, mem, excl), force=True)
        wait_until(13, 31, 0)
        fin = mis.snapshot(holding, [mk[s] for s in holding])
        book.close_all({r.stock_id: (r.z if np.isfinite(r.z) else mem[r.stock_id]["z"]) for r in fin.itertuples()})
    evf.close()
    summarize(book.trades(), ds, SIG, info)
    pub.push(live_state(ds, "已結算", info, book, mem, excl), force=True); pub.upload_day(ds)

def cmd_replay(a):
    date = pd.Timestamp(a.date)
    d1 = load_d1(since=(date - pd.DateOffset(months=4)).strftime("%Y_%m"))
    m1 = load_m1_month(date.strftime("%Y_%m"))
    elig = fetch_eligible(str(date.date())) if a.eligibility else None
    print(f"===== 回放 {date.date()} =====")
    tr, info, watch, excl = replay_day(date, d1, m1, eligible=elig, verbose=True)
    if not info["trade_ok"]:
        print("今日不交易: " + "；".join(info["reasons"]), f"(大盤跳空 {info['mkt_gap']*100:+.2f}%)"); return
    print(f"大盤跳空 {info['mkt_gap']*100:+.2f}%  候選 {info['n_cand']}  監看 {info['n_watch']}")
    out = SIG / "replay"; out.mkdir(parents=True, exist_ok=True)
    watch.to_csv(out / f"{date.date()}_watch.csv", index=False)
    summarize(tr, str(date.date()), out, info)

def cmd_verify(a):
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    exp = expected_trades(start, end)
    d1 = load_d1(since=(start - pd.DateOffset(months=4)).strftime("%Y_%m"))
    days = sorted(d for d in d1.date.unique() if start <= d <= end)
    got = []; t0 = time.time()
    for ym, grp in pd.Series(days).groupby(pd.Series(days).dt.strftime("%Y_%m")):
        m1 = load_m1_month(ym)
        for d in grp:
            tr, info, _, _ = replay_day(d, d1, m1)
            if len(tr): tr["date"] = d; got.append(tr)
        print(f"{ym} 完成 ({time.time() - t0:.0f}s)", flush=True)
    got = pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=["date", "stock_id", "r_pct"])
    m = exp.merge(got[["date", "stock_id", "entry_t", "entry_px", "r_pct"]], on=["date", "stock_id"], how="outer", suffixes=("_bt", "_live"), indicator=True)
    both = m[m._merge == "both"]
    diff = (both.r_pct_bt - both.r_pct_live).abs()
    lines = [f"# 對帳 {start.date()} ~ {end.date()}", "",
             f"- 回測筆數 {len(exp)}, 回放筆數 {len(got)}, 兩邊都有 {len(both)}, 只在回測 {int((m._merge == 'left_only').sum())}, 只在回放 {int((m._merge == 'right_only').sum())}",
             f"- 兩邊都有的筆: 每筆報酬差 |Δ| 平均 {diff.mean():.4f} 個百分點, 最大 {diff.max():.4f}；差 >0.01 的 {int((diff > 0.01).sum())} 筆",
             f"- 平均每筆: 回測 {exp.r_pct.mean():.3f}%  回放 {got.r_pct.mean():.3f}%"]
    print("\n".join(lines))
    bad = m[(m._merge != "both") | ((m.r_pct_bt - m.r_pct_live).abs() > 0.01)]
    if len(bad): print(bad.head(20).to_string(index=False))
    (SIG / "replay").mkdir(parents=True, exist_ok=True)
    m.to_csv(SIG / "replay" / f"verify_{start.date()}_{end.date()}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("refresh"); p.add_argument("--months", type=int, default=3); p.add_argument("--m1-months", type=int, default=1)
    p = sp.add_parser("pool"); p.add_argument("--date")
    p = sp.add_parser("run")
    p.add_argument("--capital", type=float, default=3_000_000, help="總資金, 每檔額度 = 資金/10")
    p.add_argument("--poll", type=float, default=10, help="輪詢秒數")
    p.add_argument("--notify", action="store_true", help="macOS 通知 (進場/停損/收盤提醒)")
    p.add_argument("--no-eligibility", action="store_true", help="不查現沖券賣資格")
    p.add_argument("--force", action="store_true", help="週一或大盤跳空 < -1%% 仍跑 (觀察用)")
    p.add_argument("--now", action="store_true", help="不等 09:02 / 09:05 (測試用)")
    p.add_argument("--publish", action="store_true", help="把盤中狀態上傳到 HF db/orb/live.json (需環境變數 HF_TOKEN)")
    p = sp.add_parser("replay"); p.add_argument("--date", required=True); p.add_argument("--eligibility", action="store_true")
    p = sp.add_parser("verify"); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    a = ap.parse_args()
    {"refresh": lambda a: refresh(a.months, a.m1_months), "pool": cmd_pool, "run": cmd_run, "replay": cmd_replay, "verify": cmd_verify}[a.cmd](a)

if __name__ == "__main__":
    main()
