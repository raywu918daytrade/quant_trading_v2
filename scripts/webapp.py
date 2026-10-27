#!/usr/bin/env python3
"""歷史交易瀏覽網頁 (本機). 用法: python3 scripts/webapp.py [--port 8790] [--live-dir signals]  → 開 http://127.0.0.1:8790
資料: data/processed/orb_history/ (先跑 scripts/build_history.py)；K 線即時讀 RAW M1 / D1。只綁 127.0.0.1。
實盤紀錄分頁: 每次請求即時讀 --live-dir (預設 signals/, 即 orb_live.py run 的輸出) 與回測對齊 (export_live.py)。"""
import argparse, json, math, sys, threading
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_live as L
import export_live as XL
from config import ROOT, RAW, PROC

HIST = PROC / "orb_history"; WEB = ROOT / "web"; LIVE_DIR = ROOT / "signals"
print("載入歷史 ...", flush=True)
DAYS = pd.read_parquet(HIST / "days.parquet"); TR = pd.read_parquet(HIST / "trades.parquet"); WT = pd.read_parquet(HIST / "watch.parquet")
CAP = 300.0                                                   # 萬; 整數張、固定 300 萬不複利 (與 plot_equity.py 預設相同)
DAYS["pnl_w"] = DAYS.pnl / 1e4
DAYS["ret_int"] = DAYS.pnl_w / CAP * 100                      # 當日損益占 300 萬 %
DAYS["equity"] = CAP + DAYS.pnl_w.cumsum()
DAYS["dd_w"] = DAYS.equity - DAYS.equity.cummax()
DAYS["dd"] = DAYS.dd_w / CAP * 100
print("載入 D1 ...", flush=True)
D1 = L.load_d1(since=(DAYS.date.min() - pd.DateOffset(months=4)).strftime("%Y_%m"))[["stock_id", "date", "open", "high", "low", "close", "volume"]]
D1_BY = {k: g.reset_index(drop=True) for k, g in D1.groupby("stock_id", sort=False)}
_m1_cache, _lock = OrderedDict(), threading.Lock()


def m1_table(ym):
    with _lock:
        if ym in _m1_cache:
            _m1_cache.move_to_end(ym); return _m1_cache[ym]
    tb = pq.read_table(L.m1_path(ym))
    with _lock:
        _m1_cache[ym] = tb
        while len(_m1_cache) > 4: _m1_cache.popitem(last=False)
    return tb


def clean(v):
    if isinstance(v, (float, np.floating)): return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (pd.Timestamp,)): return str(v.date())
    return v


def records(df):
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def api_days():
    stats = dict(n_days=len(DAYS), n_trade_days=int((DAYS.n_trades > 0).sum()), n_trades=len(TR), win=float((TR.r_pct > 0).mean() * 100), per_trade=float(TR.r_pct.mean()),
                 mdd=float(DAYS.dd.min()), mdd_w=float(DAYS.dd_w.min()), final=float(DAYS.equity.iloc[-1]), total=float(DAYS.pnl_w.sum()),
                 first=str(DAYS.date.min().date()), last=str(DAYS.date.max().date()),
                 sharpe=float(DAYS.ret_int.mean() / DAYS.ret_int.std() * np.sqrt(245)))
    cols = ["date", "trade_ok", "reasons", "mkt_gap", "n_cand", "n_watch", "n_trades", "ret_int", "pnl_w", "ret_avg", "n_win", "n_stop", "equity", "dd", "dd_w"]
    return dict(stats=stats, days=records(DAYS[cols]))


def api_day(date):
    d = pd.Timestamp(date)
    row = DAYS[DAYS.date == d]
    if row.empty: return dict(error=f"{date} 不在歷史中")
    tr = TR[TR.date == d].sort_values(["entry_t", "stock_id"])
    w = WT[WT.date == d].copy()
    state_name = {"CLOSED": "收盤回補", "STOPPED": "停損", "WAIT": "未跌破", "EXPIRED": "未跌破", "FULL": "跌破但額滿", "EXCLUDED": "剔除"}
    w["status"] = [state_name.get(s, s) + (f" ({e})" if s == "EXCLUDED" and e else "") for s, e in zip(w.state, w.excluded)]
    w = w[~w.stock_id.isin(tr.stock_id)].sort_values("gap", ascending=False)
    summ = {k: clean(v) for k, v in row.iloc[0].to_dict().items()}
    return dict(summary=summ, trades=records(tr), others=records(w))


def api_bars(date, sid):
    d = pd.Timestamp(date)
    tb = m1_table(d.strftime("%Y_%m"))
    t = tb.filter(pc.equal(tb["stock_id"], sid)).to_pandas()
    t = t[t.date.str.startswith(str(d.date()))]
    ts = pd.to_datetime(t.date); t["t"] = ts.dt.hour * 60 + ts.dt.minute - 540
    t = t[(t.t >= 0) & (t.t <= 270)].sort_values("t")
    intraday = [[int(a), float(o), float(h), float(l), float(c), int(v)] for a, o, h, l, c, v in zip(t.t, t.open, t.high, t.low, t.close, t.volume)]
    g = D1_BY.get(sid)
    daily = []
    if g is not None:
        g = g[g.date <= d].tail(80)
        daily = [[str(x.date()), float(o), float(h), float(l), float(c), int(v)] for x, o, h, l, c, v in zip(g.date, g.open, g.high, g.low, g.close, g.volume)]
    info = {}
    r = TR[(TR.date == d) & (TR.stock_id == sid)]
    w = WT[(WT.date == d) & (WT.stock_id == sid)]
    src = r if len(r) else w
    if len(src):
        info = {k: clean(v) for k, v in src.iloc[0].to_dict().items()}
        if len(w): info.update({k: clean(w.iloc[0][k]) for k in ["pc", "gap", "atr20"] if k in w.columns})
        pcv = info.get("pc") or (g.iloc[-2].close if g is not None and len(g) > 1 else None)
        if pcv: info["pc"] = pcv; info["limit_up"] = L.limit_up(pcv); info["force_px"] = pcv * 1.08
    return dict(intraday=intraday, daily=daily, info=info)


def api_live_index():
    src = XL.LocalSrc(LIVE_DIR)
    return XL.build_index([XL.compare_day(ds, XL.read_live(src, ds), TR, WT, DAYS)[0] for ds in src.dates()])


def api_live_day(date):
    src = XL.LocalSrc(LIVE_DIR)
    if date not in src.dates(): return dict(error=f"{date} 沒有實盤檔")
    return L._jclean(XL.compare_day(date, XL.read_live(src, date), TR, WT, DAYS)[1])


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/days": out = api_days()
            elif u.path == "/api/day": out = api_day(q["date"])
            elif u.path == "/api/bars": out = api_bars(q["date"], q["sid"])
            elif u.path == "/api/live_index": out = L._jclean(api_live_index())
            elif u.path == "/api/live_day": out = api_live_day(q["date"])
            elif u.path == "/api/live":
                lp = ROOT / "signals" / "live.json"
                if not lp.exists(): return self._send(404, b"{}", "application/json")
                return self._send(200, lp.read_bytes(), "application/json; charset=utf-8")
            else: return self._send(404, b"not found", "text/plain")
            self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": repr(e)}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, *a): pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--live-dir", default=str(LIVE_DIR), help="實盤檔資料夾 (<date>_trades.csv 等), 預設 signals/")
    a = ap.parse_args(); LIVE_DIR = Path(a.live_dir)
    print(f"實盤紀錄讀 {LIVE_DIR}", flush=True)
    print(f"開啟 http://127.0.0.1:{a.port}  (Ctrl+C 結束)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
