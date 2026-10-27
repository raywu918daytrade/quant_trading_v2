"""用 orb_live 的回放邏輯建立每日交易歷史 (供 webapp.py 瀏覽).
輸出 data/processed/orb_history/: days.parquet (每日摘要), trades.parquet (進場的每筆), watch.parquet (監看名單與狀態, 含剔除).
預設增量: 只補 days.parquet 最後一天之後的日子; --full 全部重建.  用法: python3 build_history.py [--full] [--start 2020-01-01]"""
import argparse, time, numpy as np, pandas as pd
import orb_live as L
from config import PROC, RAW
OUT = PROC / "orb_history"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--start", default="2020-01-01"); a = ap.parse_args()

old = {k: (pd.read_parquet(OUT / f"{k}.parquet") if (OUT / f"{k}.parquet").exists() and not a.full else None) for k in ["days", "trades", "watch"]}
start = pd.Timestamp(a.start)
if old["days"] is not None and len(old["days"]): start = old["days"].date.max() + pd.Timedelta(days=1)
d1 = L.load_d1(since=(start - pd.DateOffset(months=6)).strftime("%Y_%m"))
m1_months = {p.stem for p in (RAW / "m1").glob("*.parquet")} | {p.stem for p in (L.LIVE / "m1").glob("*.parquet")}
days = [d for d in np.sort(d1.date.unique()) if d >= start and pd.Timestamp(d).strftime("%Y_%m") in m1_months]
print(f"要補 {len(days)} 個交易日 ({pd.Timestamp(days[0]).date() if days else '-'} ~ {pd.Timestamp(days[-1]).date() if days else '-'})")
D, T, W = [], [], []; t0 = time.time(); cur_ym, m1 = None, None
dvals = d1.date.values
for d in days:
    d = pd.Timestamp(d); ym = d.strftime("%Y_%m")
    if ym != cur_ym:
        if cur_ym: print(f"{cur_ym} 完成 ({time.time() - t0:.0f}s)", flush=True)
        m1 = L.load_m1_month(ym); cur_ym = ym
    if not (m1.date == d).any():
        print(f"{d.date()} 還沒有分 K, 停在這裡 (之後 refresh 再補)"); break
    sl = d1[(dvals >= np.datetime64(d - pd.Timedelta(days=150))) & (dvals <= np.datetime64(d))]
    tr, info, watch, excl = L.replay_day(d, sl, m1)
    D.append(dict(date=d, trade_ok=info["trade_ok"], reasons="；".join(info["reasons"]), mkt_gap=info["mkt_gap"], breadth=info["breadth"],
                  n_cand=info.get("n_cand", 0), n_watch=info.get("n_watch", 0), n_trades=len(tr),
                  ret_cap=tr.r_pct.sum() / 10 if len(tr) else 0.0, pnl=tr.pnl.sum() if len(tr) else 0.0, ret_avg=tr.r_pct.mean() if len(tr) else np.nan,
                  n_win=int((tr.r_pct > 0).sum()) if len(tr) else 0, n_stop=int(tr.reason.str.startswith("停損").sum()) if len(tr) else 0))
    if len(tr): T.append(tr.assign(date=d))
    if "states" in info and len(info["states"]):
        st = info["states"][["sid", "state", "rh", "rl", "lu", "fpx", "lots"]].rename(columns={"sid": "stock_id"})
        w = watch[["stock_id", "name", "pc", "open", "gap", "atr20", "dv20"]].merge(st, on="stock_id", how="left")
        W.append(w.assign(date=d, excluded=""))
    if len(excl):
        W.append(excl[["stock_id", "name", "pc", "open", "gap", "atr20", "dv20", "rh", "rl", "excluded"]].assign(date=d, state="EXCLUDED"))
if cur_ym: print(f"{cur_ym} 完成 ({time.time() - t0:.0f}s)")
new = {"days": pd.DataFrame(D), "trades": pd.concat(T, ignore_index=True) if T else pd.DataFrame(), "watch": pd.concat(W, ignore_index=True) if W else pd.DataFrame()}
for k, v in new.items():
    df = pd.concat([old[k], v], ignore_index=True) if old[k] is not None and len(v) else (old[k] if old[k] is not None and not len(v) else v)
    if df is not None and len(df):
        df = df.sort_values(["date"] + (["entry_t", "stock_id"] if k == "trades" else ["stock_id"] if k == "watch" else [])).reset_index(drop=True)
        df.to_parquet(OUT / f"{k}.parquet", index=False); print(k, len(df))
dd = pd.read_parquet(OUT / "days.parquet")
eq = 3e6 + dd.pnl.cumsum()
print(f"交易日 {len(dd)}  有交易 {(dd.n_trades > 0).sum()}  筆數 {int(dd.n_trades.sum())}  300 萬整數張不複利: 累計損益 {dd.pnl.sum()/1e4:,.0f} 萬  MDD {((eq / eq.cummax()) - 1).min() * 100:.2f}%")
