"""M1 逐月 -> 每檔每日日內摘要 (只算 universe 內股票日) -> data/processed/m1_daily/YYYY_MM.parquet
欄位: 開盤區間 (5/15/30/60 分 高低收量), 09:00~09:02 VWAP (開盤進場價), 各時點價格, 13:30 收盤, 全日高低量,
      ORB 突破 (視窗 5/15/30): 向上/向下突破的時間、進場價、突破後最高/最低."""
import glob, sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from config import RAW, PROC
OUT = PROC / "m1_daily"; OUT.mkdir(exist_ok=True)
UNIV = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ"])
UNIV = UNIV[UNIV.univ][["stock_id", "date"]]

def window(df, mask, name):
    s = df[mask].groupby(["stock_id", "date"], sort=False)
    return pd.DataFrame({f"{name}_h": s.high.max(), f"{name}_l": s.low.min(), f"{name}_c": s.close.last(),
                         f"{name}_v": s.volume.sum(), f"{name}_dv": s.dv.sum()})

def orb(df, base, W):
    """開盤 W 分鐘區間突破. 突破 = t>=W 的 bar high > 區間高 (向上) / low < 區間低 (向下). 進場價 = max(區間高, 該 bar open) (向上) / min(區間低, bar open) (向下)."""
    key = ["stock_id", "date"]
    hl = base[[f"o{W:02d}_h", f"o{W:02d}_l"]].rename(columns={f"o{W:02d}_h": "rh", f"o{W:02d}_l": "rl"})
    x = df.join(hl, on=key)
    res = {}
    for d_, cond, lvl, fn in [("up", x.high > x.rh, "rh", np.maximum), ("dn", x.low < x.rl, "rl", np.minimum)]:
        m = (x.t >= W) & (x.t <= 270) & cond
        first = x[m].groupby(key, sort=False).head(1).set_index(key)
        px = fn(first[lvl].values, first.open.values)
        tr = pd.DataFrame({f"orb{W}_{d_}_t": first.t.values, f"orb{W}_{d_}_px": px, f"orb{W}_{d_}_vcum": first.cumv.values}, index=first.index)
        # 突破後 (含突破 bar 之後的 bar) 的極值 與 10 分鐘後價格
        y = x.join(tr[[f"orb{W}_{d_}_t"]], on=key)
        after = y[(y.t > y[f"orb{W}_{d_}_t"]) & (y.t <= 270)].groupby(key, sort=False)
        ext = pd.DataFrame({f"orb{W}_{d_}_hi_after": after.high.max(), f"orb{W}_{d_}_lo_after": after.low.min()})
        res[d_] = tr.join(ext)
    return res["up"].join(res["dn"], how="outer")

def process(f):
    ym = Path(f).stem
    of = OUT / f"{ym}.parquet"
    if of.exists(): return ym, "skip"
    df = pd.read_parquet(f)
    df["ts"] = pd.to_datetime(df.date); df["date"] = df.ts.dt.normalize()
    df = df.merge(UNIV, on=["stock_id", "date"], how="inner")
    if df.empty: return ym, "no universe rows"
    df["t"] = (df.ts.dt.hour * 60 + df.ts.dt.minute - 540).astype(np.int16)   # 09:00 = 0, 13:30 = 270
    df = df[(df.t >= 0) & (df.t <= 270)].sort_values(["stock_id", "date", "t"]).reset_index(drop=True)
    df["dv"] = df.close * df.volume
    key = ["stock_id", "date"]
    g = df.groupby(key, sort=False)
    df["cumv"] = g.volume.cumsum()
    base = pd.DataFrame({"m1_open": g.open.first(), "m1_high": g.high.max(), "m1_low": g.low.min(), "m1_close": g.close.last(),
                         "m1_vol": g.volume.sum(), "m1_dv": g.dv.sum(), "n_bars": g.close.size(), "t_last": g.t.last()})
    base["t_high"] = g.apply(lambda x: x.t.values[x.high.values.argmax()])
    base["t_low"] = g.apply(lambda x: x.t.values[x.low.values.argmin()])
    for W in [3, 5, 15, 30, 60]:
        base = base.join(window(df, df.t < W, f"o{W:02d}"))
    base["entry_open"] = base.o03_dv / base.o03_v            # 09:00~09:02 VWAP
    base["vwap30"] = base.o30_dv / base.o30_v
    for hh, nm in [(60, "c1000"), (120, "c1100"), (180, "c1200"), (240, "c1300")]:
        base[nm] = df[df.t < hh].groupby(key, sort=False).close.last()
    for W in [3, 5, 30, 60]:                                 # 進場後 (t >= W) 的最高/最低, 供強制回補/鎖漲停判斷
        a = df[df.t >= W].groupby(key, sort=False)
        base[f"a{W:02d}_h"] = a.high.max(); base[f"a{W:02d}_l"] = a.low.min()
    base = base.join(window(df, df.t == 270, "auc"))        # 13:30 收盤競價 bar
    base = base.join(window(df, (df.t >= 240), "last30"))   # 13:00 以後
    for W in [5, 15, 30]:
        base = base.join(orb(df, base, W))
    base = base.reset_index()
    base.to_parquet(of, index=False)
    return ym, f"{len(df)} bars -> {len(base)} stock-days"

if __name__ == "__main__":
    fs = sorted(glob.glob(str(RAW / "m1" / "*.parquet")))
    if len(sys.argv) > 1: fs = [f for f in fs if Path(f).stem in sys.argv[1:]]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ym, msg in ex.map(process, fs):
            print(ym, msg, flush=True)
