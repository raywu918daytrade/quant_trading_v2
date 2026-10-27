"""M1 -> 每檔每日「開盤區間向下突破」細節 (universe 全部股票日), 視窗 W=5/10/15/30 分.
每個 W: 區間高低量; 第一根 low < 區間低 的 bar (t<=180): 時間/進場價 (min(區間低, bar open))/bar 量/累積量/累積成交額;
        突破後 5/15/30 分收盤; 收盤確認版 (第一根 close < 區間低) 的時間與收盤; 突破後最高/最低;
        三種停損: 反彈收回區間內 (close > 區間低), 站回 VWAP (close > 當時 VWAP), 突破區間高 (high > 區間高) → 各自的時間與出場價.
通用: 09:00~09:02 VWAP, 收盤競價, 11:00/12:00/13:00 收盤, 全日量. -> data/processed/orb_dn/YYYY_MM.parquet"""
import glob, sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from config import RAW, PROC
OUT = PROC / "orb_dn"; OUT.mkdir(exist_ok=True)
UNIV = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ"]); UNIV = UNIV[UNIV.univ][["stock_id", "date"]]
WS = [5, 10, 15, 30]; DEADLINE = 180

def first_after(y, cond, key, name_t, name_px, px_col):
    b = y[cond].groupby(key, sort=False).head(1).set_index(key)
    return pd.DataFrame({name_t: b.t, name_px: b[px_col], name_t.replace("_t", "_open"): b.open})

def process(f):
    ym = Path(f).stem; of = OUT / f"{ym}.parquet"
    if of.exists(): return ym, "skip"
    df = pd.read_parquet(f); df["ts"] = pd.to_datetime(df.date); df["date"] = df.ts.dt.normalize()
    df = df.merge(UNIV, on=["stock_id", "date"], how="inner")
    if df.empty: return ym, "none"
    df["t"] = (df.ts.dt.hour * 60 + df.ts.dt.minute - 540).astype(np.int16)
    df = df[(df.t >= 0) & (df.t <= 270)].sort_values(["stock_id", "date", "t"]).reset_index(drop=True)
    df["dv"] = df.close * df.volume
    key = ["stock_id", "date"]; g = df.groupby(key, sort=False)
    df["cumv"] = g.volume.cumsum(); df["cumdv"] = g.dv.cumsum(); df["vwap"] = df.cumdv / df.cumv
    o3 = df[df.t < 3].groupby(key, sort=False)
    base = pd.DataFrame({"m1_open": g.open.first(), "auc_c": g.close.last(), "m1_high": g.high.max(), "m1_low": g.low.min(), "m1_vol": g.volume.sum(), "m1_dv": g.dv.sum(),
                         "entry_open": o3.dv.sum() / o3.volume.sum(), "o03_h": o3.high.max()})
    for hh, nm in [(120, "c1100"), (180, "c1200"), (240, "c1300")]:
        base[nm] = df[df.t < hh].groupby(key, sort=False).close.last()
    for W in WS:
        w = df[df.t < W].groupby(key, sort=False)
        base[f"o{W}_h"] = w.high.max(); base[f"o{W}_l"] = w.low.min(); base[f"o{W}_v"] = w.volume.sum(); base[f"o{W}_dv"] = w.dv.sum()
        x = df.join(base[[f"o{W}_h", f"o{W}_l"]].rename(columns={f"o{W}_h": "rh", f"o{W}_l": "rl"}), on=key)
        m = (x.t >= W) & (x.t <= DEADLINE) & (x.low < x.rl)
        b = x[m].groupby(key, sort=False).head(1).set_index(key)
        p = f"dn{W}_"
        base[p + "t"] = b.t; base[p + "px"] = np.minimum(b.rl, b.open); base[p + "barv"] = b.volume; base[p + "cumv"] = b.cumv; base[p + "cumdv"] = b.cumdv; base[p + "vwap_at"] = b.vwap
        base[p + "bar_c"] = b.close
        y = x.join(b[["t"]].rename(columns={"t": "bt"}), on=key); y = y[y.bt.notna()]
        after = y[y.t > y.bt]
        a = after.groupby(key, sort=False); base[p + "hi_after"] = a.high.max(); base[p + "lo_after"] = a.low.min()
        for k in [5, 15, 30]:
            base[p + f"c{k}"] = y[(y.t > y.bt) & (y.t <= y.bt + k)].groupby(key, sort=False).close.last()
        # 收盤確認版: 第一根 close < 區間低 (t>=W, <=DEADLINE)
        cc = x[(x.t >= W) & (x.t <= DEADLINE) & (x.close < x.rl)].groupby(key, sort=False).head(1).set_index(key)
        base[p + "cc_t"] = cc.t; base[p + "cc_px"] = cc.close; base[p + "cc_cumv"] = cc.cumv
        yc = x.join(cc[["t"]].rename(columns={"t": "bt"}), on=key); yc = yc[yc.bt.notna()]; ac = yc[yc.t > yc.bt].groupby(key, sort=False)
        base[p + "cc_hi_after"] = ac.high.max(); base[p + "cc_lo_after"] = ac.low.min()
        # 停損 (突破 bar 之後): 收回區間內 / 站回 VWAP / 突破區間高
        s1 = first_after(after, after.close > after.rl, key, p + "stop_in_t", p + "stop_in_px", "close"); base = base.join(s1)
        s2 = first_after(after, after.close > after.vwap, key, p + "stop_vwap_t", p + "stop_vwap_px", "close"); base = base.join(s2)
        s3 = first_after(after, after.high > after.rh, key, p + "stop_hi_t", p + "stop_hi_px", "rh"); base = base.join(s3)
        # 停損高度格點: 區間低 + k × 區間寬 (k=0.5, 1.5, 2)
        for k, kn in [(0.5, "k05"), (1.5, "k15"), (2.0, "k20")]:
            lvl = after.rl + k * (after.rh - after.rl)
            b2 = after[after.high > lvl].groupby(key, sort=False).head(1).set_index(key)
            l2 = (b2.rl + k * (b2.rh - b2.rl))
            base = base.join(pd.DataFrame({p + f"stop_{kn}_t": b2.t, p + f"stop_{kn}_px": l2, p + f"stop_{kn}_open": b2.open}))
    base = base.reset_index(); base.to_parquet(of, index=False)
    return ym, f"{len(df)} bars -> {len(base)}"

if __name__ == "__main__":
    fs = sorted(glob.glob(str(RAW / "m1" / "*.parquet")))
    if len(sys.argv) > 1: fs = [f for f in fs if Path(f).stem in sys.argv[1:]]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ym, msg in ex.map(process, fs): print(ym, msg, flush=True)
