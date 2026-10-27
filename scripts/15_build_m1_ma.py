"""M1 -> 分鐘均線 (MA10/MA20 of close) 的斜率快照與交叉觸發 (universe 全部股票日).
快照 (09:30, 10:00): MA10, MA20, MA20 5 分斜率 (%), MA10 5 分斜率, 價/MA20-1, 當時價, 當時累積量.
死叉 (做空): 第一根 t in [20,180], MA10 < MA20 且前一根 MA10 >= MA20 且 MA20 斜率(5分) < 0: 時間、收盤價 (進場)、進場前當日最高 (停損位)、之後最高/最低、停損 bar (high > 進場前最高) 的時間/開盤.
黃金交叉 (做多): 對稱.  -> data/processed/m1_ma/YYYY_MM.parquet"""
import glob, sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from config import RAW, PROC
OUT = PROC / "m1_ma"; OUT.mkdir(exist_ok=True)
UNIV = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "univ"]); UNIV = UNIV[UNIV.univ][["stock_id", "date"]]

def process(f):
    ym = Path(f).stem; of = OUT / f"{ym}.parquet"
    if of.exists(): return ym, "skip"
    df = pd.read_parquet(f); df["ts"] = pd.to_datetime(df.date); df["date"] = df.ts.dt.normalize()
    df = df.merge(UNIV, on=["stock_id", "date"], how="inner")
    if df.empty: return ym, "none"
    df["t"] = (df.ts.dt.hour * 60 + df.ts.dt.minute - 540).astype(np.int16)
    df = df[(df.t >= 0) & (df.t <= 270)].sort_values(["stock_id", "date", "t"]).reset_index(drop=True)
    key = ["stock_id", "date"]; g = df.groupby(key, sort=False)
    # 以 bar 序 (非時間) 算 MA; 缺 bar 的分鐘視為沒交易
    df["ma10"] = g.close.transform(lambda s: s.rolling(10).mean()); df["ma20"] = g.close.transform(lambda s: s.rolling(20).mean())
    df["ma20_s5"] = (df.ma20 / g.ma20.shift(5) - 1) * 100; df["ma10_s5"] = (df.ma10 / g.ma10.shift(5) - 1) * 100
    df["ma10_prev"] = g.ma10.shift(1); df["ma20_prev"] = g.ma20.shift(1)
    df["run_hi"] = g.high.cummax(); df["run_lo"] = g.low.cummin(); df["cumv"] = g.volume.cumsum()
    df["n"] = g.cumcount()
    o3 = df[df.t < 3].groupby(key, sort=False)
    base = pd.DataFrame({"m1_open": g.open.first(), "auc_c": g.close.last(), "m1_high": g.high.max(), "m1_low": g.low.min(), "entry_open": (o3.close.sum() * 0 + (df[df.t < 3].close * df[df.t < 3].volume).groupby([df[df.t < 3].stock_id, df[df.t < 3].date]).sum() / o3.volume.sum())})
    for T, nm in [(30, "s0930"), (60, "s1000")]:
        snap = df[df.t < T].groupby(key, sort=False).tail(1).set_index(key)
        base[nm + "_px"] = snap.close; base[nm + "_ma20s"] = snap.ma20_s5; base[nm + "_ma10s"] = snap.ma10_s5; base[nm + "_pma20"] = snap.close / snap.ma20 - 1; base[nm + "_nbar"] = snap.n
        base[nm + "_runhi"] = snap.run_hi; base[nm + "_runlo"] = snap.run_lo
    for sg, nm in [(-1, "dc"), (1, "gc")]:
        if sg < 0: cond = (df.ma10 < df.ma20) & (df.ma10_prev >= df.ma20_prev) & (df.ma20_s5 < 0)
        else:      cond = (df.ma10 > df.ma20) & (df.ma10_prev <= df.ma20_prev) & (df.ma20_s5 > 0)
        b = df[cond & (df.n >= 20) & (df.t <= 180)].groupby(key, sort=False).head(1).set_index(key)
        base[nm + "_t"] = b.t; base[nm + "_px"] = b.close; base[nm + "_cumv"] = b.cumv; base[nm + "_ma20s"] = b.ma20_s5
        stop_lvl = b.run_hi if sg < 0 else b.run_lo
        base[nm + "_stop_lvl"] = stop_lvl
        y = df.join(b[["t"]].rename(columns={"t": "bt"}), on=key).join(pd.DataFrame({"sl": stop_lvl}), on=key); y = y[y.bt.notna()]
        after = y[y.t > y.bt]; a = after.groupby(key, sort=False)
        base[nm + "_hi_after"] = a.high.max(); base[nm + "_lo_after"] = a.low.min()
        sc = after[(after.high > after.sl)] if sg < 0 else after[(after.low < after.sl)]
        sb = sc.groupby(key, sort=False).head(1).set_index(key)
        base[nm + "_stop_t"] = sb.t; base[nm + "_stop_open"] = sb.open
    base = base.reset_index(); base.to_parquet(of, index=False)
    return ym, f"{len(df)} -> {len(base)}"

if __name__ == "__main__":
    fs = sorted(glob.glob(str(RAW / "m1" / "*.parquet")))
    if len(sys.argv) > 1: fs = [f for f in fs if Path(f).stem in sys.argv[1:]]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ym, msg in ex.map(process, fs): print(ym, msg, flush=True)
