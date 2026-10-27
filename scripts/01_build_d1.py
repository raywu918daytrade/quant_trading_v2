"""D1 月檔合併 -> 特徵 + D1 型態/量能標籤 (全部只用「前一日以前」資料, gap 類另用今日開盤價) -> data/processed/d1_feat.parquet"""
import glob, numpy as np, pandas as pd
from config import RAW, PROC, MIN_DV20, MIN_PRICE

fs = sorted(glob.glob(str(RAW / "d1" / "*.parquet")))
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d["date"] = pd.to_datetime(d.date)
d = d.drop_duplicates(["stock_id", "date"]).sort_values(["stock_id", "date"]).reset_index(drop=True)
info = pd.read_parquet(RAW / "meta" / "info.parquet")
d = d.merge(info[["stock_id", "name", "market_type"]], on="stock_id", how="left")
d = d[d.market_type.isin(["twse", "tpex"]) & d.stock_id.str.fullmatch(r"\d{4}") & ~d.stock_id.str.startswith("00")].reset_index(drop=True)
print("D1 rows", len(d), d.date.min().date(), d.date.max().date(), "stocks", d.stock_id.nunique())

g = d.groupby("stock_id", sort=False)
d["pc"] = g.close.shift(1)                       # 昨收
d["ret"] = d.close / d.pc - 1
d["gap"] = d.open / d.pc - 1                     # 今日跳空 (用今日 open)
d["intra"] = d.close / d.open - 1               # 今日 開→收 (目標)
d["rng"] = (d.high - d.low) / d.pc
d["body"] = (d.close - d.open).abs() / d.pc
d["uw"] = (d.high - d[["open", "close"]].max(axis=1)) / d.pc       # 上影
d["lw"] = (d[["open", "close"]].min(axis=1) - d.low) / d.pc        # 下影
d["dv"] = d.close * d.volume
d["uplim"] = d.ret >= 0.095
d["dnlim"] = d.ret <= -0.095
d["touch_up"] = d.high >= (d.pc * 1.10 - 1e-6)     # 盤中觸及漲停 (近似)
d["touch_dn"] = d.low <= (d.pc * 0.90 + 1e-6)

def roll(col, n, fn="mean", shift=1):
    return g[col].transform(lambda s: getattr(s.shift(shift).rolling(n), fn)())

# ---- 量能 (全部 shift 1 = 昨日以前) ----
d["dv20"] = roll("dv", 20)
d["vol20"] = roll("volume", 20)
d["p_volume"] = g.volume.shift(1)
d["p_vr"] = d.p_volume / g.volume.transform(lambda s: s.shift(2).rolling(20).mean())   # 昨日量 / 前 20 日均量
d["vr5_20"] = roll("volume", 5) / roll("volume", 20, shift=6)                           # 近 5 日均量 / 其前 20 日均量
d["atr20"] = roll("rng", 20)
d["sd20"] = roll("ret", 20, "std")

# ---- 昨日 K 線 ----
for c in ["ret", "intra", "rng", "body", "uw", "lw", "uplim", "dnlim", "gap", "touch_up"]:
    d[f"p_{c}"] = g[c].shift(1)
d["p2_ret"] = g.ret.shift(2); d["p3_ret"] = g.ret.shift(3)
d["p2_uplim"] = g.uplim.shift(2)
d["ret5"] = g.close.shift(1) / g.close.shift(6) - 1
d["ret20"] = g.close.shift(1) / g.close.shift(21) - 1
d["hi20"] = roll("high", 20, "max", shift=2); d["lo20"] = roll("low", 20, "min", shift=2)    # 前 20 日 (不含昨日)
d["hi60"] = roll("high", 60, "max", shift=2); d["lo60"] = roll("low", 60, "min", shift=2)
d["hi10"] = roll("high", 10, "max", shift=2); d["lo10"] = roll("low", 10, "min", shift=2)
d["p_close"] = d.pc
d["p_high"] = g.high.shift(1); d["p_low"] = g.low.shift(1); d["p_open"] = g.open.shift(1)
d["ma5"] = roll("close", 5); d["ma20"] = roll("close", 20); d["ma60"] = roll("close", 60)
d["range10"] = (roll("high", 10, "max") / roll("low", 10, "min") - 1)                      # 近 10 日 (含昨日) 區間
d["pos20"] = (d.pc - roll("low", 20, "min")) / (roll("high", 20, "max") - roll("low", 20, "min"))

# ---- 市場 ----
liq = d[d.dv20 >= MIN_DV20]
mk = liq.groupby("date").agg(mkt_gap=("gap", "mean"), mkt_pret=("p_ret", "mean"), mkt_breadth=("gap", lambda s: (s > 0).mean()))
d = d.merge(mk, left_on="date", right_index=True, how="left")
d["dow"] = d.date.dt.dayofweek

# ---- universe ----
d["univ"] = (d.dv20 >= MIN_DV20) & (d.pc >= MIN_PRICE) & d.open.notna() & d.pc.notna()

# ================= D1 型態 / 量能 標籤 (布林) =================
S = {}
S["all"] = d.univ
S["brk20_vol"] = (d.p_close > d.hi20) & (d.p_vr >= 1.5)              # 昨收創 20 日新高 + 放量
S["brk20_lowvol"] = (d.p_close > d.hi20) & (d.p_vr < 1.0)           # 新高但量縮
S["brk60_vol"] = (d.p_close > d.hi60) & (d.p_vr >= 1.5)              # 60 日新高放量
S["bigred_vol"] = (d.p_ret >= 0.04) & (d.p_vr >= 2.0)                # 長紅爆量 (未漲停)
S["bigred_vol_noLU"] = S["bigred_vol"] & ~d.p_uplim.astype(bool)
S["uplim"] = d.p_uplim.astype(bool)                                  # 昨日漲停
S["uplim_2d"] = d.p_uplim.astype(bool) & d.p2_uplim.astype(bool)     # 連兩日漲停
S["uplim_first"] = d.p_uplim.astype(bool) & ~d.p2_uplim.astype(bool) & (d.ret5 - d.p_ret < 0.05)   # 第一根漲停 (之前沒漲多)
S["touchLU_fail"] = d.p_touch_up.astype(bool) & ~d.p_uplim.astype(bool)   # 昨日摸到漲停沒鎖住
S["biggreen_vol"] = (d.p_ret <= -0.04) & (d.p_vr >= 2.0)             # 長黑爆量
S["dnlim"] = d.p_dnlim.astype(bool)                                  # 昨日跌停
S["squeeze"] = (d.range10 <= 0.08) & (d.vr5_20 < 0.8)                # 10 日區間 ≤8% 且量縮 (盤整收斂)
S["squeeze_brk"] = (d.range10 <= 0.10) & (d.p_close > d.hi10) & (d.p_vr >= 1.5)   # 盤整後放量突破
S["squeeze_bkdn"] = (d.range10 <= 0.10) & (d.p_close < d.lo10) & (d.p_vr >= 1.5)
S["up3"] = (d.p_ret > 0) & (d.p2_ret > 0) & (d.p3_ret > 0)           # 連漲 3 日
S["dn3"] = (d.p_ret < 0) & (d.p2_ret < 0) & (d.p3_ret < 0)
S["bull_ma"] = (d.pc > d.ma5) & (d.ma5 > d.ma20) & (d.ma20 > d.ma60)  # 多頭排列
S["bear_ma"] = (d.pc < d.ma5) & (d.ma5 < d.ma20) & (d.ma20 < d.ma60)
S["vol_dry"] = d.p_vr <= 0.5                                         # 昨日量縮一半以下
S["vol_climax"] = (d.p_vr >= 3.0) & (d.p_uw >= 0.03)                 # 爆量長上影 (出貨)
S["vol_climax_dn"] = (d.p_vr >= 3.0) & (d.p_lw >= 0.03)              # 爆量長下影
S["hammer"] = (d.p_lw >= 2 * d.p_body) & (d.p_lw >= 0.02) & (d.ret5 < -0.03)     # 跌勢中錘子
S["shooting"] = (d.p_uw >= 2 * d.p_body) & (d.p_uw >= 0.02) & (d.ret5 > 0.03)    # 漲勢中流星
S["gap_up3"] = d.gap >= 0.03
S["gap_up3_vol"] = (d.gap >= 0.03) & (d.p_vr >= 1.5)
S["gap_dn3"] = d.gap <= -0.03
S["gap_up_afterLU"] = d.p_uplim.astype(bool) & (d.gap >= 0.03)
S["gap_up_squeeze"] = (d.range10 <= 0.10) & (d.gap >= 0.02)          # 盤整後跳空開高
S["hi_atr"] = d.atr20 >= 0.05                                        # 高波動
S["lo_atr"] = d.atr20 <= 0.02
S["ret20_hot"] = d.ret20 >= 0.30                                     # 20 日漲 30%+
S["ret20_cold"] = d.ret20 <= -0.20
S["pos20_top"] = d.pos20 >= 0.95
S["pos20_bot"] = d.pos20 <= 0.05
S["mkt_gap_up"] = d.mkt_gap >= 0.01
S["mkt_gap_dn"] = d.mkt_gap <= -0.01
for k, v in S.items():
    d[f"S_{k}"] = (v.fillna(False).astype(bool) & d.univ)
SETUPS = list(S.keys())
pd.Series(SETUPS).to_csv(PROC / "setups.txt", index=False, header=False)

keep = ["stock_id", "date", "name", "market_type", "open", "high", "low", "close", "volume", "pc", "ret", "gap", "intra", "rng",
        "dv20", "vol20", "p_vr", "vr5_20", "atr20", "sd20", "p_ret", "p_intra", "p_rng", "p_uplim", "p_dnlim", "ret5", "ret20", "pos20",
        "range10", "touch_up", "touch_dn", "mkt_gap", "mkt_pret", "mkt_breadth", "dow", "univ"] + [f"S_{k}" for k in SETUPS]
out = d[keep]
out.to_parquet(PROC / "d1_feat.parquet", index=False)
u = out[out.univ]
print("universe stock-days", len(u), "avg per day", round(len(u) / u.date.nunique()))
print("\n各型態 (universe 內) 樣本數與 日內 開→收 平均:")
rows = []
for k in SETUPS:
    m = out[f"S_{k}"]
    rows.append((k, int(m.sum()), round(out.loc[m, "gap"].mean() * 100, 2), round(out.loc[m, "intra"].mean() * 100, 2), round((out.loc[m, "intra"] > 0).mean() * 100, 1)))
print(pd.DataFrame(rows, columns=["setup", "n", "gap%", "intra%", "intra>0%"]).to_string(index=False))
