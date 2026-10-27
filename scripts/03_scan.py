"""D1 型態 × M1 日內觸發 × 多/空 全組合掃描.
每個組合: 進場價 (觸發定義), 13:30 收盤競價出場, 來回成本 COST. 探索期 (<=2023) / 驗證期 (2024~) 分開統計.
現實結算 (v1 教訓): 空單進場後價格到昨收 +8% 立即回補 (+0.2% 滑價); 收盤仍鎖漲停 額外 -7% (標借/違約). 多單對稱 (-8% 強制, 鎖跌停 -7%).
進場前已到 +8%/-8% 或已摸到漲/跌停的不進場. 另附 naive (天真: 收盤出, 不處理) 供對照.
輸出 stats/scan_all.csv (全部組合) 與 stats/pattern_scan.md (摘要)."""
import glob, numpy as np, pandas as pd
from config import PROC, STATS, COST, EXPLORE_END

d1 = pd.read_parquet(PROC / "d1_feat.parquet")
d1 = d1[d1.univ & (d1.date >= "2020-01-01")]
m1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "m1_daily" / "*.parquet")))], ignore_index=True)
df = d1.merge(m1, on=["stock_id", "date"], how="inner")
df = df[df.auc_c.notna() & df.entry_open.notna()].reset_index(drop=True)
df["exit"] = df.auc_c                                   # 13:30 收盤競價價
df["lu_px"] = df.pc * 1.10; df["ld_px"] = df.pc * 0.90   # 漲跌停價 (近似)
df["period"] = np.where(df.date <= EXPLORE_END, "explore", "validate")
df["year"] = df.date.dt.year
FORCE = 0.08; FORCE_SLIP = 0.002; LOCK_PENALTY = 0.07
df["force_up"] = df.pc * (1 + FORCE); df["force_dn"] = df.pc * (1 - FORCE)
SETUPS = [c[2:] for c in df.columns if c.startswith("S_")]
print("stock-days", len(df), df.date.min().date(), df.date.max().date(), "setups", len(SETUPS))

# ================= M1 觸發 (回傳 entry price Series, NaN = 未觸發) =================
# 每個觸發同時定義給多/空用的進場價; 方向由掃描決定. 另附 "t" 進場時間 (供統計) 與觸發後的 hi/lo 供停損分析 (這版只用收盤出場)
T = {}
T["open"] = dict(px=df.entry_open, t=pd.Series(1, index=df.index), after="a03", pre_h=df.o03_h, pre_l=df.o03_l, note="09:00~09:02 VWAP 進場")
T["t0930"] = dict(px=df.o30_c, t=pd.Series(30, index=df.index), after="a30", pre_h=df.o30_h, pre_l=df.o30_l, note="09:30 價進場 (不看方向)")
T["t1000"] = dict(px=df.c1000, t=pd.Series(60, index=df.index), after="a60", pre_h=df.o60_h, pre_l=df.o60_l, note="10:00 價進場")
for W in [5, 15, 30]:
    for dr in ["up", "dn"]:
        tcol = f"orb{W}_{dr}_t"
        ok = df[tcol] <= 180                                           # 12:00 前突破才算
        T[f"orb{W}_{dr}"] = dict(px=df[f"orb{W}_{dr}_px"].where(ok), t=df[tcol].where(ok), after=f"orb{W}_{dr}", pre_h=df[f"o{W:02d}_h"], pre_l=df[f"o{W:02d}_l"], note=f"開盤 {W} 分區間{'向上' if dr=='up' else '向下'}突破 (12:00 前), 突破價進場")
        ok_e = ok & (df[tcol] <= 60)                                   # 10:00 前的早突破
        T[f"orb{W}_{dr}_early"] = dict(px=df[f"orb{W}_{dr}_px"].where(ok_e), t=df[tcol].where(ok_e), after=f"orb{W}_{dr}", pre_h=df[f"o{W:02d}_h"], pre_l=df[f"o{W:02d}_l"], note=f"開盤 {W} 分區間{'向上' if dr=='up' else '向下'}突破 (10:00 前)")
# 09:30 相對位置
r30 = (df.o30_c - df.o30_l) / (df.o30_h - df.o30_l)
T["o30_strong"] = dict(after="a30", pre_h=df.o30_h, pre_l=df.o30_l, px=df.o30_c.where((r30 >= 0.8) & (df.o30_c > df.vwap30) & (df.o30_c > df.m1_open)), t=pd.Series(30, index=df.index), note="09:30 收在 30 分區間頂 20% 且 > VWAP > 開盤")
T["o30_weak"] = dict(after="a30", pre_h=df.o30_h, pre_l=df.o30_l, px=df.o30_c.where((r30 <= 0.2) & (df.o30_c < df.vwap30) & (df.o30_c < df.m1_open)), t=pd.Series(30, index=df.index), note="09:30 收在 30 分區間底 20% 且 < VWAP < 開盤")
T["o30_up2"] = dict(after="a30", pre_h=df.o30_h, pre_l=df.o30_l, px=df.o30_c.where(df.o30_c / df.m1_open - 1 >= 0.02), t=pd.Series(30, index=df.index), note="開盤 30 分漲 ≥2%")
T["o30_dn2"] = dict(after="a30", pre_h=df.o30_h, pre_l=df.o30_l, px=df.o30_c.where(df.o30_c / df.m1_open - 1 <= -0.02), t=pd.Series(30, index=df.index), note="開盤 30 分跌 ≥2%")
T["o05_up1_vol"] = dict(after="a05", pre_h=df.o05_h, pre_l=df.o05_l, px=df.o05_c.where((df.o05_c / df.m1_open - 1 >= 0.01) & (df.o05_v * 1000 / df.vol20 >= 0.15)), t=pd.Series(5, index=df.index), note="開盤 5 分漲 ≥1% 且 5 分量 ≥ 15% 日均量")
T["o05_dn1_vol"] = dict(after="a05", pre_h=df.o05_h, pre_l=df.o05_l, px=df.o05_c.where((df.o05_c / df.m1_open - 1 <= -0.01) & (df.o05_v * 1000 / df.vol20 >= 0.15)), t=pd.Series(5, index=df.index), note="開盤 5 分跌 ≥1% 且 5 分量 ≥ 15% 日均量")
T["gapfill_up"] = dict(after="a05", pre_h=df.o05_h, pre_l=df.o05_l, px=df.pc.where((df.gap >= 0.02) & (df.o60_l <= df.pc)), t=pd.Series(45, index=df.index), note="跳空開高 ≥2% 且 10:00 前回補到昨收, 以昨收價進場")
T["gapfill_dn"] = dict(after="a05", pre_h=df.o05_h, pre_l=df.o05_l, px=df.pc.where((df.gap <= -0.02) & (df.o60_h >= df.pc)), t=pd.Series(45, index=df.index), note="跳空開低 ≥2% 且 10:00 前反彈到昨收, 以昨收價進場")
T["hold_hi_1000"] = dict(after="a60", pre_h=df.o60_h, pre_l=df.o60_l, px=df.c1000.where((df.c1000 >= df.o60_h * 0.995) & (df.c1000 > df.m1_open)), t=pd.Series(60, index=df.index), note="10:00 價在前 60 分高點 0.5% 內 (強勢續抱)")
T["hold_lo_1000"] = dict(after="a60", pre_h=df.o60_h, pre_l=df.o60_l, px=df.c1000.where((df.c1000 <= df.o60_l * 1.005) & (df.c1000 < df.m1_open)), t=pd.Series(60, index=df.index), note="10:00 價在前 60 分低點 0.5% 內 (弱勢)")
T["above_vwap_1000"] = dict(after="a60", pre_h=df.o60_h, pre_l=df.o60_l, px=df.c1000.where((df.c1000 > df.vwap30) & (df.c1000 > df.m1_open) & (df.c1000 < df.o60_h * 0.995)), t=pd.Series(60, index=df.index), note="10:00 在 VWAP 與開盤之上但未創高 (拉回後)")
T["below_vwap_1000"] = dict(after="a60", pre_h=df.o60_h, pre_l=df.o60_l, px=df.c1000.where((df.c1000 < df.vwap30) & (df.c1000 < df.m1_open) & (df.c1000 > df.o60_l * 1.005)), t=pd.Series(60, index=df.index), note="10:00 在 VWAP 與開盤之下但未破低")
TRIGS = list(T.keys())
pd.DataFrame([(k, v["note"]) for k, v in T.items()], columns=["trigger", "note"]).to_csv(STATS / "triggers.csv", index=False)

# 多/空各自的「進場價受漲跌停限制」: 進場價已在漲停 (多無法買 / 空的回補風險) 或跌停 (空無法賣) 的排除
def stats_block(r, day, lab):
    """r: 每筆淨報酬 (%), day: 每日平均 (%) -> dict"""
    n = len(r); nd = len(day)
    if n == 0: return {f"{lab}_n": 0}
    sd = day.std(ddof=1) if nd > 1 else np.nan
    return {f"{lab}_n": n, f"{lab}_days": nd, f"{lab}_mean": r.mean(), f"{lab}_med": r.median(), f"{lab}_win": (r > 0).mean() * 100,
            f"{lab}_daymean": day.mean(), f"{lab}_t": day.mean() / sd * np.sqrt(nd) if sd and sd > 0 else np.nan,
            f"{lab}_sharpe": day.mean() / sd * np.sqrt(245) if sd and sd > 0 else np.nan}

rows = []; keep_trades = []
for s in SETUPS:
    ms = df[f"S_{s}"].values
    for tg in TRIGS:
        px = T[tg]["px"]; tt = T[tg]["t"]
        base_m = ms & px.notna().values & (px.values > 0)
        if base_m.sum() < 100: continue
        aft = T[tg]["after"]
        hi_after = df[f"{aft}_h"] if aft.startswith("a") else df[f"{aft}_hi_after"]
        lo_after = df[f"{aft}_l"] if aft.startswith("a") else df[f"{aft}_lo_after"]
        pre_h, pre_l = T[tg]["pre_h"], T[tg]["pre_l"]
        for side, sgn in [("long", 1), ("short", -1)]:
            m = base_m.copy()
            if side == "long":  m &= (px.values < df.lu_px.values * 0.999) & (pre_l.values > df.force_dn.values) & (px.values > df.force_dn.values)
            else:               m &= (px.values > df.ld_px.values * 1.001) & (pre_h.values < df.force_up.values) & (px.values < df.force_up.values)
            sub = df.loc[m, ["stock_id", "date", "period", "year", "exit", "lu_px", "ld_px", "force_up", "force_dn", "gap", "atr20", "dv20", "pc", "mkt_gap", "dow"]].copy()
            if len(sub) < 100: continue
            e = px.values[m]
            sub["naive"] = (sgn * (sub.exit.values / e - 1) - COST) * 100
            if side == "short":
                forced = (hi_after.values[m] >= sub.force_up.values)
                ex = np.where(forced, sub.force_up.values * (1 + FORCE_SLIP), sub.exit.values)
                locked = (~forced) & (ex >= sub.lu_px.values * 0.999)
            else:
                forced = (lo_after.values[m] <= sub.force_dn.values)
                ex = np.where(forced, sub.force_dn.values * (1 - FORCE_SLIP), sub.exit.values)
                locked = (~forced) & (ex <= sub.ld_px.values * 1.001)
            sub["r"] = (sgn * (ex / e - 1) - COST - np.where(locked, LOCK_PENALTY, 0.0)) * 100
            sub["t"] = tt.values[m]
            row = dict(setup=s, trigger=tg, side=side, n=len(sub), per_day=len(sub) / sub.date.nunique(), t_entry=sub.t.mean(),
                       forced=forced.mean() * 100, locked=locked.mean() * 100, naive=sub.naive.mean(), mean=sub.r.mean(), win=(sub.r > 0).mean() * 100)
            for lab, p in [("ex", "explore"), ("va", "validate")]:
                q = sub[sub.period == p]
                row.update(stats_block(q.r, q.groupby("date").r.mean(), lab))
            yr = sub.groupby("year").r.mean()
            for y in range(2020, 2027): row[f"y{y}"] = yr.get(y, np.nan)
            row["yrs_pos"] = int((yr > 0).sum()); row["yrs_n"] = len(yr)
            rows.append(row)
            if min(row.get("ex_t", -9) or -9, row.get("va_t", -9) or -9) >= 1.0:          # 保留候選組合的逐筆交易供 05 組合分析
                sub["setup"] = s; sub["trigger"] = tg; sub["side"] = side; sub["entry"] = e; sub["forced"] = forced; sub["locked"] = locked
                keep_trades.append(sub[["setup", "trigger", "side", "stock_id", "date", "t", "entry", "r", "naive", "forced", "locked", "gap", "atr20", "dv20", "pc", "mkt_gap", "dow"]])
res = pd.DataFrame(rows)
pd.concat(keep_trades, ignore_index=True).to_parquet(PROC / "scan_trades.parquet", index=False)
res.to_csv(STATS / "scan_all.csv", index=False)
print("combos", len(res))

# ================= 報告 =================
def fmt(x, nd=2):
    return "" if pd.isna(x) else f"{x:.{nd}f}"
def table(r, cols):
    hdr = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, x in r.iterrows():
        body += "| " + " | ".join((fmt(x[c[1]], c[2]) if len(c) > 2 else (fmt(x[c[1]]) if isinstance(x[c[1]], (float, np.floating)) else str(x[c[1]]))) for c in cols) + " |\n"
    return hdr + body
COLS = [("D1 型態", "setup"), ("M1 觸發", "trigger"), ("方向", "side"), ("筆數", "n", 0), ("每日筆", "per_day", 1),
        ("探索 每筆%", "ex_mean"), ("探索 勝率", "ex_win", 0), ("探索 t", "ex_t", 1), ("探索 Sharpe", "ex_sharpe", 1),
        ("驗證 每筆%", "va_mean"), ("驗證 勝率", "va_win", 0), ("驗證 t", "va_t", 1), ("驗證 Sharpe", "va_sharpe", 1),
        ("正年/總年", "yrs_pos", 0), ("強制回補%", "forced", 0), ("鎖死%", "locked", 1), ("天真每筆%", "naive")]
ok = res[(res.ex_n >= 300) & (res.va_n >= 200)].copy()
ok["score"] = ok[["ex_t", "va_t"]].min(axis=1)                     # 兩段期間都要顯著
md = ["# D1 型態 × M1 觸發 × 多空 掃描結果", "",
      f"資料: universe (前 20 日均成交額 ≥1 億、昨收 ≥10、上市櫃非 ETF) 2020-01 ~ {df.date.max().date()}，{len(df):,} 股票日。",
      f"出場一律 13:30 收盤競價；來回成本 {COST*100:.2f}%；探索期 ≤{EXPLORE_END}，驗證期之後。",
      f"共 {len(SETUPS)} 種 D1 型態 × {len(TRIGS)} 種 M1 觸發 × 2 方向 = {len(res)} 個組合 (樣本 ≥100)。",
      "注意: 多重檢定 — 幾百個組合裡挑最好的一定會過度樂觀，所以只看探索與驗證**兩段都顯著**的；排名用兩段 t 值的較小者。",
      f"**現實結算**: 空單進場後價格到昨收 +{FORCE*100:.0f}% 立即回補 (+{FORCE_SLIP*100:.1f}% 滑價)，收盤仍鎖漲停額外 −{LOCK_PENALTY*100:.0f}%；多單對稱。進場前已到 ±{FORCE*100:.0f}% 或摸到漲跌停的不進。「天真每筆%」= 不處理漲跌停、收盤出場的數字，供對照掃描為何不能天真。", "",
      "## 1. 兩段期間都顯著 (min(t) 最高前 30)", "", table(ok.sort_values("score", ascending=False).head(30), COLS), "",
      "## 2. 多單 (long) 最好的前 20 (min t)", "", table(ok[ok.side == "long"].sort_values("score", ascending=False).head(20), COLS), "",
      "## 3. 空單 (short) 最好的前 20 (min t)", "", table(ok[ok.side == "short"].sort_values("score", ascending=False).head(20), COLS), "",
      "## 4. 每日筆數 ≥ 3 且兩段都顯著 (有容量的)", "", table(ok[ok.per_day >= 3].sort_values("score", ascending=False).head(20), COLS), ""]
# 基準: 每個 D1 型態在 open 觸發的多/空
b = res[(res.trigger == "open")].copy()
md += ["## 5. 各 D1 型態的基準 (09:00 開盤進, 收盤出)", "", "正值 = 該方向有日內漂移。", "",
       table(b.sort_values(["side", "ex_mean"], ascending=[True, False]), COLS), ""]
# 每個觸發在 all 型態下
b = res[(res.setup == "all")].copy()
md += ["## 6. 各 M1 觸發的基準 (不分 D1 型態)", "", table(b.sort_values(["side", "ex_mean"], ascending=[True, False]), COLS), ""]
# 年度穩定: 前 30 的逐年
top = ok.sort_values("score", ascending=False).head(30)
yc = [("D1 型態", "setup"), ("M1 觸發", "trigger"), ("方向", "side")] + [(str(y), f"y{y}") for y in range(2020, 2027)]
md += ["## 7. 前 30 組合的逐年每筆淨 %", "", table(top, yc), "",
       "## 觸發定義", ""] + [f"- `{k}`: {v['note']}" for k, v in T.items()]
(STATS / "pattern_scan.md").write_text("\n".join(md), encoding="utf-8")
print("written", STATS / "pattern_scan.md")
print(table(ok.sort_values("score", ascending=False).head(15), COLS))
