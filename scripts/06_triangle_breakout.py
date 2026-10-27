"""D1 三角收斂 → 日內突破 (M1 逐分鐘找第一根穿越趨勢線的 bar) 的回測. 同時給當沖 (13:30 出) 與波段 (收盤後持有 1/3/5 日) 的結果.
三角定義 (回看 L=20 日, 全部用昨日以前資料):
  高點回歸斜率 < 0 且 低點回歸斜率 > 0 (對稱) ; 末 5 日區間 / 首 5 日區間 <= CONTR ; 昨收在上下緣之間 ; 上緣 > 下緣 (未過頂點).
  上緣/下緣外推到今日 = 今日突破價 res / sup.
-> data/processed/triangle_trades.parquet, stats/triangle_breakout.md"""
import glob, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from config import RAW, PROC, STATS, COST, EXPLORE_END
L = 20; CONTR = 0.5; FORCE = 0.08; FORCE_SLIP = 0.002; LOCK_PENALTY = 0.07

d = pd.read_parquet(PROC / "d1_feat.parquet")
d = d.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = d.groupby("stock_id", sort=False)
# ---- 回歸斜率 (視窗內 x = 0..L-1), 用全域 index 技巧向量化 ----
d["i"] = g.cumcount().astype(float)
Sx_var = L * (L * L - 1) / 12.0                               # Σ(x-x̄)² for x=0..L-1
def roll_sum(col): return g[col].transform(lambda s: s.rolling(L).sum())
for nm, col in [("h", "high"), ("l", "low")]:
    d[f"iy_{nm}"] = d.i * d[col]
    Sy = roll_sum(col); Siy = roll_sum(f"iy_{nm}")
    ibar = d.i - (L - 1) / 2.0                                 # 視窗內 i 的平均 (以全域 index 表示)
    slope = (Siy - ibar * Sy) / Sx_var                         # Σ(i-ī)(y-ȳ)/Σ(i-ī)²  = Σ i·y - ī·Σy  (因 Σ(i-ī)=0)
    ybar = Sy / L
    d[f"slope_{nm}"] = slope / d.pc                            # 每日斜率 (占昨收比例)
    d[f"line_{nm}_last"] = ybar + slope * ((L - 1) / 2.0)      # 視窗最後一天 (x=L-1) 的線值
    d[f"line_{nm}_next"] = ybar + slope * ((L + 1) / 2.0)      # 外推一天 (x=L) 的線值
# 以上在 "昨日" 計算 (視窗結束於昨日), 所以 shift(1) 後成為今日可用的資訊
for c in ["slope_h", "slope_l", "line_h_last", "line_l_last", "line_h_next", "line_l_next"]:
    d[c] = g[c].shift(1)
rng5_last = g.high.transform(lambda s: s.shift(1).rolling(5).max()) - g.low.transform(lambda s: s.shift(1).rolling(5).min())
rng5_first = g.high.transform(lambda s: s.shift(L - 4).rolling(5).max()) - g.low.transform(lambda s: s.shift(L - 4).rolling(5).min())
d["contr"] = rng5_last / rng5_first
d["res"] = d.line_h_next; d["sup"] = d.line_l_next
d["width"] = (d.res - d.sup) / d.pc
inside = (d.pc < d.line_h_last) & (d.pc > d.line_l_last)
d["tri_sym"] = d.univ & (d.slope_h < 0) & (d.slope_l > 0) & (d.contr <= CONTR) & inside & (d.width > 0.005)
d["tri_asc"] = d.univ & (d.slope_h.abs() < 0.001) & (d.slope_l > 0) & (d.contr <= CONTR) & inside & (d.width > 0.005)     # 上緣水平, 低點上升
d["tri_desc"] = d.univ & (d.slope_h < 0) & (d.slope_l.abs() < 0.001) & (d.contr <= CONTR) & inside & (d.width > 0.005)    # 下緣水平, 高點下降
d["tri_loose"] = d.univ & (d.slope_h < 0) & (d.slope_l > 0) & (d.contr <= 0.7) & inside & (d.width > 0.005)
d["tri_any"] = d.tri_sym | d.tri_asc | d.tri_desc
# 波段用: 未來收盤
for k in [1, 3, 5]: d[f"c_fwd{k}"] = g.close.shift(-k)
d["touch_up_any"] = d.touch_up
flag = d[d.tri_any | d.tri_loose]
print("universe days", int(d.univ.sum()), "| 三角 (對稱)", int(d.tri_sym.sum()), "上升", int(d.tri_asc.sum()), "下降", int(d.tri_desc.sum()), "寬鬆", int(d.tri_loose.sum()))
cols = ["stock_id", "date", "pc", "open", "high", "low", "close", "gap", "vol20", "dv20", "atr20", "p_vr", "vr5_20", "mkt_gap", "dow", "res", "sup", "width", "contr", "slope_h", "slope_l",
        "tri_sym", "tri_asc", "tri_desc", "tri_loose", "c_fwd1", "c_fwd3", "c_fwd5"]
flag = flag[cols].copy(); flag = flag[flag.date >= "2020-01-01"]
flag.to_parquet(PROC / "triangle_flags.parquet", index=False)

# ---- M1: 找第一根穿越 res (向上) / sup (向下) 的 bar ----
def process(f):
    ym = f.split("/")[-1][:7]
    fl = flag[(flag.date.dt.strftime("%Y_%m") == ym)]
    if fl.empty: return None
    m = pd.read_parquet(f)
    m["ts"] = pd.to_datetime(m.date); m["date"] = m.ts.dt.normalize()
    m = m.merge(fl[["stock_id", "date", "res", "sup", "vol20"]], on=["stock_id", "date"], how="inner")
    if m.empty: return None
    m["t"] = (m.ts.dt.hour * 60 + m.ts.dt.minute - 540).astype(int)
    m = m[(m.t >= 0) & (m.t <= 270)].sort_values(["stock_id", "date", "t"])
    key = ["stock_id", "date"]; gg = m.groupby(key, sort=False)
    m["cumv"] = gg.volume.cumsum()
    out = pd.DataFrame({"m1_open": gg.open.first(), "auc_c": gg.close.last(), "m1_high": gg.high.max(), "m1_low": gg.low.min(), "m1_vol": gg.volume.sum(),
                        "o03_dv": m[m.t < 3].assign(dv=lambda x: x.close * x.volume).groupby(key).dv.sum(), "o03_v": m[m.t < 3].groupby(key).volume.sum()})
    for dr, cond, lvl, fn in [("up", m.high > m.res, "res", np.maximum), ("dn", m.low < m.sup, "sup", np.minimum)]:
        b = m[cond & (m.t <= 180)].groupby(key, sort=False).head(1).set_index(key)
        out[f"{dr}_t"] = b.t; out[f"{dr}_px"] = fn(b[lvl], b.open); out[f"{dr}_barv"] = b.volume; out[f"{dr}_cumv"] = b.cumv
        y = m.join(b[["t"]].rename(columns={"t": "bt"}), on=key)
        a = y[y.t > y.bt].groupby(key, sort=False)
        out[f"{dr}_hi_after"] = a.high.max(); out[f"{dr}_lo_after"] = a.low.min()
        # 突破後 30 分鐘的收盤 (觀察是否假突破)
        c30 = y[(y.t > y.bt) & (y.t <= y.bt + 30)].groupby(key, sort=False).close.last(); out[f"{dr}_c30"] = c30
    return out.reset_index()

if __name__ == "__main__":
    fs = sorted(glob.glob(str(RAW / "m1" / "*.parquet")))
    with ProcessPoolExecutor(5) as ex: parts = [p for p in ex.map(process, fs) if p is not None]
    m1 = pd.concat(parts, ignore_index=True)
    tr = flag.merge(m1, on=["stock_id", "date"], how="inner")
    tr["entry_open"] = tr.o03_dv / tr.o03_v
    tr["period"] = np.where(tr.date <= EXPLORE_END, "explore", "validate"); tr["year"] = tr.date.dt.year
    tr["lu"] = tr.pc * 1.10; tr["ld"] = tr.pc * 0.90
    tr.to_parquet(PROC / "triangle_trades.parquet", index=False)

    def settle(s, side, ent, hi_after, lo_after, exit_px):
        sgn = 1 if side == "long" else -1
        if side == "long":
            forced = lo_after <= s.pc * (1 - FORCE); ex = np.where(forced, s.pc * (1 - FORCE) * (1 - FORCE_SLIP), exit_px); locked = (~forced) & (ex <= s.ld * 1.001)
        else:
            forced = hi_after >= s.pc * (1 + FORCE); ex = np.where(forced, s.pc * (1 + FORCE) * (1 + FORCE_SLIP), exit_px); locked = (~forced) & (ex >= s.lu * 0.999)
        r = (sgn * (ex / ent - 1) - COST - np.where(locked, LOCK_PENALTY, 0)) * 100
        naive = (sgn * (exit_px / ent - 1) - COST) * 100
        return pd.DataFrame({"date": s.date.values, "period": s.period.values, "year": s.year.values, "r": r, "naive": naive, "forced": forced, "stock_id": s.stock_id.values})
    def summ(x, label):
        if len(x) < 30: return dict(label=label, n=len(x))
        o = dict(label=label, n=len(x), per_day=len(x) / x.date.nunique(), mean=x.r.mean(), med=x.r.median(), win=(x.r > 0).mean() * 100, naive=x.naive.mean(), forced=x.forced.mean() * 100)
        for lab, p in [("ex", "explore"), ("va", "validate")]:
            q = x[x.period == p]; dq = q.groupby("date").r.mean()
            o[f"{lab}_mean"] = q.r.mean(); o[f"{lab}_t"] = dq.mean() / dq.std() * np.sqrt(len(dq)) if len(dq) > 2 else np.nan
        for y in range(2020, 2027): o[f"y{y}"] = x[x.year == y].r.mean()
        return o
    def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
    def table(rows, cols):
        rr = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
        for _, x in rr.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
        return out
    C = [("條件", "label"), ("筆數", "n"), ("每日筆", "per_day"), ("每筆淨%", "mean"), ("中位", "med"), ("勝率", "win"), ("天真%", "naive"), ("強制%", "forced"),
         ("探索%", "ex_mean"), ("探索t", "ex_t"), ("驗證%", "va_mean"), ("驗證t", "va_t")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

    md = [f"# D1 三角收斂 → 日內突破 回測", "",
          f"三角: 回看 {L} 日, 高點斜率<0 且 低點斜率>0, 末 5 日區間/首 5 日區間 ≤ {CONTR}, 昨收仍在三角內。上/下緣外推到今日 = 突破價。universe 同掃描。",
          f"進場: M1 第一根穿越突破價的 bar (12:00 前), 進場價 = max(突破價, 該 bar 開) (向上) / min (向下)；開盤就已在上緣之上 (跳空突破) 另列。出場 13:30 收盤競價；±8% 強制、鎖死 −7%、成本 {COST*100:.2f}%。",
          f"樣本: universe 股票日 {int(d.univ.sum()):,}, 對稱三角 {int(d.tri_sym.sum()):,} ({d.tri_sym.sum()/d.univ.sum()*100:.1f}%), 2020 起有 M1 的 {len(tr):,}。", ""]
    rows = []
    for nm, mask_name in [("對稱三角", "tri_sym"), ("上升三角 (上緣水平)", "tri_asc"), ("下降三角 (下緣水平)", "tri_desc"), ("寬鬆 (收斂 ≤0.7)", "tri_loose")]:
        s = tr[tr[mask_name]]
        # 向上突破 做多
        u = s[s.up_t.notna() & (s.m1_open < s.res)]
        rows.append(summ(settle(u, "long", u.up_px, u.up_hi_after, u.up_lo_after, u.auc_c), f"{nm} | 日內向上突破 → 多"))
        rows.append(summ(settle(u, "short", u.up_px, u.up_hi_after, u.up_lo_after, u.auc_c), f"{nm} | 日內向上突破 → 空 (假突破賭)"))
        gu = s[s.m1_open >= s.res]
        rows.append(summ(settle(gu, "long", gu.entry_open, gu.m1_high, gu.m1_low, gu.auc_c), f"{nm} | 跳空開在上緣之上 → 開盤多"))
        rows.append(summ(settle(gu, "short", gu.entry_open, gu.m1_high, gu.m1_low, gu.auc_c), f"{nm} | 跳空開在上緣之上 → 開盤空"))
        dn = s[s.dn_t.notna() & (s.m1_open > s.sup)]
        rows.append(summ(settle(dn, "short", dn.dn_px, dn.dn_hi_after, dn.dn_lo_after, dn.auc_c), f"{nm} | 日內向下突破 → 空"))
        rows.append(summ(settle(dn, "long", dn.dn_px, dn.dn_hi_after, dn.dn_lo_after, dn.auc_c), f"{nm} | 日內向下突破 → 多 (假跌破賭)"))
        gd = s[s.m1_open <= s.sup]
        rows.append(summ(settle(gd, "short", gd.entry_open, gd.m1_high, gd.m1_low, gd.auc_c), f"{nm} | 跳空開在下緣之下 → 開盤空"))
        ins = s[(s.m1_open < s.res) & (s.m1_open > s.sup)]
        rows.append(summ(settle(ins, "long", ins.entry_open, ins.m1_high, ins.m1_low, ins.auc_c), f"{nm} | 開在三角內, 不等突破 開盤多 (對照)"))
    md += ["## 1. 當沖: 各種三角 × 突破方向", "", table(rows, C), ""]

    s = tr[tr.tri_sym]; u = s[s.up_t.notna() & (s.m1_open < s.res)].copy()
    # ---- 2. 向上突破做多 細分 ----
    u["barv_ratio"] = u.up_barv * 1000 / (u.vol20 / 270)           # 突破 bar 量 / 日均每分鐘量
    u["cumv_ratio"] = u.up_cumv * 1000 / u.vol20                   # 突破時累積量 / 日均量
    u["c30_hold"] = u.up_c30 > u.res                                # 突破後 30 分仍在上緣之上
    rows = []
    def add(mask, lab): x = u[mask]; rows.append(summ(settle(x, "long", x.up_px, x.up_hi_after, x.up_lo_after, x.auc_c), lab))
    add(u.up_t <= 15, "突破時間 ≤09:15"); add((u.up_t > 15) & (u.up_t <= 60), "09:15~10:00"); add((u.up_t > 60) & (u.up_t <= 120), "10:00~11:00"); add(u.up_t > 120, "11:00~12:00")
    add(u.barv_ratio >= 5, "突破 bar 量 ≥5× 日均分鐘量"); add(u.barv_ratio < 5, "突破 bar 量 <5×")
    add(u.cumv_ratio >= 0.3, "突破時累積量 ≥30% 日均量"); add(u.cumv_ratio < 0.3, "累積量 <30%")
    add(u.vr5_20 < 0.7, "三角期間量縮 (5 日/20 日 <0.7)"); add(u.vr5_20 >= 0.7, "三角期間未量縮")
    add(u.contr <= 0.35, "收斂很緊 (≤0.35)"); add(u.contr > 0.35, "收斂 0.35~0.5")
    add(u.width <= 0.03, "三角寬度 ≤3%"); add(u.width > 0.03, "寬度 >3%")
    add(u.atr20 >= 0.03, "ATR20 ≥3%"); add(u.atr20 < 0.03, "ATR20 <3%")
    add(u.mkt_gap >= 0, "大盤跳空 ≥0"); add(u.mkt_gap < 0, "大盤跳空 <0")
    add(u.gap >= 0.01, "個股跳空 ≥1% (開在三角上半)"); add(u.gap < 0.01, "個股跳空 <1%")
    add(u.c30_hold, "[事後] 突破後 30 分仍站上緣"); add(~u.c30_hold, "[事後] 30 分內跌回三角內")
    md += ["## 2. 對稱三角 日內向上突破做多 的細分", "", table(rows, C), "",
           "最後兩列是事後資訊 (看到 30 分後才知道), 只用來判斷「假突破比例與代價」, 不能當進場條件。", ""]
    # 等 30 分確認再進
    conf = u[u.c30_hold].copy()
    rows = [summ(settle(conf, "long", conf.up_c30, conf.up_hi_after, conf.up_lo_after, conf.auc_c), "突破 30 分後仍站上緣 → 以 30 分後的價進多 (可執行)")]
    fail = u[~u.c30_hold].copy()
    rows.append(summ(settle(fail, "short", fail.up_c30, fail.up_hi_after, fail.up_lo_after, fail.auc_c), "突破 30 分後跌回 → 以 30 分後的價空 (假突破)"))
    md += ["## 3. 等 30 分鐘確認", "", table(rows, C), ""]

    # ---- 4. 波段: 突破日收盤買, 持有 1/3/5 日 (無成本以外調整, 單邊成本 0.585% 來回 ≈ 0.6%) ----
    rows = []
    SW_COST = 0.6
    for lab, x, ent in [("對稱三角 日內向上突破, 突破價進", u, u.up_px), ("對稱三角 日內向上突破, 當日收盤進", u, u.auc_c),
                        ("對稱三角 跳空開上緣之上, 開盤進", s[s.m1_open >= s.res], s[s.m1_open >= s.res].entry_open)]:
        for k in [1, 3, 5]:
            r = ((x[f"c_fwd{k}"] / ent - 1) * 100 - SW_COST)
            q = pd.DataFrame({"date": x.date.values, "period": x.period.values, "year": x.year.values, "r": r.values, "naive": r.values, "forced": False})
            rows.append(summ(q.dropna(subset=["r"]), f"{lab} → 持有到 T+{k} 收盤"))
    dn = s[s.dn_t.notna() & (s.m1_open > s.sup)]
    for k in [1, 3, 5]:
        r = (-(dn[f"c_fwd{k}"] / dn.dn_px - 1) * 100 - SW_COST)
        q = pd.DataFrame({"date": dn.date.values, "period": dn.period.values, "year": dn.year.values, "r": r.values, "naive": r.values, "forced": False})
        rows.append(summ(q.dropna(subset=["r"]), f"對稱三角 日內向下突破 空 → 持有到 T+{k} 收盤 (不計借券費)"))
    md += ["## 4. 波段對照: 突破後持有 1/3/5 日 (來回成本 0.6%, 不處理漲跌停)", "", table(rows, C), "",
           "用來分辨「型態無效」vs「型態有效但不適合當沖」。", ""]
    # ---- 5. 假突破統計 ----
    md += ["## 5. 假突破比例 (對稱三角, 日內向上突破)", "",
           f"- 突破後 30 分鐘仍在上緣之上: {u.c30_hold.mean()*100:.0f}%；收盤在上緣之上: {(u.auc_c > u.res).mean()*100:.0f}%；收盤高於突破價: {(u.auc_c > u.up_px).mean()*100:.0f}%。",
           f"- 突破當日收盤 ≥ 昨收 +5%: {(u.auc_c/u.pc-1 >= 0.05).mean()*100:.1f}%；收盤漲停: {(u.auc_c >= u.lu*0.999).mean()*100:.1f}%。",
           f"- 突破時間中位數: 09:00 + {u.up_t.median():.0f} 分。", ""]
    (STATS / "triangle_breakout.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
