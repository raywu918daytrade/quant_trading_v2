"""擺盪點版三角 (07_triangle_blocks) 的突破回測. 進場: M1 第一根穿越上緣 res (向上) / 下緣 sup (向下) 的 bar (12:00 前), 13:30 收盤出.
現實結算 ±8% 強制、鎖死 −7%、成本 0.23%. 另附波段 (持有 1/3/5 日) 對照. -> stats/triangle_blocks_backtest.md"""
import glob, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from config import RAW, PROC, STATS, COST, EXPLORE_END
FORCE = 0.08; FORCE_SLIP = 0.002; LOCK_PENALTY = 0.07; SW_COST = 0.6
flag = pd.read_parquet(PROC / "triangle_blocks.parquet"); flag = flag[flag.date >= "2020-01-01"].copy()
d1 = pd.read_parquet(PROC / "d1_feat.parquet", columns=["stock_id", "date", "close"]).sort_values(["stock_id", "date"])
g = d1.groupby("stock_id", sort=False)
for k in [1, 3, 5]: d1[f"c_fwd{k}"] = g.close.shift(-k)
flag = flag.merge(d1[["stock_id", "date", "c_fwd1", "c_fwd3", "c_fwd5"]], on=["stock_id", "date"], how="left")

def process(f):
    ym = f.split("/")[-1][:7]
    fl = flag[flag.date.dt.strftime("%Y_%m") == ym]
    if fl.empty: return None
    m = pd.read_parquet(f); m["ts"] = pd.to_datetime(m.date); m["date"] = m.ts.dt.normalize()
    m = m.merge(fl[["stock_id", "date", "res", "sup", "vol20"]], on=["stock_id", "date"], how="inner")
    if m.empty: return None
    m["t"] = (m.ts.dt.hour * 60 + m.ts.dt.minute - 540).astype(int)
    m = m[(m.t >= 0) & (m.t <= 270)].sort_values(["stock_id", "date", "t"])
    key = ["stock_id", "date"]; gg = m.groupby(key, sort=False); m["cumv"] = gg.volume.cumsum()
    o3 = m[m.t < 3]
    out = pd.DataFrame({"m1_open": gg.open.first(), "auc_c": gg.close.last(), "m1_high": gg.high.max(), "m1_low": gg.low.min(), "m1_vol": gg.volume.sum(),
                        "o03_dv": (o3.close * o3.volume).groupby([o3.stock_id, o3.date]).sum(), "o03_v": o3.groupby(key).volume.sum()})
    for dr, cond, lvl, fn in [("up", m.high > m.res, "res", np.maximum), ("dn", m.low < m.sup, "sup", np.minimum)]:
        b = m[cond & (m.t <= 180)].groupby(key, sort=False).head(1).set_index(key)
        out[f"{dr}_t"] = b.t; out[f"{dr}_px"] = fn(b[lvl], b.open); out[f"{dr}_barv"] = b.volume; out[f"{dr}_cumv"] = b.cumv
        y = m.join(b[["t"]].rename(columns={"t": "bt"}), on=key); a = y[y.t > y.bt].groupby(key, sort=False)
        out[f"{dr}_hi_after"] = a.high.max(); out[f"{dr}_lo_after"] = a.low.min()
        out[f"{dr}_c30"] = y[(y.t > y.bt) & (y.t <= y.bt + 30)].groupby(key, sort=False).close.last()
    return out.reset_index()

def settle(s, side, ent, hi_after, lo_after, exit_px):
    sgn = 1 if side == "long" else -1
    if side == "long":
        forced = lo_after <= s.pc * (1 - FORCE); ex = np.where(forced, s.pc * (1 - FORCE) * (1 - FORCE_SLIP), exit_px); locked = (~forced) & (ex <= s.pc * 0.9 * 1.001)
    else:
        forced = hi_after >= s.pc * (1 + FORCE); ex = np.where(forced, s.pc * (1 + FORCE) * (1 + FORCE_SLIP), exit_px); locked = (~forced) & (ex >= s.pc * 1.1 * 0.999)
    r = (sgn * (ex / ent - 1) - COST - np.where(locked, LOCK_PENALTY, 0)) * 100
    return pd.DataFrame({"date": s.date.values, "period": s.period.values, "year": s.year.values, "r": r, "naive": (sgn * (exit_px / ent - 1) - COST) * 100, "forced": forced})
def summ(x, label):
    if len(x) < 20: return dict(label=label, n=len(x))
    o = dict(label=label, n=len(x), mean=x.r.mean(), med=x.r.median(), win=(x.r > 0).mean() * 100, forced=x.forced.mean() * 100)
    day = x.groupby("date").r.mean(); o["t_all"] = day.mean() / day.std() * np.sqrt(len(day))
    for lab, p in [("ex", "explore"), ("va", "validate")]:
        q = x[x.period == p]; dq = q.groupby("date").r.mean(); o[f"{lab}_n"] = len(q); o[f"{lab}_mean"] = q.r.mean(); o[f"{lab}_t"] = dq.mean() / dq.std() * np.sqrt(len(dq)) if len(dq) > 2 else np.nan
    for y in range(2020, 2027): o[f"y{y}"] = x[x.year == y].r.mean()
    return o
def fmt(v): return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
def table(rows, cols):
    rr = pd.DataFrame(rows); out = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in rr.iterrows(): out += "| " + " | ".join(fmt(x.get(c[1])) for c in cols) + " |\n"
    return out
C = [("條件", "label"), ("筆數", "n"), ("每筆淨%", "mean"), ("中位", "med"), ("勝率", "win"), ("強制%", "forced"), ("t(全)", "t_all"), ("探索n", "ex_n"), ("探索%", "ex_mean"), ("探索t", "ex_t"), ("驗證n", "va_n"), ("驗證%", "va_mean"), ("驗證t", "va_t")] + [(str(y), f"y{y}") for y in range(2020, 2027)]

if __name__ == "__main__":
    fs = sorted(glob.glob(str(RAW / "m1" / "*.parquet")))
    with ProcessPoolExecutor(5) as ex: parts = [p for p in ex.map(process, fs) if p is not None]
    tr = flag.merge(pd.concat(parts, ignore_index=True), on=["stock_id", "date"], how="inner")
    tr["entry_open"] = tr.o03_dv / tr.o03_v; tr["period"] = np.where(tr.date <= EXPLORE_END, "explore", "validate"); tr["year"] = tr.date.dt.year
    tr.to_parquet(PROC / "triangle_blocks_trades.parquet", index=False)
    md = ["# 三角收斂 (擺盪點版) 突破回測", "",
          "三角定義 (07_triangle_blocks): 5 根 K 內最高/最低 = 擺盪高/低點; 最近 2 高 + 2 低, 後高<前高、後低>前低; 兩高點之間無其他高點、兩低點之間無其他低點; 4 點時間交錯; 昨收在線內; 中間 K 棒不穿線。",
          f"進場: M1 第一根穿越上緣/下緣的 bar (12:00 前), 進場價 = max(線, bar 開) / min; 開盤就在線外者另列。出場 13:30; ±{FORCE*100:.0f}% 強制、鎖死 −{LOCK_PENALTY*100:.0f}%、成本 {COST*100:.2f}%。t 以每日平均報酬計。", ""]
    SETS = [("收斂比 ≤0.8 (主)", tr.contr <= 0.8), ("收斂比 ≤0.7", tr.contr <= 0.7), ("不限收斂比", tr.contr <= 9)]
    rows = []
    for nm, mk in SETS:
        s = tr[mk]
        u = s[s.up_t.notna() & (s.m1_open < s.res)]; dn = s[s.dn_t.notna() & (s.m1_open > s.sup)]
        gu = s[s.m1_open >= s.res]; gd = s[s.m1_open <= s.sup]; ins = s[(s.m1_open < s.res) & (s.m1_open > s.sup)]
        rows += [summ(settle(u, "long", u.up_px, u.up_hi_after, u.up_lo_after, u.auc_c), f"{nm} | 向上突破 → 多"),
                 summ(settle(u, "short", u.up_px, u.up_hi_after, u.up_lo_after, u.auc_c), f"{nm} | 向上突破 → 空 (賭假突破)"),
                 summ(settle(gu, "long", gu.entry_open, gu.m1_high, gu.m1_low, gu.auc_c), f"{nm} | 跳空開在上緣上 → 開盤多"),
                 summ(settle(gu, "short", gu.entry_open, gu.m1_high, gu.m1_low, gu.auc_c), f"{nm} | 跳空開在上緣上 → 開盤空"),
                 summ(settle(dn, "short", dn.dn_px, dn.dn_hi_after, dn.dn_lo_after, dn.auc_c), f"{nm} | 向下突破 → 空"),
                 summ(settle(dn, "long", dn.dn_px, dn.dn_hi_after, dn.dn_lo_after, dn.auc_c), f"{nm} | 向下突破 → 多 (賭假跌破)"),
                 summ(settle(gd, "short", gd.entry_open, gd.m1_high, gd.m1_low, gd.auc_c), f"{nm} | 跳空開在下緣下 → 開盤空"),
                 summ(settle(ins, "long", ins.entry_open, ins.m1_high, ins.m1_low, ins.auc_c), f"{nm} | 開在三角內 不等突破 開盤多 (對照)"),
                 summ(settle(ins, "short", ins.entry_open, ins.m1_high, ins.m1_low, ins.auc_c), f"{nm} | 開在三角內 不等突破 開盤空 (對照)")]
        n_day = len(s); n_up = len(u) + len(gu); n_dn = len(dn) + len(gd)
        rows.append(dict(label=f"{nm} | (樣本: 三角日 {n_day}, 向上突破 {n_up} = {n_up/n_day*100:.0f}%, 向下 {n_dn} = {n_dn/n_day*100:.0f}%, 都沒 {n_day-len(s[s.up_t.notna()|(s.m1_open>=s.res)|s.dn_t.notna()|(s.m1_open<=s.sup)])})", n=""))
    md += ["## 1. 當沖", "", table(rows, C), ""]
    s = tr[tr.contr <= 0.8]; u = s[s.up_t.notna() & (s.m1_open < s.res)].copy(); dn = s[s.dn_t.notna() & (s.m1_open > s.sup)].copy()
    u["barv_ratio"] = u.up_barv * 1000 / (u.vol20 / 270); u["c30_hold"] = u.up_c30 > u.res
    rows = []
    def add(x, side, ent, h, l, lab): rows.append(summ(settle(x, side, ent, h, l, x.auc_c), lab))
    add(u[u.up_t <= 30], "long", u[u.up_t <= 30].up_px, u[u.up_t <= 30].up_hi_after, u[u.up_t <= 30].up_lo_after, "向上突破 多 | 09:30 前突破")
    add(u[u.up_t > 30], "long", u[u.up_t > 30].up_px, u[u.up_t > 30].up_hi_after, u[u.up_t > 30].up_lo_after, "向上突破 多 | 09:30 後突破")
    for lab, m in [("突破 bar 量 ≥5× 日均分鐘量", u.barv_ratio >= 5), ("突破 bar 量 <5×", u.barv_ratio < 5), ("三角期間量縮 (vr5_20<0.7)", u.vr5_20 < 0.7), ("未量縮", u.vr5_20 >= 0.7),
                   ("下降高點 ≥3 個", u.n_desc_h >= 3), ("上升低點 ≥3 個", u.n_asc_l >= 3), ("寬度 ≤5%", u.width <= 0.05), ("寬度 >5%", u.width > 0.05), ("大盤跳空 ≥0", u.mkt_gap >= 0), ("大盤跳空 <0", u.mkt_gap < 0),
                   ("[事後] 30 分後仍在線上", u.c30_hold), ("[事後] 30 分內跌回", ~u.c30_hold)]:
        x = u[m]; add(x, "long", x.up_px, x.up_hi_after, x.up_lo_after, f"向上突破 多 | {lab}")
    conf = u[u.c30_hold]; add(conf, "long", conf.up_c30, conf.up_hi_after, conf.up_lo_after, "向上突破 多 | 等 30 分確認仍在線上, 以 30 分後價進 (可執行)")
    for lab, m in [("09:30 前跌破", dn.dn_t <= 30), ("09:30 後跌破", dn.dn_t > 30), ("寬度 ≤5%", dn.width <= 0.05), ("寬度 >5%", dn.width > 0.05), ("下降高點 ≥3 個", dn.n_desc_h >= 3), ("大盤跳空 <0", dn.mkt_gap < 0)]:
        x = dn[m]; add(x, "short", x.dn_px, x.dn_hi_after, x.dn_lo_after, f"向下突破 空 | {lab}")
    md += ["## 2. 收斂比 ≤0.8 的細分", "", table(rows, C), ""]
    rows = []
    for lab, x, ent, sgn in [("向上突破 多, 突破價進", u, u.up_px, 1), ("向上突破 多, 當日收盤進", u, u.auc_c, 1), ("向下突破 空, 突破價進", dn, dn.dn_px, -1)]:
        for k in [1, 3, 5]:
            r = (sgn * (x[f"c_fwd{k}"] / ent - 1) * 100 - SW_COST)
            q = pd.DataFrame({"date": x.date.values, "period": x.period.values, "year": x.year.values, "r": r.values, "naive": r.values, "forced": False}).dropna(subset=["r"])
            rows.append(summ(q, f"{lab} → T+{k} 收盤出"))
    md += ["## 3. 波段對照 (收斂比 ≤0.8; 來回 0.6%, 不處理漲跌停)", "", table(rows, C), "",
           "## 4. 假突破 (收斂比 ≤0.8, 向上突破)", "",
           f"- 30 分後仍在上緣之上 {u.c30_hold.mean()*100:.0f}%；收盤在上緣之上 {(u.auc_c > u.res).mean()*100:.0f}%；收盤高於突破價 {(u.auc_c > u.up_px).mean()*100:.0f}%；收盤 ≥ 昨收+5% {(u.auc_c/u.pc-1 >= 0.05).mean()*100:.1f}%；收盤漲停 {(u.auc_c >= u.pc*1.1*0.999).mean()*100:.1f}%。",
           f"- 向下突破: 收盤在下緣之下 {(dn.auc_c < dn.sup).mean()*100:.0f}%；收盤低於跌破價 {(dn.auc_c < dn.dn_px).mean()*100:.0f}%。", ""]
    (STATS / "triangle_blocks_backtest.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
