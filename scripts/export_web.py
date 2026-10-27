#!/usr/bin/env python3
"""把歷史交易匯出成靜態網頁用的 JSON (雲端網頁直接從 HF 讀, 不需伺服器).
輸出 data/web_export/: index.json (統計 + 每日摘要 + 資金曲線), days/<YYYY-MM-DD>.json (當日摘要、進場股票、其他監看、每檔 1 分 K 與近 60 日日 K)。
用法:
  python3 scripts/export_web.py --all                 # 全部日子 (首次)
  python3 scripts/export_web.py --dates 2026-10-27    # 指定日子 (可多個)
  python3 scripts/export_web.py --since 2026-10-23    # 某日之後
  HF_TOKEN=... python3 scripts/export_web.py --all --upload    # 匯出後上傳到 HF db/orb/web/ (分批 commit)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import webapp as W                       # 載入歷史與 D1 (不會啟動伺服器)
import orb_live as L
from config import ROOT, HF_REPO, HF_PREFIX

OUT = ROOT / "data" / "web_export"; (OUT / "days").mkdir(parents=True, exist_ok=True)
DAILY_N = 60


def r2(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 2)


def bars_for_month(ym, pairs):
    """pairs: set of (stock_id, date_str). 回傳 {(sid, date): [[t,o,h,l,c,v], ...]}"""
    sids = sorted({s for s, _ in pairs})
    m = pd.read_parquet(L.m1_path(ym), filters=[("stock_id", "in", sids)])
    m["ts"] = pd.to_datetime(m.date); m["d"] = m.ts.dt.strftime("%Y-%m-%d"); m["t"] = m.ts.dt.hour * 60 + m.ts.dt.minute - 540
    m = m[(m.t >= 0) & (m.t <= 270)]
    m = m[[(s, d) in pairs for s, d in zip(m.stock_id, m.d)]].sort_values(["stock_id", "d", "t"])
    out = {}
    for (s, d), g in m.groupby(["stock_id", "d"], sort=False):
        out[(s, d)] = [[int(t), r2(o), r2(h), r2(l), r2(c), int(v)] for t, o, h, l, c, v in zip(g.t, g.open, g.high, g.low, g.close, g.volume)]
    return out


def stock_info(d, sid):
    r = W.TR[(W.TR.date == d) & (W.TR.stock_id == sid)]
    w = W.WT[(W.WT.date == d) & (W.WT.stock_id == sid)]
    src = r if len(r) else w
    info = {k: W.clean(v) for k, v in src.iloc[0].to_dict().items()} if len(src) else {}
    if len(w): info.update({k: W.clean(w.iloc[0][k]) for k in ["pc", "gap", "atr20"] if k in w.columns})
    if info.get("pc"):
        info["limit_up"] = round(L.limit_up(info["pc"]), 2); info["force_px"] = round(info["pc"] * 1.08, 2)
    info.pop("date", None)
    return info


def export_dates(dates):
    dates = sorted(pd.Timestamp(x) for x in dates)
    written = []
    for ym, grp in pd.Series(dates).groupby(pd.Series(dates).dt.strftime("%Y_%m")):
        day_objs, pairs = {}, set()
        for d in grp:
            j = W.api_day(str(d.date()))
            if "error" in j: continue
            sids = [x["stock_id"] for x in j["trades"]] + [x["stock_id"] for x in j["others"]]
            day_objs[d] = (j, sids); pairs |= {(s, str(d.date())) for s in sids}
        bars = bars_for_month(ym, pairs) if pairs else {}
        for d, (j, sids) in day_objs.items():
            ds = str(d.date()); stocks = {}
            for sid in sids:
                g = W.D1_BY.get(sid)
                daily = []
                if g is not None:
                    g = g[g.date <= d].tail(DAILY_N)
                    daily = [[str(x.date()), r2(o), r2(h), r2(l), r2(c), int(v)] for x, o, h, l, c, v in zip(g.date, g.open, g.high, g.low, g.close, g.volume)]
                stocks[sid] = dict(i=bars.get((sid, ds), []), d=daily, info=stock_info(d, sid))
            j["bars"] = stocks
            p = OUT / "days" / f"{ds}.json"
            p.write_text(json.dumps(j, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"); written.append(p)
        print(f"{ym}: {len(day_objs)} 天", flush=True)
    return written


def export_index():
    p = OUT / "index.json"
    p.write_text(json.dumps(W.api_days(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return p


def upload(paths, batch=300, msg="orb web"):
    from huggingface_hub import HfApi, CommitOperationAdd
    tok = os.environ.get("HF_TOKEN")
    if not tok: sys.exit("缺環境變數 HF_TOKEN (需 HF 寫入權限)")
    api = HfApi(token=tok)
    paths = list(paths)
    for i in range(0, len(paths), batch):
        ops = [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/web/{p.relative_to(OUT).as_posix()}", path_or_fileobj=str(p)) for p in paths[i:i + batch]]
        for k in range(3):
            try:
                api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops, commit_message=f"{msg} ({i + len(ops)}/{len(paths)})"); break
            except Exception as e:
                print(f"上傳失敗 ({k + 1}/3): {e}"); time.sleep(10)
        print(f"已上傳 {min(i + batch, len(paths))}/{len(paths)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true"); ap.add_argument("--since"); ap.add_argument("--dates", nargs="*")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    all_dates = list(W.DAYS.date)
    if a.all: dates = all_dates
    elif a.since: dates = [d for d in all_dates if d >= pd.Timestamp(a.since)]
    else: dates = [pd.Timestamp(x) for x in (a.dates or [])]
    t0 = time.time()
    files = export_dates(dates) + [export_index()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"匯出 {len(files)} 個檔案, {size:.1f} MB, {time.time() - t0:.0f} 秒 → {OUT}")
    if a.upload: upload(files)
