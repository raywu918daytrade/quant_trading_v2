#!/usr/bin/env python3
"""雲端每晚更新 (GitHub Actions): 歷史回放補新交易日 → 匯出網頁 JSON → 上傳 HF.
步驟:
  1. 從 HF db/orb/history/ 下載 days/trades/watch.parquet (上次的歷史)
  2. orb_live.refresh: 最近 6 個月 D1 + 最近 2 個月 M1 + info
  3. build_history.py (增量; 沒有分 K 的日子會停下, 下次再補)
  4. export_web.py 匯出新日子的 days/<date>.json 與 index.json
  5. 一個 commit 上傳 history parquet + 新 JSON 到 HF
需環境變數 HF_TOKEN (HF 寫入權限)。本機也可跑 (會用本機已有的歷史, 不先下載): python3 scripts/cloud_nightly.py --local
首次部署 (本機, 一次): HF_TOKEN=... python3 scripts/cloud_nightly.py --seed   → 上傳本機歷史 parquet + data/web_export/ 全部 (先跑 export_web.py --all)
"""
import argparse, os, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, PROC, HF_REPO, HF_PREFIX

HIST = PROC / "orb_history"; S = Path(__file__).resolve().parent


def run(cmd):
    print("$", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--local", action="store_true", help="不從 HF 下載歷史"); ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--seed", action="store_true", help="首次部署: 上傳本機歷史與全部網頁檔")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    if not a.no_upload and not tok: sys.exit("缺環境變數 HF_TOKEN")
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    if a.seed:
        api = HfApi(token=tok)
        ops = [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/history/{k}.parquet", path_or_fileobj=str(HIST / f"{k}.parquet")) for k in ["days", "trades", "watch"]]
        api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops, commit_message="orb seed history"); print("已上傳歷史 parquet")
        sys.path.insert(0, str(S)); import export_web as E
        files = sorted((ROOT / "data" / "web_export" / "days").glob("*.json")) + [ROOT / "data" / "web_export" / "index.json"]
        E.upload(files, msg="orb seed web"); print(f"完成: {len(files)} 個網頁檔"); return
    HIST.mkdir(parents=True, exist_ok=True)
    if not a.local:
        for k in ["days", "trades", "watch"]:
            p = hf_hub_download(HF_REPO, f"{HF_PREFIX}/history/{k}.parquet", repo_type="dataset", force_download=True, token=tok)
            (HIST / f"{k}.parquet").write_bytes(Path(p).read_bytes())
    before = pd.read_parquet(HIST / "days.parquet").date.max()
    print(f"歷史最後一天: {before.date()}")
    run([sys.executable, "-W", "ignore", str(S / "orb_live.py"), "refresh", "--months", "6", "--m1-months", "2"])
    run([sys.executable, "-W", "ignore", str(S / "build_history.py")])
    after = pd.read_parquet(HIST / "days.parquet").date.max()
    if after <= before:
        print("沒有新的交易日 (分 K 可能還沒更新), 結束。"); return
    print(f"新增 {before.date()} 之後 ~ {after.date()}")
    run([sys.executable, "-W", "ignore", str(S / "export_web.py"), "--since", str((before + pd.Timedelta(days=1)).date())])
    if a.no_upload: return
    out = ROOT / "data" / "web_export"
    new_days = [p for p in (out / "days").glob("*.json") if pd.Timestamp(p.stem) > before]
    ops = [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/history/{k}.parquet", path_or_fileobj=str(HIST / f"{k}.parquet")) for k in ["days", "trades", "watch"]]
    ops += [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/web/days/{p.name}", path_or_fileobj=str(p)) for p in new_days]
    ops += [CommitOperationAdd(path_in_repo=f"{HF_PREFIX}/web/index.json", path_or_fileobj=str(out / "index.json"))]
    HfApi(token=tok).create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops, commit_message=f"orb nightly {after.date()}")
    print(f"已上傳: history 3 檔 + {len(new_days)} 天 + index.json")


if __name__ == "__main__":
    main()
