"""用 cron-job.org API 建立/更新 v2 的定時工作 (平日, 台灣時間), 觸發 GitHub workflow_dispatch:
  08:50 live.yml     盤中監看 (腳本自己等到 09:02 / 09:05, 跑到 13:31)
  21:00 nightly.yml  歷史回放補新交易日 + 匯出網頁資料
  08:10 nightly.yml  補跑 (前一晚 HF 分 K 還沒更新時)
需要兩把 token (值絕不印出):
  CRONJOB_API_KEY     cron-job.org → Settings → API 產生的 key (可沿用 v1 那把)
  GH_DISPATCH_TOKEN   GitHub fine-grained token, Repository access 只選 quant_trading_v2, 權限 Actions: Read and write
用法:
  CRONJOB_API_KEY=... GH_DISPATCH_TOKEN=... python3 scripts/setup_cronjob.py [--test]
  python3 scripts/setup_cronjob.py --env-file /path/to/.env [--test]      # .env 內放上面兩個變數
--test 會先直接觸發一次 nightly.yml 驗證 GitHub token。
"""
import os, sys, json, requests

OWNER, REPO = "raywu918daytrade", "quant_trading_v2"
JOBS = [("fade v2 live 08:50", "live.yml", 8, 50), ("fade v2 nightly 21:00", "nightly.yml", 21, 0), ("fade v2 nightly retry 08:10", "nightly.yml", 8, 10)]
API = "https://api.cron-job.org"
def dispatch_url(wf): return f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{wf}/dispatches"

def read_env_file(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        out[k] = v
    return out

def main():
    env = dict(os.environ)
    for i, a in enumerate(sys.argv):
        if a == "--env-file": env.update(read_env_file(sys.argv[i + 1]))
    key = env.get("CRONJOB_API_KEY") or env.get("CRONJOB_API_KEY_DAY_TRADE"); gh = env.get("GH_DISPATCH_TOKEN")
    if not key or not gh: sys.exit("缺 CRONJOB_API_KEY 或 GH_DISPATCH_TOKEN")
    if "--test" in sys.argv:
        r = requests.post(dispatch_url("nightly.yml"), headers={"Authorization": f"Bearer {gh}", "Accept": "application/vnd.github+json"}, json={"ref": "main"}, timeout=20)
        print("GitHub dispatch 測試:", r.status_code, "(204 = 成功觸發 nightly.yml)" if r.status_code == 204 else r.text[:200])
        if r.status_code != 204: sys.exit(f"這把 GitHub token 無法觸發 {OWNER}/{REPO} (需 fine-grained token, 只選 {REPO}, Actions: Read and write)")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.get(f"{API}/jobs", headers=h, timeout=20); r.raise_for_status()
    existing = {j.get("title"): j["jobId"] for j in r.json().get("jobs", [])}
    for title, wf, hh, mm in JOBS:
        job = {"title": title, "url": dispatch_url(wf), "enabled": True, "saveResponses": True, "requestMethod": 1,
               "extendedData": {"headers": {"Authorization": f"Bearer {gh}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"}, "body": '{"ref":"main"}'},
               "schedule": {"timezone": "Asia/Taipei", "expiresAt": 0, "hours": [hh], "minutes": [mm], "mdays": [-1], "months": [-1], "wdays": [1, 2, 3, 4, 5]}}
        if title in existing:
            jid = existing[title]; requests.patch(f"{API}/jobs/{jid}", headers=h, json={"job": job}, timeout=20).raise_for_status(); print(f"已更新 job {jid}: {title}")
        else:
            r = requests.put(f"{API}/jobs", headers=h, json={"job": job}, timeout=20); r.raise_for_status(); jid = r.json().get("jobId"); print(f"已建立 job {jid}: {title}")
        jd = requests.get(f"{API}/jobs/{jid}", headers=h, timeout=20).json().get("jobDetails", {})
        print("  schedule:", json.dumps(jd.get("schedule"), ensure_ascii=False), "| enabled:", jd.get("enabled"))

if __name__ == "__main__":
    main()
