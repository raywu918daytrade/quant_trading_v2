#!/bin/bash
# fade 容器入口: 一個映像, 用第一個參數選工作 (K3s CronJob 的 args)。
#   live     orb_live.py refresh (D1 3 個月) → run --publish --line      (08:50 啟動, 跑到 13:31)
#   paper    refresh → paper_orb.py run --publish (濾掉含身分證字號的 Session Property 行)   (08:50 啟動, 跑到 13:35)
#   nightly  cloud_nightly.py                                              (21:00 / 08:10)
#   diag     refresh → diag_gap.py
#   shell    其餘參數當指令執行 (除錯用, 例: shell python3 scripts/orb_live.py replay --date 2026-10-22)
# 環境變數:
#   HC_URL          healthchecks.io ping 網址 (有設才 ping): 開始 /start, 成功 空, 失敗 /<exit code>
#   FADE_RUN_ARGS   附加給 run / cloud_nightly 的參數 (例 --force, --rebuild-live)
# 密鑰 (HF_TOKEN, LINE_*, SINOPAC_*) 只由程式從環境變數讀, 這裡不印出。
set -uo pipefail
cd /app
MODE="${1:-help}"; shift || true
EXTRA="${FADE_RUN_ARGS:-}"

hc() { [ -n "${HC_URL:-}" ] && curl -fsS -m 10 --retry 3 -o /dev/null "${HC_URL}$1" || true; }
refresh() { python3 scripts/orb_live.py refresh --months 3 --m1-months 0; }

hc /start
case "$MODE" in
  live)    refresh && python3 scripts/orb_live.py run --publish --line $EXTRA ;;
  paper)   refresh && { python3 scripts/paper_orb.py run --publish $EXTRA 2>&1 | grep --line-buffered -v "Session Property"; } ;;
  nightly) python3 scripts/cloud_nightly.py $EXTRA ;;
  diag)    refresh && python3 scripts/diag_gap.py ;;
  shell)   "$@" ;;
  *)       sed -n '2,12p' "$0"; exit 64 ;;
esac
rc=$?
if [ "$rc" -eq 0 ]; then hc ""; else hc "/$rc"; fi
exit "$rc"
