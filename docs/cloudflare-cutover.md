# fade 網站搬到 Cloudflare (fade-web) — 搬遷計畫

2026-11-21 準備 (分支 `local/cf-web`, 從 `origin/main` 切出, 本機 commit, 沒 push、沒部署)。
**時機: 11-23 (一) fade-live 切到 k3s 穩定之後才開始。** 每一步都要 Ray 核准, 每一步都有回復方式。
試點只搬 fade; quant、watch365 照舊 (GitHub Pages + 各自的 Worker), 不動。

## 現況 → 目標

| | 現在 | 搬完 |
|---|---|---|
| 網站 | GitHub Pages (repo 根 `CNAME`, `index.html` → `web/index.html`, `web/chart.html`, `web/today.html`) | Worker `fade-web` 的 static assets (同一份檔案, 網址不變) |
| LINE webhook | Worker `orb-line-bot` (`https://orb-line-bot.raywu918daytrade.workers.dev`, 由 `line_bot.yml` 部署) | 同一個 Worker `fade-web` (`https://fade-web.raywu918daytrade.workers.dev/`) |
| 資料 | 網頁在瀏覽器直接讀 HF `db/orb/...` | 不變 (另有選用的 `/api/hf/` 邊緣快取代理, 預設沒用) |
| DNS | GoDaddy (`ns53/ns54.domaincontrol.com`) | Cloudflare 免費方案 (網域註冊仍在 GoDaddy, 不轉移) |
| `fade.just1stock.com` | CNAME → `raywu918daytrade.github.io` | Worker Custom Domain → `fade-web` |

## 程式 (這個分支)

- `cf_web/worker.js` — 路由:
  - `GET/HEAD` 一般路徑 → static assets。檔案由 `cf_web/build.mjs` 在 `wrangler deploy/dev` 前把 repo 的 `index.html` + `web/` 複製到 `cf_web/public/` (白名單; `CNAME`、`scripts/`、`README.md` 等不會公開)。結尾 `/` 補 `index.html`, `.html` 不轉址 → 網址和 GitHub Pages 完全一樣 (`/web/today.html`、`/web/chart.html?d=..&s=..`、`/web/index.html#live`)。
  - `POST` (非 `/api`) → `line_bot/worker.js` 原封不動 (import 同一個檔案, 不是複製)。指令、Flex 卡片、Quick Reply、簽章驗證、只回 `LINE_USER_ID` 全部一樣。舊 bot 也是「任何路徑的 POST 都是 webhook」, 這裡保留。唯一差別: `GET /` 以前回 `orb LINE bot OK`, 現在回網站首頁。
  - `/api/*` — 以後的 API 命名空間。現在只有 `/api/health` 與 `/api/hf/<路徑>` (HF `db/orb/` 下 `.json`/`.csv` 的唯讀代理, 邊緣快取 15 秒, 網頁加的 `?t=` 不算進快取 key)。**網頁預設沒改, 照舊直接讀 HF**; 想試: `today.html?live=/api/hf/live.json&paper=/api/hf/paper/live.json`。Cache API 在 `*.workers.dev` 不生效, 綁自訂網域後才真的快取。
  - 注意: `web/index.html` 在 `localhost` / `127.0.0.1` 會改讀本機 `webapp.py` 的 `/api/days` 等 → 用 `wrangler dev` 看網頁時請開 `http://fade.localhost:8787/` (不是 localhost, 才會讀 HF)。正式網域不受影響。
- `cf_web/wrangler.toml` — `name = "fade-web"`, 同一個 account, `run_worker_first = true`, `html_handling = "none"`; Custom Domain 那行先註解。
- `cf_web/test/routes.mjs` — `node cf_web/test/routes.mjs` (不連網): 每個網頁路由 200、repo 其他檔案 404、POST 簽章錯 401、**同一批 13 個事件丟給 fade-web 和舊 orb-line-bot, 送出的 reply 內容逐字相同**、`/api` 檢查。有 `LINE_CHANNEL_ACCESS_TOKEN` 時再把每則 reply 丟 LINE validate API (不送出、不算額度)。
- `.github/workflows/cf_web.yml` — **只能手動 Run workflow**: validate.mjs → routes.mjs → wrangler-action 部署 (`wranglerVersion 4.159.0`, 同一個 `CLOUDFLARE_API_TOKEN` + 三個 LINE secrets) → 對部署網址 curl 每個路由。勾 `set_webhook` 才會把 LINE webhook 改到 fade-web。push 觸發先註解, c1 之後才打開。

### 2026-11-21 本機測過
- `node line_bot/test/validate.mjs` (LINE validate API): 9 個指令全部 ✓
- `line_bot/test/validate_push.py` (LINE validate API): 每種推播卡片 200
- `node cf_web/test/routes.mjs` + LINE validate: 全部通過 (13 則 reply validate ✓)
- `wrangler dev` (4.159.0, 假 secrets): 6 個網頁路由 200、`CNAME`/`README.md`/`scripts/` 404、`/api/hf/live.json` 和 HF 原檔逐位元組相同、POST 正確簽章 200 / 錯簽章 401 / PUT 405; 瀏覽器開 `fade.localhost:8787` 首頁 (資金曲線)、`today.html` (經 `/api/hf`)、`chart.html` (K 線) 都正常, console 無錯誤
- `wrangler deploy --dry-run`: build 與 assets 設定 OK (沒上傳)

---

## 步驟 a — 部署 fade-web, 在 workers.dev 測 (舊的全部不動)

前提: 這個分支併進 `main` 並 push (cf_web.yml 只有 workflow_dispatch, push 不會自動部署; GitHub Pages 會照常重建, 內容沒變)。

1. GitHub → Actions → **fade-web deploy** → Run workflow (`set_webhook` 不勾)。
   - 會在 Cloudflare 建新的 Worker `fade-web` 並設三個 secrets; `orb-line-bot`、LINE webhook、DNS、GitHub Pages 都不動。
   - workflow 最後一步會 curl 每個路由, 全 ✓ 才算成功。
2. 手機 / 桌機開 `https://fade-web.raywu918daytrade.workers.dev/` → 首頁、`/web/today.html`、`/web/index.html#live`、某天的 `chart.html` 看起來要和 `https://fade.just1stock.com/` 一樣 (盤中看 today.html 20 秒更新)。
3. (選) `https://fade-web.raywu918daytrade.workers.dev/web/today.html?live=/api/hf/live.json&paper=/api/hf/paper/live.json` 和沒帶參數的版本應顯示相同內容。

**回復**: Cloudflare dashboard → Workers & Pages → `fade-web` → Settings → Delete (或 `npx wrangler delete --name fade-web`)。沒有任何東西依賴它。

## 步驟 c2 — LINE webhook 換到 fade-web (可在 b 之前做, 和 DNS 無關)

webhook 指到 `*.workers.dev`, 不依賴 just1stock.com 的 DNS, 所以 DNS 出問題也不影響機器人。**建議盤後或假日做。**

1. 同一個 commit 改兩個 workflow, 避免舊 workflow 把 webhook 改回去:
   - `line_bot.yml`: 刪掉 `push:` 觸發 (只留 workflow_dispatch → 它變成「切回舊 bot」的回復按鈕)。⚠ 不刪的話, 之後任何 `line_bot/**` 的 push 都會重新部署 orb-line-bot **並把 webhook 設回舊 Worker**。
   - `cf_web.yml`: 打開 `push:` 觸發 (改 `line_bot/worker.js` 就會部署 fade-web)。
2. Actions → **fade-web deploy** → Run workflow, 勾 `set_webhook`。它跑 `setup_line.py webhook https://fade-web.raywu918daytrade.workers.dev/` 並用 LINE 的 webhook test (不算額度) 測連線, log 要看到「webhook 測試: 成功」。rich menu 不動 (按鈕連的網址沒變)。
3. 手機傳「說明」「今日」「績效」「2330」→ 卡片要和以前一樣 (reply 免費, 不吃 200 則 push 額度)。Cloudflare dashboard → fade-web → Logs 看得到請求。

**回復**: Actions → **LINE bot deploy** → Run workflow (重新部署 orb-line-bot 並把 webhook 設回 `https://orb-line-bot.raywu918daytrade.workers.dev`)。或本機: `set -a; . ./.env; set +a; python3 line_bot/setup_line.py webhook https://orb-line-bot.raywu918daytrade.workers.dev`。兩個 Worker 都在, 切換是即時的。

## 步驟 b — just1stock.com 的 DNS 從 GoDaddy 搬到 Cloudflare (免費方案)

只換 nameserver, 網域註冊留在 GoDaddy。目標是「搬完每個名字解析結果和現在一模一樣」, 所以所有紀錄先設 **DNS only (灰雲)**, 不經 Cloudflare proxy。

### 2026-11-21 從外面查得到的紀錄 (直接問 `ns53.domaincontrol.com`)

| 名稱 | 類型 | 值 | 用途 | Cloudflare 上 |
|---|---|---|---|---|
| `just1stock.com` (@) | A | `216.24.57.1` | Render (`server: cloudflare`, Render 自己的邊緣) | 重建, DNS only |
| `www` | CNAME | `just1stock.onrender.com` | Render, 301 → apex | 重建, DNS only |
| `day-trade` | CNAME | `vwap-9dqj.onrender.com` | Render (vwap) | 重建, DNS only |
| `fade` | CNAME | `raywu918daytrade.github.io` | fade GitHub Pages | 重建, DNS only (步驟 c1 才換) |
| `quant` | CNAME | `raywu918daytrade.github.io` | quant GitHub Pages | 重建, DNS only |
| `watch365` | CNAME | `raywu918daytrade.github.io` | watch365 GitHub Pages | 重建, DNS only |
| `_dmarc` | TXT | `v=DMARC1; p=quarantine; adkim=r; aspf=r; rua=mailto:dmarc_rua@onsecureserver.net;` | GoDaddy 預設 DMARC | 照抄 (無害; 這個網域沒寄信) |
| `_domainconnect` | CNAME | `_domainconnect.gd.domaincontrol.com` | GoDaddy 專用 (Domain Connect) | **不用建** |
| @ | NS / SOA | `ns53/ns54.domaincontrol.com` | | Cloudflare 自己的, 不用建 |

查過但沒有: apex AAAA / MX / TXT (沒有 SPF) / CAA、DS (**沒開 DNSSEC**, 不用先關)、萬用字元 (`*`)、`www` 以外約 40 個常見名稱 (mail、api、app、status、dev、staging、`_acme-challenge`、`default._domainkey`、`google._domainkey`、`_github-pages-challenge-*` 等)。

### 從外面看不到的 (搬之前一定要在 GoDaddy 後台看)
- **完整紀錄清單**: GoDaddy 拒絕 zone transfer (AXFR), 我只能猜名稱。名字沒猜到的子網域、其他 DKIM selector、SRV 都看不到 → GoDaddy → 網域 → DNS → **匯出 Zone File** 存一份 (也是回復用的備份), 逐行對照上表與 Cloudflare 掃到的結果。
- GoDaddy 的「轉址 (Forwarding)」、Parked 頁: 不是 DNS 紀錄, 換 nameserver 後會失效。apex 指向 Render (不是 GoDaddy 轉址 IP), 看起來沒用, 但請在後台確認。
- 各紀錄的 TTL (apex 是 600 秒, 其他沒逐一記)。
- Render / GitHub 端的網域設定 (不在 DNS 裡; 只要解析結果不變就不受影響)。

### 步驟
1. GoDaddy 匯出 Zone File (備份)。
2. Cloudflare dashboard → Add a domain → `just1stock.com` → Free。它會自動掃描紀錄 → 對照上表: 缺的補上、`_domainconnect` 刪掉、**每一筆的 Proxy status 都改成 DNS only (灰雲)**。
   - 為什麼灰雲: Render 本身就在 Cloudflare 後面且自己簽憑證; GitHub Pages 要靠流量直達 GitHub 才能續 Let's Encrypt 憑證。灰雲 = 和現在完全相同的行為。
3. 換 nameserver 前驗證 Cloudflare 那邊的答案 (Cloudflare 會給兩個 `xxx.ns.cloudflare.com`):
   ```bash
   CF=<cloudflare 給的第一個 ns>
   while read t n; do printf "%-6s %-30s " $t $n; dig +short $t $n @$CF | tr '\n' ' '; echo; done <<'EOF'
   A just1stock.com
   CNAME www.just1stock.com
   CNAME day-trade.just1stock.com
   CNAME fade.just1stock.com
   CNAME quant.just1stock.com
   CNAME watch365.just1stock.com
   TXT _dmarc.just1stock.com
   EOF
   ```
   每一行都要和上表一樣。
4. GoDaddy → 網域 → Nameservers → 「使用我自己的 nameserver」→ 填 Cloudflare 給的兩個。.com 的 NS TTL 是 2 天, 生效期間有些解析器問 GoDaddy、有些問 Cloudflare —— 兩邊內容一樣所以不會斷。**GoDaddy 上的紀錄先不要刪。**
5. 等 Cloudflare 寄「just1stock.com is now active」(通常幾小時內)。驗證: `dig +short NS just1stock.com` 是 Cloudflare 的; `curl -sI https://fade.just1stock.com/ | grep server` 仍是 `GitHub.com`; quant / watch365 / day-trade / www / apex 都能開。
6. Cloudflare 其他設定先不動 (SSL/TLS、Always Use HTTPS 等只影響橘雲紀錄, 現在沒有橘雲)。

**回復**: GoDaddy → Nameservers → 改回「GoDaddy 預設 nameserver」(`ns53/ns54.domaincontrol.com`)。GoDaddy 的紀錄沒刪, 會直接恢復 (若不見了, 用第 1 步的 Zone File 匯回)。同樣最多 2 天生效, 期間兩邊內容一樣不會斷。Cloudflare 的 zone 等回復完成再刪。

## 步驟 c1 — fade.just1stock.com 綁到 fade-web (需要 b 已完成、Cloudflare 顯示 Active)

**建議盤後或假日做** (中間可能有 1~5 分鐘打不開)。

1. Cloudflare → Workers & Pages → `fade-web` → Settings → Domains & Routes → Add → Custom Domain → `fade.just1stock.com`。若提示該名稱已有 DNS 紀錄: 先到 DNS 刪掉 `fade` CNAME 再加。Cloudflare 會自建紀錄並簽憑證 (幾分鐘)。
   - 用 dashboard 加, 不用改 `CLOUDFLARE_API_TOKEN` (它現在只有 Account 權限, 沒有 Zone 權限)。`wrangler.toml` 的 `routes` 那行維持註解: 沒寫 routes 時 wrangler deploy 不會動已存在的網域。**第一次重新部署後確認 Custom Domain 還在**。(若想改成寫在 `wrangler.toml`: token 要加 just1stock.com 的 Zone → Workers Routes: Edit, 再打開那行。)
2. 驗證: `curl -sI https://fade.just1stock.com/web/today.html` → `server: cloudflare`、200; 首頁、`#live`、chart、today 都正常; LINE 卡片上的連結 (都是 `https://fade.just1stock.com/web/...`) 點得開。
3. 打開 `cf_web.yml` 的 `push:` 觸發 (若 c2 時還沒開)。之後改 `web/*.html` 要等 fade-web 部署才上線 (GitHub Pages 不再是正式站)。

**回復**: 刪掉 fade-web 的 Custom Domain → DNS 加回 `fade` CNAME `raywu918daytrade.github.io` (DNS only)。GitHub Pages 一直開著、repo 的 `CNAME` 還在, 所以幾分鐘就回到原狀。注意 GitHub Pages 的憑證到 **2027-01-25**, DNS 不指向 GitHub 時它無法續期 —— 回復若發生在那之後, HTTPS 可能要等 GitHub 重簽 (約 1 小時)。所以備援期不要拖過 2027-01 中。

## 步驟 c3 — 穩定 (建議 2 週, 含幾個交易日盤中) 後退役舊路徑

1. 停用 GitHub Pages: repo Settings → Pages → Source 改 None; 刪掉 repo 根的 `CNAME` (root `index.html` 留著, 它也是 fade-web 的首頁)。README「開發流程」那句「main 一更新 GitHub Pages 自動部署」改成 fade-web。
2. 刪 Worker `orb-line-bot` (dashboard 或 `npx wrangler delete --name orb-line-bot`); `line_bot.yml` 刪掉或改成只做 `richmenu`; `line_bot/wrangler.toml` 刪掉 (`line_bot/worker.js`、`setup_line.py`、`test/` 留著, fade-web 在用)。
3. GoDaddy 那邊的 DNS 紀錄可以清掉 (nameserver 已不是它)。

**回復**: 退役之後要回去就是重做 (GitHub Pages 重新開 + 加回 CNAME、LINE bot deploy 重新部署 orb-line-bot), 所以 c3 前要確定 c1、c2 都跑過一段時間。

---

## 之後 (不在這次範圍)
- DNS 在 Cloudflare 後, 可用 Cloudflare Tunnel + Access 把 k3s 上的狀態頁 (online status dashboard) 掛在 `status.just1stock.com` 之類, 不開任何對外 port。
- quant、watch365 若要照這個模式搬, 各自一個 Worker; DNS 已在 Cloudflare, 只差 c1/c2。
- (選) 在 GitHub 帳號設定驗證 `just1stock.com` (Settings → Pages → Verified domains), 避免子網域被別人的 Pages 認領。
