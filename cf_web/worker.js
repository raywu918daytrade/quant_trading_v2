// fade-web: 網站 (Workers static assets) + LINE 指令機器人 (webhook) 放在同一個 Cloudflare Worker。
// 部署: GitHub Actions → fade-web deploy (.github/workflows/cf_web.yml)。搬遷步驟: docs/cloudflare-cutover.md
//
// 路由 (wrangler.toml 設 run_worker_first = true, 每個請求都先到這裡):
//   GET/HEAD /api/*  → api() (現在只有 /api/health 與 /api/hf/<路徑>; 以後的 API 都放這個命名空間)
//   GET/HEAD 其他    → 靜態檔 (build.mjs 從 repo 的 index.html + web/ 複製到 public/, 和 GitHub Pages 同一份檔案)
//   POST (非 /api)   → LINE webhook, 直接交給 line_bot/worker.js (同一份程式: 指令、Flex 卡片、簽章驗證、ALLOWED_USERS 都不變)
// 網址和 GitHub Pages 一樣: /web/index.html、/web/today.html、/web/chart.html?d=..&s=..; 結尾 / 補 index.html (html_handling = "none", 不轉址成無副檔名)。
import lineBot from "../line_bot/worker.js";

const HF = "https://huggingface.co/datasets/raywu918daytrade/daytrade/resolve/main/db/orb/";
const HF_TTL = 15;   // 秒。live.json 盤中每 2~3 分更新, 網頁每 20 秒重讀 → 最多晚 15 秒
// 只代理網頁會讀的 JSON / CSV, 不讓人拿這個 Worker 當任意 HF 下載器 (parquet 等大檔不代理)
const HF_PATH = /^(?:[A-Za-z0-9_-][A-Za-z0-9_.-]*\/)*[A-Za-z0-9_-][A-Za-z0-9_.-]*\.(?:json|csv)$/;

const json = (obj, status = 200, headers = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", ...headers } });

// HF 唯讀代理 (邊緣快取 HF_TTL 秒)。網頁預設仍直接讀 HF, 沒有改; 要試用: today.html?live=/api/hf/live.json&paper=/api/hf/paper/live.json
// 注意 Cache API 在 *.workers.dev 不生效 (每次都回源), 綁 fade.just1stock.com 之後才會快取。
async function hfProxy(req, path, ectx) {
  if (!HF_PATH.test(path) || path.split("/").some((p) => p === "." || p === "..")) return json({ error: "path not allowed" }, 400);
  const cache = caches.default;
  const key = new Request(new URL("/api/hf/" + path, req.url).toString(), { method: "GET" });   // 網頁加的 ?t= 不算進 key
  const hit = await cache.match(key);
  if (hit) return hit;
  const up = await fetch(HF + path, { cf: { cacheTtl: 0 } });
  if (!up.ok) return json({ error: `HF ${up.status}` }, up.status === 404 ? 404 : 502);
  const res = new Response(up.body, { status: 200, headers: {
    "Content-Type": path.endsWith(".csv") ? "text/csv; charset=utf-8" : "application/json; charset=utf-8",
    "Cache-Control": `public, max-age=${HF_TTL}`, "X-Fade-Proxy": "hf" } });
  ectx.waitUntil(cache.put(key, res.clone()));
  return res;
}

async function api(req, url, env, ectx) {
  const p = url.pathname;
  if (p === "/api/health") return json({ ok: true, worker: "fade-web", time: new Date().toISOString() });
  if (p.startsWith("/api/hf/")) return hfProxy(req, decodeURIComponent(p.slice("/api/hf/".length)), ectx);
  return json({ error: "not found" }, 404);
}

async function asset(req, url, env) {
  // GitHub Pages 行為: /web/ → /web/index.html。其他路徑原樣交給 ASSETS (找不到 → 404)
  if (url.pathname.endsWith("/")) {
    const u = new URL(url);
    u.pathname += "index.html";
    req = new Request(u.toString(), req);
  }
  return env.ASSETS.fetch(req);
}

export default {
  async fetch(req, env, ectx) {
    const url = new URL(req.url);
    const isApi = url.pathname === "/api" || url.pathname.startsWith("/api/");
    if (req.method === "GET" || req.method === "HEAD") return isApi ? api(req, url, env, ectx) : asset(req, url, env);
    if (req.method === "POST" && !isApi) return lineBot.fetch(req, env, ectx);   // 舊 orb-line-bot 也是「任何 POST = webhook」
    return new Response("method not allowed", { status: 405, headers: { Allow: isApi ? "GET, HEAD" : "GET, HEAD, POST" } });
  },
};
