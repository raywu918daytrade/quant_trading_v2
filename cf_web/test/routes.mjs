// fade-web 路由測試 (本機 / CI, 不連網、不推 LINE): node cf_web/test/routes.mjs
// 1. 每個網頁路由都回 200 + 正確內容, repo 其他檔案不會被公開
// 2. POST = LINE webhook: 簽章錯 → 401; 同一批事件丟給 fade-web 與舊 orb-line-bot (line_bot/worker.js), 送出的 reply 內容必須一模一樣
// 3. /api/health、/api/hf 代理 (白名單、快取)
// 有 LINE_CHANNEL_ACCESS_TOKEN 時, 再把 fade-web 產生的 reply 丟給 LINE 的 validate API 檢查格式 (不會送出, 不算額度)。
import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";

const here = path.dirname(new URL(import.meta.url).pathname), cf = path.join(here, ".."), fx = path.join(cf, "..", "line_bot", "test");
execFileSync(process.execPath, [path.join(cf, "build.mjs")], { stdio: "inherit" });
const pub = path.join(cf, "public");
const { default: web } = await import("../worker.js");
const { default: oldBot } = await import("../../line_bot/worker.js");

// ---- 假的 Cloudflare 環境 ----
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".json": "application/json", ".png": "image/png" };
const ASSETS = { async fetch(req) {   // 模擬 html_handling = "none": 只有檔名完全對上才回
  const p = decodeURIComponent(new URL(req.url).pathname);
  const f = path.join(pub, p);
  if (!f.startsWith(pub + path.sep) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) return new Response("Not Found", { status: 404 });
  return new Response(req.method === "HEAD" ? null : fs.readFileSync(f), { headers: { "Content-Type": TYPES[path.extname(f)] || "application/octet-stream" } });
} };
const store = new Map();
globalThis.caches = { default: { match: async (k) => store.get(k.url)?.clone(), put: async (k, r) => void store.set(k.url, r) } };
const realFetch = globalThis.fetch;
const today = new Date(Date.now() + 8 * 3600e3).toISOString().slice(0, 10);
const fixture = (f) => { const j = JSON.parse(fs.readFileSync(path.join(fx, f))); if (j.date) j.date = today; return JSON.stringify(j); };
const FIX = { "live.json": fixture("live.json"), "paper/live.json": fixture("paper.json"), "web/live_index.json": fixture("live_index.json") };
let replies = [], upstream = 0;
globalThis.fetch = async (input, init = {}) => {
  const u = new URL(typeof input === "string" ? input : input.url);
  if (u.hostname === "api.line.me") { replies.push(JSON.parse(init.body)); return new Response("{}"); }
  if (u.hostname === "huggingface.co") {
    upstream++;
    const k = u.pathname.split("/db/orb/")[1];
    return k in FIX ? new Response(FIX[k], { headers: { "Content-Type": "application/json" } }) : new Response("Entry not found", { status: 404 });
  }
  throw new Error("unexpected fetch " + u);
};
const SECRET = "test-secret", USER = "Utestowner";
const env = { ASSETS, LINE_CHANNEL_SECRET: SECRET, LINE_CHANNEL_ACCESS_TOKEN: "dummy", LINE_USER_ID: USER };
const call = async (worker, req) => {
  const waits = [];
  const res = await worker.fetch(req, env, { waitUntil: (p) => waits.push(p) });
  await Promise.all(waits);
  return res;
};
const BASE = "https://fade-web.example.workers.dev";
const get = (p, method = "GET") => call(web, new Request(BASE + p, { method }));

let bad = 0;
const check = (name, cond, extra = "") => { console.log(cond ? "✓" : "✗", name, cond ? "" : extra); if (!cond) bad++; };

// ---- 1. 網頁路由 ----
const PAGES = [   // [路徑, 頁面裡一定有的字]
  ["/", "web/index.html"], ["/index.html", "web/index.html"],
  ["/web/", "lightweight-charts"], ["/web/index.html", "lightweight-charts"], ["/web/index.html?data=x#live", "lightweight-charts"],
  ["/web/today.html", "paper/live.json"], ["/web/chart.html?d=2026-11-20&s=2603", "lightweight-charts"],
];
for (const [p, needle] of PAGES) {
  const r = await get(p), t = await r.text();
  check(`GET ${p} → 200 html`, r.status === 200 && r.headers.get("content-type").startsWith("text/html") && t.includes(needle), `${r.status} ${r.headers.get("content-type")}`);
}
check("HEAD /web/today.html → 200", (await get("/web/today.html", "HEAD")).status === 200);
// 每個 web/*.html 都要能開 (之後新增的頁面也自動涵蓋)
for (const f of fs.readdirSync(path.join(cf, "..", "web")).filter((f) => f.endsWith(".html")))
  check(`GET /web/${f} → 200`, (await get("/web/" + f)).status === 200);
for (const p of ["/CNAME", "/README.md", "/CLAUDE.md", "/scripts/orb_live.py", "/line_bot/worker.js", "/cf_web/worker.js", "/web/nope.html", "/../README.md"])
  check(`GET ${p} → 404 (不公開)`, (await get(p)).status === 404);

// ---- 2. LINE webhook ----
const sign = async (body, secret = SECRET) => {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return Buffer.from(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body))).toString("base64");
};
const hook = async (worker, events, { secret = SECRET, p = "/" } = {}) => {
  const body = JSON.stringify({ destination: "x", events });
  replies = [];
  const res = await call(worker, new Request(BASE + p, { method: "POST", body, headers: { "Content-Type": "application/json", "x-line-signature": await sign(body, secret) } }));
  return { status: res.status, text: await res.text(), replies };
};
const msg = (text, userId = USER, i = 0) => ({ type: "message", replyToken: `tok${i}`, source: { userId }, message: { type: "text", text } });
check("POST 簽章錯 → 401", (await hook(web, [msg("說明")], { secret: "wrong" })).status === 401);
const noSig = await call(web, new Request(BASE + "/", { method: "POST", body: "{}" }));
check("POST 沒簽章 → 401", noSig.status === 401);
check("PUT / → 405", (await call(web, new Request(BASE + "/", { method: "PUT", body: "x" }))).status === 405);
check("POST /api/health → 405 (不是 webhook)", (await call(web, new Request(BASE + "/api/health", { method: "POST", body: "{}" }))).status === 405);

const INPUTS = ["持倉", "損益", "今日", "狀態", "網頁", "績效", "說明", "/help", "2603", "9999", "亂打"];
const events = [...INPUTS.map((t, i) => msg(t, USER, i)), msg("說明", "Ustranger", 99),
  { type: "postback", replyToken: "pb", source: { userId: USER }, postback: { data: "狀態" } }, { type: "follow", source: { userId: USER } }];
const a = await hook(web, events), b = await hook(oldBot, events);
check("POST / → 200 ok", a.status === 200 && a.text === "ok");
check(`webhook 回覆數 ${a.replies.length} = 舊 bot ${b.replies.length} (= ${INPUTS.length + 2})`, a.replies.length === b.replies.length && a.replies.length === INPUTS.length + 2);
const norm = (rs) => rs.map((r) => JSON.stringify(r)).sort();
check("每則 reply 內容和舊 orb-line-bot 完全相同", JSON.stringify(norm(a.replies)) === JSON.stringify(norm(b.replies)));
check("非擁有者只收到「只回應擁有者」", a.replies.some((r) => r.replyToken === "tok99" && r.messages[0].text === "這個機器人只回應擁有者。"));
check("POST 到其他路徑也是 webhook (和舊 bot 一樣)", (await hook(web, [msg("說明")], { p: "/line/webhook" })).replies.length === 1);

// ---- 3. /api ----
const h = await get("/api/health");
check("GET /api/health", h.status === 200 && (await h.json()).worker === "fade-web");
check("GET /api/nope → 404", (await get("/api/nope")).status === 404);
upstream = 0;
const p1 = await get("/api/hf/live.json?t=1"), j1 = await p1.json();
const p2 = await get("/api/hf/live.json?t=2");
check("/api/hf/live.json = HF 的 live.json", p1.status === 200 && j1.date === today && JSON.stringify(j1) === FIX["live.json"]);
check("/api/hf 第二次走快取 (?t= 不算)", p2.status === 200 && upstream === 1, `upstream=${upstream}`);
check("/api/hf Cache-Control max-age=15", p1.headers.get("cache-control") === "public, max-age=15");
check("/api/hf/paper/live.json", (await get("/api/hf/paper/live.json")).status === 200);
check("/api/hf 找不到 → 404", (await get("/api/hf/live/1999-01-01/x.json")).status === 404);
for (const p of ["/api/hf/history/2026.parquet", "/api/hf/..%2F..%2Fconfig.json", "/api/hf/a%2F..%2Flive.json", "/api/hf/", "/api/hf/.env"])
  check(`GET ${p} → 400 (不代理)`, (await get(p)).status === 400);

// ---- 4. (選用) LINE validate API ----
const tok = process.env.LINE_CHANNEL_ACCESS_TOKEN;
if (tok) {
  for (const r of a.replies) {
    const v = await realFetch("https://api.line.me/v2/bot/message/validate/reply", { method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${tok}` }, body: JSON.stringify({ messages: r.messages }) });
    check(`LINE validate ${r.replyToken}`, v.ok, (await v.text()).slice(0, 300));
  }
} else console.log("(沒有 LINE_CHANNEL_ACCESS_TOKEN, 跳過 LINE validate)");

console.log(bad ? `${bad} 項失敗` : "全部通過");
process.exit(bad ? 1 : 0);
