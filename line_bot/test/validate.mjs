// 用範例資料產生每個指令的回覆, 丟給 LINE 的 validate API 檢查格式 (不會真的送出)。CI 部署前跑; 本機: node line_bot/test/validate.mjs
// 沒有 LINE_CHANNEL_ACCESS_TOKEN 時只印出 JSON。
import fs from "fs";
import { handle } from "../worker.js";
const dir = new URL(".", import.meta.url).pathname;
const live = JSON.parse(fs.readFileSync(dir + "live.json")), paper = JSON.parse(fs.readFileSync(dir + "paper.json"));
const today = new Date(Date.now() + 8 * 3600e3).toISOString().slice(0, 10);
live.date = paper.date = today;
const index = JSON.parse(fs.readFileSync(dir + "live_index.json"));
const ctx = async (need) => (need === "index" ? { index } : { live, paper });
const tok = process.env.LINE_CHANNEL_ACCESS_TOKEN;
let bad = 0;
for (const q of ["持倉", "損益", "狀態", "網頁", "績效", "說明", "2603", "9999"]) {
  const messages = await handle(q, ctx);
  if (!tok) { console.log(q, JSON.stringify(messages).length, "bytes"); continue; }
  const r = await fetch("https://api.line.me/v2/bot/message/validate/reply", { method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${tok}` }, body: JSON.stringify({ messages }) });
  const t = await r.text();
  console.log(r.ok ? "✓" : "✗", q, r.ok ? "" : t.slice(0, 500)); if (!r.ok) bad++;
}
process.exit(bad ? 1 : 0);
