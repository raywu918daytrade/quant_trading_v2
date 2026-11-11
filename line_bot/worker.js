// LINE 指令機器人 (Cloudflare Worker)。LINE 把使用者傳的訊息 POST 到這裡 (webhook), 這裡讀 HF 上的 live.json / paper/live.json 後用 reply API 回覆。
// reply 訊息免費, 不吃每月 push 額度。只回 ALLOWED_USERS (預設 = LINE_USER_ID) 的訊息。
//
// 加新指令: 在 COMMANDS 加一筆 { keys, desc, run(ctx, arg) }, run 回傳 LINE message 物件陣列 (最多 5 個), 用 text() / withButtons() 組。
// 環境變數 (wrangler secret): LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, LINE_USER_ID (可逗號分隔多人)。

const HF = "https://huggingface.co/datasets/raywu918daytrade/daytrade/resolve/main/db/orb/";
const SITE = "https://fade.just1stock.com/web/";
const COST = 0.0023;
const ACTIVE = ["盤前", "等待 09:05", "監看中", "等待收盤", "收盤回補提醒"];

// ---------------- 指令表 ----------------
const COMMANDS = [
  { keys: ["持倉", "部位", "p", "pos"], desc: "持倉　目前持有中的部位 (訊號 + 模擬單)", run: (ctx) => positions(ctx, false) },
  { keys: ["損益", "pnl", "$"], desc: "損益　今日已實現 + 未實現損益", run: (ctx) => positions(ctx, true) },
  { keys: ["狀態", "status", "s"], desc: "狀態　程式現在跑到哪一步", run: status },
  { keys: ["網頁", "web", "w"], desc: "網頁　今日持倉頁 / 訊號頁 / 回測頁", run: () => [withButtons("網頁", [["📋 今日持倉 / 損益", SITE + "today.html"], ["📈 即時訊號 vs 模擬單", SITE + "index.html#live"], ["📊 歷史回測", SITE + "index.html"]])] },
  { keys: ["說明", "help", "?", "？", "指令"], desc: "說明　列出所有指令", run: () => [help()] },
];
// 四碼數字 = 查個股 (今天的監看狀態 + K 線按鈕)
const STOCK = /^\d{4}$/;

// ---------------- 資料 ----------------
async function getJSON(url) {
  const r = await fetch(url + "?t=" + Date.now(), { cf: { cacheTtl: 0 } });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
async function loadCtx() {
  const [live, paper] = await Promise.all([getJSON(HF + "live.json").catch(() => null), getJSON(HF + "paper/live.json").catch(() => null)]);
  return { live, paper: paper && live && paper.date === live.date ? paper : null };
}
const ok = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(+v);
const f2 = (v) => (ok(v) ? (+v).toFixed(2).replace(/\.?0+$/, "") : "–");
const yuan = (v) => (ok(v) ? (v > 0 ? "+" : "") + Math.round(v).toLocaleString("en-US") : "–");
const pct = (v) => (ok(v) ? (v > 0 ? "+" : "") + (+v).toFixed(2) + "%" : "–");
const pnl = (sell, ref, lots) => (ok(sell) && ok(ref) && lots ? (sell - ref) * lots * 1000 - COST * sell * lots * 1000 : null);
const hm = (t) => { if (!ok(t)) return ""; const s = Math.round(t * 60) + 9 * 3600; return `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}`; };
const chartUrl = (d, sid) => `${SITE}chart.html?d=${d}&s=${sid}`;

function sigRows(live) {
  return (live?.watch || []).filter((r) => ok(r.entry_px)).map((r) => {
    const hold = r.state === "SHORT", ref = hold ? r.last : r.exit_px;
    return { sid: r.stock_id, name: r.name, hold, stop: r.state === "STOPPED", lots: r.lots, entry: r.entry_px, t: r.entry_t, ref, rh: r.rh, pnl: pnl(r.entry_px, ref, r.lots), pct: r.pnl_pct };
  });
}
function simRows(paper) {
  return (paper?.positions || []).filter((r) => ok(r.sell_px)).map((r) => {
    const hold = !ok(r.buy_px), ref = hold ? r.last : r.buy_px;
    return { sid: r.stock_id, name: r.name, hold, stop: r.state === "STOPPED", lots: r.sell_lots, entry: +r.sell_px, ref, rh: r.rh, pnl: pnl(+r.sell_px, ref, r.sell_lots), pct: r.pnl_pct };
  });
}
function notTrading(live) {
  if (!live) return "讀不到今天的資料 (live.json)";
  const today = new Date(Date.now() + 8 * 3600e3).toISOString().slice(0, 10);
  if (live.date !== today) return `今天還沒開始跑 (最後一筆資料是 ${live.date} ${live.phase})`;
  if (["不交易", "非交易日"].includes(live.phase)) return `今日不交易${live.note ? "：" + live.note : ""}`;
  if (["盤前", "等待 09:05"].includes(live.phase)) return `${live.phase}，09:05 定區間後才會有部位`;
  return null;
}

// ---------------- 指令實作 ----------------
function positions(ctx, withClosed) {
  const { live, paper } = ctx;
  const why = notTrading(live);
  if (why) return [text(`📋 ${why}`)];
  const S = sigRows(live), M = simRows(paper);
  const sum = (rs, f) => rs.filter(f).reduce((a, r) => a + (r.pnl || 0), 0);
  const line = (r) => `${r.hold ? "🔵" : r.stop ? "🟠" : "⚪"} ${r.sid} ${r.name} ${r.lots}張\n　空 ${f2(r.entry)} → ${r.hold ? "現價" : "出場"} ${f2(r.ref)}　${yuan(r.pnl)} (${pct(r.pct)})`;
  const pick = (rs) => (withClosed ? rs : rs.filter((r) => r.hold));
  const head = `${withClosed ? "💰 今日損益" : "📋 目前持倉"}　${live.date} ${live.phase}${live.updated_at ? "　" + live.updated_at.slice(11, 16) : ""}`;
  const parts = [head];
  const sec = (title, rs) => {
    const xs = pick(rs);
    parts.push(`\n【${title}】` + (withClosed ? `　合計 ${yuan(sum(rs, () => true))} (已實現 ${yuan(sum(rs, (r) => !r.hold))}、未實現 ${yuan(sum(rs, (r) => r.hold))})` : `　持有 ${xs.length} 檔、未實現 ${yuan(sum(rs, (r) => r.hold))}`));
    parts.push(xs.length ? xs.map(line).join("\n") : withClosed ? "今天沒有進場" : "目前沒有持倉");
  };
  sec("即時訊號", S);
  if (paper) sec("永豐模擬單", M);
  const shown = [...new Map([...pick(S), ...pick(M)].map((r) => [r.sid, r])).values()];
  return [withButtons(parts.join("\n"), [...shown.slice(0, 8).map((r) => [`📈 ${r.sid} ${r.name} K線`, chartUrl(live.date, r.sid)]), ["📋 今日持倉頁", SITE + "today.html"]])];
}
function status(ctx) {
  const { live, paper } = ctx;
  if (!live) return [text("讀不到 live.json")];
  const lines = [`🤖 ${live.date} 即時訊號：${live.phase}${live.updated_at ? "（更新 " + live.updated_at.slice(11, 19) + "）" : ""}`];
  if (ok(live.mkt_gap)) lines.push(`大盤平均跳空 ${pct(live.mkt_gap * 100)}　候選 ${live.n_cand ?? "–"}　監看 ${live.n_watch ?? "–"}`);
  if (live.note) lines.push(live.note);
  lines.push(paper ? `模擬單：${paper.phase}` : "模擬單：今天還沒有資料");
  if (ACTIVE.includes(live.phase)) lines.push("（盤中約每 2~3 分鐘更新一次）");
  return [text(lines.join("\n"))];
}
function stock(ctx, sid) {
  const { live } = ctx;
  const r = (live?.watch || []).find((x) => x.stock_id === sid);
  if (!r) return [withButtons(`${sid} 不在今天的監看名單`, [[`📈 ${sid} K線 (Yahoo)`, `https://tw.stock.yahoo.com/quote/${sid}`]])];
  const p = ok(r.entry_px) ? pnl(r.entry_px, r.state === "SHORT" ? r.last : r.exit_px, r.lots) : null;
  const t = [`${r.stock_id} ${r.name}　${r.state_txt}`, `進場線 ${f2(r.rl)}　停損線 ${f2(r.rh)}　現價 ${f2(r.last)}`];
  if (ok(r.entry_px)) t.push(`${hm(r.entry_t)} 空 ${f2(r.entry_px)} × ${r.lots} 張` + (ok(r.exit_px) ? `　${hm(r.exit_t)} 出 ${f2(r.exit_px)}` : "") + `\n損益 ${yuan(p)} (${pct(r.pnl_pct)})`);
  return [withButtons(t.join("\n"), [[`📈 ${sid} ${r.name} K線`, chartUrl(live.date, sid)]])];
}
function help() {
  return text("可以打這些指令：\n" + COMMANDS.map((c) => "・" + c.desc).join("\n") + "\n・四碼代號 (例如 2330)　查這檔今天的狀態與 K 線");
}

// ---------------- LINE 訊息格式 ----------------
function text(t) { return { type: "text", text: t.slice(0, 5000) }; }
function withButtons(t, btns) {
  if (!btns.length) return text(t);
  return { type: "flex", altText: t.slice(0, 400), contents: { type: "bubble", size: "mega",
    body: { type: "box", layout: "vertical", contents: [{ type: "text", text: t.slice(0, 2000), wrap: true, size: "sm" }] },
    footer: { type: "box", layout: "vertical", spacing: "none", contents: btns.slice(0, 12).map(([label, uri]) => ({ type: "button", style: "link", height: "sm", action: { type: "uri", label: label.slice(0, 40), uri } })) } } };
}

// ---------------- webhook ----------------
async function verify(body, sig, secret) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body));
  const b64 = btoa(String.fromCharCode(...new Uint8Array(mac)));
  return sig && b64.length === sig.length && b64 === sig;
}
async function reply(env, token, messages) {
  const r = await fetch("https://api.line.me/v2/bot/message/reply", { method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.LINE_CHANNEL_ACCESS_TOKEN}` }, body: JSON.stringify({ replyToken: token, messages: messages.slice(0, 5) }) });
  if (!r.ok) console.log("reply failed", r.status, await r.text());
}
export async function handle(input, ctxLoader = loadCtx) {
  const s = input.trim().replace(/^[\/!！]/, "");
  const cmd = COMMANDS.find((c) => c.keys.some((k) => k.toLowerCase() === s.toLowerCase()));
  if (cmd) return cmd.run(await ctxLoader(), "");
  if (STOCK.test(s)) return stock(await ctxLoader(), s);
  return [help()];
}
export default {
  async fetch(req, env, ectx) {
    if (req.method !== "POST") return new Response("orb LINE bot OK");
    const body = await req.text();
    if (!(await verify(body, req.headers.get("x-line-signature"), env.LINE_CHANNEL_SECRET))) return new Response("bad signature", { status: 401 });
    const allowed = (env.ALLOWED_USERS || env.LINE_USER_ID || "").split(",").map((x) => x.trim()).filter(Boolean);
    const events = JSON.parse(body).events || [];
    ectx.waitUntil(Promise.all(events.map(async (e) => {
      if (!e.replyToken) return;
      if (allowed.length && !allowed.includes(e.source?.userId)) return reply(env, e.replyToken, [text("這個機器人只回應擁有者。")]);
      const input = e.type === "message" && e.message.type === "text" ? e.message.text : e.type === "postback" ? e.postback.data : "";
      if (!input) return;
      try { await reply(env, e.replyToken, await handle(input)); }
      catch (err) { await reply(env, e.replyToken, [text(`查詢失敗：${err.message}`)]); }
    })));
    return new Response("ok");   // LINE 要求 1 秒內回 200; 實際回覆在 waitUntil 裡做
  },
};
