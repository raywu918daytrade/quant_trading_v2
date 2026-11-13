// LINE 指令機器人 (Cloudflare Worker)。LINE 把使用者傳的訊息 POST 到這裡 (webhook), 這裡讀 HF 上的 live.json / paper/live.json 後用 reply API 回覆。
// reply 訊息免費, 不吃每月 push 額度。只回 ALLOWED_USERS (預設 = LINE_USER_ID) 的訊息。
//
// 加新指令: 在 COMMANDS 加一筆 { keys, desc, run(ctx, arg) }, run 回傳 LINE message 物件陣列 (最多 5 個), 用 card({ title, sub, blocks, footer }) 組;
// blocks 用 kv() / stockRow() / tile() / note() / sep() 這些小積木, 外觀才會一致。
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
  { keys: ["網頁", "web", "w"], desc: "網頁　今日持倉頁 / 訊號頁 / 回測頁", run: () => [card({ title: "網頁", blocks: [link("今日持倉 / 損益", SITE + "today.html"), link("即時訊號 vs 模擬單", SITE + "index.html#live"), link("歷史回測", SITE + "index.html")] })] },
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
  // 已觸發 (state 非 WAIT) 但模擬環境還沒回成交 → 也列出來, 標「送單 待成交」(11-12: 聯一光 / 鈺創 09:05 送單, 09:08 還沒回報成交, 頁面上就像沒下單)
  return (paper?.positions || []).filter((r) => ok(r.sell_px) || (r.state !== "WAIT" && r.state !== "EXPIRED" && r.state !== "FULL")).map((r) => {
    const out = r.state === "STOPPED" || r.state === "CLOSED";   // 已停損 / 已收盤但券賣始終沒成交 (委託已取消): 不再算持有, 也不再顯示「待成交」
    if (!ok(r.sell_px)) return { sid: r.stock_id, name: r.name, hold: !out, pending: r.skipped ? `未下單 ${r.skipped}` : out ? "券賣未成交 (已取消)" : "送單 待成交", lots: r.lots, entry: r.rl, ref: r.last, rh: r.rh, pnl: null, pct: null };
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
  if (why) return [card({ title: withClosed ? "今日損益" : "目前持倉", sub: live ? `${live.date}` : "", blocks: [note(why)] })];
  const S = sigRows(live), M = simRows(paper);
  const total = (rs, f = () => true) => rs.filter(f).reduce((a, r) => a + (r.pnl || 0), 0);
  const pick = (rs) => (withClosed ? rs : rs.filter((r) => r.hold));
  const tiles = [tile("即時訊號", withClosed ? total(S) : total(S, (r) => r.hold), withClosed ? `持有 ${S.filter((r) => r.hold).length} / 進場 ${S.length}` : `持有 ${S.filter((r) => r.hold).length} 檔`)];
  const nHeld = M.filter((r) => r.hold && !r.pending).length, nWait = M.filter((r) => r.pending).length, wait = nWait ? `　待成交/未下單 ${nWait}` : "";
  if (paper) tiles.push(tile("永豐模擬單", withClosed ? total(M) : total(M, (r) => r.hold), (withClosed ? `持有 ${nHeld} / 成交 ${M.length - nWait}` : `持有 ${nHeld} 檔`) + wait));
  const blocks = [{ type: "box", layout: "horizontal", spacing: "md", contents: tiles }];
  const sec = (title, rs) => {
    const xs = pick(rs);
    blocks.push(sep(), label(title));
    blocks.push(...(xs.length ? xs.map((r) => stockRow(live.date, r)) : [note(withClosed ? "今天沒有進場" : "目前沒有持倉")]));
  };
  const at = (j) => (j?.updated_at ? `　(資料 ${j.updated_at.slice(11, 16)})` : "");   // 兩邊各自約 2~3 分鐘上傳一次, 標出資料時間免得以為是即時
  sec("即時訊號" + at(live), S);
  if (paper) sec("永豐模擬單" + at(paper), M);
  if (withClosed) {
    const sumLine = (rs) => `已實現 ${yuan(total(rs, (r) => !r.hold))}　未實現 ${yuan(total(rs, (r) => r.hold))}`;
    blocks.push(sep(), small(`訊號 ${sumLine(S)}` + (paper ? `\n模擬 ${sumLine(M)}` : "")));
  }
  const upd = live.updated_at ? live.updated_at.slice(11, 16) : "";
  return [card({ title: withClosed ? "今日損益" : "目前持倉", sub: `${live.date}　${live.phase}${upd ? "　" + upd + " 更新" : ""}`, blocks,
    footer: ["完整持倉頁", SITE + "today.html"], hint: "點股票看 K 線" })];
}
function status(ctx) {
  const { live, paper } = ctx;
  if (!live) return [card({ title: "系統狀態", blocks: [note("讀不到 live.json")] })];
  const rows = [kv("即時訊號", live.phase), kv("模擬單", paper ? paper.phase : "今天還沒有資料")];
  if (ok(live.mkt_gap)) rows.push(kv("大盤平均跳空", pct(live.mkt_gap * 100)));
  if (live.n_cand !== undefined) rows.push(kv("候選 / 監看", `${live.n_cand ?? "–"} / ${live.n_watch ?? "–"}`));
  if (live.updated_at) rows.push(kv("最後更新", live.updated_at.slice(11, 19)));
  if (live.note) rows.push(sep(), small(live.note));
  return [card({ title: "系統狀態", sub: live.date, blocks: rows, footer: ["今日訊號頁", SITE + "index.html#live"] })];
}
function stock(ctx, sid) {
  const { live } = ctx;
  const r = (live?.watch || []).find((x) => x.stock_id === sid);
  if (!r) return [card({ title: sid, blocks: [note("不在今天的監看名單")], footer: ["Yahoo 股市", `https://tw.stock.yahoo.com/quote/${sid}`] })];
  const p = ok(r.entry_px) ? pnl(r.entry_px, r.state === "SHORT" ? r.last : r.exit_px, r.lots) : null;
  const rows = [kv("狀態", r.state_txt), kv("進場線 / 停損線", `${f2(r.rl)} / ${f2(r.rh)}`), kv("現價", f2(r.last))];
  if (ok(r.entry_px)) rows.push(kv("放空", `${hm(r.entry_t)}　${f2(r.entry_px)} × ${r.lots} 張`));
  if (ok(r.exit_px)) rows.push(kv(r.state === "STOPPED" ? "停損" : "回補", `${hm(r.exit_t)}　${f2(r.exit_px)}`));
  if (ok(p)) rows.push(kv("損益", `${yuan(p)}（${pct(r.pnl_pct)}）`, color(p)));
  return [card({ title: `${r.stock_id} ${r.name}`, sub: live.date, blocks: rows, footer: ["看 K 線", chartUrl(live.date, sid)] })];
}
function help() {
  return card({ title: "指令", blocks: [...COMMANDS.map((c) => { const [k, ...d] = c.desc.split("　"); return kv(k, d.join(" ")); }), kv("四碼代號", "例如 2330，查今天狀態與 K 線")] });
}

// ---------------- LINE 訊息格式 (Flex) ----------------
const UP = "#d33a33", DN = "#1a9e6a", INK = "#121212", INK2 = "#6b6a65", LINE_C = "#ececE8";
const color = (v) => (ok(v) && v > 0 ? UP : ok(v) && v < 0 ? DN : INK);
const link = (t, uri) => ({ type: "button", style: "link", height: "sm", action: { type: "uri", label: t, uri } });
function text(t) { return { type: "text", text: t.slice(0, 5000) }; }
const sep = () => ({ type: "separator", color: LINE_C, margin: "md" });
const label = (t) => ({ type: "text", text: t, size: "xs", weight: "bold", color: INK2, margin: "md" });
const note = (t) => ({ type: "text", text: t, size: "sm", color: INK2, wrap: true, margin: "sm" });
const small = (t) => ({ type: "text", text: t, size: "xxs", color: INK2, wrap: true, margin: "sm" });
const kv = (k, v, c = INK) => ({ type: "box", layout: "horizontal", margin: "sm", contents: [
  { type: "text", text: k, size: "sm", color: INK2, flex: 4 }, { type: "text", text: String(v ?? "–"), size: "sm", color: c, flex: 6, align: "end", wrap: true }] });
function tile(name, v, sub) {
  return { type: "box", layout: "vertical", flex: 1, backgroundColor: "#f6f5f1", cornerRadius: "8px", paddingAll: "8px", contents: [
    { type: "text", text: name, size: "xxs", color: INK2 },
    { type: "text", text: yuan(v), size: "lg", weight: "bold", color: color(v) },
    { type: "text", text: sub, size: "xxs", color: INK2 }] };
}
function stockRow(d, r) {
  const st = r.hold ? "持有" : r.stop ? "停損" : "回補";
  return { type: "box", layout: "horizontal", margin: "md", action: { type: "uri", label: "K線", uri: chartUrl(d, r.sid) }, contents: [
    { type: "box", layout: "vertical", flex: 7, contents: [
      { type: "text", text: `${r.sid} ${r.name}`, size: "sm", weight: "bold", color: INK },
      { type: "text", text: r.pending ? `${r.pending}　區間低 ${f2(r.entry)} ${r.lots}張` : `${st} ${f2(r.entry)}→${f2(r.ref)} ${r.lots}張`, size: "xxs", color: INK2, wrap: true }] },
    { type: "box", layout: "vertical", flex: 3, contents: [
      { type: "text", text: yuan(r.pnl), size: "sm", weight: "bold", color: color(r.pnl), align: "end" },
      { type: "text", text: pct(r.pct), size: "xxs", color: color(r.pnl), align: "end" }] }] };
}
function card({ title, sub = "", blocks, footer, hint }) {
  const bubble = { type: "bubble", size: "mega",
    header: { type: "box", layout: "vertical", paddingBottom: "6px", contents: [
      { type: "box", layout: "horizontal", contents: [{ type: "text", text: title, weight: "bold", size: "md", color: INK, flex: 0 }, ...(hint ? [{ type: "text", text: hint, size: "xxs", color: INK2, align: "end", gravity: "center" }] : [])] },
      ...(sub ? [{ type: "text", text: sub, size: "xxs", color: INK2 }] : [])] },
    body: { type: "box", layout: "vertical", paddingTop: "0px", contents: blocks } };
  if (footer) bubble.footer = { type: "box", layout: "vertical", paddingTop: "0px", contents: [{ type: "button", style: "link", height: "sm", action: { type: "uri", label: footer[0], uri: footer[1] } }] };
  return { type: "flex", altText: `${title} ${sub}`.slice(0, 400), contents: bubble };
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
