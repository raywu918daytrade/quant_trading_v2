// wrangler deploy / dev 前自動執行 ([build] command): 把要公開的網站檔案複製到 public/。
// 只複製白名單 (index.html + web/), repo 其他東西 (scripts/、CNAME、CLAUDE.md...) 不會被公開。
import fs from "fs";
import path from "path";
const here = path.dirname(new URL(import.meta.url).pathname), root = path.join(here, "..");
const out = path.join(here, "public");
export const FILES = ["index.html", "web"];
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });
for (const f of FILES) fs.cpSync(path.join(root, f), path.join(out, f), { recursive: true });
const n = (d) => fs.readdirSync(d, { withFileTypes: true }).reduce((a, e) => a + (e.isDirectory() ? n(path.join(d, e.name)) : 1), 0);
console.log(`build: ${n(out)} files → cf_web/public/`);
