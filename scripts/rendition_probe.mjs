// Politely load ONE listing page per site and dump every image URL/srcset on it. node rendition_probe.mjs site url
import { createRequire } from 'node:module';
const require = createRequire('C:/repos/projects/auto card lister/package.json');
const { chromium } = require('playwright');
const [site, url] = process.argv.slice(2);
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const b = await chromium.launch({ headless: false });
const ctx = await b.newContext({ userAgent: UA, viewport: { width: 1400, height: 1000 } });
const p = await ctx.newPage();
const r = await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await p.waitForTimeout(4000);
console.log('status', r?.status(), await p.title());
const out = await p.evaluate(() => {
  const o = [];
  for (const i of document.querySelectorAll('img')) o.push({ cur: i.currentSrc, src: i.src, srcset: i.srcset, w: i.naturalWidth, cw: i.clientWidth });
  const bg = [];
  for (const el of document.querySelectorAll('*')) { const v = getComputedStyle(el).backgroundImage; if (v && v !== 'none' && v.includes('url')) bg.push(v.slice(0, 400)); }
  return { imgs: o, bg, ogs: [...document.querySelectorAll('meta[property="og:image"],link[rel=preload][as=image]')].map(m => m.content || m.href) };
});
console.log(JSON.stringify(out, null, 1));
await b.close();
