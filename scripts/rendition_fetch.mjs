// Polite rendition fetcher for the rendition study. node rendition_fetch.mjs <site>
// 1 request at a time, >=3.3 s apart, browser UA, public image URLs only, STOP the site on first 403/429/challenge.
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
const require = createRequire('C:/repos/projects/auto card lister/package.json');
const { chromium } = require('playwright');
const sharp = require('sharp');
const ROOT = 'C:/repos/projects/card-scanner-data/data';
const site = process.argv[2];
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const CH = /just a moment|attention required|captcha|access denied|checking your browser|verify you are human|are you a robot|pardon our interruption|unusual traffic|press & hold|security check|request blocked/i;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const GAP = site === 'hibid' ? 4000 : 3300;
const DHASH_MAX = 14; // bits of 64; same photo, different size, stays well below this
let last = 0;
const pace = async () => { const w = last + GAP + Math.random() * 1200 - Date.now(); if (w > 0) await sleep(w); last = Date.now(); };
class Blocked extends Error {}

const SITES = {
  mercari: {
    pageUrls: () => [...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => s.includes('/photos/m')).map(s => s.split('?')[0].replace('/thumb/', '/')),
    pickUrls: (all, id) => [...new Set(all.filter(s => s.includes(`/photos/${id}_`)))],
    renditions: b => [1280, 768, 256, 80].map(w => [`w${w}`, `${b}?width=${w}&quality=75`]),
  },
  hibid: {
    pageUrls: () => {
      const out = new Set(), clean = u => u.replace(/&h=\d+&w=\d+$/, '');
      for (const el of document.querySelectorAll('*')) for (const m of getComputedStyle(el).backgroundImage.matchAll(/url\(["']?(.*?)["']?\)/g)) if (m[1].includes('img.axd') && m[1].includes('sz=MAX')) out.add(clean(m[1]));
      for (const i of document.querySelectorAll('img')) { const s = i.currentSrc || i.src || ''; if (s.includes('img.axd') && s.includes('sz=MAX')) out.add(clean(s)); }
      return [...out];
    },
    pickUrls: a => a,
    renditions: b => [['g720', b.replace('&w=0&h=0&', '&w=720&h=720&').replace('&sz=MAX', '')], ['t200', b + '&h=200&w=200']],
  },
  craigslist: {
    pageUrls: () => [...new Set([...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => /images\.craigslist\.org\/.*_\d+x\d+c?\.jpg/.test(s)))],
    pickUrls: a => [...new Set(a.map(u => u.replace(/_\d+x\d+c?\.jpg/, '')))],
    renditions: b => ['600x450', '300x300', '50x50c'].map(s => [s, `${b}_${s}.jpg`]),
  },
  poshmark: {
    pageUrls: () => [...new Set([...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => /cloudfront\.net\/posts\/.*\/[a-z]_wp_[0-9a-f]+\.(webp|jpg)/.test(s)))],
    pickUrls: a => [...new Set(a.map(u => u.replace(/\/[a-z]_wp_/, '/L_wp_')))],
    renditions: b => [['l', b.replace('/L_wp_', '/l_').replace(/\.webp$/, '.jpg')], ['m', b.replace('/L_wp_', '/m_').replace(/\.webp$/, '.jpg')], ['s', b.replace('/L_wp_', '/s_').replace(/\.webp$/, '.jpg')], ['s_wp', b.replace('/L_wp_', '/s_wp_')]],
  },
  shopgoodwill: {
    renditions: b => [['thumb', b.replace(/(\d{4})(\d+)\.jpg$/i, '$1t$2.jpeg')]],
  },
};
const cfg = SITES[site];
if (!cfg) throw new Error('unknown site');
const sel = JSON.parse(await readFile(`${ROOT}/reports/rendition/selection.json`, 'utf8')).filter(s => s.site === site);
const OUT = `${ROOT}/harvest-20261022/_renditions/${site}`;
await mkdir(OUT, { recursive: true });
const mpath = `${OUT}/rendition-manifest.json`;
const man = [];

async function dhash(buf) {
  const { data } = await sharp(buf).rotate().greyscale().resize(9, 8, { fit: 'fill' }).raw().toBuffer({ resolveWithObject: true });
  const h = [];
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) h.push(data[y * 9 + x] > data[y * 9 + x + 1] ? 1 : 0);
  return h;
}
const ham = (a, b) => a.reduce((s, v, i) => s + (v !== b[i]), 0);

let browser, ctx, page;
if (site !== 'shopgoodwill') {
  browser = await chromium.launch({ headless: false });
  ctx = await browser.newContext({ userAgent: UA, viewport: { width: 1400, height: 1000 } });
  page = await ctx.newPage();
}
async function getBytes(u, referer) {
  await pace();
  let res;
  if (ctx) res = await ctx.request.get(u, { headers: { Referer: referer } });
  else {
    const r = await fetch(u, { headers: { 'User-Agent': UA } });
    res = { status: () => r.status, ok: () => r.ok, headers: () => ({ 'content-type': r.headers.get('content-type') || '' }), body: async () => Buffer.from(await r.arrayBuffer()) };
  }
  if ([403, 429].includes(res.status())) throw new Blocked(`${u} HTTP ${res.status()}`);
  return res;
}
const log = (...a) => console.log(new Date().toISOString(), site, ...a);
let stopped = null;
try {
  const byListing = new Map();
  for (const s of sel) { if (!byListing.has(s.listing)) byListing.set(s.listing, { url: s.url, photos: [] }); byListing.get(s.listing).photos.push(s.photo); }
  for (const [id, L] of byListing) {
    let urls;
    if (site === 'shopgoodwill') {
      await pace();
      const r = await fetch('https://buyerapi.shopgoodwill.com/api/ItemDetail/GetItemDetailModelByItemId/' + id, { headers: { 'User-Agent': 'Mozilla/5.0' } });
      if ([403, 429].includes(r.status)) throw new Blocked(`api ${id} HTTP ${r.status}`);
      const d = await r.json();
      const base = d.imageServer || 'https://shopgoodwillimages.azureedge.net/production/';
      urls = (d.imageUrlString || '').split(';').filter(u => u.trim()).map(u => (u.startsWith('http') ? u : base.replace(/\/$/, '') + '/' + u.replace(/^\//, '')).replace(/\\/g, '/').replace(/ /g, '%20'));
    } else {
      await pace();
      const r = await page.goto(L.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForTimeout(3500); last = Date.now();
      const title = await page.title(), text = await page.evaluate(() => document.body?.innerText ?? '');
      if ([403, 429, 503].includes(r?.status() ?? 0) || CH.test(title) || CH.test(text.slice(0, 1500))) throw new Blocked(`${L.url} challenge "${title}" ${r?.status()}`);
      urls = cfg.pickUrls(await page.evaluate(cfg.pageUrls), id).slice(0, 12);
    }
    log(id, 'urls', urls.length);
    for (const n of L.photos) {
      const lh = await dhash(await readFile(`${ROOT}/harvest-20261022/${site}/train/${id}/photo-${n}.jpg`));
      let ok = false;
      for (const k of [n - 1, n, n - 2, n + 1]) {
        if (k < 0 || k >= urls.length) continue;
        const got = [];
        let bad = false;
        for (const [tag, u] of cfg.renditions(urls[k])) {
          const res = await getBytes(u, L.url);
          const ct = res.headers()['content-type'] || '';
          if (!res.ok() || !ct.startsWith('image/')) { got.push({ tag, u, status: res.status(), ct }); continue; }
          const buf = await res.body();
          const meta = await sharp(buf).metadata();
          const d = ham(lh, await dhash(buf));
          const ext = (ct.split('/')[1] || 'jpg').replace('jpeg', 'jpg');
          const f = `${id}/photo-${n}__${tag}.${ext}`;
          if (site === 'craigslist' && tag === '50x50c') { got.push({ tag, u, status: 200, w: meta.width, h: meta.height, d, file: null }); continue; }
          if (d > DHASH_MAX) { got.push({ tag, u, status: res.status(), w: meta.width, h: meta.height, d, mismatch: true }); bad = true; break; }
          await mkdir(`${OUT}/${id}`, { recursive: true });
          await writeFile(`${OUT}/${f}`, buf);
          got.push({ tag, u, status: res.status(), w: meta.width, h: meta.height, bytes: buf.length, d, file: f });
        }
        if (bad) { log(id, n, 'dhash mismatch at url index', k); continue; }
        man.push({ site, listing: id, photo: n, urlIndex: k, renditions: got });
        log(id, n, got.map(g => `${g.tag}:${g.w ?? g.status}x${g.h ?? ''}/d${g.d ?? ''}`).join(' '));
        ok = true; break;
      }
      if (!ok) log(id, n, 'NO MATCH');
      await writeFile(mpath, JSON.stringify(man, null, 1));
    }
  }
} catch (e) {
  stopped = e instanceof Blocked ? `BLOCKED: ${e.message}` : `ERROR: ${e.stack}`;
  log('STOPPED', stopped);
} finally {
  await writeFile(mpath, JSON.stringify(man, null, 1));
  await writeFile(`${OUT}/stopped.txt`, String(stopped));
  await browser?.close();
  log('done', stopped);
}
