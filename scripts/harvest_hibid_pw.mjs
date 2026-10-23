// Paced, stop-on-challenge HiBid photo harvester (adapted from the lister repo's harvest-hibid-photos.mjs).
// Run from the lister repo folder: node <this file> [--max-lots=400]
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile, readdir, access } from 'node:fs/promises';
import path from 'node:path';

const require = createRequire('C:/repos/projects/auto card lister/package.json');
const { chromium } = require('playwright');

const OUT = 'C:/repos/projects/card-scanner-data/data/hibid-pack';
const DATE = '2026-10-21';
const MAX_LOTS = Number(process.argv.find(a => a.startsWith('--max-lots='))?.split('=')[1] ?? 400);
const MIN_GAP_MS = 6000, MAX_GAP_MS = 12000;
const QUERIES = ['pokemon card lot', 'pokemon cards', 'pokemon binder', 'pokemon collection', 'pokemon holo', 'vintage pokemon cards']
  .map(q => `https://hibid.com/lots?q=${encodeURIComponent(q)}`);
QUERIES.push('https://hibid.com/lots/700074/antiques-and-collectibles/collectibles/trading-cards/pokemon-cards');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const exists = p => access(p).then(() => true, () => false);
const log = (...a) => console.log(new Date().toISOString(), ...a);

const CHALLENGE = /just a moment|attention required|captcha|access denied|checking your browser|verify you are human|cf-chl|cloudflare ray id/i;

class Challenge extends Error {}

let lastLoad = 0, loads = 0;
async function pacedGoto(page, url, waitMs) {
  const gap = MIN_GAP_MS + Math.random() * (MAX_GAP_MS - MIN_GAP_MS);
  const wait = lastLoad + gap - Date.now();
  if (wait > 0) await sleep(wait);
  lastLoad = Date.now();
  loads++;
  const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(waitMs);
  lastLoad = Date.now();
  const status = resp?.status() ?? 0;
  const title = await page.title();
  const text = await page.evaluate(() => document.body?.innerText ?? '');
  if (status === 403 || status === 429) throw new Challenge(`${url} HTTP ${status}`);
  if (CHALLENGE.test(title) || CHALLENGE.test(text.slice(0, 3000))) throw new Challenge(`${url} challenge text; title="${title}"`);
  if (text.trim().length < 200) throw new Challenge(`${url} empty page (status ${status}, ${text.trim().length} chars)`);
}

const KEEP = /\b(lots?|binders?|collections?|stacks?|toploaders?|top loaders?|bulk|cards)\b/i;
const POKE = /pok[eé]mon/i;
const DROP = /\b(plush|toys?|figures?|figurines?|video games?|nintendo|game ?boy|switch|sealed|booster box|etb|elite trainer|tin|empty|funko|squishmallow|shirt|poster|blanket|backpack|keychain)\b/i;
const SLAB = /\b(psa|bgs|cgc|sgc|graded|slab)\b/i;
const MULTI = /\b(lots?|slabs|collections?|stacks?)\b/i;
function titleOk(t) {
  if (!POKE.test(t) || !KEEP.test(t) || DROP.test(t)) return false;
  if (SLAB.test(t)) return /\b(lot|lots|slabs|\d+\s*x|\d{2,}\s*(psa|bgs|cgc|sgc|graded))\b/i.test(t) && /\b(lot|lots|slabs|multiple)\b/i.test(t);
  return true;
}
const splitOf = id => (['0', '1', '2'].includes(id.slice(-1)) ? 'eval' : 'train');

function jpegSize(buf) {
  if (buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  let i = 2;
  while (i + 9 < buf.length) {
    if (buf[i] !== 0xff) { i++; continue; }
    const m = buf[i + 1];
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
    i += 2 + buf.readUInt16BE(i + 2);
  }
  return null;
}

async function lotLinks(page, url) {
  await pacedGoto(page, url, 7000);
  return page.evaluate(() => {
    const byUrl = new Map();
    for (const a of document.querySelectorAll('a[href*="/lot/"]')) {
      const url = a.href.split('?')[0];
      const title = a.textContent.replace(/\s+/g, ' ').replace(/^Lot \S+ \| /, '').trim();
      if (!byUrl.has(url) || title.length > byUrl.get(url).length) byUrl.set(url, title);
    }
    return [...byUrl].map(([url, title]) => ({ url, title }));
  });
}

async function gallery(page, url) {
  await pacedGoto(page, url, 5000);
  return page.evaluate(() => {
    const out = new Set();
    const clean = u => u.replace(/&h=\d+&w=\d+$/, '');
    for (const el of document.querySelectorAll('*')) {
      for (const m of getComputedStyle(el).backgroundImage.matchAll(/url\(["']?(.*?)["']?\)/g)) {
        if (m[1].includes('img.axd') && m[1].includes('sz=MAX')) out.add(clean(m[1]));
      }
    }
    for (const img of document.querySelectorAll('img')) {
      const s = img.currentSrc || img.src || '';
      if (s.includes('img.axd') && s.includes('sz=MAX')) out.add(clean(s));
    }
    return { title: document.title.split('|')[0].trim(), urls: [...out], host: location.host };
  });
}

async function loadManifest() {
  try { return JSON.parse(await readFile(path.join(OUT, 'MANIFEST.json'), 'utf8')); } catch { return []; }
}

await mkdir(OUT, { recursive: true });
const manifest = await loadManifest();
const done = new Set(manifest.map(m => m.lotId));
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1400, height: 1000 } });
const page = await context.newPage();
let lots = 0, photos = 0, stopped = null;
const seenImgHosts = new Set();
page.on('response', r => { if (r.url().includes('img.axd')) seenImgHosts.add(new URL(r.url()).host); });
try {
  outer: for (const q of QUERIES) {
    const links = await lotLinks(page, q);
    log('query', q, 'links', links.length);
    for (const { url: lot, title } of links) {
      if (lots >= MAX_LOTS) break outer;
      const lotId = lot.match(/\/lot\/(\d+)/)?.[1];
      if (!lotId || done.has(lotId) || !titleOk(title)) continue;
      const split = splitOf(lotId);
      lots++; done.add(lotId);
      const g = await gallery(page, lot);
      if (!g.urls.length) { log('no photos', lotId); continue; }
      const dir = path.join(OUT, split, lotId);
      await mkdir(dir, { recursive: true });
      let n = 0;
      for (const u of g.urls) {
        const host = new URL(u, lot).host;
        if (!seenImgHosts.has(host)) continue;
        const res = await context.request.get(u);
        if (!res.ok() || !(res.headers()['content-type'] || '').startsWith('image/')) continue;
        const buf = await res.body();
        const sz = jpegSize(buf) ?? { w: 0, h: 0 };
        n++;
        await writeFile(path.join(dir, `photo-${n}.jpg`), buf);
        manifest.push({ lotId, split, title: g.title, url: lot, photo: n, width: sz.w, height: sz.h, bytes: buf.length, date: DATE });
        photos++;
        await sleep(1000);
      }
      await writeFile(path.join(OUT, 'MANIFEST.json'), JSON.stringify(manifest, null, 1));
      log(`${split}\t${lotId}\t${n}\t${g.title}`);
    }
  }
} catch (e) {
  stopped = e instanceof Challenge ? `CHALLENGE: ${e.message}` : `ERROR: ${e.message}`;
  log('STOPPED', stopped);
} finally {
  await writeFile(path.join(OUT, 'MANIFEST.json'), JSON.stringify(manifest, null, 1));
  await browser.close();
  log(`done lots=${lots} photos=${photos} pageLoads=${loads} stopped=${stopped}`);
}
