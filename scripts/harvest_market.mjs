// Paced, stop-on-block marketplace photo harvester (public pages, no login). Output is gitignored.
// node harvest_market.mjs <mercari|ebay|hibid|...> [--max-photos=N] [--max-listings=N]
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile, access } from 'node:fs/promises';
import crypto from 'node:crypto';
import path from 'node:path';
const require = createRequire('C:/repos/projects/auto card lister/package.json');
const { chromium } = require('playwright');
const sharp = require('sharp');

const ROOT = 'C:/repos/projects/card-scanner-data/data/harvest-20261022';
const site = process.argv[2];
const arg = (k, d) => Number(process.argv.find(a => a.startsWith(`--${k}=`))?.split('=')[1] ?? d);
const MAX_PHOTOS = arg('max-photos', 400), MAX_LISTINGS = arg('max-listings', 250);
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const CH = /just a moment|attention required|captcha|access denied|checking your browser|verify you are human|are you a robot|pardon our interruption|unusual traffic|press & hold|security check|request blocked/i;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString(), site, ...a);
class Blocked extends Error {}

const enc = s => encodeURIComponent(s);
const KW = ['pokemon card lot', 'pokemon cards bulk', 'pokemon binder', 'pokemon cards collection', 'pokemon holo lot', 'pokemon cards toploaders', 'vintage pokemon cards lot', 'pokemon cards slabs lot'];
const POKE = /pok[eé]mon/i, DROP = /\b(plush|funko|figure|figurine|video game|nintendo|booster box|sealed box|etb|elite trainer|shirt|poster|blanket|backpack|keychain|squishmallow|lego|yu-?gi-?oh|magic the gathering|mtg|digimon|one piece|sports)\b/i;
const titleOk = t => POKE.test(t) && !DROP.test(t);
const priceOf = t => (t.match(/\$\s?[\d,]+(?:\.\d{2})?/) || [''])[0];

const SITES = {
  mercari: {
    gap: 3200, pages: 2,
    searches: p => KW.map(k => `https://www.mercari.com/search/?keyword=${enc(k)}${p > 1 ? `&page=${p}` : ''}`),
    link: /mercari\.com\/us\/item\/(m\d+)/,
    photos: (id) => [...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => s.includes(`/photos/${id}_`)).map(s => s.split('?')[0].replace('/thumb/', '/')),
    big: u => u,
  },
  ebay: {
    gap: 3200, pages: 2,
    searches: p => KW.flatMap(k => [`https://www.ebay.com/sch/i.html?_nkw=${enc(k)}&LH_Sold=1&LH_Complete=1&_pgn=${p}`, `https://www.ebay.com/sch/i.html?_nkw=${enc(k)}&_pgn=${p}`]),
    link: /ebay\.com\/itm\/(\d+)/,
    photos: () => [...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => /i\.ebayimg\.com\/(thumbs\/)?images\/g\//.test(s)),
    big: u => u.replace(/\/s-l\d+\.(jpg|webp|png)/, '/s-l1600.jpg').replace('/thumbs/', '/'),
  },
  craigslist: {
    gap: 3200, pages: 1,
    searches: () => ['cincinnati', 'chicago', 'dallas', 'losangeles', 'atlanta']
      .flatMap(c => ['pokemon card lot', 'pokemon cards binder'].map(k => `https://${c}.craigslist.org/search/sss?query=${enc(k)}`)),
    link: /craigslist\.org\/view\/d\/[^/]+\/([A-Za-z0-9]+)$/,
    photos: () => [...new Set([...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => /images\.craigslist\.org\/.*_\d+x\d+c?\.jpg/.test(s)))],
    big: u => u.replace(/_\d+x\d+c?\.jpg/, '_1200x900.jpg'),
  },
  poshmark: {
    gap: 3200, pages: 1,
    searches: () => KW.slice(0, 5).map(k => `https://poshmark.com/search?query=${enc(k)}&type=listings`),
    link: /poshmark\.com\/listing\/[^/]*?-?([0-9a-f]{24})$/,
    photos: () => [...new Set([...document.querySelectorAll('img')].map(i => i.currentSrc || i.src).filter(s => /cloudfront\.net\/posts\/.*\/[a-z]_wp_[0-9a-f]+\.(webp|jpg)/.test(s)))],
    big: u => u.replace(/\/[a-z]_wp_/, '/l_wp_'),
  },
  hibid: {
    gap: 4000, pages: 1,
    searches: () => [...KW.slice(0, 6).map(k => `https://hibid.com/lots?q=${enc(k)}`), 'https://hibid.com/lots/700074/antiques-and-collectibles/collectibles/trading-cards/pokemon-cards'],
    link: /hibid\.com\/lot\/(\d+)/,
    photos: () => {
      const out = new Set(), clean = u => u.replace(/&h=\d+&w=\d+$/, '');
      for (const el of document.querySelectorAll('*')) for (const m of getComputedStyle(el).backgroundImage.matchAll(/url\(["']?(.*?)["']?\)/g)) if (m[1].includes('img.axd') && m[1].includes('sz=MAX')) out.add(clean(m[1]));
      for (const i of document.querySelectorAll('img')) { const s = i.currentSrc || i.src || ''; if (s.includes('img.axd') && s.includes('sz=MAX')) out.add(clean(s)); }
      return [...out];
    },
    big: u => u,
    exclude: async () => (await readFile('C:/repos/projects/card-scanner-data/data/hibid-pack/MANIFEST.json', 'utf8').then(JSON.parse)).map(m => String(m.lotId)),
  },
};
const cfg = SITES[site];
if (!cfg) throw new Error('unknown site ' + site);

const OUT = path.join(ROOT, '_raw', site);
await mkdir(OUT, { recursive: true });
const mpath = path.join(OUT, 'raw-manifest.json');
const manifest = await readFile(mpath, 'utf8').then(JSON.parse, () => []);
const done = new Set(manifest.map(m => m.listingId));
const hashes = new Set(manifest.map(m => m.sha256));
for (const id of (await cfg.exclude?.()) ?? []) done.add(id);

let last = 0;
const pace = async () => { const w = last + cfg.gap + Math.random() * 1500 - Date.now(); if (w > 0) await sleep(w); last = Date.now(); };

const browser = await chromium.launch({ headless: false });
const ctx = await browser.newContext({ userAgent: UA, viewport: { width: 1400, height: 1000 } });
const page = await ctx.newPage();

async function load(url) {
  await pace();
  const r = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(3500);
  last = Date.now();
  const st = r?.status() ?? 0;
  const title = await page.title();
  const text = await page.evaluate(() => document.body?.innerText ?? '');
  if (st === 403 || st === 429 || st === 503) throw new Blocked(`${url} HTTP ${st}`);
  if (CH.test(title) || CH.test(text.slice(0, 1500))) throw new Blocked(`${url} challenge "${title}"`);
  return text;
}

let photos = 0, listings = 0, stopped = null;
try {
  outer: for (let p = 1; p <= cfg.pages; p++) for (const s of cfg.searches(p)) {
    if (photos >= MAX_PHOTOS || listings >= MAX_LISTINGS) break outer;
    await load(s);
    await page.evaluate(() => window.scrollBy(0, 3000)); await page.waitForTimeout(1500);
    const found = await page.evaluate(() => [...document.querySelectorAll('a[href]')].map(a => ({
      href: a.href.split('?')[0], text: (a.closest('li, article, [data-testid], [class*=item]') || a).innerText.replace(/\s+/g, ' ').slice(0, 200) })));
    const cands = new Map();
    for (const f of found) { const m = f.href.match(cfg.link); if (m && !cands.has(m[1])) cands.set(m[1], f); }
    log('search', s, 'cands', cands.size);
    for (const [id, f] of cands) {
      if (photos >= MAX_PHOTOS || listings >= MAX_LISTINGS) break outer;
      if (done.has(id)) continue;
      done.add(id);
      const text = await load(f.href);
      const rawTitle = await page.title();
      const title = (site === 'poshmark' ? rawTitle.split(' | ').slice(-2, -1)[0] ?? rawTitle : rawTitle).replace(/\s*\|.*$/, '').replace(/ for sale online.*$/i, '');
      if (!titleOk(title)) { log('skip title', id, title.slice(0, 60)); continue; }
      const urls = [...new Set((await page.evaluate(cfg.photos, id)).map(cfg.big))];
      if (!urls.length) { log('no photos', id); continue; }
      const dir = path.join(OUT, id);
      await mkdir(dir, { recursive: true });
      let n = 0;
      for (const u of urls.slice(0, 12)) {
        await pace();
        const res = await ctx.request.get(u, { headers: { Referer: f.href } });
        if ([403, 429].includes(res.status())) throw new Blocked(`${u} HTTP ${res.status()}`);
        if (!res.ok() || !(res.headers()['content-type'] || '').startsWith('image/')) continue;
        let buf = await res.body();
        let meta = await sharp(buf).metadata().catch(() => null);
        if (!meta?.width || meta.width < 200) continue;
        if (meta.format !== 'jpeg') { buf = await sharp(buf).jpeg({ quality: 95 }).toBuffer(); meta = await sharp(buf).metadata(); }
        const sha = crypto.createHash('sha256').update(buf).digest('hex');
        if (hashes.has(sha)) continue;
        hashes.add(sha); n++;
        await writeFile(path.join(dir, `photo-${n}.jpg`), buf);
        manifest.push({ site, listingId: id, url: f.href, title, price: priceOf(f.text), photo: n, width: meta.width, height: meta.height, sha256: sha });
        photos++;
      }
      listings++;
      await writeFile(mpath, JSON.stringify(manifest, null, 1));
      log(id, n, 'photos', title.slice(0, 50), `total=${photos}/${listings}`);
    }
  }
} catch (e) {
  stopped = e instanceof Blocked ? `BLOCKED: ${e.message}` : `ERROR: ${e.message}`;
  log('STOPPED', stopped);
} finally {
  await writeFile(mpath, JSON.stringify(manifest, null, 1));
  await browser.close();
  log(`done listings=${listings} photos=${photos} stopped=${stopped}`);
}
