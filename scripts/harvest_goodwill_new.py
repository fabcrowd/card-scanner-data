"""Harvest public ShopGoodwill listing photos into data/goodwill-pack2 (private, gitignored)."""
import json, os, re, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = r'C:\repos\projects\card-scanner-data\data'
OUT = os.path.join(ROOT, 'harvest-20261022', '_raw', 'shopgoodwill')
TMP = os.path.join(ROOT, 'goodwill-pack2', '_tmp')
CAND_DONE = {str(m['itemId']) for m in json.load(open(os.path.join(ROOT, 'goodwill-pack2', 'MANIFEST.json')))}
API = 'https://buyerapi.shopgoodwill.com/api/'
QUERIES = ['pokemon card lot', 'pokemon cards lot', 'pokemon binder', 'pokemon card collection', 'pokemon cards bulk',
           'pokemon trading cards', 'pokemon holo lot', 'vintage pokemon cards', 'pokemon cards wotc']
CAP = 250
MAX_PHOTOS = 330
MAX_PAGES = 8
INC = re.compile(r'card|binder|collection|tcg|holo|\blot\b', re.I)
EXC = re.compile(r'plush|figure|funko|toy|nintendo|game ?boy|\bds\b|switch|video game|booster box|sealed|elite trainer|\betb\b|'
                 r'tin\b|empty|psa \d|bgs \d|cgc \d|beckett|magic|yu-?gi-?oh|digimon|one piece|sports|baseball|football|basketball|'
                 r'hockey|pillow|shirt|backpack|keychain|sticker|poster|lego|sleeves? only|pack of|booster pack|blister|display', re.I)
last = [0.0]
blocked = []


def curl(url, out):
    r = subprocess.run(['curl', '-s', '-A', 'Mozilla/5.0', '--max-time', '30', '-w', '%{http_code}', url, '-o', out],
                       capture_output=True)
    return r.stdout.decode('latin1')[-3:]


def api_wait():
    d = 2.2 - (time.time() - last[0])
    if d > 0:
        time.sleep(d)
    last[0] = time.time()


def search(q, page, closed):
    body = {"searchText": q, "page": page, "pageSize": 40, "selectedCategoryIds": "", "selectedSellerIds": "", "lowPrice": 0,
            "highPrice": 999999, "searchBuyNowOnly": "", "searchPickupOnly": "", "searchNoPickupOnly": "",
            "searchOneCentShippingOnly": "", "searchDescriptions": "", "searchClosedAuctions": "true" if closed else "false",
            "closedAuctionEndingDate": "1/1/2026", "closedAuctionDaysBack": "30", "searchCanadaShipping": "false",
            "searchInternationalShippingOnly": "false", "sortColumn": "1", "sortDescending": "false", "savedSearchId": 0,
            "useBuyerPrefs": "true", "searchUSOnlyShipping": "false", "categoryLevelNo": "1", "categoryLevel": 1,
            "categoryId": 0, "partNumber": "", "catIds": ""}
    api_wait()
    r = subprocess.run(['curl', '-s', '-A', 'Mozilla/5.0', '--max-time', '30', '-X', 'POST', '-H', 'Content-Type: application/json',
                        '-d', json.dumps(body), '-w', '\n%{http_code}', API + 'Search/ItemListing'], capture_output=True)
    txt = r.stdout.decode('utf8', 'replace')
    txt, code = txt.rsplit('\n', 1)
    if code != '200':
        blocked.append(('search', q, page, code))
        return None
    try:
        return json.loads(txt)['searchResults']['items']
    except Exception:
        blocked.append(('searchparse', q, page, txt[:100]))
        return None


def collect():
    cp = os.path.join(TMP, 'cands.json')
    cands = json.load(open(cp)) if os.path.exists(cp) else {}
    for closed in (False, True):
        for q in QUERIES:
            for page in range(1, MAX_PAGES + 1):
                items = search(q, page, closed)
                if blocked:
                    return cands
                if not items:
                    break
                for it in items:
                    cands.setdefault(str(it['itemId']), {'title': it['title'], 'ended': True if closed else None})
                if len(items) < 40:
                    break
            print(closed, q, len(cands), flush=True)
            json.dump(cands, open(cp, 'w'))
    return cands


def keep(t):
    return bool(re.search(r'pok[eé]mon', t, re.I) and INC.search(t) and not EXC.search(t))


def split(i):
    return 'eval' if int(i) % 10 in (0, 1) else 'train'


def main():
    os.makedirs(OUT, exist_ok=True)
    if '--collect' in sys.argv:
        c = collect()
        print('cands', len(c), 'blocked', blocked)
        return
    cands = json.load(open(os.path.join(TMP, 'cands.json')))
    existing = set(os.listdir(os.path.join(ROOT, 'goodwill-pack'))) | CAND_DONE
    sel = [(i, v) for i, v in cands.items() if i not in existing and keep(v['title'])]
    import random
    random.Random(20261022).shuffle(sel)
    sel = sel[:CAP]
    print('selected', len(sel), flush=True)
    mpath = os.path.join(OUT, 'raw-manifest.json')
    man = json.load(open(mpath)) if os.path.exists(mpath) else []
    done_items = {m['listingId'] for m in man}
    for n, (iid, v) in enumerate(sel):
        if len(man) >= MAX_PHOTOS:
            break
        if iid in done_items:
            continue
        api_wait()
        r = subprocess.run(['curl', '-s', '-A', 'Mozilla/5.0', '--max-time', '30', '-w', '\n%{http_code}',
                            API + 'ItemDetail/GetItemDetailModelByItemId/' + iid], capture_output=True)
        txt = r.stdout.decode('utf8', 'replace')
        txt, code = txt.rsplit('\n', 1)
        if code in ('403', '429') or 'captcha' in txt.lower()[:2000]:
            print('BLOCK', code)
            blocked.append(('detail', iid, code))
            break
        if code != '200':
            print('detail', iid, code)
            continue
        try:
            d = json.loads(txt)
        except Exception:
            print('parse fail', iid)
            continue
        base = d.get('imageServer') or 'https://shopgoodwillimages.azureedge.net/production/'
        urls = [u for u in (d.get('imageUrlString') or '').split(';') if u.strip()]
        sp = split(iid)
        dd = os.path.join(OUT, iid)
        os.makedirs(dd, exist_ok=True)

        def dl(args):
            k, u = args
            f = os.path.join(dd, f'photo-{k}.jpg')
            if not (os.path.exists(f) and os.path.getsize(f) > 0):
                full = u if u.startswith('http') else base.rstrip('/') + '/' + u.lstrip('/')
                full = full.replace('\\', '/').replace(' ', '%20')
                c = curl(full, f)
                time.sleep(1)
                if c != '200':
                    if os.path.exists(f):
                        os.remove(f)
                    return (k, c, None)
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except Exception:
                os.remove(f)
                return (k, 'bad', None)
            return (k, '200', (w, h, os.path.getsize(f)))

        with ThreadPoolExecutor(2) as ex:
            res = list(ex.map(dl, list(enumerate(urls, 1))))
        bad = [x for x in res if x[1] in ('403', '429')]
        if bad:
            print('BLOCK img', bad)
            blocked.append(('img', iid, bad))
            break
        link = f'https://shopgoodwill.com/item/{iid}'
        ok = 0
        for k, c, info in res:
            if info:
                ok += 1
                import hashlib
                man.append({'site': 'shopgoodwill', 'listingId': iid, 'url': link, 'title': d.get('title') or v['title'],
                            'price': '', 'photo': k, 'width': info[0], 'height': info[1],
                            'sha256': hashlib.sha256(open(os.path.join(dd, f'photo-{k}.jpg'), 'rb').read()).hexdigest()})
        json.dump(man, open(mpath, 'w'), indent=1)
        print(n, iid, sp, ok, flush=True)
    print('blocked', blocked)


main()
