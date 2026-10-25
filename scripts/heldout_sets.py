"""Pick the held-out set H: newest whole sets (TCGdex releaseDate) among physical cards with an image, ~800-1200 cards."""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
META = r"C:\repos\projects\auto card lister\data\card-catalog\meta.json"
OUT = os.path.join(ROOT, "train", "out", "ho", "heldout.json")
LO, HI = 800, 1200


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    import pilot_data as D
    from pipeline.catalog import Catalog
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    have, _ = D.eval_rows(cat)
    phys = {cat.ids[r] for r in have if not D.is_digital(cat.ids[r])}
    meta = {m["id"]: m for m in json.load(open(META, encoding="utf-8"))}
    sets = {}
    for cid in phys:
        sets.setdefault(meta[cid]["setCode"], []).append(cid)
    dated = {}
    for sc in sets:
        try:
            d = fetch(f"https://api.tcgdex.net/v2/en/sets/{sc}")
            dated[sc] = d.get("releaseDate")
        except Exception as e:  # noqa: BLE001
            dated[sc] = None
            print("no date", sc, e)
    order = sorted((sc for sc in sets if dated[sc]), key=lambda s: dated[s], reverse=True)
    chosen, n = [], 0
    for sc in order:
        if n + len(sets[sc]) > HI and n >= LO:
            break
        chosen.append(sc)
        n += len(sets[sc])
        if n >= LO:
            break
    info = [{"setCode": s, "set": meta[sets[s][0]]["set"], "release": dated[s], "cards": len(sets[s])} for s in chosen]
    print(json.dumps(info, indent=1), n)
    print("next newest:", [(s, dated[s], len(sets[s])) for s in order[len(chosen):len(chosen) + 3]])
    print("undated:", [s for s in sets if not dated[s]])
    json.dump({"sets": info, "ids": sorted(i for s in chosen for i in sets[s]), "all_dates": dated}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
