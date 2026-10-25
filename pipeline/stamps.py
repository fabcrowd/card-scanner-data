"""Stamped TCGplayer products joined to one catalog card, and EX-era prices keyed by catalog id.

TCGplayer lists a stamped printing (Prerelease, Staff, League, ...) as its own product; the stamp is only in the
product name. A name tag counts as a stamp only when it is exactly one of the tags below, so a trainer card called
"Pokemon Center" or an unlisted tag such as "(E3 Stamped)" is never a stamp. A product is kept only when its name
and number pick exactly one catalog card (the release meta.json the extension downloads).

The seven EX sets whose reverse holos all carry the set logo are priced per catalog card from their TCGplayer
group, joined by set, number and name, because TCGdex has no TCGplayer product id for many of those cards.
"""
from __future__ import annotations

import json
import re
import unicodedata

# Earlier wins when a name carries two stamp tags: "(Prerelease) [Staff]" is a staff card.
STAMPS = ("staff", "prerelease", "championship", "league", "pokemoncenter", "wstamp", "setlogo", "playpokemon")
PROMO_PREFIX = {"BW": "bwp", "XY": "xyp", "SM": "smp", "SWSH": "swshp", "DP": "dpp", "HGSS": "hgssp"}
SV_PROMO_GROUP = 22872
PRIZE_PACK_GROUP = 22880
EX_STAMP_GROUPS = {1398: "ex10", 1429: "ex11", 1378: "ex12", 1379: "ex13", 1395: "ex14", 1411: "ex15", 1383: "ex16"}

_LEAGUE = {"league", "league promo", "pokemon league", "league challenge", "league cup"}
_POKEMON_CENTER = {"pokemon center", "pokemon center exclusive", "pokemon center ny"}
_TAG_RULES = (
    ("staff", re.compile(r"(?:prerelease |\d{4} )?staff")),
    ("prerelease", re.compile(r"(?:[a-z0-9&' ]+ )?prerelease")),
    ("championship", re.compile(
        r"worlds \d{2}|world championships? \d{4}"
        r"|(?:(?:city|state|regional|national|north america|latin america|south america|europe|oceania|asia|arena cup)"
        r"(?: international)? )?championships?(?: series| promo)?(?: \d{2}-\d{2})?"
    )),
)
_SET_LOGO = re.compile(r"(.+) stamp(?:ed)?")
_TAG = re.compile(r"\(([^)]*)\)|\[([^\]]*)\]")
_TAGS = re.compile(r"\s*\([^)]*\)|\s*\[[^\]]*\]")
_SYMBOLS = {"\u2605": " star ", "\u2606": " star ", "\u2640": " f ", "\u2642": " m "}
_PROMO_NUMBER = re.compile(r"([A-Z]+)(\d+)")
_PRINTED = re.compile(r"(\d+)/(\d+)")


def _fold(text: str) -> str:
    """Lowercase without accents, single spaces."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(text.split())


def norm(text: str, delta: str = "delta") -> str:
    """Letters and digits only, with the catalog's star, gender and delta symbols spelled out like TCGplayer's."""
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    return re.sub(r"[^a-z0-9]", "", _fold(text.replace("\u03b4", f" {delta} ")))


def _strip_zeros(number: str) -> str:
    return re.sub(r"^0+(?=.)", "", number.strip().lower())


def base_name(name: str) -> str:
    """Card name without "(...)" or "[...]" tags and without the " - <number>" part."""
    return _TAGS.sub("", name).split(" - ")[0].strip()


def _after_number(name: str) -> str | None:
    """The text after the first " - " that is outside every "(...)" and "[...]", or None."""
    depth = 0
    for i, c in enumerate(name):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and name.startswith(" - ", i):
            return name[i + 3:]
    return None


def stamp_tag(name: str) -> str | None:
    """Every tag after the number, as plain text: "Tsareena - 20/149 (League Challenge) [1st Place]" gives
    "League Challenge 1st Place". Tags before the number ("Boss's Orders [Cyrus] - ...") are part of the name."""
    after = _after_number(name)
    found = [m.group(1) if m.group(1) is not None else m.group(2) for m in _TAG.finditer(name if after is None else after)]
    text = unicodedata.normalize("NFD", " ".join(found))
    text = " ".join("".join(c for c in text if not unicodedata.combining(c)).split())
    return text or None


def tags(name: str) -> list[str]:
    """Folded text of every "(...)" and "[...]" tag in a product name."""
    return [_fold(m.group(1) if m.group(1) is not None else m.group(2)) for m in _TAG.finditer(name)]


def _tag_stamp(tag: str, set_names: frozenset[str]) -> str | None:
    for stamp, rule in _TAG_RULES:
        if rule.fullmatch(tag):
            return stamp
    if tag in _LEAGUE:
        return "league"
    if tag in _POKEMON_CENTER:
        return "pokemoncenter"
    if tag == "w stamped":
        return "wstamp"
    logo = _SET_LOGO.fullmatch(tag)
    if logo and norm(logo.group(1)) in set_names:
        return "setlogo"
    return None


def stamp_of(name: str, group_id: int, set_names: frozenset[str]) -> str | None:
    """Stamp key of a TCGplayer product, or None when it is not a stamped printing.

    ``set_names`` are normalized TCGdex set names: "(<Set> Stamp)" is a set-logo stamp only for a real set name.
    """
    found = {_tag_stamp(tag, set_names) for tag in tags(name)} - {None}
    for stamp in STAMPS:
        if stamp in found:
            return stamp
    return "playpokemon" if group_id == PRIZE_PACK_GROUP else None


def number_of(product: dict) -> str | None:
    """The collector number TCGplayer lists for a product."""
    for field in product.get("extendedData") or []:
        if isinstance(field, dict) and field.get("name") == "Number" and isinstance(field.get("value"), str):
            return field["value"].strip() or None
    return None


def _money(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return round(float(value), 2)


def _rows_by_product(prices_payload: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in (prices_payload or {}).get("results") or []:
        if not isinstance(row, dict) or not isinstance(row.get("subTypeName"), str):
            continue
        product_id = row.get("productId")
        if isinstance(product_id, int) and not isinstance(product_id, bool):
            out.setdefault(product_id, []).append(row)
    return out


def _products(products_payload: dict) -> list[dict]:
    rows = (products_payload or {}).get("results") or []
    return [p for p in rows if isinstance(p, dict) and isinstance(p.get("productId"), int) and isinstance(p.get("name"), str)]


class Catalog:
    """Catalog cards (id, name, setCode, number) and TCGdex sets (id, name, printed total) for the join."""

    def __init__(self, cards: list[dict], sets: list[dict]) -> None:
        self._keys: dict[tuple[str, str, str], list[str]] = {}
        self._by_set: dict[str, list[dict]] = {}
        for card in cards or []:
            if not isinstance(card, dict) or not all(isinstance(card.get(k), str) for k in ("id", "name", "setCode", "number")):
                continue
            self._by_set.setdefault(card["setCode"], []).append(card)
            for key in self.name_keys(card["name"]):
                ids = self._keys.setdefault((card["setCode"], _strip_zeros(card["number"]), key), [])
                if card["id"] not in ids:
                    ids.append(card["id"])
        if not self._by_set:
            raise ValueError("catalog has no cards")
        self._by_total: dict[int, list[str]] = {}
        names = set()
        for entry in sets or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            if isinstance(entry.get("name"), str) and norm(entry["name"]):
                names.add(norm(entry["name"]))
            official = (entry.get("cardCount") or {}).get("official")
            if isinstance(official, int) and not isinstance(official, bool) and official > 0:
                self._by_total.setdefault(official, []).append(entry["id"])
        if not self._by_total:
            raise ValueError("TCGdex returned no sets with a printed total")
        self.set_names = frozenset(names)

    @classmethod
    def load(cls, meta_path: str, sets: list[dict]) -> "Catalog":
        """Catalog from the release meta.json (a list of card rows)."""
        with open(meta_path, encoding="utf-8") as handle:
            cards = json.load(handle)
        if not isinstance(cards, list):
            raise ValueError("catalog meta.json is not a list")
        return cls(cards, sets)

    @staticmethod
    def name_keys(name: str) -> set[str]:
        """A catalog name, with "δ" both spelled out ("δ Rainbow Energy") and dropped ("Beedrill δ")."""
        return {key for key in (norm(name), norm(name, "")) if key}

    def cards_in(self, set_code: str) -> list[dict]:
        return self._by_set.get(set_code, [])

    def _ids(self, set_codes, number: str, name: str) -> list[str]:
        key = norm(name)
        out: list[str] = []
        for code in set_codes:
            for card_id in self._keys.get((code, _strip_zeros(number), key), []):
                if card_id not in out:
                    out.append(card_id)
        return out

    def match(self, name: str, number: str, group_id: int) -> list[str]:
        """Catalog ids a product could be; the caller keeps the product only when there is exactly one."""
        number = (number or "").strip()
        promo = _PROMO_NUMBER.fullmatch(number)
        if promo and promo.group(1) in PROMO_PREFIX:
            return self._ids([PROMO_PREFIX[promo.group(1)]], number, name)
        if group_id == SV_PROMO_GROUP and re.fullmatch(r"\d{1,3}", number):
            return self._ids(["svp"], number, name)
        printed = _PRINTED.fullmatch(number)
        if printed:
            return self._ids(self._by_total.get(int(printed.group(2)), []), printed.group(1), name)
        return []


_REPORT_FIELDS = {
    "stamped": ("products", "joined", "ambiguous", "unmatched", "unpriced", "rows"),
    "byCatalog": ("cards", "joined", "ambiguous", "unmatched", "priced", "reverse"),
}


def _count(report: dict, section: str, key: str, field: str, n: int = 1) -> None:
    entry = report.setdefault(section, {}).setdefault(key, dict.fromkeys(_REPORT_FIELDS[section], 0))
    entry[field] += n


def stamped_rows(catalog: Catalog, group_id: int, products_payload: dict, prices_payload: dict, report: dict) -> list[dict]:
    """One row per priced subtype of every stamped product in a group that joins to exactly one catalog card."""
    prices = _rows_by_product(prices_payload)
    out: list[dict] = []
    for product in _products(products_payload):
        stamp = stamp_of(product["name"], group_id, catalog.set_names)
        if stamp is None:
            continue
        _count(report, "stamped", stamp, "products")
        number = number_of(product)
        ids = catalog.match(base_name(product["name"]), number, group_id) if number else []
        if len(ids) != 1:
            _count(report, "stamped", stamp, "ambiguous" if ids else "unmatched")
            continue
        _count(report, "stamped", stamp, "joined")
        tag = stamp_tag(product["name"])
        rows = []
        for row in prices.get(product["productId"], []):
            market, mid, low = (_money(row.get(k)) for k in ("marketPrice", "midPrice", "lowPrice"))
            if market is None and mid is None and low is None:
                continue
            rows.append({"catalogId": ids[0], "stamp": stamp, "tag": tag, "productId": product["productId"],
                         "subtype": row["subTypeName"], "market": market, "mid": mid, "low": low})
        _count(report, "stamped", stamp, "rows" if rows else "unpriced", len(rows) or 1)
        out += rows
    return out


def by_catalog(catalog: Catalog, group_id: int, products_payload: dict, prices_payload: dict, report: dict) -> dict[str, dict[str, float]]:
    """Subtype prices (market, else mid) per catalog card of one EX set-logo-stamp set, joined by number and name."""
    set_code = EX_STAMP_GROUPS.get(group_id)
    if set_code is None:
        return {}
    by_key: dict[tuple[str, str], list[dict]] = {}
    for product in _products(products_payload):
        number = number_of(product)
        if number:
            by_key.setdefault((_strip_zeros(number.split("/")[0]), norm(base_name(product["name"]))), []).append(product)
    prices = _rows_by_product(prices_payload)
    out: dict[str, dict[str, float]] = {}
    for card in catalog.cards_in(set_code):
        _count(report, "byCatalog", set_code, "cards")
        found = {p["productId"]: p for key in Catalog.name_keys(card["name"])
                 for p in by_key.get((_strip_zeros(card["number"]), key), [])}
        if len(found) != 1:
            _count(report, "byCatalog", set_code, "ambiguous" if found else "unmatched")
            continue
        _count(report, "byCatalog", set_code, "joined")
        subtypes: dict[str, float] = {}
        for row in prices.get(next(iter(found)), []):
            price = _money(row.get("marketPrice")) or _money(row.get("midPrice"))
            if price is not None:
                subtypes[row["subTypeName"]] = price
        if subtypes:
            out[card["id"]] = subtypes
            _count(report, "byCatalog", set_code, "priced")
            if "Reverse Holofoil" in subtypes:
                _count(report, "byCatalog", set_code, "reverse")
    return out
