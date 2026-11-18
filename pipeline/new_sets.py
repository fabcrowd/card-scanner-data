"""New cards: find cards the extension's catalog does not have yet, embed their art like the extension does, and
write them as catalogExtra.bin (the memory layout plus a "cards" list, see tools/extension-wasm/lib/bundle.js).

The card source is TCGdex (English), the same public source and id scheme the shipped catalog was built from
(tools/vellum-ai/catalog/build_catalog.py): a card id is TCGdex's own, like "base1-4". Every network call goes
through fetch / fetch_image, so tests pass stubs. At most MAX_NEW cards are added per run; cards whose image
fails stay unknown and are retried the next night.

catalogExtra.bin is cumulative: each run starts from the previous file (from the live release) and appends.
"""
import json
import re
import struct
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .pack import DIM, MAX_HEADER, SCALE, VERSION

TCGDEX = "https://api.tcgdex.net/v2/en"
# Same pattern as DIGITAL_ONLY in tools/extension-wasm/lib/search.js (tests/test_drift.py keeps them equal): TCG Pocket
# sets are digital-only, and the extension drops those ids at search time.
DIGITAL_ONLY = re.compile(r"^(A\d|B\d|P-A)")
# Cards TCGdex lists without a picture get one from pokemontcg.io when number and name match (see _fallback_pictures).
# The source is the static per-set files of github.com/PokemonTCG/pokemon-tcg-data (one cheap GET per set and spelling).
POKEMONTCG_DATA = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en"
POKEMONTCG_IMAGE_HOST = "https://images.pokemontcg.io/"
# Every TCGdex picture in the shipped files so far comes from this host.
TCGDEX_IMAGE_HOST = "https://assets.tcgdex.net/"
MAX_FALLBACK_SETS = 150
FALLBACK_SECONDS = 600  # wall-clock budget of the whole picture-fallback step in one run
EMBED_SECONDS = 1800  # wall-clock budget of the picture download and embedding loop in one run
SET_FAILURES = 25  # picture failures in one set after which the rest of its cards wait for the next night
ID_MAX = 64  # the signer refuses a bundle with a longer card id (scripts/sign_release.py)
# TCGdex set id -> pokemon-tcg-data id where the spelling rules do not reach it (found by comparing set names). Tried after
# the rule-based spellings; a wrong alias only gives no match, because number and name must still agree.
SET_ALIASES = {
    "cel25cc": "cel25c", "bog": "bp", "hgssp": "hsp",
    "tk-ex-latia": "tk1a", "tk-ex-latio": "tk1b", "tk-ex-p": "tk2a", "tk-ex-m": "tk2b",
    "2011bw": "mcd11", "2012bw": "mcd12", "2014xy": "mcd14", "2015xy": "mcd15", "2016xy": "mcd16",
    "2017sm": "mcd17", "2018sm": "mcd18", "2019sm": "mcd19", "2021swsh": "mcd21", "2022swsh": "mcd22",
}
RETRY_DELAY = 5  # seconds before the one retry of a failed fallback request
IMAGE_URL_MAX = 300
MAX_NEW = 500
MAX_TOTAL = 20000
REQUEST_DELAY = 0.1  # seconds between requests to the card source
MAX_IMAGE_BYTES = 8 * 1024 * 1024
FIELD_MAX = 200


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    """A download may be redirected only within the host (and https) it asked for; anything else is a failed fetch."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(target)
        if new.scheme != "https" or new.netloc.lower() != old.netloc.lower():
            raise urllib.error.HTTPError(req.full_url, code, f"redirect to {new.netloc or target} refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, target)


_OPENER = urllib.request.build_opener(_SameHostRedirects)


def _get(url, limit):
    if not url.startswith("https://"):
        raise ValueError(f"not https: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "card-scanner-nightly/1"})
    with _OPENER.open(req, timeout=60) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url} is larger than {limit} bytes")
    return data


def fetch_json(url):
    return json.loads(_get(url, 16 * 1024 * 1024).decode("utf-8"))


def fetch_bytes(url):
    return _get(url, MAX_IMAGE_BYTES)


def set_of(card_id):
    return card_id.rsplit("-", 1)[0]


def _plain(x):
    """No control (Cc) or format (Cf) characters: accented letters and symbols such as é, ♀, δ are fine."""
    return not any(unicodedata.category(ch) in ("Cc", "Cf") for ch in str(x))


def _text(x):
    return isinstance(x, str) and 0 < len(x) <= FIELD_MAX and _plain(x)


def _has_control_text(summary, detail, raw):
    """Whether any text field of a raw TCGdex card has a control or format character."""
    values = (raw.get("id"), raw.get("name"), raw.get("localId"), detail.get("name"), summary.get("id"), summary.get("name"))
    return any(v is not None and not _plain(v) for v in values)


def _card(set_summary, detail, raw, need_picture=True):
    """The catalogExtra entry for one TCGdex card, or None when it lacks an id, number or picture, has an id the signer
    or the extension would refuse, or a picture from another host. With need_picture=False a card whose only gap is a
    missing picture comes back with "image": None."""
    image = raw.get("image")
    local = raw.get("localId")
    has_picture = isinstance(image, str) and image != ""
    card = {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "set": set_summary.get("id"),
        "setName": detail.get("name") or set_summary.get("name"),
        "number": str(local) if local is not None else None,
        "image": f"{image}/high.webp" if has_picture else None,
    }
    if not all(_text(card[k]) for k in ("id", "name", "set", "setName", "number")):
        return None
    if len(card["id"]) > ID_MAX or not card["id"].startswith(f"{card['set']}-") or DIGITAL_ONLY.match(card["id"]):
        return None
    if card["image"] is None:
        return None if need_picture else card
    return card if card["image"].startswith(TCGDEX_IMAGE_HOST) and len(card["image"]) <= IMAGE_URL_MAX else None


def _set_ids(set_code):
    """TCGdex set id -> pokemontcg.io spellings, in the order tried (sm7.5 -> sm75, swsh12.5gg -> swsh12pt5gg, sv03 -> sv3).
    Follows tools/vellum-ai/catalog/backfill_coverage.py."""
    forms = [set_code, set_code.replace(".", ""), set_code.replace(".5", "pt5")]
    forms += [re.sub(r"([a-z])0+(\d)", r"\1\2", f) for f in list(forms)]
    if set_code in SET_ALIASES:
        forms.append(SET_ALIASES[set_code])
    return list(dict.fromkeys(forms))


def _number_key(number):
    return re.sub(r"^([A-Za-z]*)0+(\d)", r"\1\2", str(number)).casefold()


def _name_key(name):
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(ch))
    return "".join(ch for ch in plain.casefold() if ch.isalnum())


def _is_404(err):
    # The real fetch (urllib.request.urlopen) raises HTTPError carrying the status code.
    return isinstance(err, urllib.error.HTTPError) and err.code == 404


def _pokemontcg_file(sid, fetch, sleep):
    """The card list of one pokemontcg.io set file, or None when there is no file under that id (HTTP 404). Any other
    failure is retried once after RETRY_DELAY seconds, then raised."""
    url = f"{POKEMONTCG_DATA}/{urllib.parse.quote(sid, safe='')}.json"
    for attempt in (1, 2):
        sleep(REQUEST_DELAY)
        try:
            cards = fetch(url)
            return cards if isinstance(cards, list) else []
        except Exception as err:  # noqa: BLE001 - only a 404 is an answer; the rest is retried, then reported
            if _is_404(err):
                return None
            if attempt == 2:
                raise
            sleep(RETRY_DELAY)


def _fallback_pictures(code, pictureless, fetch, sleep, tcgdex_cards=()):
    """The pictureless cards of one set with a pokemontcg.io picture where both number and name match one card there.
    A (number, name) key that occurs more than once on the pokemontcg.io side, or among `tcgdex_cards` (the raw cards of
    this set on the TCGdex side), is not used. Any error means no picture for this set tonight.
    Logs one line: which file was used and how many cards matched."""
    try:
        key_counts = {}
        for raw in tcgdex_cards:
            if isinstance(raw, dict) and raw.get("localId") is not None and raw.get("name") is not None:
                key = (_number_key(raw["localId"]), _name_key(raw["name"]))
                key_counts[key] = key_counts.get(key, 0) + 1
        ambiguous = {k for k, n in key_counts.items() if n > 1}
        theirs, used = [], None
        for sid in _set_ids(code):
            theirs = _pokemontcg_file(sid, fetch, sleep) or []
            if theirs:
                used = sid
                break
        by_key = {}
        for t in theirs:
            # Anything that is not a card with string number and name and an images dict holding a string is skipped.
            images = t.get("images") if isinstance(t, dict) else None
            if not isinstance(images, dict) or not isinstance(images.get("large"), str):
                continue
            if not isinstance(t.get("number"), str) or not isinstance(t.get("name"), str):
                continue
            by_key.setdefault((_number_key(t["number"]), _name_key(t["name"])), []).append(images["large"])
        out = []
        for card in pictureless:
            key = (_number_key(card["number"]), _name_key(card["name"]))
            urls = [] if key in ambiguous else by_key.get(key, [])
            url = urls[0] if len(urls) == 1 else None
            if isinstance(url, str) and url.startswith(POKEMONTCG_IMAGE_HOST) and len(url) <= IMAGE_URL_MAX:
                out.append({**card, "image": url})
    except Exception as err:  # noqa: BLE001 - one bad set must not sink the night; these cards are retried the next night
        print(f"skip picture fallback for set {code}: {err}")
        return []
    print(f"fallback {code}: file {used or 'none'}, matched {len(out)} of {len(pictureless)} cards")
    return out


def find_new_cards(known_ids, fetch=fetch_json, sleep=time.sleep, cap=MAX_NEW, clock=time.monotonic, per_set_cap=None):
    """([card dicts not in known_ids, at most cap], [set codes they belong to]). A set whose known cards already
    reach its official count is not even opened, so a quiet night costs one request. Digital-only (Pocket) sets are
    never opened. Cards TCGdex lists without a picture get one from pokemontcg.io (at most MAX_FALLBACK_SETS sets a run,
    and only while the fallback has used less than FALLBACK_SECONDS of wall-clock time, measured with `clock`).
    No id is returned twice, and none that is in known_ids. One set gives at most per_set_cap cards (default: cap), so
    a newest set whose pictures all fail cannot fill the whole list."""
    per_set_cap = cap if per_set_cap is None else per_set_cap
    seen = set(known_ids)
    per_set = {}
    for card_id in seen:
        per_set[set_of(card_id)] = per_set.get(set_of(card_id), 0) + 1
    try:
        try:
            listing = fetch(f"{TCGDEX}/sets")
        except Exception:  # noqa: BLE001 - the set list is retried once, then the night adds no new cards
            sleep(RETRY_DELAY)
            listing = fetch(f"{TCGDEX}/sets")
        if not isinstance(listing, list):
            raise ValueError("the set list is not a list")
    except Exception as err:  # noqa: BLE001 - nothing new tonight; the rest of the bundle is unaffected
        print(f"new sets skipped tonight: {err}")
        return [], []
    found, sets, fallback_sets, started, out_of_time, refused = [], [], 0, None, False, 0
    # The set list carries no release date, only its own order, which is release order: walk it newest first so the
    # cards users meet most are not starved by old sets when the cap bites.
    for summary in reversed(listing):
        if len(found) >= cap:
            break
        if not isinstance(summary, dict):
            continue
        code = summary.get("id")
        count = summary.get("cardCount")
        total = count.get("total") if isinstance(count, dict) else None
        if not _text(code) or code in (".", "..") or DIGITAL_ONLY.match(code) or (isinstance(total, int) and per_set.get(code, 0) >= total):
            continue
        sleep(REQUEST_DELAY)
        try:
            detail = fetch(f"{TCGDEX}/sets/{urllib.parse.quote(code, safe='')}")
        except Exception as err:  # noqa: BLE001 - one bad set must not stop the night
            print(f"skip set {code}: {err}")
            continue
        raw_cards = detail.get("cards") if isinstance(detail, dict) else None
        if not isinstance(raw_cards, list):
            print(f"skip set {code}: unexpected set detail")
            continue
        pictureless, queued, taken = [], set(), 0
        for raw in raw_cards:
            if not isinstance(raw, dict):
                print(f"skip card in set {code}: not an object")
                continue
            if _has_control_text(summary, detail, raw):
                refused += 1
                continue
            card = _card(summary, detail, raw)
            if card:
                if card["id"] not in seen and len(found) < cap and taken < per_set_cap:
                    seen.add(card["id"])
                    found.append(card)
                    taken += 1
                    if code not in sets:
                        sets.append(code)
                continue
            bare = _card(summary, detail, raw, need_picture=False)
            if bare and bare["image"] is None and bare["id"] not in seen and bare["id"] not in queued:
                queued.add(bare["id"])
                pictureless.append(bare)
        if not pictureless or len(found) >= cap or fallback_sets >= MAX_FALLBACK_SETS or out_of_time:
            continue
        now = clock()
        if started is None:
            started = now
        elif now - started >= FALLBACK_SECONDS:
            print(f"fallback time budget ({FALLBACK_SECONDS} s) used up: the remaining sets get no picture fallback tonight")
            out_of_time = True
            continue
        fallback_sets += 1
        for card in _fallback_pictures(code, pictureless, fetch, sleep, raw_cards):
            if card["id"] not in seen and len(found) < cap and taken < per_set_cap:
                seen.add(card["id"])
                found.append(card)
                taken += 1
                if code not in sets:
                    sets.append(code)
    if refused:
        print(f"refused {refused} cards with control or format characters in their text")
    return found, sets


def write_extra(cards, vectors, tag=None):
    """catalogExtra.bin bytes: cards (list of dicts) and their int8 rows (n, 512). tag = {"model", "crop"} of the
    recognizer that made the rows; it goes into the header so a later run can tell."""
    vectors = np.asarray(vectors, dtype=np.int8).reshape(-1, DIM)
    if len(cards) != len(vectors):
        raise ValueError("one row per card is required")
    head = {"v": VERSION, "dim": DIM, "entries": len(cards), "cardIds": [c["id"] for c in cards], "cards": cards}
    if tag:
        head = {**head, "model": tag["model"], "crop": tag["crop"]}
    header = json.dumps(head, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack("<I", len(header)) + header + vectors.tobytes()


def extra_tag(data):
    """(model, crop) from a catalogExtra.bin header; None for a key the header does not have."""
    if len(data) < 4:
        raise ValueError("catalogExtra too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise ValueError("bad header length")
    try:
        header = json.loads(data[4 : 4 + n].decode("utf-8"))
    except ValueError as err:
        raise ValueError("bad header") from err
    if not isinstance(header, dict):
        raise ValueError("bad header")
    return header.get("model"), header.get("crop")


def read_extra(data):
    """(cards, int8 rows) of a catalogExtra.bin; ValueError when it is malformed."""
    if len(data) < 4:
        raise ValueError("catalogExtra too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise ValueError("bad header length")
    try:
        header = json.loads(data[4 : 4 + n].decode("utf-8"))
    except ValueError as err:
        raise ValueError("bad header") from err
    if not isinstance(header, dict):
        raise ValueError("bad header")
    cards = header.get("cards")
    if header.get("v") != VERSION or header.get("dim") != DIM or not isinstance(cards, list) or header.get("entries") != len(cards):
        raise ValueError("unsupported catalogExtra")
    if not all(isinstance(c, dict) and isinstance(c.get("id"), str) for c in cards):
        raise ValueError("cards must be objects with a string id")
    if header.get("cardIds") != [c["id"] for c in cards]:
        raise ValueError("card ids do not match cards")
    body = data[4 + n :]
    if len(body) != len(cards) * DIM:
        raise ValueError("body size does not match entries")
    return cards, np.frombuffer(body, dtype=np.int8).reshape(len(cards), DIM)


def build(known_ids, previous, embedder, fetch=fetch_json, fetch_image=fetch_bytes, sleep=time.sleep, cap=MAX_NEW, tag=None,
          clock=time.monotonic):
    """Appends up to cap new cards to the previous catalogExtra.bin (bytes or None). A card whose picture fails does not
    use a slot: the next candidate takes it, and at most 2 * cap cards are tried, so a broken host cannot keep the step running.

    known_ids: ids in the shipped catalog. Returns {"bytes": new file or None when there is nothing at all,
    "added": [ids], "sets": [set codes of the added cards], "total": rows in the file}.
    tag: {"model", "crop"} of the recognizer; new files carry it, and a previous file with another (or no) tag is
    dropped so rows made with an old model are never mixed in.
    """
    cards, rows = [], np.zeros((0, DIM), dtype=np.int8)
    if previous:
        try:
            old_tag = extra_tag(previous)
            cards, rows = read_extra(previous)
        except Exception:  # noqa: BLE001 - a corrupt previous file is treated like one from another model
            print("previous catalogExtra is unreadable: rebuilding")
            previous, cards, rows = None, [], np.zeros((0, DIM), dtype=np.int8)
        else:
            if tag is not None and old_tag != (tag["model"], tag["crop"]):
                print("catalogExtra was made with another model: rebuilding")
                previous, cards, rows = None, [], np.zeros((0, DIM), dtype=np.int8)
    dropped = []
    if cards:
        keep, taken = [], set(known_ids)
        for i, card in enumerate(cards):
            if _previous_row_ok(card, taken):
                taken.add(card["id"])
                keep.append(i)
            else:
                dropped.append(card["id"])
        if dropped:
            print(f"dropped {len(dropped)} previous catalogExtra rows: {', '.join(dropped[:5])}{' ...' if len(dropped) > 5 else ''}")
            cards, rows = [cards[i] for i in keep], rows[keep]
    room = max(0, min(cap, MAX_TOTAL - len(cards)))
    wanted = find_new_cards(set(known_ids) | {c["id"] for c in cards}, fetch, sleep, 2 * room, clock, per_set_cap=room)[0] if room else []
    added, new_rows = [], []
    failures, skipped = {}, set()
    started = clock()
    for card in wanted:
        if len(added) >= room:
            break
        if clock() - started >= EMBED_SECONDS:
            print(f"embed time budget ({EMBED_SECONDS} s) used up: the remaining cards wait for the next night")
            break
        if card["set"] in skipped:
            continue
        sleep(REQUEST_DELAY)
        try:
            vector = np.asarray(embedder(fetch_image(card["image"]), trusted=True), dtype=np.float32)
        except Exception as err:  # noqa: BLE001 - a missing or undecodable picture is retried tomorrow
            print(f"skip card {card['id']}: {err}")
            failures[card["set"]] = failures.get(card["set"], 0) + 1
            if failures[card["set"]] >= SET_FAILURES:
                print(f"skip the rest of set {card['set']} tonight: {failures[card['set']]} pictures failed")
                skipped.add(card["set"])
            continue
        if vector.shape != (DIM,) or not np.all(np.isfinite(vector)):
            continue
        new_rows.append(np.clip(np.rint(vector * SCALE), -SCALE, SCALE).astype(np.int8))
        added.append(card)
    if not added:
        # Nothing new keeps the previous bytes as they were, unless rows were dropped: then the file is written again without them.
        if dropped:
            kept_file = write_extra(cards, rows, tag) if cards else None
        else:
            kept_file = previous
        return {"bytes": kept_file or None, "added": [], "sets": [], "total": len(cards), "dropped": len(dropped)}
    return {
        "bytes": write_extra(cards + added, np.vstack([rows] + [r[None] for r in new_rows]), tag),
        "added": [c["id"] for c in added],
        "sets": sorted({c["set"] for c in added}),
        "total": len(cards) + len(added),
        "dropped": len(dropped),
    }


def _previous_row_ok(card, taken):
    """The checks a fresh card passes, for a row carried over from the previous file; `taken` holds the catalog ids and the
    ids kept so far (so a duplicate, or an id the catalog now has, is dropped)."""
    cid = card["id"]
    image = card.get("image")
    return bool(
        all(_text(card.get(k)) for k in ("id", "name", "set", "setName", "number"))
        and len(cid) <= ID_MAX and cid.startswith(f"{card['set']}-") and not DIGITAL_ONLY.match(cid)
        and isinstance(image, str) and image.startswith((TCGDEX_IMAGE_HOST, POKEMONTCG_IMAGE_HOST)) and len(image) <= IMAGE_URL_MAX
        and cid not in taken
    )
