"""New cards: find cards the extension's catalog does not have yet, embed their art like the extension does, and
write them as catalogExtra.bin (the memory layout plus a "cards" list, see tools/extension-wasm/lib/bundle.js).

The card source is TCGdex (English), the same public source and id scheme the shipped catalog was built from
(tools/vellum-ai/catalog/build_catalog.py): a card id is TCGdex's own, like "base1-4". Every network call goes
through fetch / fetch_image, so tests pass stubs. At most MAX_NEW cards are added per run; cards whose image
fails stay unknown and are retried the next night.

catalogExtra.bin is cumulative: each run starts from the previous file (from the live release) and appends.
"""
import json
import struct
import time
import urllib.request

import numpy as np

from .pack import DIM, MAX_HEADER, SCALE, VERSION

TCGDEX = "https://api.tcgdex.net/v2/en"
MAX_NEW = 500
MAX_TOTAL = 20000
REQUEST_DELAY = 0.1  # seconds between requests to the card source
MAX_IMAGE_BYTES = 8 * 1024 * 1024
FIELD_MAX = 200


def _get(url, limit):
    if not url.startswith("https://"):
        raise ValueError(f"not https: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "card-scanner-nightly/1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
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


def _text(x):
    return isinstance(x, str) and 0 < len(x) <= FIELD_MAX


def _card(set_summary, detail, raw):
    """The catalogExtra entry for one TCGdex card, or None when it lacks an id, number or picture."""
    image = raw.get("image")
    local = raw.get("localId")
    card = {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "set": set_summary.get("id"),
        "setName": detail.get("name") or set_summary.get("name"),
        "number": str(local) if local is not None else None,
        "image": f"{image}/high.webp" if isinstance(image, str) and image.startswith("https://") else None,
    }
    ok = all(_text(card[k]) for k in ("id", "name", "set", "setName", "number")) and card["image"] and len(card["image"]) <= 300
    return card if ok else None


def find_new_cards(known_ids, fetch=fetch_json, sleep=time.sleep, cap=MAX_NEW):
    """([card dicts not in known_ids, at most cap], [set codes they belong to]). A set whose known cards already
    reach its official count is not even opened, so a quiet night costs one request."""
    known_ids = set(known_ids)
    per_set = {}
    for card_id in known_ids:
        per_set[set_of(card_id)] = per_set.get(set_of(card_id), 0) + 1
    found, sets = [], []
    for summary in fetch(f"{TCGDEX}/sets"):
        if len(found) >= cap:
            break
        code = summary.get("id")
        total = (summary.get("cardCount") or {}).get("total")
        if not _text(code) or (isinstance(total, int) and per_set.get(code, 0) >= total):
            continue
        sleep(REQUEST_DELAY)
        try:
            detail = fetch(f"{TCGDEX}/sets/{code}")
        except Exception as err:  # noqa: BLE001 - one bad set must not stop the night
            print(f"skip set {code}: {err}")
            continue
        for raw in detail.get("cards") or []:
            card = _card(summary, detail, raw)
            if card and card["id"] not in known_ids and len(found) < cap:
                found.append(card)
                if code not in sets:
                    sets.append(code)
    return found, sets


def write_extra(cards, vectors):
    """catalogExtra.bin bytes: cards (list of dicts) and their int8 rows (n, 512)."""
    vectors = np.asarray(vectors, dtype=np.int8).reshape(-1, DIM)
    if len(cards) != len(vectors):
        raise ValueError("one row per card is required")
    header = json.dumps(
        {"v": VERSION, "dim": DIM, "entries": len(cards), "cardIds": [c["id"] for c in cards], "cards": cards},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return struct.pack("<I", len(header)) + header + vectors.tobytes()


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
    cards = header.get("cards")
    if header.get("v") != VERSION or header.get("dim") != DIM or not isinstance(cards, list) or header.get("entries") != len(cards):
        raise ValueError("unsupported catalogExtra")
    if header.get("cardIds") != [c.get("id") for c in cards if isinstance(c, dict)]:
        raise ValueError("card ids do not match cards")
    body = data[4 + n :]
    if len(body) != len(cards) * DIM:
        raise ValueError("body size does not match entries")
    return cards, np.frombuffer(body, dtype=np.int8).reshape(len(cards), DIM)


def build(known_ids, previous, embedder, fetch=fetch_json, fetch_image=fetch_bytes, sleep=time.sleep, cap=MAX_NEW):
    """Appends up to cap new cards to the previous catalogExtra.bin (bytes or None).

    known_ids: ids in the shipped catalog. Returns {"bytes": new file or None when there is nothing at all,
    "added": [ids], "sets": [set codes of the added cards], "total": rows in the file}.
    """
    cards, rows = read_extra(previous) if previous else ([], np.zeros((0, DIM), dtype=np.int8))
    room = max(0, min(cap, MAX_TOTAL - len(cards)))
    wanted = find_new_cards(set(known_ids) | {c["id"] for c in cards}, fetch, sleep, room)[0] if room else []
    added, new_rows = [], []
    for card in wanted:
        sleep(REQUEST_DELAY)
        try:
            vector = np.asarray(embedder(fetch_image(card["image"]), trusted=True), dtype=np.float32)
        except Exception as err:  # noqa: BLE001 - a missing or undecodable picture is retried tomorrow
            print(f"skip card {card['id']}: {err}")
            continue
        if vector.shape != (DIM,) or not np.all(np.isfinite(vector)):
            continue
        new_rows.append(np.clip(np.rint(vector * SCALE), -SCALE, SCALE).astype(np.int8))
        added.append(card)
    if not added:
        return {"bytes": previous or None, "added": [], "sets": [], "total": len(cards)}
    return {
        "bytes": write_extra(cards + added, np.vstack([rows] + [r[None] for r in new_rows])),
        "added": [c["id"] for c in added],
        "sets": sorted({c["set"] for c in added}),
        "total": len(cards) + len(added),
    }
