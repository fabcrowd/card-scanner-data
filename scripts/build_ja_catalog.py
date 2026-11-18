"""Builds a Japanese Pokemon card catalog (card list, pictures, embeddings) from TCGCSV, a keyless TCGplayer mirror.

Stages, chosen by sub-command; everything lands under data/ja-catalog/ and every stage can be rerun:
  list     groups + products -> products.jsonl (resumable per group via groups_done.json)
  images   download the 1000x1000 picture of each card -> images/<productId>.jpg
  dedupe   find placeholder pictures (one hash shared by more than 20 products) and non-card-shaped pictures
  embed    CLIP embeddings of the real card pictures -> emb_<tag>.npz (ids, vectors), resumable
  pack     int8 catalog the extension downloads -> catalog-ja.bin.gz (+ .json with its sha256)

Network access goes only through fetch_json and fetch_bytes (https only); tests replace them.
"""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = str(ROOT / "data" / "ja-catalog")
CATEGORY = 85
GROUPS_URL = f"https://tcgcsv.com/tcgplayer/{CATEGORY}/groups"
IMAGE_URL = "https://tcgplayer-cdn.tcgplayer.com/product/{pid}_in_1000x1000.jpg"
USER_AGENT = "card-scanner-catalog/1"
TIMEOUT_S = 60
LIST_DELAY_S = 0.3
LIST_RETRY_WAIT_S = 5
IMAGE_RETRY_WAIT_S = 3
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MIN_IMAGE_BYTES = 2000
MAX_WORKERS = 4
MAX_RATE = 6.0
REFUSAL_STATUSES = (403, 429)
REFUSAL_LIMIT = 25
PROGRESS_EVERY = 500
PLACEHOLDER_MIN_PRODUCTS = 21  # a hash shared by more than 20 products
PLACEHOLDER_EXAMPLES = 5
SHAPE_EXAMPLES = 10
CARD_RATIO = (0.68, 0.75)
SAVE_EVERY = 2000
MAX_THREADS = 8
EXIT_REFUSED = 3
BELOW_NORMAL_PRIORITY_CLASS = 0x4000

sleep = time.sleep


class FetchError(Exception):
    """A failed request; `status` is the HTTP status or None."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


# ---- network (the only two functions that touch it) ----

def _read_limited(url, limit):
    if not url.startswith("https://"):
        raise FetchError(f"refusing non-https url {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
            data = resp.read(limit + 1)
    except urllib.error.HTTPError as err:
        raise FetchError(f"HTTP {err.code} for {url}", err.code) from err
    except (urllib.error.URLError, OSError) as err:
        raise FetchError(f"{type(err).__name__} for {url}: {err}") from err
    if len(data) > limit:
        raise FetchError(f"response over {limit} bytes for {url}")
    return data


def fetch_json(url):
    data = _read_limited(url, MAX_JSON_BYTES)
    try:
        return json.loads(data)
    except ValueError as err:
        raise FetchError(f"bad JSON from {url}: {err}") from err


def fetch_bytes(url):
    return _read_limited(url, MAX_IMAGE_BYTES)


def fetch_json_retry(url):
    """fetch_json with one retry after a pause, for any error (incl. 429/503)."""
    try:
        return fetch_json(url)
    except FetchError as err:
        print(f"  retrying in {LIST_RETRY_WAIT_S}s after: {err}")
        sleep(LIST_RETRY_WAIT_S)
        return fetch_json(url)


# ---- small file helpers ----

def write_atomic(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- list stage ----

def clean_name(name, number):
    suffix = f" - {number}"
    return name[: -len(suffix)] if number and name.endswith(suffix) else name


def card_row(product, group):
    """One products.jsonl row, or None for a product without a Number (sealed product)."""
    ext = {e.get("name"): e.get("value") for e in product.get("extendedData") or []}
    number = ext.get("Number")
    if not number:
        return None
    return {
        "productId": int(product["productId"]),
        "groupId": int(group["groupId"]),
        "group": group.get("name"),
        "abbr": group.get("abbreviation"),
        "published": (group.get("publishedOn") or "")[:10],
        "name": clean_name(product.get("name") or "", number),
        "number": number,
        "rarity": ext.get("Rarity") or None,
        "imageCount": int(product.get("imageCount") or 0),
    }


def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        return {"done": state.get("done", {}), "failed": state.get("failed", {})}
    return {"done": {}, "failed": {}}


def prune_products(path, done):
    """Drop rows of groups not marked done (a crash between the append and the state save)."""
    rows = read_jsonl(path)
    kept = [r for r in rows if str(r["groupId"]) in done]
    if len(kept) != len(rows):
        write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode("utf-8"))


def list_group(group, products_path, state, state_path):
    gid = str(group["groupId"])
    try:
        products = fetch_json_retry(f"https://tcgcsv.com/tcgplayer/{CATEGORY}/{gid}/products")["results"]
    except (FetchError, KeyError, TypeError) as err:
        print(f"  group {gid} ({group.get('name')}) failed, skipped: {err}")
        state["failed"][gid] = str(err)
        write_atomic(state_path, json.dumps(state).encode("utf-8"))
        return
    rows = [r for r in (card_row(p, group) for p in products) if r]
    append_jsonl(products_path, rows)
    state["done"][gid] = {"cards": len(rows), "dropped": len(products) - len(rows)}
    state["failed"].pop(gid, None)
    write_atomic(state_path, json.dumps(state).encode("utf-8"))


def list_summary(groups_total, state, rows):
    return {
        "groups_total": groups_total,
        "groups_done": len(state["done"]),
        "groups_failed": len(state["failed"]),
        "cards": len(rows),
        "dropped_no_number": sum(g["dropped"] for g in state["done"].values()),
        "image_count_zero": sum(1 for r in rows if r["imageCount"] == 0),
        "per_year": dict(sorted(Counter((r["published"] or "unknown")[:4] for r in rows).items())),
    }


def list_stage(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    products_path = os.path.join(out_dir, "products.jsonl")
    state_path = os.path.join(out_dir, "groups_done.json")
    state = load_state(state_path)
    prune_products(products_path, state["done"])
    groups = fetch_json_retry(GROUPS_URL)["results"]
    sleep(LIST_DELAY_S)
    for group in groups:
        if str(group["groupId"]) in state["done"]:
            continue
        list_group(group, products_path, state, state_path)
        sleep(LIST_DELAY_S)
    summary = list_summary(len(groups), state, read_jsonl(products_path))
    print_summary(summary)
    return summary


def print_summary(summary):
    for key, value in summary.items():
        print(f"{key}: {value}")


# ---- images stage ----

def clamp_workers(n):
    return max(1, min(int(n), MAX_WORKERS))


def clamp_rate(rate):
    if rate <= 0:
        raise ValueError("--rate must be above 0")
    return min(float(rate), MAX_RATE)


def make_limiter(rate, clock=time.monotonic, sleep=time.sleep):
    """Returns wait(): blocks so that all threads together stay under `rate` calls per second (capped at 6)."""
    interval = 1.0 / clamp_rate(rate)
    lock = threading.Lock()
    next_slot = [0.0]

    def wait():
        with lock:
            now = clock()
            slot = max(now, next_slot[0])
            next_slot[0] = slot + interval
        if slot > now:
            sleep(slot - now)

    return wait


def get_image(pid, limiter):
    """(bytes, None) or (None, FetchError). One retry after 3 s, except on a 403 refusal."""
    err = None
    for attempt in range(2):
        limiter()
        try:
            return fetch_bytes(IMAGE_URL.format(pid=pid)), None
        except FetchError as e:
            err = e
            if e.status == 403 or attempt == 1:
                break
            sleep(IMAGE_RETRY_WAIT_S)
    return None, err


def download_images(cards, out_dir, workers=MAX_WORKERS, rate=5, limiter=None):
    """Downloads each card's picture. Returns counts plus stopped=True if the host refused 25 times in a row."""
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    failures_path = os.path.join(out_dir, "image_failures.jsonl")
    limiter = limiter or make_limiter(rate)
    stats = {"downloaded": 0, "already": 0, "failed": 0, "bytes": 0, "stopped": False}
    streak = [0]
    stop = threading.Event()
    lock = threading.Lock()
    done_count = [0]

    def record(kind, size=0, failure=None):
        with lock:
            stats[kind] += 1
            stats["bytes"] += size
            done_count[0] += 1
            streak[0] = streak[0] + 1 if failure and failure.get("status") in REFUSAL_STATUSES else 0
            if failure:
                append_jsonl(failures_path, [failure])
            if streak[0] >= REFUSAL_LIMIT:
                stats["stopped"] = True
                stop.set()
            if done_count[0] % PROGRESS_EVERY == 0:
                print(f"  {done_count[0]} files: {stats['downloaded']} downloaded, {stats['already']} there, "
                      f"{stats['failed']} failed")

    def work(card):
        if stop.is_set():
            return
        pid = int(card["productId"])
        dest = os.path.join(images_dir, f"{pid}.jpg")
        if os.path.exists(dest) and os.path.getsize(dest) > MIN_IMAGE_BYTES:
            return record("already")
        data, err = get_image(pid, limiter)
        if err is None and len(data) <= MIN_IMAGE_BYTES:
            err = FetchError(f"only {len(data)} bytes", None)
        if err is not None:
            return record("failed", failure={"productId": pid, "status": err.status, "error": str(err)})
        write_atomic(dest, data)
        record("downloaded", len(data))

    with ThreadPoolExecutor(max_workers=clamp_workers(workers)) as pool:
        list(pool.map(work, cards))
    return stats


def images_stage(out_dir, limit=None, workers=MAX_WORKERS, rate=5):
    rows = read_jsonl(os.path.join(out_dir, "products.jsonl"))
    cards = sorted((r for r in rows if r["imageCount"] > 0), key=lambda r: r["productId"])
    if limit is not None:
        cards = cards[:limit]
    print(f"images: {len(cards)} cards to check, {clamp_workers(workers)} workers, {clamp_rate(rate)}/s")
    stats = download_images(cards, out_dir, workers, rate)
    if stats["stopped"]:
        print("the image host is refusing requests; stopped")
    print(f"downloaded: {stats['downloaded']}, already there: {stats['already']}, failed: {stats['failed']}, "
          f"bytes: {stats['bytes']}")
    return stats


# ---- dedupe stage ----

def is_card_shaped(width, height):
    return height > 0 and CARD_RATIO[0] <= width / height <= CARD_RATIO[1]


def find_placeholders(hashes):
    """hash -> {count, examples} for every picture hash shared by more than 20 products."""
    groups = {}
    for pid, digest in hashes.items():
        groups.setdefault(digest, []).append(pid)
    return {
        digest: {"count": len(pids), "examples": sorted(pids)[:PLACEHOLDER_EXAMPLES]}
        for digest, pids in groups.items()
        if len(pids) >= PLACEHOLDER_MIN_PRODUCTS
    }


def image_files(out_dir):
    images_dir = os.path.join(out_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    ids = [int(n[:-4]) for n in os.listdir(images_dir) if n.endswith(".jpg") and n[:-4].isdigit()]
    return [(pid, os.path.join(images_dir, f"{pid}.jpg")) for pid in sorted(ids)]


def image_size(data):
    """(width, height) from the header, or None if the bytes do not decode."""
    try:
        return Image.open(io.BytesIO(data)).size
    except (OSError, ValueError) as err:
        print(f"  cannot read picture header: {err}")
        return None


def dedupe_stage(out_dir):
    hashes, odd, unreadable = {}, [], 0
    for pid, path in image_files(out_dir):
        with open(path, "rb") as fh:
            data = fh.read()
        hashes[pid] = hashlib.sha256(data).hexdigest()
        size = image_size(data)
        if size is None:
            unreadable += 1
        elif not is_card_shaped(*size):
            odd.append(pid)
    placeholders = find_placeholders(hashes)
    write_atomic(os.path.join(out_dir, "placeholders.json"), json.dumps(placeholders, indent=1).encode("utf-8"))
    summary = {
        "pictures": len(hashes),
        "placeholder_hashes": len(placeholders),
        "placeholder_products": sum(p["count"] for p in placeholders.values()),
        "not_card_shaped": len(odd),
        "not_card_shaped_examples": odd[:SHAPE_EXAMPLES],
        "unreadable": unreadable,
    }
    print_summary(summary)
    return summary


# ---- embed stage ----

def load_embeddings(out_dir, tag):
    """(ids list, vectors list) from the partial file, else from the final file, else empty."""
    for name in (f"emb_{tag}.partial.npz", f"emb_{tag}.npz"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            with np.load(path) as data:
                return list(data["ids"].tolist()), list(data["vectors"])
    return [], []


def save_embeddings(path, ids, vectors):
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, ids=np.asarray(ids, dtype=np.int64)[order],
             vectors=np.asarray(vectors, dtype=np.float32).reshape(-1, 512)[order])
    os.replace(tmp, path)


def embeddable(data, placeholders):
    """'ok', 'placeholder', 'shape' or 'decode' for one picture's bytes."""
    if hashlib.sha256(data).hexdigest() in placeholders:
        return "placeholder"
    size = image_size(data)
    if size is None:
        return "decode"
    return "ok" if is_card_shaped(*size) else "shape"


def embed_stage(out_dir, tag, embedder, save_every=SAVE_EVERY):
    placeholders = {}
    ph_path = os.path.join(out_dir, "placeholders.json")
    if os.path.exists(ph_path):
        with open(ph_path, encoding="utf-8") as fh:
            placeholders = json.load(fh)
    ids, vectors = load_embeddings(out_dir, tag)
    have = set(ids)
    counts = Counter()
    partial = os.path.join(out_dir, f"emb_{tag}.partial.npz")
    for pid, path in image_files(out_dir):
        if pid in have:
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        verdict = embeddable(data, placeholders)
        if verdict == "ok":
            try:
                vectors.append(embedder(data, trusted=True))
                ids.append(pid)
                counts["embedded"] += 1
            except ValueError as err:
                print(f"  {pid}: cannot decode, skipped: {err}")
                verdict = "decode"
        if verdict != "ok":
            counts[verdict if verdict != "decode" else "decode_failed"] += 1
        if counts["embedded"] and counts["embedded"] % save_every == 0 and verdict == "ok":
            save_embeddings(partial, ids, vectors)
            print(f"  {len(ids)} embedded (partial saved)")
    save_embeddings(os.path.join(out_dir, f"emb_{tag}.npz"), ids, vectors)
    if os.path.exists(partial):
        os.remove(partial)
    summary = {"total": len(ids), "new": counts["embedded"], "placeholder": counts["placeholder"],
               "not_card_shaped": counts["shape"], "decode_failed": counts["decode_failed"]}
    print_summary(summary)
    return summary


def build_embedder(model_path, threads):
    """ONNX CLIP session limited to `threads` threads, reusing pipeline.embed.preprocess (pipeline/embed.py untouched)."""
    sys.path.insert(0, str(ROOT))
    import onnxruntime as ort
    from pipeline.embed import preprocess

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(int(threads), MAX_THREADS)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def embed(jpeg_bytes, trusted=False):
        v = session.run(None, {input_name: preprocess(jpeg_bytes, trusted)})[0].reshape(-1).astype(np.float32)
        return v / np.linalg.norm(v)

    return embed


# ---- pack stage ----

PACK_VERSION = 1
PACK_DIM = 512
PACK_MAX_CARDS = 100_000
PACK_MAX_TEXT = 200
PACK_HASH_CHARS = 16
EXIT_PACK_FAILED = 1


class PackError(Exception):
    """The pack cannot be written; the message says why."""


def file_sha256_prefix(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:PACK_HASH_CHARS]


def bad_text(row):
    for key in ("name", "group", "number"):
        value = row.get(key)
        if not isinstance(value, str) or not value or len(value) > PACK_MAX_TEXT:
            return True
    return False


def is_placeholder_picture(out_dir, pid, placeholders):
    """True if the product's downloaded picture hashes to a placeholder; no picture on disk counts as not one."""
    path = os.path.join(out_dir, "images", f"{pid}.jpg")
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest() in placeholders


def select_pack_cards(out_dir, tag, placeholders):
    """(cards sorted by groupId, number, productId, their float rows, left-out counts)."""
    emb_path = os.path.join(out_dir, f"emb_{tag}.npz")
    if not os.path.exists(emb_path):
        raise PackError(f"{emb_path} is missing; run embed first")
    with np.load(emb_path) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32).reshape(-1, PACK_DIM)
        row_of = {int(pid): i for i, pid in enumerate(data["ids"].tolist())}
    left_out = {"placeholder": 0, "noRow": 0, "badText": 0}
    kept = []
    for product in read_jsonl(os.path.join(out_dir, "products.jsonl")):
        pid = int(product["productId"])
        if pid not in row_of:
            left_out["noRow"] += 1
        elif is_placeholder_picture(out_dir, pid, placeholders):
            left_out["placeholder"] += 1
        elif bad_text(product):
            left_out["badText"] += 1
        else:
            kept.append(product)
    kept.sort(key=lambda p: (int(p["groupId"]), p["number"], int(p["productId"])))
    if len(kept) > PACK_MAX_CARDS:
        raise PackError(f"{len(kept)} cards is over the limit of {PACK_MAX_CARDS}")
    return kept, vectors[[row_of[int(p["productId"])] for p in kept]], left_out


def pack_rows(vectors):
    """(int8 rows, float32 scales) with q * scale a unit vector; quantised exactly as japanese_study does."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from japanese_study import quantize_rows

    q, _ = quantize_rows(vectors)
    norms = np.linalg.norm(q.astype(np.float32), axis=1)
    if (norms == 0).any():
        raise PackError("a row quantised to all zero")
    return q, (1.0 / norms).astype(np.float32)


def pack_bytes(header, scales, q):
    head = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw = len(head).to_bytes(4, "little") + head + scales.astype("<f4").tobytes() + q.astype(np.int8).tobytes()
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue(), len(raw)


def pack_stage(out_dir, tag, rows_model, query_model, crop, built=None, dest=None):
    ph_path = os.path.join(out_dir, "placeholders.json")
    if not os.path.exists(ph_path):
        raise PackError(f"{ph_path} is missing; run dedupe first")
    with open(ph_path, encoding="utf-8") as fh:
        placeholders = json.load(fh)
    cards, vectors, left_out = select_pack_cards(out_dir, tag, placeholders)
    q, scales = pack_rows(vectors)
    rows_hash, query_hash = file_sha256_prefix(rows_model), file_sha256_prefix(query_model)
    built = built or datetime.date.today().isoformat()
    header = {
        "v": PACK_VERSION, "dim": PACK_DIM, "entries": len(cards), "rows": rows_hash, "query": query_hash,
        "crop": int(crop), "built": built,
        "cards": [{"pid": int(c["productId"]), "gid": int(c["groupId"]), "name": c["name"], "set": c["group"],
                   "number": c["number"]} for c in cards],
    }
    blob, raw_bytes = pack_bytes(header, scales, q)
    dest = dest or os.path.join(out_dir, "catalog-ja.bin.gz")
    summary = {
        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob), "rawBytes": raw_bytes,
        "entries": len(cards), "rows": rows_hash, "query": query_hash, "crop": int(crop), "built": built,
        "leftOut": left_out,
    }
    write_atomic(dest, blob)
    write_atomic(f"{dest}.json", json.dumps(summary, indent=1).encode("utf-8"))
    print_summary(summary)
    return summary


def read_pack(path):
    """(header dict, float32 scales (n,), int8 rows (n, 512)) from a pack file; raises PackError if malformed."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise PackError("pack is too short")
    head_len = int.from_bytes(raw[:4], "little")
    header = json.loads(raw[4:4 + head_len].decode("utf-8"))
    n, dim = header["entries"], header["dim"]
    body = raw[4 + head_len:]
    if len(body) != n * 4 + n * dim:
        raise PackError(f"pack body is {len(body)} bytes, expected {n * 4 + n * dim}")
    scales = np.frombuffer(body[:n * 4], dtype="<f4").astype(np.float32)
    rows = np.frombuffer(body[n * 4:], dtype=np.int8).reshape(n, dim)
    return header, scales, rows


def lower_priority():
    if sys.platform != "win32":
        return
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except ImportError:
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


# ---- command line ----

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default data/ja-catalog)")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("list")
    images = sub.add_parser("images")
    images.add_argument("--limit", type=int)
    images.add_argument("--workers", type=int, default=MAX_WORKERS)
    images.add_argument("--rate", type=float, default=5)
    sub.add_parser("dedupe")
    embed = sub.add_parser("embed")
    embed.add_argument("--model", required=True)
    embed.add_argument("--tag", required=True)
    embed.add_argument("--threads", type=int, default=MAX_THREADS)
    pack = sub.add_parser("pack")
    pack.add_argument("--tag", required=True)
    pack.add_argument("--rows-model", required=True)
    pack.add_argument("--query-model", required=True)
    pack.add_argument("--crop", type=int, required=True)
    pack.add_argument("--built")
    pack.add_argument("--dest")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "list":
        list_stage(args.out)
    elif args.stage == "images":
        if images_stage(args.out, args.limit, args.workers, args.rate)["stopped"]:
            return EXIT_REFUSED
    elif args.stage == "dedupe":
        dedupe_stage(args.out)
    elif args.stage == "pack":
        try:
            pack_stage(args.out, args.tag, args.rows_model, args.query_model, args.crop, args.built, args.dest)
        except PackError as err:
            print(f"pack failed: {err}", file=sys.stderr)
            return EXIT_PACK_FAILED
    else:
        lower_priority()
        embed_stage(args.out, args.tag, build_embedder(args.model, args.threads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
