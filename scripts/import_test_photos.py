"""Import a card-scanner test-photos export into a folder of JPEGs plus boxes.json and review sheets.

Usage: python scripts/import_test_photos.py <export.json> --out data/fb-test-photos [--force]
"""
import argparse
import base64
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

DATA_URL_PREFIX = "data:image/jpeg;base64,"
BUCKETS = (("<100", 0, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, math.inf))
SHEET_SIZE = 12
SHEET_COLS = 4
CELL = (480, 480)


def bucket_counts(heights):
    return {label: sum(1 for h in heights if lo <= h < hi) for label, lo, hi in BUCKETS}


def decode_photo(data_url):
    if not isinstance(data_url, str) or not data_url.startswith(DATA_URL_PREFIX):
        raise ValueError("photo is not a JPEG data URL")
    return base64.b64decode(data_url[len(DATA_URL_PREFIX):])


def build_sheet(items, photos_dir):
    rows = math.ceil(len(items) / SHEET_COLS)
    sheet = Image.new("RGB", (CELL[0] * SHEET_COLS, CELL[1] * rows), "white")
    for n, (photo_id, meta) in enumerate(items):
        img = Image.open(photos_dir / f"{photo_id}.jpg").convert("RGB")
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in meta["boxes"]:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, img.width // 400))
        img.thumbnail(CELL)
        sheet.paste(img, ((n % SHEET_COLS) * CELL[0], (n // SHEET_COLS) * CELL[1]))
    return sheet


def import_export(export_path, out_dir, force=False):
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    photos = export.get("photos")
    if not isinstance(photos, list):
        raise ValueError("export has no photos array")
    out = Path(out_dir)
    sheets_dir = out / "sheets"
    targets = [out / "boxes.json"] + [out / f"{p['id']}.jpg" for p in photos]
    clash = [t for t in targets if t.exists()]
    if clash and not force:
        raise FileExistsError(f"{len(clash)} output files exist (first: {clash[0]}); use --force to overwrite")
    if sheets_dir.exists() and any(sheets_dir.glob("sheet-*.jpg")) and not force:
        raise FileExistsError(f"{sheets_dir} has sheets; use --force to overwrite")
    sheets_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    for p in photos:
        jpeg = decode_photo(p["photo"])
        (out / f"{p['id']}.jpg").write_bytes(jpeg)
        width, height = Image.open(io.BytesIO(jpeg)).size
        boxes = [list(b) for b in p["boxes"]]
        index[p["id"]] = {
            "pageUrl": p.get("pageUrl"),
            "host": p.get("host"),
            "ts": p.get("ts"),
            "width": width,
            "height": height,
            "boxes": boxes,
            "box_heights": [b[3] - b[1] for b in boxes],
        }
    (out / "boxes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    items = list(index.items())
    for n in range(0, len(items), SHEET_SIZE):
        build_sheet(items[n:n + SHEET_SIZE], out).save(sheets_dir / f"sheet-{n // SHEET_SIZE + 1}.jpg", quality=85)
    return index


def summarize(index):
    heights = [h for meta in index.values() for h in meta["box_heights"]]
    return {
        "photos": len(index),
        "boxes": len(heights),
        "height_buckets": bucket_counts(heights),
        "hosts": dict(Counter(meta["host"] or "(none)" for meta in index.values())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        index = import_export(args.export, args.out, args.force)
    except (FileExistsError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    s = summarize(index)
    print(f"photos: {s['photos']}\nboxes: {s['boxes']}\nbox height (px, stored photo):")
    for label, n in s["height_buckets"].items():
        print(f"  {label}: {n}")
    print("per host:")
    for host, n in sorted(s["hosts"].items(), key=lambda kv: -kv[1]):
        print(f"  {host}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
