"""Detection only (no embedding) over every image under a folder tree; prints box-height buckets."""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cards import DEFAULT_MODEL, detect_cards  # noqa: E402

BUCKETS = [("<100", 0, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, 10**9)]
EXT = (".jpg", ".jpeg", ".png", ".webp")


def main(root, model=DEFAULT_MODEL):
    counts = {b[0]: 0 for b in BUCKETS}
    photos = zero = boxes = skipped = 0
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.lower().endswith(EXT):
                continue
            try:
                with Image.open(os.path.join(dp, f)) as im:
                    dets = detect_cards(im.convert("RGB"), model)
            except Exception:
                skipped += 1
                continue
            photos += 1
            zero += not dets
            for d in dets:
                h = d["box"][3] - d["box"][1]
                boxes += 1
                for label, lo, hi in BUCKETS:
                    if lo <= h < hi:
                        counts[label] += 1
    print({"photos": photos, "unreadable": skipped, "zero_box_photos": zero, "boxes": boxes, "buckets": counts})


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
