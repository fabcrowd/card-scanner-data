#!/usr/bin/env python3
"""Downloads the pinned model and catalog from the code repo's GitHub releases and checks their sha256.

    python scripts/fetch_assets.py --pins pins.json --out assets
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request


def fetch(url, dest, sha256):
    h = hashlib.sha256()
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as res, open(tmp, "wb") as f:
        for block in iter(lambda: res.read(1 << 20), b""):
            h.update(block)
            f.write(block)
    if h.hexdigest() != sha256:
        os.remove(tmp)
        sys.exit(f"{url}: sha256 {h.hexdigest()} does not match the pin {sha256}")
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pins", default="pins.json")
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    with open(args.pins, encoding="utf-8") as f:
        pins = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    for a in pins["assets"]:
        url = f"https://github.com/{pins['codeRepo']}/releases/download/{a['release']}/{a['name']}"
        dest = os.path.join(args.out, a["name"])
        if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == a["sha256"]:
            continue
        print("fetching", url, file=sys.stderr)
        fetch(url, dest, a["sha256"])


if __name__ == "__main__":
    main()
