#!/usr/bin/env python3
"""Makes the pack signing key pair.

    python scripts/keygen.py --out C:\\secure\\pack-signing-key.pem

Writes the private key (PKCS8 PEM, ECDSA P-256) to --out and prints the public key (base64 SPKI) as the last
line. Paste the public key into PACK_PUBLIC_KEY in the extension's lib/shared-config.js and the PEM text into the
data repo's PACK_SIGNING_KEY secret. The private key is refused a path inside this repo and never overwritten.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.signing import generate_keypair  # noqa: E402


def inside(path, folder):
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(folder)]) == os.path.realpath(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where to write the private key PEM (outside any repo)")
    args = ap.parse_args()

    target = os.path.abspath(args.out)
    if inside(target, ROOT) or inside(target, os.getcwd()) and os.path.exists(os.path.join(os.getcwd(), ".git")):
        sys.exit("refusing to write the private key inside a repository; choose a path elsewhere")
    if os.path.exists(target):
        sys.exit(f"{target} already exists; refusing to overwrite a key")

    pem, spki = generate_keypair()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    print(f"private key written to {target}", file=sys.stderr)
    print("public key (PACK_PUBLIC_KEY):", file=sys.stderr)
    print(spki)


if __name__ == "__main__":
    main()
