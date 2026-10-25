"""Daily TCGPlayer price file, built once a night and downloaded by every install.

The extension does not call the price source. This writes prices.json; the sign job
publishes it on the data repo's ``prices`` pre-release. A product id from TCGdex is
the only join, so a row can only price that product.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable

UPDATED_URL = "https://tcgcsv.com/last-updated.txt"
GROUPS_URL = "https://tcgcsv.com/tcgplayer/3/groups"
USER_AGENT = "card-scanner-nightly/1"
UPDATED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
MIN_PRODUCTS = 1000
MAX_SNAPSHOT_BYTES = 8_000_000
MAX_BODY = 4 * 1024 * 1024


class Missing(Exception):
    """The price source has no file at this address."""


def prices_url(group_id: int) -> str:
    """Address of one set's price list."""
    return f"https://tcgcsv.com/tcgplayer/3/{group_id}/prices"


def normalize_updated(text: str) -> str:
    """UTC timestamp from the price source's last-updated file.

    Raises:
        ValueError: the text is not a UTC timestamp.
    """
    raw = text.strip().splitlines()[0].strip() if text else ""
    if raw.endswith("+0000"):
        raw = raw[:-5] + "Z"
    if not UPDATED_AT.fullmatch(raw):
        raise ValueError(f"price snapshot time is not UTC: {text!r}")
    return raw


def _price(row: dict) -> float | None:
    """Market price, or mid when the market is empty. Non-positive numbers are dropped."""
    for key in ("marketPrice", "midPrice"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value > 0:
            return round(float(value), 2)
    return None


def prices_from_group(payload: dict) -> dict[str, dict[str, float]]:
    """Map product id to subtype prices from one group's price file."""
    out: dict[str, dict[str, float]] = {}
    rows = payload.get("results") if isinstance(payload, dict) else None
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        product_id = row.get("productId")
        subtype = row.get("subTypeName")
        price = _price(row)
        if isinstance(product_id, bool) or not isinstance(product_id, int) or not isinstance(subtype, str) or price is None:
            continue
        out.setdefault(str(product_id), {})[subtype] = price
    return out


def group_ids(payload: dict) -> list[int]:
    """TCGPlayer group ids from the groups file, in the order given."""
    ids: list[int] = []
    rows = payload.get("results") if isinstance(payload, dict) else None
    for row in rows or []:
        group_id = row.get("groupId") if isinstance(row, dict) else None
        if isinstance(group_id, int) and not isinstance(group_id, bool):
            ids.append(group_id)
    return ids


def build_snapshot(updated_at: str, group_files: list[dict]) -> dict:
    """Merge group price files into the snapshot the extension stores."""
    prices: dict[str, dict[str, float]] = {}
    for payload in group_files:
        for product_id, variants in prices_from_group(payload).items():
            prices.setdefault(product_id, {}).update(variants)
    return {"v": 1, "updatedAt": normalize_updated(updated_at), "prices": prices}


def collect_snapshot(
    get_text: Callable[[str], str],
    get_json: Callable[[str], dict],
    *,
    min_products: int = MIN_PRODUCTS,
) -> dict:
    """Fetch the snapshot.

    A missing group is skipped. Fewer than ``min_products`` is refused, so a bad
    night does not replace the file installs already have.

    Raises:
        ValueError: the source listed no groups, or the result is too small or too big.
        Missing: raised by ``get_json`` for a group that has no price file.
    """
    updated = normalize_updated(get_text(UPDATED_URL))
    ids = group_ids(get_json(GROUPS_URL))
    if not ids:
        raise ValueError("price source returned no groups")
    files: list[dict] = []
    missed = 0
    for group_id in ids:
        try:
            files.append(get_json(prices_url(group_id)))
        except Missing:
            missed += 1
    snapshot = build_snapshot(updated, files)
    if len(snapshot["prices"]) < min_products:
        raise ValueError(
            f"price snapshot has {len(snapshot['prices'])} products, need {min_products} (missed {missed} groups)"
        )
    encoded = json.dumps(snapshot, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"price snapshot is {len(encoded)} bytes")
    return snapshot


def _read(url: str, limit: int) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"not https: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise Missing(url) from err
        raise
    if len(data) > limit:
        raise ValueError(f"{url} is larger than {limit} bytes")
    return data


def fetch_text(url: str) -> str:
    """GET a small text file."""
    return _read(url, 4096).decode("utf-8")


def fetch_json(url: str) -> dict:
    """GET a JSON file."""
    payload = json.loads(_read(url, MAX_BODY).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{url} is not an object")
    return payload


def write_snapshot(path: str, snapshot: dict) -> None:
    """Write prices.json. The parent directory must already exist."""
    text = json.dumps(snapshot, separators=(",", ":"), sort_keys=True) + "\n"
    if len(text.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        raise ValueError("price snapshot is too large")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    """Write the daily snapshot to ``--out``."""
    parser = argparse.ArgumentParser(description="Write the daily TCGPlayer price snapshot.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    snapshot = collect_snapshot(fetch_text, fetch_json)
    directory = os.path.dirname(args.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_snapshot(args.out, snapshot)
    print(f"prices {snapshot['updatedAt']} products {len(snapshot['prices'])} -> {args.out}")


if __name__ == "__main__":
    main()
