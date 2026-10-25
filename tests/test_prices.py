"""The nightly price file: prices keyed by TCGPlayer product id, plus stamped rows and EX prices keyed by catalog id."""
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

import _paths

from pipeline import prices, stamps

UPDATED = "2026-10-18T20:05:40+0000"
GROUPS = {"results": [{"groupId": 604, "name": "Base Set"}, {"groupId": 999, "name": "Missing"}]}
BASE = {
    "results": [
        {"productId": 42382, "subTypeName": "Holofoil", "marketPrice": 954.65, "midPrice": 500},
        {"productId": 89401, "subTypeName": "Normal", "marketPrice": None, "midPrice": 0.46},
        {"productId": 89401, "subTypeName": "Reverse Holofoil", "marketPrice": 0, "midPrice": 9.99},
        {"productId": True, "subTypeName": "Holofoil", "marketPrice": 5},
        {"productId": 1, "subTypeName": "Holofoil", "marketPrice": 0, "midPrice": 0},
    ]
}


def stub_text(url):
    assert url == prices.UPDATED_URL
    return UPDATED


def stub_json(url):
    if url == prices.GROUPS_URL:
        return GROUPS
    if url == prices.prices_url(604):
        return BASE
    if url == prices.prices_url(999):
        raise prices.Missing(url)
    raise AssertionError(url)


class Prices(unittest.TestCase):
    def test_the_snapshot_uses_the_source_time_and_market_then_mid(self):
        snapshot = prices.collect_snapshot(stub_text, stub_json, min_products=1)
        self.assertEqual(snapshot["updatedAt"], "2026-10-18T20:05:40Z")
        self.assertEqual(snapshot["v"], 1)
        self.assertEqual(snapshot["prices"]["42382"], {"Holofoil": 954.65})
        self.assertEqual(snapshot["prices"]["89401"], {"Normal": 0.46, "Reverse Holofoil": 9.99})
        self.assertNotIn("1", snapshot["prices"])
        self.assertNotIn("true", snapshot["prices"])

    def test_a_missing_group_is_skipped_and_too_few_products_is_refused(self):
        snapshot = prices.collect_snapshot(stub_text, stub_json, min_products=1)
        self.assertEqual(set(snapshot["prices"]), {"42382", "89401"})
        with self.assertRaisesRegex(ValueError, "need 1000"):
            prices.collect_snapshot(stub_text, stub_json)

    def test_a_bad_timestamp_and_an_empty_group_list_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not UTC"):
            prices.normalize_updated("yesterday")
        with self.assertRaisesRegex(ValueError, "no groups"):
            prices.collect_snapshot(stub_text, lambda url: {} if url == prices.GROUPS_URL else BASE, min_products=1)

    def test_the_file_round_trips(self):
        snapshot = prices.build_snapshot(UPDATED, [BASE])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "prices.json")
            prices.write_snapshot(path, snapshot)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), snapshot)


FIXTURES = os.path.join(_paths.ROOT, "tests", "fixtures", "prices")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


TCGCSV = fixture("tcgcsv.json")


def tcgcsv_json(missing_products=()):
    def get(url):
        if url == prices.GROUPS_URL:
            return {"results": [{"groupId": int(g)} for g in TCGCSV]}
        for group_id, files in TCGCSV.items():
            if url == prices.prices_url(int(group_id)):
                return files["prices"]
            if url == prices.products_url(int(group_id)):
                if int(group_id) in missing_products:
                    raise prices.Missing(url)
                return files["products"]
        raise AssertionError(url)

    return get


def catalog():
    return stamps.Catalog(fixture("catalog-meta.json"), fixture("tcgdex-sets.json"))


def collect(**kwargs):
    args = {"min_products": 1, "min_stamped": 1, "min_by_catalog": 1, "catalog": catalog()}
    args.update(kwargs)
    get_json = args.pop("get_json", tcgcsv_json())
    return prices.collect_snapshot(stub_text, get_json, **args)


class StampedSnapshot(unittest.TestCase):
    def test_without_a_catalog_no_product_file_is_read_and_the_shape_is_unchanged(self):
        snapshot = prices.collect_snapshot(stub_text, stub_json, min_products=1)
        self.assertEqual(set(snapshot), {"v", "updatedAt", "prices"})

    def test_new_keys_sit_beside_an_unchanged_prices_map(self):
        snapshot = collect()
        plain = prices.collect_snapshot(stub_text, tcgcsv_json(), min_products=1)
        self.assertEqual(set(snapshot), {"v", "updatedAt", "prices", "stamped", "byCatalog"})
        # lib/price-snapshot.js keeps only v, updatedAt and prices, so old builds read exactly what they read today.
        self.assertEqual({k: snapshot[k] for k in ("v", "updatedAt", "prices")}, plain)
        self.assertEqual(snapshot["v"], 1)

    def test_stamped_rows_are_sorted_and_byCatalog_holds_ex_prices(self):
        snapshot = collect()
        rows = snapshot["stamped"]
        keys = [(r["catalogId"], r["stamp"], r["productId"], r["subtype"]) for r in rows]
        self.assertEqual(keys, sorted(keys))
        self.assertIn({"catalogId": "svp-027", "stamp": "pokemoncenter", "tag": "Pokemon Center Exclusive",
                       "productId": 500264, "subtype": "Holofoil",
                       "market": 161.94, "mid": 166.1, "low": 99.99}, rows)
        self.assertEqual(snapshot["byCatalog"]["ex13-3"], {"Holofoil": 82.37, "Reverse Holofoil": 1498.99})

    def test_a_missing_product_file_drops_only_that_groups_stamps(self):
        snapshot = collect(get_json=tcgcsv_json(missing_products={22872}))
        self.assertFalse([r for r in snapshot["stamped"] if r["catalogId"].startswith("svp-")])
        self.assertTrue([r for r in snapshot["stamped"] if r["catalogId"] == "bwp-BW48"])
        self.assertIn("500264", snapshot["prices"])

    def test_too_few_stamped_rows_or_ex_cards_is_refused(self):
        with self.assertRaisesRegex(ValueError, "stamped rows"):
            collect(min_stamped=10_000)
        with self.assertRaisesRegex(ValueError, "byCatalog"):
            collect(min_by_catalog=10_000)

    def test_the_report_is_filled_for_the_job_log(self):
        report = {}
        collect(report=report)
        self.assertEqual(report["stamped"]["staff"]["joined"], 5)
        self.assertEqual(report["byCatalog"]["ex15"]["joined"], 5)
        self.assertEqual(report["missingProducts"], 0)
        self.assertGreaterEqual(report["collisions"], 1)
        self.assertEqual(report["collisionsWithTag"], 0)

    def test_the_size_cap_covers_the_new_keys(self):
        size = len(json.dumps(collect(), separators=(",", ":"), sort_keys=True).encode("utf-8"))
        with mock.patch.object(prices, "MAX_SNAPSHOT_BYTES", size - 1):
            with self.assertRaisesRegex(ValueError, "bytes"):
                collect()
        # lib/price-snapshot.js MAX_SNAPSHOT_CHARS: a bigger file is rejected whole by every install.
        self.assertLessEqual(prices.MAX_SNAPSHOT_BYTES, 8_000_000)

class PinnedCatalog(unittest.TestCase):
    META = json.dumps(fixture("catalog-meta.json")).encode("utf-8")

    def pin(self, data=META):
        return {"url": "https://example.test/meta.json", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def test_the_price_step_has_its_own_pin_and_the_pack_assets_do_not_list_it(self):
        pin = prices.CATALOG_PIN
        self.assertEqual(pin["url"], "https://github.com/fabcrowd/pokemon-auto-lister/releases/download/v2.0/meta.json")
        self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(pin["bytes"], 0)
        with open(os.path.join(_paths.ROOT, "pins.json"), encoding="utf-8") as f:
            pins = json.load(f)
        # A failed price download must never stop the pack: fetch_assets and pipeline.run check every pins.json asset.
        self.assertNotIn("meta.json", [a["name"] for a in pins["assets"]])

    def test_a_verified_download_gives_the_card_rows(self):
        seen = []

        def get_bytes(url, limit):
            seen.append((url, limit))
            return self.META

        cards = prices.fetch_catalog(get_bytes, self.pin())
        self.assertEqual(cards, fixture("catalog-meta.json"))
        self.assertEqual(seen, [("https://example.test/meta.json", len(self.META))])

    def test_a_hash_or_size_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            prices.fetch_catalog(lambda url, limit: self.META, {**self.pin(), "sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "bytes"):
            prices.fetch_catalog(lambda url, limit: self.META, {**self.pin(), "bytes": len(self.META) + 1})

    def test_a_failed_download_is_refused(self):
        def down(url, limit):
            raise prices.Missing(url)

        with self.assertRaises(prices.Missing):
            prices.fetch_catalog(down, self.pin())

    def test_a_catalog_that_is_not_a_list_is_refused(self):
        data = b'{"cards": []}'
        with self.assertRaisesRegex(ValueError, "list"):
            prices.fetch_catalog(lambda url, limit: data, self.pin(data))


if __name__ == "__main__":
    unittest.main()
