"""The nightly price file: one snapshot, joined only by TCGPlayer product id."""
import json
import os
import tempfile
import unittest

import _paths  # noqa: F401

from pipeline import prices

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


if __name__ == "__main__":
    unittest.main()
