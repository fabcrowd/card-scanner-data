import json
import unittest

import _paths  # noqa: F401
import numpy as np

from pipeline import new_sets

KNOWN = {"base1-1", "base1-2", "sv9-1"}
SETS = [
    {"id": "base1", "name": "Base Set", "cardCount": {"total": 2}},
    {"id": "sv9", "name": "Journey Together", "cardCount": {"total": 3}},
    {"id": "sv10", "name": "Destined Rivals", "cardCount": {"total": 2}},
]
DETAIL = {
    "sv9": {"name": "Journey Together", "cards": [
        {"id": "sv9-1", "localId": "1", "name": "Pikachu", "image": "https://assets.tcgdex.net/en/sv/sv9/1"},
        {"id": "sv9-2", "localId": "2", "name": "Raichu", "image": "https://assets.tcgdex.net/en/sv/sv9/2"},
        {"id": "sv9-3", "localId": "3", "name": "No Picture"},
    ]},
    "sv10": {"name": "Destined Rivals", "cards": [
        {"id": "sv10-1", "localId": "1", "name": "Mew", "image": "https://assets.tcgdex.net/en/sv/sv10/1"},
        {"id": "sv10-2", "localId": 2, "name": "Eevee", "image": "https://assets.tcgdex.net/en/sv/sv10/2"},
    ]},
}


class Net:
    def __init__(self, details=DETAIL, sets=SETS, bad_images=()):
        self.requests, self.details, self.sets, self.bad_images = [], details, sets, set(bad_images)

    def json(self, url):
        self.requests.append(url)
        if url.endswith("/sets"):
            return self.sets
        code = url.rsplit("/", 1)[1]
        if code not in self.details:
            raise OSError("down")
        return self.details[code]

    def image(self, url):
        self.requests.append(url)
        if url in self.bad_images:
            raise OSError("404")
        return url.encode()


def embedder(data, trusted=False):
    assert trusted
    v = np.zeros(512, dtype=np.float32)
    v[sum(data) % 512] = 1.0
    return v


def run(net=None, previous=None, known=KNOWN, cap=500):
    net = net or Net()
    out = new_sets.build(known, previous, embedder, fetch=net.json, fetch_image=net.image, sleep=lambda s: None, cap=cap)
    return net, out


class FindNew(unittest.TestCase):
    def test_only_cards_missing_from_the_catalog_are_found_and_complete_sets_are_not_opened(self):
        net = Net()
        cards, sets = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual([c["id"] for c in cards], ["sv9-2", "sv10-1", "sv10-2"])  # sv9-3 has no picture
        self.assertEqual(sets, ["sv9", "sv10"])
        self.assertNotIn(f"{new_sets.TCGDEX}/sets/base1", net.requests)  # base1: 2 known of 2
        self.assertEqual(cards[0], {"id": "sv9-2", "name": "Raichu", "set": "sv9", "setName": "Journey Together", "number": "2", "image": "https://assets.tcgdex.net/en/sv/sv9/2/high.webp"})
        self.assertEqual(cards[2]["number"], "2")

    def test_cap_and_a_failing_set(self):
        cards, _ = new_sets.find_new_cards(KNOWN, Net().json, lambda s: None, cap=2)
        self.assertEqual(len(cards), 2)
        net = Net(details={"sv10": DETAIL["sv10"]})  # sv9 detail fails
        cards, _ = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual([c["id"] for c in cards], ["sv10-1", "sv10-2"])

    def test_a_quiet_night_is_one_request(self):
        net = Net(sets=[SETS[0]])
        cards, _ = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual((cards, net.requests), ([], [f"{new_sets.TCGDEX}/sets"]))


class Build(unittest.TestCase):
    def test_writes_rows_in_the_extension_layout(self):
        _, out = run()
        self.assertEqual(out["added"], ["sv9-2", "sv10-1", "sv10-2"])
        self.assertEqual((out["sets"], out["total"]), (["sv10", "sv9"], 3))
        cards, rows = new_sets.read_extra(out["bytes"])
        self.assertEqual([c["id"] for c in cards], out["added"])
        self.assertEqual(rows.shape, (3, 512))
        self.assertEqual(int(rows[0].max()), 127)
        head = json.loads(out["bytes"][4 : 4 + int.from_bytes(out["bytes"][:4], "little")])
        self.assertEqual(head["cardIds"], out["added"])
        self.assertEqual(head["entries"], 3)

    def test_is_cumulative_and_does_not_repeat_earlier_cards(self):
        _, first = run(cap=1)
        self.assertEqual(first["added"], ["sv9-2"])
        _, second = run(previous=first["bytes"], cap=1)
        self.assertEqual(second["added"], ["sv10-1"])
        cards, rows = new_sets.read_extra(second["bytes"])
        self.assertEqual([c["id"] for c in cards], ["sv9-2", "sv10-1"])
        self.assertTrue(np.array_equal(rows[0], new_sets.read_extra(first["bytes"])[1][0]))  # old rows are copied, not re-embedded

    def test_a_picture_that_fails_is_skipped_and_left_for_tomorrow(self):
        _, out = run(Net(bad_images={"https://assets.tcgdex.net/en/sv/sv10/1/high.webp"}))
        self.assertEqual(out["added"], ["sv9-2", "sv10-2"])
        _, again = run(previous=out["bytes"])
        self.assertEqual(again["added"], ["sv10-1"])

    def test_nothing_new_keeps_the_previous_file_byte_for_byte(self):
        _, first = run()
        _, again = run(previous=first["bytes"])
        self.assertEqual((again["added"], again["bytes"]), ([], first["bytes"]))
        _, none = run(Net(sets=[SETS[0]]))
        self.assertIsNone(none["bytes"])

    def test_unusable_embeddings_are_dropped(self):
        def bad(data, trusted=False):
            return np.full(512, np.nan, dtype=np.float32)

        net = Net()
        out = new_sets.build(KNOWN, None, bad, fetch=net.json, fetch_image=net.image, sleep=lambda s: None)
        self.assertEqual(out["added"], [])

    def test_read_extra_rejects_damage(self):
        _, out = run()
        for broken in (out["bytes"][:-1], b"\x01", out["bytes"] + b"x"):
            with self.assertRaises(ValueError):
                new_sets.read_extra(broken)

    def test_requests_are_paced(self):
        pauses = []
        net = Net()
        new_sets.build(KNOWN, None, embedder, fetch=net.json, fetch_image=net.image, sleep=pauses.append)
        self.assertTrue(pauses and all(p == new_sets.REQUEST_DELAY for p in pauses))
        self.assertGreaterEqual(len(pauses), 2 + 3)  # two sets opened, three pictures


if __name__ == "__main__":
    unittest.main()
