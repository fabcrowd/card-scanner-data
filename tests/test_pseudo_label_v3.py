import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pseudo_label_v3 as V3  # noqa: E402

FULL = 0xFFFFFFFFFFFFFFFF


def row(level, reason=None, best="sv1-1"):
    return {"level": level, "reason": reason, "best_id": best}


class Cliques(unittest.TestCase):
    def test_no_chaining(self):
        # a~b and b~c but a!~c: chaining would give one group of 3; cliques must not.
        groups = V3.greedy_cliques(["a", "b", "c"], {frozenset("ab"), frozenset("bc")})
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])
        for g in groups:
            for x in g:
                for y in g:
                    self.assertTrue(x == y or {x, y} in ({"a", "b"}, {"b", "c"}))

    def test_full_clique_merges(self):
        groups = V3.greedy_cliques(["a", "b", "c"], {frozenset("ab"), frozenset("bc"), frozenset("ac")})
        self.assertEqual([sorted(g) for g in groups], [["a", "b", "c"]])

    def test_build_groups_chain_split_and_identical_merge(self):
        e = np.eye(3, dtype=np.float32)
        ids = ["p1", "p2", "p3", "q1", "q2"]
        names = ["Pikachu"] * 3 + ["Charizard ex"] * 2
        # p1~p2 and p2~p3 by hash and cosine, p1 vs p3 far on both -> no chain.
        hashes = [0, 0b1111111111, 0b1111111111111111111, 0, FULL]
        v = np.stack([e[0], e[0], e[0], e[1], e[1]])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        v[2] = np.array([0.0, 1.0, 0.0])  # p3 orthogonal to p1/p2 in CLIP: only reachable by hash
        g = V3.build_groups_v3(ids, names, hashes, v)
        self.assertEqual(g["p1"], g["p2"])
        self.assertNotEqual(g["p3"], g["p1"])  # cosine 0 and hash 19 bits from p1: no edge
        # q1/q2: far hashes, identical vectors, but SIFT on the catalog images says same print.
        g2 = V3.build_groups_v3(ids, names, hashes, v, sift_fn=lambda i, j: 150 if {i, j} == {3, 4} else 0)
        self.assertEqual(g2["q1"], g2["q2"])
        g3 = V3.build_groups_v3(ids, names, hashes, v, sift_fn=lambda i, j: 20)
        self.assertNotEqual(g3["q1"], g3["q2"])

    def test_different_names_never_group(self):
        v = np.ones((2, 2), dtype=np.float32) / np.sqrt(2)
        g = V3.build_groups_v3(["a", "b"], ["Mew", "Mewtwo"], [0, 0], v, sift_fn=lambda i, j: 999)
        self.assertNotEqual(g["a"], g["b"])


class SuspectJa(unittest.TestCase):
    def test_title(self):
        for t in ("Japanese Pokemon lot", "JP cards", "Pokemon from Japan", "lot jp holo"):
            self.assertTrue(V3.suspect_ja(t, [row("id")])[0], t)
        self.assertFalse(V3.suspect_ja("Pokemon jpeg scan lot", [row("id")])[0])

    def test_geometry_and_new_sets(self):
        rows = [row("reject", "geometry"), row("reject", "geometry"), row("id", best="me04-010"), row("name", best="me05-1")]
        self.assertEqual(V3.suspect_ja("Pokemon lot", rows), (True, "geometry_new_sets"))

    def test_needs_both_conditions(self):
        geom_old = [row("reject", "geometry"), row("reject", "geometry"), row("id", best="base1-4")]
        self.assertFalse(V3.suspect_ja("Pokemon lot", geom_old)[0])
        new_no_geom = [row("id", best="me04-010"), row("id", best="me05-1"), row("reject", "low_inliers")]
        self.assertFalse(V3.suspect_ja("Pokemon lot", new_no_geom)[0])
        self.assertFalse(V3.suspect_ja("Pokemon lot", [row("reject", "geometry")])[0])  # nothing accepted

    def test_exclusion_rewrites_accepted_rows_only(self):
        r = {**row("id"), "ids": ["sv1-1", "sv1-2"]}
        out = V3.finalize_row(r, "L1", True)
        self.assertEqual((out["level"], out["reason"], out["train_id"], out["no_negative"]), ("reject", "suspect_ja_listing", None, []))
        self.assertEqual(out["level_pre_exclusion"], "id")
        keep = V3.finalize_row(r, "L2", False)
        self.assertEqual((keep["level"], keep["train_id"], keep["no_negative"]), ("id", "sv1-1", ["sv1-2"]))


class Paths(unittest.TestCase):
    def test_eval_refused(self):
        for p in (r"C:\x\data\goodwill-pack2\eval", r"C:\x\data\hibid-pack\eval\1\photo-1.jpg", "C:/x/data/sealed-test/a.jpg"):
            with self.assertRaises(ValueError):
                V3.assert_allowed_path(p)
        V3.assert_allowed_path(r"C:\x\data\goodwill-pack2\train\1\photo-1.jpg")

    def test_listing_folders_refuses_eval_and_skips_recent(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                V3.listing_folders(os.path.join(d, "eval"), {})
            for n in ("1", "2", "3", "_tmp"):
                os.makedirs(os.path.join(d, n))
            old = 1000.0
            for n in ("1", "3"):
                os.utime(os.path.join(d, n), (old, old))
            ready, recent, nomf = V3.listing_folders(d, {"1": "a", "2": "b"}, now=old + 10_000)
            self.assertEqual((ready, nomf), (["1"], ["3"]))
            ready, recent, _ = V3.listing_folders(d, {"1": "a", "2": "b"}, now=os.path.getmtime(os.path.join(d, "2")) + 5)
            self.assertEqual(recent, ["2"])


if __name__ == "__main__":
    unittest.main()
