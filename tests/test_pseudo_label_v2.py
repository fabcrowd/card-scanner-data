import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pseudo_label_v2 as V2  # noqa: E402


def cand(cid, name, rank, inl, art=60, ok=True, cos=0.8):
    return {"id": cid, "name": name, "clip_rank": rank, "cos": cos, "inliers": inl, "art_inliers": art, "geom_ok": ok}


class ArtGroups(unittest.TestCase):
    def test_grouping_on_synthetic_case(self):
        e = np.eye(4, dtype=np.float32)
        ids = ["a1", "a2", "a3", "b1", "c1", "c2"]
        names = ["Bellsprout", "Erika's Bellsprout", "Bellsprout", "Dipplin", "Pikachu", "Pikachu"]
        hashes = [0, 0b1111, 0xFFFFFFFFFFFFFFFF, 0, 0, 0xFFFFFFFFFFFFFFFF]
        vecs = np.stack([e[0], e[1], e[2], e[0], e[3], e[3]])  # c1/c2 share a vector; a-ids otherwise orthogonal
        g = V2.build_groups(ids, names, hashes, vecs)
        self.assertEqual(g["a1"], g["a2"])  # hash dist 4, owner possessive stripped
        self.assertNotEqual(g["a1"], g["a3"])  # hash 64 apart, cosine 0
        self.assertNotEqual(g["a1"], g["b1"])  # same art, different name
        self.assertEqual(g["c1"], g["c2"])  # CLIP cosine 1.0 despite far hashes

    def test_group_name_strips_owner(self):
        self.assertEqual(V2.group_name("Erika's Bellsprout"), "bellsprout")
        self.assertEqual(V2.group_name("Bellsprout"), "bellsprout")


class Decide(unittest.TestCase):
    groups = {"g1": "g1", "g2": "g1"}

    def test_id_level_stores_group_members(self):
        c = [cand("g1", "Dipplin", 1, 150), cand("x", "Other", 2, 20)]
        d = V2.decide_v2(c, self.groups, set())
        self.assertEqual((d["level"], d["ids"]), ("id", ["g1", "g2"]))

    def test_gappy_name_capped_at_name_level(self):
        c = [cand("g1", "Dipplin", 1, 150), cand("x", "Other", 2, 20)]
        d = V2.decide_v2(c, self.groups, {"dipplin"})
        self.assertEqual((d["level"], d["reason"], d["ids"]), ("name", "gappy_name", []))

    def test_same_group_runner_up_does_not_block_id(self):
        c = [cand("g1", "Dipplin", 1, 150), cand("g2", "Dipplin", 2, 140)]
        self.assertEqual(V2.decide_v2(c, self.groups, set())["level"], "id")

    def test_outside_group_runner_up_blocks_id(self):
        c = [cand("g1", "Dipplin", 1, 150), cand("x", "Other", 2, 100)]
        self.assertNotEqual(V2.decide_v2(c, self.groups, set())["level"], "id")

    def test_clip_agreement_for_40_to_99(self):
        agree = [cand("a", "Koffing", 1, 60), cand("b", "Weezing", 2, 10)]
        self.assertEqual(V2.decide_v2(agree, {}, set())["level"], "name")
        disagree = [cand("a", "Koffing", 2, 60), cand("b", "Weezing", 1, 10)]
        d = V2.decide_v2(disagree, {}, set())
        self.assertEqual((d["level"], d["reason"]), ("reject", "clip_disagree"))

    def test_clip_agreement_not_needed_at_100_plus(self):
        c = [cand("a", "Koffing", 2, 120), cand("b", "Weezing", 1, 70)]  # lead < 2x -> name level, 120 >= 100
        self.assertEqual(V2.decide_v2(c, {}, set())["reason"], "ambiguous_name")

    def test_energy_rejected(self):
        d = V2.decide_v2([cand("e", "Basic Fire Energy", 1, 200)], {}, set())
        self.assertEqual(d["reason"], "energy")

    def test_geometry_failure_rejects(self):
        d = V2.decide_v2([cand("a", "Koffing", 1, 200, ok=False)], {}, set())
        self.assertEqual((d["level"], d["reason"]), ("reject", "geometry"))


class Geometry(unittest.TestCase):
    def test_identity_ok(self):
        ok, why, _ = V2.geometry_check(np.eye(3), 50)
        self.assertTrue(ok, why)

    def test_degenerate_homographies_rejected(self):
        self.assertEqual(V2.geometry_check(None, 50)[1], "no_homography")
        self.assertEqual(V2.geometry_check(np.array([[1, 2, 0], [2, 4, 0], [0, 0, 1.0]]), 50)[1], "bad_determinant")
        self.assertEqual(V2.geometry_check(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1.0]]), 50)[1], "bad_determinant")
        self.assertEqual(V2.geometry_check(np.diag([3.0, 3.0, 1.0]), 50)[1], "bad_scale")
        th = np.radians(40)
        rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        self.assertEqual(V2.geometry_check(rot, 50)[1], "bad_rotation")
        self.assertEqual(V2.geometry_check(np.eye(3), 24)[1], "few_art_inliers")


if __name__ == "__main__":
    unittest.main()
