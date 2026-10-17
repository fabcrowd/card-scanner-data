import unittest

import _paths  # noqa: F401
import numpy as np

import helpers as h
from pipeline import gate
from pipeline.catalog import Catalog
from pipeline.pack import write_pack
from pipeline.train_adapter import pack_adapter

IDS = ["c-0", "c-1", "c-2", "c-3"]


def catalog():
    return Catalog(np.stack([h.unit(k) for k in range(4)]), IDS)


def blend(a, b, wa):
    v = wa * h.unit(a) + (1 - wa) * h.unit(b)
    return (v / np.linalg.norm(v)).astype(np.float32)


def boost_c0():
    """Rank-1 adapter that doubles the c-0 component: a 0.4/0.6 blend of c-0/c-1 flips from c-1 to c-0."""
    u = np.zeros((512, 1), dtype=np.float32)
    v = np.zeros((512, 1), dtype=np.float32)
    u[0, 0] = v[0, 0] = 1.0
    return pack_adapter(u, v)


def sets(n, a, b, wa, truth):
    return np.stack([blend(a, b, wa)] * n), [truth] * n


def near_duplicates():
    """c-1 is a close look-alike of c-0 (cosine 0.8), so a query between them scores high against both."""
    rows = [h.unit(0), (0.8 * h.unit(0) + 0.6 * h.unit(1)).astype(np.float32), h.unit(2), h.unit(3)]
    return Catalog(np.stack(rows), IDS)


def between_01():
    """Raw cosine 0.944 with c-0 and 0.953 with c-1: the catalog answers c-1 (high enough to accept), boost_c0 answers c-0."""
    v = 0.944 * h.unit(0) + 0.329 * h.unit(1)
    return (v / np.linalg.norm(v)).astype(np.float32)


EMPTY = (np.zeros((0, 512), dtype=np.float32), [])


class Outcomes(unittest.TestCase):
    """The gate scores what a user sees: accepted-right, accepted-wrong and abstained, not only top-1."""

    def test_an_adapter_that_lowers_scores_does_not_make_the_scanner_abstain(self):
        # Adapted cosines fall to 0.87, raw ones stay 0.994. Thresholds read the raw cosine, as lib/search.js does.
        u = np.zeros((512, 1), dtype=np.float32)
        v = np.zeros((512, 1), dtype=np.float32)
        u[1, 0] = 5.0
        v[2, 0] = 1.0
        q = h.unit(0) * 0.9 + h.unit(2) * 0.1
        q = (q / np.linalg.norm(q)).astype(np.float32)
        res = gate.evaluate({"adapter": pack_adapter(u, v)}, {}, EMPTY, (np.stack([q] * 4), ["c-0"] * 4), catalog=catalog())
        after = res["outcomes"]["regression"]["after"]
        self.assertEqual((after["acceptedRight"], after["acceptedWrong"], after["abstained"]), (4, 0, 0))

    def test_a_fix_that_the_scanner_would_abstain_on_is_not_an_improvement(self):
        # top-1 flips to the right card, but its raw cosine is only 0.55: the user sees an abstain, not a right answer
        res = gate.evaluate({"adapter": boost_c0()}, {}, EMPTY, sets(10, 0, 1, 0.4, "c-0"), catalog=catalog())
        self.assertEqual(res["regression"], {"before": 0.0, "after": 1.0, "n": 10})
        reg = res["outcomes"]["regression"]
        self.assertEqual(reg["before"], {"acceptedRight": 0, "acceptedWrong": 10, "abstained": 0, "n": 10})
        self.assertEqual(reg["after"], {"acceptedRight": 0, "acceptedWrong": 0, "abstained": 10, "n": 10})
        self.assertFalse(res["improved"])

    def test_a_fix_the_user_sees_is_an_improvement(self):
        reg = (np.stack([between_01()] * 10), ["c-0"] * 10)
        res = gate.evaluate({"adapter": boost_c0()}, {}, EMPTY, reg, catalog=near_duplicates())
        self.assertEqual(res["regression"], {"before": 0.0, "after": 1.0, "n": 10})
        self.assertEqual(res["outcomes"]["regression"]["after"]["acceptedRight"], 10)
        self.assertTrue(res["improved"])

    def test_improvement_rules(self):
        base = {"acceptedRight": 10, "acceptedWrong": 5, "abstained": 5, "n": 20}
        better = lambda **over: gate.better_outcome(base, {**base, **over}, top1_before=10, top1_after=12)
        self.assertTrue(better(acceptedRight=12, acceptedWrong=3))
        self.assertFalse(better())  # accepted-right must rise
        self.assertFalse(better(acceptedRight=12, acceptedWrong=6))  # accepted-wrong must not rise
        self.assertFalse(better(acceptedRight=12, abstained=6))  # abstain rate may rise by 2 points at most: 1/20 = 5
        self.assertTrue(gate.better_outcome({**base, "n": 100, "abstained": 5}, {**base, "n": 100, "abstained": 7, "acceptedRight": 12}, 10, 12))
        self.assertFalse(gate.better_outcome(base, {**base, "acceptedRight": 12}, top1_before=10, top1_after=10))  # top-1 must rise


class Gate(unittest.TestCase):
    def test_adapter_that_fixes_regression_is_improved(self):
        reg = (np.stack([between_01()] * 10), ["c-0"] * 10)
        res = gate.evaluate({"adapter": boost_c0()}, {}, EMPTY, reg, catalog=near_duplicates())
        self.assertEqual(res["regression"], {"before": 0.0, "after": 1.0, "n": 10})
        self.assertTrue(res["improved"])
        self.assertEqual(res["exam"]["n"], 0)

    def test_equal_regression_is_not_improved(self):
        reg = sets(10, 0, 1, 0.7, "c-0")
        res = gate.evaluate({}, {}, (np.zeros((0, 512), dtype=np.float32), []), reg, catalog=catalog())
        self.assertFalse(res["improved"])

    def test_exam_loss_blocks_once_n_reaches_30_but_not_before(self):
        reg = (np.stack([between_01()] * 10), ["c-0"] * 10)
        hurt = lambda n: (np.stack([between_01()] * n), ["c-1"] * n)  # catalog says c-1 (right); boost_c0 flips it to c-0
        blocked = gate.evaluate({"adapter": boost_c0()}, {}, hurt(30), reg, catalog=near_duplicates())
        self.assertEqual((blocked["exam"]["before"], blocked["exam"]["after"]), (1.0, 0.0))
        self.assertFalse(blocked["improved"])
        small = gate.evaluate({"adapter": boost_c0()}, {}, hurt(29), reg, catalog=near_duplicates())
        self.assertTrue(small["improved"])

    def test_missing_regression_is_never_improved(self):
        empty = (np.zeros((0, 512), dtype=np.float32), [])
        res = gate.evaluate({"adapter": boost_c0()}, {}, empty, empty, catalog=catalog())
        self.assertFalse(res["improved"])
        self.assertEqual(res["regression"]["n"], 0)

    def test_memory_row_fixes_a_miss_but_is_never_high(self):
        q = blend(0, 1, 0.4)  # catalog says c-1, truth c-0
        mem = write_pack(["c-0"], np.stack([q]))
        reg = (np.stack([q] * 5), ["c-0"] * 5)
        res = gate.evaluate({"memory": mem}, {}, (np.zeros((0, 512), dtype=np.float32), []), reg, catalog=catalog())
        self.assertEqual(res["regression"]["after"], 1.0)
        self.assertEqual(res["wrongHigh"], {"before": 1.0, "after": 0.0})

    def test_adapter_changed_answers_are_never_high(self):
        # raw read is c-1 with a high, clear lead; a wrong adapter flip to c-0 must not count as a wrong "high" read
        cat = Catalog(np.stack([h.unit(k) for k in range(4)]), IDS)
        q = np.stack([blend(0, 1, 0.4)] * 4)
        res = gate.evaluate({"adapter": boost_c0()}, {}, (np.zeros((0, 512), dtype=np.float32), []), (q, ["c-1"] * 4), catalog=cat)
        self.assertEqual(res["regression"]["after"], 0.0)
        self.assertEqual(res["wrongHigh"], {"before": 0.0, "after": 0.0})

    def test_memory_beats_the_raw_catalog_top_not_the_adapted_one(self):
        # The extension compares memory with the raw catalog top score (lib/references.js applyMemory). An adapter
        # that lowers the top score must not make a memory row easier to accept.
        u = np.zeros((512, 1), dtype=np.float32)
        v = np.zeros((512, 1), dtype=np.float32)
        u[1, 0] = 5.0  # adds 5 * (the e2 part) to e1: the adapted top cosine falls to 0.87, the raw one is 0.994
        v[2, 0] = 1.0
        q = h.unit(0) * 0.9 + h.unit(2) * 0.1
        q = (q / np.linalg.norm(q)).astype(np.float32)
        mem = write_pack(["c-3"], np.stack([q]))  # sim 1.0: 0.13 clear of the adapted top, only 0.006 clear of the raw top
        reg = (np.stack([q] * 3), ["c-3"] * 3)
        res = gate.evaluate({"adapter": pack_adapter(u, v), "memory": mem}, {}, (np.zeros((0, 512), dtype=np.float32), []), reg, catalog=catalog())
        self.assertEqual(res["regression"]["after"], 0.0)

    def test_unchanged_wrong_high_reads_are_not_an_improvement(self):
        q = np.stack([blend(0, 1, 0.97)] * 5)  # raw: c-0 high and clear; truth c-1 -> wrong high in both
        res = gate.evaluate({}, {}, (np.zeros((0, 512), dtype=np.float32), []), (q, ["c-1"] * 5), catalog=catalog())
        self.assertEqual(res["wrongHigh"], {"before": 1.0, "after": 1.0})
        self.assertFalse(res["improved"])


if __name__ == "__main__":
    unittest.main()
