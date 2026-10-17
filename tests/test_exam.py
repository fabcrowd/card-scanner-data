import unittest

import numpy as np

import helpers as h
from pipeline.catalog import Catalog
from pipeline.contributions import build_samples
from pipeline.exam import exam_cases, is_high, score_exam
from pipeline.manifest import is_improved

IDS = ["c-0", "c-1", "c-2", "c-3"]


def catalog():
    """Four orthogonal cards; c-3's row is deliberately pulled toward c-2 to make a confusable pair."""
    return Catalog(np.stack([h.unit(k) for k in range(4)]), IDS)


def blend(a, b, wa):
    v = wa * h.unit(a) + (1 - wa) * h.unit(b)
    return (v / np.linalg.norm(v)).astype(np.float32)


class ExamCases(unittest.TestCase):
    def test_only_exam_contributions_with_every_card_answered_are_used(self):
        ok = h.item(split="exam", photo=h.photo_id(1, "exam"), cards=[h.card("c-1")])
        with_unknown = h.item(split="exam", photo=h.photo_id(2, "exam"), cards=[h.card("c-1"), h.card(answer={"unknown": True}, box=(0.5, 0.5, 0.2, 0.2))])
        teach = h.item(split="teach", photo=h.photo_id(3, "teach"), cards=[h.card("c-1")])
        cases = exam_cases([ok, with_unknown, teach], valid_ids=set(IDS))
        self.assertEqual([(s.photo_id, truth) for s, truth in cases], [(h.photo_id(1, "exam"), "c-1")])

    def test_not_a_card_cards_are_not_scored_but_do_not_disqualify(self):
        it = h.item(split="exam", photo=h.photo_id(1, "exam"), cards=[h.card("c-1"), h.card(answer={"notACard": True}, box=(0.5, 0.5, 0.2, 0.2))])
        self.assertEqual(len(exam_cases([it], valid_ids=set(IDS))), 1)

    def test_conflicting_labels_for_one_crop_are_skipped(self):
        p = h.photo_id(1, "exam")
        a = h.item(install=1, split="exam", photo=p, cards=[h.card("c-1")])
        b = h.item(install=2, split="exam", photo=p, cards=[h.card("c-2")])
        self.assertEqual(exam_cases([a, b], valid_ids=set(IDS)), [])


class HighTier(unittest.TestCase):
    def test_needs_sim_and_a_clear_lead(self):
        self.assertTrue(is_high([0.9, 0.8]))
        self.assertFalse(is_high([0.81, 0.5]))  # below the high sim
        self.assertFalse(is_high([0.9, 0.88]))  # lead too small
        self.assertTrue(is_high([0.9]))


class Scoring(unittest.TestCase):
    def test_pack_row_that_beats_the_catalog_fixes_a_miss(self):
        cat = catalog()
        q = blend(0, 1, 0.55)  # catalog picks c-0, truth is c-1
        pack_ids, pack_vecs = ["c-1"], np.stack([q])  # near-exact pack row
        res = score_exam([q], ["c-1"], cat, pack_ids, pack_vecs)
        self.assertEqual((res["n"], res["before"], res["after"]), (1, 0.0, 1.0))

    def test_pack_cannot_break_a_correct_catalog_answer_unless_it_wins_by_the_margin(self):
        cat = catalog()
        q = h.unit(0)
        res = score_exam([q], ["c-0"], cat, ["c-1"], np.stack([blend(0, 1, 0.5)]))
        self.assertEqual((res["before"], res["after"]), (1.0, 1.0))

    def test_pack_row_for_a_card_outside_the_catalog_top_k_is_ignored(self):
        cat = catalog()
        q = h.unit(0)
        res = score_exam([q], ["c-0"], cat, ["not-in-catalog"], np.stack([q]), ref_min=0.0, ref_margin=-1.0)
        self.assertEqual(res["after"], 1.0)

    def test_wrong_at_high_counts_confident_misses_and_pack_hits_are_never_high(self):
        cat = catalog()
        q = h.unit(0)  # catalog: c-0 at 1.0, others 0.0 -> high tier
        res = score_exam([q], ["c-1"], cat, [], np.zeros((0, 512), dtype=np.float32))
        self.assertEqual(res["wrongAtHigh"], {"before": 1.0, "after": 1.0})
        res = score_exam([q], ["c-1"], cat, ["c-1"], np.stack([q]), ref_min=0.5, ref_margin=-1.0)
        self.assertEqual(res["after"], 1.0)
        self.assertEqual(res["wrongAtHigh"]["after"], 0.0)

    def test_empty_exam(self):
        res = score_exam(np.zeros((0, 512), dtype=np.float32), [], catalog(), [], np.zeros((0, 512), dtype=np.float32))
        self.assertEqual((res["n"], res["before"], res["after"]), (0, 0.0, 0.0))


class Improved(unittest.TestCase):
    REG = {"before": 0.8, "after": 0.8, "n": 10}

    def test_needs_a_gain_and_at_least_30_cases(self):
        self.assertTrue(is_improved({"before": 0.5, "after": 0.6, "n": 30}, self.REG))
        self.assertFalse(is_improved({"before": 0.5, "after": 0.6, "n": 29}, self.REG))
        self.assertFalse(is_improved({"before": 0.5, "after": 0.5, "n": 100}, self.REG))
        self.assertFalse(is_improved({"before": 0.6, "after": 0.5, "n": 100}, self.REG))


class ExamNeverEntersThePack(unittest.TestCase):
    def test_exam_samples_cannot_be_promoted(self):
        from pipeline.consensus import promote

        exam_item = h.item(split="exam", photo=h.photo_id(1, "exam"), cards=[h.card("c-1")])
        samples = build_samples([exam_item])
        with self.assertRaises(ValueError):
            promote(samples, np.stack([h.unit(1)]), [{"c-1"}])

    def test_build_samples_can_filter_by_split(self):
        items = [h.item(split="exam", photo=h.photo_id(1, "exam")), h.item(split="teach", photo=h.photo_id(2, "teach"))]
        self.assertEqual([s.split for s in build_samples(items, split="teach")], ["teach"])


if __name__ == "__main__":
    unittest.main()
