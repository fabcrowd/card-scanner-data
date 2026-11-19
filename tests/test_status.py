import json
import os
import tempfile
import unittest

import _paths  # noqa: F401
from pipeline import status


def rec(seq, **kw):
    r = {"date": "2026-10-16", "seq": seq, "contributors": 3, "promoted": 12, "newSets": ["sv9"],
         "regression": {"before": 0.5, "after": 0.6}, "exam": {"before": 0.4, "after": 0.45, "n": 40}, "shipped": True}
    r.update(kw)
    return r


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rows(path):
    return [l for l in read(path).splitlines() if l.startswith("| 2026")]


class Status(unittest.TestCase):
    def test_writes_table_row_and_history_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            status.write(p, rec(7))
            self.assertEqual(rows(p), ["| 2026-10-16 | 7 | 3 | 12 | sv9 | 0.5 -> 0.6 | 0.4 -> 0.45 (n=40) | yes |"])
            self.assertEqual(load(os.path.join(d, "status.json"))[0]["seq"], 7)

    def test_newest_first_and_only_last_60_kept(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            for i in range(65):
                status.write(p, rec(i))
            hist = load(os.path.join(d, "status.json"))
            self.assertEqual((len(hist), hist[-1]["seq"]), (60, 64))
            self.assertEqual(len(rows(p)), 60)
            self.assertTrue(rows(p)[0].startswith("| 2026-10-16 | 64 |"))

    def test_missing_exam_and_not_shipped_render_dashes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            status.write(p, rec(1, exam=None, shipped=False, newSets=[]))
            self.assertEqual(rows(p), ["| 2026-10-16 | 1 | 3 | 12 | - | 0.5 -> 0.6 | - | no |"])

    def test_pipes_in_cells_are_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            status.write(p, rec(1, newSets=["a|b"]))
            self.assertIn("a\\|b", read(p))

    def test_held_cards_are_kept_in_the_history_and_listed_under_the_table_for_the_latest_night_only(self):
        held = [{"id": "sv9-1", "name": "Thief", "reason": "lost a correct read"}]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            status.write(p, rec(7, held=held))
            self.assertEqual(load(os.path.join(d, "status.json"))[0]["held"], held)
            self.assertEqual(rows(p), ["| 2026-10-16 | 7 | 3 | 12 | sv9 | 0.5 -> 0.6 | 0.4 -> 0.45 (n=40) | yes |"])  # the table is unchanged
            self.assertIn("- sv9-1 (Thief): lost a correct read", read(p))
            status.write(p, rec(8))
            self.assertNotIn("sv9-1", read(p))
            self.assertEqual(load(os.path.join(d, "status.json"))[0]["held"], held)  # the history keeps it

    def test_a_run_without_held_cards_writes_the_same_file_as_before(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            status.write(p, rec(7))
            self.assertNotIn("Held back", read(p))
            self.assertNotIn("held", load(os.path.join(d, "status.json"))[0])

    def test_corrupt_history_starts_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATUS.md")
            with open(os.path.join(d, "status.json"), "w") as f:
                f.write("{nope")
            status.write(p, rec(2))
            self.assertEqual(len(load(os.path.join(d, "status.json"))), 1)


if __name__ == "__main__":
    unittest.main()
