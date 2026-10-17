import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import helpers as h
from pipeline.catalog import Catalog
from pipeline.exam import score_exam
from pipeline.regression import load_regression, save_regression

IDS = ["c-0", "c-1", "c-2", "c-3"]


def blend(a, b, wa):
    v = wa * h.unit(a) + (1 - wa) * h.unit(b)
    return (v / np.linalg.norm(v)).astype(np.float32)


class RegressionFile(unittest.TestCase):
    def test_round_trip_holds_embeddings_and_ids_only(self):
        emb = np.stack([h.unit(0), h.unit(1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "regression.npz")
            save_regression(path, emb, ["c-0", "c-1"])
            with np.load(path) as raw:
                self.assertEqual(sorted(raw.files), ["cardIds", "embeddings"])  # no images, no paths
            back, ids = load_regression(path)
        np.testing.assert_allclose(back, emb)
        self.assertEqual(ids, ["c-0", "c-1"])

    def test_bad_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.npz")
            np.savez(path, embeddings=np.zeros((2, 512), dtype=np.float32), cardIds=np.array(["a"]))
            with self.assertRaises(ValueError):
                load_regression(path)
            np.savez(path, embeddings=np.ones((1, 512), dtype=np.float32), cardIds=np.array(["a"]))
            with self.assertRaises(ValueError):  # not unit length
                load_regression(path)
            np.savez(path, embeddings=np.zeros((1, 100), dtype=np.float32), cardIds=np.array(["a"]))
            with self.assertRaises(ValueError):
                load_regression(path)

    def test_a_pack_that_wins_wrongly_shows_as_a_drop(self):
        catalog = Catalog(np.stack([h.unit(k) for k in range(4)]), IDS)
        q = blend(0, 1, 0.55)  # the catalog correctly says c-0
        res = score_exam([q], ["c-0"], catalog, ["c-1"], np.stack([q]))  # a poisoned row claims this look is c-1
        self.assertEqual((res["before"], res["after"]), (1.0, 0.0))


class BuildScript(unittest.TestCase):
    def test_it_refuses_the_sealed_test_folder(self):
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_regression.py")
        with tempfile.TemporaryDirectory() as tmp:
            labels = os.path.join(tmp, "labels.json")
            with open(labels, "w") as f:
                f.write("[]")
            sealed = os.path.join(tmp, "data", "sealed-test")
            os.makedirs(sealed)
            r = subprocess.run([sys.executable, script, "--labels", labels, "--root", sealed, "--model", "m", "--out", os.path.join(tmp, "o.npz")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sealed", r.stderr.lower())


if __name__ == "__main__":
    unittest.main()
