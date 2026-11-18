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

    def test_tagged_round_trip_stores_the_model_and_crop(self):
        emb = np.stack([h.unit(0), h.unit(1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "regression.npz")
            save_regression(path, emb, ["c-0", "c-1"], model="0dba1302580da2a6", crop=3)
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(sorted(raw.files), ["cardIds", "crop", "embeddings", "model"])
                self.assertEqual((str(raw["model"]), int(raw["crop"])), ("0dba1302580da2a6", 3))
            back, ids = load_regression(path)
        np.testing.assert_allclose(back, emb)
        self.assertEqual(ids, ["c-0", "c-1"])

    def test_other_key_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.npz")
            ok = dict(embeddings=np.stack([h.unit(0)]), cardIds=np.array(["a"]))
            np.savez(path, **ok, model=np.array("0dba1302580da2a6"))  # a model without a crop
            with self.assertRaises(ValueError):
                load_regression(path)
            np.savez(path, **ok, model=np.array("0dba1302580da2a6"), crop=np.array(3), extra=np.array(1))
            with self.assertRaises(ValueError):
                load_regression(path)

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
    def test_it_tags_the_file_with_the_models_hash_and_the_pins_crop(self):
        import hashlib
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("build_regression", os.path.join(root, "scripts", "build_regression.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.onnx")
            with open(model, "wb") as f:
                f.write(b"not really a model")
            pins = os.path.join(tmp, "pins.json")
            with open(pins, "w") as f:
                f.write('{"cropVersion": 7}')
            self.assertEqual(mod.tag_for(model, pins), (hashlib.sha256(b"not really a model").hexdigest()[:16], 7))
            default = mod.tag_for(model, None)
            with open(os.path.join(root, "pins.json"), encoding="utf-8") as f:
                self.assertEqual(default[1], __import__("json").load(f)["cropVersion"])

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
