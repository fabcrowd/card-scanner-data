import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import helpers as h
from pipeline.catalog import Catalog
from pipeline.exam import score_exam
from pipeline.names import NameTruth
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


MODEL = "0dba1302580da2a6"
NAME_KEYS = ["acceptIds", "acceptOffsets", "cardIds", "crop", "embeddings", "model", "names"]


def name_file(**over):
    """The arrays of a valid mixed file (an id row, a name row with two accept ids, a name row with none); a value of
    None in `over` drops that key."""
    arrays = dict(
        embeddings=np.stack([h.unit(0), h.unit(1), h.unit(2)]),
        cardIds=np.array(["c-0", "", ""]),
        names=np.array(["", "pikachu", "raichu"]),
        acceptIds=np.array(["a-1", "a-2"]),
        acceptOffsets=np.array([0, 0, 2, 2], dtype=np.int32),
        model=np.array(MODEL),
        crop=np.array(3),
    )
    arrays.update(over)
    return {k: v for k, v in arrays.items() if v is not None}


class NameRows(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "r.npz")

    def write(self, **over):
        np.savez(self.path, **name_file(**over))

    def test_a_mixed_file_round_trips_with_exactly_the_new_key_set(self):
        truths = ["c-0", NameTruth("pikachu", frozenset({"a-1", "a-2"})), NameTruth("raichu", frozenset())]
        emb = np.stack([h.unit(0), h.unit(1), h.unit(2)])
        save_regression(self.path, emb, truths, model=MODEL, crop=3)
        with np.load(self.path, allow_pickle=False) as raw:
            self.assertEqual(sorted(raw.files), NAME_KEYS)
            self.assertEqual(raw["acceptOffsets"].dtype, np.int32)
            self.assertEqual((str(raw["model"]), int(raw["crop"])), (MODEL, 3))
        back, loaded = load_regression(self.path)
        np.testing.assert_allclose(back, emb)
        self.assertEqual(loaded, truths)

    def test_a_file_of_only_name_rows_round_trips(self):
        truths = [NameTruth("pikachu", frozenset({"a-1"}))]
        save_regression(self.path, np.stack([h.unit(0)]), truths, model=MODEL, crop=3)
        self.assertEqual(load_regression(self.path)[1], truths)

    def test_saving_name_rows_without_a_tag_is_refused(self):
        with self.assertRaises(ValueError):
            save_regression(self.path, np.stack([h.unit(0)]), [NameTruth("pikachu", frozenset())])

    def test_an_all_id_file_is_still_written_in_the_old_key_set(self):
        save_regression(self.path, np.stack([h.unit(0)]), ["c-0"], model=MODEL, crop=3)
        with np.load(self.path, allow_pickle=False) as raw:
            self.assertEqual(sorted(raw.files), ["cardIds", "crop", "embeddings", "model"])

    def test_the_valid_fixture_loads(self):
        self.write()
        self.assertEqual(load_regression(self.path)[1], ["c-0", NameTruth("pikachu", frozenset({"a-1", "a-2"})), NameTruth("raichu", frozenset())])

    def test_every_malformed_file_is_rejected(self):
        bad = {
            "untagged": dict(model=None, crop=None),
            "model without crop": dict(crop=None),
            "extra key": dict(extra=np.array(1)),
            "missing names": dict(names=None),
            "missing acceptIds": dict(acceptIds=None),
            "names length": dict(names=np.array(["", "pikachu"])),
            "cardIds length": dict(cardIds=np.array(["c-0", ""])),
            "offsets length": dict(acceptOffsets=np.array([0, 0, 2], dtype=np.int32)),
            "offsets do not start at 0": dict(acceptOffsets=np.array([1, 1, 2, 2], dtype=np.int32)),
            "offsets decrease": dict(acceptOffsets=np.array([0, 2, 1, 2], dtype=np.int32)),
            "offsets run past the ids": dict(acceptOffsets=np.array([0, 0, 2, 3], dtype=np.int32)),
            "offsets stop short of the ids": dict(acceptOffsets=np.array([0, 0, 1, 1], dtype=np.int32)),
            "negative offset": dict(acceptOffsets=np.array([0, -1, 2, 2], dtype=np.int32)),
            "offsets are floats": dict(acceptOffsets=np.array([0, 0, 2, 2], dtype=np.float32)),
            "offsets are 2-d": dict(acceptOffsets=np.zeros((2, 2), dtype=np.int32)),
            "both labels": dict(cardIds=np.array(["c-0", "c-1", ""])),
            "no label": dict(names=np.array(["", "", "raichu"])),
            "an id row with accept ids": dict(acceptOffsets=np.array([0, 2, 2, 2], dtype=np.int32)),
            "an accept id that is empty": dict(acceptIds=np.array(["a-1", ""])),
            "an overlong accept id": dict(acceptIds=np.array(["a-1", "x" * 65])),
            "an overlong card id": dict(cardIds=np.array(["x" * 65, "", ""])),
            "an overlong name": dict(names=np.array(["", "x" * 129, "raichu"])),
            "ids that are not strings": dict(acceptIds=np.array([1, 2])),
            "names that are not strings": dict(names=np.array([0, 1, 2])),
            "not unit length": dict(embeddings=np.ones((3, 512), dtype=np.float32)),
            "wrong dimension": dict(embeddings=np.stack([h.unit(0)] * 3)[:, :100]),
            "model is not a string": dict(model=np.array(5)),
            "model is an array": dict(model=np.array([MODEL])),
            "crop is a float": dict(crop=np.array(3.0)),
            "crop is a string": dict(crop=np.array("3")),
        }
        for label, over in bad.items():
            with self.subTest(label):
                np.savez(self.path, **name_file(**over))
                with self.assertRaises(ValueError):
                    load_regression(self.path)

    def test_a_pickled_object_array_is_rejected(self):
        np.savez(self.path, **name_file(names=np.array(["", {"a": 1}, ""], dtype=object)))
        with self.assertRaises(ValueError):
            load_regression(self.path)

    def test_old_key_sets_still_load_and_come_back_as_plain_ids(self):
        np.savez(self.path, embeddings=np.stack([h.unit(0)]), cardIds=np.array(["c-0"]))
        self.assertEqual(load_regression(self.path)[1], ["c-0"])
        np.savez(self.path, embeddings=np.stack([h.unit(0)]), cardIds=np.array(["c-0"]), model=np.array(MODEL), crop=np.array(3))
        ids = load_regression(self.path)[1]
        self.assertEqual(ids, ["c-0"])
        self.assertIsInstance(ids[0], str)


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


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_regression.py")


class BuildScriptNameMode(unittest.TestCase):
    """--embeddings mode: embeddings that already exist, labelled by card name."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def put(self, name, data):
        path = os.path.join(self.tmp, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if name.endswith(".npy"):
            np.save(path, data)
        elif isinstance(data, bytes):
            with open(path, "wb") as f:
                f.write(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        return path

    def args(self, **over):
        files = dict(
            embeddings=self.put("emb.npy", np.stack([h.unit(0), h.unit(1), h.unit(2), h.unit(3)])),
            labels=self.put("labels.json", [{"row": 2, "name": "Pikachu ex"}, {"row": 0, "name": "Raichu"}, {"row": 1, "name": "Mewthree"}]),
            catalog_names=self.put("names.json", {"p-1": "Pikachu", "p-2": "Pikachu V", "r-1": "Raichu", "x-9": "Pikachu", "z-1": "Other"}),
            id_map=self.put("id_map.json", ["p-1", "p-2", "r-1", "z-1"]),
            model=self.put("m.onnx", b"not really a model"),
            out=os.path.join(self.tmp, "o.npz"),
        )
        files.update(over)
        flags = {"catalog_names": "--catalog-names", "id_map": "--id-map"}
        return [a for k, v in files.items() if v is not None for a in (flags.get(k, f"--{k}"), v)]

    def run_script(self, args):
        return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)

    def test_it_writes_name_rows_with_accept_lists_limited_to_the_id_map(self):
        pins = self.put("pins.json", {"cropVersion": 7})
        r = self.run_script(self.args() + ["--pins", pins])
        self.assertEqual(r.returncode, 0, r.stderr)
        emb, truths = load_regression(os.path.join(self.tmp, "o.npz"))
        np.testing.assert_allclose(emb, np.stack([h.unit(2), h.unit(0), h.unit(1)]))
        self.assertEqual(truths, [
            NameTruth("pikachu", frozenset({"p-1", "p-2"})),  # x-9 is not in the id map; the V suffix is dropped from "Pikachu V"
            NameTruth("raichu", frozenset({"r-1"})),
            NameTruth("mewthree", frozenset()),
        ])
        self.assertIn("wrote 3 rows", r.stderr)
        self.assertIn("rows with an empty accept list: 1", r.stderr)
        self.assertIn("distinct base names: 3", r.stderr)

    def test_the_file_is_tagged_with_the_models_hash_and_the_pins_crop(self):
        import hashlib

        pins = self.put("pins.json", {"cropVersion": 7})
        self.assertEqual(self.run_script(self.args() + ["--pins", pins]).returncode, 0)
        with np.load(os.path.join(self.tmp, "o.npz"), allow_pickle=False) as raw:
            self.assertEqual((str(raw["model"]), int(raw["crop"])), (hashlib.sha256(b"not really a model").hexdigest()[:16], 7))

    def test_it_refuses_a_sealed_path_in_any_argument(self):
        for key in ("embeddings", "labels", "catalog_names", "id_map", "model", "out"):
            with self.subTest(key):
                ext = {"embeddings": "e.npy", "out": "o.npz", "model": "m.onnx"}.get(key, "x.json")
                base = self.args()
                sealed = os.path.join(self.tmp, "data", "sealed-test", ext)
                if key != "out":
                    self.put(os.path.join("data", "sealed-test", ext), np.stack([h.unit(0)]) if ext == "e.npy" else ([] if ext == "x.json" else b"m"))
                flag = {"catalog_names": "--catalog-names", "id_map": "--id-map"}.get(key, f"--{key}")
                base[base.index(flag) + 1] = sealed
                r = self.run_script(base)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("sealed", r.stderr.lower())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o.npz")))

    def test_bad_inputs_stop_the_build(self):
        cases = {
            "a row past the embeddings": dict(labels=self.put("l1.json", [{"row": 9, "name": "Raichu"}])),
            "a negative row": dict(labels=self.put("l2.json", [{"row": -1, "name": "Raichu"}])),
            "a label with no base name": dict(labels=self.put("l3.json", [{"row": 0, "name": "ex"}])),
            "embeddings that are not unit length": dict(embeddings=self.put("e2.npy", np.ones((4, 512), dtype=np.float32))),
        }
        for label, over in cases.items():
            with self.subTest(label):
                r = self.run_script(self.args(**over))
                self.assertNotEqual(r.returncode, 0)
                self.assertFalse(os.path.exists(os.path.join(self.tmp, "o.npz")))


if __name__ == "__main__":
    unittest.main()
