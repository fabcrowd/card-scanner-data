import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

import _paths  # noqa: F401
import numpy as np
from PIL import Image

SCRIPT = os.path.join(_paths.ROOT, "scripts", "build_seed.py")
spec = importlib.util.spec_from_file_location("build_seed", SCRIPT)
build_seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_seed)

META = [{"id": "base1-4", "name": "Charizard", "set": "Base Set", "number": "4"}, {"id": "base1-5", "name": "Mewtwo", "set": "Base Set", "number": "5"}]


def embedder(data, trusted=False):
    v = np.zeros(512, dtype=np.float32)
    v[sum(data[:50]) % 512] = 1.0
    return v


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (100, 140), (seed, 60, 90)).save(buf, "JPEG")
    return buf.getvalue()


def world(root):
    os.makedirs(os.path.join(root, "variant-photos", "crops"))
    with open(os.path.join(root, "variant-photos", "manifest.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"listing": "123", "title": "Charizard 4/102 Base Set"}) + "\n")
    with open(os.path.join(root, "variant-photos", "crops", "123_00_0.jpg"), "wb") as f:
        f.write(jpeg(10))
    os.makedirs(os.path.join(root, "images"))
    for i, card in enumerate(META):
        Image.new("RGB", (100, 140), (i * 80, 50, 90)).save(os.path.join(root, "images", card["id"] + ".png"))
    meta = os.path.join(root, "meta.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(META, f)
    return meta


class BuildSeed(unittest.TestCase):
    def test_writes_embeddings_ids_and_the_model_tag_and_reads_back_without_pickle(self):
        with tempfile.TemporaryDirectory() as root:
            meta = world(root)
            out = os.path.join(root, "seed.npz")
            model = os.path.join(root, "m.onnx")
            with open(model, "wb") as f:
                f.write(b"model bytes")
            pins = os.path.join(root, "pins.json")
            with open(pins, "w") as f:
                f.write('{"cropVersion": 3}')
            n = build_seed.main(["--data-dir", root, "--meta", meta, "--catalog-images", os.path.join(root, "images"), "--model", model,
                                 "--pins", pins, "--out", out, "--per-card", "2"], embedder=embedder)
            self.assertEqual(n, 1 + 2 * 2)
            with np.load(out, allow_pickle=False) as z:
                self.assertEqual(sorted(z.files), ["cardIds", "crop", "embeddings", "model"])
                self.assertEqual(z["embeddings"].shape, (5, 512))
                self.assertEqual(z["embeddings"].dtype, np.float32)
                self.assertEqual(list(z["cardIds"]).count("base1-4"), 3)
                self.assertEqual((str(z["model"]), int(z["crop"])), (hashlib.sha256(b"model bytes").hexdigest()[:16], 3))
            from pipeline.run import load_seed

            self.assertEqual(len(load_seed(out)), 5)  # the extra keys do not trouble the loader

    def test_forbidden_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("hibid-crops", "sealed-test", "realworld", "Sealed"):
                bad = os.path.join(root, name)
                os.makedirs(bad)
                meta = world(bad)
                with self.assertRaises(SystemExit, msg=name):
                    build_seed.main(["--data-dir", bad, "--meta", meta, "--catalog-images", os.path.join(bad, "images"), "--model", "m", "--out", os.path.join(root, "x.npz")], embedder=embedder)
                self.assertFalse(os.path.exists(os.path.join(root, "x.npz")))

    def test_is_forbidden(self):
        self.assertTrue(build_seed.is_forbidden("/a/data/HiBid/x"))
        self.assertFalse(build_seed.is_forbidden("/a/data/variant-photos"))


if __name__ == "__main__":
    unittest.main()
