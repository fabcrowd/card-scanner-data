"""End to end with a fake embedder: contributions on disk in, pack and unsigned manifest out."""
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

import numpy as np

import helpers as h
from pipeline.catalog import Catalog
from pipeline.pack import read_pack
from pipeline.run import load_seed, optional_modules, run_pipeline

IDS = [f"c-{k}" for k in range(8)]
PINS = {"cropVersion": 2, "dim": 512, "reference": {"min": 0.93, "margin": 0.02}, "highTier": {"sim": 0.82, "gap": 0.025}}
COMMON = dict(clip_hash="77538aade655907f", base_url="https://example.com/dl/", seq=9, issued_at="2026-10-15T03:17:00Z")


def read(path, mode="rb"):
    with open(path, mode) as f:
        return f.read()


class FakeEmbedder:
    """Embeds a crop as the basis direction named by its JPEG's red channel (seed = card index)."""

    def __call__(self, jpeg_bytes):
        import io

        from PIL import Image

        red = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").getpixel((0, 0))[0]
        return h.unit(int(round(red / 10)) % 8)


def card_for(k, **kw):
    return h.card(IDS[k], seed=k * 10, **kw)  # red channel ~ seed -> basis k


def catalog():
    return Catalog(np.stack([h.unit(k) for k in range(8)]), IDS)


class Run(unittest.TestCase):
    def build(self, root):
        # 10 installs, each teaching c-1 on its own photo on its own server day.
        for i in range(1, 11):
            it = h.item(install=i, photo=h.photo_id(i, "teach"), received=f"2026-10-{10 + i:02d}T12:00:00.000Z", cards=[card_for(1, box=(0.1, 0.1 + i * 0.01, 0.3, 0.4))])
            h.write_item(root, it, f"t{i}.json")
        # An exam photo labelled c-2 that must never reach the pack.
        h.write_item(root, h.item(install=1, split="exam", photo=h.photo_id(50, "exam"), cards=[card_for(2)]), "e1.json")

    def test_pack_and_manifest(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            summary = run_pipeline(data_dir=root, out_dir=out, pins=PINS, embedder=FakeEmbedder(), modules={}, catalog=catalog(), **COMMON)
            pack_bytes = read(os.path.join(out, "community-pack.bin"))
            pack = read_pack(pack_bytes)
            self.assertEqual(sorted(set(pack.card_ids)), ["c-1"])  # c-2 (exam only) is absent
            self.assertEqual(len(pack.card_ids), 5)  # ten rows qualified; five rows per card is the cap
            manifest = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual(manifest["exam"]["n"], 1)
            self.assertEqual((manifest["seq"], manifest["issuedAt"]), (9, "2026-10-15T03:17:00Z"))
            self.assertIsNone(manifest["regression"])
            self.assertIs(manifest["improved"], False)
            self.assertEqual(manifest["v"], 2)
            self.assertEqual(list(manifest["assets"]), ["memory"])
            self.assertEqual(manifest["assets"]["memory"], {"url": "https://example.com/dl/community-pack.bin", "sha256": hashlib.sha256(pack_bytes).hexdigest(), "bytes": len(pack_bytes)})
            self.assertEqual(summary["entries"], 5)
            self.assertFalse(os.path.exists(os.path.join(out, "manifest.sig")))  # signing is another job's work

    def test_regression_set_is_scored_and_reported(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            regression = (np.stack([h.unit(3), h.unit(4)]), ["c-3", "c-4"])
            run_pipeline(data_dir=root, out_dir=out, pins=PINS, embedder=FakeEmbedder(), modules={}, catalog=catalog(), regression=regression, **COMMON)
            manifest = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual(manifest["regression"], {"before": 1.0, "after": 1.0, "n": 2})

    def test_no_contributions_gives_an_empty_valid_pack(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            run_pipeline(data_dir=root, out_dir=out, pins=PINS, embedder=FakeEmbedder(), modules={}, catalog=catalog(), **COMMON)
            self.assertEqual(read_pack(read(os.path.join(out, "community-pack.bin"))).card_ids, [])
            m = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual((m["exam"]["n"], m["improved"]), (0, False))

    def test_a_hostile_image_is_skipped_not_fatal(self):
        class Strict(FakeEmbedder):
            def __call__(self, jpeg_bytes):
                if len(jpeg_bytes) % 2:
                    raise ValueError("bad image")
                return super().__call__(jpeg_bytes)

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            summary = run_pipeline(data_dir=root, out_dir=out, pins=PINS, embedder=Strict(), modules={}, catalog=catalog(), **COMMON)
            self.assertIn("embedFailed", summary)


class Parts(unittest.TestCase):
    """The optional trainers, gate, seed and status modules are called with the right data and their output is shipped."""

    def setUp(self):
        self.seen = {}
        seen = self.seen

        def train_adapter(pairs, cat):
            seen["pairs"] = pairs
            return b"ADAPTER"

        def train_check(pos, neg):
            seen["pos"], seen["neg"] = pos, neg
            return b"CHECK"

        def evaluate(candidate, live, exam, regression, catalog):
            seen["candidate"], seen["live"], seen["exam_n"], seen["regression_n"] = candidate, live, len(exam[1]), len(regression[1])
            return {"improved": seen.get("improved", True), "regression": {"before": 0.5, "after": 0.6, "n": 2},
                    "exam": {"before": 0.4, "after": 0.5, "n": 1}, "wrongHigh": {"before": 0.1, "after": 0.1},
                    "outcomes": {"regression": {"before": {"abstained": 0}}}}

        def write(path, record):
            seen["status"] = (path, record)

        self.modules = dict(
            train_adapter=types.SimpleNamespace(train=train_adapter),
            train_check=types.SimpleNamespace(train=train_check),
            gate=types.SimpleNamespace(evaluate=evaluate),
            status=types.SimpleNamespace(write=write),
        )

    def go(self, root, out, **over):
        args = dict(data_dir=root, out_dir=out, pins=PINS, embedder=FakeEmbedder(), modules=self.modules, catalog=catalog(),
                    regression=(np.stack([h.unit(3), h.unit(4)]), ["c-3", "c-4"]), **COMMON)
        args.update(over)
        return run_pipeline(**args), json.loads(read(os.path.join(out, "manifest.json")))

    def build(self, root, not_a_card_installs=2):
        Run().build(root)
        for i in range(1, not_a_card_installs + 1):
            h.write_item(root, h.item(install=i, photo=h.photo_id(60, "teach"), received=f"2026-10-{20 + i}T12:00:00.000Z", cards=[h.card(answer={"notACard": True}, seed=0)]), f"n{i}.json")

    def test_every_part_is_written_listed_and_taken_from_the_gate(self):
        extra = {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            summary, m = self.go(root, out, new_sets_fn=lambda: extra, seed_pairs=[(h.unit(3), "c-3")])
            self.assertEqual(sorted(m["assets"]), ["adapter", "catalogExtra", "check", "memory"])
            for name, filename in (("adapter", "adapter.bin"), ("check", "check.bin"), ("catalogExtra", "catalog-extra.bin")):
                data = read(os.path.join(out, filename))
                self.assertEqual(m["assets"][name]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(read(os.path.join(out, "catalog-extra.bin")), b"EXTRA")
            self.assertIs(m["improved"], True)
            self.assertEqual(m["regression"], {"before": 0.5, "after": 0.6, "n": 2})
            self.assertEqual(m["exam"], {"before": 0.4, "after": 0.5, "n": 1, "wrongAtHigh": {"before": 0.1, "after": 0.1}})
            self.assertEqual((summary["newCards"], summary["newSets"]), (1, ["sv9"]))
            self.assertEqual(summary["outcomes"], {"regression": {"before": {"abstained": 0}}})
            self.assertEqual(set(self.seen["candidate"]), {"memory", "adapter"})  # check and catalogExtra do not change answers
            self.assertEqual((self.seen["exam_n"], self.seen["regression_n"]), (1, 2))

    def test_adapter_pairs_are_the_promoted_rows_plus_the_seed(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            self.go(root, out, seed_pairs=[(h.unit(3), "c-3")])
            self.assertEqual(len(self.seen["pairs"]), 5 + 1)
            self.assertEqual(self.seen["pairs"][-1][1], "c-3")
            self.go(root, out)  # no seed sources: no seed
            self.assertEqual(len(self.seen["pairs"]), 5)

    def test_check_learns_from_promoted_cards_and_from_crops_two_installs_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root, not_a_card_installs=2)
            self.go(root, out)
            self.assertEqual((len(self.seen["pos"]), len(self.seen["neg"])), (5, 1))
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root, not_a_card_installs=1)
            self.go(root, out)
            self.assertEqual(len(self.seen["neg"]), 0)  # one install is not enough

    def test_a_trainer_with_too_little_data_leaves_its_part_out(self):
        self.modules["train_adapter"] = types.SimpleNamespace(train=lambda pairs, cat: None)
        self.modules["train_check"] = types.SimpleNamespace(train=lambda pos, neg: None)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            _, m = self.go(root, out)
            self.assertEqual(list(m["assets"]), ["memory"])
            self.assertFalse(os.path.exists(os.path.join(out, "adapter.bin")))

    def test_the_gate_decides_improved_and_no_regression_set_means_never(self):
        self.seen["improved"] = False
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            self.assertIs(self.go(root, out)[1]["improved"], False)
            self.seen["improved"] = True
            _, m = self.go(root, out, regression=None)
            self.assertIsNone(m["regression"])

    def test_load_seed_reads_embeddings_and_ids_without_pickle_and_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seed.npz")
            self.assertEqual(load_seed(path), [])
            np.savez_compressed(path, embeddings=np.stack([h.unit(1) * 3, h.unit(2)]), cardIds=np.array(["c-1", "c-2"], dtype="<U64"))
            pairs = load_seed(path)
            self.assertEqual([c for _, c in pairs], ["c-1", "c-2"])
            self.assertAlmostEqual(float(np.linalg.norm(pairs[0][0])), 1.0, places=5)
            np.savez_compressed(path, embeddings=np.zeros((2, 8), dtype=np.float32), cardIds=np.array(["a", "b"]))
            with self.assertRaises(ValueError):
                load_seed(path)
            np.savez_compressed(path, embeddings=np.full((1, 512), np.nan, dtype=np.float32), cardIds=np.array(["a"]))
            with self.assertRaises(ValueError):
                load_seed(path)

    def test_the_live_bundle_goes_to_the_gate(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            self.go(root, out, live={"adapter": b"OLD"})
            self.assertEqual(self.seen["live"], {"adapter": b"OLD"})

    def test_a_night_that_only_adds_cards_ships_when_nothing_learned_changed(self):
        self.modules["gate"] = types.SimpleNamespace(evaluate=lambda candidate, live, exam, regression, catalog: {
            "improved": False, "regression": {"before": 0.6, "after": 0.6, "n": 2}, "exam": {"before": 0.0, "after": 0.0, "n": 0}, "wrongHigh": {"before": 0.0, "after": 0.0}})
        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            self.go(root, out, new_sets_fn=extra)
            same = {"memory": read(os.path.join(out, "community-pack.bin")), "adapter": b"ADAPTER"}
            _, m = self.go(root, out, new_sets_fn=extra, live=same)
            self.assertIs(m["improved"], True)
            _, m = self.go(root, out, new_sets_fn=extra, live={**same, "adapter": b"OLD"})
            self.assertIs(m["improved"], False)  # the learning changed and the gate did not approve it
            _, m = self.go(root, out, new_sets_fn=lambda: {"bytes": None, "added": [], "sets": []}, live=same)
            self.assertIs(m["improved"], False)  # nothing new either

    def test_status_is_written_with_the_run_record(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            history = os.path.join(out, "earlier.json")
            with open(history, "w") as f:
                f.write("[]")
            self.go(root, out, status_path=os.path.join(out, "STATUS.md"), status_history=history, new_sets_fn=lambda: {"bytes": b"E", "added": ["x-1"], "sets": ["x"]})
            path, record = self.seen["status"]
            self.assertEqual(path, os.path.join(out, "STATUS.md"))
            self.assertEqual(record, {
                "date": "2026-10-15", "seq": 9, "contributors": 10, "promoted": 5, "newSets": ["x"],
                "regression": {"before": 0.5, "after": 0.6}, "exam": {"before": 0.4, "after": 0.5, "n": 1}, "shipped": True,
            })
            self.assertTrue(os.path.exists(os.path.join(out, "status.json")))

    def test_a_missing_optional_module_is_not_an_error_but_a_broken_one_is(self):
        with unittest.mock.patch("importlib.import_module", side_effect=lambda full: (_ for _ in ()).throw(ModuleNotFoundError(full, name=full))):
            self.assertEqual(optional_modules(), {})
        with unittest.mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("numpy", name="numpy")):
            with self.assertRaises(ModuleNotFoundError):
                optional_modules()


class Keygen(unittest.TestCase):
    def test_refuses_to_write_inside_the_repo_and_prints_the_public_key(self):
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "keygen.py")
        inside = os.path.join(os.path.dirname(script), "key.pem")
        r = subprocess.run([sys.executable, script, "--out", inside], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(inside))
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "key.pem")
            r = subprocess.run([sys.executable, script, "--out", target], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(read(target).startswith(b"-----BEGIN PRIVATE KEY-----"))
            self.assertTrue(base64.b64decode(r.stdout.strip().splitlines()[-1]))
            again = subprocess.run([sys.executable, script, "--out", target], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)  # never overwrite a key


if __name__ == "__main__":
    unittest.main()
