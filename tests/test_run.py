"""End to end with a fake embedder: contributions on disk in, pack and unsigned manifest out."""
import base64
import contextlib
import hashlib
import io
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
from pipeline.regression import save_regression
from pipeline.run import load_seed, model_tag_ok, optional_modules, run_pipeline

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

    def test_box_signal_rows_change_no_output_and_are_counted(self):
        with tempfile.TemporaryDirectory() as plain, tempfile.TemporaryDirectory() as extra, tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
            self.build(plain)
            self.build(extra)
            # Signals beside real cards, inside the exam photo (an "unknown" row must not disqualify it), and alone from a new install.
            h.write_item(extra, h.item(install=1, split="exam", photo=h.photo_id(50, "exam"), received="2026-10-15T13:00:00.000Z",
                                       cards=[h.signal_card("drawn", box=(0.6, 0.6, 0.2, 0.2)), h.signal_card("removed", box=(0.7, 0.1, 0.2, 0.2))]), "e2.json")
            h.write_item(extra, h.item(install=11, photo=h.photo_id(60, "teach"), cards=[h.signal_card("put-back"), h.signal_card("drawn", answer={"cardId": "c-1"}, box=(0.5, 0.5, 0.2, 0.2))]), "s1.json")
            a = run_pipeline(data_dir=plain, out_dir=out1, pins=PINS, embedder=FakeEmbedder(), modules={}, catalog=catalog(), **COMMON)
            b = run_pipeline(data_dir=extra, out_dir=out2, pins=PINS, embedder=FakeEmbedder(), modules={}, catalog=catalog(), **COMMON)
            for name in ("community-pack.bin", "manifest.json"):
                self.assertEqual(read(os.path.join(out1, name)), read(os.path.join(out2, name)), name)
            self.assertEqual(b.pop("boxSignals"), {"drawn": 2, "removed": 1, "putBack": 1})
            self.assertEqual(a.pop("boxSignals"), {"drawn": 0, "removed": 0, "putBack": 0})
            self.assertEqual(a, b)  # includes contributionsDropped: valid signal files are not dropped
            self.assertEqual(b["contributionsDropped"], 0)

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
            self.assertEqual(set(self.seen["candidate"]), {"memory", "adapter", "catalogExtra"})  # check does not change answers; the new rows are searched
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
            np.savez_compressed(path, embeddings=np.stack([h.unit(1) * 3, h.unit(2)]), cardIds=np.array(["c-1", "c-2"], dtype="<U64"),
                                model=np.array("0dba1302580da2a6"), crop=np.array(3))  # the tag keys do not trouble the loader
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

    def only_cards_gate(self):
        self.modules["gate"] = types.SimpleNamespace(evaluate=lambda candidate, live, exam, regression, catalog: {
            "improved": False, "regression": {"before": 0.6, "after": 0.6, "n": 2}, "exam": {"before": 0.0, "after": 0.0, "n": 0}, "wrongHigh": {"before": 0.0, "after": 0.0}})
        self.modules["train_adapter"] = types.SimpleNamespace(train=lambda pairs, cat: None)
        self.modules["train_check"] = types.SimpleNamespace(train=lambda pos, neg: None)

    def test_a_model_switch_night_ships_new_cards_with_a_regression_set_and_no_adapter(self):
        """Empty inbox, the old live bundle ignored (live = {}), a tagged regression set, new cards: the empty memory pack
        counts as equal to no live memory part."""
        self.only_cards_gate()
        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=extra, live={})
            self.assertIs(m["improved"], True)
            self.assertEqual(sorted(m["assets"]), ["catalogExtra", "memory"])
            self.assertFalse(os.path.exists(os.path.join(out, "adapter.bin")))
            # a non-empty memory pack is learning and still needs the gate
            self.build(root)
            _, m = self.go(root, out, new_sets_fn=extra, live={})
            self.assertIs(m["improved"], False)

    def test_an_empty_pack_does_not_ship_through_the_cards_only_rule_over_a_live_memory_part(self):
        self.only_cards_gate()
        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=extra, live={"memory": b"LIVE PACK"})
            self.assertIs(m["improved"], False)

    def test_an_empty_pack_with_a_live_adapter_that_differs_does_not_ship_through_the_rule(self):
        self.only_cards_gate()  # tonight trains no adapter
        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=extra, live={"adapter": b"OLD"})
            self.assertIs(m["improved"], False)

    def test_the_real_gate_ships_a_model_switch_night_with_new_cards_and_no_adapter(self):
        from pipeline import gate

        self.modules = {"gate": gate}
        result = self.extra_result([h.unit(100)])
        extra = lambda: result  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=extra, live={})
            self.assertIs(m["improved"], True)
            self.assertEqual(sorted(m["assets"]), ["catalogExtra", "memory"])
            self.assertFalse(os.path.exists(os.path.join(out, "adapter.bin")))

    def test_a_failed_set_list_gives_a_night_with_no_new_cards_and_no_catalog_extra(self):
        from pipeline import new_sets

        def down(url):
            raise OSError("set list down")

        def step():
            return new_sets.build(set(), None, lambda data, trusted=False: None, fetch=down, sleep=lambda s: None)

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            summary, m = self.go(root, out, new_sets_fn=step, live={}, status_path=os.path.join(out, "STATUS.md"))
            self.assertEqual((summary["newCards"], summary["newSets"]), (0, []))
            self.assertNotIn("catalogExtra", m["assets"])
            self.assertFalse(os.path.exists(os.path.join(out, "catalog-extra.bin")))
            self.assertEqual(self.seen["status"][1]["newSets"], [])
            # with a previous same-model file the part is kept as it was, and the summary still claims no new cards
            previous = new_sets.write_extra([{"id": "sv9-2", "name": "R", "set": "sv9", "setName": "J", "number": "2", "image": "https://assets.tcgdex.net/x"}],
                                            np.zeros((1, 512), dtype=np.int8), tag={"model": "m" * 16, "crop": 3})
            summary, m = self.go(root, out, new_sets_fn=lambda: new_sets.build(set(), previous, lambda d, trusted=False: None, fetch=down, sleep=lambda s: None,
                                                                            tag={"model": "m" * 16, "crop": 3}), live={})
            self.assertEqual((summary["newCards"], summary["newSets"]), (0, []))
            self.assertEqual(read(os.path.join(out, "catalog-extra.bin")), previous)

    def test_new_cards_alone_never_ship_without_a_regression_set(self):
        self.only_cards_gate()
        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=extra, live={}, regression=None)
            self.assertIs(m["improved"], False)

    # --- the replay sees the new-card rows ---------------------------------------------------------------------------
    def spy_gate(self):
        from pipeline import gate

        def evaluate(candidate, live, exam, regression, catalog):
            self.seen["candidate"], self.seen["live"] = candidate, live
            return gate.evaluate(candidate, live, exam, regression, catalog=catalog)

        self.modules = {"gate": types.SimpleNamespace(evaluate=evaluate)}

    @staticmethod
    def extra_result(vectors, ids=("sv9-1",)):
        from pipeline import new_sets

        cards = [{"id": i, "name": f"New {i}", "set": "sv9", "setName": "S", "number": str(k), "image": "https://assets.tcgdex.net/x"} for k, i in enumerate(ids)]
        data = new_sets.write_extra(cards, np.rint(np.stack(vectors) * 127).astype(np.int8))
        return {"bytes": data, "added": list(ids), "sets": ["sv9"], "total": len(ids)}

    def crop_regression(self):
        crop = h.unit(3, noise=0.05)
        return (np.stack([crop, crop]), ["c-3", "c-3"]), crop

    def test_a_cards_only_night_whose_new_row_steals_a_regression_crop_does_not_ship(self):
        self.spy_gate()
        regression, crop = self.crop_regression()
        extra = self.extra_result([crop])
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(log):
            _, m = self.go(root, out, regression=regression, new_sets_fn=lambda: extra, live={})
        self.assertIs(m["improved"], False)
        self.assertEqual(m["regression"], {"before": 1.0, "after": 0.0, "n": 2})
        self.assertIn("catalogExtra" , self.seen["candidate"])
        self.assertIn("new-card rows are the first hit on 0 of 2 regression rows before and 2 after; correct reads lost 2, gained 0", log.getvalue())

    def test_a_cards_only_night_with_a_harmless_new_row_ships(self):
        self.spy_gate()
        regression, _ = self.crop_regression()
        extra = self.extra_result([h.unit(100)])
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(log):
            _, m = self.go(root, out, regression=regression, new_sets_fn=lambda: extra, live={})
        self.assertIs(m["improved"], True)
        self.assertIn("first hit on 0 of 2 regression rows before and 0 after; correct reads lost 0, gained 0", log.getvalue())

    def test_the_live_bundles_rows_are_in_the_before_replay(self):
        self.spy_gate()
        regression, crop = self.crop_regression()
        mine = self.extra_result([crop])
        harmless = self.extra_result([h.unit(100)])
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, regression=regression, new_sets_fn=lambda: harmless, live={"catalogExtra": mine["bytes"]})
        self.assertEqual(self.seen["live"]["catalogExtra"], mine["bytes"])
        self.assertEqual(m["regression"], {"before": 0.0, "after": 1.0, "n": 2})

    def test_a_night_where_memory_changed_is_judged_with_the_extra_rows_present(self):
        self.spy_gate()
        regression, crop = self.crop_regression()
        extra = self.extra_result([crop])
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            self.build(root)
            _, m = self.go(root, out, regression=regression, new_sets_fn=lambda: extra, live={})
        self.assertIn("catalogExtra", self.seen["candidate"])
        self.assertEqual(m["regression"]["after"], 0.0)
        self.assertIs(m["improved"], False)

    # --- no silent shrink --------------------------------------------------------------------------------------------
    def test_a_candidate_with_fewer_cards_than_the_live_bundle_does_not_ship(self):
        self.only_cards_gate()
        live = self.extra_result([h.unit(100), h.unit(101), h.unit(102)], ("sv9-1", "sv9-2", "sv9-3"))
        small = self.extra_result([h.unit(100)])
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(log):
            _, m = self.go(root, out, new_sets_fn=lambda: small, live={"catalogExtra": live["bytes"]})
        self.assertIs(m["improved"], False)
        self.assertIn("not shipping: the new-card file has 1 cards, the live one 3", log.getvalue())

    def test_an_unreadable_live_file_with_matching_tags_does_not_ship(self):
        self.only_cards_gate()
        small = self.extra_result([h.unit(100)])
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(log):
            _, m = self.go(root, out, new_sets_fn=lambda: small, live={"catalogExtra": b"NOT A FILE"})
        self.assertIs(m["improved"], False)
        self.assertIn("not shipping: the live catalogExtra is unreadable", log.getvalue())

    def test_rows_dropped_on_purpose_are_not_a_shrink_but_an_unexplained_loss_still_is(self):
        self.only_cards_gate()
        live = self.extra_result([h.unit(100), h.unit(101), h.unit(102)], ("sv9-1", "sv9-2", "sv9-3"))
        two = self.extra_result([h.unit(100), h.unit(101)], ("sv9-1", "sv9-2"))
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=lambda: {**two, "dropped": 1}, live={"catalogExtra": live["bytes"]})
            self.assertIs(m["improved"], True)
            _, m = self.go(root, out, new_sets_fn=lambda: {**two, "dropped": 0}, live={"catalogExtra": live["bytes"]})
            self.assertIs(m["improved"], False)
            _, m = self.go(root, out, new_sets_fn=lambda: {**two, "dropped": 1}, live={"catalogExtra": b"NOT A FILE"})
            self.assertIs(m["improved"], False)  # an unreadable file still blocks

    def test_build_reports_the_rows_it_dropped_so_the_guard_can_allow_them(self):
        from pipeline import new_sets

        card = lambda n, cid: {"id": cid, "name": f"C{n}", "set": "sv9", "setName": "S", "number": str(n), "image": "https://assets.tcgdex.net/x"}  # noqa: E731
        cards = [card(1, "sv9-1"), card(2, "sv9-2")]
        previous = new_sets.write_extra(cards, np.rint(np.stack([h.unit(100), h.unit(101)]) * 127).astype(np.int8))
        out = new_sets.build({"sv9-2"}, previous, lambda *a, **k: h.unit(5), fetch=lambda url: [], sleep=lambda s: None)
        self.assertEqual((out["total"], out["dropped"]), (1, 1))
        self.assertTrue(self._shrinks_ok(previous, out))

    @staticmethod
    def _shrinks_ok(live_bytes, out):
        from pipeline.run import _shrinks

        return not _shrinks(live_bytes, out["bytes"], out)

    def test_a_cards_only_night_with_an_exam_drop_does_not_ship_and_does_not_raise(self):
        def gate_with(n):
            return types.SimpleNamespace(evaluate=lambda candidate, live, exam, regression, catalog: {
                "improved": False, "regression": {"before": 0.6, "after": 0.6, "n": 2}, "exam": {"before": 0.9, "after": 0.8, "n": n},
                "wrongHigh": {"before": 0.0, "after": 0.0}})

        extra = lambda: {"bytes": b"EXTRA", "added": ["sv9-1"], "sets": ["sv9"]}  # noqa: E731
        for n, ships in ((30, False), (29, True)):
            self.only_cards_gate()
            self.modules["gate"] = gate_with(n)
            with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
                _, m = self.go(root, out, new_sets_fn=extra, live={})
                self.assertIs(m["improved"], ships, n)

    def test_the_exam_minimum_is_the_same_number_in_the_gate_the_manifest_and_the_signer(self):
        import importlib.util

        from pipeline import gate, manifest

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sign_release.py")
        spec = importlib.util.spec_from_file_location("sign_release_pin", path)
        signer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(signer)
        self.assertEqual(gate.MIN_EXAM_CASES, manifest.MIN_EXAM_CASES)
        self.assertEqual(signer.MIN_EXAM_CASES, manifest.MIN_EXAM_CASES)

    def test_a_candidate_with_as_many_cards_as_live_ships_and_a_tag_change_rebuild_is_exempt(self):
        self.only_cards_gate()
        live = self.extra_result([h.unit(100), h.unit(101)], ("sv9-1", "sv9-2"))
        same = self.extra_result([h.unit(100), h.unit(101), h.unit(102)], ("sv9-1", "sv9-2", "sv9-3"))
        small = self.extra_result([h.unit(100)])
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            _, m = self.go(root, out, new_sets_fn=lambda: same, live={"catalogExtra": live["bytes"]})
            self.assertIs(m["improved"], True)
            _, m = self.go(root, out, new_sets_fn=lambda: small, live={})  # live ignored: model or crop changed
            self.assertIs(m["improved"], True)

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


MODEL = hashlib.sha256(b"clip").hexdigest()[:16]
CROP = 3


class Tags(unittest.TestCase):
    """Live bundle, regression set and seed made with another model are ignored; main passes the tag to new_sets."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        assets = os.path.join(self.tmp, "assets")
        os.makedirs(assets)
        for name, data in (("clip_visual_int8.onnx", b"clip"), ("catalog.bin.gz", b"cat"), ("id_map.json", b"{}")):
            with open(os.path.join(assets, name), "wb") as f:
                f.write(data)
        with open(os.path.join(self.tmp, "pins.json"), "w") as f:
            json.dump({"cropVersion": CROP, "assets": [{"name": n, "sha256": hashlib.sha256(d).hexdigest()}
                                                      for n, d in (("clip_visual_int8.onnx", b"clip"), ("catalog.bin.gz", b"cat"), ("id_map.json", b"{}"))]}, f)
        self.live = os.path.join(self.tmp, "live")
        os.makedirs(self.live)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def live_file(self, name, data):
        with open(os.path.join(self.live, name), "wb") as f:
            f.write(data if isinstance(data, bytes) else json.dumps(data).encode())

    def go(self, live_dir=None, **argv):
        """Runs main with the heavy parts stubbed. Returns (what run_pipeline got, what new_sets.build got, the log)."""
        import pipeline.new_sets
        import pipeline.run as run_module

        seen, built = {}, {}

        def fake_pipeline(**kw):
            seen.update(kw)
            return {}

        def fake_build(*a, **kw):
            built.update(kw)
            built["live_extra"] = a[1]
            return {"bytes": None, "added": [], "sets": []}

        args = ["--assets", self.path("assets"), "--pins", self.path("pins.json"), "--out", self.path("out"), "--base-url", "https://example.invalid/",
                "--seq", "1", "--data", self.tmp, "--live-dir", live_dir or self.live, "--regression", self.path("regression.npz"),
                "--seed", self.path("seed.npz"), "--status-history", self.path("none")]
        log = io.StringIO()
        with unittest.mock.patch.object(run_module, "ClipEmbedder", lambda p: FakeEmbedder()), \
                unittest.mock.patch.object(Catalog, "load", return_value=catalog()), \
                unittest.mock.patch.object(run_module, "run_pipeline", fake_pipeline), \
                unittest.mock.patch.object(pipeline.new_sets, "build", fake_build), \
                contextlib.redirect_stdout(log):
            run_module.main(args)
            seen["new_sets_fn"]()
        return seen, built, log.getvalue()

    def test_a_live_bundle_of_another_model_or_crop_is_ignored(self):
        for model, crop in (("1111111111111111", CROP), (MODEL, CROP + 1)):
            self.live_file("manifest.json", {"model": model, "crop": crop})
            self.live_file("adapter.bin", b"OLD")
            self.live_file("catalog-extra.bin", b"OLDEXTRA")
            seen, built, log = self.go()
            self.assertEqual(seen["live"], {})
            self.assertIsNone(built["live_extra"])
            self.assertIn(f"live bundle is model {model} crop {crop}, not ours: starting from the plain catalog", log)

    def test_a_live_bundle_without_a_readable_manifest_is_ignored(self):
        self.live_file("adapter.bin", b"OLD")
        seen, _, log = self.go()
        self.assertEqual(seen["live"], {})
        self.assertIn("no readable manifest", log)
        self.live_file("manifest.json", b"{not json")
        seen, _, log = self.go()
        self.assertEqual(seen["live"], {})
        self.assertIn("no readable manifest", log)

    def test_a_matching_live_bundle_is_used(self):
        self.live_file("manifest.json", {"model": MODEL, "crop": CROP})
        self.live_file("adapter.bin", b"OLD")
        self.live_file("catalog-extra.bin", b"EXTRA")
        seen, built, log = self.go()
        self.assertEqual(seen["live"], {"adapter": b"OLD", "catalogExtra": b"EXTRA"})
        self.assertEqual(built["live_extra"], b"EXTRA")
        self.assertNotIn("not ours", log)

    def test_a_missing_live_folder_is_quiet(self):
        seen, _, log = self.go(live_dir=self.path("nowhere"))
        self.assertEqual(seen["live"], {})
        self.assertNotIn("live bundle", log)

    def test_new_sets_gets_the_model_and_crop_tag(self):
        _, built, _ = self.go()
        self.assertEqual(built["tag"], {"model": MODEL, "crop": CROP})

    def test_regression_and_seed_made_with_another_model_or_untagged_are_ignored_and_named(self):
        emb = np.stack([h.unit(3)])
        for kw in ({}, {"model": "1111111111111111", "crop": CROP}, {"model": MODEL, "crop": CROP + 1}):
            save_regression(self.path("regression.npz"), emb, ["c-3"], **kw)
            np.savez_compressed(self.path("seed.npz"), embeddings=emb, cardIds=np.array(["c-3"]), **{k: np.array(v) for k, v in kw.items()})
            seen, _, log = self.go()
            self.assertIsNone(seen["regression"], kw)
            self.assertEqual(seen["seed_pairs"], [], kw)
            self.assertIn("regression.npz was made with another model (or is untagged): ignored", log)
            self.assertIn("seed.npz was made with another model (or is untagged): ignored", log)

    def test_a_matching_regression_set_and_seed_are_used(self):
        emb = np.stack([h.unit(3)])
        save_regression(self.path("regression.npz"), emb, ["c-3"], model=MODEL, crop=CROP)
        np.savez_compressed(self.path("seed.npz"), embeddings=emb, cardIds=np.array(["c-3"]), model=np.array(MODEL), crop=np.array(CROP))
        seen, _, log = self.go()
        self.assertEqual(seen["regression"][1], ["c-3"])
        self.assertEqual([c for _, c in seen["seed_pairs"]], ["c-3"])
        self.assertNotIn("ignored", log)

    def test_missing_regression_and_seed_files_behave_as_before(self):
        seen, _, log = self.go()
        self.assertEqual((seen["regression"], seen["seed_pairs"]), (None, []))
        self.assertNotIn("ignored", log)

    def test_model_tag_ok_never_raises_and_wants_a_string_model_and_an_integer_crop(self):
        path = self.path("t.npz")
        with open(path, "wb") as f:
            f.write(b"this is not an npz file")
        bad = [
            dict(model=np.array(MODEL, dtype=object), crop=np.array(CROP)),  # a pickled object array
            dict(model=np.array(MODEL), crop=np.array([CROP, CROP])),  # more than one element
            dict(model=np.array(MODEL), crop=np.array(3.7)),
            dict(model=np.array(MODEL), crop=np.array(float(CROP))),
            dict(model=np.array(MODEL), crop=np.array(str(CROP))),
            dict(model=np.array(5), crop=np.array(CROP)),
            dict(model=np.array([MODEL, MODEL]), crop=np.array(CROP)),
        ]
        for corrupt in (True, False):
            for arrays in bad if not corrupt else [None]:
                if arrays is not None:
                    np.savez(path, **arrays)
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    self.assertFalse(model_tag_ok(path, MODEL, CROP), arrays)
                self.assertIn("was made with another model (or is untagged): ignored", log.getvalue())
        np.savez(path, model=np.array(MODEL), crop=np.array(CROP))
        self.assertTrue(model_tag_ok(path, MODEL, CROP))

    def test_model_tag_ok_needs_both_keys_equal(self):
        path = self.path("t.npz")
        np.savez(path, embeddings=np.zeros((0, 512), dtype=np.float32))
        self.assertFalse(model_tag_ok(path, MODEL, CROP))
        np.savez(path, model=np.array(MODEL))
        self.assertFalse(model_tag_ok(path, MODEL, CROP))
        np.savez(path, model=np.array(MODEL), crop=np.array(CROP))
        self.assertTrue(model_tag_ok(path, MODEL, CROP))
        self.assertFalse(model_tag_ok(path, "2" * 16, CROP))


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
