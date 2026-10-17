import io
import tempfile
import unittest

import _paths  # noqa: F401
import numpy as np
from PIL import Image

from pipeline.seed import ForbiddenSource, RealCrop, Sources, build_seed, label_title


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (100, 140), (seed % 256, (seed * 7) % 256, 90)).save(buf, "JPEG")
    return buf.getvalue()


def fake_embedder(data, trusted=False):
    v = np.zeros(512, dtype=np.float32)
    v[sum(data[:50]) % 512] = 1.0
    return v


def png_path(tmp, name, seed):
    p = f"{tmp}/{name}.png"
    Image.new("RGB", (100, 140), (seed, 50, 90)).save(p)
    return p


def catalog_images(tmp, n):
    return [(png_path(tmp, f"c{i}", i * 20), f"c-{i}") for i in range(n)]


class BuildSeed(unittest.TestCase):
    def test_real_crops_and_synthetic_variants_are_returned_with_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Sources(embedder=fake_embedder, real=[RealCrop(jpeg(3), "c-1", "variant:1/a.jpg")],
                          catalog_images=catalog_images(tmp, 5), synthetic_per_card=2)
            out = build_seed(src)
            self.assertEqual(len(out), 1 + 5 * 2)
            self.assertTrue(all(e.dtype == np.float32 and e.shape == (512,) for e, _ in out))
            self.assertEqual([i for _, i in out].count("c-1"), 3)

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            def make():
                return build_seed(Sources(embedder=fake_embedder, real=[], catalog_images=catalog_images(tmp, 4), synthetic_per_card=2))
            a, b = make(), make()
            self.assertTrue(all(np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b)))

    def test_max_synthetic_caps_catalog_cards_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = build_seed(Sources(embedder=fake_embedder, real=[], catalog_images=catalog_images(tmp, 10), synthetic_per_card=1, max_synthetic=3))
            self.assertEqual(len(out), 3)

    def test_hibid_sealed_and_realworld_origins_are_refused(self):
        origins = ("hibid:318/00.jpg", r"C:\r\data\detect-hibid-real\3\00.jpg", "data/sealed-test/x.jpg", "data/realworld/fresh-pages.json")
        for origin in origins:
            with self.assertRaises(ForbiddenSource, msg=origin):
                build_seed(Sources(embedder=fake_embedder, real=[RealCrop(jpeg(1), "c-1", origin)], catalog_images=[]))

    def test_real_crop_that_cannot_decode_is_skipped(self):
        def flaky(data, trusted=False):
            raise ValueError("bad")

        out = build_seed(Sources(embedder=flaky, real=[RealCrop(b"x", "c-1", "variant:1/a.jpg")], catalog_images=[]))
        self.assertEqual(out, [])


META = [
    {"id": "neo1-80", "name": "Totodile", "set": "Neo Genesis", "number": "80"},
    {"id": "base1-46", "name": "Totodile", "set": "Base Set", "number": "46"},
    {"id": "sv1-80", "name": "Pikachu ex", "set": "Scarlet & Violet", "number": "80"},
    {"id": "sv2-80", "name": "Pikachu ex", "set": "Paldea Evolved", "number": "80"},
    {"id": "A1-80", "name": "Totodile", "set": "Genetic Apex", "number": "80"},
]


class LabelTitle(unittest.TestCase):
    def test_name_and_number_pick_one_physical_card(self):
        self.assertEqual(label_title("Pokemon Totodile 1st Edition Neo Genesis 80/111 Holo", META), "neo1-80")

    def test_ambiguous_title_is_dropped(self):
        self.assertIsNone(label_title("Pikachu ex 80/198 holo", META))

    def test_set_name_breaks_a_tie(self):
        self.assertEqual(label_title("Pikachu ex 80/193 Paldea Evolved", META), "sv2-80")

    def test_no_number_or_no_name_is_dropped(self):
        self.assertIsNone(label_title("Totodile holo lot", META))
        self.assertIsNone(label_title("Mystery card 80/111", META))


if __name__ == "__main__":
    unittest.main()
