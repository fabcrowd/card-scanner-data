import unittest

import _paths  # noqa: F401
import numpy as np
from PIL import Image

from pipeline.augment import photo_like


def card_image():
    rng = np.random.RandomState(3)
    arr = (rng.rand(140, 100, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def raw(img):
    return np.asarray(img, dtype=np.uint8).tobytes()


class PhotoLike(unittest.TestCase):
    def test_same_seed_gives_identical_pixels(self):
        self.assertEqual(raw(photo_like(card_image(), seed=5)), raw(photo_like(card_image(), seed=5)))

    def test_different_seeds_differ(self):
        self.assertNotEqual(raw(photo_like(card_image(), seed=1)), raw(photo_like(card_image(), seed=2)))

    def test_output_is_changed_rgb_of_the_same_size(self):
        src = card_image()
        out = photo_like(src, seed=0)
        self.assertEqual((out.mode, out.size), ("RGB", src.size))
        self.assertNotEqual(raw(out), raw(src))

    def test_input_is_not_modified(self):
        src = card_image()
        before = raw(src)
        photo_like(src, seed=0)
        self.assertEqual(raw(src), before)

    def test_accepts_non_rgb_input(self):
        self.assertEqual(photo_like(card_image().convert("RGBA"), seed=0).mode, "RGB")

    def test_default_seed_is_deterministic(self):
        self.assertEqual(raw(photo_like(card_image())), raw(photo_like(card_image())))


if __name__ == "__main__":
    unittest.main()
