import io
import unittest

import numpy as np
from PIL import Image

import _paths  # noqa: F401
from pipeline import embed as embed_module
from pipeline.embed import art_rect, preprocess


class ArtCrop(unittest.TestCase):
    def test_matches_lib_embed_js_art_source_rect(self):
        # 750x1050: strip = floor(1050 * 0.45) = 472, a centred 472-px square of that strip.
        self.assertEqual(art_rect(750, 1050), (139, 0, 472))
        # Narrow card: side limited by width, centred vertically inside the strip.
        self.assertEqual(art_rect(100, 400), (0, 40, 100))

    def test_tiny_images_keep_a_one_pixel_strip(self):
        x, y, side = art_rect(10, 1)
        self.assertEqual((y, side), (0, 1))
        self.assertGreaterEqual(x, 0)

    def test_preprocess_gives_a_normalised_chw_float_tensor(self):
        img = Image.new("RGB", (200, 280), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        x = preprocess(buf.getvalue())
        self.assertEqual(x.shape, (1, 3, 224, 224))
        self.assertEqual(x.dtype, np.float32)
        # white pixel, channel 0: (1 - 0.48145466) / 0.26862954
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), (1 - 0.48145466) / 0.26862954, places=1)

    def test_only_the_top_strip_matters(self):
        top = Image.new("RGB", (200, 280), (10, 200, 30))
        a, b = io.BytesIO(), io.BytesIO()
        top.save(a, "PNG")
        bottom_changed = top.copy()
        bottom_changed.paste((250, 0, 0), (0, 150, 200, 280))  # below the 45% strip (126 px)
        bottom_changed.save(b, "PNG")
        np.testing.assert_array_equal(preprocess(a.getvalue()), preprocess(b.getvalue()))

    def test_undecodable_bytes_raise(self):
        with self.assertRaises(ValueError):
            preprocess(b"not an image")


def jpeg(size, color=(10, 200, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class HostileImages(unittest.TestCase):
    def test_pixel_limit_is_set_for_the_whole_process(self):
        self.assertEqual(Image.MAX_IMAGE_PIXELS, 1_000_000)
        self.assertEqual(embed_module.MAX_SIDE, 400)

    def test_sizes_outside_the_relay_limits_are_skipped(self):
        for size in ((401, 140), (100, 401), (63, 63), (30, 20)):
            with self.assertRaises(ValueError, msg=str(size)):
                preprocess(jpeg(size))
        preprocess(jpeg((400, 400)))
        preprocess(jpeg((64, 10)))

    def test_a_header_that_claims_a_huge_image_is_refused_without_decoding_it(self):
        data = bytearray(jpeg((100, 140)))
        at = data.index(bytes([0xFF, 0xC0]))
        data[at + 5 : at + 9] = (60000).to_bytes(2, "big") * 2  # height, width
        with self.assertRaises(ValueError):
            preprocess(bytes(data))

    def test_trusted_images_such_as_the_regression_set_may_be_full_size(self):
        self.assertEqual(preprocess(jpeg((750, 1050)), trusted=True).shape, (1, 3, 224, 224))
        with self.assertRaises(ValueError):
            preprocess(jpeg((750, 1050)))


if __name__ == "__main__":
    unittest.main()
