import json
import struct
import unittest

import numpy as np

import helpers as h
from pipeline.pack import SCALE, read_pack, write_pack


class PackFormat(unittest.TestCase):
    def test_round_trip(self):
        ids = ["base1-4", "base1-5", "sv1-100"]
        vecs = np.stack([h.unit(0), h.unit(7, 0.01, 1), h.unit(100, 0.01, 2)])
        data = write_pack(ids, vecs)
        pack = read_pack(data)
        self.assertEqual(pack.card_ids, ids)
        self.assertEqual(pack.vectors.shape, (3, 512))
        self.assertEqual(pack.vectors.dtype, np.int8)
        back = pack.dequantized()
        self.assertGreater(float((back * vecs).sum(axis=1).min()), 0.995)
        np.testing.assert_allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-5)

    def test_byte_layout(self):
        data = write_pack(["a-1"], np.stack([h.unit(3)]))
        (n,) = struct.unpack("<I", data[:4])
        header = json.loads(data[4 : 4 + n].decode("utf-8"))
        self.assertEqual(header, {"v": 1, "dim": 512, "entries": 1, "cardIds": ["a-1"]})
        self.assertEqual(SCALE, 127)
        body = data[4 + n :]
        self.assertEqual(len(body), 512)
        self.assertEqual(np.frombuffer(body, dtype=np.int8)[3], 127)

    def test_quantisation_clips_to_the_scale(self):
        v = np.zeros((1, 512), dtype=np.float32)
        v[0, 0], v[0, 1] = 1.0, -1.0
        body = np.frombuffer(write_pack(["a"], v)[-512:], dtype=np.int8)
        self.assertEqual((body[0], body[1]), (127, -127))

    def test_matches_the_fixture_the_extension_test_parses(self):
        import os

        fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        rng = np.random.RandomState(7)
        vecs = rng.randn(3, 512).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        data = write_pack(["base1-4", "sv1-100", "xy1-7"], vecs)
        os.makedirs(fixtures, exist_ok=True)
        path = os.path.join(fixtures, "pack-v1.bin")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_empty_pack_round_trips(self):
        pack = read_pack(write_pack([], np.zeros((0, 512), dtype=np.float32)))
        self.assertEqual(pack.card_ids, [])
        self.assertEqual(pack.vectors.shape, (0, 512))
        self.assertEqual(pack.dequantized().shape, (0, 512))

    def test_output_is_deterministic(self):
        vecs = np.stack([h.unit(1), h.unit(2)])
        self.assertEqual(write_pack(["a", "b"], vecs), write_pack(["a", "b"], vecs))

    def test_corrupt_packs_are_rejected(self):
        good = write_pack(["a-1"], np.stack([h.unit(3)]))
        for bad in (good[:-1], good + b"\0", good[:3], b"\xff\xff\xff\x7f" + good[4:]):
            with self.assertRaises(ValueError):
                read_pack(bad)

    def test_mismatched_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            write_pack(["a", "b"], np.stack([h.unit(1)]))


if __name__ == "__main__":
    unittest.main()
