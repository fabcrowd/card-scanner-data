import json
import struct
import unittest

import _paths  # noqa: F401
import numpy as np

from pipeline.train_check import parse_check, score_check, train

DIM = 512


def cluster(center_seed, n, seed, noise=0.4):
    c = np.random.RandomState(center_seed).randn(DIM).astype(np.float32)
    c /= np.linalg.norm(c)
    x = c + np.random.RandomState(seed).randn(n, DIM).astype(np.float32) * noise / np.sqrt(DIM)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TrainCheck(unittest.TestCase):
    def setUp(self):
        self.pos = cluster(1, 300, 10)
        self.neg = cluster(2, 120, 11)

    def test_format_matches_check_bin(self):
        data = train(self.pos, self.neg)
        (h,) = struct.unpack("<I", data[:4])
        head = json.loads(data[4 : 4 + h])
        self.assertEqual((head["v"], head["dim"]), (1, 512))
        self.assertIn("bias", head)
        self.assertIn("threshold", head)
        self.assertEqual(len(data), 4 + h + 512 * 4)

    def test_separates_cards_from_non_cards(self):
        data = train(self.pos, self.neg)
        pos, neg = cluster(1, 200, 20), cluster(2, 200, 21)
        self.assertGreaterEqual(np.mean(score_check(data, pos) >= parse_check(data)["threshold"]), 0.97)
        self.assertLessEqual(np.mean(score_check(data, neg) >= parse_check(data)["threshold"]), 0.2)

    def test_none_with_fewer_than_50_negatives(self):
        self.assertIsNone(train(self.pos, self.neg[:49]))
        self.assertIsNotNone(train(self.pos, self.neg[:50]))

    def test_none_with_too_few_positives(self):
        self.assertIsNone(train(self.pos[:5], self.neg))

    def test_deterministic(self):
        self.assertEqual(train(self.pos, self.neg), train(self.pos, self.neg))

    def test_parse_rejects_bad_files(self):
        with self.assertRaises(ValueError):
            parse_check(b"\x01")
        with self.assertRaises(ValueError):
            parse_check(train(self.pos, self.neg)[:-4])


if __name__ == "__main__":
    unittest.main()
