import json
import struct
import unittest

import _paths  # noqa: F401
import numpy as np

from pipeline.catalog import Catalog
from pipeline.train_adapter import apply_adapter, parse_adapter
from pipeline.train_adapter import train as _train

DIM = 512


def unit_rows(n, seed):
    x = np.random.RandomState(seed).randn(n, DIM).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def world(n_cards=60, per_card=8, seed=0):
    """Cards live in a 24-d subspace; photo queries are a fixed rotation of the clean vector plus noise."""
    rng = np.random.RandomState(seed)
    basis = np.linalg.qr(rng.randn(DIM, 24))[0].astype(np.float32)  # (512, 24)
    cat = unit_rows(n_cards, seed + 1)[:, :24] @ basis.T
    cat /= np.linalg.norm(cat, axis=1, keepdims=True)
    skew = np.random.RandomState(99).randn(24, 24)
    rot = np.linalg.qr(np.eye(24) + 0.6 * (skew - skew.T))[0].astype(np.float32)
    ids = [f"c-{i}" for i in range(n_cards)]
    noise = np.random.RandomState(seed + 7)
    pairs = []
    for i in range(n_cards):
        for _ in range(per_card):
            q = (cat[i] @ basis) @ rot @ basis.T + noise.randn(DIM).astype(np.float32) * 0.02
            pairs.append((q / np.linalg.norm(q), ids[i]))
    return Catalog(cat.astype(np.float32), ids), pairs


def train(pairs, cat):
    """The toy world has few pairs, so it needs the longer, faster schedule the production defaults avoid."""
    return _train(pairs, cat, lr=1e-2, max_epochs=60)


def top1(cat, queries, truths):
    idx, _ = cat.top_k_batch(queries, 1)
    return np.mean([cat.ids[i[0]] == t for i, t in zip(idx, truths)])


class TrainAdapter(unittest.TestCase):
    def test_format_matches_adapter_bin(self):
        cat, pairs = world()
        data = train(pairs, cat)
        self.assertIsNotNone(data)
        (h,) = struct.unpack("<I", data[:4])
        head = json.loads(data[4 : 4 + h])
        self.assertEqual((head["v"], head["dim"], head["rank"]), (1, 512, 32))
        self.assertGreater(head["scale"], 0)
        self.assertEqual(len(data), 4 + h + 2 * 512 * 32)

    def test_adapter_beats_identity_on_fresh_queries(self):
        cat, pairs = world()
        adapter = parse_adapter(train(pairs, cat))
        _, fresh_pairs = world(per_card=3)
        q = np.stack([p[0] for p in fresh_pairs])
        truths = [p[1] for p in fresh_pairs]
        before = top1(cat, q, truths)
        after = top1(Catalog(apply_adapter(adapter, cat.vectors), cat.ids), apply_adapter(adapter, q), truths)
        self.assertGreater(after, before)

    def test_apply_keeps_rows_unit_length(self):
        cat, pairs = world(n_cards=30)
        adapter = parse_adapter(train(pairs, cat))
        out = apply_adapter(adapter, cat.vectors)
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-4))

    def test_returns_none_when_identity_is_already_perfect(self):
        cat = Catalog(unit_rows(40, 5), [f"c-{i}" for i in range(40)])
        pairs = [(cat.vectors[i], cat.ids[i]) for i in range(40) for _ in range(3)]
        self.assertIsNone(train(pairs, cat))

    def test_too_few_pairs_or_unknown_ids_return_none(self):
        cat, pairs = world(n_cards=10)
        self.assertIsNone(train(pairs[:5], cat))
        self.assertIsNone(train([(p[0], "nope") for p in pairs], cat))

    def test_is_deterministic(self):
        cat, pairs = world(n_cards=30, per_card=4)
        self.assertEqual(train(pairs, cat), train(pairs, cat))

    def test_parse_rejects_bad_files(self):
        with self.assertRaises(ValueError):
            parse_adapter(b"\x01\x00")
        cat, pairs = world(n_cards=30, per_card=4)
        with self.assertRaises(ValueError):
            parse_adapter(train(pairs, cat)[:-1])


if __name__ == "__main__":
    unittest.main()
