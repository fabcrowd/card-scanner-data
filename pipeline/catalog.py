"""The extension's card catalog: L2-normalised float32 rows (catalog.bin.gz) and the card id of each row (id_map.json)."""
import gzip
import json

import numpy as np

DIM = 512


class Catalog:
    def __init__(self, vectors, ids):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != DIM or vectors.shape[0] != len(ids):
            raise ValueError("catalog rows and ids do not line up")
        self.vectors = vectors
        self.ids = list(ids)

    @classmethod
    def load(cls, bin_gz_path, id_map_path):
        with gzip.open(bin_gz_path, "rb") as f:
            raw = f.read()
        with open(id_map_path, encoding="utf-8") as f:
            ids = json.load(f)
        return cls(np.frombuffer(raw, dtype=np.float32).reshape(-1, DIM), ids)

    def top_k_batch(self, queries, k=20, chunk=256):
        """(indices, scores) of the k best rows per query, best first; both shaped (len(queries), min(k, rows))."""
        queries = np.asarray(queries, dtype=np.float32).reshape(-1, DIM)
        k = min(k, len(self.ids))
        idx = np.zeros((len(queries), k), dtype=np.int64)
        score = np.zeros((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            sims = queries[start : start + chunk] @ self.vectors.T
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.tile(np.arange(sims.shape[1]), (len(sims), 1))
            part_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            idx[start : start + chunk] = np.take_along_axis(part, order, axis=1)
            score[start : start + chunk] = np.take_along_axis(part_scores, order, axis=1)
        return idx, score
