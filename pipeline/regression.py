"""The fixed, server-owned regression set: embeddings and catalog ids of labelled crops, no images.

Contributors cannot touch it. The pack must not lower top-1 accuracy on it (see manifest.is_improved).
"""
import numpy as np

DIM = 512
UNIT_TOLERANCE = 1e-3


def save_regression(path, embeddings, card_ids):
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM)
    if len(embeddings) != len(card_ids):
        raise ValueError("one card id per embedding is required")
    np.savez_compressed(path, embeddings=embeddings, cardIds=np.array(list(card_ids), dtype="U64"))


def load_regression(path):
    """(embeddings float32 (n, 512), card ids). Raises ValueError for a file that is not a regression set."""
    with np.load(path, allow_pickle=False) as raw:
        if sorted(raw.files) != ["cardIds", "embeddings"]:
            raise ValueError("regression file must hold exactly embeddings and cardIds")
        embeddings = np.asarray(raw["embeddings"], dtype=np.float32)
        ids = [str(i) for i in raw["cardIds"]]
    if embeddings.ndim != 2 or embeddings.shape[1] != DIM or len(embeddings) != len(ids):
        raise ValueError("regression embeddings and ids do not line up")
    if len(embeddings) and not np.all(np.abs(np.linalg.norm(embeddings, axis=1) - 1) <= UNIT_TOLERANCE):
        raise ValueError("regression embeddings must be unit length")
    return embeddings, ids
