"""Community pack byte format (v1), as read by tools/extension-wasm/lib/community-pack.js.

    bytes 0..3        little-endian uint32 H, the length of the header
    bytes 4..4+H      UTF-8 JSON {"v":1,"dim":512,"entries":N,"cardIds":[N strings]}
    then              N * 512 int8, row-major; row i belongs to cardIds[i]

Each value is round(127 * x) of the L2-normalised float component, clamped to -127..127; there is no scale
field. The file is exactly 4 + H + N * 512 bytes. Readers divide by 127 and should L2-normalise the row.
"""
import json
import struct
from dataclasses import dataclass

import numpy as np

SCALE = 127
DIM = 512
VERSION = 1
MAX_HEADER = 64 * 1024 * 1024


@dataclass
class Pack:
    card_ids: list
    vectors: np.ndarray  # int8, shape (entries, dim)

    def dequantized(self):
        """L2-normalised float32 rows."""
        out = self.vectors.astype(np.float32) / SCALE
        if len(out):
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


def write_pack(card_ids, embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM)
    if len(card_ids) != len(embeddings):
        raise ValueError("one card id per row is required")
    q = np.clip(np.rint(embeddings * SCALE), -SCALE, SCALE).astype(np.int8)
    header = json.dumps(
        {"v": VERSION, "dim": DIM, "entries": len(card_ids), "cardIds": list(card_ids)},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return struct.pack("<I", len(header)) + header + q.tobytes()


def read_pack(data):
    if len(data) < 4:
        raise ValueError("pack too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise ValueError("bad header length")
    try:
        header = json.loads(data[4 : 4 + n].decode("utf-8"))
    except ValueError as err:
        raise ValueError("bad header") from err
    if header.get("v") != VERSION or header.get("dim") != DIM:
        raise ValueError("unsupported pack")
    ids = header.get("cardIds")
    if not isinstance(ids, list) or header.get("entries") != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("bad card ids")
    body = data[4 + n :]
    if len(body) != len(ids) * DIM:
        raise ValueError("body size does not match entries")
    return Pack(ids, np.frombuffer(body, dtype=np.int8).reshape(len(ids), DIM))
