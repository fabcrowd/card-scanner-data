import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
CARD_ID = "base1-4"


def _catalog_path(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMG_DIR, card_id + ext)
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="module")
def pl():
    import pseudo_label as P

    P.init_ctx(threads_one=False)
    return P


def _synthetic_photo():
    p = _catalog_path(CARD_ID)
    if p is None:
        pytest.skip("catalog image missing")
    with Image.open(p) as im:
        card = np.asarray(im.convert("RGB").resize((286, 400), Image.BILINEAR))
    canvas = np.full((900, 1100, 3), 128, np.uint8)
    src = np.float32([[0, 0], [286, 0], [286, 400], [0, 400]])
    dst = np.float32([[300, 200], [590, 212], [584, 618], [296, 604]])  # slight perspective, ~405 px tall
    warp = cv2.warpPerspective(card, cv2.getPerspectiveTransform(src, dst), (1100, 900))
    mask = cv2.warpPerspective(np.full(card.shape[:2], 255, np.uint8), cv2.getPerspectiveTransform(src, dst), (1100, 900))
    canvas[mask > 0] = warp[mask > 0]
    return Image.fromarray(canvas)


def test_synthetic_card_is_accepted_with_right_id(pl):
    img = _synthetic_photo()
    rows = pl.process_image(img, "synthetic")
    assert rows, "detector found no box >= 300 px"
    hit = [r for r in rows if r["accepted"]]
    assert hit, rows
    assert hit[0]["card_id"] == CARD_ID or (hit[0]["name_only"] and "charizard" in hit[0]["name"].lower())
    assert hit[0]["inliers_best"] >= 40


def test_texture_box_is_rejected(pl):
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (900, 1100, 3), dtype=np.uint8), (0, 0), 6)
    row = pl.label_box(Image.fromarray(tex), [300, 200, 590, 610])
    assert not row["accepted"]
    assert row["card_id"] is None
