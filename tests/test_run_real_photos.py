import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
IDS = ["base1-1", "base1-2", "base1-3", "base1-4", "base1-5", "base1-6"]


def _path(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMG_DIR, card_id + ext)
        if os.path.exists(p):
            return p
    return None


def test_synthetic_photo_boxes_and_summary_keys():
    import run_real_photos as R
    from detect_cards import detect_cards

    paths = [p for p in map(_path, IDS) if p][:4]
    if len(paths) < 4:
        pytest.skip("catalog images missing")
    canvas = Image.new("RGB", (960, 700), (128, 128, 128))
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            card = im.convert("RGB").resize((200, 280))
        canvas.paste(card, (40 + (k % 2) * 300, 60 + (k // 2) * 320))
    assert len(detect_cards(canvas)) >= 3

    photos = os.path.join(ROOT, ".pytest_cache", "real_photos_test")
    out = os.path.join(photos, "out")
    os.makedirs(photos, exist_ok=True)
    canvas.save(os.path.join(photos, "synthetic.jpg"), quality=92)
    summary = R.main(["--photos", photos, "--out", out])
    assert {"photos", "boxes", "buckets", "hubs"} <= set(summary)
    assert set(summary["hubs"]) == {"<150", ">=150"}
    assert summary["boxes"] >= 3
    assert os.path.exists(os.path.join(out, "results.json"))
    assert os.path.exists(os.path.join(out, "sheets", "sheet-1.jpg"))
