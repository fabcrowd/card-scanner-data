"""Unit tests for the distillation pilot data side (split, height sampler, view coupling)."""
import os
import random
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import pilot_data as D  # noqa: E402


def test_digital_rule():
    assert D.is_digital("A1-001") and D.is_digital("A2b-010") and D.is_digital("B1a-005") and D.is_digital("P-A-001")
    assert not D.is_digital("sv02-5") and not D.is_digital("base1-1") and not D.is_digital("swsh3.5-1")


def test_split_has_no_overlap_and_no_same_name():
    from pipeline.catalog import Catalog
    import score_synth_fusion as S

    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = S.load_names()
    name_of = [nm.get(c) for c in cat.ids]
    split, train, ev = D.build_split(cat, name_of)
    assert not set(train) & set(ev)
    ev_names = {name_of[r] for r in ev if name_of[r]}
    assert not any(name_of[r] in ev_names for r in train)
    assert not any(D.is_digital(cat.ids[r]) for r in train)
    assert len(ev) == 1000
    print("split counts", split["counts"])


def test_height_sampler_mixture():
    rng = random.Random(1)
    hs = np.array([D.sample_height(rng) for _ in range(20000)])
    focus = float(np.mean((hs >= 70) & (hs <= 180)))
    big = float(np.mean(hs >= 250))
    print(f"fraction in 70-180: {focus:.3f}; fraction >=250: {big:.3f}")
    assert hs.min() >= 48 and hs.max() <= 320
    assert 0.55 <= focus <= 0.85


def _card():
    arr = (np.random.RandomState(0).rand(825, 600, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_large_and_degraded_views_share_the_jittered_image(monkeypatch):
    seen = []
    real = D.degrade_view

    def spy(big, *a, **k):
        seen.append(big)
        return real(big, *a, **k)

    monkeypatch.setattr(D, "degrade_view", spy)
    large, deg, h, big = D.make_pair(_card(), random.Random(5))
    assert len(seen) == 1 and seen[0] is big
    # the teacher/student large view is exactly the 320 px area-resize of that same big image
    again = D.to_tensor(D.resize_to_height(big, D.LARGE_H, Image.BOX))
    assert torch.equal(large, again)
    assert large.shape == deg.shape == (3, 224, 224)


def test_jitter_keeps_card_aspect():
    img = _card()
    ar = img.width / img.height
    for s in range(50):
        b = D.jitter_box(img, random.Random(s))
        assert abs(b.width / b.height - ar) < 0.02


def test_epoch_batches_deterministic_and_disjoint():
    s = D.EpochBatches(list(range(100)), 32, 7001)
    a = [x for b in s for x in b]
    assert [x for b in s for x in b] == a
    assert len({r for r, _ in a}) == len(a) == 96
