"""Unit tests for train/gpu_data2.py (fastC fixes + real crops); CPU always, CUDA when present."""
import os
import sys
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import gpu_data as G  # noqa: E402
import gpu_data2 as G2  # noqa: E402
from test_gpu_data import smooth_cards, cos_rows  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_height_mixture_fractions():
    h = G2.sample_heights(np.random.default_rng(1), 400_000)
    assert h.min() >= 48 and h.max() <= 320
    # 50% in 70-180 + 25% * ln(180/70)/ln(320/48) of the wide component = 0.6245
    assert abs(np.mean((h >= 70) & (h <= 180)) - 0.6245) < 0.01
    # 25% high component + 25% * ln(320/250)/ln(320/48) = 0.2825
    assert abs(np.mean(h >= 250) - 0.2825) < 0.01
    assert abs(np.mean(h < 70) - 0.25 * np.log(70 / 48) / np.log(320 / 48)) < 0.01


def test_quality_bins_uniform_and_in_range():
    q = G2.sample_qualities(np.random.default_rng(2), 200_000)
    assert set(np.unique(q)) == set(range(40, 90, 5))
    frac = np.array([np.mean(q == b) for b in range(40, 90, 5)])
    assert np.abs(frac - 0.1).max() < 0.006, frac  # the old draw gave the 85 bin 1/46 of the mass


def test_blur_sigma_in_320_units():
    p = G2.sample_degradation(np.random.default_rng(3), 100_000)
    s = p["sigmas"]
    assert abs(np.mean(s > 0) - 0.35) < 0.01
    assert s[s > 0].min() >= 0.12 and s.max() <= 0.47
    assert 0.3 * 320 / 825 > 0.115 and 1.2 * 320 / 825 < 0.47  # pilot strength expressed on a 320 px card


def test_nvjpeg_failure_raises_no_fallback():
    items = [torch.zeros(3, 16, 16, dtype=torch.uint8)]
    with mock.patch("torchvision.io.encode_jpeg", side_effect=RuntimeError("nvjpeg boom")):
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            G2.jpeg_roundtrip(items, 60, torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvjpeg_ok_on_cuda():
    dev = torch.device("cuda")
    items = [smooth_cards(1, i)[0, :, :40, :40].contiguous().to(dev) for i in range(3)]
    out = G2.jpeg_roundtrip(items, 60, dev)
    assert [o.shape for o in out] == [(3, 40, 40)] * 3


@pytest.mark.parametrize("device", DEVICES)
def test_real_crop_coupling(device):
    """A real crop as 'big': large view and degraded view derive from the same jittered crop."""
    dev = torch.device(device)
    real = smooth_cards(6, seed=5).to(dev)
    big = G.jitter(real, G.jitter_boxes(np.random.default_rng(3), 6))
    p = {"hs": np.full(6, 320), "modes": np.zeros(6, int), "sigmas": np.zeros(6), "quals": np.full(6, 85)}
    large, deg = G.art_view(big), G2.degrade(big, p, dev)
    same, other = cos_rows(large, deg), cos_rows(large, deg.roll(1, 0))
    assert (same > 0.97).all() and same.min() > other.max()
    l2, d2, hs, big2 = G2.make_batch(real, np.random.default_rng(4), dev)
    assert l2.shape == d2.shape == (6, 3, 224, 224) and big2.shape == (6, 3, 320, 229) and len(hs) == 6
    assert torch.isfinite(d2).all()


def test_ce_group_positives_are_all_same_name_rows():
    names = ["Pikachu", "pikachu!", "Raichu", None]
    grp, k2g = G2.name_groups(names)
    assert grp[0] == grp[1] != grp[2] and grp[3] < 0 and G.norm_name("Pikachu") in k2g
    catv = F.normalize(torch.randn(4, 8), dim=1)
    s = catv[:1].clone()
    gid = torch.tensor([k2g["pikachu"]])
    loss = G2.ce_group(s, gid, catv, grp, 0.04)
    logits = (s @ catv.T / 0.04)[0]
    want = torch.logsumexp(logits, 0) - torch.logsumexp(logits[:2], 0)  # rows 0,1 positive, no masked negatives
    assert torch.allclose(loss, want, atol=1e-5)


def test_leakage_filter_by_normalised_name():
    name_of = ["Pikachu V", "Raichu", "Mew", "Eevee"]
    _, k2g = G2.name_groups(name_of)
    ev_norm = {G.norm_name("PIKACHU-V")}
    names = ["Pikachu V", "Raichu", "Mew", "Unknown Card", "Eevee", None]
    row_names = ["Pikachu V", "Raichu", "pikachu v", "Unknown Card", "Eevee", None]
    keep, gid, rep = G2.filter_real(names, row_names, ev_norm, k2g)
    assert keep.tolist() == [False, True, False, False, True, False]  # leak, ok, leak via row name, no catalog row, ok, none
    assert rep == {"candidates": 6, "dropped_leakage": 2, "dropped_no_catalog_name": 2, "used": 2}
    assert gid.tolist() == [k2g["raichu"], k2g["eevee"]]


def test_label_names_id_uses_train_id_name_and_name_level_uses_label():
    name_of = ["Alpha", "Beta"]
    row_of = {"a": 0, "b": 1}
    rows = [{"level": "id", "train_id": "b", "name": "ignored"}, {"level": "name", "train_id": None, "name": "Alpha"}]
    assert G2.label_names(rows, name_of, row_of) == ["Beta", "Alpha"]


def test_cycle_covers_all_before_repeating():
    c = G2.Cycle(10, np.random.default_rng(0))
    a = np.concatenate([c.take(4), c.take(6)])
    assert sorted(a.tolist()) == list(range(10))
    assert len(c.take(25)) == 25
