"""Unit tests for the G1 CDN-realistic degrader and the 50/50 mixture in train/gpu_data2.py (--degrade-mix cdn50)."""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import gpu_data as G  # noqa: E402
import gpu_data2 as G2  # noqa: E402
from test_gpu_data import smooth_cards, cos_rows  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N = 200_000


@pytest.fixture(scope="module")
def mix():
    return G2.sample_degradation_mix(np.random.default_rng(11), N, "cdn50")


def test_mix_none_is_the_unchanged_sampler():
    a = G2.sample_degradation_mix(np.random.default_rng(5), 1000, "none")
    b = G2.sample_degradation(np.random.default_rng(5), 1000)
    assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):
        G2.sample_degradation_mix(np.random.default_rng(5), 3, "bogus")


def test_height_distribution_unchanged_by_the_mix(mix):
    base = G2.sample_degradation(np.random.default_rng(11), N)
    assert np.array_equal(mix["hs"], base["hs"])  # same rng state -> identical heights, base draw is made first
    assert np.array_equal(mix["quals"], base["quals"]) and np.array_equal(mix["modes"], base["modes"])
    assert abs(np.mean((mix["hs"] >= 70) & (mix["hs"] <= 180)) - 0.6245) < 0.01
    assert abs(np.mean(mix["hs"] >= 250) - 0.2825) < 0.01


def test_mixture_ratio_is_half(mix):
    assert abs(mix["cdn"].mean() - 0.5) < 0.005
    h = mix["hs"]  # the cdn choice is independent of height
    assert abs(mix["cdn"][h < 120].mean() - mix["cdn"][h >= 120].mean()) < 0.01


def test_resampler_choice_proportions(mix):
    k = mix["cdn_kinds"]
    assert set(np.unique(k)) == {0, 1, 2}
    assert np.abs(np.bincount(k) / len(k) - 1 / 3).max() < 0.005


def test_first_jpeg_quality_uniform_75_92(mix):
    q = mix["cdn_q1"]
    assert q.min() == 75 and q.max() == 92
    assert np.abs(np.bincount(q, minlength=93)[75:93] / len(q) - 1 / 18).max() < 0.004


def test_second_encode_probability_and_range(mix):
    assert abs(mix["cdn_second"].mean() - 0.3) < 0.005
    q2 = mix["cdn_q2"]
    assert q2.min() == 80 and q2.max() == 95
    assert set(np.unique(mix["cdn_d"])) == {-2, -1, 1, 2}


def test_blur_is_mild(mix):
    s = mix["cdn_sigmas"]
    assert s.min() >= 0.0 and s.max() <= 0.4 and abs(s.mean() - 0.2) < 0.005


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("s", [20, 50, 100, 144])
def test_resample_matrix_rows_and_constant_image(device, kind, s):
    dev = torch.device(device)
    m = G2._resample_matrix(torch.tensor([144.0], device=dev), torch.tensor([float(s)], device=dev), torch.tensor([kind], device=dev))
    assert torch.allclose(m[0, :s].sum(-1), torch.ones(s, device=dev), atol=1e-5)
    assert m[0, s:].abs().sum() == 0 and m[0, :, 144:].abs().sum() == 0
    out = G2._resize_square(torch.full((1, 3, 144, 144), 100.0, device=dev), m)
    assert torch.allclose(out[0, :, :s, :s], torch.full((3, s, s), 100.0, device=dev), atol=1e-3)


@pytest.mark.parametrize("kind,pil", [(0, Image.LANCZOS), (1, Image.BICUBIC), (2, Image.BOX)])
@pytest.mark.parametrize("s", [37, 72])
def test_resamplers_match_pil(kind, pil, s):
    arr = (smooth_cards(1, 3)[0, :, G.SY:G.SY + 144, G.SX:G.SX + 144]).float()
    ref = np.asarray(Image.fromarray(arr.permute(1, 2, 0).numpy().astype(np.uint8)).resize((s, s), pil), dtype=np.float32)
    m = G2._resample_matrix(torch.tensor([144.0]), torch.tensor([float(s)]), torch.tensor([kind]))
    mine = G2._resize_square(arr[None], m)[0, :, :s, :s].permute(1, 2, 0).numpy()
    assert np.abs(mine - ref).mean() < 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_degrade_cdn_shape_and_closer_to_large_than_harsh_synthetic(device):
    dev = torch.device(device)
    cards = smooth_cards(16).to(dev)
    big = G.jitter(cards, G.jitter_boxes(np.random.default_rng(3), 16))
    p = G2.sample_degradation_mix(np.random.default_rng(4), 16, "cdn50")
    p["hs"] = np.full(16, 100)
    p["cdn_second"] = np.arange(16) % 2 == 0
    deg = G2.degrade_cdn(big, p, dev)
    assert deg.shape == (16, 3, 224, 224) and torch.isfinite(deg).all()
    harsh = G2.degrade(big, {"hs": np.full(16, 100), "modes": np.full(16, 2), "sigmas": np.full(16, 0.47), "quals": np.full(16, 40)}, dev)
    large = G.art_view(big)
    assert cos_rows(large, deg).mean() > cos_rows(large, harsh).mean()


@pytest.mark.parametrize("device", DEVICES)
def test_second_resize_keeps_size_in_bounds(device):
    dev = torch.device(device)
    big = G.jitter(smooth_cards(6).to(dev), G.jitter_boxes(np.random.default_rng(1), 6))
    p = G2.sample_degradation_mix(np.random.default_rng(2), 6, "cdn50")
    p.update(hs=np.array([320, 320, 20, 20, 100, 100]), cdn_second=np.ones(6, bool), cdn_d=np.array([2, -2, -2, -1, 1, 2]))
    assert torch.isfinite(G2.degrade_cdn(big, p, dev)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_make_batch_mix_routes_each_sample_and_default_unchanged(device):
    dev = torch.device(device)
    cards = smooth_cards(10).to(dev)
    l0, d0, h0, _ = G2.make_batch(cards, np.random.default_rng(8), dev)
    l1, d1, h1, _ = G2.make_batch(cards, np.random.default_rng(8), dev, "none")
    assert torch.equal(d0, d1) and np.array_equal(h0, h1)
    l2, d2, h2, _ = G2.make_batch(cards, np.random.default_rng(8), dev, "cdn50")
    assert d2.shape == (10, 3, 224, 224) and np.array_equal(h2, h0) and torch.equal(l0, l2)
    big = G.jitter(cards, G.jitter_boxes(np.random.default_rng(8), 10))
    q = G2.sample_degradation_mix(np.random.default_rng(9), 10, "cdn50")
    out = G2.degrade_mixed(big, q, dev)
    syn = G2.degrade(big, q, dev)
    cdn = G2.degrade_cdn(big, q, dev)
    for i in range(10):
        assert torch.allclose(out[i], (cdn if q["cdn"][i] else syn)[i], atol=1e-4)
