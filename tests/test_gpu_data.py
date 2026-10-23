"""Unit tests for the GPU-resident augmentation (train/gpu_data.py); CPU always, CUDA when present."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import gpu_data as G  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def smooth_cards(n, seed=0):
    """Low-frequency random cards (so degraded and large views of the same card are correlated)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, 3, 8, 6, generator=g)
    return (F.interpolate(base, size=(G.CARD_H, G.CARD_W), mode="bicubic", align_corners=False).clamp(0, 1) * 255).to(torch.uint8)


def cos_rows(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


def test_art_rect_constants():
    assert (G.SX, G.SY, G.SIDE) == (42, 0, 144)


def test_height_mixture_fractions():
    h = G.sample_heights(np.random.default_rng(1), 200_000)
    assert h.min() >= 48 and h.max() <= 320
    assert 0.77 < np.mean((h >= 70) & (h <= 180)) < 0.83  # 60% focus + ~half of the 40% wide draws
    assert 0.03 < np.mean(h >= 250) < 0.08


def test_jitter_boxes_in_bounds_and_aspect():
    b = G.jitter_boxes(np.random.default_rng(2), 5000)
    assert (b[:, 0] >= -1e-6).all() and (b[:, 1] >= -1e-6).all()
    assert (b[:, 2] <= G.CARD_W + 1e-6).all() and (b[:, 3] <= G.CARD_H + 1e-6).all()
    ar = (b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1])
    assert np.allclose(ar, G.CARD_W / G.CARD_H, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_views_share_one_jittered_big(device):
    dev = torch.device(device)
    cards = smooth_cards(6).to(dev)
    big = G.jitter(cards, G.jitter_boxes(np.random.default_rng(3), 6))
    p = {"hs": np.full(6, 320), "modes": np.zeros(6, int), "sigmas": np.zeros(6), "quals": np.full(6, 85)}
    large, deg = G.art_view(big), G.degrade(big, p, dev)
    assert large.shape == deg.shape == (6, 3, 224, 224)
    same = cos_rows(large, deg)
    other = cos_rows(large, deg.roll(1, 0))
    assert (same > 0.97).all()
    assert same.min() > other.max()  # the student view is tied to ITS OWN jittered image


@pytest.mark.parametrize("device", DEVICES)
def test_make_batch_shapes_and_degradation_strength(device):
    dev = torch.device(device)
    cards = smooth_cards(8).to(dev)
    large, deg, hs, big = G.make_batch(cards, np.random.default_rng(4), dev)
    assert large.shape == deg.shape == (8, 3, 224, 224) and big.shape == (8, 3, G.CARD_H, G.CARD_W)
    assert torch.isfinite(large).all() and torch.isfinite(deg).all() and len(hs) == 8
    p = {"hs": np.full(8, 48), "modes": np.ones(8, int), "sigmas": np.zeros(8), "quals": np.full(8, 40)}
    assert cos_rows(G.art_view(big), G.degrade(big, p, dev)).mean() < cos_rows(large, G.degrade(big, {**p, "hs": np.full(8, 320), "quals": np.full(8, 85)}, dev)).mean()


def test_normalised_name():
    assert G.norm_name("Pikachu V-MAX!") == "pikachuvmax" and G.norm_name(None) == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvjpeg_on_side_stream_matches_cpu_while_default_stream_is_busy():
    dev = torch.device("cuda")
    items = [(smooth_cards(1, i)[0, :, :40 + 7 * i, :40 + 5 * i]).contiguous() for i in range(6)]
    cpu = G._jpeg_roundtrip([t.cpu() for t in items], 60, torch.device("cpu"))
    busy = torch.randn(4096, 4096, device=dev)
    side = torch.cuda.Stream()
    for _ in range(3):
        for _ in range(20):
            busy = (busy @ busy).clamp(-1, 1)  # keep the default stream occupied
        with torch.cuda.stream(side):
            gpu = G._jpeg_roundtrip([t.to(dev) for t in items], 60, dev)
        side.synchronize()
        for c, g in zip(cpu, gpu):
            assert c.shape == g.shape and (c.float() - g.cpu().float()).abs().mean() < 3.0
    assert G._USE_NVJPEG["on"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("s", [8, 31, 77, 144])
def test_matrix_resize_matches_torch_interpolate(device, s):
    from torchvision.transforms.functional import gaussian_blur

    dev = torch.device(device)
    x = torch.rand(1, 3, G.SIDE, G.SIDE, device=dev) * 255
    for mode, kw in ((0, dict(mode="bilinear", antialias=True)), (1, dict(mode="bicubic", antialias=True)), (2, dict(mode="bilinear", antialias=False))):
        r = G._down_matrix(torch.tensor([float(s)], device=dev), torch.tensor([mode], device=dev))
        ours = (r @ x @ r.transpose(1, 2))[:, :, :s, :s]
        ref = F.interpolate(x, size=(s, s), align_corners=False, **kw)
        assert (ours - ref).abs().mean() < 1.0, (mode, s)
        assert (r[:, s:] == 0).all()
    sig = 0.9
    m = G._blur_matrix(torch.tensor([sig], device=dev))
    ref = gaussian_blur(x, [7, 7], [sig, sig])
    assert (m @ x @ m.transpose(1, 2) - ref).abs().max() < 0.5
    assert torch.allclose(G._blur_matrix(torch.zeros(1, device=dev))[0], torch.eye(G.SIDE, device=dev))
    u = G._up_matrix(torch.tensor([float(s)], device=dev))
    xs = x[:, :, :s, :s]
    pad = torch.zeros_like(x)
    pad[:, :, :s, :s] = xs
    ref = F.interpolate(xs, size=(224, 224), mode="bicubic", align_corners=False)
    assert (u @ pad @ u.transpose(1, 2) - ref).abs().mean() < 0.5
