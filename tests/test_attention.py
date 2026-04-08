import numpy as np
import pytest
from dronespin.guidance.attention import compute_attention


def test_attention_sums_to_one():
    rng = np.random.default_rng(42)
    # Random drone orientation
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)

    # Random candidate directions
    dirs = []
    for _ in range(6):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        dirs.append(d)

    weights = compute_attention(q, dirs)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_attention_nonnegative():
    rng = np.random.default_rng(7)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)

    dirs = []
    for _ in range(8):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        dirs.append(d)

    weights = compute_attention(q, dirs)
    assert np.all(weights >= 0)


def test_attention_single_direction():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    dirs = [[0.0, 0.0, 1.0]]
    weights = compute_attention(q, dirs)
    assert abs(weights[0] - 1.0) < 1e-6


def test_attention_with_half_angle():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    dirs = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    weights = compute_attention(q, dirs, half_angle=np.pi / 4)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert np.all(weights >= 0)
