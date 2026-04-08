import numpy as np
import pytest
from dronespin.math_s3.quaternion import slerp, weighted_average_quaternion, look_rotation


def _random_unit_quat(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def test_slerp_endpoints():
    rng = np.random.default_rng(0)
    q1 = _random_unit_quat(rng)
    q2 = _random_unit_quat(rng)
    r0 = slerp(q1, q2, 0.0)
    r1 = slerp(q1, q2, 1.0)
    # Handle double-cover: q and -q represent same rotation
    assert np.allclose(np.abs(np.dot(r0, q1 / np.linalg.norm(q1))), 1.0, atol=1e-6)
    assert np.allclose(np.abs(np.dot(r1, q2 / np.linalg.norm(q2))), 1.0, atol=1e-6)


def test_slerp_unit():
    rng = np.random.default_rng(1)
    for _ in range(10):
        q1 = _random_unit_quat(rng)
        q2 = _random_unit_quat(rng)
        t = rng.uniform(0, 1)
        r = slerp(q1, q2, t)
        assert abs(np.linalg.norm(r) - 1.0) < 1e-6


def test_weighted_average_unit():
    rng = np.random.default_rng(2)
    quats = [_random_unit_quat(rng) for _ in range(5)]
    weights = np.abs(rng.standard_normal(5))
    result = weighted_average_quaternion(weights, quats)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


def test_look_rotation_unit():
    rng = np.random.default_rng(3)
    for _ in range(10):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        q = look_rotation(direction)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_look_rotation_aligns_forward():
    """look_rotation(d) rotates +Z to d."""
    directions = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [-1.0, 0.5, 0.5],
    ]
    for d in directions:
        d = np.array(d, dtype=float)
        d /= np.linalg.norm(d)
        q = look_rotation(d)
        # Rotate [0,0,1] by quaternion
        x, y, z, w = q
        fwd = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x*x + y*y)
        ])
        assert np.allclose(fwd, d, atol=1e-5), f"Expected {d}, got {fwd}"
