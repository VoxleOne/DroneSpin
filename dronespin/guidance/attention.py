import numpy as np
from spinstep.utils import batch_quaternion_angle, get_array_module
from dronespin.math_s3.quaternion import look_rotation


def compute_attention(drone_orientation, candidate_directions, half_angle=None):
    """
    Compute attention weights over candidate directions using SpinStep AttentionCone paradigm.

    drone_orientation: [x,y,z,w] unit quaternion
    candidate_directions: list of [x,y,z] unit vectors (target directions from drone)
    half_angle: cone half-angle in radians (default pi = full sphere)

    Returns: np.ndarray of shape (N,) with non-negative weights summing to 1.
    """
    if half_angle is None:
        half_angle = np.pi

    drone_orientation = np.array(drone_orientation, dtype=float)
    drone_orientation /= np.linalg.norm(drone_orientation)

    target_quats = []
    for d in candidate_directions:
        d = np.array(d, dtype=float)
        q = look_rotation(d)
        target_quats.append(q)

    target_quats = np.array(target_quats, dtype=float)
    drone_q_batch = drone_orientation.reshape(1, 4)

    xp = get_array_module(False)
    # angular distances shape (1, N)
    angles = batch_quaternion_angle(drone_q_batch, target_quats, xp)[0]

    in_cone = angles <= half_angle
    if not np.any(in_cone):
        weights = np.ones(len(candidate_directions)) / len(candidate_directions)
        return weights

    cone_angles = np.where(in_cone, angles, np.inf)
    logits = -cone_angles / (half_angle + 1e-8)
    logits = np.where(np.isinf(logits), -1e9, logits)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights
