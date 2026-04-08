import numpy as np
from spinstep.utils import quaternion_distance, quaternion_multiply, quaternion_conjugate, rotation_matrix_to_quaternion


def slerp(q1, q2, t):
    """SLERP between q1 and q2 at parameter t in [0,1]. Returns unit quaternion [x,y,z,w]."""
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    result = s1 * q1 + s2 * q2
    return result / np.linalg.norm(result)


def weighted_average_quaternion(weights, quats):
    """Weighted average of unit quaternions. Returns unit quaternion [x,y,z,w]."""
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    quats = np.array(quats, dtype=float)
    M = np.zeros((4, 4))
    for w, q in zip(weights, quats):
        q = q / np.linalg.norm(q)
        M += w * np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(M)
    avg = eigvecs[:, np.argmax(eigvals)]
    return avg / np.linalg.norm(avg)


def look_rotation(desired_forward, up=None):
    """Compute quaternion [x,y,z,w] that rotates world +Z axis to desired_forward."""
    desired_forward = np.array(desired_forward, dtype=float)
    norm = np.linalg.norm(desired_forward)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    desired_forward = desired_forward / norm

    forward = np.array([0.0, 0.0, 1.0])
    dot = np.dot(forward, desired_forward)

    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = np.cross(forward, desired_forward)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    q = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])
    return q / np.linalg.norm(q)
