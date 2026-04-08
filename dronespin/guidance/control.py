import numpy as np
from dronespin.math_s3.quaternion import slerp, look_rotation
from dronespin.guidance.attention import compute_attention
from dronespin.guidance.assignment import assign_drones_to_vertices


def guidance_step(drone_state, formation, traj_sample, dt, slerp_alpha=0.1, speed=1.0):
    """
    Single guidance tick. Updates drone_state in-place and returns it.

    drone_state: DroneState
    formation: Formation (possibly modified by trajectory)
    traj_sample: dict from trajectory.evaluate(t) or {}
    dt: timestep
    slerp_alpha: interpolation fraction for orientation (0 < alpha <= 1)
    speed: drone speed toward target
    """
    vertices = np.array(formation.vertices)

    if "formation_vertices" in traj_sample:
        vertices = np.array(traj_sample["formation_vertices"])

    if "formation_pose" in traj_sample:
        pose = traj_sample["formation_pose"]
        offset = np.array(pose["position"])
        vertices = vertices + offset

    target_pos = None
    if "drone_position_targets" in traj_sample:
        targets = traj_sample["drone_position_targets"]
        if drone_state.drone_id in targets:
            target_pos = np.array(targets[drone_state.drone_id])

    if "drone_pose_targets" in traj_sample:
        targets = traj_sample["drone_pose_targets"]
        if drone_state.drone_id in targets:
            pt = targets[drone_state.drone_id]
            target_pos = np.array(pt["position"])

    if target_pos is None:
        n_vertices = len(vertices)
        vertex_idx = drone_state.assigned_vertex
        if vertex_idx is None:
            vertex_idx = 0
        vertex_idx = vertex_idx % n_vertices
        target_pos = vertices[vertex_idx]

    delta = target_pos - drone_state.position
    dist = np.linalg.norm(delta)

    if dist > 1e-6:
        desired_forward = delta / dist
    else:
        desired_forward = np.array([0.0, 0.0, 1.0])

    # Compute attention weights over candidate vertex directions
    candidate_dirs = []
    for v in vertices:
        d = v - drone_state.position
        dn = np.linalg.norm(d)
        if dn > 1e-6:
            candidate_dirs.append(d / dn)
        else:
            candidate_dirs.append(np.array([0.0, 0.0, 1.0]))

    weights = compute_attention(drone_state.orientation, candidate_dirs)
    best_idx = int(np.argmax(weights))
    best_dir = candidate_dirs[best_idx]
    target_orientation = look_rotation(best_dir)

    # Update orientation via SLERP
    drone_state.orientation = slerp(drone_state.orientation, target_orientation, slerp_alpha)

    # Update position (move toward target)
    step = min(speed * dt, dist)
    if dist > 1e-6:
        drone_state.position = drone_state.position + desired_forward * step

    drone_state.velocity = desired_forward * speed if dist > 1e-6 else np.zeros(3)

    return drone_state
