import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

SCOPES = ("formation", "drone", "mixed")
KINDS = ("rigid_pose", "deforming_vertices", "drone_position_targets", "drone_pose_targets")
INTERPOLATIONS = ("linear", "slerp", "cubic")


@dataclass
class Trajectory:
    id: str
    name: str
    scope: str
    kind: str
    t0: float
    t1: Optional[float]
    interpolation: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def evaluate(self, t: float) -> Dict[str, Any]:
        """Evaluate trajectory at time t."""
        if self.kind == "rigid_pose":
            return self._eval_rigid_pose(t)
        elif self.kind == "deforming_vertices":
            return self._eval_deforming_vertices(t)
        elif self.kind == "drone_position_targets":
            return self._eval_drone_position_targets(t)
        elif self.kind == "drone_pose_targets":
            return self._eval_drone_pose_targets(t)
        return {}

    def _interp_t(self, keyframes, t):
        """Return (kf_before, kf_after, alpha) for time t."""
        if not keyframes:
            return None, None, 0.0
        if t <= keyframes[0]["t"]:
            return keyframes[0], keyframes[0], 0.0
        if t >= keyframes[-1]["t"]:
            return keyframes[-1], keyframes[-1], 0.0
        for i in range(len(keyframes) - 1):
            kf0, kf1 = keyframes[i], keyframes[i+1]
            if kf0["t"] <= t <= kf1["t"]:
                dt = kf1["t"] - kf0["t"]
                alpha = (t - kf0["t"]) / dt if dt > 0 else 0.0
                return kf0, kf1, alpha
        return keyframes[-1], keyframes[-1], 0.0

    def _eval_rigid_pose(self, t):
        import numpy as np
        from dronespin.math_s3.quaternion import slerp
        kfs = self.data.get("rigid_pose_keyframes", [])
        kf0, kf1, alpha = self._interp_t(kfs, t)
        if kf0 is None:
            return {}
        pos0 = np.array(kf0["position"])
        pos1 = np.array(kf1["position"])
        pos = pos0 + alpha * (pos1 - pos0)
        q0 = np.array(kf0["orientation"])
        q1 = np.array(kf1["orientation"])
        ori = slerp(q0, q1, alpha)
        return {"formation_pose": {"position": pos.tolist(), "orientation": ori.tolist()}}

    def _eval_deforming_vertices(self, t):
        import numpy as np
        kfs = self.data.get("vertices_keyframes", [])
        kf0, kf1, alpha = self._interp_t(kfs, t)
        if kf0 is None:
            return {}
        v0 = np.array(kf0["vertices"])
        v1 = np.array(kf1["vertices"])
        verts = (v0 + alpha * (v1 - v0)).tolist()
        return {"formation_vertices": verts}

    def _eval_drone_position_targets(self, t):
        import numpy as np
        result = {}
        for drone_id, kfs in self.data.get("drone_position_targets", {}).items():
            kf0, kf1, alpha = self._interp_t(kfs, t)
            if kf0 is None:
                continue
            pos0 = np.array(kf0["position"])
            pos1 = np.array(kf1["position"])
            result[drone_id] = (pos0 + alpha * (pos1 - pos0)).tolist()
        return {"drone_position_targets": result}

    def _eval_drone_pose_targets(self, t):
        import numpy as np
        from dronespin.math_s3.quaternion import slerp
        result = {}
        for drone_id, kfs in self.data.get("drone_pose_targets", {}).items():
            kf0, kf1, alpha = self._interp_t(kfs, t)
            if kf0 is None:
                continue
            pos0 = np.array(kf0["position"])
            pos1 = np.array(kf1["position"])
            pos = (pos0 + alpha * (pos1 - pos0)).tolist()
            q0 = np.array(kf0["orientation"])
            q1 = np.array(kf1["orientation"])
            ori = slerp(q0, q1, alpha).tolist()
            result[drone_id] = {"position": pos, "orientation": ori}
        return {"drone_pose_targets": result}


def validate_trajectory(traj: Trajectory):
    if traj.scope not in SCOPES:
        raise ValueError(f"Unknown scope: {traj.scope}")
    if traj.kind not in KINDS:
        raise ValueError(f"Unknown kind: {traj.kind}")
    if traj.interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation: {traj.interpolation}")
    if traj.t1 is not None and traj.t1 < traj.t0:
        raise ValueError("t1 must be >= t0")
