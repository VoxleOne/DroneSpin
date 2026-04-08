import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DroneState:
    drone_id: str
    position: np.ndarray  # (3,)
    orientation: np.ndarray  # (4,) [x,y,z,w] unit quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    assigned_vertex: Optional[int] = None

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.orientation = np.array(self.orientation, dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-10:
            self.orientation /= norm
        self.velocity = np.array(self.velocity, dtype=float)
