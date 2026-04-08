import numpy as np
from typing import List, Optional
from dronespin.sim.drone_state import DroneState
from dronespin.domain.formation import Formation
from dronespin.guidance.control import guidance_step


class SimRunner:
    def __init__(self, drones: List[DroneState], formation: Formation,
                 trajectories=None, dt=0.1, speed=1.0, slerp_alpha=0.1, seed=42):
        self.drones = drones
        self.formation = formation
        self.trajectories = trajectories or []
        self.dt = dt
        self.speed = speed
        self.slerp_alpha = slerp_alpha
        self.seed = seed
        self.t = 0.0
        self.step_count = 0
        self.history = []
        np.random.seed(seed)
        n_v = len(formation.vertices)
        for i, d in enumerate(drones):
            if d.assigned_vertex is None:
                d.assigned_vertex = i % n_v

    def _get_traj_sample(self):
        result = {}
        for traj in self.trajectories:
            sample = traj.evaluate(self.t)
            result.update(sample)
        return result

    def step(self):
        traj_sample = self._get_traj_sample()
        for drone in self.drones:
            guidance_step(drone, self.formation, traj_sample, self.dt,
                         slerp_alpha=self.slerp_alpha, speed=self.speed)
        self.t += self.dt
        self.step_count += 1
        self._record()

    def _record(self):
        snapshot = {
            "t": self.t,
            "step": self.step_count,
            "drones": [
                {
                    "drone_id": d.drone_id,
                    "position": d.position.tolist(),
                    "orientation": d.orientation.tolist(),
                    "assigned_vertex": d.assigned_vertex,
                }
                for d in self.drones
            ]
        }
        self.history.append(snapshot)

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()
        return self.history

    def convergence_error(self):
        """Max distance between any drone and its assigned vertex."""
        vertices = np.array(self.formation.vertices)
        errors = []
        for d in self.drones:
            if d.assigned_vertex is not None:
                v = vertices[d.assigned_vertex % len(vertices)]
                errors.append(np.linalg.norm(d.position - v))
        return max(errors) if errors else float('inf')
