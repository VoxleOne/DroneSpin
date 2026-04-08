import numpy as np
import pytest
from dronespin.domain.formation import make_tetrahedron
from dronespin.sim.drone_state import DroneState
from dronespin.sim.runner import SimRunner


def test_convergence_tetrahedron_4_drones():
    """4 drones should converge to tetrahedron vertices in < 100 steps."""
    rng = np.random.default_rng(42)
    formation = make_tetrahedron(scale=1.0)

    # Initialize drones near origin with random offsets
    drones = []
    for i in range(4):
        pos = rng.uniform(-0.5, 0.5, 3)
        drones.append(DroneState(
            drone_id=str(i),
            position=pos,
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            assigned_vertex=i,
        ))

    runner = SimRunner(
        drones=drones,
        formation=formation,
        dt=0.1,
        speed=2.0,
        slerp_alpha=0.2,
        seed=42,
    )

    for step in range(100):
        runner.step()
        err = runner.convergence_error()
        if err < 0.5:
            break

    final_error = runner.convergence_error()
    assert final_error < 0.5, (
        f"Drones did not converge after {runner.step_count} steps. "
        f"Final error: {final_error:.4f}"
    )
