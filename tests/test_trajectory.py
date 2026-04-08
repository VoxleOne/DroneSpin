import uuid
import numpy as np
import pytest
from datetime import datetime, timezone
from dronespin.domain.trajectory import Trajectory, validate_trajectory
from dronespin.store.db import init_db
from dronespin.store.trajectory_store import TrajectoryStore


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_rigid_pose_trajectory():
    return Trajectory(
        id=str(uuid.uuid4()),
        name="test_rigid",
        scope="formation",
        kind="rigid_pose",
        t0=0.0,
        t1=10.0,
        interpolation="linear",
        data={
            "rigid_pose_keyframes": [
                {"t": 0.0, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"t": 10.0, "position": [10.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            ]
        },
        metadata={},
        created_at=_now(),
        updated_at=_now(),
    )


def test_trajectory_rigid_pose_evaluate():
    traj = make_rigid_pose_trajectory()

    result_0 = traj.evaluate(0.0)
    assert "formation_pose" in result_0
    assert np.allclose(result_0["formation_pose"]["position"], [0.0, 0.0, 0.0], atol=1e-9)

    result_10 = traj.evaluate(10.0)
    assert np.allclose(result_10["formation_pose"]["position"], [10.0, 0.0, 0.0], atol=1e-9)

    result_5 = traj.evaluate(5.0)
    assert np.allclose(result_5["formation_pose"]["position"], [5.0, 0.0, 0.0], atol=1e-9)


def test_trajectory_persistence_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = TrajectoryStore(conn)

    traj = make_rigid_pose_trajectory()
    store.create(traj)

    retrieved = store.get_by_id(traj.id)
    assert retrieved is not None
    assert retrieved.id == traj.id
    assert retrieved.name == traj.name
    assert retrieved.scope == traj.scope
    assert retrieved.kind == traj.kind
    assert retrieved.interpolation == traj.interpolation
    assert abs(retrieved.t0 - traj.t0) < 1e-9

    deleted = store.delete(traj.id)
    assert deleted is True
    assert store.get_by_id(traj.id) is None

    conn.close()


def test_trajectory_list_all(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = TrajectoryStore(conn)

    t1 = make_rigid_pose_trajectory()
    t1.name = "traj1"
    t2 = make_rigid_pose_trajectory()
    t2.name = "traj2"
    store.create(t1)
    store.create(t2)

    all_trajs = store.list_all()
    assert len(all_trajs) == 2
    conn.close()
