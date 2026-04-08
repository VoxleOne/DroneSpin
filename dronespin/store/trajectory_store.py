import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from dronespin.domain.trajectory import Trajectory


def _now():
    return datetime.now(timezone.utc).isoformat()


def _row_to_trajectory(row) -> Trajectory:
    return Trajectory(
        id=row["id"],
        name=row["name"],
        scope=row["scope"],
        kind=row["kind"],
        interpolation=row["interpolation"],
        t0=row["t0"],
        t1=row["t1"],
        data=json.loads(row["data_json"]),
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class TrajectoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, traj: Trajectory) -> Trajectory:
        self.conn.execute(
            """INSERT INTO trajectories
               (id, name, scope, kind, interpolation, t0, t1, data_json, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                traj.id,
                traj.name,
                traj.scope,
                traj.kind,
                traj.interpolation,
                traj.t0,
                traj.t1,
                json.dumps(traj.data),
                json.dumps(traj.metadata),
                traj.created_at,
                traj.updated_at,
            ),
        )
        self.conn.commit()
        return traj

    def get_by_id(self, traj_id: str) -> Optional[Trajectory]:
        cur = self.conn.execute(
            "SELECT * FROM trajectories WHERE id = ?", (traj_id,)
        )
        row = cur.fetchone()
        return _row_to_trajectory(row) if row else None

    def list_all(self) -> List[Trajectory]:
        cur = self.conn.execute("SELECT * FROM trajectories ORDER BY created_at")
        return [_row_to_trajectory(row) for row in cur.fetchall()]

    def update(self, traj: Trajectory) -> Trajectory:
        traj.updated_at = _now()
        self.conn.execute(
            """UPDATE trajectories SET
               name=?, scope=?, kind=?, interpolation=?, t0=?, t1=?,
               data_json=?, metadata_json=?, updated_at=?
               WHERE id=?""",
            (
                traj.name,
                traj.scope,
                traj.kind,
                traj.interpolation,
                traj.t0,
                traj.t1,
                json.dumps(traj.data),
                json.dumps(traj.metadata),
                traj.updated_at,
                traj.id,
            ),
        )
        self.conn.commit()
        return traj

    def delete(self, traj_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM trajectories WHERE id = ?", (traj_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0
