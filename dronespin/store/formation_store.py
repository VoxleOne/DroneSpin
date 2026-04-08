import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from dronespin.domain.formation import Formation


def _now():
    return datetime.now(timezone.utc).isoformat()


def _row_to_formation(row) -> Formation:
    return Formation(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        scale=row["scale"],
        frame=row["frame"],
        vertices=json.loads(row["vertices_json"]),
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class FormationStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, formation: Formation) -> Formation:
        self.conn.execute(
            """INSERT INTO formations
               (id, name, type, scale, frame, vertices_json, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                formation.id,
                formation.name,
                formation.type,
                formation.scale,
                formation.frame,
                json.dumps(formation.vertices),
                json.dumps(formation.metadata),
                formation.created_at,
                formation.updated_at,
            ),
        )
        self.conn.commit()
        return formation

    def get_by_id(self, formation_id: str) -> Optional[Formation]:
        cur = self.conn.execute(
            "SELECT * FROM formations WHERE id = ?", (formation_id,)
        )
        row = cur.fetchone()
        return _row_to_formation(row) if row else None

    def list_all(self) -> List[Formation]:
        cur = self.conn.execute("SELECT * FROM formations ORDER BY created_at")
        return [_row_to_formation(row) for row in cur.fetchall()]

    def update(self, formation: Formation) -> Formation:
        formation.updated_at = _now()
        self.conn.execute(
            """UPDATE formations SET
               name=?, type=?, scale=?, frame=?, vertices_json=?, metadata_json=?, updated_at=?
               WHERE id=?""",
            (
                formation.name,
                formation.type,
                formation.scale,
                formation.frame,
                json.dumps(formation.vertices),
                json.dumps(formation.metadata),
                formation.updated_at,
                formation.id,
            ),
        )
        self.conn.commit()
        return formation

    def delete(self, formation_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM formations WHERE id = ?", (formation_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0
