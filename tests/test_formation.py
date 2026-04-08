import numpy as np
import pytest
from dronespin.domain.formation import make_tetrahedron, Formation, validate_formation
from dronespin.store.db import init_db
from dronespin.store.formation_store import FormationStore


def test_tetrahedron_vertices_count():
    f = make_tetrahedron(scale=1.0)
    assert len(f.vertices) == 4


def test_tetrahedron_centroid():
    f = make_tetrahedron(scale=1.0)
    vertices = np.array(f.vertices)
    centroid = vertices.mean(axis=0)
    assert np.allclose(centroid, [0, 0, 0], atol=1e-10)


def test_tetrahedron_equal_distances():
    f = make_tetrahedron(scale=1.0)
    vertices = np.array(f.vertices)
    distances = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(vertices[i] - vertices[j])
            distances.append(d)
    # All edge lengths should be equal
    assert np.allclose(distances, distances[0], atol=1e-6), f"Distances not equal: {distances}"


def test_formation_persistence_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = FormationStore(conn)

    f = make_tetrahedron(scale=2.0, name="test_tet")
    store.create(f)

    retrieved = store.get_by_id(f.id)
    assert retrieved is not None
    assert retrieved.id == f.id
    assert retrieved.name == f.name
    assert retrieved.type == f.type
    assert abs(retrieved.scale - f.scale) < 1e-9
    assert np.allclose(retrieved.vertices, f.vertices, atol=1e-10)

    conn.close()


def test_formation_list_all(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = FormationStore(conn)

    f1 = make_tetrahedron(scale=1.0, name="tet1")
    f2 = make_tetrahedron(scale=2.0, name="tet2")
    store.create(f1)
    store.create(f2)

    all_formations = store.list_all()
    assert len(all_formations) == 2
    conn.close()


def test_formation_update(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = FormationStore(conn)

    f = make_tetrahedron(scale=1.0, name="original")
    store.create(f)

    f.name = "updated"
    f.scale = 3.0
    store.update(f)

    retrieved = store.get_by_id(f.id)
    assert retrieved.name == "updated"
    assert abs(retrieved.scale - 3.0) < 1e-9
    conn.close()


def test_formation_delete(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    store = FormationStore(conn)

    f = make_tetrahedron(scale=1.0)
    store.create(f)
    deleted = store.delete(f.id)
    assert deleted is True
    assert store.get_by_id(f.id) is None
    conn.close()
