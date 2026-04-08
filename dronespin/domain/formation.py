import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

FORMATION_TYPES = ("tetrahedron", "cube", "octahedron", "icosahedron", "custom")


@dataclass
class Formation:
    id: str
    name: str
    type: str
    scale: float
    vertices: List[List[float]]
    frame: str = "world"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_tetrahedron(scale=1.0, name="tetrahedron") -> Formation:
    """Regular tetrahedron centered at origin with circumradius = scale.

    Vertex coordinates are derived analytically for a tetrahedron inscribed in
    a sphere of radius ``scale``.  The top vertex sits at (0, 0, scale) and the
    three base vertices are evenly distributed on the cone at polar angle
    arccos(-1/3) ≈ 109.47°:

    - x-factor for base vertex 0:  2√2 / 3
    - x/y-factor for base vertices 1 & 2:  √2 / 3  and  √6 / 3
    - z-component for all base vertices:  -1/3 (× scale)
    """
    # Geometric constants for a unit-circumradius regular tetrahedron
    _BASE_Z = -1.0 / 3.0           # z of base vertices (normalised)
    _BASE_XY_A = 2.0 * np.sqrt(2) / 3.0  # x of vertex 0 base (normalised)
    _BASE_XY_B = np.sqrt(2) / 3.0        # |x| of vertices 1 & 2 (normalised)
    _BASE_XY_C = np.sqrt(6) / 3.0        # |y| of vertices 1 & 2 (normalised)

    s = scale
    vertices = [
        [0.0,               0.0,                s * 1.0],
        [s * _BASE_XY_A,    0.0,                s * _BASE_Z],
        [-s * _BASE_XY_B,   s * _BASE_XY_C,     s * _BASE_Z],
        [-s * _BASE_XY_B,  -s * _BASE_XY_C,     s * _BASE_Z],
    ]
    now = _now()
    return Formation(
        id=str(uuid.uuid4()),
        name=name,
        type="tetrahedron",
        scale=scale,
        vertices=vertices,
        frame="world",
        metadata={},
        created_at=now,
        updated_at=now,
    )


def validate_formation(f: Formation):
    """Raise ValueError if formation is invalid."""
    if f.type not in FORMATION_TYPES:
        raise ValueError(f"Unknown formation type: {f.type}")
    if f.scale <= 0:
        raise ValueError(f"Scale must be positive, got {f.scale}")
    if not f.vertices:
        raise ValueError("Formation must have at least one vertex")
    for v in f.vertices:
        if len(v) != 3:
            raise ValueError(f"Each vertex must have 3 coordinates, got {v}")
    if f.frame != "world":
        raise ValueError(f"Only 'world' frame supported in v0.1.0, got {f.frame}")
