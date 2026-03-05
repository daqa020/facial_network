"""
triangulation.py – Delaunay triangulation on facial landmarks.

Takes an array of 2-D landmark points and produces a triangle mesh
suitable for network-style visualisation.
"""

from typing import Set, Tuple

import numpy as np
from scipy.spatial import Delaunay


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_triangulation(points: np.ndarray) -> np.ndarray:
    """
    Compute a Delaunay triangulation over *points*.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 2)`` containing 2-D coordinates (pixels).

    Returns
    -------
    np.ndarray
        Simplices array of shape ``(M, 3)`` where each row contains
        three indices into *points* defining one triangle.
    """
    if points.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 points for triangulation, got {points.shape[0]}."
        )

    tri = Delaunay(points)
    return tri.simplices


def extract_unique_edges(
    simplices: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    Extract deduplicated edges from a simplices array.

    Each triangle has three edges.  Edges shared by adjacent triangles
    are stored only once, halving the number of draw calls required
    during rendering.

    Returns
    -------
    set of (int, int)
        Each element is a sorted 2-tuple ``(min_idx, max_idx)``.
    """
    edges: Set[Tuple[int, int]] = set()

    for tri in simplices:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(i, k), max(i, k)))

    return edges
