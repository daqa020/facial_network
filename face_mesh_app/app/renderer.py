"""
renderer.py – Mesh visualisation renderer.

Takes facial landmark points and Delaunay edges and draws a clean
network-style mesh diagram using OpenCV drawing primitives.
Supports rendering on a black canvas or as an overlay on the
original image.
"""

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import cv2
import numpy as np


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    """Rendering parameters for the mesh visualisation."""

    # Line (edge) settings
    line_color: Tuple[int, int, int] = (255, 255, 0)   # BGR – cyan
    line_thickness: Optional[int] = None                 # None -> auto-scale

    # Node (landmark) settings
    draw_nodes: bool = True
    node_color: Tuple[int, int, int] = (0, 200, 255)   # BGR – orange
    node_radius: Optional[int] = None                    # None -> auto-scale

    # Overlay settings
    overlay_on_original: bool = False
    overlay_alpha: float = 0.4


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _auto_thickness(width: int) -> int:
    """Return an edge thickness that looks good at *width* pixels."""
    return max(1, width // 1920)


def _auto_radius(width: int) -> int:
    """Return a node radius that looks good at *width* pixels."""
    return max(1, width // 960)


# ─── Public API ──────────────────────────────────────────────────────────────

def render_mesh(
    original_image: np.ndarray,
    points: np.ndarray,
    edges: Set[Tuple[int, int]],
    config: Optional[RenderConfig] = None,
) -> np.ndarray:
    """
    Render a network-style mesh at full original resolution.

    Parameters
    ----------
    original_image : np.ndarray
        The original input image (BGR, full resolution).
    points : np.ndarray
        Pixel-space landmark coordinates, shape ``(N, 2)``.
    edges : set of (int, int)
        Unique edge pairs.
    config : RenderConfig, optional
        Rendering parameters.

    Returns
    -------
    np.ndarray
        The rendered output image (BGR, same resolution as *original_image*).
    """
    if config is None:
        config = RenderConfig()

    h, w = original_image.shape[:2]

    thickness = (
        config.line_thickness
        if config.line_thickness is not None
        else _auto_thickness(w)
    )
    radius = (
        config.node_radius
        if config.node_radius is not None
        else _auto_radius(w)
    )

    canvas = np.zeros_like(original_image)
    pts_int = np.round(points).astype(np.int32)

    # ── Draw edges ───────────────────────────────────────────────────────
    for i, j in edges:
        pt1 = (int(pts_int[i, 0]), int(pts_int[i, 1]))
        pt2 = (int(pts_int[j, 0]), int(pts_int[j, 1]))
        cv2.line(
            canvas, pt1, pt2,
            color=config.line_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # ── Draw nodes ───────────────────────────────────────────────────────
    if config.draw_nodes:
        for idx in range(pts_int.shape[0]):
            centre = (int(pts_int[idx, 0]), int(pts_int[idx, 1]))
            cv2.circle(
                canvas, centre,
                radius=radius,
                color=config.node_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # ── Optional overlay blending ────────────────────────────────────────
    if config.overlay_on_original:
        alpha = np.clip(config.overlay_alpha, 0.0, 1.0)
        output = cv2.addWeighted(
            original_image, 1.0 - alpha,
            canvas, alpha,
            gamma=0.0,
        )
        output = cv2.addWeighted(output, 1.0, canvas, alpha, gamma=0.0)
        return output

    return canvas
