"""
renderer.py – Mesh visualisation renderer.

Takes facial landmark points and edges and draws a clean
network-style mesh diagram using OpenCV drawing primitives.
Supports per-edge colour / thickness, selected-edge highlighting,
rendering on a configurable canvas colour or as an overlay on the
original image.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np


# ─── Edge Style ──────────────────────────────────────────────────────────────

@dataclass
class EdgeStyle:
    """Per-edge visual overrides."""
    color: Tuple[int, int, int] | None = None   # BGR; None → use global
    thickness: int | None = None                  # None → use global


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    """Rendering parameters for the mesh visualisation."""

    # Canvas
    canvas_color: Tuple[int, int, int] = (255, 255, 255)  # white default

    # Line (edge) settings
    line_color: Tuple[int, int, int] = (50, 50, 50)       # dark grey default
    line_thickness: Optional[int] = None                    # None -> auto-scale

    # Node (landmark) settings
    draw_nodes: bool = True
    node_color: Tuple[int, int, int] = (0, 200, 255)      # BGR – orange
    node_radius: Optional[int] = None                       # None -> auto-scale

    # Overlay settings
    overlay_on_original: bool = False
    overlay_alpha: float = 0.4

    # Per-edge overrides  {(min_idx, max_idx): EdgeStyle}
    edge_styles: Dict[Tuple[int, int], EdgeStyle] = field(
        default_factory=dict)

    # Edge currently selected in the GUI (drawn with glow highlight)
    selected_edge: Optional[Tuple[int, int]] = None


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
        Rendering parameters (including per-edge styles).

    Returns
    -------
    np.ndarray
        The rendered output image (BGR, same resolution as *original_image*).
    """
    if config is None:
        config = RenderConfig()

    h, w = original_image.shape[:2]

    default_thickness = (
        config.line_thickness
        if config.line_thickness is not None
        else _auto_thickness(w)
    )
    radius = (
        config.node_radius
        if config.node_radius is not None
        else _auto_radius(w)
    )

    # Canvas: configurable colour (default white)
    canvas = np.full_like(original_image, config.canvas_color, dtype=np.uint8)
    pts_int = np.round(points).astype(np.int32)

    # ── Draw edges ───────────────────────────────────────────────────────
    for i, j in edges:
        if i >= pts_int.shape[0] or j >= pts_int.shape[0]:
            continue
        pt1 = (int(pts_int[i, 0]), int(pts_int[i, 1]))
        pt2 = (int(pts_int[j, 0]), int(pts_int[j, 1]))

        # Per-edge style lookup
        key = (min(i, j), max(i, j))
        es = config.edge_styles.get(key)

        colour = es.color if (es and es.color is not None) else config.line_color
        thick = es.thickness if (es and es.thickness is not None) else default_thickness

        cv2.line(
            canvas, pt1, pt2,
            color=colour,
            thickness=thick,
            lineType=cv2.LINE_AA,
        )

    # ── Draw selected-edge highlight (glow) ──────────────────────────────
    if config.selected_edge is not None:
        si, sj = config.selected_edge
        if si < pts_int.shape[0] and sj < pts_int.shape[0]:
            p1 = (int(pts_int[si, 0]), int(pts_int[si, 1]))
            p2 = (int(pts_int[sj, 0]), int(pts_int[sj, 1]))
            glow_thick = max(default_thickness + 4, 3)
            # outer glow
            cv2.line(canvas, p1, p2, color=(255, 180, 0),
                     thickness=glow_thick + 2, lineType=cv2.LINE_AA)
            # inner bright line
            cv2.line(canvas, p1, p2, color=(255, 220, 60),
                     thickness=glow_thick - 1, lineType=cv2.LINE_AA)

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
        # Draw mesh lines/nodes on top of the original photo.
        # Use a mask: wherever the canvas differs from the blank background,
        # blend those pixels onto the original image.
        bg = np.full_like(original_image, config.canvas_color, dtype=np.uint8)
        mask = np.any(canvas != bg, axis=-1)  # pixels where mesh was drawn

        output = original_image.copy()
        alpha = np.clip(config.overlay_alpha, 0.0, 1.0)
        # Blend only where mesh was drawn
        output[mask] = cv2.addWeighted(
            original_image, 1.0 - alpha,
            canvas, alpha,
            gamma=0.0,
        )[mask]
        return output

    return canvas
