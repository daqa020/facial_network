"""
image_utils.py – Image loading, scaling, and saving utilities.

Handles high-resolution images (including 4K) and provides helpers
for downscaling images before face detection while preserving
coordinate mapping to the original resolution.
"""

import os
from typing import Tuple

import cv2
import numpy as np


# ─── Exceptions ──────────────────────────────────────────────────────────────

class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""


# ─── Public API ──────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load an image from *path* and return it as a BGR ``numpy`` array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImageLoadError
        If OpenCV cannot decode the file (corrupt / unsupported format).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(
            f"Failed to decode image (corrupt or unsupported format): {path}"
        )
    return image


def get_resolution(image: np.ndarray) -> Tuple[int, int]:
    """Return the resolution of *image* as ``(height, width)``."""
    return image.shape[0], image.shape[1]


def downscale_for_detection(
    image: np.ndarray,
    max_dim: int = 1280,
) -> Tuple[np.ndarray, float]:
    """
    Optionally downscale *image* so its longest edge is at most *max_dim*.

    Returns
    -------
    (resized_image, scale_factor)
        *scale_factor* is ``original_longest / resized_longest`` (>= 1.0).
        If the image is already within *max_dim*, returns the original
        array unchanged with ``scale_factor = 1.0``.
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_dim:
        return image, 1.0

    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_factor = longest / max(new_h, new_w)
    return resized, scale_factor


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB (required by MediaPipe)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str) -> None:
    """
    Write *image* to *path*.

    The output directory is created automatically if it does not exist.

    Raises
    ------
    IOError
        If the write fails.
    """
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to write image to: {path}")
