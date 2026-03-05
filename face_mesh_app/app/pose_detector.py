"""
pose_detector.py – Body pose landmark detection using MediaPipe PoseLandmarker.

Uses ``mp.tasks.vision.PoseLandmarker`` (Tasks API) with a local ``.task``
model file so the application runs fully offline.

Detects 33 body landmarks and provides predefined skeleton connections
for natural anatomical line drawing (shoulder→elbow→wrist, hip→knee→ankle, etc.).
"""

import os
from pathlib import Path
from typing import List, Tuple

import mediapipe as mp
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent / "pose_landmarker_heavy.task"
)

_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

# MediaPipe PoseLandmarker 33 landmark indices:
#  0  nose                11 left_shoulder     23 left_hip
#  1  left_eye_inner      12 right_shoulder    24 right_hip
#  2  left_eye            13 left_elbow        25 left_knee
#  3  left_eye_outer      14 right_elbow       26 right_knee
#  4  right_eye_inner     15 left_wrist        27 left_ankle
#  5  right_eye           16 right_wrist       28 right_ankle
#  6  right_eye_outer     17 left_pinky        29 left_heel
#  7  left_ear            18 right_pinky       30 right_heel
#  8  right_ear           19 left_index        31 left_foot_index
#  9  mouth_left          20 right_index       32 right_foot_index
# 10  mouth_right         21 left_thumb
#                         22 right_thumb

# Skeleton edge connections (anatomical structure)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),      # nose → left eye → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),      # nose → right eye → right ear
    (9, 10),                               # mouth

    # Torso
    (11, 12),                              # shoulder to shoulder
    (11, 23), (12, 24),                    # shoulders to hips
    (23, 24),                              # hip to hip

    # Left arm
    (11, 13), (13, 15),                    # shoulder → elbow → wrist
    (15, 17), (15, 19), (15, 21),          # wrist → pinky / index / thumb

    # Right arm
    (12, 14), (14, 16),                    # shoulder → elbow → wrist
    (16, 18), (16, 20), (16, 22),          # wrist → pinky / index / thumb

    # Left leg
    (23, 25), (25, 27),                    # hip → knee → ankle
    (27, 29), (27, 31), (29, 31),          # ankle → heel → foot index

    # Right leg
    (24, 26), (26, 28),                    # hip → knee → ankle
    (28, 30), (28, 32), (30, 32),          # ankle → heel → foot index
]


# ─── Exceptions ──────────────────────────────────────────────────────────────

class NoPoseDetectedError(Exception):
    """Raised when no body pose is found in the input image."""


class PoseModelNotFoundError(FileNotFoundError):
    """Raised when the PoseLandmarker model file is missing."""


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _resolve_model_path(model_path: str | None = None) -> str:
    """Return a verified path to the ``.task`` model file."""
    path = model_path or _DEFAULT_MODEL_PATH
    if not os.path.isfile(path):
        raise PoseModelNotFoundError(
            f"PoseLandmarker model not found at: {path}\n"
            f"Download it with:\n"
            f"  Invoke-WebRequest -Uri \"{_MODEL_DOWNLOAD_URL}\" "
            f"-OutFile \"{path}\"\n"
            f"Or run:  python installer/install.py"
        )
    return path


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_pose_landmarks(
    image_rgb: np.ndarray,
    min_detection_confidence: float = 0.5,
    model_path: str | None = None,
) -> List[Tuple[float, float]]:
    """
    Detect body pose landmarks in an RGB image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image in **RGB** colour order (height × width × 3).
    min_detection_confidence : float
        Minimum confidence for pose detection (0.0–1.0).
    model_path : str, optional
        Path to the ``pose_landmarker_heavy.task`` model file.

    Returns
    -------
    list of (x_norm, y_norm)
        Normalised landmark coordinates in the range [0, 1].

    Raises
    ------
    NoPoseDetectedError
        If no body pose is detected in the image.
    PoseModelNotFoundError
        If the model file is missing.
    """
    resolved_path = _resolve_model_path(model_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise NoPoseDetectedError(
            "No body pose detected in the image. "
            "Please provide an image containing a clearly visible human body."
        )

    pose = result.pose_landmarks[0]
    landmarks: List[Tuple[float, float]] = [
        (lm.x, lm.y) for lm in pose
    ]
    return landmarks


def get_pose_skeleton_edges(
    face_point_count: int,
) -> set:
    """
    Return the predefined skeleton connection edges, with indices offset
    by *face_point_count* so they index correctly into a merged array
    where face points come first (0 … face_point_count-1) and pose
    points follow (face_point_count … face_point_count+32).

    Returns
    -------
    set of (int, int)
        Sorted edge tuples.
    """
    edges = set()
    for a, b in POSE_CONNECTIONS:
        i = a + face_point_count
        j = b + face_point_count
        edges.add((min(i, j), max(i, j)))
    return edges


def scale_pose_landmarks(
    landmarks: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert normalised pose landmark coordinates to pixel coordinates.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with ``float64`` pixel coordinates.
    """
    points = np.array(landmarks, dtype=np.float64)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


# Body regions that should get denser subdivision
_TORSO_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso rectangle
]
_ARM_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # upper/lower arms
]
_LEG_CONNECTIONS = [
    (23, 25), (25, 27), (24, 26), (26, 28),  # upper/lower legs
]


def enrich_body_points(
    pose_points_pixel: np.ndarray,
    subdivisions_torso: int = 5,
    subdivisions_limb: int = 4,
    subdivisions_other: int = 2,
    jitter_factor: float = 0.08,
) -> np.ndarray:
    """
    Generate additional points along skeleton connections by subdivision,
    creating a denser point-cloud suitable for Delaunay triangulation.

    Adds intermediate points along each skeleton edge.  Torso edges get
    more subdivisions (to fill the rectangle), limb edges get moderate
    subdivisions, and smaller connections get fewer.

    Also fills the torso interior with a grid of extra points so the
    mesh covers the chest/abdomen area.

    Parameters
    ----------
    pose_points_pixel : np.ndarray
        The original 33 pose landmark pixel coordinates, shape (33, 2).
    subdivisions_torso : int
        Number of intermediate points per torso edge.
    subdivisions_limb : int
        Number of intermediate points per arm/leg edge.
    subdivisions_other : int
        Number of intermediate points per other connection.
    jitter_factor : float
        Random offset as fraction of edge length to avoid collinear points.

    Returns
    -------
    np.ndarray
        Enriched point array, shape ``(N, 2)`` where N >> 33.
        The first 33 rows are the originals (unchanged).
    """
    rng = np.random.default_rng(42)  # deterministic for consistency
    extra_pts = []

    torso_set = set(_TORSO_CONNECTIONS)
    arm_set = set(_ARM_CONNECTIONS)
    leg_set = set(_LEG_CONNECTIONS)

    for a, b in POSE_CONNECTIONS:
        pa = pose_points_pixel[a]
        pb = pose_points_pixel[b]

        conn = (a, b)
        if conn in torso_set:
            n_sub = subdivisions_torso
        elif conn in arm_set or conn in leg_set:
            n_sub = subdivisions_limb
        else:
            n_sub = subdivisions_other

        edge_len = np.linalg.norm(pb - pa)
        jitter_mag = edge_len * jitter_factor

        for k in range(1, n_sub + 1):
            t = k / (n_sub + 1)
            pt = pa + t * (pb - pa)
            # Small perpendicular jitter so Delaunay doesn't degenerate
            jit = rng.uniform(-jitter_mag, jitter_mag, size=2)
            extra_pts.append(pt + jit)

    # ── Fill torso interior with a grid ──
    # Torso corners: left_shoulder(11), right_shoulder(12),
    #                left_hip(23),      right_hip(24)
    ls = pose_points_pixel[11]
    rs = pose_points_pixel[12]
    lh = pose_points_pixel[23]
    rh = pose_points_pixel[24]

    grid_n = 4  # 4×4 interior grid
    for gy in range(1, grid_n + 1):
        ty = gy / (grid_n + 1)
        for gx in range(1, grid_n + 1):
            tx = gx / (grid_n + 1)
            # Bilinear interpolation across the torso quadrilateral
            top = ls + tx * (rs - ls)
            bot = lh + tx * (rh - lh)
            pt = top + ty * (bot - top)
            jit = rng.uniform(-jitter_mag * 0.5, jitter_mag * 0.5, size=2)
            extra_pts.append(pt + jit)

    if not extra_pts:
        return pose_points_pixel

    enriched = np.vstack([pose_points_pixel, np.array(extra_pts)])
    return enriched
