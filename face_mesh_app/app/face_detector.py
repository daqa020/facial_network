"""
face_detector.py – Facial landmark detection using MediaPipe FaceLandmarker.

Uses the ``mp.tasks.vision.FaceLandmarker`` (Tasks API) with a local
``.task`` model file so the application runs fully offline.
"""

import os
from pathlib import Path
from typing import List, Tuple

import mediapipe as mp
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "face_landmarker.task")

_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


# ─── Exceptions ──────────────────────────────────────────────────────────────

class NoFaceDetectedError(Exception):
    """Raised when no face is found in the input image."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when the FaceLandmarker model file is missing."""


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _resolve_model_path(model_path: str | None = None) -> str:
    """Return a verified path to the ``.task`` model file."""
    path = model_path or _DEFAULT_MODEL_PATH
    if not os.path.isfile(path):
        raise ModelNotFoundError(
            f"FaceLandmarker model not found at: {path}\n"
            f"Download it with:\n"
            f"  Invoke-WebRequest -Uri \"{_MODEL_DOWNLOAD_URL}\" "
            f"-OutFile \"{path}\"\n"
            f"Or run:  python installer/install.py"
        )
    return path


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_landmarks(
    image_rgb: np.ndarray,
    min_detection_confidence: float = 0.5,
    refine_landmarks: bool = True,
    model_path: str | None = None,
) -> List[Tuple[float, float]]:
    """
    Detect facial landmarks in an RGB image using MediaPipe FaceLandmarker.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image in **RGB** colour order (height x width x 3).
    min_detection_confidence : float
        Minimum confidence for face detection (0.0-1.0).
    refine_landmarks : bool
        Unused in the Tasks API (kept for interface compatibility).
    model_path : str, optional
        Path to the ``face_landmarker.task`` model file.

    Returns
    -------
    list of (x_norm, y_norm)
        Normalised landmark coordinates in the range [0, 1].

    Raises
    ------
    NoFaceDetectedError
        If no face is detected in the image.
    ModelNotFoundError
        If the model file is missing.
    """
    resolved_path = _resolve_model_path(model_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise NoFaceDetectedError(
            "No face detected in the image. "
            "Please provide an image containing a clearly visible human face."
        )

    face = result.face_landmarks[0]

    landmarks: List[Tuple[float, float]] = [
        (lm.x, lm.y) for lm in face
    ]
    return landmarks


def scale_landmarks(
    landmarks: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert normalised landmark coordinates to pixel coordinates.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with ``float64`` pixel coordinates.
    """
    points = np.array(landmarks, dtype=np.float64)
    points[:, 0] *= width
    points[:, 1] *= height
    return points
