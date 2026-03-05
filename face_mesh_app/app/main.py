"""
main.py – CLI entry point for the Facial Network Mesh Visualisation app.

Usage examples
--------------
    python main.py photo.jpg
    python main.py photo.jpg -o mesh_output.png
    python main.py photo.jpg --overlay --overlay-alpha 0.5
    python main.py highres.jpg --max-detection-dim 640 --no-nodes
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from face_detector import ModelNotFoundError, NoFaceDetectedError, detect_landmarks, scale_landmarks
from gpu_check import print_hardware_report
from image_utils import (
    ImageLoadError,
    bgr_to_rgb,
    downscale_for_detection,
    get_resolution,
    load_image,
    save_image,
)
from pose_detector import (NoPoseDetectedError, PoseModelNotFoundError,
                            detect_pose_landmarks, enrich_body_points,
                            scale_pose_landmarks)
from renderer import RenderConfig, render_mesh
from triangulation import compute_triangulation, extract_unique_edges, merge_edges


def _parse_color(value: str):
    """Parse a ``B,G,R`` string into a (B, G, R) int tuple."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected B,G,R colour format (e.g. 255,255,0), got: {value}"
        )
    try:
        bgr = tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Colour components must be integers, got: {value}"
        )
    for c in bgr:
        if not 0 <= c <= 255:
            raise argparse.ArgumentTypeError(
                f"Colour components must be 0-255, got: {value}"
            )
    return bgr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_mesh_app",
        description="Generate a facial network mesh visualisation from an input image.",
    )
    parser.add_argument("input", help="Path to the input image file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path. Default: <input_stem>_mesh.png")
    parser.add_argument("--max-detection-dim", type=int, default=1280,
                        help="Max dimension for detection. Default: 1280.")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="Overlay mesh on original image.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4,
                        help="Blend factor for overlay. Default: 0.4.")
    parser.add_argument("--no-nodes", action="store_true", default=False,
                        help="Do not draw landmark node dots.")
    parser.add_argument("--line-color", type=_parse_color, default="255,255,0",
                        help="Edge colour B,G,R. Default: 255,255,0 (cyan).")
    parser.add_argument("--node-color", type=_parse_color, default="0,200,255",
                        help="Node colour B,G,R. Default: 0,200,255 (orange).")
    parser.add_argument("--line-thickness", type=int, default=None,
                        help="Edge thickness in pixels. Default: auto.")
    parser.add_argument("--node-radius", type=int, default=None,
                        help="Node radius in pixels. Default: auto.")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Face detection confidence 0.0-1.0. Default: 0.5.")
    parser.add_argument("--no-refine", action="store_true", default=False,
                        help="Use 468 landmarks instead of 478.")
    parser.add_argument("--skip-gpu-check", action="store_true", default=False,
                        help="Skip the hardware detection report.")
    parser.add_argument("--body", action="store_true", default=False,
                        help="Enable full body mesh (face Delaunay + body skeleton).")
    return parser


def _default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_mesh.png"))


def run(args: argparse.Namespace) -> None:
    if not args.skip_gpu_check:
        print_hardware_report()

    print(f"[1/5] Loading image: {args.input}")
    original = load_image(args.input)
    orig_h, orig_w = get_resolution(original)
    print(f"       Resolution: {orig_w} \u00d7 {orig_h}")

    print(f"[2/5] Preparing image for detection (max dim = {args.max_detection_dim}) \u2026")
    small, scale_factor = downscale_for_detection(original, max_dim=args.max_detection_dim)
    det_h, det_w = get_resolution(small)
    if scale_factor > 1.0:
        print(f"       Downscaled to {det_w} \u00d7 {det_h} (factor {scale_factor:.2f}\u00d7)")
    else:
        print("       No downscaling needed.")

    small_rgb = bgr_to_rgb(small)

    print("[3/5] Detecting facial landmarks \u2026")
    t0 = time.perf_counter()
    landmarks = detect_landmarks(
        small_rgb,
        min_detection_confidence=args.min_confidence,
        refine_landmarks=not args.no_refine,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"       Found {len(landmarks)} landmarks in {elapsed_ms:.0f} ms.")

    face_points = scale_landmarks(landmarks, orig_w, orig_h)
    face_count = len(landmarks)

    # ── Optional body pose detection ──
    body_pts = None
    all_points = face_points

    if args.body:
        print("[3b/5] Detecting body pose \u2026")
        try:
            pose_lm = detect_pose_landmarks(
                small_rgb,
                min_detection_confidence=args.min_confidence,
            )
            raw_body = scale_pose_landmarks(pose_lm, orig_w, orig_h)
            body_pts = enrich_body_points(raw_body)
            all_points = np.vstack([face_points, body_pts])
            print(f"       Found {len(pose_lm)} body landmarks, "
                  f"enriched to {len(body_pts)} body points.")
        except NoPoseDetectedError:
            print("       No body detected \u2013 showing face only.")
        except PoseModelNotFoundError as exc:
            print(f"       \u26a0 {exc}")

    points = all_points

    print("[4/5] Computing triangulation \u2026")
    simplices = compute_triangulation(face_points)
    face_edges = extract_unique_edges(simplices)

    body_edges = set()
    if body_pts is not None and len(body_pts) >= 3:
        body_simplices = compute_triangulation(body_pts)
        body_edges_raw = extract_unique_edges(body_simplices)
        body_edges = {(a + face_count, b + face_count)
                      for a, b in body_edges_raw}

    edges = merge_edges(face_edges, body_edges)
    print(f"       {len(simplices)} triangles, {len(edges)} unique edges.")

    print("[5/5] Rendering mesh visualisation \u2026")
    config = RenderConfig(
        canvas_color=(255, 255, 255),
        line_color=args.line_color,
        line_thickness=args.line_thickness,
        draw_nodes=not args.no_nodes,
        node_color=args.node_color,
        node_radius=args.node_radius,
        overlay_on_original=args.overlay,
        overlay_alpha=args.overlay_alpha,
    )
    output_image = render_mesh(original, points, edges, config)

    output_path = args.output or _default_output_path(args.input)
    save_image(output_image, output_path)

    out_h, out_w = get_resolution(output_image)
    print(f"\n\u2713 Saved mesh image ({out_w} \u00d7 {out_h}) to: {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except (NoFaceDetectedError, NoPoseDetectedError) as exc:
        print(f"\n\u2717 {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ImageLoadError, ModelNotFoundError,
            PoseModelNotFoundError) as exc:
        print(f"\n\u2717 {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n\u2717 Unexpected error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
