# Facial Network Mesh

Fully offline Python application that generates an interactive **facial network mesh visualisation** from any image containing a human face.

Uses **MediaPipe FaceLandmarker** to detect 478 facial landmarks, then computes a **Delaunay triangulation** and renders the mesh — either on a clean black canvas or overlaid on the original photo.

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **478 facial landmarks** detected via MediaPipe FaceLandmarker (Tasks API)
- **Delaunay triangulation** from SciPy (~916 triangles, ~1 393 unique edges)
- **Black-canvas** or **photo overlay** rendering modes
- **Interactive GUI** — click and drag any landmark point to reshape the mesh in real time
- **Export PNG** at full original resolution
- **Export JSON** — landmarks (normalised + pixel), triangles, edges, and metadata
- **CLI mode** with 13 configurable options
- **GPU / CUDA detection** and hardware report
- Handles images up to **4K+** (auto-downscales for detection, renders at full res)
- **100 % offline** — no internet required after initial model download

---

## Quick Start

### 1. Install

Double-click **`install.bat`** (opens the GUI installer), or from terminal:

```bash
cd face_mesh_app
python installer/install.py
```

This will:
- Install Python dependencies (`mediapipe`, `opencv-python`, `numpy`, `scipy`, `Pillow`)
- Download the `face_landmarker.task` model (~3.6 MB, one-time)
- Print a hardware / GPU report

### 2. Run

Double-click **`facial_network.bat`** to launch the interactive GUI, or use the CLI:

```bash
python app/main.py photo.jpg
python app/main.py photo.jpg --overlay --no-nodes
python app/main.py photo.jpg -o output.png --line-color 0,255,0
```

---

## Project Structure

```
facial_network/
├── install.bat                  # Double-click installer launcher
├── facial_network.bat           # Double-click app launcher
└── face_mesh_app/
    ├── requirements.txt
    ├── app/
    │   ├── __init__.py
    │   ├── gpu_check.py         # GPU + CUDA detection
    │   ├── image_utils.py       # Image load / save / resize
    │   ├── face_detector.py     # MediaPipe FaceLandmarker wrapper
    │   ├── triangulation.py     # Delaunay triangulation
    │   ├── renderer.py          # Mesh rendering
    │   ├── main.py              # CLI entry point
    │   ├── gui.py               # Interactive Tkinter GUI
    │   └── face_landmarker.task # MediaPipe model (downloaded by installer)
    └── installer/
        ├── __init__.py
        ├── install.py           # CLI installer
        └── install_gui.py       # GUI installer
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Path to input image (required) |
| `-o`, `--output` | `<stem>_mesh.png` | Output file path |
| `--max-detection-dim` | 1280 | Max dimension for detection downscale |
| `--overlay` | off | Overlay mesh on the original image |
| `--overlay-alpha` | 0.4 | Blend factor for overlay mode |
| `--no-nodes` | off | Hide landmark dots |
| `--line-color` | 255,255,0 | Edge colour (B,G,R) |
| `--node-color` | 0,200,255 | Node colour (B,G,R) |
| `--line-thickness` | auto | Edge thickness in pixels |
| `--node-radius` | auto | Node radius in pixels |
| `--min-confidence` | 0.5 | Detection confidence threshold |
| `--no-refine` | off | Use 468 landmarks instead of 478 |
| `--skip-gpu-check` | off | Skip hardware report |

---

## GUI Features

- **Load Image** — opens a file picker for JPG / PNG / BMP / WebP
- **Click & Drag** — grab any landmark point and move it; the mesh re-triangulates in real time
- **Overlay / Nodes** — toggle checkboxes to switch rendering mode
- **Reset Points** — revert all edits back to the original detected landmarks
- **Export PNG** — save the rendered mesh at full original resolution
- **Export JSON** — save all face data (landmarks, triangles, edges, metadata) with an `edited` flag

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | ≥ 0.10.9 | Face landmark detection |
| opencv-python | ≥ 4.8.0 | Image I/O and rendering |
| numpy | ≥ 1.24.0 | Array operations |
| scipy | ≥ 1.10.0 | Delaunay triangulation |
| Pillow | ≥ 9.0.0 | GUI image display |

Tkinter is included with standard Python on Windows.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModelNotFoundError` | Run `install.bat` to download the model |
| `No face detected` | Try a clearer, front-facing photo |
| `mediapipe has no attribute 'solutions'` | Update mediapipe ≥ 0.10.9 (legacy API removed) |
| Black window in GUI | Resize the window or load a smaller image |
| CUDA not detected | Install CUDA toolkit (optional — app works on CPU) |

---

## License

MIT
