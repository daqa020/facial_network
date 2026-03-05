"""
gui.py – Interactive GUI for the Facial Network Mesh application.

Features:
  - Load an image and detect face landmarks
  - View the Delaunay mesh overlaid on the image
  - Click-and-drag any landmark point to edit the mesh in real time
  - Export the mesh as a PNG image
  - Export face data as JSON (landmarks, triangulation, metadata)
"""

import json
import os
import sys
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from face_detector import (ModelNotFoundError, NoFaceDetectedError,
                            detect_landmarks, scale_landmarks)
from gpu_check import print_hardware_report
from image_utils import (bgr_to_rgb, downscale_for_detection,
                          get_resolution, load_image, save_image)
from renderer import RenderConfig, render_mesh
from triangulation import compute_triangulation, extract_unique_edges


# ─── Constants ───────────────────────────────────────────────────────────────

CANVAS_MAX = 900
POINT_HIT_RADIUS = 8
BG = "#1e1e2e"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
SURFACE = "#313244"


class FaceMeshGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Network Mesh")
        self.root.configure(bg=BG)
        self.root.geometry("1200x820")
        self.root.minsize(800, 600)

        # ── State ──
        self.original_image = None
        self.image_path = ""
        self.orig_w = 0
        self.orig_h = 0
        self.landmarks_norm = []
        self.points = None
        self.simplices = None
        self.edges = set()
        self.display_scale = 1.0
        self.dragging_idx = None
        self._img_offset_x = 0
        self._img_offset_y = 0
        self._tk_img = None
        self.overlay_mode = tk.BooleanVar(value=False)
        self.show_nodes = tk.BooleanVar(value=True)

        self._build_ui()
        self._centre_window()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 14, "bold"))
        style.configure("Accent.TButton", background=ACCENT,
                         foreground="#1e1e2e", font=("Segoe UI", 10, "bold"),
                         padding=6)
        style.map("Accent.TButton",
                  background=[("active", "#74c7ec"), ("disabled", "#45475a")])
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))

        # ── Toolbar ──
        toolbar = ttk.Frame(self.root, style="TFrame")
        toolbar.pack(fill="x", padx=12, pady=(10, 4))

        ttk.Label(toolbar, text="Facial Network Mesh",
                  style="Title.TLabel").pack(side="left")

        ttk.Button(toolbar, text="Export JSON", style="Accent.TButton",
                   command=self._export_json).pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="Export PNG", style="Accent.TButton",
                   command=self._export_png).pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="Reset Points", style="Accent.TButton",
                   command=self._reset_points).pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="Load Image", style="Accent.TButton",
                   command=self._load_image).pack(side="right", padx=(4, 0))

        # ── Options row ──
        opts = ttk.Frame(self.root, style="TFrame")
        opts.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Checkbutton(opts, text="Overlay on photo",
                         variable=self.overlay_mode,
                         style="TCheckbutton",
                         command=self._redraw).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Show nodes",
                         variable=self.show_nodes,
                         style="TCheckbutton",
                         command=self._redraw).pack(side="left")

        self.status_var = tk.StringVar(value="Load an image to begin")
        ttk.Label(opts, textvariable=self.status_var,
                  style="TLabel").pack(side="right")

        # ── Canvas ──
        canvas_frame = ttk.Frame(self.root, style="TFrame")
        canvas_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.canvas = tk.Canvas(canvas_frame, bg="#11111b",
                                 highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _centre_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    # ── Image Loading ────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return

        self.status_var.set("Loading image \u2026")
        self.root.update_idletasks()

        try:
            self.original_image = load_image(path)
            self.image_path = path
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            self.status_var.set("Load failed")
            return

        self.orig_h, self.orig_w = get_resolution(self.original_image)
        self.status_var.set("Detecting face \u2026")
        self.root.update_idletasks()

        try:
            small, _ = downscale_for_detection(self.original_image, max_dim=1280)
            small_rgb = bgr_to_rgb(small)
            t0 = time.perf_counter()
            self.landmarks_norm = detect_landmarks(small_rgb)
            elapsed = (time.perf_counter() - t0) * 1000
        except NoFaceDetectedError:
            messagebox.showwarning("No Face", "No face detected in this image.")
            self.status_var.set("No face found")
            return
        except ModelNotFoundError as exc:
            messagebox.showerror("Model Missing", str(exc))
            self.status_var.set("Model missing \u2013 run installer first")
            return

        self.points = scale_landmarks(self.landmarks_norm,
                                       self.orig_w, self.orig_h)
        self._retriangulate()

        n = len(self.landmarks_norm)
        self.status_var.set(
            f"{n} landmarks  |  {len(self.simplices)} triangles  |  "
            f"{len(self.edges)} edges  |  {elapsed:.0f} ms  |  "
            f"{self.orig_w}\u00d7{self.orig_h}")
        self._redraw()

    # ── Triangulation ────────────────────────────────────────────────────

    def _retriangulate(self):
        self.simplices = compute_triangulation(self.points)
        self.edges = extract_unique_edges(self.simplices)

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self):
        if self.points is None or self.original_image is None:
            return

        config = RenderConfig(
            overlay_on_original=self.overlay_mode.get(),
            overlay_alpha=0.4,
            draw_nodes=self.show_nodes.get(),
        )
        rendered = render_mesh(self.original_image, self.points,
                               self.edges, config)

        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        cw = self.canvas.winfo_width() or CANVAS_MAX
        ch = self.canvas.winfo_height() or CANVAS_MAX
        scale_x = cw / self.orig_w
        scale_y = ch / self.orig_h
        self.display_scale = min(scale_x, scale_y, 1.0)

        disp_w = int(self.orig_w * self.display_scale)
        disp_h = int(self.orig_h * self.display_scale)
        pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")

        self._img_offset_x = (cw - disp_w) // 2
        self._img_offset_y = (ch - disp_h) // 2
        self.canvas.create_image(self._img_offset_x, self._img_offset_y,
                                  anchor="nw", image=self._tk_img)

    # ── Mouse Interaction ────────────────────────────────────────────────

    def _canvas_to_real(self, cx, cy):
        rx = (cx - self._img_offset_x) / self.display_scale
        ry = (cy - self._img_offset_y) / self.display_scale
        return rx, ry

    def _on_press(self, event):
        if self.points is None:
            return
        rx, ry = self._canvas_to_real(event.x, event.y)
        dists = np.sqrt((self.points[:, 0] - rx) ** 2 +
                         (self.points[:, 1] - ry) ** 2)
        hit_radius_real = POINT_HIT_RADIUS / self.display_scale
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= hit_radius_real:
            self.dragging_idx = min_idx
            self.canvas.configure(cursor="fleur")

    def _on_drag(self, event):
        if self.dragging_idx is None:
            return
        rx, ry = self._canvas_to_real(event.x, event.y)
        rx = max(0, min(rx, self.orig_w - 1))
        ry = max(0, min(ry, self.orig_h - 1))
        self.points[self.dragging_idx] = [rx, ry]
        self._retriangulate()
        self._redraw()

    def _on_release(self, _event):
        if self.dragging_idx is not None:
            self.dragging_idx = None
            self.canvas.configure(cursor="crosshair")
            self.status_var.set(
                f"{len(self.landmarks_norm)} landmarks  |  "
                f"{len(self.simplices)} triangles  |  "
                f"{len(self.edges)} edges  |  "
                f"{self.orig_w}\u00d7{self.orig_h}  |  edited")

    # ── Reset ────────────────────────────────────────────────────────────

    def _reset_points(self):
        if not self.landmarks_norm:
            return
        self.points = scale_landmarks(self.landmarks_norm,
                                       self.orig_w, self.orig_h)
        self._retriangulate()
        self._redraw()
        self.status_var.set(
            f"{len(self.landmarks_norm)} landmarks  |  reset to original")

    # ── Export PNG ────────────────────────────────────────────────────────

    def _export_png(self):
        if self.points is None:
            messagebox.showinfo("No Data", "Load an image first.")
            return

        default_name = Path(self.image_path).stem + "_mesh.png"
        path = filedialog.asksaveasfilename(
            title="Export mesh PNG",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if not path:
            return

        config = RenderConfig(
            overlay_on_original=self.overlay_mode.get(),
            overlay_alpha=0.4,
            draw_nodes=self.show_nodes.get(),
        )
        output = render_mesh(self.original_image, self.points,
                              self.edges, config)
        save_image(output, path)
        self.status_var.set(f"Exported PNG \u2192 {os.path.basename(path)}")

    # ── Export JSON ──────────────────────────────────────────────────────

    def _export_json(self):
        if self.points is None:
            messagebox.showinfo("No Data", "Load an image first.")
            return

        default_name = Path(self.image_path).stem + "_face.json"
        path = filedialog.asksaveasfilename(
            title="Export face JSON",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return

        data = self._build_json_data()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.status_var.set(f"Exported JSON \u2192 {os.path.basename(path)}")

    def _build_json_data(self) -> dict:
        pts = self.points.tolist()
        norms = self.landmarks_norm
        edge_list = sorted([list(e) for e in self.edges])
        tri_list = self.simplices.tolist() if self.simplices is not None else []

        return {
            "metadata": {
                "source_image": os.path.basename(self.image_path),
                "source_path": self.image_path,
                "image_width": self.orig_w,
                "image_height": self.orig_h,
                "num_landmarks": len(pts),
                "num_triangles": len(tri_list),
                "num_edges": len(edge_list),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "edited": any(
                    abs(pts[i][0] / self.orig_w - norms[i][0]) > 1e-4 or
                    abs(pts[i][1] / self.orig_h - norms[i][1]) > 1e-4
                    for i in range(len(norms))
                ),
            },
            "landmarks_normalized": [
                {"index": i, "x": round(x, 6), "y": round(y, 6)}
                for i, (x, y) in enumerate(norms)
            ],
            "landmarks_pixel": [
                {"index": i, "x": round(pt[0], 2), "y": round(pt[1], 2)}
                for i, pt in enumerate(pts)
            ],
            "triangles": tri_list,
            "edges": edge_list,
        }

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print_hardware_report()
    FaceMeshGUI().run()
