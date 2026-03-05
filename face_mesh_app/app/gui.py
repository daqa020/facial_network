"""
gui.py – Interactive GUI for the Facial Network Mesh application.

Features:
  - Load an image → choose "Face only" or "Full body" mesh mode
  - View the Delaunay mesh (face) + skeleton connections (body) on a white canvas
  - Click-and-drag any landmark point to edit the mesh in real time
  - Click any edge to select it → sidebar panel to change its colour / thickness
  - Global default colour / thickness controls
  - Export the mesh as a PNG image
  - Export face data as JSON (landmarks, triangulation, metadata)
"""

import json
import math
import os
import sys
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

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
from pose_detector import (NoPoseDetectedError, PoseModelNotFoundError,
                            detect_pose_landmarks, enrich_body_points,
                            scale_pose_landmarks)
from renderer import EdgeStyle, RenderConfig, render_mesh
from triangulation import (compute_triangulation, extract_unique_edges,
                            merge_edges)


# ─── Constants ───────────────────────────────────────────────────────────────

CANVAS_MAX = 900
POINT_HIT_RADIUS = 8
EDGE_HIT_RADIUS = 6           # pixels on display for edge click detection
BG = "#1e1e2e"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
SURFACE = "#313244"
SIDEBAR_W = 220


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bgr_to_hex(bgr):
    """Convert a (B, G, R) tuple to a '#RRGGBB' hex string."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def _hex_to_bgr(hex_str: str):
    """Convert '#RRGGBB' to (B, G, R) tuple."""
    h = hex_str.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# ─── Mode Selection Dialog ──────────────────────────────────────────────────

class ModeDialog:
    """Modal dialog: 'Face only' or 'Full body mesh'."""

    def __init__(self, parent):
        self.result = "face"  # default

        self.top = tk.Toplevel(parent)
        self.top.title("Select Mesh Mode")
        self.top.configure(bg=BG)
        self.top.geometry("340x190")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()

        ttk.Label(self.top, text="Which mesh do you need?",
                  style="Title.TLabel").pack(pady=(18, 10))

        self.var = tk.StringVar(value="face")
        f = ttk.Frame(self.top, style="TFrame")
        f.pack(pady=4)
        ttk.Radiobutton(f, text="Face only  (478 landmarks, Delaunay)",
                         variable=self.var, value="face",
                         style="TRadiobutton").pack(anchor="w", padx=20)
        ttk.Radiobutton(f, text="Full body mesh  (face + body skeleton)",
                         variable=self.var, value="body",
                         style="TRadiobutton").pack(anchor="w", padx=20, pady=(4, 0))

        ttk.Button(self.top, text="OK", style="Accent.TButton",
                   command=self._ok).pack(pady=(14, 0))

        # Centre on parent
        self.top.update_idletasks()
        px = parent.winfo_x() + parent.winfo_width() // 2 - 170
        py = parent.winfo_y() + parent.winfo_height() // 2 - 95
        self.top.geometry(f"+{px}+{py}")

        parent.wait_window(self.top)

    def _ok(self):
        self.result = self.var.get()
        self.top.destroy()


# ─── Main GUI ────────────────────────────────────────────────────────────────

class FaceMeshGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Network Mesh")
        self.root.configure(bg=BG)
        self.root.geometry("1420x860")
        self.root.minsize(900, 600)

        # ── State ──
        self.original_image = None
        self.image_path = ""
        self.orig_w = 0
        self.orig_h = 0
        self.mesh_mode = "face"        # "face" or "body"
        self.face_landmarks_norm = []
        self.pose_landmarks_norm = []
        self.face_point_count = 0
        self.points = None             # merged np.ndarray (N, 2)
        self.simplices = None
        self.edges = set()
        self.edge_styles: dict = {}    # {(i,j): EdgeStyle}
        self.display_scale = 1.0
        self.dragging_idx = None
        self.selected_edge = None      # (i, j) or None
        self._img_offset_x = 0
        self._img_offset_y = 0
        self._tk_img = None

        # Defaults
        self.default_line_color = (50, 50, 50)   # BGR dark grey
        self.default_thickness = None             # auto
        self.overlay_mode = tk.BooleanVar(value=False)
        self.show_nodes = tk.BooleanVar(value=True)

        self._build_ui()
        self._centre_window()

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Small.TLabel", background=BG, foreground="#a6adc8",
                         font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 14, "bold"))
        style.configure("SideTitle.TLabel", background=SURFACE,
                         foreground=ACCENT,
                         font=("Segoe UI", 11, "bold"))
        style.configure("Side.TLabel", background=SURFACE, foreground=FG,
                         font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=ACCENT,
                         foreground="#1e1e2e", font=("Segoe UI", 10, "bold"),
                         padding=6)
        style.map("Accent.TButton",
                  background=[("active", "#74c7ec"), ("disabled", "#45475a")])
        style.configure("Side.TButton", background="#45475a",
                         foreground=FG, font=("Segoe UI", 9), padding=4)
        style.map("Side.TButton",
                  background=[("active", "#585b70")])
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("TRadiobutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Side.TFrame", background=SURFACE)

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
                         command=self._redraw).pack(side="left", padx=(0, 12))

        self.status_var = tk.StringVar(value="Load an image to begin")
        ttk.Label(opts, textvariable=self.status_var,
                  style="TLabel").pack(side="right")

        # ── Main body: canvas + sidebar ──
        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Canvas (left, expanding)
        canvas_frame = ttk.Frame(body, style="TFrame")
        canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#f0f0f0",
                                 highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ── Sidebar (right, fixed width) ──
        self.sidebar = ttk.Frame(body, width=SIDEBAR_W, style="Side.TFrame")
        self.sidebar.pack(side="right", fill="y", padx=(8, 0))
        self.sidebar.pack_propagate(False)

        # ── Global Defaults section ──
        ttk.Label(self.sidebar, text="Line Defaults",
                  style="SideTitle.TLabel").pack(pady=(12, 6), padx=10,
                                                   anchor="w")

        ttk.Label(self.sidebar, text="Default Colour:",
                  style="Side.TLabel").pack(anchor="w", padx=10, pady=(4, 0))
        self._def_color_btn = tk.Button(
            self.sidebar, text="\u25a0 Line Colour",
            bg=_bgr_to_hex(self.default_line_color),
            fg="white", font=("Segoe UI", 9, "bold"), bd=1,
            command=self._pick_default_color)
        self._def_color_btn.pack(anchor="w", padx=10, pady=(2, 4))

        ttk.Label(self.sidebar, text="Default Thickness:",
                  style="Side.TLabel").pack(anchor="w", padx=10, pady=(4, 0))
        self._def_thick_var = tk.IntVar(value=1)
        self._def_thick_scale = tk.Scale(
            self.sidebar, from_=1, to=10, orient="horizontal",
            variable=self._def_thick_var, length=170,
            bg=SURFACE, fg=FG, troughcolor=BG, highlightthickness=0,
            command=lambda _: self._on_default_thickness_changed())
        self._def_thick_scale.pack(anchor="w", padx=10, pady=(0, 6))

        # Separator between defaults and edge inspector
        ttk.Separator(self.sidebar, orient="horizontal").pack(
            fill="x", padx=10, pady=6)

        # ── Edge Inspector section ──
        ttk.Label(self.sidebar, text="Edge Inspector",
                  style="SideTitle.TLabel").pack(pady=(6, 6), padx=10,
                                                   anchor="w")
        self._side_info_var = tk.StringVar(value="Click an edge to select it")
        ttk.Label(self.sidebar, textvariable=self._side_info_var,
                  style="Side.TLabel", wraplength=190).pack(
                      anchor="w", padx=10, pady=(0, 8))

        # Edge colour button
        ttk.Label(self.sidebar, text="Edge Colour:",
                  style="Side.TLabel").pack(anchor="w", padx=10, pady=(4, 0))
        self._edge_color_btn = tk.Button(
            self.sidebar, text="\u25a0 pick colour",
            bg=SURFACE, fg=FG, font=("Segoe UI", 9), bd=1,
            state="disabled", command=self._pick_edge_color)
        self._edge_color_btn.pack(anchor="w", padx=10, pady=(2, 6))

        # Edge thickness slider
        ttk.Label(self.sidebar, text="Edge Thickness:",
                  style="Side.TLabel").pack(anchor="w", padx=10, pady=(4, 0))
        self._edge_thick_var = tk.IntVar(value=1)
        self._edge_thick_scale = tk.Scale(
            self.sidebar, from_=1, to=10, orient="horizontal",
            variable=self._edge_thick_var, length=170,
            bg=SURFACE, fg=FG, troughcolor=BG, highlightthickness=0,
            state="disabled",
            command=lambda _: self._on_edge_thickness_changed())
        self._edge_thick_scale.pack(anchor="w", padx=10, pady=(0, 8))

        # Action buttons
        self._apply_all_btn = ttk.Button(
            self.sidebar, text="Apply to All Edges",
            style="Side.TButton", state="disabled",
            command=self._apply_style_to_all)
        self._apply_all_btn.pack(anchor="w", padx=10, pady=(4, 2))

        self._reset_edge_btn = ttk.Button(
            self.sidebar, text="Reset This Edge",
            style="Side.TButton", state="disabled",
            command=self._reset_selected_edge)
        self._reset_edge_btn.pack(anchor="w", padx=10, pady=(2, 4))

        self._reset_all_styles_btn = ttk.Button(
            self.sidebar, text="Reset All Styles",
            style="Side.TButton",
            command=self._reset_all_styles)
        self._reset_all_styles_btn.pack(anchor="w", padx=10, pady=(2, 4))

        # Separator
        ttk.Separator(self.sidebar, orient="horizontal").pack(
            fill="x", padx=10, pady=8)

        # Stats
        self._stats_var = tk.StringVar(value="")
        ttk.Label(self.sidebar, textvariable=self._stats_var,
                  style="Side.TLabel", wraplength=190,
                  justify="left").pack(anchor="w", padx=10)

    def _centre_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    # ── Default controls ─────────────────────────────────────────────────

    def _pick_default_color(self):
        initial = _bgr_to_hex(self.default_line_color)
        result = colorchooser.askcolor(color=initial, title="Default line colour")
        if result and result[1]:
            self.default_line_color = _hex_to_bgr(result[1])
            self._def_color_btn.configure(bg=result[1])
            self._redraw()

    def _on_default_thickness_changed(self):
        self.default_thickness = self._def_thick_var.get()
        self._redraw()

    # ── Image Loading ────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return

        # Ask mode
        dlg = ModeDialog(self.root)
        self.mesh_mode = dlg.result

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
        self.status_var.set("Detecting landmarks \u2026")
        self.root.update_idletasks()

        # ── Always detect face ──
        try:
            small, _ = downscale_for_detection(self.original_image, max_dim=1280)
            small_rgb = bgr_to_rgb(small)
            t0 = time.perf_counter()
            self.face_landmarks_norm = detect_landmarks(small_rgb)
            elapsed = (time.perf_counter() - t0) * 1000
        except NoFaceDetectedError:
            messagebox.showwarning("No Face", "No face detected in this image.")
            self.status_var.set("No face found")
            return
        except ModelNotFoundError as exc:
            messagebox.showerror("Model Missing", str(exc))
            self.status_var.set("Model missing \u2013 run installer first")
            return

        face_pts = scale_landmarks(self.face_landmarks_norm,
                                    self.orig_w, self.orig_h)
        self.face_point_count = len(self.face_landmarks_norm)

        # ── Optionally detect body pose ──
        self.pose_landmarks_norm = []
        body_pts = None
        body_edges = set()

        if self.mesh_mode == "body":
            try:
                t1 = time.perf_counter()
                self.pose_landmarks_norm = detect_pose_landmarks(small_rgb)
                elapsed += (time.perf_counter() - t1) * 1000
                raw_body = scale_pose_landmarks(
                    self.pose_landmarks_norm, self.orig_w, self.orig_h)
                body_pts = enrich_body_points(raw_body)
            except NoPoseDetectedError:
                messagebox.showwarning(
                    "No Body", "No body pose detected — showing face only.")
            except PoseModelNotFoundError as exc:
                messagebox.showerror("Pose Model Missing", str(exc))
                self.status_var.set("Pose model missing \u2013 run installer")

        # ── Merge points ──
        if body_pts is not None and len(body_pts) > 0:
            self.points = np.vstack([face_pts, body_pts])
        else:
            self.points = face_pts

        # ── Compute edges ──
        face_simplices = compute_triangulation(face_pts)
        face_edges = extract_unique_edges(face_simplices)
        self.simplices = face_simplices

        if body_pts is not None and len(body_pts) >= 3:
            fp = self.face_point_count
            body_simplices = compute_triangulation(body_pts)
            body_edges_raw = extract_unique_edges(body_simplices)
            body_edges = {(a + fp, b + fp) for a, b in body_edges_raw}
        else:
            body_edges = set()

        self.edges = merge_edges(face_edges, body_edges)

        # ── Reset edge styles ──
        self.edge_styles = {}
        self.selected_edge = None
        self._update_sidebar_state()

        stats_text = (
            f"Mode: {self.mesh_mode}\n"
            f"Face points: {self.face_point_count}\n"
            f"Body points: {len(self.pose_landmarks_norm)}\n"
            f"Triangles: {len(self.simplices)}\n"
            f"Edges: {len(self.edges)}\n"
            f"Detection: {elapsed:.0f} ms\n"
            f"Image: {self.orig_w}\u00d7{self.orig_h}"
        )
        self._stats_var.set(stats_text)

        total = self.face_point_count + len(self.pose_landmarks_norm)
        self.status_var.set(
            f"{total} landmarks  |  {len(self.edges)} edges  |  "
            f"{elapsed:.0f} ms  |  {self.orig_w}\u00d7{self.orig_h}")
        self._redraw()

    # ── Triangulation ────────────────────────────────────────────────────

    def _retriangulate(self):
        """Re-triangulate face points and body points via Delaunay."""
        if self.points is None:
            return
        fp = self.face_point_count
        face_pts = self.points[:fp]
        if face_pts.shape[0] >= 3:
            self.simplices = compute_triangulation(face_pts)
            face_edges = extract_unique_edges(self.simplices)
        else:
            self.simplices = np.empty((0, 3), dtype=np.int32)
            face_edges = set()

        body_edges = set()
        if self.mesh_mode == "body" and self.points.shape[0] > fp:
            body_pts = self.points[fp:]
            if body_pts.shape[0] >= 3:
                body_simplices = compute_triangulation(body_pts)
                body_edges_raw = extract_unique_edges(body_simplices)
                body_edges = {(a + fp, b + fp) for a, b in body_edges_raw}

        self.edges = merge_edges(face_edges, body_edges)

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self):
        if self.points is None or self.original_image is None:
            return

        config = RenderConfig(
            canvas_color=(255, 255, 255),
            line_color=self.default_line_color,
            line_thickness=self.default_thickness,
            overlay_on_original=self.overlay_mode.get(),
            overlay_alpha=0.4,
            draw_nodes=self.show_nodes.get(),
            edge_styles=self.edge_styles,
            selected_edge=self.selected_edge,
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

        # ── Node drag takes priority ──
        dists = np.sqrt((self.points[:, 0] - rx) ** 2 +
                         (self.points[:, 1] - ry) ** 2)
        hit_radius_real = POINT_HIT_RADIUS / self.display_scale
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= hit_radius_real:
            self.dragging_idx = min_idx
            self.canvas.configure(cursor="fleur")
            return

        # ── Edge selection ──
        best_edge = None
        best_dist = float("inf")
        edge_hit_real = EDGE_HIT_RADIUS / self.display_scale
        pts = self.points
        for i, j in self.edges:
            if i >= pts.shape[0] or j >= pts.shape[0]:
                continue
            d = _point_to_segment_dist(
                rx, ry, pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1])
            if d < best_dist:
                best_dist = d
                best_edge = (min(i, j), max(i, j))

        if best_edge is not None and best_dist <= edge_hit_real:
            self.selected_edge = best_edge
            self._update_sidebar_state()
            self._redraw()
        else:
            # Clicked empty area — deselect
            if self.selected_edge is not None:
                self.selected_edge = None
                self._update_sidebar_state()
                self._redraw()

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
            total = self.face_point_count + len(self.pose_landmarks_norm)
            self.status_var.set(
                f"{total} landmarks  |  {len(self.edges)} edges  |  "
                f"{self.orig_w}\u00d7{self.orig_h}  |  edited")

    # ── Sidebar / Edge Inspector ─────────────────────────────────────────

    def _update_sidebar_state(self):
        if self.selected_edge is not None:
            i, j = self.selected_edge
            self._side_info_var.set(f"Edge  {i} \u2194 {j}")
            es = self.edge_styles.get(self.selected_edge)
            colour = es.color if (es and es.color) else self.default_line_color
            thick = es.thickness if (es and es.thickness) else (
                self.default_thickness or 1)

            self._edge_color_btn.configure(
                state="normal", bg=_bgr_to_hex(colour),
                text=f"\u25a0  {_bgr_to_hex(colour)}")
            self._edge_thick_scale.configure(state="normal")
            self._edge_thick_var.set(thick)
            self._apply_all_btn.configure(state="normal")
            self._reset_edge_btn.configure(state="normal")
        else:
            self._side_info_var.set("Click an edge to select it")
            self._edge_color_btn.configure(
                state="disabled", bg=SURFACE, text="\u25a0 pick colour")
            self._edge_thick_scale.configure(state="disabled")
            self._apply_all_btn.configure(state="disabled")
            self._reset_edge_btn.configure(state="disabled")

    def _pick_edge_color(self):
        if self.selected_edge is None:
            return
        es = self.edge_styles.get(self.selected_edge)
        current = es.color if (es and es.color) else self.default_line_color
        result = colorchooser.askcolor(
            color=_bgr_to_hex(current), title="Edge colour")
        if result and result[1]:
            bgr = _hex_to_bgr(result[1])
            if self.selected_edge not in self.edge_styles:
                self.edge_styles[self.selected_edge] = EdgeStyle()
            self.edge_styles[self.selected_edge].color = bgr
            self._update_sidebar_state()
            self._redraw()

    def _on_edge_thickness_changed(self):
        if self.selected_edge is None:
            return
        if self.selected_edge not in self.edge_styles:
            self.edge_styles[self.selected_edge] = EdgeStyle()
        self.edge_styles[self.selected_edge].thickness = self._edge_thick_var.get()
        self._redraw()

    def _apply_style_to_all(self):
        if self.selected_edge is None:
            return
        es = self.edge_styles.get(self.selected_edge)
        if es is None:
            return
        for edge in self.edges:
            key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            self.edge_styles[key] = EdgeStyle(
                color=es.color, thickness=es.thickness)
        self._redraw()
        self.status_var.set("Applied style to all edges")

    def _reset_selected_edge(self):
        if self.selected_edge and self.selected_edge in self.edge_styles:
            del self.edge_styles[self.selected_edge]
            self._update_sidebar_state()
            self._redraw()

    def _reset_all_styles(self):
        self.edge_styles.clear()
        self.selected_edge = None
        self._update_sidebar_state()
        self._redraw()
        self.status_var.set("All edge styles reset to defaults")

    # ── Reset ────────────────────────────────────────────────────────────

    def _reset_points(self):
        if not self.face_landmarks_norm:
            return
        face_pts = scale_landmarks(self.face_landmarks_norm,
                                    self.orig_w, self.orig_h)
        if self.pose_landmarks_norm:
            body_pts = scale_pose_landmarks(
                self.pose_landmarks_norm, self.orig_w, self.orig_h)
            self.points = np.vstack([face_pts, body_pts])
        else:
            self.points = face_pts

        self._retriangulate()
        self.edge_styles.clear()
        self.selected_edge = None
        self._update_sidebar_state()
        self._redraw()
        total = self.face_point_count + len(self.pose_landmarks_norm)
        self.status_var.set(f"{total} landmarks  |  reset to original")

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
            canvas_color=(255, 255, 255),
            line_color=self.default_line_color,
            line_thickness=self.default_thickness,
            overlay_on_original=self.overlay_mode.get(),
            overlay_alpha=0.4,
            draw_nodes=self.show_nodes.get(),
            edge_styles=self.edge_styles,
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
        face_norms = self.face_landmarks_norm
        pose_norms = self.pose_landmarks_norm
        edge_list = sorted([list(e) for e in self.edges])
        tri_list = self.simplices.tolist() if self.simplices is not None else []

        # Serialize per-edge styles
        styles_out = {}
        for (ei, ej), es in self.edge_styles.items():
            key = f"{ei}-{ej}"
            styles_out[key] = {}
            if es.color is not None:
                styles_out[key]["color_bgr"] = list(es.color)
            if es.thickness is not None:
                styles_out[key]["thickness"] = es.thickness

        return {
            "metadata": {
                "source_image": os.path.basename(self.image_path),
                "source_path": self.image_path,
                "image_width": self.orig_w,
                "image_height": self.orig_h,
                "mesh_mode": self.mesh_mode,
                "num_face_landmarks": len(face_norms),
                "num_pose_landmarks": len(pose_norms),
                "num_landmarks_total": len(pts),
                "num_triangles": len(tri_list),
                "num_edges": len(edge_list),
                "num_styled_edges": len(styles_out),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "edited": any(
                    abs(pts[i][0] / self.orig_w - face_norms[i][0]) > 1e-4 or
                    abs(pts[i][1] / self.orig_h - face_norms[i][1]) > 1e-4
                    for i in range(len(face_norms))
                ),
            },
            "face_landmarks_normalized": [
                {"index": i, "x": round(x, 6), "y": round(y, 6)}
                for i, (x, y) in enumerate(face_norms)
            ],
            "pose_landmarks_normalized": [
                {"index": i, "x": round(x, 6), "y": round(y, 6)}
                for i, (x, y) in enumerate(pose_norms)
            ],
            "landmarks_pixel": [
                {"index": i, "x": round(pt[0], 2), "y": round(pt[1], 2)}
                for i, pt in enumerate(pts)
            ],
            "triangles": tri_list,
            "edges": edge_list,
            "edge_styles": styles_out,
        }

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print_hardware_report()
    FaceMeshGUI().run()
