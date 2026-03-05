"""
install_gui.py – Tkinter GUI installer for the Facial Network Mesh app.

Runs pip install, validates imports, downloads the face_landmarker model,
and displays a hardware report — all within a graphical interface.
"""

import importlib
import os
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import scrolledtext, ttk

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent / "app"
_REQ_FILE = _HERE.parent / "requirements.txt"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DEST = _APP_DIR / "face_landmarker.task"


class InstallerGUI:
    STEPS = 5

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Mesh Installer")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("680x520")
        self.root.minsize(500, 400)

        self._build_ui()
        self._centre()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4",
                         font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e2e",
                         foreground="#89b4fa",
                         font=("Segoe UI", 16, "bold"))
        style.configure("Accent.TButton", background="#89b4fa",
                         foreground="#1e1e2e",
                         font=("Segoe UI", 11, "bold"), padding=8)
        style.map("Accent.TButton",
                  background=[("active", "#74c7ec"),
                              ("disabled", "#45475a")])

        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(14, 4))
        ttk.Label(header, text="Face Mesh Installer",
                  style="Title.TLabel").pack(side="left")

        self.progress = ttk.Progressbar(self.root, mode="determinate",
                                         maximum=self.STEPS)
        self.progress.pack(fill="x", padx=16, pady=(6, 4))

        self.status_var = tk.StringVar(value="Press Install to begin")
        ttk.Label(self.root, textvariable=self.status_var,
                  style="TLabel").pack(anchor="w", padx=16)

        self.log = scrolledtext.ScrolledText(
            self.root, bg="#11111b", fg="#a6adc8",
            font=("Consolas", 9), wrap="word",
            insertbackground="#cdd6f4", state="disabled",
            height=18)
        self.log.pack(fill="both", expand=True, padx=16, pady=(6, 6))

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=16, pady=(0, 14))
        self.install_btn = ttk.Button(btn_frame, text="Install",
                                       style="Accent.TButton",
                                       command=self._start_install)
        self.install_btn.pack(side="left")
        ttk.Button(btn_frame, text="Close", style="Accent.TButton",
                   command=self.root.destroy).pack(side="right")

    def _centre(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    # ── Logging helpers ──────────────────────────────────────────────────

    def _append(self, text: str):
        self.root.after(0, self._append_ui, text)

    def _append_ui(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_step(self, n: int, msg: str):
        self.root.after(0, self._set_step_ui, n, msg)

    def _set_step_ui(self, n: int, msg: str):
        self.progress["value"] = n
        self.status_var.set(msg)

    # ── Install pipeline (background thread) ─────────────────────────────

    def _start_install(self):
        self.install_btn.configure(state="disabled")
        threading.Thread(target=self._install_pipeline, daemon=True).start()

    def _install_pipeline(self):
        ok = True
        self._set_step(0, "Checking Python …")
        ok = ok and self._step_check_python()

        self._set_step(1, "Installing dependencies …")
        ok = ok and self._step_pip_install()

        self._set_step(2, "Validating imports …")
        ok = ok and self._step_validate()

        self._set_step(3, "Downloading model …")
        ok = ok and self._step_download_model()

        self._set_step(4, "Hardware report …")
        self._step_hardware()

        self._set_step(self.STEPS,
                       "\u2713  Installation complete!" if ok
                       else "\u2717  Completed with errors")
        self._append("\n" + ("=" * 50))
        self._append("Done.  You may close this window.")
        self.root.after(0, lambda: self.install_btn.configure(state="normal"))

    def _step_check_python(self) -> bool:
        v = sys.version
        self._append(f"Python version: {v}")
        major, minor = sys.version_info[:2]
        if major < 3 or minor < 9:
            self._append("\u26a0  Python 3.9+ recommended!")
            return False
        self._append("\u2713  Python OK")
        return True

    def _step_pip_install(self) -> bool:
        if not _REQ_FILE.exists():
            self._append(f"\u2717  requirements.txt not found at {_REQ_FILE}")
            return False
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(_REQ_FILE),
               "--quiet"]
        self._append(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=300)
            if result.stdout.strip():
                self._append(result.stdout.strip())
            if result.returncode != 0:
                self._append(f"\u2717  pip exited with code {result.returncode}")
                if result.stderr.strip():
                    self._append(result.stderr.strip())
                return False
            self._append("\u2713  Dependencies installed")
            return True
        except subprocess.TimeoutExpired:
            self._append("\u2717  pip install timed out (300 s)")
            return False
        except Exception as exc:
            self._append(f"\u2717  pip error: {exc}")
            return False

    def _step_validate(self) -> bool:
        modules = ["mediapipe", "cv2", "numpy", "scipy", "PIL"]
        all_ok = True
        for mod in modules:
            try:
                importlib.import_module(mod)
                self._append(f"  \u2713  {mod}")
            except ImportError:
                self._append(f"  \u2717  {mod} — NOT FOUND")
                all_ok = False
        return all_ok

    def _step_download_model(self) -> bool:
        if MODEL_DEST.exists() and MODEL_DEST.stat().st_size > 1_000_000:
            size_mb = MODEL_DEST.stat().st_size / 1_048_576
            self._append(
                f"\u2713  Model already exists ({size_mb:.1f} MB)")
            return True

        self._append(f"Downloading model from:\n  {MODEL_URL}")
        try:
            os.makedirs(MODEL_DEST.parent, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_DEST))
            size_mb = MODEL_DEST.stat().st_size / 1_048_576
            self._append(f"\u2713  Model downloaded ({size_mb:.1f} MB)")
            return True
        except Exception as exc:
            self._append(f"\u2717  Download failed: {exc}")
            return False

    def _step_hardware(self):
        self._append("\n--- Hardware Report ---")
        try:
            sys.path.insert(0, str(_APP_DIR))
            from gpu_check import gather_full_report
            info = gather_full_report()
            self._append(f"OS       : {info['system']['os']}")
            self._append(f"Arch     : {info['system']['arch']}")
            self._append(f"Python   : {info['system']['python']}")
            gpu = info.get("gpu")
            if gpu and gpu.get("name"):
                self._append(f"GPU      : {gpu['name']}")
                self._append(f"  VRAM   : {gpu.get('memory_mb', '?')} MB")
                self._append(f"  Driver : {gpu.get('driver', '?')}")
            else:
                self._append("GPU      : not detected (CPU mode)")
            self._append(f"CUDA     : {'yes' if info.get('cuda_available') else 'no'}")
        except Exception as exc:
            self._append(f"  (could not gather hardware info: {exc})")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    InstallerGUI().run()
