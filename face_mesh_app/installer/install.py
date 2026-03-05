"""
install.py – Headless CLI installer for the Facial Network Mesh app.

Steps
-----
1. Check Python version
2. pip install -r requirements.txt
3. Validate critical imports
4. Download face_landmarker.task model (if missing)
5. Print hardware / GPU report
"""

import importlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent / "app"
_REQ_FILE = _HERE.parent / "requirements.txt"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DEST = _APP_DIR / "face_landmarker.task"


def _check_python_version():
    print(f"Python : {sys.version}")
    if sys.version_info < (3, 9):
        print("  \u26a0  Python 3.9+ is recommended.")
    else:
        print("  \u2713  OK")


def _install_requirements():
    if not _REQ_FILE.exists():
        print(f"\u2717  requirements.txt not found at {_REQ_FILE}")
        return False
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(_REQ_FILE)]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n\u2717  pip exited with code {result.returncode}")
        return False
    print("\n\u2713  All dependencies installed.")
    return True


def _validate_imports():
    modules = ["mediapipe", "cv2", "numpy", "scipy", "PIL"]
    all_ok = True
    print("\nValidating imports:")
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  \u2713  {mod}")
        except ImportError:
            print(f"  \u2717  {mod} — NOT FOUND")
            all_ok = False
    return all_ok


def _download_model():
    if MODEL_DEST.exists() and MODEL_DEST.stat().st_size > 1_000_000:
        size_mb = MODEL_DEST.stat().st_size / 1_048_576
        print(f"\n\u2713  Model already present ({size_mb:.1f} MB)")
        return True
    print(f"\nDownloading model …\n  {MODEL_URL}")
    try:
        os.makedirs(MODEL_DEST.parent, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_DEST))
        size_mb = MODEL_DEST.stat().st_size / 1_048_576
        print(f"\u2713  Downloaded model ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"\u2717  Download failed: {exc}")
        return False


def _run_hardware_report():
    try:
        sys.path.insert(0, str(_APP_DIR))
        from gpu_check import print_hardware_report
        print()
        print_hardware_report()
    except Exception as exc:
        print(f"\n(hardware report skipped: {exc})")


def main():
    print("=" * 50)
    print("  Facial Network Mesh — Installer")
    print("=" * 50)

    _check_python_version()
    ok = _install_requirements()
    ok = _validate_imports() and ok
    ok = _download_model() and ok
    _run_hardware_report()

    print("\n" + "=" * 50)
    if ok:
        print("\u2713  Installation complete!  Run facial_network.bat to start.")
    else:
        print("\u2717  Completed with errors.  Check messages above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
