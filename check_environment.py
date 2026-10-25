"""Check the local Python and computer-vision environment.

This script is intentionally read-only. It does not install packages, create
directories, access the webcam, or download the MediaPipe model.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_PACKAGES = {
    "opencv-python": ("cv2", "OpenCV"),
    "mediapipe": ("mediapipe", "MediaPipe"),
    "numpy": ("numpy", "NumPy"),
    "scikit-learn": ("sklearn", "scikit-learn"),
    "joblib": ("joblib", "Joblib"),
}


def print_header(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def check_python() -> bool:
    print_header("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {platform.python_version()}")

    if sys.version_info[:2] != (3, 12):
        print("ERROR: This project is configured for Python 3.12.")
        return False

    print("OK: Python 3.12 detected.")
    return True


def check_packages() -> bool:
    print_header("Required packages")
    all_available = True

    for distribution_name, (module_name, display_name) in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = importlib.metadata.version(distribution_name)
            location = Path(module.__file__).resolve().parent
            print(f"OK: {display_name} {version} ({location})")
        except Exception as exc:  # Import errors can vary by package/platform.
            print(f"ERROR: Could not load {display_name}: {exc}")
            all_available = False

    return all_available


def check_mediapipe_api() -> bool:
    print_header("MediaPipe Tasks API")

    try:
        import mediapipe as mp

        has_tasks = hasattr(mp, "tasks")
        has_landmarker = has_tasks and hasattr(mp.tasks.vision, "HandLandmarker")
        has_options = has_tasks and hasattr(mp.tasks.vision, "HandLandmarkerOptions")

        print(f"MediaPipe Tasks namespace: {has_tasks}")
        print(f"HandLandmarker class:      {has_landmarker}")
        print(f"HandLandmarkerOptions:      {has_options}")

        if not (has_tasks and has_landmarker and has_options):
            print("ERROR: The installed MediaPipe package does not expose the required Tasks API.")
            return False

        print("OK: MediaPipe Hand Landmarker API is available.")
        return True
    except Exception as exc:
        print(f"ERROR: MediaPipe API check failed: {exc}")
        return False


def check_model_file() -> bool:
    print_header("Model file")
    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"

    if model_path.is_file():
        print(f"OK: Found {model_path}")
        return True

    print(f"INFO: Model file not found at {model_path}")
    print("      Download it using the command in README.md before running hand_landmarks.py.")
    return True


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    checks_passed = [
        check_python(),
        check_packages(),
        check_mediapipe_api(),
        check_model_file(),
    ]

    print_header("Summary")
    if all(checks_passed):
        print("Environment checks passed.")
        return 0

    print("One or more required environment checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
