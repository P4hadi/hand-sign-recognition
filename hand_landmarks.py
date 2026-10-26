"""Detect and draw hand landmarks from a local webcam.

The MediaPipe Hand Landmarker Tasks API requires a local
``models/hand_landmarker.task`` model bundle. See README.md for the official
free download command.

This stage only detects landmarks. It does not classify gestures or save frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
WINDOW_TITLE = "Hand Sign Recognition - Hand Landmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect hand landmarks from a webcam.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand_landmarker.task file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        choices=range(1, 3),
        help="Maximum number of hands to detect (1 or 2).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for initial hand detection.",
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand presence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand tracking.",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture | None:
    """Open a Windows camera using DirectShow, with a generic fallback."""

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        return None

    return camera


def normalized_to_pixel(landmark: object, width: int, height: int) -> tuple[int, int]:
    """Convert a MediaPipe normalized landmark to a bounded image coordinate."""

    x = int(np.clip(float(landmark.x) * width, 0, width - 1))
    y = int(np.clip(float(landmark.y) * height, 0, height - 1))
    return x, y


def draw_landmarks(frame: np.ndarray, result: object) -> None:
    """Draw landmark points, connections, and handedness on a BGR frame."""

    height, width = frame.shape[:2]
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    hand_landmarks = getattr(result, "hand_landmarks", [])
    handedness = getattr(result, "handedness", [])

    for hand_index, landmarks in enumerate(hand_landmarks):
        points = [normalized_to_pixel(landmark, width, height) for landmark in landmarks]

        for connection in connections:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (0, 220, 0), 2, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

        label = f"Hand {hand_index + 1}"
        if hand_index < len(handedness) and handedness[hand_index]:
            category = handedness[hand_index][0]
            category_name = getattr(category, "category_name", None)
            category_score = getattr(category, "score", None)
            if category_name:
                label = f"{category_name}"
                if category_score is not None:
                    label += f" ({float(category_score):.2f})"

        wrist_x, wrist_y = points[0]
        cv2.putText(
            frame,
            label,
            (wrist_x + 10, max(25, wrist_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    if not model_path.is_file():
        print(
            f"ERROR: MediaPipe model file was not found: {model_path}\n"
            "Download it using the command in README.md, or pass a valid path with --model.",
            file=sys.stderr,
        )
        return 1

    camera = open_camera(args.camera_index)
    if camera is None:
        print(
            f"ERROR: Could not open camera index {args.camera_index}. "
            "Try another index and check Windows camera permissions.",
            file=sys.stderr,
        )
        return 1

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print(f"Using model: {model_path}")
    print(f"Camera index {args.camera_index} opened. Press q or Esc in the window to exit.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("ERROR: A video frame could not be read.", file=sys.stderr)
                    return 1

                # Mirror the view so it behaves like a normal webcam preview.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect(mp_image)

                draw_landmarks(frame, result)
                hand_count = len(getattr(result, "hand_landmarks", []))
                status = f"Hands detected: {hand_count} | Press q or Esc to exit"
                status_color = (0, 255, 0) if hand_count else (0, 165, 255)
                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except (cv2.error, RuntimeError, ValueError) as exc:
        print(f"ERROR: Realtime hand-landmark detection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Hand-landmark test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
