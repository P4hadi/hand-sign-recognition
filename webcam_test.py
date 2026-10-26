"""Open a local webcam and display a preview.

This is a camera-only diagnostic. It does not save or upload frames.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2


WINDOW_TITLE = "Hand Sign Recognition - Webcam Test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a local webcam with OpenCV.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture | None:
    """Open a Windows camera using DirectShow, with a generic fallback."""

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def main() -> int:
    args = parse_args()
    camera = open_camera(args.camera_index, args.width, args.height)

    if camera is None:
        print(
            f"ERROR: Could not open camera index {args.camera_index}. "
            "Try another index, close other camera applications, and check Windows camera permissions.",
            file=sys.stderr,
        )
        return 1

    frame_count = 0
    start_time = time.perf_counter()
    print(f"Camera index {args.camera_index} opened. Press q or Esc in the window to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("ERROR: The camera opened, but a video frame could not be read.", file=sys.stderr)
                return 1

            frame_count += 1
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                frame,
                f"Camera OK | {fps:.1f} FPS | Press q or Esc to exit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except cv2.error as exc:
        print(f"ERROR: OpenCV could not display the webcam window: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Webcam test finished after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
