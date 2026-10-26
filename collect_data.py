"""Collect normalized hand-landmark samples from a local webcam.

Only landmark coordinates and lightweight collection metadata are written to
CSV. Raw webcam frames are never saved or uploaded by this script.

The collector stores one hand per sample. A sample is accepted only when
MediaPipe detects exactly one hand and all 21 landmarks contain finite values.
Landmarks are translated so the wrist is the origin and scaled by the largest
2-D distance from the wrist. This reduces sensitivity to image position and
hand distance while preserving the hand shape.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from hand_landmarks import draw_landmarks, open_camera
from gesture_config import (
    ALL_GESTURE_IDS,
    GESTURE_NAMES,
    LEGACY_GESTURE_IDS,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "landmarks_fresh.csv"
WINDOW_TITLE = "Hand Sign Recognition - Data Collection"

GESTURES = GESTURE_NAMES

LANDMARK_COLUMNS = [
    f"landmark_{index:02d}_{axis}"
    for index in range(21)
    for axis in ("x", "y", "z")
]
CSV_COLUMNS = [
    "label",
    "session_id",
    "user_id",
    "handedness",
    *LANDMARK_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect normalized 21-point hand landmarks into a CSV file."
    )
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--label",
        choices=ALL_GESTURE_IDS,
        help="Gesture to collect. If omitted, choose one interactively.",
    )
    label_group.add_argument(
        "--all-gestures",
        action="store_true",
        help=(
            "Collect the original five gestures sequentially. This preserves "
            "the historical behavior; use --gestures for an explicit list."
        ),
    )
    label_group.add_argument(
        "--gestures",
        nargs="+",
        choices=ALL_GESTURE_IDS,
        metavar="GESTURE",
        help=(
            "Explicit gesture IDs to collect in the supplied order. "
            "Repeat no IDs; use all eight IDs for the expanded sessions."
        ),
    )
    parser.add_argument(
        "--samples-per-gesture",
        type=int,
        default=200,
        help="New valid samples to collect per gesture in this run (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path inside the project directory (default: data\\landmarks_fresh.csv).",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help=(
            "Explicitly allow appending to an existing output CSV. "
            "Omit this flag for a protected fresh dataset."
        ),
    )
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
        "--user-id",
        default="user_01",
        help="Identifier used for session-aware data splitting (default: user_01).",
    )
    parser.add_argument(
        "--session-id",
        help="Session identifier. Defaults to the current UTC timestamp.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.15,
        help="Minimum seconds between accepted/attempted samples (default: 0.15).",
    )
    parser.add_argument(
        "--min-feature-distance",
        type=float,
        default=0.0,
        help=(
            "Optional minimum L2 distance from the last accepted sample for the "
            "same gesture. 0 disables this diversity gate (default: 0)."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum hand-detection confidence.",
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum hand-presence confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum hand-tracking confidence.",
    )
    return parser.parse_args()


def project_path(path: Path, description: str) -> Path:
    """Resolve a path and prevent this script from writing outside the project."""

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()

    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{description} must be inside {PROJECT_ROOT}: {resolved}")
    return resolved


def make_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session_{timestamp}"


def normalize_landmarks(landmarks: Iterable[object]) -> np.ndarray | None:
    """Return a finite, wrist-centred 63-value feature vector or ``None``."""

    points = list(landmarks)
    if len(points) != 21:
        return None

    try:
        coordinates = np.asarray(
            [[float(point.x), float(point.y), float(point.z)] for point in points],
            dtype=np.float32,
        )
    except (AttributeError, TypeError, ValueError):
        return None

    if coordinates.shape != (21, 3) or not np.isfinite(coordinates).all():
        return None

    relative = coordinates - coordinates[0]
    scale = float(np.max(np.linalg.norm(relative[:, :2], axis=1)))
    if not math.isfinite(scale) or scale <= 1e-6:
        return None

    normalized = relative / scale
    if not np.isfinite(normalized).all():
        return None

    return normalized.astype(np.float32).reshape(-1)


def extract_single_hand_features(result: object) -> tuple[np.ndarray | None, str]:
    """Extract features only when exactly one valid hand is present."""

    hands = getattr(result, "hand_landmarks", [])
    hand_count = len(hands)
    if hand_count == 0:
        return None, "No hand detected"
    if hand_count > 1:
        return None, "Show one hand only"

    features = normalize_landmarks(hands[0])
    if features is None:
        return None, "Invalid landmark sample"
    return features, "Hand ready"


def is_diverse_enough(
    features: np.ndarray,
    previous_features: np.ndarray | None,
    minimum_distance: float,
) -> bool:
    """Return whether a sample differs enough from the last accepted sample."""

    if minimum_distance <= 0 or previous_features is None:
        return True
    if features.shape != (63,) or previous_features.shape != (63,):
        return False
    if not np.isfinite(features).all() or not np.isfinite(previous_features).all():
        return False
    return float(np.linalg.norm(features - previous_features)) > minimum_distance


def get_handedness(result: object) -> str:
    """Return MediaPipe's handedness label when available."""

    handedness = getattr(result, "handedness", [])
    if not handedness or not handedness[0]:
        return "unknown"

    category = handedness[0][0]
    category_name = getattr(category, "category_name", None)
    return str(category_name).lower() if category_name else "unknown"


def feature_signature(label: str, features: np.ndarray) -> tuple[str, ...]:
    """Create a stable duplicate key using the CSV precision."""

    return (label, *(f"{float(value):.8f}" for value in features))


def validate_existing_csv(
    output_path: Path,
) -> tuple[set[tuple[str, ...]], Counter[str], int]:
    """Validate an existing collection file without modifying it.

    A malformed existing file stops collection rather than being silently
    repaired or partially appended to.
    """

    signatures: set[tuple[str, ...]] = set()
    counts: Counter[str] = Counter()
    duplicate_count = 0

    if not output_path.exists():
        return signatures, counts, duplicate_count
    if output_path.is_dir():
        raise ValueError(f"Output path is a directory, not a CSV file: {output_path}")
    if output_path.stat().st_size == 0:
        return signatures, counts, duplicate_count

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                "Existing CSV has an incompatible header. "
                "Use a new --output path; existing data was not changed."
            )

        for line_number, row in enumerate(reader, start=2):
            if row is None or any(row.get(column) in (None, "") for column in CSV_COLUMNS):
                raise ValueError(f"Incomplete record at CSV line {line_number}.")

            label = row["label"]
            if label not in GESTURES:
                raise ValueError(f"Unknown label at CSV line {line_number}: {label}")

            try:
                values = np.asarray([float(row[column]) for column in LANDMARK_COLUMNS])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric landmark value at CSV line {line_number}.") from exc

            if values.shape != (63,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid landmark vector at CSV line {line_number}.")

            signature = feature_signature(label, values)
            if signature in signatures:
                duplicate_count += 1
                continue
            signatures.add(signature)
            counts[label] += 1

    return signatures, counts, duplicate_count


def create_writer(
    output_path: Path,
    append_existing: bool,
) -> tuple[object, csv.DictWriter[str]]:
    """Create a fresh CSV or explicitly append to an existing validated CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not append_existing:
        raise FileExistsError(
            f"Refusing to append to existing CSV without --append-existing: {output_path}"
        )

    if append_existing and output_path.exists():
        needs_header = output_path.stat().st_size == 0
        file = output_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
            file.flush()
    else:
        # Exclusive creation prevents a race from turning a fresh collection
        # into an accidental append. The caller has already validated that the
        # path does not exist; mode "x" enforces that guarantee at the OS level.
        file = output_path.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        file.flush()
    return file, writer


def choose_labels(args: argparse.Namespace) -> list[str]:
    if args.gestures:
        if len(set(args.gestures)) != len(args.gestures):
            raise ValueError("--gestures cannot contain duplicate gesture IDs.")
        return list(args.gestures)
    if args.all_gestures:
        return list(LEGACY_GESTURE_IDS)
    if args.label:
        return [args.label]

    print("Choose a gesture:")
    labels = list(GESTURES)
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {GESTURES[label]} ({label})")

    try:
        selection = int(input("Enter a number: ").strip())
    except (EOFError, ValueError) as exc:
        raise ValueError("A valid gesture selection is required.") from exc

    if not 1 <= selection <= len(labels):
        raise ValueError("Gesture selection is outside the available choices.")
    return [labels[selection - 1]]


def append_sample(
    writer: csv.DictWriter[str],
    output_file: object,
    label: str,
    session_id: str,
    user_id: str,
    handedness: str,
    features: np.ndarray,
) -> None:
    """Validate one complete record immediately before writing it."""

    if label not in GESTURES:
        raise ValueError(f"Unsupported gesture label: {label}")
    if not session_id.strip() or not user_id.strip() or not handedness.strip():
        raise ValueError("Session, user, and handedness metadata must be non-empty")
    if features.shape != (63,) or not np.isfinite(features).all():
        raise ValueError("A sample must contain exactly 63 finite landmark features")

    row = {
        "label": label,
        "session_id": session_id,
        "user_id": user_id,
        "handedness": handedness,
    }
    row.update(
        {
            column: f"{float(value):.8f}"
            for column, value in zip(LANDMARK_COLUMNS, features, strict=True)
        }
    )
    writer.writerow(row)
    output_file.flush()


def put_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_collection_overlay(
    frame: np.ndarray,
    label: str,
    current_count: int,
    target_count: int,
    collecting: bool,
    status: str,
    total_count: int,
) -> None:
    """Render instructions and collection progress on the webcam frame."""

    height = frame.shape[0]
    state = "COLLECTING" if collecting else "PAUSED"
    state_color = (0, 255, 0) if collecting else (0, 165, 255)
    put_text(frame, f"Gesture: {GESTURES[label]} ({label})", 35, (255, 255, 0))
    put_text(frame, f"Progress: {current_count}/{target_count} | Total valid rows: {total_count}", 70, (255, 255, 255))
    put_text(frame, f"State: {state} | {status}", 105, state_color)
    put_text(frame, "S: start/resume  P: pause  Q/Esc: save and exit", height - 45, (255, 255, 255))
    put_text(frame, "Show exactly one hand; vary angle, distance, lighting, and background.", height - 15, (220, 220, 220))


def configure_landmarker(args: argparse.Namespace, model_path: Path) -> object:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def collect_for_labels(
    args: argparse.Namespace,
    labels: list[str],
    output_path: Path,
    model_path: Path,
    session_id: str,
    existing_signatures: set[tuple[str, ...]],
    counts: Counter[str],
    append_existing: bool,
) -> int:
    camera = open_camera(args.camera_index)
    if camera is None:
        print(
            f"ERROR: Could not open camera index {args.camera_index}. "
            "Check Windows camera permissions or try another index.",
            file=sys.stderr,
        )
        return 1

    try:
        output_file, writer = create_writer(output_path, append_existing)
    except (FileExistsError, OSError) as exc:
        camera.release()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    total_count = sum(counts.values())
    active_labels = labels
    new_counts: Counter[str] = Counter()

    mode = "append mode" if append_existing else "fresh exclusive mode"
    print(f"Writing to: {output_path} ({mode})")
    print(f"Existing valid rows before this run: {total_count}")
    print(f"Session: {session_id} | User: {args.user_id}")
    print("The window starts paused. Press S after positioning one hand in view.")

    try:
        with configure_landmarker(args, model_path) as landmarker:
            for label_index, label in enumerate(active_labels):
                collecting = False
                status = "Press S to start"
                last_attempt_time = 0.0
                last_accepted_features: np.ndarray | None = None

                while True:
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        print("ERROR: A video frame could not be read.", file=sys.stderr)
                        return 1

                    frame = cv2.flip(frame, 1)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect(mp_image)
                    features, detection_status = extract_single_hand_features(result)

                    # Reuse the detector's landmark rendering for visual confirmation.
                    draw_landmarks(frame, result)

                    now = cv2.getTickCount() / cv2.getTickFrequency()
                    if collecting and now - last_attempt_time >= args.sample_interval:
                        last_attempt_time = now
                        if features is None:
                            status = detection_status
                        else:
                            signature = feature_signature(label, features)
                            if signature in existing_signatures:
                                status = "Duplicate skipped - move hand slightly"
                            elif not is_diverse_enough(
                                features,
                                last_accepted_features,
                                args.min_feature_distance,
                            ):
                                status = "Too similar - vary hand angle, position, or distance"
                            else:
                                append_sample(
                                    writer,
                                    output_file,
                                    label,
                                    session_id,
                                    args.user_id,
                                    get_handedness(result),
                                    features,
                                )
                                existing_signatures.add(signature)
                                last_accepted_features = features.copy()
                                counts[label] += 1
                                new_counts[label] += 1
                                total_count += 1
                                status = f"Saved sample {new_counts[label]}/{args.samples_per_gesture}"

                    draw_collection_overlay(
                        frame,
                        label,
                        new_counts[label],
                        args.samples_per_gesture,
                        collecting,
                        status,
                        total_count,
                    )
                    cv2.imshow(WINDOW_TITLE, frame)

                    if new_counts[label] >= args.samples_per_gesture:
                        print(
                            f"Completed {GESTURES[label]}: "
                            f"{new_counts[label]} new samples; {counts[label]} valid samples total."
                        )
                        break

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("Collection stopped by user. Samples already written remain intact.")
                        return 0
                    if key == ord("s"):
                        collecting = True
                        status = "Collecting when one valid hand is visible"
                    elif key == ord("p"):
                        collecting = False
                        status = "Paused - press S to resume"

                if label_index < len(active_labels) - 1:
                    print(f"Next gesture: {GESTURES[active_labels[label_index + 1]]}. Press S in the window to begin.")

    except (cv2.error, RuntimeError, ValueError) as exc:
        print(f"ERROR: Data collection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        output_file.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"Collection complete. CSV: {output_path}")
    return 0


def main() -> int:
    args = parse_args()

    if args.samples_per_gesture <= 0:
        print("ERROR: --samples-per-gesture must be greater than zero.", file=sys.stderr)
        return 2
    if args.sample_interval < 0:
        print("ERROR: --sample-interval cannot be negative.", file=sys.stderr)
        return 2
    if args.min_feature_distance < 0:
        print("ERROR: --min-feature-distance cannot be negative.", file=sys.stderr)
        return 2
    if not args.user_id.strip():
        print("ERROR: --user-id cannot be empty.", file=sys.stderr)
        return 2

    try:
        output_path = project_path(args.output, "Output path")
        model_path = project_path(args.model, "Model path")
        labels = choose_labels(args)

        if output_path.exists() and not args.append_existing:
            raise ValueError(
                f"Refusing to use existing output without --append-existing: {output_path}. "
                "Choose a new path or explicitly opt in to appending."
            )

        existing_signatures, counts, duplicate_count = validate_existing_csv(output_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not model_path.is_file():
        print(
            f"ERROR: MediaPipe model file was not found: {model_path}\n"
            "Download it using the README instructions before collecting data.",
            file=sys.stderr,
        )
        return 2

    if duplicate_count:
        print(
            f"WARNING: Existing CSV contains {duplicate_count} duplicate row(s). "
            "They were preserved; new duplicate rows will be skipped.",
            file=sys.stderr,
        )

    session_id = args.session_id.strip() if args.session_id else make_session_id()
    if not session_id:
        print("ERROR: --session-id cannot be empty.", file=sys.stderr)
        return 2

    return collect_for_labels(
        args,
        labels,
        output_path,
        model_path,
        session_id,
        existing_signatures,
        counts,
        args.append_existing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
