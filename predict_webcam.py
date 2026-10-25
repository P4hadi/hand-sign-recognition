"""Run realtime hand-gesture prediction from a local webcam.

The script loads the saved Random Forest artifact and never retrains it. It
reuses the collector's wrist-centred, scale-normalized landmark conversion so
the runtime feature vector has the same 63-value order used during training.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np

from collect_data import normalize_landmarks
from hand_landmarks import draw_landmarks, open_camera
from train_model import FEATURE_COLUMNS, SUPPORTED_LABELS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gesture_rf_baseline_v1.joblib"
DEFAULT_HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
WINDOW_TITLE = "Hand Sign Recognition - Realtime Prediction"


class ModelValidationError(ValueError):
    """Raised when a saved classifier is incompatible with this application."""


@dataclass(frozen=True)
class LoadedModel:
    estimator: Any
    labels: tuple[str, ...]
    feature_columns: tuple[str, ...]
    probability_supported: bool


@dataclass(frozen=True)
class FramePrediction:
    label: str
    confidence: float | None


class TimingStats:
    """Collect per-stage timing diagnostics for a live run."""

    STAGES = (
        "camera_read",
        "frame_prepare",
        "mediapipe",
        "features",
        "classifier",
        "drawing",
        "display",
        "total",
    )

    def __init__(self, report_interval: float = 5.0) -> None:
        if report_interval < 0:
            raise ValueError("report_interval cannot be negative")
        self.report_interval = report_interval
        self._started_at = time.perf_counter()
        self._last_report_at = self._started_at
        self._frame_count = 0
        self._totals = {stage: 0.0 for stage in self.STAGES}

    def record(self, timings: dict[str, float]) -> None:
        self._frame_count += 1
        for stage in self.STAGES:
            self._totals[stage] += timings.get(stage, 0.0)

    def report(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if self._frame_count == 0:
            return
        if not force and (self.report_interval == 0 or now - self._last_report_at < self.report_interval):
            return

        elapsed = now - self._last_report_at
        average_ms = {
            stage: self._totals[stage] / self._frame_count * 1000
            for stage in self.STAGES
        }
        interval_fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        print(
            "TIMING "
            f"frames={self._frame_count} interval_fps={interval_fps:.2f} "
            + " ".join(f"{stage}_ms={average_ms[stage]:.2f}" for stage in self.STAGES)
        )
        self._last_report_at = now
        self._frame_count = 0
        self._totals = {stage: 0.0 for stage in self.STAGES}


class PredictionSmoother:
    """Return the majority label from a short history of valid predictions."""

    def __init__(self, window_size: int = 7, labels: tuple[str, ...] = SUPPORTED_LABELS) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if not labels:
            raise ValueError("labels must not be empty")
        self.labels = tuple(labels)
        self._history: deque[str] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, label: str) -> str:
        if label not in self.labels:
            raise ValueError(f"Unsupported prediction label: {label}")
        self._history.append(label)
        counts = Counter(self._history)
        highest_count = max(counts.values())
        # In a tie, prefer the most recent label to keep the display responsive.
        for recent_label in reversed(self._history):
            if counts[recent_label] == highest_count:
                return recent_label
        raise RuntimeError("Prediction history unexpectedly empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a gesture from the labels stored in a saved model bundle."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved Joblib classifier bundle (default: models\\gesture_rf_baseline_v1.joblib).",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_MODEL_PATH,
        help="MediaPipe hand-landmarker.task path.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested webcam width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested webcam height (default: 480).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Number of valid predictions used for majority smoothing (default: 7).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Seconds between per-stage timing reports; 0 disables reports (default: 5).",
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


def resolve_project_path(path: Path, description: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ModelValidationError(f"{description} must be inside {PROJECT_ROOT}: {resolved}")
    return resolved


def _estimator_classes(estimator: Any) -> tuple[str, ...] | None:
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    if classes is None:
        return None
    return tuple(str(value) for value in classes)


def _estimator_feature_count(estimator: Any) -> int | None:
    count = getattr(estimator, "n_features_in_", None)
    if count is None and hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            count = getattr(step, "n_features_in_", None)
            if count is not None:
                break
    return int(count) if count is not None else None


def load_model_bundle(model_path: Path) -> LoadedModel:
    """Load and validate the saved classifier contract before prediction."""

    if not model_path.is_file():
        raise ModelValidationError(f"Classifier model was not found: {model_path}")

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise ModelValidationError(f"Could not load classifier model: {exc}") from exc

    if not isinstance(bundle, dict):
        raise ModelValidationError("Classifier artifact must be a dictionary bundle.")

    estimator = bundle.get("model")
    labels = tuple(bundle.get("labels", ()))
    feature_columns = tuple(bundle.get("feature_columns", ()))
    if estimator is None or not hasattr(estimator, "predict"):
        raise ModelValidationError("Classifier bundle does not contain a usable model.")
    if not labels or len(set(labels)) != len(labels):
        raise ModelValidationError("Classifier label mapping must contain unique labels.")
    if feature_columns != FEATURE_COLUMNS:
        raise ModelValidationError("Classifier feature-column order does not match training.")

    feature_count = _estimator_feature_count(estimator)
    if feature_count != len(FEATURE_COLUMNS):
        raise ModelValidationError(
            f"Classifier expects {feature_count} features; expected {len(FEATURE_COLUMNS)}."
        )

    estimator_classes = _estimator_classes(estimator)
    if estimator_classes is not None and tuple(sorted(estimator_classes)) != tuple(sorted(labels)):
        raise ModelValidationError(
            f"Classifier classes do not match supported labels: {estimator_classes!r}"
        )

    # The saved baseline was trained with n_jobs=-1. For one-sample realtime
    # predictions, thread-pool scheduling can cost more than tree evaluation.
    # This changes only the in-memory loaded estimator, never the saved file.
    for step in getattr(estimator, "named_steps", {}).values():
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1

    probability_supported = callable(getattr(estimator, "predict_proba", None))
    return LoadedModel(
        estimator=estimator,
        labels=labels,
        feature_columns=feature_columns,
        probability_supported=probability_supported,
    )


def extract_model_features(result: object) -> tuple[np.ndarray | None, str]:
    """Extract one valid 63-value vector, rejecting zero or multiple hands."""

    hands = getattr(result, "hand_landmarks", [])
    if len(hands) == 0:
        return None, "No hand detected"
    if len(hands) > 1:
        return None, "Show one hand only"

    features = normalize_landmarks(hands[0])
    if features is None or features.shape != (len(FEATURE_COLUMNS),):
        return None, "Invalid landmark data"
    if not np.isfinite(features).all():
        return None, "Invalid landmark data"
    return features, "Hand detected"


def _probabilities_for_features(estimator: Any, features: np.ndarray) -> tuple[tuple[str, ...], np.ndarray] | None:
    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        return None

    try:
        probabilities = np.asarray(predict_proba(features.reshape(1, -1)), dtype=float)
        classes = _estimator_classes(estimator)
        if classes is None or probabilities.shape[0] != 1 or probabilities.shape[1] != len(classes):
            return None
        row = probabilities[0]
        if not np.isfinite(row).all() or (row < 0).any() or (row > 1).any():
            return None
        return classes, row
    except (IndexError, TypeError, ValueError):
        return None


def predict_features(model: LoadedModel, features: np.ndarray) -> FramePrediction:
    """Predict a supported label from exactly 63 finite features."""

    values = np.asarray(features, dtype=np.float32)
    if values.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(values).all():
        raise ValueError("Prediction requires exactly 63 finite landmark features.")

    probability_result = _probabilities_for_features(model.estimator, values)
    if probability_result is not None:
        classes, probabilities = probability_result
        class_index = int(np.argmax(probabilities))
        raw_label = classes[class_index]
        confidence = float(probabilities[class_index])
    else:
        raw_label = str(model.estimator.predict(values.reshape(1, -1))[0])
        confidence = None

    if raw_label not in model.labels:
        raise ModelValidationError(f"Model returned unsupported label: {raw_label}")
    return FramePrediction(label=raw_label, confidence=confidence)


def predict_result(model: LoadedModel, result: object) -> tuple[FramePrediction | None, str]:
    """Return no prediction for missing/ambiguous/invalid hand landmarks."""

    features, status = extract_model_features(result)
    if features is None:
        return None, status
    return predict_features(model, features), status


def configure_landmarker(args: argparse.Namespace, hand_model_path: Path) -> object:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(hand_model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def configure_camera(camera: cv2.VideoCapture, width: int, height: int) -> tuple[int, int]:
    """Request a smaller frame and return the backend's actual dimensions."""

    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return actual_width, actual_height


def draw_prediction_overlay(
    frame: np.ndarray,
    status: str,
    smoothed_label: str | None,
    frame_prediction: FramePrediction | None,
    fps: float,
) -> None:
    if smoothed_label is None:
        prediction_text = "Prediction: --"
    else:
        prediction_text = f"Prediction: {smoothed_label}"
    cv2.putText(
        frame,
        prediction_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0) if smoothed_label else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    if frame_prediction is not None and frame_prediction.confidence is not None:
        confidence_text = (
            f"Current model probability estimate: {frame_prediction.confidence:.2f} "
            f"({frame_prediction.label})"
        )
    elif frame_prediction is not None:
        confidence_text = f"Current model output: {frame_prediction.label} (no probability API)"
    else:
        confidence_text = "Current model probability estimate: --"

    cv2.putText(
        frame,
        confidence_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Status: {status} | {fps:.1f} FPS",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q or Esc: quit | Probability is a model estimate, not guaranteed accuracy",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    try:
        model_path = resolve_project_path(args.model, "Classifier model path")
        hand_model_path = resolve_project_path(args.hand_model, "Hand landmark model path")
        if args.smoothing_window <= 0:
            raise ValueError("--smoothing-window must be greater than zero.")
        if args.stats_interval < 0:
            raise ValueError("--stats-interval cannot be negative.")
        model = load_model_bundle(model_path)
    except (ModelValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not hand_model_path.is_file():
        print(f"ERROR: MediaPipe hand-landmarker model was not found: {hand_model_path}", file=sys.stderr)
        return 1

    camera = open_camera(args.camera_index)
    if camera is None:
        print(
            f"ERROR: Could not open camera index {args.camera_index}. "
            "Check Windows camera permissions or try another index.",
            file=sys.stderr,
        )
        return 1

    try:
        actual_width, actual_height = configure_camera(camera, args.width, args.height)
    except ValueError as exc:
        camera.release()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    smoother = PredictionSmoother(args.smoothing_window, model.labels)
    timing = TimingStats(args.stats_interval)
    frame_count = 0
    started_at = time.perf_counter()
    print(f"Classifier: {model_path}")
    print(f"Hand model: {hand_model_path}")
    print(f"Requested camera size: {args.width}x{args.height}; actual: {actual_width}x{actual_height}")
    print("MediaPipe mode: VIDEO; realtime classifier inference uses one CPU worker")
    print(f"Camera index {args.camera_index} opened. Press q or Esc in the window to exit.")

    try:
        with configure_landmarker(args, hand_model_path) as landmarker:
            previous_timestamp_ms = -1
            while True:
                frame_started_at = time.perf_counter()
                read_started_at = frame_started_at
                ok, frame = camera.read()
                read_finished_at = time.perf_counter()
                if not ok or frame is None:
                    print("ERROR: The camera opened, but a video frame could not be read.", file=sys.stderr)
                    return 1

                frame_count += 1
                frame_prepare_started_at = read_finished_at
                frame = cv2.flip(frame, 1)
                rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_prepare_finished_at = time.perf_counter()

                # VIDEO mode requires monotonically increasing timestamps.
                timestamp_ms = int((time.perf_counter() - started_at) * 1000)
                if frame_count > 1 and timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1
                previous_timestamp_ms = timestamp_ms
                mediapipe_started_at = frame_prepare_finished_at
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                mediapipe_finished_at = time.perf_counter()

                feature_started_at = mediapipe_finished_at
                features, status = extract_model_features(result)
                feature_finished_at = time.perf_counter()
                drawing_started_at = feature_finished_at
                draw_landmarks(frame, result)
                drawing_finished_at = time.perf_counter()

                try:
                    classifier_started_at = feature_finished_at
                    frame_prediction = (
                        predict_features(model, features)
                        if features is not None
                        else None
                    )
                    classifier_finished_at = time.perf_counter()
                except (ModelValidationError, ValueError, RuntimeError) as exc:
                    print(f"ERROR: Prediction failed: {exc}", file=sys.stderr)
                    return 1

                if frame_prediction is None:
                    # Do not display a stale label when no usable hand is present.
                    smoother.reset()
                    smoothed_label = None
                else:
                    smoothed_label = smoother.update(frame_prediction.label)

                elapsed = time.perf_counter() - started_at
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                draw_prediction_overlay(frame, status, smoothed_label, frame_prediction, fps)
                display_started_at = time.perf_counter()
                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                display_finished_at = time.perf_counter()
                timing.record(
                    {
                        "camera_read": read_finished_at - read_started_at,
                        "frame_prepare": frame_prepare_finished_at - frame_prepare_started_at,
                        "mediapipe": mediapipe_finished_at - mediapipe_started_at,
                        "features": feature_finished_at - feature_started_at,
                        "classifier": classifier_finished_at - classifier_started_at,
                        "drawing": drawing_finished_at - drawing_started_at,
                        "display": display_finished_at - display_started_at,
                        "total": display_finished_at - frame_started_at,
                    }
                )
                timing.report()
                if key in (ord("q"), 27):
                    break
    except (cv2.error, RuntimeError, ValueError) as exc:
        print(f"ERROR: Realtime prediction failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    timing.report(force=True)
    print(f"Realtime prediction stopped after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
