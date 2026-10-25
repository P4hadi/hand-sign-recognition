"""Webcam-independent tests for realtime prediction helpers."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collect_data import is_diverse_enough, normalize_landmarks
from gesture_config import ALL_GESTURE_IDS
from predict_webcam import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    SUPPORTED_LABELS,
    FramePrediction,
    LoadedModel,
    PredictionSmoother,
    extract_model_features,
    load_model_bundle,
    predict_features,
    predict_result,
)


def fake_landmarks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=float(index % 5) / 10.0,
            y=float(index // 5) / 10.0,
            z=float(index) / 100.0,
        )
        for index in range(21)
    ]


class FakeEstimator:
    """Small deterministic estimator used without a webcam or retraining."""

    n_features_in_ = 63
    classes_ = np.asarray(SUPPORTED_LABELS)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.last_shape = features.shape
        return np.asarray(["peace_sign"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.last_probability_shape = features.shape
        probabilities = np.zeros((1, len(SUPPORTED_LABELS)), dtype=float)
        probabilities[0, SUPPORTED_LABELS.index("peace_sign")] = 0.8
        probabilities[0, SUPPORTED_LABELS.index("ok_sign")] = 0.2
        return probabilities


class PredictWebcamTests(unittest.TestCase):
    def test_feature_shape_and_order(self) -> None:
        features = normalize_landmarks(fake_landmarks())
        self.assertIsNotNone(features)
        assert features is not None
        self.assertEqual(features.shape, (63,))
        self.assertEqual(FEATURE_COLUMNS[0], "landmark_00_x")
        self.assertEqual(FEATURE_COLUMNS[1], "landmark_00_y")
        self.assertEqual(FEATURE_COLUMNS[2], "landmark_00_z")
        self.assertEqual(FEATURE_COLUMNS[-1], "landmark_20_z")
        self.assertTrue(np.isfinite(features).all())

    def test_collected_csv_row_matches_runtime_normalization(self) -> None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "landmarks_fresh.csv"
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))

        points = [
            SimpleNamespace(
                x=float(row[f"landmark_{index:02d}_x"]),
                y=float(row[f"landmark_{index:02d}_y"]),
                z=float(row[f"landmark_{index:02d}_z"]),
            )
            for index in range(21)
        ]
        runtime_features = normalize_landmarks(points)
        stored_features = np.asarray(
            [float(row[column]) for column in FEATURE_COLUMNS],
            dtype=np.float32,
        )

        self.assertIsNotNone(runtime_features)
        assert runtime_features is not None
        np.testing.assert_allclose(runtime_features, stored_features, rtol=0, atol=1e-7)
        self.assertIn(row["handedness"], {"left", "right"})

    def test_saved_model_loads_with_expected_contract(self) -> None:
        self.assertTrue(Path(DEFAULT_MODEL_PATH).is_file())
        loaded = load_model_bundle(DEFAULT_MODEL_PATH)
        self.assertEqual(loaded.labels, SUPPORTED_LABELS)
        self.assertEqual(loaded.feature_columns, FEATURE_COLUMNS)
        self.assertTrue(loaded.probability_supported)

    def test_final_v2_model_loads_with_expected_contract(self) -> None:
        final_path = Path(__file__).resolve().parents[1] / "models" / "gesture_rf_v2.joblib"
        self.assertTrue(final_path.is_file())
        loaded = load_model_bundle(final_path)
        self.assertEqual(loaded.labels, SUPPORTED_LABELS)
        self.assertEqual(loaded.feature_columns, FEATURE_COLUMNS)
        self.assertTrue(loaded.probability_supported)

    def test_final_v3_eight_gesture_model_loads_with_actual_mapping(self) -> None:
        final_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "gesture_rf_v3_eight_gestures.joblib"
        )
        self.assertTrue(final_path.is_file())
        loaded = load_model_bundle(final_path)
        self.assertEqual(loaded.labels, ALL_GESTURE_IDS)
        self.assertEqual(loaded.feature_columns, FEATURE_COLUMNS)
        self.assertTrue(loaded.probability_supported)

    def test_session06_candidate_and_final_models_load_with_eight_mapping(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "gesture_rf_session04_plus_session06_eval_eight_v1.joblib",
            "gesture_rf_v4_eight_gestures_session06.joblib",
        ):
            loaded = load_model_bundle(root / "models" / filename)
            self.assertEqual(loaded.labels, ALL_GESTURE_IDS)
            self.assertEqual(loaded.feature_columns, FEATURE_COLUMNS)
            self.assertTrue(loaded.probability_supported)

    def test_smoother_accepts_labels_from_loaded_eight_class_model(self) -> None:
        smoother = PredictionSmoother(window_size=3, labels=ALL_GESTURE_IDS)
        self.assertEqual(smoother.update("call_me"), "call_me")

    def test_no_hand_returns_no_prediction(self) -> None:
        loaded = LoadedModel(
            estimator=FakeEstimator(),
            labels=SUPPORTED_LABELS,
            feature_columns=FEATURE_COLUMNS,
            probability_supported=True,
        )
        result = SimpleNamespace(hand_landmarks=[])

        prediction, status = predict_result(loaded, result)

        self.assertIsNone(prediction)
        self.assertEqual(status, "No hand detected")

    def test_invalid_landmark_count_returns_no_prediction(self) -> None:
        loaded = LoadedModel(
            estimator=FakeEstimator(),
            labels=SUPPORTED_LABELS,
            feature_columns=FEATURE_COLUMNS,
            probability_supported=True,
        )
        result = SimpleNamespace(hand_landmarks=[fake_landmarks()[:20]])

        prediction, status = predict_result(loaded, result)

        self.assertIsNone(prediction)
        self.assertEqual(status, "Invalid landmark data")

    def test_prediction_mapping_and_probability(self) -> None:
        estimator = FakeEstimator()
        loaded = LoadedModel(
            estimator=estimator,
            labels=SUPPORTED_LABELS,
            feature_columns=FEATURE_COLUMNS,
            probability_supported=True,
        )

        prediction = predict_features(loaded, np.zeros(63, dtype=np.float32))

        self.assertIsInstance(prediction, FramePrediction)
        self.assertEqual(prediction.label, "peace_sign")
        self.assertEqual(prediction.confidence, 0.8)
        self.assertEqual(estimator.last_probability_shape, (1, 63))

    def test_smoothing_resets_and_uses_majority(self) -> None:
        smoother = PredictionSmoother(window_size=3)
        self.assertEqual(smoother.update("fist"), "fist")
        self.assertEqual(smoother.update("open_palm"), "open_palm")
        self.assertEqual(smoother.update("fist"), "fist")
        smoother.reset()
        self.assertEqual(smoother.update("ok_sign"), "ok_sign")

    def test_collector_diversity_gate_is_optional_and_finite(self) -> None:
        baseline = np.zeros(63, dtype=np.float32)
        nearby = np.full(63, 0.001, dtype=np.float32)
        different = np.full(63, 0.02, dtype=np.float32)

        self.assertTrue(is_diverse_enough(nearby, baseline, 0.0))
        self.assertFalse(is_diverse_enough(nearby, baseline, 0.1))
        self.assertTrue(is_diverse_enough(different, baseline, 0.1))
        self.assertFalse(is_diverse_enough(np.full(62, 0.1), baseline, 0.1))


if __name__ == "__main__":
    unittest.main()
