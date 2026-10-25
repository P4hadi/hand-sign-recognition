"""Tests for CSV validation and baseline training without webcam access."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from train_model import (
    CSV_COLUMNS,
    FEATURE_COLUMNS,
    SUPPORTED_LABELS,
    DatasetValidationError,
    load_dataset,
    save_artifacts,
    train_and_evaluate,
)


def write_fixture(path: Path, rows_per_class: int = 10) -> None:
    """Write a small deterministic CSV fixture for unit tests."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for label_index, label in enumerate(SUPPORTED_LABELS):
            for row_index in range(rows_per_class):
                features = [
                    f"{label_index * 10.0 + row_index * 0.01 + feature_index * 0.001:.8f}"
                    for feature_index in range(len(FEATURE_COLUMNS))
                ]
                writer.writerow(
                    [
                        label,
                        "test_session",
                        "test_user",
                        "left",
                        *features,
                    ]
                )


class TrainModelTests(unittest.TestCase):
    def test_schema_and_feature_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_fixture(path)

            dataset = load_dataset(path, require_exact_class_balance=False)

            self.assertEqual(dataset.features.shape, (50, 63))
            self.assertEqual(dataset.labels.shape, (50,))
            self.assertEqual(dataset.class_counts, {label: 10 for label in SUPPORTED_LABELS})
            self.assertEqual(dataset.session_ids, ("test_session",))
            self.assertEqual(dataset.user_ids, ("test_user",))
            self.assertTrue(np.isfinite(dataset.features).all())

    def test_bad_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_header.csv"
            write_fixture(path)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace("landmark_00_x", "wrong_feature", 1)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(DatasetValidationError):
                load_dataset(path, require_exact_class_balance=False)

    def test_missing_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_class.csv"
            write_fixture(path)
            rows = path.read_text(encoding="utf-8").splitlines()
            rows = [rows[0], *[row for row in rows[1:] if not row.startswith("ok_sign,")]]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            with self.assertRaises(DatasetValidationError):
                load_dataset(path, require_exact_class_balance=False)

    def test_exact_class_balance_is_required_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            write_fixture(path, rows_per_class=10)

            with self.assertRaises(DatasetValidationError):
                load_dataset(path)

    def test_training_evaluation_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "fixture.csv"
            write_fixture(csv_path)
            dataset = load_dataset(csv_path, require_exact_class_balance=False)

            evaluation = train_and_evaluate(dataset, test_size=0.2, random_state=42)
            self.assertEqual(evaluation.confusion.shape, (5, 5))
            self.assertIn("accuracy", evaluation.report["metrics"])
            self.assertEqual(evaluation.report["split"]["test_rows"], 10)
            self.assertNotIn("session_id", evaluation.report["feature_policy"]["used_columns"])
            self.assertNotIn("user_id", evaluation.report["feature_policy"]["used_columns"])

            model_path = root / "model.joblib"
            report_path = root / "report.json"
            confusion_path = root / "confusion.csv"
            save_artifacts(
                evaluation,
                csv_path,
                model_path,
                report_path,
                confusion_path,
                random_state=42,
            )

            bundle = joblib.load(model_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["labels"], list(SUPPORTED_LABELS))
            self.assertEqual(len(bundle["feature_columns"]), 63)
            self.assertEqual(report["evaluation_scope"], "provisional within-session estimate")
            self.assertTrue(confusion_path.is_file())


if __name__ == "__main__":
    unittest.main()
