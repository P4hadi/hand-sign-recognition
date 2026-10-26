"""Tests for session-aware dataset preparation and evaluation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import combine_sources
from gesture_config import ALL_GESTURE_IDS
from train_model import CSV_COLUMNS, SUPPORTED_LABELS, load_dataset
from train_session_model import (
    SUPPLEMENTAL_LABELS,
    ensure_supplemental_dataset,
    evaluate_holdout,
    merge_datasets,
)


class SessionTrainingTests(unittest.TestCase):
    def _require_files(self, *paths: Path) -> None:
        missing = [path for path in paths if not path.is_file()]
        if missing:
            self.skipTest(
                "optional dataset fixtures are not included: "
                + ", ".join(str(path) for path in missing)
            )

    def test_real_session_files_are_complete_and_separate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(
            root / "data" / "landmarks_fresh_session02.csv",
            root / "data" / "landmarks_fresh_session03.csv",
        )
        session02 = load_dataset(
            root / "data" / "landmarks_fresh_session02.csv",
            require_exact_class_balance=False,
        )
        session03 = load_dataset(
            root / "data" / "landmarks_fresh_session03.csv",
            require_exact_class_balance=False,
        )

        self.assertEqual(session02.features.shape, (750, 63))
        self.assertEqual(session03.features.shape, (750, 63))
        self.assertEqual(session02.session_ids, ("user01_session02",))
        self.assertEqual(session03.session_ids, ("user01_session03",))
        self.assertEqual(session02.class_counts, {label: 150 for label in SUPPORTED_LABELS})
        self.assertEqual(session03.class_counts, {label: 150 for label in SUPPORTED_LABELS})

    def test_session_holdout_uses_disjoint_sessions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(
            root / "data" / "landmarks_fresh_session02.csv",
            root / "data" / "landmarks_fresh_session03.csv",
        )
        session02 = load_dataset(
            root / "data" / "landmarks_fresh_session02.csv",
            require_exact_class_balance=False,
        )
        session03 = load_dataset(
            root / "data" / "landmarks_fresh_session03.csv",
            require_exact_class_balance=False,
        )

        model, report, confusion = evaluate_holdout(session02, session03, random_state=42)

        self.assertEqual(confusion.shape, (5, 5))
        self.assertEqual(report["training_session_ids"], ["user01_session02"])
        self.assertEqual(report["test_session_ids"], ["user01_session03"])
        self.assertEqual(report["test_rows"], 750)
        self.assertEqual(len(model.predict(session03.features[:2])), 2)

    def test_combine_sources_writes_new_1500_row_dataset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(
            root / "data" / "landmarks_fresh_session02.csv",
            root / "data" / "landmarks_fresh_session03.csv",
        )
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            source_a = temp_root / "session02.csv"
            source_b = temp_root / "session03.csv"
            for source, original in (
                (source_a, root / "data" / "landmarks_fresh_session02.csv"),
                (source_b, root / "data" / "landmarks_fresh_session03.csv"),
            ):
                source.write_bytes(original.read_bytes())

            output = temp_root / "combined.csv"
            summary = combine_sources([source_a, source_b], output)
            self.assertEqual(summary["rows"], 1500)
            self.assertEqual(summary["class_counts"], {label: 300 for label in SUPPORTED_LABELS})
            with output.open("r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                self.assertEqual(next(reader), list(CSV_COLUMNS))
                self.assertEqual(sum(1 for _ in reader), 1500)

    def test_supplemental_session_has_balanced_focus_labels_and_both_hands(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv")
        supplemental = ensure_supplemental_dataset(
            root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv"
        )
        self.assertEqual(supplemental.class_counts, {label: 150 for label in SUPPLEMENTAL_LABELS})
        self.assertEqual(supplemental.session_ids, ("user01_session06_callme_thumbsup",))
        for label in SUPPLEMENTAL_LABELS:
            self.assertIn("left", supplemental.handedness[supplemental.labels == label])
            self.assertIn("right", supplemental.handedness[supplemental.labels == label])

    def test_supplemental_merge_preserves_all_eight_labels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(
            root / "data" / "landmarks_fresh_session04.csv",
            root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv",
        )
        session04 = load_dataset(
            root / "data" / "landmarks_fresh_session04.csv",
            require_exact_class_balance=False,
            supported_labels=ALL_GESTURE_IDS,
        )
        supplemental = ensure_supplemental_dataset(
            root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv"
        )
        merged = merge_datasets([session04, supplemental], ALL_GESTURE_IDS)
        self.assertEqual(merged.features.shape, (1500, 63))
        self.assertEqual(merged.class_counts["call_me"], 300)
        self.assertEqual(merged.class_counts["thumbs_up"], 300)
        self.assertEqual(merged.session_ids, ("user01_session04", "user01_session06_callme_thumbsup"))

    def test_supplemental_holdout_evaluation_keeps_all_eight_classes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self._require_files(
            root / "data" / "landmarks_fresh_session04.csv",
            root / "data" / "landmarks_fresh_session05.csv",
            root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv",
        )
        session04 = load_dataset(
            root / "data" / "landmarks_fresh_session04.csv",
            require_exact_class_balance=False,
            supported_labels=ALL_GESTURE_IDS,
        )
        session05 = load_dataset(
            root / "data" / "landmarks_fresh_session05.csv",
            require_exact_class_balance=False,
            supported_labels=ALL_GESTURE_IDS,
        )
        supplemental = ensure_supplemental_dataset(
            root / "data" / "landmarks_fresh_session06_callme_thumbsup.csv"
        )
        merged = merge_datasets([session04, supplemental], ALL_GESTURE_IDS)
        _, report, confusion = evaluate_holdout(
            merged,
            session05,
            random_state=42,
            labels=ALL_GESTURE_IDS,
            comparative_diagnostic=True,
        )
        self.assertEqual(confusion.shape, (8, 8))
        self.assertEqual(report["training_rows"], 1500)
        self.assertTrue(report["comparative_diagnostic"])
        self.assertGreaterEqual(report["per_class"]["call_me"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
