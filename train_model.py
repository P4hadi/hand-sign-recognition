"""Train and evaluate a baseline gesture classifier from landmark CSV data.

This script intentionally uses only the 63 normalized landmark columns as
predictive inputs. Metadata columns are validated and reported, but never pass
to the classifier.

The current fresh dataset contains one user and one recording session. It is
therefore evaluated with a fixed-seed stratified random row holdout. This is a
provisional within-session estimate, not an independent-session or unseen-user
generalization result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "landmarks_fresh.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"
ARTIFACT_STEM = "gesture_rf_baseline"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_SAMPLES_PER_CLASS = 200

SUPPORTED_LABELS = (
    "open_palm",
    "fist",
    "thumbs_up",
    "peace_sign",
    "ok_sign",
)
METADATA_COLUMNS = ("label", "session_id", "user_id", "handedness")
FEATURE_COLUMNS = tuple(
    f"landmark_{index:02d}_{axis}"
    for index in range(21)
    for axis in ("x", "y", "z")
)
CSV_COLUMNS = (*METADATA_COLUMNS, *FEATURE_COLUMNS)


class DatasetValidationError(ValueError):
    """Raised when the training CSV does not satisfy the project contract."""


@dataclass(frozen=True)
class LandmarkDataset:
    """Validated dataset arrays and metadata summary."""

    features: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    users: np.ndarray
    handedness: np.ndarray
    class_counts: dict[str, int]
    session_ids: tuple[str, ...]
    user_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outputs kept separate from artifact serialization."""

    model: Pipeline
    report: dict[str, Any]
    confusion: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the five-class landmark Random Forest baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Validated landmark CSV (default: data\\landmarks_fresh.csv).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for versioned model artifacts (default: models).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for versioned evaluation artifacts (default: reports\\metrics).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction for the provisional stratified test holdout (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for splitting and the classifier (default: 42).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and summarize the dataset without training or writing artifacts.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path, description: str) -> Path:
    """Resolve a path and keep all script-managed files in the project."""

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{description} must be inside {PROJECT_ROOT}: {resolved}")
    return resolved


def _feature_signature(label: str, values: list[float]) -> tuple[str, ...]:
    return (label, *(f"{value:.8f}" for value in values))


def load_dataset(
    path: Path,
    *,
    require_exact_class_balance: bool = True,
    supported_labels: tuple[str, ...] | None = None,
) -> LandmarkDataset:
    """Load and strictly validate the landmark CSV.

    Metadata is retained for audit reporting but excluded from the feature
    matrix. The default requires exactly 200 valid rows for every supported
    class, matching the fresh-dataset contract.
    """

    labels_contract = supported_labels or SUPPORTED_LABELS

    if not path.is_file():
        raise DatasetValidationError(f"Training dataset was not found: {path}")

    feature_rows: list[list[float]] = []
    label_rows: list[str] = []
    session_rows: list[str] = []
    user_rows: list[str] = []
    handedness_rows: list[str] = []
    class_counts: Counter[str] = Counter()
    full_signatures: set[tuple[str, ...]] = set()
    label_feature_signatures: set[tuple[str, ...]] = set()

    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, strict=True)
            header = next(reader, None)
            if header != list(CSV_COLUMNS):
                actual_count = len(header) if header is not None else 0
                raise DatasetValidationError(
                    f"Invalid CSV header: expected exactly {len(CSV_COLUMNS)} columns "
                    f"in the canonical order, found {actual_count}."
                )

            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(CSV_COLUMNS):
                    raise DatasetValidationError(
                        f"Malformed row at CSV line {line_number}: expected "
                        f"{len(CSV_COLUMNS)} fields, found {len(values)}."
                    )

                row = dict(zip(CSV_COLUMNS, values))
                missing_columns = [column for column in CSV_COLUMNS if row[column] == ""]
                if missing_columns:
                    raise DatasetValidationError(
                        f"Missing value at CSV line {line_number}: {missing_columns[0]}"
                    )

                label = row["label"]
                if label not in labels_contract:
                    raise DatasetValidationError(
                        f"Unsupported label at CSV line {line_number}: {label!r}"
                    )

                for metadata_name in ("session_id", "user_id", "handedness"):
                    if not row[metadata_name].strip():
                        raise DatasetValidationError(
                            f"Empty {metadata_name} at CSV line {line_number}."
                        )

                try:
                    features = [float(row[column]) for column in FEATURE_COLUMNS]
                except ValueError as exc:
                    raise DatasetValidationError(
                        f"Non-numeric feature at CSV line {line_number}."
                    ) from exc

                if len(features) != 63 or not all(math.isfinite(value) for value in features):
                    raise DatasetValidationError(
                        f"Non-finite or incomplete feature vector at CSV line {line_number}."
                    )

                full_signature = tuple(values)
                if full_signature in full_signatures:
                    raise DatasetValidationError(
                        f"Duplicate full record at CSV line {line_number}."
                    )
                full_signatures.add(full_signature)

                label_feature_signature = _feature_signature(label, features)
                if label_feature_signature in label_feature_signatures:
                    raise DatasetValidationError(
                        f"Duplicate label-feature record at CSV line {line_number}."
                    )
                label_feature_signatures.add(label_feature_signature)

                feature_rows.append(features)
                label_rows.append(label)
                session_rows.append(row["session_id"])
                user_rows.append(row["user_id"])
                handedness_rows.append(row["handedness"])
                class_counts[label] += 1
    except csv.Error as exc:
        raise DatasetValidationError(f"CSV parsing failed: {exc}") from exc

    missing_labels = [label for label in labels_contract if class_counts[label] == 0]
    unexpected_labels = sorted(set(class_counts) - set(labels_contract))
    if missing_labels or unexpected_labels:
        raise DatasetValidationError(
            f"Class validation failed. Missing={missing_labels}; unexpected={unexpected_labels}."
        )

    if require_exact_class_balance:
        incorrect_counts = {
            label: class_counts[label]
            for label in labels_contract
            if class_counts[label] != EXPECTED_SAMPLES_PER_CLASS
        }
        if incorrect_counts:
            raise DatasetValidationError(
                "Expected exactly 200 samples per class; "
                f"observed {incorrect_counts}."
            )

    if not feature_rows:
        raise DatasetValidationError("The dataset contains no valid data rows.")

    return LandmarkDataset(
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(label_rows, dtype=str),
        sessions=np.asarray(session_rows, dtype=str),
        users=np.asarray(user_rows, dtype=str),
        handedness=np.asarray(handedness_rows, dtype=str),
        class_counts={label: class_counts[label] for label in labels_contract},
        session_ids=tuple(sorted(set(session_rows))),
        user_ids=tuple(sorted(set(user_rows))),
    )


def build_classifier(random_state: int) -> Pipeline:
    """Build the reproducible Random Forest baseline.

    Landmark features are already normalized by the collector. Random Forest
    does not require standardization and is robust for a small numeric feature
    matrix, so no learned preprocessing step is needed here.
    """

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
    )
    return Pipeline([("classifier", classifier)])


def train_and_evaluate(
    dataset: LandmarkDataset,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> EvaluationResult:
    """Fit on the training partition and evaluate once on the held-out rows."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    train_features, test_features, train_labels, test_labels = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )

    model = build_classifier(random_state)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    accuracy = float(accuracy_score(test_labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=list(SUPPORTED_LABELS),
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        test_labels,
        predictions,
        labels=list(SUPPORTED_LABELS),
        target_names=list(SUPPORTED_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=list(SUPPORTED_LABELS),
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    evaluation_report: dict[str, Any] = {
        "schema_version": "1.0",
        "model_name": ARTIFACT_STEM,
        "classifier": "sklearn.ensemble.RandomForestClassifier",
        "random_state": random_state,
        "test_size": test_size,
        "evaluation_method": "stratified random row holdout",
        "evaluation_scope": "provisional within-session estimate",
        "dataset_summary": {
            "rows": int(dataset.features.shape[0]),
            "feature_count": int(dataset.features.shape[1]),
            "class_counts": dataset.class_counts,
            "session_ids": list(dataset.session_ids),
            "user_ids": list(dataset.user_ids),
            "handedness_counts": dict(Counter(dataset.handedness.tolist())),
        },
        "feature_policy": {
            "used_columns": list(FEATURE_COLUMNS),
            "excluded_metadata_columns": list(METADATA_COLUMNS),
        },
        "split": {
            "train_rows": int(train_features.shape[0]),
            "test_rows": int(test_features.shape[0]),
            "train_class_counts": dict(Counter(train_labels.tolist())),
            "test_class_counts": dict(Counter(test_labels.tolist())),
        },
        "metrics": metrics,
        "per_class": report,
        "confusion_matrix_labels": list(SUPPORTED_LABELS),
        "limitations": [
            "All samples come from one user.",
            "All samples come from one recording session.",
            "This stratified row holdout is not an independent-session evaluation.",
            "This result does not measure unseen-user generalization.",
            "Adjacent or similar samples from the same session may be correlated.",
        ],
    }
    return EvaluationResult(model=model, report=evaluation_report, confusion=matrix)


def next_artifact_paths(models_dir: Path, reports_dir: Path) -> tuple[Path, Path, Path]:
    """Allocate matching model/report paths without overwriting artifacts."""

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        model_path = models_dir / f"{ARTIFACT_STEM}_v{version}.joblib"
        report_path = reports_dir / f"{ARTIFACT_STEM}_v{version}.json"
        confusion_path = reports_dir / f"{ARTIFACT_STEM}_v{version}_confusion_matrix.csv"
        if not any(path.exists() for path in (model_path, report_path, confusion_path)):
            return model_path, report_path, confusion_path
        version += 1


def save_artifacts(
    evaluation: EvaluationResult,
    dataset_path: Path,
    model_path: Path,
    report_path: Path,
    confusion_path: Path,
    random_state: int,
) -> None:
    """Save model, metrics, and confusion matrix using exclusive file creation."""

    bundle = {
        "model": evaluation.model,
        "labels": list(SUPPORTED_LABELS),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "classifier": ARTIFACT_STEM,
        "random_state": random_state,
        "training_dataset": str(dataset_path),
        "evaluation_scope": "provisional within-session estimate",
    }
    with model_path.open("xb") as file:
        joblib.dump(bundle, file)

    report = dict(evaluation.report)
    report["artifacts"] = {
        "model": str(model_path),
        "evaluation_report": str(report_path),
        "confusion_matrix": str(confusion_path),
    }
    with report_path.open("x", encoding="utf-8", newline="") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    with confusion_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *SUPPORTED_LABELS])
        for label, row in zip(SUPPORTED_LABELS, evaluation.confusion.tolist(), strict=True):
            writer.writerow([label, *row])


def print_summary(
    dataset_path: Path,
    dataset: LandmarkDataset,
    evaluation: EvaluationResult,
    model_path: Path,
    report_path: Path,
    confusion_path: Path,
) -> None:
    metrics = evaluation.report["metrics"]
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {dataset.features.shape[0]} | Features used: {dataset.features.shape[1]}")
    print(f"Classes: {dataset.class_counts}")
    print(f"Sessions: {list(dataset.session_ids)}")
    print(f"Users: {list(dataset.user_ids)}")
    print("Evaluation: stratified random row holdout (80% train / 20% test)")
    print("Scope: provisional within-session estimate; not unseen-user generalization")
    print(f"Accuracy:         {metrics['accuracy']:.6f}")
    print(f"Macro precision:  {metrics['macro_precision']:.6f}")
    print(f"Macro recall:     {metrics['macro_recall']:.6f}")
    print(f"Macro F1-score:   {metrics['macro_f1']:.6f}")
    print("Confusion matrix labels: " + ", ".join(SUPPORTED_LABELS))
    print(f"Model artifact:   {model_path}")
    print(f"Evaluation JSON:   {report_path}")
    print(f"Confusion CSV:     {confusion_path}")


def main() -> int:
    args = parse_args()
    try:
        data_path = resolve_project_path(args.data, "Dataset path")
        models_dir = resolve_project_path(args.models_dir, "Models directory")
        reports_dir = resolve_project_path(args.reports_dir, "Reports directory")
        if not 0.0 < args.test_size < 1.0:
            raise ValueError("--test-size must be between 0 and 1.")

        dataset = load_dataset(data_path)
        if args.audit_only:
            print(f"Dataset audit passed: {data_path}")
            print(f"Rows: {dataset.features.shape[0]} | Features: {dataset.features.shape[1]}")
            print(f"Class counts: {dataset.class_counts}")
            print(f"Sessions: {list(dataset.session_ids)}")
            print(f"Users: {list(dataset.user_ids)}")
            print("No model training or artifact writing was performed.")
            return 0

        evaluation = train_and_evaluate(
            dataset,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        model_path, report_path, confusion_path = next_artifact_paths(models_dir, reports_dir)
        save_artifacts(
            evaluation,
            data_path,
            model_path,
            report_path,
            confusion_path,
            args.random_state,
        )
        print_summary(data_path, dataset, evaluation, model_path, report_path, confusion_path)
        return 0
    except (DatasetValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
