"""Train session-aware evaluation and final models without overwriting artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from gesture_config import ALL_GESTURE_IDS
from train_model import (
    FEATURE_COLUMNS,
    LandmarkDataset,
    METADATA_COLUMNS,
    SUPPORTED_LABELS,
    DatasetValidationError,
    build_classifier,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "landmarks_fresh_session02.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "landmarks_fresh_session03.csv"
DEFAULT_COMBINED = PROJECT_ROOT / "data" / "landmarks_combined_v1.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"
RANDOM_STATE = 42
SUPPLEMENTAL_LABELS = ("call_me", "thumbs_up")
EXPECTED_SUPPLEMENTAL_SAMPLES_PER_CLASS = 150


def resolve_project_path(path: Path, description: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{description} must be inside {PROJECT_ROOT}: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a session-held-out model or a final combined-session model."
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--final-data", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument(
        "--supplemental-data",
        type=Path,
        help="Validated two-class supplemental CSV containing only call_me and thumbs_up.",
    )
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--confusion-out", type=Path)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--eight-gestures",
        action="store_true",
        help="Use the eight-label collection vocabulary for sessions 04 and 05.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Fit the final model on --final-data without test metrics.",
    )
    parser.add_argument(
        "--audit-supplemental-only",
        action="store_true",
        help="Validate and summarize --supplemental-data without training or writing artifacts.",
    )
    return parser.parse_args()


def ensure_complete_dataset(path: Path, description: str, labels: tuple[str, ...]):
    dataset = load_dataset(
        path,
        require_exact_class_balance=False,
        supported_labels=labels,
    )
    if any(dataset.class_counts[label] == 0 for label in labels):
        raise DatasetValidationError(f"{description} is missing a gesture class: {path}")
    return dataset


def ensure_supplemental_dataset(path: Path) -> LandmarkDataset:
    """Validate the focused two-class supplement without changing label mapping."""

    dataset = load_dataset(
        path,
        require_exact_class_balance=False,
        supported_labels=SUPPLEMENTAL_LABELS,
    )
    incorrect_counts = {
        label: dataset.class_counts[label]
        for label in SUPPLEMENTAL_LABELS
        if dataset.class_counts[label] != EXPECTED_SUPPLEMENTAL_SAMPLES_PER_CLASS
    }
    if incorrect_counts:
        raise DatasetValidationError(
            "Supplemental dataset must contain exactly 150 samples for each of "
            f"{SUPPLEMENTAL_LABELS}; observed {incorrect_counts}."
        )

    handedness_counts = {
        label: {
            handedness: int(
                np.sum(
                    (dataset.labels == label)
                    & (dataset.handedness == handedness)
                )
            )
            for handedness in ("left", "right")
        }
        for label in SUPPLEMENTAL_LABELS
    }
    missing_handedness = {
        label: counts
        for label, counts in handedness_counts.items()
        if any(counts[handedness] == 0 for handedness in ("left", "right"))
    }
    if missing_handedness:
        raise DatasetValidationError(
            "Supplemental dataset must contain both left and right samples for "
            f"each focused label: {missing_handedness}"
        )
    return dataset


def merge_datasets(
    datasets: list[LandmarkDataset],
    labels: tuple[str, ...],
) -> LandmarkDataset:
    """Merge validated sources while rejecting cross-source duplicate poses."""

    if not datasets:
        raise ValueError("At least one validated dataset is required.")

    feature_blocks = [dataset.features for dataset in datasets]
    label_blocks = [dataset.labels for dataset in datasets]
    session_blocks = [dataset.sessions for dataset in datasets]
    user_blocks = [dataset.users for dataset in datasets]
    handedness_blocks = [dataset.handedness for dataset in datasets]
    merged_features = np.concatenate(feature_blocks, axis=0)
    merged_labels = np.concatenate(label_blocks, axis=0)

    signatures: set[tuple[str, ...]] = set()
    for label, features in zip(merged_labels.tolist(), merged_features, strict=True):
        signature = (label, *(f"{float(value):.8f}" for value in features))
        if signature in signatures:
            raise DatasetValidationError(
                "Duplicate label-feature record detected across training sources."
            )
        signatures.add(signature)

    counts = {label: int(np.sum(merged_labels == label)) for label in labels}
    if any(counts[label] == 0 for label in labels):
        raise DatasetValidationError(f"Merged training data is missing a gesture class: {counts}")

    return LandmarkDataset(
        features=merged_features,
        labels=merged_labels,
        sessions=np.concatenate(session_blocks, axis=0),
        users=np.concatenate(user_blocks, axis=0),
        handedness=np.concatenate(handedness_blocks, axis=0),
        class_counts=counts,
        session_ids=tuple(sorted(set(np.concatenate(session_blocks).tolist()))),
        user_ids=tuple(sorted(set(np.concatenate(user_blocks).tolist()))),
    )


def supplemental_summary(dataset: LandmarkDataset) -> dict[str, Any]:
    """Return lightweight diversity and metadata evidence for the supplement."""

    summary: dict[str, Any] = {
        "rows": int(dataset.features.shape[0]),
        "class_counts": dataset.class_counts,
        "session_ids": list(dataset.session_ids),
        "user_ids": list(dataset.user_ids),
        "handedness_counts": {
            handedness: int(np.sum(dataset.handedness == handedness))
            for handedness in ("left", "right")
        },
        "label_handedness_counts": {},
        "pose_diversity": {},
    }
    for label in SUPPLEMENTAL_LABELS:
        mask = dataset.labels == label
        features = dataset.features[mask]
        consecutive = np.linalg.norm(features[1:] - features[:-1], axis=1)
        summary["label_handedness_counts"][label] = {
            handedness: int(np.sum(mask & (dataset.handedness == handedness)))
            for handedness in ("left", "right")
        }
        summary["pose_diversity"][label] = {
            "mean_feature_std": float(features.std(axis=0).mean()),
            "mean_consecutive_distance": float(consecutive.mean()),
            "min_consecutive_distance": float(consecutive.min()),
            "consecutive_at_or_below_0_10": int(np.sum(consecutive <= 0.10)),
        }
    return summary


def evaluate_holdout(
    train_dataset,
    test_dataset,
    random_state: int,
    labels: tuple[str, ...] = SUPPORTED_LABELS,
    model_name: str | None = None,
    training_data_paths: list[Path] | None = None,
    comparative_diagnostic: bool = False,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    train_sessions = set(train_dataset.session_ids)
    test_sessions = set(test_dataset.session_ids)
    if train_sessions & test_sessions:
        raise DatasetValidationError(
            f"Training/test session overlap detected: {sorted(train_sessions & test_sessions)}"
        )

    model = build_classifier(random_state)
    model.fit(train_dataset.features, train_dataset.labels)
    predictions = model.predict(test_dataset.features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_dataset.labels,
        predictions,
        labels=list(labels),
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        test_dataset.labels,
        predictions,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_dataset.labels, predictions, labels=list(labels))
    evaluation = {
        "schema_version": "1.0",
        "model_name": model_name or (
            "gesture_rf_session04_train_session05_test"
            if labels == ALL_GESTURE_IDS
            else "gesture_rf_session02_train_session03_test"
        ),
        "classifier": "sklearn.ensemble.RandomForestClassifier",
        "random_state": random_state,
        "evaluation_method": "whole-session holdout",
        "evaluation_scope": "same-user session-aware held-out evaluation",
        "training_session_ids": sorted(train_sessions),
        "test_session_ids": sorted(test_sessions),
        "training_user_ids": sorted(set(train_dataset.user_ids)),
        "test_user_ids": sorted(set(test_dataset.user_ids)),
        "label_order": list(labels),
        "feature_policy": {
            "used_columns": list(FEATURE_COLUMNS),
            "excluded_metadata_columns": list(METADATA_COLUMNS),
        },
        "training_rows": int(train_dataset.features.shape[0]),
        "test_rows": int(test_dataset.features.shape[0]),
        "training_class_counts": train_dataset.class_counts,
        "test_class_counts": test_dataset.class_counts,
        "training_data_paths": [str(path) for path in (training_data_paths or [])],
        "comparative_diagnostic": comparative_diagnostic,
        "metrics": {
            "accuracy": float(accuracy_score(test_dataset.labels, predictions)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        },
        "per_class": report,
        "confusion_matrix_labels": list(labels),
        "limitations": [
            "Both sessions belong to user_01.",
            "This does not measure unseen-user generalization.",
            "The held-out session is independent by session_id, but not by user.",
        ],
    }
    if comparative_diagnostic:
        evaluation["limitations"].extend(
            [
                "Session 05 was examined during earlier development, so this is not a pristine final test.",
                "This comparison changes the training distribution with a focused supplemental dataset.",
            ]
        )
    return model, evaluation, matrix


def final_report(
    dataset,
    random_state: int,
    labels: tuple[str, ...],
    training_data_paths: list[Path] | None = None,
    model_name: str | None = None,
    supplemental: bool = False,
) -> dict[str, Any]:
    report = {
        "schema_version": "1.0",
        "model_name": model_name or (
            "gesture_rf_v3_final_eight_gesture_sessions"
            if labels == ALL_GESTURE_IDS
            else "gesture_rf_v2_final_combined_sessions"
        ),
        "classifier": "sklearn.ensemble.RandomForestClassifier",
        "random_state": random_state,
        "training_session_ids": list(dataset.session_ids),
        "training_user_ids": list(dataset.user_ids),
        "training_rows": int(dataset.features.shape[0]),
        "training_data_paths": [str(path) for path in (training_data_paths or [])],
        "class_counts": dataset.class_counts,
        "label_order": list(labels),
        "feature_policy": {
            "used_columns": list(FEATURE_COLUMNS),
            "excluded_metadata_columns": list(METADATA_COLUMNS),
        },
        "evaluation_scope": "final model fit on combined sessions; no independent test accuracy claimed",
        "limitations": [
            "Both available sessions belong to user_01.",
            "Both sessions are used for final-model training.",
            "Use the separate session-held-out report for metrics.",
            "Unseen-user performance remains unverified.",
        ],
    }
    if supplemental:
        report["limitations"].append(
            "The focused supplemental session contains only call_me and thumbs_up samples."
        )
    return report


def save_bundle(
    model,
    model_path: Path,
    dataset_path: Path | list[Path],
    model_name: str,
    random_state: int,
    scope: str,
    labels: tuple[str, ...],
) -> None:
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "labels": list(labels),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "classifier": model_name,
        "random_state": random_state,
        "training_dataset": str(dataset_path) if isinstance(dataset_path, Path) else None,
        "training_datasets": [str(path) for path in ([dataset_path] if isinstance(dataset_path, Path) else dataset_path)],
        "evaluation_scope": scope,
    }
    with model_path.open("xb") as file:
        joblib.dump(bundle, file)


def save_evaluation(report: dict[str, Any], matrix: np.ndarray, report_path: Path, confusion_path: Path) -> None:
    if report_path.exists() or confusion_path.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation artifact.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    with confusion_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        labels = report["label_order"]
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def print_evaluation(report: dict[str, Any], model_path: Path, report_path: Path, confusion_path: Path) -> None:
    print(f"Evaluation model: {model_path}")
    print(f"Training sessions: {report['training_session_ids']}")
    print(f"Test sessions: {report['test_session_ids']}")
    print(f"Training rows: {report['training_rows']} | Test rows: {report['test_rows']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.6f}")
    print(f"Evaluation report: {report_path}")
    print(f"Confusion matrix: {confusion_path}")


def main() -> int:
    args = parse_args()
    labels = ALL_GESTURE_IDS if args.eight_gestures else SUPPORTED_LABELS
    try:
        models_dir = resolve_project_path(args.models_dir, "Models directory")
        reports_dir = resolve_project_path(args.reports_dir, "Reports directory")
        if args.audit_supplemental_only:
            if args.supplemental_data is None:
                raise ValueError("--audit-supplemental-only requires --supplemental-data.")
            supplemental_path = resolve_project_path(args.supplemental_data, "Supplemental dataset")
            supplemental = ensure_supplemental_dataset(supplemental_path)
            print(json.dumps(supplemental_summary(supplemental), indent=2))
            print("No model or report artifacts were written.")
            return 0
        if args.final_only:
            final_data = resolve_project_path(args.final_data, "Final dataset")
            dataset = ensure_complete_dataset(final_data, "Final dataset", labels)
            training_data_paths = [final_data]
            if args.supplemental_data is not None:
                if not args.eight_gestures:
                    raise ValueError("--supplemental-data requires --eight-gestures.")
                supplemental_path = resolve_project_path(args.supplemental_data, "Supplemental dataset")
                supplemental = ensure_supplemental_dataset(supplemental_path)
                dataset = merge_datasets([dataset, supplemental], labels)
                training_data_paths.append(supplemental_path)
            default_model = (
                "gesture_rf_v4_eight_gestures_session06.joblib"
                if args.supplemental_data is not None
                else ("gesture_rf_v3_eight_gestures.joblib" if args.eight_gestures else "gesture_rf_v2.joblib")
            )
            default_report = (
                "gesture_rf_v4_eight_gestures_session06_final.json"
                if args.supplemental_data is not None
                else ("gesture_rf_v3_eight_gestures_final.json" if args.eight_gestures else "gesture_rf_v2_final.json")
            )
            model_path = resolve_project_path(args.model_out or (models_dir / default_model), "Model output")
            report_path = resolve_project_path(args.report_out or (reports_dir / default_report), "Report output")
            if report_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
            model = build_classifier(args.random_state)
            model.fit(dataset.features, dataset.labels)
            save_bundle(
                model,
                model_path,
                training_data_paths,
                "gesture_rf_v4_final_eight_gesture_sessions_with_session06"
                if args.supplemental_data is not None
                else ("gesture_rf_v3_final_eight_gesture_sessions" if args.eight_gestures else "gesture_rf_v2_final_combined_sessions"),
                args.random_state,
                "final model fit on combined sessions; no independent test accuracy claimed",
                labels,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8", newline="") as file:
                json.dump(
                    final_report(
                        dataset,
                        args.random_state,
                        labels,
                        training_data_paths,
                        model_name=(
                            "gesture_rf_v4_final_eight_gesture_sessions_with_session06"
                            if args.supplemental_data is not None
                            else None
                        ),
                        supplemental=args.supplemental_data is not None,
                    ),
                    file,
                    indent=2,
                )
                file.write("\n")
            print(f"Final model: {model_path}")
            print(f"Final report: {report_path}")
            print("No independent test accuracy was calculated for the final model.")
            return 0

        train_data = resolve_project_path(args.train_data, "Training dataset")
        test_data = resolve_project_path(args.test_data, "Test dataset")
        train_dataset = ensure_complete_dataset(train_data, "Training dataset", labels)
        training_data_paths = [train_data]
        if args.supplemental_data is not None:
            if not args.eight_gestures:
                raise ValueError("--supplemental-data requires --eight-gestures.")
            supplemental_path = resolve_project_path(args.supplemental_data, "Supplemental dataset")
            supplemental = ensure_supplemental_dataset(supplemental_path)
            train_dataset = merge_datasets([train_dataset, supplemental], labels)
            training_data_paths.append(supplemental_path)
        test_dataset = ensure_complete_dataset(test_data, "Test dataset", labels)
        model, report, matrix = evaluate_holdout(
            train_dataset,
            test_dataset,
            args.random_state,
            labels,
            model_name=(
                "gesture_rf_session04_plus_session06_train_session05_test"
                if args.supplemental_data is not None
                else None
            ),
            training_data_paths=training_data_paths,
            comparative_diagnostic=args.supplemental_data is not None,
        )
        default_model = (
            "gesture_rf_session04_plus_session06_eval_eight_v1.joblib"
            if args.supplemental_data is not None
            else ("gesture_rf_session04_eval_eight_v1.joblib" if args.eight_gestures else "gesture_rf_session02_eval_v1.joblib")
        )
        default_report = (
            "gesture_rf_session04_plus_session06_eval_eight_v1.json"
            if args.supplemental_data is not None
            else ("gesture_rf_session04_eval_eight_v1.json" if args.eight_gestures else "gesture_rf_session02_eval_v1.json")
        )
        default_confusion = (
            "gesture_rf_session04_plus_session06_eval_eight_v1_confusion_matrix.csv"
            if args.supplemental_data is not None
            else ("gesture_rf_session04_eval_eight_v1_confusion_matrix.csv" if args.eight_gestures else "gesture_rf_session02_eval_v1_confusion_matrix.csv")
        )
        model_path = resolve_project_path(args.model_out or (models_dir / default_model), "Model output")
        report_path = resolve_project_path(args.report_out or (reports_dir / default_report), "Report output")
        confusion_path = resolve_project_path(args.confusion_out or (reports_dir / default_confusion), "Confusion output")
        save_bundle(
            model,
            model_path,
            training_data_paths,
            "gesture_rf_session04_plus_session06_eval_eight_v1"
            if args.supplemental_data is not None
            else ("gesture_rf_session04_eval_eight_v1" if args.eight_gestures else "gesture_rf_session02_eval_v1"),
            args.random_state,
            "session-aware held-out evaluation",
            labels,
        )
        report["artifacts"] = {
            "model": str(model_path),
            "evaluation_report": str(report_path),
            "confusion_matrix": str(confusion_path),
        }
        save_evaluation(report, matrix, report_path, confusion_path)
        print_evaluation(report, model_path, report_path, confusion_path)
        return 0
    except (DatasetValidationError, FileExistsError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
