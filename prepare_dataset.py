"""Validate and safely combine independently audited landmark session CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from train_model import (
    CSV_COLUMNS,
    DatasetValidationError,
    SUPPORTED_LABELS,
    load_dataset,
)
from gesture_config import ALL_GESTURE_IDS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "landmarks_fresh_session02.csv",
    PROJECT_ROOT / "data" / "landmarks_fresh_session03.csv",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "landmarks_combined_v1.csv"
EIGHT_GESTURE_INPUTS = [
    PROJECT_ROOT / "data" / "landmarks_fresh_session04.csv",
    PROJECT_ROOT / "data" / "landmarks_fresh_session05.csv",
]
EIGHT_GESTURE_OUTPUT = PROJECT_ROOT / "data" / "landmarks_combined_v2_eight_gestures.csv"


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
        description="Validate and combine session landmark CSVs without modifying sources."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        help="Input session CSV; repeat for each source file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New derived CSV path (default: data\\landmarks_combined_v1.csv).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate inputs and print a summary without writing the combined CSV.",
    )
    parser.add_argument(
        "--eight-gestures",
        action="store_true",
        help="Use sessions 04 and 05 and require the eight-gesture contract.",
    )
    return parser.parse_args()


def read_validated_rows(
    path: Path,
    supported_labels: tuple[str, ...] = SUPPORTED_LABELS,
) -> tuple[list[str], list[dict[str, str]]]:
    """Validate one source with the existing strict loader and return CSV rows."""

    dataset = load_dataset(
        path,
        require_exact_class_balance=False,
        supported_labels=supported_labels,
    )
    if set(dataset.class_counts) != set(supported_labels):
        raise DatasetValidationError(f"Missing supported label in {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise DatasetValidationError(f"Invalid header in {path}")
        rows.extend(dict(row) for row in reader)
    return list(CSV_COLUMNS), rows


def combine_sources(
    input_paths: list[Path],
    output_path: Path,
    supported_labels: tuple[str, ...] = SUPPORTED_LABELS,
) -> dict[str, object]:
    if len(input_paths) < 2:
        raise ValueError("At least two input session CSVs are required.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived dataset: {output_path}")

    all_rows: list[dict[str, str]] = []
    signatures: set[tuple[str, ...]] = set()
    source_summaries: list[dict[str, object]] = []
    for input_path in input_paths:
        _, rows = read_validated_rows(input_path, supported_labels)
        source_summaries.append(
            {
                "path": str(input_path),
                "rows": len(rows),
                "session_ids": sorted({row["session_id"] for row in rows}),
                "user_ids": sorted({row["user_id"] for row in rows}),
            }
        )
        for row in rows:
            signature = tuple(row[column] for column in CSV_COLUMNS)
            if signature in signatures:
                raise DatasetValidationError(
                    f"Duplicate full record across sources near session {row['session_id']}"
                )
            signatures.add(signature)
            all_rows.append(row)

    label_counts = {label: 0 for label in supported_labels}
    for row in all_rows:
        label_counts[row["label"]] += 1
    expected_total = sum(label_counts.values())
    expected_per_class = 300
    expected_total_rows = expected_per_class * len(supported_labels)
    if expected_total != expected_total_rows or any(count != expected_per_class for count in label_counts.values()):
        raise DatasetValidationError(
            f"Combined class contract failed: rows={expected_total}, counts={label_counts}"
        )

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    return {
        "output": str(output_path),
        "rows": expected_total,
        "class_counts": label_counts,
        "source_summaries": source_summaries,
        "session_ids": sorted({row["session_id"] for row in all_rows}),
        "user_ids": sorted({row["user_id"] for row in all_rows}),
    }


def main() -> int:
    args = parse_args()
    try:
        default_inputs = EIGHT_GESTURE_INPUTS if args.eight_gestures else DEFAULT_INPUTS
        default_output = EIGHT_GESTURE_OUTPUT if args.eight_gestures else DEFAULT_OUTPUT
        input_paths = [resolve_project_path(path, "Input path") for path in (args.inputs or default_inputs)]
        output_path = resolve_project_path(args.output if args.output != DEFAULT_OUTPUT else default_output, "Output path")
        supported_labels = ALL_GESTURE_IDS if args.eight_gestures else SUPPORTED_LABELS
        if args.audit_only:
            summaries = []
            for path in input_paths:
                _, rows = read_validated_rows(path, supported_labels)
                summaries.append((str(path), len(rows)))
            print(f"Validated inputs: {summaries}")
            print("No combined file was written.")
            return 0

        summary = combine_sources(input_paths, output_path, supported_labels)
        print(f"Combined dataset: {summary['output']}")
        print(f"Rows: {summary['rows']}")
        print(f"Class counts: {summary['class_counts']}")
        print(f"Sessions: {summary['session_ids']}")
        print(f"Users: {summary['user_ids']}")
        return 0
    except (DatasetValidationError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
