#!/usr/bin/env python3
"""Add native and English interruption inputs to response score files.

For each interruptions sample with an existing ``response_score.json``, this
script copies ``context_text`` and ``interrupt_text`` from the sample's benchmark
``metadata.json`` into ``context`` and ``interrupt``. The corresponding English
sample supplies ``context_en`` and ``interrupt_en``. English samples use their
English metadata for both field pairs.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple


CATEGORY = "interruptions"
ENGLISH = "english"
DEFAULT_WORKERS = 16


def load_metadata_texts(metadata_path: Path) -> Tuple[str, str]:
    """Return context_text and interrupt_text from one metadata file."""
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a JSON object: {metadata_path}")

    values = []
    for field in ("context_text", "interrupt_text"):
        if field not in metadata:
            raise KeyError(f"'{field}' is missing from {metadata_path}")
        value = metadata[field]
        if not isinstance(value, str):
            raise TypeError(f"'{field}' must be a string in {metadata_path}")
        values.append(value)
    return values[0], values[1]


def update_sample(
    sample_dir: Path,
    input_category_dir: Path,
    english_category_dir: Path,
) -> Path:
    """Add input text fields to one response_score.json atomically."""
    sample_id = sample_dir.name
    native_metadata_path = input_category_dir / sample_id / "metadata.json"
    native_context, native_interrupt = load_metadata_texts(native_metadata_path)

    if input_category_dir == english_category_dir:
        english_context = native_context
        english_interrupt = native_interrupt
    else:
        english_metadata_path = english_category_dir / sample_id / "metadata.json"
        english_context, english_interrupt = load_metadata_texts(
            english_metadata_path
        )

    score_path = sample_dir / "response_score.json"
    with score_path.open("r", encoding="utf-8") as score_file:
        score = json.load(score_file)
    if not isinstance(score, dict):
        raise ValueError(f"response score must be a JSON object: {score_path}")

    score.update(
        {
            "context": native_context,
            "interrupt": native_interrupt,
            "context_en": english_context,
            "interrupt_en": english_interrupt,
        }
    )

    temporary_path = sample_dir / (
        f".response_score.{os.getpid()}.{threading.get_ident()}.json"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as score_file:
            json.dump(score, score_file, ensure_ascii=False, indent=2)
            score_file.write("\n")
        os.replace(temporary_path, score_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return score_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="model output root",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="benchmark input root",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="language directory to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("error: --workers must be at least 1", file=sys.stderr)
        return 2

    input_category_dir = args.input_dir / args.language / CATEGORY
    english_category_dir = args.input_dir / ENGLISH / CATEGORY
    output_category_dir = args.output_dir / args.language / CATEGORY
    required_dirs = list(
        dict.fromkeys(
            (input_category_dir, english_category_dir, output_category_dir)
        )
    )
    missing = [path for path in required_dirs if not path.is_dir()]
    if missing:
        print("Skipping: missing directory/directories: " + ", ".join(map(str, missing)))
        return 0

    all_sample_dirs = sorted(
        path for path in output_category_dir.iterdir() if path.is_dir()
    )
    sample_dirs = [
        path
        for path in all_sample_dirs
        if (path / "response_score.json").is_file()
    ]
    skipped = len(all_sample_dirs) - len(sample_dirs)
    if not sample_dirs:
        print(
            f"Skipping: no response_score.json files found in "
            f"{output_category_dir}"
        )
        return 0

    failures = []
    updated = 0
    completed_count = 0
    progress_interval = max(1, len(sample_dirs) // 20)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample: Dict[Any, Path] = {
            executor.submit(
                update_sample,
                sample_dir,
                input_category_dir,
                english_category_dir,
            ): sample_dir
            for sample_dir in sample_dirs
        }
        for future in as_completed(future_to_sample):
            sample_dir = future_to_sample[future]
            try:
                future.result()
                updated += 1
            except Exception as error:
                failures.append((sample_dir, error))
                print(f"Failed {sample_dir}: {error}", file=sys.stderr)
            finally:
                completed_count += 1
                if (
                    completed_count % progress_interval == 0
                    or completed_count == len(sample_dirs)
                ):
                    percentage = 100.0 * completed_count / len(sample_dirs)
                    print(
                        f"Progress: {completed_count}/{len(sample_dirs)} "
                        f"samples ({percentage:.0f}%)",
                        flush=True,
                    )

    print(
        f"Updated {updated}/{len(sample_dirs)} response scores in "
        f"{output_category_dir} with {args.workers} workers; "
        f"skipped {skipped} samples without response_score.json; "
        f"failures: {len(failures)}."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
