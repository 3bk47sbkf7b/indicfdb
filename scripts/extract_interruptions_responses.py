#!/usr/bin/env python3
"""Extract normalized post-interruption responses from evaluated outputs.

For every interruptions sample with a non-empty ``responses`` list in
``score.json``, audio is extracted from the earliest response start through the
end of ``output.wav``. The result is loudness-normalized with FFmpeg, resampled
to 24 kHz, encoded as signed 16-bit PCM, and written beside the source as
``response.wav``.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


CATEGORY = "interruptions"
DEFAULT_WORKERS = 16


def earliest_response_start(score_path: Path) -> Optional[float]:
    """Return the earliest scored response start, or None if there is none."""
    with score_path.open("r", encoding="utf-8") as score_file:
        score = json.load(score_file)

    if score.get("response") is False:
        return None

    responses = score.get("responses")
    if not isinstance(responses, list):
        raise ValueError(f"'responses' must be a list in {score_path}")
    if not responses:
        return None

    starts = []
    for index, response in enumerate(responses):
        if not isinstance(response, dict) or "start" not in response:
            raise ValueError(
                f"responses[{index}] must contain a start timestamp in {score_path}"
            )
        try:
            start = float(response["start"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"responses[{index}].start must be numeric in {score_path}"
            ) from error
        if not math.isfinite(start) or start < 0:
            raise ValueError(
                f"responses[{index}].start must be finite and nonnegative in "
                f"{score_path}"
            )
        starts.append(start)

    return min(starts)


def extract_sample(sample_dir: Path, ffmpeg: str) -> Optional[Path]:
    """Extract one response, returning None when no response was scored."""
    score_path = sample_dir / "score.json"
    response_start = earliest_response_start(score_path)
    if response_start is None:
        return None

    output_path = sample_dir / "output.wav"
    response_path = sample_dir / "response.wav"
    temporary_path = sample_dir / (
        f".response.{os.getpid()}.{threading.get_ident()}.wav"
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(output_path),
        "-map",
        "0:a:0",
        "-af",
        f"atrim=start={response_start:.6f},asetpts=PTS-STARTPTS,loudnorm",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"ffmpeg failed for {output_path}: {completed.stderr.strip()}"
            )
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 44:
            raise RuntimeError(f"extracted response is empty for {output_path}")
        os.replace(temporary_path, response_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return response_path


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
    output_category_dir = args.output_dir / args.language / CATEGORY
    missing = [
        path
        for path in (input_category_dir, output_category_dir)
        if not path.is_dir()
    ]
    if missing:
        print("Skipping: missing directory/directories: " + ", ".join(map(str, missing)))
        return 0

    sample_dirs = sorted(
        path
        for path in output_category_dir.iterdir()
        if path.is_dir() and (path / "output.wav").is_file()
    )
    if not sample_dirs:
        print(f"Skipping: no output.wav files found in {output_category_dir}")
        return 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("error: ffmpeg is not installed or not on PATH", file=sys.stderr)
        return 2

    failures = []
    extracted = 0
    skipped = 0
    completed_count = 0
    progress_interval = max(1, len(sample_dirs) // 20)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample: Dict[Any, Path] = {
            executor.submit(extract_sample, sample_dir, ffmpeg): sample_dir
            for sample_dir in sample_dirs
        }
        for future in as_completed(future_to_sample):
            sample_dir = future_to_sample[future]
            try:
                if future.result() is None:
                    skipped += 1
                else:
                    extracted += 1
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
        f"Extracted {extracted}/{len(sample_dirs)} responses in "
        f"{output_category_dir} with {args.workers} workers; "
        f"skipped {skipped} samples without scored responses."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
