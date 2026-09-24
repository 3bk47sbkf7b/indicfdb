#!/usr/bin/env python3
"""Evaluate whether an agent responds after a user's interruption.

Only raw Silero VAD regions whose starts are strictly after ``interrupt_start``
are considered. Regions crossing or starting exactly at the boundary are
discarded rather than split. Eligible regions separated by less than 1.5
seconds are joined, and merged regions lasting at least 1.0 second are
responses. Latency is measured from ``interrupt_end`` and clipped to zero.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


CATEGORY = "interruptions"
SAMPLE_RATE = 16_000
RESPONSE_MAX_JOIN_GAP_SECONDS = 1.5
RESPONSE_MIN_DURATION_SECONDS = 1.0
VAD_THRESHOLD = 0.7
DEFAULT_WORKERS = 16

_thread_state = threading.local()


def merge_speech_segments(
    segments: Sequence[Dict[str, int]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[int, int]]:
    """Join VAD regions whose gap is strictly below 1.5 seconds."""
    if not segments:
        return []

    max_gap_samples = RESPONSE_MAX_JOIN_GAP_SECONDS * sample_rate
    merged: List[Tuple[int, int]] = []
    current_start = int(segments[0]["start"])
    current_end = int(segments[0]["end"])

    for segment in segments[1:]:
        start = int(segment["start"])
        end = int(segment["end"])
        if start - current_end < max_gap_samples:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def score_segments(
    segments: Sequence[Dict[str, int]],
    interrupt_start: float,
    interrupt_end: float,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """Build an interruption-response score from raw Silero timestamps."""
    boundary_sample = round(interrupt_start * sample_rate)
    eligible_segments = [
        segment
        for segment in segments
        if int(segment["start"]) > boundary_sample
    ]

    responses = []
    first_response_start = None
    for start, end in merge_speech_segments(eligible_segments, sample_rate):
        if (end - start) / sample_rate >= RESPONSE_MIN_DURATION_SECONDS:
            if first_response_start is None:
                first_response_start = start / sample_rate
            responses.append(
                {
                    "start": round(start / sample_rate, 3),
                    "end": round(end / sample_rate, 3),
                }
            )

    response = bool(responses)
    latency = (
        round(max(0.0, first_response_start - interrupt_end), 3)
        if response
        else None
    )
    return {
        "response": response,
        "responses": responses,
        "latency": latency,
        "success": response,
    }


def _get_model() -> Any:
    from silero_vad import load_silero_vad

    model = getattr(_thread_state, "model", None)
    if model is None:
        model = load_silero_vad()
        _thread_state.model = model
    return model


def evaluate_sample(output_sample_dir: Path, input_sample_dir: Path) -> Path:
    """Evaluate one sample and atomically overwrite its score.json."""
    from silero_vad import get_speech_timestamps, read_audio

    metadata_path = input_sample_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    for field in ("interrupt_start", "interrupt_end"):
        if field not in metadata:
            raise KeyError(f"'{field}' is missing from {metadata_path}")
    interrupt_start = float(metadata["interrupt_start"])
    interrupt_end = float(metadata["interrupt_end"])
    if interrupt_start < 0 or interrupt_end < 0:
        raise ValueError(
            f"interrupt_start and interrupt_end must be nonnegative in {metadata_path}"
        )
    if interrupt_start > interrupt_end:
        raise ValueError(
            f"interrupt_start must not exceed interrupt_end in {metadata_path}"
        )

    audio = read_audio(
        str(output_sample_dir / "output.wav"),
        sampling_rate=SAMPLE_RATE,
    )
    speech_segments = get_speech_timestamps(
        audio,
        _get_model(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        return_seconds=False,
    )
    score = score_segments(speech_segments, interrupt_start, interrupt_end)

    score_path = output_sample_dir / "score.json"
    temporary_path = output_sample_dir / (
        f".score.json.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as score_file:
            json.dump(score, score_file, indent=2)
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
        help="language directory to evaluate",
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

    try:
        import torch
        import torchaudio  # noqa: F401
        import soundfile  # noqa: F401
        import silero_vad  # noqa: F401
    except ImportError as error:
        print(f"error: missing dependency: {error}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(
                evaluate_sample,
                sample_dir,
                input_category_dir / sample_dir.name,
            ): sample_dir
            for sample_dir in sample_dirs
        }
        for future in as_completed(future_to_sample):
            sample_dir = future_to_sample[future]
            try:
                future.result()
            except Exception as error:
                failures.append((sample_dir, error))
                print(f"Failed {sample_dir}: {error}", file=sys.stderr)

    succeeded = len(sample_dirs) - len(failures)
    print(
        f"Evaluated {succeeded}/{len(sample_dirs)} samples in "
        f"{output_category_dir} with {args.workers} workers."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
