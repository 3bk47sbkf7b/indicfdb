#!/usr/bin/env python3
"""Generate dual-confidence Silero VAD JSON for one language folder.

For every ``CONVERSATION_ID/audios/speaker*.wav`` directly beneath the given
language directory, Silero is run twice:

* threshold 0.7 produces ``confidence: "high"`` segments;
* threshold 0.1 produces candidate ``confidence: "low"`` segments.

A low-confidence candidate is retained only when it has no positive-duration
overlap with any high-confidence segment.  Timestamps are returned to three
decimal places (1 ms representation).  The combined records are sorted by start
time and written to the matching ``vads/speaker*.json``.  Speaker IDs are copied
from the corresponding ``metadata/speaker*.json``.

Example:

    .venv/bin/python run_silero_vad.py \
        call_wise_delivery/bn_batch1 --workers 8 --overwrite
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NumPy/OpenBLAS can otherwise create dozens of threads in every spawned worker.
# Set these before a child imports Torch or NumPy.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


_MODEL = None
_SAMPLE_RATE = 16_000
_TIME_RESOLUTION = 3
_LOW_THRESHOLD = 0.1
_HIGH_THRESHOLD = 0.7


def _initialize_worker() -> None:
    """Load one model per process and keep CPU oversubscription under control."""
    global _MODEL

    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    _MODEL = load_silero_vad()


def _output_path(audio_path: Path) -> Path:
    conversation_dir = audio_path.parent.parent
    return conversation_dir / "vads" / f"{audio_path.stem}.json"


def _metadata_path(audio_path: Path) -> Path:
    conversation_dir = audio_path.parent.parent
    return conversation_dir / "metadata" / f"{audio_path.stem}.json"


def _has_current_schema(output_path: Path) -> bool:
    """Return whether an existing output uses labels and millisecond times."""
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    expected_fields = {"start", "end", "speaker_id", "confidence"}
    return isinstance(records, list) and all(
        isinstance(record, dict)
        and set(record) == expected_fields
        and record["confidence"] in {"high", "low"}
        and isinstance(record["start"], (int, float))
        and isinstance(record["end"], (int, float))
        and record["start"] == round(record["start"], _TIME_RESOLUTION)
        and record["end"] == round(record["end"], _TIME_RESOLUTION)
        for record in records
    )


def _has_positive_overlap(
    first: Dict[str, Any], second: Dict[str, Any]
) -> bool:
    """Return whether two timestamp dictionaries overlap for positive time."""
    return first["start"] < second["end"] and second["start"] < first["end"]


def _combine_confidence_segments(
    high_timestamps: Sequence[Dict[str, Any]],
    low_timestamps: Sequence[Dict[str, Any]],
    speaker_id: Any,
) -> List[Dict[str, Any]]:
    """Label high segments and add only non-overlapping low segments."""
    # Silero normally honors ``time_resolution`` itself, but an utterance that
    # reaches the physical end of a WAV can retain the exact sample-derived end
    # time. Normalize every boundary here so the JSON contract is unconditional.
    high = sorted(
        (
            {
                "start": round(item["start"], _TIME_RESOLUTION),
                "end": round(item["end"], _TIME_RESOLUTION),
            }
            for item in high_timestamps
        ),
        key=lambda item: (item["start"], item["end"]),
    )
    low = sorted(
        (
            {
                "start": round(item["start"], _TIME_RESOLUTION),
                "end": round(item["end"], _TIME_RESOLUTION),
            }
            for item in low_timestamps
        ),
        key=lambda item: (item["start"], item["end"]),
    )
    low_only: List[Dict[str, Any]] = []
    high_index = 0
    for low_segment in low:
        while (
            high_index < len(high)
            and high[high_index]["end"] <= low_segment["start"]
        ):
            high_index += 1
        overlaps_high = (
            high_index < len(high)
            and _has_positive_overlap(low_segment, high[high_index])
        )
        if not overlaps_high:
            low_only.append(low_segment)

    segments: List[Dict[str, Any]] = [
        {
            "start": timestamp["start"],
            "end": timestamp["end"],
            "speaker_id": speaker_id,
            "confidence": "high",
        }
        for timestamp in high
    ]
    segments.extend(
        {
            "start": timestamp["start"],
            "end": timestamp["end"],
            "speaker_id": speaker_id,
            "confidence": "low",
        }
        for timestamp in low_only
    )
    return sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
            0 if item["confidence"] == "high" else 1,
        ),
    )


def _process_one(
    task: Tuple[str, bool]
) -> Tuple[str, str, int, int, Optional[str]]:
    """Process one WAV and report high/low output segment counts."""
    audio_path = Path(task[0])
    overwrite = task[1]
    output_path = _output_path(audio_path)

    if (
        output_path.exists()
        and not overwrite
        and _has_current_schema(output_path)
    ):
        return ("skipped", str(output_path), 0, 0, None)

    try:
        from silero_vad import get_speech_timestamps, read_audio

        metadata_path = _metadata_path(audio_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "speaker_id" not in metadata:
            raise KeyError(f"'speaker_id' is missing from {metadata_path}")
        speaker_id = metadata["speaker_id"]

        waveform = read_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
        high_timestamps = get_speech_timestamps(
            waveform,
            _MODEL,
            sampling_rate=_SAMPLE_RATE,
            return_seconds=True,
            time_resolution=_TIME_RESOLUTION,
            threshold=_HIGH_THRESHOLD,
        )
        low_timestamps = get_speech_timestamps(
            waveform,
            _MODEL,
            sampling_rate=_SAMPLE_RATE,
            return_seconds=True,
            time_resolution=_TIME_RESOLUTION,
            threshold=_LOW_THRESHOLD,
        )
        segments = _combine_confidence_segments(
            high_timestamps,
            low_timestamps,
            speaker_id,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(
            f"{output_path.suffix}.tmp.{os.getpid()}"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(segments, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        high_count = sum(
            segment["confidence"] == "high" for segment in segments
        )
        return (
            "written",
            str(output_path),
            high_count,
            len(segments) - high_count,
            None,
        )
    except Exception as exc:
        return (
            "failed",
            str(audio_path),
            0,
            0,
            f"{type(exc).__name__}: {exc}",
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silero at thresholds 0.7 (high confidence) and 0.1 "
            "(non-overlapping low confidence) for one language folder."
        )
    )
    parser.add_argument(
        "language_dir",
        type=Path,
        help="one language folder, e.g. call_wise_delivery/bn_batch1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel worker processes (default: min(8, available CPUs))",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "regenerate all JSON files; otherwise current-schema files are "
            "skipped and missing/old-schema files are generated"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N WAVs (useful for a smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    language_dir = args.language_dir.resolve()
    if not language_dir.is_dir():
        raise SystemExit(f"Language folder does not exist: {language_dir}")

    audio_paths = sorted(language_dir.glob("*/audios/*.wav"))
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]
    if not audio_paths:
        raise SystemExit(f"No */audios/*.wav files found in {language_dir}")

    missing_metadata = [
        str(_metadata_path(path))
        for path in audio_paths
        if not _metadata_path(path).is_file()
    ]
    if missing_metadata:
        examples = "\n".join(f"  {path}" for path in missing_metadata[:10])
        raise SystemExit(
            f"Missing metadata for {len(missing_metadata)} WAV(s):\n{examples}"
        )

    total = len(audio_paths)
    print(
        f"Found {total} WAV file(s); using {args.workers} worker(s), "
        f"Silero sample rate {_SAMPLE_RATE} Hz, thresholds "
        f"high={_HIGH_THRESHOLD}, low={_LOW_THRESHOLD}, timestamp "
        f"decimals={_TIME_RESOLUTION}.",
        flush=True,
    )

    counts = {"written": 0, "skipped": 0, "failed": 0}
    high_segment_count = 0
    low_segment_count = 0
    started_at = time.monotonic()
    tasks = ((str(path), args.overwrite) for path in audio_paths)
    context = mp.get_context("spawn")

    with context.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(_process_one, tasks),
            start=1,
        ):
            status, path, high_segments, low_segments, error = result
            counts[status] += 1
            high_segment_count += high_segments
            low_segment_count += low_segments
            if error is not None:
                print(f"ERROR: {path}: {error}", file=sys.stderr, flush=True)
            if completed == 1 or completed % 25 == 0 or completed == total:
                elapsed = time.monotonic() - started_at
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"[{completed}/{total}] written={counts['written']} "
                    f"skipped={counts['skipped']} failed={counts['failed']} "
                    f"rate={rate:.2f} files/s",
                    flush=True,
                )

    print(
        f"Done: written={counts['written']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}, high_segments={high_segment_count}, "
        f"low_segments={low_segment_count}.",
        flush=True,
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
