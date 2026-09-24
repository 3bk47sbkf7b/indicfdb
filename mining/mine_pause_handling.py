#!/usr/bin/env python3
"""Mine examples of a speaker talking through short pauses.

This script processes one language directory at a time.  It expects each
conversation to have two speaker-separated audio files and two matching VAD
(voice activity detection) files:

    LANGUAGE_DIR/
      CONVERSATION_ID/
        audios/
          speaker1.wav
          speaker2.wav
        vads/
          speaker1.json
          speaker2.json

Each VAD JSON is a list of ``start``, ``end``, ``speaker_id``, and
``confidence`` records.  Confidence is either ``"high"`` or ``"low"``.  VAD
times are seconds on the original conversation timeline.  The script does not
read transcripts or the source metadata directory: mining decisions and
speaker IDs come exclusively from ``vads/``, while ``audios/`` is used only to
extract the selected waveforms.

What is a pause-handling clip?
--------------------------------
For each conversation, the script tries both speaker 1 and speaker 2 as the
"main" speaker.  The other speaker must remain silent for the entire clip.

Raw VAD regions separated by a gap strictly smaller than 0.5 seconds are first
merged into a *continuous speech segment*.  A qualifying clip then:

* is strictly longer than 5 seconds;
* contains at least two continuous speech segments and therefore at least one
  small pause;
* has only small pauses, defined as gaps from 0.5 through 1.5 seconds,
  inclusive (a gap over 1.5 seconds is a large pause);
* contains only continuous speech segments strictly longer than 1 second; and
* has no VAD activity from the other speaker anywhere in its interval,
  including inside a pause.
* has more than 3 seconds of main-speaker speech before its first pause and
  more than 3 seconds of main-speaker speech after its last pause.
* has no low-confidence VAD segment from either speaker anywhere in the clip.

Only high-confidence VAD segments participate in speech and pause construction.
Low-confidence segments instead act as exclusion regions: any candidate with
positive-duration overlap with one is rejected.

All qualifying subintervals are considered.  Candidates are ranked by duration
and accepted longest-first, rejecting any candidate that overlaps a previously
accepted one.  This implements the requirement to prefer larger clips while
ensuring that mined clips from a conversation never overlap.  The accepted
clips are finally ordered longest-to-shortest within that conversation, so
``CONVERSATION_ID_0`` is its longest clip.

Output
------
Results are written beneath ``LANGUAGE_DIR/pause_handling/``:

    pause_handling/
      CONVERSATION_ID_0/
        main.wav
        other.wav
        metadata.json

``main.wav`` and ``other.wav`` cover the same original time interval.  If one
source WAV ends slightly earlier than the other, it is zero-padded so the
outputs remain sample-aligned.  In ``metadata.json``, ``clip_start`` and
``clip_end`` use the original conversation timeline, ``clip_duration`` is their
difference in seconds, and pause timestamps are relative to the beginning of
the extracted clip (time zero is ``clip_start``).  All metadata times are
written to three decimal places of numeric precision (1 ms representation).

Typical usage:

    python mine_pause_handling.py call_wise_delivery/bn_batch1 --workers 8
    python mine_pause_handling.py call_wise_delivery/hi_batch1 --dry-run

An existing ``pause_handling`` directory is never changed unless
``--overwrite`` is explicitly supplied.  With that option, the existing output
directory is removed and regenerated in full.
"""

from __future__ import annotations

import argparse
import bisect
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Mining thresholds.  Comparisons intentionally match the definitions above:
# join gaps are strictly below 0.5 s, while speech and clip durations are
# strictly above their minimums.
JOIN_GAP_SECONDS = 0.5
MAX_SMALL_PAUSE_SECONDS = 1.5
MIN_CONTINUOUS_SPEECH_SECONDS = 1.0
MIN_CLIP_SECONDS = 5.0
MIN_PAUSE_CONTEXT_SECONDS = 3.0
TIMESTAMP_DECIMALS = 3


@dataclass(frozen=True)
class Segment:
    """A half-open time interval ``[start, end)`` in conversation seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Candidate:
    """A qualifying clip before longest-first overlap removal.

    ``start``, ``end``, and ``pauses`` are stored on the original conversation
    timeline.  Pause times are converted to the clip timeline only when the
    output metadata is serialized.
    """

    start: float
    end: float
    main_index: int
    other_index: int
    main_speaker_id: Any
    other_speaker_id: Any
    pauses: Tuple[Tuple[float, float], ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


class SegmentIndex:
    """Efficient positive-overlap queries for sorted VAD segments.

    The prefix maximum of segment ends makes each query logarithmic.  Merely
    touching an interval boundary is not considered speech inside the interval.
    """

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.starts: List[float] = []
        self.maximum_ends: List[float] = []
        maximum_end = float("-inf")
        for segment in segments:
            self.starts.append(segment.start)
            maximum_end = max(maximum_end, segment.end)
            self.maximum_ends.append(maximum_end)

    def overlaps(self, start: float, end: float) -> bool:
        """Return whether any segment has positive-duration overlap."""
        index = bisect.bisect_left(self.starts, end) - 1
        return index >= 0 and self.maximum_ends[index] > start


def load_vad(path: Path) -> Tuple[List[Segment], List[Segment], Any]:
    """Load and validate one speaker's VAD file.

    Returns time-sorted high-confidence segments, low-confidence exclusion
    segments, and the single speaker ID shared by every record.  Empty VAD
    lists are rejected because they do not contain the required speaker ID.
    """
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    segments: List[Segment] = []
    low_segments: List[Segment] = []
    speaker_ids = set()
    for record in records:
        if set(record) != {"start", "end", "speaker_id", "confidence"}:
            raise ValueError(f"Unexpected VAD fields in {path}: {sorted(record)}")
        confidence = record["confidence"]
        if confidence not in {"high", "low"}:
            raise ValueError(f"Invalid confidence {confidence!r} in {path}")
        start = float(record["start"])
        end = float(record["end"])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid VAD interval [{start}, {end}] in {path}")
        segment = Segment(start, end)
        if confidence == "high":
            segments.append(segment)
        else:
            low_segments.append(segment)
        speaker_ids.add(record["speaker_id"])

    segments.sort(key=lambda segment: (segment.start, segment.end))
    low_segments.sort(key=lambda segment: (segment.start, segment.end))
    if len(speaker_ids) != 1:
        raise ValueError(f"Expected exactly one speaker_id in {path}")
    return segments, low_segments, next(iter(speaker_ids))


def merge_continuous_speech(segments: Sequence[Segment]) -> List[Segment]:
    """Turn raw VAD regions into continuous speech segments.

    Neighboring or overlapping VAD regions are joined when the uncovered gap
    is strictly less than :data:`JOIN_GAP_SECONDS`.  A gap of exactly 0.5
    seconds is deliberately retained as a small pause.
    """
    if not segments:
        return []

    merged: List[Segment] = []
    current_start = segments[0].start
    current_end = segments[0].end
    for segment in segments[1:]:
        if segment.start - current_end < JOIN_GAP_SECONDS:
            current_end = max(current_end, segment.end)
        else:
            merged.append(Segment(current_start, current_end))
            current_start = segment.start
            current_end = segment.end
    merged.append(Segment(current_start, current_end))
    return merged


def candidates_for_main(
    main_vad: Sequence[Segment],
    other_vad: Sequence[Segment],
    main_low_vad: Sequence[Segment],
    other_low_vad: Sequence[Segment],
    main_index: int,
    main_speaker_id: Any,
    other_speaker_id: Any,
) -> List[Candidate]:
    """Generate all qualifying clips for one choice of main speaker.

    First, maximal valid blocks are built.  Every speech segment in a block is
    longer than one second, every link is a small pause, and no other-speaker
    VAD overlaps either the speech or its intervening pauses.  Every contiguous
    subrange of at least two speech segments is then emitted if its total span
    is longer than five seconds, its first pause starts more than three seconds
    after the clip starts, and its last pause ends more than three seconds
    before the clip ends.

    Emitting subranges as well as maximal blocks matters when a long candidate
    for the opposite speaker wins during global overlap removal: a shorter,
    non-overlapping portion may still be usable.
    """
    continuous = merge_continuous_speech(main_vad)
    other_index = SegmentIndex(other_vad)
    low_confidence_index = SegmentIndex(
        sorted(
            [*main_low_vad, *other_low_vad],
            key=lambda segment: (segment.start, segment.end),
        )
    )

    # Build maximal blocks in which every continuous segment is >1 s, every
    # intervening gap is a small pause, and the other speaker is silent.
    blocks: List[List[Segment]] = []
    current_block: List[Segment] = []
    previous: Optional[Segment] = None
    for segment in continuous:
        clean_segment = (
            segment.duration > MIN_CONTINUOUS_SPEECH_SECONDS
            and not other_index.overlaps(segment.start, segment.end)
        )
        connected = False
        if clean_segment and current_block and previous is not None:
            gap = segment.start - previous.end
            connected = (
                JOIN_GAP_SECONDS <= gap <= MAX_SMALL_PAUSE_SECONDS
                and not other_index.overlaps(previous.end, segment.start)
            )

        if not clean_segment:
            if current_block:
                blocks.append(current_block)
            current_block = []
        elif connected:
            current_block.append(segment)
        else:
            if current_block:
                blocks.append(current_block)
            current_block = [segment]
        previous = segment
    if current_block:
        blocks.append(current_block)

    candidates: List[Candidate] = []
    for block in blocks:
        # At least two continuous segments are needed to contain a small pause.
        for first in range(len(block) - 1):
            for last in range(first + 1, len(block)):
                start = block[first].start
                end = block[last].end
                if end - start <= MIN_CLIP_SECONDS:
                    continue
                if low_confidence_index.overlaps(start, end):
                    continue
                pauses = tuple(
                    (block[index].end, block[index + 1].start)
                    for index in range(first, last)
                )
                speech_before_first_pause = pauses[0][0] - start
                speech_after_last_pause = end - pauses[-1][1]
                if (
                    speech_before_first_pause <= MIN_PAUSE_CONTEXT_SECONDS
                    or speech_after_last_pause <= MIN_PAUSE_CONTEXT_SECONDS
                ):
                    continue
                candidates.append(
                    Candidate(
                        start=start,
                        end=end,
                        main_index=main_index,
                        other_index=3 - main_index,
                        main_speaker_id=main_speaker_id,
                        other_speaker_id=other_speaker_id,
                        pauses=pauses,
                    )
                )
    return candidates


def intervals_overlap(first: Candidate, second: Candidate) -> bool:
    """Return whether two candidate clips share positive-duration audio."""

    return first.start < second.end and second.start < first.end


def select_non_overlapping(candidates: Iterable[Candidate]) -> List[Candidate]:
    """Choose non-overlapping clips using longest-first greedy selection.

    Equal-duration ties are resolved deterministically by start time, end time,
    and main-speaker index.  The chosen clips are returned in the duration order
    used to assign the ``_0``, ``_1``, ... output suffixes.
    """
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.duration,
            item.start,
            item.end,
            item.main_index,
        ),
    )
    selected: List[Candidate] = []
    for candidate in ranked:
        if not any(intervals_overlap(candidate, existing) for existing in selected):
            selected.append(candidate)
    return sorted(
        selected,
        key=lambda item: (-item.duration, item.start, item.main_index),
    )


def _read_padded_frames(
    source: wave.Wave_read,
    start_frame: int,
    frame_count: int,
    bytes_per_frame: int,
) -> bytes:
    """Read exactly ``frame_count`` frames, padding unavailable tail with zero."""

    source.setpos(min(start_frame, source.getnframes()))
    available = max(0, min(frame_count, source.getnframes() - start_frame))
    frames = source.readframes(available)
    missing = frame_count * bytes_per_frame - len(frames)
    if missing > 0:
        frames += b"\0" * missing
    return frames


def extract_wav(source_path: Path, output_path: Path, start: float, end: float) -> None:
    """Extract ``[start, end)`` from an uncompressed PCM WAV.

    Seconds are converted to the nearest sample boundary.  Reading past the
    source's final frame produces silence, which keeps main and other clips
    aligned when the two original recordings differ slightly in length.
    """

    with wave.open(str(source_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"Only uncompressed PCM WAV is supported: {source_path}")
        sample_rate = source.getframerate()
        start_frame = round(start * sample_rate)
        end_frame = round(end * sample_rate)
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            raise ValueError(f"Empty clip requested from {source_path}: {start}-{end}")
        bytes_per_frame = source.getnchannels() * source.getsampwidth()
        frames = _read_padded_frames(
            source,
            start_frame=start_frame,
            frame_count=frame_count,
            bytes_per_frame=bytes_per_frame,
        )
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        compression_type = source.getcomptype()
        compression_name = source.getcompname()

    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.setcomptype(compression_type, compression_name)
        output.writeframes(frames)


def mine_conversation(conversation_dir: Path) -> List[Candidate]:
    """Mine both possible main-speaker orientations for one conversation.

    Speaker IDs are copied from the VAD records.  Candidate sets for speaker 1
    and speaker 2 are combined before overlap removal, so clips cannot overlap
    even when their main-speaker roles differ.
    """

    speaker1_vad, speaker1_low_vad, speaker1_id = load_vad(
        conversation_dir / "vads" / "speaker1.json"
    )
    speaker2_vad, speaker2_low_vad, speaker2_id = load_vad(
        conversation_dir / "vads" / "speaker2.json"
    )
    candidates = candidates_for_main(
        speaker1_vad,
        speaker2_vad,
        speaker1_low_vad,
        speaker2_low_vad,
        main_index=1,
        main_speaker_id=speaker1_id,
        other_speaker_id=speaker2_id,
    )
    candidates.extend(
        candidates_for_main(
            speaker2_vad,
            speaker1_vad,
            speaker2_low_vad,
            speaker1_low_vad,
            main_index=2,
            main_speaker_id=speaker2_id,
            other_speaker_id=speaker1_id,
        )
    )
    return select_non_overlapping(candidates)


def write_clip(
    output_root: Path,
    conversation_dir: Path,
    clip_index: int,
    candidate: Candidate,
) -> None:
    """Atomically write one selected clip directory.

    Audio is cut using original-timeline candidate boundaries.  Metadata keeps
    those original ``clip_start``/``clip_end`` values, while each pause is
    shifted by ``candidate.start`` to make it relative to ``main.wav`` and
    ``other.wav``.  A temporary directory prevents partially written clips from
    appearing under their final names.
    """

    clip_name = f"{conversation_dir.name}_{clip_index}"
    final_dir = output_root / clip_name
    temporary_dir = output_root / f".{clip_name}.tmp.{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(f"Output already exists: {final_dir}")

    temporary_dir.mkdir(parents=False)
    try:
        extract_wav(
            conversation_dir / "audios" / f"speaker{candidate.main_index}.wav",
            temporary_dir / "main.wav",
            candidate.start,
            candidate.end,
        )
        extract_wav(
            conversation_dir / "audios" / f"speaker{candidate.other_index}.wav",
            temporary_dir / "other.wav",
            candidate.start,
            candidate.end,
        )
        metadata: Dict[str, Any] = {
            "clip_start": round(candidate.start, TIMESTAMP_DECIMALS),
            "clip_end": round(candidate.end, TIMESTAMP_DECIMALS),
            "clip_duration": round(candidate.duration, TIMESTAMP_DECIMALS),
            "main_speaker_id": candidate.main_speaker_id,
            "other_speaker_id": candidate.other_speaker_id,
            "main_speaker_index": candidate.main_index,
            "other_speaker_index": candidate.other_index,
            "conversation_id": conversation_dir.name,
            "pauses": [
                {
                    "pause_start": round(
                        start - candidate.start, TIMESTAMP_DECIMALS
                    ),
                    "pause_end": round(
                        end - candidate.start, TIMESTAMP_DECIMALS
                    ),
                }
                for start, end in candidate.pauses
            ],
        }
        with (temporary_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def process_conversation(task: Tuple[str, str, bool]) -> Tuple[str, int, float, Optional[str]]:
    """Worker entry point that mines and optionally writes one conversation."""

    conversation_dir = Path(task[0])
    output_root = Path(task[1])
    dry_run = task[2]
    try:
        selected = mine_conversation(conversation_dir)
        if not dry_run:
            for clip_index, candidate in enumerate(selected):
                write_clip(output_root, conversation_dir, clip_index, candidate)
        return (
            conversation_dir.name,
            len(selected),
            sum(candidate.duration for candidate in selected),
            None,
        )
    except Exception as exc:
        return (
            conversation_dir.name,
            0,
            0.0,
            f"{type(exc).__name__}: {exc}",
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build and parse the command-line interface."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Mine clips in which one speaker continues talking across one or "
            "more short pauses while the other speaker remains silent.\n\n"
            "Mining thresholds:\n"
            "  continuous speech: merge VAD gaps < 0.5 s\n"
            "  small pause:       gap from 0.5 s through 1.5 s\n"
            "  large pause:       gap > 1.5 s (not allowed in a clip)\n"
            "  speech segment:    must be > 1 s\n"
            "  pause context:     > 3 s before first and after last pause\n"
            "  extracted clip:    must be > 5 s and contain a small pause\n\n"
            "The command processes one language directory. It reads only its "
            "conversation-level vads/ and audios/ folders and writes "
            "LANGUAGE_DIR/pause_handling/."
        ),
        epilog=(
            "Examples:\n"
            "  python mine_pause_handling.py "
            "call_wise_delivery/bn_batch1 --workers 8\n"
            "  python mine_pause_handling.py "
            "call_wise_delivery/hi_batch1 --dry-run\n"
            "  python mine_pause_handling.py "
            "call_wise_delivery/bn_batch1 --workers 8 --overwrite"
        ),
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
        help="parallel conversations to process (default: min(8, CPUs))",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove and regenerate an existing pause_handling output folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mine candidates and report counts without writing output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N conversations (for smoke tests)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run language-level discovery, parallel mining, and summary reporting."""

    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    language_dir = args.language_dir.resolve()
    if not language_dir.is_dir():
        raise SystemExit(f"Language folder does not exist: {language_dir}")

    conversations = sorted(
        path
        for path in language_dir.iterdir()
        if path.is_dir() and (path / "vads").is_dir() and (path / "audios").is_dir()
    )
    if args.limit is not None:
        conversations = conversations[: args.limit]
    if not conversations:
        raise SystemExit(f"No conversation folders found in {language_dir}")

    output_root = language_dir / "pause_handling"
    if not args.dry_run:
        if output_root.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"Output folder already exists: {output_root}\n"
                    "Use --overwrite to regenerate it."
                )
            shutil.rmtree(output_root)
        output_root.mkdir()

    total_clips = 0
    total_duration = 0.0
    failed = 0
    started = time.monotonic()
    tasks = (
        (str(conversation), str(output_root), args.dry_run)
        for conversation in conversations
    )
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(process_conversation, tasks),
            start=1,
        ):
            conversation_id, clip_count, duration, error = result
            if error:
                failed += 1
                print(
                    f"ERROR: conversation {conversation_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            total_clips += clip_count
            total_duration += duration
            if completed == 1 or completed % 25 == 0 or completed == len(conversations):
                elapsed = time.monotonic() - started
                print(
                    f"[{completed}/{len(conversations)}] clips={total_clips} "
                    f"duration_hours={total_duration / 3600:.2f} "
                    f"failed={failed} rate={completed / elapsed:.2f} conv/s",
                    flush=True,
                )

    action = "Would write" if args.dry_run else "Wrote"
    print(
        f"{action} {total_clips} clips ({total_duration / 3600:.2f} hours) "
        f"from {len(conversations)} conversations; failed={failed}.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
