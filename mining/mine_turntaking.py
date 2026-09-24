#!/usr/bin/env python3
"""Mine clean two-speaker turn transitions from VAD and aligned WAV files.

The script processes exactly one language directory at a time.  Each
conversation must have this layout:

    LANGUAGE_DIR/
      CONVERSATION_ID/
        audios/
          speaker1.wav
          speaker2.wav
        vads/
          speaker1.json
          speaker2.json

Each VAD JSON is a non-empty list of ``start``, ``end``, ``speaker_id``, and
``confidence`` records.  Times are seconds on the original conversation
timeline, and confidence is either ``"high"`` or ``"low"``.  Mining decisions
and speaker IDs come only from ``vads/``.  The source ``audios/`` are used only
to extract selected intervals.

Turn construction
-----------------
For each speaker independently, high-confidence VAD regions separated by a gap
strictly smaller than 0.5 seconds are merged into continuous speech segments.
The two speakers' continuous segments are then sorted together by start time.
Consecutive entries belonging to the same speaker form one turn; a speaker
change creates a turn boundary.

Every adjacent pair of turns is evaluated in its temporal direction.  The
speaker owning the first turn is called ``main`` and the speaker owning the
second is called ``other``.  A qualifying pair:

* contains exactly the complete main turn followed by the complete other turn;
* has a main-turn span strictly longer than 5 seconds;
* has an other-turn span strictly longer than 5 seconds;
* has no gap over 1.5 seconds inside either turn;
* has no overlap between the main and other continuous speech segments;
* has a non-negative handoff gap strictly smaller than 0.5 seconds; and
* has no positive-duration overlap with any low-confidence VAD segment from
  either speaker.

A turn's span is measured from its first continuous segment's start through its
last continuous segment's end, including permitted internal pauses.  Clip
boundaries are the main turn's start and the other turn's end, so both turns
are included completely and no high-confidence VAD is cut at an edge.

All qualifying pairs from a conversation are ranked by clip duration and
accepted longest-first.  A candidate overlapping an already accepted clip is
discarded.  Accepted clips are named in duration order, making
``CONVERSATION_ID_0`` the longest clip from that conversation.

Output
------
Results are written beneath ``LANGUAGE_DIR/turntaking/``:

    turntaking/
      CONVERSATION_ID_0/
        main.wav
        other.wav
        metadata.json

The WAV files contain the same original time interval.  A source WAV that ends
slightly early is zero-padded so the pair remains sample-aligned.
``clip_start`` and ``clip_end`` use the original conversation timeline.
``turn_start`` and ``turn_end`` are relative to the extracted clip.  All
metadata times are written to three decimal places of numeric precision
(1 ms representation).

Examples:

    python mine_turntaking.py call_wise_delivery/bn_batch1 --workers 8
    python mine_turntaking.py call_wise_delivery/hi_batch1 --dry-run

An existing ``turntaking`` folder is left untouched unless ``--overwrite`` is
supplied, in which case it is removed and regenerated completely.
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


JOIN_GAP_SECONDS = 0.5
MAX_INTERNAL_PAUSE_SECONDS = 1.5
MAX_HANDOFF_GAP_SECONDS = 0.5
MIN_TURN_SECONDS = 5.0
TIMESTAMP_DECIMALS = 3


def seconds_between(later: float, earlier: float) -> float:
    """Subtract decimal timestamps without binary floating-point artifacts."""
    return round(later - earlier, 10)


@dataclass(frozen=True)
class Segment:
    """A half-open interval ``[start, end)`` in conversation seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return seconds_between(self.end, self.start)


@dataclass(frozen=True)
class LabeledSegment:
    """A continuous speech segment associated with speaker 1 or speaker 2."""

    start: float
    end: float
    speaker_index: int


@dataclass(frozen=True)
class Turn:
    """One maximal run of same-speaker continuous segments."""

    speaker_index: int
    segments: Tuple[Segment, ...]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return seconds_between(self.end, self.start)

    @property
    def has_large_pause(self) -> bool:
        return any(
            seconds_between(current.start, previous.end)
            > MAX_INTERNAL_PAUSE_SECONDS
            for previous, current in zip(self.segments, self.segments[1:])
        )


@dataclass(frozen=True)
class Candidate:
    """A qualifying adjacent pair of main and other turns."""

    main_turn: Turn
    other_turn: Turn
    main_speaker_id: Any
    other_speaker_id: Any

    @property
    def start(self) -> float:
        return self.main_turn.start

    @property
    def end(self) -> float:
        return self.other_turn.end

    @property
    def duration(self) -> float:
        return seconds_between(self.end, self.start)

    @property
    def main_index(self) -> int:
        return self.main_turn.speaker_index

    @property
    def other_index(self) -> int:
        return self.other_turn.speaker_index


class SegmentIndex:
    """Answer positive-overlap queries for sorted segments in logarithmic time."""

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.starts: List[float] = []
        self.maximum_ends: List[float] = []
        maximum_end = float("-inf")
        for segment in segments:
            self.starts.append(segment.start)
            maximum_end = max(maximum_end, segment.end)
            self.maximum_ends.append(maximum_end)

    def overlaps(self, start: float, end: float) -> bool:
        """Return whether any segment positively overlaps ``[start, end)``."""
        index = bisect.bisect_left(self.starts, end) - 1
        return index >= 0 and self.maximum_ends[index] > start


def load_vad(path: Path) -> Tuple[List[Segment], List[Segment], Any]:
    """Return high segments, low exclusion segments, and the one speaker ID."""
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    high: List[Segment] = []
    low: List[Segment] = []
    speaker_ids = set()
    expected_fields = {"start", "end", "speaker_id", "confidence"}
    for record in records:
        if set(record) != expected_fields:
            raise ValueError(f"Unexpected VAD fields in {path}: {sorted(record)}")
        confidence = record["confidence"]
        if confidence not in {"high", "low"}:
            raise ValueError(f"Invalid confidence {confidence!r} in {path}")
        start = float(record["start"])
        end = float(record["end"])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid VAD interval [{start}, {end}] in {path}")
        segment = Segment(start, end)
        (high if confidence == "high" else low).append(segment)
        speaker_ids.add(record["speaker_id"])

    high.sort(key=lambda segment: (segment.start, segment.end))
    low.sort(key=lambda segment: (segment.start, segment.end))
    if len(speaker_ids) != 1:
        raise ValueError(f"Expected exactly one speaker_id in {path}")
    return high, low, next(iter(speaker_ids))


def merge_continuous_speech(segments: Sequence[Segment]) -> List[Segment]:
    """Join VAD regions whose intervening gap is strictly below 0.5 seconds."""
    if not segments:
        return []
    merged: List[Segment] = []
    start = segments[0].start
    end = segments[0].end
    for segment in segments[1:]:
        if seconds_between(segment.start, end) < JOIN_GAP_SECONDS:
            end = max(end, segment.end)
        else:
            merged.append(Segment(start, end))
            start, end = segment.start, segment.end
    merged.append(Segment(start, end))
    return merged


def build_turns(
    speaker1_segments: Sequence[Segment],
    speaker2_segments: Sequence[Segment],
) -> List[Turn]:
    """Sort both speakers' continuous segments and group same-speaker runs."""
    labeled = [
        LabeledSegment(segment.start, segment.end, 1)
        for segment in speaker1_segments
    ]
    labeled.extend(
        LabeledSegment(segment.start, segment.end, 2)
        for segment in speaker2_segments
    )
    labeled.sort(
        key=lambda segment: (
            segment.start,
            segment.end,
            segment.speaker_index,
        )
    )

    turns: List[Turn] = []
    current_speaker: Optional[int] = None
    current_segments: List[Segment] = []
    for segment in labeled:
        plain_segment = Segment(segment.start, segment.end)
        if segment.speaker_index == current_speaker:
            current_segments.append(plain_segment)
        else:
            if current_segments and current_speaker is not None:
                turns.append(Turn(current_speaker, tuple(current_segments)))
            current_speaker = segment.speaker_index
            current_segments = [plain_segment]
    if current_segments and current_speaker is not None:
        turns.append(Turn(current_speaker, tuple(current_segments)))
    return turns


def candidates_from_turns(
    turns: Sequence[Turn],
    low_segments: Sequence[Segment],
    speaker_ids: Dict[int, Any],
) -> List[Candidate]:
    """Evaluate every adjacent directed turn pair against all mining rules."""
    low_index = SegmentIndex(low_segments)
    candidates: List[Candidate] = []
    for turn_index, (main_turn, other_turn) in enumerate(
        zip(turns, turns[1:])
    ):
        if main_turn.speaker_index == other_turn.speaker_index:
            raise AssertionError("Turn grouping produced adjacent equal speakers")
        if (
            main_turn.duration <= MIN_TURN_SECONDS
            or other_turn.duration <= MIN_TURN_SECONDS
            or main_turn.has_large_pause
            or other_turn.has_large_pause
        ):
            continue

        # Since each turn contains one speaker, complete cross-speaker
        # non-overlap is equivalent to the first turn ending no later than the
        # second turn starts.
        handoff_gap = seconds_between(other_turn.start, main_turn.end)
        if not 0 <= handoff_gap < MAX_HANDOFF_GAP_SECONDS:
            continue
        has_extra_turn_activity = any(
            segment.start < other_turn.end
            and main_turn.start < segment.end
            for other_index, turn in enumerate(turns)
            if other_index not in {turn_index, turn_index + 1}
            for segment in turn.segments
        )
        if has_extra_turn_activity:
            continue
        if low_index.overlaps(main_turn.start, other_turn.end):
            continue
        candidates.append(
            Candidate(
                main_turn=main_turn,
                other_turn=other_turn,
                main_speaker_id=speaker_ids[main_turn.speaker_index],
                other_speaker_id=speaker_ids[other_turn.speaker_index],
            )
        )
    return candidates


def intervals_overlap(first: Candidate, second: Candidate) -> bool:
    """Return whether two candidate clips share positive-duration audio."""
    return first.start < second.end and second.start < first.end


def select_non_overlapping(candidates: Iterable[Candidate]) -> List[Candidate]:
    """Select candidates longest-first and return them in output rank order."""
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


def mine_conversation(conversation_dir: Path) -> List[Candidate]:
    """Construct turns and return longest-first non-overlapping candidates."""
    speaker1_high, speaker1_low, speaker1_id = load_vad(
        conversation_dir / "vads" / "speaker1.json"
    )
    speaker2_high, speaker2_low, speaker2_id = load_vad(
        conversation_dir / "vads" / "speaker2.json"
    )
    turns = build_turns(
        merge_continuous_speech(speaker1_high),
        merge_continuous_speech(speaker2_high),
    )
    candidates = candidates_from_turns(
        turns,
        sorted(
            [*speaker1_low, *speaker2_low],
            key=lambda segment: (segment.start, segment.end),
        ),
        {1: speaker1_id, 2: speaker2_id},
    )
    return select_non_overlapping(candidates)


def _read_padded_frames(
    source: wave.Wave_read,
    start_frame: int,
    frame_count: int,
    bytes_per_frame: int,
) -> bytes:
    """Read the requested frames and zero-pad an unavailable source tail."""
    source.setpos(min(start_frame, source.getnframes()))
    available = max(0, min(frame_count, source.getnframes() - start_frame))
    frames = source.readframes(available)
    missing = frame_count * bytes_per_frame - len(frames)
    if missing > 0:
        frames += b"\0" * missing
    return frames


def extract_wav(source_path: Path, output_path: Path, start: float, end: float) -> None:
    """Extract a sample-aligned interval from an uncompressed PCM WAV."""
    with wave.open(str(source_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"Only uncompressed PCM WAV is supported: {source_path}")
        sample_rate = source.getframerate()
        start_frame = round(start * sample_rate)
        end_frame = round(end * sample_rate)
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            raise ValueError(f"Empty clip requested from {source_path}: {start}-{end}")
        frames = _read_padded_frames(
            source,
            start_frame,
            frame_count,
            source.getnchannels() * source.getsampwidth(),
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


def write_clip(
    output_root: Path,
    conversation_dir: Path,
    clip_index: int,
    candidate: Candidate,
) -> None:
    """Atomically write aligned WAVs and clip-relative turn metadata."""
    clip_name = f"{conversation_dir.name}_{clip_index}"
    final_dir = output_root / clip_name
    temporary_dir = output_root / f".{clip_name}.tmp.{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(f"Output already exists: {final_dir}")
    temporary_dir.mkdir()
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
            "turns": [
                {
                    "turn_start": round(
                        candidate.main_turn.start - candidate.start,
                        TIMESTAMP_DECIMALS,
                    ),
                    "turn_end": round(
                        candidate.main_turn.end - candidate.start,
                        TIMESTAMP_DECIMALS,
                    ),
                    "speaker": "main",
                },
                {
                    "turn_start": round(
                        candidate.other_turn.start - candidate.start,
                        TIMESTAMP_DECIMALS,
                    ),
                    "turn_end": round(
                        candidate.other_turn.end - candidate.start,
                        TIMESTAMP_DECIMALS,
                    ),
                    "speaker": "other",
                },
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
    """Worker entry point for mining and optionally writing one conversation."""
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
        return conversation_dir.name, 0, 0.0, f"{type(exc).__name__}: {exc}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the one-language-folder command-line interface."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Mine adjacent >5 s speaker turns with a non-overlapping <0.5 s "
            "handoff, no >1.5 s internal pause, and no low-confidence VAD.\n\n"
            "Mining reads only conversation vads/ folders; audios/ supplies "
            "the aligned output WAVs."
        ),
        epilog=(
            "Examples:\n"
            "  python mine_turntaking.py "
            "call_wise_delivery/bn_batch1 --workers 8\n"
            "  python mine_turntaking.py "
            "call_wise_delivery/hi_batch1 --dry-run\n"
            "  python mine_turntaking.py "
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
        help="remove and regenerate an existing turntaking output folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mine and report counts without writing output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N conversations (for smoke tests)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Discover conversations, run workers, and report language-level totals."""
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

    output_root = language_dir / "turntaking"
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
            pool.imap_unordered(process_conversation, tasks), start=1
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
