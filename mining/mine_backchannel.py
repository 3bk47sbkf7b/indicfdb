#!/usr/bin/env python3
"""Mine clips in which one speaker gives short backchannel responses.

The script processes one language directory at a time.  Each conversation must
contain speaker-separated WAV and VAD files:

    LANGUAGE_DIR/
      CONVERSATION_ID/
        audios/
          speaker1.wav
          speaker2.wav
        vads/
          speaker1.json
          speaker2.json

Each VAD JSON is a non-empty list of ``start``, ``end``, ``speaker_id``, and
``confidence`` records.  Confidence is ``"high"`` or ``"low"``.  Times are
seconds on the original conversation timeline.  Mining uses only ``vads/``;
``audios/`` is read only after intervals have been chosen.

Definitions and mining rules
----------------------------
Raw VAD regions separated by a gap strictly smaller than 0.5 seconds are
merged into a *continuous speech segment*.  A continuous segment at most 1.2
seconds long is a backchannel; a longer one is a non-backchannel.

For each conversation, both speakers are tried as the "main" speaker.  A
qualifying clip:

* is strictly longer than 15 seconds;
* contains no gap over 1.5 seconds between main-speaker continuous segments;
* contains only main-speaker continuous segments strictly longer than 1 second;
* contains one or more other-speaker backchannel segments;
* contains no other-speaker non-backchannel segment;
* starts from 3 through 20 seconds before the first backchannel starts;
* ends from 3 through 20 seconds after the last backchannel ends; and
* does not start or end inside either speaker's raw VAD segment.
* contains no low-confidence VAD segment from either speaker.

Only high-confidence VAD segments construct main speech and backchannels.
Low-confidence segments are exclusion regions: a candidate with
positive-duration overlap with one is rejected.

Clip boundaries are the start of the first and end of the last included main
continuous segment.  Every qualifying contiguous main-speech subrange is
considered.  Candidates from both main-speaker orientations are ranked by
duration and accepted longest-first; candidates overlapping an accepted clip
are discarded.  Selected clips are ordered longest-to-shortest within each
conversation, so ``CONVERSATION_ID_0`` is that conversation's longest clip.

Output
------
Results are written to ``LANGUAGE_DIR/backchannel/``:

    backchannel/
      CONVERSATION_ID_0/
        main.wav
        other.wav
        metadata.json

The two WAVs cover the same original interval.  A source that ends slightly
early is zero-padded so both outputs remain sample-aligned.  ``clip_start`` and
``clip_end`` use the original conversation timeline.  Backchannel timestamps
are relative to the extracted clip, where time zero equals ``clip_start``.
All metadata times are written to three decimal places of numeric precision
(1 ms representation).

Examples:

    python mine_backchannel.py call_wise_delivery/bn_batch1 --workers 8
    python mine_backchannel.py call_wise_delivery/hi_batch1 --dry-run

An existing ``backchannel`` directory is left untouched unless ``--overwrite``
is supplied, in which case it is removed and regenerated in full.
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
MAX_MAIN_PAUSE_SECONDS = 1.5
MIN_MAIN_SPEECH_SECONDS = 1.0
MAX_BACKCHANNEL_SECONDS = 1.2
MIN_CLIP_SECONDS = 15.0
MIN_BACKCHANNEL_CONTEXT_SECONDS = 3.0
MAX_BACKCHANNEL_CONTEXT_SECONDS = 20.0
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
class Candidate:
    """A qualifying clip before longest-first overlap removal."""

    start: float
    end: float
    main_index: int
    other_index: int
    main_speaker_id: Any
    other_speaker_id: Any
    backchannels: Tuple[Segment, ...]

    @property
    def duration(self) -> float:
        return seconds_between(self.end, self.start)


class ContinuousSegmentIndex:
    """Locate continuous speech segments overlapping a candidate interval."""

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.segments = list(segments)
        self.starts = [segment.start for segment in segments]
        self.ends = [segment.end for segment in segments]
        non_backchannel_count = 0
        self.non_backchannel_prefix = [0]
        for segment in segments:
            if segment.duration > MAX_BACKCHANNEL_SECONDS:
                non_backchannel_count += 1
            self.non_backchannel_prefix.append(non_backchannel_count)

    def overlapping_range(self, start: float, end: float) -> Tuple[int, int]:
        """Return slice bounds for segments with positive overlap."""
        first = bisect.bisect_right(self.ends, start)
        last = bisect.bisect_left(self.starts, end)
        return first, last

    def overlaps(self, start: float, end: float) -> bool:
        """Return whether at least one indexed segment positively overlaps."""
        first, last = self.overlapping_range(start, end)
        return first < last

    def backchannels_only(
        self, start: float, end: float
    ) -> Optional[Tuple[Segment, ...]]:
        """Return overlapping backchannels, or ``None`` if the clip is invalid."""
        first, last = self.overlapping_range(start, end)
        if first == last:
            return None
        non_backchannels = (
            self.non_backchannel_prefix[last]
            - self.non_backchannel_prefix[first]
        )
        if non_backchannels:
            return None
        return tuple(self.segments[first:last])


class VadBoundaryIndex:
    """Check whether a timestamp falls strictly inside any raw VAD segment."""

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.starts: List[float] = []
        self.maximum_ends: List[float] = []
        maximum_end = float("-inf")
        for segment in segments:
            self.starts.append(segment.start)
            maximum_end = max(maximum_end, segment.end)
            self.maximum_ends.append(maximum_end)

    def cuts_segment(self, timestamp: float) -> bool:
        """Return true only when ``start < timestamp < end`` for some VAD."""
        index = bisect.bisect_left(self.starts, timestamp) - 1
        return index >= 0 and self.maximum_ends[index] > timestamp


def load_vad(path: Path) -> Tuple[List[Segment], List[Segment], Any]:
    """Return high segments, low exclusion segments, and the one speaker ID."""
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
    """Merge raw VAD segments when their gap is strictly below 0.5 seconds."""
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


def _valid_main_blocks(segments: Sequence[Segment]) -> List[List[Segment]]:
    """Split main speech into maximal blocks satisfying its local rules."""
    blocks: List[List[Segment]] = []
    block: List[Segment] = []
    for segment in segments:
        if segment.duration <= MIN_MAIN_SPEECH_SECONDS:
            if block:
                blocks.append(block)
            block = []
            continue
        if (
            block
            and seconds_between(segment.start, block[-1].end)
            <= MAX_MAIN_PAUSE_SECONDS
        ):
            block.append(segment)
        else:
            if block:
                blocks.append(block)
            block = [segment]
    if block:
        blocks.append(block)
    return blocks


def candidates_for_main(
    main_vad: Sequence[Segment],
    other_vad: Sequence[Segment],
    main_low_vad: Sequence[Segment],
    other_low_vad: Sequence[Segment],
    main_index: int,
    main_speaker_id: Any,
    other_speaker_id: Any,
) -> List[Candidate]:
    """Generate all qualifying candidates for one main-speaker orientation."""
    main_continuous = merge_continuous_speech(main_vad)
    other_continuous = merge_continuous_speech(other_vad)
    other_index = ContinuousSegmentIndex(other_continuous)
    low_confidence_index = ContinuousSegmentIndex(
        sorted(
            [*main_low_vad, *other_low_vad],
            key=lambda segment: (segment.start, segment.end),
        )
    )
    main_boundaries = VadBoundaryIndex(main_vad)
    other_boundaries = VadBoundaryIndex(other_vad)
    candidates: List[Candidate] = []

    for block in _valid_main_blocks(main_continuous):
        # A single long continuous main segment can itself form a valid clip.
        for first in range(len(block)):
            for last in range(first, len(block)):
                start = block[first].start
                end = block[last].end
                if seconds_between(end, start) <= MIN_CLIP_SECONDS:
                    continue
                if low_confidence_index.overlaps(start, end):
                    continue
                backchannels = other_index.backchannels_only(start, end)
                if not backchannels:
                    continue
                context_before = seconds_between(backchannels[0].start, start)
                context_after = seconds_between(end, backchannels[-1].end)
                if (
                    context_before < MIN_BACKCHANNEL_CONTEXT_SECONDS
                    or context_before > MAX_BACKCHANNEL_CONTEXT_SECONDS
                    or context_after < MIN_BACKCHANNEL_CONTEXT_SECONDS
                    or context_after > MAX_BACKCHANNEL_CONTEXT_SECONDS
                ):
                    continue
                if (
                    main_boundaries.cuts_segment(start)
                    or main_boundaries.cuts_segment(end)
                    or other_boundaries.cuts_segment(start)
                    or other_boundaries.cuts_segment(end)
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
                        backchannels=backchannels,
                    )
                )
    return candidates


def intervals_overlap(first: Candidate, second: Candidate) -> bool:
    """Return whether two clips share positive-duration audio."""
    return first.start < second.end and second.start < first.end


def select_non_overlapping(candidates: Iterable[Candidate]) -> List[Candidate]:
    """Accept candidates longest-first and return them in output rank order."""
    ranked = sorted(
        candidates,
        key=lambda item: (-item.duration, item.start, item.end, item.main_index),
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


def mine_conversation(conversation_dir: Path) -> List[Candidate]:
    """Mine both main-speaker orientations, then remove overlaps globally."""
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
        1,
        speaker1_id,
        speaker2_id,
    )
    candidates.extend(
        candidates_for_main(
            speaker2_vad,
            speaker1_vad,
            speaker2_low_vad,
            speaker1_low_vad,
            2,
            speaker2_id,
            speaker1_id,
        )
    )
    return select_non_overlapping(candidates)


def write_clip(
    output_root: Path,
    conversation_dir: Path,
    clip_index: int,
    candidate: Candidate,
) -> None:
    """Atomically write aligned WAVs and clip-relative backchannel metadata."""
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
            "backchannels": [
                {
                    "backchannel_start": round(
                        segment.start - candidate.start, TIMESTAMP_DECIMALS
                    ),
                    "backchannel_end": round(
                        segment.end - candidate.start, TIMESTAMP_DECIMALS
                    ),
                }
                for segment in candidate.backchannels
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
            "Mine >15 s clips in which the main speaker has no >1.5 s pause "
            "and the other speaker produces only <=1.2 s backchannels.\n\n"
            "The first backchannel must start 3-20 s into the clip and the "
            "clip must end 3-20 s after the last backchannel. Clip boundaries "
            "cannot cut through either speaker's VAD. Mining reads only vads/; "
            "audios/ supplies the aligned output WAVs."
        ),
        epilog=(
            "Examples:\n"
            "  python mine_backchannel.py "
            "call_wise_delivery/bn_batch1 --workers 8\n"
            "  python mine_backchannel.py "
            "call_wise_delivery/hi_batch1 --dry-run\n"
            "  python mine_backchannel.py "
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
        help="remove and regenerate an existing backchannel output folder",
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

    output_root = language_dir / "backchannel"
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
