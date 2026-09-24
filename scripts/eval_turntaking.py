#!/usr/bin/env python3
"""Evaluate whether and how quickly an agent responds after a user's turn.

Raw Silero VAD regions ending before ``input_turn_end`` are evaluated for an
early takeover with the pause-handling thresholds.  Regions ending after
``input_turn_end`` are evaluated as responses with the turn-taking thresholds.
A raw region crossing the boundary is split and supplied to both sides.
"""

import argparse
import json
import os
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


CATEGORY = "turntaking"
SAMPLE_RATE = 16_000
RESPONSE_MAX_JOIN_GAP_SECONDS = 1.5
RESPONSE_MIN_DURATION_SECONDS = 1.0
EARLY_MAX_JOIN_GAP_SECONDS = 0.2
EARLY_MAX_BACKCHANNEL_SECONDS = 2.0
VAD_THRESHOLD = 0.7
DEFAULT_WORKERS = 16

_thread_state = threading.local()


def merge_speech_segments(
    segments: Sequence[Dict[str, int]],
    sample_rate: int = SAMPLE_RATE,
    max_gap_seconds: float = RESPONSE_MAX_JOIN_GAP_SECONDS,
) -> List[Tuple[int, int]]:
    """Join VAD regions whose gap is strictly below ``max_gap_seconds``."""
    if not segments:
        return []

    max_gap_samples = max_gap_seconds * sample_rate
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
    input_turn_end: float,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """Build the turn-taking score from sample-based Silero timestamps."""
    turn_end_sample = round(input_turn_end * sample_rate)
    early_segments = []
    response_segments = []
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start < turn_end_sample < end:
            early_segments.append({"start": start, "end": turn_end_sample})
            response_segments.append({"start": turn_end_sample, "end": end})
        elif end < turn_end_sample:
            early_segments.append({"start": start, "end": end})
        elif end > turn_end_sample:
            response_segments.append({"start": start, "end": end})

    early_takeovers = []
    for start, end in merge_speech_segments(
        early_segments,
        sample_rate,
        EARLY_MAX_JOIN_GAP_SECONDS,
    ):
        duration = (end - start) / sample_rate
        if duration > EARLY_MAX_BACKCHANNEL_SECONDS:
            early_takeovers.append(
                {
                    "start": round(start / sample_rate, 3),
                    "end": round(end / sample_rate, 3),
                }
            )

    responses = []
    first_response_start = None
    for start, end in merge_speech_segments(
        response_segments,
        sample_rate,
        RESPONSE_MAX_JOIN_GAP_SECONDS,
    ):
        duration = (end - start) / sample_rate
        end_seconds = end / sample_rate
        if duration >= RESPONSE_MIN_DURATION_SECONDS:
            if first_response_start is None:
                first_response_start = start / sample_rate
            responses.append(
                {
                    "start": round(start / sample_rate, 3),
                    "end": round(end_seconds, 3),
                }
            )

    response = bool(responses)
    early_takeover = bool(early_takeovers)
    success = response and not early_takeover
    latency = (
        round(max(0.0, first_response_start - input_turn_end), 3)
        if success
        else None
    )
    return {
        "response": response,
        "responses": responses,
        "latency": latency,
        "early_takeover": early_takeover,
        "early_takeovers": early_takeovers,
        "success": success,
    }


def _load_pcm_wav(path: Path) -> Any:
    """Load an uncompressed PCM WAV as mono and resample it to 16 kHz."""
    import torch
    import torchaudio.functional as audio_functional

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(
                f"unsupported compressed WAV type: {wav_file.getcomptype()}"
            )
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        audio_bytes = wav_file.readframes(frame_count)

    if channels < 1 or source_rate < 1:
        raise ValueError("WAV must have at least one channel and a positive sample rate")

    buffer = bytearray(audio_bytes)
    if sample_width == 1:
        audio = torch.frombuffer(buffer, dtype=torch.uint8).clone().float()
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.frombuffer(buffer, dtype=torch.int16).clone().float()
        audio /= 32768.0
    elif sample_width == 3:
        raw = torch.frombuffer(buffer, dtype=torch.uint8).clone().reshape(-1, 3)
        values = (
            raw[:, 0].to(torch.int32)
            | (raw[:, 1].to(torch.int32) << 8)
            | (raw[:, 2].to(torch.int32) << 16)
        )
        values = torch.where(values >= 0x800000, values - 0x1000000, values)
        audio = values.float() / 8388608.0
    elif sample_width == 4:
        audio = torch.frombuffer(buffer, dtype=torch.int32).clone().float()
        audio /= 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")

    if audio.numel() % channels:
        raise ValueError("WAV data does not contain a whole number of frames")
    audio = audio.reshape(-1, channels).mean(dim=1)
    if source_rate != SAMPLE_RATE:
        audio = audio_functional.resample(audio, source_rate, SAMPLE_RATE)
    return audio


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
    if "input_turn_end" not in metadata:
        raise KeyError(f"'input_turn_end' is missing from {metadata_path}")
    input_turn_end = float(metadata["input_turn_end"])
    if input_turn_end < 0:
        raise ValueError(f"input_turn_end must be nonnegative in {metadata_path}")

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
    score = score_segments(speech_segments, input_turn_end)

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
