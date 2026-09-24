#!/usr/bin/env python3
"""Evaluate pause-handling outputs with Silero VAD.

For every output sample, speech regions separated by less than 0.2 seconds are
joined.  Any resulting region longer than 2.0 seconds is a takeover.
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


CATEGORY = "pausehandling"
SAMPLE_RATE = 16_000
MAX_JOIN_GAP_SECONDS = 0.2
MAX_BACKCHANNEL_SECONDS = 2.0
VAD_THRESHOLD = 0.7
DEFAULT_WORKERS = 16

_thread_state = threading.local()


def merge_speech_segments(
    segments: Sequence[Dict[str, int]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[int, int]]:
    """Join VAD regions whose intervening gap is strictly less than 0.2 s."""
    if not segments:
        return []

    max_gap_samples = MAX_JOIN_GAP_SECONDS * sample_rate
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


def find_takeovers(
    segments: Sequence[Dict[str, int]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Dict[str, float]]:
    """Return continuous speech regions longer than the backchannel limit."""
    takeovers = []
    for start, end in merge_speech_segments(segments, sample_rate):
        if (end - start) / sample_rate > MAX_BACKCHANNEL_SECONDS:
            takeovers.append(
                {
                    "start": round(start / sample_rate, 3),
                    "end": round(end / sample_rate, 3),
                }
            )
    return takeovers


def _load_pcm_wav(path: Path) -> Any:
    """Load a PCM WAV as a mono float tensor and resample it to 16 kHz."""
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

    # bytearray makes the buffer writable and suppresses torch.frombuffer's
    # non-writable-buffer warning.  Clone detaches the tensor from that buffer.
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


def evaluate_sample(sample_dir: Path) -> Path:
    """Evaluate one sample and overwrite its score.json."""
    from silero_vad import get_speech_timestamps, read_audio

    audio_path = sample_dir / "output.wav"
    audio = read_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    speech_segments = get_speech_timestamps(
        audio,
        _get_model(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        return_seconds=False,
    )
    takeovers = find_takeovers(speech_segments)
    takeover = bool(takeovers)
    score = {
        "takeover": takeover,
        "takeovers": takeovers,
        "success": not takeover,
    }

    score_path = sample_dir / "score.json"
    temporary_path = sample_dir / f".score.json.{os.getpid()}.{threading.get_ident()}"
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
        help="model output root (for example, human)",
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

    # Silero recommends one Torch computation thread. Parallelism is supplied
    # by the executor, with an independent stateful VAD model in each worker.
    try:
        import torch
        import torchaudio  # noqa: F401 - verify the required resampler is available
        import soundfile  # noqa: F401 - provide read_audio's fallback backend
        import silero_vad  # noqa: F401 - provide a clear dependency failure up front
    except ImportError as error:
        print(f"error: missing dependency: {error}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_sample, sample_dir): sample_dir
            for sample_dir in sample_dirs
        }
        for future in as_completed(future_to_sample):
            sample_dir = future_to_sample[future]
            try:
                future.result()
            except Exception as error:  # Keep processing independent samples.
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
