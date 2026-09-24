#!/usr/bin/env python3
"""Evaluate agent speech during backchanneling samples with Silero VAD.

For every output sample, speech regions separated by less than 0.5 seconds are
joined. A resulting region at most 1.2 seconds long is a backchannel; a longer
region is a takeover. Agent backchannel timing is compared with the reference
human backchannels using FullDuplexBench-compatible Jensen-Shannon distance.
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


CATEGORY = "backchanneling"
SAMPLE_RATE = 16_000
MAX_JOIN_GAP_SECONDS = 0.5
MAX_BACKCHANNEL_SECONDS = 1.2
VAD_THRESHOLD = 0.7
JSD_WINDOW_SECONDS = 0.2
JSD_EPSILON = 1e-10
DEFAULT_WORKERS = 16

_thread_state = threading.local()


def merge_speech_segments(
    segments: Sequence[Dict[str, int]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[int, int]]:
    """Join VAD regions whose intervening gap is strictly less than 0.5 s."""
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


def score_segments(
    segments: Sequence[Dict[str, int]],
    sample_duration: float,
    human_backchannels: Sequence[Dict[str, float]],
    clip_duration: float,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """Classify merged VAD regions and build the backchanneling score."""
    if sample_duration <= 0:
        raise ValueError("sample duration must be positive")

    backchannels = []
    takeovers = []
    for start, end in merge_speech_segments(segments, sample_rate):
        scored_segment = {
            "start": round(start / sample_rate, 3),
            "end": round(end / sample_rate, 3),
        }
        if (end - start) / sample_rate <= MAX_BACKCHANNEL_SECONDS:
            backchannels.append(scored_segment)
        else:
            takeovers.append(scored_segment)

    takeover = bool(takeovers)
    return {
        "backchannels": backchannels,
        "backchannel_frequency": round(len(backchannels) / sample_duration, 5),
        "jsd": calculate_jsd(backchannels, human_backchannels, clip_duration),
        "takeover": takeover,
        "takeovers": takeovers,
        "success": not takeover,
    }


def calculate_jsd(
    backchannels: Sequence[Dict[str, float]],
    human_backchannels: Sequence[Dict[str, float]],
    clip_duration: float,
) -> float:
    """Return FullDuplexBench-style Jensen-Shannon distance for one sample."""
    import numpy as np
    from scipy.spatial.distance import jensenshannon

    if clip_duration <= 0:
        raise ValueError("clip duration must be positive")
    if not human_backchannels:
        raise ValueError("human_backchannels must not be empty")
    if not backchannels:
        return 1.0

    bin_count = int(clip_duration / JSD_WINDOW_SECONDS) + 1

    def distribution(intervals: Sequence[Dict[str, float]]) -> Any:
        histogram = np.zeros(bin_count, dtype=float)
        for interval in intervals:
            start_time = float(
                interval.get("start", interval.get("backchannel_start"))
            )
            end_time = float(interval.get("end", interval.get("backchannel_end")))
            if start_time < 0 or end_time <= start_time or end_time > clip_duration:
                raise ValueError(
                    f"invalid backchannel interval [{start_time}, {end_time}] "
                    f"for clip duration {clip_duration}"
                )
            start_bin = int(start_time / JSD_WINDOW_SECONDS)
            end_bin = int(end_time / JSD_WINDOW_SECONDS)
            for index in range(start_bin, end_bin + 1):
                if index < bin_count:
                    histogram[index] += 1
        histogram += JSD_EPSILON
        return histogram / histogram.sum()

    predicted_distribution = distribution(backchannels)
    human_distribution = distribution(human_backchannels)
    return round(
        float(jensenshannon(predicted_distribution, human_distribution)),
        5,
    )


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
    if "clip_duration" not in metadata or "human_backchannels" not in metadata:
        raise KeyError(
            f"'clip_duration' and 'human_backchannels' are required in {metadata_path}"
        )
    clip_duration = float(metadata["clip_duration"])
    human_backchannels = metadata["human_backchannels"]

    audio = read_audio(
        str(output_sample_dir / "output.wav"),
        sampling_rate=SAMPLE_RATE,
    )
    sample_duration = audio.numel() / SAMPLE_RATE
    speech_segments = get_speech_timestamps(
        audio,
        _get_model(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        return_seconds=False,
    )
    score = score_segments(
        speech_segments,
        sample_duration,
        human_backchannels,
        clip_duration,
    )

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
        import torchaudio  # noqa: F401 - verify the required resampler is available
        import soundfile  # noqa: F401 - provide read_audio's fallback backend
        import silero_vad  # noqa: F401 - provide a clear dependency failure up front
        import scipy  # noqa: F401 - use FDB's Jensen-Shannon implementation
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
