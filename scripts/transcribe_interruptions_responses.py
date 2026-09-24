#!/usr/bin/env python3
"""Transcribe extracted interruption responses with Indic Transcribe Core.

For every ``response.wav`` in one model/language interruptions directory, this
script transcribes the audio serially and atomically writes an indented
``response_score.json`` beside it. The model's bundled long-form helper is used
so responses longer than the model's single-pass limit are chunked safely.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


CATEGORY = "interruptions"
MODEL_ID = "bodhan-ai/indic-transcribe-core"
LANGUAGE_CODES = {
    "bengali": "bn",
    "english": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
}


def transcribe_sample(
    sample_dir: Path,
    asr: Any,
    transcribe_long: Callable[..., str],
    language_code: str,
) -> Path:
    """Transcribe one response.wav and atomically write response_score.json."""
    response_path = sample_dir / "response.wav"
    transcription = transcribe_long(asr, str(response_path), lang=language_code)
    if not isinstance(transcription, str):
        raise TypeError(
            f"transcription must be a string for {response_path}, "
            f"got {type(transcription).__name__}"
        )

    score_path = sample_dir / "response_score.json"
    temporary_path = sample_dir / f".response_score.{os.getpid()}.json"
    try:
        with temporary_path.open("w", encoding="utf-8") as score_file:
            json.dump(
                {"transcription": transcription.strip()},
                score_file,
                ensure_ascii=False,
                indent=2,
            )
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
        "--overwrite",
        action="store_true",
        help="regenerate existing nonempty response_score.json files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    language_name = args.language.lower()
    language_code = LANGUAGE_CODES.get(language_name)
    if language_code is None:
        supported = ", ".join(sorted(LANGUAGE_CODES))
        print(
            f"error: unsupported language {args.language!r}; supported: {supported}",
            file=sys.stderr,
        )
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
        if path.is_dir() and (path / "response.wav").is_file()
    )
    if not sample_dirs:
        print(f"Skipping: no response.wav files found in {output_category_dir}")
        return 0

    pending_sample_dirs = [
        sample_dir
        for sample_dir in sample_dirs
        if args.overwrite
        or not (sample_dir / "response_score.json").is_file()
        or (sample_dir / "response_score.json").stat().st_size == 0
    ]
    skipped = len(sample_dirs) - len(pending_sample_dirs)
    if not pending_sample_dirs:
        print(
            f"Skipping: all {len(sample_dirs)} responses already have nonempty "
            "response_score.json files"
        )
        return 0
    if skipped:
        print(f"Skipping {skipped} existing transcriptions", flush=True)

    try:
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as error:
        print(f"error: missing dependency: {error}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "error: CUDA is not available; run this script in a GPU allocation",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {MODEL_ID} from the local Hugging Face cache", flush=True)
    try:
        model_dir = Path(
            snapshot_download(repo_id=MODEL_ID, local_files_only=True)
        )
    except Exception as error:
        print(f"error: could not load cached model: {error}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(model_dir))
    try:
        from indic_transcribe import IndicTranscribe
        from long_form import transcribe_long
    except ImportError as error:
        print(f"error: cached model is missing inference code: {error}", file=sys.stderr)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model snapshot: {model_dir}", flush=True)
    print(
        f"Found {len(sample_dirs)} response files; "
        f"transcribing {len(pending_sample_dirs)}",
        flush=True,
    )
    try:
        asr = IndicTranscribe.from_pretrained(model_dir)
    except Exception as error:
        print(f"error: failed to load model: {error}", file=sys.stderr)
        return 2

    failures = []
    completed = 0
    progress_interval = max(1, len(pending_sample_dirs) // 20)
    for sample_dir in pending_sample_dirs:
        try:
            transcribe_sample(sample_dir, asr, transcribe_long, language_code)
            completed += 1
        except Exception as error:
            failures.append((sample_dir, error))
            print(
                f"Failed {sample_dir}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

        processed = completed + len(failures)
        if processed % progress_interval == 0 or processed == len(pending_sample_dirs):
            percentage = 100.0 * processed / len(pending_sample_dirs)
            print(
                f"Progress: {processed}/{len(pending_sample_dirs)} samples "
                f"({percentage:.0f}%)",
                flush=True,
            )

    print(
        f"Transcribed {completed}/{len(pending_sample_dirs)} pending responses in "
        f"{output_category_dir}; skipped existing: {skipped}; "
        f"failures: {len(failures)}.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
