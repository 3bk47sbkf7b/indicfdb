#!/usr/bin/env python3
"""Translate interruption response transcriptions into English with Gemma 4.

For each existing ``response_score.json`` in one model/language interruptions
directory, this script reads ``transcription`` and adds ``transcription_en``.
English transcriptions are copied without model processing. Other languages use
the plain-prompt style from the tested Gemma 4 31B IT translation recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CATEGORY = "interruptions"
MODEL_ID = "google/gemma-4-31B-it"
LANGUAGE_NAMES = {
    "bengali": "Bengali",
    "english": "English",
    "gujarati": "Gujarati",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "marathi": "Marathi",
    "punjabi": "Punjabi",
    "tamil": "Tamil",
    "telugu": "Telugu",
}
TARGET_LANGUAGE = "English"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 512


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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Gemma generation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "maximum generated tokens per translation "
            f"(default: {DEFAULT_MAX_NEW_TOKENS})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate existing transcription_en values",
    )
    return parser.parse_args()


def read_score(score_path: Path) -> dict[str, Any]:
    """Read and validate the source transcription in one response score."""
    with score_path.open("r", encoding="utf-8") as score_file:
        score = json.load(score_file)
    if not isinstance(score, dict):
        raise ValueError(f"response score must be a JSON object: {score_path}")
    if "transcription" not in score:
        raise KeyError(f"'transcription' is missing from {score_path}")
    if not isinstance(score["transcription"], str):
        raise TypeError(f"'transcription' must be a string in {score_path}")
    return score


def write_score_atomic(score_path: Path, score: dict[str, Any]) -> None:
    """Atomically replace one response score with indented UTF-8 JSON."""
    temporary_path = score_path.with_name(
        f".{score_path.stem}.{os.getpid()}.json"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as score_file:
            json.dump(score, score_file, ensure_ascii=False, indent=2)
            score_file.write("\n")
        os.replace(temporary_path, score_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def translation_prompt(source_language: str, source_text: str) -> str:
    """Return the tested Gemma plain-style translation prompt."""
    return (
        f"Translate the following text from {source_language} into "
        f"{TARGET_LANGUAGE}.\n"
        f"Output only the translation in {TARGET_LANGUAGE}, with no label or "
        "additional commentary.\n\n"
        f"{source_text}"
    )


def extract_parsed_text(parsed: Any) -> str | None:
    """Extract text from the processor's version-dependent parsed response."""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, (tuple, list)):
        for item in reversed(parsed):
            result = extract_parsed_text(item)
            if result:
                return result
    if isinstance(parsed, dict):
        for key in ("final", "answer", "content", "text", "response"):
            if key in parsed:
                result = extract_parsed_text(parsed[key])
                if result:
                    return result
    for attribute in ("final", "answer", "content", "text", "response"):
        if hasattr(parsed, attribute):
            result = extract_parsed_text(getattr(parsed, attribute))
            if result:
                return result
    return None


def decode_completion(processor: Any, token_ids: Any, prefix: Any) -> str:
    """Decode one Gemma completion using its structured parser when available."""
    if hasattr(processor, "parse_response"):
        raw = processor.decode(token_ids, skip_special_tokens=False)
        try:
            parsed = extract_parsed_text(
                processor.parse_response(raw, prefix=prefix)
            )
            if parsed:
                return parsed.strip()
        except Exception as error:
            print(
                f"Warning: processor.parse_response failed ({error}); "
                "using decoded text",
                flush=True,
            )
    return processor.decode(token_ids, skip_special_tokens=True).strip()


def tokenize_prompts(processor: Any, prompt_texts: list[str]) -> Any:
    """Apply the Gemma IT chat template with thinking disabled."""
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompt_texts
    ]
    return processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )


def load_model() -> tuple[Any, Any]:
    """Load the cached Gemma 4 31B IT processor and model."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    model_dir = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    common = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    try:
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    model = model_class.from_pretrained(model_dir, **common).eval()
    return processor, model


def report_progress(processed: int, total: int, interval: int) -> None:
    """Print progress at approximately five-percent intervals."""
    if processed % interval == 0 or processed == total:
        percentage = 100.0 * processed / total
        print(f"Progress: {processed}/{total} samples ({percentage:.0f}%)", flush=True)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("error: --batch-size must be at least 1", file=sys.stderr)
        return 2
    if args.max_new_tokens < 1:
        print("error: --max-new-tokens must be at least 1", file=sys.stderr)
        return 2

    language_key = args.language.lower()
    source_language = LANGUAGE_NAMES.get(language_key)
    if source_language is None:
        supported = ", ".join(sorted(LANGUAGE_NAMES))
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

    all_sample_dirs = sorted(path for path in output_category_dir.iterdir() if path.is_dir())
    score_paths = [
        sample_dir / "response_score.json"
        for sample_dir in all_sample_dirs
        if (sample_dir / "response_score.json").is_file()
    ]
    missing_scores = len(all_sample_dirs) - len(score_paths)
    if not score_paths:
        print(
            f"Skipping: no response_score.json files found in {output_category_dir}; "
            f"samples without scores: {missing_scores}"
        )
        return 0

    failures: list[tuple[Path, Exception]] = []
    pending: list[tuple[Path, dict[str, Any]]] = []
    skipped_existing = 0
    for score_path in score_paths:
        try:
            score = read_score(score_path)
            if "transcription_en" in score and not args.overwrite:
                skipped_existing += 1
                continue
            pending.append((score_path, score))
        except Exception as error:
            failures.append((score_path, error))
            print(
                f"Failed {score_path}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    if not pending:
        print(
            f"Nothing to translate in {output_category_dir}; "
            f"missing scores: {missing_scores}; skipped existing: "
            f"{skipped_existing}; failures: {len(failures)}."
        )
        return 1 if failures else 0

    print(
        f"Found {len(score_paths)} response scores; pending: {len(pending)}; "
        f"missing scores: {missing_scores}; skipped existing: {skipped_existing}",
        flush=True,
    )

    completed = 0
    copied = 0
    processed = 0
    progress_interval = max(1, len(pending) // 20)

    if language_key == "english":
        for score_path, score in pending:
            score["transcription_en"] = score["transcription"]
            try:
                write_score_atomic(score_path, score)
                completed += 1
                copied += 1
            except Exception as error:
                failures.append((score_path, error))
                print(
                    f"Failed {score_path}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            processed += 1
            report_progress(processed, len(pending), progress_interval)
    else:
        model_work: list[tuple[Path, dict[str, Any], str]] = []
        for score_path, score in pending:
            source_text = score["transcription"].strip()
            if source_text:
                model_work.append(
                    (
                        score_path,
                        score,
                        translation_prompt(source_language, source_text),
                    )
                )
            else:
                score["transcription_en"] = ""
                try:
                    write_score_atomic(score_path, score)
                    completed += 1
                    copied += 1
                except Exception as error:
                    failures.append((score_path, error))
                    print(
                        f"Failed {score_path}: {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                processed += 1
                report_progress(processed, len(pending), progress_interval)

        if model_work:
            try:
                import torch
            except ImportError as error:
                print(f"error: missing dependency: {error}", file=sys.stderr)
                return 2
            if not torch.cuda.is_available():
                print(
                    "error: CUDA is not available; run this script in a GPU allocation",
                    file=sys.stderr,
                )
                return 2

            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Loading {MODEL_ID} from the local Hugging Face cache", flush=True)
            try:
                processor, model = load_model()
            except Exception as error:
                print(f"error: failed to load model: {error}", file=sys.stderr)
                return 2

            for offset in range(0, len(model_work), args.batch_size):
                batch = model_work[offset : offset + args.batch_size]
                try:
                    inputs = tokenize_prompts(
                        processor,
                        [prompt for _, _, prompt in batch],
                    ).to(model.device)
                    input_width = inputs["input_ids"].shape[1]
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )

                    decoded = [
                        decode_completion(
                            processor,
                            row[input_width:],
                            inputs["input_ids"][index],
                        )
                        for index, row in enumerate(generated)
                    ]
                    empty_paths = [
                        score_path
                        for translation, (score_path, _, _) in zip(decoded, batch)
                        if not translation
                    ]
                    if empty_paths:
                        raise RuntimeError(
                            "Gemma returned an empty translation for "
                            + ", ".join(map(str, empty_paths))
                        )
                    for translation, (score_path, score, _) in zip(decoded, batch):
                        score["transcription_en"] = translation
                        write_score_atomic(score_path, score)
                        completed += 1
                except Exception as error:
                    for score_path, _, _ in batch:
                        failures.append((score_path, error))
                        print(
                            f"Failed {score_path}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if args.batch_size > 1:
                        break
                finally:
                    processed += len(batch)
                    report_progress(processed, len(pending), progress_interval)

    print(
        f"Translated {completed}/{len(pending)} pending response scores in "
        f"{output_category_dir}; copied without model: {copied}; "
        f"missing scores: {missing_scores}; skipped existing: "
        f"{skipped_existing}; failures: {len(failures)}.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
